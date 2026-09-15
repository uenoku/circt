//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the LowerSyncProcesses pass, which lowers clocked
// synchronous processes with conditional drives to `seq.firreg` registers.
//
// Deseq handles canonical async-reset flops (observed clock+reset) and
// Sig2Reg handles signals with unconditional drives. What remains in
// practice (e.g. Chisel TileLink adapters) are multi-result processes with a
// single observed clock, a rising-edge resume gate, and one conditional drive
// per state signal. Since the context is synchronous, sync reset, enable and
// hold all reduce to next-state muxes feeding plain `seq.firreg` ops -- there
// are no flop reset pins involved, mirroring the reference mapped shape for the
// same RTL given Liberty flops without sync-reset pins.
//
// Soundness contract (violations skip the process loudly, changing nothing):
// - Single wait, no delay, no halt; exactly one observed i1 clock; the
//   single resume operand carries the past clock (threaded as wait arg 0
//   or passed straight from the observed value). Async contexts (extra
//   observed values) stay Deseq's domain.
// - The resume block ends with cond_br(GATE, sample, waitBlock) where GATE
//   structurally matches rising-edge detection `and(xor(past,true),clk)`
//   on the observed clock, and the sample entry takes no arguments. The
//   gate is proven but not materialized: plain `seq.firreg` holds by
//   itself on wakes without an edge, so next-state logic only needs to be
//   right when the sample runs (flop-for-flop for LEC).
// - The entry block branches directly to the wait block with matching
//   arity and holds only constants (cycle-0 values, ignored for synthesis
//   where power-up is unknown, matching reference-mapped netlists).
// - Wait yields pass the wait-block arguments through in order (state
//   slots past an optional past-clock arg 0).
// - Sample region (forward-reachable from the sample side, excluding the
//   wait and entry blocks) holds only clonable data ops (comb/arith,
//   side-effect-free ops, probes) plus br/cond_br; cycles, entry
//   successors, and other terminators bail out.
// - Every signal driven by process results is adopted with a single
//   (value, enable) result pair (plus ignored constant init drives);
//   ambiguous or unadoptable drives bail out, as does any leftover drive
//   that would dangle or fight a flop. Two signals sharing one value
//   result bail out (divergence unrepresentable with one flop).
// - Next-state evaluation has three cooperating parts: evalSample walks
//   the sample DAG forward from its entry to the wait-block completion
//   operands (nested conditionals become nested muxes); evalBlockArg
//   folds intermediate merges as reachability-selected cascades
//   (deterministic CFG: exactly one predecessor edge executes, and some
//   always does); evalValue clones data ops, reads flop outputs for past
//   state, and passes module-level values (ports, instance outputs,
//   other flops, input-wire probes) through opaquely, which keeps
//   cross-process reads order-independent under fixpoint iteration.
// - Driven signals update by their enable position (bypass paths carry
//   constant-false enables, so dummy completion values are never
//   selected); undriven live results thread completion values through.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/LLHD/LLHDOps.h"
#include "circt/Dialect/LLHD/LLHDPasses.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

namespace circt {
namespace llhd {
#define GEN_PASS_DEF_LOWERSYNCPROCESSESPASS
#include "circt/Dialect/LLHD/LLHDPasses.h.inc"
} // namespace llhd
} // namespace circt

using namespace mlir;
using namespace circt;

#define DEBUG_TYPE "llhd-lower-sync-processes"

namespace {

// Allowed inside sample regions (and the resume block modulo the gate
// structure enforced separately): pure combinational logic, constants,
// control flow, and signal probes. Everything else bails the process out.
// Clonable into module-level next-state logic: combinational dialects,
// side-effect-free ops (constants, bitcast, array/struct accessors, ...),
// and llhd probes (resolved or cloned). Control flow is handled
// separately; multi-result ops and anything stateful bails out.
bool isAllowedDataOp(Operation *op) {
  if (isa<cf::BranchOp, cf::CondBranchOp>(op) || op->getNumResults() != 1 ||
      op->getNumRegions() != 0)
    return false;
  return isa<comb::CombDialect>(op->getDialect()) ||
         isa<arith::ArithDialect>(op->getDialect()) || isa<llhd::ProbeOp>(op) ||
         mlir::isMemoryEffectFree(op);
}

struct SyncLowering {
  SyncLowering(hw::HWModuleOp module, llhd::ProcessOp process)
      : module(module), process(process), builder(process.getContext()),
        loc(process.getLoc()) {}

  LogicalResult tryLower();

private:
  hw::HWModuleOp module;
  llhd::ProcessOp process;
  OpBuilder builder;
  Location loc;

  // Phase A: structural match.
  llhd::WaitOp waitOp = nullptr;
  Block *waitBlock = nullptr;
  Block *entryBlock = nullptr;
  Block *resumeBlock = nullptr;
  Block *sampleBlock = nullptr;
  Value clockVal;
  unsigned numResults = 0;
  // Wait-block argument layout: variant (a) threads the past clock as
  // arg[0] (`pastShift == 1`, dest operand is that block argument);
  // variant (b) passes the observed clock straight to resume
  // (`pastShift == 0`, dest operand is the observed value itself).
  unsigned pastShift = 0;
  DenseSet<Block *> sampleBlocks;

  // Phase B: signal census (module block only).
  struct AdoptedSig {
    llhd::SignalOp sigOp;
    unsigned resultIdx = 0; // process result driving the value
    int enableIdx = -2;     // process result driving the enable,
                            // or -2 for unconditional (no enable operand)
    SmallVector<llhd::DriveOp> drives;
  };
  SmallVector<AdoptedSig> adopted;

  // Phase C: evaluation state.
  DenseMap<Value, Value> memo;
  DenseMap<std::pair<Block *, unsigned>, Value> sampleMemo;
  DenseMap<Block *, Value> pathMemo;
  // Separate recursion guards: forward sample evaluation and backward
  // argument merging share (block, index) keys but have independent
  // recursion schemes; sharing one set causes false cycle reports when
  // sample evaluation references the same block's arguments.
  DenseSet<std::pair<Block *, unsigned>> sampleInProgress;
  DenseSet<std::pair<Block *, unsigned>> argInProgress;
  DenseSet<Block *> pathInProgress;
  SmallVector<Operation *> created;              // erased on failure
  DenseMap<unsigned, seq::FirRegOp> resultFlops; // result idx -> flop

  LogicalResult analyzeStructure();
  LogicalResult analyzeDrives();
  LogicalResult buildAndRewrite();
  FailureOr<Value> evalValue(Value v);
  FailureOr<Value> evalBlockArg(Block *block, unsigned idx);
  // Forward completion evaluation: the value result position `pos` will
  // carry if control is at `block` (sample region), following taken
  // branches to the wait-block completion operands.
  FailureOr<Value> evalSample(Block *block, unsigned pos);
  // Reachability predicate: under which condition does control reach
  // `block` on a sample wake (materialized as combinational logic).
  FailureOr<Value> evalPath(Block *block);
  FailureOr<Value> edgeCond(Block *pred, bool takenOnTrue);
  FailureOr<Value> edgeSelector(Block *pred, bool takenOnTrue);
  Value trueConst();
  static bool isTrueConst(Value v);
  FailureOr<Value> resolveProbe(llhd::ProbeOp prb);
  void eraseCreated();
  LogicalResult skip(const Twine &msg);
};

LogicalResult SyncLowering::skip(const Twine &msg) {
  LLVM_DEBUG(llvm::dbgs() << "Skipping " << process.getLoc() << ": " << msg
                          << "\n");
  return failure();
}

LogicalResult SyncLowering::tryLower() {
  if (failed(analyzeStructure()))
    return failure();
  if (failed(analyzeDrives()))
    return failure();
  if (failed(buildAndRewrite())) {
    eraseCreated();
    return failure();
  }
  return success();
}

void SyncLowering::eraseCreated() {
  // Unconditional reverse erase is safe: on the failure path nothing
  // outside `created` references these ops yet (flop inputs are assigned
  // and uses rewired only on success).
  for (auto *op : llvm::reverse(created))
    op->erase();
  created.clear();
  memo.clear();
  sampleMemo.clear();
  resultFlops.clear();
}

//===----------------------------------------------------------------------===//
// Phase A: structural match
//===----------------------------------------------------------------------===//

LogicalResult SyncLowering::analyzeStructure() {
  auto &body = process.getBody();
  entryBlock = &body.front();

  // Single wait, no delay, no halt anywhere.
  for (auto &block : body) {
    for (auto &op : block) {
      if (isa<llhd::HaltOp>(op))
        return skip("contains halt");
      if (auto wait = dyn_cast<llhd::WaitOp>(op)) {
        if (waitOp)
          return skip("multiple waits");
        if (block.getTerminator() != &op)
          return skip("wait is not a terminator");
        waitOp = wait;
      }
    }
  }
  if (!waitOp)
    return skip("no wait");
  if (waitOp.getDelay())
    return skip("wait has delay");
  waitBlock = waitOp->getBlock();

  // Exactly one observed clock. The single dest operand carries the past
  // clock state into the resume block (verified against the edge gate
  // below). Anything else (e.g. an observed async reset) stays Deseq's
  // domain.
  if (waitOp.getObserved().size() != 1 ||
      !waitOp.getObserved()[0].getType().isInteger(1))
    return skip("wait does not observe exactly one i1 clock");
  if (waitOp.getDestOperands().size() != 1)
    return skip("wait dest arity is not one");
  if (waitOp.getDest() == waitBlock || waitOp.getDest() == entryBlock)
    return skip("degenerate wait destination");
  clockVal = waitOp.getObserved()[0];
  resumeBlock = waitOp.getDest();

  // Yields must be the wait block's arguments in order, with matching
  // counts: yields pass the wait-block arguments through in order. The
  // past clock may occupy arg[0] (variant a) or be passed straight to
  // resume (variant b).
  numResults = process.getNumResults();
  unsigned numWaitArgs = waitBlock->getNumArguments();
  if (waitOp.getYieldOperands().size() != numResults ||
      (numWaitArgs != numResults && numWaitArgs != numResults + 1))
    return skip("yield/argument/result count mismatch");
  pastShift = numWaitArgs - numResults;
  if (pastShift == 1) {
    if (waitOp.getDestOperands()[0] != waitBlock->getArgument(0))
      return skip("past clock is not wait arg 0");
  } else {
    if (waitOp.getDestOperands()[0] != waitOp.getObserved()[0])
      return skip("resume is not fed the observed clock");
  }
  for (unsigned i = 0; i < numResults; ++i) {
    if (waitOp.getYieldOperands()[i] != waitBlock->getArgument(pastShift + i))
      return skip("wait yields are not block arguments");
  }

  // The resume block computes the gate and branches to sample vs. hold; the
  // hold side must go directly back to the wait block. Only combinational
  // logic and constants may appear besides the branch.
  if (resumeBlock->getNumArguments() != 1)
    return skip("resume block takes more than the past clock");
  for (auto &op : resumeBlock->without_terminator()) {
    if (isa<cf::BranchOp, cf::CondBranchOp, llhd::ProbeOp>(op))
      return skip("unsupported op in resume block");
    if (!isAllowedDataOp(&op))
      return skip("unsupported op in resume block");
  }
  auto resumeBr = dyn_cast<cf::CondBranchOp>(resumeBlock->getTerminator());
  if (!resumeBr)
    return skip("resume block does not end with cond_br");
  if (resumeBr.getFalseDest() != waitBlock)
    return skip("resume hold side is not the wait block");
  sampleBlock = resumeBr.getTrueDest();
  if (sampleBlock == waitBlock || sampleBlock == entryBlock)
    return skip("degenerate sample block");
  if (sampleBlock->getNumArguments() != 0)
    return skip("sample block takes arguments");

  // The gate must be rising-edge detection `and(xor(past,true),clk)` on the
  // observed clock, in either operand order.
  Value pastArg = resumeBlock->getArgument(0);
  auto isTrueConst = [](Value v) {
    if (auto constOp = v.getDefiningOp<hw::ConstantOp>()) {
      if (auto intTy = dyn_cast<IntegerType>(constOp.getType()))
        return intTy.getWidth() == 1 && constOp.getValue() == 1;
    }
    return false;
  };
  auto matchXorPast = [&](Value v) {
    auto xorOp = v.getDefiningOp<comb::XorOp>();
    if (!xorOp || xorOp.getInputs().size() != 2)
      return false;
    auto inputs = xorOp.getInputs();
    return (inputs[0] == pastArg && isTrueConst(inputs[1])) ||
           (inputs[1] == pastArg && isTrueConst(inputs[0]));
  };
  auto gateAnd = resumeBr.getCondition().getDefiningOp<comb::AndOp>();
  if (!gateAnd || gateAnd.getInputs().size() != 2)
    return skip("resume condition is not an edge gate");
  auto gateInputs = gateAnd.getInputs();
  bool gateOk = (matchXorPast(gateInputs[0]) && gateInputs[1] == clockVal) ||
                (matchXorPast(gateInputs[1]) && gateInputs[0] == clockVal);
  if (!gateOk)
    return skip("resume condition is not rising-edge detection");

  // The entry block only provides cycle-0 values, which synthesis ignores
  // (matching reference-mapped netlists where power-up is unknown). It must
  // branch directly to the wait block and hold no live logic.
  auto entryBr = dyn_cast<cf::BranchOp>(entryBlock->getTerminator());
  if (!entryBr || entryBr.getDest() != waitBlock)
    return skip("entry does not branch to the wait block");
  if (entryBr.getDestOperands().size() != waitBlock->getNumArguments())
    return skip("entry arity mismatch");
  for (auto &op : entryBlock->without_terminator()) {
    if (!isa<hw::ConstantOp>(op))
      return skip("live logic in entry block");
  }

  // Collect the sample region: forward-reachable from the sample block,
  // never entering the wait or entry blocks. Depth-first search with
  // gray/black coloring: revisiting a block on the current path (gray)
  // is a genuine cycle, while revisiting a finished block (black) is an
  // ordinary diamond join.
  DenseSet<Block *> gray, black;
  black.insert(waitBlock);
  black.insert(entryBlock);
  std::function<LogicalResult(Block *)> visit = [&](Block *block) {
    if (!gray.insert(block).second)
      return skip("cycle in sample region");
    if (!black.insert(block).second) {
      gray.erase(block);
      return success(); // Join: already validated.
    }
    for (auto &op : *block) {
      if (isa<cf::BranchOp, cf::CondBranchOp, llhd::WaitOp>(op))
        continue;
      if (!isAllowedDataOp(&op))
        return skip("unsupported op in sample region");
    }
    auto *terminator = block->getTerminator();
    if (!isa<cf::BranchOp, cf::CondBranchOp>(terminator))
      return skip("unsupported terminator in sample region");
    for (unsigned i = 0; i < terminator->getNumSuccessors(); ++i) {
      Block *succ = terminator->getSuccessor(i);
      // A path back to the entry block would re-run the gate computation
      // with no wait in between (two samples per wake); completions
      // assume every wake runs the sample region at most once.
      if (succ == entryBlock)
        return skip("sample region returns to entry");
      if (succ == waitBlock)
        continue;
      if (failed(visit(succ)))
        return failure();
    }
    gray.erase(block);
    return success();
  };
  if (failed(visit(sampleBlock)))
    return failure();
  black.erase(waitBlock);
  black.erase(entryBlock);
  sampleBlocks = std::move(black);
  return success();
}

//===----------------------------------------------------------------------===//
// Phase B: drive and probe census
//===----------------------------------------------------------------------===//

LogicalResult SyncLowering::analyzeDrives() {
  Block *topBlock = module.getBodyBlock();
  // Map each signal to the drives targeting it (same block only).
  DenseMap<Value, SmallVector<llhd::DriveOp>> drivesBySig;
  for (auto &op : *topBlock) {
    auto drive = dyn_cast<llhd::DriveOp>(op);
    if (!drive)
      continue;
    drivesBySig[drive.getSignal()].push_back(drive);
  }

  // Adopt signals driven by exactly one of our results (plus ignored
  // constant init drives). Anything else bails the process out.
  for (auto &op : *topBlock) {
    auto sig = dyn_cast<llhd::SignalOp>(op);
    if (!sig)
      continue;
    auto it = drivesBySig.find(sig.getResult());
    if (it == drivesBySig.end())
      continue; // Undriven or externally driven; not ours.
    int valueResult = -1;
    int enableResult = -2; // -2 = unconditional (no enable operand yet)
    bool ok = true;
    SmallVector<llhd::DriveOp> owned;
    for (auto drive : it->second) {
      Value driveVal = drive.getValue();
      // Unconditional drives: constant values are init residue (ignored);
      // process-result values are unconditional state updates (adopted).
      if (!drive.getEnable()) {
        if (driveVal.getDefiningOp<hw::ConstantOp>()) {
          owned.push_back(drive);
          continue;
        }
        auto resVal = dyn_cast<OpResult>(driveVal);
        if (!resVal || resVal.getOwner() != process) {
          ok = false;
          break;
        }
        if (valueResult >= 0 &&
            (unsigned)valueResult != resVal.getResultNumber()) {
          ok = false;
          break;
        }
        valueResult = resVal.getResultNumber();
        owned.push_back(drive);
        continue;
      }
      // Conditional drives must carry our results. All drives to one
      // signal must share a single (value, enable) result pair: the
      // next-state mux selects by the enable, which is only sound for one
      // driver. (HoistSignals gives each drive fresh results, so sharing
      // already implies a single conditional drive plus init residue.)
      auto resVal = dyn_cast<OpResult>(driveVal);
      if (!resVal || resVal.getOwner() != process) {
        ok = false;
        break;
      }
      if (valueResult >= 0 &&
          (unsigned)valueResult != resVal.getResultNumber()) {
        ok = false;
        break;
      }
      valueResult = resVal.getResultNumber();
      Value enable = drive.getEnable();
      auto enRes = dyn_cast<OpResult>(enable);
      // Constant-true enables cannot distinguish bypass paths (which must
      // hold) from drive paths; refuse. Anything but a process result is
      // unprovable.
      if (!enRes || enRes.getOwner() != process) {
        ok = false;
        break;
      }
      if (enableResult >= 0 && enableResult != (int)enRes.getResultNumber()) {
        ok = false;
        break;
      }
      enableResult = enRes.getResultNumber();
      owned.push_back(drive);
    }
    if (!ok || valueResult < 0)
      continue; // Not ours (or ambiguous); leave the signal alone.
    // Two signals sharing one value result could diverge through different
    // enables, which a single flop per result cannot represent.
    for (auto &other : adopted)
      if (other.resultIdx == (unsigned)valueResult)
        return skip("two signals share one value result");
    adopted.push_back(
        {sig, (unsigned)valueResult, enableResult, std::move(owned)});
  }
  if (adopted.empty())
    return skip("no state signals driven by process results");

  // Ownership check: every module-level drive touching our results (as
  // value or enable) must target an adopted signal and be owned. Lowering
  // erases the process, so a leftover drive referencing our results would
  // dangle; a leftover drive to an adopted signal would fight the flop.
  DenseSet<llhd::DriveOp> ownedDrives;
  for (auto &adoptedSig : adopted)
    for (auto drive : adoptedSig.drives)
      ownedDrives.insert(drive);
  for (auto &op : *topBlock) {
    auto drive = dyn_cast<llhd::DriveOp>(op);
    if (!drive)
      continue;
    bool touchesOurs = false;
    if (auto res = dyn_cast<OpResult>(drive.getValue()))
      touchesOurs |= res.getOwner() == process;
    if (drive.getEnable())
      if (auto res = dyn_cast<OpResult>(drive.getEnable()))
        touchesOurs |= res.getOwner() == process;
    if (!touchesOurs)
      continue;
    if (!ownedDrives.contains(drive))
      return skip("unadoptable drive references process results");
    // Owned drives to adopted signals are erased with the flop rewiring;
    // drives to non-adopted signals would have been adopted above.
    bool toAdopted = false;
    for (auto &adoptedSig : adopted)
      if (adoptedSig.sigOp.getResult() == drive.getSignal())
        toAdopted = true;
    if (!toAdopted)
      return skip("owned drive targets non-adopted signal");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Phase C: symbolic evaluation and rewrite
//===----------------------------------------------------------------------===//

FailureOr<Value> SyncLowering::resolveProbe(llhd::ProbeOp prb) {
  Value sig = prb.getSignal();
  // Lowered signals read back their flop output. Signals adopted here
  // resolve once their flop exists; signals from already-lowered processes
  // were rewired away, so an unresolvable probe means its process has not
  // lowered yet (fixpoint retries) or never will (skip).
  for (auto &adoptedSig : adopted) {
    if (adoptedSig.sigOp.getResult() != sig)
      continue;
    auto flopIt = resultFlops.find(adoptedSig.resultIdx);
    if (flopIt == resultFlops.end())
      return failure();
    return flopIt->second.getData();
  }
  return failure();
}

FailureOr<Value> SyncLowering::evalSample(Block *block, unsigned pos) {
  auto key = std::make_pair(block, pos);
  auto it = sampleMemo.find(key);
  if (it != sampleMemo.end())
    return it->second;
  if (!sampleBlocks.count(block)) {
    return failure();
  }
  if (!sampleInProgress.insert(key).second) {
    return failure(); // Cycle (rejected structurally; guard anyway).
  }
  struct EraseOnExit {
    DenseSet<std::pair<Block *, unsigned>> &set;
    std::pair<Block *, unsigned> key;
    ~EraseOnExit() { set.erase(key); }
  } guard{sampleInProgress, key};
  // Collect one value per successor: completion operands at the wait
  // block, recursion into sample successors. Entry/resume successors
  // cannot occur (rejected during the structural scan).
  struct SuccVal {
    bool takenOnTrue;
    Value value;
  };
  SmallVector<SuccVal> succs;
  auto *term = block->getTerminator();
  auto takeSuccessor = [&](Block *succ, OperandRange operands,
                           bool takenOnTrue) -> LogicalResult {
    if (succ == waitBlock) {
      if (pos >= operands.size()) {
        return failure();
      }
      auto val = evalValue(operands[pos]);
      if (failed(val)) {
        return failure();
      }
      succs.push_back({takenOnTrue, *val});
      return success();
    }
    if (!sampleBlocks.count(succ)) {
      return failure();
    }
    auto val = evalSample(succ, pos);
    if (failed(val)) {
      return failure();
    }
    succs.push_back({takenOnTrue, *val});
    return success();
  };
  if (auto br = dyn_cast<cf::BranchOp>(term)) {
    if (failed(takeSuccessor(br.getDest(), br.getDestOperands(), true)))
      return failure();
  } else if (auto condBr = dyn_cast<cf::CondBranchOp>(term)) {
    if (failed(takeSuccessor(condBr.getTrueDest(), condBr.getTrueDestOperands(),
                             true)))
      return failure();
    if (failed(takeSuccessor(condBr.getFalseDest(),
                             condBr.getFalseDestOperands(), false)))
      return failure();
  } else {
    return failure();
  }
  if (succs.empty()) {
    return failure();
  }
  if (succs.size() == 1 || succs[0].value == succs[1].value)
    return sampleMemo[key] = succs[0].value;
  // Diamond (established by the arity of cf.cond_br): the shared
  // predecessor's condition decides.
  auto condBr = cast<cf::CondBranchOp>(term);
  auto cond = evalValue(condBr.getCondition());
  if (failed(cond)) {
    return failure();
  }
  Value trueVal = succs[0].value;
  Value falseVal = succs[1].value;
  auto mux = comb::MuxOp::create(builder, loc, *cond, trueVal, falseVal);
  created.push_back(mux);
  return sampleMemo[key] = mux;
}

FailureOr<Value> SyncLowering::evalValue(Value v) {
  auto it = memo.find(v);
  if (it != memo.end())
    return it->second;
  // Module-level values (ports, outer constants, lowered flop outputs) are
  // opaque inputs usable as-is.
  if (auto blockArg = dyn_cast<BlockArgument>(v)) {
    if (blockArg.getOwner()->getParentOp() != process)
      return memo[v] = v;
    // In-process block arguments go through the merge folding (wait-block
    // arguments are state slots; anything else must be a sample merge).
    auto lowered = evalBlockArg(blockArg.getOwner(), blockArg.getArgNumber());
    if (failed(lowered))
      return failure();
    return memo[v] = *lowered;
  }
  // Process results read back their flop output (past state). Module-level
  // values (outer constants, comb logic, instance outputs, already-lowered
  // flop outputs) are opaque inputs usable as-is. The defining op stays
  // live: if another process later lowers, its own result-rewiring updates
  // references inside our data trees, and vice versa, so cross-process
  // reads are order-independent. In-region values fall through to the
  // probe/clone handling below.
  if (auto res = dyn_cast<OpResult>(v)) {
    if (res.getOwner() == process) {
      auto flopIt = resultFlops.find(res.getResultNumber());
      if (flopIt == resultFlops.end()) {
        return failure();
      }
      return memo[v] = flopIt->second.getData();
    }
    if (res.getOwner()->getParentOp() != process)
      // Module-level values are opaque inputs usable as-is, including
      // probes of input wires and not-yet-lowered state: if another
      // process later lowers its signal, its own probe-rewiring updates
      // references inside our data trees.
      return memo[v] = v;
  }
  Operation *defOp = v.getDefiningOp();
  if (!defOp)
    return failure();
  // In-region probes of adopted signals read back the flop output.
  // Anything else probes a live module-level signal and is cloned like
  // other combinational reads (another process's later lowering rewires
  // the clone to its flop output).
  if (auto prb = dyn_cast<llhd::ProbeOp>(defOp)) {
    auto lowered = resolveProbe(prb);
    if (succeeded(lowered))
      return memo[v] = *lowered;
  }
  // In-region combinational logic is cloned at the ambient insertion point
  // (module level, established by the caller); anything else bails out.
  // The structural scan guaranteed only allowed ops occur.
  if (defOp->getParentOp() != process || !isAllowedDataOp(defOp)) {
    return failure();
  }
  SmallVector<Value> operands;
  for (auto operand : defOp->getOperands()) {
    auto lowered = evalValue(operand);
    if (failed(lowered)) {
      return failure();
    }
    operands.push_back(*lowered);
  }
  Operation *cloned = builder.clone(*defOp);
  for (auto [i, lowered] : llvm::enumerate(operands))
    cloned->setOperand(i, lowered);
  created.push_back(cloned);
  return memo[v] = cloned->getResult(0);
}

FailureOr<Value> SyncLowering::evalBlockArg(Block *block, unsigned idx) {
  // Wait-block state slots are established by the yield check. The past
  // clock slot (variant a arg 0) and the resume argument have no lowered
  // counterpart (no past-clock flop is built; the edge gate is proven
  // structurally but not materialized), so uses of them bail out.
  if (block == waitBlock) {
    if (idx < pastShift || idx >= pastShift + numResults) {
      return failure();
    }
    auto flopIt = resultFlops.find(idx - pastShift);
    if (flopIt == resultFlops.end()) {
      return failure();
    }
    return flopIt->second.getData();
  }
  if (block == resumeBlock) {
    return failure();
  }
  if (!sampleBlocks.count(block)) {
    return failure();
  }
  if (idx >= block->getNumArguments()) {
    return failure();
  }
  if (!argInProgress.insert({block, idx}).second) {
    return failure(); // Use-def cycle through block arguments.
  }
  struct EraseOnExit {
    DenseSet<std::pair<Block *, unsigned>> &set;
    std::pair<Block *, unsigned> key;
    ~EraseOnExit() { set.erase(key); }
  } guard{argInProgress, {block, idx}};
  // Mux over predecessor edges. All preds must be in the sample region.
  // Sound merges only: a single edge, or exactly two edges from opposite
  // sides of one cond_br (diamond). Anything else would need path
  // predicates beyond local edge conditions.
  struct InEdge {
    Block *pred;
    bool takenOnTrue; // meaningful for cond_br preds
    Value operand;
  };
  SmallVector<InEdge> incoming;
  // Predecessor iteration yields per-edge duplicates (a cond_br targeting
  // this block on both sides appears twice); same-side duplicates carry
  // identical operands and are folded into one.
  DenseSet<std::pair<Block *, unsigned>> seenEdges;
  for (auto *pred : block->getPredecessors()) {
    if (!sampleBlocks.count(pred)) {
      return failure();
    }
    auto *term = pred->getTerminator();
    if (auto br = dyn_cast<cf::BranchOp>(term)) {
      if (br.getDest() != block)
        return failure();
      if (idx >= br.getDestOperands().size())
        return failure();
      if (seenEdges.insert({pred, 1}).second)
        incoming.push_back({pred, true, br.getDestOperands()[idx]});
    } else if (auto condBr = dyn_cast<cf::CondBranchOp>(term)) {
      if (condBr.getTrueDest() == block) {
        if (idx >= condBr.getTrueDestOperands().size())
          return failure();
        if (seenEdges.insert({pred, 1}).second)
          incoming.push_back({pred, true, condBr.getTrueDestOperands()[idx]});
      }
      if (condBr.getFalseDest() == block) {
        if (idx >= condBr.getFalseDestOperands().size())
          return failure();
        if (seenEdges.insert({pred, 0}).second)
          incoming.push_back({pred, false, condBr.getFalseDestOperands()[idx]});
      }
    } else {
      return failure();
    }
  }
  if (incoming.empty()) {
    return failure();
  }
  // Cascade over the incoming edges. Exactly one predecessor edge is
  // taken per execution (deterministic CFG; loops are rejected), so
  // selecting by per-edge reachability is sound: the taken edge's
  // operand is valid, and some edge is always taken (exhaustiveness).
  // The last edge is the fallthrough (no selector needed).
  Value acc;
  for (int i = (int)incoming.size() - 1; i >= 0; --i) {
    auto &edge = incoming[i];
    auto val = evalValue(edge.operand);
    if (failed(val)) {
      return failure();
    }
    if (i == (int)incoming.size() - 1) {
      acc = *val;
      continue;
    }
    auto sel = edgeSelector(edge.pred, edge.takenOnTrue);
    if (failed(sel)) {
      return failure();
    }
    auto mux = comb::MuxOp::create(builder, loc, *sel, *val, acc);
    created.push_back(mux);
    acc = mux.getResult();
  }
  return acc;
}

// Edge condition: under which condition is the edge from `pred`'s
// `takenOnTrue` side taken, given `pred` executes.
FailureOr<Value> SyncLowering::edgeCond(Block *pred, bool takenOnTrue) {
  auto *term = pred->getTerminator();
  if (isa<cf::BranchOp>(term))
    return trueConst();
  auto condBr = dyn_cast<cf::CondBranchOp>(term);
  if (!condBr)
    return failure();
  auto cond = evalValue(condBr.getCondition());
  if (failed(cond))
    return failure();
  if (takenOnTrue)
    return *cond;
  auto one = hw::ConstantOp::create(builder, loc, builder.getI1Type(), 1);
  created.push_back(one);
  auto neg = comb::XorOp::create(builder, loc, *cond, one);
  created.push_back(neg);
  return neg.getResult();
}

Value SyncLowering::trueConst() {
  auto one = hw::ConstantOp::create(builder, loc, builder.getI1Type(), 1);
  created.push_back(one);
  return one.getResult();
}

bool SyncLowering::isTrueConst(Value v) {
  if (auto constOp = v.getDefiningOp<hw::ConstantOp>()) {
    if (auto intTy = dyn_cast<IntegerType>(constOp.getType()))
      return intTy.getWidth() == 1 && constOp.getValue() == 1;
  }
  return false;
}

// Selector for reaching `block` through `pred`'s side: the predecessor
// must execute (its own reachability) and take this side.
FailureOr<Value> SyncLowering::edgeSelector(Block *pred, bool takenOnTrue) {
  auto path = evalPath(pred);
  if (failed(path))
    return failure();
  auto cond = edgeCond(pred, takenOnTrue);
  if (failed(cond))
    return failure();
  if (isTrueConst(*path))
    return *cond;
  if (isTrueConst(*cond))
    return *path;
  auto both = comb::AndOp::create(builder, loc, *path, *cond);
  created.push_back(both);
  return both.getResult();
}

FailureOr<Value> SyncLowering::evalPath(Block *block) {
  auto it = pathMemo.find(block);
  if (it != pathMemo.end())
    return it->second;
  if (block == sampleBlock)
    return pathMemo[block] = trueConst();
  if (!sampleBlocks.count(block) || !pathInProgress.insert(block).second)
    return failure();
  struct EraseOnExit {
    DenseSet<Block *> &set;
    Block *key;
    ~EraseOnExit() { set.erase(key); }
  } guard{pathInProgress, block};
  // Disjunction over sample predecessors of (predecessor reachability
  // AND side taken). Non-sample predecessors cannot occur (entry feeds
  // only the wait block; resume feeds only the sample entry, which is
  // the base case above).
  SmallVector<Value> selectors;
  for (auto *pred : block->getPredecessors()) {
    if (!sampleBlocks.count(pred))
      return failure();
    auto *term = pred->getTerminator();
    if (auto br = dyn_cast<cf::BranchOp>(term)) {
      if (br.getDest() != block)
        return failure();
      auto sel = edgeSelector(pred, true);
      if (failed(sel))
        return failure();
      selectors.push_back(*sel);
    } else if (auto condBr = dyn_cast<cf::CondBranchOp>(term)) {
      if (condBr.getTrueDest() == block) {
        auto sel = edgeSelector(pred, true);
        if (failed(sel))
          return failure();
        selectors.push_back(*sel);
      }
      if (condBr.getFalseDest() == block) {
        auto sel = edgeSelector(pred, false);
        if (failed(sel))
          return failure();
        selectors.push_back(*sel);
      }
    } else {
      return failure();
    }
  }
  if (selectors.empty())
    return failure();
  if (selectors.size() == 1)
    return pathMemo[block] = selectors[0];
  auto either = comb::OrOp::create(builder, loc, ValueRange(selectors),
                                   /*twoState=*/false);
  created.push_back(either);
  return pathMemo[block] = either.getResult();
}

LogicalResult SyncLowering::buildAndRewrite() {
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPoint(process);
  loc = process.getLoc();

  // Create one flop per adopted result up front so feedback references have
  // something to point at (the same reason seq.firreg feedback is legal).
  // Results with no uses at all are dead state and get no flop.
  DenseSet<unsigned> liveResults;
  for (auto &adoptedSig : adopted)
    liveResults.insert(adoptedSig.resultIdx);
  // Results used as drive enables or downstream are live too (their uses
  // rewire to flop outputs below).
  for (unsigned i = 0; i < numResults; ++i) {
    Value result = process->getResult(i);
    if (!result.use_empty())
      liveResults.insert(i);
  }
  auto seqClock = seq::ToClockOp::create(builder, loc, clockVal);
  created.push_back(seqClock);
  for (unsigned idx : liveResults) {
    Type resultTy = process->getResult(idx).getType();
    auto intTy = dyn_cast<IntegerType>(resultTy);
    if (!intTy)
      return skip("non-integer register");
    auto zero = hw::ConstantOp::create(builder, loc, resultTy,
                                       builder.getIntegerAttr(resultTy, 0));
    created.push_back(zero);
    auto flop = seq::FirRegOp::create(
        builder, loc, zero, seqClock, builder.getStringAttr(""),
        hw::InnerSymAttr{}, IntegerAttr{}, Value(), Value(), false);
    created.push_back(flop);
    resultFlops[idx] = flop;
  }

  // Next-state data per live result. `evalSample` evaluates the sample
  // region forward from its entry, following taken branches down to the
  // wait-block completion operands, so nested conditionals (completions
  // from multiple blocks) fold into nested muxes. For driven signals the
  // enable position selects between the driven value and hold: bypass
  // paths carry constant-false enables (HoistSignals convention), so
  // their dummy value operands are never selected. Plain `seq.firreg`
  // holds past state by itself on wakes without a rising edge, so no gate
  // mux is needed (flop-for-flop for LEC).
  //
  // Completion operand positions shift past the past-clock slot (variant
  // a): wait-block arg[pastShift+i] carries result[i].
  for (unsigned idx : liveResults) {
    auto valPos = evalSample(sampleBlock, pastShift + idx);
    if (failed(valPos)) {
      return failure();
    }
    Value acc = *valPos;
    for (auto &adoptedSig : adopted) {
      if (adoptedSig.resultIdx != idx)
        continue;
      if (adoptedSig.enableIdx >= 0) {
        auto enPos =
            evalSample(sampleBlock, pastShift + (unsigned)adoptedSig.enableIdx);
        if (failed(enPos))
          return failure();
        auto upd = comb::MuxOp::create(builder, loc, *enPos, acc,
                                       resultFlops[idx].getData());
        created.push_back(upd);
        acc = upd;
      }
      break;
    }
    resultFlops[idx].getNextMutable().assign(acc);
  }

  // Rewire: drives and sigs go away, probes and downstream uses see flops.
  // Uses inside our own region vanish with it. Every use of an adopted
  // signal (probed reads anywhere in the module, direct value uses) sees
  // the flop output; only the adopted drives keep referencing the signal
  // until they are erased below.
  for (auto &adoptedSig : adopted) {
    Value sigVal = adoptedSig.sigOp.getResult();
    Value q = resultFlops[adoptedSig.resultIdx].getData();
    DenseSet<Operation *> driveOps;
    for (auto drive : adoptedSig.drives)
      driveOps.insert(drive.getOperation());
    // Direct value uses of the signal see the flop output. Probe signal
    // operands are handled below (the probe goes away, its value uses see
    // the flop). Drive signal operands keep referencing the signal until
    // the drives are erased.
    SmallVector<OpOperand *> uses;
    SmallVector<llhd::ProbeOp> probes;
    for (auto &use : sigVal.getUses()) {
      Operation *owner = use.getOwner();
      if (driveOps.contains(owner))
        continue;
      if (auto prb = dyn_cast<llhd::ProbeOp>(owner)) {
        prb.getResult().replaceAllUsesWith(q);
        probes.push_back(prb);
        continue;
      }
      uses.push_back(&use);
    }
    for (auto *use : uses)
      use->getOwner()->setOperand(use->getOperandNumber(), q);
    for (auto prb : probes)
      prb.erase();
    for (auto drive : adoptedSig.drives)
      drive.erase();
    adoptedSig.sigOp.erase();
  }
  // Results used downstream (outside drives, which are gone) see flops.
  // In-region uses vanish with the process.
  for (unsigned i = 0; i < numResults; ++i) {
    Value result = process->getResult(i);
    auto it = resultFlops.find(i);
    if (it == resultFlops.end())
      continue; // Dead result: no flop was built.
    Value q = it->second.getData();
    // Collect out-of-region uses first (erasing while iterating is unsafe).
    SmallVector<OpOperand *> uses;
    for (auto &use : result.getUses()) {
      if (use.getOwner()->getParentOp() == process)
        continue;
      uses.push_back(&use);
    }
    for (auto *use : uses)
      use->getOwner()->setOperand(use->getOperandNumber(), q);
  }
  process.erase();
  created.clear();
  return success();
}

struct LowerSyncProcessesPass
    : public llhd::impl::LowerSyncProcessesPassBase<LowerSyncProcessesPass> {
  using LowerSyncProcessesPassBase<
      LowerSyncProcessesPass>::LowerSyncProcessesPassBase;

  void runOnOperation() override {
    auto module = getOperation();
    bool changed = true;
    // Fixpoint: a process may read state lowered from another process in an
    // earlier iteration. Lowering is monotonic (lowered stays lowered).
    unsigned budget = 0;
    module.walk([&](llhd::ProcessOp op) {
      ++budget;
      (void)op;
    });
    for (unsigned iter = 0; changed && iter <= budget; ++iter) {
      changed = false;
      SmallVector<llhd::ProcessOp> processes;
      module.getBodyBlock()->walk(
          [&](llhd::ProcessOp op) { processes.push_back(op); });
      for (auto process : processes) {
        SyncLowering lowering(module, process);
        if (succeeded(lowering.tryLower())) {
          ++numLowered;
          changed = true;
        } else {
          ++numSkipped;
        }
      }
    }
  }
};

} // namespace
