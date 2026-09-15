//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the SeqMemMapper pass, which maps `seq.firmem`
// memories to instances of SRAM macro cells, the standard memory mapper
// equivalent for the macro-mapping part.
//
// SRAM macros are `hw.module.extern` ops carrying a `synth.sram.macro`
// descriptor. Only exact matches map: equal width and depth, equal read and
// write port counts in order, a single shared clock, equal latencies,
// undefined read/write-under-write behavior, no power-on init, and no
// read-write ports. Single-port strobe macros needing read/write
// arbitration are future work. Without a macro match, memories fall back
// to flop arrays with address decode (the standard memory-mapper default).
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Dialect/Synth/SynthAttributes.h"
#include "circt/Dialect/Synth/SynthOps.h"
#include "circt/Dialect/Synth/Transforms/SynthPasses.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

namespace circt {
namespace synth {
#define GEN_PASS_DEF_SEQMEMMAPPER
#include "circt/Dialect/Synth/Transforms/SynthPasses.h.inc"
} // namespace synth
} // namespace circt

using namespace circt;
using namespace circt::synth;
using namespace mlir;

#define DEBUG_TYPE "synth-seq-mem-mapper"

namespace {

// A macro port with pin names resolved to port indices is validated during
// extraction; this keeps the resolved widths for matching.
struct MacroPort {
  synth::SRAMPortAttr attr;
  bool isRead() const { return attr.getKind() == synth::SRAMPortKind::Read; }
};

struct MacroInfo {
  hw::HWModuleExternOp module;
  int64_t width;
  int64_t depth;
  SmallVector<synth::SRAMPortAttr> reads;
  SmallVector<synth::SRAMPortAttr> writes;
};

// Extract a macro descriptor, validating that every port is accounted for:
// inputs must be known role pins, outputs must be read-data pins.
FailureOr<MacroInfo> extractMacro(hw::HWModuleExternOp module) {
  MacroInfo info;
  info.module = module;
  auto macro = module->getAttrOfType<synth::SRAMMacroAttr>("synth.sram.macro");
  if (!macro)
    return failure();
  info.width = macro.getWidth();
  info.depth = macro.getDepth();
  if (info.width <= 0 || info.depth <= 0)
    return failure();
  llvm::DenseSet<StringAttr> rolePins;
  for (auto portAttr : macro.getPorts()) {
    auto port = dyn_cast<synth::SRAMPortAttr>(portAttr);
    if (!port)
      return failure();
    if (port.getAddrWidth() <= 0 || port.getDataWidth() != info.width)
      return failure();
    if (port.getMaskWidth() < 0)
      return failure();
    auto note = [&](StringAttr pin) {
      if (!pin || pin.getValue().empty())
        return;
      rolePins.insert(pin);
    };
    note(port.getClock());
    note(port.getEnable());
    note(port.getAddress());
    note(port.getData());
    note(port.getMask());
    if (port.getKind() == synth::SRAMPortKind::Read)
      info.reads.push_back(port);
    else
      info.writes.push_back(port);
  }
  // Every port of the extern must play a described role.
  for (const auto &port : module.getPortList()) {
    if (port.isInput()) {
      if (!rolePins.count(port.name))
        return failure();
      continue;
    }
    // Outputs must be read-data pins.
    bool isReadData = false;
    for (auto read : info.reads)
      if (read.getData() == port.name)
        isReadData = true;
    if (!isReadData)
      return failure();
  }
  return info;
}

struct MemPorts {
  seq::FirMemOp mem;
  SmallVector<seq::FirMemReadOp> reads;
  SmallVector<seq::FirMemWriteOp> writes;
  bool hasReadWrite = false;
};

Value clockToI1(OpBuilder &builder, Location loc, Value clock) {
  if (auto toClock = clock.getDefiningOp<seq::ToClockOp>())
    return toClock.getInput();
  return seq::FromClockOp::create(builder, loc, clock);
}

struct SeqMemMapperPass
    : public circt::synth::impl::SeqMemMapperBase<SeqMemMapperPass> {
  using SeqMemMapperBase<SeqMemMapperPass>::SeqMemMapperBase;

  void runOnOperation() override {
    auto top = getOperation();

    SmallVector<MacroInfo> lib;
    for (auto externOp : top.getOps<hw::HWModuleExternOp>()) {
      auto info = extractMacro(externOp);
      if (failed(info))
        continue;
      lib.push_back(std::move(*info));
    }
    // Collect memories with their ports (flop-array fallback needs no
    // macros, so an empty library only skips macro matching).
    SmallVector<MemPorts> memories;
    DenseMap<Value, size_t> memIndex;
    for (auto module : top.getOps<hw::HWModuleOp>()) {
      module.walk([&](seq::FirMemOp mem) {
        memIndex[mem.getResult()] = memories.size();
        memories.push_back({mem, {}, {}});
      });
    }
    if (memories.empty())
      return markAllAnalysesPreserved();
    for (auto module : top.getOps<hw::HWModuleOp>()) {
      module.walk([&](Operation *op) {
        if (auto read = dyn_cast<seq::FirMemReadOp>(op)) {
          auto it = memIndex.find(read.getMemory());
          if (it != memIndex.end())
            memories[it->second].reads.push_back(read);
        } else if (auto write = dyn_cast<seq::FirMemWriteOp>(op)) {
          auto it = memIndex.find(write.getMemory());
          if (it != memIndex.end())
            memories[it->second].writes.push_back(write);
        } else if (auto rw = dyn_cast<seq::FirMemReadWriteOp>(op)) {
          auto it = memIndex.find(rw.getMemory());
          if (it != memIndex.end())
            memories[it->second].hasReadWrite = true;
        }
      });
    }

    OpBuilder builder(top.getContext());
    for (auto &entry : memories)
      mapMemory(builder, entry, lib);
  }

  // Clock identity key: through `to_clock` to the i1 signal when possible.
  Value clockKey(Value clock) {
    if (auto toClock = clock.getDefiningOp<seq::ToClockOp>())
      return toClock.getInput();
    return clock;
  }

  void mapMemory(OpBuilder &builder, MemPorts &entry,
                 SmallVectorImpl<MacroInfo> &lib) {
    auto mem = entry.mem;
    Location loc = mem.getLoc();
    auto memType = cast<seq::FirMemType>(mem.getResult().getType());

    auto skip = [&] { ++numMemsSkipped; };
    // Preconditions: no init, undefined collision behavior, sane geometry.
    if (mem.getInitAttr())
      return skip();
    if (mem.getRuw() != seq::RUW::Undefined ||
        mem.getWuw() != seq::WUW::Undefined)
      return skip();
    // Read-write ports have no macro counterpart yet.
    if (entry.hasReadWrite)
      return skip();

    // All ports share one clock (compared through `to_clock`).
    Value clock;
    Value clockKey0;
    auto noteClock = [&](Value clk) -> bool {
      Value key = clockKey(clk);
      if (!clock) {
        clock = clk;
        clockKey0 = key;
        return true;
      }
      return clockKey0 == key;
    };
    for (auto read : entry.reads)
      if (!noteClock(read.getClk()))
        return skip();
    for (auto write : entry.writes)
      if (!noteClock(write.getClk()))
        return skip();
    if (!clock)
      return skip();

    // Find a compatible macro: geometry, port counts, latencies; masks and
    // enables checked per port below.
    MacroInfo *best = nullptr;
    for (auto &macro : lib) {
      if (macro.width != static_cast<int64_t>(memType.getWidth()) ||
          macro.depth != static_cast<int64_t>(memType.getDepth()))
        continue;
      if (macro.reads.size() != entry.reads.size() ||
          macro.writes.size() != entry.writes.size())
        continue;
      bool latenciesOk = true;
      for (auto read : macro.reads)
        latenciesOk &= read.getLatency() == mem.getReadLatency();
      for (auto write : macro.writes)
        latenciesOk &= write.getLatency() == mem.getWriteLatency();
      if (!latenciesOk)
        continue;
      if (!best || macro.module.getModuleName() < best->module.getModuleName())
        best = &macro;
    }
    if (!best) {
      // No macro match: flop-array fallback builds depth*width flops
      // plus per-bit address-decode muxes, which downstream lowering and
      // mapping cannot swallow at scale (a 16384x64 SRAM becomes ~1M
      // flops and ~2M muxes and hangs the flow). Fail fast above
      // `max-fallback-bits` instead of exploding silently; provide a
      // matching `synth.sram.macro` extern or raise the limit explicitly.
      uint64_t depth = memType.getDepth();
      uint64_t width = memType.getWidth();
      if (width != 0 && depth > maxFallbackBits / width) {
        unsigned __int128 numBits = (unsigned __int128)depth * width;
        mem.emitError("memory '")
            << depth << "x" << width << "' (" << static_cast<uint64_t>(numBits)
            << " bits) has no matching SRAM macro and exceeds "
               "max-fallback-bits ("
            << maxFallbackBits.getValue()
            << "): refusing flop-array fallback; provide a matching "
               "#synth.sram.macro extern or raise max-fallback-bits";
        signalPassFailure();
        return;
      }
      return explodeToFlops(builder, entry, clock, memType);
    }

    // Per-port feature checks.
    for (auto [read, macro] : llvm::zip(entry.reads, best->reads)) {
      if (read.getEnable() && macro.getEnable().empty())
        return skip(); // Held output needs a register; out of scope.
      if (static_cast<int64_t>(
              cast<IntegerType>(read.getAddress().getType()).getWidth()) !=
          macro.getAddrWidth())
        return skip();
    }
    for (auto [write, macro] : llvm::zip(entry.writes, best->writes)) {
      if (write.getEnable() && macro.getEnable().empty())
        return skip();
      if (static_cast<int64_t>(
              cast<IntegerType>(write.getAddress().getType()).getWidth()) !=
          macro.getAddrWidth())
        return skip();
      if (static_cast<int64_t>(
              cast<IntegerType>(write.getData().getType()).getWidth()) !=
          macro.getDataWidth())
        return skip();
      if (write.getMask()) {
        // Exact mask width required; no read-modify-write emulation.
        if (static_cast<int64_t>(
                cast<IntegerType>(write.getMask().getType()).getWidth()) !=
            macro.getMaskWidth())
          return skip();
      }
    }

    // Wire the instance in input-port order.
    builder.setInsertionPoint(mem);
    Value clkI1 = clockToI1(builder, loc, clock);
    auto constInt = [&](Type type, uint64_t v) -> Value {
      return hw::ConstantOp::create(
          builder, loc, type,
          builder.getIntegerAttr(type,
                                 APInt(cast<IntegerType>(type).getWidth(), v)));
    };
    DenseMap<StringAttr, Value> pinValues;
    auto tieOne = [&](Type type) -> Value {
      return hw::ConstantOp::create(
          builder, loc, type,
          builder.getIntegerAttr(
              type, APInt::getAllOnes(cast<IntegerType>(type).getWidth())));
    };
    for (auto [read, macro] : llvm::zip(entry.reads, best->reads)) {
      pinValues[macro.getClock()] = clkI1;
      pinValues[macro.getAddress()] = read.getAddress();
      if (!macro.getEnable().empty())
        pinValues[macro.getEnable()] =
            read.getEnable()
                ? read.getEnable()
                : constInt(IntegerType::get(builder.getContext(), 1), 1);
    }
    for (auto [write, macro] : llvm::zip(entry.writes, best->writes)) {
      pinValues[macro.getClock()] = clkI1;
      pinValues[macro.getAddress()] = write.getAddress();
      pinValues[macro.getData()] = write.getData();
      if (!macro.getEnable().empty())
        pinValues[macro.getEnable()] =
            write.getEnable()
                ? write.getEnable()
                : constInt(IntegerType::get(builder.getContext(), 1), 1);
      if (!macro.getMask().empty()) {
        if (write.getMask()) {
          pinValues[macro.getMask()] = write.getMask();
        } else {
          auto maskTy =
              IntegerType::get(builder.getContext(), macro.getMaskWidth());
          pinValues[macro.getMask()] = tieOne(maskTy);
        }
      }
    }
    SmallVector<Value> inputs;
    for (const auto &port : best->module.getPortList()) {
      if (!port.isInput())
        continue;
      auto it = pinValues.find(port.name);
      if (it == pinValues.end())
        return skip(); // Unreachable: extraction validated roles.
      inputs.push_back(it->second);
    }

    StringAttr instName = builder.getStringAttr("mem_mapped");
    if (auto name = mem.getNameAttr(); name && !name.getValue().empty())
      instName = name;
    auto inst =
        hw::InstanceOp::create(builder, loc, best->module, instName, inputs,
                               builder.getArrayAttr({}), mem.getInnerSymAttr());
    // Results follow output ports in order; every output is a read-data pin.
    unsigned resultIdx = 0;
    for (const auto &port : best->module.getPortList()) {
      if (!port.isOutput())
        continue;
      for (auto [read, macro] : llvm::zip(entry.reads, best->reads)) {
        if (macro.getData() == port.name)
          read.getData().replaceAllUsesWith(inst.getResult(resultIdx));
      }
      ++resultIdx;
    }
    for (auto read : entry.reads)
      read.erase();
    for (auto write : entry.writes)
      write.erase();
    mem.erase();
    ++numMemsMapped;
  }

  // Standard memory-mapper default: with no SRAM macro available, memories
  // become flop arrays with address-decode muxes. Strict preconditions
  // beyond mapMemory's: asynchronous read (latency 0; a synchronous
  // read would need a held-output register), single-cycle write, at
  // most one write port, and no read enables. Masks are supported when
  // lanes are equal-width and aligned (word width divisible by mask
  // width); each lane bit gates its lane's flops like a per-bit write
  // enable.
  void explodeToFlops(OpBuilder &builder, MemPorts &entry, Value clock,
                      seq::FirMemType memType) {
    auto mem = entry.mem;
    Location loc = mem.getLoc();
    auto skip = [&] { ++numMemsSkipped; };
    uint64_t depth = memType.getDepth();
    unsigned width = memType.getWidth();
    if (depth == 0 || width == 0)
      return skip();
    if (mem.getReadLatency() != 0 || mem.getWriteLatency() != 1)
      return skip();
    auto maskWidth = memType.getMaskWidth();
    // Masked words need equal-width aligned lanes; a present mask must
    // cover exactly the declared mask width. An absent mask writes the
    // full word.
    unsigned laneWidth = width;
    if (maskWidth && *maskWidth != 1) {
      if (width % *maskWidth != 0)
        return skip();
      laneWidth = width / *maskWidth;
      for (auto write : entry.writes) {
        if (!write.getMask())
          continue;
        auto maskTy = dyn_cast<IntegerType>(write.getMask().getType());
        if (!maskTy || maskTy.getWidth() != *maskWidth)
          return skip();
      }
    } else {
      for (auto write : entry.writes)
        if (write.getMask())
          return skip();
    }
    if (entry.writes.size() > 1)
      return skip();
    for (auto read : entry.reads)
      if (read.getEnable())
        return skip();

    // Dead memory: no readers, nothing to build.
    if (entry.reads.empty()) {
      for (auto write : entry.writes)
        write.erase();
      mem.erase();
      ++numMemsExploded;
      return;
    }

    auto i1 = builder.getI1Type();
    auto constInt = [&](Type type, uint64_t v) -> Value {
      return hw::ConstantOp::create(
          builder, loc, type,
          builder.getIntegerAttr(type,
                                 APInt(cast<IntegerType>(type).getWidth(), v)));
    };
    std::string baseName = "mem";
    if (auto name = mem.getNameAttr(); name && !name.getValue().empty())
      baseName = name.getValue().str();

    // One flop per bit, built at the memory: the memory dominates its
    // ports, so every flop output dominates every port position.
    builder.setInsertionPoint(mem);
    Value zero = constInt(i1, 0);
    seq::FirMemWriteOp write =
        entry.writes.empty() ? seq::FirMemWriteOp() : entry.writes[0];
    Value wren =
        write && write.getEnable() ? write.getEnable() : constInt(i1, 1);
    // Storage[row][bit].
    SmallVector<SmallVector<seq::FirRegOp>> storage;
    for (uint64_t r = 0; r < depth; ++r) {
      SmallVector<seq::FirRegOp> row;
      for (unsigned b = 0; b < width; ++b) {
        auto flop = seq::FirRegOp::create(
            builder, loc, zero, clock,
            builder.getStringAttr(baseName + "_r" + Twine(r).str() + "_b" +
                                  Twine(b).str()));
        row.push_back(flop);
      }
      storage.push_back(std::move(row));
    }
    // Write path, built at the write port where every operand dominates.
    // With a mask, each lane bit gates its lane's flops.
    if (write) {
      builder.setInsertionPoint(write);
      Value wdata = write.getData();
      Value waddr = write.getAddress();
      auto waddrTy = cast<IntegerType>(waddr.getType());
      Value wmask = write.getMask();
      for (uint64_t r = 0; r < depth; ++r) {
        Value match = comb::ICmpOp::create(
            builder, loc, comb::ICmpPredicate::eq, waddr, constInt(waddrTy, r));
        Value rowFire = comb::AndOp::create(builder, loc, wren, match);
        for (unsigned b = 0; b < width; ++b) {
          Value fire = rowFire;
          if (wmask) {
            Value laneBit =
                comb::ExtractOp::create(builder, loc, wmask, b / laneWidth, 1);
            fire = comb::AndOp::create(builder, loc, rowFire, laneBit);
          }
          Value dataBit = comb::ExtractOp::create(builder, loc, wdata, b, 1);
          Value next = comb::MuxOp::create(builder, loc, fire, dataBit,
                                           storage[r][b].getData());
          storage[r][b].getNextMutable().assign(next);
        }
      }
    }
    // Read path: per-bit mux cascade over rows, built at each read port.
    // Out-of-range addresses match nothing and read row 0 (don't-care,
    // deterministic, mirroring standard undefined-address behavior).
    for (auto read : entry.reads) {
      builder.setInsertionPoint(read);
      Value raddr = read.getAddress();
      auto raddrTy = cast<IntegerType>(raddr.getType());
      SmallVector<Value> bits;
      for (unsigned b = 0; b < width; ++b) {
        Value acc = storage[0][b].getData();
        for (uint64_t r = 1; r < depth; ++r) {
          Value match =
              comb::ICmpOp::create(builder, loc, comb::ICmpPredicate::eq, raddr,
                                   constInt(raddrTy, r));
          acc = comb::MuxOp::create(builder, loc, match,
                                    storage[r][b].getData(), acc);
        }
        bits.push_back(acc);
      }
      SmallVector<Value> reversed(bits.rbegin(), bits.rend());
      Value data = comb::ConcatOp::create(builder, loc, reversed);
      read.getData().replaceAllUsesWith(data);
      read.erase();
    }
    for (auto writeOp : entry.writes)
      writeOp.erase();
    mem.erase();
    ++numMemsExploded;
  }
};

} // namespace
