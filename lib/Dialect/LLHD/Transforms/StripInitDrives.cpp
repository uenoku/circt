//===- StripInitDrives.cpp - Drop subsumed constant drives ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Drops `llhd.drv` ops with constant values that only specify a signal's
// power-up value when the signal also has a real (non-constant) driver.
//
// Verilog `initial reg = const` lowers to a module-level constant drive next
// to the driver from the `always` block. The two unconditional drives look
// like a multi-driver conflict to Sig2Reg, so the signal gets stuck. For
// synthesis the constant only selects power-up state, which ASIC cells
// cannot realize (the reference flow likewise drops init values at techmap when
// the cell has no init pin), so the drive can go once a real driver exists.
// This changes simulation semantics and therefore only runs in synthesis
// flows (gated by flag, like strip-assert-like).
//
// Strictness: every drive to the signal must be unconditional, carry a
// known constant-time delay, and target the full signal directly in the
// module block. Constant drives with differing values are a genuine
// conflict and bail out. All-constant identical drives are deduplicated
// to one so Sig2Reg can fold the tie.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/LLHD/LLHDOps.h"
#include "circt/Dialect/LLHD/LLHDPasses.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "llhd-strip-init-drives"

namespace circt {
namespace llhd {
#define GEN_PASS_DEF_STRIPINITDRIVESPASS
#include "circt/Dialect/LLHD/LLHDPasses.h.inc"
} // namespace llhd
} // namespace circt

using namespace mlir;
using namespace circt;

namespace {

struct StripInitDrivesPass
    : public llhd::impl::StripInitDrivesPassBase<StripInitDrivesPass> {
  void runOnOperation() override {
    hw::HWModuleOp module = getOperation();
    Block *body = module.getBodyBlock();

    // Map each signal to its module-level drives.
    DenseMap<Value, SmallVector<llhd::DriveOp>> drivesBySig;
    for (auto &op : *body) {
      auto drive = dyn_cast<llhd::DriveOp>(op);
      if (!drive)
        continue;
      drivesBySig[drive.getSignal()].push_back(drive);
    }

    // Collect signals up front: erasing drives while iterating the
    // body would invalidate the iteration (erased drives may sit after
    // the current op).
    SmallVector<llhd::SignalOp> sigs;
    for (auto &op : *body)
      if (auto sig = dyn_cast<llhd::SignalOp>(op))
        sigs.push_back(sig);

    for (auto sig : sigs) {
      auto it = drivesBySig.find(sig.getResult());
      if (it == drivesBySig.end())
        continue;
      auto &drives = it->second;

      // Every user must be a direct drive or probe in the module block.
      // Slices, foreign blocks, or anything else bails the signal out.
      bool ok = true;
      for (auto *user : sig.getResult().getUsers()) {
        if (user->getBlock() != body ||
            (!isa<llhd::DriveOp>(user) && !isa<llhd::ProbeOp>(user))) {
          ok = false;
          break;
        }
      }
      if (!ok)
        continue;

      // All drives must be unconditional with known delays. Anything else
      // (conditional drives, unknown timing) bails the signal out.
      for (auto drive : drives) {
        Value time = drive.getTime();
        if (drive.getEnable() || !time ||
            !time.getDefiningOp<llhd::ConstantTimeOp>()) {
          ok = false;
          break;
        }
      }
      if (!ok)
        continue;

      // Partition into constant and real drivers. Only integer signals;
      // aggregates keep today's behavior.
      SmallVector<llhd::DriveOp> constDrives;
      bool hasRealDrive = false;
      IntegerAttr constValue;
      for (auto drive : drives) {
        auto constOp = drive.getValue().getDefiningOp<hw::ConstantOp>();
        if (!constOp || !isa<IntegerType>(constOp.getType())) {
          hasRealDrive = true;
          continue;
        }
        if (!constValue) {
          constValue = constOp.getValueAttr();
        } else if (constValue.getType() != constOp.getType() ||
                   constValue.getValue() != constOp.getValue()) {
          ok = false;
          break;
        }
        constDrives.push_back(drive);
      }
      if (!ok)
        continue;

      if (hasRealDrive) {
        // Power-up values are unrealizable once a real driver exists.
        for (auto drive : constDrives) {
          LLVM_DEBUG(llvm::dbgs() << "Erasing init drive: " << drive << "\n");
          drive.erase();
          ++numDrivesErased;
        }
      } else if (constDrives.size() > 1) {
        // Identical constants: dedupe to one so Sig2Reg can fold the tie.
        for (auto drive : llvm::drop_begin(constDrives)) {
          LLVM_DEBUG(llvm::dbgs() << "Deduping tie drive: " << drive << "\n");
          drive.erase();
          ++numDrivesErased;
        }
      }
    }
  }
};

} // namespace
