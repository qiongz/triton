//===- AttnFwdSchedule.cpp - attention-FORWARD LDS-read prefetch ----------===//
//
// Forward-safe variant of AttnBwdScheduling.cpp. Hoists MFMA operand LDS reads
// (llvm.amdgcn.ds.read* / addrspace(3) loads) EARLY to hide ds_read latency and
// cut s_waitcnt(lgkmcnt) stalls before the consuming MFMA -- the asm-verified gap
// (aiter ~0.12 waits/MFMA vs Triton ~0.97).
//
// WHY a separate pass: AttnBwdScheduling only tracks SSA deps, so on the forward
// kernel it hoists a ds_read ABOVE the async wait_group / buffer_load_to_shared
// that fills that LDS buffer (no SSA edge at the LLVM level -> the ordering is via
// s_waitcnt), reading STALE data -> wrong results. This pass adds a FENCE
// constraint: a ds_read is never reordered across a wait / barrier / LDS-write /
// inline-asm, so the async-write -> read ordering is preserved.
//
// Safety: dependency-preserving list schedule + fence barrier. A load is emitted
// only once (a) its in-block SSA operands are emitted AND (b) every fence that
// originally preceded it has been emitted. SSA dominance + LDS memory order hold.
//
// Enabled by TRITON_ENABLE_ATTN_FWD_SCHED=1. Strategy via TRITON_ATTN_FWD_MODE:
//   early (default) | lead<N> | even.
//===----------------------------------------------------------------------===//

#include "TritonAMDGPUToLLVM/Passes.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Pass.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <string>

#define DEBUG_TYPE "tritonamdgpu-attn-fwd-schedule"

namespace {
using namespace llvm;

enum class Mode { Early, Even, Lead };

static Mode getMode(unsigned &leadOut) {
  leadOut = 2;
  const char *e = std::getenv("TRITON_ATTN_FWD_MODE");
  if (!e)
    return Mode::Early;
  StringRef s(e);
  if (s == "even")
    return Mode::Even;
  if (s.starts_with("lead")) {
    int v = 0;
    if (!s.drop_front(4).getAsInteger(10, v) && v > 0)
      leadOut = (unsigned)v;
    return Mode::Lead;
  }
  return Mode::Early;
}

// MFMA operand load = ds_read / ds_load intrinsic, or any addrspace(3) load.
static bool isLDSRead(const Instruction &I) {
  if (const auto *CI = dyn_cast<CallInst>(&I)) {
    if (CI->isInlineAsm())
      return false;
    const Function *C = CI->getCalledFunction();
    if (C && C->isIntrinsic()) {
      StringRef N = C->getName();
      return N.contains("ds.read") || N.contains("ds.load");
    }
    return false;
  }
  if (const auto *LI = dyn_cast<LoadInst>(&I))
    return LI->getPointerAddressSpace() == 3;
  return false;
}

// A FENCE that a ds_read must not be hoisted across: any sync / LDS-write /
// inline-asm (s_waitcnt, s_barrier). Conservative: preserves async-DMA -> read.
static bool isFence(const Instruction &I) {
  if (const auto *CI = dyn_cast<CallInst>(&I)) {
    if (CI->isInlineAsm())
      return true; // s_waitcnt / s_barrier etc.
    const Function *C = CI->getCalledFunction();
    if (C && C->isIntrinsic()) {
      StringRef N = C->getName();
      if (N.contains("barrier") || N.contains("waitcnt") || N.contains("fence") ||
          N.contains("load.lds") || N.contains("ds.write") || N.contains("ds.store") ||
          N.contains("async") || N.contains("sched.barrier"))
        return true;
    }
    return false;
  }
  if (const auto *SI = dyn_cast<StoreInst>(&I))
    return SI->getPointerAddressSpace() == 3; // LDS write
  if (isa<FenceInst>(&I))
    return true;
  return false;
}

static Loop *findMainLoop(LoopInfo &LI) {
  Loop *main = nullptr;
  size_t best = 0;
  for (Loop *L : LI) {
    size_t n = 0;
    for (BasicBlock *BB : L->blocks())
      n += BB->size();
    if (n > best) { best = n; main = L; }
  }
  return main;
}

static bool scheduleBB(BasicBlock &BB) {
  SmallVector<Instruction *, 256> body;
  Instruction *term = BB.getTerminator();
  for (Instruction &I : BB) {
    if (isa<PHINode>(&I) || &I == term)
      continue;
    body.push_back(&I);
  }
  if (body.size() < 4)
    return false;

  unsigned lead = 2;
  const Mode mode = getMode(lead);
  SmallVector<Instruction *, 64> loads, others;
  SmallPtrSet<Instruction *, 32> inBody;
  // fence index per load = number of fences (in `others`) before it originally.
  SmallVector<unsigned, 64> loadFenceReq;
  unsigned fenceSeen = 0;
  for (Instruction *I : body) {
    inBody.insert(I);
    if (isLDSRead(*I)) {
      loads.push_back(I);
      loadFenceReq.push_back(fenceSeen);
    } else {
      if (isFence(*I))
        ++fenceSeen;
      others.push_back(I);
    }
  }
  if (loads.empty() || others.empty())
    return false;

  SmallPtrSet<Value *, 32> emitted;
  auto ready = [&](Instruction *I) {
    for (Value *op : I->operands()) {
      auto *opI = dyn_cast<Instruction>(op);
      if (opI && inBody.count(opI) && !emitted.count(opI))
        return false;
    }
    return true;
  };

  SmallVector<Instruction *, 256> order;
  order.reserve(body.size());
  size_t li = 0, oj = 0, emittedL = 0;
  unsigned fencesEmitted = 0;
  const size_t nL = loads.size(), nO = others.size();
  while (li < nL || oj < nO) {
    // a load is eligible only if SSA-ready AND its required fences are emitted
    bool rl = (li < nL) && ready(loads[li]) && (fencesEmitted >= loadFenceReq[li]);
    bool ro = (oj < nO) && ready(others[oj]);
    bool pickL;
    if (rl && ro) {
      switch (mode) {
      case Mode::Early: pickL = true; break;
      case Mode::Lead: {
        size_t desired = (oj * nL) / nO + lead;
        pickL = (emittedL < desired);
        break;
      }
      case Mode::Even:
      default: {
        size_t desired = (oj * nL) / nO;
        pickL = (emittedL < desired);
        break;
      }
      }
    } else {
      pickL = rl;
    }
    if (!rl && !ro)
      return false; // blocked (e.g. load waiting on a fence) -> abort, IR untouched
    if (pickL) {
      order.push_back(loads[li]);
      emitted.insert(loads[li]);
      ++li; ++emittedL;
    } else {
      Instruction *o = others[oj];
      if (isFence(*o))
        ++fencesEmitted;
      order.push_back(o);
      emitted.insert(o);
      ++oj;
    }
  }
  if (order.size() != body.size())
    return false;
  for (Instruction *I : order)
    I->moveBefore(term->getIterator());
  return true;
}

struct AttnFwdSchedulePass : FunctionPass {
  static char ID;
  AttnFwdSchedulePass() : FunctionPass(ID) {}
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<LoopInfoWrapperPass>();
  }
  bool runOnFunction(Function &F) override {
    if (F.isDeclaration())
      return false;
    auto &LI = getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
    if (LI.empty())
      return false;
    Loop *main = findMainLoop(LI);
    if (!main)
      return false;
    bool changed = false;
    for (BasicBlock *BB : main->blocks())
      changed |= scheduleBB(*BB);
    return changed;
  }
};
} // namespace

char AttnFwdSchedulePass::ID = 0;

namespace mlir::triton::AMD {
void runAttnFwdSchedulePass(llvm::Function &F, llvm::StringRef arch) {
  llvm::legacy::FunctionPassManager FPM(F.getParent());
  FPM.add(new AttnFwdSchedulePass());
  FPM.doInitialization();
  FPM.run(F);
  FPM.doFinalization();
  if (llvm::verifyFunction(F, &llvm::errs())) {
    llvm::errs() << "[attn-fwd-sched] produced invalid IR!\n";
    assert(false && "attn-fwd schedule pass must preserve valid IR");
  }
}
} // namespace mlir::triton::AMD
