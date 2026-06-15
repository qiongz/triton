//===- AttnBwdScheduling.cpp - attention-backward LDS-read prefetch -------===//
//
// A *second*, orthogonal attention-backward LLIR scheduling pass, distinct from
// both the GEMM-tuned LLIRSchedule.cpp and the MFMA-placement AttnSchedule.cpp.
//
// Where AttnSchedule.cpp moves the MFMAs among fixed anchors, this pass moves
// the *LDS reads* (the MFMA operand loads: `llvm.amdgcn.ds.read*` and any
// addrspace(3) load) as EARLY as their address is ready -- i.e. it prefetches
// the operands so the data has landed by the time the consuming MFMA issues
// (latency hiding for the LDS-read-bound dkdv loop; FA3 §B.2 issues operand
// loads ahead of the matmul). The two passes are orthogonal and stackable:
// enable this first (hoist loads), then AttnSchedule (interleave MFMA).
//
// Safety: dependency-preserving list scheduler. Non-load instructions keep their
// original relative order (memory / side-effect ordering preserved); a load is
// only emitted once its address operand has already been emitted. Only READY
// instructions are ever emitted, so SSA dominance always holds and the function
// always verifies.
//
// Enabled by TRITON_ENABLE_ATTN_BWD_SCHED=1. Misched is disabled when
// TRITON_ATTN_BWD_NOMISCHED is set (see python/src/llvm.cc) so the order sticks.
// Strategy via TRITON_ATTN_BWD_MODE:
//   load-movable (prefetch operands): early (default) | lead<N> | even
//   convert-movable (anti-cluster the Px8 converts, FA3 §B.2 interleave):
//     conv-late (defer each convert behind anchors) | conv-even
//
//===----------------------------------------------------------------------===//

#include "TritonAMDGPUToLLVM/Passes.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <string>

#define DEBUG_TYPE "tritonamdgpu-attn-bwd-schedule"

namespace {

using namespace llvm;

// Strategy (TRITON_ATTN_BWD_MODE):
//   LOAD-movable (movable class = LDS reads -> prefetch operands):
//     early   - emit each LDS read the moment its address is ready (ASAP prefetch)
//     even    - spread the LDS reads uniformly across the non-load anchors
//     lead<N> - even-spread target, but run N loads ahead so they prefetch
//   CONVERT-movable (movable class = layout converts shuffle/bpermute/perm ->
//   break the clustered Px8 convert block so it interleaves with MFMA, FA3 §B.2):
//     conv-late - defer each convert to just before its consumer (anti-cluster)
//     conv-even - spread converts uniformly across the non-convert anchors
enum class Mode { Early, Even, Lead, ConvLate, ConvEven };

static bool isConvertMode(Mode m) { return m == Mode::ConvLate || m == Mode::ConvEven; }

static Mode getMode(unsigned &leadOut) {
  leadOut = 2;
  const char *e = std::getenv("TRITON_ATTN_BWD_MODE");
  if (!e)
    return Mode::Early;
  StringRef s(e);
  if (s == "even")
    return Mode::Even;
  if (s == "conv-late")
    return Mode::ConvLate;
  if (s == "conv-even")
    return Mode::ConvEven;
  if (s.starts_with("lead")) {
    int v = 0;
    if (!s.drop_front(4).getAsInteger(10, v) && v > 0)
      leadOut = (unsigned)v;
    return Mode::Lead;
  }
  return Mode::Early;
}

// Layout convert at the LLVM-IR level = the shuffle / cross-lane permute that
// realizes a result->DotOperand conversion: shufflevector, or the AMDGPU
// ds.bpermute / perm intrinsics. (These lower to the asm `Px8` v_perm/ds_bpermute.)
static bool isConvertIR(const Instruction &I) {
  if (isa<ShuffleVectorInst>(&I))
    return true;
  if (const auto *CI = dyn_cast<CallInst>(&I)) {
    if (CI->isInlineAsm())
      return false;
    const Function *Callee = CI->getCalledFunction();
    if (Callee && Callee->isIntrinsic()) {
      StringRef N = Callee->getName();
      return N.contains("ds.bpermute") || N.contains(".perm");
    }
  }
  return false;
}

// MFMA operand load = ds_read / ds_load intrinsic, or any LDS (addrspace 3) load.
static bool isLDSRead(const Instruction &I) {
  if (const auto *CI = dyn_cast<CallInst>(&I)) {
    if (CI->isInlineAsm())
      return false;
    const Function *Callee = CI->getCalledFunction();
    if (Callee && Callee->isIntrinsic()) {
      StringRef N = Callee->getName();
      // ds.read* (incl ds.read.tr16/8) and ds.load*; NOT ds.bpermute (a permute).
      return N.contains("ds.read") || N.contains("ds.load");
    }
    return false;
  }
  if (const auto *LI = dyn_cast<LoadInst>(&I))
    return LI->getPointerAddressSpace() == 3;
  return false;
}

// Largest loop = the attention reduction loop.
static Loop *findMainLoop(LoopInfo &LI) {
  Loop *main = nullptr;
  size_t best = 0;
  for (Loop *L : LI) {
    size_t n = 0;
    for (BasicBlock *BB : L->blocks())
      n += BB->size();
    if (n > best) {
      best = n;
      main = L;
    }
  }
  return main;
}

// Dependency-preserving prefetch of one basic block's LDS reads.
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
  // Movable class by mode: LDS reads (prefetch operands) OR layout converts
  // (anti-cluster, FA3 §B.2). Kept-in-order = everything else (MFMA, VALU, GEPs).
  // Dependency-preserving: a movable inst is emitted only once its in-block
  // operands are emitted, so SSA dominance always holds.
  SmallVector<Instruction *, 64> loads, others;
  SmallPtrSet<Instruction *, 32> inBody;
  for (Instruction *I : body) {
    inBody.insert(I);
    bool movable = isConvertMode(mode) ? isConvertIR(*I) : isLDSRead(*I);
    if (movable)
      loads.push_back(I);
    else
      others.push_back(I);
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
  const size_t nL = loads.size(), nO = others.size();
  while (li < nL || oj < nO) {
    bool rl = (li < nL) && ready(loads[li]);
    bool ro = (oj < nO) && ready(others[oj]);
    bool pickL;
    if (rl && ro) {
      switch (mode) {
      case Mode::Early:
        pickL = true; // hoist the load the moment its address is ready
        break;
      case Mode::ConvLate:
        pickL = false; // defer the convert behind anchors -> break Px8 cluster
        break;
      case Mode::Lead: {
        size_t desired = (oj * nL) / nO + lead;
        pickL = (emittedL < desired);
        break;
      }
      case Mode::Even:
      case Mode::ConvEven:
      default: {
        size_t desired = (oj * nL) / nO;
        pickL = (emittedL < desired);
        break;
      }
      }
    } else {
      pickL = rl; // exactly one ready (the earlier-in-original-order one)
    }
    if (!rl && !ro)
      return false; // unexpected -> abort, IR untouched
    if (pickL) {
      order.push_back(loads[li]);
      emitted.insert(loads[li]);
      ++li; ++emittedL;
    } else {
      order.push_back(others[oj]);
      emitted.insert(others[oj]);
      ++oj;
    }
  }
  if (order.size() != body.size())
    return false;

  for (Instruction *I : order)
    I->moveBefore(term->getIterator());
  return true;
}

struct AttnBwdSchedulePass : FunctionPass {
  static char ID;
  AttnBwdSchedulePass() : FunctionPass(ID) {}

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

char AttnBwdSchedulePass::ID = 0;

namespace mlir::triton::AMD {

void runAttnBwdSchedulePass(llvm::Function &F, llvm::StringRef arch) {
  llvm::legacy::FunctionPassManager FPM(F.getParent());
  FPM.add(new AttnBwdSchedulePass());
  FPM.doInitialization();
  FPM.run(F);
  FPM.doFinalization();

  if (llvm::verifyFunction(F, &llvm::errs())) {
    llvm::errs() << "[attn-bwd-sched] produced invalid IR!\n";
    assert(false && "attn-bwd schedule pass must preserve valid IR");
  }
}

} // namespace mlir::triton::AMD
