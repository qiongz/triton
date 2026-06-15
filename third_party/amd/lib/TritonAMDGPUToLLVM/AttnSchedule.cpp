//===- AttnSchedule.cpp - attention-backward LLIR interleave --------------===//
//
// A *separate*, attention-specific LLIR scheduling pass (distinct from the
// GEMM-tuned LLIRSchedule.cpp). Goal: realize the hand-written AC/MFMA/LR
// pipeline of the Gluon attention-backward kernels by interleaving MFMA with
// the LDS reads / softmax (VALU) / layout converts in the main loop body, then
// rely on misched being disabled (see python/src/llvm.cc, gated by the same env
// var) so the order sticks through to ISA.
//
// Safety: this is a dependency-preserving list scheduler. Non-MFMA instructions
// keep their original relative order (so all memory / side-effect ordering is
// preserved), and each (pure) MFMA is only placed once all of its operands have
// already been placed. Therefore SSA dominance is never violated and the result
// always verifies -- unlike the GEMM pass, which assumes an already
// software-pipelined GEMM loop and produced invalid IR on attention.
//
// Enabled by TRITON_ENABLE_ATTN_SCHED=1.
//
//===----------------------------------------------------------------------===//

#include "TritonAMDGPUToLLVM/Passes.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdlib>
#include <string>

#define DEBUG_TYPE "tritonamdgpu-attn-schedule"

namespace {

using namespace llvm;

// MFMA placement strategy (selected via TRITON_ATTN_SCHED_MODE):
//   even    - spread MFMA uniformly across the non-MFMA anchors (draft default)
//   early   - emit each MFMA as soon as its operands are ready (issue ASAP)
//   late    - delay MFMA until no other instruction is ready (fill VALU bubbles)
//   ld<N>   - keep each MFMA <lead> anchors ahead of its even-spread position
//             so its ds_read operands have time to land (latency hiding)
enum class SchedMode { Even, Early, Late, Lead };

static SchedMode getSchedMode(unsigned &leadOut) {
  leadOut = 2;
  const char *e = std::getenv("TRITON_ATTN_SCHED_MODE");
  if (!e)
    return SchedMode::Even;
  StringRef s(e);
  if (s == "early")
    return SchedMode::Early;
  if (s == "late")
    return SchedMode::Late;
  if (s.starts_with("lead")) {
    int v = 0;
    if (!s.drop_front(4).getAsInteger(10, v) && v > 0)
      leadOut = (unsigned)v;
    return SchedMode::Lead;
  }
  return SchedMode::Even;
}

static bool isMFMAorWMMA(const Instruction &I) {
  const auto *CI = dyn_cast<CallInst>(&I);
  if (!CI || CI->isInlineAsm())
    return false;
  Function *Callee = CI->getCalledFunction();
  if (!Callee || !Callee->isIntrinsic())
    return false;
  StringRef Name = Callee->getName();
  return Name.contains("mfma") || Name.contains("wmma");
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

// Dependency-preserving interleave of one basic block.
// Returns true if it reordered anything.
static bool scheduleBB(BasicBlock &BB) {
  // Partition: PHIs stay at top, terminator stays at bottom.
  SmallVector<Instruction *, 256> body;
  Instruction *term = BB.getTerminator();
  for (Instruction &I : BB) {
    if (isa<PHINode>(&I) || &I == term)
      continue;
    body.push_back(&I);
  }
  if (body.size() < 4)
    return false;

  // Split into pure-MFMA (movable) and everything else (kept in order).
  SmallVector<Instruction *, 128> mfmas, others;
  SmallPtrSet<Instruction *, 32> inBody;
  for (Instruction *I : body) {
    inBody.insert(I);
    if (isMFMAorWMMA(*I))
      mfmas.push_back(I);
    else
      others.push_back(I);
  }
  // Nothing to interleave if there are no MFMAs or no anchors to spread among.
  if (mfmas.empty() || others.empty())
    return false;

  // An instruction is "ready" once every operand defined inside this block has
  // already been emitted (operands defined outside the block are available).
  SmallPtrSet<Value *, 32> emitted;
  auto ready = [&](Instruction *I) {
    for (Value *op : I->operands()) {
      auto *opI = dyn_cast<Instruction>(op);
      if (opI && inBody.count(opI) && !emitted.count(opI))
        return false;
    }
    return true;
  };

  // Ready-based list schedule. `others` keep their original relative order (so
  // all memory / side-effect ordering is preserved); MFMAs are released
  // according to the selected strategy. We only ever emit a READY instruction,
  // so def-before-use (SSA dominance) always holds.
  unsigned lead = 2;
  const SchedMode mode = getSchedMode(lead);
  SmallVector<Instruction *, 256> order;
  order.reserve(body.size());
  size_t mi = 0, oj = 0, emittedM = 0;
  const size_t nM = mfmas.size(), nO = others.size();
  while (mi < nM || oj < nO) {
    bool rm = (mi < nM) && ready(mfmas[mi]);
    bool ro = (oj < nO) && ready(others[oj]);
    bool pickM;
    if (rm && ro) {
      switch (mode) {
      case SchedMode::Early:
        pickM = true; // issue the MFMA the moment it is ready
        break;
      case SchedMode::Late:
        pickM = false; // prefer anchors; MFMA only when nothing else is ready
        break;
      case SchedMode::Lead: {
        // even-spread target, but run `lead` MFMAs ahead so their ds_read
        // operands have time to land before the MFMA issues (latency hiding).
        size_t desired = (oj * nM) / nO + lead;
        pickM = (emittedM < desired);
        break;
      }
      case SchedMode::Even:
      default: {
        size_t desired = (oj * nM) / nO;
        pickM = (emittedM < desired);
        break;
      }
      }
    } else {
      pickM = rm; // exactly one is ready (the earlier-in-original-order one)
    }
    if (!rm && !ro)
      return false; // unexpected: abort with no moves -> IR untouched
    if (pickM) {
      order.push_back(mfmas[mi]);
      emitted.insert(mfmas[mi]);
      ++mi; ++emittedM;
    } else {
      order.push_back(others[oj]);
      emitted.insert(others[oj]);
      ++oj;
    }
  }
  if (order.size() != body.size())
    return false; // safety: never partially move

  // Apply: move each instruction to just before the terminator, in `order`.
  // (PHIs are untouched at the top; the terminator stays last.)
  for (Instruction *I : order)
    I->moveBefore(term->getIterator());
  return true;
}

struct AttnSchedulePass : FunctionPass {
  static char ID;
  AttnSchedulePass() : FunctionPass(ID) {}

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

char AttnSchedulePass::ID = 0;

namespace mlir::triton::AMD {

void runAttnSchedulePass(llvm::Function &F, llvm::StringRef arch) {
  llvm::legacy::FunctionPassManager FPM(F.getParent());
  FPM.add(new AttnSchedulePass());
  FPM.doInitialization();
  FPM.run(F);
  FPM.doFinalization();

  if (llvm::verifyFunction(F, &llvm::errs())) {
    llvm::errs() << "[attn-sched] produced invalid IR!\n";
    assert(false && "attn schedule pass must preserve valid IR");
  }
}

} // namespace mlir::triton::AMD
