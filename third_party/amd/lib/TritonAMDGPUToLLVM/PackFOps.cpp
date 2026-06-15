//===- PackFOps.cpp - pack scalar f32 fops into <2xf32> (v_pk_*) ----------===//
//
// INVERSE of ScalarizePackedFOps.cpp. In basic blocks that contain MFMAs, pack
// pairs of independent scalar f32 fmul/fadd/fsub into a single <2 x float>
// binop. These codegen to packed v_pk_mul_f32 / v_pk_add_f32 (2 fp32 / instr),
// ~2x the VALU throughput on the softmax / acc-rescale path -- matching what the
// aiter hand-asm does (asm-verified: aiter uses 152 v_pk, Triton emits 0).
//
// Safety: a pair (a,b) is only fused when (1) same opcode + scalar f32, (2) b
// does not use a, and (3) a has no use strictly between a and b. Then the packed
// op is inserted right before b (both operands dominate it) and a/b are RAUW'd
// with extractelement; SSA dominance is preserved and the result verifies.
//
// Gated by TRITON_ENABLE_PACK_FOPS=1.
//
//===----------------------------------------------------------------------===//

#include "TritonAMDGPUToLLVM/Passes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Verifier.h"

using namespace llvm;

namespace {

static bool isMFMAorWMMA(Instruction &I) {
  auto *CI = dyn_cast<CallInst>(&I);
  if (!CI || CI->isInlineAsm())
    return false;
  Function *F = CI->getCalledFunction();
  if (!F || !F->isIntrinsic())
    return false;
  StringRef N = F->getName();
  return N.contains("mfma") || N.contains("wmma");
}

static bool isPackableFOp(Instruction *I) {
  unsigned op = I->getOpcode();
  if (op != Instruction::FMul && op != Instruction::FAdd &&
      op != Instruction::FSub)
    return false;
  return I->getType()->isFloatTy(); // scalar f32 only
}

// True if `a` is used by any instruction in (a, b) (exclusive..exclusive).
static bool usedBetween(Instruction *a, Instruction *b) {
  for (Instruction *it = a->getNextNode(); it && it != b;
       it = it->getNextNode()) {
    for (Value *op : it->operands())
      if (op == a)
        return true;
  }
  return false;
}

static bool packBB(BasicBlock &BB) {
  if (!llvm::any_of(BB, isMFMAorWMMA))
    return false;
  SmallVector<Instruction *, 64> cands;
  for (Instruction &I : BB)
    if (isPackableFOp(&I))
      cands.push_back(&I);
  if (cands.size() < 2)
    return false;

  IRBuilder<> builder(BB.getContext());
  SmallVector<Instruction *, 64> toErase;
  bool changed = false;
  VectorType *v2f32 = FixedVectorType::get(Type::getFloatTy(BB.getContext()), 2);

  size_t i = 0;
  while (i + 1 < cands.size()) {
    Instruction *a = cands[i];
    Instruction *b = cands[i + 1];
    // pair guards: same opcode; b must not use a; a not used between a and b.
    bool bUsesA = false;
    for (Value *op : b->operands())
      if (op == a)
        bUsesA = true;
    if (a->getOpcode() != b->getOpcode() || bUsesA || usedBetween(a, b)) {
      ++i; // can't pair a with b; advance by one and retry
      continue;
    }
    // build <2xf32> binop right before b (both a,b operands dominate b).
    builder.SetInsertPoint(b);
    Value *vlhs = builder.CreateInsertElement(
        UndefValue::get(v2f32), a->getOperand(0), builder.getInt32(0));
    vlhs = builder.CreateInsertElement(vlhs, b->getOperand(0), builder.getInt32(1));
    Value *vrhs = builder.CreateInsertElement(
        UndefValue::get(v2f32), a->getOperand(1), builder.getInt32(0));
    vrhs = builder.CreateInsertElement(vrhs, b->getOperand(1), builder.getInt32(1));
    Value *vres;
    if (a->getOpcode() == Instruction::FMul)
      vres = builder.CreateFMul(vlhs, vrhs);
    else if (a->getOpcode() == Instruction::FAdd)
      vres = builder.CreateFAdd(vlhs, vrhs);
    else
      vres = builder.CreateFSub(vlhs, vrhs);
    if (auto *vi = dyn_cast<Instruction>(vres))
      vi->copyFastMathFlags(a);
    Value *r0 = builder.CreateExtractElement(vres, builder.getInt32(0));
    Value *r1 = builder.CreateExtractElement(vres, builder.getInt32(1));
    a->replaceAllUsesWith(r0);
    b->replaceAllUsesWith(r1);
    toErase.push_back(a);
    toErase.push_back(b);
    changed = true;
    i += 2;
  }
  for (Instruction *I : toErase)
    I->eraseFromParent();
  return changed;
}

} // namespace

namespace mlir::triton::AMD {
void runPackFOpsPass(llvm::Function &F) {
  if (F.isDeclaration())
    return;
  for (BasicBlock &BB : F)
    packBB(BB);
  if (llvm::verifyFunction(F, &llvm::errs())) {
    llvm::errs() << "[pack-fops] produced invalid IR!\n";
    assert(false && "pack fops pass must preserve valid IR");
  }
}
} // namespace mlir::triton::AMD
