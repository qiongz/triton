#include "TritonAMDGPUToLLVM/MembarUtility.h"
#include "AsyncUtility.h"
#include "Dialect/TritonAMDGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include <cstdlib>

namespace mlir::triton::AMD {
namespace {
// Returns true if one of the operands is a LocalLoad synced via AsyncWait.
bool filterAsyncLocalLoadsDependencies(Operation *op1, Operation *op2,
                                       Allocation *allocation) {
  auto isAsyncLoad = [](Operation *op) {
    return llvm::isa<triton::gpu::AsyncCopyGlobalToLocalOp,
                     triton::amdgpu::BufferLoadToLocalOp,
                     triton::amdgpu::AsyncTDMCopyLocalToGlobalOp>(op);
  };
  auto isLocalLoadWithAsyncWaitToken = [](Operation *op) {
    auto localLoad = llvm::dyn_cast<triton::gpu::LocalLoadOp>(op);
    return localLoad && isSyncedViaAsyncWait(localLoad);
  };
  auto getMemdescValue = [](Operation *op) -> Value {
    return llvm::TypeSwitch<Operation *, Value>(op)
        .Case<triton::amdgpu::BufferLoadToLocalOp>(
            [](auto op) { return op.getDest(); })
        .Case<triton::gpu::AsyncCopyGlobalToLocalOp>(
            [](auto op) { return op.getResult(); })
        .Case<triton::gpu::LocalLoadOp>([](auto op) { return op.getSrc(); })
        .Default([](Operation *) { return Value(); });
  };

  // Early return if neither or both operands are an AsyncLoad
  if (isAsyncLoad(op1) == isAsyncLoad(op2)) {
    return false;
  }

  Value op1Memdesc = getMemdescValue(op1);
  Value op2Memdesc = getMemdescValue(op2);
  if (!op1Memdesc || !op2Memdesc)
    return false;
  auto op1BufferIds = allocation->getAllBufferIdsWithAliases(op1Memdesc);
  auto op2BufferIds = allocation->getAllBufferIdsWithAliases(op2Memdesc);

  // Check if operations access the same buffer
  bool sameBuffer = llvm::any_of(
      op1BufferIds, [&](auto id) { return op2BufferIds.count(id); });

  if (!sameBuffer)
    return false;

  return isLocalLoadWithAsyncWaitToken(op1) ||
         isLocalLoadWithAsyncWaitToken(op2);
}

// Returns true if both operands are async loads (cp.async / buffer_load_to_lds).
// Two async-DMA writes are ordered by the async queue and commit/wait tokens, so
// a CTA-wide barrier between them is redundant (matches autopipeliner output).
// Gated by env var so the behavior is opt-in and A/B testable.
bool filterAsyncVsAsyncDependencies(Operation *op1, Operation *op2) {
  // Read each call (no static cache) so a kernel launcher can scope it via
  // os.environ around its own compile without affecting other kernels.
  const char *e = std::getenv("TRITON_AMD_MEMBAR_SKIP_ASYNC_ASYNC");
  if (!(e && e[0] == '1'))
    return false;
  auto isAsyncLoad = [](Operation *op) {
    return llvm::isa<triton::gpu::AsyncCopyGlobalToLocalOp,
                     triton::amdgpu::BufferLoadToLocalOp,
                     triton::amdgpu::AsyncTDMCopyLocalToGlobalOp>(op);
  };
  return isAsyncLoad(op1) && isAsyncLoad(op2);
}

bool filterLDSMemoryBarriersDependencies(Operation *op1, Operation *op2) {
  auto isLDSMemoryBarrierOp = [](Operation *op) {
    return llvm::isa<triton::amdgpu::InitBarrierOp,
                     triton::amdgpu::ArriveBarrierOp,
                     triton::amdgpu::AsyncCopyMbarrierArriveOp,
                     triton::amdgpu::WaitBarrierOp>(op);
  };

  return (isLDSMemoryBarrierOp(op1) && isLDSMemoryBarrierOp(op2));
}
} // namespace

bool membarFilter(Operation *op1, Operation *op2, bool /*op1IsRead*/,
                  bool /*op2IsRead*/, Allocation *allocation) {
  return (filterAsyncLocalLoadsDependencies(op1, op2, allocation) ||
          filterAsyncVsAsyncDependencies(op1, op2) ||
          filterLDSMemoryBarriersDependencies(op1, op2));
}
} // namespace mlir::triton::AMD
