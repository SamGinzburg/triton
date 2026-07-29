#ifndef TRITON_THIRD_PARTY_AMD_LIB_TRITONAMDGPUTOLLVM_ATOMICRMWOPSEMITTER_H_
#define TRITON_THIRD_PARTY_AMD_LIB_TRITONAMDGPUTOLLVM_ATOMICRMWOPSEMITTER_H_

#include "TargetInfo.h"

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "triton/Analysis/Utility.h"
#include "llvm/ADT/STLFunctionalExtras.h"

namespace mlir::LLVM::AMD {

class AtomicRMWEmitter {
public:
  using AtomicCallback =
      llvm::function_ref<Value(RewriterBase &, Value, Value)>;

  AtomicRMWEmitter(const mlir::triton::AMD::TargetInfo &targetInfo,
                   LLVM::AtomicBinOp binOp, LLVM::AtomicOrdering memOrder,
                   StringRef scopeStr)
      : targetInfo(targetInfo), binOp(binOp), memOrder(memOrder),
        scopeStr(scopeStr) {}

  Value emitAtomicRMW(RewriterBase &rewriter, Value rmwPtr, Value valElem,
                      Value rmwMask, std::optional<Value> sharedMemBase,
                      bool enableIntraWaveReduce) const;

  Value emitPairedAtomicForEvenTID(RewriterBase &rewriter, Value rmwPtr,
                                   Value valElem, Value rmwMask) const;

  // Group active lanes by a 64-bit address/offset key, reduce each group's
  // operand, and invoke emitAtomic once per group leader.
  Value emitIntraWaveReducedAtomic(RewriterBase &rewriter, Value key,
                                   Value operand, Value mask,
                                   int64_t adjacentKeyStride,
                                   AtomicCallback emitAtomic) const;
  void setAtomicOrdering(LLVM::AtomicOrdering memOrder) {
    this->memOrder = memOrder;
  }

private:
  const mlir::triton::AMD::TargetInfo &targetInfo;

  mlir::LLVM::AtomicBinOp binOp;
  mlir::LLVM::AtomicOrdering memOrder;
  std::string scopeStr;

  Value atomicIntraWaveReduce(RewriterBase &rewriter, Value key, Value operand,
                              int64_t adjacentKeyStride,
                              AtomicCallback emitAtomic) const;
};

} // namespace mlir::LLVM::AMD

#endif // TRITON_THIRD_PARTY_AMD_LIB_TRITONAMDGPUTOLLVM_ATOMICRMWEMITTER_H_
