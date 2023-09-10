//===--------------------- Passes.h - Hello Ops Header ----------------------===//
//
//===-------------------------- corelab heelim ----------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_Hello_PASSES_H_
#define MLIR_Hello_PASSES_H_
#include "mlir/Pass/Pass.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LLVM.h"

#include <functional>
#include <memory>

namespace mlir {
class Pass;

class FuncOp;
class ModuleOp;
template <typename T>
class OperationPass;

std::unique_ptr<Pass> createRemoveRedundantAddSubPass();
std::unique_ptr<Pass> createConvertToyToHelloPass();

} // end namespace mlir

#endif
