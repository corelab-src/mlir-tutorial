#ifndef MLIR_CONVERSION_ToyToHello_ToyToHello_H_
#define MLIR_CONVERSION_ToyToHello_ToyToHello_H_
#include "Pass/Passes.h"
#include "Dialect/Toy/IR/ToyOps.hpp"
#include "Dialect/Hello/IR/HelloOps.hpp"

namespace mlir {

class LLVMTypeConverter;
class MLIRContext;
class ModuleOp;
template <typename T>
class OperationPass;
class RewritePatternSet;

/// Populate the given list with patterns that convert from Toy to Hello.
void populateLoweringToyAddOpToHelloPatterns(
    RewritePatternSet& patterns, MLIRContext* context);
void populateLoweringToySubOpToHelloPatterns(
    RewritePatternSet& patterns, MLIRContext* context);

} // namespace mlir

#endif // MLIR_CONVERSION_ToyToHello_ToyToHello_H_
