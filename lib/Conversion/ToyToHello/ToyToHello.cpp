//===- ToyToHello.cpp - conversion from Toy to Hello dialect ----------===//
//
//
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/ErrorHandling.h"

#include "Conversion/ToyToHello/ToyToHello.h"
#include "Pass/Passes.h"

#include "Dialect/Hello/IR/HelloOps.hpp"
#include "Dialect/Toy/IR/ToyOps.hpp"

#include <iostream>

using namespace mlir;
//===----------------------------------------------------------------------===//
// PrintOp
//===----------------------------------------------------------------------===//
// struct ToyPrintOpToHello : public mlir::ConversionPattern {
//   ToyPrintOpToHello(MLIRContext* context)
//       : ConversionPattern(mlir::PrintOp::getOperationName(), 1, context)
//   {
//   }
//
//   LogicalResult matchAndRewrite(mlir::Operation* op, mlir::ArrayRef<Value> operands,
//       mlir::ConversionPatternRewriter& rewriter) const final
//   {
//     PrintOpAdaptor operandAdaptor(operands);
//     rewriter.replaceOpWithNewOp<HelloWorldOp>(op, operandAdaptor.input());
//     return success();
//   }
// };
//
// void mlir::populateLoweringToyPrintOpToHelloPatterns(
//     RewritePatternSet& patterns, MLIRContext* context)
// {
//   patterns.insert<ToyPrintOpToHello>(context);
// }
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------------===//
struct ToyAddOpToHello : public mlir::ConversionPattern {
  ToyAddOpToHello(MLIRContext* context)
    : ConversionPattern(mlir::AddOp::getOperationName(), 1, context)
  {
  }

  LogicalResult matchAndRewrite(mlir::Operation* op, mlir::ArrayRef<Value> operands,
      mlir::ConversionPatternRewriter& rewriter) const final
  {
    auto loc = op->getLoc();

    auto lhs = op->getOperand(0);
    auto rhs = op->getOperand(1);
    auto addSignOp = rewriter.create<AddSignOp>(loc, lhs.getType(), lhs, rhs, rewriter.getI32IntegerAttr(0));

    rewriter.replaceOp(op, addSignOp->getResult(0));
    return success();
  }
};

void mlir::populateLoweringToyAddOpToHelloPatterns(
    RewritePatternSet& patterns, MLIRContext* context)
{
  patterns.insert<ToyAddOpToHello>(context);
}
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// SubOp
//===----------------------------------------------------------------------===//
struct ToySubOpToHello : public mlir::ConversionPattern {
  ToySubOpToHello(MLIRContext* context)
    : ConversionPattern(mlir::SubOp::getOperationName(), 1, context)
  {
  }

  LogicalResult matchAndRewrite(mlir::Operation* op, mlir::ArrayRef<Value> operands,
      mlir::ConversionPatternRewriter& rewriter) const final
  {
    auto loc = op->getLoc();

    auto lhs = op->getOperand(0);
    auto rhs = op->getOperand(1);
    auto addSignOp = rewriter.create<AddSignOp>(loc, lhs.getType(), lhs, rhs, rewriter.getI32IntegerAttr(1));

    rewriter.replaceOp(op, addSignOp->getResult(0));
    return success();
  }
};

void mlir::populateLoweringToySubOpToHelloPatterns(
    RewritePatternSet& patterns, MLIRContext* context)
{
  patterns.insert<ToySubOpToHello>(context);
}
//===----------------------------------------------------------------------===//

namespace {
struct ConvertToyToHelloPass
    : public PassWrapper<ConvertToyToHelloPass, OperationPass<ModuleOp>> {
  void getDependentDialects(mlir::DialectRegistry& registry) const override
  {
    registry.insert<ToyDialect, HelloDialect>();
  }
  void runOnOperation() final;
  StringRef getArgument() const override
  {
    return "toy";
  }
  StringRef getDescription() const override
  {
    return "toy to hello lowering pass";
  }
};
}

void ConvertToyToHelloPass::runOnOperation()
{
  ModuleOp module = getOperation();
  ConversionTarget target(getContext());

  target.addIllegalDialect<ToyDialect>();
  target.addLegalDialect<HelloDialect>();

  target.addLegalOp<ToyReturnOp>();
  // target.addLegalOp<AddOp>();

  RewritePatternSet patterns(&getContext());

  // ----------- Adding Patterns for Lowering Pass ----------- //
  populateLoweringToyAddOpToHelloPatterns(patterns, &getContext());
  populateLoweringToySubOpToHelloPatterns(patterns, &getContext());
  // --------------------------------------------------------- //
  if (mlir::failed(applyPartialConversion(module, target, std::move(patterns)))) {
    signalPassFailure();
  }
}
std::unique_ptr<mlir::Pass> mlir::createConvertToyToHelloPass()
{
  return std::make_unique<ConvertToyToHelloPass>();
}
