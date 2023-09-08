//===- ToyOps.cpp - Toy dialect ops ---------------*- C++ -*-===//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"

#include "mlir/Support/TypeID.h"

#include "mlir/Dialect/Traits.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"

#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/InliningUtils.h"

#include "Dialect/Toy/IR/ToyOps.hpp"
#include "Dialect/Toy/IR/ToyOpsDialect.cpp.inc"

using namespace mlir;
using namespace mlir::func;

//===----------------------------------------------------------------------===//
// Toy dialect.
//===----------------------------------------------------------------------===//

void ToyDialect::initialize()
{
  addOperations<
#define GET_OP_LIST
#include "Dialect/Toy/IR/ToyOps.cpp.inc"
      >();
}

#define GET_OP_CLASSES
#include "Dialect/Toy/IR/ToyOps.cpp.inc"

//===----------------------------------------------------------------------===//
// AddOp

void AddOp::build(mlir::OpBuilder& builder, mlir::OperationState& state,
    mlir::Value lhs, mlir::Value rhs)
{
  state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
  state.addOperands({ lhs, rhs });
}
//===----------------------------------------------------------------------===//
// SubOp

void SubOp::build(mlir::OpBuilder& builder, mlir::OperationState& state,
    mlir::Value lhs, mlir::Value rhs)
{
  state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
  state.addOperands({ lhs, rhs });
}
