//===--------- HelloAPIOps.cpp - HelloAPI dialect ops ---------------*- C++ -*-===//
//===----------------------------------------------------------------------===//
#include <iostream>
#include <queue>

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

#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallBitVector.h"

#include "Dialect/Hello/IR/HelloOps.hpp"
#include "Dialect/Hello/IR/HelloOpsDialect.cpp.inc"

using namespace mlir;
using namespace mlir::func;

//===----------------------------------------------------------------------===//
// Hello dialect.
//===----------------------------------------------------------------------===//

void HelloDialect::initialize()
{
  addOperations<
#define GET_OP_LIST
#include "Dialect/Hello/IR/HelloOps.cpp.inc"
    >();
}

#define GET_OP_CLASSES
#include "Dialect/Hello/IR/HelloOps.cpp.inc"

//===----------------------------------------------------------------------===//
// HelloWorldOp
/*
   void HelloWorldOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
   mlir::Value input) {
   state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
   state.addOperands({input});
   }
   */
//===----------------------------------------------------------------------===//

