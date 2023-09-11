

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Operation.h"

#include "Pass/Passes.h"

#include "Dialect/Toy/IR/ToyOps.hpp"

#include <iostream>

using namespace mlir;

namespace{
struct createSubOpPass
  : public PassWrapper<createSubOpPass, OperationPass<ModuleOp>>{
  void runOnOperation() final;

  StringRef getArgument() const override {
    return "create-subop";
  }

  StringRef getDescription() const override {
    return "create sub";
  }
};
}

void createSubOpPass::runOnOperation() {
  OpBuilder builder(&getContext());
  int counter = 0;

  auto moduleOp = getOperation();
  auto loc = moduleOp.getLoc();
  Operation* firstOp = nullptr;

  moduleOp.walk([&](Operation *op) {
    if (isa<AddOp>(op)) {
      counter++;
      if (counter == 1) {
        firstOp = op;
      }
      if (counter == 2) {
        // SubOp generation
        builder.setInsertionPointAfter(op);
        auto newOp = builder.create<SubOp>(loc, firstOp->getResult(0).getType(),
            // 에러를 방지하기 위해 일단 두번째 argument 까지 firstOp->getOperand(0) 로 설정
            firstOp->getResult(0), firstOp->getResult(0));
        op->getResult(0).replaceAllUsesWith(newOp);
        // 그 후 나중에 두번째 argument 를 다시 op->getResult(0) 로 돌려줌
        // 이렇게 하는 이유는 replaceAllUsesWith 을 수행하면 %3의 두번째 argument인 %2까지 replace 되어서
        // %3이 정의되기 전에 사용되는 문제가 생김. (에러코드: @@@ dominant its use)
        newOp.setOperand(1, op->getResult(0));
      }
    }
  });
}

std::unique_ptr<mlir::Pass> mlir::createCreateSubOpPass() {
  return std::make_unique<createSubOpPass>();
}

