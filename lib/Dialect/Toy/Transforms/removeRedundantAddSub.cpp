

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Operation.h"

#include "Pass/Passes.h"

#include "Dialect/Toy/IR/ToyOps.hpp"

#include <iostream>

using namespace mlir;

namespace{
struct removeRedundantAddSubPass
  : public PassWrapper<removeRedundantAddSubPass, OperationPass<ModuleOp>>{
  void runOnOperation() final;

  StringRef getArgument() const override {
    return "remove-add-sub";
  }

  StringRef getDescription() const override {
    return "remove redundant add and sub";
  }
};
}

void removeRedundantAddSubPass::runOnOperation() {
  SmallVector<Operation*, 1> removeVec;

  auto moduleOp = getOperation();
  moduleOp.walk([&](Operation *op) {
    if (isa<AddOp>(op)) {
      for (Operation* user: op->getUsers()) {
        if (isa<SubOp>(user)) {
          auto addArg1 = op->getOperand(1);
          auto subArg1 = user->getOperand(1);
          if (addArg1 == subArg1) {
            user->getResult(0).replaceAllUsesWith(op->getOperand(0));
            if (user->use_empty()) {
              removeVec.emplace_back(user);
            }
          }
        }
      }
    }
  });

  for (size_t i=0; i<removeVec.size(); i++) {
    removeVec[i]->erase();
  }
}

std::unique_ptr<mlir::Pass> mlir::createRemoveRedundantAddSubPass() {
  return std::make_unique<removeRedundantAddSubPass>();
}

