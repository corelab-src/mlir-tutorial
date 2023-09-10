//===--------- HelloOps.td - Hello Dialect Operation Definitions ----------===//
//
//===-------------------------- corelab jaeho ----------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef Hello_OPS
#define Hello_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
//===----------------------------------------------------------------------===//
// Hello dialect definition.
//===----------------------------------------------------------------------===//

def Hello_Dialect : Dialect {
  let name = "hello";
  let cppNamespace = "::mlir";
}

//===----------------------------------------------------------------------===//
// Base Hello operation definition.
//===----------------------------------------------------------------------===//

//class Op<Hello_Dialect, string mnemonic, list<OpTrait> traits = []> :
//        Op<Hello_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
//  Hello operations definition.
//===----------------------------------------------------------------------===//

def AddSignOp : Op<Hello_Dialect, "addsign", [NoSideEffect, SameOperandsShape, SameOperandsAndResultShape]> {
  let summary = "element-wise addition operation";
  let description = [{
    The "add" operation performs element-wise addition between two tensors.
      The shapes of the tensor operands are expected to match.
  }];

  let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs, I32Attr:$sign);
  let results = (outs F64Tensor);
}

#endif // Hello_OPS
