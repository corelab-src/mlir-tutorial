//===----------- ToyOps.td - Toy Dialect Operation Definitions ------------===//
//
//===-------------------------- corelab heelim ----------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef Toy_OPS
#define Toy_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
//===----------------------------------------------------------------------===//
// Toy dialect definition.
//===----------------------------------------------------------------------===//

def Toy_Dialect : Dialect {
  let name = "toy";
  let cppNamespace = "::mlir";
}

//===----------------------------------------------------------------------===//
// Base Toy operation definition.
//===----------------------------------------------------------------------===//

//class Op<Toy_Dialect, string mnemonic, list<OpTrait> traits = []> :
//        Op<Toy_Dialect, mnemonic, traits>;

def PrintOp : Op<Toy_Dialect, "print",[]> {
  let summary = "print operation";
  let description = [{
    The "print" builtin operation prints a given input tensor, and produces
      no results.
  }];

  // The print operation takes an input tensor to print.
  let arguments = (ins F64Tensor:$input);

  let assemblyFormat = "$input attr-dict `:` type($input)";
}

def AddOp : Op<Toy_Dialect, "add", [NoSideEffect]> {
  let summary = "element-wise addition operation";
  let description = [{
    The "add" operation performs element-wise addition between two tensors.
      The shapes of the tensor operands are expected to match.
  }];

  let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs);
  let results = (outs F64Tensor);

  // Allow building an AddOp with from the two input operands.
  let builders = [
    OpBuilder<(ins "Value":$lhs, "Value":$rhs)>
  ];
}

def SubOp : Op<Toy_Dialect, "sub", [NoSideEffect]> {
  let summary = "element-wise subtraction operation";
  let description = [{
    The "sub" operation performs element-wise subtraction between two tensors.
      The shapes of the tensor operands are expected to match.
  }];

  let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs);
  let results = (outs F64Tensor);

  // Allow building an SubOp with from the two input operands.
  let builders = [
    OpBuilder<(ins "Value":$lhs, "Value":$rhs)>
  ];
}

def ToyReturnOp : Op<Toy_Dialect,"return",
    [NoSideEffect, ReturnLike, Terminator]> {
      let summary = "return operation";
      let description = [{
        The "return" operation represents a return operation within a function.
          The operation takes an optional tensor operand and produces no results.
          The operand type must match the signature of the function that contains
          the operation. For example:

          ```mlir
          func @foo() -> tensor<2xf64> {
            ...
              toy.return %0 : tensor<2xf64>
          }
        ```
      }];

      // The return operation takes an optional input operand to return. This
      // value must match the return type of the enclosing function.
      let arguments = (ins Variadic<AnyTensor>:$input);

      // The return operation only emits the input in the format if it is present.
      let assemblyFormat = "($input^ `:` type($input))? attr-dict ";

      // Allow building a ReturnOp with no return operand.
      let builders = [
        OpBuilder<(ins), [{ build($_builder, $_state, llvm::None); }]>
      ];

      // Provide extra utility definitions on the c++ operation class definition.
      let extraClassDeclaration = [{
        bool hasOperand() { return getNumOperands() != 0; }
      }];
    }
#endif // Toy_OPS
