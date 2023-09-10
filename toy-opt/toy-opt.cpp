//===- standalone-opt.cpp ---------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
/* #include "mlir/Parser.h" */
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

#include "Pass/Passes.h"

#include "Conversion/ToyToHello/ToyToHello.h"
#include "Dialect/Hello/IR/HelloOps.hpp"
#include "Dialect/Toy/IR/ToyOps.hpp"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include <iostream>
namespace cl = llvm::cl;
using namespace mlir;

static cl::opt<std::string> input_filename(
    cl::Positional, cl::desc("<input file>"), cl::init("-"));

static cl::opt<std::string> output_filename("o",
    cl::desc("Output filename"), cl::value_desc("filename"),
    cl::init("-"));

void initPasses()
{
  mlir::registerPass(
    []() -> std::unique_ptr<mlir::Pass> {
      return mlir::createRemoveRedundantAddSubPass();
  });
  // mlir::registerPass(
  //   []() -> std::unique_ptr<mlir::Pass> {
  //     // Add another Passes
  // });
}

int main(int argc, char** argv)
{
  mlir::DialectRegistry registry;
  registry.insert<mlir::HelloDialect>();
  registry.insert<mlir::ToyDialect>();
  registry.insert<mlir::func::FuncDialect>();
  initPasses();

  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  mlir::registerPassManagerCLOptions();
  mlir::PassPipelineCLParser passPipeline("", "Compiler passes to run");
  cl::ParseCommandLineOptions(argc, argv, "toy compiler \n");

  // Set up the input file.
  std::string error_message;
  auto file = mlir::openInputFile(input_filename, &error_message);
  assert(file);

  auto output = mlir::openOutputFile(output_filename, &error_message);
  assert(output);

  return failed(mlir::MlirOptMain(output->os(), std::move(file), passPipeline,
      registry, 0, 0, 1, 0, true));
}
