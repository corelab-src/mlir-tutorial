set(LLVM_TARGET_DEFINITIONS ToyOps.td)
mlir_tablegen(ToyOps.hpp.inc -gen-op-decls "-I${PROJECT_SOURCE_DIR}/include")
mlir_tablegen(ToyOps.cpp.inc -gen-op-defs "-I${PROJECT_SOURCE_DIR}/include")
mlir_tablegen(ToyOpsDialect.hpp.inc -gen-dialect-decls "-I${PROJECT_SOURCE_DIR}/include")
mlir_tablegen(ToyOpsDialect.cpp.inc -gen-dialect-defs "-I${PROJECT_SOURCE_DIR}/include")

set(GEN_DOC_FILE ${CMAKE_BINARY_DIR}/docs/Dialects/toy.md)
add_public_tablegen_target(ToyOpsIncGen)

# Header dependancies target for ToyOps.hpp
add_custom_target(ToyOpsInc
  DEPENDS ToyOpsIncGen)

