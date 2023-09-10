set(LLVM_TARGET_DEFINITIONS HelloOps.td)
mlir_tablegen(HelloOps.hpp.inc -gen-op-decls "-I${PROJECT_SOURCE_DIR}/include")
mlir_tablegen(HelloOps.cpp.inc -gen-op-defs "-I${PROJECT_SOURCE_DIR}/include")
mlir_tablegen(HelloOpsDialect.hpp.inc -gen-dialect-decls "-I${PROJECT_SOURCE_DIR}/include")
mlir_tablegen(HelloOpsDialect.cpp.inc -gen-dialect-defs "-I${PROJECT_SOURCE_DIR}/include")

set(GEN_DOC_FILE ${CMAKE_BINARY_DIR}/docs/Dialects/hello.md)
add_public_tablegen_target(HelloOpsIncGen)

# Header dependancies target for HelloOps.hpp
add_custom_target(HelloOpsInc
  DEPENDS HelloOpsIncGen)

