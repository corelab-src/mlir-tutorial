add_mlir_dialect_library(HelloOps
  HelloOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include

  DEPENDS
  HelloOpsInc

  LINK_LIBS PUBLIC
  MLIRIR
  )
target_include_directories(HelloOps
  PRIVATE
  ${PROJECT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_BINARY_DIR}
  ${PROJECT_BINARY_DIR}/include
  )
add_dependencies(HelloOps HelloOpsInc)
install(TARGETS HelloOps DESTINATION lib)

