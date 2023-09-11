add_mlir_dialect_library(ToyTrans
  removeRedundantAddSub.cpp
  createSubOp.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_BINARY_DIR}/
  ${PROJECT_BINARY_DIR}/include

  DEPENDS
  ToyOpsInc

  LINK_LIBS PUBLIC
  MLIRIR
  ToyOps
  )
target_include_directories(ToyTrans
  PRIVATE
  ${PROJECT_SOURCE_DIR}/
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_BINARY_DIR}/
  ${PROJECT_BINARY_DIR}/include
  )

#Header dependancies
add_dependencies(ToyTrans
  ToyOpsInc
  )
# Linking dependancies
add_dependencies(ToyTrans
  ToyOps)
install(TARGETS ToyTrans DESTINATION lib)


