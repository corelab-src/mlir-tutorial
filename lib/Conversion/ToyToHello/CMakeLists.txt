add_mlir_dialect_library(ToyToHello
  ToyToHello.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_BINARY_DIR}/
  ${PROJECT_BINARY_DIR}/include

  DEPENDS
  HelloOpsInc
  ToyOpsInc

  LINK_LIBS PUBLIC
  MLIRIR
  HelloOps 
  ToyOps
  )
target_include_directories(ToyToHello
  PRIVATE
  ${PROJECT_SOURCE_DIR}/
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_BINARY_DIR}/
  ${PROJECT_BINARY_DIR}/include
  )

#Header dependancies
add_dependencies(ToyToHello
  ToyOpsInc
  HelloOpsInc)

# Linking dependancies
add_dependencies(ToyToHello 
  HelloOps 
  ToyOps)
install(TARGETS ToyToHello DESTINATION lib)

