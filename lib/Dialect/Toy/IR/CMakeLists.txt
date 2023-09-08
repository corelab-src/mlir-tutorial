#add_library(ToyOps
#    ToyOps.cpp)
add_mlir_dialect_library(ToyOps
	ToyOps.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include

        DEPENDS
				ToyOpsInc

	LINK_LIBS PUBLIC
	MLIRIR
	)
target_include_directories(ToyOps
    PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_BINARY_DIR}
    ${PROJECT_BINARY_DIR}/include)
add_dependencies(ToyOps ToyOpsInc)
install(TARGETS ToyOps DESTINATION lib)

