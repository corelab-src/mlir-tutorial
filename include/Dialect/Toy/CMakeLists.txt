add_subdirectory(IR)

#set(LLVM_TARGET_DEFINITIONS Passes.td)
#mlir_tablegen(Passes.h.inc -gen-pass-decls -name PIMAPI)
#mlir_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix PIMAPI)
#mlir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix PIMAPI)
#add_public_tablegen_target(MLIRPIMAPIPassIncGen)
#add_mlir_doc(Passes PIMAPIPasses ./ -gen-pass-doc)
