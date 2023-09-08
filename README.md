# An out-of-tree MLIR dialect example 

This is an example of an out-of-tree [MLIR](https://mlir.llvm.org/) dialect along with a `toy-opt` tool to operate on that dialect.

## Building

This setup assumes that you have built and installed LLVM and MLIR. To build and launch the tests, run
```sh
mkdir build && cd build
cmake -G Ninja ..
cmake --build . --target toy-opt
```
**Note**: If command `cmake -G Ninja ..` not working properly, Open `mlir101/CMakeList.txt` file and check if `LLVM_DIR` and `MLIR_DIR` is properly defined.


