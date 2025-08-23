rm -rf build


cmake -S . -B build -G Ninja \
  -DMLIR_DIR=/usr/lib/llvm-18/lib/cmake/mlir \
  -DLLVM_DIR=/usr/lib/llvm-18/lib/cmake/llvm


cmake --build build 



## poly-build-repr
/usr/bin/mlir-opt-18 test/input.mlir -linalg-generalize-named-ops \
| /usr/bin/mlir-opt-18 - -one-shot-bufferize="bufferize-function-boundaries" \
| /usr/bin/mlir-opt-18 - -convert-linalg-to-affine-loops -canonicalize -cse \
| /usr/bin/mlir-opt-18 - \
    -load-pass-plugin=./build/GpuTileILPSelection.so \
    -pass-pipeline='builtin.module(func.func(poly-build-repr))'


## poly-depend-analyze
/usr/bin/mlir-opt-18 test/input.mlir -linalg-generalize-named-ops \
| /usr/bin/mlir-opt-18 - -one-shot-bufferize="bufferize-function-boundaries" \
| /usr/bin/mlir-opt-18 - -convert-linalg-to-affine-loops -canonicalize -cse \
| /usr/bin/mlir-opt-18 - \
    -load-pass-plugin=./build/GpuTileILPSelection.so \
    -pass-pipeline='builtin.module(func.func(poly-depend-analyze))'




### gpu-apply-tiles
/usr/bin/mlir-opt-18 test/input_tile.mlir -linalg-generalize-named-ops \
| /usr/bin/mlir-opt-18 - -one-shot-bufferize="bufferize-function-boundaries" \
| /usr/bin/mlir-opt-18 - -convert-linalg-to-affine-loops -canonicalize -cse \
| /usr/bin/mlir-opt-18 - \
    -load-pass-plugin=./build/GpuTileILPSelection.so \
    -pass-pipeline='builtin.module(func.func(gpu-apply-tiles))'


### gpu-ilp-select-tiles

/usr/bin/mlir-opt-18 test/input.mlir -linalg-generalize-named-ops \
| /usr/bin/mlir-opt-18 - -one-shot-bufferize="bufferize-function-boundaries" \
| /usr/bin/mlir-opt-18 - -convert-linalg-to-affine-loops -canonicalize -cse \
| /usr/bin/mlir-opt-18 - \
    -load-pass-plugin=./build/GpuTileILPSelection.so \
    -pass-pipeline='builtin.module(func.func(gpu-ilp-select-tiles))'




### test 
