
## build:
```
rm -rf build

cmake -S . -B build -G Ninja \
  -DMLIR_DIR=/usr/lib/llvm-18/lib/cmake/mlir \
  -DLLVM_DIR=/usr/lib/llvm-18/lib/cmake/llvm


cmake --build build 

```

## Pass
### poly-build-repr
```
/usr/bin/mlir-opt-18 test/input.mlir -linalg-generalize-named-ops \
| /usr/bin/mlir-opt-18 - -one-shot-bufferize="bufferize-function-boundaries" \
| /usr/bin/mlir-opt-18 - -convert-linalg-to-affine-loops -canonicalize -cse \
| /usr/bin/mlir-opt-18 - \
    -load-pass-plugin=./build/GpuTileILPSelection.so \
    -pass-pipeline='builtin.module(func.func(poly-build-repr))'
```

### poly-depend-analyze
```
/usr/bin/mlir-opt-18 test/input.mlir -linalg-generalize-named-ops \
| /usr/bin/mlir-opt-18 - -one-shot-bufferize="bufferize-function-boundaries" \
| /usr/bin/mlir-opt-18 - -convert-linalg-to-affine-loops -canonicalize -cse \
| /usr/bin/mlir-opt-18 - \
    -load-pass-plugin=./build/GpuTileILPSelection.so \
    -pass-pipeline='builtin.module(func.func(poly-depend-analyze))'
```



### gpu-apply-tiles
```
/usr/bin/mlir-opt-18 test/input_tile.mlir -linalg-generalize-named-ops \
| /usr/bin/mlir-opt-18 - -one-shot-bufferize="bufferize-function-boundaries" \
| /usr/bin/mlir-opt-18 - -convert-linalg-to-affine-loops -canonicalize -cse \
| /usr/bin/mlir-opt-18 - \
    -load-pass-plugin=./build/GpuTileILPSelection.so \
    -pass-pipeline='builtin.module(func.func(gpu-apply-tiles))'
```    


### gpu-ilp-select-tiles
```
/usr/bin/mlir-opt-18 test/input.mlir -linalg-generalize-named-ops \
| /usr/bin/mlir-opt-18 - -one-shot-bufferize="bufferize-function-boundaries" \
| /usr/bin/mlir-opt-18 - -convert-linalg-to-affine-loops -canonicalize -cse \
| /usr/bin/mlir-opt-18 - \
    -load-pass-plugin=./build/GpuTileILPSelection.so \
    -pass-pipeline='builtin.module(func.func(gpu-ilp-select-tiles))'
```    






## bench test


/usr/bin/mlir-opt-18 test/mm_bench_memref.mlir \
  -linalg-generalize-named-ops \
  -convert-linalg-to-affine-loops -canonicalize -cse \
  -affine-loop-tile="tile-sizes=8,8,32" \
  -lower-affine \
  -convert-scf-to-cf \
  -convert-to-llvm \
  > test/mm_tiled.ll

SHLIB_DIR=$(llvm-config-18 --libdir)

### 预热一次
mlir-cpu-runner-18 test/mm_tiled.ll \
  -e main -entry-point-result=void \
  -shared-libs=$SHLIB_DIR/libmlir_runner_utils.so \
  -shared-libs=$SHLIB_DIR/libmlir_c_runner_utils.so >/dev/null

### 计时
/usr/bin/time -p mlir-cpu-runner-18 test/mm_tiled.ll \
  -e main -entry-point-result=void \
  -shared-libs=$SHLIB_DIR/libmlir_runner_utils.so \
  -shared-libs=$SHLIB_DIR/libmlir_c_runner_utils.so \
  >/dev/null




## bench_gemm.sh 

## 用法

* 指定输入 + 多组 tile：

  ```bash
  ./bench_gemm.sh test/mm_bench_memref.mlir  8,8,32  16,16,16  32,32,8
  ```
* 不给输入（默认 `test/mm_bench_memref.mlir`）：

  ```bash
  ./bench_gemm.sh 8,8,32 16,16,16 32,32,8
  ```
* 控制 tiler / 次数 / 预热：

  ```bash
  TILER=affine REPEATS=7 PREHEAT=2 ./bench_gemm.sh test/mm_bench_memref.mlir 16,16,16 32,32,16
  ```
* 只改显示用的 FLOPs 规模（不改 IR）：

  ```bash
  M=256 N=256 K=256 N_RUNS=500 ./bench_gemm.sh test/mm_bench_memref.mlir 16,16,16 32,32,16
  ```

### test result 
#### 记录gpu-ilp-select-tiles v2版的 性能对比数据
```
/usr/bin/mlir-opt-18 test/input.mlir -linalg-generalize-named-ops \
| /usr/bin/mlir-opt-18 - -one-shot-bufferize="bufferize-function-boundaries" \
| /usr/bin/mlir-opt-18 - -convert-linalg-to-affine-loops -canonicalize -cse \
| /usr/bin/mlir-opt-18 - \
    -load-pass-plugin=./build/GpuTileILPSelection.so \
    -pass-pipeline='builtin.module(func.func(gpu-ilp-select-tiles))'
<stdin>:3:5: remark: 已选择分块大小并写回属性 tile.sizes.vector 与 tile.sizes.
    affine.for %arg3 = 0 to 64 {
    ^
<stdin>:3:5: note: see current operation: 
affine.for %arg3 = 0 to 64 {
  affine.for %arg4 = 0 to 32 {
    affine.for %arg5 = 0 to 128 {
      %0 = affine.load %arg0[%arg3, %arg5] : memref<64x128xf32, strided<[?, ?], offset: ?>>
      %1 = affine.load %arg1[%arg5, %arg4] : memref<128x32xf32, strided<[?, ?], offset: ?>>
      %2 = affine.load %arg2[%arg3, %arg4] : memref<64x32xf32, strided<[?, ?], offset: ?>>
      %3 = arith.mulf %0, %1 : f32
      %4 = arith.addf %2, %3 : f32
      affine.store %4, %arg2[%arg3, %arg4] : memref<64x32xf32, strided<[?, ?], offset: ?>>
    }
  }
} {tile.sizes = [64, 1, 128], tile.sizes.reuse = [64, 32, 32], tile.sizes.vector = [64, 1, 128]}
module {
  func.func @mm(%arg0: memref<64x128xf32, strided<[?, ?], offset: ?>>, %arg1: memref<128x32xf32, strided<[?, ?], offset: ?>>, %arg2: memref<64x32xf32, strided<[?, ?], offset: ?>>) -> memref<64x32xf32, strided<[?, ?], offset: ?>> {
    affine.for %arg3 = 0 to 64 {
      affine.for %arg4 = 0 to 32 {
        affine.for %arg5 = 0 to 128 {
          %0 = affine.load %arg0[%arg3, %arg5] : memref<64x128xf32, strided<[?, ?], offset: ?>>
          %1 = affine.load %arg1[%arg5, %arg4] : memref<128x32xf32, strided<[?, ?], offset: ?>>
          %2 = affine.load %arg2[%arg3, %arg4] : memref<64x32xf32, strided<[?, ?], offset: ?>>
          %3 = arith.mulf %0, %1 : f32
          %4 = arith.addf %2, %3 : f32
          affine.store %4, %arg2[%arg3, %arg4] : memref<64x32xf32, strided<[?, ?], offset: ?>>
        }
      }
    } {tile.sizes = [64, 1, 128], tile.sizes.reuse = [64, 32, 32], tile.sizes.vector = [64, 1, 128]}
    return %arg2 : memref<64x32xf32, strided<[?, ?], offset: ?>>
  }
}
```

```
./bench_gemm.sh 8,8,32 16,16,16 32,32,8
>>> Using:
INPUT=test/mm_bench_memref.mlir
TILER=affine
SHLIB_DIR=/usr/lib/llvm-18/lib
Dims: M=64, N=32, K=128, N_RUNS=200

tile-sizes=8,8,32 | repeats=5 | median=0.11s | mean=0.110000s | GFLOPS=0.953
tile-sizes=16,16,16 | repeats=5 | median=0.09s | mean=0.090000s | GFLOPS=1.165
tile-sizes=32,32,8 | repeats=5 | median=0.08s | mean=0.078000s | GFLOPS=1.311
sunteng@zhimin-M12SWA-TF:~/codespace/mlir-gpu-tiler$ 
sunteng@zhimin-M12SWA-TF:~/codespace/mlir-gpu-tiler$ ./bench_gemm.sh 64,32,32 64,1,128
>>> Using:
INPUT=test/mm_bench_memref.mlir
TILER=affine
SHLIB_DIR=/usr/lib/llvm-18/lib
Dims: M=64, N=32, K=128, N_RUNS=200

tile-sizes=64,32,32 | repeats=5 | median=0.11s | mean=0.110000s | GFLOPS=0.953
tile-sizes=64,1,128 | repeats=5 | median=0.16s | mean=0.160000s | GFLOPS=0.655
```
使用gpu-ilp-select-tiles生成的tile size 效果并不好 