module {
  func.func @mm(%A: memref<64x128xf32>,
                %B: memref<128x32xf32>,
                %C: memref<64x32xf32>) {
    linalg.matmul ins(%A, %B : memref<64x128xf32>, memref<128x32xf32>)
                  outs(%C : memref<64x32xf32>)
    func.return
  }

  func.func @main() {
    %c1 = arith.constant 1.0 : f32
    %c0 = arith.constant 0.0 : f32
    %A = memref.alloc() : memref<64x128xf32>
    %B = memref.alloc() : memref<128x32xf32>
    %C = memref.alloc() : memref<64x32xf32>

    linalg.fill ins(%c1 : f32) outs(%A : memref<64x128xf32>)
    linalg.fill ins(%c1 : f32) outs(%B : memref<128x32xf32>)
    linalg.fill ins(%c0 : f32) outs(%C : memref<64x32xf32>)

    %n    = arith.constant 200 : index
    %zero = arith.constant 0 : index
    %one  = arith.constant 1 : index
    scf.for %i = %zero to %n step %one {
      func.call @mm(%A, %B, %C)
        : (memref<64x128xf32>, memref<128x32xf32>, memref<64x32xf32>) -> ()
      scf.yield
    }

    // 取一个元素避免未来 DCE
    %i0 = arith.constant 0 : index
    %j0 = arith.constant 0 : index
    %v  = memref.load %C[%i0, %j0] : memref<64x32xf32>
    func.return
  }
}
