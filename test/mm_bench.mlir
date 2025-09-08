module {
  // 纯 tensor 版 mm，便于用 -linalg-tile 对 linalg.matmul 直接分块
  func.func @mm(%A: tensor<64x128xf32>,
                %B: tensor<128x32xf32>,
                %C: tensor<64x32xf32>) -> tensor<64x32xf32> {
    %0 = linalg.matmul
          ins(%A, %B : tensor<64x128xf32>, tensor<128x32xf32>)
         outs(%C : tensor<64x32xf32>) -> tensor<64x32xf32>
    return %0 : tensor<64x32xf32>
  }

  // 多次调用 @mm 放大运行时间（N_RUNS 次）
  func.func @main() {
    %c1 = arith.constant 1.0 : f32
    %c0 = arith.constant 0.0 : f32

    // 准备 A, B, C（tensor 版本）
    %Ainit = tensor.empty() : tensor<64x128xf32>
    %Binit = tensor.empty() : tensor<128x32xf32>
    %Cinit = tensor.empty() : tensor<64x32xf32>

    %A = linalg.fill ins(%c1 : f32) outs(%Ainit : tensor<64x128xf32>) -> tensor<64x128xf32>
    %B = linalg.fill ins(%c1 : f32) outs(%Binit : tensor<128x32xf32>) -> tensor<128x32xf32>
    %C0 = linalg.fill ins(%c0 : f32) outs(%Cinit : tensor<64x32xf32>) -> tensor<64x32xf32>

    // N_RUNS = 200
    %c200 = arith.constant 200 : index
    %c0i  = arith.constant 0 : index
    %c1i  = arith.constant 1 : index

    %Cfinal = scf.for %it = %c0i to %c200 step %c1i
                iter_args(%Ccur = %C0) -> (tensor<64x32xf32>) {
      %Cnext = func.call @mm(%A, %B, %Ccur)
                : (tensor<64x128xf32>, tensor<128x32xf32>, tensor<64x32xf32>)
                  -> tensor<64x32xf32>
      scf.yield %Cnext : tensor<64x32xf32>
    }

    // 可选：防止将来可能的 DCE
    %i0 = arith.constant 0 : index
    %j0 = arith.constant 0 : index
    %e00 = tensor.extract %Cfinal[%i0, %j0] : tensor<64x32xf32>

    func.return
  }
}
