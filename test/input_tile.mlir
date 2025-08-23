module {
  func.func @mm(%A: tensor<64x128xf32>,
                %B: tensor<128x32xf32>,
                %C: tensor<64x32xf32>)
      -> tensor<64x32xf32>
      attributes { tile.sizes = [8, 8, 32] } {   
    %0 = linalg.matmul
          ins(%A, %B : tensor<64x128xf32>, tensor<128x32xf32>)
         outs(%C : tensor<64x32xf32>) -> tensor<64x32xf32>
    return %0 : tensor<64x32xf32>
  }
}
