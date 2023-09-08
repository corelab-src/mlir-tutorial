module {
  func.func @main(%arg0: tensor<2x3xf64>, %arg1:tensor<2x3xf64>) {
    %0 = "toy.add"(%arg0, %arg1) : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<2x3xf64>
    %1 = "toy.sub"(%0, %arg1) : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<2x3xf64>
    %2 = "toy.add"(%1, %arg1) : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<2x3xf64>
    toy.return %2 : tensor<2x3xf64>
  }
}
