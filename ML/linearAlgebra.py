def matrixDotProduct(a: list[list[int|float]], b: list[int|float]) -> list[int|float]:
  # This function takes in a matrix 'a' and a vector 'b', and returns the result of the matrix-vector multiplication. -1 is returned if encounter invalid inputs
  if len(b) != len(a[0]):
    return -1
  else:
    result = [i for i in range(len(a))]
    for i in range(len(a)):
      result[i] = sum([b[j]*a[i][j] for j in range(len(b))])
    
    return result

def transposeMatrix(a: list[list[int|float]]) -> list[list[int|float]]:
  #This function takes in a matrix 'a' and returns its tranpose. The transpose of a matrix is obtained by swapping its rows with its columns. For example, if the input matrix 'a' is:
  # [[1,2,3],[4,5,6]]
  # then the output of the function will be:
  # [[1,4],[2,5],[3,6]]
  return [[a[i][j] for i in range(len(a))] for j in range(len(a[0]))]