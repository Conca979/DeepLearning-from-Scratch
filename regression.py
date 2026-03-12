import matplotlib.pyplot as plt

class basicLinearRegression:
  def __init__(self, inputTrainingSet, inputParameters = None, epsilon = 0.001, learningRate = 0.001):
    if isinstance(inputTrainingSet, str):
      with open(inputTrainingSet) as f:
        self.trainingSet = [[float(val) for val in line.split()] for line in f]
    else:
      self.trainingSet = inputTrainingSet
    #
    if inputParameters is None:
      self.parameters = [0]*len(self.trainingSet[0])
    else:
      self.parameters = inputParameters
    #
    self.learningRate = learningRate
    self.epsilon = epsilon
    self.iterationCount = 0

  def predict(self, sampleIndex): # from the hypothesis function
    result = self.parameters[0]
    for i in range(len(self.trainingSet[0]) - 1):
      result += self.parameters[i + 1]*self.trainingSet[sampleIndex][i]

    return result
  
  def computeMse(self): # the mean squared error
    e = 0
    m = len(self.trainingSet)
    for sampleIndex in range(m):
      e += ((self.predict(sampleIndex)) - self.trainingSet[sampleIndex][-1])**2

    return e / (m*2)
  
  def computeGradient(self, featureIndex): # from cost function
    m = len(self.trainingSet)
    result = 0
    for sampleIndex in range(m):
      result += (self.trainingSet[sampleIndex][-1] - self.predict(sampleIndex))*(1 if featureIndex == 0 else self.trainingSet[sampleIndex][featureIndex - 1])

    return result*(self.learningRate / m)
  
  def fitModel(self):
    currentCost = self.computeMse()

    while True:
      self.iterationCount += 1
      updatePar = self.parameters.copy()
      for featureIndex in range(len(self.parameters)):
        updatePar[featureIndex] += self.computeGradient(featureIndex)

      self.parameters = updatePar
      updateCost = self.computeMse()
      if ((currentCost - updateCost) / currentCost) < self.epsilon:
        break
      else:
        currentCost = updateCost

  def showModel(self):
    predictedValue = [self.predict(sampleIndex) for sampleIndex in range(len(self.trainingSet))]
    actualValue = [ys[-1] for ys in self.trainingSet]

    plt.scatter(actualValue, predictedValue, c = 'r')
    plt.xlabel('Predicted Values')
    plt.ylabel('Actual Values')
    # draw the accurary lines
    upperBound = max((max(predictedValue), max(actualValue)))
    lowerBound = min((min(predictedValue), min(actualValue)))
    plt.plot([lowerBound, upperBound], [lowerBound, upperBound], c = 'b', alpha = 0.5)

    t1 = [[actualValue[k], predictedValue[k]] for k in range(len(self.trainingSet))]
    model = basicLinearRegression(t1, epsilon= self.epsilon, learningRate= self.learningRate)
    model.fitModel()
    t = model.parameters
    plt.plot([lowerBound, upperBound], [lowerBound*t[1] + t[0], upperBound*t[1] + t[0]], c = 'r')
    #
    plt.title('Prediction Accurary')
    plt.show()


model = basicLinearRegression('data.txt', epsilon = 0.00000000000001, learningRate = 0.0001)
model.fitModel()
print(model.iterationCount)
model.showModel()