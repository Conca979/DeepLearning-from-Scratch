import matplotlib.pyplot as plt
import math

class basicLinearRegression: # gradinet decsent approach
  def __init__(self, inputTrainingSet, inputParameters = None, epsilon = 0.001, learningRate = 0.001, DataSplittingMethod = None):
    # get dataSet
    if isinstance(inputTrainingSet, str):
      with open(inputTrainingSet) as f:
        next(f)
        self.trainingSet = [[float(val) for val in (line.split()[1:] if DataSplittingMethod is None else line.split(DataSplittingMethod)[1:])] for line in f]
    else:
      self.trainingSet = inputTrainingSet
    #
    if inputParameters is None:
      self.parameters = [0]*len(self.trainingSet[0])
    else:
      self.parameters = inputParameters
    #
    self.sizeDataSet = len(self.trainingSet)
    self.learningRate = learningRate
    self.epsilon = epsilon
    self.iterationCount = 0
    self._featureScaling()

  def _featureScaling(self): # Z-score standardization
    # new scaled feature = (feature - theMean) / theStandardDeviation
    for featureIndex in range(len(self.trainingSet[0]) - 1):
      mean = sum([self.trainingSet[i][featureIndex] for i in range(self.sizeDataSet)]) / self.sizeDataSet
      stdDeviation = (sum([(self.trainingSet[i][featureIndex] - mean)**2 for i in range(self.sizeDataSet)]) / self.sizeDataSet)**0.5
      for i in range(len(self.trainingSet)):
        if stdDeviation > 0:
          self.trainingSet[i][featureIndex] = (self.trainingSet[i][featureIndex] - mean) / stdDeviation
        else:
          self.trainingSet[i][featureIndex] = 0     

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
    result = 0
    for sampleIndex in range(self.sizeDataSet):
      result += (self.trainingSet[sampleIndex][-1] - self.predict(sampleIndex))*(1 if featureIndex == 0 else self.trainingSet[sampleIndex][featureIndex - 1])

    return result*(self.learningRate / self.sizeDataSet)

  def fitModel(self, log = None):
    currentCost = self.computeMse()
    self.costHistory = [currentCost]
    if log:
      with open('iterations.txt', mode = 'w') as f: pass
      with open('iterations.txt', mode = 'a') as f:
        f.write(f"[weights], current MSE, shifting in MSE \n")
        while True:
          self.iterationCount += 1
          updatePar = self.parameters.copy()
          for featureIndex in range(len(self.parameters)):
            updatePar[featureIndex] += self.computeGradient(featureIndex)

          self.parameters = updatePar
          updateCost = self.computeMse()
          self.costHistory.append(updateCost)
          shifting = abs((currentCost - updateCost) / currentCost)
          #
          f.write(f"{self.parameters}, {updateCost}, {shifting} \n")
          #
          if (shifting) < self.epsilon or self.iterationCount > 200000:
            break
          else:
            currentCost = updateCost
    else:
      while True:
        self.iterationCount += 1
        updatePar = self.parameters.copy()
        for featureIndex in range(len(self.parameters)):
          updatePar[featureIndex] += self.computeGradient(featureIndex)

        self.parameters = updatePar
        updateCost = self.computeMse()
        self.costHistory.append(updateCost)
        shifting = abs((currentCost - updateCost) / currentCost)
        #
        #
        if (shifting) < self.epsilon or self.iterationCount > 200000:
          break
        else:
          currentCost = updateCost


  def showModel(self, graph = None): # type = Prediction Accurary by default, Cost vs Iteration for any passed value
    if graph is None:
      predictedValue = [self.predict(sampleIndex) for sampleIndex in range(len(self.trainingSet))]
      actualValue = [ys[-1] for ys in self.trainingSet]

      plt.scatter(actualValue, predictedValue, c = 'r')
      plt.xlabel('Predicted Values')
      plt.ylabel('Actual Values')
      # draw the accurary lines
      upperBound = max((max(predictedValue), max(actualValue)))
      lowerBound = min((min(predictedValue), min(actualValue)))
      plt.plot([lowerBound, upperBound], [lowerBound, upperBound], c = 'b', alpha = 0.5)

      # t1 = [[actualValue[k], predictedValue[k]] for k in range(len(self.trainingSet))]
      # model = basicLinearRegression(t1, epsilon= self.epsilon, learningRate= self.learningRate)
      # model.fitModel()
      # t = model.parameters
      # plt.plot([lowerBound, upperBound], [lowerBound*t[1] + t[0], upperBound*t[1] + t[0]], c = 'r')
      #
      plt.title('Prediction Accurary')
    else:
      plt.xlabel('Iteration (Log scale)')
      plt.ylabel('Cost - MSE')
      plt.xscale('log')
      plt.title('Cost vs iteration')
      plt.plot(self.costHistory)

    plt.show()

class basicLogisticRegression: # grandient ascent approach
  def __init__(self, dataSet, weights = None, epsilon = 0.00001,  learningRate = 0.001):
    self.learningRate = learningRate
    self.epsilon = epsilon
    self.iterationCount = 0
    #
    if isinstance(dataSet, str):
      with open(dataSet) as f:
        self.dataSet = [[float(val) for val in line.split()] for line in f]
    self.sizeDataSet = len(self.dataSet)
    #
    if weights is None:
      self.weights = [0]*len(self.dataSet[0])
    else:
      self.weights = weights
    #

  def predict(self, sampleIndex):
    result = self.weights[0]
    for i in range(len(self.weights) - 1):
      result += self.weights[i + 1]*self.dataSet[sampleIndex][i]
    
    return 1 / (1 + math.e**(-result))
  
  def computeLogLikelihood(self):
    llh = 0 # log Likelihood
    for sampleIndex in range(self.sizeDataSet):
      llh += self.dataSet[sampleIndex][-1]*math.log(self.predict(sampleIndex)) + (1 - self.dataSet[sampleIndex][-1])*math.log(1 - self.predict(sampleIndex))

    return llh
  
  def computeGradient(self, freatureIndex):
    grd = 0
    
    for sampleIndex in range(self.sizeDataSet):
      grd += (self.dataSet[sampleIndex][-1] - self.predict(sampleIndex))*(1 if freatureIndex == 0 else self.dataSet[sampleIndex][freatureIndex - 1])

    return self.learningRate*grd
  
  def fitModel(self):
    curLlh = self.computeLogLikelihood()

    with open('iterations.txt', mode = 'w') as f: pass
    with open('iterations.txt', mode = 'a') as f:
      f.write(f"""["Weights"], current log likelihood, shifting in Llh \n {self.weights}, {curLlh}, 0 \n""")
      while True:
        #
        updateWeights = self.weights.copy()
        #
        self.iterationCount += 1
        for freatureIndex in range(len(self.weights)):
          updateWeights[freatureIndex] += self.computeGradient(freatureIndex) 

        self.weights = updateWeights
        newLlh = self.computeLogLikelihood()
        
        change = abs((newLlh - curLlh) / curLlh)
        f.write(f"{self.weights}, {curLlh}, {change} \n")
        if change < self.epsilon:
          break
        else:
          curLlh = newLlh

  def showTestTable(self):
    with open('data2TestSet.txt') as f:
      testSet = [[float(val) for val in line.split()] for line in f]

    self.dataSet = testSet
    predictedProbabilities = [self.predict(sampleIndex) for sampleIndex in range(len(testSet))]
    predictedLabels = [(1 if val >= 0.5 else 0) for val in predictedProbabilities]
    tableData = []
    for i in range(len(testSet)):
      isCorrect = "Yes" if predictedLabels[i] == testSet[i][-1] else "No"
      tableData.append([
        testSet[i][0], 
        testSet[i][1], 
        round(predictedProbabilities[i], 7), 
        predictedLabels[i], 
        isCorrect
      ])

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis('tight')
    ax.axis('off')

    columns = ("Feature (x)", "Actual Label", "Probability", "Predicted", "Correct?")
    table = ax.table(cellText=tableData, colLabels=columns, loc='center', cellLoc='center')

    # Color coding for the "Correct?" column
    for i in range(len(tableData)):
        cell = table[(i + 1, 4)] # +1 because of header row
        if tableData[i][4] == "No":
            cell.set_facecolor("#ffcccc") # Light red for errors
        else:
            cell.set_facecolor("#ccffcc") # Light green for success

    plt.title("Model Verification Table", fontsize=14, pad=20)
    plt.show()
    # print([True if (1 if predictedProbability[i] >= 0.5 else 0) == testSet[i][-1] else False for i in range(len(testSet))])


model = basicLinearRegression('Advertising.csv', epsilon = 0.00000001, learningRate = 0.0001, DataSplittingMethod= ',')
model.fitModel(log= False)
print(model.iterationCount)
model.showModel(graph= 'C')

# model = basicLogisticRegression('data2.txt', epsilon= 0.00001, learningRate= 0.001)
# model.fitModel()
# print(model.iterationCount)
# model.showTestTable()