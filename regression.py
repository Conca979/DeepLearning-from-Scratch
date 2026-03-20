import matplotlib.pyplot as plt
import math
import random

class modelSetup:
  def __init__(self, inputDataSet, initWeights, epsilon, learningRate, parsingData, featureScaling):
    # get dataSet
    self.parsingData = parsingData
    self._dataPreparation(inputDataSet)
    # model initialization
    self.sizeDataSet = len(self.dataSet)
    self.featureCount = len(self.dataSet[0]) - 1
    self.learningRate = learningRate
    self.epsilon = epsilon
    self.iterationCount = 0
    # initialize weights
    if initWeights is None:
      self.weights = [0]*len(self.dataSet[0])
    else:
      self.weights = initWeights
    # Scaling data to avoid overFlow
    self.featureScaling = False
    if featureScaling:
      self.featureScaling = self._featureScaling()

  def _featureScaling(self): # Z-score standardization
    # new scaled feature = (oldFeature - theMean) / theStandardDeviation
    scalingParams = [] # store the (mean, stdDev)
    for featureIndex in range(self.featureCount):
      mean = sum([self.dataSet[i][featureIndex] / self.sizeDataSet for i in range(self.sizeDataSet)])
      stdDev = (sum([((self.dataSet[i][featureIndex] - mean)**2) / self.sizeDataSet for i in range(self.sizeDataSet)]))**0.5
      scalingParams.append((mean, stdDev))
      for i in range(self.sizeDataSet):
        if stdDev > 0:
          self.dataSet[i][featureIndex] = (self.dataSet[i][featureIndex] - mean) / stdDev
        else:
          self.dataSet[i][featureIndex] = 0

    return scalingParams

  def _dataPreparation(self, inputDataSet):
    # preparing the dataSet
    if isinstance(inputDataSet, str):
      with open(inputDataSet) as f:
        if self.parsingData[1]: next(f) # skip the header line
        offSet = 1 if self.parsingData[2] else 0 # skip the index column
        a = [[float(val) for val in line.split(self.parsingData[0])[offSet:]] for line in f]
      random.shuffle(a)
    # split dataSet into "80 training / 20 testing"
      self.dataSet = a[:int(len(a)*0.8)]
      self.testSet = a[int(len(a)*0.8):]
    else:
      self.dataSet = inputDataSet[:int(len(inputDataSet)*0.8)]
      self.testSet = inputDataSet[int(len(inputDataSet)*0.8):]

class basicLinearRegression(modelSetup): # gradinet decsent approach
  def __init__(self, 
               inputDataSet, 
               initWeights= None, 
               epsilon= 0.000001, 
               learningRate= 0.001, 
               parsingData= (None,None,None),  # [0]: comma/pip/semicolumn/space-separated value
                                                # [1]: header included in file
                                                # [2]: indexing column included
               featureScaling= True,
               outComeScaling= True):
    super().__init__(inputDataSet, initWeights, epsilon, learningRate, parsingData, featureScaling)
    # scaling outPut/outCome also. But why? Because we are doing the arithmetic computation
    # and we have to deal with the overflow lol
    self.outComeScaling = False
    if outComeScaling:
      self.outComeScaling = self._outComeScaling()

  def _outComeScaling(self): # Z-score standardization
    # new scaled feature = (oldFeature - theMean) / theStandardDeviation
    mean = sum([self.dataSet[i][-1] / self.sizeDataSet for i in range(self.sizeDataSet)])
    stdDev = (sum([((self.dataSet[i][-1] - mean)**2) / self.sizeDataSet for i in range(self.sizeDataSet)]))**0.5
    for i in range(self.sizeDataSet):
      if stdDev > 0:
        self.dataSet[i][-1] = (self.dataSet[i][-1] - mean) / stdDev
      else:
        self.dataSet[i][-1] = 0

    return mean, stdDev
  
  def _modelEvalutaion(self): # R-square evaluation
    actualMean = sum([val[-1] for val in self.dataSet]) / self.sizeDataSet
    sst = 0 # sumOfSquareErrors: proportional to the variance of the actual data
    sse = 0 # sumOfSquareOfResidualErrors: proportional to the variance of the prodicted data
    for sampleIndex in range(self.sizeDataSet):
      sst += (self.dataSet[sampleIndex][-1] - actualMean)**2
      sse += (self.dataSet[sampleIndex][-1] - self._predict(sampleIndex))**2
    
    goodnessOfFit = (1 - (sse / sst))
    return sst, sse, goodnessOfFit
  
  def predict(self, inputData):
    if len(inputData) == self.featureCount:
      result = self.weights[0]
      for i in range(self.featureCount):
        if self.featureScaling:
          mean, stdDev = self.featureScaling[i][0], self.featureScaling[i][1]
          inputData[i] = (inputData[i] - mean) / stdDev
        result += self.weights[i + 1]*inputData[i]

      if self.outComeScaling:
        result = result*self.outComeScaling[1] + self.outComeScaling[0]
      return result
    else:
      print("Input size does not match the model! pls check again")
      return None

  def _predict(self, sampleIndex): # from the hypothesis function
    result = self.weights[0]
    for i in range(self.featureCount):
      result += self.weights[i + 1]*self.dataSet[sampleIndex][i]

    return result
  
  def computeMse(self): # the mean squared error
    e = 0
    for sampleIndex in range(self.sizeDataSet):
      e += ((self._predict(sampleIndex)) - self.dataSet[sampleIndex][-1])**2

    return e / (self.sizeDataSet*2)
  
  def computeGradient(self, featureIndex):
    result = 0
    for sampleIndex in range(self.sizeDataSet):
      result += (self.dataSet[sampleIndex][-1] - self._predict(sampleIndex))*(1 if featureIndex == 0 else self.dataSet[sampleIndex][featureIndex - 1])

    return result*(self.learningRate / self.sizeDataSet)

  def fitModel(self, log = None):
    currentCost = self.computeMse()
    self.costHistory = [currentCost]
    if log:
      with open('iterations.txt', mode = 'w') as f: pass
      with open('iterations.txt', mode = 'a') as f:
        f.write(f"[weights], current MSE, shifting in MSE \n")
        f.write(f"{self.weights}, {currentCost}, 0 \n")
        while True:
          self.iterationCount += 1
          updWeights = self.weights.copy()
          for weightIndex in range(self.featureCount + 1):
            updWeights[weightIndex] += self.computeGradient(weightIndex)

          self.weights = updWeights
          updateCost = self.computeMse()
          self.costHistory.append(updateCost)
          shifting = abs((currentCost - updateCost) / currentCost)
          #
          f.write(f"{self.weights}, {updateCost}, {shifting} \n")
          #
          if (shifting) < self.epsilon or self.iterationCount > 200000:
            break
          else:
            currentCost = updateCost
    else:
      while True:
        self.iterationCount += 1
        updatePar = self.weights.copy()
        for weightIndex in range(self.featureCount + 1):
          updatePar[weightIndex] += self.computeGradient(weightIndex)

        self.weights = updatePar
        updateCost = self.computeMse()
        self.costHistory.append(updateCost)
        shifting = abs((currentCost - updateCost) / currentCost)
        #
        #
        if (shifting) < self.epsilon or self.iterationCount > 200000:
          break
        else:
          currentCost = updateCost

  def showModel(self, graph= None, log= True): # graphType = "Prediction Accurary" by default, "Cost vs Iteration" for any passed value
    if graph is None:
      predictedValue = [self._predict(sampleIndex) for sampleIndex in range(self.sizeDataSet)]
      actualValue = [ys[-1] for ys in self.dataSet]

      plt.scatter(actualValue, predictedValue, c = 'r')
      plt.xlabel('Predicted Values')
      plt.ylabel('Actual Values')
      # draw the accurary lines
      upperBound = max((max(predictedValue), max(actualValue)))
      lowerBound = min((min(predictedValue), min(actualValue)))
      plt.plot([lowerBound, upperBound], [lowerBound, upperBound], c = 'b', alpha = 0.5)

      # t1 = [[actualValue[k], predictedValue[k]] for k in range(self.sizeDataSet)]
      # model = basicLinearRegression(t1, epsilon= self.epsilon, learningRate= self.learningRate)
      # model.fitModel()
      # t = model.weights
      # plt.plot([lowerBound, upperBound], [lowerBound*t[1] + t[0], upperBound*t[1] + t[0]], c = 'r')
      #
      plt.title('Prediction Accurary')
    else:
      plt.xlabel(f'Iterations {'Log scalse' if log else ''}')
      plt.ylabel('Cost - MSE')
      plt.title('Cost vs iteration')
      if log: plt.xscale('log')
      plt.plot(self.costHistory)

    plt.show()

class basicLogisticRegression(modelSetup): # grandient ascent approach
  def __init__(self, 
               inputDataSet, 
               initWeights= None, 
               epsilon= 0.000001, 
               learningRate= 0.001, 
               parsingData= (None,None,None),  # [0]: comma/pip/semicolumn/space-separated value
                                                # [1]: header included in file
                                                # [2]: indexing column included
               featureScaling= True):
    super().__init__(inputDataSet, initWeights, epsilon, learningRate, parsingData, featureScaling)

  def predict(self, inputData):
    result = self.weights[0]
    for i in range(self.featureCount):
      if self.featureScaling:
        mean, stdDev = self.featureScaling[i]
        inputData[i] = (inputData[i] - mean) / stdDev      
      result += self.weights[i + 1]*inputData[i]

    return 1 / (1 + math.e**(-result))

  def _predict(self, sampleIndex):
    result = self.weights[0]
    for i in range(self.featureCount):
      result += self.weights[i + 1]*self.dataSet[sampleIndex][i]
    
    return 1 / (1 + math.e**(-result))
  
  def computeLogLikelihood(self):
    llh = 0 # log Likelihood
    for sampleIndex in range(self.sizeDataSet):
      llh += self.dataSet[sampleIndex][-1]*math.log(self._predict(sampleIndex)) + (1 - self.dataSet[sampleIndex][-1])*math.log(1 - self._predict(sampleIndex))

    return llh
  
  def computeGradient(self, freatureIndex):
    grd = 0
    for sampleIndex in range(self.sizeDataSet):
      grd += (self.dataSet[sampleIndex][-1] - self._predict(sampleIndex))*(1 if freatureIndex == 0 else self.dataSet[sampleIndex][freatureIndex - 1])

    return self.learningRate*grd
  
  def fitModel(self, log= None):
    curLlh = self.computeLogLikelihood()
    self.costHistory = [math.e**curLlh]
    if log:
      with open('iterations.txt', mode = 'w') as f: pass
      with open('iterations.txt', mode = 'a') as f:
        f.write(f"""["Weights"], current log likelihood, shifting in Llh \n {self.weights}, {curLlh}, 0 \n""")
        while True:
          updateWeights = self.weights.copy()
          self.iterationCount += 1
          for freatureIndex in range(len(self.weights)):
            updateWeights[freatureIndex] += self.computeGradient(freatureIndex) 

          self.weights = updateWeights
          newLlh = self.computeLogLikelihood()
          self.costHistory.append(math.e**newLlh)
          change = abs((newLlh - curLlh) / curLlh)
          f.write(f"{self.weights}, {curLlh}, {change} \n")
          if change < self.epsilon:
            break
          else:
            curLlh = newLlh
    else:
      while True:
        updateWeights = self.weights.copy()
        self.iterationCount += 1
        for freatureIndex in range(len(self.weights)):
          updateWeights[freatureIndex] += self.computeGradient(freatureIndex) 

        self.weights = updateWeights
        newLlh = self.computeLogLikelihood()
        self.costHistory.append(math.e**newLlh)
        change = abs((newLlh - curLlh) / curLlh)
        if change < self.epsilon:
          break
        else:
          curLlh = newLlh

  def showCostTrend(self, log= True):
    plt.xlabel(f'Iterations {'Log scalse' if log else ''}')
    plt.ylabel('Cost - MSE')
    plt.title('Cost vs iteration')
    if log: plt.xscale('log')
    plt.plot(self.costHistory)
    plt.show()

  def showTestTable(self):
    predictedProbabilities = [self.predict(inputData) for inputData in self.testSet]
    predictedLabels = [(1 if val >= 0.5 else 0) for val in predictedProbabilities]
    tableData = []
    for i in range(len(self.testSet)):
      isCorrect = "Yes" if predictedLabels[i] == self.testSet[i][-1] else "No"
      tableData.append([
        self.testSet[i][-1], 
        round(predictedProbabilities[i], 7), 
        predictedLabels[i], 
        isCorrect
      ])

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis('tight')
    ax.axis('off')

    columns = ("Actual Label", "Probability", "Predicted", "Correct?")
    table = ax.table(cellText=tableData, colLabels=columns, loc='center', cellLoc='center')

    # color coding for the "Correct?" column
    for i in range(len(tableData)):
        cell = table[(i + 1, 3)] # +1 because of header row
        if tableData[i][3] == "No":
            cell.set_facecolor("#ffcccc") # red for errors
        else:
            cell.set_facecolor("#ccffcc") # green for success

    plt.title("Model Verification Table", fontsize=14, pad=20)
    plt.show()
    # print([True if (1 if predictedProbability[i] >= 0.5 else 0) == testSet[i][-1] else False for i in range(len(testSet))])
