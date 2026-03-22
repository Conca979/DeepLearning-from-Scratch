import matplotlib.pyplot as plt
import seaborn as sborn
import numpy as np
import pandas as pd
import json
from datetime import datetime

class PolyFeatureTransform:
  def __init__(self, featureCount, degree= 1):
    self.relations = []
    self.degree = degree
    self.featureCount = featureCount
    # initializing relations
    self._getFeatureRelation()

  def _gen(self, cur, start, degNeeded):
    if degNeeded == 0:
      self.relations.append(cur.copy())
      return
    for e in range(start, self.featureCount):
      cur.append(e)
      self._gen(cur, e, degNeeded - 1)
      cur.pop()

  def _getFeatureRelation(self):
    for degree in range(self.degree + 1):
      self._gen([], 0, degree)
  
  def transform(self, inputFeature):
    newFeatures = []
    for relation in self.relations:
      val = 1
      for featureIndex in relation:
        val *= inputFeature[featureIndex]
      newFeatures.append(val)
    return newFeatures

class ModelSetup:
  def __init__(self, filePath, initWeights, epsilon, learningRate, parsingData, featureScaling, modelDegree, iterationLogTrigger):
    # Preparing dataSet
    self.filePath = filePath
    self.parsingData = parsingData
    self._dataPreparation(self.filePath)
    self.sizeDataSet = self.dataSet.shape[0]

    # Separate feature (X) and label/outCome (Y) early for easier matrix math
    self.xTrain = self.dataSet[:, :-1]
    self.yTrain = self.dataSet[:, -1]
    self.xTest = self.testSet[:, :-1]
    self.yTest = self.testSet[:, -1]

    # Scaling data to avoid overFlow
    self.scalers = []
    self.featureScaling = featureScaling
    self._featureScaling()

    # Model polynomial transformation
    self.degree = modelDegree
    self.polyTransform()
    self._featureScaling(skipBias= True)

    # Model initialization
    self.featureCount = self.xTrain.shape[1]
    self.learningRate = learningRate
    self.epsilon = epsilon
    self.iterationCount = 0
    self.logTrigger = iterationLogTrigger

    # Initialize weights
    if initWeights is None:
      self.weights = np.zeros(self.featureCount)
    else:
      self.weights = np.array(initWeights, dtype= float)

  def _featureScaling(self, skipBias= False): # Z-score standardization
    # new scaled feature = (oldFeature - theMean) / theStandardDeviation
    if self.featureScaling:
      offSet = 1 if skipBias else 0
      means = np.mean(self.xTrain[:, offSet:], axis= 0)
      stdDevs = np.std(self.xTrain[:, offSet:], axis= 0)
      
      # Prevent division by 0
      stdDevs[stdDevs == 0] = 1

      # Vectorize subtraction and division
      self.xTrain[:, offSet:] = (self.xTrain[:, offSet:] - means) / stdDevs
      self.scalers.append((means, stdDevs))

  def _dataPreparation(self, filePath):
    skipHeadLine = 1 if self.parsingData[1] else 0
    data = np.loadtxt(filePath,
                      delimiter= self.parsingData[0],
                      skiprows= skipHeadLine)
    if self.parsingData[2]: data = data[:, 1:]
    # Splitting data into 80/20 training/testting
    np.random.shuffle(data) # Optional
    splitIndex = int(data.shape[0]*0.8)
    self.dataSet = data[:splitIndex]
    self.testSet = data[splitIndex:]

  def polyTransform(self):
    self.polyEngine = PolyFeatureTransform(self.xTrain.shape[1], degree= self.degree)
    newInput = []
    for xSet in self.xTrain:
      newInput.append(self.polyEngine.transform(xSet))
    
    self.xTrain = np.array(newInput)

  def saveModel(self, modelSaveFile, model):
    state = {
              'ModelId': datetime.now().strftime('%Y%m%d-%H%M%S'),
              'modelType': self.__class__.__name__,
              'dataSet': self.filePath,
              'hyperparameters': {
                                    'learningRate': self.learningRate,
                                    'epsilon': self.epsilon,
                                    'degree': self.degree,
                                    'featureCount': self.featureCount,
              },
              'trainingResult': {
                                  'iterations': self.iterationCount,
                                  'weights': self.weights.tolist()
              }
      }
    
    print("-"*20, model.__class__.__name__, "-"*20)
    if model.__class__.__name__ == "BasicLinearRegression":
      state['trainingResult']['goodnessOfFit'] = model.goodnessOfFit
    elif model.__class__.__name__ == "BasicLogisticRegression":
      state['trainingResult']['ConfusionMatrix'] = model.modelEvaluation

    with open(modelSaveFile, mode= 'a') as f:
      json.dump(state, f, indent= 2)
      f.write(f"\n{"-"*100}\n")
    print(f"Model'state saved to file path {modelSaveFile}")

class BasicLinearRegression(ModelSetup): # gradient decsent approach
  def __init__(self, 
               filePath,
               initWeights= None, 
               epsilon= 0.000001, 
               learningRate= 0.001, 
               parsingData= (None,None,None),   # [0]: delimiter
                                                # [1]: header included in file
                                                # [2]: indexing column included
               featureScaling= True,
               outComeScaling= True,
               modelDegree= 1,
               iterationLogTrigger= 1000):
    super().__init__(filePath, initWeights, epsilon, learningRate, parsingData, featureScaling, modelDegree, iterationLogTrigger)
    # scaling outPut/outCome also. But why? Because we are doing the arithmetic computation
    # and we have to deal with the overflow lol
    self.outComeScaling = False
    if outComeScaling:
      self.outComeScaling = self._outComeScaling()

  def _outComeScaling(self): # Z-score standardization
    # new scaled feature = (oldFeature - theMean) / theStandardDeviation
    mean = np.mean(self.yTrain)
    stdDev = np.std(self.yTrain)
    self.yTrain = (self.yTrain - mean) / stdDev

    return mean, stdDev
  
  def _modelEvaluaion(self): # R-square evaluation
    actualMean = np.mean(self.yTrain)

    # SumOfSquareErrors: proportional to the variance of the actual data
    sst = np.sum((self.yTrain - actualMean)**2)

    # SumOfSquareOfResidualErrors: proportional to the variance of the prodicted data
    sse = np.sum((self.yTrain - self._predict())**2)
    
    goodnessOfFit = (1 - (sse / sst))
    self.goodnessOfFit = goodnessOfFit
    return sst, sse, goodnessOfFit

  def predict(self, inputData):
    if len(inputData) == len(self.scalers[0][0]):
    # input -> scaleBefore -> poly -> scalseAfter -> prediction
    # ----------------------------------------------------------
    # 1st scale / scaleBefore
      means, stdDevs = self.scalers[0]
      inputData = (inputData - means) / stdDevs
    # polynomial extention
      inputData = self.polyEngine.transform(inputData)
    # 2nd scale / scaleAfter
      means, stdDevs = self.scalers[1]
      inputData[1:] = (inputData[1:] - means) / stdDevs
    # predict
      result = np.dot(self.weights, inputData)

      if self.outComeScaling:
        result = result*self.outComeScaling[1] + self.outComeScaling[0]
      return result
    else:
      print("Input size does not match the model! pls check again")
      return None

  def _predict(self): # predict the whole dataSet
    return np.dot(self.xTrain, self.weights)
  
  def computeMse(self, prediction): # the mean squared error
    errors = prediction - self.yTrain
    return np.sum(errors**2) / (2*self.yTrain.shape[0])
  
  def computeGradient(self, prediction):
    errors = prediction - self.yTrain
    grd = np.dot(self.xTrain.T, errors)
    return grd*(self.learningRate / self.yTrain.shape[0])

  def fitModel(self, log= None, saveModelState= None):
    predictions = self._predict()
    currentCost = self.computeMse(predictions)
    self.costHistory = [currentCost]
    # Checking file before fitting
    if saveModelState:
      with open(saveModelState): pass

    # User want to store model training states
    if log:
      with open('data/iterations.txt', mode = 'w') as f: pass
      with open('data/iterations.txt', mode = 'a') as f:
        f.write(f"[weights], current MSE, shifting in MSE \n")
        f.write(f"{self.weights}, {currentCost}, 0 \n")

    # Start training
    while True:
      self.iterationCount += 1
      
      # Update weights
      self.weights -= self.computeGradient(predictions)
      predictions = self._predict()

      # Capture model state
      updateCost = self.computeMse(predictions)
      self.costHistory.append(updateCost)
      costShifting = abs((currentCost - updateCost) / currentCost)

      # Log
      if self.iterationCount % self.logTrigger == 0:
        print(f"||| {self.iterationCount}\t iterations passed; model'current cost: {currentCost}")
        if log :f.write(f"{self.weights}, {updateCost}, {costShifting} \n")
      
      # Stopping condition
      if (costShifting) < self.epsilon or updateCost > currentCost:
        self._modelEvaluaion()
        if saveModelState:
          self.saveModel(modelSaveFile= saveModelState, model= self)
        break
      else:
        currentCost = updateCost

  def showModel(self, graph= None, logScale= True): # graphType = "Prediction Accurary" by default, "Cost vs Iteration" for any passed value
    if graph is None:
      predictedValue = self._predict()

      plt.scatter(self.yTrain, predictedValue, c = 'r')
      plt.ylabel('Predicted Values')
      plt.xlabel('Actual Values')
      plt.title('Prediction Accurary')

      # Draw the accurary lines
      upperBound = max((max(predictedValue), max(self.yTrain)))
      lowerBound = min((min(predictedValue), min(self.yTrain)))
      plt.plot([lowerBound, upperBound], [lowerBound, upperBound], c = 'b', alpha = 0.5)
    else:
      plt.xlabel(f'Iterations {'Log scalse' if logScale else ''}')
      plt.ylabel('Cost - MSE')
      plt.title('Cost vs iteration')
      if logScale: plt.xscale('log')
      plt.plot(self.costHistory)

    plt.show()

class BasicLogisticRegression(ModelSetup): # grandient ascent approach
  def __init__(self, 
               filePath, 
               initWeights= None, 
               epsilon= 0.000001, 
               learningRate= 0.001, 
               parsingData= (None,None,None),   # [0]: delimiter
                                                # [1]: header included in file
                                                # [2]: indexing column included
               featureScaling= True,
               modelDegree= 1,
               logEventTrigger= 1000):
    super().__init__(filePath, initWeights, epsilon, learningRate, parsingData, featureScaling, modelDegree, logEventTrigger)

  def predict(self, inputData):
    if len(inputData) == len(self.scalers[0][0]):
    # input -> scaleBefore -> poly -> scalseAfter -> prediction
    # ----------------------------------------------------------
    # 1st scale / scaleBefore
      means, stdDevs = self.scalers[0]
      inputData = (inputData - means) / stdDevs
    # polynomial extention
      inputData = self.polyEngine.transform(inputData)
    # 2nd scale / scaleAfter
      means, stdDevs = self.scalers[1]
      inputData[1:] = (inputData[1:] - means) / stdDevs
    # predict
      return 1 / (1 + np.exp(-np.dot(self.weights, inputData)))
    else:
      print("Input size does not match the model! pls check again")
      return None

  def _predict(self):
    return 1 / (1 + np.exp(-np.dot(self.xTrain, self.weights)))
  
  def computeLogLikelihood(self, predictions):
    predictions = np.clip(predictions, a_min= 1e-15, a_max= 1 - 1e-15)
    return sum(self.yTrain*np.log(predictions) + (1 - self.yTrain)*np.log(1 - predictions))
  
  def computeGradient(self, predictions):
    grd = np.dot(self.xTrain.T, predictions - self.yTrain)
    return self.learningRate*grd
  
  def fitModel(self, log= None, saveModelState= None):
    predictions = self._predict()
    curLlh = self.computeLogLikelihood(predictions)
    self.costHistory = [np.e**curLlh]

    # User want to store model tranining states
    if log:
      with open('data/iterations.txt', mode = 'w') as f: pass
      with open('data/iterations.txt', mode = 'a') as f:
        f.write(f"""["Weights"], current log likelihood, shifting in Llh \n {self.weights}, {curLlh}, 0 \n""")

    # Start training
    while True:
      self.iterationCount += 1

      # Update weights
      self.weights -= self.computeGradient(predictions)
      predictions = self._predict()

      # Capture model states
      newLlh = self.computeLogLikelihood(predictions)
      self.costHistory.append(np.e**newLlh)
      llhDif = abs((newLlh - curLlh) / curLlh)

      # Log
      if self.iterationCount % self.logTrigger == 0:
        print(f"||| {self.iterationCount}\t iterations passed; model'current log-likelihood: {curLlh}")
        if log:
          with open('data/iterations.txt', mode = 'a') as f:
            f.write(f"{self.weights}, {curLlh}, {llhDif} \n")

      # Stopping condition
      if llhDif < self.epsilon or newLlh < curLlh:
        self._modelEvaluation() # Return model evaluation result
        if saveModelState:
          self.saveModel(modelSaveFile= saveModelState, model= self)
        break
      else:
        curLlh = newLlh

  def _modelEvaluation(self):
    predictions = np.array([self.predict(inputData) for inputData in self.xTest])
    preLabels = predictions >= 0.5
    actLabels = self.yTest >= 0.5
    
    # Evaluate
    truePos = sum(1 for a, p in zip(actLabels, preLabels) if a == 1 and p == 1)
    trueNeg = sum(1 for a, p in zip(actLabels, preLabels) if a == 0 and p == 0)
    falsePos = sum(1 for a, p in zip(actLabels, preLabels) if a == 0 and p == 1)
    falseNeg = sum(1 for a, p in zip(actLabels, preLabels) if a == 1 and p == 0)

    # F-score components
    precision = truePos / (truePos + falsePos) if (truePos + falsePos) > 0 else 0
    recall = truePos / (truePos + falseNeg) if (truePos + falseNeg) > 0 else 0
    specificity = trueNeg / (trueNeg + falsePos) if (trueNeg + falsePos) > 0 else 0
    total = (truePos + trueNeg + falsePos + falseNeg)
    accuracy = (truePos + trueNeg) / total if total > 0 else 0
    fScore = 2*(precision*recall / (precision + recall)) if (precision + recall) > 0 else 0

    self.modelEvaluation = {'tp': truePos, 'tn': trueNeg, 'fp': falsePos, 'fn': falseNeg,
                            'precision': precision, 'recall': recall, 'spe': specificity, 'accuracy': accuracy,
                            'fScore': fScore}
    # return np.array([[trueNeg, falsePos],[falseNeg, truePos]])

  def showConfusionMatrix(self):
    m = self.modelEvaluation
    # confusion matrix
    cmData = np.array([[m['tp'], m['fn']],
                       [m['fp'], m['tn']]])
    
    # Plot the confusion matrix
    cmDataFrame = pd.DataFrame(data= cmData, index= [1,0], columns= [1,0])
    sborn.heatmap(cmDataFrame, annot= True, cmap= 'Greens')
    plt.title(f'Consusion Matrix - (f1 = {round(m['fScore']*100,4)}%)')
    plt.xlabel('Predicted Labels')
    plt.ylabel('Actual Labels')
    plt.show()

  def showCostTrend(self, log= True):
    plt.xlabel(f'Iterations {'Log scalse' if log else ''}')
    plt.ylabel('Cost - likelihood')
    plt.title('Cost vs iteration')
    if log: plt.xscale('log')
    plt.plot(self.costHistory)
    plt.show()

  def showTestTable(self):
    predictedProbabilities = [self.predict(inputData) for inputData in self.xTest]
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
