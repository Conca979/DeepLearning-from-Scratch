from regression import BasicLinearRegression, BasicLogisticRegression
from itertools import combinations
import numpy as np

# -------------------------------------------------------------------

dataSet = np.loadtxt('data/Advertising.csv', skiprows= 1, delimiter= ',', usecols= range(1, 5))

# Splitting data into 80 training / 20 testing
splitIndex = int(dataSet.shape[0]*0.8)
trSet = dataSet[:splitIndex, :]
tSet = dataSet[splitIndex:, :]

# Cross-validation on the training set
# K-fold = 5 is widely used
k = 5

# Splitting training set into k folders
folders = {}
splitIndex = int(trSet.shape[0] / k)
for i in range(k):
  if i == k - 1: # The last folders
    folders[i+1] = trSet[splitIndex*i:, ::]
  else:
    folders[i+1] = trSet[splitIndex*i:splitIndex*(i+1), ::]

# Hyperparameters for coss-validation
degrees = [3,4,5,6]

# Split k folders into (1 for validating set) and (k-1 for trainingSet)
a = list(combinations(range(1, k + 1), 4))
AMSEs = {}
for d in degrees:
  MSEs = []
  for fs in a:
    track = set(range(1, k + 1))
    for f in fs:
      if len(track) == k: # Must create training set in the first file
        trainingSet = folders[f].copy()
      else:
        trainingSet = np.concatenate((trainingSet, folders[f]))
      # Tracking files
      track.remove(f)
    # The last file for validating
    for lastFolder in track:
      validatingSet = folders[lastFolder]

    # Start training
    model = BasicLinearRegression(trainingSet= trainingSet,
                                  testSet= validatingSet,
                                  epsilon= 1e-7, 
                                  learningRate= 0.0001, 
                                  modelDegree= d,
                                  iterationLogTrigger= -1,
                                  initWeights= None)

    model.fitModel()
    Mean2Error = model.modelValidating()
    print(f"""{'-'*100}
    |||Training iterations:{model.iterationCount}
    |||Model degree, #weights: {model.degree}, {model.xTrain.shape[1]}
    |||Model goodness of fit (Training Error): {model._modelEvaluaion()[2]}
    |||Mean square error (Test Error): {Mean2Error}""")
    MSEs.append(Mean2Error)

  # Average(Mean square errors)
  AMSEs[model.degree] = float(np.mean(MSEs))

print(AMSEs)