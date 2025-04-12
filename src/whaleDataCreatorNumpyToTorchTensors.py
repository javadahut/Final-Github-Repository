from helperFunctions import *
import argparse
import numpy as np
import torch

class directory:
    pass
class filename:
    pass
class I:
    pass    
class N:
    pass

parser = argparse.ArgumentParser(description='Settings')
parser.add_argument('-numpyDataDir', dest='numpyDataDir', required=True, type=str)
parser.add_argument('-valPercentage', dest='valPercentage', default=0.2, type=float)
parser.add_argument('-testPercentage', dest='testPercentage', default=0.1, type=float)
args = parser.parse_args()

directory.loadNumpyDataFrom = args.numpyDataDir
N.valPercentage = args.valPercentage
N.testPercentage = args.testPercentage

def splitData(testND, valND, trainND, pDataPos, pDataNeg):
    dataShape = pDataPos.shape[1:4]
    # Extract the test set.
    pTestData = np.zeros((2 * testND,) + dataShape, dtype=np.float32)
    pTestLabels = -99 * np.ones(2 * testND, dtype=np.int64)
    pTestData[0:testND, :, :, :] = np.copy(pDataPos[0:testND, :, :, :])
    pTestLabels[0:testND] = 1
    pTestData[testND:2 * testND, :, :, :] = np.copy(pDataNeg[0:testND, :, :, :])
    pTestLabels[testND:2 * testND] = 0

    # Extract the validation set.
    pValData = np.zeros((2 * valND,) + dataShape, dtype=np.float32)
    pValLabels = -99 * np.ones(2 * valND, dtype=np.int64)
    pValData[0:valND, :, :, :] = np.copy(pDataPos[testND:testND+valND, :, :, :])
    pValLabels[0:valND] = 1
    pValData[valND:2 * valND, :, :, :] = np.copy(pDataNeg[testND:testND+valND, :, :, :])
    pValLabels[valND:2 * valND] = 0

    # Extract the training set.
    pTrainingDataPos = np.copy(pDataPos[(testND + valND):, :, :, :])
    pTrainingDataNeg = np.copy(pDataNeg[(testND + valND):, :, :, :])

    # Normalize the data: compute mean and std from the training set.
    trainingPosNegConcat = np.concatenate((pTrainingDataPos, pTrainingDataNeg), 0)
    pTrainingMean = np.mean(trainingPosNegConcat, axis=0)
    pTrainingStd = np.std(trainingPosNegConcat - pTrainingMean, axis=0)

    # De-mean and normalize.
    pTrainingDataPos -= pTrainingMean 
    pTrainingDataNeg -= pTrainingMean 
    pValData -= pTrainingMean 
    pTestData -= pTrainingMean 

    pTrainingDataPos /= (pTrainingStd + 1e-6)
    pTrainingDataNeg /= (pTrainingStd + 1e-6)
    pValData /= (pTrainingStd + 1e-6)
    pTestData /= (pTrainingStd + 1e-6)

    return pTrainingDataPos, pTrainingDataNeg, (pValData, pValLabels), (pTestData, pTestLabels), pTrainingMean, pTrainingStd

def minimumSamples(percentage, nNonDominant):
    samples = np.round(percentage * nNonDominant).astype(np.int64)
    return samples

N.trainPercentage = 1 - (N.testPercentage + N.valPercentage)
np.random.seed(1)

pData = np.load(directory.loadNumpyDataFrom + 'pData.npy')
pLabels = np.load(directory.loadNumpyDataFrom + 'pLabels.npy')

I.randomIndices = np.random.permutation(pData.shape[0])
pData = pData[I.randomIndices, :, :, :]
pLabels = pLabels[I.randomIndices]

pDataPos = np.copy(pData[pLabels == 1, :, :, :])
pDataNeg = np.copy(pData[pLabels == 0, :, :, :])

if pDataPos.shape[0] >= pDataNeg.shape[0]:
    N.nonDominant = pDataNeg.shape[0]
else:
    N.nonDominant = pDataPos.shape[0]

N.testND = minimumSamples(N.testPercentage, N.nonDominant)
N.valND = minimumSamples(N.valPercentage, N.nonDominant)
N.trainND = N.nonDominant - (N.valND + N.testND)

pTrainingDataPos, pTrainingDataNeg, valTuple, testTuple, _, _ = splitData(N.testND, N.valND, N.trainND, pDataPos, pDataNeg)

tTrainingDataPos = torch.Tensor(pTrainingDataPos)
tTrainingDataNeg = torch.Tensor(pTrainingDataNeg)
tValData = torch.Tensor(valTuple[0])
tValLabels = torch.Tensor(valTuple[1]).long()
tTestData = torch.Tensor(testTuple[0])
tTestLabels = torch.Tensor(testTuple[1]).long()

torch.save(tTrainingDataPos, directory.loadNumpyDataFrom + 'tTrainingDataPos')
torch.save(tTrainingDataNeg, directory.loadNumpyDataFrom + 'tTrainingDataNeg')
torch.save(tValData, directory.loadNumpyDataFrom + 'tValData')
torch.save(tValLabels, directory.loadNumpyDataFrom + 'tValLabels')
torch.save(tTestData, directory.loadNumpyDataFrom + 'tTestData')
torch.save(tTestLabels, directory.loadNumpyDataFrom + 'tTestLabels')

print('FIN')
