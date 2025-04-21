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

if not (0 <= args.valPercentage <= 1) or not (0 <= args.testPercentage <= 1):
    raise ValueError("Percentages must be between 0 and 1")
if args.valPercentage + args.testPercentage >= 1:
    raise ValueError("Sum of val and test percentages must be < 1")

directory.loadNumpyDataFrom = args.numpyDataDir
N.valPercentage = args.valPercentage
N.testPercentage = args.testPercentage

'''
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
'''
def splitData(testND, valND, trainND, pDataPos, pDataNeg):
    # Split data FIRST (no normalization yet)
    test_pos = pDataPos[:testND]
    test_neg = pDataNeg[:testND]
    val_pos = pDataPos[testND:testND+valND]
    val_neg = pDataNeg[testND:testND+valND]
    train_pos = pDataPos[testND+valND:]
    train_neg = pDataNeg[testND+valND:]

    # Compute stats ONLY from training data
    training_concat = np.concatenate((train_pos, train_neg), axis=0)
    pTrainingMean = np.mean(training_concat, axis=0)
    pTrainingStd = np.std(training_concat, axis=0)

    # Normalize ALL splits
    train_pos = (train_pos - pTrainingMean) / (pTrainingStd + 1e-6)
    train_neg = (train_neg - pTrainingMean) / (pTrainingStd + 1e-6)
    
    val_data = np.concatenate([
        (val_pos - pTrainingMean) / (pTrainingStd + 1e-6),
        (val_neg - pTrainingMean) / (pTrainingStd + 1e-6)
    ], axis=0)
    
    test_data = np.concatenate([
        (test_pos - pTrainingMean) / (pTrainingStd + 1e-6),
        (test_neg - pTrainingMean) / (pTrainingStd + 1e-6)
    ], axis=0)

    # Labels
    val_labels = np.concatenate([np.ones(val_pos.shape[0]), np.zeros(val_neg.shape[0])])
    test_labels = np.concatenate([np.ones(test_pos.shape[0]), np.zeros(test_neg.shape[0])])

    return train_pos, train_neg, (val_data, val_labels), (test_data, test_labels), pTrainingMean, pTrainingStd

def minimumSamples(percentage, nNonDominant):
    samples = np.round(percentage * nNonDominant).astype(np.int64)
    return samples

def verify_tensor(tensor, expected_shape, name=""):
    if tensor.shape != expected_shape:
        raise ValueError(f"{name} tensor shape mismatch. Got {tensor.shape}, expected {expected_shape}")
    
N.trainPercentage = 1 - (N.testPercentage + N.valPercentage)
np.random.seed(1)

pData = np.load(directory.loadNumpyDataFrom + 'pData.npy', mmap_mode='r')
pLabels = np.load(directory.loadNumpyDataFrom + 'pLabels.npy', mmap_mode='r')

I.randomIndices = np.random.permutation(pLabels.shape[0])
pLabels = pLabels[I.randomIndices]

# Use indices to slice without copying all data into memory
pData = np.load(directory.loadNumpyDataFrom + 'pData.npy', mmap_mode='r')
pDataPosIndices = I.randomIndices[pLabels == 1]
pDataNegIndices = I.randomIndices[pLabels == 0]
pDataPos = np.copy(pData[pDataPosIndices])
pDataNeg = np.copy(pData[pDataNegIndices])

# Add class balance check
pos_ratio = len(pDataPos) / (len(pDataPos) + len(pDataNeg))
if not 0.4 <= pos_ratio <= 0.6:
    print(f"[WARNING] Class imbalance detected: {pos_ratio:.1%} positive samples")

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

expected_shape = pTrainingDataPos.shape[1:]  # Get (C,H,W) from numpy array
verify_tensor(tTrainingDataPos, (len(pTrainingDataPos), *expected_shape), "TrainingPos")
verify_tensor(tValData, (len(valTuple[0]), *expected_shape), "Validation")

print("[INFO] Saving tensors...")
torch.save(tTrainingDataPos, directory.loadNumpyDataFrom + 'tTrainingDataPos')
torch.save(tTrainingDataNeg, directory.loadNumpyDataFrom + 'tTrainingDataNeg')
torch.save(tValData, directory.loadNumpyDataFrom + 'tValData')
torch.save(tValLabels, directory.loadNumpyDataFrom + 'tValLabels')
torch.save(tTestData, directory.loadNumpyDataFrom + 'tTestData')
torch.save(tTestLabels, directory.loadNumpyDataFrom + 'tTestLabels')

print("[SUCCESS] Saved all tensors:")
print(f"- Training: {len(pTrainingDataPos)} pos, {len(pTrainingDataNeg)} neg")
print(f"- Validation: {len(valTuple[0])} samples")
print(f"- Test: {len(testTuple[0])} samples")
print('FIN')
