from dependencies import *
import numpy as np
import sys
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

def cross_entropy_loss(yEst, labels, gpuFlag=None):
    """
    Computes the cross entropy loss.
    The extra argument gpuFlag is ignored.
    """
    return F.cross_entropy(yEst, labels)

class STFT(object):
    """
    Computes the Short-Time-Fourier-Transform (STFT) of a signal.
    """

    def __init__(self, fs, xT, overlapT, frameT, fftLength=None, flagDebug=False, window='rect'):
        class F: pass
        class T: pass
        class N: pass

        self.F = F()
        self.T = T()
        self.N = N()

        self.F.fs = fs
        self.T.x = xT
        self.T.olap = overlapT
        self.T.frame = frameT

        self.N.frameLength = int(np.floor(self.F.fs * self.T.frame))
        self.N.olap = int(np.floor(self.F.fs * self.T.olap))
        self.N.stride = self.N.frameLength - self.N.olap

        if fftLength is None or fftLength < self.N.frameLength:
            self.N.fft = self.N.frameLength
        else:
            self.N.fft = fftLength

        self.N.stftRows = int(np.ceil(self.N.fft / 2.0))

        if window in ['rect', 'none']:
            self.window = np.ones(self.N.frameLength, dtype=np.float32)
        elif window in ['hanning', 'hann']:
            nn = np.arange(0, self.N.frameLength)
            self.window = 0.5 * (1 - np.cos((2.0 * np.pi * nn) / (self.N.frameLength - 1))).astype(np.float32)
        else:
            print(f'[ERROR] Unsupported window type: {window}')
            sys.exit(1)

        if flagDebug:
            print("[DEBUG] STFT/MEL Parameters:")
            print("Frame Length:", self.N.frameLength, "| Stride:", self.N.stride, "| FFT Length:", self.N.fft)

    def computeSTFT(self, x):
        signal_length = len(x)
        self.N.frames = max(0, (signal_length - self.N.frameLength) // self.N.stride)

        self.stftMatrix = np.zeros((self.N.stftRows, self.N.frames), dtype=np.complex64)

        for ff in range(self.N.frames):
            start = ff * self.N.stride
            end = start + self.N.frameLength
            if end > signal_length:
                break
            frame = x[start:end] * self.window
            self.stftMatrix[:, ff] = np.fft.fft(frame, self.N.fft)[:self.N.stftRows]


def extractForwardPropResults_binary(theNet, theDataLoader, gpuFlag='0'):
    """
    Forward propagates a neural net and returns predictions, targets, and loss.
    """
    lossValue = 0.0
    softPredictions = []
    targets = []
    sm = torch.nn.Softmax(dim=1)   # Define Softmax function here

    for ff, data in enumerate(theDataLoader, 0):
        images, labels = data

        if gpuFlag == '1':
            images = images.cuda()
            labels = labels.cuda()

        with torch.no_grad():
            yEst = theNet(images)
            loss = cross_entropy_loss(yEst, labels, gpuFlag)
            lossValue += loss.item()

            softPreds = sm(yEst)[:, 1]
            softPredictions.extend(softPreds.detach().cpu().numpy())
            targets.extend(labels.detach().cpu().numpy())

    lossValue /= max(1, ff + 1)
    return np.array(softPredictions), np.array(targets), lossValue


def info(data):
    """ Prints info on numpy arrays or torch tensors. """
    if isinstance(data, torch.Tensor):
        print(data.size(), 'Torch', type(data).__name__, 'Max:', torch.max(data).item(), 'Min:', torch.min(data).item())
    elif isinstance(data, np.ndarray):
        print(data.shape, type(data).__name__, data.dtype, 'Max:', np.max(data), 'Min:', np.min(data))
    return ''


def qImage(image):
    """ Quickly display a 2D numpy array as an image. """
    plt.imshow(image, interpolation='none', aspect='auto')
    plt.show()

from sklearn.metrics import precision_recall_curve, average_precision_score, roc_curve, roc_auc_score
import numpy as np

def extractMetrics(softPredictions, targets):
    """
    Given soft predictions and true target labels, compute evaluation metrics:
      - precision, recall: for Precision-Recall curve,
      - AP: average precision score,
      - FPR, TPR: false positive and true positive rates for ROC curve,
      - AUCROC: area under the ROC curve.
      
    Arguments:
      softPredictions: numpy array of soft output predictions.
      targets: numpy array of binary true labels (0 or 1).
      
    Returns:
      precision, recall, ap, fpr, tpr, aucroc
    """
    sp = softPredictions.ravel()
    t = targets.ravel()
    precision, recall, _ = precision_recall_curve(t, sp)
    ap = average_precision_score(t, sp)
    fpr, tpr, _ = roc_curve(t, sp)
    aucroc = roc_auc_score(t, sp)
    return precision, recall, ap, fpr, tpr, aucroc
