import os
import sys
import pdb
import gzip
import time
import glob
import pickle
import random
import hashlib
import datetime
import numpy as np
import h5py
import torch
import aifc
import argparse
import torchvision
import matplotlib
import scipy
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn import svm, datasets, metrics
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    roc_curve,
)
from collections import Counter
from scipy.fftpack import fft
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import csv

# Enable interactive plotting
plt.ion()

# Platform check for logging
if sys.platform.startswith("linux"):
    print("LINUX OS")
elif sys.platform == "darwin":
    print("MAC OSX")
    matplotlib.use("TkAgg")
