# ------------------------- Argument Documentation --------------------------
# | Argument          | Type     | Default | Description                    |
# |-------------------|----------|---------|--------------------------------|
# | -dataDir          | str      | None    | Path to input data directory   |
# | -labelcsv         | str      | None    | Path to CSV file with labels   |
# | -dataDirProcessed | str      | None    | Directory to save processed data |
# | -fs               | float    | 2000.0  | Sampling frequency in Hz       |
# | -tx               | float    | 2.0     | Duration of time segment in sec|
# | -tf               | float    | 0.071   | Frame length in seconds        |
# | -po               | float    | 0.75    | Frame overlap as a proportion  |
# | -fftl             | int      | 512     | FFT length                     |
# | -fftw             | str      | 'rect'  | FFT window type                |
# | -rk               | (int,int)| (20,128)| Range of FFT rows to keep      |
# | -ds               | float    | -1.0    | Downsample factor (-1 = no DS) |
# | -s                | int      | 0       | Save flag (1 to save npy files)|
# | -ins              | int      | 0       | Inspect flag (1=debug, 2=plot) |
# ---------------------------------------------------------------------------

from helperFunctions import *
import os
import numpy as np
import argparse
import csv
import hashlib
import aifc
import matplotlib.pyplot as plt
import pdb
from PIL import Image

# -- Helper Function: Detect Duplicate Files --
def id_duplicates(directory):
    """ Detect duplicate files based on MD5 hash. """
    seen_hashes = set()
    duplicates = []
    for fname in os.listdir(directory):
        full_path = os.path.join(directory, fname)
        if os.path.isfile(full_path):
            with open(full_path, 'rb') as f:
                filehash = hashlib.md5(f.read()).hexdigest()
            if filehash in seen_hashes:
                duplicates.append(fname)
            else:
                seen_hashes.add(filehash)
    return duplicates


# -- Helper Namespaces for Data Parameters --
class directory: pass
class filename: pass
class F: pass
class T: pass
class N: pass
class I: pass


# -- Argument Parsing --
parser = argparse.ArgumentParser(description='Whale Data Preprocessor')
parser.add_argument('-dataDir', required=True, type=str)
parser.add_argument('-labelcsv', required=True, type=str)
parser.add_argument('-dataDirProcessed', required=True, type=str)
parser.add_argument('-fs', default=2000.0, type=float)
parser.add_argument('-tx', default=2.0, type=float)
parser.add_argument('-tf', default=0.071, type=float)
parser.add_argument('-po', default=0.75, type=float)
parser.add_argument('-fftl', default=512, type=int)
parser.add_argument('-fftw', default='rect', type=str)
parser.add_argument('-rk', default=(20, 128), nargs='+', type=int)
parser.add_argument('-ds', default=-1.0, type=float)
parser.add_argument('-s', default=0, type=int)
parser.add_argument('-ins', default=0, type=int)
args = parser.parse_args()


# -- Configuration --
directory.dataDir = args.dataDir
filename.labelcsv = args.labelcsv
directory.dataDirProcessed = args.dataDirProcessed
F.fs = args.fs
T.x = args.tx
T.frameLength = args.tf
T.olap = args.po * T.frameLength
N.fftLength = args.fftl
I.rowsKept = np.asarray(range(args.rk[0], args.rk[1]))
fftWindow = args.fftw

# -- Create STFT Object --
stftObj = STFT(F.fs, T.x, T.olap, T.frameLength, fftLength=N.fftLength, window=fftWindow, flagDebug=True)

# -- Read CSV Labels --
with open(filename.labelcsv, 'r', newline='') as f:
    reader = csv.reader(f)
    csvList = list(reader)[1:]  # Skip header
N.data = len(csvList)

# -- Detect Duplicates --
dupes = id_duplicates(directory.dataDir)

# -- Initialize Empty Variables (Delay Allocation) --
frame_shape = None
pData = None
pLabels = None

# -- Main Processing Loop --
cc = 0
for ii in range(N.data):
    curr_filename = csvList[ii][0]
    filename.currentTrainingFile = os.path.join(directory.dataDir, curr_filename)

    if curr_filename in dupes:
        print(f"[DUPE]: {filename.currentTrainingFile}")
        continue

    # Read audio signal
    with aifc.open(filename.currentTrainingFile, 'r') as f:
        audio = f.readframes(f.getnframes())
        signal = np.frombuffer(audio, dtype=np.int16).byteswap().astype(np.float32)

    # Normalize
    signal -= np.mean(signal)
    signal /= np.std(signal)

    # Compute STFT
    stftObj.computeSTFT(signal)
    stftImage = np.abs(stftObj.stftMatrix[I.rowsKept, :])

    # -- Allocate pData/pLabels after knowing frame count --
    if frame_shape is None:
        frame_shape = (
            int(np.floor(args.ds * len(I.rowsKept))) if args.ds != -1 else len(I.rowsKept),
            int(np.floor(args.ds * stftObj.N.frames)) if args.ds != -1 else stftObj.N.frames
        )
        pData = np.zeros((N.data, 1, frame_shape[0], frame_shape[1]), dtype=np.float32)
        pLabels = -1 * np.ones(N.data, dtype=np.int64)

    # Optional downsampling
    if args.ds != -1.0:
        stftImage = Image.fromarray(stftImage)
        stftImage = stftImage.resize((frame_shape[1], frame_shape[0]), Image.BICUBIC)
        stftImage = np.asarray(stftImage)

    # Pad or trim to expected shape
    expected_shape = (frame_shape[0], frame_shape[1])
    if stftImage.shape[1] < expected_shape[1]:
        pad = expected_shape[1] - stftImage.shape[1]
        stftImage = np.pad(stftImage, ((0, 0), (0, pad)), mode='constant')
    elif stftImage.shape[1] > expected_shape[1]:
        stftImage = stftImage[:, :expected_shape[1]]

    # Store result
    pData[cc, 0, :, :] = stftImage
    pLabels[cc] = int(csvList[ii][1])

    # Optional inspection
    if args.ins == 1:
        pdb.set_trace()
    elif args.ins == 2:
        plt.cla()
        plt.imshow(pData[cc, 0, :, :], interpolation='none', aspect='auto')
        plt.title(f'ii: {ii}, label: {pLabels[cc]}', fontsize=20, fontweight='bold')
        plt.pause(0.2)

    print(f"[OK]: {filename.currentTrainingFile}")
    cc += 1

# -- Trim arrays for skipped duplicates --
pData = pData[:cc]
pLabels = pLabels[:cc]

# -- Save processed data if requested --
if args.s == 1:
    print("Saving pData to disk...")
    np.save(os.path.join(directory.dataDirProcessed, 'pData.npy'), pData)
    print("Saving pLabels to disk...")
    np.save(os.path.join(directory.dataDirProcessed, 'pLabels.npy'), pLabels)

print("FINISHED PROCESSING")
