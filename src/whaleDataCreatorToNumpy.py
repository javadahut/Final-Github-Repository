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
# New argument to choose feature type:
parser.add_argument('-feature', dest='featureType', default='stft', type=str, choices=['stft', 'mel'],
                    help='Feature extraction type: "stft" or "mel"')
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

# Create processed data directory if it doesn't exist
if not os.path.exists(directory.dataDirProcessed):
    os.makedirs(directory.dataDirProcessed)

# -- Create STFT Object (for use when featureType is 'stft') --
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

    # Compute feature (STFT or Mel spectrogram)
    if args.featureType == 'stft':
        stftObj.computeSTFT(signal)
        processedImage = np.abs(stftObj.stftMatrix[I.rowsKept, :])
    elif args.featureType == 'mel':
        import librosa
        n_fft = args.fftl
        hop_length = int(np.floor(args.fs * args.tf * (1 - args.po)))
        n_mels = 128  # You can later make this configurable
        melSpectrogram = librosa.feature.melspectrogram(y=signal, sr=args.fs, n_fft=n_fft,
                                                        hop_length=hop_length, n_mels=n_mels)
        processedImage = librosa.power_to_db(melSpectrogram, ref=np.max)
    else:
        raise ValueError("Unknown feature type: " + args.featureType)

    # Optional downsampling if requested
    if args.ds != -1.0:
        from PIL import Image
        procImgPIL = Image.fromarray(processedImage)
        new_dims = (int(np.floor(args.ds * processedImage.shape[1])),
                    int(np.floor(args.ds * processedImage.shape[0])))
        procImgPIL = procImgPIL.resize(new_dims, Image.BICUBIC)
        processedImage = np.asarray(procImgPIL)

    # Resize to fixed dimensions for network compatibility (108 x 108)  
    from PIL import Image
    procImgPIL = Image.fromarray(processedImage)
    fixed_dims = (108, 108)
    procImgPIL = procImgPIL.resize(fixed_dims, Image.BICUBIC)
    processedImage = np.asarray(procImgPIL)

    # -- Allocate pData/pLabels after knowing frame count --
    if frame_shape is None:
        frame_shape = processedImage.shape  # Should now be (108, 108)
        pData = np.zeros((N.data, 1, frame_shape[0], frame_shape[1]), dtype=np.float32)
        pLabels = -1 * np.ones(N.data, dtype=np.int64)

    # Save processedImage into pData array (corrected variable name)
    pData[cc, 0, :, :] = processedImage
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
