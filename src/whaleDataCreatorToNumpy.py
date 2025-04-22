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
parser.add_argument('-dnn', required=True, type=str, help='DNN architecture name')
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

# architecture dimensions
arch_dims = {
    'cnn_108x108': (108, 108),
    'inceptionModuleV1_108x108': (108, 108),
    'inceptionModuleV1_75x45': (75, 45),
    'inceptionTwoModulesV1_75x45': (75, 45),
    'inceptionTwoModulesV1_root1_75x45': (75, 45),
    'inceptionV1_modularized': (108, 108),
    'inceptionV1_modularized_mnist': (108, 108),
    'centerlossSimple': (108, 108),
    'efficientnetV2_S': (128, 128),
    'ast': (128, 256)
}

fixed_dims = arch_dims.get(args.dnn)
if fixed_dims is None:
    raise ValueError(f"Unsupported architecture '{args.dnn}'.")

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
valid_rows = [row for row in csvList if row[0] not in dupes]
N.data = len(valid_rows)

# -- Initialize Empty Variables (Delay Allocation) --
frame_shape = None
pData = None
pLabels = None

# -- Main Processing Loop --
cc = 0
print(f"Processing {N.data} files...")
for ii in range(N.data):
    if ii % 10 == 0:
        print(f"Progress: {ii}/{N.data} ({ii/N.data:.1%})")
    curr_filename = valid_rows[ii][0]
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
        n_mels = fixed_dims[0]  # configurable
        melSpectrogram = librosa.feature.melspectrogram(y=signal, sr=args.fs, n_fft=n_fft,
                                                        hop_length=hop_length, n_mels=n_mels,
                                                        fmin=20, fmax=args.fs // 2) # Biological hearing range
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
    #fixed_dims = (128, 256)  # Mel bins x time frames
    procImgPIL = Image.fromarray(processedImage)
    #fixed_dims = (args.fftl // 2, int(F.fs * T.x))  # Frequency bins x time frames
    #if args.featureType == 'mel':
    #    fixed_dims = (128, fixed_dims[1])  # Mel bins x time frames
    procImgPIL = procImgPIL.resize(fixed_dims, Image.BICUBIC)
    processedImage = np.asarray(procImgPIL)

    # -- Allocate pData/pLabels after knowing frame count --
    if frame_shape is None:
        frame_shape = processedImage.shape  # 
        pData = np.memmap(os.path.join(directory.dataDirProcessed, 'temp.dat'),
            dtype=np.float32, mode='w+', shape=(N.data, 1, *frame_shape)
        )
        pLabels = -1 * np.ones(N.data, dtype=np.int64)

    # Save processedImage into pData array (corrected variable name)
    pData[cc, 0, :, :] = processedImage
    pLabels[cc] = int(valid_rows[ii][1])

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
    np.save(os.path.join(directory.dataDirProcessed, 'pData.npy'), pData[:cc])
    print("Saving pLabels to disk...")
    np.save(os.path.join(directory.dataDirProcessed, 'pLabels.npy'), pLabels[:cc])

# flush and delete the memmap before removal
if pData is not None:
    pData.flush()
    del pData

try:
    os.remove(os.path.join(directory.dataDirProcessed, 'temp.dat'))
except FileNotFoundError:
    pass
except PermissionError as e:
    print(f"[WARNING] Could not remove temp.dat: {e}")

print("FINISHED PROCESSING")
