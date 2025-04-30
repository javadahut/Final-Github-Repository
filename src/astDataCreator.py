import os
import argparse
import aifc
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import csv
import librosa

def read_aiff(path):
    with aifc.open(path, 'r') as f:
        audio = f.readframes(f.getnframes())
        signal = np.frombuffer(audio, dtype=np.int16).byteswap().astype(np.float32)
    signal -= np.mean(signal)
    signal /= (np.std(signal) + 1e-8)
    return signal

def compute_spectrogram(signal, sr, n_fft=512, n_mels=128):
    mel = librosa.feature.melspectrogram(
        y=signal, sr=sr, n_fft=n_fft, hop_length=128,
        n_mels=n_mels, fmin=20, fmax=sr // 2
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db

def spectrogram_to_tensor(spec, size=(224, 224)):
    img = Image.fromarray(spec).convert("RGB")
    img = img.resize(size, Image.BICUBIC)
    img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
    return img_tensor  # [3, 224, 224]

def save_shards(data_list, label_list, out_dir, prefix, shard_size=5000):
    os.makedirs(out_dir, exist_ok=True)
    total = len(data_list)
    print(f"[Saving] Total samples: {total}, Shard size: {shard_size}")

    for i in range(0, total, shard_size):
        shard_data = torch.stack(data_list[i:i+shard_size])
        shard_labels = torch.tensor(label_list[i:i+shard_size])
        torch.save(
            {'data': shard_data, 'labels': shard_labels},
            os.path.join(out_dir, f"{prefix}_shard_{i//shard_size}.pt")
        )
        print(f"  ↪ Saved: {prefix}_shard_{i//shard_size}.pt with {len(shard_labels)} examples")

def main():
    parser = argparse.ArgumentParser(description='AST Whale Data Preprocessor (sharded for Colab)')
    parser.add_argument('-dataDir', required=True, type=str)
    parser.add_argument('-labelcsv', required=True, type=str)
    parser.add_argument('-dataDirProcessed', required=True, type=str)
    parser.add_argument('-fs', default=2000, type=int)
    parser.add_argument('-s', default=0, type=int, help='Save processed data if 1')
    parser.add_argument('-shard_size', default=5000, type=int, help='Max samples per shard')
    args = parser.parse_args()

    os.makedirs(args.dataDirProcessed, exist_ok=True)

    with open(args.labelcsv, 'r') as f:
        reader = csv.reader(f)
        rows = list(reader)[1:]  # skip header

    data, labels = [], []

    for fname, label in tqdm(rows, desc="Processing audio files"):
        path = os.path.join(args.dataDir, fname)
        if not os.path.exists(path):
            print(f"Skipping missing file: {path}")
            continue
        try:
            signal = read_aiff(path)
            spec = compute_spectrogram(signal, args.fs)
            tensor = spectrogram_to_tensor(spec)
            data.append(tensor)
            labels.append(int(label))
        except Exception as e:
            print(f"[ERROR] {fname}: {e}")

    print(f"[Summary] Total samples: {len(data)}")

    if args.s:
        save_shards(data, labels, args.dataDirProcessed, "whale_ast", shard_size=args.shard_size)

if __name__ == "__main__":
    main()
