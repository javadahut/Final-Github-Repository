
# 🐋 Whale Vocalization Detection with Deep Learning

This project builds on previous research from the [Kaggle Whale Detection Challenge](https://www.kaggle.com/competitions/whale-detection-challenge), which improved whale call detection accuracy from 72% to 98%—a critical advancement used to help prevent ship-whale collisions in busy shipping lanes.

---

## 🔬 Project Overview

Our goal is to develop a deep learning system that detects whale vocalizations by:
- Converting raw audio into robust spectrogram representations using an optimized Short-Time Fourier Transform (STFT) implementation.
- Transitioning to Mel-spectrograms to emphasize biologically relevant frequency bands.
- Evaluating state-of-the-art architectures including:
  - **EfficientNetV2-S:** A lightweight, high-performance CNN.
  - **Audio Spectrogram Transformer (AST):** A transformer-based model pretrained on [AudioSet](https://research.google.com/audioset/).

These models will be benchmarked against an InceptionV1 baseline, with a target AUROC > 0.98.

---

## ✅ Goals

- **Preprocessing:** Enhance feature extraction using Mel-frequency or MFCC techniques.
- **Modeling:** Improve generalization via EfficientNetV2-S and AST.
- **Visualization:** Develop an interactive UI to display spectrograms and model predictions.
- **Scalability:** Expand from binary detection to multi-class classification (e.g., multiple whale species and other marine sounds).

---

## 📁 Project Structure

```
Final-Whale-Detection-Project/
 ├── README.md # Project overview and documentation
 ├── requirements.txt # Python dependencies
 ├── .gitignore # Files/folders to ignore in GitHub
 ├── data/
 │ ├── train/ # Raw audio files (.aif, .aiff, etc.)
 │ ├── processed/ # Processed outputs (e.g., pData.npy, pLabels.npy)
 │ └── train.csv # CSV file mapping audio filenames to labels
 ├── src/
 │ ├── dependencies.py # Unified dependencies and environment setup
 │ ├── helperFunctions.py # STFT class, forward propagation utilities, etc.
 │ └── whaleDataCreatorToNumpy.py # Main script to preprocess audio data
 ├── models/ # Directory for storing trained model files
 ├── notebooks/ # Jupyter notebooks for experiments, visualization, etc.
 ├── outputs/ # Training logs, ROC curves, evaluation plots, etc.
 └── scripts/ # Helper shell or batch scripts (optional)
```

---

## 📦 Installation

1. **Clone the repository**

```bash
git clone https://github.com/javadahut/Final-Github-Repository.git
cd Final-Github-Repository
```

2. **Set up a virtual environment (recommended)**

```python3 -m venv venv
source venv/bin/activate   # Linux/Mac
# or on Windows:
venv\Scripts\activate

```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Download and prepare the dataset**  
Follow the dataset instructions at [Kaggle competition page](https://www.kaggle.com/competitions/whale-detection-challenge).

Place your raw 2-second audio files (16kHz, ~32,000 samples per file) in data/train/.

Ensure your train.csv file (located in data/) contains the filenames and corresponding labels.
---

## 🚀 Getting Started

Preprocessing Audio Files
The main preprocessing script converts audio files to STFT (or Mel-spectrogram) representations and saves them as NumPy arrays.

Example command:
```
  cd src/
  python whaleDataCreatorToNumpy.py \
    -dataDir ../data/train/ \
    -labelcsv ../data/train.csv \
    -dataDirProcessed ../data/processed/ \
    -fs 16000 \
    -tx 2.0 \
    -tf 0.071 \
    -po 0.75 \
    -fftl 1024 \
    -fftw hanning \
    -rk 30 200 \
    -s 1 \
    -ins 2
```

  -fs: Sampling frequency (16,000 Hz)
  
  -tx: Duration to process from each file (2.0 sec)
  
  -tf: Frame length (0.071 sec)
  
  -po: Frame overlap (75%)
  
  -fftl: FFT length (1024)
  
  -fftw: Window type ('hanning' supported)
  
  -rk: Range of FFT rows to keep (indices 30 to 200)
  
  -s: Save output flag (1 means outputs will be saved)
  
  -ins: Inspect flag (2 enables visualization during processing)

Processed NumPy arrays (pData.npy and pLabels.npy) will be saved in data/processed/

```
python whaleDataCreatorNumpyToTorchTensors.py -numpyDataDir ../data/processed/ -valPercentage 0.2 -testPercentage 0.1
```


## Available DNN Architectures

- **cnn_108x108**  
  A standard convolutional neural network designed for 108×108 input images. It consists of several layers of convolution, batch normalization, ReLU activation, pooling, and dropout, followed by a series of fully connected layers. This model serves as a baseline architecture for classification.

- **inceptionModuleV1_108x108**  
  An inception-style network designed specifically for 108×108 inputs. It starts with convolutional “root” layers that downsample the image, then branches out into multiple parallel paths with different kernel sizes (1×1, 3×3, 5×5, and pooling). The outputs of these branches are concatenated, passed through a reduction stage, and finally classified.

- **inceptionModuleV1_75x45**  
  A variant of the inception module aimed at smaller input sizes (75×45). Similar in design to the 108×108 version, this network also employs parallel branches to extract multi-scale features. It is suitable when the processing pipeline outputs spectrograms at a lower resolution.

- **inceptionTwoModulesV1_75x45**  
  This network extends the single inception module design by stacking two sequential inception modules for 75×45 inputs. The first module processes the input and the second further refines the concatenated multi-scale features. This additional depth enhances feature extraction for more complex patterns.

- **inceptionTwoModulesV1_root1_75x45**  
  A variation on the two-module inception architecture with an alternate “root” (initial convolution) configuration for 75×45 inputs. This adjustment alters the receptive field and overall feature extraction process, offering an alternative design for experimentation.

- **inceptionV1_modularized**  
  A modular, configurable version of the Inception V1 architecture designed for 75×45 inputs. It allows the user to specify the number of inception layers (using the `-nils` argument), so the network depth can be adjusted according to task complexity. After processing through multiple modular inception blocks, the features are aggregated via a redux layer and classified through fully connected layers.

- **inceptionV1_modularized_mnist**  
  A simplified modular inception model optimized for very small-scale inputs (such as MNIST images, typically 28×28). It uses a tailored “root” network and a more compact design to produce a latent embedding that is then mapped to a small number of output classes. This design is useful for benchmark or low-resolution tasks.

- **centerlossSimple**  
  This network integrates center loss to promote intra-class compactness in the feature space. A CNN “root” network first extracts features from the input, then a linear layer produces a latent representation with maintained class centroids. Finally, another linear layer performs classification. This model is intended to enhance feature discrimination by reducing intra-class variability.

## How to Select a Model
When you run the training script (e.g. whaleClassifier.py), you can choose one of these architectures by specifying the appropriate value for the -dnn parameter. For example:

For a standard CNN designed for 108×108 inputs, run:
```
python whaleClassifier.py -dataDirProcessed ../data/processed/ -dnn "cnn_108x108" ...
```

For an inception-style model on 108×108 inputs, run:
```
python whaleClassifier.py -dataDirProcessed ../data/processed/ -dnn "inceptionModuleV1_108x108" ...
```

For lower resolution spectrograms (75×45), you can similarly choose from the other available options.

Choose the model that best matches your data and desired complexity. This flexibility allows you to benchmark different architectures and experiment with their performance on your whale vocalization detection task.


```
python whaleClassifier.py -dataDirProcessed ../data/processed/ -cpu 1 -e 10 -lr 5e-5 -L2 0.001 -mb 16 -s 3 -dnn "cnn_108x108"
DNN arch: cnn_108x108
```

---

## 📊 Results & Visualizations

After training, you can generate:

  Training and validation performance curves.
  ROC & Precision-Recall curves.
  Confusion matrices.
  Interactive plots for manual inspection of spectrograms.

All results are saved to the outputs/ directory.

---

## 📈 Benchmark Targets

| Model                               | Input              | AUROC Goal | Current Status |
|-------------------------------------|--------------------|------------|----------------|
| CNN (Baseline)                      | STFT Spectrogram   | 0.94       | ✔️ Implemented |
| InceptionV1 (Baseline)              | STFT Spectrogram   | 0.96       | ✔️ Implemented |
| EfficientNetV2-S                    | Mel Spectrogram    | >0.98      | ⏳ In Progress |
| Audio Spectrogram Transformer (AST) | Mel Spectrogram    | >0.98      | ⏳ In Progress |

---

## 🧠 Pretrained Models

We use pretrained weights for:
- AST (pretrained on AudioSet)
- EfficientNetV2-S (from torchvision or timm)

Pretrained models are automatically downloaded during training unless specified otherwise.

---

## 🛠 Future Work & Roadmap
- Enhance Preprocessing: Transition from raw STFTs to Mel-spectrograms and explore MFCCs.
- Advanced Modeling: Evaluate and fine-tune EfficientNetV2-S and AST models.
- Visualization Tools: Develop an interactive UI to help interpret model predictions.
- Extension to Multi-Class: Expand detection to include multiple whale species and other marine sounds.

Refer to our Gantt-style project tracker https://github.com/javadahut/Final-Github-Repository/blob/main/project_plan.csv for a detailed week-by-week plan (April 4 – May 10).

---

## 🙌 Contributing

We welcome contributions! To get started:

1. Fork the repository
2. Create a new branch: `git checkout -b feature/my-feature`
3. Commit your changes: `git commit -m "Add new feature"`
4. Push your branch: `git push origin feature/my-feature`
5. Open a Pull Request

---

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.

---

## 📚 Citation

If you use this project, please consider citing:

> Kaggle Whale Detection Challenge:  
> https://www.kaggle.com/competitions/whale-detection-challenge/

> Gong et al. (2021). AST: Audio Spectrogram Transformer  
> https://arxiv.org/abs/2104.01778
