
# 🐋 Whale Vocalization Detection with Deep Learning

This project builds on previous research from the [Kaggle Whale Detection Challenge](https://www.kaggle.com/competitions/whale-detection-challenge), which improved whale call detection accuracy from 72% to 98%—a critical advancement used to help prevent ship-whale collisions in busy shipping lanes.

---

## 🔬 Project Overview

We aim to push the boundaries of this work by:

- Replacing raw STFT spectrograms with **Mel-spectrograms**, which emphasize biologically relevant frequency bands for marine mammals.
- Evaluating two state-of-the-art deep learning architectures:
  - **EfficientNetV2-S**: A lightweight, high-performance CNN.
  - **Audio Spectrogram Transformer (AST)**: A transformer-based model pretrained on [AudioSet](https://research.google.com/audioset/).

These models will be benchmarked against the original InceptionV1 baseline used in the competition, targeting an **AUROC > 0.98**.

---

## ✅ Goals

- ✅ Enhance spectrogram preprocessing using Mel-frequency and MFCC techniques  
- ✅ Improve model generalization with EfficientNetV2 and AST  
- ✅ Develop an interactive UI to visualize model predictions and spectrograms  
- ✅ Expand beyond binary classification to identify multiple whale species and marine sounds

---

## 📁 Project Structure

```
.
├── data/                       # Data loading and augmentation scripts
├── models/                     # Network architecture definitions
├── utils/                      # Spectrogram processing and helpers
├── notebooks/                  # Jupyter analysis & prototype notebooks
├── train.py                    # CLI training script
├── evaluate.py                 # Evaluation and visualization tools
├── requirements.txt            # Package dependencies
├── config.yaml                 # Model and training configuration
└── README.md                   # Project documentation
```

---

## 📦 Installation

1. **Clone the repository**

```bash
git clone https://github.com/your-username/whale-vocalization-detector.git
cd whale-vocalization-detector
```

2. **Set up a virtual environment (recommended)**

```bash
python3 -m venv venv
source venv/bin/activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Download and prepare the dataset**  
Follow the dataset instructions in `data/README.md` or refer to the [Kaggle competition page](https://www.kaggle.com/competitions/whale-detection-challenge).

---

## 🚀 Getting Started

To train the model using a configuration:

```bash
python train.py --config config.yaml
```

To evaluate:

```bash
python evaluate.py --checkpoint models/best_model.pth
```

---

## 📊 Results & Visualizations

- Training and validation curves
- ROC & PR curves
- Confusion matrices
- Audio + spectrogram playback for manual inspection (UI coming soon)

All plots are saved to `outputs/` during evaluation.

---

## 📈 Benchmark Targets

| Model              | Input              | AUROC Goal | Current Status |
|--------------------|--------------------|------------|----------------|
| InceptionV1 (baseline) | STFT Spectrogram | 0.96        | ✔️ Implemented |
| EfficientNetV2-S   | Mel Spectrogram     | >0.98       | ⏳ In Progress |
| AST                | Mel Spectrogram     | >0.98       | ⏳ In Progress |

---

## 🧠 Pretrained Models

We use pretrained weights for:
- AST (pretrained on AudioSet)
- EfficientNetV2-S (from torchvision or timm)

Pretrained models are automatically downloaded during training unless specified otherwise.

---

## 📌 Roadmap

See our [Gantt-style project tracker](https://github.com/your-username/whale-vocalization-detector/blob/main/project_plan.csv) for a detailed week-by-week plan (April 4 – May 10).

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
