# Flower Species Classification — A Comparative Deep Learning Study

> Can a linear model compete with ResNet? This project finds out — systematically.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Keras](https://img.shields.io/badge/Keras-green)
![W&B](https://img.shields.io/badge/Weights%20%26%20Biases-tracked-yellow)
![Colab](https://img.shields.io/badge/Google%20Colab-GPU-red)
![Accuracy](https://img.shields.io/badge/Best%20Accuracy-94.54%25-brightgreen)

---

## What This Project Is

Most deep learning tutorials pick one model, train it, and call it done. This project does something different — it asks a research question:

**"How does classification performance evolve as architectural complexity increases, from a simple linear model to state-of-the-art pretrained networks?"**

To answer that, seven architectures were implemented, trained, and evaluated under identical conditions on the same 5-class flower dataset. Every decision — from image resolution to optimiser choice — is deliberate and documented.

---

## Results at a Glance

| # | Model | Type | Val Accuracy |
|---|-------|------|-------------|
| 1 | Linear NN | Baseline | 46.45% |
| 2 | Hidden Layer NN | Fully connected | 48.27% |
| 3 | Improved NN | Tuned FC + Bayesian sweep | — |
| 4 | **EfficientNetB0** | Transfer learning | **94.54% 🏆** |
| 5 | AlexNet | Custom CNN | 74.86% |
| 6 | VGG16 | Transfer learning | 91.99% |
| 7 | ResNet50 | Transfer learning | 93.62% |

The jump from **46% → 94%** is not random — it tells a clear story about what these architectures can and cannot do, and why.

---

## The Story Behind the Numbers

**Models 1 & 2 (46–48%)** confirm the theoretical expectation: dense networks flatten spatial structure. A 64×64 image becomes a vector of 12,288 numbers — all pixel relationships are lost. No amount of hidden layers rescues that.

**AlexNet (74.86%)** shows that convolutions change everything. Spatial hierarchies — edges → textures → shapes — are preserved and learned. But training from scratch on 4,317 images limits how far it can go.

**VGG16, ResNet50, EfficientNetB0 (91–94%)** demonstrate the power of transfer learning. These models arrive pre-loaded with knowledge extracted from 1.2 million ImageNet images. Fine-tuning redirects that knowledge toward flowers in a fraction of the training time.

**EfficientNetB0 wins** because it scales depth, width, and resolution together — not independently. It achieves the highest accuracy with the fewest parameters, which is exactly why it is the standard choice in modern research.

---

## Dataset

**Source:** [Kaggle — Flowers Dataset](https://www.kaggle.com/datasets/imsparsh/flowers-dataset)
**Size:** 4,317 images · 5 classes · 80/20 train-validation split (seed = 42)

| Class | Label |
|-------|-------|
| Daisy | 0 |
| Dandelion | 1 |
| Roses | 2 |
| Sunflowers | 3 |
| Tulips | 4 |

---

## Architecture Overview

<img width="1409" height="667" alt="Model comparison results" src="https://github.com/user-attachments/assets/279b30a0-62c7-4427-98f6-d93cfd1e5fca" />

---

## Key Technical Decisions — and Why

Every design choice in this project has a reason. Here are the most important ones:

**Why 64×64 for dense models, 224×224 for CNNs?**
Flattening a 224×224 image produces 150,528 input neurons. A single Dense(512) layer on that input creates 77 million parameters — enough to exhaust Colab's GPU RAM instantly. Reducing to 64×64 brings this to 6.3 million — manageable, and sufficient for dense layers which cannot exploit spatial structure anyway. CNN models are unaffected because convolutions do not explode with image size.

**Why Bayesian optimisation over grid search?**
Grid search evaluates every combination exhaustively. With 6 hyperparameters, that means hundreds of runs. Bayesian search builds a probabilistic model of the search space and proposes the next configuration based on what has worked before — finding better results in 10 trials than grid search finds in 100.

**Why two-phase fine-tuning for transfer learning?**
Unfreezing a pretrained network immediately and training at full learning rate destroys the ImageNet weights — a phenomenon called catastrophic forgetting. Phase 1 trains only the new classification head while the base is frozen. Phase 2 selectively unfreezes the top layers and trains at 10× lower learning rate, gently adapting pretrained features to the flower domain.

**Why EfficientNetB0 specifically?**
VGG16 scales by making networks deeper. ResNets scale by adding residual blocks. EfficientNetB0 uses compound scaling — depth, width, and resolution are increased together using a fixed ratio derived from a neural architecture search. This produces better accuracy per parameter than any of its contemporaries.

---

## Experiment Tracking — Weights & Biases

Professional ML research requires reproducibility. Every run in this project is tracked via [Weights & Biases](https://wandb.ai).

- **Model 2** is logged as a single named run — demonstrating manual experiment tracking
- **Model 3** runs a full Bayesian sweep: 10 trials, 6 hyperparameters, optimising `val_accuracy`. The best configuration is then retrained with **CosineDecay** learning rate scheduling and **EarlyStopping**

| Setting | Value |
|---------|-------|
| Project | `flower-classification` |
| Sweep method | Bayesian optimisation |
| Trials | 10 |
| Metric | `val_accuracy` → maximize |
| LR schedule (best config) | CosineDecay |

---

## Evaluation Methodology

All models are evaluated identically — no cherry-picking:

1. **Training curves** — loss and accuracy per epoch, training vs. validation
2. **Confusion matrix** — seaborn heatmap showing per-class prediction breakdown
3. **Classification report** — precision, recall, F1-score per class + macro/weighted averages
4. **Final comparison** — unified bar chart ranking all 7 models

---

