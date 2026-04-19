# Flower Species Classification — A Comparative Deep Learning Study

> Systematic evaluation of seven neural network architectures on the Oxford 5-class flower dataset, with experiment tracking via Weights & Biases.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Keras](https://img.shields.io/badge/Keras-green)
![W&B](https://img.shields.io/badge/Weights%20%26%20Biases-tracked-yellow)
![Colab](https://img.shields.io/badge/Google%20Colab-GPU-red)
![Dataset](https://img.shields.io/badge/Dataset-5%20Classes%20%C2%B7%204317%20Images-lightgrey)

---
<img width="1409" height="667" alt="image" src="https://github.com/user-attachments/assets/279b30a0-62c7-4427-98f6-d93cfd1e5fca" />

## Overview

This project presents a structured research comparison of seven deep learning architectures applied to the task of flower species classification. Beginning from a simple linear baseline and progressing to state-of-the-art pretrained networks, each model is evaluated under identical conditions — same dataset, same preprocessing pipeline, and the same evaluation metrics (accuracy curves, confusion matrices, and per-class classification reports).

The primary objective is not just to achieve high accuracy, but to rigorously document *how* and *why* performance evolves as architectural complexity increases — from linear decision boundaries to deep residual networks with transfer learning.

Hyperparameter optimisation is performed using Bayesian search via Weights & Biases Sweeps, and all experiment runs are tracked and logged for full reproducibility.

---

## Models Compared

| # | Model | Architecture Type | Key Feature | Input Size |
|---|-------|------------------|-------------|------------|
| 1 | Linear NN | Fully connected | Baseline — no hidden layers, linear boundary only | 64×64 |
| 2 | Hidden Layer NN | Fully connected | Two Dense layers, BatchNorm, Dropout — logged to W&B | 64×64 |
| 3 | Improved NN | Fully connected | Bayesian sweep (W&B), CosineDecay LR, EarlyStopping | 64×64 |
| 4 | EfficientNetB0 | Transfer learning | ImageNet pretrained, two-phase fine-tuning | 224×224 |
| 5 | AlexNet | Custom CNN | 5 conv layers, historical ImageNet 2012 winner | 224×224 |
| 6 | VGG16 | Transfer learning | 13 conv layers with 3×3 filters, block5 fine-tuning | 224×224 |
| 7 | ResNet50 | Transfer learning | Residual connections, conv5 block fine-tuning | 224×224 |

---

## Dataset
https://www.kaggle.com/datasets/imsparsh/flowers-dataset
The **TF Flowers dataset** contains 4,317 labelled images across five species. All images are resized to a consistent resolution per model type. An 80/20 train-validation split is applied with a fixed seed (42) for reproducibility.

| Class | Label |
|-------|-------|
| Daisy | 0 |
| Dandelion | 1 |
| Roses | 2 |
| Sunflowers | 3 |
| Tulips | 4 |

Two parallel data pipelines are maintained:
- **64×64 normalised pipeline** — for Dense/flat models, to avoid GPU out-of-memory on the flatten operation (224×224 flatten = 150,528 neurons → GPU OOM; 64×64 flatten = 12,288 neurons → safe)
- **224×224 raw pipeline** — for CNN and transfer learning models, which apply their own built-in preprocessing (`preprocess_input`)

---

## Experiment Tracking — Weights & Biases

All experiment runs are tracked using W&B under the project `flower-classification`.

- **Model 2 (Hidden Layer NN)** — logged as a single named run to demonstrate manual experiment tracking
- **Model 3 (Improved NN)** — full Bayesian sweep across 6 hyperparameters, 10 trials. Best configuration is used to retrain a final model with CosineDecay learning rate scheduling and EarlyStopping

| Setting | Value |
|---------|-------|
| W&B project | `flower-classification` |
| Sweep method | Bayesian optimisation |
| Trials | 10 |
| Optimise metric | `val_accuracy` (maximize) |
| LR scheduler on best config | CosineDecay |

---


## Setup & Usage

This project is designed to run in **Google Colab** with a GPU runtime.

```bash
# 1. Clone repository
git clone https://github.com/your-username/flower-classification.git

# 2. Upload archive.zip (flowers dataset) to /content/ in Colab

# 3. Add WANDB_API_KEY to Colab Secrets
#    (click the key icon on the left panel in Colab)

# 4. Open flower_research_complete.py and run all sections sequentially
```

---

## Evaluation Methodology

Every model is evaluated using the same three tools applied consistently to the held-out validation set:

1. **Training curves** — loss and accuracy per epoch, training vs. validation
2. **Confusion matrix** — per-class prediction breakdown visualised as a seaborn heatmap
3. **Classification report** — precision, recall, F1-score per class, plus macro and weighted averages
4. **Final comparison** — a unified bar chart and table ranking all 7 models by validation accuracy

---

## Key Technical Decisions

| Decision | Rationale |
|----------|-----------|
| 64×64 for Dense models | 224×224 flatten → 150,528 neurons causes GPU OOM. 64×64 → 12,288 neurons is sufficient for Dense layers. |
| Bayesian over grid search | Bayesian optimisation finds better configurations in fewer trials by modelling the search space probabilistically. |
| Two-phase fine-tuning | Freeze base → train head first, then unfreeze top layers at 10× lower LR. Prevents catastrophic forgetting of ImageNet weights. |
| EfficientNetB0 for TL baseline | Best accuracy-per-parameter ratio among CNNs. Compound scaling balances width, depth, and resolution simultaneously. |
| `keras.backend.clear_session()` | Called between W&B sweep runs to release GPU memory and prevent OOM accumulation across trials. |
| `gc.collect()` + `del model` | Explicit Python garbage collection after each sweep run ensures clean memory state. |

---

## Requirements

```
tensorflow>=2.15
wandb>=0.25
scikit-learn>=1.3
seaborn>=0.13
matplotlib>=3.8
numpy>=1.24
```

Install with:
```bash
pip install tensorflow wandb scikit-learn seaborn matplotlib numpy
```

---

## Results

Final validation accuracies are logged to the console and plotted as a comparative bar chart at the end of the notebook. Individual confusion matrices and classification reports are generated for each model.

> Expected ordering by performance: Linear NN < Hidden NN < Improved NN < AlexNet < VGG16 ≈ ResNet50 < EfficientNetB0

---


## References

- Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). ImageNet Classification with Deep Convolutional Neural Networks. *NeurIPS*.
- Simonyan, K., & Zisserman, A. (2014). Very Deep Convolutional Networks for Large-Scale Image Recognition. *ICLR 2015*.
- He, K., Zhang, X., Ren, S., & Sun, J. (2015). Deep Residual Learning for Image Recognition. *CVPR 2016*.
- Tan, M., & Le, Q. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. *ICML 2019*.
- TensorFlow Flowers Dataset — [tensorflow.org/datasets/catalog/tf_flowers](https://www.tensorflow.org/datasets/catalog/tf_flowers)
