# Transfer Learning for Multi-Class Image Classification

A production-style deep learning pipeline for **multi-class image classification** using **transfer learning**, built and evaluated across multiple ImageNet-pretrained architectures.  
This project benchmarks modern CNN backbones on a **17-class jute pest image dataset**, emphasizing generalization, empirical regularization, and rigorous evaluation.

---

## Overview

This project implements an end-to-end **transfer learning framework** for image classification using TensorFlow/Keras.  
Five pretrained convolutional neural networks are repurposed as **fixed feature extractors**, with custom classification heads trained on top to efficiently learn from a limited dataset.

The pipeline includes:

- Stratified train/validation/test splits
- GPU-accelerated image augmentation
- Model-specific preprocessing
- Early stopping and checkpointing
- Comprehensive multi-class evaluation

---

## Models Evaluated

The following ImageNet-pretrained architectures were evaluated under a unified training and evaluation pipeline:

- **ResNet50**
- **ResNet101**
- **EfficientNetB0**
- **VGG16**
- **DenseNet201**

For each model:

- All backbone layers were **frozen**
- A custom classification head was attached using:
  - Global Average Pooling
  - Batch Normalization
  - ReLU-activated dense layers
  - Dropout (0.2)
  - L2 regularization
- Models were optimized using **Adam** with **categorical cross-entropy**

---

## Data Processing & Augmentation

- Images resized to **224 × 224**
- Labels encoded using **one-hot encoding**
- Empirical regularization via on-GPU augmentation:
  - Random crop & resize
  - Horizontal flip
  - Rotation, zoom, translation
  - Contrast adjustment
- Stratified validation split (20% per class)

---

## Training Strategy

- Trained for **50 epochs** with **early stopping**
- Best model selected based on **lowest validation loss**
- Separate generators used for:
  - Augmented training
  - Clean evaluation
- Deterministic training via fixed random seeds

---

## Evaluation Metrics

All models were evaluated on **training, validation, and test sets** using:

- Accuracy
- Precision (weighted)
- Recall (weighted)
- **F1-Score (weighted)**
- **Multi-class ROC-AUC (OvR)**

Confusion matrices and learning curves were generated for detailed error analysis.

---

## Results Summary

- All architectures achieved **~97% test accuracy**
- **EfficientNetB0** achieved the strongest overall performance:
  - **Test F1-Score:** 0.974
  - **Test AUC:** 0.9999
- Performance differences across models were marginal, indicating strong generalization across architectures
- Training and validation curves remained closely aligned, showing minimal overfitting

---

## Key Takeaways

- Transfer learning is highly effective for multi-class classification on limited datasets
- Lightweight architectures (EfficientNetB0) can outperform deeper networks when properly regularized
- F1-Score and AUC provide more reliable insight than accuracy alone for multi-class problems

---

## Tech Stack

- **Python**
- **Pytorch**
- **TensorFlow / Keras**
- **NumPy, Pandas**
- **scikit-learn**
- **OpenCV**
- **Matplotlib, Seaborn**

---

## Project Context

This project was completed as the **final project for DSCI 552 (Machine Learning for Data Science)** at the University of Southern California.  
Starter dataset provided by course staff; all modeling, experimentation, and analysis were completed independently.

---

## Author

**Mahin Mohan**  
M.S. Computer Science, University of Southern California
