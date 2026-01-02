# DroidFlow: Learning Behavioral Densities from Android API Sequences

This is the official PyTorch implementation of the paper:
**"DroidFlow: Learning Behavioral Densities from Android API Sequences"** (Submitted to *Journal of Systems Architecture*)

**Authors:** Wanyi Gu, Guojun Wang, et al.

## 🚀 Overview
DroidFlow is a deep generative framework for Out-of-Distribution (OOD) Android malware detection. It combines a **BiLSTM** encoder for sequence representation learning with a **class-conditional Flow-based model** for high-fidelity density estimation.

## 📂 Repository Structure
* `model.py`: Implementation of the BiLSTM encoder.
* `cancha_advanced.py`: Implementation of the Advanced Flow-based density estimator.
* `bilstm_resflow_pipeline1.py`: Pipeline for training the BiLSTM encoder and extracting features.
* `111adv.py`: Script for training the Flow-based density estimators (IND).
* `222adv.py`: Script for OOD inference and score calculation.
* `extract_ood_features.py`: Helper script to extract features from OOD samples.
* `evaluate_ood_scores.py`: Script to compute AUROC, TNR@95TPR, and visualize results.

## 🛠️ Requirements
This code is implemented in Python 3.8+ and PyTorch 1.10+.
To install dependencies:
```bash
pip install -r requirements.txt
