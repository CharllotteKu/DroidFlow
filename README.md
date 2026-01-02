
# DroidFlow: Learning Behavioral Densities from Android API Sequences

This is the official PyTorch implementation of the paper:
**"DroidFlow: Learning Behavioral Densities from Android API Sequences"** (Submitted to *Journal of Systems Architecture*)

**Authors:** Wanyi Gu, Guojun Wang, et al.

## 🚀 Overview
DroidFlow is a deep generative framework for Out-of-Distribution (OOD) Android malware detection. It combines a **BiLSTM** encoder for sequence representation learning with a **class-conditional Flow-based model** for high-fidelity density estimation.

## 📂 Repository Structure
* **Core Models:**
  * `model.py`: Implementation of the BiLSTM feature extractor.
  * `cancha_advanced.py`: Implementation of the Advanced Flow-based density estimator.
* **Training & Inference Pipelines:**
  * `bilstm_resflow_pipeline1.py`: Main pipeline for training the BiLSTM encoder and extracting features from the IND dataset.
  * `111adv.py`: Script for training the Flow-based density estimators (IND).
  * `222adv.py`: Script for OOD inference and score calculation.
* **Evaluation:**
  * `evaluate_ood_scores.py`: Computes AUROC, TNR@95TPR, and visualizes results.
  * `benchmark_efficiency.py`: **(New)** Benchmarks computational efficiency (Latency/FLOPs/Memory).
* **Utils:**
  * `extract_ood_features.py`: Helper to extract features from OOD samples.

## 🛠️ Requirements
* Python 3.8+
* PyTorch >= 1.10
* Dependencies are listed in `requirements.txt`. To install:
```bash
pip install -r requirements.txt

```

## ⚡ Quick Start

### 1. Train BiLSTM Encoder & Extract IND Features

Prepare your IND dataset (ensure it is named `ind4.csv` or update the path in the script).

```bash
python bilstm_resflow_pipeline1.py

```

> **Output:** `bilstm_model4.pth` (Model weights), `features_for_resflow4.pt` (Extracted IND features)

### 2. Train Flow Models

Train the density estimators on the extracted IND features.

```bash
python 111adv.py --feature_file features_for_resflow4.pt --outf saved_flow_models

```

### 3. Prepare OOD Features

Extract features from your OOD dataset using the trained BiLSTM model.

```bash
python extract_ood_features.py --csv_path ood_data.csv --model_path bilstm_model4.pth --outf features_ood.pt

```

*(Note: You may need to adjust the arguments based on your OOD data path)*

### 4. OOD Detection & Evaluation

Run inference on mixed IND and OOD samples to calculate anomaly scores and evaluation metrics.

```bash
python 222adv.py --feature_val features_for_resflow4.pt --feature_ood features_ood.pt --outf results
python evaluate_ood_scores.py

```


## 📊 Dataset

Due to privacy and copyright concerns, the raw APK files cannot be shared. However, to facilitate reproducibility, we provide the processed API sequences used in our experiments.

Data Access: The dataset is available at: https://pan.quark.cn/s/d020f384a92d

File Naming Logic: Leave-One-Out

The files are named based on the class that is treated as Out-of-Distribution (OOD).

ood1.csv: Contains samples from Class 1 only. In this experiment, Class 1 is treated as the anomaly (OOD).

ind1.csv: Contains samples from all other classes (excluding Class 1). These are treated as the normal (In-Distribution) data.

Ideally, indX.csv represents the normal data when Class X is held out as the anomaly.

<img width="480" height="327" alt="image" src="https://github.com/user-attachments/assets/088a7a6d-976c-4023-9022-e651f0318772" />


## 📜 Citation

If you find this code useful, please cite our paper:

```bibtex
@article{droidflow2026,
  title={DroidFlow: Learning Behavioral Densities from Android API Sequences},
  author={Gu, Wanyi and Wang, Guojun and Chen, Mingfei and others},
  journal={Journal of Systems Architecture},
  year={2026}
}

```



