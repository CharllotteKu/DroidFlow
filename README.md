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
* See `requirements.txt` for details.

## ⚡ Quick Start

### 1. Train BiLSTM Encoder
Prepare your IND dataset as `ind4.csv` (or modify the path in the script).
```bash
python bilstm_resflow_pipeline1.py
Output: bilstm_model4.pth, features_for_resflow4.pt

2. Train Flow Models
Train the density estimators on the extracted features.

Bash

python 111adv.py --feature_file features_for_resflow4.pt --outf saved_flow_models
3. OOD Detection & Evaluation
Run inference on mixed IND and OOD samples and calculate metrics.

Bash

python 222adv.py --feature_val features_for_resflow4.pt --feature_ood features_ood.pt --outf results
python evaluate_ood_scores.py
📊 Dataset
Due to privacy and copyright concerns, the raw APK files cannot be shared. We provide the processed API sequences (integer-mapped) used in our experiments.

📜 Citation
If you find this code useful, please cite our paper:

Code snippet

@article{droidflow2026,
  title={DroidFlow: Learning Behavioral Densities from Android API Sequences},
  author={Gu, Wanyi, et al.},
  journal={Journal of Systems Architecture},
  year={2026}
}

---

### **2. `requirements.txt` 
**操作**：新建 `requirements.txt`，粘贴以下内容：

```text
torch>=1.10.0
numpy>=1.20.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
seaborn>=0.11.0
tqdm>=4.60.0
thop>=0.1.0  
psutil       # 用于计算内存占用
