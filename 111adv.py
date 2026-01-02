# 111adv.py (最终版：支持按块定义深度和多GPU)
import os
import json
import copy
import math
import argparse
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from collections import OrderedDict
# 确保导入的是最新的模型定义
from cancha_advanced import AdvancedFlow

# ---------------- 参数 ----------------
parser = argparse.ArgumentParser(description='Advanced Flow Training (Multi-GPU, Block-wise definition)')
parser.add_argument('--feature_file', type=str, required=True, help='特征文件路径')
parser.add_argument('--outf', type=str, required=True, help='输出目录')
parser.add_argument('--length_hidden', type=int, default=1, help='Flow 子网络隐层宽度超参 h')
parser.add_argument('--num_iter', type=int, default=2000, help='训练总迭代步数')
parser.add_argument('--batch_size', type=int, default=128, help='【总】批处理大小，会被均分到各个GPU')
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--eval_interval', type=int, default=100, help='验证频率（步）')
parser.add_argument('--patience', type=int, default=5, help='早停耐心')

# --- MODIFIED: 明确定义模型结构参数 ---
parser.add_argument('--num_blocks', type=int, default=5, help='Flow模型中线性+非线性块的数量')
parser.add_argument('--layers_per_block', type=int, default=3, help='每个块中包含的非线性耦合层数量')
# ----------------------------------------
args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- 数据加载（仅 IND） ----------------
print(f"[INFO] 加载 IND 特征与标签: {args.feature_file}")
data = torch.load(args.feature_file, map_location=DEVICE)
features, labels = data['features'], data['labels']
num_classes = int(labels.max().item() + 1)

# 类内中心化 & 训练/验证拆分
X_train, X_val, means = [], [], []
print("[INFO] 按类别进行中心化并拆分训练/验证...")
for c in range(num_classes):
    feats_c = features[labels == c]
    n_total = len(feats_c)
    if n_total == 0:
        X_train.append(torch.empty(0, features.shape[1], device=DEVICE))
        X_val.append(torch.empty(0, features.shape[1], device=DEVICE))
        means.append(torch.zeros(1, features.shape[1], device=DEVICE))
        continue
    mean_c = feats_c.mean(dim=0, keepdim=True)
    means.append(mean_c.to(DEVICE))
    n_val = max(1, int(0.20 * n_total))
    n_discard = int(0.10 * n_total)
    n_train_end = n_total - n_discard
    n_train_start = n_val
    feats_val = feats_c[:n_val]
    feats_train = feats_c[n_train_start:n_train_end]
    X_val.append((feats_val - mean_c).to(DEVICE))
    X_train.append((feats_train - mean_c).to(DEVICE))

# --- REMOVED: 移除固定的掩码创建逻辑 ---
# ... (原mask_np, mask创建代码已删除)

# ---------------- 构建模型与优化器 ----------------
print(f"[INFO] 构建 AdvancedFlow 模型 (num_blocks={args.num_blocks}, layers_per_block={args.layers_per_block})...")

# --- MODIFIED: 使用新的初始化方式 ---
flows_raw = [
    AdvancedFlow(
        dim=features.shape[1],
        h=args.length_hidden,
        num_blocks=args.num_blocks,
        layers_per_block=args.layers_per_block
    ) for _ in range(num_classes)
]
# ------------------------------------

use_data_parallel = torch.cuda.is_available() and torch.cuda.device_count() > 1
if use_data_parallel:
    print(f"Let's use {torch.cuda.device_count()} GPUs!")
    flows = [nn.DataParallel(m).to(DEVICE) for m in flows_raw]
else:
    flows = [m.to(DEVICE) for m in flows_raw]

opts = [torch.optim.Adam(m.parameters(), lr=args.lr) for m in flows]


# ---------------- 验证指标 ----------------
@torch.no_grad()
def compute_ind_val_nll(flow_list, val_list):
    nlls = []
    for i, (m, xval) in enumerate(zip(flow_list, val_list)):
        if xval is None or len(xval) == 0: continue
        m.eval()
        parts = []
        for k in range(0, len(xval), 128):
            log_probs = m.module.log_prob(xval[k:k + 128]) if use_data_parallel else m.log_prob(xval[k:k + 128])
            parts.append(-log_probs.detach())
        nlls.append(torch.cat(parts).mean().item())
    if len(nlls) == 0: return float('inf')
    return float(np.mean(nlls))


# ---------------- 训练循环 ----------------
os.makedirs(args.outf, exist_ok=True)
print(f"[INFO] 开始训练。模型将保存至 '{args.outf}'")
history, best_val, best_states, epochs_no_improve = [], float('inf'), None, 0

pbar = tqdm(range(args.num_iter), desc="Training")
for it in pbar:
    for i in range(num_classes):
        xi = X_train[i]
        if xi is None or len(xi) == 0: continue
        flows[i].train()
        idx = torch.randint(low=0, high=xi.size(0), size=(min(args.batch_size, xi.size(0)),), device=DEVICE)
        batch = xi[idx]
        log_prob = flows[i].module.log_prob(batch) if use_data_parallel else flows[i].log_prob(batch)
        loss = -log_prob.mean()
        opts[i].zero_grad()
        loss.backward()
        opts[i].step()

    if (it + 1) % args.eval_interval == 0 or it == args.num_iter - 1:
        val_nll = compute_ind_val_nll(flows, X_val)
        history.append((it + 1, val_nll))
        pbar.set_postfix({'val_nll': f"{val_nll:.4f}"})

        if val_nll < best_val:
            best_val = val_nll
            epochs_no_improve = 0

            print(f"\n✨ New best IND-NLL: {best_val:.6f}. Saving states to CPU memory.")
            current_states = []
            for m in flows:
                model_to_save = m.module if use_data_parallel else m
                state_dict_cpu = {k: v.to('cpu') for k, v in model_to_save.state_dict().items()}
                current_states.append(state_dict_cpu)
            best_states = current_states

        else:
            epochs_no_improve += 1
        if epochs_no_improve >= args.patience:
            print(f"\n[INFO] 早停触发：连续 {epochs_no_improve} 次无改进。最佳 IND-NLL: {best_val:.6f}")
            break

# ---------------- 保存最佳模型 ----------------
if best_states is not None:
    for i, sd in enumerate(best_states):
        torch.save(sd, os.path.join(args.outf, f"resflow_class_{i}.pth"))
    print(f"[INFO] 已保存最佳模型到: {args.outf}")
else:
    print("[WARN] 未能保存模型")

# ---------------- 保存训练曲线 ----------------
hist_path = os.path.join(args.outf, "training_history_ind_only.json")
with open(hist_path, "w") as f:
    json.dump(history, f)
print(f"✅ 训练历史已保存: {hist_path}")