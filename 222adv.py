# 222adv.py (最终版：支持按块定义深度)
import os
import torch
import numpy as np
from torch import nn
from tqdm import tqdm
import argparse
import math
# 确保导入的是最新的模型定义
from cancha_advanced import AdvancedFlow

# ---------------- 参数解析 ----------------
parser = argparse.ArgumentParser(description="Advanced Flow OOD Inference")
parser.add_argument('--feature_val', type=str, required=True, help='验证/测试集 IND 特征')
parser.add_argument('--feature_ood', type=str, required=True, help='OOD特征')
parser.add_argument('--outf', type=str, required=True, help='输出目录')
parser.add_argument('--model_dir', type=str, required=True, help='每类 flow 模型目录')
parser.add_argument('--length_hidden', type=int, default=1, help='Flow 子网络隐层宽度超参 h')

# --- MODIFIED: 与 111adv.py 同步模型结构参数 ---
parser.add_argument('--num_blocks', type=int, default=5, help='Flow模型中线性+非线性块的数量')
parser.add_argument('--layers_per_block', type=int, default=3, help='每个块中包含的非线性耦合层数量')
# ---------------------------------------------
args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- 加载特征 ----------------
print(f"[INFO] 加载特征文件: {args.feature_val} 和 {args.feature_ood}")
data = torch.load(args.feature_val, map_location=DEVICE)
features, labels = data['features'], data['labels']
num_classes = labels.max().item() + 1

# 按类别取每类后10%作为IND测试集
X_val_list = []
print("[INFO] 按类别取每类后10%样本作为 IND 测试集 ...")
for i in range(num_classes):
    class_feats = features[labels == i]
    n_total = len(class_feats)
    if n_total == 0: continue
    mean = class_feats.mean(dim=0, keepdim=True)
    n_last = max(1, int(0.10 * n_total))
    start_idx = n_total - n_last
    feats_last10 = class_feats[start_idx:]
    X_val_list.append(feats_last10 - mean)

X_val = torch.cat(X_val_list, dim=0).to(DEVICE)
print(f"[INFO] 总IND测试样本数(后10%): {len(X_val)}")

X_ood = torch.load(args.feature_ood, map_location=DEVICE)['features'].to(DEVICE)

# ---------------- 构建并加载模型 ----------------
print(f"[INFO] 从目录 '{args.model_dir}' 加载 AdvancedFlow 模型...")
dim = X_val.shape[1]

# --- REMOVED: 移除固定的掩码创建逻辑 ---
# ... (原mask_np, mask创建代码已删除)

flow = []
for i in range(num_classes):
    # --- MODIFIED: 使用新的初始化方式 ---
    model = AdvancedFlow(
        dim=dim,
        h=args.length_hidden,
        num_blocks=args.num_blocks,
        layers_per_block=args.layers_per_block
    ).to(DEVICE)
    # ------------------------------------
    
    model_path = os.path.join(args.model_dir, f"resflow_class_{i}.pth")

    if not os.path.exists(model_path):
        print(f"[ERROR] 模型文件不存在: {model_path}")
        exit()

    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    flow.append(model)


# ---------------- 推理：使用 max 聚合每类 log_prob ----------------
def get_logp(X):
    if len(X) == 0:
        return np.array([])
    scores = []
    with torch.no_grad():
        for m in flow:
            s = []
            for i in range(0, len(X), 128):
                s.append(m.log_prob(X[i:i + 128]).cpu())
            scores.append(torch.cat(s))
    stacked = torch.stack(scores, dim=1)
    max_scores, _ = torch.max(stacked, dim=1)
    return max_scores.detach().numpy()


print("[INFO] 正在计算 log_prob 得分（max）...")
s_val = get_logp(X_val)
s_ood = get_logp(X_ood)

# ---------------- 保存 logp ----------------
os.makedirs(args.outf, exist_ok=True)
output_file = os.path.join(args.outf, "rnn_resflow_1.npz") # 使用一个通用的文件名
np.savez(output_file, ind=s_val, ood=s_ood)

print(f"✅ 推理完成，logp (max) 得分保存至 {output_file}")