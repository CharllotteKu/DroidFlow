# evaluate_ood_scores_debug.py
# (增强版：在图上绘制TNR@TPR=95的决策阈值)

import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve
from scipy.integrate import trapezoid
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# ---------- 加载得分 ----------
# 请确保这里的文件名是您正在分析的那个
NPZ_FILE = "kuaiceng_ood_scores/bilstm_resflow_53_2.npz"
print(f"[INFO] 正在加载得分文件: {NPZ_FILE}")
try:
    data = np.load(NPZ_FILE)
    s_ind, s_ood = data["ind"], data["ood"]
except FileNotFoundError:
    print(f"[ERROR] 文件 '{NPZ_FILE}' 未找到！请检查路径。")
    exit()

# 确保数据不为空
if len(s_ind) == 0 or len(s_ood) == 0:
    print("[ERROR] IND 或 OOD 分数为空，无法进行评估。")
    exit()

labels = np.concatenate([np.ones_like(s_ind), np.zeros_like(s_ood)])
scores = np.concatenate([s_ind, s_ood])

# ---------- 1. 计算各项指标 (逻辑与之前相同) ----------
fpr, tpr, thresholds = roc_curve(labels, scores)
auroc = roc_auc_score(labels, scores)

# 找到满足TPR>=0.95的第一个点
# 添加一个检查，以防在某些情况下找不到满足条件的点
tpr95_indices = np.where(tpr >= 0.95)[0]
if len(tpr95_indices) == 0:
    print("[ERROR] 未能找到TPR>=0.95的阈值点，无法计算TNR@TPR=95。")
    tpr95_idx = -1 # 使用最后一个点作为备用
else:
    tpr95_idx = tpr95_indices[0]

tnr_at_tpr95 = 1 - fpr[tpr95_idx]
decision_threshold = thresholds[tpr95_idx] # <<<<<< 获取对应的决策阈值

y_pred = scores >= decision_threshold
dtacc = (y_pred == labels).mean()

prec_in, rec_in, _ = precision_recall_curve(labels, scores)
rec_in_sorted, prec_in_sorted = zip(*sorted(zip(rec_in, prec_in)))
auin = trapezoid(prec_in_sorted, rec_in_sorted)

prec_out, rec_out, _ = precision_recall_curve(1 - labels, -scores)
rec_out_sorted, prec_out_sorted = zip(*sorted(zip(rec_out, prec_out)))
auout = trapezoid(prec_out_sorted, rec_out_sorted)

# ---------- 打印结果 ----------
print("\n✅ OOD 检测评估指标：")
print(f"AUROC          : {auroc:.4f}")
print(f"TNR@TPR=95     : {tnr_at_tpr95:.4f}")
print(f"Detection Acc  : {dtacc:.4f}")
print(f"AUIN           : {auin:.4f}")
print(f"AUOUT          : {auout:.4f}")
print("-" * 20)
print(f"📊 计算TNR@TPR=95时使用的决策阈值为: {decision_threshold:.4f}")
print("-" * 20)


# ---------- 2. 分布图可视化 (核心修改) ----------
df = pd.DataFrame({
    "Score": scores,
    "Type": ["IND"] * len(s_ind) + ["OOD"] * len(s_ood)
})

plt.figure(figsize=(10, 6))
sns.histplot(data=df, x="Score", hue="Type", bins=100, kde=True, stat="density", palette="muted")

# ========== 在图上绘制决策阈值 ==========
plt.axvline(x=decision_threshold, color='red', linestyle='--', linewidth=2,
            label=f'TNR@TPR=95 Threshold = {decision_threshold:.2f}')
# =========================================

plt.title("IND vs OOD Score Distribution with Decision Threshold")
plt.xlabel("Score")
plt.ylabel("Density")
plt.legend()
plt.grid(True)
plt.tight_layout()

output_filename = "score_distribution_with_threshold.png"
plt.savefig(output_filename)
plt.close()

print(f"✅ 带有决策边界的分布图已保存至: {output_filename}")