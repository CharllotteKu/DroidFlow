# extract_ood_features_variants.py (多模型通用版)
import pandas as pd
import torch
import pickle
import argparse
import os
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# --- 导入所有模型定义 ---
from model import BiLSTMClassifier
from model_variants import GRUClassifier, CNNClassifier, RNNClassifier

# ------------------------------------------------------------
class APIOODDataset(Dataset):
    def __init__(self, df):
        self.X = df['input_ids'].tolist()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx])

def main():
    # ---------------------- 参数解析 ----------------------
    parser = argparse.ArgumentParser(description="Extract OOD features using BiLSTM/GRU/RNN/CNN")

    # --- 核心参数：模型类型 ---
    parser.add_argument('--model_type', type=str, default='BiLSTM', choices=['BiLSTM', 'GRU', 'CNN', 'RNN'], 
                        help='模型架构类型')

    # --- 文件路径 ---
    parser.add_argument('--vocab_path', type=str, default='vocab4.pkl', help='词表文件路径')
    parser.add_argument('--model_path', type=str, required=True, help='预训练的模型权重文件 (.pth)')
    parser.add_argument('--ind_features_path', type=str, required=True,
                        help='对应的 IND 特征文件 (用于计算中心化均值)')
    parser.add_argument('--ood_csv', type=str, default='ood.csv', help='包含OOD数据的CSV文件')
    parser.add_argument('--output_path', type=str, required=True, help='输出的OOD特征文件名 (.pt)')

    # --- 模型和数据参数 ---
    parser.add_argument('--max_len', type=int, default=5000, help='序列最大长度')
    parser.add_argument('--embed_dim', type=int, default=128, help='Embedding dimension')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension (CNN为num_filters)')
    parser.add_argument('--batch_size', type=int, default=64, help='批处理大小')

    args = parser.parse_args()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {DEVICE}")
    print(f"[INFO] Selected Model Architecture: {args.model_type}")

    # ---------------------- 加载IND特征 (用于获取类别数和中心化) ----------------------
    print(f"[INFO] Loading IND features from {args.ind_features_path}...")
    if not os.path.exists(args.ind_features_path):
        print(f"[ERROR] IND feature file not found: {args.ind_features_path}")
        return

    ind_data = torch.load(args.ind_features_path, map_location='cpu')
    ind_features = ind_data['features']
    ind_labels = ind_data['labels']
    num_classes = len(ind_labels.unique())
    feature_dim = ind_features.shape[1] # IND 特征的维度

    print(f"[INFO] Auto-detected: num_classes={num_classes}, feature_dim={feature_dim}")

    # ---------------------- 加载词表 ----------------------
    if not os.path.exists(args.vocab_path):
        print(f"[ERROR] Vocab file not found: {args.vocab_path}")
        return
        
    with open(args.vocab_path, "rb") as f:
        vocab = pickle.load(f)
    vocab_size = len(vocab)

    # ---------------------- 初始化模型 ----------------------
    print(f"[INFO] Initializing {args.model_type} model...")
    
    if args.model_type == 'BiLSTM':
        # BiLSTM 特征是 hidden*2，所以传入 hidden = feature_dim // 2
        model = BiLSTMClassifier(vocab_size, args.embed_dim, args.hidden_dim, num_classes).to(DEVICE)
    elif args.model_type == 'GRU':
        model = GRUClassifier(vocab_size, args.embed_dim, args.hidden_dim, num_classes).to(DEVICE)
    elif args.model_type == 'RNN':
        model = RNNClassifier(vocab_size, args.embed_dim, args.hidden_dim, num_classes).to(DEVICE)
    elif args.model_type == 'CNN':
        # CNN 的 feature_dim 是 num_filters * len(kernel_sizes)
        # 这里假设 args.hidden_dim 是 num_filters，kernel_sizes=[3,4,5] (默认)
        model = CNNClassifier(vocab_size, args.embed_dim, num_classes, num_filters=args.hidden_dim).to(DEVICE)

    # ---------------------- 加载权重 ----------------------
    if not os.path.exists(args.model_path):
        print(f"[ERROR] Model weights not found: {args.model_path}")
        return

    print(f"[INFO] Loading weights from {args.model_path}...")
    model.load_state_dict(torch.load(args.model_path, map_location=DEVICE))
    model.eval()

    # ---------------------- 编码函数 ----------------------
    def encode(tokens):
        ids = [vocab.get(tok, 0) for tok in tokens]
        if len(ids) < args.max_len:
            return ids + [0] * (args.max_len - len(ids))
        return ids[:args.max_len]

    # ---------------------- 加载 OOD 数据 ----------------------
    if not os.path.exists(args.ood_csv):
        print(f"[ERROR] OOD CSV not found: {args.ood_csv}")
        return

    print(f"[INFO] Processing OOD samples from {args.ood_csv}...")
    df_ood = pd.read_csv(args.ood_csv)
    # 确保是字符串并分词
    df_ood['tokens'] = df_ood['sequence'].astype(str).apply(lambda x: x.strip().split())
    df_ood['input_ids'] = df_ood['tokens'].apply(encode)

    test_loader = DataLoader(APIOODDataset(df_ood), batch_size=args.batch_size)

    # ---------------------- 提取特征 ----------------------
    print(f"[INFO] Extracting features...")
    all_features_raw = []
    with torch.no_grad():
        for x in tqdm(test_loader):
            x = x.to(DEVICE)
            # 兼容所有模型接口：forward 返回 (logits, features)
            _, features = model(x)
            all_features_raw.append(features.cpu())

    features_raw_tensor = torch.cat(all_features_raw, dim=0)
    print(f"[INFO] Extracted features shape: {features_raw_tensor.shape}")

    # 验证维度匹配
    if features_raw_tensor.shape[1] != feature_dim:
        print(f"[WARNING] OOD feature dim ({features_raw_tensor.shape[1]}) != IND feature dim ({feature_dim})")
        print("这可能意味着 IND 特征文件和当前加载的模型不匹配！中心化可能会出错。")

    # ---------------------- 中心化处理 ----------------------
    # 逻辑：Flow 模型是在 (X - Mean) 上训练的
    # OOD 数据也必须减去同样的 Mean (所有 IND 类中心的平均)
    
    print("[INFO] Calculating global mean from IND features...")
    class_means = []
    for i in range(num_classes):
        # 找出属于第 i 类的 IND 特征
        feats_i = ind_features[ind_labels == i]
        if len(feats_i) > 0:
            class_mean = feats_i.mean(dim=0, keepdim=True)
            class_means.append(class_mean)
    
    if len(class_means) > 0:
        avg_mean = torch.mean(torch.stack(class_means), dim=0)
    else:
        avg_mean = torch.zeros(1, feature_dim)

    print("[INFO] Applying centralization...")
    features_centered = features_raw_tensor - avg_mean

    # ---------------------- 保存结果 ----------------------
    print(f"[INFO] Saving to {args.output_path}...")
    torch.save({
        "features": features_centered,
        "raw": features_raw_tensor, # 同时保存原始特征以备不时之需
    }, args.output_path)

    print("✅ Done!")

if __name__ == '__main__':
    main()