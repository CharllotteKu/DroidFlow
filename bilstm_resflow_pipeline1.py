# bilstm_resflow_pipeline1.py (修改版)

import os
import csv
import math
import pickle
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from collections import Counter
from model import BiLSTMClassifier   # 保持原有模型不变

# ----------------------
# CONFIG
# ----------------------
CSV_PATH   = "ind4.csv"       # 你的CSV文件
MAX_LEN    = 5000
EMBED_DIM  = 128
HIDDEN_DIM = 128
BATCH_SIZE = 64
EPOCHS     = 20
LR         = 1e-3
WEIGHT_DECAY = 1e-4           # 适度权重衰减缓解过拟合
SEED       = 42

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 定义与.pt文件匹配的模型文件名
MODEL_SAVE_PATH = "bilstm_model4.pth"
VOCAB_SAVE_PATH = "vocab4.pkl"
FEATURES_SAVE_PATH = "features_for_resflow4.pt"


torch.manual_seed(SEED)
np.random.seed(SEED)

print("[INFO] Step 1: Loading CSV and building vocab...")

# ----------------------
# STEP 1: LOAD CSV & BUILD VOCAB
# ----------------------
df = pd.read_csv(CSV_PATH)
assert 'sequence' in df.columns and 'label' in df.columns, \
    "CSV 必须包含列 'sequence' 与 'label'"

# 分词（空格分隔的 API token）
df['tokens'] = df['sequence'].astype(str).apply(lambda x: x.strip().split())

# 构建词表
all_tokens = [tok for tokens in df['tokens'] for tok in tokens]
token_freq = Counter(all_tokens)
vocab = {token: idx + 1 for idx, (token, _) in enumerate(token_freq.items())}
vocab['<PAD>'] = 0

with open(VOCAB_SAVE_PATH, "wb") as f:
    pickle.dump(vocab, f)

def encode(tokens):
    ids = [vocab.get(tok, 0) for tok in tokens]
    if len(ids) >= MAX_LEN:
        return ids[:MAX_LEN]
    return ids + [0] * (MAX_LEN - len(ids))

df['input_ids'] = df['tokens'].apply(encode)

# 标签编码
label_encoder = LabelEncoder()
df['label_id'] = label_encoder.fit_transform(df['label'])
NUM_CLASSES = df['label_id'].nunique()

print("[INFO] Vocab size:", len(vocab))
print("[INFO] Total samples:", len(df))
print("[INFO] Class count:", NUM_CLASSES)

# ----------------------
# STEP 2: Dataset & DataLoader
# ----------------------
print("[INFO] Step 2: Splitting dataset...")

class APIDataset(Dataset):
    def __init__(self, df_slice):
        self.X = df_slice['input_ids'].tolist()
        self.y = df_slice['label_id'].tolist()
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.long), torch.tensor(self.y[idx], dtype=torch.long)

df_train, df_val = train_test_split(
    df, test_size=0.2, stratify=df['label_id'], random_state=SEED
)

train_loader = DataLoader(APIDataset(df_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(APIDataset(df_val), batch_size=BATCH_SIZE, shuffle=False)

print("[INFO] Train samples:", len(df_train))
print("[INFO] Validation samples:", len(df_val))

# ----------------------
# STEP 3: BiLSTM Model
# ----------------------
print("[INFO] Step 3: Building BiLSTM model...")

model = BiLSTMClassifier(
    vocab_size=len(vocab),
    embed_dim=EMBED_DIM,
    hidden_dim=HIDDEN_DIM,
    num_classes=NUM_CLASSES
).to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

# ----------------------
# helpers
# ----------------------
def evaluate(m, crit, loader, device):
    m.eval()
    total, correct, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits, _ = m(xb)
            loss = crit(logits, yb)
            loss_sum += loss.item() * yb.size(0)
            pred = logits.argmax(1)
            correct += (pred == yb).sum().item()
            total += yb.size(0)
    return loss_sum / max(total, 1), correct / max(total, 1)

# ----------------------
# STEP 4: Train BiLSTM (with val eval, early stop, LR scheduler)
# ----------------------
print("[INFO] Step 4: Training BiLSTM...")

best_val_acc = -math.inf
patience, bad_epochs = 5, 0

log_path = "train_val_log.csv"
with open(log_path, "w", newline="", encoding="utf-8") as f:
    csv.writer(f).writerow(["epoch","train_loss","train_acc","val_loss","val_acc","lr"])

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=2, verbose=True
)

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        logits, _ = model(xb)
        loss = criterion(logits, yb)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item() * yb.size(0)
        correct += (logits.argmax(1) == yb).sum().item()
        total += yb.size(0)

    train_loss = total_loss / total
    train_acc  = correct / total
    val_loss, val_acc = evaluate(model, criterion, val_loader, DEVICE)

    lr = optimizer.param_groups[0]["lr"]
    print(f"Epoch {epoch}/{EPOCHS} | Train Loss {train_loss:.4f} Acc {train_acc:.4f} | "
          f"Val Loss {val_loss:.4f} Acc {val_acc:.4f} | LR {lr:.6f}")

    with open(log_path, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([epoch, f"{train_loss:.6f}", f"{train_acc:.6f}",
                                f"{val_loss:.6f}", f"{val_acc:.6f}", f"{lr:.6f}"])

    scheduler.step(val_acc)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        # --- MODIFIED: Save with the new filename ---
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        bad_epochs = 0
    else:
        bad_epochs += 1
        if bad_epochs >= patience:
            print(f"[Early Stopping] Best Val Acc={best_val_acc:.4f}")
            break

# --- REMOVED: No longer saving the last model ---
print(f"[INFO] Best BiLSTM model saved to {MODEL_SAVE_PATH}.")

# ----------------------
# STEP 5: Extract Features for Residual Flow (all data, one file)
# ----------------------
print("[INFO] Step 5: Extracting features for residual flow (whole dataset, one file)...")

# --- MODIFIED: Load from the new filename to ensure consistency ---
if os.path.exists(MODEL_SAVE_PATH):
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
else:
    print(f"[ERROR] Saved model '{MODEL_SAVE_PATH}' not found. Cannot extract features.")
    # exit() # You might want to uncomment this to stop the script if the model is missing
model.eval()

# 用全数据的 DataLoader
full_loader = DataLoader(APIDataset(df), batch_size=BATCH_SIZE, shuffle=False)
feats, labs = [], []
with torch.no_grad():
    for xb, yb in full_loader:
        xb = xb.to(DEVICE)
        _, h = model(xb)
        feats.append(h.cpu())
        labs.extend(yb.numpy())

feats = torch.cat(feats, dim=0)
labs = torch.tensor(labs, dtype=torch.long)
torch.save({"features": feats, "labels": labs}, FEATURES_SAVE_PATH)
print(f"✅ Saved {FEATURES_SAVE_PATH}  features={tuple(feats.shape)}  labels={tuple(labs.shape)}")