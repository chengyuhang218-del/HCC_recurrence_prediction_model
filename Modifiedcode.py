import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


# =========================
# 0. 参数
# =========================
EXPR_PATH = "/Users/chengyuhang/Desktop/GSE76427_expr_gene_RFS_108_clean.csv"
LABEL_PATH = "/Users/chengyuhang/Desktop/GSE76427_rfs_label_108.csv"

TOPK_GENES = 90
EPOCHS = 260
LR = 5e-4
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


# =========================
# 1. 读取数据
# =========================
print("Loading data...")

df = pd.read_csv(EXPR_PATH, index_col=0)

label_df = pd.read_csv(LABEL_PATH)
label_df = label_df.rename(columns={
    "sample": "sample_id",
    "recurrence": "label"
})
label_df["label"] = label_df["label"].astype(int)

common_samples = [s for s in df.columns if s in set(label_df["sample_id"])]

df = df[common_samples]
label_df = label_df.set_index("sample_id").loc[common_samples].reset_index()

gene_names_all = np.array(df.index.tolist())
sample_names = np.array(df.columns.tolist())

X = df.values.T.astype(np.float32)
y = label_df["label"].values.astype(int)

print(f"Samples with labels: {X.shape[0]}")
print(f"Genes: {X.shape[1]}")
print(f"Recurrence samples: {y.sum()}")
print(f"Non-recurrence samples: {(y == 0).sum()}")


# =========================
# 2. 分层划分 train / val / test
# =========================
print("\nSplitting train / val / test...")

X_trainval, X_test, y_trainval, y_test, sample_trainval, sample_test = train_test_split(
    X,
    y,
    sample_names,
    test_size=0.2,
    random_state=RANDOM_SEED,
    stratify=y
)

X_train, X_val, y_train, y_val, sample_train, sample_val = train_test_split(
    X_trainval,
    y_trainval,
    sample_trainval,
    test_size=0.25,
    random_state=RANDOM_SEED,
    stratify=y_trainval
)

print(f"Train samples: {X_train.shape[0]}")
print(f"Val samples: {X_val.shape[0]}")
print(f"Test samples: {X_test.shape[0]}")
print(f"Train recurrence: {y_train.sum()}")
print(f"Train non-recurrence: {(y_train == 0).sum()}")
print(f"Val recurrence: {y_val.sum()}")
print(f"Val non-recurrence: {(y_val == 0).sum()}")
print(f"Test recurrence: {y_test.sum()}")
print(f"Test non-recurrence: {(y_test == 0).sum()}")


# =========================
# 3. 只用训练集筛选差异基因
# =========================
def select_genes_by_tscore(X_train, y_train, gene_names, topk=100):
    pos = X_train[y_train == 1]
    neg = X_train[y_train == 0]

    pos_mean = pos.mean(axis=0)
    neg_mean = neg.mean(axis=0)

    pos_var = pos.var(axis=0)
    neg_var = neg.var(axis=0)

    n_pos = pos.shape[0]
    n_neg = neg.shape[0]

    se = np.sqrt(pos_var / n_pos + neg_var / n_neg)
    se[se == 0] = 1e-6

    t_score = (pos_mean - neg_mean) / se
    abs_t = np.abs(t_score)

    top_idx = np.argsort(abs_t)[::-1][:topk]

    selected_info = pd.DataFrame({
        "gene": gene_names[top_idx],
        "t_score": t_score[top_idx],
        "abs_t_score": abs_t[top_idx],
        "recurrence_mean": pos_mean[top_idx],
        "non_recurrence_mean": neg_mean[top_idx]
    })

    return top_idx, selected_info


print("\nSelecting recurrence-related genes...")

top_idx, selected_gene_info = select_genes_by_tscore(
    X_train,
    y_train,
    gene_names_all,
    TOPK_GENES
)

selected_genes = gene_names_all[top_idx]

X_train = X_train[:, top_idx]
X_val = X_val[:, top_idx]
X_test = X_test[:, top_idx]

print(f"Selected genes: {len(selected_genes)}")


# =========================
# 4. 标准化
# =========================
print("\nStandardizing data...")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)


# =========================
# 5. MLP 模型
# =========================
class MLPClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.classifier = nn.Linear(16, 1)

    def forward(self, x):
        embedding = self.feature_extractor(x)
        logits = self.classifier(embedding).squeeze(1)
        return logits, embedding


model = MLPClassifier(input_dim=TOPK_GENES)

x_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)

x_val_t = torch.tensor(X_val_scaled, dtype=torch.float32)
y_val_t = torch.tensor(y_val, dtype=torch.float32)

x_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)

pos_weight = torch.tensor(
    [(y_train == 0).sum() / (y_train == 1).sum()],
    dtype=torch.float32
)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LR,
    weight_decay=1e-3
)

print(f"\nPositive class weight: {pos_weight.item():.4f}")


# =========================
# 6. MLP 训练 + Early stopping
# =========================
print("\nTraining MLP...")

best_val_loss = float("inf")
best_state = None
patience = 20
wait = 0

for epoch in tqdm(range(EPOCHS), desc="Training MLP"):
    model.train()

    optimizer.zero_grad()

    train_logits, train_embedding = model(x_train_t)
    loss = criterion(train_logits, y_train_t)

    loss.backward()
    optimizer.step()

    model.eval()

    with torch.no_grad():
        val_logits, val_embedding = model(x_val_t)
        val_loss = criterion(val_logits, y_val_t).item()

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = model.state_dict()
        wait = 0
    else:
        wait += 1

    if (epoch + 1) % 20 == 0:
        print(
            f"\nEpoch [{epoch + 1}/{EPOCHS}] "
            f"Train Loss: {loss.item():.4f} "
            f"Val Loss: {val_loss:.4f}"
        )

    if wait >= patience:
        print(f"\nEarly stopping at epoch {epoch + 1}")
        break

model.load_state_dict(best_state)

print("MLP training finished.")


# =========================
# 7. Logistic Regression
# =========================
print("\nTraining Logistic Regression...")

logreg = LogisticRegression(
    penalty="elasticnet",
    l1_ratio=0.5,
    C=0.5,
    class_weight="balanced",
    solver="saga",
    max_iter=3000,
    random_state=RANDOM_SEED
)
logreg.fit(X_train_scaled, y_train)

print("Logistic Regression finished.")


# =========================
# 8. MLP + Logistic 预测概率
# =========================
print("\nRunning ensemble prediction...")

model.eval()

with torch.no_grad():
    mlp_val_logits, mlp_val_embedding = model(x_val_t)
    mlp_test_logits, mlp_test_embedding = model(x_test_t)

    mlp_val_prob = torch.sigmoid(mlp_val_logits).numpy()
    mlp_test_prob = torch.sigmoid(mlp_test_logits).numpy()

logreg_val_prob = logreg.predict_proba(X_val_scaled)[:, 1]
logreg_test_prob = logreg.predict_proba(X_test_scaled)[:, 1]

val_prob = (mlp_val_prob + logreg_val_prob) / 2
test_prob = (mlp_test_prob + logreg_test_prob) / 2


# =========================
# 9. 验证集自动寻找最佳阈值
# =========================
print("\nSearching best threshold on validation set...")

best_threshold = 0.5
best_f1 = -1

for threshold in tqdm(np.arange(0.40, 0.61, 0.01), desc="Searching threshold"):
    val_pred = (val_prob >= threshold).astype(int)
    f1 = f1_score(y_val, val_pred, zero_division=0)

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print(f"\nBest threshold from validation set: {best_threshold:.2f}")
print(f"Best validation F1: {best_f1:.4f}")


# =========================
# 10. 测试集评估
# =========================
print("\nEvaluating on test set...")

test_pred = (test_prob >= best_threshold).astype(int)

acc = accuracy_score(y_test, test_pred)
precision = precision_score(y_test, test_pred, zero_division=0)
recall = recall_score(y_test, test_pred, zero_division=0)
f1 = f1_score(y_test, test_pred, zero_division=0)

tn, fp, fn, tp = confusion_matrix(y_test, test_pred).ravel()

print("\n=== Test Result ===")
print(f"Test Accuracy: {acc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")

print("\nProbability summary:")
print(f"Min probability: {test_prob.min():.4f}")
print(f"Max probability: {test_prob.max():.4f}")
print(f"Mean probability: {test_prob.mean():.4f}")

print("\nConfusion Matrix:")
print(f"TP: {tp}, FP: {fp}")
print(f"FN: {fn}, TN: {tn}")


# =========================
# 11. 保存预测结果
# =========================
print("\nSaving prediction result...")

pred_df = pd.DataFrame({
    "sample_id": sample_test,
    "true_label": y_test,
    "pred_label": test_pred,
    "ensemble_probability": test_prob,
    "mlp_probability": mlp_test_prob,
    "logistic_probability": logreg_test_prob
})

pred_output_path = "/Users/chengyuhang/Desktop/ensemble_prediction_result.csv"
pred_df.to_csv(pred_output_path, index=False)

print(f"Prediction result saved to: {pred_output_path}")


# =========================
# 12. 保存样本 embedding
# =========================
print("\nSaving sample embedding...")

embedding_df = pd.DataFrame(
    mlp_test_embedding.detach().numpy(),
    index=sample_test
)

embedding_output_path = "/Users/chengyuhang/Desktop/ensemble_mlp_sample_embedding.csv"
embedding_df.to_csv(embedding_output_path)

print(f"Sample embedding saved to: {embedding_output_path}")


# =========================
# 13. 保存筛选基因
# =========================
print("\nSaving selected genes...")

gene_output_path = "/Users/chengyuhang/Desktop/ensemble_selected_genes.csv"
selected_gene_info.to_csv(gene_output_path, index=False)

print(f"Selected genes saved to: {gene_output_path}")


# =========================
# 14. 保存综合基因重要性
# =========================
print("\nSaving gene importance...")

mlp_weight = model.feature_extractor[0].weight.detach().numpy()
mlp_importance = np.mean(np.abs(mlp_weight), axis=0)

logreg_importance = np.abs(logreg.coef_[0])

importance_df = pd.DataFrame({
    "gene": selected_genes,
    "mlp_importance": mlp_importance,
    "logistic_importance": logreg_importance
})

importance_df["mean_importance"] = importance_df[
    ["mlp_importance", "logistic_importance"]
].mean(axis=1)

importance_df = importance_df.sort_values(
    by="mean_importance",
    ascending=False
)

importance_output_path = "/Users/chengyuhang/Desktop/ensemble_gene_importance.csv"
importance_df.to_csv(importance_output_path, index=False)

print(f"Gene importance saved to: {importance_output_path}")

print("\nAll done.")