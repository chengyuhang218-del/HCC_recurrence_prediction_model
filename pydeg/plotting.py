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

EPOCHS = 200
LR = 5e-4
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


# =========================
# 1. 手动定义通路基因集
# =========================
PATHWAY_GENES = {
    "Proliferation_CellCycle": [
        "MKI67", "TOP2A", "PCNA", "CCNB1", "CCNB2", "CDK1", "CDK2",
        "AURKA", "AURKB", "BUB1", "BUB1B", "CDC20", "MCM2", "MCM3",
        "MCM4", "MCM5", "MCM6", "MCM7"
    ],
    "EMT_Invasion": [
        "VIM", "SNAI1", "SNAI2", "TWIST1", "ZEB1", "ZEB2", "CDH2",
        "MMP2", "MMP9", "MMP14", "ITGA5", "ITGB1", "FN1", "COL1A1",
        "COL1A2", "TGFB1", "TGFB2"
    ],
    "Angiogenesis_Hypoxia": [
        "VEGFA", "VEGFB", "KDR", "FLT1", "HIF1A", "EPAS1", "ANGPT1",
        "ANGPT2", "PDGFA", "PDGFB", "PDGFRB", "SERPINE1", "CA9",
        "SLC2A1", "ENO1", "LDHA"
    ],
    "Inflammation_NFKB": [
        "NFKB1", "NFKB2", "RELA", "REL", "TNF", "IL1B", "IL6", "CXCL8",
        "CCL2", "CCL3", "CCL4", "CXCL10", "ICAM1", "VCAM1", "PTGS2"
    ],
    "Immune_Cytotoxic": [
        "CD8A", "CD8B", "GZMB", "GZMA", "PRF1", "NKG7", "GNLY",
        "IFNG", "CXCL9", "CXCL10", "TBX21", "EOMES"
    ],
    "Immune_Checkpoint": [
        "PDCD1", "CD274", "CTLA4", "LAG3", "TIGIT", "HAVCR2",
        "VSIR", "IDO1", "CD80", "CD86"
    ],
    "Treg_ImmuneSuppression": [
        "FOXP3", "IL2RA", "CTLA4", "IKZF2", "TIGIT", "CCR8",
        "ENTPD1", "NT5E", "TGFB1", "IL10"
    ],
    "Macrophage_Myeloid": [
        "CD68", "CD163", "MRC1", "LYZ", "LST1", "FCGR3A", "ITGAM",
        "CSF1R", "C1QA", "C1QB", "C1QC", "SPP1", "APOE"
    ],
    "Stemness": [
        "EPCAM", "PROM1", "SOX2", "NANOG", "POU5F1", "KRT19",
        "ALDH1A1", "CD44", "THY1", "LGR5"
    ],
    "WNT_BetaCatenin": [
        "CTNNB1", "MYC", "CCND1", "AXIN2", "LEF1", "TCF7", "WNT3A",
        "WNT5A", "FZD7", "DKK1", "GSK3B"
    ],
    "PI3K_AKT_MTOR": [
        "PIK3CA", "PIK3CB", "AKT1", "AKT2", "MTOR", "RPTOR",
        "RICTOR", "PTEN", "TSC1", "TSC2", "RPS6KB1", "EIF4EBP1"
    ],
    "P53_Apoptosis": [
        "TP53", "MDM2", "BAX", "BAK1", "CASP3", "CASP8", "CASP9",
        "BBC3", "PMAIP1", "CDKN1A", "FAS", "FASLG"
    ],
    "DNA_Repair": [
        "BRCA1", "BRCA2", "RAD51", "ATM", "ATR", "CHEK1", "CHEK2",
        "PARP1", "MSH2", "MSH6", "MLH1", "XRCC1"
    ],
    "Metabolism_Glycolysis": [
        "HK1", "HK2", "PFKP", "ALDOA", "GAPDH", "PGK1", "ENO1",
        "PKM", "LDHA", "SLC2A1", "SLC2A3"
    ],
    "Liver_Function": [
        "ALB", "APOA1", "APOA2", "TTR", "CYP3A4", "CYP2E1",
        "CYP2C9", "HNF4A", "FABP1", "FGA", "FGB", "FGG"
    ]
}


# =========================
# 2. 读取数据
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

sample_names = np.array(df.columns.tolist())
y = label_df["label"].values.astype(int)

print(f"Samples with labels: {df.shape[1]}")
print(f"Genes: {df.shape[0]}")
print(f"Recurrence samples: {y.sum()}")
print(f"Non-recurrence samples: {(y == 0).sum()}")


# =========================
# 3. 通路打分
# =========================
print("\nCalculating pathway scores...")

df.index = df.index.astype(str).str.upper()

pathway_score_dict = {}
pathway_gene_used = {}

for pathway, genes in PATHWAY_GENES.items():
    genes_upper = [g.upper() for g in genes]
    genes_exist = [g for g in genes_upper if g in df.index]

    pathway_gene_used[pathway] = genes_exist

    if len(genes_exist) < 3:
        print(f"Warning: {pathway} has fewer than 3 matched genes, skipped.")
        continue

    pathway_expr = df.loc[genes_exist]

    pathway_score = pathway_expr.mean(axis=0)

    pathway_score_dict[pathway] = pathway_score

pathway_df = pd.DataFrame(pathway_score_dict)

print(f"Pathways used: {pathway_df.shape[1]}")

for pathway, genes in pathway_gene_used.items():
    if pathway in pathway_df.columns:
        print(f"{pathway}: {len(genes)} genes")


X = pathway_df.values.astype(np.float32)
pathway_names = np.array(pathway_df.columns.tolist())


# =========================
# 4. 分层划分 train / val / test
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
# 5. 标准化
# =========================
print("\nStandardizing pathway scores...")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)


# =========================
# 6. MLP 模型
# =========================
class PathwayMLPClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(16, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.classifier = nn.Linear(8, 1)

    def forward(self, x):
        embedding = self.feature_extractor(x)
        logits = self.classifier(embedding).squeeze(1)
        return logits, embedding


model = PathwayMLPClassifier(input_dim=X_train_scaled.shape[1])

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
# 7. 训练 MLP
# =========================
print("\nTraining pathway MLP...")

best_val_loss = float("inf")
best_state = None
patience = 25
wait = 0

for epoch in tqdm(range(EPOCHS), desc="Training pathway MLP"):
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

print("Pathway MLP training finished.")


# =========================
# 8. Logistic Regression
# =========================
print("\nTraining Logistic Regression...")

logreg = LogisticRegression(
    penalty="l2",
    C=1.0,
    class_weight="balanced",
    solver="liblinear",
    max_iter=300,
    random_state=RANDOM_SEED
)

logreg.fit(X_train_scaled, y_train)

print("Logistic Regression finished.")


# =========================
# 9. MLP + Logistic 集成预测
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
# 10. 验证集寻找最佳阈值
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
# 11. 测试集评估
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
# 12. 保存预测结果
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

pred_output_path = "/Users/chengyuhang/Desktop/pathway_prediction_result.csv"
pred_df.to_csv(pred_output_path, index=False)

print(f"Prediction result saved to: {pred_output_path}")


# =========================
# 13. 保存通路分数
# =========================
print("\nSaving pathway score matrix...")

pathway_score_output = "/Users/chengyuhang/Desktop/pathway_score_matrix.csv"
pathway_df.to_csv(pathway_score_output)

print(f"Pathway score matrix saved to: {pathway_score_output}")


# =========================
# 14. 保存样本 embedding
# =========================
print("\nSaving sample embedding...")

embedding_df = pd.DataFrame(
    mlp_test_embedding.detach().numpy(),
    index=sample_test
)

embedding_output_path = "/Users/chengyuhang/Desktop/pathway_mlp_sample_embedding.csv"
embedding_df.to_csv(embedding_output_path)

print(f"Sample embedding saved to: {embedding_output_path}")


# =========================
# 15. 保存通路重要性
# =========================
print("\nSaving pathway importance...")

mlp_weight = model.feature_extractor[0].weight.detach().numpy()
mlp_importance = np.mean(np.abs(mlp_weight), axis=0)

logreg_importance = np.abs(logreg.coef_[0])

importance_df = pd.DataFrame({
    "pathway": pathway_names,
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

importance_output_path = "/Users/chengyuhang/Desktop/pathway_importance.csv"
importance_df.to_csv(importance_output_path, index=False)

print(f"Pathway importance saved to: {importance_output_path}")

print("\nAll done.")