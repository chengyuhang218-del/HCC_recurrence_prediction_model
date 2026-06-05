import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)

# =========================
# 0. 参数
# =========================
EXPR_PATH = "/Users/chengyuhang/Desktop/HCC复发预测项目/GSE76427_expr_gene_RFS_108_clean.csv"
LABEL_PATH = "/Users/chengyuhang/Desktop/HCC复发预测项目/GSE76427_rfs_label_108.csv"

TOPK_GENES = 90
EPOCHS = 260
LR = 5e-4
RANDOM_SEED = 42

FIG_DIR = "/Users/chengyuhang/Desktop/model_figures"
os.makedirs(FIG_DIR, exist_ok=True)

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
# 2. 划分数据集
# =========================
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

# =========================
# 3. 差异基因筛选
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

# =========================
# 6. Tensor
# =========================
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

# =========================
# 7. 训练 MLP
# =========================
best_val_loss = float("inf")
best_state = None
patience = 20
wait = 0

train_loss_history = []
val_loss_history = []

train_acc_history = []
val_acc_history = []

train_f1_history = []
val_f1_history = []

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

        train_prob_epoch = torch.sigmoid(train_logits).detach().numpy()
        val_prob_epoch = torch.sigmoid(val_logits).detach().numpy()

        train_pred_epoch = (train_prob_epoch >= 0.5).astype(int)
        val_pred_epoch = (val_prob_epoch >= 0.5).astype(int)

        train_acc = accuracy_score(y_train, train_pred_epoch)
        val_acc = accuracy_score(y_val, val_pred_epoch)

        train_f1_epoch = f1_score(y_train, train_pred_epoch, zero_division=0)
        val_f1_epoch = f1_score(y_val, val_pred_epoch, zero_division=0)

    train_loss_history.append(loss.item())
    val_loss_history.append(val_loss)

    train_acc_history.append(train_acc)
    val_acc_history.append(val_acc)

    train_f1_history.append(train_f1_epoch)
    val_f1_history.append(val_f1_epoch)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = model.state_dict()
        wait = 0
    else:
        wait += 1

    if (epoch + 1) % 20 == 0:
        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"Train Loss: {loss.item():.4f} "
            f"Val Loss: {val_loss:.4f} "
            f"Train Acc: {train_acc:.4f} "
            f"Val Acc: {val_acc:.4f}"
        )

    if wait >= patience:
        print(f"Early stopping at epoch {epoch + 1}")
        break

model.load_state_dict(best_state)

print("MLP training finished.")

# =========================
# 8. 训练过程曲线
# =========================
plt.figure(figsize=(8, 5))
plt.plot(train_loss_history, label="Train Loss")
plt.plot(val_loss_history, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("MLP Training and Validation Loss")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "loss_curve.png"), dpi=300)
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(train_acc_history, label="Train Accuracy")
plt.plot(val_acc_history, label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("MLP Training and Validation Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "accuracy_curve.png"), dpi=300)
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(train_f1_history, label="Train F1-score")
plt.plot(val_f1_history, label="Validation F1-score")
plt.xlabel("Epoch")
plt.ylabel("F1-score")
plt.title("MLP Training and Validation F1-score")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "f1_curve.png"), dpi=300)
plt.show()

# =========================
# 9. 训练 Logistic Regression
# =========================
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
# 10. 三种模型预测概率
# =========================
model.eval()

with torch.no_grad():
    mlp_val_logits, mlp_val_embedding = model(x_val_t)
    mlp_test_logits, mlp_test_embedding = model(x_test_t)

    mlp_val_prob = torch.sigmoid(mlp_val_logits).numpy()
    mlp_test_prob = torch.sigmoid(mlp_test_logits).numpy()

logreg_val_prob = logreg.predict_proba(X_val_scaled)[:, 1]
logreg_test_prob = logreg.predict_proba(X_test_scaled)[:, 1]

ensemble_val_prob = (mlp_val_prob + logreg_val_prob) / 2
ensemble_test_prob = (mlp_test_prob + logreg_test_prob) / 2

# =========================
# 11. 在验证集上分别找最佳阈值
# =========================
def find_best_threshold(y_true, y_prob):
    best_threshold = 0.5
    best_f1 = -1

    threshold_list = np.arange(0.01, 1.00, 0.01)

    f1_list = []
    precision_list = []
    recall_list = []

    for threshold in threshold_list:
        pred = (y_prob >= threshold).astype(int)

        current_f1 = f1_score(y_true, pred, zero_division=0)
        current_precision = precision_score(y_true, pred, zero_division=0)
        current_recall = recall_score(y_true, pred, zero_division=0)

        f1_list.append(current_f1)
        precision_list.append(current_precision)
        recall_list.append(current_recall)

        if current_f1 > best_f1:
            best_f1 = current_f1
            best_threshold = threshold

    return best_threshold, best_f1, threshold_list, f1_list, precision_list, recall_list


mlp_best_threshold, mlp_best_val_f1, _, _, _, _ = find_best_threshold(
    y_val,
    mlp_val_prob
)

logreg_best_threshold, logreg_best_val_f1, _, _, _, _ = find_best_threshold(
    y_val,
    logreg_val_prob
)

ensemble_best_threshold, ensemble_best_val_f1, threshold_list, threshold_f1_list, threshold_precision_list, threshold_recall_list = find_best_threshold(
    y_val,
    ensemble_val_prob
)

print("MLP best threshold:", round(mlp_best_threshold, 2))
print("Logistic best threshold:", round(logreg_best_threshold, 2))
print("Ensemble best threshold:", round(ensemble_best_threshold, 2))

# =========================
# 12. 阈值曲线
# =========================
plt.figure(figsize=(8, 5))
plt.plot(threshold_list, threshold_f1_list, label="Ensemble Validation F1")
plt.axvline(
    ensemble_best_threshold,
    linestyle="--",
    label=f"Best threshold = {ensemble_best_threshold:.2f}"
)
plt.xlabel("Threshold")
plt.ylabel("F1-score")
plt.title("Ensemble Threshold vs F1-score")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "threshold_f1_curve.png"), dpi=300)
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(threshold_list, threshold_precision_list, label="Precision")
plt.plot(threshold_list, threshold_recall_list, label="Recall")
plt.axvline(
    ensemble_best_threshold,
    linestyle="--",
    label=f"Best threshold = {ensemble_best_threshold:.2f}"
)
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title("Ensemble Threshold vs Precision and Recall")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "threshold_precision_recall_curve.png"), dpi=300)
plt.show()

# =========================
# 13. 模型评估函数
# =========================
def evaluate_model_by_prob(y_true, y_prob, threshold, model_name):
    y_pred = (y_prob >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    precision_value = precision_score(y_true, y_pred, zero_division=0)
    recall_value = recall_score(y_true, y_pred, zero_division=0)
    f1_value = f1_score(y_true, y_pred, zero_division=0)

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc_value = auc(fpr, tpr)

    ap_value = average_precision_score(y_true, y_prob)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        "model": model_name,
        "threshold": threshold,
        "accuracy": acc,
        "precision": precision_value,
        "recall": recall_value,
        "f1": f1_value,
        "auc": roc_auc_value,
        "average_precision": ap_value,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn
    }


mlp_result = evaluate_model_by_prob(
    y_test,
    mlp_test_prob,
    mlp_best_threshold,
    "MLP"
)

logreg_result = evaluate_model_by_prob(
    y_test,
    logreg_test_prob,
    logreg_best_threshold,
    "Logistic Regression"
)

ensemble_result = evaluate_model_by_prob(
    y_test,
    ensemble_test_prob,
    ensemble_best_threshold,
    "MLP + Logistic"
)

model_compare_df = pd.DataFrame([
    mlp_result,
    logreg_result,
    ensemble_result
])

model_compare_path = "/Users/chengyuhang/Desktop/model_performance_comparison.csv"
model_compare_df.to_csv(model_compare_path, index=False)

print("\n=== Model Performance Comparison ===")
print(model_compare_df)

# =========================
# 14. 三种模型性能柱状图
# =========================
metrics_to_plot = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "auc",
    "average_precision"
]

plot_df = model_compare_df.set_index("model")[metrics_to_plot]

plt.figure(figsize=(11, 6))

x = np.arange(len(metrics_to_plot))
width = 0.25

plt.bar(x - width, plot_df.loc["MLP"], width, label="MLP")
plt.bar(x, plot_df.loc["Logistic Regression"], width, label="Logistic Regression")
plt.bar(x + width, plot_df.loc["MLP + Logistic"], width, label="MLP + Logistic")

plt.xticks(x, metrics_to_plot, rotation=30)
plt.ylabel("Score")
plt.ylim(0, 1.05)
plt.title("Model Performance Comparison")
plt.legend()
plt.tight_layout()

plt.savefig(os.path.join(FIG_DIR, "model_performance_comparison.png"), dpi=300)
plt.show()

# =========================
# 15. 三种模型 ROC 曲线对比
# =========================
plt.figure(figsize=(6, 5))

for name, prob in {
    "MLP": mlp_test_prob,
    "Logistic Regression": logreg_test_prob,
    "MLP + Logistic": ensemble_test_prob
}.items():
    fpr, tpr, _ = roc_curve(y_test, prob)
    model_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{name} AUC = {model_auc:.3f}")

plt.plot([0, 1], [0, 1], linestyle="--", label="Random")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.tight_layout()

plt.savefig(os.path.join(FIG_DIR, "roc_curve_model_comparison.png"), dpi=300)
plt.show()

# =========================
# 16. 三种模型 PR 曲线对比
# =========================
plt.figure(figsize=(6, 5))

for name, prob in {
    "MLP": mlp_test_prob,
    "Logistic Regression": logreg_test_prob,
    "MLP + Logistic": ensemble_test_prob
}.items():
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, prob)
    ap = average_precision_score(y_test, prob)
    plt.plot(recall_curve, precision_curve, label=f"{name} AP = {ap:.3f}")

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve Comparison")
plt.legend()
plt.tight_layout()

plt.savefig(os.path.join(FIG_DIR, "pr_curve_model_comparison.png"), dpi=300)
plt.show()

# =========================
# 17. 三种模型混淆矩阵
# =========================
def plot_confusion_matrix_from_result(result, filename):
    cm = np.array([
        [result["TN"], result["FP"]],
        [result["FN"], result["TP"]]
    ])

    plt.figure(figsize=(5, 4))
    plt.imshow(cm)
    plt.xticks([0, 1], ["Pred 0", "Pred 1"])
    plt.yticks([0, 1], ["True 0", "True 1"])
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(result["model"] + " Confusion Matrix")

    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, filename), dpi=300)
    plt.show()


plot_confusion_matrix_from_result(mlp_result, "confusion_matrix_mlp.png")
plot_confusion_matrix_from_result(logreg_result, "confusion_matrix_logistic.png")
plot_confusion_matrix_from_result(ensemble_result, "confusion_matrix_ensemble.png")

# =========================
# 18. 三种模型预测概率分布
# =========================
def plot_probability_distribution(y_true, y_prob, threshold, title, filename):
    plt.figure(figsize=(8, 5))
    plt.hist(y_prob[y_true == 0], bins=10, alpha=0.7, label="Non-recurrence")
    plt.hist(y_prob[y_true == 1], bins=10, alpha=0.7, label="Recurrence")
    plt.axvline(threshold, linestyle="--", label=f"Threshold = {threshold:.2f}")
    plt.xlabel("Predicted recurrence probability")
    plt.ylabel("Sample count")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, filename), dpi=300)
    plt.show()


plot_probability_distribution(
    y_test,
    mlp_test_prob,
    mlp_best_threshold,
    "MLP Probability Distribution",
    "probability_distribution_mlp.png"
)

plot_probability_distribution(
    y_test,
    logreg_test_prob,
    logreg_best_threshold,
    "Logistic Regression Probability Distribution",
    "probability_distribution_logistic.png"
)

plot_probability_distribution(
    y_test,
    ensemble_test_prob,
    ensemble_best_threshold,
    "Ensemble Probability Distribution",
    "probability_distribution_ensemble.png"
)

# =========================
# 19. MLP vs Logistic 概率相关性
# =========================
plt.figure(figsize=(6, 6))
plt.scatter(mlp_test_prob, logreg_test_prob, alpha=0.8)
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("MLP probability")
plt.ylabel("Logistic probability")
plt.title("MLP vs Logistic Regression Probability")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "mlp_vs_logistic_probability.png"), dpi=300)
plt.show()

# =========================
# 20. Ensemble 概率排序图
# =========================
prob_order = np.argsort(ensemble_test_prob)

sorted_prob = ensemble_test_prob[prob_order]
sorted_y = y_test[prob_order]

plt.figure(figsize=(9, 5))
plt.scatter(range(len(sorted_prob)), sorted_prob, c=sorted_y)
plt.axhline(
    ensemble_best_threshold,
    linestyle="--",
    label=f"Threshold = {ensemble_best_threshold:.2f}"
)
plt.xlabel("Test samples sorted by probability")
plt.ylabel("Predicted recurrence probability")
plt.title("Ensemble Sorted Prediction Probability")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "sorted_prediction_probability_ensemble.png"), dpi=300)
plt.show()

# =========================
# 21. PCA Embedding
# =========================
embedding_array = mlp_test_embedding.detach().numpy()

pca = PCA(
    n_components=2,
    random_state=RANDOM_SEED
)

embedding_pca = pca.fit_transform(embedding_array)

plt.figure(figsize=(6, 5))
plt.scatter(
    embedding_pca[:, 0],
    embedding_pca[:, 1],
    c=y_test,
    alpha=0.8
)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA of MLP Sample Embedding")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "mlp_embedding_pca.png"), dpi=300)
plt.show()

# =========================
# 22. 保存预测结果
# =========================
pred_df = pd.DataFrame({
    "sample_id": sample_test,
    "true_label": y_test,

    "mlp_probability": mlp_test_prob,
    "mlp_pred_label": (mlp_test_prob >= mlp_best_threshold).astype(int),

    "logistic_probability": logreg_test_prob,
    "logistic_pred_label": (logreg_test_prob >= logreg_best_threshold).astype(int),

    "ensemble_probability": ensemble_test_prob,
    "ensemble_pred_label": (ensemble_test_prob >= ensemble_best_threshold).astype(int)
})

pred_output_path = "/Users/chengyuhang/Desktop/model_prediction_result_three_models.csv"
pred_df.to_csv(pred_output_path, index=False)

print(f"Prediction result saved to: {pred_output_path}")
print(pred_df.head())

# =========================
# 23. 保存 MLP embedding
# =========================
embedding_df = pd.DataFrame(
    mlp_test_embedding.detach().numpy(),
    index=sample_test
)

embedding_output_path = "/Users/chengyuhang/Desktop/ensemble_mlp_sample_embedding.csv"
embedding_df.to_csv(embedding_output_path)

print(f"Embedding saved to: {embedding_output_path}")

# =========================
# 24. 保存筛选基因
# =========================
gene_output_path = "/Users/chengyuhang/Desktop/ensemble_selected_genes.csv"

selected_gene_info.to_csv(
    gene_output_path,
    index=False
)

print(f"Selected genes saved to: {gene_output_path}")

# =========================
# 25. 基因重要性
# =========================
mlp_weight = model.feature_extractor[0].weight.detach().numpy()

mlp_importance = np.mean(
    np.abs(mlp_weight),
    axis=0
)

logreg_importance = np.abs(
    logreg.coef_[0]
)

importance_df = pd.DataFrame({
    "gene": selected_genes,
    "mlp_importance": mlp_importance,
    "logistic_importance": logreg_importance
})

importance_df["mlp_importance_norm"] = (
    importance_df["mlp_importance"]
    - importance_df["mlp_importance"].min()
) / (
    importance_df["mlp_importance"].max()
    - importance_df["mlp_importance"].min()
)

importance_df["logistic_importance_norm"] = (
    importance_df["logistic_importance"]
    - importance_df["logistic_importance"].min()
) / (
    importance_df["logistic_importance"].max()
    - importance_df["logistic_importance"].min()
)

importance_df["mean_importance"] = importance_df[
    [
        "mlp_importance_norm",
        "logistic_importance_norm"
    ]
].mean(axis=1)

importance_df = importance_df.sort_values(
    by="mean_importance",
    ascending=False
)

importance_output_path = "/Users/chengyuhang/Desktop/ensemble_gene_importance.csv"

importance_df.to_csv(
    importance_output_path,
    index=False
)

print(f"Gene importance saved to: {importance_output_path}")
print(importance_df.head(20))

# =========================
# 26. Top20 基因重要性图
# =========================
top_gene_df = importance_df.head(20).sort_values(
    "mean_importance"
)

plt.figure(figsize=(8, 6))
plt.barh(
    top_gene_df["gene"],
    top_gene_df["mean_importance"]
)
plt.xlabel("Mean normalized importance")
plt.ylabel("Gene")
plt.title("Top 20 Important Genes")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "top20_gene_importance.png"), dpi=300)
plt.show()

# =========================
# 27. 输出所有图
# =========================
print("\nAll figures saved to:")
print(FIG_DIR)

print("\nGenerated figures:")

for file in os.listdir(FIG_DIR):
    if file.endswith(".png"):
        print(file)

print("\nAll done.")