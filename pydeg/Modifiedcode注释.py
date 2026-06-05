import os  # 导入系统环境模块
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 允许重复加载 OpenMP，避免部分环境报错
import torch  # 导入 PyTorch
import torch.nn as nn  # 导入神经网络模块
import numpy as np  # 导入数值计算库
import pandas as pd  # 导入表格处理库
from tqdm import tqdm  # 导入进度条工具
from sklearn.model_selection import train_test_split  # 导入数据集划分函数
from sklearn.preprocessing import StandardScaler  # 导入标准化工具
from sklearn.linear_model import LogisticRegression  # 导入逻辑回归模型
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix  # 导入评价指标


# =========================  # 分隔符
# 0. 参数  # 参数设置部分
# =========================  # 分隔符
EXPR_PATH = "/Users/chengyuhang/Desktop/GSE76427_expr_gene_RFS_108_clean.csv"  # 表达矩阵文件路径
LABEL_PATH = "/Users/chengyuhang/Desktop/GSE76427_rfs_label_108.csv"  # 标签文件路径

TOPK_GENES = 90  # 筛选前 90 个差异基因
EPOCHS = 260  # 最大训练轮数
LR = 5e-4  # 学习率
RANDOM_SEED = 42  # 随机种子

np.random.seed(RANDOM_SEED)  # 固定 numpy 随机性
torch.manual_seed(RANDOM_SEED)  # 固定 PyTorch 随机性


# =========================  # 分隔符
# 1. 读取数据  # 数据读取部分
# =========================  # 分隔符
print("Loading data...")  # 打印提示信息

df = pd.read_csv(EXPR_PATH, index_col=0)  # 读取表达矩阵，第一列作为基因名索引

label_df = pd.read_csv(LABEL_PATH)  # 读取样本标签文件
label_df = label_df.rename(columns={  # 重命名标签表列名
    "sample": "sample_id",  # 将 sample 改为 sample_id
    "recurrence": "label"  # 将 recurrence 改为 label
})  # 结束重命名
label_df["label"] = label_df["label"].astype(int)  # 将标签转成整数类型

common_samples = [s for s in df.columns if s in set(label_df["sample_id"])]  # 找到表达矩阵和标签共有样本

df = df[common_samples]  # 只保留有标签的样本
label_df = label_df.set_index("sample_id").loc[common_samples].reset_index()  # 按表达矩阵样本顺序重排标签

gene_names_all = np.array(df.index.tolist())  # 保存所有基因名
sample_names = np.array(df.columns.tolist())  # 保存所有样本名

X = df.values.T.astype(np.float32)  # 转置为 样本 × 基因 的特征矩阵
y = label_df["label"].values.astype(int)  # 提取标签数组

print(f"Samples with labels: {X.shape[0]}")  # 打印有标签样本数
print(f"Genes: {X.shape[1]}")  # 打印基因数量
print(f"Recurrence samples: {y.sum()}")  # 打印复发样本数
print(f"Non-recurrence samples: {(y == 0).sum()}")  # 打印非复发样本数


# =========================  # 分隔符
# 2. 分层划分 train / val / test  # 数据集划分部分
# =========================  # 分隔符
print("\nSplitting train / val / test...")  # 打印提示信息

X_trainval, X_test, y_trainval, y_test, sample_trainval, sample_test = train_test_split(  # 先划分训练验证集和测试集
    X,  # 输入全部表达数据
    y,  # 输入全部标签
    sample_names,  # 同步划分样本名
    test_size=0.2,  # 测试集占 20%
    random_state=RANDOM_SEED,  # 固定划分随机性
    stratify=y  # 按标签比例分层划分
)  # 结束第一次划分

X_train, X_val, y_train, y_val, sample_train, sample_val = train_test_split(  # 再划分训练集和验证集
    X_trainval,  # 输入训练验证特征
    y_trainval,  # 输入训练验证标签
    sample_trainval,  # 输入训练验证样本名
    test_size=0.25,  # 验证集占 trainval 的 25%
    random_state=RANDOM_SEED,  # 固定划分随机性
    stratify=y_trainval  # 按标签比例分层划分
)  # 结束第二次划分

print(f"Train samples: {X_train.shape[0]}")  # 打印训练样本数
print(f"Val samples: {X_val.shape[0]}")  # 打印验证样本数
print(f"Test samples: {X_test.shape[0]}")  # 打印测试样本数
print(f"Train recurrence: {y_train.sum()}")  # 打印训练集复发数
print(f"Train non-recurrence: {(y_train == 0).sum()}")  # 打印训练集非复发数
print(f"Val recurrence: {y_val.sum()}")  # 打印验证集复发数
print(f"Val non-recurrence: {(y_val == 0).sum()}")  # 打印验证集非复发数
print(f"Test recurrence: {y_test.sum()}")  # 打印测试集复发数
print(f"Test non-recurrence: {(y_test == 0).sum()}")  # 打印测试集非复发数


# =========================  # 分隔符
# 3. 只用训练集筛选差异基因  # 防止数据泄露
# =========================  # 分隔符
def select_genes_by_tscore(X_train, y_train, gene_names, topk=100):  # 定义按 t-score 筛基因函数
    pos = X_train[y_train == 1]  # 取复发组样本
    neg = X_train[y_train == 0]  # 取非复发组样本

    pos_mean = pos.mean(axis=0)  # 计算复发组每个基因均值
    neg_mean = neg.mean(axis=0)  # 计算非复发组每个基因均值

    pos_var = pos.var(axis=0)  # 计算复发组每个基因方差
    neg_var = neg.var(axis=0)  # 计算非复发组每个基因方差

    n_pos = pos.shape[0]  # 复发组样本数
    n_neg = neg.shape[0]  # 非复发组样本数

    se = np.sqrt(pos_var / n_pos + neg_var / n_neg)  # 计算标准误
    se[se == 0] = 1e-6  # 避免除以 0

    t_score = (pos_mean - neg_mean) / se  # 计算每个基因 t-score
    abs_t = np.abs(t_score)  # 取 t-score 绝对值

    top_idx = np.argsort(abs_t)[::-1][:topk]  # 按绝对值从大到小取前 topk 个基因索引

    selected_info = pd.DataFrame({  # 构建筛选基因信息表
        "gene": gene_names[top_idx],  # 筛选出的基因名
        "t_score": t_score[top_idx],  # 对应 t-score
        "abs_t_score": abs_t[top_idx],  # 对应绝对 t-score
        "recurrence_mean": pos_mean[top_idx],  # 复发组平均表达
        "non_recurrence_mean": neg_mean[top_idx]  # 非复发组平均表达
    })  # 结束表格构建

    return top_idx, selected_info  # 返回基因索引和信息表


print("\nSelecting recurrence-related genes...")  # 打印提示信息

top_idx, selected_gene_info = select_genes_by_tscore(  # 调用函数筛选差异基因
    X_train,  # 只输入训练集特征
    y_train,  # 只输入训练集标签
    gene_names_all,  # 输入全部基因名
    TOPK_GENES  # 输入筛选数量
)  # 结束函数调用

selected_genes = gene_names_all[top_idx]  # 获取筛选出的基因名

X_train = X_train[:, top_idx]  # 训练集只保留筛选基因
X_val = X_val[:, top_idx]  # 验证集只保留筛选基因
X_test = X_test[:, top_idx]  # 测试集只保留筛选基因

print(f"Selected genes: {len(selected_genes)}")  # 打印筛选基因数量


# =========================  # 分隔符
# 4. 标准化  # 特征标准化部分
# =========================  # 分隔符
print("\nStandardizing data...")  # 打印提示信息

scaler = StandardScaler()  # 创建标准化器

X_train_scaled = scaler.fit_transform(X_train)  # 用训练集拟合并标准化训练集
X_val_scaled = scaler.transform(X_val)  # 用训练集参数标准化验证集
X_test_scaled = scaler.transform(X_test)  # 用训练集参数标准化测试集


# =========================  # 分隔符
# 5. MLP 模型  # 神经网络模型部分
# =========================  # 分隔符
class MLPClassifier(nn.Module):  # 定义 MLP 分类器
    def __init__(self, input_dim):  # 初始化函数
        super().__init__()  # 调用父类初始化

        self.feature_extractor = nn.Sequential(  # 定义特征提取网络
            nn.Linear(input_dim, 32),  # 第一层全连接：输入基因数到 32 维
            nn.BatchNorm1d(32),  # 对 32 维特征做批标准化
            nn.ReLU(),  # 使用 ReLU 激活函数
            nn.Dropout(0.4),  # 随机丢弃 40% 神经元防止过拟合

            nn.Linear(32, 16),  # 第二层全连接：32 维到 16 维
            nn.BatchNorm1d(16),  # 对 16 维特征做批标准化
            nn.ReLU(),  # 使用 ReLU 激活函数
            nn.Dropout(0.3)  # 随机丢弃 30% 神经元防止过拟合
        )  # 结束特征提取网络

        self.classifier = nn.Linear(16, 1)  # 输出层：16 维到 1 个 logit

    def forward(self, x):  # 定义前向传播
        embedding = self.feature_extractor(x)  # 提取样本低维特征
        logits = self.classifier(embedding).squeeze(1)  # 输出二分类 logit 并压缩维度
        return logits, embedding  # 返回预测值和 embedding


model = MLPClassifier(input_dim=TOPK_GENES)  # 创建 MLP 模型

x_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)  # 转换训练特征为 tensor
y_train_t = torch.tensor(y_train, dtype=torch.float32)  # 转换训练标签为 tensor

x_val_t = torch.tensor(X_val_scaled, dtype=torch.float32)  # 转换验证特征为 tensor
y_val_t = torch.tensor(y_val, dtype=torch.float32)  # 转换验证标签为 tensor

x_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)  # 转换测试特征为 tensor

pos_weight = torch.tensor(  # 创建正类权重
    [(y_train == 0).sum() / (y_train == 1).sum()],  # 用负类数除以正类数平衡类别
    dtype=torch.float32  # 设置数据类型
)  # 结束权重创建

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)  # 定义带类别权重的二分类损失

optimizer = torch.optim.Adam(  # 创建 Adam 优化器
    model.parameters(),  # 输入模型参数
    lr=LR,  # 设置学习率
    weight_decay=1e-3  # 设置 L2 正则化
)  # 结束优化器创建

print(f"\nPositive class weight: {pos_weight.item():.4f}")  # 打印正类权重


# =========================  # 分隔符
# 6. MLP 训练 + Early stopping  # 训练神经网络
# =========================  # 分隔符
print("\nTraining MLP...")  # 打印提示信息

best_val_loss = float("inf")  # 初始化最佳验证损失为无穷大
best_state = None  # 初始化最佳模型参数
patience = 20  # 允许验证损失不下降的轮数
wait = 0  # 当前等待轮数

for epoch in tqdm(range(EPOCHS), desc="Training MLP"):  # 循环训练多个 epoch
    model.train()  # 设置模型为训练模式
    optimizer.zero_grad()  # 清空上一轮梯度

    train_logits, train_embedding = model(x_train_t)  # 前向计算训练集输出
    loss = criterion(train_logits, y_train_t)  # 计算训练损失

    loss.backward()  # 反向传播计算梯度
    optimizer.step()  # 更新模型参数

    model.eval()  # 设置模型为评估模式

    with torch.no_grad():  # 关闭梯度计算
        val_logits, val_embedding = model(x_val_t)  # 前向计算验证集输出
        val_loss = criterion(val_logits, y_val_t).item()  # 计算验证损失并转成数值

    if val_loss < best_val_loss:  # 如果验证损失更小
        best_val_loss = val_loss  # 更新最佳验证损失
        best_state = model.state_dict()  # 保存当前最佳模型参数
        wait = 0  # 重置等待轮数
    else:  # 如果验证损失没有下降
        wait += 1  # 等待轮数加一

    if (epoch + 1) % 20 == 0:  # 每 20 轮打印一次
        print(  # 打印训练信息
            f"\nEpoch [{epoch + 1}/{EPOCHS}] "  # 当前训练轮数
            f"Train Loss: {loss.item():.4f} "  # 当前训练损失
            f"Val Loss: {val_loss:.4f}"  # 当前验证损失
        )  # 结束打印

    if wait >= patience:  # 如果等待轮数超过 patience
        print(f"\nEarly stopping at epoch {epoch + 1}")  # 打印早停轮数
        break  # 提前结束训练

model.load_state_dict(best_state)  # 加载验证集表现最好的模型参数

print("MLP training finished.")  # 打印训练完成


# =========================  # 分隔符
# 7. Logistic Regression  # 逻辑回归模型部分
# =========================  # 分隔符
print("\nTraining Logistic Regression...")  # 打印提示信息

logreg = LogisticRegression(  # 创建逻辑回归模型
    penalty="elasticnet",  # 使用 Elastic Net 正则化
    l1_ratio=0.5,  # L1 和 L2 各占一部分
    C=0.5,  # 正则化强度参数，越小正则越强
    class_weight="balanced",  # 自动平衡类别权重
    solver="saga",  # 支持 elasticnet 的求解器
    max_iter=3000,  # 最大迭代次数
    random_state=RANDOM_SEED  # 固定随机性
)  # 结束模型创建
logreg.fit(X_train_scaled, y_train)  # 用训练集训练逻辑回归

print("Logistic Regression finished.")  # 打印训练完成


# =========================  # 分隔符
# 8. MLP + Logistic 预测概率  # 集成预测部分
# =========================  # 分隔符
print("\nRunning ensemble prediction...")  # 打印提示信息

model.eval()  # 设置 MLP 为评估模式

with torch.no_grad():  # 关闭梯度计算
    mlp_val_logits, mlp_val_embedding = model(x_val_t)  # 计算验证集 MLP 输出
    mlp_test_logits, mlp_test_embedding = model(x_test_t)  # 计算测试集 MLP 输出

    mlp_val_prob = torch.sigmoid(mlp_val_logits).numpy()  # 将验证集 logit 转为概率
    mlp_test_prob = torch.sigmoid(mlp_test_logits).numpy()  # 将测试集 logit 转为概率

logreg_val_prob = logreg.predict_proba(X_val_scaled)[:, 1]  # 计算逻辑回归验证集复发概率
logreg_test_prob = logreg.predict_proba(X_test_scaled)[:, 1]  # 计算逻辑回归测试集复发概率

val_prob = (mlp_val_prob + logreg_val_prob) / 2  # 平均两个模型的验证集概率
test_prob = (mlp_test_prob + logreg_test_prob) / 2  # 平均两个模型的测试集概率


# =========================  # 分隔符
# 9. 验证集自动寻找最佳阈值  # 阈值选择部分
# =========================  # 分隔符
print("\nSearching best threshold on validation set...")  # 打印提示信息

best_threshold = 0.5  # 初始化默认阈值
best_f1 = -1  # 初始化最佳 F1

for threshold in tqdm(np.arange(0.40, 0.61, 0.01), desc="Searching threshold"):  # 遍历候选阈值
    val_pred = (val_prob >= threshold).astype(int)  # 根据阈值得到验证集预测标签
    f1 = f1_score(y_val, val_pred, zero_division=0)  # 计算验证集 F1

    if f1 > best_f1:  # 如果当前 F1 更高
        best_f1 = f1  # 更新最佳 F1
        best_threshold = threshold  # 更新最佳阈值

print(f"\nBest threshold from validation set: {best_threshold:.2f}")  # 打印最佳阈值
print(f"Best validation F1: {best_f1:.4f}")  # 打印最佳验证 F1


# =========================  # 分隔符
# 10. 测试集评估  # 模型最终评估
# =========================  # 分隔符
print("\nEvaluating on test set...")  # 打印提示信息

test_pred = (test_prob >= best_threshold).astype(int)  # 用最佳阈值得到测试集预测标签

acc = accuracy_score(y_test, test_pred)  # 计算准确率
precision = precision_score(y_test, test_pred, zero_division=0)  # 计算精确率
recall = recall_score(y_test, test_pred, zero_division=0)  # 计算召回率
f1 = f1_score(y_test, test_pred, zero_division=0)  # 计算 F1 分数

tn, fp, fn, tp = confusion_matrix(y_test, test_pred).ravel()  # 拆分混淆矩阵四个值

print("\n=== Test Result ===")  # 打印结果标题
print(f"Test Accuracy: {acc:.4f}")  # 打印测试准确率
print(f"Precision: {precision:.4f}")  # 打印精确率
print(f"Recall: {recall:.4f}")  # 打印召回率
print(f"F1-score: {f1:.4f}")  # 打印 F1 分数

print("\nProbability summary:")  # 打印概率统计标题
print(f"Min probability: {test_prob.min():.4f}")  # 打印最小预测概率
print(f"Max probability: {test_prob.max():.4f}")  # 打印最大预测概率
print(f"Mean probability: {test_prob.mean():.4f}")  # 打印平均预测概率

print("\nConfusion Matrix:")  # 打印混淆矩阵标题
print(f"TP: {tp}, FP: {fp}")  # 打印真正例和假正例
print(f"FN: {fn}, TN: {tn}")  # 打印假负例和真负例


# =========================  # 分隔符
# 11. 保存预测结果  # 输出预测结果文件
# =========================  # 分隔符
print("\nSaving prediction result...")  # 打印提示信息

pred_df = pd.DataFrame({  # 构建预测结果表
    "sample_id": sample_test,  # 测试集样本 ID
    "true_label": y_test,  # 真实标签
    "pred_label": test_pred,  # 预测标签
    "ensemble_probability": test_prob,  # 集成模型预测概率
    "mlp_probability": mlp_test_prob,  # MLP 预测概率
    "logistic_probability": logreg_test_prob  # 逻辑回归预测概率
})  # 结束表格构建

pred_output_path = "/Users/chengyuhang/Desktop/ensemble_prediction_result.csv"  # 预测结果输出路径
pred_df.to_csv(pred_output_path, index=False)  # 保存预测结果表

print(f"Prediction result saved to: {pred_output_path}")  # 打印保存路径


# =========================  # 分隔符
# 12. 保存样本 embedding  # 输出 MLP 低维特征
# =========================  # 分隔符
print("\nSaving sample embedding...")  # 打印提示信息

embedding_df = pd.DataFrame(  # 构建 embedding 表
    mlp_test_embedding.detach().numpy(),  # 将测试集 embedding 转为 numpy
    index=sample_test  # 使用测试样本名作为行名
)  # 结束表格构建

embedding_output_path = "/Users/chengyuhang/Desktop/ensemble_mlp_sample_embedding.csv"  # embedding 输出路径
embedding_df.to_csv(embedding_output_path)  # 保存 embedding 表

print(f"Sample embedding saved to: {embedding_output_path}")  # 打印保存路径


# =========================  # 分隔符
# 13. 保存筛选基因  # 输出筛选基因信息
# =========================  # 分隔符
print("\nSaving selected genes...")  # 打印提示信息

gene_output_path = "/Users/chengyuhang/Desktop/ensemble_selected_genes.csv"  # 筛选基因输出路径
selected_gene_info.to_csv(gene_output_path, index=False)  # 保存筛选基因信息

print(f"Selected genes saved to: {gene_output_path}")  # 打印保存路径


# =========================  # 分隔符
# 14. 保存综合基因重要性  # 输出基因重要性
# =========================  # 分隔符
print("\nSaving gene importance...")  # 打印提示信息

mlp_weight = model.feature_extractor[0].weight.detach().numpy()  # 提取 MLP 第一层权重
mlp_importance = np.mean(np.abs(mlp_weight), axis=0)  # 用权重绝对值均值表示 MLP 基因重要性

logreg_importance = np.abs(logreg.coef_[0])  # 用逻辑回归系数绝对值表示基因重要性

importance_df = pd.DataFrame({  # 构建重要性表
    "gene": selected_genes,  # 基因名
    "mlp_importance": mlp_importance,  # MLP 重要性
    "logistic_importance": logreg_importance  # 逻辑回归重要性
})  # 结束表格构建

importance_df["mlp_importance_norm"] = (  # 归一化 MLP 重要性
    importance_df["mlp_importance"] -  # 减去最小值
    importance_df["mlp_importance"].min()  # MLP 重要性最小值
) / (  # 除以极差
    importance_df["mlp_importance"].max() -  # MLP 重要性最大值
    importance_df["mlp_importance"].min()  # MLP 重要性最小值
)  # 结束归一化

importance_df["logistic_importance_norm"] = (  # 归一化逻辑回归重要性
    importance_df["logistic_importance"] -  # 减去最小值
    importance_df["logistic_importance"].min()  # 逻辑回归重要性最小值
) / (  # 除以极差
    importance_df["logistic_importance"].max() -  # 逻辑回归重要性最大值
    importance_df["logistic_importance"].min()  # 逻辑回归重要性最小值
)  # 结束归一化

importance_df["mean_importance"] = importance_df[  # 计算两个模型平均重要性
    ["mlp_importance_norm", "logistic_importance_norm"]  # 选择两个归一化重要性列
].mean(axis=1)  # 按行求平均

importance_df = importance_df.sort_values(  # 按综合重要性排序
    by="mean_importance",  # 排序依据列
    ascending=False  # 从大到小排序
)  # 结束排序

importance_output_path = "/Users/chengyuhang/Desktop/ensemble_gene_importance.csv"  # 基因重要性输出路径
importance_df.to_csv(importance_output_path, index=False)  # 保存基因重要性表

print(f"Gene importance saved to: {importance_output_path}")  # 打印保存路径

print("\nAll done.")  # 打印全部完成