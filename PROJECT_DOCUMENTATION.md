# HCC 复发与非复发预测项目说明

## 1. 项目简介

本项目基于 HCC（hepatocellular carcinoma，肝细胞癌）患者的基因表达矩阵与复发标签，构建一个用于预测患者是否复发的二分类模型。当前主程序为 `Modifiedcode.py`，整体思路是先在训练集内筛选与复发相关的基因，再分别训练 MLP 神经网络和 ElasticNet Logistic Regression，最后对两个模型的预测概率取平均形成集成预测。

从你的描述看，当前训练结果测试准确率大约在 73% 左右。考虑到该数据集只有 108 个样本，而基因数量达到 31,333 个，这是一个典型的小样本、高维度生物信息学预测问题，因此 73% 左右的准确率说明模型已经捕捉到了一部分复发相关信号，但仍需要通过交叉验证、外部验证集、特征稳定性分析等方式进一步确认泛化能力。

## 2. 项目文件结构

```text
scRNAseq-HCC/
├── Modifiedcode.py                         # 主训练与预测脚本
├── GSE76427_expr_gene_RFS_108_clean.csv    # 基因表达矩阵，行为基因，列为样本
├── GSE76427_rfs_label_108.csv              # 样本复发标签，1=复发，0=非复发
├── README.md                               # 原始 README
└── PROJECT_DOCUMENTATION.md                # 本项目说明文档
```

## 3. 数据说明

### 3.1 表达矩阵

文件：`GSE76427_expr_gene_RFS_108_clean.csv`

表达矩阵的基本格式为：

- 行：基因，例如 `CYP2E1`、`CRP` 等。
- 列：样本 ID，例如 `GSM2011285`、`GSM2011286` 等。
- 数值：对应样本中该基因的表达值。

当前文件包含：

- 样本数：108
- 基因数：31,333

### 3.2 标签文件

文件：`GSE76427_rfs_label_108.csv`

标签文件包含两列：

| 字段 | 含义 |
| --- | --- |
| `sample` | 样本 ID |
| `recurrence` | 复发标签，`1` 表示复发，`0` 表示非复发 |

当前标签分布为：

| 类别 | 样本数 |
| --- | ---: |
| 复发 `1` | 48 |
| 非复发 `0` | 60 |
| 总计 | 108 |

## 4. 建模任务

本项目要解决的问题是：给定一个 HCC 患者样本的基因表达谱，预测该患者是否会发生复发。

这是一个监督学习二分类任务：

- 输入 `X`：样本的基因表达特征。
- 输出 `y`：复发标签，`1` 为复发，`0` 为非复发。
- 目标：学习表达模式与复发风险之间的关系，并输出复发概率。

## 5. 整体流程

主程序 `Modifiedcode.py` 的流程可以概括为：

1. 读取基因表达矩阵和复发标签。
2. 对表达矩阵和标签文件取共同样本，保证样本顺序一致。
3. 使用分层抽样划分训练集、验证集和测试集。
4. 只在训练集上计算 t-score，筛选与复发差异最明显的前 90 个基因。
5. 使用训练集拟合 `StandardScaler`，再转换验证集和测试集，避免数据泄漏。
6. 训练一个 MLP 二分类神经网络。
7. 训练一个 ElasticNet Logistic Regression。
8. 分别获得 MLP 和 Logistic Regression 的复发预测概率。
9. 将两个模型的概率平均，得到集成概率。
10. 在验证集上搜索最佳分类阈值。
11. 在测试集上评估 Accuracy、Precision、Recall、F1 和混淆矩阵。
12. 保存测试集预测结果、MLP embedding、筛选基因和综合基因重要性。

## 6. 代码模块说明及对应逐行注释代码

下面按 `Modifiedcode.py` 的实际执行顺序说明每个模块。每个模块后面都附上该模块对应的逐行注释代码，代码逻辑来自 `Modifiedcode.py`，注释版本来自 `Modifiedcode_annotated.py`。

### 模块 0：参数设置

该模块设置输入数据路径、模型超参数和随机种子。`TOPK_GENES` 控制筛选多少个复发相关基因，`EPOCHS` 和 `LR` 控制 MLP 训练过程，`RANDOM_SEED` 用于提高实验可复现性。

需要注意的是，当前路径仍写死为 `/Users/chengyuhang/Desktop/...`，如果在本仓库中运行，建议改为仓库内路径或相对路径。

```python
# =========================  # 模块分隔线，用于让代码结构更清晰
# 0. 参数  # 0. 参数模块标题或说明，标记当前代码功能区域
# =========================  # 模块分隔线，用于让代码结构更清晰
EXPR_PATH = "/Users/chengyuhang/Desktop/GSE76427_expr_gene_RFS_108_clean.csv"  # 设置基因表达矩阵文件路径，行为基因、列为样本
LABEL_PATH = "/Users/chengyuhang/Desktop/GSE76427_rfs_label_108.csv"  # 设置样本复发标签文件路径，包含 sample 和 recurrence 两列

TOPK_GENES = 90  # 设置 t-score 筛选的基因数量，只保留差异最明显的前 90 个基因
EPOCHS = 260  # 设置 MLP 最大训练轮数，实际可能因 early stopping 提前结束
LR = 5e-4  # 设置 Adam 优化器学习率，控制每次参数更新步长
RANDOM_SEED = 42  # 设置随机种子，使数据划分和模型初始化更可复现

np.random.seed(RANDOM_SEED)  # 固定 NumPy 随机数种子，保证 NumPy 相关随机过程可复现
torch.manual_seed(RANDOM_SEED)  # 固定 PyTorch 随机数种子，保证模型初始化等过程尽量可复现
```

### 模块 1：读取数据

该模块读取基因表达矩阵和复发标签，并统一标签列名。表达矩阵中行为基因、列为样本；标签文件中 `recurrence` 被转换成整数型标签。

随后代码取表达矩阵和标签文件共有的样本，保证 `X`、`y` 和 `sample_names` 的顺序严格一致。这一步非常关键，否则模型可能会把某个样本的表达数据对应到另一个样本的标签。

```python
# =========================  # 模块分隔线，用于让代码结构更清晰
# 1. 读取数据  # 1. 读取数据模块标题或说明，标记当前代码功能区域
# =========================  # 模块分隔线，用于让代码结构更清晰
print("Loading data...")  # 打印提示信息，表示开始读取数据

df = pd.read_csv(EXPR_PATH, index_col=0)  # 读取基因表达矩阵，并把第一列作为基因名索引

label_df = pd.read_csv(LABEL_PATH)  # 读取复发标签表格
label_df = label_df.rename(columns={  # 开始一个多行字典定义
    "sample": "sample_id",  # 把标签文件中的 sample 列重命名为 sample_id，方便后续统一处理
    "recurrence": "label"  # 把 recurrence 列重命名为 label，作为模型训练标签
})  # 结束当前多行函数调用或字典定义
label_df["label"] = label_df["label"].astype(int)  # 把复发标签转换为整数类型，确保 0/1 标签可用于模型训练和指标计算

common_samples = [s for s in df.columns if s in set(label_df["sample_id"])]  # 取表达矩阵和标签文件共有的样本，避免无标签样本进入训练

df = df[common_samples]  # 按照共有样本筛选并重发表达矩阵列顺序
label_df = label_df.set_index("sample_id").loc[common_samples].reset_index()  # 按照表达矩阵样本顺序重排标签，确保 X 和 y 一一对应

gene_names_all = np.array(df.index.tolist())  # 保存全部基因名称，后续根据索引提取筛选基因名
sample_names = np.array(df.columns.tolist())  # 保存样本名称，后续用于输出测试集预测结果

X = df.values.T.astype(np.float32)  # 将表达矩阵转置为 样本 x 基因，并转换为 float32 以便模型训练
y = label_df["label"].values.astype(int)  # 提取标签数组，并确保标签为整数类型

print(f"Samples with labels: {X.shape[0]}")  # 打印有标签样本数量
print(f"Genes: {X.shape[1]}")  # 打印基因特征数量
print(f"Recurrence samples: {y.sum()}")  # 打印复发样本数量，标签 1 的总和即复发数
print(f"Non-recurrence samples: {(y == 0).sum()}")  # 打印非复发样本数量，即标签为 0 的样本数
```

### 模块 2：分层划分训练集、验证集和测试集

该模块使用 `train_test_split` 进行两次划分。第一次划出 20% 测试集，第二次从剩余数据中划出验证集，因此最终比例约为训练集 60%、验证集 20%、测试集 20%。

`stratify=y` 用于保持复发和非复发样本比例，避免某个集合中类别分布明显偏斜。

```python
# =========================  # 模块分隔线，用于让代码结构更清晰
# 2. 分层划分 train / val / test  # 2. 分层划分 train / val / test模块标题或说明，标记当前代码功能区域
# =========================  # 模块分隔线，用于让代码结构更清晰
print("\nSplitting train / val / test...")  # 打印提示信息，表示开始划分训练集、验证集和测试集

X_trainval, X_test, y_trainval, y_test, sample_trainval, sample_test = train_test_split(  # 开始一个多行函数调用或对象创建
    X,  # 传入完整特征矩阵作为待划分数据
    y,  # 传入标签数组作为待划分标签
    sample_names,  # 同时划分样本名，方便后续追踪测试集样本 ID
    test_size=0.2,  # 设置测试集比例为 20%
    random_state=RANDOM_SEED,  # 使用固定随机种子，使数据划分可复现
    stratify=y  # 按 y 的类别比例进行分层抽样，保持复发/非复发比例
)  # 结束当前多行函数调用

X_train, X_val, y_train, y_val, sample_train, sample_val = train_test_split(  # 开始一个多行函数调用或对象创建
    X_trainval,  # 传入训练验证合并集特征，准备继续划分训练集和验证集
    y_trainval,  # 传入训练验证合并集标签
    sample_trainval,  # 传入训练验证合并集样本名
    test_size=0.25,  # 从训练验证合并集中划出 25% 作为验证集，整体约等于 20%
    random_state=RANDOM_SEED,  # 使用固定随机种子，使数据划分可复现
    stratify=y_trainval  # 按训练验证合并集标签比例继续分层划分
)  # 结束当前多行函数调用

print(f"Train samples: {X_train.shape[0]}")  # 打印训练集样本数量
print(f"Val samples: {X_val.shape[0]}")  # 打印验证集样本数量
print(f"Test samples: {X_test.shape[0]}")  # 打印测试集样本数量
print(f"Train recurrence: {y_train.sum()}")  # 打印训练集中复发样本数量
print(f"Train non-recurrence: {(y_train == 0).sum()}")  # 打印训练集中非复发样本数量
print(f"Val recurrence: {y_val.sum()}")  # 打印验证集中复发样本数量
print(f"Val non-recurrence: {(y_val == 0).sum()}")  # 打印验证集中非复发样本数量
print(f"Test recurrence: {y_test.sum()}")  # 打印测试集中复发样本数量
print(f"Test non-recurrence: {(y_test == 0).sum()}")  # 打印测试集中非复发样本数量
```

### 模块 3：只用训练集筛选差异基因

该模块定义 `select_genes_by_tscore` 函数。函数分别计算复发组和非复发组每个基因的均值、方差和标准误，再计算 t-score。

代码按 `abs_t_score` 从大到小排序，选择前 `TOPK_GENES` 个基因。这里特意只使用训练集做基因筛选，避免验证集和测试集信息泄漏到特征选择步骤中。

```python
# =========================  # 模块分隔线，用于让代码结构更清晰
# 3. 只用训练集筛选差异基因  # 3. 只用训练集筛选差异基因模块标题或说明，标记当前代码功能区域
# =========================  # 模块分隔线，用于让代码结构更清晰
def select_genes_by_tscore(X_train, y_train, gene_names, topk=100):  # 定义基于 t-score 的基因筛选函数，只使用训练集计算差异程度
    pos = X_train[y_train == 1]  # 提取训练集中复发样本的表达矩阵
    neg = X_train[y_train == 0]  # 提取训练集中非复发样本的表达矩阵

    pos_mean = pos.mean(axis=0)  # 计算复发组每个基因的平均表达值
    neg_mean = neg.mean(axis=0)  # 计算非复发组每个基因的平均表达值

    pos_var = pos.var(axis=0)  # 计算复发组每个基因的表达方差
    neg_var = neg.var(axis=0)  # 计算非复发组每个基因的表达方差

    n_pos = pos.shape[0]  # 记录复发组样本数量
    n_neg = neg.shape[0]  # 记录非复发组样本数量

    se = np.sqrt(pos_var / n_pos + neg_var / n_neg)  # 计算两组均值差的标准误，作为 t-score 分母
    se[se == 0] = 1e-6  # 防止标准误为 0 导致除零错误，用极小值替代

    t_score = (pos_mean - neg_mean) / se  # 计算每个基因的 t-score，表示复发组与非复发组差异方向和强度
    abs_t = np.abs(t_score)  # 取 t-score 绝对值，只关注差异强度，不关心上调或下调方向

    top_idx = np.argsort(abs_t)[::-1][:topk]  # 按绝对 t-score 从大到小排序，取前 topk 个基因索引

    selected_info = pd.DataFrame({  # 创建表格保存筛选基因及统计信息
        "gene": gene_names[top_idx],  # 保存被选中的基因名称
        "t_score": t_score[top_idx],  # 保存被选中基因的 t-score，保留差异方向
        "abs_t_score": abs_t[top_idx],  # 保存被选中基因的绝对 t-score，表示差异强度
        "recurrence_mean": pos_mean[top_idx],  # 保存复发组中这些基因的平均表达
        "non_recurrence_mean": neg_mean[top_idx]  # 保存非复发组中这些基因的平均表达
    })  # 结束当前多行函数调用或字典定义

    return top_idx, selected_info  # 返回筛选基因索引和对应统计信息表


print("\nSelecting recurrence-related genes...")  # 打印提示信息，表示开始筛选复发相关基因

top_idx, selected_gene_info = select_genes_by_tscore(  # 开始一个多行函数调用或对象创建
    X_train,  # 传入训练集表达矩阵用于基因筛选
    y_train,  # 传入训练集标签用于区分复发和非复发组
    gene_names_all,  # 传入全部基因名称，用于把索引映射回基因名
    TOPK_GENES  # 传入需要保留的基因数量
)  # 结束当前多行函数调用

selected_genes = gene_names_all[top_idx]  # 根据筛选索引得到最终入选基因名称

X_train = X_train[:, top_idx]  # 训练集只保留筛选出的基因特征
X_val = X_val[:, top_idx]  # 验证集使用同一批筛选基因，保持特征空间一致
X_test = X_test[:, top_idx]  # 测试集使用同一批筛选基因，避免特征不一致

print(f"Selected genes: {len(selected_genes)}")  # 打印最终筛选出的基因数量
```

### 模块 4：标准化

该模块使用 `StandardScaler` 对筛选后的表达矩阵进行标准化。标准化的均值和方差只从训练集学习，然后应用到验证集和测试集。

这样可以让模型训练更稳定，同时避免把验证集或测试集的统计信息提前暴露给模型。

```python
# =========================  # 模块分隔线，用于让代码结构更清晰
# 4. 标准化  # 4. 标准化模块标题或说明，标记当前代码功能区域
# =========================  # 模块分隔线，用于让代码结构更清晰
print("\nStandardizing data...")  # 打印提示信息，表示开始标准化数据

scaler = StandardScaler()  # 创建标准化器，用于把每个基因特征转为均值 0、方差 1

X_train_scaled = scaler.fit_transform(X_train)  # 在训练集上拟合标准化参数，并转换训练集
X_val_scaled = scaler.transform(X_val)  # 使用训练集标准化参数转换验证集，避免数据泄漏
X_test_scaled = scaler.transform(X_test)  # 使用训练集标准化参数转换测试集，避免数据泄漏
```

### 模块 5：MLP 模型定义

该模块定义 `MLPClassifier`。模型先通过 `feature_extractor` 把输入基因特征映射到低维 embedding，再通过 `classifier` 输出一个二分类 logit。

网络中使用了 `BatchNorm1d`、`ReLU` 和 `Dropout`。其中 BatchNorm 有助于稳定训练，Dropout 有助于降低小样本场景下的过拟合。

```python
# =========================  # 模块分隔线，用于让代码结构更清晰
# 5. MLP 模型  # 5. MLP 模型模块标题或说明，标记当前代码功能区域
# =========================  # 模块分隔线，用于让代码结构更清晰
class MLPClassifier(nn.Module):  # 定义 MLP 二分类模型，继承 PyTorch 的 nn.Module
    def __init__(self, input_dim):  # 定义模型初始化函数，input_dim 为输入基因特征数量
        super().__init__()  # 调用父类初始化方法，注册 PyTorch 模型基础结构

        self.feature_extractor = nn.Sequential(  # 定义特征提取网络，用于把基因表达压缩成低维 embedding
            nn.Linear(input_dim, 32),  # 第一层全连接层，将输入特征映射到 32 维隐藏表示
            nn.BatchNorm1d(32),  # 对 32 维隐藏层做批归一化，提高训练稳定性
            nn.ReLU(),  # 使用 ReLU 激活函数，引入非线性表达能力
            nn.Dropout(0.4),  # 训练时随机丢弃 40% 神经元，降低过拟合风险

            nn.Linear(32, 16),  # 第二层全连接层，将 32 维隐藏表示压缩到 16 维
            nn.BatchNorm1d(16),  # 对 16 维隐藏层做批归一化
            nn.ReLU(),  # 使用 ReLU 激活函数，引入非线性表达能力
            nn.Dropout(0.3)  # 训练时随机丢弃 30% 神经元，进一步正则化模型
        )  # 结束当前多行函数调用

        self.classifier = nn.Linear(16, 1)  # 定义输出层，将 16 维 embedding 映射为单个二分类 logit

    def forward(self, x):  # 定义前向传播过程，描述输入如何经过网络得到输出
        embedding = self.feature_extractor(x)  # 通过特征提取器得到样本的 16 维 embedding
        logits = self.classifier(embedding).squeeze(1)  # 通过分类层得到 logit，并压缩为一维向量
        return logits, embedding  # 同时返回分类 logit 和中间层 embedding，便于训练和后续分析


model = MLPClassifier(input_dim=TOPK_GENES)  # 实例化 MLP 模型，输入维度等于筛选后的基因数量

x_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)  # 将训练集特征转换为 PyTorch float32 张量
y_train_t = torch.tensor(y_train, dtype=torch.float32)  # 将训练集标签转换为 float32 张量，以匹配 BCE 损失函数要求

x_val_t = torch.tensor(X_val_scaled, dtype=torch.float32)  # 将验证集特征转换为 PyTorch 张量
y_val_t = torch.tensor(y_val, dtype=torch.float32)  # 将验证集标签转换为 PyTorch 张量

x_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)  # 将测试集特征转换为 PyTorch 张量，用于最终预测

pos_weight = torch.tensor(  # 开始定义正类权重，用于处理复发/非复发样本不均衡
    [(y_train == 0).sum() / (y_train == 1).sum()],  # 用负类数量除以正类数量，得到正类在损失中的加权系数
    dtype=torch.float32  # 指定权重张量类型为 float32
)  # 结束当前多行函数调用

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)  # 定义二分类损失函数，内部会把 logit 转换为概率并计算 BCE

optimizer = torch.optim.Adam(  # 创建 Adam 优化器，用于更新 MLP 参数
    model.parameters(),  # 把模型所有可训练参数交给优化器
    lr=LR,  # 设置优化器学习率
    weight_decay=1e-3  # 设置 L2 权重衰减，帮助降低过拟合
)  # 结束当前多行函数调用

print(f"\nPositive class weight: {pos_weight.item():.4f}")  # 打印正类权重，便于确认类别加权强度
```

### 模块 6：MLP 训练与 Early Stopping

该模块把标准化后的数据转换成 PyTorch tensor，并使用 `BCEWithLogitsLoss` 训练 MLP。由于类别数量不完全均衡，`pos_weight` 会提高正类样本在损失函数中的权重。

训练过程中，代码监控验证集 loss。如果验证集 loss 连续 `patience=20` 个 epoch 没有下降，就提前停止训练，并恢复验证集表现最好的模型参数。

```python
# =========================  # 模块分隔线，用于让代码结构更清晰
# 6. MLP 训练 + Early stopping  # 6. MLP 训练 + Early stopping模块标题或说明，标记当前代码功能区域
# =========================  # 模块分隔线，用于让代码结构更清晰
print("\nTraining MLP...")  # 打印提示信息，表示开始训练 MLP

best_val_loss = float("inf")  # 初始化最佳验证集损失为无穷大，后续只要更小就更新
best_state = None  # 用于保存验证集表现最好的模型参数
patience = 20  # 设置 early stopping 容忍轮数，连续 20 轮不提升就停止
wait = 0  # 验证集性能提升后，重置 early stopping 等待计数

for epoch in tqdm(range(EPOCHS), desc="Training MLP"):  # 循环训练最多 EPOCHS 轮，并显示进度条
    model.train()  # 切换到训练模式，启用 Dropout 和 BatchNorm 的训练行为

    optimizer.zero_grad()  # 清空上一轮梯度，避免梯度累积

    train_logits, train_embedding = model(x_train_t)  # 前向传播训练集，得到训练 logit 和 embedding
    loss = criterion(train_logits, y_train_t)  # 计算训练集二分类损失

    loss.backward()  # 反向传播，计算每个参数的梯度
    optimizer.step()  # 根据梯度更新模型参数

    model.eval()  # 切换到评估模式，关闭 Dropout 并固定 BatchNorm 行为

    with torch.no_grad():  # 关闭梯度计算，节省内存并避免验证阶段影响梯度
        val_logits, val_embedding = model(x_val_t)  # 前向传播验证集，得到验证 logit 和 embedding
        val_loss = criterion(val_logits, y_val_t).item()  # 计算验证集损失，并转换为 Python 数值

    if val_loss < best_val_loss:  # 如果当前验证集损失优于历史最佳，则保存模型
        best_val_loss = val_loss  # 更新历史最佳验证集损失
        best_state = model.state_dict()  # 保存当前模型参数状态
        wait = 0  # 验证集性能提升后，重置 early stopping 等待计数
    else:  # 如果当前验证集损失没有提升，则进入等待计数逻辑
        wait += 1  # 验证集未提升轮数加 1

    if (epoch + 1) % 20 == 0:  # 每训练 20 轮打印一次训练和验证损失
        print(  # 开始打印多行格式化训练日志
            f"\nEpoch [{epoch + 1}/{EPOCHS}] "  # 打印当前 epoch 和总 epoch 数
            f"Train Loss: {loss.item():.4f} "  # 打印当前训练损失
            f"Val Loss: {val_loss:.4f}"  # 打印当前验证损失
        )  # 结束当前多行函数调用

    if wait >= patience:  # 如果验证集连续未提升轮数达到 patience，则提前停止
        print(f"\nEarly stopping at epoch {epoch + 1}")  # 打印 early stopping 发生的训练轮数
        break  # 跳出训练循环，结束 MLP 训练

model.load_state_dict(best_state)  # 恢复验证集损失最低时的模型参数

print("MLP training finished.")  # 打印提示信息，表示 MLP 训练完成
```

### 模块 7：Logistic Regression 模型

该模块训练一个 ElasticNet Logistic Regression。ElasticNet 同时包含 L1 和 L2 正则，适合基因表达这类高维、小样本数据。

`class_weight="balanced"` 用于处理类别不均衡，`solver="saga"` 是 scikit-learn 中支持 ElasticNet Logistic Regression 的优化器。

```python
# =========================  # 模块分隔线，用于让代码结构更清晰
# 7. Logistic Regression  # 7. Logistic Regression模块标题或说明，标记当前代码功能区域
# =========================  # 模块分隔线，用于让代码结构更清晰
print("\nTraining Logistic Regression...")  # 打印提示信息，表示开始训练逻辑回归模型

logreg = LogisticRegression(  # 创建逻辑回归分类器
    penalty="elasticnet",  # 使用 ElasticNet 正则化，同时结合 L1 和 L2 约束
    l1_ratio=0.5,  # 设置 L1 与 L2 的混合比例，0.5 表示二者权重均衡
    C=0.5,  # 设置正则化强度的倒数，数值越小正则越强
    class_weight="balanced",  # 按类别频率自动平衡样本权重，缓解类别不均衡
    solver="saga",  # 使用 saga 优化器，因为它支持 ElasticNet 正则
    max_iter=3000,  # 设置最大迭代次数，保证模型有足够机会收敛
    random_state=RANDOM_SEED  # 设置随机种子，提高逻辑回归训练可复现性
)  # 结束当前多行函数调用
logreg.fit(X_train_scaled, y_train)  # 在标准化训练集上拟合逻辑回归模型

print("Logistic Regression finished.")  # 打印提示信息，表示逻辑回归训练完成
```

### 模块 8：MLP 与 Logistic Regression 集成预测

该模块分别计算 MLP 和 Logistic Regression 在验证集、测试集上的复发概率。MLP 输出的是 logit，因此先用 `torch.sigmoid` 转换为概率。

最终集成方式是简单平均：`(mlp_prob + logreg_prob) / 2`。这种方法直观、稳定，可以降低单个模型在小样本数据上的波动。

```python
# =========================  # 模块分隔线，用于让代码结构更清晰
# 8. MLP + Logistic 预测概率  # 8. MLP + Logistic 预测概率模块标题或说明，标记当前代码功能区域
# =========================  # 模块分隔线，用于让代码结构更清晰
print("\nRunning ensemble prediction...")  # 打印提示信息，表示开始集成预测

model.eval()  # 切换到评估模式，关闭 Dropout 并固定 BatchNorm 行为

with torch.no_grad():  # 关闭梯度计算，节省内存并避免验证阶段影响梯度
    mlp_val_logits, mlp_val_embedding = model(x_val_t)  # 使用 MLP 预测验证集 logit 和 embedding
    mlp_test_logits, mlp_test_embedding = model(x_test_t)  # 使用 MLP 预测测试集 logit 和 embedding

    mlp_val_prob = torch.sigmoid(mlp_val_logits).numpy()  # 将验证集 MLP logit 通过 sigmoid 转换为复发概率
    mlp_test_prob = torch.sigmoid(mlp_test_logits).numpy()  # 将测试集 MLP logit 通过 sigmoid 转换为复发概率

logreg_val_prob = logreg.predict_proba(X_val_scaled)[:, 1]  # 获取逻辑回归在验证集上预测为复发类别的概率
logreg_test_prob = logreg.predict_proba(X_test_scaled)[:, 1]  # 获取逻辑回归在测试集上预测为复发类别的概率

val_prob = (mlp_val_prob + logreg_val_prob) / 2  # 对 MLP 和逻辑回归的验证集概率取平均，得到集成概率
test_prob = (mlp_test_prob + logreg_test_prob) / 2  # 对 MLP 和逻辑回归的测试集概率取平均，得到集成概率
```

### 模块 9：验证集自动寻找最佳阈值

该模块不固定使用 0.5 作为分类阈值，而是在验证集上从 0.40 到 0.60 搜索最佳阈值。

评价标准使用 F1-score，因为它同时考虑 Precision 和 Recall，适合复发预测这种需要兼顾误报和漏报的任务。

```python
# =========================  # 模块分隔线，用于让代码结构更清晰
# 9. 验证集自动寻找最佳阈值  # 9. 验证集自动寻找最佳阈值模块标题或说明，标记当前代码功能区域
# =========================  # 模块分隔线，用于让代码结构更清晰
print("\nSearching best threshold on validation set...")  # 打印提示信息，表示开始在验证集上搜索最佳阈值

best_threshold = 0.5  # 初始化最佳阈值为常用默认值 0.5
best_f1 = -1  # 初始化最佳 F1 为 -1，确保第一次计算后会更新

for threshold in tqdm(np.arange(0.40, 0.61, 0.01), desc="Searching threshold"):  # 在 0.40 到 0.60 之间按 0.01 步长遍历候选阈值
    val_pred = (val_prob >= threshold).astype(int)  # 根据当前阈值把验证集概率转换为 0/1 预测标签
    f1 = f1_score(y_val, val_pred, zero_division=0)  # 计算当前阈值下的验证集 F1-score，避免除零时报错

    if f1 > best_f1:  # 如果当前 F1 优于历史最佳，则更新最佳阈值
        best_f1 = f1  # 保存当前最佳 F1-score
        best_threshold = threshold  # 保存当前最佳分类阈值

print(f"\nBest threshold from validation set: {best_threshold:.2f}")  # 打印验证集搜索得到的最佳阈值
print(f"Best validation F1: {best_f1:.4f}")  # 打印最佳阈值对应的验证集 F1-score
```

### 模块 10：测试集评估

该模块使用验证集选出的最佳阈值，在测试集上生成最终预测标签，并计算 Accuracy、Precision、Recall、F1-score 和混淆矩阵。

其中 TP、FP、FN、TN 可以帮助进一步理解模型错误类型：模型是更容易漏掉复发样本，还是更容易把非复发样本误判为复发。

```python
# =========================  # 模块分隔线，用于让代码结构更清晰
# 10. 测试集评估  # 10. 测试集评估模块标题或说明，标记当前代码功能区域
# =========================  # 模块分隔线，用于让代码结构更清晰
print("\nEvaluating on test set...")  # 打印提示信息，表示开始测试集评估

test_pred = (test_prob >= best_threshold).astype(int)  # 使用验证集选出的最佳阈值，把测试集概率转换为预测标签

acc = accuracy_score(y_test, test_pred)  # 计算测试集准确率
precision = precision_score(y_test, test_pred, zero_division=0)  # 计算测试集精确率，即预测复发样本中真实复发的比例
recall = recall_score(y_test, test_pred, zero_division=0)  # 计算测试集召回率，即真实复发样本中被识别出来的比例
f1 = f1_score(y_test, test_pred, zero_division=0)  # 计算测试集 F1-score，综合衡量精确率和召回率

tn, fp, fn, tp = confusion_matrix(y_test, test_pred).ravel()  # 展开混淆矩阵，得到 TN、FP、FN、TP 四个数值

print("\n=== Test Result ===")  # 打印测试集结果标题
print(f"Test Accuracy: {acc:.4f}")  # 打印测试集准确率
print(f"Precision: {precision:.4f}")  # 打印测试集精确率
print(f"Recall: {recall:.4f}")  # 打印测试集召回率
print(f"F1-score: {f1:.4f}")  # 打印测试集 F1-score

print("\nProbability summary:")  # 打印预测概率统计标题
print(f"Min probability: {test_prob.min():.4f}")  # 打印测试集集成概率最小值
print(f"Max probability: {test_prob.max():.4f}")  # 打印测试集集成概率最大值
print(f"Mean probability: {test_prob.mean():.4f}")  # 打印测试集集成概率平均值

print("\nConfusion Matrix:")  # 打印混淆矩阵标题
print(f"TP: {tp}, FP: {fp}")  # 打印真正例和假正例数量
print(f"FN: {fn}, TN: {tn}")  # 打印假负例和真负例数量
```

### 模块 11：保存预测结果

该模块把测试集每个样本的真实标签、预测标签、集成概率、MLP 概率和 Logistic Regression 概率保存为 CSV。

这个文件适合用于后续人工检查错分样本，或者进一步画 ROC 曲线、概率分布图等。

```python
# =========================  # 模块分隔线，用于让代码结构更清晰
# 11. 保存预测结果  # 11. 保存预测结果模块标题或说明，标记当前代码功能区域
# =========================  # 模块分隔线，用于让代码结构更清晰
print("\nSaving prediction result...")  # 打印提示信息，表示开始保存预测结果

pred_df = pd.DataFrame({  # 创建预测结果表格
    "sample_id": sample_test,  # 保存测试集样本 ID
    "true_label": y_test,  # 保存测试集真实标签
    "pred_label": test_pred,  # 保存测试集预测标签
    "ensemble_probability": test_prob,  # 保存集成模型预测复发概率
    "mlp_probability": mlp_test_prob,  # 保存 MLP 单模型预测复发概率
    "logistic_probability": logreg_test_prob  # 保存逻辑回归单模型预测复发概率
})  # 结束当前多行函数调用或字典定义

pred_output_path = "/Users/chengyuhang/Desktop/ensemble_prediction_result.csv"  # 设置测试集预测结果输出路径
pred_df.to_csv(pred_output_path, index=False)  # 将预测结果保存为 CSV，不额外保存行索引

print(f"Prediction result saved to: {pred_output_path}")  # 打印预测结果保存位置
```

### 模块 12：保存样本 embedding

该模块保存 MLP 在测试集上提取到的中间层 embedding。

这些 embedding 可以看作模型学习到的低维样本表示，后续可以用于 PCA、t-SNE、UMAP 可视化，观察复发和非复发样本是否在特征空间中出现分离趋势。

```python
# =========================  # 模块分隔线，用于让代码结构更清晰
# 12. 保存样本 embedding  # 12. 保存样本 embedding模块标题或说明，标记当前代码功能区域
# =========================  # 模块分隔线，用于让代码结构更清晰
print("\nSaving sample embedding...")  # 打印提示信息，表示开始保存样本 embedding

embedding_df = pd.DataFrame(  # 创建 embedding 表格，用于保存 MLP 中间层样本表示
    mlp_test_embedding.detach().numpy(),  # 从计算图中分离测试集 embedding，并转换为 NumPy 数组
    index=sample_test  # 使用测试集样本 ID 作为 embedding 表格行索引
)  # 结束当前多行函数调用

embedding_output_path = "/Users/chengyuhang/Desktop/ensemble_mlp_sample_embedding.csv"  # 设置 MLP 样本 embedding 输出路径
embedding_df.to_csv(embedding_output_path)  # 将测试集 embedding 保存为 CSV

print(f"Sample embedding saved to: {embedding_output_path}")  # 打印 embedding 保存位置
```

### 模块 13：保存筛选基因

该模块保存 t-score 筛选出的基因列表及其统计信息，包括 t-score、绝对 t-score、复发组均值和非复发组均值。

该文件可用于后续生物学解释，例如检查这些基因是否与 HCC 复发、肿瘤进展、免疫微环境或预后相关。

```python
# =========================  # 模块分隔线，用于让代码结构更清晰
# 13. 保存筛选基因  # 13. 保存筛选基因模块标题或说明，标记当前代码功能区域
# =========================  # 模块分隔线，用于让代码结构更清晰
print("\nSaving selected genes...")  # 打印提示信息，表示开始保存筛选基因

gene_output_path = "/Users/chengyuhang/Desktop/ensemble_selected_genes.csv"  # 设置筛选基因统计表输出路径
selected_gene_info.to_csv(gene_output_path, index=False)  # 将筛选基因及 t-score 信息保存为 CSV

print(f"Selected genes saved to: {gene_output_path}")  # 打印筛选基因文件保存位置
```

### 模块 14：保存综合基因重要性

该模块分别从 MLP 第一层权重和 Logistic Regression 系数中估计基因重要性，然后取平均作为综合重要性分数。

这个结果可以帮助筛选潜在关键基因。不过需要注意，神经网络权重和 Logistic 系数的可解释性有限，最终仍建议结合交叉验证稳定性和生物学证据来判断关键基因。

```python
# =========================  # 模块分隔线，用于让代码结构更清晰
# 14. 保存综合基因重要性  # 14. 保存综合基因重要性模块标题或说明，标记当前代码功能区域
# =========================  # 模块分隔线，用于让代码结构更清晰
print("\nSaving gene importance...")  # 打印提示信息，表示开始保存基因重要性

mlp_weight = model.feature_extractor[0].weight.detach().numpy()  # 提取 MLP 第一层权重，用于估计输入基因的重要性
mlp_importance = np.mean(np.abs(mlp_weight), axis=0)  # 对第一层权重取绝对值并按隐藏单元求平均，得到 MLP 基因重要性

logreg_importance = np.abs(logreg.coef_[0])  # 取逻辑回归系数绝对值，作为逻辑回归基因重要性

importance_df = pd.DataFrame({  # 创建综合基因重要性表格
    "gene": selected_genes,  # 保存筛选后的基因名称
    "mlp_importance": mlp_importance,  # 保存 MLP 估计的基因重要性
    "logistic_importance": logreg_importance  # 保存逻辑回归估计的基因重要性
})  # 结束当前多行函数调用或字典定义

importance_df["mean_importance"] = importance_df[  # 新增平均重要性列，综合两个模型的基因重要性
    ["mlp_importance", "logistic_importance"]  # 选择 MLP 和逻辑回归两列重要性用于求平均
].mean(axis=1)  # 按行计算两个模型重要性的平均值

importance_df = importance_df.sort_values(  # 按照综合重要性对基因进行排序
    by="mean_importance",  # 指定按 mean_importance 列排序
    ascending=False  # 设置为降序，使重要性最高的基因排在最前面
)  # 结束当前多行函数调用

importance_output_path = "/Users/chengyuhang/Desktop/ensemble_gene_importance.csv"  # 设置综合基因重要性输出路径
importance_df.to_csv(importance_output_path, index=False)  # 将综合基因重要性表保存为 CSV，不保存行索引

print(f"Gene importance saved to: {importance_output_path}")  # 打印基因重要性文件保存位置

print("\nAll done.")  # 打印完成提示，表示整个训练、评估和保存流程结束
```

## 7. 如何运行

建议先安装依赖：

```bash
pip install numpy pandas torch scikit-learn tqdm
```

如果继续使用当前代码，需要确认 `Modifiedcode.py` 中的路径存在：

```python
EXPR_PATH = "/Users/chengyuhang/Desktop/GSE76427_expr_gene_RFS_108_clean.csv"
LABEL_PATH = "/Users/chengyuhang/Desktop/GSE76427_rfs_label_108.csv"
```

如果要使用当前仓库中的数据，建议改成：

```python
BASE_DIR = "/Users/chengyuhang2/GitHub/scRNAseq-HCC"
EXPR_PATH = f"{BASE_DIR}/GSE76427_expr_gene_RFS_108_clean.csv"
LABEL_PATH = f"{BASE_DIR}/GSE76427_rfs_label_108.csv"
```

然后运行：

```bash
python Modifiedcode.py
```

## 8. 当前模型结果解读

你提到当前训练准确率大约在 73% 左右。对于该项目，需要注意以下几点：

1. 样本量只有 108，测试集约 22 个样本，单个样本预测变化就会明显影响准确率。
2. 基因数超过 3 万，远大于样本量，模型非常容易过拟合。
3. 当前代码已经使用训练集内筛基因、Dropout、权重衰减、Early Stopping 和正则化 Logistic Regression 来降低过拟合风险。
4. 73% 准确率可以作为初步结果，但不建议单次划分结果作为最终结论。
5. 更推荐后续使用重复分层 K 折交叉验证，并报告 Accuracy、AUC、F1、Recall、Specificity 的均值和标准差。

## 9. 后续优化建议

可以优先考虑以下方向：

1. 将单次 train/val/test 划分改为重复分层 K 折交叉验证。
2. 增加 ROC-AUC 和 PR-AUC，更适合医学二分类任务。
3. 对 `TOPK_GENES` 做网格搜索，例如 30、50、90、120、200。
4. 对 MLP 的隐藏层维度、Dropout、学习率和 weight decay 做系统调参。
5. 比较更多传统模型，如 SVM、Random Forest、XGBoost、LightGBM。
6. 保存训练日志和随机种子，增强结果可复现性。
7. 对高频入选基因做生物学解释，例如通路富集分析、文献验证和 HCC 复发相关性讨论。
