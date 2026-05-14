# HCC 复发与非复发预测项目说明

## 1. 项目简介

利用 HCC（hepatocellular carcinoma，肝细胞癌）患者的基因表达矩阵与复发标签，构建一个用于预测患者是否复发的二分类模型。整体思路是先在训练集内筛选与复发相关的基因，再分别训练 MLP （多层感知机非线性）神经网络和 ElasticNet Logistic Regression（线性模型），最后对两个模型的预测概率取平均形成集成预测。

## 2. 项目文件结构
项目地址：https://github.com/chengyuhang218-del/HCC_recurrence_prediction_model
```text
scRNAseq-HCC/
├── Modifiedcode.py                         # 深度学习代码
├── GSE76427_expr_gene_RFS_108_clean.csv    # 基因表达矩阵
├── GSE76427_rfs_label_108.csv              # 样本复发标签
├── README.md                               # 简要的项目介绍
└── PROJECT_DOCUMENTATION.md                # 本项目说明文档
```

## 3. 数据说明

### 3.1 表达矩阵

文件：`GSE76427_expr_gene_RFS_108_clean.csv`

表达矩阵的基本格式为：行：基因，列：样本 ID，数值：对应样本中该基因的表达值。
当前文件包含：样本数：108； 基因数：31,333

### 3.2 标签文件

文件：`GSE76427_rfs_label_108.csv`

标签文件包含两列：

| 字段 | 含义 |
| --- | --- |
| `sample`样本数 | 样本 ID |
| `recurrence` 复发标记 | 复发标签，`1` 表示复发，`0` 表示非复发 |

## 4. 建模任务

#####本项目要解决的问题是：给定一个 HCC 患者样本的基因表达谱，预测该患者是否会发生复发。

这是一个监督学习二分类任务：

- 输入 `X`：样本的基因表达特征。
- 输出 `y`：复发标签，`1` 为复发，`0` 为非复发。
- 目标：利用深度学习方法学习表达模式与复发风险之间的关系，并输出复发概率。

## 5. 整体流程

项目的主要思路流程可以概括为：

1. 数据预处理后，读取基因表达矩阵和复发标签。
2. 对表达矩阵和标签文件取共同样本，保证样本顺序一致。
3. 使用分层抽样划分训练集、验证集和测试集。
4. 在训练集上计算 t-score，筛选与复发差异最明显的前 90 个基因（测试发现90基因准确率更高）
5. 使用训练集拟合 `StandardScaler`，再转换验证集和测试集，避免数据泄漏。
6. 训练一个 MLP 二分类神经网络（过两遍 Linear + RELU，两层神经网络）。
7. 训练一个 ElasticNet Logistic Regression，即在线性模型原本的loss上额外加入正则化约束的模型。
8. 分别获得 MLP 和 Logistic Regression 的复发预测概率。
9. 将两个模型的概率平均，得到集成概率。
10. 在验证集上搜索最佳分类阈值。
11. 在测试集上评估 Accuracy、Precision、Recall、F1 和混淆矩阵。
12. 保存测试集预测结果、MLP embedding、筛选基因等结果。

# 从零开始做肝癌复发与非复发预测项目

从处理数据开始一步步开始吧，基于 GSE76427数据肝癌肿瘤Bulk样本、构建复发标签并使用简单模型进行复发预测。

## 一、数据来源

使用 GEO 数据集 **GSE76427**。该数据集包含 115 名患者的原发性肝细胞癌肿瘤组织（还有部分癌旁组织已筛掉），并提供 RFS/OS 等预后信息，数据网址 https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE76427 ，我已经把需要的数据上传到github项目中，无需再下载。

| 文件 | 作用 |
|---|---|
| GSE76427_series_matrix.txt.gz | 表达矩阵和样本临床信息 |
| GPL10558 HumanHT-12 V4.0 annotation | Illumina 芯片探针注释文件，用于基因 ID 转 Gene Symbol |

**说明：** GSE76427 是芯片数据，不是 RNA-seq counts。表达值出现负数是正常的，因为数据已经经过标准化。

## 二、数据处理
R 中读取下载的数据集 GSE76427，GEOquery包可直接提取文件中的临床信息和表达矩阵。
```r
library(GEOquery)

gset <- getGEO(
  filename = "/Users/chengyuhang/Desktop/GSE76427_series_matrix.txt.gz",
  AnnotGPL = FALSE,
  getGPL = FALSE)

exprSet <- exprs(gset)   # 表达矩阵：probe × sample
pdata <- pData(gset)     # 临床信息

colnames(pdata)
dim(exprSet)
```

读取后表达矩阵约为：

```text
47322 × 167
```

其中 167 个样本包括肿瘤样本和癌旁样本。

## 三、提取复发标签

pdata中关键字段用来提取复发标签：

| 字段 | 含义 |
|---|---|
| tissue:ch1 | 组织类型 |
| event_rfs:ch1 | RFS 事件，1=复发，0=未复发 |
| duryears_rfs:ch1 | 无复发生存时间，单位为年 |

```r
unique(pdata$`tissue:ch1`)
unique(pdata$`event_rfs:ch1`)
```

我们只保留肿瘤样本即可：

```r
rfs_label <- data.frame(
  sample = rownames(pdata),
  tissue = pdata$`tissue:ch1`,
  recurrence = pdata$`event_rfs:ch1`)

rfs_label <- rfs_label[
  rfs_label$tissue == "primary hepatocellular carcinoma tumor",]

rfs_label$tissue <- NULL

rfs_label <- rfs_label[!is.na(rfs_label$recurrence), ]
rfs_label <- rfs_label[rfs_label$recurrence != "NA", ]
rfs_label$recurrence <- as.numeric(rfs_label$recurrence)

table(rfs_label$recurrence)
dim(rfs_label)
```

输出：

```text
0  1
60 48
```

**结果：** 最终得到 108 个有明确复发标签的真实肝癌肿瘤样本（115个肿瘤样本，但有些无复发标签）。

## 四、表达矩阵只保留肿瘤样本

```r
tumor_samples <- rfs_label$sample
expr_tumor <- exprSet[, tumor_samples]

dim(expr_tumor)
dim(rfs_label)
all(colnames(expr_tumor) == rfs_label$sample)
```
输出：

```text
expr_tumor: 47322 × 108 #基因和样本
rfs_label: 108 × 2 #样本和标签
```

## 五、探针 ID 转基因名

表达矩阵中行名类似 `ILMN_1651199`，这是 Illumina probe ID，不是基因名。需要用 GPL10558 注释表转换。

```r
anno <- read.delim(
  "/Users/chengyuhang/Desktop/GPL10558_HumanHT-12_V4_0_R1_15002873_B.txt",
  skip = 8,
  header = TRUE,
  sep = "\t",
  quote = "",
  fill = TRUE,
  check.names = FALSE)

head(anno[, c("Probe_Id", "Symbol")])
```

转换：

```r
probe2gene <- anno[, c("Probe_Id", "Symbol")]
probe2gene <- probe2gene[match(rownames(expr_tumor), probe2gene$Probe_Id),]
rownames(expr_tumor) <- probe2gene$Symbol
head(rownames(expr_tumor))
```

## 六、删除空基因名和重复基因

删除空基因名

```r
expr_tumor <- expr_tumor[rownames(expr_tumor) != "" & !is.na(rownames(expr_tumor)),]
```

重复基因保留方差最大的 probe

```r
probe_var <- apply(expr_tumor, 1, var)
expr_tumor_ordered <- expr_tumor[order(probe_var, decreasing = TRUE), ]
expr_final <- expr_tumor_ordered[!duplicated(rownames(expr_tumor_ordered)), ]

sum(duplicated(rownames(expr_final)))
dim(expr_final)
```

**说明：** 同一个基因可能对应多个 probe，保留方差最大的 probe 是芯片数据中常用的处理方式，就是在多个样本中保留表达变化最明显的那个。

## 七、清理 NA / Inf

如果表达矩阵有 NA 或 Inf，Python 模型会报错，这个一定要清理。

```r
expr_final <- as.matrix(expr_final)
mode(expr_final) <- "numeric"

cat("NA count before clean:", sum(is.na(expr_final)), "\n")
cat("Inf count before clean:", sum(is.infinite(expr_final)), "\n")

expr_final[is.infinite(expr_final)] <- NA
expr_final <- expr_final[complete.cases(expr_final), ]

cat("NA count after clean:", sum(is.na(expr_final)), "\n")
cat("Inf count after clean:", sum(is.infinite(expr_final)), "\n")

dim(expr_final)
```

## 八、保存最终数据

```r
# 这是处理好的用于训练最终数据，也在github有存
write.csv(expr_final,"/Users/chengyuhang/Desktop/GSE76427_expr_gene_RFS_108_clean.csv")
# 这是标签
write.csv(rfs_label,"/Users/chengyuhang/Desktop/GSE76427_rfs_label_108.csv",row.names = FALSE)
```

| 文件 | 说明 |
|---|---|
| GSE76427_expr_gene_RFS_108_clean.csv | 清理后的表达矩阵，gene × sample |
| GSE76427_rfs_label_108.csv | 复发标签，sample × recurrence |


## 数据处理遇见问题

##### 1. 表达矩阵中为什么有负值？
GSE76427 是标准化后的芯片数据，负值表示相对表达较低，不是错误。

##### 3. 为什么要删除空基因名，为什么 Loss 会变成 nan？
通常是表达矩阵中存在 NA、Inf 或非数字值，需要先在 R 中清理，不然 y=w1​x1​+w2​x2 计算会有na，模型会炸。


## 九、开始训练模型

上面完成了从 GSE76427 数据读取、真实复发标签构建、表达矩阵清理、探针注释，最终得到数据：108 个真实肝癌肿瘤样本，其中 48 个复发，60 个非复发，用于构建复发预测模型。

下面按 `Modifiedcode.py` 的实际执行顺序说明每个模块。我们来一步一步跑这个基础模型。
### 模块 0：参数设置

该模块设置输入数据路径、模型超参数和随机种子。`TOPK_GENES` 控制筛选多少个复发相关基因，`EPOCHS` 和 `LR` 控制 MLP 训练过程，`RANDOM_SEED` 用于提高实验可复现性。

需要注意的是，当前路径仍我的路径，你需要在Github（https://github.com/chengyuhang218-del/scRNAseq-HCC) 下载我的原始数据导入你自己的文件夹路径,如果在本仓库中运行，建议改为仓库内路径或相对路径。

```python
EXPR_PATH = "/Users/chengyuhang/Desktop/GSE76427_expr_gene_RFS_108_clean.csv"  # 基因表达矩阵文件路径，行为基因、列为样本
LABEL_PATH = "/Users/chengyuhang/Desktop/GSE76427_rfs_label_108.csv"  # 样本复发标签文件路径，包含 sample 和 recurrence 两列

TOPK_GENES = 90  # 设置 t-score 筛选的基因数量，只保留差异最明显的前 90 个基因
EPOCHS = 260  # 设置 MLP 最大训练轮数，实际可能因 early stopping 提前结束
LR = 5e-4  # 设置 Adam 优化器学习率，控制每次参数更新步长，Adam 是最经典优化器，自动调整每个参数更新速度
RANDOM_SEED = 42  # 设置随机种子，使数据划分和模型初始化更可复现

np.random.seed(RANDOM_SEED)  # 固定 NumPy 随机数种子，保证 NumPy 相关随机过程可复现
torch.manual_seed(RANDOM_SEED)  # 固定 PyTorch 随机数种子，保证模型初始化等过程尽量可复现
```

### 模块 1：读取数据

该模块读取基因表达矩阵和复发标签，并统一标签列名。表达矩阵中行为基因、列为样本；标签文件中 `recurrence` 被转换成整数型标签。

随后代码取表达矩阵和标签文件共有的样本，保证 `X`、`y` 和 `sample_names` 的顺序严格一致。这一步非常关键，否则模型可能会把某个样本的表达数据对应到另一个样本的标签。

```python
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

```python
print("\nSplitting train / val / test...")  # 打印提示信息，表示开始划分训练集、验证集和测试集

X_trainval, X_test, y_trainval, y_test, sample_trainval, sample_test = train_test_split(  # 开始一个多行函数调用或对象创建
    X,  # 传入完整特征矩阵作为待划分数据
    y,  # 传入标签数组作为待划分标签
    sample_names,  # 同时划分样本名，方便后续追踪测试集样本 ID
    test_size=0.2,  # 设置测试集比例为 20%
    random_state=RANDOM_SEED,  # 使用固定随机种子，使数据划分可复现
    stratify=y  # 按 y 的类别比例进行分层抽样，保持复发/非复发比例
)

X_train, X_val, y_train, y_val, sample_train, sample_val = train_test_split(  # 开始一个多行函数调用或对象创建
    X_trainval,  # 传入训练验证合并集特征，准备继续划分训练集和验证集
    y_trainval,  # 传入训练验证合并集标签
    sample_trainval,  # 传入训练验证合并集样本名
    test_size=0.25,  # 从训练验证合并集中划出 25% 作为验证集，整体约等于 20%
    random_state=RANDOM_SEED,  # 使用固定随机种子，使数据划分可复现
    stratify=y_trainval  # 按训练验证合并集标签比例继续分层划分
)

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

该模块定义 `select_genes_by_tscore` 函数。函数分别计算复发组和非复发组每个基因的均值、方差和标准误，再计算 t-score。按 `abs_t_score` 从大到小排序，选择前 `TOPK_GENES` 个基因（就是我之前设置的90）。这里特意只使用训练集做基因筛选，避免验证集和测试集信息泄漏到特征选择步骤中。

```python
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
    })

    return top_idx, selected_info  # 返回筛选基因索引和对应统计信息表


print("\nSelecting recurrence-related genes...")  # 打印提示信息，表示开始筛选复发相关基因

top_idx, selected_gene_info = select_genes_by_tscore(  # 开始一个多行函数调用或对象创建
    X_train,  # 传入训练集表达矩阵用于基因筛选
    y_train,  # 传入训练集标签用于区分复发和非复发组
    gene_names_all,  # 传入全部基因名称，用于把索引映射回基因名
    TOPK_GENES  # 传入需要保留的基因数量
)

selected_genes = gene_names_all[top_idx]  # 根据筛选索引得到最终入选基因名称

X_train = X_train[:, top_idx]  # 训练集只保留筛选出的基因特征
X_val = X_val[:, top_idx]  # 验证集使用同一批筛选基因，保持特征空间一致
X_test = X_test[:, top_idx]  # 测试集使用同一批筛选基因，避免特征不一致

print(f"Selected genes: {len(selected_genes)}")  # 打印最终筛选出的基因数量
```

### 模块 4：标准化

之前计算t-score是单个基因维度看复发和非复发变化幅度，基因表达差距很大（原始数据）没关系，下面再做标准化即可，标准化前后长这样，本质是把每个基因减去平均值再除以标准差（数据波动大标准差大，数据波动小标准差小），0=平均表达；>0高于平均；<0低于平均。

```python
             LOC440864   LOC648638  ...      TTTY15     PLEKHN1
GSM2011306   91.290001  339.559998  ...  115.360001  155.869995
GSM2011327   92.820000  875.549988  ...  112.860001  149.929993
GSM2011342  102.000000  514.010010  ...  111.419998  139.330002
GSM2011332  117.160004  661.609985  ...  129.619995  129.149994
GSM2011335   99.269997  378.730011  ...  105.470001  160.399994
```
```python
             LOC440864  LOC648638   ...       TTTY15   PLEKHN1
GSM2011306   -1.463980  -1.119647   ...    0.413690   0.473567
GSM2011327   -1.213936   1.123463   ...    0.173583   0.086652
GSM2011342    0.286331  -0.389576   ...    0.035281  -0.603801
GSM2011332    2.763896   0.228127   ...    1.783262  -1.266899
GSM2011335   -0.159828  -0.955721   ...   -0.536175   0.768639
```

该模块使用 `StandardScaler` 对筛选后的表达矩阵进行标准化。标准化的均值和方差只从训练集学习，然后应用到验证集和测试集，这样可以让模型训练更稳定，同时避免把验证集或测试集的统计信息提前暴露给模型。标准化就是把表达拉到同一尺度，求平均和标准差，将每个基因在不同样本表达带入标准化公式，得到各个基因可以同一尺度比较的表达矩阵。


```python
print("\nStandardizing data...")  # 打印提示信息，表示开始标准化数据

scaler = StandardScaler()  # 创建标准化器，用于把每个基因特征转为均值 0、方差 1

X_train_scaled = scaler.fit_transform(X_train)  # 在训练集上拟合标准化参数，并转换训练集
X_val_scaled = scaler.transform(X_val)  # 使用训练集标准化参数转换验证集，避免数据泄漏
X_test_scaled = scaler.transform(X_test)  # 使用训练集标准化参数转换测试集，避免数据泄漏
```

### 模块 5：MLP 模型定义

该模块定义 `MLPClassifier`。模型训练时，所有训练样本会一起输入 MLP 网络（多层感知机），按顺序每个病人的 90 个复发相关基因表达值依次经过第一层全连接层，由模型自动学习不同基因之间的组合关系，并将原始基因表达逐渐压缩为更抽象的隐藏特征（低维 embedding）；随后再经过第二层特征提取，最终形成 16 维 embedding，用于表示该病人的低维生物学状态，最后分类层会输出每个病人的复发风险 logit（16维 embedding压成的1个数），并通过 sigmoid 函数转换为复发概率。

模型得到所有样本的预测结果后，会与真实复发标签（复发=1，非复发=0）进行比较，计算每个样本的预测误差。随后，BCEWithLogitsLoss 会将所有训练样本的误差进行平均，得到当前模型的整体损失（loss），该 loss 用于衡量模型当前整体预测效果的好坏，如果模型对复发样本预测错误，loss 会增大；预测越准确，loss 越小。

接下来，模型通过反向传播（backpropagation）将 loss 从输出层逐层传回网络内部，自动计算每个参数对最终误差的贡献大小，并据此调整各层神经元对应的权重参数例如，如果某些基因组合能够更好地区分复发与非复发，相关权重会逐渐增大；而对预测贡献较小甚至造成错误预测的基因，其权重会被减弱。完成一次参数更新后，所有训练样本会再次进入模型重新进行预测，并重新计算新的平均 loss，如此不断循环迭代，使模型逐渐学习到与肿瘤复发最相关的隐藏表达模式，最终提高对复发与非复发样本的区分能力。

在参数更新过程中，模型还会受到正则化项的约束。正则化公式 Loss=Lpred​+λ∑∣w∣ ，也就是某个基因的权重大了之后，在计算loss时会加大损失值，使得违背了降低损失的目的，从而不会过度加大某个基因的权重，防止过拟合，有用的基因权重不会太高，会牵制；没用的基因给高权重和低权重其实损失不会变化很大，但正则化损失权重下降后，损失能下降，所以不断降低没用的基因权重，这样既能够减少模型对噪声基因的过度依赖，又能够降低小样本高维数据中的过拟合风险，从而提高模型的泛化能力。

BatchNorm1d 原理是每个样本在各个神经元输出差异巨大，标准化拉到同一标准，ReLU(x)=max(0,X)，Dropout为随机关闭神经元，防止模型过度依赖某几个神经元防止过拟合，Linear为简单线性函数。

```python
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
        )  

        self.classifier = nn.Linear(16, 1)  # 定义输出层，将 16 维 embedding 映射为单个二分类 logit

    def forward(self, x):  # 定义前向传播过程，描述输入如何经过网络得到输出
        embedding = self.feature_extractor(x)  # 通过特征提取器得到样本的 16 维 embedding
        logits = self.classifier(embedding).squeeze(1)  # 通过分类层得到 logit，并压缩为一维向量，方便后面计算损失
        return logits, embedding  # 同时返回分类 logit 和中间层 embedding，便于训练和后续分析


model = MLPClassifier(input_dim=TOPK_GENES)  # 实例化 MLP 模型，输入维度等于筛选后的基因数量
```
将数据转变为tensor格式（PyTorch 神经网络处理格式为 tensor），并创建优化器和损失函数。

使用 `BCEWithLogitsLoss` （二分类任务的损失函数计算模块）训练 MLP。由于类别数量不完全均衡，`pos_weight` 会提高正类样本在损失函数中的权重。

其中假如不复发样本多，就算全预测不复发预测正确率也不低，所以要对复发样本进行额外加权，代码中使用“非复发样本数 ÷ 复发样本数”计算正类权重，并传入 BCEWithLogitsLoss，使模型在训练过程中对复发样本预测错误给予更大的 loss 惩罚，从而提高模型对少数类复发样本的学习能力，避免模型仅偏向多数类而忽视复发风险。
```python
x_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)  # 将训练集特征转换为 PyTorch float32 张量
y_train_t = torch.tensor(y_train, dtype=torch.float32)  # 将训练集标签转换为 float32 张量，以匹配 BCE 损失函数要求

x_val_t = torch.tensor(X_val_scaled, dtype=torch.float32)  # 将验证集特征转换为 PyTorch 张量
y_val_t = torch.tensor(y_val, dtype=torch.float32)  # 将验证集标签转换为 PyTorch 张量

x_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)  # 将测试集特征转换为 PyTorch 张量，用于最终预测

pos_weight = torch.tensor(  # 开始定义正类权重，用于处理复发/非复发样本不均衡
    [(y_train == 0).sum() / (y_train == 1).sum()],  # 用负类数量除以正类数量，得到正类在损失中的加权系数
    dtype=torch.float32  # 指定权重张量类型为 float32
)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)  # 定义二分类损失函数，内部会把 logit 转换为概率并计算 BCE，Binary Cross Entropy：二分类交叉熵用于：比较预测结果vs真实标签，它内部会自动：logit->sigmoid->概率->计算误差

optimizer = torch.optim.Adam(  # 创建 Adam 优化器，用于更新 MLP 参数
    model.parameters(),  # 把模型所有可训练参数交给优化器,所有Linear层权重都会被更新
    lr=LR,  # 设置优化器学习率
    weight_decay=1e-3  # 设置 L2 权重衰减，帮助降低过拟合
)

print(f"\nPositive class weight: {pos_weight.item():.4f}")  # 打印正类权重，便于确认类别加权强度
```

### 模块 6：MLP 训练与 Early Stopping

训练过程中，代码监控验证集 loss。如果验证集 loss 连续 `patience=20` 个 epoch 没有下降，就提前停止训练，并恢复验证集表现最好的模型参数，真正最好的模型：往往不是最后一轮，而是验证集效果最好那一轮。每训练一轮（epoch）就用当前模型去验证集测试一次，然后：记录对应的验证集 loss，验证集验证集的损失最低才可以。

```python
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

该模块训练一个 ElasticNet Logistic Regression。ElasticNet 同时包含 L1 和 L2 正则，适合基因表达这类高维、小样本数据。L2 正则化不让某个基因权重过大，防止过拟合过度依赖一个基因，L1 它会倾向：直接把部分基因权重压成0，把无关的基因去掉。也就是某个基因的权重大了之后，在计算loss时会加大损失值，从而降低损失的幅度没那么大，起到牵制作用防止过拟合，因此，模型只有在“该基因确实能够显著提高预测能力”时，才会保留较大的权重；如果权重过大带来的预测收益不足以抵消正则化惩罚，总 loss 反而会上升，优化器就会倾向于减小该基因权重。没用的基因给高权重和低权重其实损失不会变化很大（表达变化不定），但正则化损失权重下降后，损失能下降，所以不断降低没用的基因权重（这里再重复一下，我正则化理解了挺久）


`class_weight="balanced"` 用于处理类别不均衡，`solver="saga"` 是 scikit-learn 中支持 ElasticNet Logistic Regression 的优化器。

```python
print("\nTraining Logistic Regression...")  # 打印提示信息，表示开始训练逻辑回归模型

logreg = LogisticRegression(  # 创建逻辑回归分类器
    penalty="elasticnet",  # 使用 ElasticNet 正则化，同时结合 L1 和 L2 约束
    l1_ratio=0.5,  # 设置 L1 与 L2 的混合比例，0.5 表示二者权重均衡
    C=0.5,  # 设置正则化强度的倒数，数值越小正则越强
    class_weight="balanced",  # 按类别频率自动平衡样本权重，缓解类别不均衡
    solver="saga",  # 使用 saga 优化器，因为它支持 ElasticNet 正则
    max_iter=3000,  # 设置最大迭代次数，保证模型有足够机会收敛（更新一次权重=一次迭代）
    random_state=RANDOM_SEED  # 设置随机种子，提高逻辑回归训练可复现性
)
logreg.fit(X_train_scaled, y_train)  # 在标准化训练集上拟合逻辑回归模型

print("Logistic Regression finished.")  # 打印提示信息，表示逻辑回归训练完成
```

### 模块 8：MLP 与 Logistic Regression 集成预测

该模块分别计算 MLP 和 Logistic Regression 在验证集、测试集上的复发概率。MLP 输出的是 logit，因此先用 `torch.sigmoid` 转换为概率，所以这里就把测试集跑了，用的是验证集 loss 最低时保存下来的那组最佳参数。

最终集成方式是简单平均：`(mlp_prob + logreg_prob) / 2`。这种方法直观、稳定，可以降低单个模型在小样本数据上的波动。

```python
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

评价标准使用 F1-score寻找最佳分类阈值，因为它同时考虑 Precision 和 Recall，适合复发预测这种需要兼顾误报和漏报的任务，Precision 和 Recall 都高F1 才会高。

```python
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

其中 TP（模型预测复发实际上也复发）、FP（模型预测复发实际上没复发）、FN（模型预测不复发实际上复发）、TN（模型预测不复发实际上也不复发） 可以帮助进一步理解模型错误类型：模型是更容易漏掉复发样本，还是更容易把非复发样本误判为复发。

```python
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
print("\nSaving prediction result...")  # 打印提示信息，表示开始保存预测结果

pred_df = pd.DataFrame({  # 创建预测结果表格
    "sample_id": sample_test,  # 保存测试集样本 ID
    "true_label": y_test,  # 保存测试集真实标签
    "pred_label": test_pred,  # 保存测试集预测标签
    "ensemble_probability": test_prob,  # 保存集成模型预测复发概率
    "mlp_probability": mlp_test_prob,  # 保存 MLP 单模型预测复发概率
    "logistic_probability": logreg_test_prob  # 保存逻辑回归单模型预测复发概率
})

pred_output_path = "/Users/chengyuhang/Desktop/ensemble_prediction_result.csv"  # 设置测试集预测结果输出路径
pred_df.to_csv(pred_output_path, index=False)  # 将预测结果保存为 CSV，不额外保存行索引

print(f"Prediction result saved to: {pred_output_path}")  # 打印预测结果保存位置
```

### 模块 12：保存样本 embedding

该模块保存 MLP 在测试集上提取到的中间层 embedding。

这些 embedding 可以看作模型学习到的低维样本表示，后续可以用于 PCA、t-SNE、UMAP 可视化，观察复发和非复发样本是否在特征空间中出现分离趋势（所以后面可以补模块，写一个完整的python包）

```python
print("\nSaving sample embedding...")  # 打印提示信息，表示开始保存样本 embedding

embedding_df = pd.DataFrame(  # 创建 embedding 表格，用于保存 MLP 中间层样本表示
    mlp_test_embedding.detach().numpy(),  # 从计算图中分离测试集 embedding，并转换为 NumPy 数组
    index=sample_test  # 使用测试集样本 ID 作为 embedding 表格行索引
) 

embedding_output_path = "/Users/chengyuhang/Desktop/ensemble_mlp_sample_embedding.csv"  # 设置 MLP 样本 embedding 输出路径
embedding_df.to_csv(embedding_output_path)  # 将测试集 embedding 保存为 CSV

print(f"Sample embedding saved to: {embedding_output_path}")  # 打印 embedding 保存位置
```

### 模块 13：保存筛选基因

该模块保存 t-score 筛选出的基因列表及其统计信息，包括 t-score、绝对 t-score、复发组均值和非复发组均值。

该文件可用于后续生物学解释，例如检查这些基因是否与 HCC 复发相关。

```python
print("\nSaving selected genes...")  # 打印提示信息，表示开始保存筛选基因

gene_output_path = "/Users/chengyuhang/Desktop/ensemble_selected_genes.csv"  # 设置筛选基因统计表输出路径
selected_gene_info.to_csv(gene_output_path, index=False)  # 将筛选基因及 t-score 信息保存为 CSV

print(f"Selected genes saved to: {gene_output_path}")  # 打印筛选基因文件保存位置
```

### 模块 14：保存综合基因重要性

该模块分别从 MLP 第一层权重和 Logistic Regression 系数中估计基因重要性，然后取平均作为综合重要性分数。

就是对每个基因在 MLP 第一层连接到 32 个隐藏神经元的权重取绝对值并求平均，作为该基因在神经网络中的整体影响程度；同时提取 Logistic Regression 的线性系数绝对值作为线性模型的重要性。标准化后，随后对两个模型的重要性分数再取平均，得到综合基因重要性排序用于筛选对复发预测贡献较大的候选基因。不过需要注意，神经网络权重和 Logistic 系数的可解释性有限，最终要结合生物学证据来判断关键基因。

```python
print("\nSaving gene importance...")  # 开始保存基因重要性

mlp_weight = model.feature_extractor[0].weight.detach().numpy()  # 提取 MLP 第一层权重
mlp_importance = np.mean(np.abs(mlp_weight), axis=0)  # 计算 MLP 基因重要性

logreg_importance = np.abs(logreg.coef_[0])  # 提取逻辑回归基因权重

importance_df = pd.DataFrame({
    "gene": selected_genes,  # 基因名称
    "mlp_importance": mlp_importance,  # MLP 重要性
    "logistic_importance": logreg_importance  # Logistic 重要性
})

importance_df["mlp_importance_norm"] = (
    importance_df["mlp_importance"] -
    importance_df["mlp_importance"].min()
) / (
    importance_df["mlp_importance"].max() -
    importance_df["mlp_importance"].min()
)  # MLP 重要性归一化到 0~1

importance_df["logistic_importance_norm"] = (
    importance_df["logistic_importance"] -
    importance_df["logistic_importance"].min()
) / (
    importance_df["logistic_importance"].max() -
    importance_df["logistic_importance"].min()
)  # Logistic 重要性归一化到 0~1

importance_df["mean_importance"] = importance_df[
    ["mlp_importance_norm", "logistic_importance_norm"]
].mean(axis=1)  # 计算综合基因重要性

importance_df = importance_df.sort_values(
    by="mean_importance",
    ascending=False
)  # 按综合重要性降序排序

importance_output_path = "/Users/chengyuhang/Desktop/ensemble_gene_importance.csv"  # 输出路径

importance_df.to_csv(importance_output_path, index=False)  # 保存 CSV 文件

print(f"Gene importance saved to: {importance_output_path}")  # 打印保存位置

print("\nAll done.")  # 全部流程结束
```

## 十、如何运行（我准备写成包，之后上传到pip，后续在补充内容）

建议先安装依赖：

```bash
pip install numpy pandas torch scikit-learn tqdm
```


## 十一、结果与学习心得

我自己训练准确率大约在 72.7% 左右。对于该项目，主要有以下几点：

1. 样本量只有 108，测试集约 22 个样本，单个样本预测变化就会明显影响准确率，而且我找样本患者治疗预后信息很少，回访复发信息感觉少之又少。
2. 基因数超过 3 万，远大于样本量，模型非常容易过拟合。但当前代码已经使用训练集内筛基因、Dropout、Early Stopping 和正则化 Logistic Regression 来降低过拟合风险。

跑完这个流程，深度学习模型应用于表达矩阵的过程，很多模块、流程、环节、概念也都可以理清清楚了，感觉可以进行更复杂的工作了。

## 十二、优化建议

现在的模型很简单，但师兄鼓励我说知道不同的数据要用不同的模型已经进步很大了，近期抓紧把之前的师兄给的文献读完吧，感觉浪费了很多时间。

2026/5/14


