# 从零开始做肝癌复发与非复发预测项目

基于 GSE76427，提取真实肝癌肿瘤样本、构建 RFS 复发标签，并使用 Attention 模型进行复发预测。

## 一、项目目标

本项目使用真实肝细胞癌患者表达谱数据，构建一个二分类模型，用于预测患者术后是否复发。

| 步骤 | 内容 |
|---|---|
| 1 | 下载 GSE76427 表达矩阵和临床信息 |
| 2 | 提取肿瘤样本和 RFS 复发标签 |
| 3 | 探针 ID 转换为基因名 |
| 4 | 删除空基因名、处理重复基因、清理 NA/Inf |
| 5 | 用 Python 构建 Attention 二分类模型 |

## 二、数据来源

使用 GEO 数据集 **GSE76427**。该数据集包含 115 名肝细胞癌患者的原发肿瘤组织和部分癌旁组织表达谱，并提供 RFS/OS 等预后信息。

| 文件 | 作用 |
|---|---|
| GSE76427_series_matrix.txt.gz | 表达矩阵和样本临床信息 |
| GPL10558 HumanHT-12 V4.0 annotation | Illumina 芯片探针注释文件，用于 Probe ID 转 Gene Symbol |

> **说明：** GSE76427 是芯片数据，不是 RNA-seq counts。表达值出现负数是正常的，因为数据已经经过标准化。

## 三、R 读取 GSE76427

```r
library(GEOquery)

gset <- getGEO(
  filename = "/Users/chengyuhang/Desktop/GSE76427_series_matrix.txt.gz",
  AnnotGPL = FALSE,
  getGPL = FALSE
)

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

## 四、提取复发标签

关键字段：

| 字段 | 含义 |
|---|---|
| tissue:ch1 | 组织类型 |
| event_rfs:ch1 | RFS 事件，1=复发，0=未复发 |
| duryears_rfs:ch1 | 无复发生存时间，单位为年 |

```r
unique(pdata$`tissue:ch1`)
unique(pdata$`event_rfs:ch1`)
```

只保留肿瘤样本：

```r
rfs_label <- data.frame(
  sample = rownames(pdata),
  tissue = pdata$`tissue:ch1`,
  recurrence = pdata$`event_rfs:ch1`
)

rfs_label <- rfs_label[
  rfs_label$tissue == "primary hepatocellular carcinoma tumor",
]

rfs_label$tissue <- NULL

rfs_label <- rfs_label[!is.na(rfs_label$recurrence), ]
rfs_label <- rfs_label[rfs_label$recurrence != "NA", ]
rfs_label$recurrence <- as.numeric(rfs_label$recurrence)

table(rfs_label$recurrence)
dim(rfs_label)
```

结果：

```text
0  1
60 48
```

> **结果：** 最终得到 108 个有明确复发标签的真实肝癌肿瘤样本。

## 五、表达矩阵只保留肿瘤样本

```r
tumor_samples <- rfs_label$sample
expr_tumor <- exprSet[, tumor_samples]

dim(expr_tumor)
dim(rfs_label)
all(colnames(expr_tumor) == rfs_label$sample)
```

理想结果：

```text
expr_tumor: 47322 × 108
rfs_label: 108 × 2
```

## 六、探针 ID 转基因名

表达矩阵中行名类似 `ILMN_1651199`，这是 Illumina probe ID，不是基因名。需要用 GPL10558 注释表转换。

```r
anno <- read.delim(
  "/Users/chengyuhang/Desktop/GPL10558_HumanHT-12_V4_0_R1_15002873_B.txt",
  skip = 8,
  header = TRUE,
  sep = "\t",
  quote = "",
  fill = TRUE,
  check.names = FALSE
)

head(anno[, c("Probe_Id", "Symbol")])
```

转换：

```r
probe2gene <- anno[, c("Probe_Id", "Symbol")]

probe2gene <- probe2gene[
  match(rownames(expr_tumor), probe2gene$Probe_Id),
]

rownames(expr_tumor) <- probe2gene$Symbol
head(rownames(expr_tumor))
```

## 七、删除空基因名和重复基因

### 1. 删除空基因名

```r
expr_tumor <- expr_tumor[
  rownames(expr_tumor) != "" & !is.na(rownames(expr_tumor)),
]
```

### 2. 重复基因保留方差最大的 probe

```r
probe_var <- apply(expr_tumor, 1, var)
expr_tumor_ordered <- expr_tumor[order(probe_var, decreasing = TRUE), ]
expr_final <- expr_tumor_ordered[!duplicated(rownames(expr_tumor_ordered)), ]

sum(duplicated(rownames(expr_final)))
dim(expr_final)
```

> **说明：** 同一个基因可能对应多个 probe，保留方差最大的 probe 是芯片数据中常用的处理方式。

## 八、清理 NA / Inf

如果表达矩阵有 NA 或 Inf，Python 模型可能出现 `Loss = nan`。

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

## 九、保存最终数据

```r
write.csv(
  expr_final,
  "/Users/chengyuhang/Desktop/GSE76427_expr_gene_RFS_108_clean.csv"
)

write.csv(
  rfs_label,
  "/Users/chengyuhang/Desktop/GSE76427_rfs_label_108.csv",
  row.names = FALSE
)
```

| 文件 | 说明 |
|---|---|
| GSE76427_expr_gene_RFS_108_clean.csv | 清理后的表达矩阵，gene × sample |
| GSE76427_rfs_label_108.csv | 复发标签，sample × recurrence |

## 十、Python 读取建模数据

```python
EXPR_PATH = "/Users/chengyuhang/Desktop/GSE76427_expr_gene_RFS_108_clean.csv"
LABEL_PATH = "/Users/chengyuhang/Desktop/GSE76427_rfs_label_108.csv"

import pandas as pd
import torch

df = pd.read_csv(EXPR_PATH, index_col=0)
label_df = pd.read_csv(LABEL_PATH)

label_df = label_df.rename(columns={
    "sample": "sample_id",
    "recurrence": "label"
})
label_df["label"] = label_df["label"].astype(float)

common_samples = [s for s in df.columns if s in set(label_df["sample_id"])]
df = df[common_samples]
label_df = label_df.set_index("sample_id").loc[common_samples].reset_index()

gene_names = df.index.tolist()
sample_names = df.columns.tolist()

data = torch.tensor(df.values.T, dtype=torch.float32)  # samples × genes
labels = torch.tensor(label_df["label"].values, dtype=torch.float32)

print("Samples with labels:", data.shape[0])
print("Genes:", data.shape[1])
print("Recurrence samples:", int(labels.sum().item()))
print("Non-recurrence samples:", int((labels == 0).sum().item()))
```

## 十一、筛选高变基因

```python
def select_variable_genes_by_scaled_variance(data, gene_names, topk=1000):
    topk = min(topk, data.shape[1])
    data_for_var = data.clone()

    data_min = data_for_var.min(dim=0, keepdim=True).values
    data_max = data_for_var.max(dim=0, keepdim=True).values
    denom = data_max - data_min
    denom[denom == 0] = 1

    data_scaled = 2 * (data_for_var - data_min) / denom - 1
    var = torch.var(data_scaled, dim=0)
    top_idx = torch.topk(var, topk).indices

    data_selected = data[:, top_idx]
    gene_names_selected = [gene_names[i] for i in top_idx.tolist()]
    return data_selected, gene_names_selected

data, gene_names = select_variable_genes_by_scaled_variance(
    data=data,
    gene_names=gene_names,
    topk=1000
)
```

## 十二、Attention 模型逻辑

| 模块 | 作用 |
|---|---|
| Linear(1 → 32) | 把每个基因表达值映射成 32 维向量 |
| MultiheadAttention | 学习基因之间的关系 |
| Mean pooling | 得到每个样本的整体 embedding |
| Classifier | 输出复发概率 |

模型输入是：

```text
samples × genes
```

输出是每个样本的复发概率。

## 十三、模型输出文件解释

| 输出文件 | 含义 |
|---|---|
| attention_prediction_result.csv | 测试集样本真实标签、预测标签和复发概率 |
| sample_attention_embedding.csv | 样本经过 Attention 后的低维特征 |
| gene_attention_matrix_test.csv | 测试集中基因之间的平均 attention 权重 |
| selected_attention_genes.csv | 进入模型的高变基因列表 |

## 十四、评价指标解释

| 指标 | 含义 |
|---|---|
| Accuracy | 所有样本中预测正确的比例 |
| Precision | 预测为复发的样本中，真正复发的比例 |
| Recall | 真正复发的样本中，被模型找回来的比例 |
| F1-score | Precision 和 Recall 的综合指标 |

> **提醒：** 复发预测中 Recall 很重要，因为它代表真正复发患者被模型找出来的比例。

## 十五、常见问题

### 1. 表达矩阵中有负值正常吗？

正常。GSE76427 是标准化后的芯片数据，负值表示相对表达较低，不是错误。

### 2. 为什么要删除癌旁组织？

本项目预测肿瘤患者术后复发，因此输入应为肿瘤组织表达谱。

### 3. 为什么要删除空基因名？

空基因名无法做 GO/KEGG，也无法解释 Attention 结果。

### 4. 为什么 Loss 会变成 nan？

通常是表达矩阵中存在 NA、Inf 或非数字值，需要先在 R 中清理。

## 十六、项目总结

本项目完成了从 GSE76427 数据读取、真实复发标签构建、表达矩阵清理、探针注释，到 Attention 模型预测复发的完整流程。

> **最终数据：** 108 个真实肝癌肿瘤样本，其中 48 个复发，60 个非复发，可用于 Attention 复发预测模型。
