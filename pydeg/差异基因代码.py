import torch
import numpy as np
from scipy.stats import ttest_ind

file_path = "/Users/chengyuhang/Desktop/TCGALIHC.tsv"

tumor_np = tumor_data.numpy()
normal_np = normal_data.numpy()

tumor_mean = np.mean(tumor_np, axis=0)
normal_mean = np.mean(normal_np, axis=0)

#  计算 log2 Fold Change
# 避免除0，加一个很小的数
log2fc = np.log2((tumor_mean + 1e-6) / (normal_mean + 1e-6))


#  t检验（每个基因）
p_values = []
for i in range(tumor_np.shape[1]):
    t, p = ttest_ind(tumor_np[:, i], normal_np[:, i], equal_var=False)
    p_values.append(p)

p_values = np.array(p_values)


#  筛选差异基因
# 条件：
# |log2FC| > 1  (2倍变化)
# p < 0.05
deg_mask = (np.abs(log2fc) > 1) & (p_values < 0.05)
deg_indices = np.where(deg_mask)[0]
print(f"筛选到差异基因数量: {len(deg_indices)}")


# 6️⃣ 提取差异基因表达矩阵
tumor_deg = tumor_data[:, deg_indices]
normal_deg = normal_data[:, deg_indices]


# 7️⃣ 如果你有 gene_names
deg_genes = [gene_names[i] for i in deg_indices]
print("前10个差异基因：", deg_genes[:10])