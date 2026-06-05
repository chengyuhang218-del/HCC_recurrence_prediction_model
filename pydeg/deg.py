import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, fcluster

file_path = "/Users/chengyuhang/Desktop/TCGALIHC.tsv"
df = pd.read_csv(file_path, sep="\t", index_col=0)
gene_names = df.index.tolist()
data = torch.tensor(df.values.T, dtype=torch.float32)
n_samples, n_genes = data.shape
print(f"Samples: {n_samples}, Genes: {n_genes}")


#降维（非常重要！！！）
var = torch.var(data, dim=0)
topk = 1000
top_idx = torch.topk(var, topk).indices
data = data[:, top_idx]
gene_names = [gene_names[i] for i in top_idx]
n_genes = topk
print(f"Selected top {n_genes} variable genes")


# 注意力模型

class GeneAttention(nn.Module):
    def __init__(self, n_genes, embed_dim=8):
        super().__init__()
        self.linears = nn.ModuleList([nn.Linear(1, embed_dim) for _ in range(n_genes)])

    def forward(self, x):
        out_list = []
        for i, lin in enumerate(self.linears):
            gene_column = x[:, i:i+1]
            out_list.append(lin(gene_column))
        out = torch.stack(out_list, dim=1)  # samples x genes x dim
        out_mean = out.mean(dim=0)          # genes x dim
        return out_mean


# 模块划分

def gene_modules_from_embedding(embedding, gene_names, threshold=0.2):
    emb = embedding.detach().numpy()
    # 归一化
    norm_emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    # 余弦距离
    dist_vector = pdist(norm_emb, metric='cosine')
    # 防止负值（关键！！）
    dist_vector = np.clip(dist_vector, 0, None)
    Z = linkage(dist_vector, method='average')
    clusters = fcluster(Z, t=threshold, criterion='distance')
    modules = {}
    module_features = {}
    for idx, cluster_id in enumerate(clusters):
        modules.setdefault(cluster_id, []).append(gene_names[idx])
    for cluster_id, genes in modules.items():
        idxs = [gene_names.index(g) for g in genes]
        module_features[cluster_id] = torch.tensor(np.mean(emb[idxs], axis=0))
    return modules, module_features


# 运行
attention = GeneAttention(n_genes, embed_dim=8)
embedding = attention(data)
modules, module_features = gene_modules_from_embedding(
    embedding, gene_names, threshold=0.6
)

# 输出
print("\n=== Gene Modules ===")
for cluster_id, genes in modules.items():
    print(f"\nModule {cluster_id} ({len(genes)} genes):")
    print(genes[:10], "...")  # 只显示前10个
print("\nTotal modules:", len(modules))

# 计算模块平均表达（关键）
def compute_module_activity(data, modules, gene_names):
    module_activity = {}
    for cluster_id, genes in modules.items():
        idxs = [gene_names.index(g) for g in genes]
        # 模块平均表达（每个样本）
        module_activity[cluster_id] = data[:, idxs].mean(dim=1).numpy()
    return module_activity
module_activity = compute_module_activity(data, modules, gene_names)

# 按模块平均表达排序
module_mean_expression = {}
for cluster_id, values in module_activity.items():
    module_mean_expression[cluster_id] = np.mean(values)
# 排序（从高到低）
sorted_modules = sorted(module_mean_expression.items(), key=lambda x: x[1], reverse=True)
print("\n=== Modules sorted by average expression ===")
for cluster_id, mean_val in sorted_modules[:20]:  # 只看前20
    genes = modules[cluster_id]
    print(f"\nModule {cluster_id} ({len(genes)} genes)")
    print(f"Mean expression: {mean_val:.3f}")
    print("Genes:", genes[:10], "...")