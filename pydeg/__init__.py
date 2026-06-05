import torch
import torch.nn as nn
import numpy as np
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, fcluster

# =========================
# 数据生成
# =========================
n_samples = 100
n_genes = 10
torch.manual_seed(42)

# 模拟正常/肿瘤数据，基因间有微小差异
normal_data = torch.randn(n_samples, n_genes) + torch.linspace(0, 5, n_genes)
tumor_data = torch.randn(n_samples, n_genes) + torch.linspace(2, 7, n_genes)
gene_names = [f'gene{i + 1}' for i in range(n_genes)]


# =========================
# 注意力机制生成基因 embedding
# =========================
class GeneAttention(nn.Module):
    def __init__(self, n_genes, embed_dim=2):
        super().__init__()
        # 每个基因独立映射
        self.linears = nn.ModuleList([nn.Linear(1, embed_dim) for _ in range(n_genes)])

    def forward(self, x):
        # x: samples x genes
        out_list = []
        for i, lin in enumerate(self.linears):
            gene_column = x[:, i:i + 1]  # samples x 1
            out_list.append(lin(gene_column))
        out = torch.stack(out_list, dim=1)  # samples x genes x embed_dim
        out_mean = out.mean(dim=0)  # genes x embed_dim
        return out_mean


# =========================
# 模块划分函数
# =========================
def gene_modules_from_embedding(embedding, gene_names, threshold=0.5):
    """
    embedding: genes x embed_dim
    threshold: 聚类距离阈值
    """
    emb = embedding.detach().numpy()
    # 归一化
    norm_emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    # 计算余弦距离
    dist_vector = pdist(norm_emb, metric='cosine')
    # 层次聚类
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


# =========================
# 运行注意力 + 聚类
# =========================
attention = GeneAttention(n_genes, embed_dim=2)

for data_name, data_matrix in [('Normal', normal_data), ('Tumor', tumor_data)]:
    embedding = attention(data_matrix)
    modules, module_features = gene_modules_from_embedding(embedding, gene_names, threshold=0.3)

    print(f"=== {data_name} Data ===")
    print("Gene embedding:\n", embedding.detach().numpy())
    for cluster_id, genes in modules.items():
        print(f"Module {cluster_id}: {genes}, embedding={module_features[cluster_id].detach().numpy()}")
    print("\n")