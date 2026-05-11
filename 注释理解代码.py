import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch  # PyTorch，用于张量计算
import torch.nn as nn  # 神经网络模块
import numpy as np  # 数值计算
import pandas as pd  # 处理表格数据
from scipy.spatial.distance import pdist  # 计算成对距离
from scipy.cluster.hierarchy import linkage, fcluster  # 层次聚类

file_path = "/Users/chengyuhang/Desktop/TCGALIHC.tsv"  # 表达矩阵路径
df = pd.read_csv(file_path, sep="\t", index_col=0)
gene_names = df.index.tolist() #将index的第一列基因名，转化为基因名列表
data = torch.tensor(df.values.T, dtype=torch.float32)# 转换为tensor，并转置
n_samples, n_genes = data.shape # 获取样本数和基因数（先横坐标，再纵坐标）
print(f"Samples: {n_samples}, Genes: {n_genes}")

# 2. 降维（选高变基因）
var = torch.var(data, dim = 0) # 计算每个基因在所有样本中的方差（变化程度）dim0就是一维，基因
topk = 1000 # 只保留前1000个高变基因（降维，去噪）
top_idx = torch.topk(var, topk).indices # 找到方差最大的1000个基因的索引
data = data[:, top_idx] # 只保留这些高变基因
n_genes = topk # 更新基因数量
print(f"Selected top {n_genes} variable genes")

# 3. 注意力模型（本质：学基因embedding）
class GeneAttention(nn.Module):
    def __init__(self, n_genes, embed_dim=8):
        super().__init__()
        # 为每个基因单独定义一个线性层
        # 注意：这是“每个基因独立编码”，没有真正交互
        self.linears = nn.ModuleList([nn.Linear(1, embed_dim) for _ in range(n_genes)])

    def forward(self, x):
        out_list = []

        for i, lin in enumerate(self.linears):
            gene_column = x[:, i:i+1]
            # 取第i个基因在所有样本中的表达（shape: samples × 1）
            out_list.append(lin(gene_column))
            # 每个基因通过自己的线性层 → 映射到embedding空间
        out = torch.stack(out_list, dim=1)
        # shape: samples × genes × embed_dim
        out_mean = out.mean(dim=0)
        # 对所有样本取平均 → 得到每个基因的embedding（genes × dim）
        return out_mean
        # 输出：每个基因一个向量（embedding）

# 4. 基因模块划分（聚类）
def gene_modules_from_embedding(embedding, gene_names, threshold=0.2):
    emb = embedding.detach().numpy()
    # tensor → numpy
    # 归一化（变成单位向量）
    norm_emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    # 计算余弦距离（基因之间的相似性）
    dist_vector = pdist(norm_emb, metric='cosine')
    # 防止数值误差导致负距离（很关键！）
    dist_vector = np.clip(dist_vector, 0, None)
    # 层次聚类（average linkage）
    Z = linkage(dist_vector, method='average')
    # 根据距离阈值划分模块
    clusters = fcluster(Z, t=threshold, criterion='distance')
    modules = {}  # 存储：模块ID → 基因列表
    module_features = {}  # 存储：模块ID → embedding特征
    for idx, cluster_id in enumerate(clusters):
        modules.setdefault(cluster_id, []).append(gene_names[idx])
    # 计算每个模块的平均embedding
    for cluster_id, genes in modules.items():
        idxs = [gene_names.index(g) for g in genes]
        # 找到这些基因的索引
        module_features[cluster_id] = torch.tensor(np.mean(emb[idxs], axis=0))
        # 模块特征 = 所有基因embedding平均
    return modules, module_features

# 5. 运行模型

attention = GeneAttention(n_genes, embed_dim=8)
# 初始化模型（每个基因 → 8维向量）
embedding = attention(data)
# 得到每个基因的embedding（genes × 8）
modules, module_features = gene_modules_from_embedding(
    embedding, gene_names, threshold=0.6
)
# 根据embedding聚类成模块


# =========================
# 6. 输出模块
# =========================
print("\n=== Gene Modules ===")

for cluster_id, genes in modules.items():
    print(f"\nModule {cluster_id} ({len(genes)} genes):")
    print(genes[:10], "...")
    # 只显示前10个基因（防止太长）

print("\nTotal modules:", len(modules))


# =========================
# 7. 计算模块表达（关键步骤🔥）
# =========================
def compute_module_activity(data, modules, gene_names):
    module_activity = {}

    for cluster_id, genes in modules.items():
        idxs = [gene_names.index(g) for g in genes]
        # 找到模块中基因的位置

        module_activity[cluster_id] = data[:, idxs].mean(dim=1).numpy()
        # 每个样本中，该模块的平均表达（module score）

    return module_activity


module_activity = compute_module_activity(data, modules, gene_names)


# =========================
# 8. 按模块表达排序
# =========================
module_mean_expression = {}

for cluster_id, values in module_activity.items():
    module_mean_expression[cluster_id] = np.mean(values)
    # 模块整体平均表达（跨所有样本）

# 按表达从高到低排序
sorted_modules = sorted(module_mean_expression.items(), key=lambda x: x[1], reverse=True)

print("\n=== Modules sorted by average expression ===")

for cluster_id, mean_val in sorted_modules[:20]:
    # 只显示前20个模块

    genes = modules[cluster_id]

    print(f"\nModule {cluster_id} ({len(genes)} genes)")
    print(f"Mean expression: {mean_val:.3f}")
    print("Genes:", genes[:10], "...")