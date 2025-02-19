import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
import os
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# 1. 读取数据
file_path = r'F:\图神经网络\开题准备\GNN\最终\标志物分析/0.79\诊断标志物/5.ROC\python/major.csv'
data = pd.read_csv(file_path, index_col=0)

# 2. 构建标签
labels = [0 if 'treat' in col else 1 for col in data.columns]
labels = pd.Series(labels)

output_folder = r'F:\图神经网络\开题准备\GNN\最终\标志物分析/0.79\诊断标志物/5.ROC\python'
os.makedirs(output_folder, exist_ok=True)

# 创建一个新的图形
fig, ax = plt.subplots(figsize=(10, 8))

# 收集所有 AUC 值，用于颜色映射
all_aucs = []

# 3. 对于每个基因计算和绘制 ROC 曲线
for gene in data.index:
    gene_values = data.loc[gene].astype(float)

    # 检查样本数量是否一致
    if len(gene_values) != len(labels):
        print(f"Warning: The number of samples in gene {gene} is inconsistent with labels. Skipping...")
        continue

    # 计算 ROC 曲线和 AUC
    auc = roc_auc_score(labels, gene_values)
    if auc < 0.5:
        gene_values = -gene_values  # 取反预测得分
        auc = 1 - auc
    fpr, tpr, _ = roc_curve(labels, gene_values)
    all_aucs.append(auc)

# 定义蓝色渐变颜色映射
norm = Normalize(vmin=min(all_aucs), vmax=max(all_aucs))
cmap = plt.get_cmap('Blues')

# 再次循环绘制曲线，应用颜色映射
legend_handles = []
for i, gene in enumerate(data.index):
    gene_values = data.loc[gene].astype(float)
    if len(gene_values) != len(labels):
        continue

    auc = roc_auc_score(labels, gene_values)
    if auc < 0.5:
        gene_values = -gene_values  # 取反预测得分
        auc = 1 - auc
    fpr, tpr, _ = roc_curve(labels, gene_values)

    color = cmap(norm(auc))
    line, = ax.plot(fpr, tpr, color=color, label=f'ROC Curve (AUC = {auc:.2f}) for {gene}')
    legend_handles.append(line)

# 绘制对角线
ax.plot([0, 1], [0, 1], color='navy', linestyle='--')

# 设置坐标轴标签和标题
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
#ax.set_title('Receiver Operating Characteristic for All Genes', fontsize=14)

# 创建颜色条图例
sm = ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, label='AUC Value')

# 调整图例位置到右下角
ax.legend(handles=legend_handles, loc="lower right", fontsize=8)

# 保存图像到文件夹中
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'ROC_Curve_All_Genes.png'), dpi=300)

# 关闭图像
plt.close()