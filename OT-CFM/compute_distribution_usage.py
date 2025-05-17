import sys
sys.path.append('/home/nkd/ouyangzl/conditional-flow-matching/runner/src')  # 添加到更高一层路径
import torch
from models.components.distribution_distances import compute_distribution_distances

# 假设我们有16个样本、5个时间点和2个维度
batch_size = 16
times = 5
dims = 2

# 随机生成预测数据和真实数据，形状为 [batch, times, dims]
pred = torch.randn(batch_size, times, dims)
true = torch.randn(batch_size, times, dims)

# 调用函数计算各项分布距离
names, dists = compute_distribution_distances(pred, true)

print("Distance metric names:")
print(names)
print("Computed distances:")
print(dists)
