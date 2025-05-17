import pandas as pd
import numpy as np

# 读取 CSV 文件
df = pd.read_csv("/home/nkd/ouyangzl/conditional-flow-matching/results/CFM_results.csv")  # 替换为你的文件名

# 按照 source 和 target 分组
grouped = df.groupby(["source", "target"])

# 需要统计的列
metrics = ["W2", "PE", "NPE", "time", "ov"]

# 定义一个函数将 μ 和 σ 格式化为 μ ± σ
def format_mu_sigma(x):
    return f"{x['mean']:.2f} ± {x['std']:.2f}"

# 计算均值和标准差并格式化
summary = grouped[metrics].agg(['mean', 'std'])

# 使用 pandas 的 MultiIndex，可以 flatten 为单层列名
summary.columns = ['_'.join(col).strip() for col in summary.columns.values]

# 为每个指标生成 μ ± σ 的格式
formatted_summary = pd.DataFrame(index=summary.index)
for metric in metrics:
    formatted_summary[metric] = summary.apply(
        lambda row: f"{row[metric + '_mean']:.2f} ± {row[metric + '_std']:.2f}", axis=1
    )

# 打印结果
print(formatted_summary)

# 如果你想保存为 CSV：
formatted_summary.to_csv("CFM_0.1_256.csv")
