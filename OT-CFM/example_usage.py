import torch
from torchcfm.conditional_flow_matching import ExactOptimalTransportConditionalFlowMatcher

# 创建一个实例，设置 sigma 参数（例如0.1）
matcher = ExactOptimalTransportConditionalFlowMatcher(sigma=0.1)

# 准备示例输入数据
# x0, x1 分别表示 source 和 target minibatch，假设每个样本维度为2，batch_size为8
x0 = torch.randn(8, 2)
x1 = torch.randn(8, 2)

# 使用 sample_location_and_conditional_flow 进行采样
t, xt, ut = matcher.sample_location_and_conditional_flow(x0, x1)
print("Sample without labels:")
print("t:", t)
print("xt:", xt)
print("ut:", ut)

# 如果需要同时传入标签，使用 guided_sample_location_and_conditional_flow
y0 = torch.randint(0, 10, (8,))  # 示例标签
y1 = torch.randint(0, 10, (8,))
# 当 return_noise 为 True 时，会返回额外的 epsilon
t, xt, ut, y0_out, y1_out, eps = matcher.guided_sample_location_and_conditional_flow(x0, x1, y0, y1, return_noise=True)
print("\nGuided sample with labels:")
print("t:", t)
print("xt:", xt)
print("ut:", ut)
print("y0:", y0_out)
print("y1:", y1_out)
print("eps:", eps)
