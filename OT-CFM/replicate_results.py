import time
import numpy as np
from omegaconf import OmegaConf
from hydra import initialize, compose

# 注意：请确保configs目录下的train.yaml配置符合论文实验要求，
# 同时所有需要的模块（模型、数据模块等）均已正确实现。

def run_experiment(overrides):
    # 用hydra加载配置（配置路径根据你的项目结构调整）
    with initialize(config_path="configs"):
        cfg = compose(config_name="train", overrides=overrides)
    start = time.time()
    # 调用训练函数，返回指标和对象字典（这里要求train函数已返回训练指标中包含测试2Wasserstein误差和路径能量）
    from src.train import train
    metric_dict, _ = train(cfg)
    duration = time.time() - start
    # 假定指标键为"test_2Wasserstein_error"和"normalized_path_energy"
    test_error = metric_dict.get("test_2Wasserstein_error", np.nan)
    path_energy = metric_dict.get("normalized_path_energy", np.nan)
    return test_error, path_energy, duration

def main():
    # 每个实验设定：label, 模型override, 数据模块override（确保与论文中对应数据一致）
    experiments = [
        ("OT-CFM", "model=otcfm", "datamodule=8gaussians,moons,twodim,gaussians"),
        ("CFM",    "model=cfm",   "datamodule=8gaussians,moons,twodim,gaussians"),
        ("FM",     "model=fm",    "datamodule=8gaussians,moons,gaussians"),
        ("Reg. CNF", "model=reg_cnf", "datamodule=8gaussians,moons,twodim,gaussians"),
        ("CNF",    "model=cnf",   "datamodule=8gaussians,moons,twodim,gaussians"),
        ("ICNN",   "model=icnn",  "datamodule=8gaussians,moons,twodim,gaussians"),
    ]
    # 使用多个随机种子，保证复现性（与论文中一致的种子）
    seeds = [42, 43, 44, 45, 46]
    
    results = {exp[0]: {"test_error": [], "path_energy": [], "train_time": []} for exp in experiments}
    
    for label, model_override, datamodule_override in experiments:
        for seed in seeds:
            overrides = [
                "experiment=cfm",       # 实验名称（可调整）
                model_override,         # 模型配置override
                datamodule_override,    # 数据模块配置override
                "model.sigma_min=0.1",    # 超参数设置，与论文一致
                f"seed={seed}",
            ]
            print(f"Running {label} with seed {seed} overrides: {overrides}")
            te, pe, tt = run_experiment(overrides)
            results[label]["test_error"].append(te)
            results[label]["path_energy"].append(pe)
            results[label]["train_time"].append(tt)
    
    # 汇总结果：计算均值和标准差
    print("\nFinal Results:")
    header = ("Method", "Test 2W Error", "Path Energy", "Train Time (s x1e3)")
    print("{:<12s}{:<24s}{:<24s}{:<24s}".format(*header))
    for label, metrics in results.items():
        te_arr = np.array(metrics["test_error"])
        pe_arr = np.array(metrics["path_energy"])
        tt_arr = np.array(metrics["train_time"]) * 1e-3  # 转换为×1e3秒
        line = "{:<12s}{:<12.3f}±{:<10.3f}{:<12.3f}±{:<10.3f}{:<12.3f}±{:<10.3f}".format(
            label,
            te_arr.mean(), te_arr.std(),
            pe_arr.mean(), pe_arr.std(),
            tt_arr.mean(), tt_arr.std(),
        )
        print(line)

if __name__ == "__main__":
    main()
