import os
import numpy as np
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)

class OUActionNoise:
    def __init__(self, mu, sigma=0.15, theta=0.2, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()
    
    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        
        return x
    
    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)
    
def create_directory(path: str, sub_paths: list):
    for sub_path in sub_paths:
        if not os.path.exists(path + sub_path):
            os.makedirs(path + sub_path, exist_ok=True)
            print('Create path: {} successfully'.format(path+sub_path))
        else:
            print('Path: {} already exists'.format(path+sub_path))

def plot_learning_curve(episodes, records, title, ylabel, figure_file):
    # 创建目录
    dir_path = os.path.dirname(figure_file)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        logger.info(f"目录 '{dir_path}' 不存在，已自动创建。")
    else:
        logger.info(f"目录 '{dir_path}' 已存在。")
    
    plt.figure()
    plt.plot(episodes, records, color='r', linestyle='-')
    plt.title(title)
    plt.xlabel('episode')
    plt.ylabel(ylabel)
    
    plt.show()
    
    plt.savefig(figure_file)
    logger.info(f"图像已保存至 {figure_file}")
    plt.close()

def scale_action(action, high, low):
    action = np.clip(action, -1, 1)
    weight = (high - low) / 2
    bias = (high + low) / 2
    action_new = action * weight + bias
    
    return action_new

def setup_logger(log_dir=None, log_filename="train.log", level=logging.INFO):
    # 创建 Logger
    logger = logging.getLogger()
    logger.setLevel(level)
    
    # 防止多次添加 Handler（例如你多次调用 setup_logger 会重复输出）
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # 日志格式
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(filename)s[line%(lineno)d] - %(levelname)s: %(message)s',
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    # 控制台输出 Handler（StreamHandler）
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    
    # 文件输出 Handler（FileHandler）
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, log_filename)
        file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger