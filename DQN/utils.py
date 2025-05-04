import os
import logging
import matplotlib.pyplot as plt
logger = logging.getLogger(__name__)

def plot_learning_curve(episodes, rewards, title, ylabel, figure_file):
    dir_name = os.path.dirname(figure_file)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        logger.info(f'Dir {dir_name} does not exists !')
        logger.info(f'{dir_name} is created automaticaly !')
    else:
        logger.info(f'{dir_name} has already exists !')
        
    plt.figure()
    plt.plot(episodes, rewards, color='r', linestyle='-')
    plt.title(title)
    plt.xlabel('episodes')
    plt.ylabel(ylabel)
    plt.show()
    plt.savefig(figure_file)
    logging.info(f'Figure has been saved to path {figure_file}')
    plt.close()

def create_directory(path: str, sub_dirs: list):
    for sub_dir in sub_dirs:
        if os.path.exists(path + sub_dir):
            logger.info(path + sub_dir + 'already exists !' )
        else:
            os.makedirs(path + sub_dir, exist_ok=True)
            logger.info(path + sub_dir + 'created successfully !')

def setup_logger(log_dir=None, log_filename="dqn_train.log", level=logging.INFO):
    logger = logging.getLogger()
    logger.setLevel(level)
    
    if logger.hasHandlers():
        logger.handlers.clear()
    
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(filename)s[line%(lineno)d] - %(levelname)s: %(message)s',
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, log_filename)
        file_handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger