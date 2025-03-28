import yaml
import os

# 默认配置字典，如果 config.yaml 文件不存在或加载失败时使用
# (Default configuration dictionary, used if config.yaml file doesn't exist or fails to load)
DEFAULT_CONFIG = {
    'data': {
        'raw_path': 'data/raw/online_retail_10_11.csv', # 默认原始数据路径 (Default raw data path)
        'processed_path': 'data/processed',           # 处理后数据的保存路径 (Path to save processed data)
        'min_days_threshold': 30                      # 产品有效性阈值 (Product validity threshold)
    },
    'features': {
        'windows': [7, 14, 30],                       # 滚动窗口大小 (Rolling window sizes)
        'sequence_length': 30                         # 模型输入序列长度 (Model input sequence length)
    },
    'training': {
        'model_type': 'hybrid',                       # 模型类型 ('hybrid' 或 'lstm') (Model type ('hybrid' or 'lstm'))
        'epochs': 30,                                 # 训练轮数 (Number of training epochs)
        'batch_size': 32,                             # 批量大小 (Batch size)
        'learning_rate': 0.001,                       # 学习率 (Learning rate)
        'test_size': 0.2,                             # 测试集比例 (Test set proportion)
        'random_state': 42,                           # 随机种子，用于可复现性 (Random seed for reproducibility)
        'patience': 10                                # Early stopping 的耐心值 (Patience for early stopping)
    },
    'evaluation': {
        'n_visualization_samples': 5,                 # 可视化样本数量 (Number of visualization samples)
        'output_dir': 'results'                       # 输出目录 (Output directory)
    },
    'excel': {
        'output_filename': 'product_lifecycle_results.xlsx' # Excel 输出文件名 (Excel output filename)
    }
}

def load_config(config_path='config.yaml'):
    """
    从 YAML 文件加载配置。
    (Loads configuration from a YAML file.)

    Args:
        config_path (str): YAML 配置文件的路径。 (Path to the YAML configuration file.)

    Returns:
        dict: 加载的配置字典。如果加载失败，则返回默认配置。
              (The loaded configuration dictionary. Returns default config if loading fails.)
    """
    # 检查配置文件是否存在 (Check if the configuration file exists)
    if not os.path.exists(config_path):
        print(f"警告: 配置文件 '{config_path}' 未找到。将使用默认配置。") # WARNING: Config file '...' not found. Using default config.
        # 可选：如果配置文件不存在，可以创建一个默认的
        # (Optional: Create a default config file if it doesn't exist)
        # try:
        #     with open(config_path, 'w') as f:
        #         yaml.dump(DEFAULT_CONFIG, f, default_flow_style=False, sort_keys=False)
        #     print(f"默认配置文件已创建于: {config_path}") # Default config file created at...
        # except IOError as e:
        #     print(f"错误: 无法创建默认配置文件: {e}") # ERROR: Could not create default config file...
        return DEFAULT_CONFIG # 返回默认配置 (Return default config)

    try:
        # 尝试打开并加载 YAML 文件 (Try to open and load the YAML file)
        with open(config_path, 'r', encoding='utf-8') as f: # 指定编码 (Specify encoding)
            config = yaml.safe_load(f)
        # 可选：将加载的配置与默认配置合并，以处理缺失的键
        # (Optional: Merge loaded config with defaults to handle missing keys)
        # merged_config = {**DEFAULT_CONFIG, **config} # 简单的浅合并 (Simple shallow merge)
        # 注意：对于嵌套字典，可能需要深度合并 (Note: Deep merging might be needed for nested dictionaries)
        print(f"配置已成功从 '{config_path}' 加载。") # Configuration successfully loaded from '...'.
        # 返回加载的配置，如果加载成功但文件为空，则返回默认配置
        # (Return loaded config, or default if loading succeeded but file was empty)
        return config if config else DEFAULT_CONFIG
    except yaml.YAMLError as e:
        # 处理 YAML 解析错误 (Handle YAML parsing errors)
        print(f"错误: 解析配置文件 '{config_path}' 失败: {e}。将使用默认配置。") # ERROR: Failed to parse config file '...': ... Using default config.
        return DEFAULT_CONFIG
    except IOError as e:
        # 处理文件读取错误 (Handle file reading errors)
        print(f"错误: 读取配置文件 '{config_path}' 失败: {e}。将使用默认配置。") # ERROR: Failed to read config file '...': ... Using default config.
        return DEFAULT_CONFIG

# 可以在此处加载配置使其全局可用，但通常更好的做法是在 main.py 中加载并按需传递。
# (Config can be loaded here to make it globally available, but it's often better practice
#  to load it in main.py and pass it as needed.)
# config = load_config()