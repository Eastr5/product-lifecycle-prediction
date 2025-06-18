# product_lifecycle_predictor_vA/src/utils/config.py
import yaml
import logging
import os
import sys 
from typing import Dict, Any

# print("DEBUG: utils/config.py - 开始执行") 

def load_config(config_path: str = 'config.yaml') -> Dict[str, Any]:
    """
    加载 YAML 配置文件。
    """
    # print(f"DEBUG: load_config - 开始执行，config_path: {config_path}") 
    abs_config_path = os.path.abspath(config_path) 
    # print(f"DEBUG: load_config - 尝试加载绝对路径: {abs_config_path}")

    if not os.path.exists(abs_config_path):
        # print(f"错误: 配置文件未找到于绝对路径: {abs_config_path}") 
        logging.error(f"配置文件未找到: {abs_config_path}")
        raise FileNotFoundError(f"配置文件未找到: {abs_config_path}")
    try:
        with open(abs_config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        # print(f"DEBUG: load_config - 配置文件 '{abs_config_path}' 加载成功。") 
        logging.info(f"配置文件 '{abs_config_path}' 加载成功。")
        return config
    except yaml.YAMLError as e:
        # print(f"错误: 加载配置文件 '{abs_config_path}' 失败 (YAML解析错误): {e}") 
        logging.error(f"加载配置文件 '{abs_config_path}' 失败: {e}")
        raise
    except Exception as e:
        # print(f"错误: 加载配置文件 '{abs_config_path}' 时发生未知错误: {e}") 
        logging.error(f"加载配置文件时发生未知错误: {e}")
        raise

def setup_logging(config: Dict[str, Any]):
    """
    根据配置设置日志。
    """
    # print("DEBUG: setup_logging - 开始执行") 
    log_config = config.get('logging', {})
    log_level_str = log_config.get('level', 'INFO').upper()
    log_format = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_file = log_config.get('log_file', None)
    
    numeric_level = getattr(logging, log_level_str, None)
    if not isinstance(numeric_level, int):
        # print(f"警告: 无效的日志级别 '{log_level_str}'. 将使用 INFO 级别。")
        numeric_level = logging.INFO
    
    effective_log_file = None
    if log_file:
        if not os.path.isabs(log_file):
            project_root_dir = os.getcwd() 
            effective_log_file = os.path.join(project_root_dir, log_file)
            # print(f"DEBUG: setup_logging - 日志文件相对路径 '{log_file}' 解析为 '{effective_log_file}'")
        else:
            effective_log_file = log_file
            # print(f"DEBUG: setup_logging - 日志文件使用绝对路径 '{effective_log_file}'")

        log_dir = os.path.dirname(effective_log_file)
        if log_dir and not os.path.exists(log_dir):
            try:
                os.makedirs(log_dir, exist_ok=True)
                # print(f"DEBUG: setup_logging - 日志目录 '{log_dir}' 创建成功。")
            except OSError as e_mkdir:
                # print(f"警告: 无法创建日志目录 '{log_dir}': {e_mkdir}。日志可能无法写入文件。")
                effective_log_file = None 

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(level=numeric_level, 
                        format=log_format,
                        filename=effective_log_file if effective_log_file else None,
                        filemode='a' if effective_log_file else None, 
                        force=True 
                        )
    
    if effective_log_file:
        console_handler = logging.StreamHandler(sys.stdout) 
        console_handler.setLevel(numeric_level)
        formatter = logging.Formatter(log_format)
        console_handler.setFormatter(formatter)
        logging.getLogger('').addHandler(console_handler) 
        # print(f"DEBUG: setup_logging - 日志将输出到文件 '{effective_log_file}' 和控制台。")
        logging.info(f"日志同时输出到文件: {effective_log_file} 和控制台。")
    else:
        # print("DEBUG: setup_logging - 日志将输出到控制台。")
        logging.info("日志输出到控制台。")

# (移除了 if __name__ == '__main__': 部分的测试代码，因为它依赖于特定的文件结构)
