import os
import traceback
import warnings
import pandas as pd # 添加导入 (Added import)

# 导入工具和配置加载器 (Import utilities and config loader)
from src.utils.config import load_config
from src.utils.excel_export import export_results_to_excel

# 导入工作流的各个步骤 (Import steps of the workflow)
from src.data.load import load_and_preprocess_data, create_daily_sales_data
from src.data.features import create_features
from src.lifecycle.simple import simple_lifecycle_labels
from src.evaluation.metrics import evaluate_lifecycle_labels_quality
from src.evaluation.visualization import visualize_lifecycle_labels
from src.models.training import prepare_sequences, train_and_evaluate_model

# 忽略警告 (可选) (Ignore warnings (optional))
warnings.filterwarnings("ignore")

def main(config):
    """
    执行整个产品生命周期预测工作流的主函数。
    (Main function to run the entire product lifecycle prediction workflow.)

    Args:
        config (dict): 加载的配置字典。 (Loaded configuration dictionary.)
    """
    # 获取输出目录并创建 (Get output directory and create it)
    output_dir = config['evaluation']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    print(f"输出目录: {output_dir}") # Output directory...

    # 用于存储所有结果以供 Excel 导出的字典
    # (Dictionary to store all results for Excel export)
    all_results = {'config': config}

    try:
        # --- 1. 加载和预处理数据 (Load and Preprocess Data) ---
        file_path = config['data']['raw_path']
        print(f"从 '{file_path}' 加载数据...") # Loading data from '...'
        df = load_and_preprocess_data(file_path)
        if df is None or df.empty:
             print("错误: 无法加载或预处理数据。程序中止。") # ERROR: Failed to load or preprocess data. Aborting.
             return # 提前退出 (Exit early)
        print(f"数据加载并预处理完毕，形状: {df.shape}") # Data loaded and preprocessed, shape...
        all_results['initial_data_shape'] = df.shape # 记录形状 (Record shape)

        # --- 2. 创建每日销售数据 (Create Daily Sales Data) ---
        daily_sales, product_desc = create_daily_sales_data(df)
        all_results['product_descriptions'] = product_desc # 存储产品描述 (Store product descriptions)
        if daily_sales.empty:
             print("错误: 未能创建每日销售数据。程序中止。") # ERROR: Failed to create daily sales data. Aborting.
             return

        # --- 3. 特征工程 (Feature Engineering) ---
        windows = config['features']['windows']
        min_days_threshold = config['data']['min_days_threshold']
        product_features_df, valid_products, feature_importance = create_features(daily_sales, product_desc, df, windows, min_days_threshold)
        if product_features_df.empty:
             print("错误: 未生成任何特征。请检查数据和参数。程序中止。") # ERROR: No features generated. Check data and parameters. Aborting.
             return
        all_results['valid_products'] = valid_products # 存储有效产品列表 (Store list of valid products)
        print(f"用于分析的有效产品数量: {len(valid_products)}") # Number of valid products for analysis...

        # --- 4. 生成生命周期标签 (Generate Lifecycle Labels) ---
        print("生成生命周期标签...") # Generating lifecycle labels...
        labeled_features_list = [] # 存储带标签的产品数据 (List to store labeled product data)
        for product in valid_products:
            product_data = product_features_df[product_features_df['StockCode'] == product].copy()
            if not product_data.empty:
                try:
                    # 应用标签函数 (Apply labeling function)
                    product_data['LifecyclePhase'] = simple_lifecycle_labels(product_data)
                    labeled_features_list.append(product_data)
                except Exception as e:
                    # 如果标签生成失败，打印错误并设置默认标签 (If labeling fails, print error and set default label)
                    print(f"错误: 为产品 {product} 生成标签时出错: {e}。将标签设置为 2 (成熟期)。") # ERROR: Error labeling product ... Setting label to 2 (Maturity).
                    product_data['LifecyclePhase'] = 2 # 默认成熟期 (Default to Maturity)
                    labeled_features_list.append(product_data)

        if not labeled_features_list:
             print("错误: 标签化后没有剩余的产品。程序中止。") # ERROR: No products left after labeling. Aborting.
             return

        # 合并带标签的数据 (Concatenate labeled data)
        product_features_df = pd.concat(labeled_features_list, ignore_index=True)
        all_results['product_features'] = product_features_df # 存储带标签的 DataFrame (Store the labeled DataFrame)

        # 打印阶段分布 (Print phase distribution)
        if 'LifecyclePhase' in product_features_df.columns:
            phase_counts = product_features_df['LifecyclePhase'].value_counts().sort_index()
            print("\n生命周期阶段分布 (基于标签):") # Lifecycle Phase Distribution (Based on Labels):
            print(phase_counts)
        else:
            print("警告: 在 DataFrame 中未找到 'LifecyclePhase' 列。") # WARNING: 'LifecyclePhase' column not found in DataFrame.

        # --- 5. 评估标签质量 (Evaluate Label Quality) ---
        quality_metrics, _ = evaluate_lifecycle_labels_quality(product_features_df, valid_products)
        all_results['quality_metrics'] = quality_metrics # 存储质量指标 (Store quality metrics)

        # --- 6. 可视化样本产品的生命周期标签 (Visualize Sample Product Lifecycles) ---
        n_vis = config['evaluation']['n_visualization_samples']
        sample_products_vis = visualize_lifecycle_labels(product_features_df, valid_products, product_desc, n_vis, output_dir)
        all_results['sample_products_vis'] = sample_products_vis # 存储可视化的样本列表 (Store list of visualized samples)

        # --- 7. 准备序列数据 (Prepare Sequence Data) ---
        seq_length = config['features']['sequence_length']
        X, y, product_ids, dates, scalers = prepare_sequences(product_features_df, valid_products, seq_length)
        all_results['X'] = X # 存储序列 (Store sequences)
        all_results['y'] = y # 存储标签 (Store labels)
        all_results['product_ids_seq'] = product_ids # 存储序列对应的产品ID (Store product IDs corresponding to sequences)
        all_results['dates_seq'] = dates         # 存储序列对应的日期 (Store dates corresponding to sequences)
        all_results['scalers'] = scalers       # 存储缩放器 (Store scalers)

        # --- 8. 训练和评估模型 (Train and Evaluate Model) ---
        print("\n开始模型训练和评估...") # Starting model training and evaluation...
        model, history, test_eval_results, report_str, cm, report_dict = train_and_evaluate_model(X, y, config)
        all_results['model_state_dict'] = model.state_dict() # 存储模型状态字典 (Store model state dictionary - optional)
        all_results['history'] = history                    # 存储训练历史 (Store training history)
        all_results['test_results'] = test_eval_results       # 存储测试结果 (y_true, y_pred) (Store test results)
        all_results['classification_report_str'] = report_str # 存储报告字符串 (Store report string)
        all_results['confusion_matrix'] = cm                # 存储混淆矩阵 (Store confusion matrix)
        all_results['report_dict'] = report_dict              # 存储报告字典 (Store report dictionary)
        print("模型训练和评估完成。") # Model training and evaluation finished.

        # --- 9. 将结果导出到 Excel (Export Results to Excel) ---
        excel_filename = config['excel']['output_filename']
        excel_path = os.path.join(output_dir, excel_filename)
        export_results_to_excel(all_results, excel_path)

        print("\n工作流成功完成！") # Workflow successfully completed!
        print(f"结果、图表和 Excel 文件已保存在目录: {output_dir}") # Results, plots, and Excel file saved in directory...

    # --- 错误处理 (Error Handling) ---
    except FileNotFoundError as e:
        print(f"\n错误: 数据文件未找到: {e}") # ERROR: Data file not found...
        print(f"请确保原始数据文件位于 '{config['data']['raw_path']}' 指定的位置，并且路径在 `config.yaml` 中正确无误。")
        # Please ensure the raw data file is located at the path specified by '...' and the path in `config.yaml` is correct.
    except ValueError as e:
        print(f"\n错误: 数据处理过程中发生错误: {e}") # ERROR: Error occurred during data processing...
        print(traceback.format_exc())
    except Exception as e:
        print(f"\n发生意外错误: {type(e).__name__} - {e}") # An unexpected error occurred...
        print("错误追踪:") # Traceback:
        print(traceback.format_exc())

# --- 脚本入口点 (Script Entry Point) ---
if __name__ == "__main__":
    print("启动产品生命周期预测工作流...") # Starting Product Lifecycle Prediction Workflow...
    # 加载配置 (Load configuration)
    # 从项目根目录的 config.yaml 加载 (Load from config.yaml in the project root)
    app_config = load_config('config.yaml')

    # 执行主工作流 (Execute the main workflow)
    main(app_config)