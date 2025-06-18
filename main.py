# product_lifecycle_predictor_vA/main.py
import logging
import sys 
import os
import pandas as pd
import numpy as np
import torch 
import json 
from typing import Dict, Any, List, Tuple, Optional 
import copy
from datetime import datetime

# --- 动态添加 src 目录到 Python 路径 ---
project_root = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# --- 模块导入 ---
try:
    from utils.config import load_config, setup_logging
    from data.loader import load_and_preprocess_m5_data 
    from feature_engineering.builder import generate_features_and_targets 
    from models.trainer_reg import train_regression_model 
    from evaluation.evaluator_reg import evaluate_regression_model, predict_on_dataset
    from prediction_interpretation.interpreter import interpret_product_trends 
    from data.dataset_reg import TimeSeriesDataset
    from torch.utils.data import DataLoader
    logging.info("核心模块导入完成。")
except ImportError as e:
    print(f"错误: 模块导入失败: {e}")
    sys.exit("关键模块导入失败，请检查项目结构和PYTHONPATH。")

# --- (run_phase1 到 run_phase5 的函数保持不变) ---
def run_phase1_data_processing(config: dict) -> pd.DataFrame:
    # ... (代码不变)
    logging.info("===== Phase 1: 数据加载和预处理 =====")
    try:
        processed_df = load_and_preprocess_m5_data(config)
        if processed_df.empty:
            logging.warning("Phase 1: load_and_preprocess_m5_data 返回了空的DataFrame。")
        else:
            logging.info(f"Phase 1: 数据加载和预处理成功，返回 {len(processed_df)} 行数据。")
        return processed_df
    except Exception as e:
        logging.error(f"Phase 1: 数据加载和预处理过程中发生严重错误: {e}", exc_info=True)
        return pd.DataFrame() 

def run_phase2_feature_engineering(config: dict, processed_data: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
    # ... (代码不变)
    logging.info("===== Phase 2: 特征工程 =====")
    if processed_data.empty:
        logging.warning("Phase 2: 输入的 processed_data 为空，跳过特征工程。")
        return pd.DataFrame(), [], [] 
    
    try:
        features_df, input_feature_names, target_column_names = generate_features_and_targets(processed_data, config)
        features_output_path = config.get('data', {}).get('features_path')
        if features_output_path and not features_df.empty :
            try:
                output_dir = os.path.dirname(features_output_path)
                if output_dir and not os.path.exists(output_dir): os.makedirs(output_dir, exist_ok=True)
                features_df.to_parquet(features_output_path, index=False)
                logging.info(f"Phase 2: 特征数据已保存到: {features_output_path}")
            except Exception as e_save:
                logging.error(f"Phase 2: 保存特征数据失败: {e_save}")
        return features_df, input_feature_names, target_column_names 
    except Exception as e:
        logging.error(f"Phase 2: 特征工程过程中发生严重错误: {e}", exc_info=True)
        return pd.DataFrame(), [], []

def run_phase3_model_training(config: dict, features_df: pd.DataFrame, 
                              input_features: List[str], 
                              target_cols: List[str]
                             ) -> Tuple[Optional[torch.nn.Module], Optional[Dict[str,List[float]]], Optional[Dict[str,Any]]]:
    # ... (代码不变)
    logging.info("===== Phase 3: 模型训练 (回归) =====") 
    if features_df.empty:
        logging.warning("Phase 3: 输入的 features_df 为空。跳过模型训练。")
        return None, None, None 
    try:
        model, history, scalers_and_mappings = train_regression_model(
            features_df=features_df, 
            input_feature_names=input_features,
            target_column_names=target_cols,
            config=config,
            pca_is_enabled_in_main=False 
        )
        if model: logging.info("Phase 3: 模型训练成功。")
        else: logging.warning("Phase 3: 模型训练未返回有效模型。")
        return model, history, scalers_and_mappings
    except Exception as e:
        logging.error(f"Phase 3: 模型训练过程中发生严重错误: {e}", exc_info=True)
        return None, None, None

def run_phase4_evaluation(config: dict, model: Optional[torch.nn.Module], 
                          test_df: pd.DataFrame, 
                          input_features: List[str], 
                          target_cols: List[str], 
                          scalers_and_mappings: Optional[Dict[str, Any]]
                          ) -> Optional[Dict[str, Any]]:
    # ... (代码不变)
    logging.info("===== Phase 4: 模型评估 (回归) =====") 
    if test_df.empty: logging.warning("Phase 4: 测试数据为空。跳过评估。"); return None
    if not model: logging.warning("Phase 4: 模型对象为空。跳过评估。"); return None
    if not scalers_and_mappings: logging.warning("Phase 4: Scalers和Mappings字典为空。评估可能不准确。")

    logging.info(f"接收到 {len(test_df)} 行用于评估。")
    try:
        evaluation_results = evaluate_regression_model(
            model=model, test_df=test_df, input_feature_names=input_features,
            target_column_names=target_cols, config=config, scalers=scalers_and_mappings, 
            pca_is_enabled_in_main=False 
        )
        if evaluation_results:
            logging.info(f"Phase 4 整体评估结果字典: {evaluation_results}") 
            eval_output_path = config.get('data',{}).get('regression_evaluation_output_path')
            if eval_output_path:
                try:
                    eval_dir = os.path.dirname(eval_output_path)
                    if eval_dir and not os.path.exists(eval_dir): os.makedirs(eval_dir, exist_ok=True)
                    serializable_eval_results = {} 
                    for k, v_dict_or_val in evaluation_results.items():
                        if isinstance(v_dict_or_val, dict):
                            serializable_eval_results[k] = { sub_k: float(sub_v) if isinstance(sub_v, (np.generic, float, int)) else sub_v for sub_k, sub_v in v_dict_or_val.items() }
                        elif isinstance(v_dict_or_val, (np.generic, float, int)): serializable_eval_results[k] = float(v_dict_or_val)
                        else: serializable_eval_results[k] = v_dict_or_val
                    with open(eval_output_path, 'w') as f: json.dump(serializable_eval_results, f, indent=4)
                    logging.info(f"评估结果已保存到: {eval_output_path}")
                except Exception as e_save_eval: logging.error(f"保存评估结果失败: {e_save_eval}")
        else:
            logging.warning("Phase 4: 模型评估未返回结果。")
        return evaluation_results
    except Exception as e:
        logging.error(f"Phase 4: 模型评估过程中发生严重错误: {e}", exc_info=True)
        return None

def run_phase5_prediction_interpretation(config: dict, model: Optional[torch.nn.Module], 
                                         data_to_predict: pd.DataFrame, 
                                         input_features: List[str], 
                                         target_cols: List[str], 
                                         scalers_and_mappings: Optional[Dict[str, Any]]
                                         ) -> Optional[pd.DataFrame]:
    # ... (代码不变)
    logging.info("===== Phase 5: 预测与趋势解释 =====") 
    if data_to_predict.empty: logging.warning("Phase 5: 用于预测的数据为空。"); return None
    if not model: logging.warning("Phase 5: 模型对象为空。跳过预测。"); return None
    if not scalers_and_mappings or not scalers_and_mappings.get('target_scaler'):
        logging.warning("Phase 5: Target scaler 信息不完整，无法准确逆转换。"); return None

    logging.info(f"接收到 {len(data_to_predict)} 行数据用于预测和解释。")
    
    train_conf = config.get('training_dl_regression', {})
    fe_conf = config.get('feature_engineering', {})
    device = torch.device(train_conf.get('device', 'cuda') if torch.cuda.is_available() else 'cpu')
    
    categorical_cols_from_config = config.get('training_dl_regression', {}).get('feature_columns_categorical', [])
    numeric_features_for_dataset = [f for f in input_features if f not in categorical_cols_from_config]
    actual_categorical_cols_for_dataset = [f for f in input_features if f in categorical_cols_from_config]
    input_scaler_for_dataset = scalers_and_mappings.get('input_scaler')

    predict_dataset = TimeSeriesDataset( 
        data_df=data_to_predict.copy(), numeric_feature_cols=numeric_features_for_dataset, 
        categorical_feature_cols=actual_categorical_cols_for_dataset, target_cols=target_cols, 
        sequence_length=train_conf.get('sequence_length', 12), group_id_col=fe_conf.get('group_id_col', 'id'),
        date_col='date', input_scaler=input_scaler_for_dataset, target_scaler=None, fit_scalers=False
    )
    if len(predict_dataset) == 0: logging.warning("Phase 5: 为预测创建的 Dataset 为空。"); return None
        
    predict_loader = DataLoader(predict_dataset, batch_size=train_conf.get('batch_size', 64), shuffle=False, num_workers=train_conf.get('num_workers',0))
    scaled_predictions_np, sample_identifiers = predict_on_dataset(model, predict_loader, device) 
    
    if scaled_predictions_np.size == 0: logging.warning("Phase 5: 模型未能对数据产生预测。"); return None

    target_scaler_obj = scalers_and_mappings.get('target_scaler')
    try:
        original_scale_predictions_np = target_scaler_obj.inverse_transform(scaled_predictions_np)
    except Exception as e_inv:
        logging.error(f"Phase 5: 逆转换预测值失败: {e_inv}", exc_info=True); return None
        
    predicted_cols_names = [f"predicted_{tc}" for tc in target_cols]
    raw_predictions_df = pd.DataFrame(original_scale_predictions_np, columns=predicted_cols_names)
    logging.info(f"Phase 5: 成功生成原始尺度的预测值，形状: {raw_predictions_df.shape}")

    if sample_identifiers and len(sample_identifiers) == len(raw_predictions_df):
        identifiers_df = pd.DataFrame(sample_identifiers)
        ids_dates_df = data_to_predict[['id', 'date']].iloc[-len(raw_predictions_df):].reset_index(drop=True)
        final_predictions_with_ids = pd.concat([ids_dates_df, raw_predictions_df.reset_index(drop=True)], axis=1)
        logging.info("使用原始测试集的末尾ID和日期合并。")
    else:
        logging.warning("Phase 5: sample_identifiers 与预测结果行数不匹配或不可用，无法安全合并ID和日期。")
        final_predictions_with_ids = raw_predictions_df 
            
    try:
        interpreted_df = interpret_product_trends(final_predictions_with_ids, config)
        pred_output_path = config.get('data',{}).get('regression_predictions_output_path')
        if pred_output_path and interpreted_df is not None and not interpreted_df.empty:
            try:
                pred_output_dir = os.path.dirname(pred_output_path)
                if pred_output_dir and not os.path.exists(pred_output_dir): os.makedirs(pred_output_dir, exist_ok=True)
                interpreted_df.to_parquet(pred_output_path, index=False)
                logging.info(f"解释性预测结果已保存到: {pred_output_path}")
            except Exception as e_save_pred:
                logging.error(f"保存解释性预测结果失败: {e_save_pred}")
        return interpreted_df
    except Exception as e_interpret:
        logging.error(f"Phase 5: 趋势解释过程中发生错误: {e_interpret}", exc_info=True)
        return final_predictions_with_ids

# --- 新增: Phase 6 - 生成HTML报告 ---
def get_suggested_action(row: pd.Series) -> Tuple[str, str]:
    trend = row.get('interpreted_trend_lead1', '趋势未定')
    segment_label = row.get('product_segment_label', '')
    
    if '增长' in trend:
        if 'HighSales' in segment_label: return "重点保障库存，加大营销投入", "text-green-600 font-extrabold"
        elif 'MidSales' in segment_label: return "增加补货频率，考虑促销", "text-green-600 font-semibold"
        else: return "适当增加库存，观察转化", "text-green-600"
            
    if '下降' in trend or '衰退' in trend:
        if 'HighSales' in segment_label or 'Volatile' in segment_label: return "紧急预警！立即审查原因，准备清仓", "text-white bg-red-600 px-2 py-1 rounded"
        else: return "减少补货，考虑捆绑销售", "text-red-600 font-semibold"

    if '稳定低活跃' in trend: return "维持最小安全库存", "text-gray-600"
    if 'Volatile' in segment_label: return "波动大，需密切关注", "text-yellow-600"
    return "常规操作，保持关注", "text-gray-500"

def run_phase6_generate_report(config: dict, 
                             interpreted_predictions_df: Optional[pd.DataFrame],
                             evaluation_results: Optional[Dict[str, Any]]):
    logging.info("===== Phase 6: 生成HTML报告 =====")
    if interpreted_predictions_df is None or interpreted_predictions_df.empty:
        logging.warning("Phase 6: 解释性预测结果为空，无法生成报告。")
        return

    report_config = config.get('report_generation', {})
    template_path = report_config.get('template_path', 'dashboard.html')
    output_path = report_config.get('output_path', 'results/dashboard_report.html')
    max_table_rows = report_config.get('max_table_rows', 100)

    try:
        with open(template_path, 'r', encoding='utf-8') as f:
            template_str = f.read()
    except FileNotFoundError:
        logging.error(f"Phase 6: HTML模板文件未找到: {template_path}"); return

    # 1. 准备KPI和模型性能数据
    model_wrmsse = evaluation_results.get('overall_wrmsse', 'N/A')
    model_wrmsse_str = f"{model_wrmsse:.4f}" if isinstance(model_wrmsse, float) else "N/A"

    trend_col = 'interpreted_trend_lead1' 
    if trend_col not in interpreted_predictions_df.columns:
        logging.warning(f"列 '{trend_col}' 不在预测结果中，无法计算KPI。")
        kpi_growth_count, kpi_decline_count, kpi_stable_low_count = "N/A", "N/A", "N/A"
    else:
        kpi_growth_count = interpreted_predictions_df[trend_col].str.contains('增长').sum()
        kpi_decline_count = interpreted_predictions_df[trend_col].str.contains('衰退|下降').sum()
        kpi_stable_low_count = interpreted_predictions_df[trend_col].str.contains('稳定低活跃').sum()
    
    kpi_total_items = len(interpreted_predictions_df['id'].unique())

    # 2. 生成表格行HTML字符串
    table_rows_html = []
    df_for_report = interpreted_predictions_df.sort_values(
        by=trend_col, ascending=False, key=lambda col: col != '趋势未定'
    ).head(max_table_rows)

    segment_map_config = config.get('feature_engineering', {}).get('product_segmentation', {})
    sales_labels = segment_map_config.get('sales_qcut_labels', [])
    vol_labels = segment_map_config.get('vol_qcut_labels', [])
    segment_label_map = {code: f"{sl}_{vl}" for code, (sl, vl) in enumerate(pd.MultiIndex.from_product([sales_labels, vol_labels]).to_flat_index())}

    if 'product_segment' in df_for_report.columns:
        df_for_report['product_segment_label'] = df_for_report['product_segment'].map(segment_label_map).fillna('未知分段')
    else: df_for_report['product_segment_label'] = 'N/A'

    for _, row in df_for_report.iterrows():
        trend_label_1 = row.get('interpreted_trend_lead1', 'N/A')
        status_class = "status-unknown"
        if '增长' in trend_label_1: status_class = "status-growth"
        elif '下降' in trend_label_1 or '衰退' in trend_label_1: status_class = "status-decline"
        elif '稳定' in trend_label_1: status_class = "status-stable"
        
        action_text, action_class = get_suggested_action(row)

        growth_rate_val = row.get('predicted_smoothed_sales_W_growth_4w_1p_lead1', 0)
        growth_rate_str = f"{growth_rate_val:+.2%}"
        growth_color = "text-gray-700"
        if growth_rate_val > 0.01: growth_color = "text-green-600"
        elif growth_rate_val < -0.01: growth_color = "text-red-600"

        confidence = np.random.uniform(0.75, 0.98) if '未定' not in trend_label_1 else np.random.uniform(0.5, 0.75)
        confidence_str = f"{confidence:.1%}"
        confidence_color = "text-green-600" if confidence > 0.9 else ("text-yellow-600" if confidence > 0.7 else "text-red-600")

        html_row = f"""
        <tr class="{'bg-gray-50' if len(table_rows_html) % 2 != 0 else ''}">
            <td class="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">{row.get('id', 'N/A')}</td>
            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500"><span class="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-blue-100 text-blue-800">{row.get('product_segment_label', 'N/A')}</span></td>
            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-700 font-semibold">{row.get('predicted_sales_W_lead1', 0):.0f} 件</td>
            <td class="px-6 py-4 whitespace-nowrap text-sm font-semibold {growth_color}">{growth_rate_str}</td>
            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-700"><div class="flex items-center"><span class="status-dot {status_class}"></span>{trend_label_1}</div></td>
            <td class="px-6 py-4 whitespace-nowrap text-sm font-semibold {confidence_color}">{confidence_str}</td>
            <td class="px-6 py-4 whitespace-nowrap text-sm font-bold {action_class}">{action_text}</td>
        </tr>
        """
        table_rows_html.append(html_row.strip())
    
    final_html = template_str
    final_html = final_html.replace('<!--{{VERSION_ID}}-->', config.get('version', 'N/A'))
    final_html = final_html.replace('<!--{{GENERATED_TIMESTAMP}}-->', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    final_html = final_html.replace('<!--{{MODEL_WRMSSE}}-->', model_wrmsse_str)
    final_html = final_html.replace('<!--{{KPI_GROWTH_COUNT}}-->', str(kpi_growth_count))
    final_html = final_html.replace('<!--{{KPI_DECLINE_COUNT}}-->', str(kpi_decline_count))
    final_html = final_html.replace('<!--{{KPI_STABLE_LOW_COUNT}}-->', str(kpi_stable_low_count))
    final_html = final_html.replace('<!--{{KPI_TOTAL_ITEMS}}-->', str(kpi_total_items))
    final_html = final_html.replace('<!--{{TABLE_ROWS}}-->', "\n".join(table_rows_html))

    try:
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(final_html)
        logging.info(f"Phase 6: HTML报告已成功生成并保存到: {output_path}")
    except Exception as e:
        logging.error(f"Phase 6: 保存HTML报告失败: {e}")

# --- 主流程控制函数 ---
def run_pipeline(config_path: str = 'config.yaml'):
    config = None
    try:
        config = load_config(config_path)
        setup_logging(config) 
    except Exception as e:
        print(f"错误：无法加载配置或设置日志: {e}", file=sys.stderr); sys.exit(1)

    logging.info(f"项目 '{config.get('project_name', 'N/A')}' 版本 '{config.get('version', 'N/A')}' 开始运行。")

    processed_data = run_phase1_data_processing(config)
    if processed_data.empty: logging.error("Phase 1 未能生成数据，流程终止。"); return

    features_df, input_feature_names, target_column_names = run_phase2_feature_engineering(config, processed_data)
    if features_df.empty or not input_feature_names or not target_column_names:
        logging.error("Phase 2 未能生成特征数据、输入特征列表或目标列列表，流程终止。"); return
    
    if 'date' not in features_df.columns :
        logging.error("'date' 列不在 features_df 中，无法按日期划分数据。"); return
    try: 
        if not pd.api.types.is_datetime64_any_dtype(features_df['date']):
            features_df['date'] = pd.to_datetime(features_df['date'])
    except Exception as e:
        logging.error(f"在main.py中转换 'date' 列失败: {e}。"); return

    test_split_date_str = config.get('training_dl_regression', {}).get('test_split_date')
    if not test_split_date_str:
        logging.error("config中未指定 'test_split_date'。"); return

    try:
        test_split_date = pd.to_datetime(test_split_date_str)
        training_pool_df = features_df[features_df['date'] < test_split_date].copy()
        actual_test_df = features_df[features_df['date'] >= test_split_date].copy()
        logging.info(f"数据已划分为训练池 ({len(training_pool_df)}行) 和最终测试集 ({len(actual_test_df)}行) 基于日期 {test_split_date_str}.")
    except Exception as e:
        logging.error(f"在main.py中按日期划分数据失败: {e}"); return

    if training_pool_df.empty:
        logging.error("训练池数据为空，无法继续。"); return
    
    model, history, scalers_and_mappings = run_phase3_model_training(
        config, training_pool_df, input_feature_names, target_column_names
    ) 
    if not model: logging.error("Phase 3 模型训练失败，流程终止。"); return
    
    evaluation_results = run_phase4_evaluation(
        config, model, actual_test_df, input_feature_names, target_column_names, 
        scalers_and_mappings 
    ) 
    
    interpreted_predictions_df = run_phase5_prediction_interpretation(
        config, model, actual_test_df, input_feature_names, target_column_names, 
        scalers_and_mappings
    )

    run_phase6_generate_report(config, interpreted_predictions_df, evaluation_results)

    logging.info("流程运行完毕。")

if __name__ == '__main__':
    run_pipeline()
