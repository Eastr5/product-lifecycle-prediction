# product_lifecycle_predictor_vA/src/evaluation/evaluator_reg.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import logging
# import joblib # Not used directly in this file for loading, but good for consistency
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, Any, List, Optional, Tuple

# Assuming TimeSeriesDataset is in data.dataset_reg
from data.dataset_reg import TimeSeriesDataset 

def weighted_mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-9) -> float:
    sum_abs_true = np.sum(np.abs(y_true))
    if sum_abs_true < epsilon: 
        return np.nan if np.sum(np.abs(y_true - y_pred)) > epsilon else 0.0
    return np.sum(np.abs(y_true - y_pred)) / sum_abs_true * 100

def symmetric_mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-9) -> float:
    numerator = np.abs(y_true - y_pred)
    denominator = np.abs(y_true) + np.abs(y_pred)
    smape_terms = np.zeros_like(y_true, dtype=float)
    mask = denominator > epsilon
    smape_terms[mask] = 2 * numerator[mask] / denominator[mask]
    smape_terms[(np.abs(y_true) < epsilon) & (np.abs(y_pred) < epsilon)] = 0.0
    return np.mean(smape_terms) * 100 

def predict_on_dataset(model: nn.Module, 
                       data_loader: DataLoader, 
                       device: torch.device) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    model.eval() 
    all_predictions = []
    all_identifiers = [] 
    with torch.no_grad(): 
        for batch_idx, batch in enumerate(data_loader): 
            logging.debug(f"DEBUG_EVAL: predict_on_dataset - Processing batch {batch_idx}")
            numerical_feats = batch['numerical_features'].to(device)
            
            # Ensure model.embeddings exists before trying to access it
            categorical_feats_batch = {}
            if hasattr(model, 'embeddings') and model.embeddings is not None:
                 categorical_feats_batch = {k: v.to(device) for k, v in batch['categorical_features'].items() if k in model.embeddings}
            elif batch['categorical_features']: # If there are cat features but no embeddings in model
                logging.warning("Categorical features found in batch, but model has no embeddings dict or it's None.")


            outputs = model(numerical_features=numerical_feats, categorical_features=categorical_feats_batch)
            all_predictions.append(outputs.cpu().numpy()) 
            
            if 'identifier' in batch:
                batch_identifiers_current_batch = [] 
                keys_in_identifier = list(batch['identifier'].keys())
                if keys_in_identifier: 
                    num_in_batch = 0
                    # Check if the first identifier value is a list to determine batch structure
                    first_id_val = batch['identifier'][keys_in_identifier[0]]
                    if isinstance(first_id_val, list) or (isinstance(first_id_val, torch.Tensor) and first_id_val.ndim > 0):
                        num_in_batch = len(first_id_val)
                    elif isinstance(first_id_val, (str, int, float)): # Single item in "batch" (e.g. if batch_size=1 and not wrapped in list)
                        num_in_batch = 1 # Should ideally be handled by DataLoader's batching

                    if num_in_batch > 0:
                        for i in range(num_in_batch):
                            sample_id_dict = {}
                            for key in keys_in_identifier:
                                id_val_for_key = batch['identifier'][key]
                                if isinstance(id_val_for_key, list) or (isinstance(id_val_for_key, torch.Tensor) and id_val_for_key.ndim > 0):
                                    if i < len(id_val_for_key):
                                        sample_id_dict[key] = id_val_for_key[i]
                                    else: # Should not happen if batching is correct
                                        sample_id_dict[key] = None 
                                elif num_in_batch == 1: # Single item in batch
                                    sample_id_dict[key] = id_val_for_key
                            batch_identifiers_current_batch.append(sample_id_dict)
                        all_identifiers.extend(batch_identifiers_current_batch)
                    elif num_in_batch == 0 and len(keys_in_identifier) > 0 : # If not list, assume single item batch
                         sample_id_dict = {key: batch['identifier'][key] for key in keys_in_identifier}
                         all_identifiers.append(sample_id_dict)


    if not all_predictions: 
        return np.array([]), []
        
    predictions_np = np.concatenate(all_predictions, axis=0) 
    return predictions_np, all_identifiers

def evaluate_regression_model(
    model: nn.Module, 
    test_df: pd.DataFrame, 
    input_feature_names: List[str], 
    target_column_names: List[str], 
    config: Dict[str, Any], 
    scalers: Dict[str, Any], 
    pca_is_enabled_in_main: bool = False 
) -> Optional[Dict[str, Any]]:
    logging.info("开始回归模型评估 (包含标准RMSSE/WRMSSE)...")
    
    logging.info(f"DEBUG_EVAL: Config object received in evaluate_regression_model: {type(config)}")
    if not isinstance(config, dict):
        logging.error("DEBUG_EVAL: CRITICAL - config object is not a dictionary!")
        return None

    eval_conf = config.get('evaluation_regression', {})
    logging.info(f"DEBUG_EVAL: eval_conf from config.get('evaluation_regression', {{}}): {eval_conf}")
    
    if not isinstance(eval_conf, dict):
        logging.error(f"DEBUG_EVAL: CRITICAL - eval_conf is not a dictionary (type: {type(eval_conf)}). Cannot get 'metrics'.")
        metrics_to_calculate = []
    else:
        metrics_to_calculate = eval_conf.get('metrics', []) 
    
    logging.info(f"DEBUG_EVAL: metrics_to_calculate list to be used: {metrics_to_calculate}")
    logging.info(f"DEBUG_EVAL: Type of metrics_to_calculate: {type(metrics_to_calculate)}")
    if isinstance(metrics_to_calculate, list) and metrics_to_calculate:
        logging.info(f"DEBUG_EVAL: First element of metrics_to_calculate: '{metrics_to_calculate[0]}', type: {type(metrics_to_calculate[0])}")
    elif not metrics_to_calculate:
        logging.warning("DEBUG_EVAL: metrics_to_calculate list is EMPTY. No metrics will be calculated if this is the case.")

    if test_df.empty: logging.warning("测试数据集为空。评估跳过。"); return None
    if not model: logging.warning("模型对象为空。评估跳过。"); return None
    
    target_scaler_obj = scalers.get('target_scaler')
    rmsse_scales_dict = scalers.get('rmsse_scales', {})      
    wrmsse_weights_dict = scalers.get('wrmsse_weights', {}) 

    if not target_scaler_obj: logging.warning("目标缩放器未提供，原始尺度指标可能无法计算准确。")
    if not rmsse_scales_dict: logging.warning("RMSSE缩放因子未提供，RMSSE/WRMSSE将无法计算。")
    if not wrmsse_weights_dict: logging.warning("WRMSSE权重未提供，WRMSSE将无法计算。")

    train_conf = config.get('training_dl_regression', {})
    fe_conf = config.get('feature_engineering', {})
    group_id_col = fe_conf.get('group_id_col', 'id')
    rmsse_base_sales_col = eval_conf.get('rmsse_base_sales_col', 'sales_W') 
    
    device = torch.device(train_conf.get('device', 'cuda') if torch.cuda.is_available() else 'cpu')
    model.to(device) 

    sequence_length = train_conf.get('sequence_length', 12)
    batch_size = train_conf.get('batch_size', 64) 

    input_scaler_for_dataset = None 
    if pca_is_enabled_in_main:
        numeric_features_for_dataset = [f for f in input_feature_names if f.startswith("PC_")]
        logging.info(f"PCA已启用 (评估)，数值特征为 {len(numeric_features_for_dataset)} 个主成分。")
    else:
        raw_numeric_cols_from_config = config.get('training_dl_regression', {}).get('feature_columns_numeric', [])
        numeric_features_for_dataset = [col for col in input_feature_names if col in raw_numeric_cols_from_config]
        logging.info(f"PCA未启用 (评估)，使用 {len(numeric_features_for_dataset)} 个数值特征。")
        input_scaler_for_dataset = scalers.get('scaler_for_pca', scalers.get('input_scaler'))

    categorical_embedding_info = scalers.get('categorical_embedding_info', {})
    actual_categorical_cols_for_dataset = [col for col in input_feature_names if col in categorical_embedding_info and not col.startswith("PC_")]

    test_dataset = TimeSeriesDataset(
        data_df=test_df.copy(), 
        numeric_feature_cols=numeric_features_for_dataset,
        categorical_feature_cols=actual_categorical_cols_for_dataset,
        target_cols=target_column_names, 
        sequence_length=sequence_length,
        group_id_col=group_id_col,
        date_col='date', 
        input_scaler=input_scaler_for_dataset, 
        target_scaler=target_scaler_obj, 
        fit_scalers=False 
    )
    
    if len(test_dataset) == 0: logging.warning("测试 Dataset 为空。评估跳过。"); return None
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=train_conf.get('num_workers',0))
    logging.info(f"测试 DataLoader 创建完成，包含 {len(test_dataset)} 个样本。")
    scaled_predictions_np, sample_identifiers = predict_on_dataset(model, test_loader, device) 
    if scaled_predictions_np.shape[0] != len(test_dataset): logging.error(f"预测数量 ({scaled_predictions_np.shape[0]}) 与测试样本数量 ({len(test_dataset)}) 不匹配。评估中止。"); return None
    
    scaled_true_targets_list = []
    for i in range(len(test_dataset)): 
        sample_data = test_dataset[i]
        if 'target' not in sample_data or sample_data['target'] is None:
            logging.error(f"测试数据样本 {i} 中缺少 'target'。")
            if scaled_predictions_np.ndim > 1: scaled_true_targets_list.append(np.full(scaled_predictions_np.shape[1:], np.nan))
            else: scaled_true_targets_list.append(np.nan)
            continue
        scaled_true_targets_list.append(sample_data['target'].numpy())

    valid_indices = [idx for idx, arr in enumerate(scaled_true_targets_list) if not (isinstance(arr, float) and np.isnan(arr)) and not (isinstance(arr, np.ndarray) and np.isnan(arr).all())]
    if len(valid_indices) < len(scaled_true_targets_list):
        logging.warning(f"移除了 {len(scaled_true_targets_list) - len(valid_indices)} 个因目标无效而无法评估的样本。")
        scaled_true_targets_list = [scaled_true_targets_list[i] for i in valid_indices]
        scaled_predictions_np = scaled_predictions_np[valid_indices] if len(valid_indices) > 0 else np.array([])
        if scaled_predictions_np.size == 0 and len(scaled_true_targets_list) > 0: 
             logging.error("所有预测因对应无效目标被移除，但仍有真实目标。评估中止。")
             return None
    if not scaled_true_targets_list: logging.error("没有有效的真实目标值用于评估。"); return None
    scaled_true_targets_np = np.array(scaled_true_targets_list)
    if scaled_predictions_np.shape != scaled_true_targets_np.shape: logging.error(f"预测形状 ({scaled_predictions_np.shape}) 与真实目标形状 ({scaled_true_targets_np.shape}) 不匹配。评估中止。"); return None
    
    evaluation_results = {} 
    original_scale_predictions = None
    original_scale_true_targets = None
    if target_scaler_obj: 
        try:
            original_scale_predictions = target_scaler_obj.inverse_transform(scaled_predictions_np)
            original_scale_true_targets = target_scaler_obj.inverse_transform(scaled_true_targets_np)
            logging.info(f"DEBUG_EVAL: original_scale_predictions shape: {original_scale_predictions.shape if original_scale_predictions is not None else 'None'}")
            logging.info(f"DEBUG_EVAL: original_scale_true_targets shape: {original_scale_true_targets.shape if original_scale_true_targets is not None else 'None'}")
        except Exception as e_inv_transform: logging.error(f"逆转换预测值或真实目标值失败: {e_inv_transform}", exc_info=True)
    else: logging.warning("无目标缩放器，原始尺度指标可能不准确或无法计算。")

    all_preds_true_list = []
    if original_scale_predictions is not None and original_scale_true_targets is not None and \
       len(sample_identifiers) >= len(original_scale_predictions) and \
       original_scale_predictions.shape == original_scale_true_targets.shape:
        num_valid_preds = len(original_scale_predictions)
        valid_sample_identifiers = sample_identifiers[:num_valid_preds] if len(sample_identifiers) >= num_valid_preds else []
        for sample_idx in range(num_valid_preds): 
            identifier = valid_sample_identifiers[sample_idx] if valid_sample_identifiers else {'group_id': f'unknown_group_{sample_idx}'}
            for target_idx, target_col_name in enumerate(target_column_names):
                all_preds_true_list.append({
                    'group_id': identifier.get('group_id', f'unknown_group_{sample_idx}'), 
                    'target_column': target_col_name,   
                    'y_true_orig': original_scale_true_targets[sample_idx, target_idx], 
                    'y_pred_orig': original_scale_predictions[sample_idx, target_idx]   
                })
    all_preds_true_df = pd.DataFrame(all_preds_true_list) 
    
    if not target_column_names: logging.warning("DEBUG_EVAL: target_column_names is empty. No per-target metrics will be calculated.")

    for i, target_name in enumerate(target_column_names):
        logging.debug(f"DEBUG_EVAL: Processing metrics for target: {target_name}")
        pred_col_s = scaled_predictions_np[:, i]
        true_col_s = scaled_true_targets_np[:, i]
        target_metrics_scaled = {} 
        if 'mean_squared_error' in metrics_to_calculate: 
            logging.debug(f"DEBUG_EVAL: Calculating mean_squared_error (scaled) for {target_name}")
            try: target_metrics_scaled['mse'] = mean_squared_error(true_col_s, pred_col_s)
            except Exception as e_metric: logging.error(f"Error calculating mse (scaled) for {target_name}: {e_metric}")
        if 'mean_absolute_error' in metrics_to_calculate: 
            logging.debug(f"DEBUG_EVAL: Calculating mean_absolute_error (scaled) for {target_name}")
            try: target_metrics_scaled['mae'] = mean_absolute_error(true_col_s, pred_col_s)
            except Exception as e_metric: logging.error(f"Error calculating mae (scaled) for {target_name}: {e_metric}")
        if 'r2_score' in metrics_to_calculate: 
            logging.debug(f"DEBUG_EVAL: Calculating r2_score (scaled) for {target_name}")
            try: target_metrics_scaled['r2'] = r2_score(true_col_s, pred_col_s)
            except Exception as e_metric: logging.error(f"Error calculating r2 (scaled) for {target_name}: {e_metric}")
        if 'wmape' in metrics_to_calculate: 
            logging.debug(f"DEBUG_EVAL: Calculating wmape (scaled) for {target_name}")
            try: target_metrics_scaled['wmape'] = weighted_mean_absolute_percentage_error(true_col_s, pred_col_s)
            except Exception as e_metric: logging.error(f"Error calculating wmape (scaled) for {target_name}: {e_metric}")
        if 'smape' in metrics_to_calculate: 
            logging.debug(f"DEBUG_EVAL: Calculating smape (scaled) for {target_name}")
            try: target_metrics_scaled['smape'] = symmetric_mean_absolute_percentage_error(true_col_s, pred_col_s)
            except Exception as e_metric: logging.error(f"Error calculating smape (scaled) for {target_name}: {e_metric}")
        evaluation_results[f'{target_name}_scaled_metrics'] = target_metrics_scaled
        logging.info(f"指标 (缩放尺度) for {target_name}: {target_metrics_scaled}")

        if original_scale_predictions is not None and original_scale_true_targets is not None:
            pred_col_o = original_scale_predictions[:, i]
            true_col_o = original_scale_true_targets[:, i]
            target_metrics_original = {} 
            if 'mean_squared_error' in metrics_to_calculate: 
                logging.debug(f"DEBUG_EVAL: Calculating mean_squared_error (original) for {target_name}")
                try: target_metrics_original['mse'] = mean_squared_error(true_col_o, pred_col_o)
                except Exception as e_metric: logging.error(f"Error calculating mse (original) for {target_name}: {e_metric}")
            if 'mean_absolute_error' in metrics_to_calculate: 
                logging.debug(f"DEBUG_EVAL: Calculating mean_absolute_error (original) for {target_name}")
                try: target_metrics_original['mae'] = mean_absolute_error(true_col_o, pred_col_o)
                except Exception as e_metric: logging.error(f"Error calculating mae (original) for {target_name}: {e_metric}")
            if 'r2_score' in metrics_to_calculate: 
                logging.debug(f"DEBUG_EVAL: Calculating r2_score (original) for {target_name}")
                try: target_metrics_original['r2'] = r2_score(true_col_o, pred_col_o)
                except Exception as e_metric: logging.error(f"Error calculating r2 (original) for {target_name}: {e_metric}")
            if 'wmape' in metrics_to_calculate: 
                logging.debug(f"DEBUG_EVAL: Calculating wmape (original) for {target_name}")
                try: target_metrics_original['wmape'] = weighted_mean_absolute_percentage_error(true_col_o, pred_col_o)
                except Exception as e_metric: logging.error(f"Error calculating wmape (original) for {target_name}: {e_metric}")
            if 'smape' in metrics_to_calculate: 
                logging.debug(f"DEBUG_EVAL: Calculating smape (original) for {target_name}")
                try: target_metrics_original['smape'] = symmetric_mean_absolute_percentage_error(true_col_o, pred_col_o)
                except Exception as e_metric: logging.error(f"Error calculating smape (original) for {target_name}: {e_metric}")

            if 'rmsse' in metrics_to_calculate and not all_preds_true_df.empty:
                current_target_df = all_preds_true_df[all_preds_true_df['target_column'] == target_name]
                per_series_rmsse_for_this_target = []
                for group, series_data in current_target_df.groupby('group_id'):
                    scale_sq = rmsse_scales_dict.get(str(group), rmsse_scales_dict.get(group)) 
                    if scale_sq is not None and scale_sq > 1e-9: 
                        try:
                            mse_series = mean_squared_error(series_data['y_true_orig'], series_data['y_pred_orig'])
                            rmsse_s = np.sqrt(mse_series / scale_sq)
                            if not np.isnan(rmsse_s): per_series_rmsse_for_this_target.append(rmsse_s)
                        except Exception as e_rmsse: logging.error(f"Error calculating RMSSE for series {group}, target {target_name}: {e_rmsse}")
                if per_series_rmsse_for_this_target: target_metrics_original['rmsse'] = np.nanmean(per_series_rmsse_for_this_target)
            evaluation_results[f'{target_name}_original_metrics'] = target_metrics_original
            logging.info(f"指标 (原始尺度) for {target_name}: {target_metrics_original}")
    
    if 'rmsse' in metrics_to_calculate: 
        all_rmsse_values_across_targets = [] 
        for target_name_iter in target_column_names:
            rmsse_val = evaluation_results.get(f'{target_name_iter}_original_metrics', {}).get('rmsse')
            if rmsse_val is not None and not np.isnan(rmsse_val): all_rmsse_values_across_targets.append(rmsse_val)
        if all_rmsse_values_across_targets: 
            evaluation_results['overall_mean_rmsse_per_target'] = np.nanmean(all_rmsse_values_across_targets)
            logging.info(f"总体平均目标RMSSE (各目标RMSSE的均值): {evaluation_results['overall_mean_rmsse_per_target']:.4f}")
        else: logging.warning("DEBUG_EVAL: No valid RMSSE values found across targets for overall_mean_rmsse_per_target.")

    if 'wrmsse' in metrics_to_calculate and not all_preds_true_df.empty and rmsse_scales_dict and wrmsse_weights_dict:
        rmsse_sum_weighted_sq_numerator = 0.0; total_weights_sum_denominator = 0.0    
        sales_related_targets = [t for t in target_column_names if t.startswith(rmsse_base_sales_col + "_lead")]
        logging.info(f"DEBUG_EVAL: Sales related targets for WRMSSE: {sales_related_targets}")
        if sales_related_targets:
            relevant_preds_true_df = all_preds_true_df[all_preds_true_df['target_column'].isin(sales_related_targets)]
            if relevant_preds_true_df.empty: logging.warning("DEBUG_EVAL: relevant_preds_true_df for WRMSSE is empty.")
            for group_id_key, group_data in relevant_preds_true_df.groupby('group_id'):
                scale_sq = rmsse_scales_dict.get(str(group_id_key), rmsse_scales_dict.get(group_id_key))
                weight = wrmsse_weights_dict.get(str(group_id_key), wrmsse_weights_dict.get(group_id_key, 0.0))
                if scale_sq is None or scale_sq < 1e-9 or weight < 1e-9: logging.debug(f"DEBUG_EVAL: Skipping group {group_id_key} for WRMSSE due to invalid scale ({scale_sq}) or weight ({weight})."); continue
                try:
                    y_true_g = group_data['y_true_orig'].values; y_pred_g = group_data['y_pred_orig'].values
                    mse_g = mean_squared_error(y_true_g, y_pred_g) 
                    rmsse_g_sq = mse_g / scale_sq 
                    rmsse_sum_weighted_sq_numerator += weight * rmsse_g_sq 
                    total_weights_sum_denominator += weight  
                except Exception as e_wrmsse_calc: logging.error(f"Error calculating WRMSSE component for group {group_id_key}: {e_wrmsse_calc}")
            if total_weights_sum_denominator > 1e-9: 
                wrmsse = np.sqrt(rmsse_sum_weighted_sq_numerator / total_weights_sum_denominator)
                evaluation_results['overall_wrmsse'] = wrmsse
                logging.info(f"总体 WRMSSE (基于 '{rmsse_base_sales_col}' 相关目标): {wrmsse:.4f}")
            else: logging.warning("无法计算 WRMSSE (总权重为零或无有效序列组件)。")
        else: logging.warning(f"未找到与 '{rmsse_base_sales_col}' 相关的目标列用于计算WRMSSE。")

    logging.info("模型评估完成。")
    return evaluation_results
