# product_lifecycle_predictor_vA/src/models/trainer_reg.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler 
import pandas as pd
import numpy as np
import logging
import os
import copy 
import joblib 
import json 
from typing import Dict, Any, List, Tuple, Optional

from data.dataset_reg import TimeSeriesDataset 
from models.builders_reg import build_lstm_regressor 


def train_regression_model(
    features_df: pd.DataFrame, 
    input_feature_names: List[str],
    target_column_names: List[str], 
    config: Dict[str, Any],
    pca_is_enabled_in_main: bool = False 
) -> Tuple[Optional[nn.Module], Optional[Dict[str,List[float]]], Optional[Dict[str,Any]]]:
    logging.info("开始回归模型训练流程...")
    if features_df.empty: logging.error("输入的 features_df 为空。"); return None, None, None
    if not input_feature_names: logging.error("输入特征列表为空。"); return None, None, None
    if not target_column_names: logging.error("目标列列表为空。"); return None, None, None

    train_conf = config.get('training_dl_regression', {})
    data_conf = config.get('data', {})
    fe_conf = config.get('feature_engineering', {}) 
    eval_conf = config.get('evaluation_regression', {}) 
    group_id_col = fe_conf.get('group_id_col','id') 
    date_col = 'date' 

    rmsse_base_sales_col = eval_conf.get('rmsse_base_sales_col', 'sales_W') 
    wrmsse_weight_col = eval_conf.get('wrmsse_weight_column', rmsse_base_sales_col) 
    wrmsse_weight_period_weeks = eval_conf.get('wrmsse_weight_period_weeks', 4)

    device = torch.device(train_conf.get('device', 'cuda') if torch.cuda.is_available() else 'cpu')
    logging.info(f"使用设备: {device}")

    sequence_length = train_conf.get('sequence_length', 12)
    batch_size = train_conf.get('batch_size', 64)
    epochs = train_conf.get('epochs', 50)
    learning_rate = train_conf.get('learning_rate', 0.001)
    patience = train_conf.get('early_stopping_patience', 10)
    
    categorical_cols_from_config = config.get('training_dl_regression', {}).get('feature_columns_categorical', [])
    actual_categorical_cols = [col for col in input_feature_names if col in categorical_cols_from_config and not col.startswith("PC_")]
    
    categorical_embedding_info = {}
    if actual_categorical_cols:
        embedding_dims_config = train_conf.get('embedding_dims', {})
        valid_cats_for_emb_info = []
        for cat_col in actual_categorical_cols:
            if cat_col not in features_df.columns:
                logging.warning(f"配置的类别特征 '{cat_col}' 不在 features_df 中，跳过。")
                continue
            try:
                if pd.api.types.is_numeric_dtype(features_df[cat_col]):
                    vocab_size = int(features_df[cat_col].max() + 1) if not features_df[cat_col].empty else 0
                else:
                    codes = features_df[cat_col].astype('category').cat.codes
                    vocab_size = int(codes.max() + 1) if not codes.empty else 0
                
                if vocab_size <= 0 : 
                    logging.warning(f"类别特征 '{cat_col}' 计算得到的 vocab_size ({vocab_size}) 无效。跳过此特征。")
                    continue
                emb_dim = embedding_dims_config.get(cat_col, 10) 
                categorical_embedding_info[cat_col] = {'vocab_size': vocab_size, 'embedding_dim': emb_dim}
                valid_cats_for_emb_info.append(cat_col)
            except Exception as e:
                logging.error(f"计算类别特征 '{cat_col}' 的 vocab_size 失败: {e}. 跳过此特征。")
        actual_categorical_cols = valid_cats_for_emb_info
        logging.info(f"准备好的类别特征嵌入信息: {categorical_embedding_info}")

    if date_col not in features_df.columns:
        logging.error(f"日期列 '{date_col}' 不在 features_df 中。")
        return None, None, None
    
    features_df[date_col] = pd.to_datetime(features_df[date_col])
    
    val_split_date_str = train_conf.get('validation_split_date')
    train_df_full = pd.DataFrame() 
    val_df = pd.DataFrame()

    if not val_split_date_str:
        unique_ids = features_df[group_id_col].unique()
        if len(unique_ids) < 2 :
             logging.warning(f"唯一ID数量 ({len(unique_ids)}) 过少，无法进行基于ID的随机划分。将使用所有数据进行训练，不设验证集。")
             train_df_full = features_df.copy()
             val_df = pd.DataFrame(columns=features_df.columns) 
        elif len(unique_ids) < 5 : 
             logging.warning(f"唯一ID数量 ({len(unique_ids)}) 较少。尝试按时间划分80/20。")
             df_to_split = features_df.sort_values(by=[group_id_col, date_col]).reset_index(drop=True) 
             split_point = int(len(df_to_split) * 0.8)
             train_df_full = df_to_split.iloc[:split_point].copy()
             val_df = df_to_split.iloc[split_point:].copy()
        else:
            train_ids, val_ids = train_test_split(unique_ids, test_size=0.2, random_state=42)
            train_df_full = features_df[features_df[group_id_col].isin(train_ids)].copy()
            val_df = features_df[features_df[group_id_col].isin(val_ids)].copy()
    else: 
        val_split_date = pd.to_datetime(val_split_date_str)
        train_df_full = features_df[features_df[date_col] < val_split_date].copy()
        val_df = features_df[features_df[date_col] >= val_split_date].copy()
            
    logging.info(f"训练集 (用于统计和训练) 大小: {len(train_df_full)}, 验证集大小: {len(val_df)}")

    if train_df_full.empty:
        logging.error("训练集为空，无法继续训练。")
        return None, None, None
    if val_df.empty:
        logging.warning("验证集为空。模型将仅在训练集上训练和评估（如果适用），早停和学习率调度器可能不起作用。")

    rmsse_scales = {}
    wrmsse_weights = {}
    base_sales_col_for_stats = rmsse_base_sales_col
    if base_sales_col_for_stats not in train_df_full.columns:
        logging.warning(f"用于RMSSE/WRMSSE的基准销售列 '{base_sales_col_for_stats}' 不在训练数据中。将尝试使用 'sales' 列（如果存在）。")
        if 'sales' in train_df_full.columns: base_sales_col_for_stats = 'sales'
        else: logging.error(f"也未找到 'sales' 列。RMSSE/WRMSSE统计量可能不准确。"); 
    
    if base_sales_col_for_stats in train_df_full.columns:
        for group, group_data in train_df_full.groupby(group_id_col):
            group_data_sorted = group_data.sort_values(by=date_col)
            series_for_scale = group_data_sorted[base_sales_col_for_stats].astype(float) 
            if len(series_for_scale) >= 2: 
                diffs = series_for_scale.diff(1).iloc[1:] 
                if not diffs.empty and not diffs.isnull().all():
                     scale_denominator_sq = np.mean(np.square(diffs.dropna())) 
                     rmsse_scales[group] = float(scale_denominator_sq) if scale_denominator_sq > 1e-12 else 1e-12 
                else: rmsse_scales[group] = 1.0 
            else: rmsse_scales[group] = 1.0 
            
            current_weight_col_to_use = wrmsse_weight_col if wrmsse_weight_col in group_data_sorted.columns else base_sales_col_for_stats
            if current_weight_col_to_use in group_data_sorted.columns:
                 recent_sales = group_data_sorted[current_weight_col_to_use].tail(wrmsse_weight_period_weeks)
                 wrmsse_weights[group] = float(recent_sales.abs().sum())
                 if wrmsse_weights[group] < 1e-9: wrmsse_weights[group] = 1e-9
            else: wrmsse_weights[group] = 1.0 
    logging.info(f"为 {len(rmsse_scales)} 个序列计算了RMSSE缩放分母。")
    logging.info(f"为 {len(wrmsse_weights)} 个序列计算了WRMSSE权重。")

    numeric_features_for_dataset = [f for f in input_feature_names if f.startswith("PC_")] if pca_is_enabled_in_main else \
                                   [col for col in input_feature_names if col in config.get('training_dl_regression', {}).get('feature_columns_numeric', [])]

    input_scaler_for_dataset = None if pca_is_enabled_in_main else StandardScaler()
    target_scaler = StandardScaler() 

    train_dataset = TimeSeriesDataset(
        data_df=train_df_full, 
        numeric_feature_cols=numeric_features_for_dataset,
        categorical_feature_cols=actual_categorical_cols,
        target_cols=target_column_names,
        sequence_length=sequence_length,
        group_id_col=group_id_col,
        date_col=date_col, 
        input_scaler=input_scaler_for_dataset, 
        target_scaler=target_scaler, 
        fit_scalers=True 
    )
    
    val_loader = None
    if not val_df.empty:
        val_dataset = TimeSeriesDataset(
            data_df=val_df, 
            numeric_feature_cols=numeric_features_for_dataset,
            categorical_feature_cols=actual_categorical_cols,
            target_cols=target_column_names,
            sequence_length=sequence_length,
            group_id_col=group_id_col, 
            date_col=date_col, 
            input_scaler=train_dataset.input_scaler, 
            target_scaler=train_dataset.target_scaler, 
            fit_scalers=False
        )
        if len(val_dataset) > 0:
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=train_conf.get('num_workers',0), pin_memory=True if device.type=='cuda' else False)
        else:
            logging.warning("验证集 Dataset 为空（即使val_df不为空）。")
    else:
        logging.info("验证集DataFrame为空，不创建验证DataLoader。")

    logging.info(f"训练集 DataLoader 创建完成。训练样本数: {len(train_dataset)}")
    if val_loader: logging.info(f"验证集 DataLoader 创建完成。验证样本数: {len(val_dataset)}")

    num_model_numerical_features = len(numeric_features_for_dataset)
    num_outputs = len(target_column_names)

    model = build_lstm_regressor( 
        config=config, 
        num_numerical_features=num_model_numerical_features,
        num_outputs=num_outputs,
        categorical_embedding_info=categorical_embedding_info 
    )
    model.to(device)
    
    loss_function_name = train_conf.get('loss_function', 'MSELoss')
    if loss_function_name == 'MSELoss': criterion = nn.MSELoss()
    elif loss_function_name == 'L1Loss': criterion = nn.L1Loss()
    else: raise ValueError(f"不支持的损失函数: {loss_function_name}")
    
    optimizer_name = train_conf.get('optimizer', 'Adam')
    if optimizer_name == 'Adam': optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'AdamW': optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    else: raise ValueError(f"不支持的优化器: {optimizer_name}")

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=max(1, patience // 2 -1), factor=0.1, verbose=False) 
    logging.info(f"使用损失函数: {loss_function_name}, 优化器: {optimizer_name} (lr={learning_rate})")

    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    training_history = {'train_loss': [], 'val_loss': [], 'lr': []}

    logging.info(f"开始训练模型，共 {epochs} 个 epochs...")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=train_conf.get('num_workers',0), pin_memory=True if device.type=='cuda' else False)

    for epoch in range(epochs):
        model.train() 
        running_train_loss = 0.0
        for i, batch in enumerate(train_loader):
            numerical_feats = batch['numerical_features'].to(device)
            categorical_feats_batch = {k: v.to(device) for k, v in batch['categorical_features'].items() if k in categorical_embedding_info}
            targets = batch['target'].to(device)
            optimizer.zero_grad()
            outputs = model(numerical_features=numerical_feats, categorical_features=categorical_feats_batch)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * targets.size(0) 
        epoch_train_loss = running_train_loss / len(train_loader.dataset) if len(train_loader.dataset) > 0 else 0.0

        epoch_val_loss = float('inf') 
        if val_loader and len(val_loader.dataset) > 0 :
            model.eval() 
            running_val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    numerical_feats = batch['numerical_features'].to(device)
                    categorical_feats_batch = {k: v.to(device) for k, v in batch['categorical_features'].items() if k in categorical_embedding_info}
                    targets = batch['target'].to(device)
                    outputs = model(numerical_features=numerical_feats, categorical_features=categorical_feats_batch)
                    loss = criterion(outputs, targets)
                    running_val_loss += loss.item() * targets.size(0)
            epoch_val_loss = running_val_loss / len(val_loader.dataset)
        
        current_lr = optimizer.param_groups[0]['lr']
        val_loss_log_str = f"{epoch_val_loss:.6f}" if epoch_val_loss != float('inf') else "N/A (无验证集)"
        logging.info(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {epoch_train_loss:.6f} | Val Loss: {val_loss_log_str} | LR: {current_lr:.7f}")
        
        training_history['train_loss'].append(float(epoch_train_loss)) 
        training_history['val_loss'].append(float(epoch_val_loss))    
        training_history['lr'].append(float(current_lr))             

        if val_loader and len(val_loader.dataset) > 0:
            old_lr = optimizer.param_groups[0]['lr']
            scheduler.step(epoch_val_loss) 
            if optimizer.param_groups[0]['lr'] < old_lr:
                logging.info(f"  学习率从 {old_lr:.7f} 降低到 {optimizer.param_groups[0]['lr']:.7f}")

            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                epochs_no_improve = 0
                if train_conf.get('save_best_model', True): 
                    best_model_state = copy.deepcopy(model.state_dict())
                    logging.info(f"  验证损失改善，保存最佳模型状态。Best Val Loss: {best_val_loss:.6f}")
            else:
                epochs_no_improve += 1
                logging.info(f"  验证损失未改善 {epochs_no_improve}轮。")

            if epochs_no_improve >= patience:
                logging.info(f"触发早停！在 {patience} 轮内验证损失未改善。")
                break
        elif epoch + 1 == epochs : 
             if train_conf.get('save_best_model', True): 
                 best_model_state = copy.deepcopy(model.state_dict())
                 logging.info(f"已达到最大epoch且无验证集，保存当前模型状态。")

    if best_model_state and train_conf.get('save_best_model', True):
        model.load_state_dict(best_model_state)
        logging.info("已加载早停（或最终epoch）得到的最佳模型权重。")
    
    scalers_and_mappings_to_return = {
        'input_scaler': train_dataset.input_scaler, 
        'target_scaler': train_dataset.target_scaler,
        'categorical_embedding_info': categorical_embedding_info,
        'rmsse_scales': rmsse_scales,     
        'wrmsse_weights': wrmsse_weights
    }
    
    if train_dataset.input_scaler is not None:
        input_scaler_path = data_conf.get('scaler_input_path') 
        if input_scaler_path: 
            os.makedirs(os.path.dirname(input_scaler_path), exist_ok=True)
            joblib.dump(train_dataset.input_scaler, input_scaler_path)
            logging.info(f"TimeSeriesDataset内部的输入缩放器已保存到: {input_scaler_path}")
    
    target_scaler_path = data_conf.get('scaler_target_path')
    if target_scaler_path and scalers_and_mappings_to_return.get('target_scaler'): 
        os.makedirs(os.path.dirname(target_scaler_path), exist_ok=True)
        joblib.dump(scalers_and_mappings_to_return['target_scaler'], target_scaler_path)
        logging.info(f"目标缩放器已保存到: {target_scaler_path}")
        
    rmsse_scales_path = data_conf.get('rmsse_scales_path')
    wrmsse_weights_path = data_conf.get('wrmsse_weights_path')
    try:
        if rmsse_scales and rmsse_scales_path: 
            os.makedirs(os.path.dirname(rmsse_scales_path), exist_ok=True)
            serializable_rmsse_scales = {str(k): float(v) if isinstance(v, (np.generic, float, int)) else v for k, v in rmsse_scales.items()}
            with open(rmsse_scales_path, 'w') as f: json.dump(serializable_rmsse_scales, f, indent=4)
            logging.info(f"RMSSE scales 已保存到: {rmsse_scales_path}")
        if wrmsse_weights and wrmsse_weights_path: 
            os.makedirs(os.path.dirname(wrmsse_weights_path), exist_ok=True)
            serializable_wrmsse_weights = {str(k): float(v) if isinstance(v, (np.generic, float, int)) else v for k, v in wrmsse_weights.items()}
            with open(wrmsse_weights_path, 'w') as f: json.dump(serializable_wrmsse_weights, f, indent=4)
            logging.info(f"WRMSSE weights 已保存到: {wrmsse_weights_path}")
    except Exception as e_save_stats:
        logging.error(f"保存RMSSE/WRMSSE统计量失败: {e_save_stats}")

    logging.info("模型训练流程完成。")
    return model, training_history, scalers_and_mappings_to_return
