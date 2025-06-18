# src/feature_engineering/builder.py
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Tuple

# 注意：这里已移除任何不必要的 "from ..utils.config import ..." 相对导入语句。
# 主流程 main.py 会将 config 对象作为参数传递给 generate_features_and_targets 函数。

GROWTH_CLIP_VALUE = 100.0

def _calculate_ewma(series: pd.Series, span: int) -> pd.Series:
    """计算指数加权移动平均"""
    if series.empty or span <= 0:
        return pd.Series([np.nan] * len(series), index=series.index)
    return series.ewm(span=span, adjust=False, min_periods=1).mean().fillna(0)

def _calculate_growth(series: pd.Series, window: int, period: int) -> pd.Series:
    """计算增长率"""
    if series.empty or window <= 0 or period <= 0:
        return pd.Series([np.nan] * len(series), index=series.index)
    ma = series.rolling(window=window, min_periods=1).mean()
    ma_shifted = ma.shift(period)
    growth = (ma - ma_shifted) / ma_shifted.replace(0, np.nan)
    growth = growth.replace([np.inf, -np.inf], np.nan)
    growth = growth.clip(lower=-GROWTH_CLIP_VALUE, upper=GROWTH_CLIP_VALUE)
    return growth.fillna(0)

def _calculate_volatility(series: pd.Series, window: int) -> pd.Series:
    """计算波动率（标准差）"""
    if series.empty or window <= 0:
        return pd.Series([np.nan] * len(series), index=series.index)
    volatility = series.rolling(window=window, min_periods=1).std()
    return volatility.fillna(0)

def generate_features_and_targets(processed_data: pd.DataFrame, config: Dict[str, Any]) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    基于预处理后的数据，生成所有用于模型训练的特征和目标变量。
    """
    logging.info("开始生成特征和目标变量 (包含产品分段特征)...")
    if processed_data.empty:
        logging.warning("输入 processed_data 为空，无法生成特征。")
        return pd.DataFrame(), [], []

    fe_config = config.get('feature_engineering', {})
    training_config = config.get('training_dl_regression', {}) # 从主回归模型配置中获取特征列定义
    df = processed_data.sort_values(by=['id', 'date']).copy()

    source_metrics_config = fe_config.get('target_continuous_metrics', {})
    smoothing_span_sales = fe_config.get('smoothing_span_sales', 4)
    clip_ratio_value = fe_config.get('clip_ratio_value', 10.0)

    # 产品分段相关配置
    segmentation_config = fe_config.get('product_segmentation', {})
    enable_segmentation = segmentation_config.get('enable', False)
    sales_level_window = segmentation_config.get('sales_level_window', 26)
    volatility_level_window = segmentation_config.get('volatility_level_window', 12)
    sales_qcut_bins = segmentation_config.get('sales_qcut_bins', [0, 0.33, 0.66, 1.0])
    sales_qcut_labels = segmentation_config.get('sales_qcut_labels', ['LowSales', 'MidSales', 'HighSales'])
    vol_qcut_bins = segmentation_config.get('vol_qcut_bins', [0, 0.5, 1.0])
    vol_qcut_labels = segmentation_config.get('vol_qcut_labels', ['Stable', 'Volatile'])

    metrics_to_generate_derivatives_for = []
    calculated_metrics_successfully = {}

    # --- Pass 1: 计算基础指标、平滑指标、增长率、波动率 ---
    # 这些指标名称应该与 config.yaml -> feature_engineering -> target_continuous_metrics 中的键名对应
    metrics_calc_order = [
        'sales_W', 'price_W_mean',
        'smoothed_sales_W',
        'sales_W_volatility_4w', # 假设这个波动率是4周窗口
        'sales_W_growth_4w_1p', # 假设这个增长率是基于4周均值，周期为1
        'smoothed_sales_W_growth_4w_1p',
    ]

    for metric_name in metrics_calc_order:
        if metric_name not in source_metrics_config: continue # 只处理在配置中定义的指标
        logging.debug(f"尝试计算或获取基础指标: {metric_name}")
        column_generated_or_found = False
        if metric_name == 'sales_W':
            if 'sales' not in df.columns: logging.error("'sales' 列未在 processed_data 中找到 ('sales_W' 指标)。"); continue
            df[metric_name] = df['sales']
            column_generated_or_found = True
        elif metric_name == 'price_W_mean':
            if 'sell_price' not in df.columns: logging.error("'sell_price' 列未在 processed_data 中找到 ('price_W_mean' 指标)。"); continue
            df[metric_name] = df['sell_price']
            column_generated_or_found = True
        elif metric_name == 'smoothed_sales_W':
            if 'sales_W' not in df.columns: logging.error(f"基础列 'sales_W' 未找到，无法计算 '{metric_name}'。"); continue
            df[metric_name] = df.groupby('id')['sales_W'].transform(lambda x: _calculate_ewma(x, span=smoothing_span_sales))
            column_generated_or_found = True
        elif metric_name == 'sales_W_volatility_4w': # 明确窗口为4周
            if 'sales_W' not in df.columns: logging.error(f"基础列 'sales_W' 未找到，无法计算 '{metric_name}'。"); continue
            df[metric_name] = df.groupby('id')['sales_W'].transform(lambda x: _calculate_volatility(x, window=4))
            column_generated_or_found = True
        elif metric_name == 'sales_W_growth_4w_1p': # 明确窗口和周期
            if 'sales_W' not in df.columns: logging.error(f"基础列 'sales_W' 未找到，无法计算 '{metric_name}'。"); continue
            df[metric_name] = df.groupby('id')['sales_W'].transform(lambda x: _calculate_growth(x, window=4, period=1))
            column_generated_or_found = True
        elif metric_name == 'smoothed_sales_W_growth_4w_1p': # 明确窗口和周期
            if 'smoothed_sales_W' not in df.columns: logging.error(f"基础列 'smoothed_sales_W' 未找到，无法计算 '{metric_name}'。"); continue
            df[metric_name] = df.groupby('id')['smoothed_sales_W'].transform(lambda x: _calculate_growth(x, window=4, period=1))
            column_generated_or_found = True
        else:
            # 处理其他在 source_metrics_config 中定义但未在上面明确列出的指标
            # 假设这些指标如果存在，应该已经由 loader.py 生成并存在于 df 中
            if metric_name in df.columns:
                column_generated_or_found = True
            elif not metric_name.startswith("item_div_"): # item_div_* 是后面计算的
                logging.warning(f"指标 '{metric_name}' 在 source_metrics_config 中定义，但在 metrics_calc_order 中没有明确计算逻辑，且不存在于输入数据中。")
                continue # 跳过这个无法处理的指标

        if column_generated_or_found:
            calculated_metrics_successfully[metric_name] = True
            if metric_name not in metrics_to_generate_derivatives_for:
                metrics_to_generate_derivatives_for.append(metric_name)
            logging.info(f"已准备/计算源指标 (用于衍生特征): {metric_name}")
        else:
            # 如果到这里 column_generated_or_found 仍然是 False，说明指标处理失败
            logging.error(f"基础指标 '{metric_name}' 计算失败或未找到。")
            calculated_metrics_successfully[metric_name] = False

    # --- Pass 2: 计算相对层级特征 ---
    # 例如 item_div_dept_sales_W, item_div_store_sales_W
    if 'sales_W' in df.columns and calculated_metrics_successfully.get('sales_W', False):
        # 假设层级总销量列 (如 dept_sales_sum_W, store_sales_sum_W) 已由 loader.py 添加到 df 中
        hier_sums_cols_map = {
            'item_div_dept_sales_W': 'dept_sales_sum_W', # 假设这是部门总销量的列名
            'item_div_store_sales_W': 'store_sales_sum_W', # 假设这是店铺总销量的列名
            # 可以根据需要添加更多层级
        }
        for ratio_col_name, sum_col_name in hier_sums_cols_map.items():
            if ratio_col_name in source_metrics_config: # 仅当配置中需要此指标时才计算
                if sum_col_name in df.columns:
                    df[ratio_col_name] = (df['sales_W'] / (df[sum_col_name] + 1e-9)) # 加一个小的epsilon避免除以零
                    df[ratio_col_name] = df[ratio_col_name].replace([np.inf, -np.inf], np.nan).fillna(0)
                    df[ratio_col_name] = df[ratio_col_name].clip(lower=-clip_ratio_value, upper=clip_ratio_value) # 裁剪极端值
                    if ratio_col_name not in metrics_to_generate_derivatives_for:
                        metrics_to_generate_derivatives_for.append(ratio_col_name)
                    logging.info(f"已计算相对层级特征: {ratio_col_name}")
                    calculated_metrics_successfully[ratio_col_name] = True
                else:
                    logging.warning(f"计算相对层级特征 '{ratio_col_name}' 失败: 基础总和列 '{sum_col_name}' 未在DataFrame中找到。")
                    calculated_metrics_successfully[ratio_col_name] = False # 标记为失败
    else:
        logging.warning("'sales_W' 列未成功计算或找到，无法计算相对层级特征。")

    logging.info(f"将为以下成功计算/获取并配置的源指标生成滞后和滚动特征: {metrics_to_generate_derivatives_for}")

    # --- Pass 3: 生成滞后和滚动特征 ---
    lag_config = fe_config.get('weekly_lags_input', [])
    for metric_to_lag in metrics_to_generate_derivatives_for:
        if not calculated_metrics_successfully.get(metric_to_lag, False): continue # 跳过计算失败的指标
        for lag in lag_config:
            feature_col_name = f'{metric_to_lag}_lag_{lag}'
            df[feature_col_name] = df.groupby('id')[metric_to_lag].shift(lag)
            logging.debug(f"创建滞后特征: {feature_col_name}")

    rolling_window_config = fe_config.get('weekly_rolling_windows_input', [])
    rolling_stats_config = fe_config.get('weekly_rolling_stats_input', ['mean']) # 默认为均值

    for metric_to_roll in metrics_to_generate_derivatives_for:
        if not calculated_metrics_successfully.get(metric_to_roll, False): continue # 跳过计算失败的指标
        for window in rolling_window_config:
            for stat in rolling_stats_config:
                feature_col_name = f'{metric_to_roll}_roll_{stat}_{window}w'
                if stat == 'mean':
                    df[feature_col_name] = df.groupby('id')[metric_to_roll].transform(
                        lambda x: x.rolling(window=window, min_periods=1).mean()
                    )
                elif stat == 'std':
                    df[feature_col_name] = df.groupby('id')[metric_to_roll].transform(
                        lambda x: x.rolling(window=window, min_periods=1).std()
                    )
                # 可以根据需要添加更多统计量，如 'median', 'sum', 'min', 'max'
                logging.debug(f"创建滚动特征: {feature_col_name}")

    # --- Pass 4: 创建产品分段特征 (Product Segmentation) ---
    if enable_segmentation:
        logging.info("开始创建产品分段特征...")
        if 'sales_W' not in df.columns: # 依赖周销量
            logging.error("'sales_W'列不存在，无法进行产品分段。")
        else:
            # 计算用于分段的销量水平和波动性水平
            df['avg_sales_level_temp'] = df.groupby('id')['sales_W'].transform(
                lambda x: x.rolling(window=sales_level_window, min_periods=max(1, sales_level_window // 2)).mean()
            )
            df['product_sales_level'] = df['avg_sales_level_temp'].fillna(0)

            # 使用配置的波动率窗口计算波动性
            volatility_col_for_segment = f'sales_W_volatility_{volatility_level_window}w'
            if volatility_col_for_segment not in df.columns: # 如果之前没有计算过这个特定窗口的波动率
                 df[volatility_col_for_segment] = df.groupby('id')['sales_W'].transform(
                     lambda x: _calculate_volatility(x, window=volatility_level_window)
                 )
            df['product_volatility_level'] = df[volatility_col_for_segment].fillna(0)

            # 对销量水平进行分箱
            try:
                df['sales_segment'] = pd.qcut(df['product_sales_level'], q=sales_qcut_bins, labels=sales_qcut_labels, duplicates='drop')
                logging.info(f"销量分段完成。各段分布:\n{df['sales_segment'].value_counts(dropna=False)}")
            except Exception as e_qcut_sales:
                logging.error(f"销量分位数切割失败: {e_qcut_sales}. 将尝试使用固定阈值（例如中位数）。")
                sales_median = df['product_sales_level'].median()
                df['sales_segment'] = pd.cut(df['product_sales_level'],
                                             bins=[-np.inf, sales_median, np.inf],
                                             labels=[sales_qcut_labels[0], sales_qcut_labels[-1]], # 使用首尾标签
                                             right=False)

            # 对波动性水平进行分箱
            try:
                df['volatility_segment'] = pd.qcut(df['product_volatility_level'], q=vol_qcut_bins, labels=vol_qcut_labels, duplicates='drop')
                logging.info(f"波动性分段完成。各段分布:\n{df['volatility_segment'].value_counts(dropna=False)}")
            except Exception as e_qcut_vol:
                logging.error(f"波动性分位数切割失败: {e_qcut_vol}. 将尝试使用固定阈值（例如中位数）。")
                vol_median = df['product_volatility_level'].median()
                df['volatility_segment'] = pd.cut(df['product_volatility_level'],
                                                  bins=[-np.inf, vol_median, np.inf],
                                                  labels=[vol_qcut_labels[0], vol_qcut_labels[-1]], # 使用首尾标签
                                                  right=False)

            # 组合分段创建最终的 product_segment 特征
            df['product_segment'] = df['sales_segment'].astype(str) + "_" + df['volatility_segment'].astype(str)
            df['product_segment'] = df['product_segment'].astype('category').cat.codes # 转换为数值编码

            logging.info(f"产品分段特征 'product_segment' 创建完成。")
            # 清理临时列
            df.drop(columns=['avg_sales_level_temp', 'product_sales_level',
                             'product_volatility_level', 'sales_segment', 'volatility_segment',
                             volatility_col_for_segment], # 确保清理的是用于分段的波动率列
                    errors='ignore', inplace=True)

        # 确保 product_segment 列存在，即使创建失败也用默认值填充
        if 'product_segment' not in df.columns:
            logging.warning("产品分段特征未能成功创建，将使用默认值0填充。")
            df['product_segment'] = 0


    # --- Pass 5: 时间特征 ---
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df['weekofyear'] = df['date'].dt.isocalendar().week.astype(float)
        df['month'] = df['date'].dt.month.astype(float)
        df['year'] = df['date'].dt.year.astype(float)
        df['weekofyear_sin'] = np.sin(2 * np.pi * df['weekofyear'] / 52.0)
        df['weekofyear_cos'] = np.cos(2 * np.pi * df['weekofyear'] / 52.0)
        df['time_idx'] = df.groupby('id').cumcount() # 每个时间序列内的时间索引
        logging.info("时间相关特征创建完成。")
    else:
        logging.error("'date' 列未在DataFrame中找到，无法创建时间特征。")


    # --- Pass 6: 事件交互特征 ---
    # 确保事件计数特征 (如 num_days_any_event_in_week) 和基础销售特征 (如 sales_W_roll_mean_4w) 已存在
    logging.info("开始创建事件交互特征...")
    interaction_pairs = [
        ('num_days_any_event_in_week', 'sales_W_roll_mean_4w', 'event_any_X_sales_roll_mean_4w'),
        ('num_days_any_event_in_week', 'sales_W_volatility_4w', 'event_any_X_sales_volatility_4w'),
        # 可以为不同州的SNAP日添加交互特征，例如 CA
        ('num_snap_ca_days_in_week', 'sales_W_roll_mean_4w', 'snap_ca_X_sales_roll_mean_4w'),
        ('num_snap_ca_days_in_week', 'sales_W_volatility_4w', 'snap_ca_X_sales_volatility_4w')
        # 根据需要添加更多州的SNAP交互
    ]
    for event_col, sales_char_col, new_interact_col in interaction_pairs:
        if event_col in df.columns and sales_char_col in df.columns:
            # 确保参与乘法的列是数值类型
            if not pd.api.types.is_numeric_dtype(df[sales_char_col]):
                df[sales_char_col] = pd.to_numeric(df[sales_char_col], errors='coerce').fillna(0)
            if not pd.api.types.is_numeric_dtype(df[event_col]): # 事件计数也应是数值
                df[event_col] = pd.to_numeric(df[event_col], errors='coerce').fillna(0)

            df[new_interact_col] = df[event_col] * df[sales_char_col]
            logging.info(f"已创建事件交互特征: {new_interact_col}")
        else:
            missing_cols = [col for col in [event_col, sales_char_col] if col not in df.columns]
            logging.warning(f"创建交互特征 '{new_interact_col}' 失败: 依赖列 {missing_cols} 未找到。")


    # --- Pass 7: 生成目标变量 和 清理/选择最终列 ---
    prediction_horizon = fe_config.get('prediction_horizon', 1)
    target_column_names = []

    for metric_name, metric_conf_details in source_metrics_config.items():
        if metric_conf_details.get('as_target'): # 检查配置是否将此指标用作目标
            if not calculated_metrics_successfully.get(metric_name, False) and metric_name not in df.columns :
                logging.warning(f"用作目标的基础源列 '{metric_name}' 未能成功计算/获取。跳过为其生成lead目标。")
                continue
            for h in range(1, prediction_horizon + 1):
                target_col_name = f'{metric_name}_lead{h}'
                df[target_col_name] = df.groupby('id')[metric_name].shift(-h) # shift负值表示未来期数
                target_column_names.append(target_col_name)
                logging.debug(f"创建目标变量: {target_col_name}")

    if not target_column_names:
        logging.warning("未生成任何目标列。请检查config中 'as_target: true' 的指标是否能被成功计算。")
    else:
        logging.info(f"目标变量列: {target_column_names}")

    original_rows = len(df)
    if target_column_names: # 仅当有目标列时才移除NaN
        df.dropna(subset=target_column_names, inplace=True) # 移除包含NaN目标值的行
    rows_removed = original_rows - len(df)
    logging.info(f"因目标列中的NaN（来自未来期数shift）移除了 {rows_removed} 行。")

    # 从主回归模型配置中获取期望的输入特征列
    input_numeric_features_config = training_config.get('feature_columns_numeric', [])
    input_categorical_features_config = training_config.get('feature_columns_categorical', [])

    # 筛选出DataFrame中实际存在的特征
    final_input_features_numeric = [col for col in input_numeric_features_config if col in df.columns]
    final_input_features_categorical = [col for col in input_categorical_features_config if col in df.columns]

    # 记录配置中存在但DF中缺失的特征，以供调试
    missing_numeric = set(input_numeric_features_config) - set(final_input_features_numeric)
    if missing_numeric:
        logging.error(f"配置的数值输入特征在DataFrame中最终未找到: {sorted(list(missing_numeric))}")

    missing_categorical = set(input_categorical_features_config) - set(final_input_features_categorical)
    if missing_categorical:
        logging.error(f"配置的类别输入特征在DataFrame中最终未找到: {sorted(list(missing_categorical))}")

    final_input_feature_names = final_input_features_numeric + final_input_features_categorical

    essential_cols = ['id', 'date'] # 确保ID和日期列被保留
    final_df_cols_to_select = list(dict.fromkeys(essential_cols + final_input_feature_names + target_column_names))
    
    # 再次确认所有选定列都在DataFrame中，以防万一
    final_df_cols_present_in_df = [col for col in final_df_cols_to_select if col in df.columns]

    features_df_final = df[final_df_cols_present_in_df].copy()

    # 填充最终数值特征中的NaN（例如，来自lag或rolling操作在序列开头的NaN）
    for col in final_input_features_numeric: # 只处理最终被选为数值输入的列
        if features_df_final[col].isnull().any():
            fill_value = 0 # 或者使用中位数/均值等更复杂的填充策略
            features_df_final[col] = features_df_final[col].fillna(fill_value)
            logging.debug(f"在最终数值特征 '{col}' 中用 {fill_value} 填充了NaN。")
    
    # 对类别特征中的NaN进行处理（如果需要）
    for col in final_input_features_categorical:
        if features_df_final[col].isnull().any():
            if col == 'product_segment': # 特别处理产品分段，确保是整数
                features_df_final[col] = features_df_final[col].fillna(0).astype(int)
                logging.warning(f"类别特征 '{col}' 包含NaN值，已用0填充并转换为整数。")
            else:
                # 对于其他类别特征，可以填充一个特殊值如 'UNK' 或最常见值，然后进行编码
                # 当前假设类别特征在输入时已经是编码好的整数，或者在后续Dataset中处理
                logging.warning(f"最终类别特征 '{col}' 包含NaN值。当前未做特殊填充（除product_segment外）。")


    logging.info(f"特征工程完成。最终DataFrame形状: {features_df_final.shape}")
    
    # 再次确认最终的输入和输出列，以防在选择或填充过程中发生意外
    actual_final_input_features = [f for f in final_input_feature_names if f in features_df_final.columns]
    actual_final_target_columns = [t for t in target_column_names if t in features_df_final.columns]

    logging.info(f"实际最终输入特征列 ({len(actual_final_input_features)}): {actual_final_input_features[:10]}...") # 显示前10个
    logging.info(f"实际最终目标列 ({len(actual_final_target_columns)}): {actual_final_target_columns}")
    
    if not features_df_final.empty and logging.getLogger().isEnabledFor(logging.DEBUG):
        logging.debug(f"最终特征DataFrame样本 (前5行):\n{features_df_final.head()}")
        if 'product_segment' in features_df_final.columns:
            logging.debug(f"产品分段特征 'product_segment' 分布:\n{features_df_final['product_segment'].value_counts(normalize=True, dropna=False).head()}")
        
        # 检查最终选择的列中是否还有NaN
        check_nan_cols = [col for col in (actual_final_input_features + actual_final_target_columns) if col in features_df_final.columns]
        if check_nan_cols:
            nan_sum_series = features_df_final[check_nan_cols].isnull().sum()
            if nan_sum_series.sum() > 0:
                logging.warning("最终选择的输入特征或目标列中仍存在NaN值！")
                logging.debug(f"NaN 统计 (最终选择列):\n{nan_sum_series[nan_sum_series > 0]}")


    return features_df_final, actual_final_input_features, actual_final_target_columns
