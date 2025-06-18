# product_lifecycle_predictor_vA/src/prediction_interpretation/interpreter.py
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List

def interpret_product_trends(
    raw_predictions_df: pd.DataFrame, 
    config: Dict[str, Any]
) -> pd.DataFrame:
    """
    根据模型预测的连续指标和配置中的规则，解释产品趋势。

    Args:
        raw_predictions_df (pd.DataFrame): 包含模型预测的连续指标的DataFrame。
                                           列名应为 "predicted_{metric_name}_lead{h}" 格式，
                                           例如 "predicted_sales_W_lead1"。
                                           这些预测值应该是已经逆转换为原始数据尺度的。
        config (Dict[str, Any]): 项目配置文件。

    Returns:
        pd.DataFrame: 原始预测DataFrame，并添加了新的趋势解释列 
                      (例如 "interpreted_trend_lead1", "interpreted_trend_lead2")。
    """
    logging.info("开始解释产品趋势...")
    if raw_predictions_df.empty:
        logging.warning("输入的预测DataFrame为空，无法进行趋势解释。")
        return raw_predictions_df

    interp_config = config.get('prediction_interpretation', {})
    rules_for_trends = interp_config.get('rules_for_trends', {})
    default_trend_label = interp_config.get('default_trend_label', "趋势未定")
    
    # 从特征工程配置中获取预测期数，以确定要为哪些lead期生成趋势标签
    fe_config = config.get('feature_engineering', {})
    prediction_horizon = fe_config.get('prediction_horizon', 1)

    # 复制DataFrame以避免修改原始输入
    interpreted_df = raw_predictions_df.copy()

    for h in range(1, prediction_horizon + 1):
        trend_col_name = f"interpreted_trend_lead{h}"
        interpreted_df[trend_col_name] = default_trend_label # 初始化为默认标签

        # 筛选适用于当前lead期 (h) 的规则
        # 我们将基于规则的键名来判断，例如 "growth_trend_lead1" 适用于 h=1
        applicable_rules = []
        for rule_name, rule_details in rules_for_trends.items():
            # 简单的后缀匹配，可以根据需要设计更复杂的规则匹配逻辑
            if rule_name.endswith(f"_lead{h}"):
                if 'condition' in rule_details and 'label' in rule_details:
                    applicable_rules.append(rule_details) 
                else:
                    logging.warning(f"规则 '{rule_name}' 缺少 'condition' 或 'label'，将跳过。")
        
        if not applicable_rules:
            logging.info(f"对于 lead{h}，未找到适用的趋势解释规则。将使用默认标签 '{default_trend_label}'。")
            continue

        logging.info(f"为 lead{h} 应用 {len(applicable_rules)} 条趋势解释规则...")
        
        # 应用规则：这里采用顺序应用，后满足的规则会覆盖先满足的规则
        # 如果需要更复杂的逻辑（如优先级、互斥），可以使用 np.select 或其他方法
        for rule_details in applicable_rules:
            condition_str = rule_details['condition']
            label = rule_details['label']
            try:
                # 使用 pandas.eval 来评估条件字符串
                # 这要求 condition_str 中的列名存在于 interpreted_df 中
                condition_mask = interpreted_df.eval(condition_str)
                interpreted_df.loc[condition_mask, trend_col_name] = label
                logging.debug(f"  应用规则 (lead{h}): '{label}' for condition '{condition_str}'")
            except Exception as e:
                logging.error(f"评估规则条件 (lead{h}) '{condition_str}' 时出错: {e}。请检查列名和表达式。")
                # 可以选择跳过此规则或停止
                continue
        
        logging.info(f"已为 {trend_col_name} 生成趋势标签。")
        if logging.getLogger().isEnabledFor(logging.DEBUG) and trend_col_name in interpreted_df:
            logging.debug(f"  {trend_col_name} 分布:\n{interpreted_df[trend_col_name].value_counts(dropna=False)}")


    logging.info("产品趋势解释完成。")
    return interpreted_df


if __name__ == '__main__':
    print("测试 src/prediction_interpretation/interpreter.py...")
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

    # 模拟模型预测输出 (已逆转换为原始尺度)
    mock_predictions = pd.DataFrame({
        'id': ['itemA', 'itemA', 'itemB', 'itemB', 'itemC'],
        'date': pd.to_datetime(['2024-01-07', '2024-01-14', '2024-01-07', '2024-01-14', '2024-01-07']),
        'predicted_sales_W_lead1':          [12, 15, 3,  2,  20],
        'predicted_sales_W_growth_4w_1p_lead1': [0.15, 0.2, -0.02, -0.15, 0.05],
        'predicted_sales_W_lead2':          [18, 20, 2,  1,  25],
        'predicted_sales_W_growth_4w_1p_lead2': [0.12, 0.1, -0.05, -0.2,  0.08],
        # 可以有其他预测列，但规则中未用到
        'predicted_price_W_mean_lead1':     [2.5, 2.5, 1.0, 1.0, 3.0] 
    })

    # 模拟 config
    mock_config = {
        'feature_engineering': {
            'prediction_horizon': 2
        },
        'prediction_interpretation': {
            'default_trend_label': "趋势不明朗",
            'rules_for_trends': {
                # Lead 1 规则
                'high_growth_lead1': {
                    'condition': "predicted_sales_W_growth_4w_1p_lead1 > 0.1 and predicted_sales_W_lead1 > 10",
                    'label': "高速增长 (未来1周)"
                },
                'stable_low_lead1': {
                    'condition': "abs(predicted_sales_W_growth_4w_1p_lead1) < 0.05 and predicted_sales_W_lead1 < 5",
                    'label': "稳定低迷 (未来1周)"
                },
                 'moderate_growth_lead1': { # 后定义的规则，如果条件也满足，会覆盖 'high_growth_lead1' 的部分情况
                    'condition': "predicted_sales_W_growth_4w_1p_lead1 > 0.0 and predicted_sales_W_lead1 > 8",
                    'label': "温和增长 (未来1周)"
                },
                'decline_lead1': {
                    'condition': "predicted_sales_W_growth_4w_1p_lead1 < -0.1",
                    'label': "显著衰退 (未来1周)"
                },
                # Lead 2 规则
                'growth_prospect_lead2': {
                    'condition': "predicted_sales_W_growth_4w_1p_lead2 > 0.05",
                    'label': "增长前景 (未来2周)"
                },
                'decline_prospect_lead2': {
                    'condition': "predicted_sales_W_growth_4w_1p_lead2 < -0.05",
                    'label': "衰退风险 (未来2周)"
                }
            }
        }
    }

    interpreted_df = interpret_product_trends(mock_predictions, mock_config)
    
    print("\n--- 测试：解释后的 DataFrame ---")
    print(interpreted_df)

    if 'interpreted_trend_lead1' in interpreted_df:
        print("\n趋势解释 (lead1) 分布:")
        print(interpreted_df['interpreted_trend_lead1'].value_counts())
    if 'interpreted_trend_lead2' in interpreted_df:
        print("\n趋势解释 (lead2) 分布:")
        print(interpreted_df['interpreted_trend_lead2'].value_counts())

