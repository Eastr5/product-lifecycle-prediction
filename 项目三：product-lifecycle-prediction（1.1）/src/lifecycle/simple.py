import numpy as np
import pandas as pd

def simple_lifecycle_labels(product_data):
    """
    基于销售水平和增长率的简化生命周期标注方法。
    (Simplified lifecycle labeling method based on sales level and growth rate.)

    Args:
        product_data (pd.DataFrame): 包含单个产品销售数据的 DataFrame。
                                     需要包含 'Sales_30d', 'Quantity', 'MA_SalesGrowth' (或类似) 列。
                                     (DataFrame containing sales data for a single product.
                                     Expected to contain 'Sales_30d', 'Quantity', 'MA_SalesGrowth' (or similar) columns.)

    Returns:
        np.ndarray: 生命周期阶段标签数组 (0=导入期, 1=成长期, 2=成熟期, 3=衰退期)。
                    (Array of lifecycle phase labels (0=Introduction, 1=Growth, 2=Maturity, 3=Decline).)
    """
    # 创建数据副本以避免修改原始数据
    # (Create a copy of the data to avoid modifying the original)
    data = product_data.copy()

    # --- 计算相对销售水平 (Calculate Relative Sales Level) ---
    # 相对于 30 天销售额的历史最大值
    # (Relative to the historical maximum of 30-day sales)
    max_sales_30d = data['Sales_30d'].max()
    if max_sales_30d > 0:
        data['RelativeSales_30d'] = data['Sales_30d'] / max_sales_30d
    else:
        data['RelativeSales_30d'] = 0 # 如果最大销售额为 0，则相对销售额也为 0 (If max sales is 0, relative sales is 0)

    # --- 使用增长率趋势指标 (Use Growth Rate Trend Indicator) ---
    # 使用增长率的移动平均值 (MA_SalesGrowth) 作为趋势指标，更稳定
    # (Use the moving average of growth rate (MA_SalesGrowth) as the trend indicator, more stable)
    if 'MA_SalesGrowth' not in data.columns:
        # 如果 MA_SalesGrowth 不存在，尝试基于其他增长率计算或设置为 0
        # (If MA_SalesGrowth doesn't exist, try calculating based on other growth rates or default to 0)
        if 'SalesGrowth_14d' in data.columns:
            data['MA_SalesGrowth'] = data['SalesGrowth_14d'].rolling(7, min_periods=1).mean().fillna(0)
        elif 'SalesGrowth_7d' in data.columns:
            data['MA_SalesGrowth'] = data['SalesGrowth_7d'].fillna(0)
        else:
            print("警告: 在 simple_lifecycle_labels 中未找到增长率列。默认增长率为 0。") # WARNING: No growth rate column found in simple_lifecycle_labels. Defaulting growth to 0.
            data['MA_SalesGrowth'] = 0

    # --- 定义简单的阶段条件 (Define Simple Phase Conditions) ---
    # 这些阈值可以根据业务需求进行调整 (These thresholds can be adjusted based on business needs)
    intro_threshold = 0.25         # 导入期销售额上限 (Introduction sales upper threshold)
    growth_sales_min = 0.25        # 成长期销售额下限 (Growth sales lower threshold)
    growth_sales_max = 0.70        # 成长期销售额上限 / 成熟期下限 (Growth sales upper threshold / Maturity lower threshold)
    maturity_threshold = 0.70      # 成熟期高销售额下限 (Maturity high sales lower threshold)
    growth_rate_positive = 0.05    # 定义为“强劲”正增长的阈值 (Threshold defining "strong" positive growth)
    growth_rate_stable_low = -0.05 # 稳定增长率下限 (Stable growth rate lower bound)
    growth_rate_stable_high = 0.05 # 稳定增长率上限 (Stable growth rate upper bound)
    growth_rate_negative = -0.05   # 定义为“显著”负增长的阈值 (Threshold defining "significant" negative growth)

    conditions = [
        # 导入期 (Introduction): 低销售额 (< intro_threshold) + 正增长 (> 0)
        # (Low sales (< intro_threshold) + Positive growth (> 0))
        (data['RelativeSales_30d'] < intro_threshold) & (data['MA_SalesGrowth'] > 0),

        # 成长期 (Growth): 中等销售额 (>= growth_sales_min and < growth_sales_max) + 强劲正增长 (>= growth_rate_positive)
        # (Medium sales (>= growth_sales_min and < growth_sales_max) + Strong positive growth (>= growth_rate_positive))
        (data['RelativeSales_30d'] >= growth_sales_min) & (data['RelativeSales_30d'] < growth_sales_max) & (data['MA_SalesGrowth'] >= growth_rate_positive),

        # 成熟期 (Maturity): 高销售额 (>= maturity_threshold) 或 (中等销售额 + 稳定增长)
        # (High sales (>= maturity_threshold) OR (Medium sales + Stable growth))
        (
            (data['RelativeSales_30d'] >= maturity_threshold) |  # 高销售额直接判定为成熟期 (High sales directly indicates maturity)
            (
                (data['RelativeSales_30d'] >= growth_sales_min) & # 中等销售额 (Medium sales)
                (data['MA_SalesGrowth'] >= growth_rate_stable_low) & # 增长率不显著下降 (Growth rate not significantly declining)
                (data['MA_SalesGrowth'] < growth_rate_positive)     # 增长率未达到强劲增长 (Growth rate not reaching strong growth)
            )
        ),

        # 衰退期 (Decline): 低/中销售额 (< maturity_threshold) + 显著负增长 (<= growth_rate_negative)
        # (Low/Medium sales (< maturity_threshold) + Significant negative growth (<= growth_rate_negative))
        (data['RelativeSales_30d'] < maturity_threshold) & (data['MA_SalesGrowth'] <= growth_rate_negative)
    ]

    choices = [0, 1, 2, 3]  # 0=导入期, 1=成长期, 2=成熟期, 3=衰退期 (0=Intro, 1=Growth, 2=Maturity, 3=Decline)

    # 使用 np.select 根据条件分配标签，默认值为 2 (成熟期)
    # (Use np.select to assign labels based on conditions, default to 2 (Maturity))
    # 默认成熟期是一种保守假设，适用于稳定产品
    # (Defaulting to maturity is a conservative assumption for stable products)
    data['LifecyclePhase'] = np.select(conditions, choices, default=2)

    # --- 可选的后处理：平滑标签 (Optional Post-processing: Smooth Labels) ---
    # 使用滚动中位数来平滑短暂的标签波动 (Use rolling median to smooth short label fluctuations)
    # window_size = 3 # 例如，3天窗口 (e.g., 3-day window)
    # data['LifecyclePhase'] = data['LifecyclePhase'].rolling(window=window_size, center=True, min_periods=1).median()
    # # 填充滚动产生的 NaN (Fill NaNs produced by rolling)
    # data['LifecyclePhase'] = data['LifecyclePhase'].fillna(method='bfill').fillna(method='ffill').astype(int)

    # 返回生命周期阶段标签的 NumPy 数组
    # (Return the NumPy array of lifecycle phase labels)
    return data['LifecyclePhase'].values