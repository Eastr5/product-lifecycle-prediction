"""
高级生命周期标签生成方法
"""
import numpy as np
from scipy.optimize import curve_fit

def rule_based_lifecycle_labels(product_data):
    """
    基于规则的生命周期标签生成方法
    """
    # 创建产品数据副本
    data = product_data.copy()
    
    # 定义生命周期阶段条件
    conditions = [
        # 导入期: 低销量 + 开始增长
        (data['RelativeSales'] < 0.3) & (data['MA_GrowthRate'] > 0),
        # 成长期: 中等销量 + 高增长率
        (data['RelativeSales'] >= 0.3) & (data['RelativeSales'] < 0.8) & (data['MA_GrowthRate'] > 0.05),
        # 成熟期: 高销量 + 稳定
        (data['RelativeSales'] >= 0.8) & (abs(data['MA_GrowthRate']) <= 0.05),
        # 衰退期: 从高点下降
        (data['RelativeSales'] < 0.8) & (data['MA_GrowthRate'] < -0.05)
    ]
    
    choices = [0, 1, 2, 3]  # 0=导入期, 1=成长期, 2=成熟期, 3=衰退期
    data['LifecyclePhase'] = np.select(conditions, choices, default=0)
    
    return data['LifecyclePhase']

def percentile_based_lifecycle_labels(product_data):
    """
    基于百分位数的生命周期标签生成方法
    """
    # 创建产品数据副本
    data = product_data.copy()
    
    # 计算销售量百分位数
    sales = data['Sales_30d'].values
    if len(sales) == 0 or sales.max() == 0:
        return np.zeros(len(data))
        
    percentiles = [0, 25, 75, 100]
    thresholds = np.percentile(sales, percentiles)
    
    # 定义生命周期阶段条件
    conditions = [
        (data['Sales_30d'] <= thresholds[1]),
        (data['Sales_30d'] > thresholds[1]) & (data['Sales_30d'] <= thresholds[2]),
        (data['Sales_30d'] > thresholds[2]) & (data['GrowthRate_30d'] >= 0),
        (data['Sales_30d'] > thresholds[1]) & (data['GrowthRate_30d'] < 0)
    ]
    
    choices = [0, 1, 2, 3]  # 0=导入期, 1=成长期, 2=成熟期, 3=衰退期
    data['LifecyclePhase'] = np.select(conditions, choices, default=0)
    
    return data['LifecyclePhase'].values

def bass_model(t, p, q, m):
    """
    Bass扩散模型，常用于描述创新产品生命周期
    """
    return m * (1 - np.exp(-(p + q) * t)) / (1 + (q/p) * np.exp(-(p + q) * t))

def model_based_lifecycle_labels(product_data):
    """
    基于Bass模型拟合的生命周期标签生成方法
    """
    # 检查数据是否充足
    if len(product_data) < 5:  # 设定最小阈值
        return np.zeros(len(product_data))

    # 创建产品数据副本
    data = product_data.copy()
    
    # 准备拟合数据
    sales = data['Sales_7d'].values
    time_points = np.arange(len(sales))
    
    # 如果销售数据不足或全为0，返回默认标签
    if len(sales) < 10 or np.sum(sales) <= 0:
        return np.zeros(len(data))
    
    try:
        # 使用更保守的参数
        initial_guess = [0.01, 0.1, np.sum(sales)]
        bounds_low = [0.0001, 0.0001, np.max(sales)]
        bounds_high = [0.2, 0.2, np.sum(sales) * 1.5]
        
        # 尝试拟合Bass模型
        popt_bass, _ = curve_fit(
            bass_model, 
            time_points, 
            sales, 
            p0=initial_guess,
            bounds=(bounds_low, bounds_high),
            maxfev=5000
        )
        
        # 生成模型预测
        bass_pred = bass_model(time_points, *popt_bass)
        
        # 基于模型预测确定生命周期阶段
        # 计算销售增速
        growth_rate = np.gradient(bass_pred)
        
        # 定义生命周期阶段
        lifecycle = np.zeros(len(time_points), dtype=int)
        
        # 导入期: 低销量，开始上升
        threshold_intro = bass_pred.max() * 0.2
        # 成长期: 快速增长
        threshold_growth = bass_pred.max() * 0.6
        # 成熟期: 接近峰值
        threshold_mature = bass_pred.max() * 0.8
        
        for i in range(len(time_points)):
            if bass_pred[i] <= threshold_intro:
                lifecycle[i] = 0  # 导入期
            elif bass_pred[i] <= threshold_growth:
                lifecycle[i] = 1  # 成长期
            elif (bass_pred[i] <= threshold_mature) or (growth_rate[i] >= 0):
                lifecycle[i] = 2  # 成熟期
            else:
                lifecycle[i] = 3  # 衰退期
                
        return lifecycle
        
    except Exception as e:
        print(f"Bass模型拟合错误: {type(e).__name__}: {str(e)}")
        return rule_based_lifecycle_labels(data)