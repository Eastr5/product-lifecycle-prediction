from src.lifecycle.simple import simple_lifecycle_labels


def create_product_attribute_features(product_data, product_desc):
    """添加产品属性相关特征"""
    
    # 价格段特征 - 基于平均单价
    if 'AvgUnitPrice_30d' in product_data.columns:
        # 创建价格四分位数
        price_quantiles = product_data['AvgUnitPrice_30d'].quantile([0.25, 0.5, 0.75])
        
        # 将产品划分为价格段
        conditions = [
            product_data['AvgUnitPrice_30d'] <= price_quantiles[0.25],
            (product_data['AvgUnitPrice_30d'] > price_quantiles[0.25]) & 
            (product_data['AvgUnitPrice_30d'] <= price_quantiles[0.5]),
            (product_data['AvgUnitPrice_30d'] > price_quantiles[0.5]) & 
            (product_data['AvgUnitPrice_30d'] <= price_quantiles[0.75]),
            product_data['AvgUnitPrice_30d'] > price_quantiles[0.75]
        ]
        choices = [0, 1, 2, 3]  # 低、中低、中高、高
        product_data['PriceCategory'] = np.select(conditions, choices, default=1)
    
    # 产品类别特征 - 基于StockCode前缀
    product_data['ProductPrefix'] = product_data['StockCode'].astype(str).str.extract(r'([A-Za-z]+)')
    
    # 季节性检测 - 基于月度销售模式
    if 'InvoiceDate' in product_data.columns:
        product_data['Month'] = product_data['InvoiceDate'].dt.month
        monthly_sales = product_data.groupby('Month')['Quantity'].sum()
        
        # 检测是否存在季节性模式
        if len(monthly_sales) >= 3:
            winter_sales = monthly_sales.get(12, 0) + monthly_sales.get(1, 0) + monthly_sales.get(2, 0)
            spring_sales = monthly_sales.get(3, 0) + monthly_sales.get(4, 0) + monthly_sales.get(5, 0)
            summer_sales = monthly_sales.get(6, 0) + monthly_sales.get(7, 0) + monthly_sales.get(8, 0)
            autumn_sales = monthly_sales.get(9, 0) + monthly_sales.get(10, 0) + monthly_sales.get(11, 0)
            
            total_sales = winter_sales + spring_sales + summer_sales + autumn_sales
            if total_sales > 0:
                product_data['WinterSalesRatio'] = winter_sales / total_sales
                product_data['SpringSalesRatio'] = spring_sales / total_sales
                product_data['SummerSalesRatio'] = summer_sales / total_sales
                product_data['AutumnSalesRatio'] = autumn_sales / total_sales
                
                # 季节性强度 (0-1)，表示销售分布的不均匀性
                seasons = [winter_sales, spring_sales, summer_sales, autumn_sales]
                std_seasons = np.std(seasons) / (np.mean(seasons) + 1e-8)  # 避免除零
                product_data['SeasonalityStrength'] = np.clip(std_seasons, 0, 1)
    
    return product_data

def create_product_attribute_features(product_data, product_desc):
    """添加产品属性相关特征"""
    
    # 价格段特征 - 基于平均单价
    if 'AvgUnitPrice_30d' in product_data.columns:
        # 创建价格四分位数
        price_quantiles = product_data['AvgUnitPrice_30d'].quantile([0.25, 0.5, 0.75])
        
        # 将产品划分为价格段
        conditions = [
            product_data['AvgUnitPrice_30d'] <= price_quantiles[0.25],
            (product_data['AvgUnitPrice_30d'] > price_quantiles[0.25]) & 
            (product_data['AvgUnitPrice_30d'] <= price_quantiles[0.5]),
            (product_data['AvgUnitPrice_30d'] > price_quantiles[0.5]) & 
            (product_data['AvgUnitPrice_30d'] <= price_quantiles[0.75]),
            product_data['AvgUnitPrice_30d'] > price_quantiles[0.75]
        ]
        choices = [0, 1, 2, 3]  # 低、中低、中高、高
        product_data['PriceCategory'] = np.select(conditions, choices, default=1)
    
    # 产品类别特征 - 基于StockCode前缀
    product_data['ProductPrefix'] = product_data['StockCode'].astype(str).str.extract(r'([A-Za-z]+)')
    
    # 季节性检测 - 基于月度销售模式
    if 'InvoiceDate' in product_data.columns:
        product_data['Month'] = product_data['InvoiceDate'].dt.month
        monthly_sales = product_data.groupby('Month')['Quantity'].sum()
        
        # 检测是否存在季节性模式
        if len(monthly_sales) >= 3:
            winter_sales = monthly_sales.get(12, 0) + monthly_sales.get(1, 0) + monthly_sales.get(2, 0)
            spring_sales = monthly_sales.get(3, 0) + monthly_sales.get(4, 0) + monthly_sales.get(5, 0)
            summer_sales = monthly_sales.get(6, 0) + monthly_sales.get(7, 0) + monthly_sales.get(8, 0)
            autumn_sales = monthly_sales.get(9, 0) + monthly_sales.get(10, 0) + monthly_sales.get(11, 0)
            
            total_sales = winter_sales + spring_sales + summer_sales + autumn_sales
            if total_sales > 0:
                product_data['WinterSalesRatio'] = winter_sales / total_sales
                product_data['SpringSalesRatio'] = spring_sales / total_sales
                product_data['SummerSalesRatio'] = summer_sales / total_sales
                product_data['AutumnSalesRatio'] = autumn_sales / total_sales
                
                # 季节性强度 (0-1)，表示销售分布的不均匀性
                seasons = [winter_sales, spring_sales, summer_sales, autumn_sales]
                std_seasons = np.std(seasons) / (np.mean(seasons) + 1e-8)  # 避免除零
                product_data['SeasonalityStrength'] = np.clip(std_seasons, 0, 1)
    
    return product_data

def create_market_context_features(df, product_data, product_desc):
    """创建市场环境相关特征"""
    
    product_id = product_data['StockCode'].iloc[0]  # 假设所有行都是同一产品
    
    # 获取产品类别信息
    # 这里使用一个简单的方法：提取产品描述的第一个单词作为类别
    # 实际应用中可能需要更复杂的分类方法
    product_category = product_desc.get(product_id, "").split()[0] if product_desc.get(product_id, "") else ""
    
    # 寻找同类别产品
    similar_products = []
    for pid, desc in product_desc.items():
        if pid != product_id and desc.startswith(product_category) and len(product_category) > 0:
            similar_products.append(pid)
    
    # 按日期计算市场特征
    market_features = []
    
    for date in product_data['InvoiceDate'].unique():
        # 截至当前日期的数据
        df_until_date = df[df['InvoiceDate'] <= date]
        
        # 同类产品的数量
        num_similar_active = sum(1 for pid in similar_products if pid in df_until_date['StockCode'].unique())
        
        # 市场整体趋势 (过去30天的总销售增长率)
        past_30d = date - pd.Timedelta(days=30)
        past_60d = date - pd.Timedelta(days=60)
        
        sales_last_30d = df_until_date[(df_until_date['InvoiceDate'] > past_30d)]['Quantity'].sum()
        sales_30d_before = df_until_date[(df_until_date['InvoiceDate'] > past_60d) & 
                                        (df_until_date['InvoiceDate'] <= past_30d)]['Quantity'].sum()
        
        market_growth = (sales_last_30d - sales_30d_before) / (sales_30d_before + 1) # 避免除零
        
        # 产品在同类别中的市场份额
        if similar_products:
            category_sales = df_until_date[(df_until_date['StockCode'].isin(similar_products)) | 
                                         (df_until_date['StockCode'] == product_id)]
            product_sales = category_sales[category_sales['StockCode'] == product_id]['Quantity'].sum()
            total_category_sales = category_sales['Quantity'].sum()
            market_share = product_sales / total_category_sales if total_category_sales > 0 else 0
        else:
            market_share = 1.0  # 如果没有同类产品，则市场份额为100%
        
        market_features.append({
            'InvoiceDate': date,
            'NumSimilarProducts': num_similar_active,
            'MarketGrowth30d': market_growth,
            'MarketShare': market_share
        })
    
    # 转换为DataFrame并合并
    market_df = pd.DataFrame(market_features)
    return pd.merge(product_data, market_df, on='InvoiceDate', how='left')

from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

def perform_feature_selection(product_features_df, valid_products, n_components=0.95):
    """
    执行特征选择和降维
    
    参数:
        product_features_df: 带特征的产品数据DataFrame
        valid_products: 有效产品ID列表
        n_components: PCA保留的方差比例或组件数
        
    返回:
        处理后的特征DataFrame，特征重要性信息
    """
    # 选择数值特征列
    feature_cols = [col for col in product_features_df.columns
                   if product_features_df[col].dtype in [np.float64, np.int64, float, int]
                   and col not in ['StockCode', 'InvoiceDate', 'LifecyclePhase']]
    
    # 1. 相关性分析 - 移除高度相关特征
    corr_matrix = product_features_df[feature_cols].corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr_cols = [col for col in upper_tri.columns if any(upper_tri[col] > 0.95)]
    print(f"移除了 {len(high_corr_cols)} 个高度相关特征")
    
    # 从特征列表中移除高度相关列
    reduced_features = [col for col in feature_cols if col not in high_corr_cols]
    
    # 2. 使用随机森林评估特征重要性
    # 为了处理大数据集，我们可以抽样
    sample_indices = product_features_df[product_features_df['StockCode'].isin(valid_products[:100])].index
    if len(sample_indices) > 10000:
        sample_indices = np.random.choice(sample_indices, 10000, replace=False)
    
    X_sample = product_features_df.loc[sample_indices, reduced_features]
    y_sample = product_features_df.loc[sample_indices, 'LifecyclePhase']
    
    # 确保数据有效
    X_sample = X_sample.fillna(0)
    
    # 训练随机森林以获取特征重要性
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_sample, y_sample)
    
    # 获取特征重要性
    importances = rf.feature_importances_
    feature_importance = pd.DataFrame({
        'Feature': reduced_features,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    # 选择重要特征
    top_features = feature_importance['Feature'][:min(30, len(reduced_features))].tolist()
    print(f"选择了 {len(top_features)} 个最重要特征")
    
    # 3. 应用PCA降维 (可选)
    apply_pca = False  # 设置为True来启用PCA
    if apply_pca:
        pca = PCA(n_components=n_components, random_state=42)
        # 在所有产品数据上拟合PCA
        all_product_data = product_features_df[top_features].fillna(0)
        pca_result = pca.fit_transform(all_product_data)
        
        # 创建PCA特征列
        pca_cols = [f'PCA_{i+1}' for i in range(pca_result.shape[1])]
        pca_df = pd.DataFrame(pca_result, columns=pca_cols, index=product_features_df.index)
        
        # 解释方差比例
        explained_var = pca.explained_variance_ratio_
        print(f"PCA保留了 {len(pca_cols)} 个成分，解释了 {sum(explained_var):.2%} 的方差")
        
        # 将原始重要特征与PCA特征合并
        combined_features = pd.concat([product_features_df[['StockCode', 'InvoiceDate', 'LifecyclePhase']], 
                                      product_features_df[top_features[:10]], 
                                      pca_df], axis=1)
    else:
        combined_features = product_features_df[['StockCode', 'InvoiceDate', 'LifecyclePhase'] + top_features]
    
    return combined_features, feature_importance

def create_nonlinear_features(product_data):
    """创建非线性特征和交互特征"""
    
    # 确保所有数值特征可用
    for col in product_data.columns:
        if product_data[col].dtype in [np.float64, np.int64, float, int]:
            product_data[col] = product_data[col].fillna(0)
    
    # 1. 销售相关比率特征
    if all(col in product_data.columns for col in ['Sales_30d', 'Sales_7d']):
        # 计算短期与长期销售比率 (>1表示销售增长，<1表示销售下降)
        product_data['SalesRatio_7d_30d'] = (product_data['Sales_7d'] / (product_data['Sales_30d'] + 1))
    
    if all(col in product_data.columns for col in ['Quantity', 'NumTransactions']):
        # 每笔交易的平均数量 (反映批量购买模式)
        product_data['AvgQuantityPerTransaction'] = (product_data['Quantity'] / 
                                                   (product_data['NumTransactions'] + 1e-8))
    
    # 2. 价格与销量交互特征
    if all(col in product_data.columns for col in ['AvgUnitPrice_30d', 'Sales_30d']):
        # 价格敏感度指标
        price_std = product_data['AvgUnitPrice_30d'].std()
        if price_std > 0:
            # 计算过去30天的价格变化率
            product_data['PriceChangeRate'] = product_data['AvgUnitPrice_30d'].pct_change(periods=7).fillna(0)
            
            # 创建价格-销量弹性特征
            # 当价格增加时销量如何变化
            product_data['PriceElasticity'] = np.where(
                product_data['PriceChangeRate'] != 0,
                -(product_data['Sales_7d'].pct_change(periods=7).fillna(0) / 
                  (product_data['PriceChangeRate'] + 1e-8)),
                0
            )
            # 限制极端值
            product_data['PriceElasticity'] = product_data['PriceElasticity'].clip(-10, 10)
    
    # 3. 增长率的非线性变换
    if 'SalesGrowth_14d' in product_data.columns:
        # 对数变换以减轻异常值影响
        product_data['LogSalesGrowth'] = np.sign(product_data['SalesGrowth_14d']) * np.log1p(
            np.abs(product_data['SalesGrowth_14d']))
        
        # 增长率平方 (放大大的变化)
        product_data['SalesGrowthSquared'] = product_data['SalesGrowth_14d'] ** 2
        
        # 增长加速度/减速度 (增长率的变化趋势)
        product_data['GrowthTrend'] = product_data['SalesGrowth_14d'].diff().fillna(0)
    
    # 4. 产品生命指标 - 综合考虑年龄、销量和增长
    if all(col in product_data.columns for col in ['ProductAge', 'RelativeSales', 'MA_SalesGrowth']):
        # 生命力指标 (考虑产品年龄、当前销量水平和增长趋势)
        max_age = product_data['ProductAge'].max()
        
        # 归一化年龄 (0-1范围)
        norm_age = product_data['ProductAge'] / max_age if max_age > 0 else 0
        
        # 生命力分数 = 相对销量 * (1 + 增长率) * exp(-年龄影响)
        # 年龄影响随时间增强，但成熟产品（中间年龄）不会受太多影响
        age_effect = np.exp(-4 * (norm_age - 0.5)**2)
        product_data['VitalityScore'] = (product_data['RelativeSales'] * 
                                       (1 + product_data['MA_SalesGrowth']) * 
                                       age_effect)
    
    return product_data

import numpy as np
from scipy import signal
from statsmodels.tsa.stattools import adfuller
import pandas as pd

def create_advanced_time_series_features(product_data):
    """创建高级时间序列特征"""
    
    # 确保数据按日期排序
    product_data = product_data.sort_values('InvoiceDate')
    
    # 1. 趋势强度特征
    if 'Quantity' in product_data.columns and len(product_data) > 14:
        # 使用线性回归计算趋势斜率
        x = np.arange(len(product_data))
        y = product_data['Quantity'].values
        
        # 处理缺失值
        mask = ~np.isnan(y)
        if sum(mask) > 2:  # 至少需要3个点才能计算趋势
            x_valid = x[mask]
            y_valid = y[mask]
            
            # 计算趋势线斜率和R²
            if len(x_valid) > 1:
                slope, intercept = np.polyfit(x_valid, y_valid, 1)
                y_pred = slope * x_valid + intercept
                ssr = np.sum((y_valid - y_pred) ** 2)
                sst = np.sum((y_valid - np.mean(y_valid)) ** 2)
                r2 = 1 - (ssr / sst) if sst != 0 else 0
                
                # 将斜率标准化为相对于平均值的百分比变化
                mean_y = np.mean(y_valid)
                norm_slope = slope / mean_y if mean_y != 0 else 0
                
                product_data['TrendSlope'] = norm_slope
                product_data['TrendStrength'] = r2  # R² 越高表示趋势越强
                
                # 趋势方向 (-1: 下降, 0: 平稳, 1: 上升)
                product_data['TrendDirection'] = np.sign(slope)
        
    # 2. 变点检测 (使用窗口方差变化检测)
    if 'Quantity' in product_data.columns and len(product_data) > 30:
        window_size = 7
        rolling_mean = product_data['Quantity'].rolling(window=window_size, min_periods=1).mean()
        rolling_std = product_data['Quantity'].rolling(window=window_size, min_periods=1).std()
        
        # 计算滚动均值的显著变化
        mean_change = rolling_mean.diff().abs()
        std_of_change = mean_change.rolling(window=window_size, min_periods=1).std()
        
        # 如果变化大于标准差的2倍，标记为变点
        product_data['ChangePoint'] = (mean_change > 2 * std_of_change).astype(int)
        
        # 自上一个变点以来的天数
        product_data['DaysSinceChangePoint'] = 0
        change_points = product_data[product_data['ChangePoint'] == 1].index.tolist()
        
        if change_points:
            last_cp_idx = 0
            for idx in range(len(product_data)):
                if idx in change_points:
                    last_cp_idx = idx
                else:
                    product_data.iloc[idx, product_data.columns.get_loc('DaysSinceChangePoint')] = idx - last_cp_idx
    
    # 3. 周期性分析 (使用自相关)
    if 'Quantity' in product_data.columns and len(product_data) > 60:
        # 填充缺失值
        qty_series = product_data['Quantity'].fillna(method='ffill').fillna(method='bfill')
        
        if len(qty_series) >= 40:  # 需要足够的数据点
            # 计算自相关函数 (ACF)
            n = min(40, len(qty_series) // 2)  # 检查的最大延迟
            acf = np.correlate(qty_series - qty_series.mean(), qty_series - qty_series.mean(), mode='full')
            acf = acf[len(acf)//2:len(acf)//2+n]  # 只保留正延迟
            acf /= acf[0]  # 归一化
            
            # 查找峰值 (潜在周期)
            peaks = signal.find_peaks(acf, height=0.2, distance=3)[0]
            
            if len(peaks) > 0:
                # 使用首个显著周期
                first_peak = peaks[0] if len(peaks) > 0 else 0
                product_data['SeasonalPeriod'] = first_peak
                product_data['SeasonalStrength'] = acf[first_peak] if first_peak > 0 else 0
                
                # 在周期中的位置
                if first_peak > 1:
                    product_data['CyclicPosition'] = (
                        pd.Series(range(len(product_data))) % first_peak) / first_peak
                else:
                    product_data['CyclicPosition'] = 0
            else:
                product_data['SeasonalPeriod'] = 0
                product_data['SeasonalStrength'] = 0
                product_data['CyclicPosition'] = 0
    
    # 4. 平稳性检测 (Augmented Dickey-Fuller test)
    if 'Quantity' in product_data.columns and len(product_data) > 30:
        qty_series = product_data['Quantity'].fillna(method='ffill').fillna(0)
        if len(qty_series) >= 30:
            try:
                # 进行 ADF 测试
                adf_result = adfuller(qty_series, maxlag=10)
                
                # p值越小表示越平稳 (拒绝单位根假设)
                adf_pvalue = adf_result[1]
                product_data['Stationarity'] = 1 - adf_pvalue  # 转换为0-1分数，越高越平稳
            except:
                product_data['Stationarity'] = 0.5  # 默认值
        else:
            product_data['Stationarity'] = 0.5
    
    return product_data

# 修改 create_features 函数
def create_features(daily_sales, product_desc, df_original, windows, min_days_threshold):
    """
    为每个产品创建增强的时间序列特征。
    
    Args:
        daily_sales (pd.DataFrame): 包含每日销售数据的DataFrame
        product_desc (dict): 将StockCode映射到Description的字典
        df_original (pd.DataFrame): 原始交易数据
        windows (list): 用于滚动计算的窗口大小列表
        min_days_threshold (int): 考虑一个产品所需的最小天数
        
    Returns:
        tuple: 包含以下内容的元组:
            - pd.DataFrame: 包含所有有效产品生成特征的DataFrame
            - list: 有效产品StockCode的列表
            - dict: 特征重要性信息
    """
    print("开始增强特征工程...") # Starting enhanced feature engineering...
    
    # 基本数据预处理和有效产品筛选 (与原始代码相同)
    products = daily_sales['StockCode'].unique()
    product_counts = daily_sales['StockCode'].value_counts()
    valid_products = product_counts[product_counts >= min_days_threshold].index.tolist()
    
    print(f"有效产品数量 (>= {min_days_threshold} 天数据): {len(valid_products)}")
    daily_sales_filtered = daily_sales[daily_sales['StockCode'].isin(valid_products)].copy()
    
    # 初始化结果列表
    all_product_features = []
    
    # 按产品分组处理
    grouped = daily_sales_filtered.groupby('StockCode')
    
    # 特征工程计数器
    processed_count = 0
    
    for product, product_data_group in grouped:
        # 处理进度显示
        processed_count += 1
        if processed_count % 50 == 0:
            print(f"已处理 {processed_count}/{len(valid_products)} 个产品...")
        
        # 按日期排序并创建副本
        product_data = product_data_group.sort_values('InvoiceDate').copy()
        
        # --- 日期连续性处理 (与原始代码相同)
        date_range = pd.date_range(
            start=product_data['InvoiceDate'].min(), 
            end=product_data['InvoiceDate'].max(), 
            freq='D'
        )
        date_df = pd.DataFrame({'InvoiceDate': date_range})
        product_data = pd.merge(date_df, product_data, on='InvoiceDate', how='left')
        product_data['StockCode'].fillna(product, inplace=True)
        product_data['Description'].fillna(product_desc.get(product, 'Unknown'), inplace=True)
        product_data[['Quantity', 'TotalPrice', 'NumTransactions']].fillna(0, inplace=True)
        
        # --- 创建基本滚动窗口特征 (与原始代码相同)
        for window in windows:
            product_data[f'Sales_{window}d'] = product_data['Quantity'].rolling(window, min_periods=1).sum()
            product_data[f'Revenue_{window}d'] = product_data['TotalPrice'].rolling(window, min_periods=1).sum()
            product_data[f'Transactions_{window}d'] = product_data['NumTransactions'].rolling(window, min_periods=1).sum()
            
            product_data[f'AvgSales_{window}d'] = product_data['Quantity'].rolling(window, min_periods=1).mean()
            product_data[f'AvgRevenue_{window}d'] = product_data['TotalPrice'].rolling(window, min_periods=1).mean()
            
            product_data[f'SalesGrowth_{window}d'] = product_data[f'Sales_{window}d'].pct_change(periods=window)
            product_data[f'RevenueGrowth_{window}d'] = product_data[f'Revenue_{window}d'].pct_change(periods=window)
        
        # --- 创建基本衍生特征 (与原始代码相同)
        long_window = max(windows)
        sales_denom = product_data[f'Sales_{long_window}d'].replace(0, np.nan)
        product_data[f'AvgUnitPrice_{long_window}d'] = (
            product_data[f'Revenue_{long_window}d'] / sales_denom).fillna(0)
        
        short_window = min(windows)
        product_data['SalesAcceleration'] = product_data[f'SalesGrowth_{short_window}d'].diff()
        
        max_sales_period = max(windows)
        max_sales_val = product_data[f'Sales_{max_sales_period}d'].max()
        if max_sales_val > 0:
            product_data['RelativeSales'] = product_data[f'Sales_{max_sales_period}d'] / max_sales_val
        else:
            product_data['RelativeSales'] = 0
            
        medium_window = sorted(windows)[len(windows)//2]
        product_data['MA_SalesGrowth'] = product_data[f'SalesGrowth_{medium_window}d'].rolling(
            short_window, min_periods=1).mean()
        
        product_data['ProductAge'] = (
            product_data['InvoiceDate'] - product_data['InvoiceDate'].min()).dt.days
            
        product_data[f'SalesVolatility_{medium_window}d'] = product_data['Quantity'].rolling(
            medium_window, min_periods=1).std()
        
        # --- 新增: 应用增强特征工程 ---
        # 1. 产品属性特征
        # product_data = create_product_attribute_features(product_data, product_desc)
        
        # 2. 客户行为特征 (需要原始数据)
        # product_data = create_customer_behavior_features(df_original, product_data)
        
        # 3. 市场环境特征
        # product_data = create_market_context_features(df_original, product_data, product_desc)
        
        # 4. 非线性特征和交互特征
        # product_data = create_nonlinear_features(product_data)
        
        # 5. 高级时序特征
        product_data = create_advanced_time_series_features(product_data)
        
        # 将处理后的产品数据添加到列表
        all_product_features.append(product_data)
    
    # 检查是否有特征数据
    if not all_product_features:
        print("警告: 未生成任何产品特征。请检查min_days_threshold和输入数据。")
        return pd.DataFrame(), [], {}
    
    # 合并所有产品特征
    print("合并所有产品特征...")
    product_features_df = pd.concat(all_product_features, ignore_index=True)

    # 处理缺失值和无限值
    product_features_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    # 填充所有数值列的NaN为0 (保留非数值列如StockCode, InvoiceDate)
    num_cols = product_features_df.select_dtypes(include=np.number).columns
    product_features_df[num_cols] = product_features_df[num_cols].fillna(0)
    # 对于非数值列，你可能需要不同的填充策略，但这里似乎主要是 StockCode/InvoiceDate/Description
    # product_features_df.fillna({'Description': 'Unknown'}, inplace=True) # 如果需要填充描述

    # --- 新增：在此处生成生命周期标签 ---
    print("生成生命周期标签...")
    labeled_dfs = []
    for product in valid_products:
        product_segment = product_features_df[product_features_df['StockCode'] == product].copy()
        if not product_segment.empty:
            try:
                # 确保用于打标签的列存在且已填充
                required_label_cols = ['Sales_30d', 'MA_SalesGrowth'] # 示例，根据 simple_lifecycle_labels 确认
                if all(col in product_segment.columns for col in required_label_cols):
                    product_segment['LifecyclePhase'] = simple_lifecycle_labels(product_segment)
                else:
                    print(f"警告: 产品 {product} 缺少用于打标签的列，设置为默认值 2。")
                    product_segment['LifecyclePhase'] = 2 # 默认成熟期
            except Exception as e:
                print(f"错误: 为产品 {product} 生成标签时出错: {e}。设置为默认值 2。")
                product_segment['LifecyclePhase'] = 2 # 默认成熟期
            labeled_dfs.append(product_segment)

    if not labeled_dfs:
        print("错误: 标签化后没有剩余的产品数据。程序中止。")
        return pd.DataFrame(), [], {} # 返回空结果

    product_features_df = pd.concat(labeled_dfs, ignore_index=True)
    # 确保标签列是整数类型
    if 'LifecyclePhase' in product_features_df.columns:
        product_features_df['LifecyclePhase'] = product_features_df['LifecyclePhase'].astype(int)
    else:
        print("错误: 'LifecyclePhase' 列未能成功添加到 DataFrame。")
        return pd.DataFrame(), [], {} # 返回空结果
    # --- 标签生成结束 ---


    # 执行特征选择与降维
    print("执行特征选择...")
    # 现在 product_features_df 应该包含 'LifecyclePhase' 列了
    selected_features_df, feature_importance = perform_feature_selection(
        product_features_df, valid_products)

    # (确保返回 feature_importance)
    print(f"特征工程完成，生成了 {len(product_features_df.columns)} 个原始特征，" +
        f"选择了 {len(selected_features_df.columns) - 3} 个最终特征") # 减去StockCode/InvoiceDate/LifecyclePhase

    # 返回 selected_features_df, valid_products, 和 feature_importance
    return selected_features_df, valid_products, feature_importance # <--- 确保返回 feature_importance                        