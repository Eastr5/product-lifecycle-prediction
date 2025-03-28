好的，这是按照您要求的项目结构，将所有添加了中文注释的代码整合到一个 Markdown 文件中。

````markdown
# Product Lifecycle Prediction Project Code

## 1. `product-lifecycle-prediction/requirements.txt`

```txt
numpy
pandas
matplotlib
seaborn
scikit-learn
torch
openpyxl
PyYAML
warnings
```

## 2. `product-lifecycle-prediction/config.yaml`

```yaml
data:
  # 根据需要更改为实际路径，或在 load.py 中组合逻辑
  # (Change this to the actual path if needed, or combine logic in load.py)
  raw_path: data/raw/online_retail_10_11.csv
  processed_path: data/processed
  min_days_threshold: 30 # 产品被视为有效的最少数据天数 (Minimum number of days of data for a product to be considered valid)

features:
  windows: [7, 14, 30]     # 用于滚动计算的窗口大小 (Window sizes for rolling calculations)
  sequence_length: 30     # 模型的输入序列长度 (Input sequence length for the model)

training:
  model_type: 'hybrid'     # 模型类型 'hybrid' 或 'lstm' (Model type 'hybrid' or 'lstm')
  epochs: 30               # 训练轮数 (Number of training epochs)
  batch_size: 32           # 批量大小 (Batch size)
  learning_rate: 0.001     # 学习率 (Learning rate)
  test_size: 0.2           # 用于测试的数据比例 (Proportion of data used for testing)
  random_state: 42         # 用于可复现的训练/测试分割 (For reproducible train/test split)
  patience: 10             # Early Stopping 等待改善的轮数 (Epochs for Early Stopping without improvement)

evaluation:
  n_visualization_samples: 5 # 用于生命周期可视化的随机抽样产品数量 (Number of products randomly sampled for lifecycle visualization)
  output_dir: results       # 保存绘图和Excel文件的目录 (Directory to save plots and Excel files)

excel:
  output_filename: 'product_lifecycle_results.xlsx' # Excel输出文件名 (Excel output filename)
```

## 3. `product-lifecycle-prediction/src/__init__.py`

```python
# 此文件可以是空的，用于将 src 标记为 Python 包
# (This file can be empty, used to mark src as a Python package)
```

## 4. `product-lifecycle-prediction/src/data/__init__.py`

```python
# 此文件可以是空的，用于将 data 标记为 Python 子包
# (This file can be empty, used to mark data as a Python sub-package)
```

## 5. `product-lifecycle-prediction/src/data/load.py`

```python
import pandas as pd
import numpy as np

def load_and_preprocess_data(file_path, unit_price_threshold=0.0, quantity_threshold=0):
    """
    加载并预处理英国零售数据集。
    (Loads and preprocesses the UK retail dataset.)

    Args:
        file_path (str): CSV 数据文件的路径。 (Path to the CSV data file.)
        unit_price_threshold (float): 被视为有效的最低单价。 (Minimum unit price considered valid.)
        quantity_threshold (int): 被视为有效的最低数量。 (Minimum quantity considered valid.)

    Returns:
        pd.DataFrame or None: 预处理后的 DataFrame，如果加载失败则返回 None。
                              (Preprocessed DataFrame, or None if loading fails.)
    """
    print("开始加载数据...") # Start loading data...

    # 尝试确定编码或显式指定
    # (Try to determine encoding or specify explicitly)
    encodings_to_try = ['utf-8', 'ISO-8859-1', 'latin1']
    df = None
    for encoding in encodings_to_try:
        try:
            # 使用指定的编码读取CSV文件
            # (Read CSV file using the specified encoding)
            df = pd.read_csv(file_path, encoding=encoding)
            print(f"数据使用编码成功加载: {encoding}") # Data loaded successfully using encoding...
            break # 成功加载后退出循环 (Exit loop after successful loading)
        except UnicodeDecodeError:
            # 如果解码失败，尝试下一个编码
            # (If decoding fails, try the next encoding)
            print(f"使用编码加载失败: {encoding}") # Failed to load using encoding...
        except FileNotFoundError:
            # 如果文件未找到，打印错误并返回 None
            # (If file not found, print error and return None)
            print(f"错误: 文件未找到于: {file_path}") # ERROR: File not found at...
            return None

    if df is None:
        # 如果所有编码都失败，打印错误并返回 None
        # (If all encodings failed, print error and return None)
        print("错误: 无法使用任何尝试的编码加载数据。") # ERROR: Could not load data using any attempted encodings.
        return None

    print(f"原始数据形状: {df.shape}") # Original data shape...

    # --- 基础清洗 (Basic Cleaning) ---
    # 删除关键列中包含NaN的记录
    # (Remove records containing NaN in key columns)
    original_rows = len(df)
    df.dropna(subset=['InvoiceNo', 'StockCode', 'Quantity', 'UnitPrice', 'CustomerID', 'InvoiceDate'], inplace=True)
    print(f"删除关键列NaN后形状: {df.shape} (移除了 {original_rows - len(df)} 行)") # Shape after removing NaNs in key columns (removed ... rows)

    # 删除无效数量的记录 (Quantity <= threshold)
    # (Remove records with invalid quantity)
    original_rows = len(df)
    df = df[df['Quantity'] > quantity_threshold]
    print(f"过滤数量 (> {quantity_threshold}) 后形状: {df.shape} (移除了 {original_rows - len(df)} 行)") # Shape after filtering quantity (> ...) (removed ... rows)

    # 确保单价有效 (UnitPrice > threshold)
    # (Ensure unit price is valid)
    original_rows = len(df)
    df = df[df['UnitPrice'] > unit_price_threshold]
    print(f"过滤单价 (> {unit_price_threshold}) 后形状: {df.shape} (移除了 {original_rows - len(df)} 行)") # Shape after filtering unit price (> ...) (removed ... rows)

    # 删除测试/异常的StockCode (例如，只有字母或特定代码)
    # (Remove test/abnormal StockCodes (e.g., only letters or specific codes))
    original_rows = len(df)
    df['StockCode'] = df['StockCode'].astype(str) # 确保StockCode是字符串类型 (Ensure StockCode is string type)
    df = df[~df['StockCode'].str.match(r'^[A-Za-z]+$')] # 移除只有字母的代码 (Remove codes with only letters)
    manual_codes = ['POST', 'C2', 'M', 'BANK CHARGES', 'PADS', 'DOT', 'CRUK'] # 手动排除的代码列表 (List of codes to manually exclude)
    df = df[~df['StockCode'].isin(manual_codes)]
    print(f"移除测试/手动StockCodes后形状: {df.shape} (移除了 {original_rows - len(df)} 行)") # Shape after removing test/manual StockCodes (removed ... rows)

    # --- 特征创建与转换 (Feature Creation & Conversion) ---
    # 计算总价 (Calculate TotalPrice)
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

    # 转换日期格式 (Convert date format)
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

    # --- 移除异常值 (Remove Outliers) ---
    # 谨慎使用：移除数量和单价的极端值 (Use with caution: remove extreme values for quantity and unit price)
    q_upper = df['Quantity'].quantile(0.99)
    p_upper = df['UnitPrice'].quantile(0.99)
    initial_rows = len(df)
    df_filtered = df[(df['Quantity'] <= q_upper) & (df['UnitPrice'] <= p_upper)]
    print(f"移除异常值 (数量 <= {q_upper:.2f}, 价格 <= {p_upper:.2f}) 后形状: {df_filtered.shape} (移除了 {initial_rows - len(df_filtered)} 行)") # Shape after removing outliers (...) (removed ... rows)

    return df_filtered # 返回过滤后的 DataFrame (Return the filtered DataFrame)

def create_daily_sales_data(df):
    """
    根据预处理后的 DataFrame 创建每日销售数据。
    (Creates daily sales data from the preprocessed DataFrame.)

    Args:
        df (pd.DataFrame): 预处理后的 DataFrame。 (Preprocessed DataFrame.)

    Returns:
        tuple: 包含以下内容的元组 (A tuple containing):
            - pd.DataFrame: 包含每日销售数据的 DataFrame。 (DataFrame with daily sales data.)
            - dict: 将 StockCode 映射到 Description 的字典。 (Dictionary mapping StockCode to Description.)
    """
    print("创建每日销售数据...") # Creating daily sales data...

    # 按 StockCode 和日期（天）分组，并聚合销售数据
    # (Group by StockCode and date (day), and aggregate sales data)
    grouped = df.groupby(['StockCode', pd.Grouper(key='InvoiceDate', freq='D')])
    daily_sales = grouped.agg(
        Quantity=('Quantity', 'sum'),           # 计算每日总销量 (Calculate total daily quantity)
        TotalPrice=('TotalPrice', 'sum'),         # 计算每日总收入 (Calculate total daily revenue)
        NumTransactions=('InvoiceNo', 'nunique') # 计算每日唯一交易数 (Calculate number of unique daily transactions)
    ).reset_index() # 将分组索引重置为列 (Reset group index to columns)

    # 创建产品描述字典，方便后续使用
    # (Create product description dictionary for later use)
    # 使用 first() 获取每个 StockCode 的第一个遇到的描述
    # (Use first() to get the first encountered description for each StockCode)
    product_desc = df.groupby('StockCode')['Description'].first().to_dict()
    # 将描述映射到每日销售数据中
    # (Map descriptions to daily sales data)
    daily_sales['Description'] = daily_sales['StockCode'].map(product_desc)

    print(f"每日销售数据形状: {daily_sales.shape}") # Daily sales data shape...

    return daily_sales, product_desc
```

## 6. `product-lifecycle-prediction/src/data/features.py`

```python
import pandas as pd
import numpy as np

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
        product_data = create_product_attribute_features(product_data, product_desc)
        
        # 2. 客户行为特征 (需要原始数据)
        # product_data = create_customer_behavior_features(df_original, product_data)
        
        # 3. 市场环境特征
        product_data = create_market_context_features(df_original, product_data, product_desc)
        
        # 4. 非线性特征和交互特征
        product_data = create_nonlinear_features(product_data)
        
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
    product_features_df.fillna(0, inplace=True)
    
    # 执行特征选择与降维
    print("执行特征选择...")
    selected_features_df, feature_importance = perform_feature_selection(
        product_features_df, valid_products)
    
    print(f"特征工程完成，生成了 {len(product_features_df.columns)} 个原始特征，" + 
          f"选择了 {len(selected_features_df.columns) - 3} 个最终特征")  # 减去StockCode/InvoiceDate/LifecyclePhase
    
    return selected_features_df, valid_products, feature_importance
```

## 7. `product-lifecycle-prediction/src/lifecycle/__init__.py`

```python
# 此文件可以是空的，用于将 lifecycle 标记为 Python 子包
# (This file can be empty, used to mark lifecycle as a Python sub-package)
```

## 8. `product-lifecycle-prediction/src/lifecycle/simple.py`

```python
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
```

## 9. `product-lifecycle-prediction/src/evaluation/__init__.py`

```python
# 此文件可以是空的，用于将 evaluation 标记为 Python 子包
# (This file can be empty, used to mark evaluation as a Python sub-package)
```

## 10. `product-lifecycle-prediction/src/evaluation/metrics.py`

```python
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

def evaluate_lifecycle_labels_quality(product_features_df, valid_products):
    """
    评估生成的生命周期标签的质量。
    (Evaluates the quality of the generated lifecycle labels.)

    Args:
        product_features_df (pd.DataFrame): 包含产品特征和 'LifecyclePhase' 标签的 DataFrame。
                                            (DataFrame containing product features and 'LifecyclePhase' labels.)
        valid_products (list): 有效产品 StockCode 的列表。
                               (List of valid product StockCodes.)

    Returns:
        tuple: 包含以下内容的元组 (A tuple containing):
            - dict: 包含质量指标 ('invalid_transition_ratio', 'stability', 'avg_phase_durations') 的字典。
                    (Dictionary containing quality metrics ('invalid_transition_ratio', 'stability', 'avg_phase_durations').)
            - list: 所有观察到的转换的列表，格式为 (from_phase, to_phase) 元组。
                    (List of all observed transitions as (from_phase, to_phase) tuples.)
    """
    print("评估生命周期标签质量...") # Evaluating lifecycle label quality...

    invalid_transitions_count = 0 # 无效转换计数 (Invalid transition count)
    total_transitions = 0         # 总转换计数 (Total transition count)
    stability_scores = []         # 每个产品的稳定性得分列表 (List of stability scores for each product)
    all_transitions = []          # 存储所有转换的列表 (List to store all transitions)
    # 存储每个阶段持续时间的列表 (Dictionary to store durations for each phase)
    phase_durations = {0: [], 1: [], 2: [], 3: []} # 0: Intro, 1: Growth, 2: Maturity, 3: Decline

    # 定义无效转换对 (可以根据业务逻辑调整)
    # (Define invalid transition pairs (can be adjusted based on business logic))
    # 例如：导入期 -> 衰退期, 衰退期 -> 导入期, 衰退期 -> 成长期
    # (e.g., Introduction -> Decline, Decline -> Introduction, Decline -> Growth)
    invalid_transition_pairs = {(0, 3), (3, 0), (3, 1)}

    for product in valid_products:
        # 获取并排序单个产品的数据 (Get and sort data for a single product)
        product_data = product_features_df[product_features_df['StockCode'] == product].sort_values('InvoiceDate')
        labels = product_data['LifecyclePhase'].values # 获取标签序列 (Get the label sequence)

        if len(labels) < 2:
            # 如果只有一个数据点，则稳定性为 1.0 (If only one data point, stability is 1.0)
            stability_scores.append(1.0)
            continue # 处理下一个产品 (Process next product)

        transitions_in_product = 0 # 当前产品的转换次数 (Transition count for the current product)
        current_phase_start_index = 0 # 当前阶段开始的索引 (Index where the current phase started)

        # 遍历标签序列以检测转换 (Iterate through the label sequence to detect transitions)
        for i in range(1, len(labels)):
            if labels[i] != labels[i-1]:
                # --- 检测到转换 (Transition Detected) ---
                from_phase = int(labels[i-1]) # 转换前的阶段 (Phase before transition)
                to_phase = int(labels[i])     # 转换后的阶段 (Phase after transition)
                transition = (from_phase, to_phase) # 创建转换元组 (Create transition tuple)

                all_transitions.append(transition) # 记录所有转换 (Record all transitions)
                total_transitions += 1             # 增加总转换计数 (Increment total transition count)
                transitions_in_product += 1        # 增加产品内转换计数 (Increment in-product transition count)

                # --- 记录前一个阶段的持续时间 (Record duration of the previous phase) ---
                duration = i - current_phase_start_index
                if from_phase in phase_durations:
                    phase_durations[from_phase].append(duration)

                # 更新新阶段的开始索引 (Update start index for the new phase)
                current_phase_start_index = i

                # --- 检查无效转换 (Check for invalid transitions) ---
                if transition in invalid_transition_pairs:
                    invalid_transitions_count += 1

        # --- 记录最后一个阶段的持续时间 (Record duration of the last phase) ---
        last_phase = int(labels[-1])
        last_duration = len(labels) - current_phase_start_index
        if last_phase in phase_durations:
            phase_durations[last_phase].append(last_duration)

        # --- 计算产品的稳定性 (Calculate product stability) ---
        # 稳定性 = 1 - 转换频率 (Stability = 1 - transition frequency)
        change_ratio = transitions_in_product / (len(labels) - 1)
        stability_scores.append(1.0 - change_ratio)

    # --- 计算总体指标 (Calculate Overall Metrics) ---
    # 无效转换率 (Invalid transition ratio)
    invalid_transition_ratio = invalid_transitions_count / max(1, total_transitions) if total_transitions > 0 else 0
    # 平均稳定性 (Average stability)
    avg_stability = np.mean(stability_scores) if stability_scores else 1.0 # 如果列表为空，则为 1.0 (1.0 if list is empty)

    # --- 计算平均阶段持续时间 (Calculate Average Phase Durations) ---
    avg_phase_durations = {}
    phase_names = {0: '导入期', 1: '成长期', 2: '成熟期', 3: '衰退期'} # Phase names
    for phase, durations in phase_durations.items():
        if durations: # 仅在有持续时间记录时计算平均值 (Only calculate average if durations were recorded)
            avg_phase_durations[phase] = np.mean(durations)
        else:
            avg_phase_durations[phase] = 0 # 如果没有记录，则平均持续时间为 0 (If no records, average duration is 0)

    # --- 打印结果 (Print Results) ---
    print("\n标签质量指标:") # Label Quality Metrics:
    print(f"- 无效转换率: {invalid_transition_ratio:.4f}") # - Invalid Transition Ratio:
    print(f"- 平均标签稳定性: {avg_stability:.4f}") # - Average Label Stability:
    for phase, avg_duration in avg_phase_durations.items():
         print(f"- '{phase_names.get(phase, phase)}' 平均持续时间: {avg_duration:.2f} 天") # - Average Duration for Phase '...': ... days

    # --- 准备返回的指标字典 (Prepare metrics dictionary to return) ---
    quality_metrics = {
        'invalid_transition_ratio': invalid_transition_ratio,
        'stability': avg_stability,
        'avg_phase_durations': avg_phase_durations # 添加平均持续时间 (Add average durations)
    }

    return quality_metrics, all_transitions


def get_classification_report(y_true, y_pred, target_names):
    """
    生成分类报告和混淆矩阵。
    (Generates a classification report and confusion matrix.)

    Args:
        y_true (list or np.ndarray): 真实标签。 (True labels.)
        y_pred (list or np.ndarray): 预测标签。 (Predicted labels.)
        target_names (list): 目标类别的名称。 (Names of the target classes.)

    Returns:
        tuple: 包含以下内容的元组 (A tuple containing):
            - str: 格式化的分类报告字符串。 (Formatted classification report string.)
            - np.ndarray: 混淆矩阵。 (Confusion matrix.)
            - dict: 作为字典的分类报告。 (Classification report as a dictionary.)
    """
    # 生成报告字符串，zero_division 控制除零行为
    # (Generate report string, zero_division controls division-by-zero behavior)
    report_str = classification_report(y_true, y_pred, target_names=target_names, zero_division=0)
    # 生成报告字典 (Generate report dictionary)
    report_dict = classification_report(y_true, y_pred, target_names=target_names, output_dict=True, zero_division=0)
    # 生成混淆矩阵 (Generate confusion matrix)
    cm = confusion_matrix(y_true, y_pred, labels=range(len(target_names))) # 确保标签顺序正确 (Ensure label order is correct)

    return report_str, cm, report_dict
```

## 11. `product-lifecycle-prediction/src/evaluation/visualization.py`

```python
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd # 为函数内数据处理添加 (Added for data handling within function)

# --- Matplotlib 设置 (Matplotlib Settings) ---
# 确保字体设置在此处或全局配置中完成
# (Ensure font settings are done here or in global config)
# 设置 Matplotlib 支持中文显示的字体 (Set Matplotlib font to support Chinese display)
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif'] # 添加 SimHei (Add SimHei)
plt.rcParams['axes.unicode_minus'] = False # 解决负号显示问题 (Resolve issue with displaying minus sign)

def visualize_lifecycle_labels(product_features_df, valid_products, product_desc, n_samples=5, output_dir='results'):
    """
    可视化样本产品的生命周期标签并保存为图片。
    (Visualizes lifecycle labels for a sample of products and saves them as images.)

    Args:
        product_features_df (pd.DataFrame): 包含产品特征和 'LifecyclePhase' 标签的 DataFrame。
                                            (DataFrame containing product features and 'LifecyclePhase' labels.)
        valid_products (list): 有效产品 StockCode 的列表。 (List of valid product StockCodes.)
        product_desc (dict): 将 StockCode 映射到 Description 的字典。 (Dictionary mapping StockCode to Description.)
        n_samples (int): 用于可视化的随机抽样产品数量。 (Number of products to randomly sample for visualization.)
        output_dir (str): 保存绘图的目录。 (Directory to save the plots.)

    Returns:
        list: 可视化的样本产品的 StockCode 列表。
              (List of StockCodes of the visualized sample products.)
    """
    print(f"可视化 {n_samples} 个随机产品的生命周期标签并保存图片...") # Visualizing lifecycle labels for ... random products and saving plots...

    # 创建输出目录 (如果不存在) (Create output directory (if it doesn't exist))
    os.makedirs(output_dir, exist_ok=True)

    # --- 随机选择样本产品 (Randomly Select Sample Products) ---
    if not valid_products: # 检查 valid_products 是否为空 (Check if valid_products is empty)
        print("警告: 没有可用于可视化的有效产品。") # WARNING: No valid products available for visualization.
        return []

    if n_samples >= len(valid_products):
        # 如果请求的样本数大于或等于有效产品数，则选择所有有效产品
        # (If requested sample count is greater than or equal to valid products, select all valid products)
        sample_products = valid_products
        n_samples = len(valid_products) # 确保 n_samples 不大于有效产品数 (Ensure n_samples is not greater than the number of valid products)
    else:
        # 否则，随机选择 n_samples 个产品
        # (Otherwise, randomly select n_samples products)
        sample_products = np.random.choice(valid_products, n_samples, replace=False).tolist()

    if n_samples == 0: # 再次检查，以防 valid_products 为空 (Double check in case valid_products was empty)
         print("警告: 没有可用于可视化的有效产品。") # WARNING: No valid products available for visualization.
         return []


    # --- 定义阶段颜色和名称 (Define Phase Colors and Names) ---
    phase_colors = {0: 'green', 1: 'blue', 2: 'orange', 3: 'red'} # 颜色映射 (Color mapping)
    phase_names = {0: '导入期', 1: '成长期', 2: '成熟期', 3: '衰退期'} # 名称映射 (Name mapping)

    # --- 为每个样本产品创建绘图 (Create Plot for Each Sample Product) ---
    num_plots = len(sample_products)
    # squeeze=False 确保即使只有一个子图，axes 也是 2D 数组
    # (squeeze=False ensures axes is always a 2D array, even with one subplot)
    fig, axes = plt.subplots(num_plots, 1, figsize=(15, 5 * num_plots), squeeze=False)

    for i, product in enumerate(sample_products):
        ax = axes[i, 0] # 获取当前子图的 Axes 对象 (Get the Axes object for the current subplot)
        try:
            # 获取并排序产品数据 (Get and sort product data)
            product_data = product_features_df[product_features_df['StockCode'] == product].sort_values('InvoiceDate')

            if product_data.empty:
                # 如果找不到产品数据，显示警告信息
                # (If product data is not found, display a warning message)
                print(f"警告: 未找到样本产品 {product} 的数据。") # WARNING: No data found for sample product...
                ax.set_title(f'产品: {product} - 未找到数据', fontsize=12) # Product: ... - Data Not Found
                continue # 处理下一个产品 (Process next product)

            # --- 绘制销售曲线 (Plot Sales Curve) ---
            sales_col = 'Sales_30d' # 要绘制的销售指标 (Sales metric to plot)
            if sales_col not in product_data.columns:
                sales_col = 'Quantity' # 如果 Sales_30d 不存在，回退到每日 Quantity (Fallback to daily Quantity if Sales_30d doesn't exist)

            # 绘制销售曲线，使用黑色实线
            # (Plot the sales curve using a black solid line)
            ax.plot(product_data['InvoiceDate'], product_data[sales_col], 'k-', linewidth=1.5, label=f'{sales_col} 销量') # Sales

            # --- 添加生命周期阶段背景色 (Add Lifecycle Phase Background Colors) ---
            legend_added = set() # 跟踪已添加到图例的阶段 (Track phases already added to the legend)
            min_date_overall = product_data['InvoiceDate'].min() # 最早日期 (Earliest date)
            max_date_overall = product_data['InvoiceDate'].max() # 最晚日期 (Latest date)

            # 遍历数据以查找阶段转换点并绘制背景区域
            # (Iterate through data to find phase transition points and draw background spans)
            last_phase = -1 # 上一个时间点的阶段 (Phase at the previous time point)
            start_date = min_date_overall # 当前阶段的开始日期 (Start date of the current phase)

            for _, row in product_data.iterrows():
                current_phase = int(row['LifecyclePhase']) # 当前阶段 (Current phase)
                current_date = row['InvoiceDate']        # 当前日期 (Current date)

                # 如果阶段发生变化 (且不是第一次迭代)
                # (If the phase has changed (and it's not the first iteration))
                if current_phase != last_phase and last_phase != -1:
                    # 绘制上一个阶段的背景区域 (Draw the background span for the previous phase)
                    phase_name = phase_names.get(last_phase, '未知') # Get phase name
                    label_text = phase_name if last_phase not in legend_added else None # 仅在第一次出现时添加图例标签 (Add legend label only on first occurrence)
                    ax.axvspan(start_date, current_date, alpha=0.3, color=phase_colors.get(last_phase, 'grey'), label=label_text)
                    legend_added.add(last_phase) # 将此阶段标记为已添加到图例 (Mark this phase as added to the legend)
                    start_date = current_date # 更新新阶段的开始日期 (Update the start date for the new phase)

                last_phase = current_phase # 更新上一个阶段 (Update the last phase)

            # 绘制最后一个阶段的背景区域 (Draw the background span for the last phase)
            if last_phase != -1:
                phase_name = phase_names.get(last_phase, '未知')
                label_text = phase_name if last_phase not in legend_added else None
                ax.axvspan(start_date, max_date_overall, alpha=0.3, color=phase_colors.get(last_phase, 'grey'), label=label_text)
                legend_added.add(last_phase)

            # --- 设置标题和标签 (Set Title and Labels) ---
            product_name = product_desc.get(product, product) # 获取产品名称 (Get product name)
            ax.set_title(f'产品: {product} - {product_name}', fontsize=14) # Product: ... - ...
            ax.set_ylabel(f'{sales_col} 销量', fontsize=12) # Sales
            ax.grid(True, linestyle='--', alpha=0.7) # 添加网格线 (Add grid lines)
            ax.tick_params(axis='x', rotation=15) # 旋转X轴标签以便阅读 (Rotate x-axis labels for readability)

            # --- 添加图例 (Add Legend) ---
            if legend_added: # 仅在有标签添加时显示图例 (Only show legend if labels were added)
                ax.legend(loc='upper left')

        except Exception as e:
            # 捕获并打印可视化过程中的任何错误
            # (Catch and print any errors during visualization)
            print(f"可视化产品 {product} 时出错: {type(e).__name__}: {str(e)}") # Error visualizing product...
            import traceback
            print(traceback.format_exc())
            ax.set_title(f'产品: {product} - 可视化错误', fontsize=12) # Product: ... - Visualization Error

    # --- 保存并关闭绘图 (Save and Close Plot) ---
    plt.tight_layout() # 调整布局以防止重叠 (Adjust layout to prevent overlap)
    plot_filename = os.path.join(output_dir, "lifecycle_visualization.png")
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight') # 保存为 PNG 文件 (Save as PNG file)
    plt.close(fig) # 关闭图形以释放内存 (Close the figure to free memory)
    print(f"生命周期可视化图已保存至: {plot_filename}") # Lifecycle visualization plot saved to...

    return sample_products # 返回实际可视化的产品列表 (Return the list of products actually visualized)


def plot_training_history(history, output_dir='results'):
    """
    绘制训练和验证损失及准确率。
    (Plots the training and validation loss and accuracy.)

    Args:
        history (dict): 包含训练历史的字典 ('loss', 'accuracy', 'val_loss', 'val_accuracy')。
                        (Dictionary containing training history ('loss', 'accuracy', 'val_loss', 'val_accuracy').)
        output_dir (str): 保存绘图的目录。 (Directory to save the plot.)
    """
    # 创建输出目录 (如果不存在) (Create output directory (if it doesn't exist))
    os.makedirs(output_dir, exist_ok=True)
    # 获取训练的轮数 (Get the number of epochs trained)
    epochs_range = range(1, len(history.get('loss', [])) + 1)

    if not epochs_range: # 如果历史记录为空，则不绘制 (If history is empty, do not plot)
        print("警告: 训练历史记录为空，无法绘制。") # WARNING: Training history is empty, cannot plot.
        return

    plt.figure(figsize=(12, 5)) # 创建新的图形 (Create a new figure)

    # --- 绘制损失曲线 (Plot Loss Curves) ---
    plt.subplot(1, 2, 1) # 创建第一个子图 (Create the first subplot)
    plt.plot(epochs_range, history['loss'], label='训练损失') # Plot training loss
    plt.plot(epochs_range, history['val_loss'], label='验证损失') # Plot validation loss
    plt.title('模型损失') # Model Loss
    plt.xlabel('轮次') # Epochs
    plt.ylabel('损失') # Loss
    plt.legend() # 显示图例 (Show legend)
    plt.grid(True) # 显示网格 (Show grid)

    # --- 绘制准确率曲线 (Plot Accuracy Curves) ---
    plt.subplot(1, 2, 2) # 创建第二个子图 (Create the second subplot)
    plt.plot(epochs_range, history['accuracy'], label='训练准确率') # Plot training accuracy
    plt.plot(epochs_range, history['val_accuracy'], label='验证准确率') # Plot validation accuracy
    plt.title('模型准确率') # Model Accuracy
    plt.xlabel('轮次') # Epochs
    plt.ylabel('准确率') # Accuracy
    plt.legend() # 显示图例 (Show legend)
    plt.grid(True) # 显示网格 (Show grid)

    # --- 保存并关闭绘图 (Save and Close Plot) ---
    plt.tight_layout() # 调整布局 (Adjust layout)
    plot_filename = os.path.join(output_dir, "training_history.png")
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight') # 保存图形 (Save the figure)
    plt.close() # 关闭图形 (Close the figure)
    print(f"训练历史图已保存至: {plot_filename}") # Training history plot saved to...


def plot_confusion_matrix(cm, target_names, output_dir='results'):
    """
    绘制混淆矩阵。
    (Plots the confusion matrix.)

    Args:
        cm (np.ndarray): 混淆矩阵。 (Confusion matrix.)
        target_names (list): 目标类别的名称。 (Names of the target classes.)
        output_dir (str): 保存绘图的目录。 (Directory to save the plot.)
    """
    # 创建输出目录 (如果不存在) (Create output directory (if it doesn't exist))
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(10, 8)) # 创建新的图形 (Create a new figure)

    # 使用 seaborn 绘制热力图 (Use seaborn to plot the heatmap)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', # annot=True 显示数值, fmt='d' 整数格式, cmap 颜色主题 (annot=True shows values, fmt='d' integer format, cmap color theme)
                xticklabels=target_names, # 设置 X 轴标签 (Set x-axis labels)
                yticklabels=target_names) # 设置 Y 轴标签 (Set y-axis labels)
    plt.xlabel('预测标签', fontsize=12) # Predicted Label
    plt.ylabel('真实标签', fontsize=12) # True Label
    plt.title('生命周期阶段预测混淆矩阵', fontsize=14) # Confusion Matrix for Lifecycle Phase Prediction

    # --- 保存并关闭绘图 (Save and Close Plot) ---
    plot_filename = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight') # 保存图形 (Save the figure)
    plt.close() # 关闭图形 (Close the figure)
    print(f"混淆矩阵图已保存至: {plot_filename}") # Confusion matrix plot saved to...
```

## 12. `product-lifecycle-prediction/src/models/__init__.py`

```python
# 此文件可以是空的，用于将 models 标记为 Python 子包
# (This file can be empty, used to mark models as a Python sub-package)
```

## 13. `product-lifecycle-prediction/src/models/builders.py`

```python
import torch
import torch.nn as nn

class HybridModel(nn.Module):
    """
    用于产品生命周期分类的混合 CNN+LSTM 模型。
    (A hybrid CNN+LSTM model for product lifecycle classification.)
    """
    def __init__(self, input_dim, seq_length, num_classes=4, cnn_filters=64, lstm_units=64, dropout_rate=0.3):
        """
        初始化 HybridModel。
        (Initializes the HybridModel.)

        Args:
            input_dim (int): 每个时间步的特征数量。 (Number of features in each time step.)
            seq_length (int): 输入序列的长度。 (Length of the input sequence.)
            num_classes (int): 输出类别的数量 (生命周期阶段)。 (Number of output classes (lifecycle phases).)
            cnn_filters (int): CNN 层中的滤波器数量。 (Number of filters in the CNN layers.)
            lstm_units (int): LSTM 层中的单元数量。 (Number of units in the LSTM layer.)
            dropout_rate (float): 用于正则化的 Dropout 率。 (Dropout rate for regularization.)
        """
        super(HybridModel, self).__init__()

        # --- CNN 部分 (CNN Part) ---
        # 第一层卷积 (First Convolutional Layer)
        self.conv1 = nn.Conv1d(input_dim, cnn_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(cnn_filters) # 添加批量归一化 (Add Batch Normalization)
        self.pool1 = nn.MaxPool1d(2) # 最大池化层 (Max Pooling Layer)
        # 第二层卷积 (Second Convolutional Layer)
        self.conv2 = nn.Conv1d(cnn_filters, cnn_filters * 2, kernel_size=3, padding=1) # 增加滤波器数量 (Increase filter count)
        self.bn2 = nn.BatchNorm1d(cnn_filters * 2) # 添加批量归一化 (Add Batch Normalization)
        self.pool2 = nn.MaxPool1d(2) # 最大池化层 (Max Pooling Layer)

        # 计算 CNN 输出维度 (Calculate CNN output dimension)
        # 经过两次池化，序列长度减半再减半 (Sequence length halved twice due to two pooling layers)
        cnn_output_len = seq_length // 4
        cnn_output_dim = (cnn_filters * 2) * cnn_output_len # 展平后的维度 (Dimension after flattening)

        # --- LSTM 部分 (LSTM Part) ---
        # 使用双向 LSTM 以可能获得更好的结果 (Use bidirectional LSTM for potentially better results)
        self.lstm = nn.LSTM(input_dim, lstm_units, batch_first=True, bidirectional=True)
        lstm_output_dim = lstm_units * 2 # 双向 LSTM 输出维度是单元数的两倍 (*2 for bidirectional)

        # --- 全连接层 (Fully Connected Layers) ---
        # 输入维度是展平的 CNN 输出和 LSTM 输出的拼接
        # (Input dimension is the concatenation of flattened CNN output and LSTM output)
        self.fc1 = nn.Linear(cnn_output_dim + lstm_output_dim, 128) # 增加 FC 层单元数 (Increased FC units)
        self.dropout = nn.Dropout(dropout_rate) # Dropout 层用于正则化 (Dropout layer for regularization)
        self.fc2 = nn.Linear(128, num_classes) # 输出层 (Output layer)

        # --- 激活函数 (Activation Functions) ---
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU() # LeakyReLU 可能有助于缓解梯度消失 (LeakyReLU might help mitigate vanishing gradients)

    def forward(self, x):
        """
        执行模型的前向传播。
        (Performs the forward pass through the model.)

        Args:
            x (torch.Tensor): 输入张量，形状为 [batch, seq_len, features]。
                              (Input tensor of shape [batch, seq_len, features].)

        Returns:
            torch.Tensor: 模型输出，形状为 [batch, num_classes]。
                          (Model output of shape [batch, num_classes].)
        """
        batch_size = x.size(0)

        # --- CNN 前向传播 (CNN Forward Pass) ---
        # 输入需要转置以匹配 Conv1d 期望的格式 [batch, channels, seq_len]
        # (Input needs to be transposed to match Conv1d expected format [batch, channels, seq_len])
        x_cnn = x.transpose(1, 2)
        # 应用 Conv1 -> BN -> Activation -> Pool
        # (Apply Conv1 -> BN -> Activation -> Pool)
        x_cnn = self.pool1(self.leaky_relu(self.bn1(self.conv1(x_cnn))))
        # 应用 Conv2 -> BN -> Activation -> Pool
        # (Apply Conv2 -> BN -> Activation -> Pool)
        x_cnn = self.pool2(self.leaky_relu(self.bn2(self.conv2(x_cnn))))
        # 展平 CNN 输出 (Flatten the CNN output)
        x_cnn = x_cnn.reshape(batch_size, -1)

        # --- LSTM 前向传播 (LSTM Forward Pass) ---
        # 输入 x 已经是 [batch, seq_len, features]，适合 batch_first=True
        # (Input x is already [batch, seq_len, features], suitable for batch_first=True)
        # lstm_out: [batch, seq_len, num_directions * hidden_size]
        # hidden: [num_layers * num_directions, batch, hidden_size]
        lstm_out, (hidden, _) = self.lstm(x)
        # 拼接最后一个时间步的前向和后向隐藏状态
        # (Concatenate the last hidden states from both forward and backward directions)
        # hidden[-2,:,:] 是最后一个时间步的前向隐藏状态 (Forward hidden state at last time step)
        # hidden[-1,:,:] 是第一个时间步的后向隐藏状态 (Backward hidden state at first time step)
        x_lstm = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        # 或者，可以使用 LSTM 输出的最后一个时间步 (Alternatively, can use the last time step of LSTM output)
        # x_lstm = torch.cat((lstm_out[:, -1, :self.lstm.hidden_size], lstm_out[:, 0, self.lstm.hidden_size:]), dim=1) # 注意后向是从序列开头取

        # --- 拼接 CNN 和 LSTM 特征 (Concatenate CNN and LSTM Features) ---
        combined = torch.cat((x_cnn, x_lstm), dim=1)

        # --- 全连接层前向传播 (Fully Connected Layers Forward Pass) ---
        out = self.leaky_relu(self.fc1(combined)) # 应用 FC1 -> Activation
        out = self.dropout(out)                   # 应用 Dropout
        out = self.fc2(out)                       # 应用 FC2 (输出层，无激活函数，因为 CrossEntropyLoss 会处理)
                                                  # (Output layer, no activation as CrossEntropyLoss handles it)

        return out


class LSTMModel(nn.Module):
    """
    用于产品生命周期分类的纯 LSTM 模型。
    (A pure LSTM model for product lifecycle classification.)
    """
    def __init__(self, input_dim, seq_length, num_classes=4, lstm_units=128, num_layers=2, dropout_rate=0.3):
        """
        初始化 LSTMModel。
        (Initializes the LSTMModel.)

        Args:
            input_dim (int): 每个时间步的特征数量。 (Number of features in each time step.)
            seq_length (int): 输入序列的长度 (模型内部不直接使用，但有助于理解)。
                              (Length of the input sequence (not used directly internally, but good for context).)
            num_classes (int): 输出类别的数量。 (Number of output classes.)
            lstm_units (int): LSTM 层中的单元数量。 (Number of units in the LSTM layers.)
            num_layers (int): 堆叠的 LSTM 层数。 (Number of stacked LSTM layers.)
            dropout_rate (float): LSTM 层之间和 FC 层之前的 Dropout 率。
                                  (Dropout rate between LSTM layers and before the FC layer.)
        """
        super(LSTMModel, self).__init__()

        # --- LSTM 层 (LSTM Layers) ---
        self.lstm = nn.LSTM(input_dim,               # 输入特征维度 (Input feature dimension)
                            lstm_units,            # LSTM 单元数 (Number of LSTM units)
                            num_layers=num_layers,   # LSTM 层数 (Number of LSTM layers)
                            batch_first=True,      # 输入格式为 [batch, seq, feature] (Input format is [batch, seq, feature])
                            dropout=dropout_rate if num_layers > 1 else 0, # 层间的 Dropout (Dropout between layers)
                            bidirectional=True)    # 使用双向 LSTM (Use bidirectional LSTM)
        lstm_output_dim = lstm_units * 2 # 双向 LSTM 输出维度 (*2 for bidirectional)

        # --- 全连接层 (Fully Connected Layers) ---
        self.fc1 = nn.Linear(lstm_output_dim, 64) # 输入维度匹配 LSTM 输出 (Input dim matches LSTM output)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(64, num_classes)

        # --- 激活函数 (Activation Functions) ---
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x):
        """
        执行模型的前向传播。
        (Performs the forward pass through the model.)

        Args:
            x (torch.Tensor): 输入张量，形状为 [batch, seq_len, features]。
                              (Input tensor of shape [batch, seq_len, features].)

        Returns:
            torch.Tensor: 模型输出，形状为 [batch, num_classes]。
                          (Model output of shape [batch, num_classes].)
        """
        # --- LSTM 前向传播 (LSTM Forward Pass) ---
        # 输入 x: [batch, seq_len, features]
        # lstm_out: [batch, seq_len, num_directions * hidden_size]
        # hidden: [num_layers * num_directions, batch, hidden_size]
        lstm_out, (hidden, _) = self.lstm(x)

        # --- 获取最终的 LSTM 表示 (Get Final LSTM Representation) ---
        # 拼接最后一层的前向和后向隐藏状态
        # (Concatenate the forward and backward hidden states from the last layer)
        # hidden 形状: [num_layers*2, batch, lstm_units]
        # (hidden shape: [num_layers*2, batch, lstm_units])
        # 最后一个前向隐藏状态索引是 -2 (Last forward hidden state index is -2)
        # 最后一个后向隐藏状态索引是 -1 (Last backward hidden state index is -1)
        forward_last_hidden = hidden[-2,:,:]
        backward_last_hidden = hidden[-1,:,:]
        x_lstm = torch.cat((forward_last_hidden, backward_last_hidden), dim=1)

        # 或者使用 LSTM 输出的最后一个时间步 (Alternatively, use the last time step of LSTM output)
        # forward_last_output = lstm_out[:, -1, :self.lstm.hidden_size]
        # backward_last_output = lstm_out[:, 0, self.lstm.hidden_size:] # 后向从序列开头取 (Backward takes from the beginning of sequence)
        # x_lstm = torch.cat((forward_last_output, backward_last_output), dim=1)

        # --- 全连接层前向传播 (Fully Connected Layers Forward Pass) ---
        x = self.leaky_relu(self.fc1(x_lstm)) # 应用 FC1 -> Activation
        x = self.dropout(x)                  # 应用 Dropout
        x = self.fc2(x)                      # 应用 FC2 (输出层) (Apply FC2 (output layer))

        return x
```

## 14. `product-lifecycle-prediction/src/models/training.py`

```python
import os
import numpy as np
import pandas as pd # 添加导入 (Added import)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# 导入模型构建器 (Import model builders)
from .builders import HybridModel, LSTMModel
# 导入评估指标和可视化工具 (Import evaluation metrics and visualization)
from src.evaluation.metrics import get_classification_report
from src.evaluation.visualization import plot_training_history, plot_confusion_matrix

class ProductLifecycleDataset(Dataset):
    """用于产品生命周期数据的 PyTorch Dataset 类。
       (PyTorch Dataset class for product lifecycle data.)"""
    def __init__(self, X, y):
        # 确保 X 和 y 是 numpy 数组或兼容类型，然后转换为 Tensor
        # (Ensure X and y are numpy arrays or compatible before converting to tensors)
        self.X = torch.FloatTensor(np.array(X)) # 特征数据 (Feature data)
        self.y = torch.LongTensor(np.array(y))   # 标签数据 (Label data)

    def __len__(self):
        # 返回数据集中的样本数量 (Return the number of samples in the dataset)
        return len(self.y)

    def __getitem__(self, idx):
        # 根据索引获取单个样本 (Get a single sample by index)
        return self.X[idx], self.y[idx]

def prepare_sequences(product_features_df, valid_products, seq_length=30):
    """
    准备用于模型训练的序列数据。按产品对特征进行缩放。
    (Prepares sequence data for model training. Scales features per product.)

    Args:
        product_features_df (pd.DataFrame): 包含产品特征的 DataFrame。
                                            (DataFrame containing product features.)
        valid_products (list): 有效产品 StockCode 的列表。
                               (List of valid product StockCodes.)
        seq_length (int): 要创建的序列长度。 (Length of the sequences to create.)

    Returns:
        tuple: 包含以下内容的元组 (A tuple containing):
            - np.ndarray: 输入序列数组 (X)。 (Array of input sequences (X).)
            - np.ndarray: 目标标签数组 (y)。 (Array of target labels (y).)
            - list: 每个序列对应的产品 ID 列表。 (List of product IDs for each sequence.)
            - list: 每个序列对应的日期列表 (结束日期)。 (List of dates for each sequence (end date).)
            - dict: 存储每个产品 StandardScaler 对象的字典。 (Dictionary storing StandardScaler objects for each product.)
    """
    print(f"准备序列数据 (序列长度={seq_length})...") # Preparing sequence data (sequence length=...)...

    X_all, y_all = [], [] # 存储所有序列和标签 (Lists to store all sequences and labels)
    product_ids_all, dates_all = [], [] # 存储对应的产品ID和日期 (Lists to store corresponding product IDs and dates)
    scalers = {} # 存储每个产品的 StandardScaler (Dictionary to store StandardScaler for each product)

    # --- 定义要使用的特征列 (Define Feature Columns to Use) ---
    # 包括在 `create_features` 中生成的所有数值特征
    # (Include all numerical features generated in `create_features`)
    # 排除非数值列和目标列 (Exclude non-numeric and target columns)
    feature_cols = [col for col in product_features_df.columns
                    if product_features_df[col].dtype in [np.float64, np.int64, float, int] # 处理 pandas 和 numpy 类型 (Handle pandas and numpy types)
                    and col not in ['StockCode', 'Description', 'InvoiceDate', 'LifecyclePhase']]
    # 排除可能噪声较大或不是良好预测因子的原始日度列
    # (Exclude original daily columns that might be noisy or not good predictors)
    cols_to_exclude = ['Quantity', 'TotalPrice', 'NumTransactions']
    feature_cols = [col for col in feature_cols if col not in cols_to_exclude]

    if not feature_cols:
         raise ValueError("在 prepare_sequences 中未找到合适的特征列。请检查 features.py。") # No suitable feature columns found... Check features.py.
    print(f"使用的特征列 ({len(feature_cols)}): {feature_cols}") # Feature columns used...

    # --- 为每个有效产品处理数据 (Process Data for Each Valid Product) ---
    for product in valid_products:
        # 获取并排序产品数据 (Get and sort product data)
        product_data = product_features_df[product_features_df['StockCode'] == product].sort_values('InvoiceDate')

        # 确保有足够的数据来创建至少一个序列 (包括标签)
        # (Ensure enough data to create at least one sequence (including the label))
        if len(product_data) < seq_length + 1:
            continue # 跳过数据不足的产品 (Skip products with insufficient data)

        # 提取特征和标签 (Extract features and labels)
        features = product_data[feature_cols].values
        labels = product_data['LifecyclePhase'].values

        # --- 检查并处理非有限值 (Check and Handle Non-finite Values) ---
        # 理论上在 create_features 中已处理，但作为安全检查
        # (Theoretically handled in create_features, but as a safety check)
        if not np.all(np.isfinite(features)):
            print(f"警告: 产品 {product} 在特征中包含非有限值，将使用 0 替换。") # WARNING: Product ... contains non-finite values in features, replacing with 0.
            features = np.nan_to_num(features) # 使用 0 替换 NaN 和 Inf (Replace NaN and Inf with 0)

        # --- 按产品缩放特征 (Scale Features Per Product) ---
        try:
            scaler = StandardScaler() # 创建标准化器 (Create a scaler)
            features_scaled = scaler.fit_transform(features) # 拟合并转换特征 (Fit and transform features)
            scalers[product] = scaler # 存储此产品的缩放器 (Store the scaler for this product)

            # 再次检查缩放后的非有限值 (Double-check for non-finite values after scaling)
            if not np.all(np.isfinite(features_scaled)):
                print(f"警告: 产品 {product} 缩放后包含非有限值，跳过此产品。") # WARNING: Product ... contains non-finite values after scaling, skipping this product.
                continue

            # --- 创建序列 (Create Sequences) ---
            product_X, product_y, product_pids, product_dates = [], [], [], []
            # 迭代数据以创建长度为 seq_length 的序列
            # (Iterate through the data to create sequences of length seq_length)
            for i in range(len(product_data) - seq_length):
                product_X.append(features_scaled[i : i + seq_length]) # 提取序列特征 (Extract sequence features)
                product_y.append(labels[i + seq_length])            # 序列结束后的标签 (Label after the sequence ends)
                product_pids.append(product)                         # 记录产品 ID (Record product ID)
                product_dates.append(product_data['InvoiceDate'].iloc[i + seq_length]) # 记录序列结束日期 (Record sequence end date)

            # 将此产品的序列添加到总列表中 (Append sequences from this product to the overall lists)
            X_all.extend(product_X)
            y_all.extend(product_y)
            product_ids_all.extend(product_pids)
            dates_all.extend(product_dates)

        except Exception as e:
            # 捕获并打印处理过程中的错误 (Catch and print errors during processing)
            print(f"处理产品 {product} 序列时出错: {type(e).__name__}: {str(e)}") # Error processing product ... sequences...

    # --- 检查是否生成了任何数据 (Check if Any Data Was Generated) ---
    if not X_all:
        raise ValueError("未生成有效的序列数据，请检查数据和参数。") # No valid sequence data generated, check data and parameters.

    # --- 转换为 NumPy 数组 (Convert to NumPy Arrays) ---
    X_all = np.array(X_all)
    y_all = np.array(y_all)

    print(f"准备好的序列数据: X形状={X_all.shape}, y形状={y_all.shape}") # Prepared sequence data: X shape=..., y shape=...

    return X_all, y_all, product_ids_all, dates_all, scalers


def train_and_evaluate_model(X, y, config):
    """
    使用 PyTorch 训练和评估生命周期预测模型。
    (Trains and evaluates the lifecycle prediction model using PyTorch.)

    Args:
        X (np.ndarray): 输入序列。 (Input sequences.)
        y (np.ndarray): 目标标签。 (Target labels.)
        config (dict): 包含训练参数的配置字典。 (Configuration dictionary containing training parameters.)

    Returns:
        tuple: 包含以下内容的元组 (A tuple containing):
            - torch.nn.Module: 训练好的 (最佳) 模型。 (The trained (best) model.)
            - dict: 包含训练历史的字典。 (Dictionary containing the training history.)
            - tuple: 包含测试数据的 (y_true_test, y_pred_test) 的元组。 (Tuple containing (y_true_test, y_pred_test) for the test data.)
            - str: 测试数据的格式化分类报告。 (Formatted classification report for the test data.)
            - np.ndarray: 测试数据的混淆矩阵。 (Confusion matrix for the test data.)
            - dict: 作为字典的测试数据分类报告。 (Test data classification report as a dictionary.)
    """
    # --- 从配置中提取参数 (Extract Parameters from Config) ---
    output_dir = config['evaluation']['output_dir']
    model_type = config['training']['model_type']
    test_size = config['training']['test_size']
    random_state = config['training']['random_state']
    batch_size = config['training']['batch_size']
    epochs = config['training']['epochs']
    lr = config['training']['learning_rate']
    patience = config['training']['patience'] # Early stopping 的耐心值 (Patience for early stopping)

    # 创建输出目录 (如果不存在) (Create output directory (if it doesn't exist))
    os.makedirs(output_dir, exist_ok=True)

    # --- 划分训练集和测试集 (Split into Training and Test Sets) ---
    # 使用 stratify=y 保持训练集和测试集中的类别比例一致
    # (Use stratify=y to maintain class proportions in train and test sets)
    try:
         X_train, X_test, y_train, y_test = train_test_split(
             X, y, test_size=test_size, random_state=random_state, stratify=y
        )
    except ValueError: # 如果类别太少无法分层，则执行普通划分 (If too few samples per class to stratify, perform normal split)
         print("警告: Stratify 失败 (可能每个类的样本不足)。执行普通划分。") # WARNING: Stratify failed (possibly not enough samples per class). Performing normal split.
         X_train, X_test, y_train, y_test = train_test_split(
             X, y, test_size=test_size, random_state=random_state
         )

    # --- 创建 PyTorch Datasets 和 DataLoaders (Create PyTorch Datasets and DataLoaders) ---
    train_dataset = ProductLifecycleDataset(X_train, y_train)
    test_dataset = ProductLifecycleDataset(X_test, y_test)

    # DataLoader 用于批量加载数据 (DataLoader for batching data)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) # 训练时打乱数据 (Shuffle data during training)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) # 测试时不需要打乱 (No need to shuffle for testing)

    print(f"训练集: X形状={X_train.shape}, y形状={y_train.shape}") # Training set: X shape=..., y shape=...
    print(f"测试集: X形状={X_test.shape}, y形状={y_test.shape}") # Test set: X shape=..., y shape=...
    # 打印类别分布以检查不平衡性 (Print class distribution to check for imbalance)
    print(f"训练集类别分布: {np.bincount(y_train)}") # Training set class distribution...
    print(f"测试集类别分布: {np.bincount(y_test)}") # Test set class distribution...

    # --- 获取模型输入维度 (Get Model Input Dimensions) ---
    seq_length, input_dim = X_train.shape[1], X_train.shape[2]
    num_classes = len(np.unique(y)) # 从数据中确定类别数量 (Determine number of classes from data)
    print(f"模型参数: input_dim={input_dim}, seq_length={seq_length}, num_classes={num_classes}") # Model parameters...

    # --- 创建模型实例 (Create Model Instance) ---
    if model_type == 'lstm':
        model = LSTMModel(input_dim, seq_length, num_classes=num_classes)
        print("构建 LSTM 模型...") # Building LSTM model...
    else: # 默认使用混合模型 (Default to hybrid model)
        model = HybridModel(input_dim, seq_length, num_classes=num_classes)
        print("构建混合 CNN+LSTM 模型...") # Building hybrid CNN+LSTM model...

    # --- 设置设备 (GPU 或 CPU) (Set Device (GPU or CPU)) ---
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}") # Using device...
    model.to(device) # 将模型移动到指定设备 (Move model to the specified device)

    # --- 定义损失函数和优化器 (Define Loss Function and Optimizer) ---
    # 如果数据集不平衡，可以添加类别权重 (Add class weights if dataset is imbalanced)
    class_counts = np.bincount(y_train)
    if len(class_counts) == num_classes and np.min(class_counts) > 0: # 确保所有类都存在 (Ensure all classes are present)
         # 计算权重：类别样本数越少，权重越高 (Calculate weights: fewer samples per class means higher weight)
         class_weights = 1. / torch.tensor(class_counts, dtype=torch.float32)
         # 归一化权重 (Normalize weights)
         class_weights = class_weights / class_weights.sum() * num_classes
         criterion = nn.CrossEntropyLoss(weight=class_weights.to(device)) # 使用带权重的损失函数 (Use weighted loss function)
         print(f"使用加权交叉熵损失，权重: {class_weights.cpu().numpy()}") # Using weighted CrossEntropyLoss with weights...
    else:
         print("警告: 无法计算类别权重，使用未加权的交叉熵损失。") # WARNING: Could not calculate class weights, using unweighted CrossEntropyLoss.
         criterion = nn.CrossEntropyLoss() # 未加权损失 (Unweighted loss)

    optimizer = optim.Adam(model.parameters(), lr=lr) # Adam 优化器 (Adam optimizer)
    # 可选：添加学习率调度器 (Optional: Add learning rate scheduler)
    # 当验证损失停止改善时降低学习率 (Reduce learning rate when validation loss stops improving)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=patience // 2, verbose=True)

    # --- 训练循环 (Training Loop) ---
    history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []} # 记录训练历史 (Record training history)
    print("开始模型训练...") # Starting model training...
    best_val_loss = float('inf') # 初始化最佳验证损失 (Initialize best validation loss)
    early_stop_counter = 0 # Early stopping 计数器 (Early stopping counter)
    best_model_path = os.path.join(output_dir, "best_model.pt") # 最佳模型保存路径 (Path to save the best model)

    for epoch in range(epochs):
        # --- 训练阶段 (Training Phase) ---
        model.train() # 设置模型为训练模式 (Set model to training mode)
        train_loss_sum = 0.0 # 累积训练损失 (Accumulated training loss)
        correct_train = 0    # 训练集正确预测数 (Correct predictions on training set)
        total_train = 0      # 训练集总样本数 (Total samples in training set)

        for inputs, targets in train_loader:
            # 将数据移动到设备 (Move data to device)
            inputs, targets = inputs.to(device), targets.to(device)

            # 前向传播 (Forward pass)
            optimizer.zero_grad() # 清除梯度 (Clear gradients)
            outputs = model(inputs) # 获取模型输出 (Get model output)
            loss = criterion(outputs, targets) # 计算损失 (Calculate loss)

            # 反向传播和优化 (Backward pass and optimization)
            loss.backward() # 计算梯度 (Compute gradients)
            # 可选：梯度裁剪，防止梯度爆炸 (Optional: Gradient clipping to prevent exploding gradients)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step() # 更新模型参数 (Update model parameters)

            # 统计损失和准确率 (Accumulate loss and accuracy stats)
            # loss.item() 是当前批次的平均损失，乘以批次大小得到总损失
            # (loss.item() is the average loss for the current batch, multiply by batch size for total loss)
            train_loss_sum += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1) # 获取预测类别 (Get predicted class)
            total_train += targets.size(0) # 累加样本数 (Accumulate sample count)
            correct_train += predicted.eq(targets).sum().item() # 累加正确预测数 (Accumulate correct predictions)

        # 计算平均训练损失和准确率 (Calculate average training loss and accuracy)
        avg_train_loss = train_loss_sum / total_train
        avg_train_accuracy = correct_train / total_train

        # --- 验证阶段 (Validation Phase) ---
        model.eval() # 设置模型为评估模式 (Set model to evaluation mode)
        val_loss_sum = 0.0   # 累积验证损失 (Accumulated validation loss)
        correct_val = 0      # 验证集正确预测数 (Correct predictions on validation set)
        total_val = 0        # 验证集总样本数 (Total samples in validation set)

        with torch.no_grad(): # 禁用梯度计算 (Disable gradient calculations)
            for inputs, targets in test_loader: # 使用 test_loader 进行验证 (Use test_loader for validation)
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets) # 计算验证损失 (Calculate validation loss)

                val_loss_sum += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total_val += targets.size(0)
                correct_val += predicted.eq(targets).sum().item()

        # 计算平均验证损失和准确率 (Calculate average validation loss and accuracy)
        avg_val_loss = val_loss_sum / total_val
        avg_val_accuracy = correct_val / total_val

        # 更新训练历史 (Update training history)
        history['loss'].append(avg_train_loss)
        history['accuracy'].append(avg_train_accuracy)
        history['val_loss'].append(avg_val_loss)
        history['val_accuracy'].append(avg_val_accuracy)

        # 打印轮次信息 (Print epoch information)
        print(f'轮次 {epoch+1}/{epochs}, 训练损失: {avg_train_loss:.4f}, 训练准确率: {avg_train_accuracy:.4f}, '
              f'验证损失: {avg_val_loss:.4f}, 验证准确率: {avg_val_accuracy:.4f}')
        # Epoch ..., Train Loss: ..., Train Acc: ..., Val Loss: ..., Val Acc: ...

        # --- 学习率调度和 Early Stopping (Learning Rate Scheduling and Early Stopping) ---
        scheduler.step(avg_val_loss) # 根据验证损失调整学习率 (Adjust learning rate based on validation loss)

        # 检查是否需要 Early Stopping (Check if early stopping is needed)
        if avg_val_loss < best_val_loss:
            # 如果验证损失改善，保存模型并重置计数器
            # (If validation loss improved, save the model and reset counter)
            best_val_loss = avg_val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), best_model_path) # 保存模型状态 (Save model state)
            print(f"  -> 验证损失改善，最佳模型已保存至 {best_model_path}") # -> Validation loss improved, best model saved to ...
        else:
            # 如果验证损失未改善，增加计数器
            # (If validation loss did not improve, increment counter)
            early_stop_counter += 1
            if early_stop_counter >= patience:
                # 如果连续 patience 轮未改善，则停止训练
                # (If no improvement for `patience` consecutive epochs, stop training)
                print(f"Early stopping 触发于轮次 {epoch+1} (验证损失 {patience} 轮未改善)") # Early stopping triggered at epoch ... (validation loss did not improve for ... epochs)
                break # 退出训练循环 (Exit training loop)

    # --- 加载最佳模型进行最终评估 (Load Best Model for Final Evaluation) ---
    if os.path.exists(best_model_path):
         print(f"加载最佳模型从: {best_model_path}") # Loading best model from...
         model.load_state_dict(torch.load(best_model_path))
    else:
         print("警告: 未找到最佳模型文件。将使用最后一轮的模型进行评估。") # WARNING: Best model file not found. Using model from the last epoch for evaluation.

    # --- 在测试集上评估最佳模型 (Evaluate Best Model on Test Set) ---
    print("在测试集上评估最佳模型...") # Evaluating best model on test set...
    model.eval() # 设置为评估模式 (Set to evaluation mode)
    y_pred_test = [] # 存储测试集预测结果 (List to store test set predictions)
    y_true_test = [] # 存储测试集真实标签 (List to store test set true labels)

    with torch.no_grad(): # 禁用梯度 (Disable gradients)
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1) # 获取预测 (Get predictions)

            y_pred_test.extend(predicted.cpu().numpy()) # 移动到 CPU 并添加到列表 (Move to CPU and add to list)
            y_true_test.extend(targets.cpu().numpy()) # targets 通常已在 CPU 上 (targets are usually already on CPU)

    # --- 生成并打印/保存评估结果 (Generate and Print/Save Evaluation Results) ---
    target_names = ['导入期', '成长期', '成熟期', '衰退期'] # 确保与标签对应 (Ensure correspondence with labels)
    # 获取分类报告和混淆矩阵 (Get classification report and confusion matrix)
    report_str, cm, report_dict = get_classification_report(y_true_test, y_pred_test, target_names)

    print("\n测试集分类报告:") # Test Set Classification Report:
    print(report_str)

    # 绘制并保存混淆矩阵和训练历史图 (Plot and save confusion matrix and training history plots)
    plot_confusion_matrix(cm, target_names, output_dir)
    plot_training_history(history, output_dir)

    # 返回训练结果 (Return training results)
    return model, history, (y_true_test, y_pred_test), report_str, cm, report_dict
```

## 15. `product-lifecycle-prediction/src/utils/__init__.py`

```python
# 此文件可以是空的，用于将 utils 标记为 Python 子包
# (This file can be empty, used to mark utils as a Python sub-package)
```

## 16. `product-lifecycle-prediction/src/utils/config.py`

```python
import yaml
import os

# 默认配置字典，如果 config.yaml 文件不存在或加载失败时使用
# (Default configuration dictionary, used if config.yaml file doesn't exist or fails to load)
DEFAULT_CONFIG = {
    'data': {
        'raw_path': 'data/raw/online_retail_10_11.csv', # 默认原始数据路径 (Default raw data path)
        'processed_path': 'data/processed',           # 处理后数据的保存路径 (Path to save processed data)
        'min_days_threshold': 30                      # 产品有效性阈值 (Product validity threshold)
    },
    'features': {
        'windows': [7, 14, 30],                       # 滚动窗口大小 (Rolling window sizes)
        'sequence_length': 30                         # 模型输入序列长度 (Model input sequence length)
    },
    'training': {
        'model_type': 'hybrid',                       # 模型类型 ('hybrid' 或 'lstm') (Model type ('hybrid' or 'lstm'))
        'epochs': 30,                                 # 训练轮数 (Number of training epochs)
        'batch_size': 32,                             # 批量大小 (Batch size)
        'learning_rate': 0.001,                       # 学习率 (Learning rate)
        'test_size': 0.2,                             # 测试集比例 (Test set proportion)
        'random_state': 42,                           # 随机种子，用于可复现性 (Random seed for reproducibility)
        'patience': 10                                # Early stopping 的耐心值 (Patience for early stopping)
    },
    'evaluation': {
        'n_visualization_samples': 5,                 # 可视化样本数量 (Number of visualization samples)
        'output_dir': 'results'                       # 输出目录 (Output directory)
    },
    'excel': {
        'output_filename': 'product_lifecycle_results.xlsx' # Excel 输出文件名 (Excel output filename)
    }
}

def load_config(config_path='config.yaml'):
    """
    从 YAML 文件加载配置。
    (Loads configuration from a YAML file.)

    Args:
        config_path (str): YAML 配置文件的路径。 (Path to the YAML configuration file.)

    Returns:
        dict: 加载的配置字典。如果加载失败，则返回默认配置。
              (The loaded configuration dictionary. Returns default config if loading fails.)
    """
    # 检查配置文件是否存在 (Check if the configuration file exists)
    if not os.path.exists(config_path):
        print(f"警告: 配置文件 '{config_path}' 未找到。将使用默认配置。") # WARNING: Config file '...' not found. Using default config.
        # 可选：如果配置文件不存在，可以创建一个默认的
        # (Optional: Create a default config file if it doesn't exist)
        # try:
        #     with open(config_path, 'w') as f:
        #         yaml.dump(DEFAULT_CONFIG, f, default_flow_style=False, sort_keys=False)
        #     print(f"默认配置文件已创建于: {config_path}") # Default config file created at...
        # except IOError as e:
        #     print(f"错误: 无法创建默认配置文件: {e}") # ERROR: Could not create default config file...
        return DEFAULT_CONFIG # 返回默认配置 (Return default config)

    try:
        # 尝试打开并加载 YAML 文件 (Try to open and load the YAML file)
        with open(config_path, 'r', encoding='utf-8') as f: # 指定编码 (Specify encoding)
            config = yaml.safe_load(f)
        # 可选：将加载的配置与默认配置合并，以处理缺失的键
        # (Optional: Merge loaded config with defaults to handle missing keys)
        # merged_config = {**DEFAULT_CONFIG, **config} # 简单的浅合并 (Simple shallow merge)
        # 注意：对于嵌套字典，可能需要深度合并 (Note: Deep merging might be needed for nested dictionaries)
        print(f"配置已成功从 '{config_path}' 加载。") # Configuration successfully loaded from '...'.
        # 返回加载的配置，如果加载成功但文件为空，则返回默认配置
        # (Return loaded config, or default if loading succeeded but file was empty)
        return config if config else DEFAULT_CONFIG
    except yaml.YAMLError as e:
        # 处理 YAML 解析错误 (Handle YAML parsing errors)
        print(f"错误: 解析配置文件 '{config_path}' 失败: {e}。将使用默认配置。") # ERROR: Failed to parse config file '...': ... Using default config.
        return DEFAULT_CONFIG
    except IOError as e:
        # 处理文件读取错误 (Handle file reading errors)
        print(f"错误: 读取配置文件 '{config_path}' 失败: {e}。将使用默认配置。") # ERROR: Failed to read config file '...': ... Using default config.
        return DEFAULT_CONFIG

# 可以在此处加载配置使其全局可用，但通常更好的做法是在 main.py 中加载并按需传递。
# (Config can be loaded here to make it globally available, but it's often better practice
#  to load it in main.py and pass it as needed.)
# config = load_config()
```

## 17. `product-lifecycle-prediction/src/utils/excel_export.py`

```python
import os
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, PatternFill
# 确保从 sklearn.metrics 导入所需内容 (Ensure required imports from sklearn.metrics)
from sklearn.metrics import classification_report, confusion_matrix
from datetime import datetime
import numpy as np
import random

def export_results_to_excel(results, file_path="product_lifecycle_results.xlsx"):
    """
    将模型结果导出到 Excel 文件，提供更详细的分析信息。
    (Exports model results to an Excel file, providing more detailed analysis information.)

    Args:
        results (dict): 从主函数返回的结果字典。 (Dictionary of results returned from the main function.)
        file_path (str): Excel 文件的保存路径。 (Save path for the Excel file.)
    """
    print(f"开始导出详细结果到 Excel: {file_path}") # Starting export of detailed results to Excel...

    # 创建工作簿 (Create Workbook)
    wb = Workbook()
    # 删除默认工作表 (Remove default sheet)
    if "Sheet" in wb.sheetnames:
        wb.remove(wb["Sheet"])

    # --- 从 results 字典中获取数据以便于访问 (Get data from results dictionary for easier access) ---
    product_features_df = results.get('product_features')
    valid_products = results.get('valid_products', [])
    quality_metrics = results.get('quality_metrics', {})
    model_history = results.get('history', {})
    test_results = results.get('test_results') # (y_true, y_pred)
    report_dict = results.get('report_dict', {}) # 分类报告字典 (Classification report dictionary)
    cm = results.get('confusion_matrix') # 混淆矩阵 (Confusion matrix)
    config = results.get('config', {}) # 配置字典 (Configuration dictionary)
    sample_products_vis = results.get('sample_products_vis', []) # 用于可视化的样本 (Samples used for visualization)
    # 您可能还需要传递其他信息到 results 字典中，例如特征列表、模型参数等
    # (You might need to pass other info into the results dict, like feature lists, model params etc.)


    # --- 辅助函数和常量 (Helper functions and constants) ---
    # 阶段名称和颜色映射 (Phase name and color mappings)
    phase_names_map = {0: "导入期", 1: "成长期", 2: "成熟期", 3: "衰退期"} # Intro, Growth, Maturity, Decline
    phase_colors_hex = {
        0: "D9F0D3", # 浅绿 (Light Green)
        1: "AED4F8", # 浅蓝 (Light Blue)
        2: "FFE699", # 浅橙/黄 (Light Orange/Yellow)
        3: "F8CECC"  # 浅红 (Light Red)
    }
    # (其他颜色映射...)
    importance_colors_hex = { # 特征重要性颜色 (Feature importance colors)
        'Sehr Hoch': 'FF8585', # Very High
        'Hoch': 'FFCC99',      # High
        'Mittel': 'FFFFCC',     # Medium
        'Niedrig': 'E6F2E6'     # Low
    }
    accuracy_colors_hex = { # 准确性颜色 (Accuracy colors)
        "Genau": "C6EFCE",     # Accurate (Green)
        "Ungenau": "FFC7CE",    # Inaccurate (Red)
        "Hoch": "C6EFCE",       # High (Green)
        "Mittel": "FFEB9C",      # Medium (Yellow)
        "Niedrig": "FFC7CE"      # Low (Red)
    }
    overfitting_colors_hex = { # 过拟合颜色 (Overfitting colors)
        "Nein": "FFFFFF",       # No (White/None)
        "Leicht": "FFEB9C",     # Slight (Yellow)
        "Deutlich": "FFC7CE"    # Obvious (Red)
    }

    def apply_fill(cell, color_hex):
        """将填充颜色应用于单元格。 (Applies a fill color to a cell.)"""
        if color_hex and color_hex != "FFFFFF": # 仅当颜色有效时应用 (Only apply if color is valid)
            cell.fill = PatternFill(start_color=color_hex, end_color=color_hex, fill_type="solid")

    # --- 1. 创建概览工作表 (Create Overview Sheet) ---
    overview_sheet = wb.create_sheet("概览") # Overview
    # 设置标题 (Set Title)
    overview_sheet['A1'] = "产品生命周期分析 - 结果概览" # Product Lifecycle Analysis - Results Overview
    overview_sheet['A1'].font = Font(bold=True, size=16)
    overview_sheet.merge_cells('A1:F1')
    overview_sheet['A1'].alignment = Alignment(horizontal='center')

    # 添加项目元信息 (Add Project Meta Information)
    overview_sheet['A3'] = "项目信息" # Project Information
    overview_sheet['A3'].font = Font(bold=True)
    overview_sheet['A4'] = "项目名称:" # Project Name:
    overview_sheet['B4'] = "产品生命周期智能分析系统" # Product Lifecycle Intelligent Analysis System
    overview_sheet['A5'] = "版本:" # Version:
    overview_sheet['B5'] = "1.0.0" # (示例版本) (Example Version)
    overview_sheet['A6'] = "生成日期:" # Generation Date:
    overview_sheet['B6'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # 添加基本数据统计 (Add Basic Data Statistics)
    overview_sheet['A8'] = "数据统计" # Data Statistics
    overview_sheet['A8'].font = Font(bold=True)
    # overview_sheet['A9'] = "原始产品总数:" # Total Original Products:
    # overview_sheet['B9'] = ... # 需要从加载步骤获取 (Need to get from loading step)
    overview_sheet['A10'] = "有效产品数 (>= 阈值天数):" # Valid Products (>= threshold days):
    overview_sheet['B10'] = len(valid_products)
    overview_sheet['A11'] = "序列样本总数:" # Total Sequence Samples:
    overview_sheet['B11'] = len(results.get('X', []))
    if results.get('X') is not None and len(results['X']) > 0:
        overview_sheet['A12'] = "特征维度数量:" # Number of Feature Dimensions:
        overview_sheet['B12'] = results['X'].shape[2]
        overview_sheet['A13'] = "序列长度:" # Sequence Length:
        overview_sheet['B13'] = results['X'].shape[1]
    overview_sheet['A14'] = "总记录数 (处理后):" # Total Records (Processed):
    overview_sheet['B14'] = len(product_features_df) if product_features_df is not None else 0

    # 生命周期阶段分布 (Lifecycle Phase Distribution)
    overview_sheet['A16'] = "生命周期阶段分布 (基于标签)" # Lifecycle Phase Distribution (Based on Labels)
    overview_sheet['A16'].font = Font(bold=True)
    overview_sheet['A17'] = "阶段" # Phase
    overview_sheet['B17'] = "天数" # Days
    overview_sheet['C17'] = "百分比" # Percentage
    current_row = 18
    if product_features_df is not None and 'LifecyclePhase' in product_features_df.columns:
        phase_counts = product_features_df['LifecyclePhase'].value_counts().sort_index()
        total_days = len(product_features_df)
        for phase_id, count in phase_counts.items():
            phase_name = phase_names_map.get(phase_id, f"未知 ({phase_id})") # Unknown
            overview_sheet[f'A{current_row}'] = phase_name
            overview_sheet[f'B{current_row}'] = count
            overview_sheet[f'C{current_row}'] = f"{count / total_days * 100:.2f}%" if total_days > 0 else "0.00%"
            apply_fill(overview_sheet[f'A{current_row}'], phase_colors_hex.get(phase_id))
            current_row += 1
    overview_sheet.row_dimensions[current_row].height = 5 # Add small gap

    # 标签质量指标 (Label Quality Metrics)
    current_row += 1
    overview_sheet[f'A{current_row}'] = "标签质量指标" # Label Quality Metrics
    overview_sheet[f'A{current_row}'].font = Font(bold=True)
    current_row += 1
    overview_sheet[f'A{current_row}'] = "无效转换率:" # Invalid Transition Ratio:
    overview_sheet[f'B{current_row}'] = f"{quality_metrics.get('invalid_transition_ratio', 0):.4f}"
    current_row += 1
    overview_sheet[f'A{current_row}'] = "标签稳定性 (平均):" # Label Stability (Average):
    overview_sheet[f'B{current_row}'] = f"{quality_metrics.get('stability', 0):.4f}"
    current_row += 1
    avg_durations = quality_metrics.get('avg_phase_durations', {})
    for phase_id, duration in avg_durations.items():
         phase_name = phase_names_map.get(phase_id, f"未知 ({phase_id})")
         overview_sheet[f'A{current_row}'] = f"'{phase_name}' 平均持续时间:" # Average Duration:
         overview_sheet[f'B{current_row}'] = f"{duration:.2f} 天" # days
         current_row += 1
    overview_sheet.row_dimensions[current_row].height = 5 # Add small gap

    # 模型评估摘要 (Model Evaluation Summary)
    current_row += 1
    overview_sheet[f'A{current_row}'] = "模型评估摘要 (测试集)" # Model Evaluation Summary (Test Set)
    overview_sheet[f'A{current_row}'].font = Font(bold=True)
    current_row += 1
    if report_dict:
        overview_sheet[f'A{current_row}'] = "总体准确率:" # Overall Accuracy:
        overview_sheet[f'B{current_row}'] = f"{report_dict.get('accuracy', 0):.4f}"
        current_row += 1
        overview_sheet[f'A{current_row}'] = "宏平均 F1 分数:" # Macro Avg F1-Score:
        overview_sheet[f'B{current_row}'] = f"{report_dict.get('macro avg', {}).get('f1-score', 0):.4f}"
        current_row += 1
        overview_sheet[f'A{current_row}'] = "加权平均 F1 分数:" # Weighted Avg F1-Score:
        overview_sheet[f'B{current_row}'] = f"{report_dict.get('weighted avg', {}).get('f1-score', 0):.4f}"

    # 设置列宽 (Set Column Widths)
    for col in ['A', 'B', 'C']: # Adjust as needed
        overview_sheet.column_dimensions[col].width = 25

    # --- 2. 创建模型评估工作表 (Create Model Evaluation Sheet) ---
    eval_sheet = wb.create_sheet("模型评估") # Model Evaluation
    # 设置标题 (Set Title)
    eval_sheet['A1'] = "生命周期预测模型 - 评估结果 (测试集)" # Lifecycle Prediction Model - Evaluation Results (Test Set)
    eval_sheet['A1'].font = Font(bold=True, size=16)
    eval_sheet.merge_cells('A1:F1')
    eval_sheet['A1'].alignment = Alignment(horizontal='center')

    # 分类报告详情 (Classification Report Details)
    eval_sheet['A3'] = "分类报告" # Classification Report
    eval_sheet['A3'].font = Font(bold=True)
    eval_sheet['A4'] = "类别" # Class
    eval_sheet['B4'] = "精确率" # Precision
    eval_sheet['C4'] = "召回率" # Recall
    eval_sheet['D4'] = "F1分数" # F1-Score
    eval_sheet['E4'] = "支持度" # Support
    current_row = 5
    if report_dict:
         # 按类别指标 (Per-class metrics)
         for phase_id, phase_name in phase_names_map.items():
             metrics = report_dict.get(phase_name, {})
             eval_sheet[f'A{current_row}'] = phase_name
             eval_sheet[f'B{current_row}'] = f"{metrics.get('precision', 0):.4f}"
             eval_sheet[f'C{current_row}'] = f"{metrics.get('recall', 0):.4f}"
             eval_sheet[f'D{current_row}'] = f"{metrics.get('f1-score', 0):.4f}"
             eval_sheet[f'E{current_row}'] = metrics.get('support', 0)
             apply_fill(eval_sheet[f'A{current_row}'], phase_colors_hex.get(phase_id))
             current_row += 1
         eval_sheet.row_dimensions[current_row].height = 5 # Add small gap
         current_row += 1

         # 总体指标 (Overall metrics)
         eval_sheet[f'A{current_row}'] = "准确率 (总体)" # Accuracy (Overall)
         eval_sheet[f'B{current_row}'] = f"{report_dict.get('accuracy', 0):.4f}"
         eval_sheet.merge_cells(f'B{current_row}:E{current_row}')
         current_row += 1
         # 宏平均 (Macro Average)
         macro_avg = report_dict.get('macro avg', {})
         eval_sheet[f'A{current_row}'] = "宏平均" # Macro Avg
         eval_sheet[f'B{current_row}'] = f"{macro_avg.get('precision', 0):.4f}"
         eval_sheet[f'C{current_row}'] = f"{macro_avg.get('recall', 0):.4f}"
         eval_sheet[f'D{current_row}'] = f"{macro_avg.get('f1-score', 0):.4f}"
         eval_sheet[f'E{current_row}'] = macro_avg.get('support', 0)
         current_row += 1
         # 加权平均 (Weighted Average)
         weighted_avg = report_dict.get('weighted avg', {})
         eval_sheet[f'A{current_row}'] = "加权平均" # Weighted Avg
         eval_sheet[f'B{current_row}'] = f"{weighted_avg.get('precision', 0):.4f}"
         eval_sheet[f'C{current_row}'] = f"{weighted_avg.get('recall', 0):.4f}"
         eval_sheet[f'D{current_row}'] = f"{weighted_avg.get('f1-score', 0):.4f}"
         eval_sheet[f'E{current_row}'] = weighted_avg.get('support', 0)
         current_row += 1
    eval_sheet.row_dimensions[current_row].height = 5 # Add small gap

    # 混淆矩阵 (Confusion Matrix)
    current_row += 1
    eval_sheet[f'A{current_row}'] = "混淆矩阵" # Confusion Matrix
    eval_sheet[f'A{current_row}'].font = Font(bold=True)
    current_row += 1
    eval_sheet[f'A{current_row}'] = "(真实 \\ 预测)" # (True \ Predicted)
    col_offset = 1 # Start from column B
    # 写列标题 (Write column headers)
    for j, phase_name in phase_names_map.items():
        eval_sheet.cell(row=current_row, column=col_offset + j + 1).value = f"预测:\n{phase_name}" # Pred: ...
        eval_sheet.cell(row=current_row, column=col_offset + j + 1).alignment = Alignment(wrap_text=True, horizontal='center')
    eval_sheet.row_dimensions[current_row].height = 30

    # 填充矩阵数据 (Fill matrix data)
    if cm is not None:
        for i, phase_name_true in phase_names_map.items():
            eval_sheet.cell(row=current_row + i + 1, column=col_offset).value = f"真实:\n{phase_name_true}" # True: ...
            eval_sheet.cell(row=current_row + i + 1, column=col_offset).alignment = Alignment(wrap_text=True, vertical='center')
            eval_sheet.row_dimensions[current_row + i + 1].height = 30
            for j, phase_name_pred in phase_names_map.items():
                cell = eval_sheet.cell(row=current_row + i + 1, column=col_offset + j + 1)
                cell.value = cm[i, j]
                cell.alignment = Alignment(horizontal='center', vertical='center')
                if i == j: # 高亮对角线 (Highlight diagonal)
                    apply_fill(cell, accuracy_colors_hex.get("Genau"))
                # 可以选择性地高亮显示严重的错误分类 (Optionally highlight severe misclassifications)
                # elif (i==0 and j==3) or (i==3 and j==1): # e.g., Intro -> Decline or Decline -> Growth
                #     apply_fill(cell, accuracy_colors_hex.get("Ungenau"))

    # 设置列宽 (Set Column Widths)
    for col_letter in ['A', 'B', 'C', 'D', 'E']:
         eval_sheet.column_dimensions[col_letter].width = 15
    eval_sheet.column_dimensions['A'].width = 20


    # --- 3. 创建特征信息工作表 (Create Feature Information Sheet) ---
    features_sheet = wb.create_sheet("特征信息") # Feature Information
    # 设置标题 (Set Title)
    features_sheet['A1'] = "模型特征信息" # Model Feature Information
    features_sheet['A1'].font = Font(bold=True, size=16)
    features_sheet.merge_cells('A1:E1')
    features_sheet['A1'].alignment = Alignment(horizontal='center')

    # 列出使用的特征 (List features used)
    features_sheet['A3'] = "序号" # No.
    features_sheet['B3'] = "特征名称" # Feature Name
    features_sheet['C3'] = "描述" # Description
    features_sheet['D3'] = "类型" # Type
    features_sheet['E3'] = "重要性 (示例)" # Importance (Example)

    # 特征列表应与 prepare_sequences 中使用的列表一致
    # (Feature list should match the one used in prepare_sequences)
    if results.get('X') is not None and len(results['X']) > 0:
         # (假设特征名称可以从某处获取，或在此重新定义)
         # (Assume feature names can be retrieved from somewhere, or redefine here)
         feature_cols = [ # 确保此列表准确 (Ensure this list is accurate)
             'Sales_7d', 'Sales_14d', 'Sales_30d',
             'Revenue_7d', 'Revenue_14d', 'Revenue_30d',
             'Transactions_7d', 'Transactions_14d', 'Transactions_30d',
             'AvgSales_7d', 'AvgSales_14d', 'AvgSales_30d',
             'AvgRevenue_7d', 'AvgRevenue_14d', 'AvgRevenue_30d',
             'SalesGrowth_7d', 'SalesGrowth_14d', 'SalesGrowth_30d',
             'RevenueGrowth_7d', 'RevenueGrowth_14d', 'RevenueGrowth_30d',
             'AvgUnitPrice_30d',
             'SalesAcceleration', 'RelativeSales', 'MA_SalesGrowth', 'ProductAge',
             'SalesVolatility_14d'
         ]
         # 再次检查 DataFrame 中是否存在这些列 (Double check if these columns exist in the DataFrame)
         if product_features_df is not None:
              feature_cols = [col for col in feature_cols if col in product_features_df.columns]
         else:
              feature_cols = []
    else:
         feature_cols = []

    # 特征描述 (Feature descriptions - update as needed)
    feature_descriptions = {
        'Sales_7d': '7天滚动销售总量', 'Sales_14d': '14天滚动销售总量', 'Sales_30d': '30天滚动销售总量',
        'Revenue_7d': '7天滚动总收入', 'Revenue_14d': '14天滚动总收入', 'Revenue_30d': '30天滚动总收入',
        'Transactions_7d': '7天滚动交易次数', 'Transactions_14d': '14天滚动交易次数', 'Transactions_30d': '30天滚动交易次数',
        'AvgSales_7d': '7天滚动日均销量', 'AvgSales_14d': '14天滚动日均销量', 'AvgSales_30d': '30天滚动日均销量',
        'AvgRevenue_7d': '7天滚动日均收入', 'AvgRevenue_14d': '14天滚动日均收入', 'AvgRevenue_30d': '30天滚动日均收入',
        'SalesGrowth_7d': '7天销量增长率(与前7天比)', 'SalesGrowth_14d': '14天销量增长率(与前14天比)', 'SalesGrowth_30d': '30天销量增长率(与前30天比)',
        'RevenueGrowth_7d': '7天收入增长率(与前7天比)', 'RevenueGrowth_14d': '14天收入增长率(与前14天比)', 'RevenueGrowth_30d': '30天收入增长率(与前30天比)',
        'AvgUnitPrice_30d': '30天滚动平均单价',
        'SalesAcceleration': '销售加速度(7天增长率的变化)',
        'RelativeSales': '相对销售额(30天销量/历史峰值30天销量)',
        'MA_SalesGrowth': '销售增长率的移动平均(例如14天增长率的7日MA)',
        'ProductAge': '产品年龄(首次销售至今的天数)',
        'SalesVolatility_14d': '销售波动性(例如14天日销量的标准差)'
    }
    # 示例重要性 - 应替换为实际分析结果
    # (Example importance - should be replaced with actual analysis results)
    feature_importance = {col: random.choice(['Hoch', 'Mittel', 'Niedrig']) for col in feature_cols}
    if 'RelativeSales' in feature_importance: feature_importance['RelativeSales'] = 'Sehr Hoch'
    if 'MA_SalesGrowth' in feature_importance: feature_importance['MA_SalesGrowth'] = 'Sehr Hoch'
    if 'SalesAcceleration' in feature_importance: feature_importance['SalesAcceleration'] = 'Hoch'
    if 'Sales_30d' in feature_importance: feature_importance['Sales_30d'] = 'Hoch'

    # 填充表格 (Fill the table)
    current_row = 4
    for i, feature in enumerate(feature_cols):
         features_sheet[f'A{current_row}'] = i + 1
         features_sheet[f'B{current_row}'] = feature
         features_sheet[f'C{current_row}'] = feature_descriptions.get(feature, "N/A")
         features_sheet[f'D{current_row}'] = "数值型" # Numeric (assuming all are numeric)
         importance = feature_importance.get(feature, "Mittel") # Default to Medium
         features_sheet[f'E{current_row}'] = importance.replace('Sehr Hoch', '非常高').replace('Hoch', '高').replace('Mittel', '中').replace('Niedrig', '低') # Translate importance
         apply_fill(features_sheet[f'E{current_row}'], importance_colors_hex.get(importance))
         current_row += 1

    # 设置列宽 (Set Column Widths)
    features_sheet.column_dimensions['A'].width = 8
    features_sheet.column_dimensions['B'].width = 25
    features_sheet.column_dimensions['C'].width = 50
    features_sheet.column_dimensions['D'].width = 12
    features_sheet.column_dimensions['E'].width = 15


    # --- 4. 创建模型架构工作表 (Create Model Architecture Sheet) ---
    model_sheet = wb.create_sheet("模型架构") # Model Architecture
    # 设置标题 (Set Title)
    model_sheet['A1'] = "生命周期预测模型 - 架构详情" # Lifecycle Prediction Model - Architecture Details
    model_sheet['A1'].font = Font(bold=True, size=16)
    model_sheet.merge_cells('A1:F1')
    model_sheet['A1'].alignment = Alignment(horizontal='center')

    # 模型类型和基本参数 (Model type and basic parameters)
    model_type = config.get('training', {}).get('model_type', 'N/A')
    seq_len = config.get('features', {}).get('sequence_length', 'N/A')
    input_dim = results.get('X').shape[2] if results.get('X') is not None and len(results['X']) > 0 else 'N/A'
    num_classes = len(phase_names_map)

    model_sheet['A3'] = "模型类型:" # Model Type:
    model_sheet['B3'] = model_type.upper()
    model_sheet['A4'] = "输入序列长度:" # Input Sequence Length:
    model_sheet['B4'] = seq_len
    model_sheet['A5'] = "输入特征维度:" # Input Feature Dimension:
    model_sheet['B5'] = input_dim
    model_sheet['A6'] = "输出类别数:" # Number of Output Classes:
    model_sheet['B6'] = num_classes

    # 添加层信息 (Add layer information - Example for Hybrid)
    model_sheet['A8'] = "模型层级详情 (示例: Hybrid)" # Model Layer Details (Example: Hybrid)
    model_sheet['A8'].font = Font(bold=True)
    model_sheet['A9'] = "层名称" # Layer Name
    model_sheet['B9'] = "类型" # Type
    model_sheet['C9'] = "主要参数" # Main Parameters
    model_sheet['D9'] = "输出形状 (估计)" # Output Shape (Estimated)
    current_row = 10
    if model_type == 'hybrid' and input_dim != 'N/A' and seq_len != 'N/A':
        # (假设参数 - 应从 builders.py 或 config 获取) (Assume parameters - should get from builders.py or config)
        cnn_filters = 64
        lstm_units = 64
        # Conv1
        model_sheet[f'A{current_row}'] = "Conv1"; model_sheet[f'B{current_row}'] = "Conv1d"; model_sheet[f'C{current_row}'] = f"in={input_dim}, out={cnn_filters}, k=3, p=1"; model_sheet[f'D{current_row}'] = f"B, {cnn_filters}, {seq_len}"; current_row+=1
        model_sheet[f'A{current_row}'] = "BN1"; model_sheet[f'B{current_row}'] = "BatchNorm1d"; model_sheet[f'C{current_row}'] = f"features={cnn_filters}"; model_sheet[f'D{current_row}'] = f"B, {cnn_filters}, {seq_len}"; current_row+=1
        model_sheet[f'A{current_row}'] = "Act1"; model_sheet[f'B{current_row}'] = "LeakyReLU"; model_sheet[f'D{current_row}'] = f"B, {cnn_filters}, {seq_len}"; current_row+=1
        model_sheet[f'A{current_row}'] = "Pool1"; model_sheet[f'B{current_row}'] = "MaxPool1d"; model_sheet[f'C{current_row}'] = "k=2"; model_sheet[f'D{current_row}'] = f"B, {cnn_filters}, {seq_len//2}"; current_row+=1
        # Conv2
        model_sheet[f'A{current_row}'] = "Conv2"; model_sheet[f'B{current_row}'] = "Conv1d"; model_sheet[f'C{current_row}'] = f"in={cnn_filters}, out={cnn_filters*2}, k=3, p=1"; model_sheet[f'D{current_row}'] = f"B, {cnn_filters*2}, {seq_len//2}"; current_row+=1
        model_sheet[f'A{current_row}'] = "BN2"; model_sheet[f'B{current_row}'] = "BatchNorm1d"; model_sheet[f'C{current_row}'] = f"features={cnn_filters*2}"; model_sheet[f'D{current_row}'] = f"B, {cnn_filters*2}, {seq_len//2}"; current_row+=1
        model_sheet[f'A{current_row}'] = "Act2"; model_sheet[f'B{current_row}'] = "LeakyReLU"; model_sheet[f'D{current_row}'] = f"B, {cnn_filters*2}, {seq_len//2}"; current_row+=1
        model_sheet[f'A{current_row}'] = "Pool2"; model_sheet[f'B{current_row}'] = "MaxPool1d"; model_sheet[f'C{current_row}'] = "k=2"; model_sheet[f'D{current_row}'] = f"B, {cnn_filters*2}, {seq_len//4}"; current_row+=1
        # Flatten CNN
        cnn_flat_dim = (cnn_filters*2) * (seq_len//4)
        model_sheet[f'A{current_row}'] = "FlattenCNN"; model_sheet[f'B{current_row}'] = "Reshape"; model_sheet[f'D{current_row}'] = f"B, {cnn_flat_dim}"; current_row+=1
        # LSTM
        lstm_out_dim = lstm_units * 2 # Bidirectional
        model_sheet[f'A{current_row}'] = "LSTM"; model_sheet[f'B{current_row}'] = "LSTM"; model_sheet[f'C{current_row}'] = f"in={input_dim}, hidden={lstm_units}, bi=True"; model_sheet[f'D{current_row}'] = f"B, {lstm_out_dim} (last state)"; current_row+=1
        # Concat
        combined_dim = cnn_flat_dim + lstm_out_dim
        model_sheet[f'A{current_row}'] = "Concat"; model_sheet[f'B{current_row}'] = "Concatenate"; model_sheet[f'D{current_row}'] = f"B, {combined_dim}"; current_row+=1
        # FC1
        fc1_units = 128
        model_sheet[f'A{current_row}'] = "FC1"; model_sheet[f'B{current_row}'] = "Linear"; model_sheet[f'C{current_row}'] = f"in={combined_dim}, out={fc1_units}"; model_sheet[f'D{current_row}'] = f"B, {fc1_units}"; current_row+=1
        model_sheet[f'A{current_row}'] = "Act3"; model_sheet[f'B{current_row}'] = "LeakyReLU"; model_sheet[f'D{current_row}'] = f"B, {fc1_units}"; current_row+=1
        # Dropout
        dp_rate = config.get('training', {}).get('dropout_rate', 0.3)
        model_sheet[f'A{current_row}'] = "Dropout"; model_sheet[f'B{current_row}'] = "Dropout"; model_sheet[f'C{current_row}'] = f"p={dp_rate}"; model_sheet[f'D{current_row}'] = f"B, {fc1_units}"; current_row+=1
        # FC2 (Output)
        model_sheet[f'A{current_row}'] = "FC2"; model_sheet[f'B{current_row}'] = "Linear"; model_sheet[f'C{current_row}'] = f"in={fc1_units}, out={num_classes}"; model_sheet[f'D{current_row}'] = f"B, {num_classes}"; current_row+=1

    # 训练参数 (Training Parameters)
    current_row += 1
    model_sheet[f'A{current_row}'] = "训练参数" # Training Parameters
    model_sheet[f'A{current_row}'].font = Font(bold=True)
    current_row += 1
    train_config = config.get('training', {})
    params_to_show = {'optimizer': 'Adam', # 假设是Adam (Assume Adam)
                      'loss_function': 'CrossEntropyLoss (Weighted)', # 假设加权 (Assume Weighted)
                      'learning_rate': train_config.get('learning_rate'),
                      'epochs': train_config.get('epochs'),
                      'batch_size': train_config.get('batch_size'),
                      'early_stopping_patience': train_config.get('patience')}
    for name, value in params_to_show.items():
        model_sheet[f'A{current_row}'] = f"{name.replace('_', ' ').title()}:"
        model_sheet[f'B{current_row}'] = value
        current_row += 1

    # 设置列宽 (Set Column Widths)
    for col_letter in ['A', 'B', 'C', 'D']:
         model_sheet.column_dimensions[col_letter].width = 25


    # --- 5. 创建流程介绍工作表 (Create Process Description Sheet) ---
    intro_sheet = wb.create_sheet("流程介绍") # Process Description
    # 设置标题 (Set Title)
    intro_sheet['A1'] = "产品生命周期分析 - 详细流程介绍" # Product Lifecycle Analysis - Detailed Process Description
    intro_sheet['A1'].font = Font(bold=True, size=16)
    intro_sheet.merge_cells('A1:D1')
    intro_sheet['A1'].alignment = Alignment(horizontal='center')

    # 添加流程步骤 (Add process steps)
    intro_sheet['A3'] = "步骤" # Step
    intro_sheet['B3'] = "阶段名称" # Phase Name
    intro_sheet['C3'] = "主要内容" # Main Content
    intro_sheet['D3'] = "关键参数/输出" # Key Parameters/Output

    process_steps = [
        (1, "数据加载与预处理", "加载原始CSV数据；清洗数据（处理缺失值、异常值、无效记录）；转换数据类型（日期）", f"原始路径: {config.get('data',{}).get('raw_path')}\n输出: 清洗后的DataFrame"),
        (2, "创建日度数据", "按产品和日期聚合数据，计算日销量、收入、交易次数", "输入: 清洗后的DataFrame\n输出: 日度销售DataFrame"),
        (3, "特征工程", "筛选有效产品；填充缺失日期；计算滚动窗口特征（销量、收入、增长率等）；计算衍生特征（相对销量、加速度、年龄、波动性等）", f"窗口: {config.get('features',{}).get('windows')}\n最小天数: {config.get('data',{}).get('min_days_threshold')}\n输出: 带特征的DataFrame"),
        (4, "生命周期标注", "基于规则（销售水平和增长率）为每个时间点分配生命周期阶段标签（导入、成长、成熟、衰退）", "规则基于: 相对销量, 增长率移动平均\n输出: 带标签的DataFrame ('LifecyclePhase'列)"),
        (5, "标签质量评估", "计算无效转换率、标签稳定性、各阶段平均持续时间", f"输出: 质量指标字典 (稳定性: {quality_metrics.get('stability', 0):.3f})"),
        (6, "序列数据准备", "将时间序列特征转换为固定长度的输入序列；按产品标准化特征", f"序列长度: {seq_len}\n输出: X (序列), y (标签), Scalers"),
        (7, "模型构建", f"构建 {model_type.upper()} 模型 (CNN+LSTM 或 LSTM)", f"输入维度: {input_dim}\n输出类别: {num_classes}"),
        (8, "模型训练与验证", "使用训练集训练模型；在验证集上监控性能；使用Early Stopping和学习率调度", f"优化器: Adam\n损失函数: CrossEntropyLoss\n轮数: {config.get('training',{}).get('epochs')}\nPatience: {config.get('training',{}).get('patience')}"),
        (9, "模型评估", "在测试集上评估最佳模型；计算准确率、精确率、召回率、F1分数；生成混淆矩阵", f"输出: 分类报告, 混淆矩阵"),
        (10, "结果可视化与导出", "绘制样本产品生命周期图、训练历史图、混淆矩阵；将所有结果和分析导出到Excel文件", f"输出目录: {config.get('evaluation',{}).get('output_dir')}\nExcel文件: {config.get('excel',{}).get('output_filename')}")
    ]

    current_row = 4
    for step_data in process_steps:
        for i, value in enumerate(step_data):
            cell = intro_sheet.cell(row=current_row, column=i+1)
            cell.value = value
            cell.alignment = Alignment(wrap_text=True, vertical='top') # 自动换行并顶部对齐 (Wrap text and align top)
        intro_sheet.row_dimensions[current_row].height = 60 # 调整行高 (Adjust row height)
        current_row += 1

    # 设置列宽 (Set Column Widths)
    intro_sheet.column_dimensions['A'].width = 8
    intro_sheet.column_dimensions['B'].width = 20
    intro_sheet.column_dimensions['C'].width = 50
    intro_sheet.column_dimensions['D'].width = 40


    # --- 6. 创建样本产品分析工作表 (Create Sample Product Analysis Sheet) ---
    samples_sheet = wb.create_sheet("样本产品分析") # Sample Product Analysis
    # 设置标题 (Set Title)
    samples_sheet['A1'] = "选定样本产品分析" # Selected Sample Product Analysis
    samples_sheet['A1'].font = Font(bold=True, size=16)
    samples_sheet.merge_cells('A1:H1')
    samples_sheet['A1'].alignment = Alignment(horizontal='center')

    # 选择要分析的样本 (Select samples to analyze)
    num_samples_to_show = 20
    samples_to_analyze = []
    if valid_products:
        # 尝试从每个阶段（基于最终标签）获取一些样本
        # (Try to get some samples from each phase (based on final label))
        if product_features_df is not None and 'LifecyclePhase' in product_features_df.columns:
            # 获取每个产品的最后一条记录 (Get the last record for each product)
            last_records = product_features_df.loc[product_features_df.groupby('StockCode')['InvoiceDate'].idxmax()]
            for phase_id in phase_names_map.keys():
                prods_in_phase = last_records[last_records['LifecyclePhase'] == phase_id]['StockCode'].tolist()
                # 添加一些来自该阶段的样本 (Add some samples from this phase)
                samples_to_analyze.extend(random.sample(prods_in_phase, min(len(prods_in_phase), num_samples_to_show // len(phase_names_map) + 1)))
        # 随机填充剩余的样本 (Randomly fill the remaining samples)
        remaining_needed = num_samples_to_show - len(samples_to_analyze)
        if remaining_needed > 0:
            remaining_prods = [p for p in valid_products if p not in samples_to_analyze]
            samples_to_analyze.extend(random.sample(remaining_prods, min(len(remaining_prods), remaining_needed)))
        # 去重并限制数量 (Make unique and limit count)
        samples_to_analyze = list(set(samples_to_analyze))[:num_samples_to_show]

    # 添加表头 (Add table headers)
    samples_sheet['A3'] = "产品代码" # Product Code
    samples_sheet['B3'] = "描述" # Description
    samples_sheet['C3'] = "销售天数" # Sales Days
    samples_sheet['D3'] = "平均日销量" # Avg Daily Sales
    samples_sheet['E3'] = "峰值日销量" # Peak Daily Sales
    samples_sheet['F3'] = "总收入" # Total Revenue
    samples_sheet['G3'] = "主要阶段(末期)" # Main Phase (End)
    samples_sheet['H3'] = "阶段转换次数" # Phase Transitions

    # 填充样本数据 (Fill sample data)
    current_row = 4
    product_desc_dict = results.get('product_descriptions', {}) # 获取描述字典 (Get description dictionary)
    if product_features_df is not None:
        for product in samples_to_analyze:
            product_data = product_features_df[product_features_df['StockCode'] == product]
            if product_data.empty: continue

            samples_sheet[f'A{current_row}'] = product
            samples_sheet[f'B{current_row}'] = product_desc_dict.get(product, "N/A")
            samples_sheet[f'C{current_row}'] = len(product_data)
            samples_sheet[f'D{current_row}'] = f"{product_data['Quantity'].mean():.2f}" if 'Quantity' in product_data else "N/A"
            samples_sheet[f'E{current_row}'] = f"{product_data['Quantity'].max():.2f}" if 'Quantity' in product_data else "N/A"
            samples_sheet[f'F{current_row}'] = f"{product_data['TotalPrice'].sum():.2f}" if 'TotalPrice' in product_data else "N/A"

            # 获取主要/最终阶段 (Get main/final phase)
            if 'LifecyclePhase' in product_data.columns:
                final_phase_id = int(product_data.sort_values('InvoiceDate')['LifecyclePhase'].iloc[-1])
                final_phase_name = phase_names_map.get(final_phase_id, "未知")
                samples_sheet[f'G{current_row}'] = final_phase_name
                apply_fill(samples_sheet[f'G{current_row}'], phase_colors_hex.get(final_phase_id))

                # 计算阶段转换次数 (Calculate phase transitions)
                phases = product_data.sort_values('InvoiceDate')['LifecyclePhase'].values
                transitions = sum(1 for i in range(len(phases) - 1) if phases[i] != phases[i+1])
                samples_sheet[f'H{current_row}'] = transitions
            else:
                samples_sheet[f'G{current_row}'] = "N/A"
                samples_sheet[f'H{current_row}'] = "N/A"

            current_row += 1

    # 设置列宽 (Set Column Widths)
    samples_sheet.column_dimensions['A'].width = 15
    samples_sheet.column_dimensions['B'].width = 40
    samples_sheet.column_dimensions['C'].width = 12
    samples_sheet.column_dimensions['D'].width = 15
    samples_sheet.column_dimensions['E'].width = 15
    samples_sheet.column_dimensions['F'].width = 15
    samples_sheet.column_dimensions['G'].width = 15
    samples_sheet.column_dimensions['H'].width = 15


    # --- 7. 创建训练历史工作表 (Create Training History Sheet) ---
    if model_history and 'loss' in model_history: # 仅当历史记录有效时创建 (Create only if history is valid)
        history_sheet = wb.create_sheet("训练历史") # Training History
        # 设置标题 (Set Title)
        history_sheet['A1'] = "模型训练历史详情" # Model Training History Details
        history_sheet['A1'].font = Font(bold=True, size=16)
        history_sheet.merge_cells('A1:F1')
        history_sheet['A1'].alignment = Alignment(horizontal='center')

        # 添加列标题 (Add column headers)
        history_sheet['A3'] = "轮次" # Epoch
        history_sheet['B3'] = "训练损失" # Train Loss
        history_sheet['C3'] = "训练准确率" # Train Accuracy
        history_sheet['D3'] = "验证损失" # Val Loss
        history_sheet['E3'] = "验证准确率" # Val Accuracy
        history_sheet['F3'] = "过拟合迹象" # Overfitting Sign

        # 填充训练历史数据 (Fill training history data)
        current_row = 4
        epochs_ran = len(model_history.get('loss', []))
        for i in range(epochs_ran):
             history_sheet[f'A{current_row+i}'] = i + 1
             # 格式化数值 (Format numbers)
             tl = model_history['loss'][i]
             ta = model_history['accuracy'][i]
             vl = model_history['val_loss'][i]
             va = model_history['val_accuracy'][i]
             history_sheet[f'B{current_row+i}'] = f"{tl:.4f}"
             history_sheet[f'C{current_row+i}'] = f"{ta:.4f}"
             history_sheet[f'D{current_row+i}'] = f"{vl:.4f}"
             history_sheet[f'E{current_row+i}'] = f"{va:.4f}"

             # 检查过拟合迹象 (Check for overfitting signs)
             # (简单规则：训练准确率显著高于验证准确率，且训练损失显著低于验证损失)
             # (Simple rule: train acc significantly higher than val acc, and train loss significantly lower than val loss)
             overfitting_status = "否" # No
             if ta > va + 0.1 and tl < vl * 0.8:
                  overfitting_status = "明显" # Obvious
             elif ta > va + 0.05 and tl < vl * 0.9:
                  overfitting_status = "轻微" # Slight
             history_sheet[f'F{current_row+i}'] = overfitting_status
             # 应用颜色标记 (Apply color flag)
             apply_fill(history_sheet[f'F{current_row+i}'], overfitting_colors_hex.get(overfitting_status.replace('否','Nein').replace('轻微','Leicht').replace('明显','Deutlich'))) # Translate back for color map

        # 设置列宽 (Set Column Widths)
        for col_letter in ['A', 'B', 'C', 'D', 'E', 'F']:
             history_sheet.column_dimensions[col_letter].width = 18


    # --- 11. 创建推荐策略工作表 (Create Recommended Strategies Sheet) ---
    # (Worksheet 8-10, 12, 13 are optional/complex, skipping detailed implementation here)
    strategy_sheet = wb.create_sheet("推荐策略") # Recommended Strategies
    # 设置标题 (Set Title)
    strategy_sheet['A1'] = "产品生命周期阶段 - 推荐策略" # Product Lifecycle Phase - Recommended Strategies
    strategy_sheet['A1'].font = Font(bold=True, size=16)
    strategy_sheet.merge_cells('A1:F1')
    strategy_sheet['A1'].alignment = Alignment(horizontal='center')

    # 添加表头 (Add headers)
    strategy_sheet['A3'] = "生命周期阶段" # Lifecycle Phase
    strategy_sheet['B3'] = "主要特征" # Key Characteristics
    strategy_sheet['C3'] = "营销建议" # Marketing Suggestions
    strategy_sheet['D3'] = "定价策略" # Pricing Strategy
    strategy_sheet['E3'] = "库存管理" # Inventory Management
    strategy_sheet['F3'] = "风险点" # Risk Points

    # 策略内容 (Strategy content - Example)
    strategies_data = [
        ("导入期", "销量低, 增长可能为正", "提高知名度, 市场教育, 吸引早期用户", "渗透定价或撇脂定价", "保持较低安全库存", "市场接受度低, 竞争出现"),
        ("成长期", "销量快速增长", "扩大市场份额, 建立品牌偏好, 增加渠道", "维持价格或随成本略增", "增加库存以满足需求, 优化供应链", "产能不足, 竞争加剧, 质量控制"),
        ("成熟期", "销量增长放缓, 达到峰值或稳定", "维持市场份额, 差异化, 细分市场", "竞争性定价, 价格战可能出现", "优化库存水平, 降低成本", "市场饱和, 替代品威胁, 价格压力"),
        ("衰退期", "销量持续下降", "削减成本, 收割策略, 专注于利基市场或核心用户", "降价清仓, 捆绑销售", "逐步减少库存, 避免积压", "利润下降, 库存贬值, 品牌形象老化")
    ]

    current_row = 4
    for i, strategy_row in enumerate(strategies_data):
        phase_name = strategy_row[0]
        phase_id = list(phase_names_map.keys())[list(phase_names_map.values()).index(phase_name)] # Find phase ID
        # 填充行数据 (Fill row data)
        for j, value in enumerate(strategy_row):
            cell = strategy_sheet.cell(row=current_row + i, column=j + 1)
            cell.value = value
            cell.alignment = Alignment(wrap_text=True, vertical='top')
        # 应用阶段颜色 (Apply phase color)
        apply_fill(strategy_sheet[f'A{current_row + i}'], phase_colors_hex.get(phase_id))
        strategy_sheet.row_dimensions[current_row + i].height = 80 # 设置行高 (Set row height)

    # 设置列宽 (Set Column Widths)
    for col_letter in ['A', 'B', 'C', 'D', 'E', 'F']:
        strategy_sheet.column_dimensions[col_letter].width = 25


    # --- 保存工作簿 (Save Workbook) ---
    try:
        wb.save(file_path)
        print(f"详细结果已成功导出到 Excel: {file_path}") # Detailed results successfully exported to Excel...
    except Exception as e:
        print(f"错误: 保存 Excel 文件失败: {e}") # ERROR: Failed to save Excel file...
        # 尝试使用备用名称保存 (Try saving with a backup name)
        alt_path = file_path.replace(".xlsx", "_backup.xlsx")
        try:
             wb.save(alt_path)
             print(f"文件已成功保存到备用路径: {alt_path}") # File successfully saved to backup path...
        except Exception as e2:
             print(f"错误: 无法保存到备用路径: {e2}") # ERROR: Could not save to backup path either...

    return file_path
```

## 18. `product-lifecycle-prediction/main.py`

```python
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
        min_days = config['data']['min_days_threshold']
        product_features_df, valid_products = create_features(daily_sales, product_desc, windows, min_days)
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
```

## 注意事项 (Notes):

* **`__init__.py` 文件:** 这些文件是必需的，即使它们是空的，Python 也需要它们来将目录识别为包。
* **`excel_export.py`:** 由于该文件非常庞大，上面的 Markdown 中只包含了它的框架和一些关键部分的注释。您需要将完整的 `export_results_to_excel` 函数（如前一个回复中所示）放入实际的 `src/utils/excel_export.py` 文件中。
* **运行代码:** 确保您已经创建了所有目录并将代码放入了正确的文件中，然后从 `product-lifecycle-prediction` 根目录运行 `python main.py`。
* **数据文件:** 不要忘记将您的 `.csv` 数据文件放在 `data/raw/` 目录下。
* **依赖项:** 运行 `pip install -r requirements.txt` 来安装所有必需的库。
* **错误处理和日志记录:** 在实际生产环境中，您可能希望添加更健壮的错误处理和日志记录机制。
````