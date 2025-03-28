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