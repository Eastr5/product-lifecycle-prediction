import os
import numpy as np
import pandas as pd # 添加导入 (Added import)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# 导入评估指标和可视化工具 (Import evaluation metrics and visualization)
from src.evaluation.metrics import get_classification_report
from src.evaluation.visualization import plot_training_history, plot_confusion_matrix

class HybridModel(nn.Module):
    def __init__(self, input_dim, seq_length, num_classes=4):
        super(HybridModel, self).__init__()
        
        # CNN部分
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(2)
        
        # 计算CNN输出尺寸
        cnn_output_len = seq_length // 4  # 因为有两个池化层，每个将长度减半
        cnn_output_dim = 128 * cnn_output_len
        
        # LSTM部分
        self.lstm = nn.LSTM(input_dim, 64, batch_first=True)
        
        # 全连接层
        self.fc1 = nn.Linear(cnn_output_dim + 64, 64)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, num_classes)
        
        # 激活函数
        self.relu = nn.ReLU()
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # CNN部分 - 需要转置输入以匹配PyTorch的Conv1d期望格式 [batch, channels, seq_len]
        x_cnn = x.transpose(1, 2)
        x_cnn = self.relu(self.conv1(x_cnn))
        x_cnn = self.pool1(x_cnn)
        x_cnn = self.relu(self.conv2(x_cnn))
        x_cnn = self.pool2(x_cnn)
        x_cnn = x_cnn.reshape(batch_size, -1)  # 展平
        
        # LSTM部分
        x_lstm, (hidden, _) = self.lstm(x)
        x_lstm = hidden[-1]  # 取最后一个时间步的隐藏状态
        
        # 融合CNN和LSTM输出
        combined = torch.cat((x_cnn, x_lstm), dim=1)
        
        # 全连接层
        out = self.relu(self.fc1(combined))
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out

class LSTMModel(nn.Module):
    def __init__(self, input_dim, seq_length, num_classes=4):
        super(LSTMModel, self).__init__()
        
        # LSTM层
        self.lstm1 = nn.LSTM(input_dim, 128, batch_first=True, return_sequences=True)
        self.lstm2 = nn.LSTM(128, 64, batch_first=True)
        
        # 全连接层
        self.fc1 = nn.Linear(64, 32)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(32, num_classes)
        
        # 激活函数
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # LSTM层
        x, (h_n, c_n) = self.lstm1(x)
        x, (h_n, c_n) = self.lstm2(x)
        
        # 取最后一个时间步的隐藏状态
        x = h_n[-1]
        
        # 全连接层
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

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