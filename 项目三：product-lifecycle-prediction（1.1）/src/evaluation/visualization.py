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