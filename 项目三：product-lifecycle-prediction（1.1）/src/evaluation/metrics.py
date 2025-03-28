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