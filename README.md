# 使用深度学习进行产品生命周期预测

本项目是一个先进的销售预测系统，利用深度学习模型来预测产品的销售趋势和生命周期。该项目最初是为深度学习课程而设计，其强大的功能和模块化的架构也使其具备作为商业智能（BI）工具进行二次开发的潜力。

项目的核心是解决复杂的、多层次的时间序列预测问题，其设计灵感和评估体系来源于零售业权威的 M5 销售预测竞赛。

## 核心功能详解

本项目并非一个简单的脚本，而是一个完整的、端到端的深度学习应用流程。

#### 1. 端到端的执行流程
项目由一个主脚本 `main.py` 驱动，它整合了所有步骤，形成了一个完整的自动化流程：
- **加载配置**: 从 `config.yaml` 读取所有设置。
- **数据加载与特征工程**: 调用 `FeatureBuilder` 生成模型所需的特征。
- **模型训练**: 调用 `Trainer` 模块进行模型的训练。
- **模型评估**: 训练结束后，调用 `Evaluator` 计算预测得分。
- **结果解读与可视化**: 调用 `Interpreter` 生成可视化的仪表板报告。

#### 2. 数据驱动的灵活配置
所有实验参数都通过 `config.yaml` 文件进行管理，实现了代码与配置的分离。这使得研究人员可以在不修改任何Python代码的情况下，进行各种实验：
- **路径管理**: 灵活配置数据、模型和结果的存储位置。
- **模型定制**: 自由定义神经网络的层数、神经元数量、嵌入维度等。
- **训练调优**: 轻松调整学习率、批处理大小（batch size）、训练周期（epochs）等超参数。

#### 3. 精细化的特征工程
特征是决定模型上限的关键。`src/feature_engineering/builder.py` 模块负责构建一个丰富的特征集，主要包括：
- **时间特征**: 年、月、日、星期几、一年中的第几天等。
- **滞后（Lag）特征**: 过去不同时间点（如前7天，前28天）的销售额，用于捕捉短期依赖关系。
- **滚动窗口（Rolling Window）统计**: 在特定窗口期（如7天、30天）内的销售额均值、标准差、最大/最小值等，用于捕捉趋势和稳定性。
- **事件与假日**: 从 `calendar.csv` 中提取的节假日、体育赛事等特殊事件的标志位。

#### 4. 先进的深度学习模型
项目核心是一个可高度定制的深度神经网络回归器 `src/models/builders_reg.py`，其架构为：
- **实体嵌入（Entity Embeddings）**: 对高基数的分类特征（如商品ID、店铺ID、州ID）进行嵌入，将它们转换为密集的向量表示，从而捕捉其内在关联。
- **多层感知机（MLP）**: 将嵌入向量与数值特征拼接后，输入到一个包含多个隐藏层的全连接网络中，进行非线性变换和最终的销售预测。
- **自编码器（Autoencoder）**: 项目中还包含一个自编码器模型 `src/models/autoencoder.py`，可用于对时间序列进行无监督的表示学习或异常检测。

#### 5. 严谨的训练与评估
- **自定义损失函数**: 为了对齐M5竞赛的目标，项目使用了 WRMSSE（加权均方根比例误差）作为核心评估指标。评估逻辑在 `src/evaluation/evaluator_reg.py` 中实现。WRMSSE能够根据不同产品和地区的销售额和波动性，对预测误差进行加权，是一种比普通RMSE更公平、更具挑战性的指标。
- **训练流程**: `src/models/trainer_reg.py` 中实现了完整的训练循环，包括学习率调度器（Learning Rate Scheduler），以帮助模型更好地收敛。

#### 6. 交互式的结果可视化
模型的好坏最终需要直观地展现。`src/prediction_interpretation/interpreter.py` 模块使用 `Plotly` 库生成一个独立的HTML仪表板（如 `dashboard_report.html`），用户可以交互式地缩放、平移，直观地对比模型预测值与真实销售数据，从而进行深入的误差分析。

## 系统工作流

项目的数据流和处理流程可以概括如下：

`原始数据 (CSVs)` -> `[1. 特征工程]` -> `特征数据` -> `[2. 模型训练]` -> `已训练模型` -> `[3. 评估]` -> `性能分数 (JSON)` -> `[4. 解读]` -> `可视化报告 (HTML)`

## 安装与使用指南

1.  **克隆仓库并安装依赖**
    ```bash
    # 建议在虚拟环境中操作
    git clone <your-repo-url>
    cd product-lifecycle-prediction
    pip install -r requirements.txt
    ```

2.  **准备数据**
    - 在项目根目录下创建一个名为 `data` 的文件夹。
    - 将M5竞赛的三个核心数据文件 `sales_train_validation.csv`, `calendar.csv`, 和 `sell_prices.csv` 放入 `data` 文件夹。
    - 如果您的数据路径不同，请在 `config.yaml` 中更新 `data_params` 下的路径。

3.  **配置您的实验**
    - 打开 `config.yaml`。
    - 设置一个独特的 `version` 名称，这将用于创建存放结果的文件夹。
    - 根据您的实验设计，调整 `model_params` 和 `train_params`。

4.  **开始训练**
    - 运行主脚本。使用 `--version` 参数指定本次运行的版本，结果将保存在 `results/<version>/` 目录下。
    - 添加 `--gpu` 标志以使用GPU进行训练（如果可用）。

    ```bash
    # 示例：运行名为 "v1_baseline" 的实验，并使用GPU
    python main.py --version v1_baseline --gpu
    ```

5.  **查看结果**
    - 训练完成后，进入 `results/v1_baseline/` 文件夹。
    - `experiment_regression_v1_baseline.log` 文件记录了详细的训练日志。
    - `pytorch_regression_evaluation_v1_baseline.json` 文件包含了最终的评估分数。
    - `dashboard_report.html` 是您可以直接在浏览器中打开的交互式可视化报告。

## 文件结构导览

```
.
├── config.yaml                   # 核心配置文件，一切实验的起点
├── main.py                       # 项目主执行脚本
├── requirements.txt              # Python依赖包列表
├── models/                       # 存放预计算的WRMSSE权重和缩放因子
├── results/                      # 保存所有实验输出：日志、分数、报告
└── src/                          # 项目源代码
    ├── feature_engineering/    # 特征工程模块
    ├── models/                 # 包含模型定义(autoencoder.py, builders_reg.py)和训练器(trainer_reg.py)
    ├── evaluation/             # 模型评估模块
    ├── prediction_interpretation/ # 结果可视化模块
    └── utils/                  # 工具函数，如配置加载
```