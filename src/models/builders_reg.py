# product_lifecycle_predictor_vA/src/models/builders_reg.py
import torch
import torch.nn as nn
import torch.nn.functional as F 
import logging
from typing import Dict, Any, List, Optional

# 移除了之前的单头 Attention 类，因为我们将使用 nn.MultiheadAttention

class LSTMRegressor(nn.Module):
    """
    一个基于 LSTM 的回归模型，用于多步时间序列预测。
    可以处理数值特征和类别特征（通过嵌入层）。
    新增了对 PyTorch 内置多头注意力机制的支持。
    """
    def __init__(self,
                 num_numerical_features: int,
                 categorical_embedding_info: Optional[Dict[str, Dict[str, int]]] = None, 
                 hidden_units_lstm: List[int] = [64], # 保持与v1.13.1一致
                 num_lstm_layers: int = 2,
                 lstm_dropout: float = 0.1,
                 bidirectional_lstm: bool = False,
                 use_attention: bool = False, 
                 attention_type: str = "simple", # "simple" (我们之前的实现) 或 "multihead"
                 num_attention_heads: int = 4,   # 仅当 attention_type == "multihead" 时使用
                 attention_dropout: float = 0.0, # 仅当 attention_type == "multihead" 时使用
                 dense_units_regressor: List[int] = [64, 32], # 保持与v1.13.1一致
                 dense_dropout_regressor: float = 0.1,
                 num_outputs: int = 1):
        super().__init__()
        self.categorical_embedding_info = categorical_embedding_info if categorical_embedding_info else {}
        self.embeddings = nn.ModuleDict()
        total_embedding_output_dim = 0

        if self.categorical_embedding_info:
            for col_name, emb_info in self.categorical_embedding_info.items():
                vocab_size = emb_info.get('vocab_size')
                embedding_dim = emb_info.get('embedding_dim')
                if vocab_size is None or embedding_dim is None:
                    logging.warning(f"类别特征 '{col_name}' 的 vocab_size 或 embedding_dim 未提供，将跳过此嵌入层。")
                    continue
                if vocab_size <= 0: 
                    logging.warning(f"类别特征 '{col_name}' 的 vocab_size ({vocab_size}) 无效，必须大于0。将跳过此嵌入层。")
                    continue
                self.embeddings[col_name] = nn.Embedding(vocab_size, embedding_dim, padding_idx=0) 
                total_embedding_output_dim += embedding_dim
                logging.info(f"为 '{col_name}' 创建嵌入层: vocab_size={vocab_size}, embedding_dim={embedding_dim}")
        
        lstm_input_dim = num_numerical_features + total_embedding_output_dim
        if lstm_input_dim == 0:
            raise ValueError("LSTM 输入维度为0。请至少提供数值特征或有效的类别特征嵌入信息。")
        logging.info(f"LSTM 输入维度: {lstm_input_dim} (数值: {num_numerical_features}, 嵌入: {total_embedding_output_dim})")

        # LSTM的隐藏单元数应为列表中的第一个元素，如果列表为空则默认为64
        first_lstm_hidden_size = hidden_units_lstm[0] if hidden_units_lstm else 64
        
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=first_lstm_hidden_size, # 使用列表中的第一个值
            num_layers=num_lstm_layers,
            batch_first=True, 
            dropout=lstm_dropout if num_lstm_layers > 1 else 0, 
            bidirectional=bidirectional_lstm
        )
        logging.info(f"创建 LSTM 层: hidden_size={first_lstm_hidden_size}, num_layers={num_lstm_layers}, bidirectional={bidirectional_lstm}")

        # LSTM的输出维度
        lstm_output_feature_dim = first_lstm_hidden_size * (2 if bidirectional_lstm else 1)

        self.use_attention = use_attention
        self.attention_layer = None
        if self.use_attention:
            if attention_type == "multihead":
                # 确保 embed_dim (即 lstm_output_feature_dim) 可以被 num_heads 整除
                if lstm_output_feature_dim % num_attention_heads != 0:
                    logging.warning(
                        f"LSTM输出维度 ({lstm_output_feature_dim}) 不能被注意力头数 ({num_attention_heads}) 整除。 "
                        f"多头注意力可能无法正确初始化或运行。请调整 hidden_units_lstm 或 num_attention_heads。"
                    )
                    # 可以选择抛出错误或禁用注意力
                    self.use_attention = False 
                    logging.warning("已禁用注意力机制，因为维度不匹配。")
                else:
                    self.attention_layer = nn.MultiheadAttention(
                        embed_dim=lstm_output_feature_dim,
                        num_heads=num_attention_heads,
                        dropout=attention_dropout,
                        batch_first=True # LSTM输出是batch_first
                    )
                    logging.info(f"PyTorch MultiheadAttention 已启用: embed_dim={lstm_output_feature_dim}, num_heads={num_attention_heads}, dropout={attention_dropout}")
            else: # 默认或未知的注意力类型，可以回退到之前的简单实现或不使用
                logging.warning(f"未知的 attention_type: '{attention_type}'. 将不使用注意力机制或回退（如果实现）。当前禁用。")
                self.use_attention = False
        
        if not self.use_attention:
             logging.info("注意力机制未启用或因配置问题被禁用。")


        regressor_input_dim = lstm_output_feature_dim # 如果使用注意力，上下文向量维度与lstm输出特征维度一致
        
        regressor_layers = []
        current_dim_for_regressor = regressor_input_dim
        
        if not dense_units_regressor: 
            logging.info(f"回归头: 直接从 LSTM/Attention 输出 ({current_dim_for_regressor}) 连接到输出层 ({num_outputs})。")
        else:
            for i, units in enumerate(dense_units_regressor):
                if units <=0: 
                    logging.warning(f"回归头中全连接层单元数 {units} 无效，跳过此层。")
                    continue
                regressor_layers.append(nn.Linear(current_dim_for_regressor, units))
                regressor_layers.append(nn.ReLU()) 
                if dense_dropout_regressor > 0 and i < len(dense_units_regressor) -1 : 
                    regressor_layers.append(nn.Dropout(dense_dropout_regressor))
                current_dim_for_regressor = units 
                logging.info(f"回归头添加全连接层: output_units={units}")
        
        if num_outputs <=0:
            raise ValueError(f"模型的输出维度 num_outputs ({num_outputs}) 必须大于0。")
        regressor_layers.append(nn.Linear(current_dim_for_regressor, num_outputs)) 
        self.regressor_head = nn.Sequential(*regressor_layers)
        logging.info(f"回归头最终输出层: output_units={num_outputs}")

    def forward(self, 
                numerical_features: torch.Tensor, 
                categorical_features: Optional[Dict[str, torch.Tensor]] = None 
               ) -> torch.Tensor:
        all_feature_sequences = []
        if numerical_features.numel() > 0 and numerical_features.shape[-1] > 0 : 
             all_feature_sequences.append(numerical_features)

        if categorical_features and self.embeddings:
            for col_name, emb_layer in self.embeddings.items():
                if col_name in categorical_features:
                    cat_input = categorical_features[col_name]
                    if cat_input.dtype != torch.long:
                        cat_input = cat_input.long()
                    embedded = emb_layer(cat_input) 
                    all_feature_sequences.append(embedded)
                else:
                    logging.debug(f"前向传播中未找到类别特征 '{col_name}' 的输入数据。") 
        
        if not all_feature_sequences:
            raise ValueError("没有任何特征提供给LSTM层。")

        concatenated_features = torch.cat(all_feature_sequences, dim=-1) 
        lstm_out, (hidden, cell) = self.lstm(concatenated_features)
        # lstm_out shape: (batch_size, seq_len, num_directions * hidden_size)
        
        regressor_input = None
        if self.use_attention and self.attention_layer and isinstance(self.attention_layer, nn.MultiheadAttention):
            # MultiheadAttention期望 query, key, value
            # 在自注意力中，它们通常是相同的，即LSTM的输出序列
            # attn_output shape: (batch_size, seq_len, embed_dim)
            # context_vector shape: (batch_size, embed_dim) - 我们需要对序列维度进行聚合
            attn_output, attn_weights = self.attention_layer(lstm_out, lstm_out, lstm_out)
            # 通常取注意力输出的最后一个时间步，或者对所有时间步的注意力输出进行平均/加权平均
            # 这里我们简单地取最后一个时间步的注意力输出，与不使用注意力时保持一致
            regressor_input = attn_output[:, -1, :] 
        else:
            regressor_input = lstm_out[:, -1, :] 
        
        predictions = self.regressor_head(regressor_input) 
        
        return predictions

def build_lstm_regressor(config: Dict[str, Any], 
                         num_numerical_features: int, 
                         num_outputs: int,
                         categorical_embedding_info: Optional[Dict[str, Dict[str, int]]] = None
                        ) -> LSTMRegressor:
    model_conf = config.get('training_dl_regression', {}) 
    
    hidden_units_lstm_cfg = model_conf.get('hidden_units_lstm', [64]) 
    if not hidden_units_lstm_cfg: hidden_units_lstm_cfg = [64] 

    num_lstm_layers_cfg = model_conf.get('num_lstm_layers', 1)
    if num_lstm_layers_cfg <=0 : num_lstm_layers_cfg = 1 

    lstm_dropout_cfg = model_conf.get('lstm_dropout', 0.0)
    bidirectional_lstm_cfg = model_conf.get('bidirectional_lstm', False)
    
    use_attention_cfg = model_conf.get('use_attention', False)
    attention_type_cfg = model_conf.get('attention_type', "simple") # 默认为之前的简单实现或无
    num_attention_heads_cfg = model_conf.get('num_attention_heads', 4)
    attention_dropout_cfg = model_conf.get('attention_dropout', 0.0)

    dense_units_regressor_cfg = model_conf.get('dense_units_regressor', []) 
    dense_dropout_regressor_cfg = model_conf.get('dense_dropout_regressor', 0.0)

    logging.info(
        f"构建 LSTMRegressor 模型: 数值特征数={num_numerical_features}, "
        f"类别嵌入信息={bool(categorical_embedding_info)}, 输出目标数={num_outputs}, "
        f"使用注意力={use_attention_cfg}, 注意力类型='{attention_type_cfg}', "
        f"注意力头数={num_attention_heads_cfg if use_attention_cfg and attention_type_cfg == 'multihead' else 'N/A'}"
    )

    model = LSTMRegressor(
        num_numerical_features=num_numerical_features,
        categorical_embedding_info=categorical_embedding_info,
        hidden_units_lstm=hidden_units_lstm_cfg,
        num_lstm_layers=num_lstm_layers_cfg,
        lstm_dropout=lstm_dropout_cfg,
        bidirectional_lstm=bidirectional_lstm_cfg,
        use_attention=use_attention_cfg,
        attention_type=attention_type_cfg,
        num_attention_heads=num_attention_heads_cfg,
        attention_dropout=attention_dropout_cfg,
        dense_units_regressor=dense_units_regressor_cfg,
        dense_dropout_regressor=dense_dropout_regressor_cfg,
        num_outputs=num_outputs
    )
    
    logging.info("LSTMRegressor 模型构建（可能带注意力机制）完成。")
    return model

if __name__ == '__main__':
    # (测试代码与 builders_reg_py_with_attention 中的类似，但需要调整以测试 multihead attention)
    print("测试 src/models/builders_reg.py (带可选的多头注意力机制)...")
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    num_numeric = 10
    num_targets = 3 
    seq_len = 12
    batch_size = 4
    
    # LSTM隐藏层大小，需要能被注意力头数整除
    lstm_hidden_for_multihead = 64 
    num_heads_test = 4 # 64 % 4 == 0

    print("\n--- 测试1: 无类别特征，不使用注意力的 LSTMRegressor ---")
    # (与之前测试用例相同)
    model1_config = {
        'training_dl_regression': { 
            'hidden_units_lstm': [64], 'num_lstm_layers': 1, 'lstm_dropout': 0, 
            'bidirectional_lstm': False, 'use_attention': False,
            'dense_units_regressor': [32], 'dense_dropout_regressor': 0
        }
    }
    model1 = build_lstm_regressor(model1_config, num_numerical_features=num_numeric, num_outputs=num_targets)
    dummy_numeric_input1 = torch.randn(batch_size, seq_len, num_numeric)
    try:
        output1 = model1(numerical_features=dummy_numeric_input1); print(f"模型1输出形状: {output1.shape}")
    except Exception as e: print(f"模型1前向传播测试失败: {e}")


    print("\n--- 测试2: 带类别特征，使用双向LSTM和多头注意力的 LSTMRegressor ---")
    cat_embed_info_test = {
        'item_id': {'vocab_size': 100, 'embedding_dim': 10},
        'store_id': {'vocab_size': 10, 'embedding_dim': 5}
    }
    model2_config = {
        'training_dl_regression': {
            'hidden_units_lstm': [lstm_hidden_for_multihead], # 例如 64
            'num_lstm_layers': 1, # 为简单起见，用单层双向
            'lstm_dropout': 0.0,
            'bidirectional_lstm': True, # 双向LSTM，输出维度为 2 * lstm_hidden_for_multihead
            'use_attention': True, 
            'attention_type': "multihead",
            'num_attention_heads': num_heads_test, # 例如 4 (如果双向LSTM输出128，则128%4==0)
            'attention_dropout': 0.1,
            'dense_units_regressor': [64, 32], 
            'dense_dropout_regressor': 0.1
        }
    }
    # 双向LSTM的输出维度是 2 * lstm_hidden_for_multihead
    # 如果 lstm_hidden_for_multihead=64, bidirectional=True, 则LSTM输出特征维度是128
    # 128 % num_heads_test (4) == 0，所以是有效的
    
    model2 = build_lstm_regressor(
        model2_config, 
        num_numerical_features=num_numeric, 
        num_outputs=num_targets, 
        categorical_embedding_info=cat_embed_info_test
    )
    # print(model2) # 可以打印模型结构查看
    dummy_numeric_input2 = torch.randn(batch_size, seq_len, num_numeric)
    dummy_categorical_input2 = {
        'item_id': torch.randint(0, cat_embed_info_test['item_id']['vocab_size'], (batch_size, seq_len)),
        'store_id': torch.randint(0, cat_embed_info_test['store_id']['vocab_size'], (batch_size, seq_len)),
    }
    try:
        output2 = model2(numerical_features=dummy_numeric_input2, categorical_features=dummy_categorical_input2)
        print(f"模型2输出形状: {output2.shape}") 
        assert output2.shape == (batch_size, num_targets)
    except Exception as e:
        print(f"模型2前向传播测试失败: {e}", exc_info=True)

    print("\n--- 测试3: 维度不匹配导致注意力禁用的情况 ---")
    bad_model_config = {
        'training_dl_regression': {
            'hidden_units_lstm': [63], # 63 不能被 4 整除
            'num_lstm_layers': 1, 
            'bidirectional_lstm': False,
            'use_attention': True, 
            'attention_type': "multihead",
            'num_attention_heads': num_heads_test, # 4
            'dense_units_regressor': [32], 
        }
    }
    model_bad = build_lstm_regressor(bad_model_config, num_numerical_features=num_numeric, num_outputs=num_targets)
    # 应该会看到日志警告注意力被禁用
    assert model_bad.use_attention == False # 验证注意力是否真的被禁用了
    try:
        output_bad = model_bad(numerical_features=dummy_numeric_input1)
        print(f"模型 (注意力禁用) 输出形状: {output_bad.shape}")
        assert output_bad.shape == (batch_size, num_targets)
    except Exception as e:
        print(f"模型 (注意力禁用) 前向传播测试失败: {e}", exc_info=True)

