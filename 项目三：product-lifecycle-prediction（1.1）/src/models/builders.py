"""
模型构建模块
"""
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Input, Concatenate, Flatten

def build_lstm_model(input_shape, n_classes=4):
    """
    构建LSTM模型
    
    参数:
    input_shape: 输入形状，如(30, 16)表示30个时间点，每个时间点16个特征
    n_classes: 输出类别数
    
    返回:
    构建好的模型
    """
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(n_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def build_hybrid_model(input_shape, n_classes=4):
    """
    构建CNN+LSTM混合模型
    
    参数:
    input_shape: 输入形状
    n_classes: 输出类别数
    
    返回:
    构建好的模型
    """
    # 输入层
    input_seq = Input(shape=input_shape)
    
    # CNN部分 - 捕捉局部时间模式
    conv1 = Conv1D(filters=64, kernel_size=3, activation='relu')(input_seq)
    pool1 = MaxPooling1D(pool_size=2)(conv1)
    conv2 = Conv1D(filters=128, kernel_size=3, activation='relu')(pool1)
    pool2 = MaxPooling1D(pool_size=2)(conv2)
    
    # LSTM部分 - 捕捉长期依赖
    lstm_out = LSTM(64, return_sequences=False)(input_seq)
    
    # 合并CNN和LSTM的输出
    merged = Concatenate()([Flatten()(pool2), lstm_out])
    
    # 全连接层
    dense1 = Dense(64, activation='relu')(merged)
    dropout = Dropout(0.3)(dense1)
    output = Dense(n_classes, activation='softmax')(dropout)
    
    # 构建模型
    model = Model(inputs=input_seq, outputs=output)
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model