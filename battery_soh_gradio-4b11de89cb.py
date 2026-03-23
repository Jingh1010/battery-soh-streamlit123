"""
基于深度学习的蓄电池健康状态预测系统 - Gradio版本
使用LSTM神经网络进行时序预测
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gradio as gr
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import io
import base64

# 设置随机种子以确保可重复性
np.random.seed(42)
tf.random.set_seed(42)

# ==================== 数据生成模块 ====================

def generate_battery_data(n_samples=1500):
    """
    生成更真实的电池循环测试数据
    模拟电池衰减的非线性特性
    """
    np.random.seed(42)
    
    # 循环次数（从小到大排序，模拟时间序列）
    cycles = np.sort(np.random.randint(1, 2000, n_samples))
    
    # SOH衰减模型：初期缓慢，中期加速，后期平缓（典型电池衰减曲线）
    cycle_norm = cycles / 2000.0  # 归一化循环次数
    
    # 使用分段函数模拟真实衰减
    base_soh = np.zeros_like(cycles)
    base_soh[cycle_norm < 0.3] = 100 - 5 * (cycle_norm[cycle_norm < 0.3] / 0.3)  # 前30%循环
    base_soh[(cycle_norm >= 0.3) & (cycle_norm < 0.7)] = 95 - 20 * ((cycle_norm[(cycle_norm >= 0.3) & (cycle_norm < 0.7)] - 0.3) / 0.4)  # 中期
    base_soh[cycle_norm >= 0.7] = 75 - 10 * ((cycle_norm[cycle_norm >= 0.7] - 0.7) / 0.3)  # 后期
    
    # 添加噪声和随机波动
    soh_noise = np.random.normal(0, 1.5, n_samples)
    soh = np.clip(base_soh + soh_noise, 40, 100)
    
    # 电压与SOH相关，但有一定随机性
    voltage = 3.7 + 0.3 * (soh / 100) + np.random.normal(0, 0.03, n_samples)
    
    # 电流（模拟充放电循环）
    current = np.where(np.random.rand(n_samples) > 0.5, 
                      1.0 + np.random.normal(0, 0.1, n_samples),  # 充电
                      -0.9 + np.random.normal(0, 0.1, n_samples))  # 放电
    
    # 温度随循环次数升高，同时受环境影响
    temperature = 25 + 10 * cycle_norm + np.random.normal(0, 3, n_samples)
    
    # 容量与SOH高度相关
    capacity = 3.0 * (soh / 100) + np.random.normal(0, 0.08, n_samples)
    
    # 内阻（随SOH下降而增加）
    internal_resistance = 0.05 + (100 - soh) / 100 * 0.15 + np.random.normal(0, 0.01, n_samples)
    
    # 创建DataFrame
    df = pd.DataFrame({
        'Cycle': cycles,
        'Voltage': voltage,
        'Current': current,
        'Temperature': temperature,
        'Capacity': capacity,
        'Internal_Resistance': internal_resistance,
        'SOH': soh
    })
    
    return df

# ==================== 特征工程模块 ====================

def engineer_features(df):
    """
    提取高级特征
    """
    df = df.copy()
    
    # 容量衰减率
    df['Capacity_Decay_Rate'] = (df['Capacity'].max() - df['Capacity']) / df['Capacity'].max()
    
    # 电压标准差（滑动窗口）
    df['Voltage_Std'] = df['Voltage'].rolling(window=10, min_periods=1).std()
    
    # 温度变化率
    df['Temp_Change_Rate'] = df['Temperature'].pct_change().fillna(0)
    
    # 循环效率（非线性）
    df['Cycle_Efficiency'] = 1 / (1 + 0.001 * df['Cycle'])
    
    # 充放电标志
    df['Charge_Flag'] = (df['Current'] > 0).astype(int)
    
    # 功率
    df['Power'] = df['Voltage'] * df['Current']
    
    # 填充可能的NaN值
    df.fillna(method='bfill', inplace=True)
    df.fillna(method='ffill', inplace=True)
    
    return df

# ==================== 数据预处理模块 ====================

def create_sequences(data, target, sequence_length=20):
    """
    为LSTM创建时序序列
    """
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:(i + sequence_length)])
        y.append(target[i + sequence_length])
    return np.array(X), np.array(y)

def preprocess_data(df, sequence_length=20, test_split=0.15, val_split=0.15):
    """
    完整的数据预处理流程
    """
    # 特征工程
    df_engineered = engineer_features(df)
    
    # 选择特征
    feature_cols = ['Cycle', 'Voltage', 'Current', 'Temperature', 'Capacity', 
                    'Internal_Resistance', 'Capacity_Decay_Rate', 'Voltage_Std',
                    'Temp_Change_Rate', 'Cycle_Efficiency', 'Charge_Flag', 'Power']
    
    X = df_engineered[feature_cols].values
    y = df_engineered['SOH'].values
    
    # 标准化
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
    
    # 划分数据集
    # 先划分训练集和临时集
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_scaled, y_scaled, test_size=(test_split + val_split), random_state=42, shuffle=False
    )
    
    # 再划分验证集和测试集
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=test_split/(test_split + val_split), 
        random_state=42, shuffle=False
    )
    
    # 创建序列
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, sequence_length)
    X_val_seq, y_val_seq = create_sequences(X_val, y_val, sequence_length)
    X_test_seq, y_test_seq = create_sequences(X_test, y_test, sequence_length)
    
    # 调整y的维度以匹配序列长度
    y_train_seq = y_train[sequence_length:]
    y_val_seq = y_val[sequence_length:]
    y_test_seq = y_test[sequence_length:]
    
    return (X_train_seq, y_train_seq), (X_val_seq, y_val_seq), (X_test_seq, y_test_seq), scaler_X, scaler_y, feature_cols

# ==================== 模型构建模块 ====================

def build_lstm_model(input_shape):
    """
    构建LSTM神经网络模型
    """
    model = Sequential([
        # 第一层LSTM
        LSTM(64, return_sequences=True, input_shape=input_shape),
        BatchNormalization(),
        Dropout(0.2),
        
        # 第二层LSTM
        LSTM(32, return_sequences=False),
        BatchNormalization(),
        Dropout(0.2),
        
        # 全连接层
        Dense(16, activation='relu'),
        Dropout(0.1),
        
        # 输出层
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    
    return model

# ==================== 可视化模块 ====================

def plot_training_history(history):
    """
    绘制训练历史
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss
    ax1.plot(history.history['loss'], label='Training Loss', linewidth=2)
    ax1.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss (MSE)', fontsize=12)
    ax1.set_title('Model Loss', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # MAE
    ax2.plot(history.history['mae'], label='Training MAE', linewidth=2)
    ax2.plot(history.history['val_mae'], label='Validation MAE', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('MAE', fontsize=12)
    ax2.set_title('Model MAE', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 转换为图片
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    
    return buf

def plot_predictions(y_true, y_pred, title="预测结果"):
    """
    绘制预测结果对比
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # 1. 时序对比（前200个点）
    axes[0, 0].plot(y_true[:200], 'b-', label='真实值', linewidth=2, alpha=0.7)
    axes[0, 0].plot(y_pred[:200], 'r--', label='预测值', linewidth=2, alpha=0.7)
    axes[0, 0].set_xlabel('样本索引', fontsize=11)
    axes[0, 0].set_ylabel('SOH (%)', fontsize=11)
    axes[0, 0].set_title('时序对比（前200个样本）', fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 散点图
    axes[0, 1].scatter(y_true, y_pred, alpha=0.5, s=20, edgecolors='black', linewidths=0.5)
    max_val = max(y_true.max(), y_pred.max())
    min_val = min(y_true.min(), y_pred.min())
    axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='理想线')
    axes[0, 1].set_xlabel('真实 SOH (%)', fontsize=11)
    axes[0, 1].set_ylabel('预测 SOH (%)', fontsize=11)
    axes[0, 1].set_title('预测值 vs 真实值', fontsize=12, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 计算R²
    r2 = r2_score(y_true, y_pred)
    axes[0, 1].text(0.05, 0.95, f'R² = {r2:.4f}', 
                    transform=axes[0, 1].transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                    fontsize=12, fontweight='bold')
    
    # 3. 残差分布
    residuals = y_true - y_pred
    axes[1, 0].hist(residuals, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    axes[1, 0].axvline(x=0, color='red', linestyle='--', linewidth=2)
    axes[1, 0].set_xlabel('残差 (%)', fontsize=11)
    axes[1, 0].set_ylabel('频数', fontsize=11)
    axes[1, 0].set_title('残差分布直方图', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # 添加统计信息
    mean_res = np.mean(residuals)
    std_res = np.std(residuals)
    axes[1, 0].text(0.95, 0.95, f'均值: {mean_res:.2f}%\n标准差: {std_res:.2f}%', 
                    transform=axes[1, 0].transAxes, verticalalignment='top',
                    horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
                    fontsize=10)
    
    # 4. 误差分布
    errors = np.abs(residuals)
    axes[1, 1].plot(errors, 'o', alpha=0.5, markersize=3)
    axes[1, 1].axhline(y=np.mean(errors), color='red', linestyle='--', linewidth=2, label=f'平均误差: {np.mean(errors):.2f}%')
    axes[1, 1].set_xlabel('样本索引', fontsize=11)
    axes[1, 1].set_ylabel('绝对误差 (%)', fontsize=11)
    axes[1, 1].set_title('误差分布', fontsize=12, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    
    return buf

# ==================== 主应用类 ====================

class BatterySOHPredictor:
    def __init__(self):
        self.model = None
        self.scaler_X = None
        self.scaler_y = None
        self.feature_cols = None
        self.is_trained = False
        self.history = None
        self.df = None
        self.sequence_length = 20
        
    def train_model(self, n_samples=1500, epochs=50, batch_size=32):
        """
        训练模型
        """
        # 生成数据
        self.df = generate_battery_data(n_samples)
        
        # 预处理
        (X_train, y_train), (X_val, y_val), (X_test, y_test), self.scaler_X, self.scaler_y, self.feature_cols = \
            preprocess_data(self.df, sequence_length=self.sequence_length)
        
        # 构建模型
        self.model = build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))
        
        # 回调函数
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
        
        # 训练
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr],
            verbose=0
        )
        
        self.is_trained = True
        
        # 评估
        y_pred = self.model.predict(X_test, verbose=0)
        y_pred = self.scaler_y.inverse_transform(y_pred).flatten()
        y_true = self.scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
        
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        return {
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'n_train': len(X_train),
            'n_val': len(X_val),
            'n_test': len(X_test)
        }
    
    def get_training_plot(self):
        """获取训练历史图"""
        if not self.is_trained or self.history is None:
            return None
        return plot_training_history(self.history)
    
    def get_prediction_plot(self):
        """获取预测结果图"""
        if not self.is_trained:
            return None
        
        # 重新预测测试集
        df = generate_battery_data(1500)
        _, _, (X_test, y_test), _, _, _ = preprocess_data(df, sequence_length=self.sequence_length)
        
        y_pred = self.model.predict(X_test, verbose=0)
        y_pred = self.scaler_y.inverse_transform(y_pred).flatten()
        y_true = self.scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
        
        return plot_predictions(y_true, y_pred, "LSTM模型预测结果")
    
    def predict_single(self, cycle, voltage, current, temperature, capacity, resistance):
        """
        单点预测
        """
        if not self.is_trained:
            return None
        
        # 创建输入数据
        input_data = pd.DataFrame({
            'Cycle': [cycle],
            'Voltage': [voltage],
            'Current': [current],
            'Temperature': [temperature],
            'Capacity': [capacity],
            'Internal_Resistance': [resistance],
            'Capacity_Decay_Rate': [0],
            'Voltage_Std': [0],
            'Temp_Change_Rate': [0],
            'Cycle_Efficiency': [1 / (1 + 0.001 * cycle)],
            'Charge_Flag': [1 if current > 0 else 0],
            'Power': [voltage * current]
        })
        
        # 特征工程
        input_data = engineer_features(input_data)
        
        # 标准化
        input_scaled = self.scaler_X.transform(input_data[self.feature_cols])
        
        # 创建序列（复制sequence_length次）
        input_seq = np.tile(input_scaled, (self.sequence_length, 1, 1))
        
        # 预测
        prediction = self.model.predict(input_seq, verbose=0)[0][0]
        
        # 反标准化
        prediction = self.scaler_y.inverse_transform([[prediction]])[0][0]
        
        return prediction

# ==================== Gradio界面 ====================

# 创建全局预测器实例
predictor = BatterySOHPredictor()

def train_interface(n_samples, epochs, batch_size, progress=gr.Progress()):
    """训练界面"""
    progress(0, desc="正在生成数据...")
    
    metrics = predictor.train_model(
        n_samples=int(n_samples),
        epochs=int(epochs),
        batch_size=int(batch_size)
    )
    
    progress(0.5, desc="正在生成图表...")
    
    # 生成图表
    train_plot = predictor.get_training_plot()
    pred_plot = predictor.get_prediction_plot()
    
    progress(1.0, desc="完成！")
    
    # 构建结果文本
    result_text = f"""
## 🎉 模型训练完成！

### 📊 模型性能指标
- **均方误差 (MSE)**: {metrics['mse']:.4f}
- **平均绝对误差 (MAE)**: {metrics['mae']:.4f}
- **决定系数 (R²)**: {metrics['r2']:.4f}

### 📈 数据集划分
- **训练集**: {metrics['n_train']} 条
- **验证集**: {metrics['n_val']} 条
- **测试集**: {metrics['n_test']} 条

### 🤖 模型架构
- **类型**: LSTM神经网络
- **层数**: 2层LSTM + 2层全连接
- **训练轮数**: {epochs}
- **批量大小**: {batch_size}
"""
    
    return result_text, train_plot, pred_plot, gr.update(visible=True)

def predict_interface(cycle, voltage, current, temperature, capacity, resistance):
    """预测界面"""
    if not predictor.is_trained:
        return "⚠️ 请先训练模型！", None
    
    prediction = predictor.predict_single(cycle, voltage, current, temperature, capacity, resistance)
    
    # 健康状态评估
    if prediction >= 90:
        status = "🟢 优秀"
        advice = "电池状态良好，继续保持正常使用和维护。"
        color = "#4CAF50"
    elif prediction >= 75:
        status = "🟡 良好"
        advice = "电池状态尚可，建议关注充放电习惯。"
        color = "#FFC107"
    elif prediction >= 60:
        status = "🟠 一般"
        advice = "电池性能有所下降，建议适当维护。"
        color = "#FF9800"
    else:
        status = "🔴 需要更换"
        advice = "电池性能严重衰减，建议尽快更换。"
        color = "#F44336"
    
    result = f"""
## 🔮 预测结果

### SOH: {prediction:.2f}%

### 健康状态: {status}

### 使用建议
{advice}

### 输入参数
- 循环次数: {cycle} 次
- 电压: {voltage} V
- 电流: {current} A
- 温度: {temperature} °C
- 容量: {capacity} Ah
- 内阻: {resistance} Ω
"""
    
    return result

def export_data():
    """导出数据"""
    if not predictor.is_trained:
        return None
    
    # 生成完整数据集
    df = generate_battery_data(1500)
    df = engineer_features(df)
    
    # 保存为CSV
    output = io.StringIO()
    df.to_csv(output, index=False)
    output.seek(0)
    
    return output.getvalue()

# 创建Gradio界面
with gr.Blocks(title="电池SOH预测系统", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🔋 蓄电池健康状态（SOH）预测系统
    ## 基于LSTM深度学习的智能预测平台
    
    本系统使用LSTM神经网络对蓄电池的健康状态进行预测，支持时序数据处理和多特征融合。
    """)
    
    with gr.Tabs():
        # Tab 1: 训练模型
        with gr.Tab("🎓 模型训练"):
            gr.Markdown("### 配置训练参数")
            
            with gr.Row():
                with gr.Column():
                    n_samples = gr.Slider(500, 3000, value=1500, label="数据集大小", step=100)
                    epochs = gr.Slider(10, 100, value=50, label="训练轮数", step=5)
                    batch_size = gr.Slider(16, 64, value=32, label="批量大小", step=8)
                    
                    train_btn = gr.Button("🚀 开始训练", variant="primary", size="lg")
            
            with gr.Row():
                train_result = gr.Markdown(label="训练结果")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 📊 训练过程")
                    train_plot = gr.Image(label="训练历史")
                with gr.Column():
                    gr.Markdown("### 🎯 预测效果")
                    pred_plot = gr.Image(label="预测结果")
            
            train_btn.click(
                train_interface,
                inputs=[n_samples, epochs, batch_size],
                outputs=[train_result, train_plot, pred_plot]
            )
        
        # Tab 2: 预测
        with gr.Tab("🔮 实时预测"):
            gr.Markdown("### 输入电池参数进行预测")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### 基本参数")
                    cycle = gr.Number(value=500, label="循环次数 (Cycle)", minimum=1, maximum=5000)
                    voltage = gr.Number(value=3.7, label="电压 (V)", minimum=2.5, maximum=4.5, step=0.01)
                    current = gr.Number(value=1.0, label="电流 (A)", minimum=-5.0, maximum=5.0, step=0.1)
                
                with gr.Column():
                    gr.Markdown("#### 性能参数")
                    temperature = gr.Number(value=30.0, label="温度 (℃)", minimum=-20, maximum=80)
                    capacity = gr.Number(value=2.5, label="容量 (Ah)", minimum=0.5, maximum=5.0, step=0.1)
                    resistance = gr.Number(value=0.08, label="内阻 (Ω)", minimum=0.01, maximum=1.0, step=0.01)
                
                with gr.Column():
                    gr.Markdown("#### 预测结果")
                    predict_btn = gr.Button("🔍 立即预测", variant="primary", size="lg")
                    predict_result = gr.Markdown(label="预测结果")
            
            predict_btn.click(
                predict_interface,
                inputs=[cycle, voltage, current, temperature, capacity, resistance],
                outputs=[predict_result]
            )
        
        # Tab 3: 数据导出
        with gr.Tab("📥 数据导出"):
            gr.Markdown("### 导出训练数据集")
            gr.Markdown("下载包含所有特征和目标变量的完整数据集（CSV格式）")
            
            export_btn = gr.Button("📊 导出数据", variant="secondary")
            data_file = gr.File(label="下载文件")
            
            export_btn.click(
                export_data,
                outputs=[data_file]
            )
        
        # Tab 4: 系统说明
        with gr.Tab("📖 使用说明"):
            gr.Markdown("""
            ## 🎯 系统简介
            
            本系统基于**LSTM（长短期记忆网络）**深度学习模型，用于预测蓄电池的健康状态（SOH）。
            
            ### 核心技术
            
            - **LSTM神经网络**: 专门处理时间序列数据，能够捕捉电池性能衰减的时序特征
            - **多特征融合**: 整合电压、电流、温度、容量、内阻等多维度信息
            - **自动特征工程**: 自动提取容量衰减率、电压变化率等高级特征
            - **端到端训练**: 从数据预处理到模型训练的完整自动化流程
            
            ### 使用流程
            
            1. **训练模型**
               - 设置数据集大小、训练轮数、批量大小
               - 点击"开始训练"按钮
               - 等待训练完成（通常需要1-3分钟）
               - 查看训练历史和预测效果
            
            2. **实时预测**
               - 输入电池的各项参数
               - 点击"立即预测"按钮
               - 获取SOH预测结果和健康状态评估
            
            3. **数据导出**
               - 导出完整的训练数据集
               - 用于离线分析或进一步研究
            
            ### 模型评估指标
            
            - **MSE (均方误差)**: 衡量预测误差的平方均值，越小越好
            - **MAE (平均绝对误差)**: 衡量预测误差的平均值，越小越好
            - **R² (决定系数)**: 衡量模型对数据的解释能力，越接近1越好
            
            ### 特征说明
            
            | 特征 | 说明 |
            |------|------|
            | Cycle | 充放电循环次数 |
            | Voltage | 电池电压 |
            | Current | 电池电流（正为充电，负为放电） |
            | Temperature | 电池温度 |
            | Capacity | 电池容量 |
            | Internal_Resistance | 电池内阻 |
            | Capacity_Decay_Rate | 容量衰减率（自动计算） |
            | Voltage_Std | 电压标准差（滑动窗口） |
            | Temp_Change_Rate | 温度变化率 |
            | Cycle_Efficiency | 循环效率 |
            | Charge_Flag | 充放电标志 |
            | Power | 功率（电压×电流） |
            
            ### 技术栈
            
            - **深度学习框架**: TensorFlow / Keras
            - **数据处理**: Pandas, NumPy
            - **可视化**: Matplotlib
            - **Web界面**: Gradio
            - **编程语言**: Python 3.8+
            
            ### 注意事项
            
            1. 首次使用必须先训练模型
            2. 训练过程可能需要较长时间，请耐心等待
            3. 预测结果仅供参考，实际应用需结合专业测试
            4. 系统会自动进行数据预处理和特征工程
            
            ---
            
            **开发者**: AI实践项目  
            **更新时间**: 2026年3月
            """)

# 启动应用
if __name__ == "__main__":
    demo.launch(share=True, server_name="0.0.0.0", server_port=7860)
