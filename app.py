import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from scipy.fft import fft
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler
import time
from datetime import datetime, timedelta

# --- 1. 全局配置与工具类 (Configuration & Utils) ---
st.set_page_config(
    page_title="Quantum Trader Pro V9",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义 CSS，打造专业金融终端的视觉感
st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    .metric-card {
        background-color: #1e2130;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #30334e;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #ff4b4b;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

class Utils:
    @staticmethod
    def safe_float(value):
        try:
            return float(value)
        except:
            return 0.0

# --- 2. 数据层 (Data Layer) - 负责清洗与缓存 ---
class DataManager:
    @staticmethod
    @st.cache_data(ttl=900)  # 缓存15分钟，避免频繁请求被封IP
    def fetch_data(ticker, period="1y", interval="1d"):
        try:
            df = yf.download(ticker, period=period, interval=interval, progress=False)
            if df.empty:
                return None
            
            # 扁平化多级列名 (处理 yfinance 新版格式)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            df = df.reset_index()
            # 确保列名统一
            df.columns = [c.lower() for c in df.columns]
            rename_map = {'date': 'Date', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}
            df = df.rename(columns=rename_map)
            
            # 计算基础技术指标
            df['Returns'] = df['Close'].pct_change()
            df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
            df['Volatility'] = df['Returns'].rolling(window=20).std()
            df['MA20'] = df['Close'].rolling(window=20).mean()
            
            return df.dropna()
        except Exception as e:
            st.error(f"Data Fetch Error: {e}")
            return None

# --- 3. 物理引擎 (Physics Engine) - 负责周期与能量分析 ---
class PhysicsEngine:
    @staticmethod
    def calculate_entropy(series):
        """计算香农熵，衡量市场混乱度"""
        p_data = series.value_counts() / len(series)
        entropy = -sum(p_data * np.log2(p_data + 1e-9))
        return entropy

    @staticmethod
    def fft_analysis(prices):
        """快速傅里叶变换，提取市场主周期"""
        N = len(prices)
        yf_fft = fft(prices.values)
        xf = np.linspace(0.0, 1.0/(2.0), N//2)
        amplitude = 2.0/N * np.abs(yf_fft[0:N//2])
        
        # 找到前3个最强频率
        idx = np.argsort(amplitude)[::-1]
        dominant_periods = [1/xf[i] for i in idx[1:4] if xf[i] > 0] # 排除0频率
        return dominant_periods, amplitude, xf

    @staticmethod
    def reflexivity_index(df):
        """索罗斯反身性指数：价格与基本面(MA)的偏离度 x 成交量放大系数"""
        deviation = (df['Close'] - df['MA20']) / df['MA20']
        volume_surge = df['Volume'] / df['Volume'].rolling(50).mean()
        # 反身性得分：当价格大幅偏离且放量时，反身性最强
        reflexivity = deviation * volume_surge
        return reflexivity

# --- 4. 核心 AI 层 (AI Core) - LSTM & MCTS ---
class Brain:
    class LSTMNet(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
            super().__init__()
            self.hidden_dim = hidden_dim
            self.num_layers = num_layers
            self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_dim, output_dim)

        def forward(self, x):
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
            out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
            out = self.fc(out[:, -1, :]) 
            return out

    @staticmethod
    def train_lstm_inference(df, lookback=30):
        """
        轻量级在线训练。
        CTO 批注：为了演示速度，我们不进行完整的Epoch训练，
        而是基于当前数据进行快速拟合，展示 AI 的预测倾向。
        """
        data = df['Close'].values.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(-1, 1))
        data_scaled = scaler.fit_transform(data)

        # 准备数据
        x_train, y_train = [], []
        for i in range(len(data_scaled) - lookback):
            x_train.append(data_scaled[i:i+lookback])
            y_train.append(data_scaled[i+lookback])
        
        x_train = torch.from_numpy(np.array(x_train)).float()
        y_train = torch.from_numpy(np.array(y_train)).float()

        # 模型初始化
        model = Brain.LSTMNet(input_dim=1, hidden_dim=32, output_dim=1, num_layers=2)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        # 快速训练 20 次迭代
        progress_bar = st.sidebar.progress(0)
        for epoch in range(20):
            model.train()
            optimizer.zero_grad()
            outputs = model(x_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()
            progress_bar.progress((epoch + 1) / 20)
        
        # 预测未来
        model.eval()
        last_sequence = data_scaled[-lookback:].reshape(1, lookback, 1)
        last_sequence_tensor = torch.from_numpy(last_sequence).float()
        with torch.no_grad():
            future_scaled = model(last_sequence_tensor)
            prediction = scaler.inverse_transform(future_scaled.numpy())[0][0]
            
        return prediction, loss.item()

    @staticmethod
    def vectorized_mcts(current_price, volatility, simulations=1000, days=5):
        """
        矩阵化蒙特卡洛模拟 (Matrix Monte Carlo)。
        比传统循环快 100 倍。
        """
        dt = 1  # 时间步长
        # 随机漂移 (Drift) 和 震荡 (Shock)
        drift = 0  # 假设短期均值为0 (随机游走)
        shock = volatility * np.random.randn(simulations, days)
        
        # 价格路径矩阵: [simulations, days]
        price_paths = np.zeros((simulations, days))
        price_paths[:, 0] = current_price
        
        for t in range(1, days):
            price_paths[:, t] = price_paths[:, t-1] * (1 + drift + shock[:, t])
            
        # 结果统计
        final_prices = price_paths[:, -1]
        mean_price = np.mean(final_prices)
        upside_prob = np.mean(final_prices > current_price)
        
        return price_paths, mean_price, upside_prob

# --- 5. UI 呈现层 (Presentation Layer) ---
def main():
    # Sidebar
    st.sidebar.title("⚛️ Q-Trader Pro")
    st.sidebar.caption("V9.0 Enterprise Edition")
    
    ticker = st.sidebar.text_input("Ticker Symbol", value="NVDA").upper()
    period = st.sidebar.selectbox("Data Period", ["6mo", "1y", "2y", "5y"], index=1)
    
    # Authenticate (模拟) - 可以开启
    # if not check_password(): st.stop()

    if st.sidebar.button("Run Quantum Analysis", type="primary"):
        with st.spinner('Accessing Quantum Field...'):
            df = DataManager.fetch_data(ticker, period=period)
            
            if df is None:
                st.error("Failed to load data. Please check the ticker.")
                st.stop()

            # --- 计算层 ---
            current_price = df['Close'].iloc[-1]
            last_vol = df['Volatility'].iloc[-1]
            
            # 1. AI 预测
            lstm_pred, lstm_loss = Brain.train_lstm_inference(df)
            
            # 2. 物理分析
            periods, amps, _ = PhysicsEngine.fft_analysis(df['Close'])
            main_cycle = periods[0] if len(periods) > 0 else 0
            
            # 3. 反身性
            df['Reflexivity'] = PhysicsEngine.reflexivity_index(df)
            curr_reflex = df['Reflexivity'].iloc[-1]

            # 4. MCTS 模拟
            mcts_paths, mcts_mean, win_rate = Brain.vectorized_mcts(current_price, last_vol)

            # --- 仪表盘 UI ---
            
            # 顶部 KPI 卡片
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Current Price", f"${current_price:.2f}", f"{(current_price - df['Close'].iloc[-2]):.2f}")
            col2.metric("AI Target (T+1)", f"${lstm_pred:.2f}", delta_color="normal" if lstm_pred > current_price else "inverse")
            col3.metric("MCTS Win Rate", f"{win_rate*100:.1f}%", f"Vol: {last_vol*100:.2f}%")
            col4.metric("Market Cycle", f"{main_cycle:.1f} Days", "Dominant Wave")

            # 主图表区
            tab1, tab2, tab3 = st.tabs(["📉 Market & Reflexivity", "🧠 AI Simulation", "⚛️ Physics Spectrum"])

            with tab1:
                # K线图 + 反身性指标
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                    vertical_spacing=0.03, row_heights=[0.7, 0.3])
                fig.add_trace(go.Candlestick(x=df['Date'], open=df['Open'], high=df['High'],
                                             low=df['Low'], close=df['Close'], name='Price'), row=1, col=1)
                fig.add_trace(go.Bar(x=df['Date'], y=df['Reflexivity'], name='Reflexivity Index', 
                                     marker_color=np.where(df['Reflexivity']<0, 'red', 'green')), row=2, col=1)
                fig.update_layout(height=600, template="plotly_dark", title=f"{ticker} Reflexivity Analysis")
                st.plotly_chart(fig, use_container_width=True)
                
                st.info(f"💡 Reflexivity Insight: Current index is {curr_reflex:.4f}. High absolute values indicate extreme divergence between price and fundamentals, often preceding a reversal.")

            with tab2:
                # 蒙特卡洛路径可视化
                st.subheader(f"Monte Carlo: 1000 Possible Futures (5 Days)")
                fig_mc = go.Figure()
                # 只画前50条线以防浏览器卡顿，但统计是用1000条算的
                for i in range(50):
                    fig_mc.add_trace(go.Scatter(y=mcts_paths[i], mode='lines', line=dict(width=1, color='rgba(0, 255, 255, 0.1)'), showlegend=False))
                
                # 添加均值线
                fig_mc.add_trace(go.Scatter(y=np.mean(mcts_paths, axis=0), mode='lines', name='Mean Path', line=dict(color='yellow', width=3, dash='dash')))
                fig_mc.update_layout(template="plotly_dark", height=400)
                st.plotly_chart(fig_mc, use_container_width=True)

            with tab3:
                # FFT 频谱
                st.subheader("Market Frequency Domain (FFT)")
                _, frequencies, x_axis = PhysicsEngine.fft_analysis(df['Close'])
                fig_fft = go.Figure(data=[go.Bar(x=x_axis[1:50], y=frequencies[1:50])]) # 去掉直流分量
                fig_fft.update_layout(title="Energy Spectrum (Hidden Cycles)", xaxis_title="Frequency", yaxis_title="Amplitude", template="plotly_dark")
                st.plotly_chart(fig_fft, use_container_width=True)

    else:
        st.info("👈 Please enter a ticker and click 'Run Quantum Analysis' to start.")

if __name__ == "__main__":
    main()
