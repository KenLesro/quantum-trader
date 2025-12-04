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

# --- 1. 全局配置与工具类 (Global Config) ---
st.set_page_config(
    page_title="Quantum Trader Pro V9",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 设置专业金融终端的深色样式
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
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. 数据层 (Data Layer) - 核心修复版 ---
class DataManager:
    @staticmethod
    @st.cache_data(ttl=900)  # 缓存 15 分钟
    def fetch_data(ticker, period="1y", interval="1d"):
        try:
            # 修复核心：改用 Ticker.history，这对云端 IP 更友好，不易报错
            stock = yf.Ticker(ticker)
            df = stock.history(period=period, interval=interval)
            
            if df.empty:
                return None
            
            # 数据清洗：重置索引，让 Date 变成一列
            df = df.reset_index()
            
            # 修复核心：处理时区问题 (TZ-aware to TZ-naive)
            # 很多报错是因为 plotly 无法处理带时区的时间
            if 'Date' in df.columns and pd.api.types.is_datetime64_any_dtype(df['Date']):
                if df['Date'].dt.tz is not None:
                    df['Date'] = df['Date'].dt.tz_localize(None)
            
            # 统一列名 (Yahoo 有时返回 Open, 有时返回 open)
            df.columns = [c.capitalize() for c in df.columns]
            
            # 容错处理：确保关键列存在
            required = ['Close', 'Volume', 'High', 'Low', 'Open']
            for col in required:
                if col not in df.columns:
                    # 如果找不到 Close，尝试找 close
                    if col.lower() in df.columns:
                        df[col] = df[col.lower()]
            
            # 计算基础指标
            df['Returns'] = df['Close'].pct_change()
            df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
            df['Volatility'] = df['Returns'].rolling(window=20).std()
            df['MA20'] = df['Close'].rolling(window=20).mean()
            
            return df.dropna()
        except Exception as e:
            st.error(f"Data Engine Error: {e}")
            return None

# --- 3. 物理引擎 (Physics Engine) ---
class PhysicsEngine:
    @staticmethod
    def fft_analysis(prices):
        try:
            N = len(prices)
            # 转换为 numpy 数组防止索引问题
            price_data = prices.values
            yf_fft = fft(price_data)
            xf = np.linspace(0.0, 1.0/(2.0), N//2)
            amplitude = 2.0/N * np.abs(yf_fft[0:N//2])
            
            # 提取主周期
            idx = np.argsort(amplitude)[::-1]
            dominant_periods = []
            for i in idx:
                if xf[i] > 0: # 排除直流分量
                    period = 1/xf[i]
                    if period < N/2: # 排除过长周期
                        dominant_periods.append(period)
                if len(dominant_periods) >= 3: break
            
            if not dominant_periods: dominant_periods = [0]
            return dominant_periods, amplitude, xf
        except:
            return [0], [], []

    @staticmethod
    def reflexivity_index(df):
        try:
            # 索罗斯反身性：价格偏离度 * 成交量放大倍数
            deviation = (df['Close'] - df['MA20']) / df['MA20']
            vol_mean = df['Volume'].rolling(50).mean().replace(0, 1) # 防止除以0
            volume_surge = df['Volume'] / vol_mean
            reflexivity = deviation * volume_surge
            return reflexivity
        except:
            return pd.Series(0, index=df.index)

# --- 4. AI 大脑 (AI Brain) ---
class Brain:
    class LSTMNet(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
            super().__init__()
            self.hidden_dim = hidden_dim
            self.num_layers = num_layers
            self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_dim, output_dim)

        def forward(self, x):
            # 初始化隐状态
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
            out, _ = self.lstm(x, (h0.detach(), c0.detach()))
            out = self.fc(out[:, -1, :]) 
            return out

    @staticmethod
    def train_lstm_inference(df, lookback=30):
        try:
            # 数据归一化 (Normalization) - 这一步对神经网络至关重要
            data = df['Close'].values.reshape(-1, 1)
            scaler = MinMaxScaler(feature_range=(-1, 1))
            data_scaled = scaler.fit_transform(data)

            x_train, y_train = [], []
            for i in range(len(data_scaled) - lookback):
                x_train.append(data_scaled[i:i+lookback])
                y_train.append(data_scaled[i+lookback])
            
            if not x_train: return df['Close'].iloc[-1], 0.0

            x_train = torch.from_numpy(np.array(x_train)).float()
            y_train = torch.from_numpy(np.array(y_train)).float()

            model = Brain.LSTMNet(input_dim=1, hidden_dim=32, output_dim=1, num_layers=2)
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

            # 快速训练 (Micro-Training)
            for epoch in range(15):
                model.train()
                optimizer.zero_grad()
                outputs = model(x_train)
                loss = criterion(outputs, y_train)
                loss.backward()
                optimizer.step()
            
            # 预测 T+1
            model.eval()
            last_sequence = data_scaled[-lookback:].reshape(1, lookback, 1)
            with torch.no_grad():
                future_scaled = model(torch.from_numpy(last_sequence).float())
                prediction = scaler.inverse_transform(future_scaled.numpy())[0][0]
                
            return prediction, loss.item()
        except Exception as e:
            st.error(f"AI Prediction Error: {e}")
            return df['Close'].iloc[-1], 0.0

    @staticmethod
    def vectorized_mcts(current_price, volatility, simulations=1000, days=5):
        try:
            # 矩阵化蒙特卡洛：一次计算 1000x5 的矩阵，极大提升速度
            if np.isnan(volatility) or volatility == 0: volatility = 0.02
            
            dt = 1
            drift = 0
            # 核心：生成随机矩阵
            shock = volatility * np.random.randn(simulations, days)
            
            price_paths = np.zeros((simulations, days))
            price_paths[:, 0] = current_price
            
            for t in range(1, days):
                price_paths[:, t] = price_paths[:, t-1] * (1 + drift + shock[:, t])
                
            final_prices = price_paths[:, -1]
            mean_price = np.mean(final_prices)
            # 计算上涨概率
            upside_prob = np.mean(final_prices > current_price)
            
            return price_paths, mean_price, upside_prob
        except:
            # 兜底返回
            return np.zeros((simulations, days)), current_price, 0.5

# --- 5. 前端 UI (User Interface) ---
def main():
    st.sidebar.title("⚛️ Q-Trader Pro")
    st.sidebar.caption("V9.1 Stable Edition")
    
    # 输入区域
    ticker = st.sidebar.text_input("Ticker Symbol", value="NVDA").upper()
    period = st.sidebar.selectbox("Data Period", ["6mo", "1y", "2y", "5y"], index=1)
    
    # 运行按钮
    if st.sidebar.button("Run Quantum Analysis", type="primary"):
        with st.spinner('Initializing Quantum Core...'):
            # 1. 获取数据
            df = DataManager.fetch_data(ticker, period=period)
            
            if df is None:
                st.error(f"⚠️ 无法获取 {ticker} 数据。请检查代码是否正确，或稍后再试。")
                st.stop()

            # 2. 核心计算
            current_price = df['Close'].iloc[-1]
            last_vol = df['Volatility'].iloc[-1]
            prev_price = df['Close'].iloc[-2]
            
            # AI & 物理引擎计算
            lstm_pred, lstm_loss = Brain.train_lstm_inference(df)
            periods, amps, x_axis = PhysicsEngine.fft_analysis(df['Close'])
            main_cycle = periods[0] if len(periods) > 0 else 0
            
            df['Reflexivity'] = PhysicsEngine.reflexivity_index(df)
            curr_reflex = df['Reflexivity'].iloc[-1]

            mcts_paths, mcts_mean, win_rate = Brain.vectorized_mcts(current_price, last_vol)

            # 3. 结果展示
            # 顶部指标卡
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Current Price", f"${current_price:.2f}", f"{current_price - prev_price:.2f}")
            col2.metric("AI Target (T+1)", f"${lstm_pred:.2f}", delta_color="normal" if lstm_pred > current_price else "inverse")
            col3.metric("MCTS Win Rate", f"{win_rate*100:.1f}%", f"Vol: {last_vol*100:.2f}%")
            col4.metric("Market Cycle", f"{main_cycle:.1f} Days", "Dominant Wave")

            # 选项卡视图
            tab1, tab2, tab3 = st.tabs(["📉 Reflexivity", "🧠 AI Simulation", "⚛️ Spectrum"])

            with tab1:
                # K线图 + 反身性
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
                fig.add_trace(go.Candlestick(x=df['Date'], open=df['Open'], high=df['High'],
                                             low=df['Low'], close=df['Close'], name='Price'), row=1, col=1)
                fig.add_trace(go.Bar(x=df['Date'], y=df['Reflexivity'], name='Reflexivity Index', 
                                     marker_color=np.where(df['Reflexivity']<0, 'red', 'green')), row=2, col=1)
                fig.update_layout(height=600, template="plotly_dark", title=f"{ticker} Reflexivity Analysis")
                st.plotly_chart(fig, use_container_width=True)

            with tab2:
                # 蒙特卡洛路径
                fig_mc = go.Figure()
                # 仅绘制前 50 条路径以优化性能
                for i in range(min(50, len(mcts_paths))):
                    fig_mc.add_trace(go.Scatter(y=mcts_paths[i], mode='lines', line=dict(width=1, color='rgba(0, 255, 255, 0.1)'), showlegend=False))
                fig_mc.add_trace(go.Scatter(y=np.mean(mcts_paths, axis=0), mode='lines', name='Mean Path', line=dict(color='yellow', width=3, dash='dash')))
                fig_mc.update_layout(template="plotly_dark", height=400, title="Monte Carlo Simulation (1000 Scenarios)")
                st.plotly_chart(fig_mc, use_container_width=True)

            with tab3:
                # FFT 频谱
                if len(x_axis) > 0:
                    fig_fft = go.Figure(data=[go.Bar(x=x_axis[1:50], y=amps[1:50])])
                    fig_fft.update_layout(title="Energy Spectrum (Hidden Cycles)", xaxis_title="Frequency", yaxis_title="Amplitude", template="plotly_dark")
                    st.plotly_chart(fig_fft, use_container_width=True)
                else:
                    st.write("Not enough data for FFT.")

    else:
        st.info("👈 Please enter a stock ticker (e.g., AAPL) and click Run.")

if __name__ == "__main__":
    main()
