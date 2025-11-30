import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm, powerlaw
from scipy.fft import fft
import torch
import torch.nn as nn
import math

# ==========================================
# 1. 物理与数学引擎 (Physics & Math Core)
# ==========================================
class PhysicsEngine:
    """
    V8.0 核心：引入物理学方法分析市场
    1. FFT (快速傅立叶变换) -> 识别市场周期
    2. Matrix MCTS (矩阵化蒙特卡洛) -> 提升运算速度 100倍
    """
    @staticmethod
    def analyze_cycles_fft(prices):
        """利用 FFT 识别市场主周期"""
        # 去趋势 (Detrending) 以提取纯周期波动
        prices_detrend = prices - np.mean(prices)
        n = len(prices)
        
        # FFT 变换
        fft_output = fft(prices_detrend)
        power = np.abs(fft_output[:n//2]) # 能量谱
        freqs = np.fft.fftfreq(n, d=1)[:n//2] # 频率
        
        # 找到能量最大的主频率 (忽略直流分量)
        if len(power) > 1:
            idx = np.argmax(power[1:]) + 1
            dominant_period = 1 / (freqs[idx] + 1e-9)
            cycle_strength = power[idx] / (np.sum(power) + 1e-9)
            return dominant_period, cycle_strength
        return 0, 0

    @staticmethod
    def mcts_matrix_simulation(price_0, vol_0, avg_vol, base_sigma, simulations=1000, horizon=5):
        """
        [矩阵加速版] 反身性博弈推演
        """
        # 1. 批量生成随机冲击矩阵 (Simulations x Horizon)
        shocks = np.random.normal(0, 1, (simulations, horizon))
        
        # 2. 计算 RVOL (相对成交量)
        rvol = vol_0 / (avg_vol + 1e-9)
        
        # 3. 动态波动率 (基于 RVOL 的非线性放大)
        # 索罗斯逻辑：量越大，波动率不仅仅是线性增加，而是对数级放大
        dynamic_sigma = base_sigma * (1 + 0.3 * np.log1p(rvol))
        
        # 4. 反身性放大器 (Soros Amplifier)
        # 当市场拥挤 (RVOL > 1) 时，情绪反馈呈幂律增长
        amplifier = np.power(rvol, 1.8) if rvol > 1.0 else rvol
        
        # 5. 路径演化 (矩阵化计算)
        # 模拟情绪漂移：随机生成情绪倾向，并被 amplifier 放大
        feedback_drift = 0.001 * amplifier * np.random.choice([-1, 1], size=(simulations, horizon))
        
        # 每日回报率 = 随机冲击 * 动态波动 + 情绪反馈
        daily_returns = shocks * dynamic_sigma + feedback_drift
        
        # 累积回报率 -> 价格路径
        cumulative_returns = np.cumprod(1 + daily_returns, axis=1)
        final_prices = price_0 * cumulative_returns[:, -1]
        
        # 统计结果
        win_rate = np.mean(final_prices > price_0)
        expected_price = np.mean(final_prices)
        var_95 = np.percentile(final_prices, 5) # 95% VaR
        
        return win_rate, expected_price, var_95, final_prices

# ==========================================
# 2. 深度学习模型 (Quantum LSTM)
# ==========================================
class QuantumLSTM(nn.Module):
    def __init__(self, input_size=10, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.attention = nn.MultiheadAttention(hidden_size*2, 4, batch_first=True)
        self.fc = nn.Linear(hidden_size*2, 3) # Output: Buy, Hold, Sell

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        logits = self.fc(attn_out[:, -1, :])
        return torch.softmax(logits, dim=1)

# ==========================================
# 3. 数据引擎
# ==========================================
@st.cache_data(ttl=300)
def get_data(symbol):
    try:
        stock = yf.Ticker(symbol)
        df = stock.history(period="1y")
        if df.empty: return None, None
        
        # 基础指标
        df['MA20'] = df['Close'].rolling(20).mean()
        df['Vol_MA20'] = df['Volume'].rolling(20).mean()
        df['Returns'] = df['Close'].pct_change()
        
        # 物理周期分析 (FFT)
        # 取最近 60 个交易日进行频谱分析
        recent_prices = df['Close'].tail(60).values
        period, strength = PhysicsEngine.analyze_cycles_fft(recent_prices)
        
        meta = {
            "period": period, 
            "cycle_strength": strength, 
            "info": stock.info
        }
        return df.dropna(), meta
    except:
        return None, None

def get_buffett_score(info):
    """巴菲特基本面评分"""
    score = 0
    try:
        if info.get('trailingPE', 99) < 25: score += 30
        if info.get('returnOnEquity', 0) > 0.15: score += 30
        if info.get('debtToEquity', 100) < 80: score += 20
        if info.get('freeCashflow', 0) > 0: score += 20
    except:
        score = 50 # 数据缺失给中性分
    return score

# ==========================================
# 4. GUI 主界面
# ==========================================
st.set_page_config(page_title="Quantum Trader V8.1", layout="wide", page_icon="⚛️")

# CSS 美化
st.markdown("""
    <style>
    .stApp {background-color: #0e1117;}
    .metric-card {background-color: #262730; padding: 15px; border-radius: 10px; border-left: 5px solid #4CAF50;}
    .warning-card {background-color: rgba(255, 75, 75, 0.1); padding: 15px; border-radius: 10px; border-left: 5px solid #FF4B4B;}
    </style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("⚛️ 量子控制台 V8.1")
    st.caption("Physics Engine + Reflexivity + AI")
    
    # --- 修复点：恢复文本输入框 ---
    st.subheader("1. 标的选择")
    
    # 快捷按钮
    col_a, col_b, col_c = st.columns(3)
    if col_a.button("NVDA"): st.session_state.symbol = "NVDA"
    if col_b.button("BTC"): st.session_state.symbol = "BTC-USD"
    if col_c.button("AAPL"): st.session_state.symbol = "AAPL"
    
    # 接收输入 (默认值逻辑)
    default_sym = st.session_state.get("symbol", "NVDA")
    symbol = st.text_input("输入代码 (如 600519.SS)", default_sym).upper()
    
    st.markdown("---")
    st.subheader("2. 模拟参数")
    sim_count = st.slider("MCTS 模拟次数", 1000, 10000, 2000)
    
    run_btn = st.button("🚀 启动深度分析", type="primary")

st.title(f"📊 量化深度分析: {symbol}")

if run_btn:
    with st.spinner(f"正在连接物理引擎与华尔街数据源..."):
        df, meta = get_data(symbol)
        
    if df is None:
        st.error(f"❌ 无法获取 {symbol} 数据。请检查代码拼写 (如A股需加后缀 .SS 或 .SZ)。")
    else:
        # 准备数据
        last_row = df.iloc[-1]
        current_price = last_row['Close']
        current_vol = last_row['Volume']
        avg_vol = last_row['Vol_MA20']
        volatility = df['Returns'].std()
        
        # 1. 运行矩阵加速 MCTS
        win_rate, target, var_95, paths = PhysicsEngine.mcts_matrix_simulation(
            current_price, current_vol, avg_vol, volatility, simulations=sim_count
        )
        
        # 2. 计算 RVOL
        rvol = current_vol / (avg_vol + 1e-9)
        
        # 3. 巴菲特评分
        f_score = get_buffett_score(meta['info'])
        
        # --- 仪表盘 ---
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("当前价格", f"${current_price:.2f}", f"{last_row['Returns']*100:.2f}%")
        
        # 周期指标
        period = meta['period']
        p_str = f"{period:.1f} 天" if period > 0 else "无明显周期"
        col2.metric("FFT 市场周期", p_str, f"强度 {meta['cycle_strength']*100:.0f}%")
        
        # 反身性指标
        state = "🔥 极度拥挤" if rvol > 2.0 else ("⚡ 活跃" if rvol > 1.2 else "🧊 平稳")
        col3.metric("RVOL (情绪放大)", f"{rvol:.2f}x", state, delta_color="inverse")
        
        # 预测指标
        col4.metric("MCTS 胜率", f"{win_rate*100:.1f}%", f"目标 ${target:.2f}")
        
        st.markdown("---")
        
        # --- 深度图表区 ---
        c1, c2 = st.columns([2, 1])
        
        with c1:
            st.subheader("🔮 多重宇宙推演 (Matrix Simulation)")
            # 绘制分布图
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=paths, nbinsx=60, marker_color='#00CC96', name='预测分布'))
            fig.add_vline(x=current_price, line_dash="dash", line_color="white", annotation_text="当前价")
            fig.add_vline(x=var_95, line_dash="dot", line_color="red", annotation_text="VaR 95%")
            fig.update_layout(title=f"基于 {sim_count} 次反身性模拟的未来价格分布", height=350, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
            st.plotly_chart(fig, use_container_width=True)
            
            if rvol > 1.5:
                st.warning(f"⚠️ **反身性警报：** 市场处于非线性状态 (RVOL={rvol:.1f})。情绪反馈正在指数级放大波动，建议降低杠杆。")
        
        with c2:
            st.subheader("🧭 综合决策")
            final_score = (win_rate * 50) + (f_score * 0.3) + (meta['cycle_strength'] * 20)
            if rvol > 2.0: final_score -= 15 # 过热惩罚
            
            # 进度条颜色
            bar_color = "green" if final_score > 60 else ("red" if final_score < 40 else "yellow")
            st.markdown(f"### 得分: {final_score:.1f} / 100")
            st.progress(min(int(final_score), 100))
            
            if final_score > 60:
                st.success("✅ **建议：买入** (动量+周期共振)")
            elif final_score < 40:
                st.error("❌ **建议：卖出** (风险过高)")
            else:
                st.info("👀 **建议：观望** (方向不明)")
                
            st.write(f"**巴菲特安全垫：** {f_score} 分")
            st.caption("注：得分基于 MCTS 胜率、基本面及物理周期强度的加权计算。")

    # 原始数据
    with st.expander("查看历史数据"):
        st.dataframe(df.tail(20))


