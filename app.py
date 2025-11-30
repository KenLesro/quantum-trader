import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
import math

# ==========================================
# 核心算法：反身性 MCTS (Reflexivity MCTS)
# ==========================================
class ReflexivityMCTS:
    def __init__(self, simulations=1000, horizon=5):
        self.simulations = simulations
        self.horizon = horizon

    def simulate(self, current_price, current_vol, avg_vol, base_sigma):
        future_outcomes = []
        rvol_start = current_vol / (avg_vol + 1e-9)
        drift = 0.0 

        for _ in range(self.simulations):
            price = current_price
            vol = current_vol
            rvol = rvol_start
            sentiment = 0.0
            
            for _ in range(self.horizon):
                # 动态波动率
                dynamic_sigma = base_sigma * (1 + 0.3 * np.log1p(rvol))
                if dynamic_sigma < 0.01: dynamic_sigma = 0.01
                
                # 非线性放大器
                amplifier = np.power(rvol, 1.8) if rvol > 1.0 else rvol
                
                # 反身性反馈
                feedback_impact = np.tanh(sentiment) * 0.02 * amplifier
                
                # 随机冲击
                shock = np.random.normal(drift, dynamic_sigma)
                ret = shock + feedback_impact
                
                price = price * (1 + ret)
                
                # 闭环演化
                vol = vol * (1 + abs(ret) * 5.0)
                rvol = vol / (avg_vol + 1e-9)
                sentiment_delta = np.sign(ret) * (abs(ret) * 10.0 * amplifier)
                sentiment = sentiment * 0.9 + sentiment_delta
            
            future_outcomes.append(price)
            
        future_outcomes = np.array(future_outcomes)
        win_rate = np.mean(future_outcomes > current_price)
        expected_price = np.mean(future_outcomes)
        sorted_prices = np.sort(future_outcomes)
        var_95_price = sorted_prices[int(self.simulations * 0.05)]
        
        return {
            'win_rate': win_rate,
            'expected_price': expected_price,
            'rvol': rvol_start,
            'var_95_price': var_95_price,
            'simulations': future_outcomes
        }

# ==========================================
# 数据引擎
# ==========================================
@st.cache_data(ttl=300)
def get_market_data(symbol, period='1y'):
    try:
        stock = yf.Ticker(symbol)
        df = stock.history(period=period)
        if df.empty: return None
        df['MA_Volume'] = df['Volume'].rolling(window=20).mean()
        df['Volatility'] = df['Close'].pct_change().rolling(window=20).std()
        return df.dropna()
    except:
        return None

# ==========================================
# GUI 界面
# ==========================================
st.set_page_config(page_title="Quantum Trader Pro", layout="wide", page_icon="⚡")

st.markdown("""
    <style>
    .stApp {background-color: #0e1117;}
    .metric-card {background-color: #262730; padding: 15px; border-radius: 10px; border-left: 5px solid #4CAF50;}
    .warning-card {background-color: rgba(255, 75, 75, 0.1); padding: 15px; border-radius: 10px; border-left: 5px solid #FF4B4B;}
    </style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("⚡ 量子控制台")
    symbol = st.text_input("股票代码", "NVDA").upper()
    sim_count = st.slider("MCTS 模拟次数", 500, 5000, 1000)
    horizon = st.slider("预测视野 (天)", 1, 10, 5)
    run_btn = st.button("🚀 启动分析", type="primary")

st.title(f"📊 Quantum Trader Pro: {symbol}")

if run_btn:
    with st.spinner(f"正在建立 {symbol} 的反身性反馈模型..."):
        df = get_market_data(symbol)
        
    if df is None:
        st.error(f"❌ 无法获取 {symbol} 数据，请检查代码。")
    else:
        last_row = df.iloc[-1]
        res = ReflexivityMCTS(sim_count, horizon).simulate(
            last_row['Close'], last_row['Volume'], last_row['MA_Volume'], last_row['Volatility'] if not np.isnan(last_row['Volatility']) else 0.02
        )
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("当前价格", f"${last_row['Close']:.2f}")
        col2.metric("RVOL", f"{res['rvol']:.2f}x", "🔥 放量" if res['rvol']>1.2 else "正常")
        col3.metric("上涨概率", f"{res['win_rate']*100:.1f}%", f"目标 ${res['expected_price']:.2f}")
        col4.metric("下行风险", f"${res['var_95_price']:.2f}")

        c1, c2 = st.columns([2, 1])
        with c1:
            st.subheader("🧬 市场微观分析")
            if res['rvol'] > 1.5:
                st.markdown(f"<div class='warning-card'><h4>⚠️ 极度拥挤 (RVOL {res['rvol']:.1f}x)</h4><p>情绪被放大，谨防逼空或踩踏。</p></div>", unsafe_allow_html=True)
            else:
                st.info("✅ 市场情绪平稳，反身性效应微弱。")
            
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=res['simulations'], nbinsx=50, marker_color='#00CC96'))
            fig.update_layout(title="未来价格概率分布", height=300, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
            st.plotly_chart(fig, use_container_width=True)
            
        with c2:
            st.subheader("📈 历史走势")
            st.line_chart(df['Close'].tail(60))
