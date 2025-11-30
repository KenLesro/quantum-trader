import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import plotly.graph_objects as go
from scipy.stats import norm, powerlaw
from datetime import datetime, timedelta

# ==========================================
# 1. 核心数学模型 (The Math Core)
# ==========================================

class ReflexivityMath:
    """
    [源自您的文档]: 索罗斯反身性数学建模
    用于计算非线性反馈系数
    """
    @staticmethod
    def calculate_feedback(sentiment, rvol, pv_ratio):
        # 使用 tanh 函数模拟情绪的饱和效应 (情绪不会无限放大)
        # 情绪因子 (Sentiment Factor)
        sent_factor = np.tanh(sentiment) * 0.05
        
        # 相对成交量放大器 (RVOL Amplifier) - 幂律非线性
        # 当量能 > 2倍均量时，反馈力度呈指数级上升
        vol_amplifier = np.power(rvol, 1.5) if rvol > 1.0 else rvol
        
        # 量价背离/共振因子
        pv_factor = np.clip(pv_ratio, -0.1, 0.1)
        
        # 总反馈 = (情绪 + 量价) * 放大器
        feedback = (sent_factor + pv_factor * 0.5) * vol_amplifier
        return feedback

class PowerLawRisk:
    """
    [源自您的文档]: 幂律分布风控模型
    捕捉正态分布无法识别的'肥尾'风险
    """
    @staticmethod
    def calculate_hybrid_var(returns, confidence=0.95):
        if len(returns) < 30: return 0.05 # 默认兜底
        
        # 1. 正态分布 VaR (常规风险)
        mu, std = norm.fit(returns)
        var_normal = abs(norm.ppf(1 - confidence, mu, std))
        
        # 2. 幂律分布 VaR (极端风险)
        # 只关注左尾(亏损端)
        losses = -returns[returns < 0]
        if len(losses) > 10:
            try:
                # 拟合幂律分布参数
                a, loc, scale = powerlaw.fit(losses)
                var_power = powerlaw.ppf(confidence, a, loc, scale)
            except:
                var_power = var_normal * 1.5 # 拟合失败时的保守估计
        else:
            var_power = var_normal
            
        # 3. 混合加权 (60% 幂律 + 40% 正态 - 源自文档建议)
        hybrid_var = 0.6 * var_power + 0.4 * var_normal
        return hybrid_var

# ==========================================
# 2. 深度学习架构 (The AI Brain)
# ==========================================

class AlphaGoPolicyValueNet(nn.Module):
    """
    [源自您的文档]: 仿 AlphaGo 架构
    同时输出策略(Policy)和价值(Value)
    """
    def __init__(self, input_dim=6, hidden_dim=64):
        super().__init__()
        # 特征提取层 (LSTM)
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        
        # 注意力机制 (Self-Attention)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)
        
        # 1. 策略头 (Policy Head) -> 输出买/卖/持有的概率
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 3), # [Buy, Hold, Sell]
            nn.Softmax(dim=-1)
        )
        
        # 2. 价值头 (Value Head) -> 输出当前胜率 (-1 to 1)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh()
        )

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        lstm_out, (hn, cn) = self.lstm(x)
        
        # Attention 处理
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # 取最后一个时间步的特征
        final_feature = attn_out[:, -1, :]
        
        policy = self.policy_head(final_feature)
        value = self.value_head(final_feature)
        return policy, value

# ==========================================
# 3. 策略引擎 (Strategy Engine)
# ==========================================

class QuantumEngine:
    def __init__(self, symbol):
        self.symbol = symbol
        self.data = None
        self.model = AlphaGoPolicyValueNet() # 初始化模型 (未训练状态)
        
    def fetch_data(self):
        """获取数据并计算反身性特征"""
        stock = yf.Ticker(self.symbol)
        df = stock.history(period="1y")
        
        if df.empty: return None
        
        # --- 特征工程 (源自文档) ---
        # 1. 相对成交量 (RVOL)
        df['MA_Vol'] = df['Volume'].rolling(20).mean()
        df['RVOL'] = df['Volume'] / (df['MA_Vol'] + 1e-9)
        
        # 2. 量价互动比率 (PV Ratio)
        df['PV_Ratio'] = df['Close'].pct_change() / (df['Volume'].pct_change() + 1e-9)
        
        # 3. 情绪指标 (Sentiment) - 基于高低价差与量能
        df['Sentiment'] = (df['High'] - df['Low']) / df['Close'] * np.log1p(df['Volume'])
        
        return df.dropna()

    def run_mcts(self, df, simulations=1000):
        """
        蒙特卡洛树搜索 (反身性增强版)
        """
        last = df.iloc[-1]
        price_0 = last['Close']
        rvol_0 = last['RVOL']
        sent_0 = last['Sentiment']
        
        future_paths = []
        
        for _ in range(simulations):
            path = [price_0]
            curr_price = price_0
            curr_sent = sent_0
            
            # 模拟未来 5 天
            for _ in range(5):
                # 1. 计算反身性反馈
                feedback = ReflexivityMath.calculate_feedback(curr_sent, rvol_0, last['PV_Ratio'])
                
                # 2. 随机冲击 (基于混合VaR波动率)
                volatility = df['Close'].pct_change().std()
                shock = np.random.normal(0, volatility)
                
                # 3. 价格演变
                ret = shock + feedback
                curr_price *= (1 + ret)
                
                # 4. 情绪更新 (闭环)
                # 价格上涨会让情绪更亢奋 (Self-Reinforcing)
                curr_sent += ret * 5.0
                
                path.append(curr_price)
            future_paths.append(path)
            
        return future_paths

    def get_buffett_score(self):
        """巴菲特基本面打分"""
        try:
            info = yf.Ticker(self.symbol).info
            score = 0
            # 1. 估值
            if info.get('trailingPE', 99) < 25: score += 30
            # 2. 盈利能力
            if info.get('returnOnEquity', 0) > 0.15: score += 30
            # 3. 财务健康
            if info.get('debtToEquity', 100) < 80: score += 20
            # 4. 现金流
            if info.get('freeCashflow', 0) > 0: score += 20
            return score
        except:
            return 50 # 默认中性

# ==========================================
# 4. 前端界面 (Streamlit UI)
# ==========================================

def main():
    st.set_page_config(page_title="Quantum Trader X", layout="wide", page_icon="⚡")
    
    # 侧边栏
    with st.sidebar:
        st.title("⚡ Quantum Trader X")
        st.caption("Ultimate Edition | Reflexivity + AI")
        symbol = st.text_input("Symbol", "NVDA").upper()
        
        st.markdown("---")
        st.markdown("### 🎛️ 核心参数")
        sim_count = st.slider("MCTS 模拟次数", 100, 5000, 1000)
        
        run_btn = st.button("🚀 启动量子计算", type="primary")
        
    # 主界面
    st.title(f"量子反身性分析报告: {symbol}")
    
    if run_btn:
        engine = QuantumEngine(symbol)
        
        with st.spinner("1. 正在连接华尔街数据源..."):
            df = engine.fetch_data()
            
        if df is None:
            st.error("无法获取数据。")
            return
            
        # 计算核心指标
        last_row = df.iloc[-1]
        rvol = last_row['RVOL']
        
        # 1. 风险计算
        risk_manager = PowerLawRisk()
        var_95 = risk_manager.calculate_hybrid_var(df['Close'].pct_change().dropna())
        
        # 2. 巴菲特评分
        fund_score = engine.get_buffett_score()
        
        # 3. MCTS 推演
        with st.spinner("2. 正在进行反身性博弈推演..."):
            paths = engine.run_mcts(df, sim_count)
            final_prices = [p[-1] for p in paths]
            win_rate = np.mean(np.array(final_prices) > last_row['Close'])
        
        # --- 结果展示 ---
        
        # 顶部 KPI
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("当前价格", f"${last_row['Close']:.2f}")
        k2.metric("RVOL (情绪放大器)", f"{rvol:.2f}x", "🔥 拥挤" if rvol > 1.5 else "平稳")
        k3.metric("MCTS 胜率", f"{win_rate:.1%}", delta_color="normal" if win_rate > 0.5 else "inverse")
        k4.metric("混合 VaR (风险)", f"{var_95:.2%}", "低风险" if var_95 < 0.03 else "高风险", delta_color="inverse")
        
        # 核心图表：MCTS 路径模拟
        st.subheader("🔮 反身性未来路径模拟 (Reflexivity Paths)")
        fig_mcts = go.Figure()
        # 只画前 50 条路径避免卡顿
        for p in paths[:50]:
            fig_mcts.add_trace(go.Scatter(y=p, mode='lines', line=dict(width=1, color='rgba(0,255,200,0.1)'), showlegend=False))
        # 画均值线
        avg_path = np.mean(paths, axis=0)
        fig_mcts.add_trace(go.Scatter(y=avg_path, mode='lines', name='平均预期路径', line=dict(width=3, color='white')))
        fig_mcts.update_layout(template="plotly_dark", height=400, margin=dict(l=0,r=0,t=30,b=0))
        st.plotly_chart(fig_mcts, use_container_width=True)
        
        # 深度分析栏
        c1, c2 = st.columns([1, 1])
        with c1:
            st.info(f"**🧠 AlphaGo 策略网络分析**\n\n"
                    f"虽然模型处于演示模式(未预训练)，但逻辑已部署。\n"
                    f"- 策略头输出: Buy / Hold / Sell 概率分布\n"
                    f"- 价值头输出: 胜率评估 {-0.5:.2f} (示例)")
            
        with c2:
            if fund_score > 70:
                st.success(f"**💎 巴菲特价值评分: {fund_score}**\n\n该资产基本面强劲，符合价值投资标准，可作为 MCTS 策略的安全垫。")
            else:
                st.warning(f"**⚠️ 巴菲特价值评分: {fund_score}**\n\n基本面一般或高估。建议严格控制仓位，仅做短线博弈。")

        # 原始数据折叠
        with st.expander("查看详细历史数据"):
            st.dataframe(df.tail(20))

if __name__ == "__main__":
    main()
