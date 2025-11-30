"""
Quantum Trader V8.1 - Physics Engine + Reflexivity + AI Dashboard

基于：
- Streamlit 作为前端
- yfinance 拉取行情
- FFT 提取市场主周期
- 矩阵化 MCTS（蒙特卡洛）模拟反身性路径
"""

from __future__ import annotations

from typing import Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from scipy.fft import fft

# 尝试导入 PyTorch（可选）
try:
    import torch
    import torch.nn as nn
except ImportError:  # 在未安装 torch 的环境下也能跑
    torch = None
    nn = None


# ==========================================
# 1. 物理与数学引擎 (Physics & Math Core)
# ==========================================
class PhysicsEngine:
    """
    V8.0 核心：引入物理学方法分析市场
    1. FFT (快速傅立叶变换) -> 识别市场周期
    2. Matrix MCTS (矩阵化蒙特卡洛) -> 提升运算速度
    """

    @staticmethod
    def analyze_cycles_fft(prices: np.ndarray) -> Tuple[float, float]:
        """
        利用 FFT 识别市场主周期

        :param prices: 收盘价序列 (numpy array)
        :return: (dominant_period, cycle_strength)
        """
        if prices is None or len(prices) < 2:
            return 0.0, 0.0

        # 去趋势 (Detrending) 以提取纯周期波动
        prices_detrend = prices - np.mean(prices)
        n = len(prices_detrend)

        # FFT 变换
        fft_output = fft(prices_detrend)
        half_n = n // 2
        if half_n < 2:
            return 0.0, 0.0

        power = np.abs(fft_output[:half_n])  # 能量谱
        freqs = np.fft.fftfreq(n, d=1)[:half_n]  # 频率

        # 找到能量最大的主频率 (忽略直流分量)
        if len(power) > 1:
            idx = np.argmax(power[1:]) + 1
            # 避免除零
            if freqs[idx] == 0:
                return 0.0, 0.0
            dominant_period = 1.0 / freqs[idx]
            cycle_strength = float(power[idx] / (np.sum(power) + 1e-9))
            return float(dominant_period), cycle_strength

        return 0.0, 0.0

    @staticmethod
    def mcts_matrix_simulation(
        price_0: float,
        vol_0: float,
        avg_vol: float,
        base_sigma: float,
        simulations: int = 1000,
        horizon: int = 5,
    ) -> Tuple[float, float, float, np.ndarray]:
        """
        [矩阵加速版] 反身性博弈推演

        :return: (win_rate, expected_price, var_95, final_prices_array)
        """
        simulations = int(simulations)
        horizon = int(horizon)

        if simulations <= 0 or horizon <= 0:
            return 0.0, price_0, price_0, np.array([price_0])

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
        feedback_drift = 0.001 * amplifier * np.random.choice(
            [-1, 1], size=(simulations, horizon)
        )

        # 每日回报率 = 随机冲击 * 动态波动 + 情绪反馈
        daily_returns = shocks * dynamic_sigma + feedback_drift

        # 累积回报率 -> 价格路径
        cumulative_returns = np.cumprod(1 + daily_returns, axis=1)
        final_prices = price_0 * cumulative_returns[:, -1]

        # 统计结果
        win_rate = float(np.mean(final_prices > price_0))
        expected_price = float(np.mean(final_prices))
        var_95 = float(np.percentile(final_prices, 5))  # 下 5% 分位，作为 95% VaR

        return win_rate, expected_price, var_95, final_prices


# ==========================================
# 2. 深度学习模型 (Quantum LSTM) - 可选
# ==========================================
if torch is not None and nn is not None:

    class QuantumLSTM(nn.Module):
        """
        占位深度模型（当前未在 GUI 中调用）
        预留接口方便后续把 LSTM 信号并入决策打分
        """

        def __init__(self, input_size: int = 10, hidden_size: int = 64) -> None:
            super().__init__()
            self.lstm = nn.LSTM(
                input_size, hidden_size, batch_first=True, bidirectional=True
            )
            self.attention = nn.MultiheadAttention(
                hidden_size * 2, num_heads=4, batch_first=True
            )
            self.fc = nn.Linear(hidden_size * 2, 3)  # Output: Buy, Hold, Sell

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            lstm_out, _ = self.lstm(x)
            attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
            logits = self.fc(attn_out[:, -1, :])
            return torch.softmax(logits, dim=1)


# ==========================================
# 3. 数据引擎
# ==========================================
@st.cache_data(ttl=300)
def get_data(symbol: str) -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, Any]]]:
    """
    从 yfinance 获取近 1 年数据，并计算基础指标 + FFT 周期信息
    """
    try:
        stock = yf.Ticker(symbol)
        df = stock.history(period="1y")

        if df.empty:
            return None, None

        # 基础指标
        df["MA20"] = df["Close"].rolling(20).mean()
        df["Vol_MA20"] = df["Volume"].rolling(20).mean()
        df["Returns"] = df["Close"].pct_change()

        # 物理周期分析 (FFT)
        recent_prices = df["Close"].tail(60).values
        period, strength = PhysicsEngine.analyze_cycles_fft(recent_prices)

        # 有些 yfinance 版本 stock.info 可能比较慢 / 不稳定，统一 try 包一下
        try:
            info = stock.info
        except Exception:
            info = {}

        meta = {
            "period": period,
            "cycle_strength": strength,
            "info": info,
        }

        # 丢掉前期 rolling 产生的 NaN
        return df.dropna(), meta

    except Exception:
        # 不在缓存函数里 log 太多，直接返回 None 即可
        return None, None


def get_buffett_score(info: Dict[str, Any]) -> int:
    """
    巴菲特基本面评分（非常粗糙的打分，仅作示意）

    :param info: yfinance 的 info 字典
    """
    score = 0
    try:
        if info.get("trailingPE", 99) < 25:
            score += 30
        if info.get("returnOnEquity", 0) > 0.15:
            score += 30
        if info.get("debtToEquity", 100) < 80:
            score += 20
        if info.get("freeCashflow", 0) > 0:
            score += 20
    except Exception:
        score = 50  # 数据缺失给中性分
    return int(score)


# ==========================================
# 4. GUI 主界面
# ==========================================


def main() -> None:
    st.set_page_config(
        page_title="Quantum Trader V8.1", layout="wide", page_icon="⚛️"
    )

    # CSS 美化
    st.markdown(
        """
        <style>
        .stApp {background-color: #0e1117;}
        .metric-card {
            background-color: #262730;
            padding: 15px;
            border-radius: 10px;
            border-left: 5px solid #4CAF50;
        }
        .warning-card {
            background-color: rgba(255, 75, 75, 0.1);
            padding: 15px;
            border-radius: 10px;
            border-left: 5px solid #FF4B4B;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # 初始化默认 symbol
    if "symbol" not in st.session_state:
        st.session_state.symbol = "NVDA"

    # ========== 侧边栏 ==========
    with st.sidebar:
        st.title("⚛️ 量子控制台 V8.1")
        st.caption("Physics Engine + Reflexivity + AI")

        st.subheader("1. 标的选择")

        # 快捷按钮
        col_a, col_b, col_c = st.columns(3)
        if col_a.button("NVDA"):
            st.session_state.symbol = "NVDA"
        if col_b.button("BTC"):
            st.session_state.symbol = "BTC-USD"
        if col_c.button("AAPL"):
            st.session_state.symbol = "AAPL"

        # 文本输入与 session_state 绑定
        symbol_input = st.text_input(
            "输入代码 (如 600519.SS)",
            value=st.session_state.symbol,
            key="symbol",
        )
        symbol = symbol_input.upper()

        st.markdown("---")
        st.subheader("2. 模拟参数")
        sim_count = st.slider("MCTS 模拟次数", 1000, 10000, 2000, step=500)

        run_btn = st.button("🚀 启动深度分析", type="primary")

    st.title(f"📊 量化深度分析: {symbol}")

    df: Optional[pd.DataFrame] = None
    meta: Optional[Dict[str, Any]] = None

    if run_btn:
        with st.spinner("正在连接物理引擎与华尔街数据源..."):
            df, meta = get_data(symbol)

        if df is None or meta is None:
            st.error(
                f"❌ 无法获取 {symbol} 数据。请检查代码拼写 (如 A股需加后缀 .SS 或 .SZ)。"
            )
        else:
            # ====== 数据准备 ======
            last_row = df.iloc[-1]
            current_price = float(last_row["Close"])
            current_vol = float(last_row["Volume"])
            avg_vol = float(last_row["Vol_MA20"])
            volatility = float(df["Returns"].std())

            # 1. 运行矩阵加速 MCTS
            win_rate, target, var_95, paths = PhysicsEngine.mcts_matrix_simulation(
                current_price,
                current_vol,
                avg_vol,
                volatility,
                simulations=sim_count,
            )

            # 2. 计算 RVOL
            rvol = current_vol / (avg_vol + 1e-9)

            # 3. 巴菲特评分
            f_score = get_buffett_score(meta["info"])

            # ====== 仪表盘 ======
            col1, col2, col3, col4 = st.columns(4)
            col1.metric(
                "当前价格",
                f"${current_price:.2f}",
                f"{last_row['Returns'] * 100:.2f}%",
            )

            # 周期指标
            period = meta["period"]
            p_str = f"{period:.1f} 天" if period and period > 0 else "无明显周期"
            col2.metric(
                "FFT 市场周期",
                p_str,
                f"强度 {meta['cycle_strength'] * 100:.0f}%",
            )

            # 反身性指标
            if rvol > 2.0:
                state = "🔥 极度拥挤"
            elif rvol > 1.2:
                state = "⚡ 活跃"
            else:
                state = "🧊 平稳"
            col3.metric(
                "RVOL (情绪放大)",
                f"{rvol:.2f}x",
                state,
                delta_color="inverse",
            )

            # 预测指标
            col4.metric("MCTS 胜率", f"{win_rate * 100:.1f}%", f"目标 ${target:.2f}")

            st.markdown("---")

            # ====== 深度图表区 ======
            c1, c2 = st.columns([2, 1])

            with c1:
                st.subheader("🔮 多重宇宙推演 (Matrix Simulation)")

                fig = go.Figure()
                fig.add_trace(
                    go.Histogram(
                        x=paths,
                        nbinsx=60,
                        marker_color="#00CC96",
                        name="预测分布",
                    )
                )
                fig.add_vline(
                    x=current_price,
                    line_dash="dash",
                    line_color="white",
                    annotation_text="当前价",
                )
                fig.add_vline(
                    x=var_95,
                    line_dash="dot",
                    line_color="red",
                    annotation_text="VaR 95%",
                )
                fig.update_layout(
                    title=f"基于 {sim_count} 次反身性模拟的未来价格分布",
                    height=350,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="white"),
                )
                st.plotly_chart(fig, use_container_width=True)

                if rvol > 1.5:
                    st.warning(
                        f"⚠️ **反身性警报：** 市场处于非线性状态 (RVOL={rvol:.1f})。"
                        "情绪反馈正在指数级放大波动，建议降低杠杆。"
                    )

            with c2:
                st.subheader("🧭 综合决策")
                final_score = (
                    win_rate * 50
                    + f_score * 0.3
                    + meta["cycle_strength"] * 20
                )
                if rvol > 2.0:
                    final_score -= 15  # 过热惩罚

                st.markdown(f"### 得分: {final_score:.1f} / 100")
                st.progress(min(int(final_score), 100))

                if final_score > 60:
                    st.success("✅ **建议：买入** (动量 + 周期共振)")
                elif final_score < 40:
                    st.error("❌ **建议：卖出** (风险过高)")
                else:
                    st.info("👀 **建议：观望** (方向不明)")

                st.write(f"**巴菲特安全垫：** {f_score} 分")
                st.caption("注：得分基于 MCTS 胜率、基本面及物理周期强度的加权计算。")

            # ====== 原始数据展示 ======
            with st.expander("查看历史数据"):
                st.dataframe(df.tail(20))

    else:
        # 首次加载 / 未点击按钮时给一个友好提示
        st.info("在左侧输入标的代码并点击 **🚀 启动深度分析**。")


if __name__ == "__main__":
    main()
