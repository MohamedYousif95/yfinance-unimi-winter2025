"""
Streamlit web app for exploring yfinance data.

This app demonstrates:
- Input/Output (yfinance API)
- Data manipulation with pandas
- Scientific computing (CAGR, volatility, VaR/CVaR, drawdowns, beta)
- Visualisation with Plotly
- Clean project structure suitable for your course evaluation
"""

from __future__ import annotations

# from pathlib import Path

import pandas as pd
import streamlit as st

from src.yfinance_app.config import (
    DEFAULT_TICKERS,
    DEFAULT_START,
    DEFAULT_END,
    # DEFAULT_INTERVAL,
)
from src.yfinance_app.data import load_yfinance_data
from src.yfinance_app.transform import (
    extract_close_prices,
    clean_closes,
    compute_daily_returns,
    monthly_aggregate,
)
from src.yfinance_app.stats import (
    build_performance_table,
    correlation_matrices,
    rolling_beta,
    total_return_index,
    compute_drawdown,
    rolling_statistics,
)
from src.yfinance_app.visualization import (
    plot_normalized,
    plot_drawdowns,
    plot_rolling_vol,
    plot_hist_daily_returns,
    plot_hist_monthly_returns,
    plot_corr_heatmap,
    plot_risk_return_scatter,
    plot_tail_risk_bars,
)

# --------------------------------------------------------------------
# Streamlit page config
# --------------------------------------------------------------------
st.set_page_config(
    page_title="YFinance Risk & Return Dashboard",
    layout="wide",
)

st.title("📈 YFinance Risk & Return Dashboard")

st.markdown(
    """
This dashboard downloads financial time series from **Yahoo Finance** and
computes a set of **risk & return metrics**:

- CAGR, annualized volatility, maximum drawdown
- Historical **VaR / CVaR** (tail risk)
- Correlation matrices (daily & monthly)
- Rolling volatility and rolling beta
- Distribution plots of daily and monthly returns

Use the left sidebar to configure tickers and date range.
"""
)

# --------------------------------------------------------------------
# Sidebar controls
# --------------------------------------------------------------------
st.sidebar.header("Configuration")

# Tickerset
tickers = st.sidebar.multiselect(
    "Tickers",
    options=DEFAULT_TICKERS,
    default=DEFAULT_TICKERS,
    help="Choose which instruments to analyse.",
)

if not tickers:
    st.warning("Please select at least one ticker in the sidebar.")
    st.stop()

# Date range
start_date = st.sidebar.date_input(
    "Start date",
    pd.to_datetime(DEFAULT_START),
)
end_default = pd.to_datetime("today") if DEFAULT_END is None else pd.to_datetime(DEFAULT_END)
end_date = st.sidebar.date_input(
    "End date",
    end_default,
)

# Interval
interval = st.sidebar.selectbox(
    "Interval",
    options=["1d", "1wk", "1mo"],
    index=0,
)

# Tail probability for VaR / CVaR
alpha = st.sidebar.slider(
    "Tail probability for VaR / CVaR",
    min_value=0.01,
    max_value=0.10,
    value=0.05,
    step=0.01,
)

st.sidebar.markdown("---")
st.sidebar.caption("Data source: Yahoo Finance via `yfinance`.")

# --------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------
st.subheader("1. Data Loading")

with st.spinner("Downloading data from Yahoo Finance..."):
    df_raw = load_yfinance_data(
        tickers=tickers,
        start=start_date.strftime("%Y-%m-%d"),
        end=end_date.strftime("%Y-%m-%d"),
        interval=interval,
    )

st.write("Raw yfinance output (head):")
st.dataframe(df_raw.head())

# --------------------------------------------------------------------
# Data manipulation / transformations
# --------------------------------------------------------------------
st.subheader("2. Data Preparation and Transformations")

# Extract and clean closes
closes_raw = extract_close_prices(df_raw, tickers=tickers)
closes, _ = clean_closes(closes_raw)



st.write("Cleaned closing prices (tail):")
st.dataframe(closes.tail())

# Daily & monthly returns
daily_returns = compute_daily_returns(closes)
monthly_close, monthly_return_pct = monthly_aggregate(closes)


# Total Return Index and drawdowns
tri = total_return_index(closes)  # starts at 1.0
drawdowns = tri.apply(compute_drawdown)

# Rolling statistics (mean & vol from daily returns)
rolling_stats = rolling_statistics(daily_returns, window=30)

st.markdown("**Derived series:** daily returns, monthly returns, TRI, drawdowns, rolling stats.")

# --------------------------------------------------------------------
# Scientific computing: metrics, correlations, rolling beta
# --------------------------------------------------------------------
st.subheader("3. Risk & Return Metrics")

# Performance summary table (CAGR, vol, max drawdown, VaR, CVaR)
summary_table = build_performance_table(
    daily_closing=closes,
    tri=tri,
    alpha=alpha,
)

# Correlation matrices
daily_corr, monthly_corr = correlation_matrices(
    daily_returns=daily_returns,
    monthly_returns_percent=monthly_return_pct,
)

# Choose a benchmark for rolling beta (if one exists)
BENCHMARK_DEFAULT = None
for candidate in tickers:
    if candidate.startswith("^"):  # typical index tickers
        BENCHMARK_DEFAULT = candidate
        break
if BENCHMARK_DEFAULT is None:
    BENCHMARK_DEFAULT = tickers[0]

benchmark = st.selectbox(
    "Benchmark for rolling beta",
    options=tickers,
    index=tickers.index(BENCHMARK_DEFAULT),
    help="Used only for the rolling beta computation.",
)

asset_for_beta = st.selectbox(
    "Asset for rolling beta (vs benchmark)",
    options=[t for t in tickers if t != benchmark] or [benchmark],
)

beta_series = rolling_beta(
    asset_returns=daily_returns[asset_for_beta],
    market_returns=daily_returns[benchmark],
    window=60,
)

# Display summary table
st.write("### Performance Summary per Ticker")

formatted = summary_table.copy()
# Format as percentages where appropriate
pct_cols = [c for c in formatted.columns if "CAGR" in c or "Vol" in c or "Drawdown" in c]
pct_cols += [c for c in formatted.columns if "VaR" in c or "CVaR" in c]

for col in pct_cols:
    formatted[col] = formatted[col].map(lambda x: f"{x:.2%}" if pd.notna(x) else "")

st.dataframe(formatted)

# --------------------------------------------------------------------
# Visualisations
# --------------------------------------------------------------------
st.subheader("4. Visualisations")

tab_prices, tab_distributions, tab_risk, tab_corr = st.tabs(
    ["Prices & TRI", "Return Distributions", "Risk Metrics", "Correlations & Beta"]
)

# --- Tab 1: Prices and TRI ---
with tab_prices:
    st.markdown("#### Normalized performance (TRI, start = 100)")

    fig_norm = plot_normalized(tri)
    st.plotly_chart(fig_norm, use_container_width=True)

    st.markdown("#### Drawdowns (peak-to-trough losses)")

    fig_dd = plot_drawdowns(tri)
    st.plotly_chart(fig_dd, use_container_width=True)

    st.markdown("#### Rolling volatility (30-day window, annualized)")
    fig_roll_vol = plot_rolling_vol(daily_returns, window=30)
    st.plotly_chart(fig_roll_vol, use_container_width=True)

# --- Tab 2: Distributions ---
with tab_distributions:
    st.markdown("#### Histograms of daily returns")

    fig_hist_daily = plot_hist_daily_returns(daily_returns)
    st.plotly_chart(fig_hist_daily, use_container_width=True)

    st.markdown("#### Histograms of monthly returns (%)")

    fig_hist_monthly = plot_hist_monthly_returns(monthly_return_pct)
    st.plotly_chart(fig_hist_monthly, use_container_width=True)

# --- Tab 3: Risk metrics (scatter, tail risk, drawdowns already shown) ---
with tab_risk:
    st.markdown("#### Risk–Return scatter (CAGR vs Volatility)")

    fig_rr = plot_risk_return_scatter(summary_table)
    st.plotly_chart(fig_rr, use_container_width=True)

    st.markdown(f"#### Tail risk: VaR and CVaR (α = {alpha:.0%})")

    fig_tail = plot_tail_risk_bars(summary_table, alpha=alpha)
    st.plotly_chart(fig_tail, use_container_width=True)

# --- Tab 4: Correlations & Rolling beta ---
with tab_corr:
    st.markdown("#### Correlation matrix – daily returns")
    fig_corr_daily = plot_corr_heatmap(daily_corr, title="Correlation (Daily Returns)")
    st.plotly_chart(fig_corr_daily, use_container_width=True)

    st.markdown("#### Correlation matrix – monthly returns")
    fig_corr_monthly = plot_corr_heatmap(monthly_corr, title="Correlation (Monthly Returns)")
    st.plotly_chart(fig_corr_monthly, use_container_width=True)

    st.markdown(
        f"#### Rolling beta of **{asset_for_beta}** vs benchmark **{benchmark}** (60-day window)"
    )
    beta_df = beta_series.to_frame(name="Rolling Beta")
    st.line_chart(beta_df)
