"""
Streamlit web app for exploring yfinance data using a modular class-based design.

This app demonstrates:
- Input/Output (yfinance API)
- Data manipulation with pandas
- Scientific computing (CAGR, volatility, VaR/CVaR, drawdowns, beta)
- Visualisation with Plotly
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional

import pandas as pd
import streamlit as st

from src.yfinance_app.config import (
    DEFAULT_TICKERS,
    DEFAULT_START,
    DEFAULT_END,
    DEFAULT_INTERVAL,
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


@dataclass
class AppState:
    """Application state for the TradeBuddy dashboard Streamlit app"""
    # Configuration chosen in the sidebar
    download_tickers: List[str] = field(default_factory=list)
    analysis_tickers: List[str] = field(default_factory=list)
    start_date: Optional[pd.Timestamp] = None
    end_date: Optional[pd.Timestamp] = None
    interval: str = DEFAULT_INTERVAL
    alpha: float = 0.05  # tail probability for VaR / CVaR

    # Data objects
    df_raw: Optional[pd.DataFrame] = None
    closes: Optional[pd.DataFrame] = None
    daily_returns: Optional[pd.DataFrame] = None
    monthly_return_pct: Optional[pd.DataFrame] = None
    tri: Optional[pd.DataFrame] = None
    drawdowns: Optional[pd.DataFrame] = None
    rolling_stats: Optional[pd.DataFrame] = None

    # Metrics
    summary_table: Optional[pd.DataFrame] = None
    daily_corr: Optional[pd.DataFrame] = None
    monthly_corr: Optional[pd.DataFrame] = None
    beta_series: Optional[pd.Series] = None
    benchmark: Optional[str] = None
    asset_for_beta: Optional[str] = None


class TradeBuddyDashboardApp:
    """Streamlit app for exploring yfinance data with risk & return metrics."""
    def __init__(self) -> None:
        st.set_page_config(
            page_title="TradeBuddy Risk & Return Dashboard",
            layout="wide",
        )
        self.state = AppState()


    def run(self) -> None:
        """Orchestrator to run TradeBuddy Web App"""
        self._render_title_and_intro()
        self._render_sidebar_config()
        self._load_data()
        self._prepare_data()
        self._compute_metrics_and_beta()
        self._render_visualisations()


    def _render_title_and_intro(self) -> None:
        st.title("📈 TradeBuddy Risk & Return Dashboard")
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

    def _render_sidebar_config(self) -> None:
        """Render sidebar controls and store choices in state."""
        st.sidebar.header("Configuration")

        # Tickers to download
        download_tickers = st.sidebar.multiselect(
            "Tickers to download",
            options=DEFAULT_TICKERS,
            default=DEFAULT_TICKERS,
            help="Choose which instruments to download and analyse.",
        )
        if not download_tickers:
            st.sidebar.error("Please select at least one ticker.")
            st.stop()

        # Date range
        start_date = st.sidebar.date_input(
            "Start date",
            pd.to_datetime(DEFAULT_START),
        )
        end_default = pd.to_datetime("today") if DEFAULT_END is None else pd.to_datetime(
            DEFAULT_END
        )
        end_date = st.sidebar.date_input("End date", end_default)

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

        # Store in state
        self.state.download_tickers = download_tickers
        self.state.start_date = pd.to_datetime(start_date)
        self.state.end_date = pd.to_datetime(end_date)
        self.state.interval = interval
        self.state.alpha = alpha

    def _load_data(self) -> None:
        """load raw price data from yfinance and display a preview."""
        st.subheader("1. Data Loading")

        with st.spinner("Downloading data from Yahoo Finance..."):
            df_raw = load_yfinance_data(
                tickers=self.state.download_tickers,
                start=self.state.start_date.strftime("%Y-%m-%d"),
                end=self.state.end_date.strftime("%Y-%m-%d"),
                interval=self.state.interval,
            )

        self.state.df_raw = df_raw

        st.write("Raw yfinance data preview:")
        st.dataframe(df_raw.head())

    def _prepare_data(self) -> None:
        """
        Transform raw data into closes, returns, TRI, drawdowns, and rolling stats.
        Also lets the user choose a subset of tickers for analysis.
        """
        st.subheader("2. Data Preparation and Transformations")

        if self.state.df_raw is None:
            st.error("No raw data loaded.")
            st.stop()

        # Extract & clean closing prices
        closes_raw = extract_close_prices(
            self.state.df_raw, tickers=self.state.download_tickers
        )
        closes, _ = clean_closes(closes_raw)

        # Use downloaded tickers for analysis
        analysis_tickers = list[Any](closes.columns)

        st.write("Cleaned closing prices:")
        st.dataframe(closes.tail())

        # Daily & monthly returns
        daily_returns = compute_daily_returns(closes)
        _, monthly_return_pct = monthly_aggregate(closes)

        # TRI & drawdowns
        tri = total_return_index(closes)
        drawdowns = tri.apply(compute_drawdown)

        # Rolling statistics
        rolling_stats = rolling_statistics(daily_returns, window=30)

        # Store in state
        self.state.analysis_tickers = analysis_tickers
        self.state.closes = closes
        self.state.daily_returns = daily_returns
        self.state.monthly_return_pct = monthly_return_pct
        self.state.tri = tri
        self.state.drawdowns = drawdowns
        self.state.rolling_stats = rolling_stats


    def _compute_metrics_and_beta(self) -> None:
        """
        Compute summary metrics, correlation matrices, and rolling beta.
        Render the performance summary table.
        """
        st.subheader("3. Risk & Return Metrics")

        if (
            self.state.closes is None
            or self.state.daily_returns is None
            or self.state.monthly_return_pct is None
            or self.state.tri is None
        ):
            st.error("Data not prepared.")
            st.stop()

        # Performance summary per ticker
        summary_table = build_performance_table(
            daily_closing=self.state.closes,
            tri=self.state.tri,
            alpha=self.state.alpha,
        )

        # Correlation matrices
        daily_corr, monthly_corr = correlation_matrices(
            daily_returns=self.state.daily_returns,
            monthly_returns_percent=self.state.monthly_return_pct,
        )

        # Choose benchmark & asset for rolling beta
        benchmark, asset_for_beta = self._select_beta_assets()
        beta_series = rolling_beta(
            asset_returns=self.state.daily_returns[asset_for_beta],
            market_returns=self.state.daily_returns[benchmark],
            window=60,
        )

        # Store metrics in state
        self.state.summary_table = summary_table
        self.state.daily_corr = daily_corr
        self.state.monthly_corr = monthly_corr
        self.state.beta_series = beta_series
        self.state.benchmark = benchmark
        self.state.asset_for_beta = asset_for_beta

        # Render performance summary table
        st.write("### Performance Summary per Ticker")
        st.dataframe(self._format_summary_table(summary_table))

    def _select_beta_assets(self) -> tuple[str, str]:
        """
        Determine benchmark and asset for rolling beta from analysis tickers.
        Uses an index-like ticker (starting with '^') as the default benchmark if available.
        """
        analysis_tickers = self.state.analysis_tickers
        if not analysis_tickers:
            st.error("No analysis tickers available for beta calculation.")
            st.stop()

        # Choose the benchmark
        benchmark_default = None
        for t in analysis_tickers:
            if t.startswith("^"):
                benchmark_default = t
                break
        if benchmark_default is None:
            benchmark_default = analysis_tickers[0]

        benchmark = st.selectbox(
            "Benchmark for rolling beta",
            options=analysis_tickers,
            index=analysis_tickers.index(benchmark_default),
            help="Used only for the rolling beta computation.",
        )

        candidate_assets = [t for t in analysis_tickers if t != benchmark] or [benchmark]

        # There is a bug here as the selected stock does not render in the rolling beta plot
        asset_for_beta = st.selectbox(
            "Asset for rolling beta (vs benchmark)",
            options=candidate_assets,
            index=0,
            key=f"asset_for_beta_{benchmark}",
        )

        return benchmark, asset_for_beta

    def _format_summary_table(self, summary_table: pd.DataFrame) -> pd.DataFrame:
        """
        Return a copy of the summary table with percentage-like columns formatted as text.
        """
        formatted = summary_table.copy()

        pct_cols = [
            c
            for c in formatted.columns
            if "CAGR" in c
            or "Annual Volatility" in c
            or "Max Drawdown" in c
            or "VaR" in c
            or "CVaR" in c
        ]

        for col in pct_cols:
            formatted[col] = formatted[col].map(
                lambda x: f"{x:.2%}" if pd.notna(x) else ""
            )

        return formatted

    def _render_visualisations(self) -> None:
        """Render all visualisation tabs."""
        st.subheader("4. Visualisations")

        tabs = st.tabs(
            ["Prices & TRI", "Return Distributions", "Risk Metrics", "Correlations & Beta"]
        )
        tab_prices, tab_distributions, tab_risk, tab_corr = tabs

        self._render_prices_and_tri_tab(tab_prices)
        self._render_distributions_tab(tab_distributions)
        self._render_risk_metrics_tab(tab_risk)
        self._render_correlations_and_beta_tab(tab_corr)

    def _render_prices_and_tri_tab(self, container) -> None:
        """Prices & TRI tab: normalised TRI, drawdowns, rolling vol."""
        # For the current tab
        with container:
            if self.state.tri is None or self.state.daily_returns is None:
                st.warning("No data for prices & TRI.")
                return

            st.markdown("#### Normalized performance (TRI, start = 100)")
            fig_norm = plot_normalized(self.state.tri)
            st.plotly_chart(fig_norm, use_container_width=True)

            st.markdown("#### Drawdowns (peak-to-trough losses)")
            fig_dd = plot_drawdowns(self.state.tri)
            st.plotly_chart(fig_dd, use_container_width=True)

            st.markdown("#### Rolling volatility (30-day window, annualized)")
            fig_roll_vol = plot_rolling_vol(self.state.daily_returns, window=30)
            st.plotly_chart(fig_roll_vol, use_container_width=True)

    def _render_distributions_tab(self, container) -> None:
        """Return distributions tab: histograms of daily & monthly returns."""
        with container:
            if (
                self.state.daily_returns is None
                or self.state.monthly_return_pct is None
            ):
                st.warning("No data for distributions.")
                return

            st.markdown("#### Histograms of daily returns")
            fig_hist_daily = plot_hist_daily_returns(self.state.daily_returns)
            st.plotly_chart(fig_hist_daily, use_container_width=True)

            st.markdown("#### Histograms of monthly returns (%)")
            fig_hist_monthly = plot_hist_monthly_returns(
                self.state.monthly_return_pct
            )
            st.plotly_chart(fig_hist_monthly, use_container_width=True)

    def _render_risk_metrics_tab(self, container) -> None:
        """Risk metrics tab: risk–return scatter and tail risk bars."""
        with container:
            if self.state.summary_table is None:
                st.warning("No summary metrics available.")
                return

            st.markdown("#### Risk–Return scatter (CAGR vs Volatility)")
            fig_rr = plot_risk_return_scatter(self.state.summary_table)
            st.plotly_chart(fig_rr, use_container_width=True)

            st.markdown(
                f"#### Tail risk: VaR and CVaR (α = {self.state.alpha:.0%})"
            )
            fig_tail = plot_tail_risk_bars(
                self.state.summary_table, alpha=self.state.alpha
            )
            st.plotly_chart(fig_tail, use_container_width=True)

    def _render_correlations_and_beta_tab(self, container) -> None:
        """Correlations & beta tab: heatmaps + rolling beta line chart."""
        with container:
            if (
                self.state.daily_corr is None
                or self.state.monthly_corr is None
                or self.state.beta_series is None
                or self.state.benchmark is None
                or self.state.asset_for_beta is None
            ):
                st.warning("No correlation/beta data available.")
                return

            st.markdown("#### Correlation matrix – daily returns")
            fig_corr_daily = plot_corr_heatmap(
                self.state.daily_corr, title="Correlation (Daily Returns)"
            )
            st.plotly_chart(fig_corr_daily, use_container_width=True)

            st.markdown("#### Correlation matrix – monthly returns")
            fig_corr_monthly = plot_corr_heatmap(
                self.state.monthly_corr, title="Correlation (Monthly Returns)"
            )
            st.plotly_chart(fig_corr_monthly, use_container_width=True)

            st.markdown(
                f"#### Rolling beta of **{self.state.asset_for_beta}** "
                f"vs benchmark **{self.state.benchmark}** (60-day window)"
            )
            beta_df = self.state.beta_series.to_frame(name="Rolling Beta")
            st.line_chart(beta_df)



if __name__ == "__main__":
    app = TradeBuddyDashboardApp()
    app.run()
