"""Plots for data analysis"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

from src.yfinance_app.stats import compute_drawdown
pio.renderers.default = "vscode"



def plot_normalized(tri: pd.DataFrame, title: str = "Normalized Total Return (start=100)"):
    """
    Plot normalized total return index starting at 100.

    Normalizes the Total Return Index so that all series start at 100, making
    it easier to compare performance across different tickers.

    Args:
        tri: DataFrame with Total Return Index values. Each column represents
             a ticker, index is datetime.
        title: Plot title. Default is "Normalized Total Return (start=100)".

    Returns:
        plotly.graph_objects.Figure: Plotly figure object with line chart showing
                                    normalized TRI over time.
    """
    norm = tri / tri.iloc[0] * 100.0
    fig = px.line(
        norm,
        title=title,
        labels={"value":"Index Level","index":"Date","variable":"Ticker"}
    )
    return fig


def plot_drawdowns(tri: pd.DataFrame, title: str = "Drawdowns (from running peak)"):
    """
    Plot drawdown series for all tickers.

    Visualizes the drawdown (decline from peak) over time for each ticker.
    Drawdowns are displayed as percentages.

    Args:
        tri: DataFrame with Total Return Index values. Each column represents
             a ticker, index is datetime.
        title: Plot title. Default is "Drawdowns (from running peak)".

    Returns:
        plotly.graph_objects.Figure: Plotly figure object with line chart showing
                                    drawdown percentages over time.
    """
    dd = compute_drawdown(tri)
    fig = px.line(dd, title=title, labels={"value":"Drawdown","index":"Date","variable":"Ticker"})
    fig.update_yaxes(tickformat=".0%")
    return fig


def plot_rolling_vol(returns: pd.DataFrame, window: int = 30, title: str | None = None):
    """
    Plot rolling annualized volatility over time.

    Calculates and visualizes rolling volatility (standard deviation) of returns,
    annualized by multiplying by the square root of 252 trading days.

    Args:
        returns: DataFrame with return values (as fractions, e.g., 0.01 = 1%).
                Each column represents a ticker, index is datetime.
        window: Rolling window size in days. Default is 30.
        title: Plot title. If None, defaults to "Rolling Volatility ({window}d, annualized)".

    Returns:
        plotly.graph_objects.Figure: Plotly figure object with line chart showing
                                    rolling annualized volatility over time.
    """
    rvol = returns.rolling(window).std() * np.sqrt(252)
    ttl = title or f"Rolling Volatility ({window}d, annualized)"
    fig = px.line(rvol, title=ttl, labels={"value":"Volatility","index":"Date","variable":"Ticker"})
    return fig


def plot_corr_heatmap(corr: pd.DataFrame, title: str = "Correlation Heatmap"):
    """
    Plot correlation matrix as a heatmap.

    Visualizes pairwise correlations between tickers using a color-coded heatmap.
    Uses a red-blue color scale with values ranging from -1 to 1.

    Args:
        corr: DataFrame with correlation matrix. Should be symmetric with tickers
              as both index and columns. Values should be between -1 and 1.
        title: Plot title. Default is "Correlation Heatmap".

    Returns:
        plotly.graph_objects.Figure: Plotly figure object with heatmap visualization
                                    showing correlation values with text labels.
    """
    fig = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdBu",
        origin="lower",
        title=title
    )
    fig.update_traces(zmin=-1, zmax=1)
    return fig


def plot_risk_return_scatter(
    summary: pd.DataFrame,
    title: str = "Risk–Return: CAGR vs Annualized Volatility"
):
    """
    Plot risk-return scatter chart showing CAGR vs annualized volatility.

    Creates a scatter plot where each point represents a ticker, with x-axis
    showing risk (volatility) and y-axis showing return (CAGR). Useful for
    comparing risk-adjusted returns across tickers.

    Args:
        summary: DataFrame with performance metrics. Must contain 'CAGR' and
                'Annual Volatility' columns. Index should be ticker names.
        title: Plot title. Default is "Risk–Return: CAGR vs Annualized Volatility".

    Returns:
        plotly.graph_objects.Figure: Plotly figure object with scatter plot showing
                                    risk-return relationship. Ticker names are
                                    displayed as text labels.
    """
    fig = px.scatter(
        summary.reset_index(),
        x="Annual Volatility", y="CAGR", text="Ticker", color="Ticker",
        title=title, labels={"x":"Annual Volatility","y":"CAGR"}
    )
    fig.update_traces(textposition="top center")
    return fig


def plot_tail_risk_bars(summary: pd.DataFrame, alpha: float = 0.05):
    """
    Plot Value at Risk (VaR) and Conditional VaR (CVaR) as grouped bar chart.

    Visualizes tail risk metrics for each ticker, showing both VaR and CVaR
    side by side for easy comparison.

    Args:
        summary: DataFrame with performance metrics. Must contain columns named
                'VaR{alpha*100}% (daily)' and 'CVaR{alpha*100}% (daily)'.
                Index should be ticker names.
        alpha: Confidence level used for VaR/CVaR calculation. Default is 0.05 (5%).

    Returns:
        plotly.graph_objects.Figure: Plotly figure object with grouped bar chart
                                    showing VaR and CVaR for each ticker.
    """
    var_col  = f"VaR{int(alpha*100)}% (daily)"
    cvar_col = f"CVaR{int(alpha*100)}% (daily)"
    sub = summary[[var_col, cvar_col]].reset_index()
    fig = go.Figure(data=[
        go.Bar(name=var_col,  x=sub["Ticker"], y=sub[var_col]),
        go.Bar(name=cvar_col, x=sub["Ticker"], y=sub[cvar_col]),
    ])
    fig.update_layout(barmode="group", title="Tail Risk (historical)")
    return fig


def plot_hist_daily_returns(daily_returns: pd.DataFrame, facet_wrap: int = 2):
    """
    Plot faceted histograms of daily returns for each ticker.

    Creates separate histogram plots for each ticker showing the distribution
    of daily returns. Returns are expected as fractions (e.g., 0.01 = 1%).
    Number of bins is determined using the square root rule.

    Args:
        daily_returns: DataFrame with daily return values (as fractions, e.g., 0.01 = 1%).
                      Each column represents a ticker, index is datetime.
        facet_wrap: Number of columns to use when wrapping facet plots. Default is 2.

    Returns:
        plotly.graph_objects.Figure: Plotly figure object with faceted histograms,
                                    one per ticker. X-axis is formatted as percentages.
    """
    long = daily_returns.melt(ignore_index=False, var_name="Ticker", value_name="Return")
    n = long["Return"].dropna().size
    nbins = max(10, int(np.sqrt(n)))  # simple √n rule
    fig = px.histogram(
        long, x="Return", facet_col="Ticker", facet_col_wrap=facet_wrap,
        nbins=nbins, title="Histograms of Daily Returns"
    )
    fig.update_xaxes(tickformat=".1%")
    return fig


def plot_hist_monthly_returns(monthly_return_pct: pd.DataFrame, facet_wrap: int = 2):
    """
    Plot faceted histograms of monthly returns for each ticker.

    Creates separate histogram plots for each ticker showing the distribution
    of monthly returns. Returns are expected as percentages (e.g., 5.0 = 5%)
    but are converted to fractions for consistent plotting with daily returns.
    Number of bins is determined using the square root rule.

    Args:
        monthly_return_pct: DataFrame with monthly return values as percentages
                           (e.g., 5.0 = 5%). Each column represents a ticker,
                           index is datetime.
        facet_wrap: Number of columns to use when wrapping facet plots. Default is 2.

    Returns:
        plotly.graph_objects.Figure: Plotly figure object with faceted histograms,
                                    one per ticker, showing monthly return distributions.
    """
    monthly_returns_fraction = monthly_return_pct / 100.0
    long = monthly_returns_fraction.melt(ignore_index=False, var_name="Ticker", value_name="Return")
    n = long["Return"].dropna().size
    nbins = max(10, int(np.sqrt(n)))
    fig = px.histogram(
        long, x="Return", facet_col="Ticker", facet_col_wrap=facet_wrap,
        nbins=nbins, title="Histograms of Monthly Returns"
    )
    fig.update_xaxes(tickformat=".1%")
    return fig
