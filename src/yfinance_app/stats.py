"""Statistical Computations"""

import pandas as pd
import numpy as np

def rolling_statistics(
        df_returns: pd.DataFrame,
        window: int = 30,
        trading_days_per_year: int = 252) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute rolling mean and rolling volatility (annualized) of daily returns.

    Calculates rolling statistics over a specified window. Volatility is annualized
    by multiplying the standard deviation by the square root of trading days per year.

    Args:
        df_returns: DataFrame with daily returns. Each column represents a ticker,
                    index is datetime.
        window: Rolling window size in days. Default is 30.
        trading_days_per_year: Number of trading days per year for annualization.
                              Default is 252.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: A tuple containing:
            - rolling_mean: DataFrame with rolling mean returns.
            - rolling_volatility: DataFrame with rolling annualized volatility.
    """
    rolling_mean = df_returns.rolling(window).mean()
    rolling_volatility = df_returns.rolling(window).std() * np.sqrt(trading_days_per_year)
    return rolling_mean, rolling_volatility


def compute_drawdown(df_return: pd.DataFrame) -> pd.DataFrame:
    """
    Compute drawdown series from price/level series.

    Calculates the drawdown at each point as the percentage decline from the
    running maximum. Formula: drawdown_t = (price_t / running_max_t) - 1

    Args:
        df_return: DataFrame with price or level series. Each column represents
                   a ticker, index is datetime.

    Returns:
        pd.DataFrame: DataFrame with drawdown values (as fractions, e.g., -0.1 = -10%).
                     Values are negative or zero (zero when at peak).
    """
    cum_max = df_return.cummax(axis=0)
    drawdown = df_return / cum_max - 1.0
    return drawdown


def total_return_index(daily_closing: pd.DataFrame) -> pd.DataFrame:
    """
    Build a Total Return Index (TRI) from daily closing prices.

    Constructs a TRI by computing the cumulative product of (1 + daily return),
    which is equivalent to normalizing prices by the first value. The index starts
    at 1.0 for all tickers.

    Args:
        daily_closing: DataFrame with daily closing prices. Each column represents
                       a ticker, index is datetime.

    Returns:
        pd.DataFrame: DataFrame with Total Return Index values, starting at 1.0.
                     Missing values are forward-filled with 1.0.
    """
    return (1.0 + daily_closing.pct_change()).cumprod().fillna(1.0)


def cagr(series: pd.Series) -> float:
    """
    Compute Compound Annual Growth Rate (CAGR) from a level series.

    Calculates the annualized return based on the first and last values of the series.
    Formula: (final_value / initial_value)^(1/years) - 1

    Args:
        series: Series with level values (e.g., prices or TRI). Index should be datetime.
               Missing values are dropped before calculation.

    Returns:
        float: CAGR as a fraction (e.g., 0.1 = 10% annual return). Returns np.nan
              if series is empty or time period is invalid.
    """
    s = series.dropna()
    if s.empty:
        return np.nan
    years = (s.index[-1] - s.index[0]).days / 365.25
    return (s.iloc[-1] / s.iloc[0])**(1/years) - 1 if years > 0 else np.nan


def max_drawdown(series: pd.Series) -> float:
    """
    Compute the maximum drawdown from a level series.

    Calculates the most negative drawdown value (maximum depth) by finding the
    minimum value in the drawdown series.

    Args:
        series: Series with level values (e.g., prices or TRI). Index should be datetime.

    Returns:
        float: Maximum drawdown as a fraction (e.g., -0.25 = -25% maximum decline).
              Always negative or zero.
    """
    drawdown_series = compute_drawdown(series)
    return float(drawdown_series.min())


def var_cvar(returns: pd.Series, alpha: float = 0.05) -> tuple[float, float]:
    """
    Compute Value at Risk (VaR) and Conditional Value at Risk (CVaR).

    Calculates historical (non-parametric) risk metrics from return series.
    VaR is the alpha-quantile of returns, and CVaR is the mean of returns
    that are less than or equal to VaR (tail risk).

    Args:
        returns: Series with return values (as fractions, e.g., 0.01 = 1%).
                Missing values are dropped and values are sorted before calculation.
        alpha: Confidence level for VaR/CVaR. Default is 0.05 (5% tail risk).

    Returns:
        tuple[float, float]: A tuple containing:
            - var: Value at Risk as a fraction (e.g., -0.02 = -2%).
            - cvar: Conditional Value at Risk as a fraction.
                   Returns np.nan for both if series is empty or no values <= VaR.
    """
    r = returns.dropna().sort_values()
    if r.empty:
        return np.nan, np.nan
    var = float(r.quantile(alpha))
    cvar = float(r[r <= var].mean()) if (r <= var).any() else np.nan
    return var, cvar



def correlation_matrices(
    daily_returns: pd.DataFrame,
    monthly_returns_percent: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute Pearson correlation matrices for daily and monthly returns.

    Calculates pairwise correlation coefficients between all tickers for both
    daily and monthly return frequencies. Monthly returns are converted from
    percentage to fraction before computing correlations.

    Args:
        daily_returns: DataFrame with daily returns (as fractions, e.g., 0.01 = 1%).
                      Each column represents a ticker, index is datetime.
        monthly_returns_percent: DataFrame with monthly returns as percentages
                                 (e.g., 5.0 = 5%). Each column represents a ticker,
                                 index is datetime.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: A tuple containing:
            - corr_daily: Correlation matrix for daily returns (symmetric matrix).
            - corr_monthly: Correlation matrix for monthly returns (symmetric matrix).
    """
    corr_daily   = daily_returns.corr()
    corr_monthly = (monthly_returns_percent / 100.0).corr()
    return corr_daily, corr_monthly


def rolling_beta(
    asset_returns: pd.Series,
    market_returns: pd.Series,
    window: int = 252
) -> pd.Series:
    """
    Compute rolling beta coefficient between an asset and market returns.

    Calculates rolling beta as the ratio of rolling covariance to rolling variance
    of market returns. Beta measures the asset's sensitivity to market movements.
    A beta of 1.0 means the asset moves with the market.

    Args:
        asset_returns: Series with asset return values (as fractions, e.g., 0.01 = 1%).
                     Index should be datetime, aligned with market_returns.
        market_returns: Series with market/benchmark return values (as fractions).
                       Index should be datetime, aligned with asset_returns.
        window: Rolling window size in trading days. Default is 252 (1 year).

    Returns:
        pd.Series: Series with rolling beta values. Index matches input series.
                  Beta = Cov(asset, market) / Var(market) over rolling window.
    """
    cov = asset_returns.rolling(window).cov(market_returns)
    var = market_returns.rolling(window).var()
    return cov / var


def build_performance_table(
    daily_closing: pd.DataFrame,
    tri: pd.DataFrame,
    alpha: float = 0.05
) -> pd.DataFrame:
    """
    Build a comprehensive performance summary table for all tickers.

    Computes key performance metrics for each ticker including CAGR, annualized
    volatility, maximum drawdown, and tail risk measures (VaR and CVaR).

    Args:
        daily_closing: DataFrame with daily closing prices. Each column represents
                       a ticker, index is datetime.
        tri: DataFrame with Total Return Index values. Each column represents
             a ticker, index is datetime. Should align with daily_closing.
        alpha: Confidence level for VaR/CVaR calculation. Default is 0.05 (5% tail risk).

    Returns:
        pd.DataFrame: DataFrame with performance metrics indexed by ticker.
                     Columns include:
                     - CAGR: Compound Annual Growth Rate
                     - Annual Volatility: Annualized volatility
                     - Max Drawdown: Maximum drawdown
                     - VaR{alpha*100}% (daily): Value at Risk
                     - CVaR{alpha*100}% (daily): Conditional Value at Risk
    """
    rets_d = daily_closing.pct_change()
    rows = []
    for t in tri.columns:
        tri_t = tri[t]
        r     = rets_d[t]
        vr, cvr = var_cvar(r, alpha=alpha)
        rows.append({
            "Ticker": t,
            "CAGR": cagr(tri_t),
            "Annual Volatility": float(r.std(ddof=1) * np.sqrt(252)),
            "Max Drawdown": max_drawdown(tri_t),
            f"VaR{int(alpha*100)}% (daily)": vr,
            f"CVaR{int(alpha*100)}% (daily)": cvr,
        })
    return pd.DataFrame(rows).set_index("Ticker")
