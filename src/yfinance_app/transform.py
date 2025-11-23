"""Transformation of data"""

import pandas as pd


def extract_close_prices(df: pd.DataFrame, tickers: list[str] | None = None) -> pd.DataFrame:
    """
    Extract closing prices from a DataFrame containing stock market data.

    Handles both MultiIndex column structures (from yfinance multi-ticker downloads)
    and simple column structures. The resulting DataFrame is sorted by index and
    has timezone-naive datetime index.

    Args:
        df: DataFrame containing market data. Can have MultiIndex columns (with 'Close'
            as level 1) or simple columns with 'Close' column.
        tickers: Optional list of ticker symbols. Used to name columns when DataFrame
                 has simple structure. If None and simple structure, uses 'Close'.

    Returns:
        pd.DataFrame: DataFrame with closing prices, sorted by index, with timezone-naive
                      datetime index.

    Raises:
        ValueError: If DataFrame has simple structure but doesn't contain 'Close' column.
    """
    if isinstance(df.columns, pd.MultiIndex):
        closes=df.xs('Close', axis=1, level=1)
    else:
        if "Close" not in df.columns:
            raise ValueError("DataFrame does not contain 'Close' prices.")
        name=tickers[0] if tickers else 'Close'
        closes=df["Close"].to_frame(name=name)
    closes=closes.sort_index()
    closes.index=pd.to_datetime(closes.index).tz_localize(None)
    return closes


def clean_closes(daily_closing: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Clean daily closing prices by forward-filling gaps and removing leading NA rows.

    Forward-fills missing values to handle gaps in trading data, then removes rows
    where all tickers have NA values. Reports NA counts before filling for analysis.

    Args:
        daily_closing: DataFrame with daily closing prices. Each column represents
                       a ticker, index is datetime.

    Returns:
        tuple[pd.DataFrame, pd.Series]: A tuple containing:
            - closes_ffill: DataFrame with forward-filled prices, sorted by index,
                           with leading all-NA rows removed.
            - na_counts: Series showing the count of NA values per ticker before
                        forward-filling (for reporting purposes).
    """
    na_counts=daily_closing.isna().sum()
    print("NA per ticker:\n", na_counts)

    closes_ffill=daily_closing.ffill().dropna(how='all')
    closes_ffill=closes_ffill.sort_index()
    print("After ffill → shape:", closes_ffill.shape)

    return closes_ffill, na_counts


def compute_daily_returns(df_closing: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily percentage returns from closing prices.

    Calculates the percentage change between consecutive closing prices.
    Drops rows where all tickers have NA values after computation.

    Args:
        df_closing: DataFrame with daily closing prices. Each column represents
                    a ticker, index is datetime.

    Returns:
        pd.DataFrame: DataFrame with daily returns (as fractions, e.g., 0.01 = 1%).
                     Rows where all tickers are NA are dropped.
    """
    returns=df_closing.pct_change().dropna(how='all')
    return returns


def monthly_aggregate(daily_closing: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aggregate daily closing prices to monthly frequency and compute monthly returns.

    Resamples daily data to month-end frequency by taking the last value of each month,
    then calculates monthly percentage returns. Returns are expressed as percentages
    (e.g., 5.0 means 5%).

    Args:
        daily_closing: DataFrame with daily closing prices. Each column represents
                       a ticker, index is datetime.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: A tuple containing:
            - monthly_close: DataFrame with month-end closing prices (resampled to 'ME').
            - monthly_return: DataFrame with monthly returns as percentages (multiplied by 100).
                            Rows where all tickers are NA are dropped.
    """
    monthly_close=daily_closing.resample('ME').last()
    monthly_return=monthly_close.pct_change().dropna(how='all') * 100.0
    return monthly_close, monthly_return
