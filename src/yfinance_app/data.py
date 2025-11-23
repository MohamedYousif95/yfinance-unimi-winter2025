"""Data fetching module for Yahoo Finance data using yfinance library."""

from pathlib import Path
import pandas as pd
import yfinance as yf



def load_yfinance_data(
        tickers,
        start,
        end,
        interval="1d",
        auto_adjust=True,
        group_by='ticker',
        progress=False,
        threads=True,
        **download_kwargs,
) -> pd.DataFrame:
    """
    Load historical price data from Yahoo Finance using yfinance library.

    Downloads historical market data for specified tickers over a given time period.
    The data is normalized to remove timezone information from the index.

    Args:
        tickers: List of ticker symbols (e.g., ['AAPL', 'MSFT']) or single ticker string.
        start: Start date for data download (format: 'YYYY-MM-DD' or datetime).
        end: End date for data download (format: 'YYYY-MM-DD' or datetime).
        interval: Data interval. Valid periods: '1d', '5d', '1wk', '1mo', '3mo'.
                  Default is '1d' (daily).
        auto_adjust: Whether to adjust prices for splits and dividends. Default is True.
        group_by: How to group columns. Default is 'ticker'.
        progress: Whether to show download progress. Default is False.
        threads: Whether to use multi-threading for downloads. Default is True.
        **download_kwargs: Additional keyword arguments passed to yf.download().

    Returns:
        pd.DataFrame: DataFrame containing historical price data with datetime index.
                      Columns depend on the data structure (MultiIndex for multiple tickers).
    """

    print(
        f"Downloading data for tickers={tickers}, "
        f"start={start}, end={end}, interval={interval}"
    )

    df = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=auto_adjust,
        group_by=group_by,
        progress=progress,
        threads=threads,
        **download_kwargs,
    )

    df.index = pd.to_datetime(df.index).tz_localize(None)
    print("Columns type:", type(df.columns), "Shape:", df.shape)
    return df


def save_yfinance_data(df: pd.DataFrame, path: Path) -> None:
    """
    Save price data to a CSV file.

    Creates parent directories if they don't exist and saves the DataFrame
    to the specified CSV path.

    Args:
        df: DataFrame containing price data to save.
        path: Destination file path (Path object). Parent directories will be created
              if they don't exist.

    Returns:
        None: Function saves data to file but returns nothing.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)