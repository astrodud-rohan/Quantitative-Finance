"""
Data Module
===========
Downloads historical OHLCV data for NSE-listed stocks using yfinance.
NSE tickers on Yahoo Finance use the ".NS" suffix (e.g., "HDFCBANK.NS").

Pre-screened cointegrated pairs from the same sector:
  Banking  : HDFCBANK, ICICIBANK, AXISBANK, KOTAKBANK, SBIN
  IT       : INFY, TCS, WIPRO, HCLTECH
  FMCG     : HINDUNILVR, NESTLEIND, BRITANNIA
  Pharma   : SUNPHARMA, DRREDDY, CIPLA, DIVISLAB
  Energy   : RELIANCE, ONGC, BPCL
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

# Pre-screened pairs known to show cointegration (use as defaults)
CANDIDATE_PAIRS = [
    ("HDFCBANK.NS", "ICICIBANK.NS"),  # Large private banks
    ("INFY.NS", "WIPRO.NS"),           # Mid-large IT
    ("SUNPHARMA.NS", "DRREDDY.NS"),    # Large pharma
    ("HINDUNILVR.NS", "BRITANNIA.NS"), # FMCG
    ("TCS.NS", "HCLTECH.NS"),          # Large IT
]

# For quick testing with liquid US stocks (if NSE data is patchy)
US_CANDIDATE_PAIRS = [
    ("GLD", "SLV"),    # Gold vs Silver ETFs
    ("XOM", "CVX"),    # Major oil companies
    ("KO", "PEP"),     # Beverage giants
    ("JPM", "BAC"),    # Large US banks
    ("MSFT", "GOOGL"), # Tech giants
]

def download_prices(
    tickers: list[str],
    start_date: str = None,
    end_date: str = None,
    years: int = 5,
) -> pd.DataFrame:
    """
    Download adjusted closing prices for a list of tickers.

    Parameters
    ----------
    tickers    : list of Yahoo Finance ticker symbols
    start_date : 'YYYY-MM-DD' string (optional, overrides years)
    end_date   : 'YYYY-MM-DD' string (optional, defaults to today)
    years      : number of years of history to download

    Returns
    -------
    DataFrame with date index and one column per ticker (Adj Close prices)
    """
    if end_date is None:
        end_date = datetime.today().strftime("%Y-%m-%d")

    if start_date is None:
        start_dt = datetime.today() - timedelta(days=years * 365)
        start_date = start_dt.strftime("%Y-%m-%d")

    print(f"Downloading {len(tickers)} tickers from {start_date} to {end_date}...")

    raw = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False,
    )

    # Extract closing prices
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})

    # Drop columns with too many NaNs (>5% missing)
    threshold = 0.05
    prices = prices.loc[:, prices.isna().mean() < threshold]

    # Forward-fill remaining gaps (weekends, holidays)
    prices = prices.ffill().dropna()

    print(f"Downloaded: {prices.shape[0]} trading days × {prices.shape[1]} tickers")
    print(f"Date range: {prices.index[0].date()} to {prices.index[-1].date()}")

    return prices

def get_pair_data(
    ticker_a: str,
    ticker_b: str,
    years: int = 4,
) -> tuple[pd.Series, pd.Series]:
    """
    Download and align price data for a single pair.

    Returns
    -------
    (prices_a, prices_b) as aligned pd.Series
    """
    prices = download_prices([ticker_a, ticker_b], years=years)

    if ticker_a not in prices.columns or ticker_b not in prices.columns:
        raise ValueError(
            f"Could not download data for {ticker_a} or {ticker_b}. "
            "Check ticker symbols — NSE stocks need '.NS' suffix."
        )

    prices_a = prices[ticker_a]
    prices_b = prices[ticker_b]

    # Align
    common_idx = prices_a.index.intersection(prices_b.index)
    return prices_a[common_idx], prices_b[common_idx]

def get_sector_universe(sector: str = "banking", years: int = 4) -> pd.DataFrame:
    """
    Download a full sector universe for pair screening.

    Sectors: 'banking', 'it', 'pharma', 'fmcg', 'energy', 'us_etf'
    """
    universes = {
        "banking": ["HDFCBANK.NS", "ICICIBANK.NS", "AXISBANK.NS", "KOTAKBANK.NS", "SBIN.NS"],
        "it": ["INFY.NS", "TCS.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS"],
        "pharma": ["SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS"],
        "fmcg": ["HINDUNILVR.NS", "NESTLEIND.NS", "BRITANNIA.NS", "DABUR.NS"],
        "energy": ["RELIANCE.NS", "ONGC.NS", "BPCL.NS", "IOC.NS"],
        "us_etf": ["GLD", "SLV", "XOM", "CVX", "KO", "PEP", "JPM", "BAC"],
    }

    if sector not in universes:
        raise ValueError(f"Sector must be one of: {list(universes.keys())}")

    tickers = universes[sector]
    return download_prices(tickers, years=years)