import os
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf

# ============================================================
# Configuration
# ============================================================

PERIOD = "20y"

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

STOCK_RETURN_FILE = DATA_DIR / "data_sp500_stocks.csv"
INDEX_RETURN_FILE = DATA_DIR / "data_sp500_index.csv"

# ============================================================
# Download S&P 500 constituents
# ============================================================

print("Downloading S&P 500 constituents...")

url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
response.raise_for_status()

table = pd.read_html(StringIO(response.text))[0]

tickers = (
    table["Symbol"]
    .dropna()
    .str.upper()
    .str.replace(".", "-", regex=False)
    .tolist()
)

print(f"Found {len(tickers)} stocks.")

# ============================================================
# Download stock prices
# ============================================================

print("Downloading stock prices...")

stock_data = yf.download(
    tickers,
    period=PERIOD,
    interval="1d",
    auto_adjust=True,
    progress=True,
    threads=True
)

if isinstance(stock_data.columns, pd.MultiIndex):
    prices = stock_data["Close"]
else:
    prices = stock_data

prices = prices.sort_index()
prices = prices.replace([np.inf, -np.inf], np.nan)

# Remove only stocks with no observations at all
prices = prices.dropna(axis=1, how="all")

print(f"Downloaded {prices.shape[1]} stocks.")

# ============================================================
# Download S&P 500 index
# ============================================================

print("Downloading S&P 500 index...")

index_data = yf.download(
    "^GSPC",
    period=PERIOD,
    interval="1d",
    auto_adjust=True,
    progress=False
)

if isinstance(index_data.columns, pd.MultiIndex):
    index_price = index_data["Close"].iloc[:, 0]
else:
    index_price = index_data["Close"]

index_price = index_price.sort_index()
index_price = index_price.replace([np.inf, -np.inf], np.nan).dropna()

# ============================================================
# Prepare price data
# ============================================================

print("Preparing price data...")

stock_prices = prices.copy()

index_prices = index_price.copy()
index_prices.name = "SP500"

# ============================================================
# Align dates
# ============================================================

print("Aligning dates...")

common_dates = stock_prices.index.intersection(index_prices.index)

stock_prices = stock_prices.loc[common_dates]
index_prices = index_prices.loc[common_dates]

# Remove only dates where every stock price is missing
valid_dates = ~stock_prices.isna().all(axis=1)

stock_prices = stock_prices.loc[valid_dates]
index_prices = index_prices.loc[stock_prices.index]

# ============================================================
# Save
# ============================================================

print("Saving files...")

stock_prices.to_csv(STOCK_RETURN_FILE)
index_prices.to_csv(INDEX_RETURN_FILE)

# ============================================================
# Summary
# ============================================================

print("\nDownload completed")
print(f"Stock prices      : {stock_prices.shape}")
print(f"Index prices      : {index_prices.shape}")
print(f"Dates matched     : {stock_prices.index.equals(index_prices.index)}")
print(f"Missing values    : {stock_prices.isna().sum().sum():,}")
print(f"Stock file        : {STOCK_RETURN_FILE}")
print(f"Index file        : {INDEX_RETURN_FILE}")