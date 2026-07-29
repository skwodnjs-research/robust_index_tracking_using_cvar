import os

import numpy as np
import pandas as pd
import yfinance as yf
import FinanceDataReader as fdr

# ============================================================
# Configuration
# ============================================================

PERIOD = "20y"

STOCK_RETURN_FILE = "data/data_kospi_stocks.csv"
INDEX_RETURN_FILE = "data/data_kospi_index.csv"

os.makedirs("data", exist_ok=True)

# ============================================================
# Download KOSPI constituents
# ============================================================

print("Downloading KOSPI constituents...")

listing = fdr.StockListing("KOSPI")

# Convert 6-digit stock codes to Yahoo Finance tickers
tickers = listing["Code"].astype(str).str.zfill(6) + ".KS"
tickers = tickers.tolist()

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
    threads=True,
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
# Download KOSPI index
# ============================================================

print("Downloading KOSPI index...")

index_data = yf.download(
    "^KS11",
    period=PERIOD,
    interval="1d",
    auto_adjust=True,
    progress=False,
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
index_prices.name = "KOSPI"

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