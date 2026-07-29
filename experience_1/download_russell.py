import csv
import os
import time
from io import StringIO

import numpy as np
import pandas as pd
import requests
import yfinance as yf


# ============================================================
# 설정
# ============================================================

PERIOD = "20y"
BATCH_SIZE = 100

STOCK_PRICE_FILE = "data/data_russell2000_stocks.csv"
INDEX_PRICE_FILE = "data/data_russell2000_index.csv"

IWM_HOLDINGS_URLS = [
    (
        "https://www.ishares.com/us/products/239710/"
        "ishares-russell-2000-etf/latest-holdings.csv"
    ),
    (
        "https://www.ishares.com/us/products/239710/"
        "ishares-russell-2000-etf/1467271812596.ajax"
        "?fileType=csv"
        "&fileName=IWM_holdings"
        "&dataType=fund"
    ),
]

os.makedirs("data", exist_ok=True)


# ============================================================
# IWM 보유 종목 다운로드
# ============================================================

def download_iwm_holdings():
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": (
            "https://www.ishares.com/us/products/239710/"
            "ishares-russell-2000-etf"
        ),
    }

    for url in IWM_HOLDINGS_URLS:
        try:
            response = requests.get(
                url,
                headers=headers,
                timeout=60,
            )
            response.raise_for_status()

            text = response.content.decode(
                "utf-8-sig",
                errors="replace",
            ).replace("\x00", "")

            lines = text.splitlines()
            header_row = None

            for i, line in enumerate(lines):
                try:
                    columns = next(csv.reader([line]))
                except csv.Error:
                    continue

                columns = [
                    str(column)
                    .replace("\ufeff", "")
                    .strip()
                    for column in columns
                ]

                lower_columns = {
                    column.lower()
                    for column in columns
                }

                if (
                    "ticker" in lower_columns
                    and "name" in lower_columns
                ):
                    header_row = i
                    break

            if header_row is None:
                continue

            holdings = pd.read_csv(
                StringIO(
                    "\n".join(lines[header_row:])
                )
            )

            holdings.columns = [
                str(column)
                .replace("\ufeff", "")
                .strip()
                for column in holdings.columns
            ]

            ticker_column = next(
                (
                    column
                    for column in holdings.columns
                    if column.lower() == "ticker"
                ),
                None,
            )

            if ticker_column is None:
                continue

            if ticker_column != "Ticker":
                holdings = holdings.rename(
                    columns={ticker_column: "Ticker"}
                )

            return holdings

        except requests.RequestException:
            continue

    raise RuntimeError("IWM 보유 종목 CSV를 다운로드하지 못했습니다.")


# ============================================================
# 티커 정리
# ============================================================

def get_tickers(holdings):
    mask = pd.Series(
        True,
        index=holdings.index,
    )

    asset_class_column = next(
        (
            column
            for column in holdings.columns
            if column.lower() == "asset class"
        ),
        None,
    )

    if asset_class_column is not None:
        mask &= (
            holdings[asset_class_column]
            .astype(str)
            .str.strip()
            .str.lower()
            .eq("equity")
        )

    tickers = (
        holdings.loc[mask, "Ticker"]
        .dropna()
        .astype(str)
        .str.strip()
        .str.upper()
        .str.replace(".", "-", regex=False)
        .str.replace("/", "-", regex=False)
        .str.replace(" ", "", regex=False)
    )

    valid_mask = tickers.str.fullmatch(
        r"[A-Z][A-Z0-9\-]*",
        na=False,
    )

    tickers = tickers.loc[valid_mask]

    tickers = tickers[
        ~tickers.isin(
            {
                "",
                "-",
                "USD",
                "CASH",
            }
        )
    ]

    return (
        tickers
        .drop_duplicates()
        .sort_values()
        .tolist()
    )


# ============================================================
# yfinance Close 가격 추출
# ============================================================

def extract_close(data, requested_tickers):
    if data is None or data.empty:
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        level_0 = data.columns.get_level_values(0)
        level_1 = data.columns.get_level_values(1)

        if "Close" in level_0:
            close = data["Close"].copy()

        elif "Close" in level_1:
            close = data.xs(
                "Close",
                axis=1,
                level=1,
            ).copy()

        else:
            return pd.DataFrame()

    else:
        if "Close" not in data.columns:
            return pd.DataFrame()

        close = data[["Close"]].copy()

        if len(requested_tickers) == 1:
            close.columns = requested_tickers

    if isinstance(close, pd.Series):
        close = close.to_frame()

    close.columns = [
        str(column).upper()
        for column in close.columns
    ]

    return close


# ============================================================
# 종목 가격 다운로드
# ============================================================

def download_stock_prices(tickers):
    batches = []

    for start in range(
        0,
        len(tickers),
        BATCH_SIZE,
    ):
        batch = tickers[
            start:start + BATCH_SIZE
        ]

        batch_number = (
            start // BATCH_SIZE + 1
        )

        total_batches = (
            len(tickers)
            + BATCH_SIZE
            - 1
        ) // BATCH_SIZE

        print(f"Downloading stocks {batch_number}/{total_batches}")

        try:
            data = yf.download(
                tickers=batch,
                period=PERIOD,
                interval="1d",
                auto_adjust=True,
                progress=True,
                threads=True,
                group_by="column",
            )

            close = extract_close(
                data,
                batch,
            )

            if not close.empty:
                batches.append(close)

        except Exception as error:
            print(f"Batch failed: {error}")

        time.sleep(0.5)

    if not batches:
        raise RuntimeError("주식 가격을 하나도 다운로드하지 못했습니다.")

    prices = pd.concat(
        batches,
        axis=1,
    )

    prices = prices.loc[
        :,
        ~prices.columns.duplicated()
    ]

    prices = (
        prices
        .sort_index()
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
    )

    # 전 기간 데이터가 전혀 없는 종목만 제거
    prices = prices.dropna(
        axis=1,
        how="all",
    )

    return prices


# ============================================================
# Download Russell 2000 index
# ============================================================

def download_index_prices():
    data = yf.download(
        "^RUT",
        period=PERIOD,
        interval="1d",
        auto_adjust=True,
        progress=True,
        threads=False,
    )

    close = extract_close(
        data,
        ["^RUT"],
    )

    if close.empty:
        raise RuntimeError("Russell 2000 지수를 다운로드하지 못했습니다.")

    index_price = (
        close
        .iloc[:, 0]
        .sort_index()
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    index_price.name = "Russell2000"

    return index_price


# ============================================================
# Download Russell 2000 constituents
# ============================================================

print("Downloading Russell 2000 constituents...")

holdings = download_iwm_holdings()
tickers = get_tickers(holdings)

print(f"Found {len(tickers)} stocks.")


# ============================================================
# Download stock prices
# ============================================================

print("Downloading stock prices...")

stock_prices = download_stock_prices(tickers)

print(f"Downloaded {stock_prices.shape[1]} stocks.")


# ============================================================
# Download Russell 2000 index
# ============================================================

print("Downloading Russell 2000 index...")

index_prices = download_index_prices()


# ============================================================
# Prepare price data
# ============================================================

print("Preparing price data...")

stock_prices = stock_prices.copy()
index_prices = index_prices.copy()


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

stock_prices.to_csv(STOCK_PRICE_FILE)
index_prices.to_csv(INDEX_PRICE_FILE)


# ============================================================
# Summary
# ============================================================

print("\nDownload completed")
print(f"Stock prices      : {stock_prices.shape}")
print(f"Index prices      : {index_prices.shape}")
print(f"Dates matched     : {stock_prices.index.equals(index_prices.index)}")
print(f"Missing values    : {stock_prices.isna().sum().sum():,}")
print(f"Stock file        : {STOCK_PRICE_FILE}")
print(f"Index file        : {INDEX_PRICE_FILE}")