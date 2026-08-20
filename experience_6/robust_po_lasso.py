import time
import argparse

import cvxpy as cp
import numpy as np
import pandas as pd


# ============================================================
# Argument
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument("--alpha", type=float, default=0.95)
parser.add_argument("--delta", type=float, default=0.01)


# ============================================================
# 투자 대상 종목
# ============================================================

SELECTED = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOG",
    "META", "BRK-B", "JPM", "JNJ", "V",
    "PG", "XOM", "UNH", "HD", "MA",
    "BAC", "ABBV", "KO", "PEP", "COST",
    "WMT", "NFLX", "CRM", "CSCO", "ACN",
    "MCD", "DIS", "ADBE", "AMD", "INTC",
]


# 선택 종목이 필요한 기간 중
# 최소 몇 %의 데이터를 가지고 있어야 하는지
MIN_AVAILABILITY = 0.95


# ============================================================
# 데이터 불러오기
# ============================================================

index = "sp500"
INDEX = "SP500"


stock_prices = pd.read_csv(
    f"../data/data_{index}_stocks.csv",
    index_col=0,
    parse_dates=True,
)


index_prices = pd.read_csv(
    f"../data/data_{index}_index.csv",
    index_col=0,
    parse_dates=True,
)[INDEX]


# ============================================================
# 데이터 전처리
# ============================================================

stock_prices = (
    stock_prices
    .sort_index()
    .replace([np.inf, -np.inf], np.nan)
)

index_prices = (
    index_prices
    .sort_index()
    .replace([np.inf, -np.inf], np.nan)
)


# 날짜가 중복된 경우 첫 번째 행만 사용
stock_prices = stock_prices.loc[
    ~stock_prices.index.duplicated(keep="first")
]

index_prices = index_prices.loc[
    ~index_prices.index.duplicated(keep="first")
]


# ============================================================
# 선택 종목 데이터 확인
# ============================================================

# 첫 리밸런싱: 2015-01
# 학습기간: 직전 12개월
# 따라서 필요한 최초 데이터: 2014-01
#
# 마지막 리밸런싱: 2020-12
# 따라서 필요한 마지막 데이터: 2020-12

required_start = pd.Timestamp("2014-01-01")
required_end = pd.Timestamp("2020-12-31")


# ------------------------------------------------------------
# 1. CSV에 선택한 티커가 존재하는지 확인
# ------------------------------------------------------------

missing_tickers = [
    ticker
    for ticker in SELECTED
    if ticker not in stock_prices.columns
]


print()
print("============================================================")
print("              Selected stock data check")
print("============================================================")

print(f"Required period   : {required_start.date()} ~ {required_end.date()}")
print(f"Selected stocks   : {len(SELECTED)}")
print(f"Min availability  : {MIN_AVAILABILITY:.0%}")
print()


if missing_tickers:

    print("Tickers not found in CSV:")

    for ticker in missing_tickers:
        print(f"  {ticker}")

    print()

    raise ValueError(
        "The selected ticker list contains stocks "
        "that do not exist in stock_prices."
    )


# ------------------------------------------------------------
# 2. 필요한 기간의 가격 데이터만 추출
# ------------------------------------------------------------

selected_required_prices = stock_prices.loc[
    required_start:required_end,
    SELECTED,
]


if len(selected_required_prices) == 0:
    raise ValueError(
        "No stock price data exists in the required period."
    )


# ------------------------------------------------------------
# 3. 종목별 데이터 보유 현황 확인
# ------------------------------------------------------------

problem_stocks = []


print(
    f"{'Ticker':<8}"
    f"{'First':<12}"
    f"{'Last':<12}"
    f"{'Missing':>10}"
    f"{'Availability':>15}"
)

print("-" * 57)


for ticker in SELECTED:

    series = selected_required_prices[ticker]

    first_valid = series.first_valid_index()
    last_valid = series.last_valid_index()

    missing_count = int(series.isna().sum())
    availability = float(series.notna().mean())


    first_text = (
        first_valid.date().isoformat()
        if first_valid is not None
        else "None"
    )

    last_text = (
        last_valid.date().isoformat()
        if last_valid is not None
        else "None"
    )


    print(
        f"{ticker:<8}"
        f"{first_text:<12}"
        f"{last_text:<12}"
        f"{missing_count:>10d}"
        f"{availability:>14.2%}"
    )


    if availability < MIN_AVAILABILITY:
        problem_stocks.append(ticker)


# ------------------------------------------------------------
# 4. 검사 결과
# ------------------------------------------------------------

print()
print("------------------------------------------------------------")


if len(problem_stocks) == 0:

    print("Data completeness : OK")

    print(
        f"All selected stocks have at least "
        f"{MIN_AVAILABILITY:.0%} data availability "
        f"during the required period."
    )


else:

    print("Data completeness : FAILED")

    print(
        "Stocks below minimum availability: "
        + ", ".join(problem_stocks)
    )

    raise ValueError(
        "Some selected stocks do not have enough data "
        "for the required backtest period."
    )


# ============================================================
# Return 데이터 생성
# ============================================================

stock_returns = stock_prices.pct_change(
    fill_method=None
)

index_returns = index_prices.pct_change(
    fill_method=None
)


# pct_change 때문에 첫 번째 행은 NaN
stock_returns = stock_returns.iloc[1:]
index_returns = index_returns.iloc[1:]


# ============================================================
# 벤치마크와 주식 수익률 날짜 정리
# ============================================================

# 벤치마크 수익률이 NaN인 날짜 제거
index_returns = index_returns.dropna()


# 모든 종목의 수익률이 NaN인 날짜 제거
stock_returns = stock_returns.loc[
    ~stock_returns.isna().all(axis=1)
]


# 주식 데이터와 벤치마크 데이터가 모두 존재하는 날짜만 사용
common_return_dates = stock_returns.index.intersection(
    index_returns.index
)


stock_returns = stock_returns.loc[
    common_return_dates
]

index_returns = index_returns.loc[
    common_return_dates
]


print()
print("============================================================")
print("                     Loaded data")
print("============================================================")

print(f"Stock returns      : {stock_returns.shape}")
print(f"Benchmark returns  : {index_returns.shape}")
print(
    f"Dates matched      : "
    f"{stock_returns.index.equals(index_returns.index)}"
)


# ============================================================
# Parameter 설정
# ============================================================

args = parser.parse_args()

alpha = args.alpha
delta = args.delta


# ============================================================
# CEIR 최적화 문제 해결 함수
# ============================================================

def solve_robust_po_lasso(
    stock_returns,
    alpha=0.95,
    delta = 0.01,
    ss = 1.5,
):

    xi = stock_returns.to_numpy(dtype=float)

    n, m = xi.shape


    # 최적화 변수
    w = cp.Variable(m)
    u = cp.Variable(m)
    
    gamma = cp.Variable()
    s = cp.Variable(n)

    lamb = cp.norm(w, 2) / (1 - alpha)

    # 목적함수
    objective = cp.Minimize(lamb * delta + cp.sum(s) / n)

    # 제약조건
    constraints = [
        cp.sum(w) == 1,
        u >= w,
        u >= -w,
        cp.sum(u) <= ss,

        # k = 1
        gamma <= s,

        # k = 2
        gamma - gamma / (1 - alpha) - xi @ w / (1 - alpha) <= s,
    ]

    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.MOSEK)

    if problem.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(
            f"Optimization failed: "
            f"{problem.status}"
        )

    if w.value is None:
        raise RuntimeError(
            "Optimization returned no weights."
        )

    return (
        np.asarray(w.value).reshape(-1),
        problem.value,
        problem.status,
    )

# ============================================================
# Rolling-window
# ============================================================

portfolio_returns = pd.Series(
    index=stock_prices.index,
    dtype=float,
    name="portfolio",
)

equal_weight_returns = pd.Series(
    index=stock_prices.index,
    dtype=float,
    name="equal_weight",
)

weight_history = []
optimization_history = []

rebalance_months = pd.period_range(start="2015-01", end="2020-12", freq="M")

start_time = time.perf_counter()

for rebalance_month in rebalance_months:

    training_end_month = rebalance_month - 1
    training_start_month = rebalance_month - 12

    training_mask = (
        (stock_returns.index.to_period("M") >= training_start_month)
        & (stock_returns.index.to_period("M") <= training_end_month)
    )

    training_returns = stock_returns.loc[training_mask]

    training_benchmark_mask = (
        (index_returns.index.to_period("M") >= training_start_month)
        & (index_returns.index.to_period("M") <= training_end_month)
    )

    training_benchmark = index_returns.loc[training_benchmark_mask]

    if len(training_returns) == 0:
        print(f"{rebalance_month} skipped: no training data for {training_start_month} ~ {training_end_month}")
        continue

    holding_dates = stock_returns.index[stock_returns.index.to_period("M") == rebalance_month]
    
    if len(holding_dates) == 0:
        print(f"{rebalance_month} skipped: no holding data")
        continue

    rebalance_date = holding_dates[0]

    # 투자 종목 설정
    selected_columns = SELECTED

    # 학습 데이터 설정
    training_returns = training_returns.loc[:, selected_columns].fillna(0.0)

    common_training_dates = training_returns.index.intersection(training_benchmark.index)

    training_returns = training_returns.loc[common_training_dates]
    training_benchmark = training_benchmark.loc[common_training_dates]

    if len(training_returns) == 0:
        print(f"{rebalance_date.date()} skipped: no matched training dates")
        continue

    # 최적화
    solve_start_time = time.perf_counter()

    try:
        w_value, objective_value, status = solve_robust_po_lasso(
            stock_returns=training_returns,
            alpha=alpha,
            delta=delta,
        )

    except RuntimeError as error:
        print(f"{rebalance_date.date()} skipped: {error}")
        continue

    solve_time = time.perf_counter() - solve_start_time
    
    weights = pd.Series(
        w_value,
        index=selected_columns,
        name=rebalance_date,
    )

    # 향후 1개월 포트폴리오 수익률 계산
    holding_returns = stock_returns.loc[
        holding_dates,
        selected_columns,
    ].fillna(0.0)

    # Robust CEIR portfolio
    monthly_portfolio_returns = holding_returns @ weights
    portfolio_returns.loc[holding_dates] = monthly_portfolio_returns

    # Equal-weight portfolio
    equal_weight = np.repeat(
        1.0 / len(selected_columns),
        len(selected_columns),
    )

    monthly_equal_weight_returns = holding_returns @ equal_weight
    equal_weight_returns.loc[holding_dates] = monthly_equal_weight_returns

    # 결과 저장
    weight_history.append(weights)

    optimization_history.append({
        "rebalance_date": rebalance_date,
        "training_start_month": training_start_month,
        "training_end_month": training_end_month,
        "training_start": training_returns.index[0],
        "training_end": training_returns.index[-1],
        "holding_year": rebalance_month,
        "holding_start": holding_dates[0],
        "holding_end": holding_dates[-1],
        "training_days": len(training_returns),
        "holding_days": len(holding_dates),
        "available_stocks": len(selected_columns),
        "number_of_stocks": int(np.count_nonzero(np.abs(w_value) > 1e-8)),
        "solve_time_seconds": solve_time,
        "objective": objective_value,
        "gross_exposure": float(np.abs(w_value).sum()),
        "short_exposure": float(-w_value[w_value < 0].sum()),
        "status": status,
    })

    print(
        f"{rebalance_date.date()} | "
        f"train={training_start_month} ~ {training_end_month} "
        f"({len(training_returns):<d} days) | "
        f"hold={rebalance_month} "
        f"({len(holding_dates):<3d} days) | "
        f"available={len(selected_columns):<3d} | "
        f"selected="
        f"{np.count_nonzero(np.abs(w_value) > 1e-8):<3d} | "
        f"time={solve_time:.3f}s"
    )

total_time = time.perf_counter() - start_time

# ============================================================
# 결과 정리
# ============================================================

portfolio_returns = portfolio_returns.dropna()

equal_weight_returns = equal_weight_returns.loc[
    portfolio_returns.index
]

benchmark_backtest_returns = index_returns.loc[
    portfolio_returns.index
].rename("benchmark")

# 각 연도에 사용하지 않은 종목의 비중 = 0
weight_history = pd.DataFrame(weight_history).fillna(0.0)

weight_history.index.name = "rebalance_date"

optimization_history = pd.DataFrame(optimization_history).set_index("rebalance_date")

backtest_returns = pd.concat(
    [
        portfolio_returns,
        equal_weight_returns,
        benchmark_backtest_returns,
    ],
    axis=1,
)


# ============================================================
# 결과 출력
# ============================================================

print("\n============================================================")
print("                      Backtest summary                      ")
print("============================================================")

print()

print(f"Total solve time : {optimization_history['solve_time_seconds'].sum():.2f} seconds")
print(f"Average time     : {optimization_history['solve_time_seconds'].mean():.3f} seconds")
print(f"Total run time   : {total_time:.2f} seconds")

print()

print(f"Start date       : {portfolio_returns.index.min().date()}")
print(f"End date         : {portfolio_returns.index.max().date()}")
print(f"Rebalancings     : {len(weight_history)}")
print(f"Average stocks   : {optimization_history['number_of_stocks'].mean():.2f}")
print(f"Minimum stocks   : {optimization_history['number_of_stocks'].min()}")
print(f"Maximum stocks   : {optimization_history['number_of_stocks'].max()}")


# ============================================================
# 저장
# ============================================================

weight_history.to_csv(f"../data/exp6_data_{index}_robust_po_lasso_alpha={alpha}_delta={delta}_weights.csv")
optimization_history.to_csv(f"../data/exp6_data_{index}_robust_po_lasso_alpha={alpha}_delta={delta}_optimization_history.csv")
backtest_returns.to_csv(f"../data/exp6_data_{index}_robust_po_lasso_alpha={alpha}_delta={delta}_returns.csv")

print("\nFiles saved")

print(f"../data/exp6_data_{index}_robust_po_lasso_alpha={alpha}_delta={delta}_weights.csv")
print(f"../data/exp6_data_{index}_robust_po_lasso_alpha={alpha}_delta={delta}_optimization_history.csv")
print(f"../data/exp6_data_{index}_robust_po_lasso_alpha={alpha}_delta={delta}_returns.csv")