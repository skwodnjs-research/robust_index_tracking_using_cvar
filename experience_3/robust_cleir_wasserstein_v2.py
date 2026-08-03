import time
import argparse

import cvxpy as cp
import numpy as np
import pandas as pd


parser = argparse.ArgumentParser()

parser.add_argument("--alpha", type=float, default=0.95)
parser.add_argument("--delta", type=float, default=0.01)

SELECTED = 100

# ============================================================
# 데이터 불러오기
# ============================================================

index = ["sp500", "russell2000", "kospi"][0]
INDEX = ["SP500", "Russell2000", "KOSPI"][0]

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

stock_prices = stock_prices.sort_index().replace([np.inf, -np.inf], np.nan)
index_prices = index_prices.sort_index().replace([np.inf, -np.inf], np.nan)

# 1. 날짜가 중복된 행이 있으면 첫 번째 행만 남기고 나머지는 삭제
stock_prices = stock_prices.loc[~stock_prices.index.duplicated(keep="first")]
index_prices = index_prices.loc[~index_prices.index.duplicated(keep="first")]

# 2. return 데이터 생성
stock_returns = stock_prices.pct_change(fill_method=None)
index_returns = index_prices.pct_change(fill_method=None)

stock_returns = stock_returns.iloc[1:]
index_returns = index_returns.iloc[1:]

# 3. 벤치마크가 NaN인 날짜 제거
index_returns = index_returns.dropna()

# 4. 모든 종목이 NaN인 날짜 제거
stock_returns = stock_returns.loc[~stock_returns.isna().all(axis=1)]

# 5. 주식 데이터와 벤치마크 데이터가 모두 유효한 날짜만을 사용
common_return_dates = stock_returns.index.intersection(index_returns.index)

stock_returns = stock_returns.loc[common_return_dates]
index_returns = index_returns.loc[common_return_dates]

print(
    "Loaded data\n"
    f"Stock returns      : {stock_returns.shape}\n"
    f"Benchmark returns  : {index_returns.shape}\n"
    f"Dates matched      : {stock_returns.index.equals(index_returns.index)}"
)

# ============================================================
# parameter 설정
# ============================================================

args = parser.parse_args()

alpha = args.alpha
delta = args.delta

ss = 1.5


# ============================================================
# CEIR 최적화 문제 해결 함수
# ============================================================

def solve_robust_cleir_wasserstein(
    stock_returns,
    index_returns,
    alpha=0.95,
    delta = 0.01,
    ss=1.5,
):

    R = stock_returns.to_numpy(dtype=float)
    Y = index_returns.to_numpy(dtype=float).reshape(-1, 1)
    xi = np.hstack((Y, R))

    n, m = R.shape

    # 최적화 변수
    w = cp.Variable(m)
    u = cp.Variable(m)
    w_tilde = cp.hstack([-1, w])
    
    gamma = cp.Variable()
    s = cp.Variable(n)

    lamb = cp.norm(w_tilde, 2) / (1 - alpha)

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
        gamma - gamma / (1 - alpha) - xi @ w_tilde / (1 - alpha) <= s,
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

weight_history = []
optimization_history = []

first_rebalance_year = stock_returns.index.min().year + 2
last_rebalance_year = min(stock_returns.index.max().year, 2025)
rebalance_years = range(first_rebalance_year, last_rebalance_year + 1)

start_time = time.perf_counter()

for rebalance_year in rebalance_years:

    training_year = rebalance_year - 1
    training_mask = (stock_returns.index.year == training_year)
    training_returns = stock_returns.loc[training_mask]
    training_benchmark = index_returns.loc[index_returns.index.year == training_year]

    if len(training_returns) == 0:
        print(f"{rebalance_year} skipped: no training data for {training_year}")
        continue

    holding_dates = stock_returns.index[stock_returns.index.year == rebalance_year]
    
    if len(holding_dates) == 0:
        print(f"{rebalance_year} skipped: no holding data")
        continue

    rebalance_date = holding_dates[0]

    # 투자 종목 설정
    valid_stocks = (training_returns.notna().mean(axis=0) >= 0.95)
    valid_columns = training_returns.columns[valid_stocks]

    selected_columns = valid_columns[:min(SELECTED, len(valid_columns))]

    if len(selected_columns) == 0:
        print(f"{rebalance_date.date()} skipped: no valid stocks")
        continue

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
        w_value, objective_value, status = solve_robust_cleir_wasserstein(
            stock_returns=training_returns,
            index_returns=training_benchmark,
            alpha=alpha,
            delta=delta,
            ss=ss
        )

    except RuntimeError as error:
        print(f"{rebalance_date.date()} skipped: {error}")
        continue

    solve_time = time.perf_counter() - solve_start_time
    
    weights = pd.Series(w_value, index=selected_columns, name=rebalance_date)

    # 향후 1개월 포트폴리오 수익률 계산
    holding_returns = stock_returns.loc[holding_dates, selected_columns]
    holding_returns = holding_returns.fillna(0.0)
    annual_portfolio_returns = holding_returns @ weights
    portfolio_returns.loc[holding_dates] = annual_portfolio_returns

    # 결과 저장
    weight_history.append(weights)

    optimization_history.append({
        "rebalance_date": rebalance_date,
        "training_year": training_year,
        "training_start": training_returns.index[0],
        "training_end": training_returns.index[-1],
        "holding_year": rebalance_year,
        "holding_start": holding_dates[0],
        "holding_end": holding_dates[-1],
        "training_days": len(training_returns),
        "holding_days": len(holding_dates),
        "valid_stocks": len(valid_columns),
        "number_of_stocks": int(np.count_nonzero(np.abs(w_value) > 1e-8)),
        "solve_time_seconds": solve_time,
        "objective": objective_value,
        "gross_exposure": float(np.abs(w_value).sum()),
        "short_exposure": float(-w_value[w_value < 0].sum()),
        "status": status,
    })

    print(
        f"{rebalance_date.date()} | "
        f"train={training_year} year "
        f"({len(training_returns):<d} days) | "
        f"hold={rebalance_year} "
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
benchmark_backtest_returns = index_returns.loc[portfolio_returns.index].rename("benchmark")

# 각 연도에 사용하지 않은 종목의 비중 = 0
weight_history = pd.DataFrame(weight_history).fillna(0.0)

weight_history.index.name = "rebalance_date"

optimization_history = pd.DataFrame(optimization_history).set_index("rebalance_date")

backtest_returns = pd.concat(
    [
        portfolio_returns,
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

weight_history.to_csv(f"../data/exp3_data_{index}_wass_cleir_alpha={alpha}_delta={delta}_weights_v2.csv")
optimization_history.to_csv(f"../data/exp3_data_{index}_wass_cleir_alpha={alpha}_delta={delta}_optimization_history_v2.csv")
backtest_returns.to_csv(f"../data/exp3_data_{index}_wass_cleir_alpha={alpha}_delta={delta}_returns_v2.csv")

print("\nFiles saved")

print(f"../data/exp3_data_{index}_wass_cleir_alpha={alpha}_delta={delta}_weights_v2.csv")
print(f"../data/exp3_data_{index}_wass_cleir_alpha={alpha}_delta={delta}_optimization_history_v2.csv")
print(f"../data/exp3_data_{index}_wass_cleir_alpha={alpha}_delta={delta}_returns_v2.csv")