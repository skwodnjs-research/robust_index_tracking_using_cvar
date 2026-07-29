import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# 결과 불러오기
# ============================================================

index = 0
INDEX = ["sp500", "russell2000", "kospi"][index]


# CEIR + Wasserstein
robust_ceir_wasserstein_weight_history = pd.read_csv(
    f"data/data_{INDEX}_robust_ceir_wasserstein_weights.csv",
    index_col=0,
    parse_dates=True
)

robust_ceir_wasserstein_optimization_history = pd.read_csv(
    f"data/data_{INDEX}_robust_ceir_wasserstein_optimization_history.csv",
    index_col=0,
    parse_dates=True
)

robust_ceir_wasserstein_backtest_returns = pd.read_csv(
    f"data/data_{INDEX}_robust_ceir_wasserstein_returns.csv",
    index_col=0,
    parse_dates=True
)

# CLEIR + Wasserstein
robust_cleir_wasserstein_weight_history = pd.read_csv(
    f"data/data_{INDEX}_robust_cleir_wasserstein_weights.csv",
    index_col=0,
    parse_dates=True
)

robust_cleir_wasserstein_optimization_history = pd.read_csv(
    f"data/data_{INDEX}_robust_cleir_wasserstein_optimization_history.csv",
    index_col=0,
    parse_dates=True
)

robust_cleir_wasserstein_backtest_returns = pd.read_csv(
    f"data/data_{INDEX}_robust_cleir_wasserstein_returns.csv",
    index_col=0,
    parse_dates=True
)

# ============================================================
# 계산
# ============================================================

robust_ceir_wasserstein_cumulative = (1 + robust_ceir_wasserstein_backtest_returns).cumprod()
robust_cleir_wasserstein_cumulative = (1 + robust_cleir_wasserstein_backtest_returns).cumprod()

tol = 1e-6
robust_ceir_wasserstein_l0_norm = (robust_ceir_wasserstein_weight_history.abs() > tol).sum(axis=1)
robust_cleir_wasserstein_l0_norm = (robust_cleir_wasserstein_weight_history.abs() > tol).sum(axis=1)


# ============================================================
# 1. 누적수익률
# ============================================================

plt.figure(figsize=(12, 6))

common_index = (
    robust_ceir_wasserstein_cumulative.index
    .intersection(robust_cleir_wasserstein_cumulative.index)
)

robust_ceir_wasserstein_cumulative = robust_ceir_wasserstein_cumulative.loc[common_index]
robust_cleir_wasserstein_cumulative = robust_cleir_wasserstein_cumulative.loc[common_index]

plt.plot(
    common_index,
    robust_ceir_wasserstein_cumulative["benchmark"],
    label=INDEX
)

plt.plot(
    common_index,
    robust_ceir_wasserstein_cumulative["portfolio"],
    label="CEIR + Wasserstein Portfolio (delta=0.01)"
)

plt.plot(
    common_index,
    robust_cleir_wasserstein_cumulative["portfolio"],
    label="CLEIR + Wasserstein Portfolio (delta=0.01)"
)

plt.xlabel("Date")
plt.ylabel("Cumulative Value")
plt.title("Out-of-sample Backtest")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

# ============================================================
# 2. 사용 종목 수
# ============================================================

plt.figure(figsize=(12, 5))

common_index = (
    robust_ceir_wasserstein_l0_norm.index
    .intersection(robust_cleir_wasserstein_l0_norm.index)
)

robust_ceir_wasserstein_l0_norm = robust_ceir_wasserstein_l0_norm.loc[common_index]
robust_cleir_wasserstein_l0_norm = robust_cleir_wasserstein_l0_norm.loc[common_index]

plt.step(
    robust_ceir_wasserstein_l0_norm.index,
    robust_ceir_wasserstein_l0_norm,
    where="post",
    label="CEIR + Wasserstein (delta=0.01)"
)

plt.step(
    robust_cleir_wasserstein_l0_norm.index,
    robust_cleir_wasserstein_l0_norm,
    where="post",
    label="CLEIR + Wasserstein (delta=0.01)"
)

plt.xlabel("Rebalancing Date")
plt.ylabel(r"$\|w\|_0$")
plt.title("Number of Stocks")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

# ============================================================
# 3. 월별 최적화 계산 시간
# ============================================================

if ("solve_time_seconds" in robust_ceir_wasserstein_optimization_history.columns
    and "solve_time_seconds" in robust_cleir_wasserstein_optimization_history.columns
):
    plt.figure(figsize=(12, 5))

    common_index = (
        robust_ceir_wasserstein_optimization_history.index
        .intersection(robust_cleir_wasserstein_optimization_history.index)
    )
    robust_ceir_wasserstein_optimization_history = robust_ceir_wasserstein_optimization_history.loc[common_index]
    robust_cleir_wasserstein_optimization_history = robust_cleir_wasserstein_optimization_history.loc[common_index]

    plt.plot(
        robust_ceir_wasserstein_optimization_history.index,
        robust_ceir_wasserstein_optimization_history["solve_time_seconds"],
        label="CEIR + Wasserstein (delta=0.01)"
    )

    plt.plot(
        robust_cleir_wasserstein_optimization_history.index,
        robust_cleir_wasserstein_optimization_history["solve_time_seconds"],
        label="CLEIR + Wasserstein (delta=0.01)"
    )

    plt.xlabel("Rebalancing Date")
    plt.ylabel("Solve Time (seconds)")
    plt.title("Monthly Optimization Time")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

plt.show()