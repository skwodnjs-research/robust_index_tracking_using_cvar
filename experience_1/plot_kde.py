import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# 결과 불러오기
# ============================================================

index = 0
INDEX = ["sp500", "russell2000", "kospi"][index]

# CEIR
ceir_weight_history = pd.read_csv(
    f"data/data_{INDEX}_ceir_weights.csv",
    index_col=0,
    parse_dates=True
)

ceir_optimization_history = pd.read_csv(
    f"data/data_{INDEX}_ceir_optimization_history.csv",
    index_col=0,
    parse_dates=True
)

ceir_backtest_returns = pd.read_csv(
    f"data/data_{INDEX}_ceir_returns.csv",
    index_col=0,
    parse_dates=True
)

# CLEIR
cleir_weight_history = pd.read_csv(
    f"data/data_{INDEX}_cleir_weights.csv",
    index_col=0,
    parse_dates=True
)

cleir_optimization_history = pd.read_csv(
    f"data/data_{INDEX}_cleir_optimization_history.csv",
    index_col=0,
    parse_dates=True
)

cleir_backtest_returns = pd.read_csv(
    f"data/data_{INDEX}_cleir_returns.csv",
    index_col=0,
    parse_dates=True
)

# CEIR + KDE(Gaussian + KL Divergence)

# CEIR + KDE(Gaussian + Hellinger)

# CEIR + KDE(Gaussian + Wasserstein)

# CEIR + KDE(Epanechnikov + KL Divergence)

# CEIR + KDE(Epanechnikov + Hellinger)

# CEIR + KDE(Epanechnikov + Wasserstein)

# ============================================================
# 계산
# ============================================================

ceir_cumulative = (1 + ceir_backtest_returns).cumprod()
cleir_cumulative = (1 + cleir_backtest_returns).cumprod()

tol = 1e-6
ceir_l0_norm = (ceir_weight_history.abs() > tol).sum(axis=1)
cleir_l0_norm = (cleir_weight_history.abs() > tol).sum(axis=1)


# ============================================================
# 1. 누적수익률
# ============================================================

plt.figure(figsize=(12, 6))

common_index = (
    ceir_cumulative.index
    .intersection(cleir_cumulative.index)
)

ceir_cumulative = ceir_cumulative.loc[common_index]
cleir_cumulative = cleir_cumulative.loc[common_index]

plt.plot(
    common_index,
    ceir_cumulative["benchmark"],
    label=INDEX
)

plt.plot(
    common_index,
    ceir_cumulative["portfolio"],
    label="CEIR Portfolio"
)

plt.plot(
    common_index,
    cleir_cumulative["portfolio"],
    label="CLEIR Portfolio"
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
    ceir_l0_norm.index
    .intersection(cleir_l0_norm.index)
)

ceir_l0_norm = ceir_l0_norm.loc[common_index]
cleir_l0_norm = cleir_l0_norm.loc[common_index]

plt.step(
    ceir_l0_norm.index,
    ceir_l0_norm,
    where="post",
    label="CEIR"
)

plt.step(
    cleir_l0_norm.index,
    cleir_l0_norm,
    where="post",
    label="CLEIR"
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

if ("solve_time_seconds" in ceir_optimization_history.columns 
    and "solve_time_seconds" in cleir_optimization_history.columns
):
    plt.figure(figsize=(12, 5))

    common_index = (
        ceir_optimization_history.index
        .intersection(cleir_optimization_history.index)
    )
    ceir_optimization_history = ceir_optimization_history.loc[common_index]
    cleir_optimization_history = cleir_optimization_history.loc[common_index]

    plt.plot(
        ceir_optimization_history.index,
        ceir_optimization_history["solve_time_seconds"],
        label="CEIR"
    )

    plt.plot(
        cleir_optimization_history.index,
        cleir_optimization_history["solve_time_seconds"],
        label="CLEIR"
    )

    plt.xlabel("Rebalancing Date")
    plt.ylabel("Solve Time (seconds)")
    plt.title("Monthly Optimization Time")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

plt.show()