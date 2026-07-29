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

# CEIR + KDE(Gaussian + KL Divergence)

robust_ceir_kde_gaussian_kl_weight_history = pd.read_csv(
    f"data/data_{INDEX}_robust_ceir_kde_gaussian_kl_weights.csv",
    index_col=0,
    parse_dates=True
)

robust_ceir_kde_gaussian_kl_optimization_history = pd.read_csv(
    f"data/data_{INDEX}_robust_ceir_kde_gaussian_kl_optimization_history.csv",
    index_col=0,
    parse_dates=True
)

robust_ceir_kde_gaussian_kl_backtest_returns = pd.read_csv(
    f"data/data_{INDEX}_robust_ceir_kde_gaussian_kl_returns.csv",
    index_col=0,
    parse_dates=True
)

# CLEIR + KDE(Gaussian + KL Divergence)

robust_cleir_kde_gaussian_kl_weight_history = pd.read_csv(
    f"data/data_{INDEX}_robust_cleir_kde_gaussian_kl_weights.csv",
    index_col=0,
    parse_dates=True
)

robust_cleir_kde_gaussian_kl_optimization_history = pd.read_csv(
    f"data/data_{INDEX}_robust_cleir_kde_gaussian_kl_optimization_history.csv",
    index_col=0,
    parse_dates=True
)

robust_cleir_kde_gaussian_kl_backtest_returns = pd.read_csv(
    f"data/data_{INDEX}_robust_cleir_kde_gaussian_kl_returns.csv",
    index_col=0,
    parse_dates=True
)

# CLEIR + KDE(Epanechnikov + KL Divergence)

robust_cleir_kde_epanechnikov_kl_weight_history = pd.read_csv(
    f"data/data_{INDEX}_robust_cleir_kde_epanechnikov_kl_weights.csv",
    index_col=0,
    parse_dates=True
)

robust_cleir_kde_epanechnikov_kl_optimization_history = pd.read_csv(
    f"data/data_{INDEX}_robust_cleir_kde_epanechnikov_kl_optimization_history.csv",
    index_col=0,
    parse_dates=True
)

robust_cleir_kde_epanechnikov_kl_backtest_returns = pd.read_csv(
    f"data/data_{INDEX}_robust_cleir_kde_epanechnikov_kl_returns.csv",
    index_col=0,
    parse_dates=True
)

# CLEIR + KDE(Epanechnikov + Hellinger distance)

robust_cleir_kde_epanechnikov_hellinger_weight_history = pd.read_csv(
    f"data/data_{INDEX}_robust_cleir_kde_epanechnikov_hellinger_weights.csv",
    index_col=0,
    parse_dates=True
)

robust_cleir_kde_epanechnikov_hellinger_optimization_history = pd.read_csv(
    f"data/data_{INDEX}_robust_cleir_kde_epanechnikov_hellinger_optimization_history.csv",
    index_col=0,
    parse_dates=True
)

robust_cleir_kde_epanechnikov_hellinger_backtest_returns = pd.read_csv(
    f"data/data_{INDEX}_robust_cleir_kde_epanechnikov_hellinger_returns.csv",
    index_col=0,
    parse_dates=True
)

# ============================================================
# 계산
# ============================================================

ceir_cumulative = (1 + ceir_backtest_returns).cumprod()
cleir_cumulative = (1 + cleir_backtest_returns).cumprod()
robust_ceir_wasserstein_cumulative = (1 + robust_ceir_wasserstein_backtest_returns).cumprod()
robust_cleir_wasserstein_cumulative = (1 + robust_cleir_wasserstein_backtest_returns).cumprod()
robust_ceir_kde_gaussian_kl_cumulative = (1 + robust_ceir_kde_gaussian_kl_backtest_returns).cumprod()
robust_cleir_kde_gaussian_kl_cumulative = (1 + robust_cleir_kde_gaussian_kl_backtest_returns).cumprod()
robust_cleir_kde_epanechnikov_kl_cumulative = (1 + robust_cleir_kde_epanechnikov_kl_backtest_returns).cumprod()
robust_cleir_kde_epanechnikov_hellinger_cumulative = (1 + robust_cleir_kde_epanechnikov_hellinger_backtest_returns).cumprod()

tol = 1e-6
ceir_l0_norm = (ceir_weight_history.abs() > tol).sum(axis=1)
cleir_l0_norm = (cleir_weight_history.abs() > tol).sum(axis=1)
robust_ceir_wasserstein_l0_norm = (robust_ceir_wasserstein_weight_history.abs() > tol).sum(axis=1)
robust_cleir_wasserstein_l0_norm = (robust_cleir_wasserstein_weight_history.abs() > tol).sum(axis=1)
robust_ceir_kde_gaussian_kl_l0_norm = (robust_ceir_kde_gaussian_kl_weight_history.abs() > tol).sum(axis=1)
robust_cleir_kde_gaussian_kl_l0_norm = (robust_cleir_kde_gaussian_kl_weight_history.abs() > tol).sum(axis=1)
robust_cleir_kde_epanechnikov_kl_l0_norm = (robust_cleir_kde_epanechnikov_kl_weight_history.abs() > tol).sum(axis=1)
robust_cleir_kde_epanechnikov_hellinger_l0_norm = (robust_cleir_kde_epanechnikov_hellinger_weight_history.abs() > tol).sum(axis=1)


# ============================================================
# 1. 누적수익률
# ============================================================

plt.figure(figsize=(12, 6))

common_index = ceir_cumulative.index

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

plt.plot(
    common_index,
    robust_ceir_wasserstein_cumulative["portfolio"],
    label="CEIR + Wasserstein Portfolio"
)

plt.plot(
    common_index,
    robust_cleir_wasserstein_cumulative["portfolio"],
    label="CLEIR + Wasserstein Portfolio"
)

plt.plot(
    common_index,
    robust_ceir_kde_gaussian_kl_cumulative["portfolio"],
    label="CEIR + KDE(Gaussian + KL) Portfolio"
)

plt.plot(
    common_index,
    robust_cleir_kde_gaussian_kl_cumulative["portfolio"],
    label="CLEIR + KDE(Gaussian + KL) Portfolio"
)

plt.plot(
    common_index,
    robust_cleir_kde_epanechnikov_kl_cumulative["portfolio"],
    label="CLEIR + KDE(Epanechnikov + KL) Portfolio"
)

plt.plot(
    common_index,
    robust_cleir_kde_epanechnikov_hellinger_cumulative["portfolio"],
    label="CLEIR + KDE(Epanechnikov + Hellinger) Portfolio"
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

common_index = ceir_l0_norm.index

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

plt.step(
    robust_ceir_wasserstein_l0_norm.index,
    robust_ceir_wasserstein_l0_norm,
    where="post",
    label="CEIR + Wasserstein"
)

plt.step(
    robust_cleir_wasserstein_l0_norm.index,
    robust_cleir_wasserstein_l0_norm,
    where="post",
    label="CLEIR + Wasserstein"
)

plt.step(
    robust_ceir_kde_gaussian_kl_l0_norm.index,
    robust_ceir_kde_gaussian_kl_l0_norm,
    where="post",
    label="CEIR + KDE(Gaussian + KL)"
)

plt.step(
    robust_cleir_kde_gaussian_kl_l0_norm.index,
    robust_cleir_kde_gaussian_kl_l0_norm,
    where="post",
    label="CLEIR + KDE(Gaussian + KL)"
)

plt.step(
    robust_cleir_kde_epanechnikov_kl_l0_norm.index,
    robust_cleir_kde_epanechnikov_kl_l0_norm,
    where="post",
    label="CLEIR + KDE(Epanechnikov + KL)"
)

plt.step(
    robust_cleir_kde_epanechnikov_hellinger_l0_norm.index,
    robust_cleir_kde_epanechnikov_hellinger_l0_norm,
    where="post",
    label="CLEIR + KDE(Epanechnikov + Hellinger)"
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

plt.figure(figsize=(12, 5))

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

plt.plot(
    robust_ceir_wasserstein_optimization_history.index,
    robust_ceir_wasserstein_optimization_history["solve_time_seconds"],
    label="CEIR + Wasserstein"
)

plt.plot(
    robust_cleir_wasserstein_optimization_history.index,
    robust_cleir_wasserstein_optimization_history["solve_time_seconds"],
    label="CLEIR + Wasserstein"
)

plt.plot(
    robust_ceir_kde_gaussian_kl_optimization_history.index,
    robust_ceir_kde_gaussian_kl_optimization_history["solve_time_seconds"],
    label="CEIR + KDE(Gaussian + KL)"
)

plt.plot(
    robust_cleir_kde_gaussian_kl_optimization_history.index,
    robust_cleir_kde_gaussian_kl_optimization_history["solve_time_seconds"],
    label="CLEIR + KDE(Gaussian + KL)"
)

plt.plot(
    robust_cleir_kde_epanechnikov_kl_optimization_history.index,
    robust_cleir_kde_epanechnikov_kl_optimization_history["solve_time_seconds"],
    label="CLEIR + KDE(Epanechnikov + KL)"
)

plt.plot(
    robust_cleir_kde_epanechnikov_hellinger_optimization_history.index,
    robust_cleir_kde_epanechnikov_hellinger_optimization_history["solve_time_seconds"],
    label="CLEIR + KDE(Epanechnikov + Hellinger)"
)

plt.xlabel("Rebalancing Date")
plt.ylabel("Solve Time (seconds)")
plt.title("Monthly Optimization Time")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.show()