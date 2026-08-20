import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# 결과 불러오기
# ============================================================

# Robust Portfolio Optimization
robust_po_weight_history = pd.read_csv(
    "../data/exp5_data_sp500_robust_po_alpha=0.95_delta=0.01_weights.csv",
    index_col=0,
    parse_dates=True,
)

robust_po_optimization_history = pd.read_csv(
    "../data/exp5_data_sp500_robust_po_alpha=0.95_delta=0.01_optimization_history.csv",
    index_col=0,
    parse_dates=True,
)

robust_po_backtest_returns = pd.read_csv(
    "../data/exp5_data_sp500_robust_po_alpha=0.95_delta=0.01_returns.csv",
    index_col=0,
    parse_dates=True,
)


# Robust Portfolio Optimization with LASSO Constraint
robust_po_lasso_weight_history = pd.read_csv(
    "../data/exp5_data_sp500_robust_po_lasso_alpha=0.95_delta=0.01_weights.csv",
    index_col=0,
    parse_dates=True,
)

robust_po_lasso_optimization_history = pd.read_csv(
    "../data/exp5_data_sp500_robust_po_lasso_alpha=0.95_delta=0.01_optimization_history.csv",
    index_col=0,
    parse_dates=True,
)

robust_po_lasso_backtest_returns = pd.read_csv(
    "../data/exp5_data_sp500_robust_po_lasso_alpha=0.95_delta=0.01_returns.csv",
    index_col=0,
    parse_dates=True,
)


# Robust Portfolio Optimization with Robust Return Constraint
robust_po_with_return_const_weight_history = pd.read_csv(
    "../data/exp5_data_sp500_robust_po_with_return_const_alpha=0.95_delta=0.01_weights.csv",
    index_col=0,
    parse_dates=True,
)

robust_po_with_return_const_optimization_history = pd.read_csv(
    "../data/exp5_data_sp500_robust_po_with_return_const_alpha=0.95_delta=0.01_optimization_history.csv",
    index_col=0,
    parse_dates=True,
)

robust_po_with_return_const_backtest_returns = pd.read_csv(
    "../data/exp5_data_sp500_robust_po_with_return_const_alpha=0.95_delta=0.01_returns.csv",
    index_col=0,
    parse_dates=True,
)


# Robust Portfolio Optimization with LASSO Constraint and Robust Return Constraint
robust_po_lasso_with_return_const_weight_history = pd.read_csv(
    "../data/exp5_data_sp500_robust_po_lasso_with_return_const_alpha=0.95_delta=0.01_weights.csv",
    index_col=0,
    parse_dates=True,
)

robust_po_lasso_with_return_const_optimization_history = pd.read_csv(
    "../data/exp5_data_sp500_robust_po_lasso_with_return_const_alpha=0.95_delta=0.01_optimization_history.csv",
    index_col=0,
    parse_dates=True,
)

robust_po_lasso_with_return_const_backtest_returns = pd.read_csv(
    "../data/exp5_data_sp500_robust_po_lasso_with_return_const_alpha=0.95_delta=0.01_returns.csv",
    index_col=0,
    parse_dates=True,
)


# ============================================================
# 계산
# ============================================================

# ------------------------------------------------------------
# 누적수익률
# ------------------------------------------------------------

wass_ceir_cumulative = (
    1 + robust_po_backtest_returns
).cumprod()

wass_cleir_cumulative = (
    1 + robust_po_lasso_backtest_returns
).cumprod()

wass_ceir_cumulative_with_constraint = (
    1 + robust_po_with_return_const_backtest_returns
).cumprod()

wass_cleir_cumulative_with_constraint = (
    1 + robust_po_lasso_with_return_const_backtest_returns
).cumprod()


# ------------------------------------------------------------
# Equal Weight
# ------------------------------------------------------------
# 네 실험 모두 동일한 30개 종목을 사용한다면
# equal weight는 한 returns 파일에서만 가져오면 충분하다.

equal_weight_returns = (
    robust_po_backtest_returns["equal_weight"]
)

equal_weight_cumulative = (
    1 + equal_weight_returns
).cumprod()


# ============================================================
# 1. 수익률 & 누적수익률
# ============================================================

# ------------------------------------------------------------
# 1.1 수익률
# ------------------------------------------------------------

plt.figure(figsize=(13, 6))


# Equal Weight
plt.plot(
    equal_weight_returns.index,
    equal_weight_returns,
    label="Equal Weight",
    linewidth=0.6,
    alpha=0.7,
)


# Benchmark
plt.plot(
    robust_po_with_return_const_backtest_returns.index,
    robust_po_with_return_const_backtest_returns["benchmark"],
    label="S&P500",
    linewidth=0.8,
    alpha=0.8,
)


# CEIR
plt.plot(
    robust_po_backtest_returns.index,
    robust_po_backtest_returns["portfolio"],
    label="CEIR",
    linewidth=0.5,
    alpha=0.5,
)


# CLEIR
plt.plot(
    robust_po_lasso_backtest_returns.index,
    robust_po_lasso_backtest_returns["portfolio"],
    label="CLEIR",
    linewidth=0.5,
    alpha=0.5,
)


# CEIR with constraint
plt.plot(
    robust_po_with_return_const_backtest_returns.index,
    robust_po_with_return_const_backtest_returns["portfolio"],
    label="CEIR with constraint",
    linewidth=0.5,
    alpha=0.5,
)


# CLEIR with constraint
plt.plot(
    robust_po_lasso_with_return_const_backtest_returns.index,
    robust_po_lasso_with_return_const_backtest_returns["portfolio"],
    label="CLEIR with constraint",
    linewidth=0.5,
    alpha=0.5,
)


plt.xlabel("Date")
plt.ylabel("Return")
plt.title("CEIR vs CLEIR")

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()


# ------------------------------------------------------------
# 1.2 누적수익률
# ------------------------------------------------------------

plt.figure(figsize=(13, 6))


# Equal Weight
plt.plot(
    equal_weight_cumulative.index,
    equal_weight_cumulative,
    label="Equal Weight",
    linewidth=0.8,
    alpha=0.8,
)


# Benchmark
plt.plot(
    wass_ceir_cumulative_with_constraint.index,
    wass_ceir_cumulative_with_constraint["benchmark"],
    label="S&P500",
    linewidth=0.8,
    alpha=0.8,
)


# CEIR
plt.plot(
    wass_ceir_cumulative.index,
    wass_ceir_cumulative["portfolio"],
    label="CEIR",
    linewidth=0.5,
    alpha=0.8,
    linestyle=":",
)


# CLEIR
plt.plot(
    wass_cleir_cumulative.index,
    wass_cleir_cumulative["portfolio"],
    label="CLEIR",
    linewidth=0.5,
    alpha=0.8,
    linestyle=":",
)


# CEIR with constraint
plt.plot(
    wass_ceir_cumulative_with_constraint.index,
    wass_ceir_cumulative_with_constraint["portfolio"],
    label="CEIR with constraint",
    linewidth=0.5,
    alpha=0.8,
    linestyle="--",
)


# CLEIR with constraint
plt.plot(
    wass_cleir_cumulative_with_constraint.index,
    wass_cleir_cumulative_with_constraint["portfolio"],
    label="CLEIR with constraint",
    linewidth=0.5,
    alpha=0.8,
    linestyle="--",
)


plt.xlabel("Date")
plt.ylabel("Cumulative Value")
plt.title("CEIR vs CLEIR")

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()


# ============================================================
# 2. Weight
# ============================================================

def plot_weight_history(
    weight_history,
    title,
    tol=1e-6,
):

    """
    포트폴리오 비중을 누적 영역 그래프로 출력한다.

    양수 비중과 음수 비중을 따로 누적하므로,
    long-only 포트폴리오와 short position을 포함하는
    포트폴리오를 모두 표시할 수 있다.
    """

    weights = (
        weight_history
        .copy()
        .fillna(0.0)
        .sort_index()
        * 100
    )


    # 솔버에서 발생한 작은 수치 오차 제거
    weights = weights.mask(
        weights.abs() < tol * 100,
        0.0,
    )


    # 전체 기간에서 한 번도 사용하지 않은 종목 제거
    active_columns = (
        weights.abs().max(axis=0) > 0
    )

    weights = weights.loc[
        :,
        active_columns,
    ]


    # Long / Short 분리
    positive_weights = weights.clip(
        lower=0.0
    )

    negative_weights = weights.clip(
        upper=0.0
    )


    positive_columns = (
        positive_weights.max(axis=0) > 0
    )

    negative_columns = (
        negative_weights.min(axis=0) < 0
    )


    positive_weights = positive_weights.loc[
        :,
        positive_columns,
    ]

    negative_weights = negative_weights.loc[
        :,
        negative_columns,
    ]


    plt.figure(figsize=(13, 6))


    # Long positions
    if not positive_weights.empty:

        plt.stackplot(
            positive_weights.index,
            positive_weights.T.to_numpy(),
            alpha=0.8,
        )


    # Short positions
    if not negative_weights.empty:

        plt.stackplot(
            negative_weights.index,
            negative_weights.T.to_numpy(),
            alpha=0.8,
        )


    plt.axhline(
        y=0,
        linewidth=0.8,
    )

    plt.xlabel("Rebalancing Date")
    plt.ylabel("Weight (%)")
    plt.title(title)

    plt.grid(alpha=0.3)
    plt.tight_layout()


# ============================================================
# Weight plot
# ============================================================

# CEIR
plot_weight_history(
    weight_history=robust_po_weight_history,
    title="Portfolio Composition : CEIR",
)


# CLEIR
plot_weight_history(
    weight_history=robust_po_lasso_weight_history,
    title="Portfolio Composition : CLEIR",
)


# CEIR with constraint
plot_weight_history(
    weight_history=robust_po_with_return_const_weight_history,
    title="Portfolio Composition : CEIR with constraint",
)


# CLEIR with constraint
plot_weight_history(
    weight_history=robust_po_lasso_with_return_const_weight_history,
    title="Portfolio Composition : CLEIR with constraint",
)


plt.show()