import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 결과 불러오기
# ============================================================

# CEIR V1
wass_ceir_v1_weight_history = pd.read_csv(
    f"../data/exp3_data_sp500_wass_ceir_alpha=0.95_delta=0.01_weights_v1.csv",
    index_col=0,
    parse_dates=True,
)

wass_ceir_v1_optimization_history = pd.read_csv(
    f"../data/exp3_data_sp500_wass_ceir_alpha=0.95_delta=0.01_optimization_history_v1.csv",
    index_col=0,
    parse_dates=True,
)

wass_ceir_v1_backtest_returns = pd.read_csv(
    f"../data/exp3_data_sp500_wass_ceir_alpha=0.95_delta=0.01_returns_v1.csv",
    index_col=0,
    parse_dates=True,
)


# CEIR V2
wass_ceir_v2_weight_history = pd.read_csv(
    f"../data/exp3_data_sp500_wass_ceir_alpha=0.95_delta=0.01_weights_v2.csv",
    index_col=0,
    parse_dates=True,
)

wass_ceir_v2_optimization_history = pd.read_csv(
    f"../data/exp3_data_sp500_wass_ceir_alpha=0.95_delta=0.01_optimization_history_v2.csv",
    index_col=0,
    parse_dates=True,
)

wass_ceir_v2_backtest_returns = pd.read_csv(
    f"../data/exp3_data_sp500_wass_ceir_alpha=0.95_delta=0.01_returns_v2.csv",
    index_col=0,
    parse_dates=True,
)


# CLEIR V1
wass_cleir_v1_weight_history = pd.read_csv(
    f"../data/exp3_data_sp500_wass_cleir_alpha=0.95_delta=0.01_weights_v1.csv",
    index_col=0,
    parse_dates=True,
)

wass_cleir_v1_optimization_history = pd.read_csv(
    f"../data/exp3_data_sp500_wass_cleir_alpha=0.95_delta=0.01_optimization_history_v1.csv",
    index_col=0,
    parse_dates=True,
)

wass_cleir_v1_backtest_returns = pd.read_csv(
    f"../data/exp3_data_sp500_wass_cleir_alpha=0.95_delta=0.01_returns_v1.csv",
    index_col=0,
    parse_dates=True,
)


# CLEIR V2
wass_cleir_v2_weight_history = pd.read_csv(
    f"../data/exp3_data_sp500_wass_cleir_alpha=0.95_delta=0.01_weights_v2.csv",
    index_col=0,
    parse_dates=True,
)

wass_cleir_v2_optimization_history = pd.read_csv(
    f"../data/exp3_data_sp500_wass_cleir_alpha=0.95_delta=0.01_optimization_history_v2.csv",
    index_col=0,
    parse_dates=True,
)

wass_cleir_v2_backtest_returns = pd.read_csv(
    f"../data/exp3_data_sp500_wass_cleir_alpha=0.95_delta=0.01_returns_v2.csv",
    index_col=0,
    parse_dates=True,
)

# ============================================================
# 계산
# ============================================================

# 누적수익률
wass_ceir_v1_cumulative = (1 + wass_ceir_v1_backtest_returns).cumprod()
wass_ceir_v2_cumulative = (1 + wass_ceir_v2_backtest_returns).cumprod()
wass_cleir_v1_cumulative = (1 + wass_cleir_v1_backtest_returns).cumprod()
wass_cleir_v2_cumulative = (1 + wass_cleir_v2_backtest_returns).cumprod()

# 사용 종목 수
tol = 1e-6

wass_ceir_v1_l0_norm = (wass_ceir_v1_weight_history.abs() > tol).sum(axis=1)
wass_ceir_v2_l0_norm = (wass_ceir_v2_weight_history.abs() > tol).sum(axis=1)
wass_cleir_v1_l0_norm = (wass_cleir_v1_weight_history.abs() > tol).sum(axis=1)
wass_cleir_v2_l0_norm = (wass_cleir_v2_weight_history.abs() > tol).sum(axis=1)

# ============================================================
# 1. 수익률 & 누적수익률
# ============================================================

# ------------------------------------------------------------
# 1.1 수익률
# ------------------------------------------------------------

# CEIR
plt.figure(figsize=(13, 6))

plt.plot(
    wass_ceir_v1_backtest_returns.index,
    wass_ceir_v1_backtest_returns["benchmark"],
    label="S&P500",
    linewidth=0.8,
    alpha=0.8,
)

plt.plot(
    wass_ceir_v1_backtest_returns.index,
    wass_ceir_v1_backtest_returns["portfolio"],
    label="version 1",
    linewidth=0.5,
    alpha=0.5,
    linestyle="--"
)

plt.plot(
    wass_ceir_v2_backtest_returns.index,
    wass_ceir_v2_backtest_returns["portfolio"],
    label="version 2",
    linewidth=0.5,
    alpha=0.5,
    linestyle=":"
)

plt.xlabel("Date")
plt.ylabel("Return")
plt.title("CEIR : Out-of-sample Portfolio Returns")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()


# CLEIR
plt.figure(figsize=(13, 6))

plt.plot(
    wass_cleir_v1_backtest_returns.index,
    wass_cleir_v1_backtest_returns["benchmark"],
    label="S&P500",
    linewidth=0.8,
    alpha=0.8,
)

plt.plot(
    wass_cleir_v1_backtest_returns.index,
    wass_cleir_v1_backtest_returns["portfolio"],
    label="version 1",
    linewidth=0.5,
    alpha=0.5,
    linestyle="--"
)

plt.plot(
    wass_cleir_v2_backtest_returns.index,
    wass_cleir_v2_backtest_returns["portfolio"],
    label="version 2",
    linewidth=0.5,
    alpha=0.5,
    linestyle=":"
)

plt.xlabel("Date")
plt.ylabel("Return")
plt.title("CLEIR : Out-of-sample Portfolio Returns")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()


# ------------------------------------------------------------
# 1.2 누적수익률
# ------------------------------------------------------------

# CEIR
plt.figure(figsize=(13, 6))

plt.plot(
    wass_ceir_v1_cumulative.index,
    wass_ceir_v1_cumulative["benchmark"],
    label="S&P500",
    linewidth=0.8,
    alpha=0.8,
)

plt.plot(
    wass_ceir_v1_cumulative.index,
    wass_ceir_v1_cumulative["portfolio"],
    label="version 1",
    linewidth=0.5,
    alpha=0.5,
    linestyle="--"
)

plt.plot(
    wass_ceir_v2_cumulative.index,
    wass_ceir_v2_cumulative["portfolio"],
    label="version 2",
    linewidth=0.5,
    alpha=0.5,
    linestyle=":"
)

plt.xlabel("Date")
plt.ylabel("Cumulative Value")
plt.title("CEIR : Out-of-sample Backtest")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()


# CLEIR
plt.figure(figsize=(13, 6))

plt.plot(
    wass_cleir_v1_cumulative.index,
    wass_cleir_v1_cumulative["benchmark"],
    label="S&P500",
    linewidth=0.8,
    alpha=0.8,
)

plt.plot(
    wass_cleir_v1_cumulative.index,
    wass_cleir_v1_cumulative["portfolio"],
    label="version 1",
    linewidth=0.5,
    alpha=0.5,
    linestyle="--"
)

plt.plot(
    wass_cleir_v2_cumulative.index,
    wass_cleir_v2_cumulative["portfolio"],
    label="version 2",
    linewidth=0.5,
    alpha=0.5,
    linestyle=":"
)

plt.xlabel("Date")
plt.ylabel("Cumulative Value")
plt.title("CLEIR : Out-of-sample Backtest")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()


# ============================================================
# 2. weight
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

    weights = weights.loc[:, active_columns]

    positive_weights = weights.clip(lower=0.0)
    negative_weights = weights.clip(upper=0.0)

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


# CEIR V1
plot_weight_history(
    weight_history=wass_ceir_v1_weight_history,
    title=("Portfolio Composition : CEIR : version 1"),
)


# CEIR V2
plot_weight_history(
    weight_history=wass_ceir_v2_weight_history,
    title=("Portfolio Composition : CEIR : version 2"),
)


# CLEIR V1
plot_weight_history(
    weight_history=wass_cleir_v1_weight_history,
    title=("Portfolio Composition : CLEIR : version 1"),
)


# CLEIR V2
plot_weight_history(
    weight_history=wass_cleir_v2_weight_history,
    title=("Portfolio Composition : CLEIR : version 2"),
)


plt.show()