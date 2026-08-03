import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# 결과 불러오기
# ============================================================

# case 1 : Rough

# CEIR
ceir_case1_weight_history = pd.read_csv(
    f"../data/data_sp500_wass_ceir_alpha=0.95_delta=0.01_weights_case1.csv",
    index_col=0,
    parse_dates=True
)

ceir_case1_optimization_history = pd.read_csv(
    f"../data/data_sp500_wass_ceir_alpha=0.95_delta=0.01_optimization_history_case1.csv",
    index_col=0,
    parse_dates=True
)

ceir_case1_backtest_returns = pd.read_csv(
    f"../data/data_sp500_wass_ceir_alpha=0.95_delta=0.01_returns_case1.csv",
    index_col=0,
    parse_dates=True
)

# CLEIR
cleir_case1_weight_history = pd.read_csv(
    f"../data/data_sp500_wass_cleir_alpha=0.95_delta=0.01_weights_case1.csv",
    index_col=0,
    parse_dates=True
)

cleir_case1_optimization_history = pd.read_csv(
    f"../data/data_sp500_wass_cleir_alpha=0.95_delta=0.01_optimization_history_case1.csv",
    index_col=0,
    parse_dates=True
)

cleir_case1_backtest_returns = pd.read_csv(
    f"../data/data_sp500_wass_cleir_alpha=0.95_delta=0.01_returns_case1.csv",
    index_col=0,
    parse_dates=True
)

# Robust Index Tracking
rit_case1_weight_history = pd.read_csv(
    f"../data/data_sp500_wass_rit_delta=0.01_weights_case1.csv",
    index_col=0,
    parse_dates=True
)

rit_case1_optimization_history = pd.read_csv(
    f"../data/data_sp500_wass_rit_delta=0.01_optimization_history_case1.csv",
    index_col=0,
    parse_dates=True
)

rit_case1_backtest_returns = pd.read_csv(
    f"../data/data_sp500_wass_rit_delta=0.01_returns_case1.csv",
    index_col=0,
    parse_dates=True
)

# case 2 : Detailde

# CEIR
ceir_case2_weight_history = pd.read_csv(
    f"../data/data_sp500_wass_ceir_alpha=0.95_delta=0.01_weights_case2.csv",
    index_col=0,
    parse_dates=True
)

ceir_case2_optimization_history = pd.read_csv(
    f"../data/data_sp500_wass_ceir_alpha=0.95_delta=0.01_optimization_history_case2.csv",
    index_col=0,
    parse_dates=True
)

ceir_case2_backtest_returns = pd.read_csv(
    f"../data/data_sp500_wass_ceir_alpha=0.95_delta=0.01_returns_case2.csv",
    index_col=0,
    parse_dates=True
)

# CLEIR
cleir_case2_weight_history = pd.read_csv(
    f"../data/data_sp500_wass_cleir_alpha=0.95_delta=0.01_weights_case2.csv",
    index_col=0,
    parse_dates=True
)

cleir_case2_optimization_history = pd.read_csv(
    f"../data/data_sp500_wass_cleir_alpha=0.95_delta=0.01_optimization_history_case2.csv",
    index_col=0,
    parse_dates=True
)

cleir_case2_backtest_returns = pd.read_csv(
    f"../data/data_sp500_wass_cleir_alpha=0.95_delta=0.01_returns_case2.csv",
    index_col=0,
    parse_dates=True
)

# Robust Index Tracking
rit_case2_weight_history = pd.read_csv(
    f"../data/data_sp500_wass_rit_delta=0.01_weights_case2.csv",
    index_col=0,
    parse_dates=True
)

rit_case2_optimization_history = pd.read_csv(
    f"../data/data_sp500_wass_rit_delta=0.01_optimization_history_case2.csv",
    index_col=0,
    parse_dates=True
)

rit_case2_backtest_returns = pd.read_csv(
    f"../data/data_sp500_wass_rit_delta=0.01_returns_case2.csv",
    index_col=0,
    parse_dates=True
)

# case 3 : Unbounded

# CEIR
ceir_case3_weight_history = pd.read_csv(
    f"../data/data_sp500_wass_ceir_alpha=0.95_delta=0.01_weights_case3.csv",
    index_col=0,
    parse_dates=True
)

ceir_case3_optimization_history = pd.read_csv(
    f"../data/data_sp500_wass_ceir_alpha=0.95_delta=0.01_optimization_history_case3.csv",
    index_col=0,
    parse_dates=True
)

ceir_case3_backtest_returns = pd.read_csv(
    f"../data/data_sp500_wass_ceir_alpha=0.95_delta=0.01_returns_case3.csv",
    index_col=0,
    parse_dates=True
)

# CLEIR
cleir_case3_weight_history = pd.read_csv(
    f"../data/data_sp500_wass_cleir_alpha=0.95_delta=0.01_weights_case3.csv",
    index_col=0,
    parse_dates=True
)

cleir_case3_optimization_history = pd.read_csv(
    f"../data/data_sp500_wass_cleir_alpha=0.95_delta=0.01_optimization_history_case3.csv",
    index_col=0,
    parse_dates=True
)

cleir_case3_backtest_returns = pd.read_csv(
    f"../data/data_sp500_wass_cleir_alpha=0.95_delta=0.01_returns_case3.csv",
    index_col=0,
    parse_dates=True
)

# Robust Index Tracking
rit_case3_weight_history = pd.read_csv(
    f"../data/data_sp500_wass_rit_delta=0.01_weights_case3.csv",
    index_col=0,
    parse_dates=True
)

rit_case3_optimization_history = pd.read_csv(
    f"../data/data_sp500_wass_rit_delta=0.01_optimization_history_case3.csv",
    index_col=0,
    parse_dates=True
)

rit_case3_backtest_returns = pd.read_csv(
    f"../data/data_sp500_wass_rit_delta=0.01_returns_case3.csv",
    index_col=0,
    parse_dates=True
)


# ============================================================
# 계산
# ============================================================

wass_ceir_case1_cumulative = (1 + ceir_case1_backtest_returns).cumprod()
wass_cleir_case1_cumulative = (1 + cleir_case1_backtest_returns).cumprod()
wass_rit_case1_cumulative = (1 + rit_case1_backtest_returns).cumprod()

wass_ceir_case2_cumulative = (1 + ceir_case2_backtest_returns).cumprod()
wass_cleir_case2_cumulative = (1 + cleir_case2_backtest_returns).cumprod()
wass_rit_case2_cumulative = (1 + rit_case2_backtest_returns).cumprod()

wass_ceir_case3_cumulative = (1 + ceir_case3_backtest_returns).cumprod()
wass_cleir_case3_cumulative = (1 + cleir_case3_backtest_returns).cumprod()
wass_rit_case3_cumulative = (1 + rit_case3_backtest_returns).cumprod()

tol = 1e-6
wass_ceir_case1_l0_norm = (ceir_case1_weight_history.abs() > tol).sum(axis=1)
wass_cleir_case1_l0_norm = (cleir_case1_weight_history.abs() > tol).sum(axis=1)
wass_rit_case1_l0_norm = (rit_case1_weight_history.abs() > tol).sum(axis=1)

wass_ceir_case2_l0_norm = (ceir_case2_weight_history.abs() > tol).sum(axis=1)
wass_cleir_case2_l0_norm = (cleir_case2_weight_history.abs() > tol).sum(axis=1)
wass_rit_case2_l0_norm = (rit_case2_weight_history.abs() > tol).sum(axis=1)

wass_ceir_case3_l0_norm = (ceir_case3_weight_history.abs() > tol).sum(axis=1)
wass_cleir_case3_l0_norm = (cleir_case3_weight_history.abs() > tol).sum(axis=1)
wass_rit_case3_l0_norm = (rit_case3_weight_history.abs() > tol).sum(axis=1)


# ============================================================
# 1. 수익률 & 누적수익률
# ============================================================

# 1.1 수익률

# 1.1.1 CEIR
plt.figure(figsize=(13, 6))

common_index = ceir_case1_backtest_returns.index

plt.plot(
    common_index,
    ceir_case1_backtest_returns["benchmark"],
    label="S&P500",
    linewidth=0.8,
    alpha=0.8,
)

plt.plot(
    common_index,
    ceir_case1_backtest_returns["portfolio"],
    label="case 1",
    linewidth=0.5,
    alpha=0.95,
)

plt.plot(
    common_index,
    ceir_case2_backtest_returns["portfolio"],
    label="case 2",
    linewidth=0.5,
    alpha=0.95,
)

plt.plot(
    common_index,
    ceir_case3_backtest_returns["portfolio"],
    label="case 3",
    linewidth=0.5,
    alpha=0.95,
)

plt.xlabel("Date")
plt.ylabel("Return")
plt.title("CEIR : Out-of-sample Portfolio Returns")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

# 1.1.2 CLEIR
plt.figure(figsize=(13, 6))

common_index = cleir_case1_backtest_returns.index

plt.plot(
    common_index,
    cleir_case1_backtest_returns["benchmark"],
    label="S&P500",
    linewidth=0.8,
    alpha=0.8,
)

plt.plot(
    common_index,
    cleir_case1_backtest_returns["portfolio"],
    label="case 1",
    linewidth=0.5,
    alpha=0.95,
)

plt.plot(
    common_index,
    cleir_case2_backtest_returns["portfolio"],
    label="case 2",
    linewidth=0.5,
    alpha=0.95,
)

plt.plot(
    common_index,
    cleir_case3_backtest_returns["portfolio"],
    label="case 3",
    linewidth=0.5,
    alpha=0.95,
)

plt.xlabel("Date")
plt.ylabel("Return")
plt.title("CLEIR : Out-of-sample Portfolio Returns")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

# 1.1.3 Robust Index Tracking
plt.figure(figsize=(13, 6))

common_index = rit_case1_backtest_returns.index

plt.plot(
    common_index,
    rit_case1_backtest_returns["benchmark"],
    label="S&P500",
    linewidth=0.8,
    alpha=0.8,
)

plt.plot(
    common_index,
    rit_case1_backtest_returns["portfolio"],
    label="case 1",
    linewidth=0.5,
    alpha=0.95,
)

plt.plot(
    common_index,
    rit_case2_backtest_returns["portfolio"],
    label="case 2",
    linewidth=0.5,
    alpha=0.95,
)

plt.plot(
    common_index,
    rit_case3_backtest_returns["portfolio"],
    label="case 3",
    linewidth=0.5,
    alpha=0.95,
)

plt.xlabel("Date")
plt.ylabel("Return")
plt.title("RIT : Out-of-sample Portfolio Returns")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

# 1.2 누적수익률

# 1.2.1 CEIR
plt.figure(figsize=(13, 6))

common_index = wass_ceir_case1_cumulative.index

plt.plot(
    common_index,
    wass_ceir_case1_cumulative["benchmark"],
    label="S&P500",
    linewidth=0.8,
    alpha=0.8,
)

plt.plot(
    common_index,
    wass_ceir_case1_cumulative["portfolio"],
    label="case 1",
    linewidth=0.5,
    alpha=0.95,
)

plt.plot(
    common_index,
    wass_ceir_case2_cumulative["portfolio"],
    label="case 2",
    linewidth=0.5,
    alpha=0.95,
)

plt.plot(
    common_index,
    wass_ceir_case3_cumulative["portfolio"],
    label="case 3",
    linewidth=0.5,
    alpha=0.95,
)

plt.xlabel("Date")
plt.ylabel("Cumulative Value")
plt.title("CEIR : Out-of-sample Backtest")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

# 1.2.2 CLEIR
plt.figure(figsize=(13, 6))

common_index = wass_cleir_case1_cumulative.index

plt.plot(
    common_index,
    wass_cleir_case1_cumulative["benchmark"],
    label="S&P500",
    linewidth=0.8,
    alpha=0.8,
)

plt.plot(
    common_index,
    wass_cleir_case1_cumulative["portfolio"],
    label="case 1",
    linewidth=0.5,
    alpha=0.95,
)

plt.plot(
    common_index,
    wass_cleir_case2_cumulative["portfolio"],
    label="case 2",
    linewidth=0.5,
    alpha=0.95,
)

plt.plot(
    common_index,
    wass_cleir_case3_cumulative["portfolio"],
    label="case 3",
    linewidth=0.5,
    alpha=0.95,
)

plt.xlabel("Date")
plt.ylabel("Cumulative Value")
plt.title("CLEIR : Out-of-sample Backtest")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

# 1.2.3 RIT
plt.figure(figsize=(13, 6))

common_index = wass_rit_case1_cumulative.index

plt.plot(
    common_index,
    wass_rit_case1_cumulative["benchmark"],
    label="S&P500",
    linewidth=0.8,
    alpha=0.8,
)

plt.plot(
    common_index,
    wass_rit_case1_cumulative["portfolio"],
    label="case 1",
    linewidth=0.5,
    alpha=0.95,
)

plt.plot(
    common_index,
    wass_rit_case2_cumulative["portfolio"],
    label="case 2",
    linewidth=0.5,
    alpha=0.95,
)

plt.plot(
    common_index,
    wass_rit_case3_cumulative["portfolio"],
    label="case 3",
    linewidth=0.5,
    alpha=0.95,
)

plt.xlabel("Date")
plt.ylabel("Cumulative Value")
plt.title("RIT : Out-of-sample Backtest")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

# ============================================================
# 2. weight
# ============================================================

case_labels = {
    1: "Case 1 : Rough",
    2: "Case 2 : Detailed",
    3: "Case 3 : Unbounded",
}

def plot_weight_history(
    weight_history,
    title,
):
    """
    종목별 포트폴리오 비중을 누적 영역 그래프로 출력한다.

    양수와 음수 비중을 따로 쌓으므로,
    공매도가 있는 CLEIR 결과도 표시할 수 있다.
    """

    weights = weight_history.copy().fillna(0.0) * 100

    # 수치 오차로 발생한 매우 작은 비중 제거
    weights = weights.mask(weights.abs() < tol, 0.0)

    positive_weights = weights.clip(lower=0.0)
    negative_weights = weights.clip(upper=0.0)

    plt.figure(figsize=(13, 6))

    # Long positions
    if (positive_weights.abs().sum(axis=0) > 0).any():
        plt.stackplot(
            positive_weights.index,
            positive_weights.T.to_numpy(),
            labels=positive_weights.columns,
            alpha=0.8,
        )

    # Short positions
    if (negative_weights.abs().sum(axis=0) > 0).any():
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


# ------------------------------------------------------------
# 2.2.1 CEIR
# ------------------------------------------------------------

# CEIR : Case 1
plot_weight_history(
    weight_history=ceir_case1_weight_history,
    title="CEIR : Portfolio Composition — Case 1 (Rough)",
)

# CEIR : Case 2
plot_weight_history(
    weight_history=ceir_case2_weight_history,
    title="CEIR : Portfolio Composition — Case 2 (Detailed)",
)

# CEIR : Case 3
plot_weight_history(
    weight_history=ceir_case3_weight_history,
    title="CEIR : Portfolio Composition — Case 3 (Unbounded)",
)


# ------------------------------------------------------------
# 2.2.2 CLEIR
# ------------------------------------------------------------

# CLEIR : Case 1
plot_weight_history(
    weight_history=cleir_case1_weight_history,
    title="CLEIR : Portfolio Composition — Case 1 (Rough)",
)

# CLEIR : Case 2
plot_weight_history(
    weight_history=cleir_case2_weight_history,
    title="CLEIR : Portfolio Composition — Case 2 (Detailed)",
)

# CLEIR : Case 3
plot_weight_history(
    weight_history=cleir_case3_weight_history,
    title="CLEIR : Portfolio Composition — Case 3 (Unbounded)",
)


# ------------------------------------------------------------
# 2.2.3 Robust Index Tracking
# ------------------------------------------------------------

# RIT : Case 1
plot_weight_history(
    weight_history=rit_case1_weight_history,
    title="RIT : Portfolio Composition — Case 1 (Rough)",
)

# RIT : Case 2
plot_weight_history(
    weight_history=rit_case2_weight_history,
    title="RIT : Portfolio Composition — Case 2 (Detailed)",
)

# RIT : Case 3
plot_weight_history(
    weight_history=rit_case3_weight_history,
    title="RIT : Portfolio Composition — Case 3 (Unbounded)",
)


plt.show()