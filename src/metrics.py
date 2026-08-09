"""D11 성과 지표 — 자본곡선을 만들고 그 위에서만 잰다.

정의는 전부 `src/config.py` §18에 있고 **결과를 보기 전에 커밋했다**
(커밋 `7d3d82d`). 이 모듈은 그 값을 읽어 쓸 뿐 스스로 정하지 않는다.

수익률의 두 종류 — 섞지 말 것
------------------------------
이 모듈은 **단순수익률(simple return)** 만 다룬다.

    단순수익률  r_t = P_t / P_{t-1} - 1      복리 누적은 곱셈
    로그수익률  ℓ_t = ln(P_t) - ln(P_{t-1})   복리 누적은 덧셈

둘을 섞으면 조용히 틀린다. D2에서 로그수익률 평균 × 252를 CAGR로 읽어
한 국면에서 1%p 넘게 어긋난 적이 있다. 그래서 **모든 함수의 docstring에 어느
쪽을 받는지 적는다.** 로그수익률이 필요한 곳은 없다 — 자본곡선이 곱셈으로
누적되므로 단순수익률이 자연스러운 표현이다.

자본곡선이 먼저다
-----------------
지표는 전부 자본곡선 하나를 입력으로 받는다. 거래 로그에서 직접 재면
**미보유 구간이 통계에서 사라진다** — 현금으로 있는 동안에도 고점 대비
회복이 안 된 상태일 수 있고, 그 낙폭은 실재한다.

`config.MDD_BASIS = "equity_curve"`가 이것을 고정한다.

사용 예 (프로젝트 루트 기준)
----------------------------
    from src import backtest, config, data, metrics, signals

    price = data.load_parquet(config.RAW_OHLCV_PATH)
    rate = data.load_parquet(config.RAW_IRX_PATH)
    sig = signals.make_signals(price)

    trades, _ = backtest.run_all(price, sig)
    curve = metrics.build_equity_curve(trades, price, rate, "S1_rsi_oversold", 20)
    summary = metrics.summarize(curve, trades_subset)
"""

import numpy as np
import pandas as pd

from src import backtest
from src import config


# 지표 표 컬럼 순서. 리포트에서 그대로 쓰므로 고정한다.
METRIC_COLUMNS = [
    "name",
    "n_trades",
    "avg_holding_days",
    "exposure_pct",
    "total_return_pct",
    "cagr_pct",
    "ann_vol_pct",
    "sharpe",
    "sortino",
    "mdd_pct",
    "calmar",
    "win_rate_pct",
    "profit_factor",
]  # -> list[str] (13,)


# ---------------------------------------------------------------------------
# 1. 자본곡선
# ---------------------------------------------------------------------------
def daily_position_returns(trades, dates, prices):
    """거래 로그를 **거래일별 단순수익률**로 편다.

    보유 구간에서는 그날의 가격 변화율이 들어가고, 미보유 구간에는 NaN이
    들어간다. NaN과 0을 구분하는 것이 핵심이다 — 0으로 두면 "포지션이 있었는데
    수익이 0이었다"와 "포지션이 없었다"가 같은 값이 되어, 노출도와 현금 수익을
    계산할 수 없다.

    거래 비용 처리
    --------------
    진입일과 청산일에 각각 편도 비용을 실효가로 반영한다
    (`backtest.apply_costs`와 같은 규칙). 비용을 마지막에 한 번 빼지 않고
    발생 시점에 넣는 이유는, 복리로 굴러가는 자본에서는 **언제 빠지는지가
    최종값을 바꾸기 때문**이다.

    Parameters
    ----------
    trades : DataFrame   한 조합(signal_id, holding_days)의 거래만
    dates : ndarray[datetime64] (T,)
    prices : ndarray[float] (T,)   종가

    Returns
    -------
    Series[float] (T,)
        거래일별 단순수익률. 미보유일은 NaN.
    """
    n_days = len(dates)  # -> int

    position_return = np.full(n_days, np.nan)  # -> ndarray[float] (T,)

    index_of = {}  # -> dict[Timestamp, int]

    for position in range(n_days):
        index_of[pd.Timestamp(dates[position])] = position

    cost_bps = config.COMMISSION_BPS + config.SLIPPAGE_BPS  # -> int
    cost_fraction = cost_bps / 10_000.0                     # -> float

    for _, trade in trades.iterrows():
        entry_index = index_of[pd.Timestamp(trade["entry_date"])]  # -> int
        exit_index = index_of[pd.Timestamp(trade["exit_date"])]    # -> int

        for position in range(entry_index + 1, exit_index + 1):
            previous_price = prices[position - 1]  # -> float
            current_price = prices[position]       # -> float

            raw_return = current_price / previous_price - 1.0  # -> float

            position_return[position] = raw_return

        # 비용은 진입일과 청산일 수익률에 각각 실린다.
        # 진입일 당일은 보유 수익이 없으므로(진입가 = 그날 종가) 비용만 남는다.
        position_return[entry_index] = -cost_fraction

        exit_value = position_return[exit_index]  # -> float
        position_return[exit_index] = (1.0 + exit_value) * (1.0 - cost_fraction) - 1.0

    series = pd.Series(position_return, index=pd.DatetimeIndex(dates))  # -> Series[float]

    return series


def price_exposure_mask(trades, dates):
    """가격 노출이 있었던 날만 True.

    `position_return`의 non-NaN과 **일부러 다르다.** 진입일에는 비용이 실리므로
    수익률 값이 들어가지만, 진입가가 그날 종가이므로 **가격 노출은 없다.**
    노출도를 non-NaN으로 세면 거래 하나당 하루씩 과대 집계된다
    (실측: h=1에서 이론 0.972% vs 1.945%로 정확히 두 배).

    노출 구간은 진입 다음 거래일부터 청산일까지 = 정확히 h일이다.
    """
    n_days = len(dates)  # -> int

    mask = np.zeros(n_days, dtype=bool)  # -> ndarray[bool] (T,)

    index_of = {}  # -> dict[Timestamp, int]

    for position in range(n_days):
        index_of[pd.Timestamp(dates[position])] = position

    for _, trade in trades.iterrows():
        entry_index = index_of[pd.Timestamp(trade["entry_date"])]  # -> int
        exit_index = index_of[pd.Timestamp(trade["exit_date"])]    # -> int

        mask[entry_index + 1:exit_index + 1] = True

    return mask


def build_equity_curve_loop(position_return, risk_free, initial_capital):
    """원리 버전 — 하루씩 굴린다.

    **이 함수가 정의다.** 아래 벡터 버전은 이 결과를 재현해야 한다.

    규칙 (config §18)
    -----------------
        포지션 보유일 → 자본 × (1 + 그날 포지션 수익률)
        미보유일      → 자본 × (1 + 그날 무위험수익률)   CASH_RETURN="risk_free"

    Parameters
    ----------
    position_return : Series[float] (T,)   단순수익률, 미보유일 NaN
    risk_free : Series[float] (T,)         일별 무위험수익률 (단순)
    initial_capital : float

    Returns
    -------
    ndarray[float] (T,)
        각 거래일 **종료 시점**의 자본.
    """
    n_days = len(position_return)  # -> int

    position_values = position_return.to_numpy()  # -> ndarray[float] (T,)
    rate_values = risk_free.to_numpy()            # -> ndarray[float] (T,)

    equity = np.empty(n_days)  # -> ndarray[float] (T,)

    capital = float(initial_capital)  # -> float

    for position in range(n_days):
        day_return = position_values[position]  # -> float

        if np.isnan(day_return):
            if config.CASH_RETURN == "risk_free":
                day_return = rate_values[position]
            elif config.CASH_RETURN == "zero":
                day_return = 0.0
            else:
                raise ValueError(f"CASH_RETURN을 처리할 수 없다: {config.CASH_RETURN!r}")

        capital = capital * (1.0 + day_return)

        equity[position] = capital

    return equity


def build_equity_curve_vectorized(position_return, risk_free, initial_capital):
    """위 루프와 같은 결과를 내는 벡터 버전.

    자본곡선은 경로 의존이 아니라 **단순 누적곱**이므로 완전 벡터화가 된다
    (D10의 상태 기계와 달리 다음 날의 규칙이 과거 결과에 의존하지 않는다).
    """
    if config.CASH_RETURN == "risk_free":
        filler = risk_free  # -> Series[float]
    elif config.CASH_RETURN == "zero":
        filler = pd.Series(0.0, index=position_return.index)  # -> Series[float]
    else:
        raise ValueError(f"CASH_RETURN을 처리할 수 없다: {config.CASH_RETURN!r}")

    combined = position_return.fillna(filler)  # -> Series[float] (T,)

    growth = 1.0 + combined            # -> Series[float]
    cumulative = growth.cumprod()      # -> Series[float]

    equity = float(initial_capital) * cumulative  # -> Series[float]

    return equity.to_numpy()


def build_equity_curve(trades, price_frame, rate_frame, signal_id, holding_days,
                       verify=True):
    """한 조합의 자본곡선을 만든다.

    Returns
    -------
    DataFrame
        (date, position_return, risk_free, equity, in_position)
    """
    prepared = backtest.prepare_price_frame(price_frame)  # -> DataFrame

    dates = prepared["date"].to_numpy()    # -> ndarray[datetime64] (T,)
    prices = prepared["close"].to_numpy()  # -> ndarray[float] (T,)

    id_mask = trades["signal_id"] == signal_id            # -> Series[bool]
    horizon_mask = trades["holding_days"] == holding_days  # -> Series[bool]
    subset = trades.loc[id_mask & horizon_mask]            # -> DataFrame

    position_return = daily_position_returns(subset, dates, prices)  # -> Series[float]

    risk_free, _ = backtest.prepare_risk_free(rate_frame, prepared["date"])  # -> (Series, dict)

    equity = build_equity_curve_vectorized(
        position_return, risk_free, config.INITIAL_CAPITAL
    )  # -> ndarray[float] (T,)

    if verify:
        slow = build_equity_curve_loop(
            position_return, risk_free, config.INITIAL_CAPITAL
        )  # -> ndarray[float] (T,)

        largest_gap = float(np.max(np.abs(slow - equity)))  # -> float

        if largest_gap > 1e-6:
            raise ValueError(
                f"루프/벡터 자본곡선 불일치: 최대 {largest_gap:.3e} "
                f"({signal_id} h={holding_days})"
            )

    frame = pd.DataFrame({
        "date": prepared["date"].to_numpy(),
        "position_return": position_return.to_numpy(),
        "risk_free": risk_free.to_numpy(),
        "equity": equity,
    })  # -> DataFrame (T, 4)

    # 노출도 전용 마스크. `position_return.notna()`와 다르다 — 위 함수 참조.
    frame["in_position"] = price_exposure_mask(subset, dates)  # -> DataFrame (T, 5)

    return frame


def build_benchmark_curve(price_frame, rate_frame, benchmark_key):
    """벤치마크 자본곡선 (Buy&Hold 또는 현금 전액).

    Buy&Hold는 **비용을 넣지 않는다.** 26년에 한 번 사서 들고 있는 것이라
    왕복 20bp는 CAGR에 사실상 영향이 없고(0.2%를 26년으로 나누면 연 0.008%p),
    "벤치마크를 불리하게 잡지 않는다"가 더 중요한 원칙이다.
    """
    definition = config.BENCHMARKS[benchmark_key]  # -> dict
    ticker = definition["ticker"]                  # -> str

    prepared = backtest.prepare_price_frame(price_frame)  # -> DataFrame, ^GSPC 거래일 기준
    calendar = prepared["date"]                            # -> Series[Timestamp] (T,)

    if ticker == config.RISK_FREE_SOURCE:
        # 현금 전액 보유 — 매일 무위험수익률로만 성장한다.
        risk_free, _ = backtest.prepare_risk_free(rate_frame, calendar)  # -> (Series, dict)

        growth = 1.0 + risk_free           # -> Series[float]
        cumulative = growth.cumprod()      # -> Series[float]
        equity = float(config.INITIAL_CAPITAL) * cumulative  # -> Series[float]

        frame = pd.DataFrame({
            "date": calendar.to_numpy(),
            "equity": equity.to_numpy(),
        })  # -> DataFrame (T, 2)

        return frame

    ticker_mask = price_frame["ticker"] == ticker  # -> Series[bool]
    series = price_frame.loc[ticker_mask]          # -> DataFrame
    series = series.sort_values("date")            # -> DataFrame
    series = series[["date", "close"]]             # -> DataFrame

    merged = pd.DataFrame({"date": calendar.to_numpy()})  # -> DataFrame (T, 1)
    merged = merged.merge(series, on="date", how="left")   # -> DataFrame (T, 2)

    missing = int(merged["close"].isna().sum())  # -> int

    if missing > 0:
        raise ValueError(
            f"{ticker}에 {missing}개 거래일의 종가가 없다. "
            "보간하지 않는다 — 원인을 먼저 규명할 것 (CLAUDE.md 규칙 3)."
        )

    close = merged["close"]              # -> Series[float]
    first_close = float(close.iloc[0])   # -> float

    equity = float(config.INITIAL_CAPITAL) * (close / first_close)  # -> Series[float]

    frame = pd.DataFrame({
        "date": merged["date"].to_numpy(),
        "equity": equity.to_numpy(),
    })  # -> DataFrame (T, 2)

    return frame


# ---------------------------------------------------------------------------
# 2. 지표 — 전부 자본곡선을 받는다
# ---------------------------------------------------------------------------
def equity_daily_returns(equity):
    """자본곡선 → 일별 **단순수익률**.

    첫날은 직전 자본이 없으므로 NaN이다. 0으로 채우면 표본 수가 하루 늘어
    표준편차가 미세하게 작아진다.
    """
    values = np.asarray(equity, dtype=float)  # -> ndarray[float] (T,)

    previous = values[:-1]  # -> ndarray[float] (T-1,)
    current = values[1:]    # -> ndarray[float] (T-1,)

    returns_after_first = current / previous - 1.0  # -> ndarray[float] (T-1,)

    daily = np.empty(len(values))  # -> ndarray[float] (T,)
    daily[0] = np.nan
    daily[1:] = returns_after_first

    return daily


def total_return_pct(equity):
    """누적수익률 (%). 단순수익률 기준."""
    values = np.asarray(equity, dtype=float)  # -> ndarray[float]

    if len(values) == 0:
        return np.nan

    ratio = values[-1] / float(config.INITIAL_CAPITAL)  # -> float

    return 100.0 * (ratio - 1.0)


def cagr_pct(equity, n_days):
    """연복리 성장률 (%).

        CAGR = (최종 / 초기) ** (252 / 거래일수) - 1

    로그수익률 평균에 252를 곱한 값이 **아니다.** D2에서 그렇게 읽어 한 국면에서
    1%p 넘게 어긋난 적이 있다.
    """
    values = np.asarray(equity, dtype=float)  # -> ndarray[float]

    if len(values) == 0 or n_days <= 0:
        return np.nan

    total_growth = values[-1] / float(config.INITIAL_CAPITAL)  # -> float

    if total_growth <= 0:
        return np.nan

    years = n_days / float(config.SHARPE_PERIODS_PER_YEAR)  # -> float
    exponent = 1.0 / years                                   # -> float

    return 100.0 * (total_growth ** exponent - 1.0)


def ann_volatility_pct(daily_returns):
    """연율변동성 (%). 일별 단순수익률의 표본표준편차 × sqrt(252)."""
    clean = daily_returns[~np.isnan(daily_returns)]  # -> ndarray[float]

    if len(clean) < 2:
        return np.nan

    daily_std = float(np.std(clean, ddof=config.SHARPE_DDOF))  # -> float
    scale = np.sqrt(config.SHARPE_PERIODS_PER_YEAR)            # -> numpy.float64

    return 100.0 * daily_std * scale


def excess_returns(daily_returns, risk_free):
    """일별 초과수익 = 일별 수익률 − 일별 무위험수익률.

    `config.RF_SUBTRACTION = "daily"`가 이것을 고정한다. 연율 단계에서 한 번
    빼면 변동성이 큰 구간에서 편향된다 — 산술평균과 기하평균의 차이가 변동성에
    비례해 커지기 때문이다.
    """
    if config.RF_SUBTRACTION != "daily":
        raise ValueError(f"RF_SUBTRACTION을 처리할 수 없다: {config.RF_SUBTRACTION!r}")

    rate_values = np.asarray(risk_free, dtype=float)  # -> ndarray[float] (T,)

    return daily_returns - rate_values


def sharpe_ratio(daily_returns, risk_free):
    """샤프 비율.

        Sharpe = mean(초과수익) / std(초과수익) * sqrt(252)

    분모가 **초과수익의** 표준편차다. 수익률 자체의 표준편차를 쓰면 무위험수익률의
    변동(2000~2007년에 실재했다)이 분모에서 빠진다.
    """
    if config.SHARPE_ANNUALIZE != "sqrt_t":
        raise ValueError(f"SHARPE_ANNUALIZE를 처리할 수 없다: {config.SHARPE_ANNUALIZE!r}")

    excess = excess_returns(daily_returns, risk_free)  # -> ndarray[float]
    clean = excess[~np.isnan(excess)]                  # -> ndarray[float]

    if len(clean) < 2:
        return np.nan

    mean_excess = float(np.mean(clean))                          # -> float
    std_excess = float(np.std(clean, ddof=config.SHARPE_DDOF))   # -> float

    if std_excess == 0:
        return np.nan

    scale = np.sqrt(config.SHARPE_PERIODS_PER_YEAR)  # -> numpy.float64

    return mean_excess / std_excess * scale


def sortino_ratio(daily_returns, risk_free):
    """소르티노 비율 — 하방 변동만 위험으로 센다.

        Sortino = mean(초과수익) / 하방편차 * sqrt(252)

    **하방편차의 분모 규약을 명시한다 (config §18):** 초과수익 < MAR인 날만
    모아 표본표준편차(ddof=1)를 낸다. 전체 일수로 나누는 방식도 흔하고 값이
    다르므로, 어느 쪽인지 적지 않으면 비교가 성립하지 않는다.
    """
    excess = excess_returns(daily_returns, risk_free)  # -> ndarray[float]
    clean = excess[~np.isnan(excess)]                  # -> ndarray[float]

    if len(clean) < 2:
        return np.nan

    downside = clean[clean < config.SORTINO_MAR]  # -> ndarray[float]

    if len(downside) < 2:
        return np.nan

    mean_excess = float(np.mean(clean))                             # -> float
    downside_std = float(np.std(downside, ddof=config.SORTINO_DDOF))  # -> float

    if downside_std == 0:
        return np.nan

    scale = np.sqrt(config.SHARPE_PERIODS_PER_YEAR)  # -> numpy.float64

    return mean_excess / downside_std * scale


def max_drawdown_pct_loop(equity):
    """최대낙폭 (%) — 원리 버전. **이 함수가 정의다.**

    고점을 기억하면서 하루씩 훑는다. 반환값은 **항상 0 이하**다.
    """
    values = np.asarray(equity, dtype=float)  # -> ndarray[float]

    if len(values) == 0:
        return np.nan

    peak = values[0]      # -> float
    worst = 0.0           # -> float

    for position in range(len(values)):
        current = values[position]  # -> float

        if current > peak:
            peak = current

        drawdown = current / peak - 1.0  # -> float, 0 이하

        if drawdown < worst:
            worst = drawdown

    return 100.0 * worst


def max_drawdown_pct(equity):
    """위 루프와 같은 결과를 내는 벡터 버전."""
    series = pd.Series(np.asarray(equity, dtype=float))  # -> Series[float]

    running_peak = series.cummax()          # -> Series[float]
    drawdown = series / running_peak - 1.0  # -> Series[float]

    if len(drawdown) == 0:
        return np.nan

    return 100.0 * float(drawdown.min())


def calmar_ratio(cagr_value, mdd_value):
    """CAGR / |MDD|. 둘 다 % 단위로 받는다."""
    if np.isnan(cagr_value) or np.isnan(mdd_value):
        return np.nan

    if mdd_value == 0:
        return np.nan

    return cagr_value / abs(mdd_value)


# ---------------------------------------------------------------------------
# 3. 거래 기반 지표
# ---------------------------------------------------------------------------
def win_rate_pct(trades):
    """승률 (%). `return_pct`가 0보다 큰 거래의 비율.

    거래 로그의 `return_pct`는 **명목과 무관**하므로 복리/고정 명목 선택에
    영향받지 않는다 (`size`·`pnl`과 달리).
    """
    if len(trades) == 0:
        return np.nan

    wins = int((trades["return_pct"] > 0).sum())  # -> int

    return 100.0 * wins / len(trades)


def profit_factor(trades):
    """손익비 = 이익 거래 합 / |손실 거래 합|. `return_pct` 기준."""
    if len(trades) == 0:
        return np.nan

    values = trades["return_pct"]  # -> Series[float]

    gains = float(values[values > 0].sum())          # -> float
    losses = float(values[values < 0].sum())         # -> float

    if losses == 0:
        return np.nan  # 손실 거래가 없으면 정의되지 않는다 (inf로 두지 않는다)

    return gains / abs(losses)


def average_holding_days(trades):
    """평균 보유기간 (거래일). 고정 보유이므로 h와 같아야 정상이다."""
    if len(trades) == 0:
        return np.nan

    return float(trades["holding_days"].mean())


# ---------------------------------------------------------------------------
# 4. 요약
# ---------------------------------------------------------------------------
def summarize(name, curve, trades):
    """자본곡선 + 거래 로그 → 지표 한 줄.

    거래가 0건이면 지표는 **NaN**이다. 0이 아니다 — "거래가 없었다"와
    "수익이 0이었다"는 다른 사실이고, 0으로 두면 평균에 섞여 들어간다.
    """
    equity = curve["equity"].to_numpy()      # -> ndarray[float] (T,)
    risk_free = curve["risk_free"].to_numpy() if "risk_free" in curve.columns else None

    daily = equity_daily_returns(equity)  # -> ndarray[float] (T,)

    if risk_free is None:
        risk_free = np.zeros(len(equity))  # -> ndarray[float] (T,)

    n_trades = len(trades) if trades is not None else np.nan  # -> int | float

    cagr_value = cagr_pct(equity, len(equity))  # -> float
    mdd_value = max_drawdown_pct(equity)        # -> float

    if "in_position" in curve.columns:
        exposure = 100.0 * float(curve["in_position"].mean())  # -> float
    else:
        exposure = np.nan

    row = {
        "name": name,
        "n_trades": n_trades,
        "avg_holding_days": average_holding_days(trades) if trades is not None else np.nan,
        "exposure_pct": exposure,
        "total_return_pct": total_return_pct(equity),
        "cagr_pct": cagr_value,
        "ann_vol_pct": ann_volatility_pct(daily),
        "sharpe": sharpe_ratio(daily, risk_free),
        "sortino": sortino_ratio(daily, risk_free),
        "mdd_pct": mdd_value,
        "calmar": calmar_ratio(cagr_value, mdd_value),
        "win_rate_pct": win_rate_pct(trades) if trades is not None else np.nan,
        "profit_factor": profit_factor(trades) if trades is not None else np.nan,
    }  # -> dict (13,)

    return row
