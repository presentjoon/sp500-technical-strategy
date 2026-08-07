"""D6 사후 진단 전담 모듈 — 순열검정의 교환가능성 가정 점검.

**이 모듈의 산출물은 전부 사후 탐색적(post-hoc exploratory) 진단이다.**
확증 검정이 아니며, `docs/signal_spec.md` §6.4의 검정 family(m=20)에 어떤 검정도
추가하지 않는다. 확증 판정은 원안을 유지한다.

왜 필요한가
-----------
순열검정은 전체 거래일에서 **균등하게** 재표집한다. 이것이 타당하려면 신호일과
비신호일이 **교환가능(exchangeable)** 해야 한다. 두 가지 경로로 이 가정이 깨질 수
있고, **둘 다 p-value를 과소 추정하는 방향으로만 작동한다.**

1. **분산 불일치** — 신호일의 사후 변동성이 무조건부보다 높으면 균등 재표집으로
   만든 귀무분포가 실제보다 좁아진다. 관측값이 실제보다 극단적인 백분위에 놓인다.
2. **국면 구성** — 신호가 특정 시기에 편중 발동하면, 검정이 잡은 것이 "사건의
   특이성"이 아니라 "그 시기의 특성"일 수 있다.

계획과 판독 기준은 `reports/day06_diagnostics.md`에 **실행 전에** 기록하고
커밋했다 (커밋 06be1c1).

사용 예 (프로젝트 루트 기준)
----------------------------
    from src import config, data, diagnostics, signals

    df = data.load_parquet(config.RAW_OHLCV_PATH)
    events, universe = diagnostics.prepare_frames(df)
    variance = diagnostics.variance_diagnostic(events, universe)
    regime = diagnostics.regime_diagnostic(events, universe)
"""

import numpy as np
import pandas as pd

from src import config
from src import returns
from src import signals


# 진단에서 쓰는 표준편차는 전부 표본표준편차로 통일한다.
# pandas 기본값은 1이지만 numpy는 0이라, 명시하지 않으면 어느 쪽인지 알 수 없다.
STD_DDOF = 1  # -> int


def prepare_frames(df, ticker=None):
    """D6와 동일한 신호·사후 수익률 프레임을 만든다.

    **재계산이 아니라 재현이다.** `signals.make_signals()`와
    `signals.forward_returns()`를 그대로 호출하므로 D6와 같은 값이 나온다.
    진단을 위해 별도의 계산 경로를 만들면 두 결과가 갈렸을 때 원인을 추적할 수
    없게 된다.

    Returns
    -------
    (DataFrame, DataFrame)
        events : 신호 long format + 사후 수익률, 분석 구간·단일 티커
        universe : 전체 거래일 + 사후 수익률, 분석 구간·단일 티커
    """
    if ticker is None:
        ticker = config.TICKERS["signal"]  # -> str ("^GSPC")

    signal_frame = signals.make_signals(df)                              # -> DataFrame (92120, 5)
    forward_frame = signals.forward_returns(df)                          # -> DataFrame (18424, 13)
    merged = signals.attach_forward_returns(signal_frame, forward_frame)  # -> DataFrame (92120, 11)

    analysis_start = pd.Timestamp(config.ANALYSIS_START)  # -> Timestamp

    ticker_mask = merged["ticker"] == ticker              # -> Series[bool] (92120,)
    date_mask = merged["date"] >= analysis_start          # -> Series[bool] (92120,)
    events = merged.loc[ticker_mask & date_mask]          # -> DataFrame (33420, 11)
    events = events.reset_index(drop=True)                # -> DataFrame

    universe_ticker_mask = forward_frame["ticker"] == ticker      # -> Series[bool] (18424,)
    universe = forward_frame.loc[universe_ticker_mask]            # -> DataFrame (9212, 13)
    universe = universe.loc[universe["date"] >= analysis_start]   # -> DataFrame (6684, 13)
    universe = universe.sort_values("date")                       # -> DataFrame
    universe = universe.reset_index(drop=True)                    # -> DataFrame

    return events, universe


# ---------------------------------------------------------------------------
# 진단 A — 분산비
# ---------------------------------------------------------------------------
def variance_diagnostic(events, universe, horizons=None, permutation_table=None):
    """신호일과 전체일의 사후 수익률 산포를 비교한다.

    두 개의 t 통계량을 나란히 낸다.

        t_naive = (평균차) / (sd_all / sqrt(n))
        t_adj   = (평균차) / (sd_sig / sqrt(n))

    `t_naive`는 **균등 재표집 귀무분포가 암묵적으로 쓰는 척도**다. 순열검정이
    전체 거래일에서 뽑으므로 그 분포의 폭은 `sd_all`로 결정된다.
    `t_adj`는 **신호일 자체의 산포**로 정규화한 것이다.

    **둘의 괴리가 대안 1(분산 불일치)의 크기다.** `sd_sig`가 `sd_all`보다 크면
    `t_adj`가 `t_naive`보다 작아지고, 그만큼 순열 p-value가 낙관적이었다는 뜻이다.

    Returns
    -------
    DataFrame
        20조합 (signal_id, h, n, mean_sig, mean_all, sd_sig, sd_all, sd_ratio,
        t_naive, t_adj, p_perm).
    """
    if horizons is None:
        horizons = config.EVENT_HORIZONS  # -> list[int] (4,)

    rows = []  # -> list[dict]

    for signal_id, label, _condition_function in signals.SIGNAL_DEFINITIONS:
        fired_mask = (events["signal_id"] == signal_id) & (events["signal"])  # -> Series[bool]
        fired = events.loc[fired_mask]                                        # -> DataFrame

        for horizon in horizons:
            # EX-POST ONLY: 사후 평가용. signals.py로 역류 금지
            column_name = f"fwd_ret_{horizon}"  # -> str

            signal_values = fired[column_name].dropna()        # -> Series[float] (n,)
            universe_values = universe[column_name].dropna()   # -> Series[float] (유효 거래일,)

            count = len(signal_values)  # -> int

            if count == 0:
                continue

            mean_signal = float(signal_values.mean())     # -> float
            mean_universe = float(universe_values.mean())  # -> float

            # ddof를 명시한다. 기본값에 맡기면 pandas(1)와 numpy(0)가 갈린다.
            sd_signal = float(signal_values.std(ddof=STD_DDOF))       # -> float
            sd_universe = float(universe_values.std(ddof=STD_DDOF))   # -> float

            sd_ratio = sd_signal / sd_universe  # -> float

            difference = mean_signal - mean_universe  # -> float
            root_n = np.sqrt(count)                   # -> numpy.float64

            t_naive = difference / (sd_universe / root_n)  # -> float, 균등 재표집이 쓰는 척도
            t_adj = difference / (sd_signal / root_n)      # -> float, 신호일 산포로 정규화

            row = {
                "signal_id": signal_id,
                "label": label,
                "h": horizon,
                "n": count,
                "mean_sig": mean_signal,
                "mean_all": mean_universe,
                "sd_sig": sd_signal,
                "sd_all": sd_universe,
                "sd_ratio": sd_ratio,
                "t_naive": t_naive,
                "t_adj": t_adj,
            }  # -> dict

            rows.append(row)

    table = pd.DataFrame(rows)  # -> DataFrame (20, 11)

    # D6 순열검정 p-value를 그대로 옮겨 붙인다. 다시 계산하지 않는다.
    if permutation_table is not None:
        keep = permutation_table[["signal_id", "h", "p"]]  # -> DataFrame (20, 3)
        keep = keep.rename(columns={"p": "p_perm"})        # -> DataFrame (20, 3)

        before_rows = len(table)  # -> int
        table = table.merge(keep, on=["signal_id", "h"], how="left")  # -> DataFrame (20, 12)

        if len(table) != before_rows:
            raise ValueError("순열 p-value 조인 후 행 수가 달라졌다. 중복을 확인할 것.")

    return table


def classify_variance(sd_ratio):
    """진단 A의 판독 기준 (실행 전 확정, day06_diagnostics.md §4).

    > 1.5   : 순열 p-value를 액면대로 해석하지 않는다
    1.2~1.5 : 주의 구간
    < 1.2   : 분산 불일치가 실질적 문제는 아니다
    """
    if sd_ratio > 1.5:
        return "액면 해석 불가"

    if sd_ratio >= 1.2:
        return "주의"

    return "양호"


# ---------------------------------------------------------------------------
# 진단 B — 국면별 발동 편중
# ---------------------------------------------------------------------------
def regime_diagnostic(events, universe, phases=None, horizon=1):
    """신호가 특정 국면에 편중 발동하는지 본다.

    핵심은 `mean_uncond_h1`과 `mean_sig_h1`을 나란히 두는 것이다.
    **국면 내에서도 신호일이 전체일보다 나쁘면 대안 2로 전부 설명되지 않는다.**
    즉 "하락 국면이라서 나빴다"가 아니라 "그 국면 안에서도 신호일이 특별히
    나빴다"는 뜻이 된다.

    국면 라벨은 사후 정보이며 (CLAUDE.md 규칙 4) 진단 전용이다.
    신호 생성에 들어가지 않는다.

    Returns
    -------
    DataFrame
        (signal_id, regime, n_days, n_sig, rate, rate_ratio, share_sig,
        share_days, mean_uncond_h1, mean_sig_h1).
    """
    if phases is None:
        phases = config.MARKET_REGIMES  # -> dict[str, dict] (6,)

    phased_universe = returns.tag_phase(universe, phases=phases)  # -> DataFrame (6684, 14)
    phased_events = returns.tag_phase(events, phases=phases)      # -> DataFrame (33420, 12)

    total_days = len(phased_universe)  # -> int (6684)

    # EX-POST ONLY: 사후 평가용. signals.py로 역류 금지
    return_column = f"fwd_ret_{horizon}"  # -> str

    phase_keys = list(phases.keys())  # -> list[str] (6,)

    rows = []  # -> list[dict]

    for signal_id, label, _condition_function in signals.SIGNAL_DEFINITIONS:
        signal_mask = phased_events["signal_id"] == signal_id  # -> Series[bool]
        signal_rows = phased_events.loc[signal_mask]           # -> DataFrame (6684, 12)

        fired_rows = signal_rows.loc[signal_rows["signal"]]  # -> DataFrame (사건 수, 12)
        total_signals = len(fired_rows)                      # -> int

        overall_rate = total_signals / total_days  # -> float

        for phase_key in phase_keys:
            phase_days_mask = phased_universe["phase"] == phase_key  # -> Series[bool]
            phase_universe = phased_universe.loc[phase_days_mask]    # -> DataFrame (국면 거래일, 14)

            phase_signal_mask = fired_rows["phase"] == phase_key  # -> Series[bool]
            phase_signals = fired_rows.loc[phase_signal_mask]     # -> DataFrame (국면 사건 수, 12)

            days_in_phase = len(phase_universe)   # -> int
            signals_in_phase = len(phase_signals)  # -> int

            if days_in_phase == 0:
                continue

            phase_rate = signals_in_phase / days_in_phase  # -> float

            if overall_rate > 0:
                rate_ratio = phase_rate / overall_rate  # -> float
            else:
                rate_ratio = float("nan")  # -> float

            if total_signals > 0:
                share_signal = signals_in_phase / total_signals  # -> float
            else:
                share_signal = float("nan")  # -> float

            share_days = days_in_phase / total_days  # -> float

            unconditional = phase_universe[return_column].dropna()  # -> Series[float]
            conditional = phase_signals[return_column].dropna()     # -> Series[float]

            if len(unconditional) > 0:
                mean_unconditional = float(unconditional.mean())  # -> float
            else:
                mean_unconditional = float("nan")  # -> float

            if len(conditional) > 0:
                mean_conditional = float(conditional.mean())  # -> float
            else:
                mean_conditional = float("nan")  # -> float

            rows.append(
                {
                    "signal_id": signal_id,
                    "label": label,
                    "regime": phase_key,
                    "regime_name": phases[phase_key]["name"],
                    "n_days": days_in_phase,
                    "n_sig": signals_in_phase,
                    "rate": phase_rate,
                    "rate_ratio": rate_ratio,
                    "share_sig": share_signal,
                    "share_days": share_days,
                    f"mean_uncond_h{horizon}": mean_unconditional,
                    f"mean_sig_h{horizon}": mean_conditional,
                }
            )

    table = pd.DataFrame(rows)  # -> DataFrame (30, 12)

    ordered = pd.Categorical(table["regime"], categories=phase_keys, ordered=True)
    table["regime"] = ordered
    table = table.sort_values(["signal_id", "regime"])  # -> DataFrame
    table = table.reset_index(drop=True)                # -> DataFrame

    return table


# 국면별 표본이 이 값 미만이면 표에 값은 적되 해석하지 않는다
# (day06_diagnostics.md §5, 실행 전 확정).
MIN_REGIME_SIGNALS = 10  # -> int


# ---------------------------------------------------------------------------
# 사전 구간 곡선
# ---------------------------------------------------------------------------
def pre_event_curve(events, universe, signal_id, window=5, price_column="close"):
    """신호일 기준 tau = -window ... +window 의 평균 누적수익률 곡선.

    **tau < 0 구간이 급락임을 보이는 것이 목적**이며, 대안 2(국면 구성)의
    시각적 근거다. 신호가 "이미 떨어진 뒤"에 발동한다면, 신호일과 무작위일의
    차이는 사건의 특이성이 아니라 진입 시점의 편중일 수 있다.

    tau=0을 기준점(누적수익률 0)으로 두고 앞뒤를 잇는다.

    Returns
    -------
    DataFrame
        (tau, mean_cum_return, n)
    """
    price_series = universe[price_column]           # -> Series[float] (6684,)
    price_values = price_series.to_numpy()          # -> ndarray[float] (6684,)

    date_to_position = universe.reset_index()               # -> DataFrame (6684, 14)
    date_to_position = date_to_position.set_index("date")["index"]  # -> Series[int] (6684,)

    fired_mask = (events["signal_id"] == signal_id) & (events["signal"])  # -> Series[bool]
    fired_dates = events.loc[fired_mask, "date"]                          # -> Series[datetime64]

    positions = date_to_position.reindex(fired_dates)  # -> Series[int] (사건 수,)
    positions = positions.dropna()                     # -> Series[float]
    positions = positions.astype(int).to_numpy()       # -> ndarray[int] (사건 수,)

    offsets = np.arange(-window, window + 1)  # -> ndarray[int] (2*window+1,)

    rows = []  # -> list[dict]

    for offset in offsets:
        target = positions + offset  # -> ndarray[int] (사건 수,)

        valid = (target >= 0) & (target < len(price_values))  # -> ndarray[bool]
        usable_base = positions[valid]                         # -> ndarray[int]
        usable_target = target[valid]                          # -> ndarray[int]

        base_price = price_values[usable_base]      # -> ndarray[float]
        target_price = price_values[usable_target]  # -> ndarray[float]

        cumulative = target_price / base_price - 1  # -> ndarray[float], tau=0 기준 누적수익률

        rows.append(
            {
                "tau": int(offset),
                "mean_cum_return": float(cumulative.mean()),
                "n": int(len(cumulative)),
            }
        )

    return pd.DataFrame(rows)  # -> DataFrame (2*window+1, 3)


# ---------------------------------------------------------------------------
# B 재실행
# ---------------------------------------------------------------------------
def permutation_pvalue(events, universe, signal_id, horizon, iterations, seed):
    """단일 조합의 순열검정을 다시 돌린다 (명세 §6.5와 동일한 절차).

    **B만 바꾸고 나머지는 전부 동일하다.** B는 추정 대상을 바꾸지 않는 순수
    몬테카를로 계산 파라미터이므로 증가 자체는 HARKing이 아니다. 다만 경계
    사례에서 원하는 답이 나올 때까지 올리는 조작이 가능하므로, 사전에 1회만
    실행하기로 약속하고 결과와 무관하게 기록한다.

    Returns
    -------
    dict
    """
    # EX-POST ONLY: 사후 평가용. signals.py로 역류 금지
    column_name = f"fwd_ret_{horizon}"  # -> str

    fired_mask = (events["signal_id"] == signal_id) & (events["signal"])  # -> Series[bool]
    observed = events.loc[fired_mask, column_name].dropna().to_numpy()    # -> ndarray[float] (n,)

    pool = universe[column_name].dropna().to_numpy()  # -> ndarray[float] (유효 거래일,)

    count = len(observed)  # -> int

    pool_mean = float(pool.mean())          # -> float
    observed_mean = float(observed.mean())  # -> float
    statistic = abs(observed_mean - pool_mean)  # -> float

    generator = np.random.default_rng(seed)  # -> Generator

    extreme_count = 0  # -> int

    for _iteration in range(iterations):
        drawn = generator.choice(pool, size=count, replace=False)  # -> ndarray[float] (n,)
        drawn_mean = drawn.mean()                                  # -> numpy.float64

        if abs(drawn_mean - pool_mean) >= statistic:
            extreme_count = extreme_count + 1

    p_value = (1 + extreme_count) / (iterations + 1)  # -> float

    standard_error = np.sqrt(p_value * (1 - p_value) / iterations)  # -> numpy.float64

    return {
        "signal_id": signal_id,
        "h": horizon,
        "n": count,
        "iterations": iterations,
        "seed": seed,
        "extreme_count": extreme_count,
        "p": p_value,
        "mc_se": float(standard_error),
        "ci_low": float(max(0.0, p_value - 1.96 * standard_error)),
        "ci_high": float(p_value + 1.96 * standard_error),
    }
