"""D12 층화 순열검정. 종속변수는 사후 수익률(post-signal return)이다.
경로변동성은 본 모듈의 대상이 아니며 pathvol.py에서 다룬다.

사전등록: `docs/prereg_d12_stratified_permutation.md` (커밋 `4c5533a`).
§1~§5는 고정이며 이 모듈은 그 규정을 구현할 뿐 스스로 정하지 않는다.

왜 파일을 나눴는가
------------------
D8 사전등록이 폐기된 원인은 **종속변수 혼동**이었다 — "사후 20일 실현변동성"을
전제했으나 근거로 삼은 D7의 값은 h=1 사후 **수익률**의 횡단면 표준편차였다.
같은 사고를 막기 위해 파일 경계로 갈라 둔다. 이 모듈에 경로변동성이 들어오면
그 자체가 설계 위반이다.

D7과 무엇이 같고 무엇이 다른가
------------------------------
    같음   검정통계량 — 스튜던트화 t 형태 (diagnostics.studentized_permutation)
           B = 10,000, 시드 42, 양측
    다름   재표집 단위 — 전체 거래일 → **층 내부**

통계량을 그대로 두는 이유는 D7 결과와 나란히 놓기 위해서다. 통계량과 재표집을
동시에 바꾸면 차이가 어느 쪽에서 왔는지 알 수 없게 된다.

층화 통계량의 형태
------------------
신호 관측 $i$ (층 $q$에 속함)에 대해 **자기 층의 비신호일 평균**을 뺀 값

    e_i = r_i − mean(pool_q)

를 만들고, 이 e에 D7과 같은 t를 적용한다.

    T = mean(e) / (sd(e) / sqrt(n))     ddof = 1

층이 하나뿐이면 D7의 식과 정확히 같아진다. `mean(e)`가 §3이 예측한
**효과 크기(%p)** 이고, 그 부호가 §4 판정 항목 2번의 대상이다.

사용 예 (프로젝트 루트 기준)
----------------------------
    from src import config, data, stratified

    df = data.load_parquet(config.RAW_OHLCV_PATH)
    bins = stratified.build_bin_frame(df)
    results, checks = stratified.run_all(df, bins)
"""

import numpy as np
import pandas as pd

from src import config
from src import pathvol
from src import signals


# 표준편차는 전부 표본표준편차. D7(`diagnostics.STD_DDOF`)과 같은 값이다.
STD_DDOF = 1  # -> int

# 사전등록 §2-2. 판독 기준이지 확증 유의수준이 아니다.
ALPHA_READ = 0.05 / 4  # -> float (0.0125)

# 사전등록 T3. 초과 관측이 이 값 미만이면 "해상도 한계"로 병기한다.
# 근거: 상대 몬테카를로 오차 1/sqrt(k)가 k=10에서 32%.
RESOLUTION_MIN_COUNT = 10  # -> int

# §5-1 기준값 — reports/day08b_pathvol.csv의 n_signal 벡터
D8B_SIGNAL_COUNTS = {1: 1, 2: 2, 3: 10, 4: 20, 5: 32}  # -> dict[int, int]

# §4 판정 항목 2번 — D6 실측 부호 (reports/day06_event_study.csv)
D6_EXCESS_PCT = {1: -0.502037, 5: -0.197561, 10: -0.026388, 20: 1.188019}  # -> dict[int, float]

BINS_COLUMNS = ["date", "quintile", "pre_vol", "is_signal", "in_analysis_period"]  # -> list[str] (5,)


# ---------------------------------------------------------------------------
# 1. 층 배정 — pathvol의 정의를 그대로 쓴다
# ---------------------------------------------------------------------------
def build_bin_frame(df, signal_id="S1_rsi_oversold", ticker=None, n_bins=5,
                    analysis_start=None):
    """전체 거래일에 대해 (date, quintile, pre_vol, is_signal, 분석구간 여부)를 만든다.

    **층 배정 규칙을 새로 정하지 않는다.** `pathvol.add_pre_volatility()`와
    `pathvol.expanding_quantile_bin()`을 그대로 호출한다.

    `pathvol.build_frame()`을 쓰지 않는 이유
    ----------------------------------------
    그 함수는 `post_vol`(경로변동성)이 유효한 행만 남긴다. D12의 종속변수는
    **수익률**이므로, 경로변동성 창이 미완성이라는 이유로 마지막 21거래일을
    버릴 근거가 없다. 층 배정의 **정의**는 재사용하되 **행 필터는 따르지 않는다.**

    분위 경계는 전체 9,212거래일을 후보 풀로 훑되 각 시점은 그 이전 자료만
    참조한다 (사전등록 §1-3). 그래서 분석구간으로 자르기 **전에** 배정한다.

    Returns
    -------
    DataFrame (거래일 수, 5)
        분위 미배정 구간(최소 과거 750거래일 미달)은 `quintile`이 NaN.
    """
    if ticker is None:
        ticker = config.TICKERS["signal"]  # -> str ("^GSPC")

    if analysis_start is None:
        analysis_start = config.ANALYSIS_START  # -> str

    with_returns = pathvol.add_log_return(df)            # -> DataFrame (18424, 9)
    with_pre = pathvol.add_pre_volatility(with_returns)  # -> DataFrame (18424, 10)

    ticker_mask = with_pre["ticker"] == ticker  # -> Series[bool] (18424,)
    frame = with_pre.loc[ticker_mask]           # -> DataFrame (9212, 10)
    frame = frame.sort_values("date")           # -> DataFrame
    frame = frame.reset_index(drop=True)        # -> DataFrame (9212, 10)

    signal_frame = signals.make_signals(df)                       # -> DataFrame (92120, 5)
    id_mask = signal_frame["signal_id"] == signal_id              # -> Series[bool]
    signal_ticker_mask = signal_frame["ticker"] == ticker         # -> Series[bool]
    one_signal = signal_frame.loc[id_mask & signal_ticker_mask]   # -> DataFrame (9212, 5)
    one_signal = one_signal[["date", "signal"]]                   # -> DataFrame (9212, 2)

    merged = frame.merge(one_signal, on="date", how="left")  # -> DataFrame (9212, 11)

    if len(merged) != len(frame):
        raise ValueError(
            f"신호 병합에서 행 수가 변했다: {len(frame)} → {len(merged)}. "
            "date 중복을 의심하라."
        )

    pre_values = merged["pre_vol"].to_numpy()                        # -> ndarray[float] (9212,)
    quintile = pathvol.expanding_quantile_bin(pre_values, n_bins)    # -> ndarray[float] (9212,)

    start = pd.Timestamp(analysis_start)  # -> Timestamp

    result = pd.DataFrame({
        "date": merged["date"].to_numpy(),
        "quintile": quintile,
        "pre_vol": merged["pre_vol"].to_numpy(),
        "is_signal": merged["signal"].fillna(False).to_numpy(dtype=bool),
        "in_analysis_period": (merged["date"] >= start).to_numpy(),
    })  # -> DataFrame (9212, 5)

    return result[BINS_COLUMNS]


def check_bin_assignment(bins):
    """§5-1 — 층별 신호 건수가 D8b 집계표와 일치하는지.

    **해시 대조가 아니라 집계표 대조다.** D8b가 `(date, quintile)` 배열을
    저장하지 않아 해시 대조가 불가능하기 때문이며, 그 경위와 검증력 한계는
    사전등록 §5-1에 적혀 있다.

    D8b 프레임은 `post_vol` 유효 행만 남겼으므로(6,663행), 대조도 **같은 조건**
    에서 해야 한다. 조건이 다르면 건수가 달라지는 것이 정상이라 비교가 성립하지 않는다.

    Returns
    -------
    (bool, dict)
        일치 여부와 층별 실측 건수.
    """
    analysis = bins.loc[bins["in_analysis_period"]]  # -> DataFrame (6684, 5)

    signal_rows = analysis.loc[analysis["is_signal"]]  # -> DataFrame (65, 5)

    counted = {}  # -> dict[int, int]

    for bin_number in sorted(D8B_SIGNAL_COUNTS):
        match = signal_rows["quintile"] == bin_number  # -> Series[bool]
        counted[bin_number] = int(match.sum())

    matches = counted == D8B_SIGNAL_COUNTS  # -> bool

    return matches, counted


# ---------------------------------------------------------------------------
# 2. 지평별 프레임 — 층 배정 × 사후 수익률
# ---------------------------------------------------------------------------
def attach_returns(df, bins, horizon, ticker=None, analysis_start=None):
    """층 배정에 사후 수익률을 붙인다.

    **여기가 실제 위험 지점이다.** 층 배정은 9,212행이고 사후 수익률은 지평마다
    유효 행이 다르다(6,682 / 6,678 / 6,673 / 6,663). 붙이는 자리에서 행이
    늘거나 줄면 조용히 다른 표본을 검정하게 된다. 그래서 병합 직후 행 수와
    신호 건수를 둘 다 검사한다.

    Returns
    -------
    (DataFrame, dict)
        (date, quintile, is_signal, ret) — 분석구간, `ret` 유효, 분위 배정된 행만.
        그리고 병합 검사 결과.
    """
    if ticker is None:
        ticker = config.TICKERS["signal"]  # -> str

    if analysis_start is None:
        analysis_start = config.ANALYSIS_START  # -> str

    forward = signals.forward_returns(df)  # -> DataFrame (18424, 13)

    ticker_mask = forward["ticker"] == ticker  # -> Series[bool]
    one = forward.loc[ticker_mask]             # -> DataFrame (9212, 13)
    one = one.sort_values("date")              # -> DataFrame

    # EX-POST ONLY: 사후 평가용. signals.py로 역류 금지
    column_name = f"fwd_ret_{horizon}"  # -> str

    one = one[["date", column_name]]  # -> DataFrame (9212, 2)

    before_rows = len(bins)  # -> int

    merged = bins.merge(one, on="date", how="left")  # -> DataFrame (9212, 6)

    after_rows = len(merged)  # -> int

    if after_rows != before_rows:
        raise ValueError(
            f"h={horizon} 병합에서 행 수가 변했다: {before_rows} → {after_rows}. "
            "date 중복을 의심하라."
        )

    merged = merged.rename(columns={column_name: "ret"})  # -> DataFrame

    start = pd.Timestamp(analysis_start)  # -> Timestamp

    keep = merged["in_analysis_period"]           # -> Series[bool]
    keep = keep & merged["ret"].notna()           # -> Series[bool]
    keep = keep & merged["quintile"].notna()      # -> Series[bool]

    usable = merged.loc[keep]                # -> DataFrame
    usable = usable.reset_index(drop=True)   # -> DataFrame

    check = {
        "h": horizon,
        "rows_before_merge": before_rows,
        "rows_after_merge": after_rows,
        "rows_unchanged": after_rows == before_rows,
        "usable_rows": len(usable),
        "signals_all_strata": int(usable["is_signal"].sum()),
    }  # -> dict

    return usable[["date", "quintile", "is_signal", "ret"]], check


def stratum_table(usable, min_signals=None):
    """층별 신호 수·비신호 수·평균과 포함 여부.

    사전등록 §1-2: 층당 신호가 `min_signals` 미만이면 제외한다.
    """
    if min_signals is None:
        min_signals = pathvol.MIN_SIGNAL_PER_BIN  # -> int (5)

    rows = []  # -> list[dict]

    for bin_number in sorted(D8B_SIGNAL_COUNTS):
        subset = usable.loc[usable["quintile"] == bin_number]  # -> DataFrame

        signal_rows = subset.loc[subset["is_signal"]]       # -> DataFrame
        pool_rows = subset.loc[~subset["is_signal"]]        # -> DataFrame

        n_signal = len(signal_rows)  # -> int
        n_pool = len(pool_rows)      # -> int

        mean_signal = float(signal_rows["ret"].mean()) if n_signal else np.nan  # -> float
        mean_pool = float(pool_rows["ret"].mean()) if n_pool else np.nan        # -> float

        rows.append({
            "quintile": bin_number,
            "n_signal": n_signal,
            "n_pool": n_pool,
            "mean_signal_pct": 100.0 * mean_signal,
            "mean_pool_pct": 100.0 * mean_pool,
            "diff_pct": 100.0 * (mean_signal - mean_pool),
            "included": n_signal >= min_signals,
        })

    return pd.DataFrame(rows)  # -> DataFrame (5, 7)


# ---------------------------------------------------------------------------
# 3. 층화 스튜던트화 순열검정
# ---------------------------------------------------------------------------
def _centered_signals(usable, included_bins):
    """신호 관측을 **자기 층의 비신호일 평균**으로 중심화한다.

    Returns
    -------
    (ndarray[float], dict, dict)
        중심화된 값 e, 층별 신호 수, 층별 비신호 풀.
    """
    centered = []       # -> list[float]
    sizes = {}          # -> dict[int, int]
    pools = {}          # -> dict[int, ndarray]

    for bin_number in included_bins:
        subset = usable.loc[usable["quintile"] == bin_number]  # -> DataFrame

        signal_values = subset.loc[subset["is_signal"], "ret"].to_numpy()   # -> ndarray[float]
        pool_values = subset.loc[~subset["is_signal"], "ret"].to_numpy()    # -> ndarray[float]

        pool_mean = float(pool_values.mean())  # -> float

        centered.extend(signal_values - pool_mean)

        sizes[bin_number] = len(signal_values)
        pools[bin_number] = pool_values

    return np.asarray(centered, dtype=float), sizes, pools


def _studentized_t(values):
    """D7과 같은 형태. `values`는 이미 중심화된 e."""
    count = len(values)  # -> int

    if count < 2:
        return np.nan

    standard_deviation = float(np.std(values, ddof=STD_DDOF))  # -> float

    if standard_deviation == 0:
        return np.nan

    root_n = np.sqrt(count)  # -> numpy.float64

    return float(np.mean(values)) / (standard_deviation / root_n)


def stratified_permutation(usable, included_bins, iterations=None, seed=None):
    """층 내에서만 재표집하는 스튜던트화 순열검정.

    각 반복에서 **층별로** 그 층의 비신호일 풀에서 `n_q`개를 비복원 추출하고,
    같은 층 평균으로 중심화한 뒤 이어 붙여 t를 다시 계산한다.
    층 경계를 넘지 않으므로 층별 신호 개수가 매 반복에서 보존된다 (§5 검증 2).

    Returns
    -------
    dict
    """
    if iterations is None:
        iterations = config.PERMUTATION_ITERATIONS  # -> int (10000)

    if seed is None:
        seed = config.PERMUTATION_SEED  # -> int (42)

    centered, sizes, pools = _centered_signals(usable, included_bins)  # -> (ndarray, dict, dict)

    observed_t = _studentized_t(centered)  # -> float
    effect_pct = 100.0 * float(np.mean(centered))  # -> float, §3이 예측한 효과 크기

    if np.isnan(observed_t):
        return {
            "observed_t": np.nan, "effect_pct": effect_pct,
            "extreme_count": 0, "p_value": np.nan,
            "n_signal": len(centered), "size_preserved": True,
        }

    statistic = abs(observed_t)  # -> float

    generator = np.random.default_rng(seed)  # -> Generator

    extreme_count = 0      # -> int
    size_preserved = True  # -> bool

    for _iteration in range(iterations):
        drawn = []  # -> list[float]

        for bin_number in included_bins:
            pool_values = pools[bin_number]  # -> ndarray[float]
            take = sizes[bin_number]         # -> int

            sample = generator.choice(pool_values, size=take, replace=False)  # -> ndarray[float]

            if len(sample) != take:
                size_preserved = False

            drawn.extend(sample - float(pool_values.mean()))

        drawn_array = np.asarray(drawn, dtype=float)  # -> ndarray[float]

        drawn_t = _studentized_t(drawn_array)  # -> float

        if np.isnan(drawn_t):
            continue

        if abs(drawn_t) >= statistic:
            extreme_count = extreme_count + 1

    p_value = (1 + extreme_count) / (iterations + 1)  # -> float

    return {
        "observed_t": observed_t,
        "effect_pct": effect_pct,
        "extreme_count": extreme_count,
        "p_value": p_value,
        "n_signal": len(centered),
        "size_preserved": size_preserved,
    }


# ---------------------------------------------------------------------------
# 4. 사전등록 §4 — 판정 항목과 기술적 보고 항목
# ---------------------------------------------------------------------------
def leave_one_out(usable, included_bins, iterations=None, seed=None):
    """판정 항목 1 — 포함된 층을 하나씩 빼고 다시 검정한다.

    **임의 백분율 없이 이진 판정**이 되도록 설계했다 (사전등록 §4-2 1-비고).
    하나라도 $p > \\alpha_{read}$면 flag.

    이 3회는 판정 재량을 제거하기 위한 진단 절차이며 새로운 확증 검정이 아니다.
    따라서 다중검정 family에 포함하지 않는다.
    """
    rows = []  # -> list[dict]

    for dropped in included_bins:
        remaining = [b for b in included_bins if b != dropped]  # -> list[int]

        if len(remaining) == 0:
            continue

        outcome = stratified_permutation(usable, remaining, iterations, seed)  # -> dict

        rows.append({
            "dropped_quintile": dropped,
            "p_value": outcome["p_value"],
            "effect_pct": outcome["effect_pct"],
            "n_signal": outcome["n_signal"],
        })

    return pd.DataFrame(rows)  # -> DataFrame (3, 4)


def stratum_gap_median(usable, included_bins, dates_index):
    """판정 항목 3 — 층 내 사건 간격 중앙값 (거래일).

    `signal_spec.md` §6.6의 "간격 중앙값이 h보다 작으면 중첩 심각" 규칙을
    **층 단위로** 적용한다. 새 임계를 만드는 것이 아니라 기존 사전등록 규칙의
    적용 범위를 좁히는 것이다.
    """
    rows = []  # -> list[dict]

    for bin_number in included_bins:
        subset = usable.loc[usable["quintile"] == bin_number]  # -> DataFrame
        signal_dates = subset.loc[subset["is_signal"], "date"]  # -> Series[Timestamp]

        positions = [dates_index[pd.Timestamp(d)] for d in signal_dates]  # -> list[int]
        positions = sorted(positions)                                      # -> list[int]

        if len(positions) < 2:
            median_gap = np.nan
        else:
            gaps = np.diff(np.asarray(positions))  # -> ndarray[int]
            median_gap = float(np.median(gaps))    # -> float

        rows.append({"quintile": bin_number, "gap_median": median_gap})

    return pd.DataFrame(rows)  # -> DataFrame


def stratum_contribution(table, included_bins):
    """T1 — 층별 기여도 (%). 판정에 쓰지 않는다.

    기여도_q = (n_q / n) * diff_q / (가중평균 diff) * 100
    """
    included = table.loc[table["quintile"].isin(included_bins)]  # -> DataFrame

    weights = included["n_signal"].to_numpy(dtype=float)  # -> ndarray[float]
    diffs = included["diff_pct"].to_numpy(dtype=float)    # -> ndarray[float]

    total_weight = weights.sum()  # -> float

    if total_weight == 0:
        return {}

    weighted = weights * diffs / total_weight  # -> ndarray[float]
    total = weighted.sum()                     # -> float

    if total == 0:
        return {int(q): np.nan for q in included["quintile"]}

    shares = 100.0 * weighted / total  # -> ndarray[float]

    return {int(q): float(s) for q, s in zip(included["quintile"], shares)}


# ---------------------------------------------------------------------------
# 5. 실행
# ---------------------------------------------------------------------------
def run_all(df, bins, horizons=None, iterations=None, seed=None, expected_signals=62):
    """4지평 전부. 결과 방향과 무관하게 전량 산출한다 (§2-3).

    Returns
    -------
    (DataFrame, list[dict])
        지평별 결과표와 검증 기록.
    """
    if horizons is None:
        horizons = config.HOLDING_DAYS  # -> list[int] (4,)

    date_order = bins["date"].tolist()  # -> list[Timestamp]
    dates_index = {}                     # -> dict[Timestamp, int]

    for position in range(len(date_order)):
        dates_index[pd.Timestamp(date_order[position])] = position

    rows = []    # -> list[dict]
    checks = []  # -> list[dict]

    for horizon in horizons:
        usable, check = attach_returns(df, bins, horizon)  # -> (DataFrame, dict)

        table = stratum_table(usable)  # -> DataFrame (5, 7)

        included_bins = [int(q) for q in table.loc[table["included"], "quintile"]]  # -> list[int]
        excluded_bins = [int(q) for q in table.loc[~table["included"], "quintile"]]  # -> list[int]

        kept = int(table.loc[table["included"], "n_signal"].sum())     # -> int
        dropped = int(table.loc[~table["included"], "n_signal"].sum())  # -> int

        check["included_strata"] = len(included_bins)
        check["excluded_strata"] = len(excluded_bins)
        check["signals_kept"] = kept
        check["signals_dropped"] = dropped
        check["signals_kept_equals_expected"] = (kept == expected_signals)

        main = stratified_permutation(usable, included_bins, iterations, seed)  # -> dict

        loo = leave_one_out(usable, included_bins, iterations, seed)  # -> DataFrame
        gaps = stratum_gap_median(usable, included_bins, dates_index)  # -> DataFrame
        shares = stratum_contribution(table, included_bins)            # -> dict

        # --- 판정 항목 1: leave-one-out ---
        loo_max_p = float(loo["p_value"].max()) if len(loo) else np.nan  # -> float
        flag_dominance = bool(loo_max_p > ALPHA_READ) if not np.isnan(loo_max_p) else False

        # --- 판정 항목 2: 부호 정합 ---
        expected_sign = np.sign(D6_EXCESS_PCT[horizon])       # -> float
        observed_sign = np.sign(main["effect_pct"])           # -> float
        flag_sign = bool(observed_sign != expected_sign)

        # --- 판정 항목 3: 층 내 중첩 ---
        below = gaps["gap_median"] < horizon                  # -> Series[bool]
        flagged_gap = [int(q) for q in gaps.loc[below, "quintile"]]  # -> list[int]
        flag_overlap = len(flagged_gap) > 0

        # --- T2: 제외 층 복원 (참고값, 판정 아님) ---
        all_bins = [int(q) for q in table.loc[table["n_signal"] > 0, "quintile"]]  # -> list[int]
        restored = stratified_permutation(usable, all_bins, iterations, seed)      # -> dict

        # --- T3: p 해상도 ---
        low_resolution = main["extreme_count"] < RESOLUTION_MIN_COUNT  # -> bool

        row = {
            "h": horizon,
            "n_signal": main["n_signal"],
            "included_strata": len(included_bins),
            "excluded_strata": len(excluded_bins),
            "signals_dropped": dropped,
            "effect_pct": main["effect_pct"],
            "observed_t": main["observed_t"],
            "p_value": main["p_value"],
            "extreme_count": main["extreme_count"],
            "alpha_read": ALPHA_READ,
            "significant": bool(main["p_value"] <= ALPHA_READ),
            # 판정 항목
            "flag_dominance": flag_dominance,
            "loo_max_p": loo_max_p,
            "flag_sign": flag_sign,
            "d6_excess_pct": D6_EXCESS_PCT[horizon],
            "flag_overlap": flag_overlap,
            "overlap_flagged_strata": ";".join(str(q) for q in flagged_gap),
            # 기술적 보고 항목
            "T1_contribution_pct": ";".join(
                f"Q{q}={shares[q]:.2f}" for q in sorted(shares)
            ),
            "T2_p_all_strata": restored["p_value"],
            "T2_delta_p": restored["p_value"] - main["p_value"],
            "T3_low_resolution": low_resolution,
        }  # -> dict

        for _, gap_row in gaps.iterrows():
            row[f"gap_median_Q{int(gap_row['quintile'])}"] = gap_row["gap_median"]

        rows.append(row)
        checks.append(check)

    return pd.DataFrame(rows), checks
