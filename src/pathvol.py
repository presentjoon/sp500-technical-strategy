"""D8b 사후 경로변동성 층화 반증 테스트 전담 모듈.

사전등록: `docs/prereg_d08b_pathvol.md` (커밋 f2f2947).
계획과 판정 기준은 **실행 전에** 확정·커밋했다.

이 모듈이 재는 양은 D7과 다르다 — 혼동 금지
--------------------------------------------
`diagnostics.variance_diagnostic()`의 `sd_ratio`와 본 모듈의 `ratio`는
이름이 비슷하지만 **다른 양이다.** D8(폐기)이 이 둘을 같은 것으로 전제했다가
사전등록 전체를 폐기했다 (`docs/prereg_d08_stratified.md` 상단 폐기 블록).

===================  =======================================  =======================
항목                 D7 `sd_ratio`                            D8b `ratio` (본 모듈)
===================  =======================================  =======================
사건당 값            수익률 **1개** (C_{t+1+h}/C_{t+1} - 1)   변동성 **1개**
                                                              (일별 로그수익률 20개의 sd)
집계 방식            65개 값의 **횡단면 표준편차**            65개 값의 **평균**
재는 것              사건 간 결과가 얼마나 흩어지는가         진입 후 경로가 얼마나 출렁였는가
비교군               전체 거래일 (**신호일 65건 포함**)       **비신호일만**
S1 실측              2.1437 (h=1) / 1.4588 (h=20)             본 모듈이 산출 (R0)
===================  =======================================  =======================

**차이는 전부 의도된 것이다.** 비교군에서 신호일을 뺀 것도 사전등록 §2에
명시한 설계이지 버그가 아니다 (D7 불일치 C 해소).

`post_vol` 창의 시작점 — 왜 t+2인가
------------------------------------
`post_vol(t)`는 ``ln(C_{t+2}/C_{t+1})`` 부터 ``ln(C_{t+21}/C_{t+20})`` 까지
**20개**의 표준편차다. 첫 항이 ``ln(C_{t+1}/C_t)`` 가 아닌 이유는 두 가지다.

1. D7 `signals.forward_returns()`가 C_{t+1}을 진입가로 삼은 것과 정합한다.
   두 모듈이 같은 진입 시점을 전제해야 나중에 값을 나란히 놓을 수 있다.
2. 프로젝트 절대 규칙(신호는 t일 종가로 확정, 진입은 t+1일)에 부합한다.
   재려는 것은 **실제 진입 후 겪는 경로변동성**이므로, 진입 전에 지나가버린
   갭 ``ln(C_{t+1}/C_t)`` 는 투자자가 겪지 않는다. 이를 포함하면 신호 확정일
   종가 C_t가 사후 창에 들어와 층화 변수 창(t-13 ~ t)과 종점을 공유하게 된다.

이는 사전등록 §2 문구("t+1 진입 이후 20거래일")의 모호성을 구현 단계에서
확정한 것이며, §3의 예측·판정 기준과 무관하다. 사전등록은 수정하지 않았다.

사용 예 (프로젝트 루트 기준)
----------------------------
    from src import config, data, pathvol

    df = data.load_parquet(config.RAW_OHLCV_PATH)
    frame = pathvol.build_frame(df, signal_id="S1_rsi_oversold")
    baseline = pathvol.baseline_ratio(frame)
    table = pathvol.stratified_table(frame)
"""

import numpy as np
import pandas as pd

from src import config
from src import signals


# 표준편차는 전부 표본표준편차로 통일한다 (diagnostics.STD_DDOF와 같은 값).
# pandas 기본값은 1이지만 numpy는 0이라, 명시하지 않으면 어느 쪽인지 알 수 없다.
STD_DDOF = 1  # -> int

# 사후 경로변동성 창 (사전등록 §2)
POST_VOL_WINDOW = 20  # -> int, 일별 로그수익률 개수
POST_VOL_OFFSET = 2   # -> int, 첫 수익률이 ln(C_{t+2}/C_{t+1}) 이므로 t+2에서 시작

# 층화 변수 창 (사전등록 §2)
PRE_VOL_WINDOW = 14  # -> int, t-13 ~ t 로그수익률 14개

# 분위 경계 산출 (사전등록 §2, 명세 C-3)
QUANTILE_HISTORY_START = "1990-01-01"  # -> str, 경계 산출에 쓰는 과거 구간의 시작
QUANTILE_MIN_HISTORY = 750             # -> int, 최소 과거 3년(약 750거래일)

# 판정 임계값 (사전등록 §3-3, R0 확인 **전** 확정)
RESIDUAL_NONE_FRACTION = 0.22      # -> float, 이하면 "잔여 정보 없음"
RESIDUAL_EXISTS_FRACTION = 0.44    # -> float, 초과면 "잔여 정보 존재"
DISCRIMINABILITY_FLOOR = 1.15      # -> float, R0가 이 미만이면 판정 미수행
MIN_SIGNAL_PER_BIN = 5             # -> int, 미만이면 가중평균에서 제외
STRATIFICATION_GAP_LIMIT = 20.0    # -> float, prevol_gap_pct가 이상이면 "층화 불완전"


# ---------------------------------------------------------------------------
# 1. 변동성 두 종류
# ---------------------------------------------------------------------------
def add_log_return(df, price_column="close"):
    """티커별 일별 로그수익률을 붙인다.

    ``returns.add_log_return()``과 같은 정의이지만, 본 모듈이 컬럼 이름과
    정렬 순서를 스스로 통제하기 위해 여기서 다시 만든다. 두 모듈이 같은
    컬럼을 서로 다른 정렬 상태로 넘겨받으면 shift가 조용히 어긋난다.
    """
    work = df.copy()                             # -> DataFrame (18424, 8)
    work = work.sort_values(["ticker", "date"])  # -> DataFrame, shift는 행 순서를 믿는다
    work = work.reset_index(drop=True)           # -> DataFrame (18424, 8)

    log_price = np.log(work[price_column])  # -> Series[float] (18424,), 원소별 연산이라 티커 무관

    grouped = log_price.groupby(work["ticker"])  # -> SeriesGroupBy

    # LOOKAHEAD GUARD
    # 티커별 shift(1)이다. groupby 없이 걸면 티커 경계에서 ^SP500TR 첫날
    # 수익률이 ln(386.16) - ln(7489.72) = -2.965 (단순수익률 -94.8%)로
    # 계산된다. 존재한 적 없는 하락이 조용히 섞인다.
    previous_log_price = grouped.shift(1)  # -> Series[float] (18424,), 티커별 첫 행 NaN

    work["log_return"] = log_price - previous_log_price  # -> DataFrame (18424, 9)

    return work


def add_pre_volatility(df, window=None):
    """층화 변수: 직전 ``window``일 실현변동성 (사전등록 §2).

    ``prevol(t)`` = ``ln(C_{t-13}/C_{t-14})`` ~ ``ln(C_t/C_{t-1})`` 14개의
    표준편차, ddof=1, 연율화 없음.

    이 값은 **t일 종가 시점에 알 수 있다.** 마지막 항 ``ln(C_t/C_{t-1})``이
    C_t와 C_{t-1}만 쓰므로 t 이후 정보가 들어가지 않는다. 층화 변수가 미래를
    보면 분위 자체가 무의미해지므로 이 점이 본 분석의 전제다.
    """
    if window is None:
        window = PRE_VOL_WINDOW  # -> int (14)

    work = df.copy()  # -> DataFrame (18424, 9)

    grouped = work.groupby("ticker")           # -> DataFrameGroupBy
    return_by_ticker = grouped["log_return"]   # -> SeriesGroupBy

    # LOOKAHEAD GUARD
    # rolling은 창의 **끝**에 값을 놓는다. 즉 index t의 값은 t-13 ~ t를 쓴다.
    # shift가 없으므로 t 이후 정보는 들어가지 않는다. groupby를 거치므로
    # 티커 경계를 넘는 창도 만들어지지 않는다.
    rolling_window = return_by_ticker.rolling(window)          # -> RollingGroupby
    pre_volatility = rolling_window.std(ddof=STD_DDOF)         # -> Series[float] MultiIndex
    pre_volatility = pre_volatility.reset_index(level=0, drop=True)  # -> Series[float] (18424,)

    work["pre_vol"] = pre_volatility  # -> DataFrame (18424, 10)

    return work


def add_post_volatility(df, window=None, offset=None):
    """사후 경로변동성 (사전등록 §2 + 모듈 docstring의 창 시작점 확정).

    ``post_vol(t)`` = ``ln(C_{t+2}/C_{t+1})`` ~ ``ln(C_{t+21}/C_{t+20})``
    20개의 표준편차, ddof=1, 연율화 없음.

    EX-POST ONLY: 사후 평가 전용. `signals.py`로 역류 금지.
    이 컬럼은 정의상 미래를 본다. 매매 규칙에 들어가는 순간 미래 참조가 된다.

    구현 메모 — shift(-(window + offset - 1))의 근거
    ------------------------------------------------
    ``log_return`` 은 ``lr_i = ln(C_i / C_{i-1})`` 이므로
    ``ln(C_{t+2}/C_{t+1}) = lr_{t+2}`` 이다. 따라서 필요한 값은
    ``lr_{t+2} .. lr_{t+21}`` 즉 **인덱스 t+2에서 시작하는 20개**다.

    rolling(20)은 창의 끝에 값을 놓으므로, 그 20개의 표준편차는 인덱스
    **t+21** 행에 놓인다. 이를 t행으로 끌어오려면 21칸 당겨야 한다.
    ``offset + window - 1 = 2 + 20 - 1 = 21``.
    """
    if window is None:
        window = POST_VOL_WINDOW  # -> int (20)

    if offset is None:
        offset = POST_VOL_OFFSET  # -> int (2)

    work = df.copy()  # -> DataFrame (18424, 10)

    grouped = work.groupby("ticker")          # -> DataFrameGroupBy
    return_by_ticker = grouped["log_return"]  # -> SeriesGroupBy

    rolling_window = return_by_ticker.rolling(window)      # -> RollingGroupby
    trailing_std = rolling_window.std(ddof=STD_DDOF)       # -> Series[float] MultiIndex
    trailing_std = trailing_std.reset_index(level=0, drop=True)  # -> Series[float] (18424,)

    work["_trailing_std"] = trailing_std  # -> DataFrame (18424, 11)

    shift_amount = offset + window - 1  # -> int (21)

    # EX-POST ONLY
    # 음수 shift는 미래를 당겨온다. `forward_returns()`의 shift(-1)과 같은
    # 성격이며, 사후 평가 컬럼에만 허용된다. 티커별로 걸어야 마지막 21행이
    # 다음 티커의 값을 끌어오지 않는다.
    post_grouped = work.groupby("ticker")["_trailing_std"]  # -> SeriesGroupBy
    post_volatility = post_grouped.shift(-shift_amount)     # -> Series[float] (18424,)

    work["post_vol"] = post_volatility          # -> DataFrame (18424, 12)
    work = work.drop(columns=["_trailing_std"])  # -> DataFrame (18424, 11)

    return work


# ---------------------------------------------------------------------------
# 2. 확장 윈도우 분위 경계
# ---------------------------------------------------------------------------
def expanding_quantile_bin(values, n_bins, min_history=None):
    """각 시점의 분위 경계를 **그 시점 이전 자료만으로** 산출한다 (명세 C-3).

    ``values[i]``의 분위는 ``values[:i]`` 의 분포에서 나온 경계로 정한다.
    전체 표본 분위를 쓰면 2026년 자료가 2001년 관측의 분위를 정하게 되어
    미래 참조가 된다.

    갱신 주기가 매 거래일인 이유
    ----------------------------
    연 단위로 경계를 고정하면 2008년처럼 연중 변동성 레짐이 급변하는 해에
    연초 경계로 연말을 분류하게 되어 분위가 의미를 잃는다. expanding
    quantile은 한 번 훑으면 되므로 계산량이 문제되지 않는다.

    Returns
    -------
    ndarray[float]
        1~n_bins의 분위 번호. 과거가 부족하거나 값이 NaN이면 NaN.
    """
    if min_history is None:
        min_history = QUANTILE_MIN_HISTORY  # -> int (750)

    if n_bins <= 1:
        # 층화 없음(기준선 R0) 경로. 전부 같은 분위에 넣는다.
        single_bin = np.where(np.isnan(values), np.nan, 1.0)  # -> ndarray[float] (n,)
        return single_bin

    probabilities = np.arange(1, n_bins) / n_bins  # -> ndarray[float] (n_bins - 1,)

    n_rows = len(values)          # -> int
    bins = np.full(n_rows, np.nan)  # -> ndarray[float] (n,)

    history = []  # -> list[float], 지금까지 관측된 유효 값

    for index in range(n_rows):
        current = values[index]  # -> numpy.float64

        # LOOKAHEAD GUARD
        # 경계를 먼저 계산하고 그 다음에 현재 값을 history에 넣는다.
        # 순서가 뒤집히면 자기 자신이 자기 분위의 경계에 기여한다.
        if not np.isnan(current) and len(history) >= min_history:
            past = np.asarray(history)                       # -> ndarray[float] (len(history),)
            edges = np.quantile(past, probabilities)          # -> ndarray[float] (n_bins - 1,)
            position = np.searchsorted(edges, current, side="right")  # -> numpy.int64
            bins[index] = position + 1                        # -> float, 1~n_bins

        if not np.isnan(current):
            history.append(current)

    return bins


# ---------------------------------------------------------------------------
# 3. 분석 프레임 조립
# ---------------------------------------------------------------------------
def build_frame(df, signal_id="S1_rsi_oversold", ticker=None, n_bins=5,
                analysis_start=None):
    """신호 여부·두 변동성·분위를 한 티커에 대해 조립한다.

    분위 경계는 **1990년부터의 전체 거래일**로 산출하고, 집계는
    ``analysis_start`` 이후로 자른다. 경계 산출까지 2000년으로 자르면
    분석 초기 몇 년이 최소 과거 요구(750일)를 못 채운다.

    Returns
    -------
    DataFrame
        (date, ticker, is_signal, pre_vol, post_vol, bin) — 분석 구간, 단일 티커.
        pre_vol·post_vol이 모두 유효한 행만 남는다.
    """
    if ticker is None:
        ticker = config.TICKERS["signal"]  # -> str ("^GSPC")

    if analysis_start is None:
        analysis_start = config.ANALYSIS_START  # -> str ("2000-01-01")

    with_returns = add_log_return(df)             # -> DataFrame (18424, 9)
    with_pre = add_pre_volatility(with_returns)   # -> DataFrame (18424, 10)
    with_post = add_post_volatility(with_pre)     # -> DataFrame (18424, 11)

    ticker_mask = with_post["ticker"] == ticker   # -> Series[bool] (18424,)
    frame = with_post.loc[ticker_mask]            # -> DataFrame (9212, 11)
    frame = frame.sort_values("date")             # -> DataFrame
    frame = frame.reset_index(drop=True)          # -> DataFrame (9212, 11)

    # 신호는 D6·D7과 같은 경로로 만든다. 재계산이 아니라 재현이다.
    signal_frame = signals.make_signals(df)                          # -> DataFrame (92120, 5)
    id_mask = signal_frame["signal_id"] == signal_id                 # -> Series[bool] (92120,)
    signal_ticker_mask = signal_frame["ticker"] == ticker            # -> Series[bool] (92120,)
    one_signal = signal_frame.loc[id_mask & signal_ticker_mask]      # -> DataFrame (9212, 5)
    one_signal = one_signal[["date", "signal"]]                      # -> DataFrame (9212, 2)

    merged = frame.merge(one_signal, on="date", how="left")  # -> DataFrame (9212, 12)

    if len(merged) != len(frame):
        raise ValueError("신호 병합에서 행 수가 변했다. date 중복을 의심하라.")

    merged["is_signal"] = merged["signal"].fillna(False)  # -> DataFrame (9212, 13)

    # 분위는 1990년부터의 전체 거래일 분포로 매긴다.
    pre_values = merged["pre_vol"].to_numpy()                       # -> ndarray[float] (9212,)
    merged["bin"] = expanding_quantile_bin(pre_values, n_bins)      # -> DataFrame (9212, 14)

    start = pd.Timestamp(analysis_start)          # -> Timestamp
    date_mask = merged["date"] >= start           # -> Series[bool] (9212,)
    analysis = merged.loc[date_mask]              # -> DataFrame (6684, 14)

    valid_mask = analysis["pre_vol"].notna() & analysis["post_vol"].notna()  # -> Series[bool]
    analysis = analysis.loc[valid_mask]           # -> DataFrame

    columns = ["date", "ticker", "is_signal", "pre_vol", "post_vol", "bin"]  # -> list[str] (6,)
    analysis = analysis[columns]                  # -> DataFrame
    analysis = analysis.reset_index(drop=True)    # -> DataFrame

    return analysis


# ---------------------------------------------------------------------------
# 4. 집계
# ---------------------------------------------------------------------------
def _bin_row(subset, bin_label):
    """한 분위의 신호일·비신호일 요약 한 줄."""
    signal_rows = subset.loc[subset["is_signal"]]          # -> DataFrame
    nonsignal_rows = subset.loc[~subset["is_signal"]]      # -> DataFrame

    n_signal = len(signal_rows)        # -> int
    n_nonsignal = len(nonsignal_rows)  # -> int

    post_vol_signal = float(signal_rows["post_vol"].mean()) if n_signal else np.nan        # -> float
    post_vol_nonsignal = float(nonsignal_rows["post_vol"].mean()) if n_nonsignal else np.nan  # -> float

    prevol_signal = float(signal_rows["pre_vol"].mean()) if n_signal else np.nan           # -> float
    prevol_nonsignal = float(nonsignal_rows["pre_vol"].mean()) if n_nonsignal else np.nan  # -> float

    if n_signal and n_nonsignal:
        ratio = post_vol_signal / post_vol_nonsignal                     # -> float
        prevol_gap_pct = (prevol_signal / prevol_nonsignal - 1) * 100    # -> float
    else:
        ratio = np.nan
        prevol_gap_pct = np.nan

    row = {
        "bin": bin_label,
        "n_signal": n_signal,
        "n_nonsignal": n_nonsignal,
        "post_vol_signal": post_vol_signal,
        "post_vol_nonsignal": post_vol_nonsignal,
        "ratio": ratio,
        "prevol_signal": prevol_signal,
        "prevol_nonsignal": prevol_nonsignal,
        "prevol_gap_pct": prevol_gap_pct,
    }  # -> dict (9,)

    return row


def baseline_ratio(frame):
    """층화하지 않은 기준선 R0 (사전등록 §2-1).

    **재현 검산이 아니라 신규 측정이다.** D7에 대응값이 없으므로 어떤 값이
    나와도 "불일치"가 아니다. D8(폐기)이 이 지점에서 2.14와의 일치를
    요구했다가 폐기됐다.
    """
    row = _bin_row(frame, bin_label="ALL")  # -> dict (9,)
    return row


def stratified_table(frame, n_bins=5):
    """분위별 표 (명세 C-3의 8개 컬럼).

    **분위별 p값을 계산하지 않는다.** 분위당 약 13건이며, 여기서 검정 5개를
    만들면 D7에서 정리한 다중검정 문제의 반복이다. 점추정치와 방향만 본다.
    """
    rows = []  # -> list[dict]

    for bin_number in range(1, n_bins + 1):
        bin_mask = frame["bin"] == bin_number   # -> Series[bool]
        subset = frame.loc[bin_mask]            # -> DataFrame

        row = _bin_row(subset, bin_label=f"Q{bin_number}")  # -> dict (9,)
        row["stratification"] = _stratification_flag(row["prevol_gap_pct"])  # -> str
        row["in_weighted_mean"] = row["n_signal"] >= MIN_SIGNAL_PER_BIN      # -> bool

        rows.append(row)

    table = pd.DataFrame(rows)  # -> DataFrame (n_bins, 11)

    return table


def _stratification_flag(prevol_gap_pct):
    """층화 완전성 진단 (사전등록 §4).

    신호일의 직전 변동성이 같은 분위의 비신호일보다 20% 이상 높으면, 분위
    안에서도 신호일이 위쪽에 몰려 있다는 뜻이므로 층화가 충분하지 않다.
    """
    if np.isnan(prevol_gap_pct):
        return "표본 없음"

    if prevol_gap_pct >= STRATIFICATION_GAP_LIMIT:
        return "층화 불완전"

    return "양호"


def weighted_ratio(table):
    """신호일 수 가중평균 (사전등록 §4).

    ``R_weighted = sum(n_q * R_q) / sum(n_q)``, 단 ``n_q >= 5`` 인 분위만.

    가중평균을 쓰는 이유는 질문이 "각 분위의 변동성이 평균적으로 어떻게
    다른가"가 아니라 "실제로 신호를 따라갔을 때의 사후 경로변동성이
    직전 변동성을 통제한 뒤에도 높은가"이기 때문이다.

    Returns
    -------
    (float, list[str])
        가중평균, 제외된 분위 라벨 목록.
    """
    included_mask = table["in_weighted_mean"] & table["ratio"].notna()  # -> Series[bool]
    included = table.loc[included_mask]                                 # -> DataFrame
    excluded = table.loc[~included_mask]                                # -> DataFrame

    weights = included["n_signal"].to_numpy()  # -> ndarray[int]
    ratios = included["ratio"].to_numpy()      # -> ndarray[float]

    if weights.sum() == 0:
        return np.nan, list(excluded["bin"])

    weighted = float((weights * ratios).sum() / weights.sum())  # -> float

    return weighted, list(excluded["bin"])


def verdict(baseline, weighted):
    """판정 (사전등록 §3-3 + 판별력 하한 조항).

    임계값은 R0 확인 **전에** 비율 형태로 확정했다. R0를 본 뒤 비율을
    바꾸지 않는다.

    Returns
    -------
    dict
        (performed, r0, floor, threshold_none, threshold_exists, label, reason)
    """
    result = {
        "r0": baseline,
        "floor": DISCRIMINABILITY_FLOOR,
        "weighted": weighted,
    }  # -> dict

    if np.isnan(baseline) or baseline < DISCRIMINABILITY_FLOOR:
        result["performed"] = False
        result["threshold_none"] = np.nan
        result["threshold_exists"] = np.nan
        result["label"] = "판정 미수행"
        result["reason"] = (
            f"R0 = {baseline:.4f} < {DISCRIMINABILITY_FLOOR} "
            "(판별력 하한 조항). 판정 구간이 n=65의 추정 정밀도보다 좁아진다."
        )
        return result

    gap = baseline - 1.0  # -> float

    threshold_none = 1.0 + RESIDUAL_NONE_FRACTION * gap      # -> float
    threshold_exists = 1.0 + RESIDUAL_EXISTS_FRACTION * gap  # -> float

    result["performed"] = True
    result["threshold_none"] = threshold_none
    result["threshold_exists"] = threshold_exists

    if weighted <= threshold_none:
        result["label"] = "잔여 정보 없음"
    elif weighted > threshold_exists:
        result["label"] = "잔여 정보 존재"
    else:
        result["label"] = "판정 보류"

    result["reason"] = (
        f"가중평균 {weighted:.4f} vs 기준 "
        f"{threshold_none:.4f} / {threshold_exists:.4f}"
    )

    return result


# ===========================================================================
# 5. D13 — 사후 경로변동성 층화 순열검정
# ===========================================================================
"""
사전등록: `docs/prereg_day13b.md`.
(원안 `docs/prereg_day13.md`는 σ_pre 정의 충돌로 2026-08-12 폐기)

아래 함수들은 §1~§4의 규정을 **구현할 뿐** 스스로 정하지 않는다.
위쪽 D8b 함수는 한 줄도 고치지 않았다.

D8b와 무엇이 다른가 — 창이 하루 다르다
--------------------------------------
    D8b  `post_vol`      : lr_{t+2} .. lr_{t+21}   (offset = 2)
    D13  `vol_post_{W}`  : lr_{t+1} .. lr_{t+W}    (offset = 1)

값이 비슷해 보여도 **다른 양이다.** 그래서 D8b 컬럼을 재사용하지 않고 새로
계산한다 (사전등록 §1.4 마지막 문단). lr_{t+1} = ln(C_{t+1}/C_t)는 t일 종가와
t+1일 종가만 쓰므로 신호일 수익률 lr_t = ln(C_t/C_{t-1})를 포함하지 않는다.
따라서 §1.2가 배제하려는 기계적 팽창은 발생하지 않는다.
"""

# 사전등록 §1.2 — 창
D13_POST_WINDOW = 20      # -> int, 주검정 W
D13_POST_OFFSET = 1       # -> int, 첫 수익률이 lr_{t+1} 이므로 t+1에서 시작
D13_POST_WINDOW_AUX = 10  # -> int, 보조 분석 W (기술 전용, 유의성 판정 없음)

# 사전등록 §2.1 — 층 포함 조건. 신호군은 D8b의 MIN_SIGNAL_PER_BIN(=5)을 승계한다.
D13_MIN_CONTROL_PER_BIN = 10  # -> int, 층 내 대조군이 이 미만이면 층 제외

# 사전등록 §2.5 / §4 — 축약과 진단 임계값. 전부 실행 전에 확정된 값이다.
D13_DECLUSTER_GAP = 20        # -> int, 축약이 요구하는 최소 거래일 간격
D13_ALPHA = 0.05              # -> float, §2.6 (다중검정 보정 없음)
D13_DOMINANCE_LIMIT = 0.50    # -> float, §4-1 max_k w_k
D13_OPPOSITE_WEIGHT_LIMIT = 0.25  # -> float, §4-2 부호 불일치 가중치 합
D13_OVERLAP_LIMIT = 0.20      # -> float, §4-3 신호군 중첩률
D13_PRE_GAP_LIMIT = 0.10      # -> float, §4-4 abs(Δ_pre)
D13_P_RATIO_LIMIT = 3.0       # -> float, §4-5 p_stud/p_raw 배율
D13_MIN_SIGNALS = 40          # -> int, §4-6 원자료 유효 신호 수

# 사전등록 §1.3 / §2.1 — 실행 전에 확정된 표본 수. 검증 실패 시 중단한다.
D13_EXPECTED_SIGNALS = 65     # -> int, 절단 후 유효 신호
D13_EXPECTED_CONTROLS = 6598  # -> int, 절단 후 대조군
D13_EXPECTED_SIGNAL_BY_BIN = {1: 1, 2: 2, 3: 10, 4: 20, 5: 32}       # -> dict[int, int]
D13_EXPECTED_CONTROL_BY_BIN = {3: 1132, 4: 1288, 5: 1586}            # -> dict[int, int]
D13_EXPECTED_DECLUSTERED_SIGNALS = 34   # -> int, §2.5
D13_EXPECTED_DECLUSTERED_CONTROLS = 281  # -> int, §2.5


# ---------------------------------------------------------------------------
# 5.1 σ_post — 창 lr_{t+1} .. lr_{t+W}
# ---------------------------------------------------------------------------
def d13_window_positions(position, window=None, offset=None):
    """이벤트 t의 σ_post 창에 들어가는 `log_return` **행 인덱스** 목록.

    검증 전용 헬퍼다. 손계산 대조에서 "창 첫 원소가 lr_{t+1}인가",
    "lr_t가 섞여 들어오지 않았는가"를 인덱스 수준에서 직접 확인하려고 둔다.
    값을 계산하지 않으므로 이 함수만으로는 미래를 보지 않는다.
    """
    if window is None:
        window = D13_POST_WINDOW  # -> int (20)

    if offset is None:
        offset = D13_POST_OFFSET  # -> int (1)

    start = position + offset       # -> int, t+1
    stop = start + window           # -> int, t+W+1 (반열림)

    return list(range(start, stop))  # -> list[int] (window,)


def post_path_volatility_loop(log_returns, window=None, offset=None):
    """**원리 버전** — for문으로 창을 직접 잘라 표본표준편차를 낸다.

    EX-POST ONLY: 사후 평가 전용. `signals.py`로 역류 금지.

    벡터 버전(`add_post_path_volatility`)과 같은 값을 내야 한다. 두 구현을
    따로 만드는 이유는 D10에서 겪은 것과 같다 — rolling + 음수 shift는
    한 칸만 어긋나도 조용히 다른 창을 재고, 그 사실이 결과 숫자만 봐서는
    드러나지 않는다. 창을 눈으로 볼 수 있는 버전을 옆에 두고 대조한다.

    Parameters
    ----------
    log_returns : ndarray[float]
        **한 티커의** 일별 로그수익률, 날짜 오름차순.

    Returns
    -------
    ndarray[float]
        각 위치 t의 σ_post. 창이 표본 끝을 넘어가거나 창 안에 NaN이 있으면 NaN.
    """
    if window is None:
        window = D13_POST_WINDOW  # -> int (20)

    if offset is None:
        offset = D13_POST_OFFSET  # -> int (1)

    values = np.asarray(log_returns, dtype=float)  # -> ndarray[float] (n,)

    n_rows = len(values)                # -> int
    result = np.full(n_rows, np.nan)    # -> ndarray[float] (n,)

    for position in range(n_rows):
        positions = d13_window_positions(position, window, offset)  # -> list[int] (window,)

        last = positions[-1]  # -> int, t+W

        if last >= n_rows:
            # 표본 말단 절단 (사전등록 §1.3). 보간하지 않는다.
            continue

        chunk = values[positions[0]:last + 1]  # -> ndarray[float] (window,)

        if np.isnan(chunk).any():
            continue

        result[position] = np.std(chunk, ddof=STD_DDOF)  # -> numpy.float64

    return result


def add_post_path_volatility(df, window=None, offset=None):
    """**벡터 버전** — rolling + 음수 shift. 컬럼 이름은 `vol_post_{window}`.

    EX-POST ONLY: 사후 평가 전용. `signals.py`로 역류 금지.

    shift 양의 근거
    ---------------
    `log_return`은 ``lr_i = ln(C_i / C_{i-1})``이므로 필요한 값은
    ``lr_{t+1} .. lr_{t+W}`` 즉 **인덱스 t+1에서 시작하는 W개**다.
    rolling(W)는 창의 끝에 값을 놓으므로 그 표준편차는 인덱스 **t+W** 행에
    놓인다. t행으로 끌어오려면 W칸 당긴다.
    ``offset + window - 1 = 1 + 20 - 1 = 20``.

    D8b `add_post_volatility()`의 21과 정확히 1 차이이며, 그 1이 lr_{t+1}의
    포함 여부다.
    """
    if window is None:
        window = D13_POST_WINDOW  # -> int (20)

    if offset is None:
        offset = D13_POST_OFFSET  # -> int (1)

    work = df.copy()  # -> DataFrame (18424, 9)

    grouped = work.groupby("ticker")          # -> DataFrameGroupBy
    return_by_ticker = grouped["log_return"]  # -> SeriesGroupBy

    rolling_window = return_by_ticker.rolling(window)              # -> RollingGroupby
    trailing_std = rolling_window.std(ddof=STD_DDOF)               # -> Series[float] MultiIndex
    trailing_std = trailing_std.reset_index(level=0, drop=True)    # -> Series[float] (18424,)

    work["_d13_trailing_std"] = trailing_std  # -> DataFrame (18424, 10)

    shift_amount = offset + window - 1  # -> int (20)

    # EX-POST ONLY
    # 음수 shift는 미래를 당겨온다. 사후 평가 컬럼에만 허용된다.
    # 티커별로 걸어야 마지막 20행이 다음 티커의 값을 끌어오지 않는다.
    post_grouped = work.groupby("ticker")["_d13_trailing_std"]  # -> SeriesGroupBy
    path_volatility = post_grouped.shift(-shift_amount)         # -> Series[float] (18424,)

    column_name = f"vol_post_{window}"  # -> str, 사전등록 §1.1의 코드 컬럼 이름

    work[column_name] = path_volatility                     # -> DataFrame (18424, 11)
    work = work.drop(columns=["_d13_trailing_std"])         # -> DataFrame (18424, 10)

    return work


def compare_volatility_versions(loop_values, vector_values, tolerance=1e-12):
    """원리 버전과 벡터 버전의 등가성 검사 (사전등록 없음 — 구현 검증 절차).

    Returns
    -------
    dict
        (nan_positions_match, n_both_valid, max_abs_diff, equivalent)
    """
    loop_array = np.asarray(loop_values, dtype=float)      # -> ndarray[float] (n,)
    vector_array = np.asarray(vector_values, dtype=float)  # -> ndarray[float] (n,)

    loop_missing = np.isnan(loop_array)      # -> ndarray[bool] (n,)
    vector_missing = np.isnan(vector_array)  # -> ndarray[bool] (n,)

    nan_match = bool(np.array_equal(loop_missing, vector_missing))  # -> bool

    both_valid = ~loop_missing & ~vector_missing  # -> ndarray[bool] (n,)

    if both_valid.sum() == 0:
        return {
            "nan_positions_match": nan_match,
            "n_both_valid": 0,
            "max_abs_diff": np.nan,
            "equivalent": False,
        }

    differences = np.abs(loop_array[both_valid] - vector_array[both_valid])  # -> ndarray[float]

    max_difference = float(differences.max())  # -> float

    return {
        "nan_positions_match": nan_match,
        "n_both_valid": int(both_valid.sum()),
        "max_abs_diff": max_difference,
        "equivalent": bool(nan_match and max_difference <= tolerance),
    }


# ---------------------------------------------------------------------------
# 5.2 분석 프레임 — 층 배정 × σ_pre × σ_post
# ---------------------------------------------------------------------------
def d13_build_frame(df, bins, window=None, ticker=None, truncation="d8b"):
    """§1.3·§1.7의 분석 프레임을 조립한다.

    `bins`는 `stratified.build_bin_frame()`의 결과를 그대로 받는다. 층 배정
    규칙을 여기서 새로 정하지 않는다 (사전등록 §2.1: D8b·D12 정의 승계).

    `truncation`
    ------------
    ``"d8b"``  §1.3이 명시한 경로. D8b `valid_mask`(`post_vol.notna()`)를 그대로
              쓴다. W=20 주검정은 이 경로이며 6,663행 / 신호 65 / 대조 6,598로
              사전등록에 확정돼 있다.
    ``"own"``  §1.3 첫 문장("t+W가 분석구간을 초과하는 이벤트는 제외")을 본
              분석의 창에 그대로 적용한 경로. 보조 분석 W=10에서 쓴다
              (§1.2 "창 절단 조건만 W=10으로 재적용").

    Returns
    -------
    DataFrame
        (pos, date, quintile, is_signal, pre_vol_14, vol_post_{W},
         log_pre_vol, log_vol_post) — 분석구간, 절단 통과, 층 배정된 행만.
        `pos`는 ^GSPC 전체 거래일(9,212행) 안에서의 행 번호다. 축약과 간격
        계산이 거래일 인덱스 차이를 쓰므로 분석 프레임으로 자르기 전 번호를
        유지한다.
    """
    if window is None:
        window = D13_POST_WINDOW  # -> int (20)

    if ticker is None:
        ticker = config.TICKERS["signal"]  # -> str ("^GSPC")

    with_returns = add_log_return(df)                                    # -> DataFrame (18424, 9)
    with_pre = add_pre_volatility(with_returns)                          # -> DataFrame (18424, 10)
    with_d8b = add_post_volatility(with_pre)                             # -> DataFrame (18424, 11)
    with_d13 = add_post_path_volatility(with_d8b, window=window)         # -> DataFrame (18424, 12)

    ticker_mask = with_d13["ticker"] == ticker  # -> Series[bool] (18424,)
    frame = with_d13.loc[ticker_mask]           # -> DataFrame (9212, 12)
    frame = frame.sort_values("date")           # -> DataFrame
    frame = frame.reset_index(drop=True)        # -> DataFrame (9212, 12)
    frame["pos"] = frame.index                  # -> DataFrame (9212, 13), 거래일 인덱스

    post_column = f"vol_post_{window}"  # -> str

    keep_columns = ["pos", "date", "pre_vol", "post_vol", post_column]  # -> list[str] (5,)
    slim = frame[keep_columns]                                          # -> DataFrame (9212, 5)

    before_rows = len(bins)  # -> int

    merged = bins.merge(slim, on="date", how="left")  # -> DataFrame (9212, 9)

    if len(merged) != before_rows:
        raise ValueError(
            f"층 배정 병합에서 행 수가 변했다: {before_rows} → {len(merged)}. "
            "date 중복을 의심하라."
        )

    # pre_vol이 두 경로에서 왔으므로 같은 값인지 확인한다. 다르면 층 배정과
    # Δ_pre가 서로 다른 σ_pre를 보게 된다.
    pre_gap = (merged["pre_vol_x"] - merged["pre_vol_y"]).abs()  # -> Series[float] (9212,)
    pre_gap_max = float(pre_gap.max(skipna=True))                # -> float

    if pre_gap_max > 1e-15:
        raise ValueError(f"σ_pre 두 경로 불일치: 최대 차이 {pre_gap_max}")

    merged = merged.rename(columns={"pre_vol_x": "pre_vol_14"})  # -> DataFrame
    merged = merged.drop(columns=["pre_vol_y"])                  # -> DataFrame (9212, 8)

    keep = merged["in_analysis_period"]           # -> Series[bool] (9212,)
    keep = keep & merged["quintile"].notna()      # -> Series[bool]
    keep = keep & merged["pre_vol_14"].notna()    # -> Series[bool]

    if truncation == "d8b":
        keep = keep & merged["post_vol"].notna()      # -> Series[bool], §1.3 명시 경로
    elif truncation == "own":
        keep = keep & merged[post_column].notna()     # -> Series[bool], t+W 절단
    else:
        raise ValueError(f"truncation은 'd8b' 또는 'own'이다: {truncation}")

    usable = merged.loc[keep]                # -> DataFrame
    usable = usable.reset_index(drop=True)   # -> DataFrame

    if usable[post_column].isna().any():
        raise ValueError(f"{post_column}에 NaN이 남았다. 절단 조건을 확인하라.")

    if (usable[post_column] <= 0).any():
        raise ValueError(f"{post_column}에 0 이하가 있다. log 변환이 정의되지 않는다.")

    # 사전등록 §1.5 — 검정은 log σ_post 위에서 수행한다.
    usable["log_vol_post"] = np.log(usable[post_column])     # -> DataFrame
    usable["log_pre_vol"] = np.log(usable["pre_vol_14"])     # -> DataFrame

    usable["quintile"] = usable["quintile"].astype(int)      # -> DataFrame

    columns = [
        "pos", "date", "quintile", "is_signal",
        "pre_vol_14", post_column, "log_pre_vol", "log_vol_post",
    ]  # -> list[str] (8,)

    return usable[columns]


def d13_check_frame(frame, expected_signals=None, expected_controls=None,
                    expected_signal_by_bin=None, expected_control_by_bin=None):
    """§1.3·§2.1의 확정 표본 수와 대조한다. **숫자를 보기 전에 통과해야 한다.**

    Returns
    -------
    dict
        항목별 (실측, 기대, 통과 여부).
    """
    if expected_signals is None:
        expected_signals = D13_EXPECTED_SIGNALS  # -> int (65)

    if expected_controls is None:
        expected_controls = D13_EXPECTED_CONTROLS  # -> int (6598)

    if expected_signal_by_bin is None:
        expected_signal_by_bin = D13_EXPECTED_SIGNAL_BY_BIN  # -> dict[int, int]

    if expected_control_by_bin is None:
        expected_control_by_bin = D13_EXPECTED_CONTROL_BY_BIN  # -> dict[int, int]

    signal_rows = frame.loc[frame["is_signal"]]       # -> DataFrame
    control_rows = frame.loc[~frame["is_signal"]]     # -> DataFrame

    n_signal = len(signal_rows)    # -> int
    n_control = len(control_rows)  # -> int

    signal_by_bin = {}   # -> dict[int, int]
    control_by_bin = {}  # -> dict[int, int]

    for bin_number in sorted(expected_signal_by_bin):
        signal_match = signal_rows["quintile"] == bin_number    # -> Series[bool]
        control_match = control_rows["quintile"] == bin_number  # -> Series[bool]

        signal_by_bin[bin_number] = int(signal_match.sum())
        control_by_bin[bin_number] = int(control_match.sum())

    control_subset = {}  # -> dict[int, int]

    for bin_number in sorted(expected_control_by_bin):
        control_subset[bin_number] = control_by_bin[bin_number]

    checks = [
        ("총 행 수", len(frame), n_signal + n_control),
        ("유효 신호 수", n_signal, expected_signals),
        ("대조군 수", n_control, expected_controls),
        ("층별 신호 수", signal_by_bin, dict(expected_signal_by_bin)),
        ("층별 대조군 수", control_subset, dict(expected_control_by_bin)),
    ]  # -> list[tuple]

    rows = []      # -> list[dict]
    all_pass = True  # -> bool

    for label, observed, expected in checks:
        passed = observed == expected  # -> bool

        if not passed:
            all_pass = False

        rows.append({"항목": label, "실측": observed, "기대": expected, "통과": passed})

    return {"all_pass": all_pass, "rows": rows}


# ---------------------------------------------------------------------------
# 5.3 층별 통계와 가중치 (§2.2)
# ---------------------------------------------------------------------------
def d13_stratum_table(frame, min_signals=None, min_controls=None):
    """층별 표본 수·log 평균·분산과 포함 여부.

    포함 조건은 §2.1의 두 가지를 **모두** 적용한다.
        n_k,sig >= MIN_SIGNAL_PER_BIN (=5)
        n_k,ctl >= D13_MIN_CONTROL_PER_BIN (=10)

    w_k는 **포함된 층의 신호 수 합**을 분모로 한다 (§2.2, §2.5의 0.161 /
    0.323 / 0.516이 62를 분모로 쓴 것과 같다).
    """
    if min_signals is None:
        min_signals = MIN_SIGNAL_PER_BIN  # -> int (5)

    if min_controls is None:
        min_controls = D13_MIN_CONTROL_PER_BIN  # -> int (10)

    rows = []  # -> list[dict]

    for bin_number in sorted(frame["quintile"].unique()):
        subset = frame.loc[frame["quintile"] == bin_number]  # -> DataFrame

        signal_rows = subset.loc[subset["is_signal"]]     # -> DataFrame
        control_rows = subset.loc[~subset["is_signal"]]   # -> DataFrame

        n_signal = len(signal_rows)    # -> int
        n_control = len(control_rows)  # -> int

        included = (n_signal >= min_signals) and (n_control >= min_controls)  # -> bool

        row = {
            "quintile": int(bin_number),
            "n_sig": n_signal,
            "n_ctl": n_control,
            "included": included,
        }  # -> dict

        if n_signal > 0 and n_control > 0:
            signal_values = signal_rows["log_vol_post"].to_numpy()    # -> ndarray[float]
            control_values = control_rows["log_vol_post"].to_numpy()  # -> ndarray[float]

            mean_signal = float(np.mean(signal_values))    # -> float
            mean_control = float(np.mean(control_values))  # -> float

            # ddof=1 명시. numpy 기본값 0은 쓰지 않는다 (§1.4, §2.3).
            # n=1이면 ddof=1 분산이 정의되지 않는다. 제외 층(Q1, 신호 1건)에서만
            # 생기며, 그 층은 어차피 Δ·SE에 들어가지 않으므로 NaN으로 둔다.
            if n_signal >= 2:
                var_signal = float(np.var(signal_values, ddof=STD_DDOF))  # -> float
            else:
                var_signal = np.nan

            if n_control >= 2:
                var_control = float(np.var(control_values, ddof=STD_DDOF))  # -> float
            else:
                var_control = np.nan

            se_squared = var_signal / n_signal + var_control / n_control  # -> float

            pre_signal = float(np.mean(signal_rows["log_pre_vol"].to_numpy()))    # -> float
            pre_control = float(np.mean(control_rows["log_pre_vol"].to_numpy()))  # -> float

            row["mean_log_post_sig"] = mean_signal
            row["mean_log_post_ctl"] = mean_control
            row["delta_k"] = mean_signal - mean_control
            row["exp_delta_k"] = float(np.exp(mean_signal - mean_control))
            row["var_log_post_sig"] = var_signal
            row["var_log_post_ctl"] = var_control
            row["se_k"] = float(np.sqrt(se_squared))
            row["mean_log_pre_sig"] = pre_signal
            row["mean_log_pre_ctl"] = pre_control
            row["delta_pre_k"] = pre_signal - pre_control
        else:
            for name in ["mean_log_post_sig", "mean_log_post_ctl", "delta_k",
                         "exp_delta_k", "var_log_post_sig", "var_log_post_ctl",
                         "se_k", "mean_log_pre_sig", "mean_log_pre_ctl",
                         "delta_pre_k"]:
                row[name] = np.nan

        rows.append(row)

    table = pd.DataFrame(rows)  # -> DataFrame (층 수, 14)

    included_mask = table["included"]                                  # -> Series[bool]
    total_signal = int(table.loc[included_mask, "n_sig"].sum())        # -> int

    if total_signal == 0:
        table["w_k"] = np.nan
    else:
        weights = table["n_sig"] / total_signal      # -> Series[float]
        table["w_k"] = weights.where(included_mask)  # -> Series[float], 제외 층은 NaN

    return table


def d13_delta_and_se(table):
    """§2.2의 Δ와 §2.3의 SE(Δ)를 포함 층에서 산출한다.

    Returns
    -------
    dict
        (delta, se, t_stud, exp_delta, delta_pre, n_sig, included)
    """
    included = table.loc[table["included"]]  # -> DataFrame

    weights = included["w_k"].to_numpy()      # -> ndarray[float] (K,)
    differences = included["delta_k"].to_numpy()  # -> ndarray[float] (K,)
    se_terms = included["se_k"].to_numpy()    # -> ndarray[float] (K,)
    pre_differences = included["delta_pre_k"].to_numpy()  # -> ndarray[float] (K,)

    delta = float(np.sum(weights * differences))  # -> float

    variance = float(np.sum(weights * weights * se_terms * se_terms))  # -> float
    standard_error = float(np.sqrt(variance))                          # -> float

    delta_pre = float(np.sum(weights * pre_differences))  # -> float

    return {
        "delta": delta,
        "se": standard_error,
        "t_stud": delta / standard_error if standard_error > 0 else np.nan,
        "exp_delta": float(np.exp(delta)),
        "delta_pre": delta_pre,
        "n_sig": int(included["n_sig"].sum()),
        "included": [int(q) for q in included["quintile"]],
    }


# ---------------------------------------------------------------------------
# 5.4 층 내 라벨 순열검정 (§2.4)
# ---------------------------------------------------------------------------
def _d13_delta_se_from_groups(groups, weights):
    """층별 (신호값, 대조값)에서 (Δ, SE)를 낸다. 관측과 순열이 같은 함수를 쓴다.

    같은 함수를 쓰는 것이 핵심이다. 관측과 순열이 서로 다른 코드 경로를 타면
    p-value가 통계량의 차이가 아니라 구현의 차이를 재게 된다.
    """
    delta = 0.0     # -> float
    variance = 0.0  # -> float

    for stratum in sorted(groups):
        signal_values, control_values = groups[stratum]  # -> (ndarray, ndarray)

        weight = weights[stratum]  # -> float

        mean_signal = float(np.mean(signal_values))    # -> float
        mean_control = float(np.mean(control_values))  # -> float

        delta = delta + weight * (mean_signal - mean_control)

        # Welch형. pooled variance를 쓰지 않는다 (§2.3).
        var_signal = float(np.var(signal_values, ddof=STD_DDOF))    # -> float
        var_control = float(np.var(control_values, ddof=STD_DDOF))  # -> float

        se_squared = var_signal / len(signal_values) + var_control / len(control_values)  # -> float

        variance = variance + weight * weight * se_squared

    return delta, float(np.sqrt(variance))


def d13_permutation(frame, table, iterations=None, seed=None, value_column="log_vol_post"):
    """§2.4 — 층 내 라벨 순열, B회, 양측.

    층 경계를 넘는 라벨 교환은 하지 않는다. 각 층에서 (신호 + 대조) 값을 모아
    라벨만 섞고 앞의 n_k,sig개를 신호로 삼는다. 따라서 n_k,sig · n_k,ctl · w_k가
    매 반복에서 보존된다.

    p_stud와 p_raw는 **같은 B개의 라벨 배치**에서 나온다 (§2.4 마지막 문단).
    두 검정을 따로 돌리지 않는다.

    Returns
    -------
    dict
    """
    if iterations is None:
        iterations = config.PERMUTATION_ITERATIONS  # -> int (10000)

    if seed is None:
        seed = config.PERMUTATION_SEED  # -> int (42)

    included = table.loc[table["included"]]  # -> DataFrame

    strata = [int(q) for q in included["quintile"]]  # -> list[int] (K,)

    weights = {}          # -> dict[int, float]
    observed_groups = {}  # -> dict[int, tuple]
    pooled = {}           # -> dict[int, ndarray]
    signal_sizes = {}     # -> dict[int, int]

    for stratum in strata:
        weight_value = float(included.loc[included["quintile"] == stratum, "w_k"].iloc[0])  # -> float
        weights[stratum] = weight_value

        subset = frame.loc[frame["quintile"] == stratum]  # -> DataFrame

        signal_values = subset.loc[subset["is_signal"], value_column].to_numpy()    # -> ndarray[float]
        control_values = subset.loc[~subset["is_signal"], value_column].to_numpy()  # -> ndarray[float]

        observed_groups[stratum] = (signal_values, control_values)
        pooled[stratum] = np.concatenate([signal_values, control_values])  # -> ndarray[float]
        signal_sizes[stratum] = len(signal_values)                         # -> int

    observed_delta, observed_se = _d13_delta_se_from_groups(observed_groups, weights)  # -> (float, float)

    observed_t = observed_delta / observed_se if observed_se > 0 else np.nan  # -> float

    statistic_stud = abs(observed_t)     # -> float
    statistic_raw = abs(observed_delta)  # -> float

    generator = np.random.default_rng(seed)  # -> Generator, 시드 명시

    extreme_stud = 0  # -> int
    extreme_raw = 0   # -> int

    sizes_preserved = True  # -> bool

    for _iteration in range(iterations):
        permuted = {}  # -> dict[int, tuple]

        for stratum in strata:
            values = pooled[stratum]         # -> ndarray[float] (n_k,)
            take = signal_sizes[stratum]     # -> int

            order = generator.permutation(len(values))  # -> ndarray[int] (n_k,)
            shuffled = values[order]                    # -> ndarray[float] (n_k,)

            signal_star = shuffled[:take]   # -> ndarray[float] (n_k,sig,)
            control_star = shuffled[take:]  # -> ndarray[float] (n_k,ctl,)

            if len(signal_star) != take:
                sizes_preserved = False

            permuted[stratum] = (signal_star, control_star)

        delta_star, se_star = _d13_delta_se_from_groups(permuted, weights)  # -> (float, float)

        if abs(delta_star) >= statistic_raw:
            extreme_raw = extreme_raw + 1

        if se_star > 0:
            t_star = delta_star / se_star  # -> float

            if abs(t_star) >= statistic_stud:
                extreme_stud = extreme_stud + 1

    p_stud = (1 + extreme_stud) / (iterations + 1)  # -> float
    p_raw = (1 + extreme_raw) / (iterations + 1)    # -> float

    return {
        "delta": observed_delta,
        "exp_delta": float(np.exp(observed_delta)),
        "se": observed_se,
        "t_stud": observed_t,
        "p_stud": p_stud,
        "p_raw": p_raw,
        "extreme_stud": extreme_stud,
        "extreme_raw": extreme_raw,
        "iterations": iterations,
        "seed": seed,
        "sizes_preserved": sizes_preserved,
        "strata": strata,
        "weights": weights,
    }


# ---------------------------------------------------------------------------
# 5.5 창 중첩 (§2.5 진단, §4-3)
# ---------------------------------------------------------------------------
def d13_gap_ratio(frame, min_gap=None):
    """층별·군별 연속 간격 중첩률 `count(g < W) / n_gap`.

    간격은 **거래일 인덱스 차이**다 (달력일 아님). `pos`를 쓰는 이유가 이것이다.

    신호군은 §4-3의 판정 대상이고, 대조군은 구조적으로 1에 가까우므로 임계값을
    적용하지 않는 기술 항목이다 (§2.5, §4.3).
    """
    if min_gap is None:
        min_gap = D13_DECLUSTER_GAP  # -> int (20)

    rows = []  # -> list[dict]

    for group_label, is_signal in [("signal", True), ("control", False)]:
        for bin_number in sorted(frame["quintile"].unique()):
            bin_mask = frame["quintile"] == bin_number         # -> Series[bool]
            group_mask = frame["is_signal"] == is_signal       # -> Series[bool]

            subset = frame.loc[bin_mask & group_mask]          # -> DataFrame

            positions = np.sort(subset["pos"].to_numpy())      # -> ndarray[int]

            if len(positions) < 2:
                rows.append({
                    "group": group_label,
                    "quintile": f"Q{int(bin_number)}",
                    "n_obs": len(positions),
                    "n_gap": 0,
                    "count_lt_W": 0,
                    "overlap_ratio": np.nan,
                })
                continue

            gaps = np.diff(positions)  # -> ndarray[int] (n-1,)

            count_below = int((gaps < min_gap).sum())  # -> int

            rows.append({
                "group": group_label,
                "quintile": f"Q{int(bin_number)}",
                "n_obs": len(positions),
                "n_gap": int(len(gaps)),
                "count_lt_W": count_below,
                "overlap_ratio": count_below / len(gaps),
            })

    return pd.DataFrame(rows)  # -> DataFrame (2 * 층 수, 6)


# ---------------------------------------------------------------------------
# 5.6 클러스터 축약 (§2.5)
# ---------------------------------------------------------------------------
def d13_decluster(frame, greedy_thin, drop_near_anchors, min_gap=None):
    """§2.5의 2단계 축약을 적용한 부분 프레임을 돌려준다.

    축약 알고리즘 자체는 `scripts/d13_counts.py`에 이미 구현돼 있고 그 결과가
    사전등록 §2.5의 확정 수치(신호 34 / 대조 281)를 만들었다. **다시 짜지 않고
    함수를 주입받는다.** 여기서 새로 구현하면 사전등록 수치를 만든 코드와
    다른 코드로 민감도를 돌리게 된다.

    Parameters
    ----------
    greedy_thin : callable
        `d13_counts.greedy_thin`
    drop_near_anchors : callable
        `d13_counts.drop_near_anchors`
    """
    if min_gap is None:
        min_gap = D13_DECLUSTER_GAP  # -> int (20)

    signal_rows = frame.loc[frame["is_signal"]]      # -> DataFrame
    control_rows = frame.loc[~frame["is_signal"]]    # -> DataFrame

    signal_positions = signal_rows["pos"].tolist()    # -> list[int]
    control_positions = control_rows["pos"].tolist()  # -> list[int]

    anchors = greedy_thin(signal_positions, min_gap)  # -> list[int], 1단계

    control_far = drop_near_anchors(control_positions, anchors, min_gap)  # -> list[int], 2단계 전반
    control_thin = greedy_thin(control_far, min_gap)                      # -> list[int], 2단계 후반

    kept = set(anchors) | set(control_thin)  # -> set[int]

    keep_mask = frame["pos"].isin(kept)  # -> Series[bool]

    declustered = frame.loc[keep_mask]              # -> DataFrame
    declustered = declustered.reset_index(drop=True)  # -> DataFrame

    return declustered, {"n_anchor": len(anchors), "n_control": len(control_thin)}


# ---------------------------------------------------------------------------
# 5.7 진단 (§4)
# ---------------------------------------------------------------------------
def d13_diagnostics(table, summary, permutation, gap_table, n_signal_raw):
    """§4-1 ~ §4-6. 임계값은 전부 사전등록에 적힌 값이고 여기서 정하지 않는다.

    Returns
    -------
    dict
    """
    included = table.loc[table["included"]]  # -> DataFrame

    # --- 4-1 층 지배도 ---
    max_weight = float(included["w_k"].max())            # -> float
    flag_dominance = bool(max_weight >= D13_DOMINANCE_LIMIT)

    # --- 4-2 층별 부호 이질성 ---
    delta_sign = np.sign(summary["delta"])               # -> numpy.float64
    stratum_signs = np.sign(included["delta_k"].to_numpy())  # -> ndarray[float] (K,)

    opposite_mask = stratum_signs != delta_sign          # -> ndarray[bool] (K,)
    weights_array = included["w_k"].to_numpy()           # -> ndarray[float] (K,)

    opposite_weight = float(weights_array[opposite_mask].sum())  # -> float
    flag_sign_heterogeneity = bool(opposite_weight >= D13_OPPOSITE_WEIGHT_LIMIT)

    # --- 4-3 신호군 창 중첩 ---
    included_labels = [f"Q{int(q)}" for q in included["quintile"]]  # -> list[str]

    signal_gaps = gap_table.loc[gap_table["group"] == "signal"]     # -> DataFrame
    signal_gaps = signal_gaps.loc[signal_gaps["quintile"].isin(included_labels)]  # -> DataFrame

    overlap_over = signal_gaps["overlap_ratio"] >= D13_OVERLAP_LIMIT  # -> Series[bool]
    flagged_strata = [q for q in signal_gaps.loc[overlap_over, "quintile"]]  # -> list[str]

    flag_overlap = len(flagged_strata) > 0  # -> bool

    # --- 4-4 층 내 잔차 편중 ---
    flag_pre_gap = bool(abs(summary["delta_pre"]) >= D13_PRE_GAP_LIMIT)

    # --- 4-5 스튜던트화 전후 p 변화 ---
    p_stud = permutation["p_stud"]  # -> float
    p_raw = permutation["p_raw"]    # -> float

    significant_stud = p_stud < D13_ALPHA  # -> bool
    significant_raw = p_raw < D13_ALPHA    # -> bool

    crossing = bool(significant_stud != significant_raw)  # -> bool

    log_ratio = float(abs(np.log(p_stud / p_raw)))        # -> float
    ratio_exceeded = bool(log_ratio >= np.log(D13_P_RATIO_LIMIT))

    flag_p_shift = bool(crossing or ratio_exceeded)  # -> bool

    # --- 4-6 유효 신호 수 (원자료) ---
    flag_power = bool(n_signal_raw < D13_MIN_SIGNALS)

    return {
        "d4_1_max_weight": max_weight,
        "d4_1_flag": flag_dominance,
        "d4_2_opposite_weight": opposite_weight,
        "d4_2_flag": flag_sign_heterogeneity,
        "d4_3_flagged_strata": ";".join(flagged_strata),
        "d4_3_flag": flag_overlap,
        "d4_4_delta_pre": summary["delta_pre"],
        "d4_4_flag": flag_pre_gap,
        "d4_5_crossing": crossing,
        "d4_5_abs_log_ratio": log_ratio,
        "d4_5_ratio_exceeded": ratio_exceeded,
        "d4_5_flag": flag_p_shift,
        "d4_6_n_signal_raw": int(n_signal_raw),
        "d4_6_flag": flag_power,
    }
