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
