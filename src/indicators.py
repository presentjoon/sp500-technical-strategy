"""기술적 지표를 정의 그대로 손으로 계산하는 모듈.

왜 라이브러리가 있는데 직접 구현하는가
--------------------------------------
1단계의 목적은 "지표 값을 얻는 것"이 아니라 "지표가 무엇을 계산하는지 아는
것"이다. 라이브러리를 그냥 부르면, 2단계에서 Backtrader와 값이 어긋났을 때
어느 쪽이 무엇을 다르게 하는지 판단할 근거가 없다.

이 파일에 shift가 없는 이유
---------------------------
지표 계산에는 shift를 넣지 않는다. diff / rolling / ewm 은 전부 "현재와 과거"만
참조하므로 미래 참조가 원천적으로 발생할 수 없다. 진입 시점을 하루 늦추는
shift(1)은 **신호 단계(signals.py)의 책임**이다. 지표에 shift를 넣으면 손계산
결과가 라이브러리보다 하루씩 밀려서, 존재하지 않는 차이를 진단하게 된다.

평활(smoothing) 방식이 세 가지라는 것이 이 파일의 핵심
------------------------------------------------------
같은 "14일"이라도 평균을 내는 방식이 다르면 다른 숫자가 나온다.

- 단순이동평균(SMA): 최근 n개를 똑같은 무게로 평균. n+1개 전 값은 무게 0.
- Wilder 평활(RMA): alpha = 1/n. 지수 감쇠하지만 완전히 사라지지 않음.
- 일반 EMA: alpha = 2/(n+1). Wilder보다 최근 값에 약 2배 민감하다.

RSI를 만든 Wilder(1978)가 쓴 것은 두 번째다. 그런데 "RSI"라는 이름으로 첫 번째를
구현한 라이브러리도 흔하다. 그래서 두 버전을 모두 만들어 대조한다.

워밍업 경계에 대한 주의
-----------------------
SMA는 "창을 채워야 값이 나온다"는 경계가 수학적으로 존재한다. 반면 EMA와 Wilder
평활은 재귀식이라 **워밍업 경계가 수학적으로 없다.** 선행 NaN 개수는 이론값이
아니라 min_periods 규약의 결과일 뿐이다 (config.INDICATOR_MIN_PERIODS 참고).

사용 예 (프로젝트 루트 기준)
----------------------------
    from src import config, data, indicators

    df = data.load_parquet(config.RAW_OHLCV_PATH)   # -> DataFrame (18424, 8)
    df = indicators.rsi_simple(df)                   # -> + rsi_simple_14
    df = indicators.rsi_wilder(df)                   # -> + rsi_wilder_14
    df = indicators.macd(df)                         # -> + macd_line/signal/hist
"""

import numpy as np
import pandas as pd

from src import config


# ---------------------------------------------------------------------------
# 1. 평활 프리미티브
# ---------------------------------------------------------------------------
# 지표를 만들기 전에 "평균 내는 방법" 세 가지를 먼저 독립 함수로 분리한다.
# 내일 ATR이 wilder_rma를, 볼린저밴드가 sma를 그대로 재사용한다. 지표마다
# 평활을 다시 구현하면 어느 지표가 어떤 평활을 쓰는지 추적할 수 없게 되고,
# 한 곳을 고쳤을 때 다른 곳이 따라오지 않는다.
#
# 세 함수 모두 Series 하나를 받아 Series 하나를 돌려준다. 티커 경계는 이
# 함수들이 아니라 _apply_per_ticker()가 책임진다.
# ---------------------------------------------------------------------------
def sma(series, n, min_periods=None):
    """단순이동평균 (Simple Moving Average).

    최근 n개를 똑같은 무게로 평균낸다. n+1개 전의 값은 무게가 갑자기 0이 되며,
    이 "뚝 떨어지는" 성질 때문에 큰 값이 창 밖으로 나가는 날 지표가 가격과
    무관하게 점프한다.

    rolling(n)은 "이 행을 포함한 직전 n개"를 본다. 미래를 보지 않는다.

    Returns
    -------
    Series[float]
        앞의 (n-1)개는 NaN. 이건 수학적 경계다 — 창을 채울 값이 실제로 없다.
    """
    if min_periods is None:
        min_periods = n  # -> int, config.INDICATOR_MIN_PERIODS 규약

    rolling_window = series.rolling(window=n, min_periods=min_periods)  # -> Rolling
    averaged = rolling_window.mean()  # -> Series[float] (행 수,)

    return averaged


def ema(series, n, adjust=config.EWM_ADJUST, min_periods=None):
    """지수이동평균 (Exponential Moving Average).

    재귀식:

        ema[t] = alpha * 값[t] + (1 - alpha) * ema[t-1],   alpha = 2/(n+1)

    adjust 인자를 명시하는 이유
    ---------------------------
    pandas ewm()의 adjust 기본값은 True다. True면 초기 구간을 가중치 합으로
    정규화해서 채우고, False면 위 재귀식을 첫 값에서 그대로 시작한다. 지표
    정의가 전제하는 것은 False이고, 기본값에 의존하면 pandas 버전이 바뀔 때
    결과가 조용히 달라진다. 그래서 config.EWM_ADJUST로 고정해 항상 넘긴다.

    Returns
    -------
    Series[float]
        앞의 (n-1)개는 NaN. 단 이건 수학적 경계가 아니라 min_periods 규약의
        결과다 — 재귀 자체는 첫 값부터 돈다.
    """
    if min_periods is None:
        min_periods = n  # -> int

    exponential = series.ewm(span=n, adjust=adjust, min_periods=min_periods)  # -> ExponentialMovingWindow
    averaged = exponential.mean()  # -> Series[float] (행 수,)

    return averaged


def wilder_rma(series, n, seed=config.WILDER_SEED, adjust=config.EWM_ADJUST, min_periods=None):
    """Wilder 평활 (Wilder's Smoothing / RMA, Running Moving Average).

    재귀식:

        rma[t] = (rma[t-1] * (n - 1) + 값[t]) / n
               = (1/n) * 값[t] + (1 - 1/n) * rma[t-1]

    두 줄은 같은 식이다. 아래 형태로 보면 alpha = 1/n인 지수평활임이 드러난다.
    일반 EMA의 alpha = 2/(n+1)과 다르다 — 같은 n에 대해 Wilder는 EMA로 치면
    span = 2n-1에 해당하므로 훨씬 느리게 반응한다.

    seed 인자 — 값이 갈리는 지점
    ----------------------------
    재귀식은 출발점이 필요한데 관례가 둘로 갈린다.

    "sma"   : 첫 n개 값의 단순평균에서 출발 (Wilder 1978 원전).
              첫 유효값이 index=n에 나온다.
    "first" : 첫 값 하나에서 바로 출발 (pandas ewm(adjust=False), ta 라이브러리).
              재귀는 index=0부터 돌고, min_periods 규약으로 앞 (n-1)개를 가린다.

    두 방식은 같은 재귀식을 쓰므로 차이는 (1-1/n)^t 로 감쇠한다. 결국 수렴하지만
    초기 수백 일 동안은 눈에 보이는 차이를 만든다.

    Returns
    -------
    Series[float]
    """
    if min_periods is None:
        min_periods = n  # -> int

    if seed == "first":
        # pandas ewm(adjust=False)가 정확히 이 방식이다.
        exponential = series.ewm(alpha=1 / n, adjust=adjust, min_periods=min_periods)  # -> ExponentialMovingWindow
        averaged = exponential.mean()  # -> Series[float] (행 수,)
        return averaged

    if seed == "sma":
        return _rma_with_sma_seed(series, n)

    raise ValueError(f"seed는 'sma' 또는 'first'여야 한다: {seed!r}")


def _rma_with_sma_seed(series, n):
    """Wilder 원전 방식 — 첫 n개의 단순평균을 출발점으로 삼는 RMA.

    pandas ewm()은 "첫 값에서 출발"만 지원하므로 이 방식은 직접 재귀를 돌린다.
    루프가 항상 바로 앞 칸만 참조하므로 미래를 보지 않는다.

    Returns
    -------
    Series[float]
    """
    values = series.to_numpy()   # -> ndarray[float] (행 수,)
    length = len(values)         # -> int
    result = np.full(length, np.nan)  # -> ndarray[float] (행 수,)

    valid_mask = ~np.isnan(values)                # -> ndarray[bool] (행 수,)
    valid_positions = np.flatnonzero(valid_mask)  # -> ndarray[int] (유효 개수,)

    if len(valid_positions) == 0:
        return pd.Series(result, index=series.index)

    first_valid = int(valid_positions[0])  # -> int
    seed_end = first_valid + n             # -> int, 슬라이스 끝(미포함)

    if seed_end > length:
        # 시드를 만들 만큼 데이터가 없다 — 전부 NaN으로 남긴다 (보간하지 않는다).
        return pd.Series(result, index=series.index)

    seed_slice = values[first_valid:seed_end]  # -> ndarray[float] (n,)
    current = float(np.mean(seed_slice))       # -> float, 첫 n개의 단순평균
    start_index = seed_end - 1                 # -> int, 시드를 놓는 위치

    result[start_index] = current

    alpha = 1 / n  # -> float

    for position in range(start_index + 1, length):
        value = values[position]  # -> numpy.float64

        if np.isnan(value):
            # 보간하지 않는다. 입력이 없는 자리는 평활값도 없는 것이 사실이다.
            continue

        current = alpha * value + (1 - alpha) * current  # -> float, 재귀 한 단계
        result[position] = current

    return pd.Series(result, index=series.index)


# ---------------------------------------------------------------------------
# 2. 티커 경계 방어
# ---------------------------------------------------------------------------
def _apply_per_ticker(df, compute_function):
    """티커별로 잘라서 계산 함수를 적용하고 결과를 다시 붙인다.

    지표 계산은 전부 앞 행을 참조하는 연산(diff, rolling, ewm)이라 티커 경계를
    넘으면 존재한 적 없는 값이 만들어진다. src/returns.py에서 로그수익률에 대해
    막았던 것과 같은 문제다. 지표마다 같은 방어 코드를 반복하면 어느 하나에서
    빠뜨리기 쉬우므로 한 곳에 모은다.

    compute_function은 티커 하나짜리 DataFrame을 받아 DataFrame(추가할 컬럼만)을
    돌려주면 된다. 그 안에서는 티커 경계를 신경 쓸 필요가 없다.

    Returns
    -------
    DataFrame
        원본 + 계산 결과 컬럼.
    """
    work = df.copy()                             # -> DataFrame (행 수, 컬럼 수), 원본 보호
    work = work.sort_values(["ticker", "date"])  # -> DataFrame, 재귀·rolling은 순서를 그대로 믿는다
    work = work.reset_index(drop=True)           # -> DataFrame

    ticker_column = work["ticker"]           # -> Series[str] (행 수,)
    unique_tickers = ticker_column.unique()  # -> ndarray[str] (티커 수,)
    tickers = sorted(unique_tickers)         # -> list[str] (티커 수,)

    pieces = []  # -> list[DataFrame]

    for ticker in tickers:
        mask = ticker_column == ticker  # -> Series[bool] (행 수,)
        subset = work.loc[mask]         # -> DataFrame (티커 행 수, 컬럼 수)
        subset = subset.copy()          # -> DataFrame, SettingWithCopy 방지

        computed = compute_function(subset)  # -> DataFrame (티커 행 수, 추가 컬럼 수)

        for column_name in computed.columns:
            subset[column_name] = computed[column_name]

        pieces.append(subset)

    combined = pd.concat(pieces, ignore_index=True)      # -> DataFrame
    combined = combined.sort_values(["ticker", "date"])  # -> DataFrame
    combined = combined.reset_index(drop=True)           # -> DataFrame

    return combined


# ---------------------------------------------------------------------------
# 3. RSI 공통 부품
# ---------------------------------------------------------------------------
def split_gain_loss(close_series):
    """종가에서 상승분과 하락분을 분리한다.

    하락분은 **양수로** 만든다. RSI 공식이 "평균 상승폭 / 평균 하락폭"이라
    둘 다 크기(magnitude)여야 비율이 의미를 갖기 때문이다.

    diff()는 바로 윗행과 비교하는 연산이다. 이 함수는 항상 티커 하나짜리
    Series만 받으므로 (_apply_per_ticker가 보장한다) 경계 문제가 없다.

    Returns
    -------
    (Series[float], Series[float])
        (상승분, 하락분). 각각 첫 원소는 NaN — 전날이 없어 변화량이 없다.
    """
    difference = close_series.diff()  # -> Series[float] (행 수,), 첫 행 NaN
    values = difference.to_numpy()    # -> ndarray[float] (행 수,)

    # np.where는 NaN을 조건에서 False로 처리하므로 그대로 쓰면 첫 행이 0.0이
    # 되어버린다. "변화량을 모르는 날"과 "변화가 0인 날"은 다르므로 NaN을 살린다.
    is_nan = np.isnan(values)  # -> ndarray[bool] (행 수,)

    gain_values = np.where(values > 0, values, 0.0)   # -> ndarray[float] (행 수,)
    loss_values = np.where(values < 0, -values, 0.0)  # -> ndarray[float] (행 수,), 부호를 뒤집어 양수화

    gain_values[is_nan] = np.nan
    loss_values[is_nan] = np.nan

    gain = pd.Series(gain_values, index=close_series.index)  # -> Series[float]
    loss = pd.Series(loss_values, index=close_series.index)  # -> Series[float]

    return gain, loss


def rsi_from_averages(average_gain, average_loss):
    """평균 상승폭/하락폭에서 RSI를 만든다.

    공식:

        RS  = 평균 상승폭 / 평균 하락폭
        RSI = 100 - 100 / (1 + RS)

    분모가 0인 경우의 규약 (이 프로젝트가 정한 것)
    ---------------------------------------------
    - 평균 하락폭 = 0, 평균 상승폭 > 0  -> RSI = 100 (RS가 무한대이므로 극한값)
    - 평균 하락폭 = 0, 평균 상승폭 = 0  -> RSI = 50  (가격이 전혀 안 움직임.
      방향이 없으므로 중립. ta 라이브러리는 이 경우에도 100을 준다 — 명시적
      규약 차이이고, notebooks/day03_indicators.ipynb 자체검증 4번에서 확인한다.)

    Returns
    -------
    Series[float]
    """
    gain_values = average_gain.to_numpy()  # -> ndarray[float] (행 수,)
    loss_values = average_loss.to_numpy()  # -> ndarray[float] (행 수,)

    length = len(gain_values)         # -> int
    result = np.full(length, np.nan)  # -> ndarray[float] (행 수,)

    gain_is_nan = np.isnan(gain_values)     # -> ndarray[bool] (행 수,)
    loss_is_nan = np.isnan(loss_values)     # -> ndarray[bool] (행 수,)
    either_nan = gain_is_nan | loss_is_nan  # -> ndarray[bool] (행 수,)

    loss_is_zero = loss_values == 0  # -> ndarray[bool] (행 수,)
    gain_is_zero = gain_values == 0  # -> ndarray[bool] (행 수,)

    # 정상 구간: 분모가 0도 NaN도 아닌 곳
    normal = ~either_nan & ~loss_is_zero  # -> ndarray[bool] (행 수,)

    relative_strength = np.full(length, np.nan)  # -> ndarray[float] (행 수,)
    relative_strength[normal] = gain_values[normal] / loss_values[normal]
    result[normal] = 100 - (100 / (1 + relative_strength[normal]))

    # 하락이 하나도 없던 구간 -> 100
    only_up = ~either_nan & loss_is_zero & ~gain_is_zero  # -> ndarray[bool] (행 수,)
    result[only_up] = 100.0

    # 움직임이 전혀 없던 구간 -> 50 (중립)
    flat = ~either_nan & loss_is_zero & gain_is_zero  # -> ndarray[bool] (행 수,)
    result[flat] = 50.0

    return pd.Series(result, index=average_gain.index)


# ---------------------------------------------------------------------------
# 4. RSI
# ---------------------------------------------------------------------------
def rsi_simple(df, period=config.RSI_PERIOD, price_column="close"):
    """단순이동평균(SMA) 기반 RSI.

    워밍업: diff가 1개, sma가 (period-1)개를 먹으므로 첫 유효값은 index=period.
    이건 SMA의 수학적 경계라 이론적으로 확정된다.

    Returns
    -------
    DataFrame
        원본 + rsi_simple_{period} 컬럼.
    """
    column_name = f"rsi_simple_{period}"  # -> str

    def compute(subset):
        close_series = subset[price_column]           # -> Series[float] (티커 행 수,)
        gain, loss = split_gain_loss(close_series)    # -> (Series, Series)

        average_gain = sma(gain, period)  # -> Series[float] (티커 행 수,)
        average_loss = sma(loss, period)  # -> Series[float] (티커 행 수,)

        rsi_values = rsi_from_averages(average_gain, average_loss)  # -> Series[float]

        return pd.DataFrame({column_name: rsi_values})  # -> DataFrame (티커 행 수, 1)

    return _apply_per_ticker(df, compute)


def rsi_wilder(df, period=config.RSI_PERIOD, price_column="close", seed=config.WILDER_SEED):
    """Wilder 평활 기반 RSI (RSI의 원래 정의).

    seed="sma"   : Wilder 1978 원전. 첫 유효값이 index=period.
    seed="first" : pandas/ta 방식. 재귀는 index=0부터, min_periods로 앞을 가림.

    Returns
    -------
    DataFrame
        원본 + rsi_wilder_{period} (seed="first"면 rsi_wilder_{period}_first).
    """
    if seed == "sma":
        column_name = f"rsi_wilder_{period}"  # -> str
    else:
        column_name = f"rsi_wilder_{period}_{seed}"  # -> str

    def compute(subset):
        close_series = subset[price_column]         # -> Series[float] (티커 행 수,)
        gain, loss = split_gain_loss(close_series)  # -> (Series, Series)

        if seed == "first":
            # ta는 diff.where(diff > 0, 0.0)을 쓰는데, NaN > 0 이 False로 평가되어
            # 첫 행이 조용히 0.0이 된다. 그 0.0이 그대로 시드가 되어 결과적으로
            # 유효 구간이 한 칸 앞당겨진다. 원인을 분리하려면 똑같이 재현해야 한다.
            gain = gain.copy()  # -> Series[float]
            loss = loss.copy()  # -> Series[float]
            gain.iloc[0] = 0.0
            loss.iloc[0] = 0.0

        average_gain = wilder_rma(gain, period, seed=seed)  # -> Series[float]
        average_loss = wilder_rma(loss, period, seed=seed)  # -> Series[float]

        rsi_values = rsi_from_averages(average_gain, average_loss)  # -> Series[float]

        return pd.DataFrame({column_name: rsi_values})  # -> DataFrame (티커 행 수, 1)

    return _apply_per_ticker(df, compute)


# ---------------------------------------------------------------------------
# 5. MACD
# ---------------------------------------------------------------------------
def macd(
    df,
    fast=config.MACD_FAST,
    slow=config.MACD_SLOW,
    signal=config.MACD_SIGNAL,
    price_column="close",
):
    """MACD (Moving Average Convergence Divergence).

        MACD선     = EMA(종가, fast) - EMA(종가, slow)
        시그널선   = EMA(MACD선, signal)
        히스토그램 = MACD선 - 시그널선

    EMA의 alpha = 2/(span+1)이라 Wilder 평활(alpha = 1/n)과 다르다. 같은
    "이동평균"이라는 말을 써도 감쇠 속도가 다르다는 점에 주의.

    시그널선의 워밍업은 MACD선이 시작된 뒤부터 세어진다. ema()에 넘기는
    Series 앞쪽이 NaN이면 pandas ewm이 그 NaN을 관측치로 세지 않기 때문에,
    별도 오프셋 계산 없이 자동으로 맞는다.

    Returns
    -------
    DataFrame
        원본 + macd_line, macd_signal, macd_hist 컬럼.
    """
    def compute(subset):
        close_series = subset[price_column]  # -> Series[float] (티커 행 수,)

        ema_fast = ema(close_series, fast)  # -> Series[float] (티커 행 수,), 앞 fast-1개 NaN
        ema_slow = ema(close_series, slow)  # -> Series[float] (티커 행 수,), 앞 slow-1개 NaN

        macd_line = ema_fast - ema_slow  # -> Series[float], 한쪽이라도 NaN이면 NaN

        signal_line = ema(macd_line, signal)  # -> Series[float]
        histogram = macd_line - signal_line   # -> Series[float]

        return pd.DataFrame(
            {
                "macd_line": macd_line,
                "macd_signal": signal_line,
                "macd_hist": histogram,
            }
        )  # -> DataFrame (티커 행 수, 3)

    return _apply_per_ticker(df, compute)


# ---------------------------------------------------------------------------
# 6. 구현 대조 — "값이 다르다"에서 "왜 다른지"로 넘어가기 위한 도구
# ---------------------------------------------------------------------------
def compare_columns(df, column_a, column_b, label, top_n=config.TOP_DIFF_ROWS):
    """두 컬럼의 차이를 티커별로 요약한다.

    비교는 **둘 다 값이 있는 행에서만** 한다. 한쪽이 NaN인 워밍업 구간을 섞으면
    "차이 없음"으로 집계되어, 비교조차 못 한 구간을 일치한 것처럼 보이게 만든다.
    그래서 실제로 몇 행을 비교했는지(n_compared)를 항상 같이 낸다.

    Returns
    -------
    (DataFrame, DataFrame)
        요약표와 차이 상위 행 목록.
    """
    work = df.copy()                             # -> DataFrame
    work = work.sort_values(["ticker", "date"])  # -> DataFrame
    work = work.reset_index(drop=True)           # -> DataFrame

    ticker_column = work["ticker"]           # -> Series[str] (행 수,)
    unique_tickers = ticker_column.unique()  # -> ndarray[str] (티커 수,)
    tickers = sorted(unique_tickers)         # -> list[str] (티커 수,)

    summary_rows = []   # -> list[dict]
    detail_frames = []  # -> list[DataFrame]

    for ticker in tickers:
        mask = ticker_column == ticker  # -> Series[bool] (행 수,)
        subset = work.loc[mask]         # -> DataFrame (티커 행 수, 컬럼 수)

        series_a = subset[column_a]  # -> Series[float] (티커 행 수,)
        series_b = subset[column_b]  # -> Series[float] (티커 행 수,)

        both_present = series_a.notna() & series_b.notna()  # -> Series[bool] (티커 행 수,)
        compared = subset.loc[both_present]                  # -> DataFrame (비교 가능 행 수, 컬럼 수)

        difference = compared[column_a] - compared[column_b]  # -> Series[float]
        absolute_difference = difference.abs()                # -> Series[float]

        n_compared = len(compared)  # -> int

        if n_compared == 0:
            max_absolute = float("nan")   # -> float
            mean_absolute = float("nan")  # -> float
            worst_date = pd.NaT           # -> NaT
        else:
            max_absolute = float(absolute_difference.max())    # -> float
            mean_absolute = float(absolute_difference.mean())  # -> float

            worst_position = absolute_difference.idxmax()      # -> int
            worst_date = compared.loc[worst_position, "date"]  # -> Timestamp

            if max_absolute == 0:
                # 차이가 전혀 없으면 "최대차 발생일"이라는 것이 존재하지 않는다.
                # idxmax가 돌려준 첫 행 날짜를 그대로 쓰면 그날 무슨 일이 있었던
                # 것처럼 읽히므로 비운다.
                worst_date = pd.NaT

        summary_rows.append(
            {
                "ticker": ticker,
                "comparison": label,
                "n_compared": n_compared,
                "max_abs_diff": max_absolute,
                "mean_abs_diff": mean_absolute,
                "worst_date": worst_date,
            }
        )

        if n_compared > 0:
            detail = compared.copy()                  # -> DataFrame
            detail["abs_diff"] = absolute_difference  # -> DataFrame
            detail = detail.sort_values("abs_diff", ascending=False)  # -> DataFrame
            detail = detail.head(top_n)               # -> DataFrame (top_n, ...)

            keep_columns = ["date", "ticker", column_a, column_b, "abs_diff"]  # -> list[str] (5,)
            detail = detail[keep_columns]             # -> DataFrame (top_n, 5)
            detail = detail.rename(columns={column_a: "value_a", column_b: "value_b"})
            detail.insert(0, "comparison", label)     # -> DataFrame (top_n, 6)
            detail_frames.append(detail)

    summary = pd.DataFrame(summary_rows)  # -> DataFrame (티커 수, 6)

    if len(detail_frames) > 0:
        details = pd.concat(detail_frames, ignore_index=True)  # -> DataFrame
    else:
        detail_columns = ["comparison", "date", "ticker", "value_a", "value_b", "abs_diff"]
        details = pd.DataFrame(columns=detail_columns)        # -> DataFrame (0, 6)

    return summary, details


def comparison_table(df, pairs, top_n=config.TOP_DIFF_ROWS):
    """여러 비교쌍을 한 표로 모은다.

    Returns
    -------
    (DataFrame, DataFrame)
    """
    summary_frames = []  # -> list[DataFrame]
    detail_frames = []   # -> list[DataFrame]

    for column_a, column_b, label in pairs:
        summary, details = compare_columns(df, column_a, column_b, label, top_n=top_n)
        summary_frames.append(summary)
        detail_frames.append(details)

    all_summary = pd.concat(summary_frames, ignore_index=True)  # -> DataFrame
    all_details = pd.concat(detail_frames, ignore_index=True)   # -> DataFrame

    return all_summary, all_details


def classify_difference(df, column_a, column_b, ticker, early_end="1995-01-01", late_start="2000-01-01"):
    """차이의 시간 구조를 보고 원인을 분류한다.

    "차이가 작다"는 것은 진단이 아니다. **차이가 시간에 따라 어떻게 변하는가**가
    원인을 가른다. 판정 기준:

    - 1990년대 초반에만 크고 지수적으로 0에 수렴  -> 초기값 씨딩 차이
    - 2000년 이후에도 일정 크기 유지               -> 평활 방식 자체가 다름
    - 1e-12 수준에서 무작위로 흔들림               -> 부동소수점 오차
    - 특정 날짜에만 튀거나 계단식으로 변함         -> 코드 버그 (원본 확인 필요)

    Returns
    -------
    dict
        판정 결과와 근거 수치.
    """
    ticker_mask = df["ticker"] == ticker  # -> Series[bool] (행 수,)
    subset = df.loc[ticker_mask]          # -> DataFrame (티커 행 수, 컬럼 수)
    subset = subset.sort_values("date")   # -> DataFrame

    series_a = subset[column_a]  # -> Series[float]
    series_b = subset[column_b]  # -> Series[float]

    both_present = series_a.notna() & series_b.notna()  # -> Series[bool]
    compared = subset.loc[both_present]                  # -> DataFrame

    absolute_difference = (compared[column_a] - compared[column_b]).abs()  # -> Series[float]
    date_column = compared["date"]                                        # -> Series[datetime64]

    early_mask = date_column < pd.Timestamp(early_end)    # -> Series[bool]
    late_mask = date_column >= pd.Timestamp(late_start)   # -> Series[bool]

    early_max = float(absolute_difference.loc[early_mask].max())  # -> float
    late_max = float(absolute_difference.loc[late_mask].max())    # -> float
    late_mean = float(absolute_difference.loc[late_mask].mean())  # -> float
    overall_max = float(absolute_difference.max())                # -> float

    if overall_max == 0:
        verdict = "완전 동일 — 차이 없음"
    elif late_max < 1e-12:
        if early_max < 1e-12:
            verdict = "부동소수점 오차"
        else:
            verdict = "초기값 씨딩 차이 (지수적으로 수렴)"
    elif late_mean > 1e-6:
        verdict = "평활 방식 자체가 다름 (2000년 이후에도 유지)"
    else:
        verdict = "판정 보류 — 차이 시계열을 직접 확인할 것"

    return {
        "ticker": ticker,
        "comparison": f"{column_a} vs {column_b}",
        "early_max": early_max,
        "late_max": late_max,
        "late_mean": late_mean,
        "overall_max": overall_max,
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# 7. 정밀도 확인 — "0.000000"이 반올림인지 진짜 0인지 가린다
# ---------------------------------------------------------------------------
def precision_check(series_a, series_b):
    """두 시계열이 어느 수준으로 같은지 소수 반올림 없이 확인한다.

    왜 필요한가
    -----------
    표에 "0.000000"으로 찍혔다고 정말 0인 것은 아니다. 소수 6자리 반올림이라
    2.8e-16도 0.000000으로 보인다. 그런데 이 구분이 결론을 바꾼다.

    - 정확히 0        -> **비트 단위 일치**. 두 구현이 같은 부동소수점 연산을
                         같은 순서로 수행했다는 뜻이다.
    - 1e-16 수준      -> **부동소수점 오차 수준의 일치**. 수학적으로는 같은
                         식이지만 연산 순서가 달라 마지막 자리가 흔들린 것.
    - 그 이상          -> 실질적 차이. 가정이 다르거나 코드가 틀렸다.

    (a)/(b)/(c) 판정의 근거로 쓰려면 이 셋을 섞으면 안 된다.

    Returns
    -------
    dict
    """
    both_present = series_a.notna() & series_b.notna()  # -> Series[bool] (행 수,)

    values_a = series_a.loc[both_present].to_numpy()  # -> ndarray[float] (비교 가능 행 수,)
    values_b = series_b.loc[both_present].to_numpy()  # -> ndarray[float] (비교 가능 행 수,)

    difference = values_a - values_b        # -> ndarray[float] (비교 가능 행 수,)
    absolute = np.abs(difference)           # -> ndarray[float] (비교 가능 행 수,)

    n_compared = len(values_a)  # -> int

    if n_compared == 0:
        return {
            "n_compared": 0,
            "max_abs_diff": float("nan"),
            "n_nonzero": 0,
            "array_equal": False,
            "nan_positions_equal": False,
            "verdict": "비교 가능 구간 없음",
        }

    max_absolute = float(absolute.max())          # -> float
    n_nonzero = int((difference != 0).sum())      # -> int
    arrays_equal = bool(np.array_equal(values_a, values_b))  # -> bool

    # NaN이 같은 자리에 있는지도 봐야 한다. 값이 같아도 유효 구간이 다르면
    # 두 구현이 같다고 말할 수 없다.
    nan_a = series_a.isna().to_numpy()  # -> ndarray[bool] (행 수,)
    nan_b = series_b.isna().to_numpy()  # -> ndarray[bool] (행 수,)
    nan_equal = bool(np.array_equal(nan_a, nan_b))  # -> bool

    if max_absolute == 0:
        verdict = "비트 단위 일치 (정확히 0)"       # -> str
    elif max_absolute < 1e-12:
        verdict = "부동소수점 오차 수준의 일치"     # -> str
    else:
        verdict = "실질적 차이 — 원인 규명 필요"    # -> str

    return {
        "n_compared": n_compared,
        "max_abs_diff": max_absolute,
        "n_nonzero": n_nonzero,
        "array_equal": arrays_equal,
        "nan_positions_equal": nan_equal,
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# 8. 신호 불일치 분해 — "몇 % 어긋난다"를 원인별로 쪼갠다
# ---------------------------------------------------------------------------
def threshold_signal(series, threshold, direction):
    """RSI 같은 지표에 임계선을 걸어 이진 신호를 만든다.

    NaN(워밍업 구간)은 False로 두지 않고 NaN으로 남긴다. False로 만들면
    "신호가 없었다"와 "판단할 수 없었다"가 같은 값이 되어, 뒤에서 신호 개수를
    셀 때 워밍업 구간이 조용히 분모에 들어간다.

    Parameters
    ----------
    direction : str
        "below" — 임계선 아래일 때 신호 (과매도)
        "above" — 임계선 위일 때 신호 (과매수)

    Returns
    -------
    Series[float]
        1.0 / 0.0 / NaN
    """
    if direction == "below":
        raw = series < threshold  # -> Series[bool] (행 수,)
    elif direction == "above":
        raw = series > threshold  # -> Series[bool] (행 수,)
    else:
        raise ValueError(f"direction은 'below' 또는 'above'여야 한다: {direction!r}")

    signal = raw.astype(float)          # -> Series[float] (행 수,)
    signal = signal.where(series.notna())  # -> Series[float], 워밍업은 NaN 유지

    return signal


def decompose_disagreement(signal_a, signal_b, label_a, label_b,
                           tolerance=config.SIGNAL_TIMING_TOLERANCE):
    """두 신호 집합의 불일치를 빈도 / 타이밍 / 진짜 불일치로 분해한다.

    왜 분해해야 하는가
    ------------------
    "신호의 72%가 어긋난다"는 문장은 성격이 다른 세 상황을 뭉갠 숫자다.

    (A) 빈도 차이   — 한쪽이 훨씬 자주 발동해서 겹치지 않는 날이 많다
    (B) 타이밍 차이 — 같은 사건을 둘 다 잡았는데 며칠 어긋났다
    (C) 진짜 불일치 — 서로 다른 사건을 가리킨다

    (A)라면 "빈도가 N배 다르다"고 써야 하고, (B)라면 "지연 차이"이며,
    (C)일 때만 비로소 "서로 다른 사건을 가리킨다"고 쓸 수 있다.

    타이밍 판정에 쓰는 rolling(center=True)에 대한 경고
    --------------------------------------------------
    center=True는 창의 중심을 현재 행에 두므로 **미래 행을 참조한다.**
    오늘 신호의 3일 뒤에 상대 신호가 있는지 보려면 미래를 봐야 하기 때문에
    피할 수 없다. 이 함수는 **사후 분석 전용**이며, 여기서 나온 어떤 값도
    매매 규칙에 들어갈 수 없다 (CLAUDE.md 규칙 1).

    Returns
    -------
    dict
    """
    both_present = signal_a.notna() & signal_b.notna()  # -> Series[bool] (행 수,)

    fired_a = (signal_a == 1) & both_present  # -> Series[bool] (행 수,)
    fired_b = (signal_b == 1) & both_present  # -> Series[bool] (행 수,)

    n_days = int(both_present.sum())  # -> int, 양쪽 모두 판단 가능했던 거래일
    n_a = int(fired_a.sum())          # -> int
    n_b = int(fired_b.sum())          # -> int

    both_fire = int((fired_a & fired_b).sum())                    # -> int
    only_a = int((fired_a & ~fired_b).sum())                      # -> int
    only_b = int((~fired_a & fired_b).sum())                      # -> int
    neither = int((~fired_a & ~fired_b & both_present).sum())     # -> int

    # ------------------------------------------------------------------
    # 타이밍 분해
    # ------------------------------------------------------------------
    window = 2 * tolerance + 1  # -> int, 앞뒤 tolerance일 + 당일

    # LOOKAHEAD (사후 분석 전용 — 매매 규칙에 절대 사용 금지)
    # center=True는 창의 중심을 현재 행에 두므로 미래 tolerance일을 참조한다.
    # "이 신호 주변에 상대 신호가 있었나"를 묻는 사후 질문에만 쓴다.
    rolling_b = fired_b.astype(float).rolling(window=window, center=True, min_periods=1)
    nearby_b = rolling_b.max()  # -> Series[float] (행 수,), 주변에 b 신호가 있으면 1.0

    # LOOKAHEAD (사후 분석 전용 — 매매 규칙에 절대 사용 금지)
    rolling_a = fired_a.astype(float).rolling(window=window, center=True, min_periods=1)
    nearby_a = rolling_a.max()  # -> Series[float] (행 수,)

    only_a_mask = fired_a & ~fired_b  # -> Series[bool] (행 수,)
    only_b_mask = ~fired_a & fired_b  # -> Series[bool] (행 수,)

    only_a_timing = int((only_a_mask & (nearby_b == 1)).sum())  # -> int, 같은 사건 다른 날
    only_a_true = int((only_a_mask & (nearby_b == 0)).sum())    # -> int, 진짜 불일치

    only_b_timing = int((only_b_mask & (nearby_a == 1)).sum())  # -> int
    only_b_true = int((only_b_mask & (nearby_a == 0)).sum())    # -> int

    union = both_fire + only_a + only_b  # -> int, 어느 한쪽이라도 발동한 날

    if union > 0:
        raw_disagreement = 100 * (only_a + only_b) / union  # -> float
        true_disagreement = 100 * (only_a_true + only_b_true) / union  # -> float
    else:
        raw_disagreement = float("nan")   # -> float
        true_disagreement = float("nan")  # -> float

    if n_b > 0:
        frequency_ratio = n_a / n_b  # -> float
        # B의 신호 중 A도 함께 발동한 비율. 100%에 가까우면 B의 신호 집합이
        # A의 부분집합이라는 뜻이고, 그렇다면 "서로 다른 사건을 가리킨다"가
        # 아니라 "A가 B의 신호를 포함하면서 더 자주 발동한다"가 맞는 서술이다.
        containment_b_in_a = 100 * both_fire / n_b  # -> float
    else:
        frequency_ratio = float("nan")     # -> float
        containment_b_in_a = float("nan")  # -> float

    if n_a > 0:
        containment_a_in_b = 100 * both_fire / n_a  # -> float
    else:
        containment_a_in_b = float("nan")  # -> float

    return {
        "label_a": label_a,
        "label_b": label_b,
        "containment_b_in_a": containment_b_in_a,
        "containment_a_in_b": containment_a_in_b,
        "n_days": n_days,
        "n_a": n_a,
        "n_b": n_b,
        "rate_a": 100 * n_a / n_days if n_days else float("nan"),
        "rate_b": 100 * n_b / n_days if n_days else float("nan"),
        "frequency_ratio": frequency_ratio,
        "both": both_fire,
        "only_a": only_a,
        "only_b": only_b,
        "neither": neither,
        "only_a_timing": only_a_timing,
        "only_a_true": only_a_true,
        "only_b_timing": only_b_timing,
        "only_b_true": only_b_true,
        "raw_disagreement_pct": raw_disagreement,
        "true_disagreement_pct": true_disagreement,
        "tolerance": tolerance,
    }


def disagreement_to_markdown(results):
    """불일치 분해 결과를 마크다운 표로 만든다 (리포트 붙여넣기용).

    Parameters
    ----------
    results : list[tuple[str, dict]]
        (신호 이름, decompose_disagreement 결과) 목록.

    Returns
    -------
    str
    """
    lines = []  # -> list[str]

    lines.append("**1) 발동 빈도**")
    lines.append("")
    lines.append("| 신호 | 판단 가능일 | A 발동 | A 비율 | B 발동 | B 비율 | 빈도비 A/B |")
    lines.append("|---|---|---|---|---|---|---|")

    for name, result in results:
        lines.append(
            f"| {name} | {result['n_days']:,} | {result['n_a']:,} | {result['rate_a']:.1f}% "
            f"| {result['n_b']:,} | {result['rate_b']:.1f}% | **{result['frequency_ratio']:.2f}배** |"
        )

    lines.append("")
    lines.append("**2) 혼동행렬과 포함 관계**")
    lines.append("")
    lines.append("| 신호 | 둘 다 | A만 | B만 | 둘 다 아님 | 비대칭 A만:B만 | B가 A에 포함된 비율 |")
    lines.append("|---|---|---|---|---|---|---|")

    for name, result in results:
        if result["only_b"] > 0:
            asymmetry = f"{result['only_a'] / result['only_b']:.1f} : 1"  # -> str
        else:
            asymmetry = "-"  # -> str

        lines.append(
            f"| {name} | {result['both']:,} | {result['only_a']:,} "
            f"| {result['only_b']:,} | {result['neither']:,} "
            f"| **{asymmetry}** | **{result['containment_b_in_a']:.1f}%** |"
        )

    lines.append("")
    lines.append("**3) 타이밍 차이 vs 진짜 불일치**")
    lines.append("")
    lines.append("| 신호 | A만 (타이밍) | A만 (진짜) | B만 (타이밍) | B만 (진짜) | 표면 불일치율 | 진짜 불일치율 |")
    lines.append("|---|---|---|---|---|---|---|")

    for name, result in results:
        lines.append(
            f"| {name} | {result['only_a_timing']:,} | {result['only_a_true']:,} "
            f"| {result['only_b_timing']:,} | {result['only_b_true']:,} "
            f"| {result['raw_disagreement_pct']:.1f}% | **{result['true_disagreement_pct']:.1f}%** |"
        )

    if len(results) > 0:
        tolerance = results[0][1]["tolerance"]  # -> int
    else:
        tolerance = config.SIGNAL_TIMING_TOLERANCE  # -> int

    lines.append("")
    lines.append("> **읽는 법**")
    lines.append(f"> - '타이밍'은 상대 신호가 ±{tolerance}거래일 이내에 존재하는 경우다"
                 " (같은 사건을 며칠 어긋나 잡음). '진짜'는 그 범위에 상대 신호가 없는 경우다.")
    lines.append("> - 타이밍 판정에는 `rolling(center=True)`가 쓰인다. 이는 **미래를 참조**하므로"
                 " 사후 분석 전용이며 매매 규칙에 들어갈 수 없다.")
    lines.append("> - 빈도비가 1에서 크게 벗어나면, 불일치의 주된 원인은 '서로 다른 사건을"
                 " 가리킨다'가 아니라 '한쪽이 훨씬 자주 발동한다'이다.")

    table = "\n".join(lines)  # -> str

    return table


def comparison_to_markdown(summary):
    """대조 요약표를 마크다운 표 문자열로 바꾼다 (리포트 붙여넣기용).

    Returns
    -------
    str
    """
    header_cells = [
        "티커", "비교", "비교행수", "최대 절대차", "평균 절대차", "최대차 발생일", "판정",
    ]  # -> list[str] (7,)

    lines = []  # -> list[str]

    header_line = "| " + " | ".join(header_cells) + " |"  # -> str
    lines.append(header_line)

    separator_cells = ["---"] * len(header_cells)           # -> list[str] (7,)
    separator_line = "|" + "|".join(separator_cells) + "|"  # -> str
    lines.append(separator_line)

    for row_index in summary.index:
        ticker = summary.loc[row_index, "ticker"]                # -> str
        label = summary.loc[row_index, "comparison"]             # -> str
        n_compared = summary.loc[row_index, "n_compared"]        # -> int
        max_absolute = summary.loc[row_index, "max_abs_diff"]    # -> float
        mean_absolute = summary.loc[row_index, "mean_abs_diff"]  # -> float
        worst_date = summary.loc[row_index, "worst_date"]        # -> Timestamp | NaT

        if max_absolute == 0:
            verdict = "완전 동일"          # -> str
        elif max_absolute < 1e-10:
            verdict = "부동소수점 오차"    # -> str
        else:
            verdict = "구현 차이 — 규명 필요"  # -> str

        if pd.isna(worst_date):
            worst_text = "-"  # -> str
        else:
            worst_text = str(worst_date.date())  # -> str

        row_cells = [
            str(ticker),
            label,
            f"{n_compared:,}",
            f"{max_absolute:.3e}",
            f"{mean_absolute:.3e}",
            worst_text,
            verdict,
        ]  # -> list[str] (7,)

        row_line = "| " + " | ".join(row_cells) + " |"  # -> str
        lines.append(row_line)

    caption_lines = [
        "",
        "> **읽는 법**",
        "> - 비교는 **양쪽 모두 값이 있는 행에서만** 한다. 워밍업 구간을 섞으면"
        " 비교하지 못한 구간이 '차이 0'으로 집계되어 일치한 것처럼 보인다.",
        "> - RSI는 0~100 눈금이라 절대차 1.0이 눈금의 1%다. MACD는 지수 포인트"
        " 단위라 절대차의 의미가 RSI와 다르다.",
        "> - **'구현 차이'는 작아도 넘어가지 않는다.** 원인이 규명돼야 한다.",
    ]  # -> list[str] (5,)

    all_lines = lines + caption_lines  # -> list[str]
    table = "\n".join(all_lines)       # -> str

    return table
