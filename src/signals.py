"""진입 신호 생성과 사후 수익률 계산 전담 모듈.

기준 문서는 `docs/signal_spec.md` 하나다. **명세에 없는 신호를 여기서 만들지
않는다.** 명세가 코드를 규정하는 것이지 그 반대가 아니며, 코드가 앞서면
"구현하기 편한 것"에 명세가 끌려간다.

세 가지를 구분한다 (명세 §2.1)
------------------------------
- **조건(condition)**: 매 거래일 참/거짓으로 평가되는 상태. "오늘 RSI < 30"
- **사건(event) = 신호(signal)**: 조건이 거짓 -> 참으로 **바뀐 날**
- **포지션(position)**: 진입 후 청산까지 유지되는 상태 (이 모듈의 범위 밖, D10)

조건을 사건으로 세면 같은 사건을 여러 번 세게 되고, 그 관측들의 사후 수익률
구간이 서로 겹쳐(중첩 관측) 유효 표본 수가 부풀려진다. 그 결과 p-value가
실제보다 작게 나온다. **그래서 모든 신호는 to_edge()를 거친다.**

사용 예 (프로젝트 루트 기준)
----------------------------
    from src import config, data, signals

    df = data.load_parquet(config.RAW_OHLCV_PATH)  # -> DataFrame (18424, 8)
    signal_frame = signals.make_signals(df)         # -> DataFrame (long format)
    returns_frame = signals.forward_returns(df)     # -> DataFrame (+ fwd_ret_h)
"""

import numpy as np
import pandas as pd

from src import config
from src import indicators


# ---------------------------------------------------------------------------
# 1. 레벨 -> 엣지 변환
# ---------------------------------------------------------------------------
def to_edge(condition):
    """레벨 조건을 엣지 사건으로 바꾼다 (명세 §3.2).

        s_t = c_t AND NOT c_{t-1}

    즉 조건이 **거짓에서 참으로 전환된 첫날만** 사건으로 센다.
    RSI가 5일 연속 30 미만이면 사건은 1개지 5개가 아니다.

    첫 거래일 처리
    --------------
    c_{-1}은 존재하지 않으므로 False로 간주한다 (명세 §3.2). 따라서 워밍업
    구간 이후 첫날에 조건이 이미 참이면 사건으로 센다.

    shift(1)에 GUARD 주석을 붙이지 않는 이유
    ----------------------------------------
    여기 shift(1)은 **어제 조건값을 참조**하는 연산이다. 과거를 보는 정상적인
    참조이지 미래 참조 방지 장치가 아니다. 진입을 하루 늦추는 shift(1)
    (`# LOOKAHEAD GUARD`가 붙는 그것)과 목적이 다르다. 아무 데나 GUARD를
    붙이면 주석이 의미를 잃는다 (명세 §3.1).

    이 함수는 티커 하나짜리 Series만 받는다. 티커 경계는 호출부가 책임진다.

    Parameters
    ----------
    condition : Series[bool]
        레벨 조건. NaN(워밍업)이 섞여 있어도 된다.

    Returns
    -------
    Series[bool]
        엣지 사건. 조건이 NaN인 자리는 False.
    """
    filled = condition.fillna(False)          # -> Series[bool] (행 수,), 워밍업은 "조건 거짓"으로 취급
    filled = filled.astype(bool)              # -> Series[bool] (행 수,)

    previous = filled.shift(1)                # -> Series[object] (행 수,), 어제 조건. 첫 행 NaN
    previous = previous.fillna(False)         # -> Series[object] (행 수,), c_{-1} = False (명세 §3.2)
    previous = previous.astype(bool)          # -> Series[bool] (행 수,)

    edge = filled & ~previous                 # -> Series[bool] (행 수,), 거짓->참 전환일만 True

    return edge


# ---------------------------------------------------------------------------
# 2. 지표 계산 (명세 §3.3.2 — 이탈 금지)
# ---------------------------------------------------------------------------
def add_indicators(df):
    """명세 §3.3.2의 규약대로 S1~S5에 필요한 지표를 붙인다.

    지표 구현은 D3~D4에서 만든 src/indicators.py를 **재사용**한다. 여기서
    평활을 다시 짜면 같은 이름의 지표가 두 벌 생기고, 한쪽만 고쳤을 때 원인을
    추적할 수 없게 된다.

    규약 (기본값에 맡기면 안 되는 자리들)
    -------------------------------------
    - RSI    : Wilder RMA, alpha=1/14, 첫 14일 단순평균 시딩
    - 볼린저 : SMA + 롤링 std, window=20, k=2, **ddof=0 명시**
               (pandas 기본값은 ddof=1이라 명시하지 않으면 밴드가 어긋난다)
    - MACD   : EMA 12/26/9, **adjust=False 명시** (config.EWM_ADJUST)
               (pandas 기본값은 adjust=True라 초기 구간이 갈린다)

    Returns
    -------
    DataFrame
        원본 + rsi_wilder_14, bb_* , macd_* 컬럼.
    """
    work = indicators.rsi_wilder(
        df,
        period=config.RSI_PERIOD,
        seed=config.WILDER_SEED,
    )  # -> DataFrame (행 수, 컬럼 수 + 1)

    work = indicators.bollinger(
        work,
        period=config.BB_PERIOD,
        num_std=config.BB_NUM_STD,
        ddof=config.BB_STD_DDOF,  # 명세 §3.3.2 — 기본값 금지
    )  # -> DataFrame (행 수, 컬럼 수 + 5)

    work = indicators.macd(
        work,
        fast=config.MACD_FAST,
        slow=config.MACD_SLOW,
        signal=config.MACD_SIGNAL,
    )  # -> DataFrame (행 수, 컬럼 수 + 3), 내부에서 adjust=config.EWM_ADJUST 사용

    return work


# ---------------------------------------------------------------------------
# 3. 신호 정의 (명세 §4.1 — S1~S5 다섯 개가 전부다)
# ---------------------------------------------------------------------------
# 각 항목: (signal_id, 사람이 읽는 이름, 조건을 만드는 함수)
# 조건 함수는 티커 하나짜리 DataFrame을 받아 Series[bool]을 돌려준다.
# ---------------------------------------------------------------------------
def _condition_s1(subset):
    """S1 — RSI 과매도. c_t = RSI_14 < 30 (무조건부 임계선)"""
    rsi_column = f"rsi_wilder_{config.RSI_PERIOD}"  # -> str
    rsi_series = subset[rsi_column]                 # -> Series[float] (행 수,)
    condition = rsi_series < config.RSI_OVERSOLD    # -> Series[bool] (행 수,)

    return condition


def _condition_s2(subset):
    """S2 — RSI 과매수. c_t = RSI_14 > 70 (무조건부 임계선)"""
    rsi_column = f"rsi_wilder_{config.RSI_PERIOD}"   # -> str
    rsi_series = subset[rsi_column]                  # -> Series[float] (행 수,)
    condition = rsi_series > config.RSI_OVERBOUGHT   # -> Series[bool] (행 수,)

    return condition


def _condition_s3(subset):
    """S3 — 볼린저 하단 이탈. c_t = C_t < LB_t (조건부 임계선)"""
    close_series = subset["close"]       # -> Series[float] (행 수,)
    lower_series = subset["bb_lower"]    # -> Series[float] (행 수,)
    condition = close_series < lower_series  # -> Series[bool] (행 수,)

    return condition


def _condition_s4(subset):
    """S4 — 볼린저 상단 이탈. c_t = C_t > UB_t (조건부 임계선)"""
    close_series = subset["close"]       # -> Series[float] (행 수,)
    upper_series = subset["bb_upper"]    # -> Series[float] (행 수,)
    condition = close_series > upper_series  # -> Series[bool] (행 수,)

    return condition


def _condition_s5(subset):
    """S5 — MACD > Signal. 조건부 임계선(임계선이 지표 자신의 EMA)"""
    macd_series = subset["macd_line"]      # -> Series[float] (행 수,)
    signal_series = subset["macd_signal"]  # -> Series[float] (행 수,)
    condition = macd_series > signal_series  # -> Series[bool] (행 수,)

    return condition


SIGNAL_DEFINITIONS = [
    ("S1_rsi_oversold", "S1 RSI<30", _condition_s1),
    ("S2_rsi_overbought", "S2 RSI>70", _condition_s2),
    ("S3_bb_lower_break", "S3 볼린저 하단", _condition_s3),
    ("S4_bb_upper_break", "S4 볼린저 상단", _condition_s4),
    ("S5_macd_cross", "S5 MACD>Signal", _condition_s5),
]  # -> list[tuple[str, str, callable]] (5,)

SIGNAL_NAMES = {signal_id: label for signal_id, label, _ in SIGNAL_DEFINITIONS}  # -> dict (5,)


# ---------------------------------------------------------------------------
# 4. 신호 생성
# ---------------------------------------------------------------------------
def make_signals(df, definitions=None, keep_condition=True):
    """S1~S5 진입 신호를 long format으로 생성한다 (명세 §4.0, §4.1).

    반환 형식이 long인 이유
    ----------------------
    `(date, ticker, signal_id, signal)`로 두면 신호가 몇 개든 스키마가 그대로다.
    wide로 두면 신호를 추가할 때마다 컬럼이 늘어 스키마가 계속 변한다.
    나중에 뉴스 데이터를 (date, ticker)로 조인할 때도 이쪽이 맞는다.

    condition 컬럼을 함께 남기는 이유
    ---------------------------------
    레벨 발동률(엣지 변환 **전**)과 사건 수(엣지 변환 **후**)를 둘 다 봐야
    "레벨 발동률과 사건 빈도는 비례하지 않는다"(명세 §7.1)를 확인할 수 있다.
    평균 지속일수도 두 값의 비로 구한다.

    티커 경계
    ---------
    to_edge()의 shift(1)은 앞 행을 참조하므로 티커 경계를 넘으면 안 된다.
    아래 루프가 티커 하나로 필터링한 subset 안에서만 to_edge()를 부른다.

    Returns
    -------
    DataFrame
        long format (date, ticker, signal_id, condition, signal).
    """
    if definitions is None:
        definitions = SIGNAL_DEFINITIONS  # -> list[tuple] (5,)

    work = add_indicators(df)                    # -> DataFrame (행 수, 컬럼 수 + 9)
    work = work.sort_values(["ticker", "date"])  # -> DataFrame, shift는 행 순서를 그대로 믿는다
    work = work.reset_index(drop=True)           # -> DataFrame

    ticker_column = work["ticker"]           # -> Series[str] (행 수,)
    unique_tickers = ticker_column.unique()  # -> ndarray[str] (티커 수,)
    tickers = sorted(unique_tickers)         # -> list[str] (티커 수,)

    pieces = []  # -> list[DataFrame]

    for ticker in tickers:
        mask = ticker_column == ticker  # -> Series[bool] (행 수,)
        subset = work.loc[mask]         # -> DataFrame (티커 행 수, 컬럼 수)
        subset = subset.copy()          # -> DataFrame, SettingWithCopy 방지

        for signal_id, _label, condition_function in definitions:
            condition = condition_function(subset)  # -> Series[bool] (티커 행 수,)
            edge = to_edge(condition)               # -> Series[bool] (티커 행 수,)

            frame = pd.DataFrame(
                {
                    "date": subset["date"].to_numpy(),
                    "ticker": ticker,
                    "signal_id": signal_id,
                    "condition": condition.to_numpy(),
                    "signal": edge.to_numpy(),
                }
            )  # -> DataFrame (티커 행 수, 5)

            pieces.append(frame)

    combined = pd.concat(pieces, ignore_index=True)  # -> DataFrame (티커 수 * 신호 수 * 행 수, 5)
    combined = combined.sort_values(["signal_id", "ticker", "date"])  # -> DataFrame
    combined = combined.reset_index(drop=True)       # -> DataFrame

    if not keep_condition:
        combined = combined.drop(columns=["condition"])  # -> DataFrame (행 수, 4)

    return combined


# ---------------------------------------------------------------------------
# 5. 사후 수익률
# ---------------------------------------------------------------------------
def forward_returns(df, horizons=None, price_column="close"):
    """사후 수익률을 계산한다 (명세 §3.1, §6.1).

        r_{t,h} = C_{t+1+h} / C_{t+1} - 1

    **분모가 C_t가 아니라 C_{t+1}이다.** 신호는 t일 종가로 확정되지만 진입은
    t+1일 종가이므로, 투자자가 실제로 얻는 수익률의 시작점은 C_{t+1}이다.
    C_t를 분모로 쓰면 신호 확정 당일의 움직임까지 수익으로 세게 되어,
    "종가를 보고 그 종가에 산" 셈이 된다 — 전형적인 미래 참조다.

    왜 t+1일 시가가 아니라 종가인가 (명세 §3.1)
    -------------------------------------------
    지수의 시가는 구성종목 500개의 첫 체결이 동시에 일어나지 않아 일부는 전일
    가격이 섞인 인위적 값이다. D4에서 ^GSPC의 22.0%가 갭 항이 장중 범위보다
    컸음을 확인했다. 종가는 마감 동시호가로 확정되므로 상대적으로 신뢰할 수 있다.

    경계 처리
    ---------
    t+1+h가 데이터 범위를 넘어가면 NaN으로 둔다. 제외는 집계 단계에서 한다
    (명세 §3.4). 지평별로 유효 표본 수가 달라지므로 §6에서 지평마다 n을
    따로 보고한다.

    Returns
    -------
    DataFrame
        원본 + entry_price, fwd_ret_1, fwd_ret_5, ... 컬럼.
    """
    if horizons is None:
        horizons = config.EVENT_HORIZONS  # -> list[int] (4,)

    work = df.copy()                             # -> DataFrame (행 수, 컬럼 수)
    work = work.sort_values(["ticker", "date"])  # -> DataFrame, shift는 행 순서를 믿는다
    work = work.reset_index(drop=True)           # -> DataFrame

    grouped = work.groupby("ticker")        # -> DataFrameGroupBy
    price_by_ticker = grouped[price_column]  # -> SeriesGroupBy

    # LOOKAHEAD GUARD
    # 진입가는 신호 다음날(t+1) 종가다. shift(-1)이 미래를 당겨오는 것처럼
    # 보이지만, 이는 "t일 신호에 대응하는 진입가는 t+1일 종가"라는 **정의**를
    # t일 행에 붙여두는 것이고, 실제 매매 결정은 t일 종가 정보만으로 내려진다.
    # 반대로 진입가를 C_t로 두면 그 순간 미래 참조가 된다 (명세 §3.1).
    entry_price = price_by_ticker.shift(-1)  # -> Series[float] (행 수,), 각 티커 마지막 행 NaN
    work["entry_price"] = entry_price        # -> DataFrame (행 수, 컬럼 수 + 1)

    for horizon in horizons:
        # 청산가는 진입일로부터 h거래일 뒤 종가 = C_{t+1+h}
        # LOOKAHEAD GUARD
        # shift(-(1+h))는 사후 수익률의 **관측 대상**을 t일 행에 붙이는 연산이다.
        # 신호 판단에는 쓰이지 않으며, 오직 "그 신호 뒤에 무슨 일이 있었나"를
        # 사후에 집계하기 위한 것이다. 이 컬럼이 신호 조건에 들어가면 안 된다.
        exit_price = price_by_ticker.shift(-(1 + horizon))  # -> Series[float] (행 수,)

        column_name = f"fwd_ret_{horizon}"  # -> str
        work[column_name] = exit_price / work["entry_price"] - 1  # -> Series[float] (행 수,)

    return work


def attach_forward_returns(signal_frame, returns_frame, horizons=None):
    """신호 long format에 사후 수익률을 붙인다.

    조인 키는 (date, ticker)다. 두 프레임 모두 이 키로 유일하게 식별되므로
    조인 후 행 수가 늘어나면 안 된다 — 늘어나면 어느 한쪽에 중복이 있는 것이고
    그대로 집계하면 같은 사건이 여러 번 세어진다.

    Returns
    -------
    DataFrame
        신호 프레임 + entry_price, fwd_ret_* 컬럼.
    """
    if horizons is None:
        horizons = config.EVENT_HORIZONS  # -> list[int] (4,)

    return_columns = [f"fwd_ret_{horizon}" for horizon in horizons]  # -> list[str] (4,)
    keep_columns = ["date", "ticker", "close", "entry_price"] + return_columns  # -> list[str]

    right = returns_frame[keep_columns]  # -> DataFrame (행 수, 8)

    before_rows = len(signal_frame)  # -> int

    merged = signal_frame.merge(right, on=["date", "ticker"], how="left")  # -> DataFrame

    after_rows = len(merged)  # -> int

    if after_rows != before_rows:
        raise ValueError(
            f"조인 후 행 수가 달라졌다: {before_rows} -> {after_rows}. "
            "(date, ticker) 중복을 확인할 것."
        )

    return merged
