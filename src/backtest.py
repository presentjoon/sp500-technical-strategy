"""D10 백테스트 엔진 — 고정 보유기간, 중복 진입 없음.

가정은 전부 `src/config.py` §17에 있고 **결과를 보기 전에 커밋했다**
(커밋 `bf4fea2`). 이 모듈은 그 값을 읽어 쓸 뿐 스스로 정하지 않는다.
숫자를 여기 하드코딩하면 2단계에서 Backtrader로 옮길 때 무엇을 옮겨야 하는지
알 수 없게 된다.

이 엔진이 하지 않는 것
----------------------
- **새 청산 규칙을 만들지 않는다.** 청산은 `config.HOLDING_DAYS`의 고정 보유기간
  하나뿐이다. 손절·익절·추적손절은 전부 새로운 자유 파라미터이고, 그것을 넣는
  순간 확증 검정 family(m=20)를 다시 세야 한다.
- **파라미터를 바꿔가며 비교하지 않는다.** D10은 엔진 제작일이지 탐색일이 아니다.

시점 규약 (`docs/signal_spec.md` §3.1)
--------------------------------------
    신호 확정   t일 종가
    진입        t+1일 종가   C_{t+1}
    청산        t+1+h일 종가 C_{t+1+h}

`config.FILL_TIMING = "next_close"`가 이것을 고정한다. 지수 시가는 구성 종목
500개가 동시에 체결되지 않아 **일부는 전일 가격이 섞인 인위적 값**이라
(D4 실측: 거래일의 22.0%에서 시가 갭이 장중 범위보다 컸다) 진입가로 쓸 수 없다.

거래 로그 스키마 (컬럼명 변경 금지)
-----------------------------------
2단계에서 Backtrader 출력과 행 단위로 대조할 표다. 이름이 다르면 대조 자체가
수작업이 된다.

    entry_date, exit_date, entry_price, exit_price, size, pnl,
    return_pct, holding_days, exit_reason

사용 예 (프로젝트 루트 기준)
----------------------------
    from src import backtest, config, data, signals

    price = data.load_parquet(config.RAW_OHLCV_PATH)
    rate = data.load_parquet(config.RAW_IRX_PATH)

    trades = backtest.run_all(price, rate)
"""

import numpy as np
import pandas as pd

from src import config
from src import signals


# 거래 로그 표준 스키마. 순서까지 고정한다.
TRADE_COLUMNS = [
    "signal_id",
    "holding_days",
    "entry_date",
    "exit_date",
    "entry_price",
    "exit_price",
    "size",
    "pnl",
    "return_pct",
    "exit_reason",
]  # -> list[str] (10,)

# 표준편차를 쓰는 자리는 전부 표본표준편차로 통일한다.
# pandas 기본값은 1이지만 numpy는 0이라, 명시하지 않으면 어느 쪽인지 알 수 없다.
STD_DDOF = 1  # -> int


# ---------------------------------------------------------------------------
# 1. 무위험수익률
# ---------------------------------------------------------------------------
def prepare_risk_free(rate_frame, price_dates):
    """`^IRX` 연율(%)을 거래일별 단리 일별 수익률로 바꾼다.

    두 가지를 여기서 처리한다.

    1. **연율 → 일별 환산** (`config.RF_CONVERSION`)
       `"simple_daily"`이면 `(연율% / 100) / 252`. 복리 환산이 아니다.
       252는 D1 실측 251.9일/년에 근거한 관례값이다.

    2. **채권시장 휴일 채우기** (`config.RF_MISSING_POLICY`)
       주식시장은 열렸는데 채권시장이 쉰 날이 분석구간에 6일 있다
       (콜럼버스 데이, 재향군인의 날). 결측이 아니라 구조적 차이이며,
       T-bill 이자는 그 날에도 발생하므로 직전 영업일 금리를 이어 쓴다.

    Returns
    -------
    (Series[float], dict)
        날짜를 인덱스로 하는 일별 무위험수익률, 그리고 진단 정보.
    """
    work = rate_frame.copy()                # -> DataFrame (행 수, 컬럼 수)
    work = work.sort_values("date")         # -> DataFrame
    work = work.reset_index(drop=True)      # -> DataFrame

    annual_percent = work["close"]  # -> Series[float], 연율 %

    if config.RF_CONVERSION == "simple_daily":
        annual_fraction = annual_percent / 100.0                       # -> Series[float]
        daily_rate = annual_fraction / config.TRADING_DAYS_PER_YEAR     # -> Series[float]
    else:
        raise ValueError(
            f"RF_CONVERSION을 처리할 수 없다: {config.RF_CONVERSION!r}. "
            "새 방식을 쓰려면 config에 정의하고 여기에 분기를 추가할 것."
        )

    if not config.RF_ALLOW_NEGATIVE:
        raise ValueError(
            "RF_ALLOW_NEGATIVE=False는 구현하지 않았다. "
            "음수 금리를 0으로 바닥 대는 것은 실측을 고치는 일이라, "
            "config만 바꾸는 것으로 조용히 적용되면 안 된다."
        )

    indexed = pd.Series(daily_rate.to_numpy(), index=work["date"])  # -> Series[float]

    target_index = pd.DatetimeIndex(price_dates)  # -> DatetimeIndex (거래일 수,)

    aligned = indexed.reindex(target_index)  # -> Series[float], 없는 날은 NaN

    missing_before = int(aligned.isna().sum())  # -> int
    missing_dates = list(aligned.index[aligned.isna()])  # -> list[Timestamp]

    if config.RF_MISSING_POLICY == "forward_fill":
        # 직전 영업일 금리를 이어 쓴다. 이것이 "보간 금지" 규칙에 걸리지 않는
        # 이유는 config §17에 적어뒀다 — 가격이 아니고 신호를 만들지도 않는다.
        aligned = aligned.ffill()  # -> Series[float]
    elif config.RF_MISSING_POLICY == "zero":
        aligned = aligned.fillna(0.0)  # -> Series[float]
    else:
        raise ValueError(f"RF_MISSING_POLICY를 처리할 수 없다: {config.RF_MISSING_POLICY!r}")

    missing_after = int(aligned.isna().sum())  # -> int, 첫 행이 결측이면 남는다

    diagnostic = {
        "n_days": len(aligned),
        "missing_before": missing_before,
        "missing_after": missing_after,
        "missing_dates": missing_dates,
        "policy": config.RF_MISSING_POLICY,
        "negative_days": int((aligned < 0).sum()),
    }  # -> dict (6,)

    return aligned, diagnostic


# ---------------------------------------------------------------------------
# 2. 거래 생성 — 원리 버전
# ---------------------------------------------------------------------------
def generate_trades_loop(dates, prices, entry_flags, holding_days):
    """포지션 상태 기계를 for 루프로 그대로 쓴 버전.

    **이 함수가 정의다.** 아래 벡터화 버전은 이 결과를 재현해야 하고,
    `check_equivalence()`가 두 결과가 같은지 확인한다. 벡터화 코드는 빠르지만
    "무슨 일이 일어나는지"가 안 보이므로, 규칙은 여기에 적고 저기서 베낀다.

    상태 기계
    ---------
        보유 안 함 + 신호 → t+1 종가에 진입
        보유 중          → 신호를 무시 (config.ALLOW_OVERLAP=False)
        보유 중 + h일 경과 → t+1+h 종가에 청산

    Parameters
    ----------
    dates : ndarray[datetime64]  (T,)
    prices : ndarray[float]      (T,)   종가
    entry_flags : ndarray[bool]  (T,)   t일에 확정된 진입 신호
    holding_days : int

    Returns
    -------
    list[dict]
        거래 하나가 dict 하나. 컬럼은 TRADE_COLUMNS 중 가격 관련 항목.
    """
    n_days = len(dates)  # -> int

    trades = []          # -> list[dict]
    position_open = False  # -> bool
    entry_index = None     # -> int | None

    for t in range(n_days):
        if position_open:
            elapsed = t - entry_index  # -> int

            if elapsed < holding_days:
                continue  # 아직 보유 중. 이 날의 신호는 무시한다

            trades.append({
                "entry_index": entry_index,
                "exit_index": t,
                "exit_reason": "time_stop",
            })
            position_open = False
            entry_index = None

            # 여기서 continue하지 않는다.
            # 청산일 종가에 확정된 신호는 그 다음 날(t+1) 진입이므로 보유 구간이
            # 겹치지 않는다. 이 날을 건너뛰면 유효한 거래를 놓치고,
            # 벡터화 버전과 결과가 갈린다.

        if not entry_flags[t]:
            continue

        # LOOKAHEAD GUARD
        # t일 종가로 확정된 신호의 진입 시점은 t+1일 종가다 (명세 §3.1).
        # entry_index를 t가 아니라 t+1로 잡는 것이 그 지연을 구현한다.
        # t로 잡으면 "종가를 보고 그 종가에 샀다"가 되어 미래 참조다.
        fill_index = t + 1  # -> int

        if fill_index >= n_days:
            # 마지막 거래일에 발생한 신호는 진입할 다음 날이 없다.
            continue

        exit_index = fill_index + holding_days  # -> int

        if exit_index >= n_days:
            # 보유기간이 데이터 범위를 넘어가는 사건은 제외한다 (명세 §3.4).
            # 여기서 잘라내지 않고 마지막 날 강제청산하면, 보유기간이 h가 아닌
            # 거래가 섞여 "고정 보유기간"이라는 규칙이 깨진다.
            continue

        position_open = True
        entry_index = fill_index

    return trades


def generate_trades_vectorized(dates, prices, entry_flags, holding_days):
    """위 루프와 같은 결과를 내는 벡터화 버전.

    중복 진입 금지가 있어 완전한 벡터화는 불가능하다 — 다음 진입 가능 시점이
    직전 청산 시점에 의존하므로 **경로 의존적**이기 때문이다. 그래서 신호가
    있는 인덱스만 훑는다 (전체 거래일이 아니라 사건 수만큼).

    S1 기준 6,684일을 훑던 것이 65건만 훑는 것으로 줄어든다.
    """
    n_days = len(dates)  # -> int

    signal_indices = np.flatnonzero(entry_flags)  # -> ndarray[int] (사건 수,)

    trades = []            # -> list[dict]
    next_free_index = 0    # -> int, 이 인덱스 이전에는 진입할 수 없다

    for t in signal_indices:
        if t < next_free_index:
            continue  # 아직 보유 중

        fill_index = t + 1  # -> int, LOOKAHEAD GUARD (명세 §3.1, 위 루프와 동일)

        if fill_index >= n_days:
            continue

        exit_index = fill_index + holding_days  # -> int

        if exit_index >= n_days:
            continue

        trades.append({
            "entry_index": fill_index,
            "exit_index": exit_index,
            "exit_reason": "time_stop",
        })

        next_free_index = exit_index  # 청산일에 다시 진입 가능

    return trades


def check_equivalence(dates, prices, entry_flags, holding_days):
    """두 구현이 같은 거래를 내는지 확인한다.

    벡터화 버전이 빠르다는 이유로 원리 버전을 지우면, 나중에 결과가 이상할 때
    "규칙이 무엇이었는가"를 코드에서 되읽을 수 없게 된다. 그래서 둘을 남기고
    같은지 검사한다.

    Returns
    -------
    (bool, str)
        일치 여부와 설명.
    """
    loop_trades = generate_trades_loop(dates, prices, entry_flags, holding_days)      # -> list[dict]
    fast_trades = generate_trades_vectorized(dates, prices, entry_flags, holding_days)  # -> list[dict]

    if len(loop_trades) != len(fast_trades):
        return False, f"거래 수 불일치: 루프 {len(loop_trades)} vs 벡터 {len(fast_trades)}"

    for position in range(len(loop_trades)):
        left = loop_trades[position]   # -> dict
        right = fast_trades[position]  # -> dict

        same_entry = left["entry_index"] == right["entry_index"]  # -> bool
        same_exit = left["exit_index"] == right["exit_index"]     # -> bool

        if not (same_entry and same_exit):
            return False, (
                f"{position}번째 거래 불일치: "
                f"루프 ({left['entry_index']}, {left['exit_index']}) vs "
                f"벡터 ({right['entry_index']}, {right['exit_index']})"
            )

    return True, f"두 구현 일치 ({len(loop_trades)}건)"


# ---------------------------------------------------------------------------
# 3. 손익 계산
# ---------------------------------------------------------------------------
def apply_costs(entry_price, exit_price):
    """수수료와 슬리피지를 반영한 실효 체결가를 낸다.

    둘 다 **편도** 기준이므로 진입과 청산에 각각 적용한다.
    매수는 불리하게(비싸게), 매도는 불리하게(싸게) 체결된다고 본다.

        진입 실효가 = C * (1 + cost)
        청산 실효가 = C * (1 - cost)

    왕복 총 비용은 config 기준 20bp = 0.20%다.
    """
    cost_bps = config.COMMISSION_BPS + config.SLIPPAGE_BPS  # -> int (10)
    cost_fraction = cost_bps / 10_000.0                     # -> float (0.001)

    effective_entry = entry_price * (1.0 + cost_fraction)  # -> float
    effective_exit = exit_price * (1.0 - cost_fraction)    # -> float

    return effective_entry, effective_exit


def build_trade_frame(signal_id, holding_days, dates, prices, raw_trades):
    """거래 인덱스 목록을 표준 스키마 DataFrame으로 만든다."""
    rows = []  # -> list[dict]

    capital = float(config.INITIAL_CAPITAL) * float(config.POSITION_SIZE)  # -> float

    for trade in raw_trades:
        entry_index = trade["entry_index"]  # -> int
        exit_index = trade["exit_index"]    # -> int

        entry_price = float(prices[entry_index])  # -> float
        exit_price = float(prices[exit_index])    # -> float

        effective_entry, effective_exit = apply_costs(entry_price, exit_price)  # -> (float, float)

        # 지수는 소수 단위로 살 수 있다고 본다 (ETF 소수점 매매 가정).
        # 정수 주식 수로 반올림하면 초기자본 10만에서 잔돈이 남고, 그 잔돈이
        # 지수 수준에 따라 달라져 시기별로 다른 효과를 만든다.
        size = capital / effective_entry  # -> float

        pnl = size * (effective_exit - effective_entry)  # -> float
        return_pct = 100.0 * (effective_exit / effective_entry - 1.0)  # -> float

        rows.append({
            "signal_id": signal_id,
            "holding_days": holding_days,
            "entry_date": dates[entry_index],
            "exit_date": dates[exit_index],
            "entry_price": entry_price,
            "exit_price": exit_price,
            "size": size,
            "pnl": pnl,
            "return_pct": return_pct,
            "exit_reason": trade["exit_reason"],
        })

    frame = pd.DataFrame(rows, columns=TRADE_COLUMNS)  # -> DataFrame (거래 수, 10)

    return frame


# ---------------------------------------------------------------------------
# 4. 실행
# ---------------------------------------------------------------------------
def prepare_price_frame(price_frame, ticker=None, analysis_start=None):
    """단일 티커의 분석구간 종가 시계열을 만든다.

    `groupby("ticker")`를 쓰지 않고 티커를 먼저 거르는 이유는, 이 함수가 낸
    결과가 인덱스 기반 상태 기계에 들어가기 때문이다. 티커가 섞인 배열에
    `t+1`을 적용하면 티커 경계에서 다음 티커의 첫날로 넘어간다.
    **경계를 없애는 것이 경계를 조심하는 것보다 안전하다.**
    """
    if ticker is None:
        ticker = config.TICKERS["signal"]  # -> str

    if analysis_start is None:
        analysis_start = config.ANALYSIS_START  # -> str

    ticker_mask = price_frame["ticker"] == ticker  # -> Series[bool]
    frame = price_frame.loc[ticker_mask]           # -> DataFrame
    frame = frame.sort_values("date")              # -> DataFrame
    frame = frame.reset_index(drop=True)           # -> DataFrame

    start = pd.Timestamp(analysis_start)          # -> Timestamp
    frame = frame.loc[frame["date"] >= start]     # -> DataFrame
    frame = frame.reset_index(drop=True)          # -> DataFrame

    return frame


def run_signal(price_frame, signal_frame, signal_id, holding_days, verify=True):
    """신호 하나 × 보유기간 하나를 백테스트한다.

    Returns
    -------
    (DataFrame, str)
        거래 로그와 등가성 검사 메시지.
    """
    id_mask = signal_frame["signal_id"] == signal_id  # -> Series[bool]
    one_signal = signal_frame.loc[id_mask]            # -> DataFrame
    one_signal = one_signal[["date", "signal"]]       # -> DataFrame

    merged = price_frame.merge(one_signal, on="date", how="left")  # -> DataFrame

    if len(merged) != len(price_frame):
        raise ValueError("신호 병합에서 행 수가 변했다. date 중복을 의심하라.")

    dates = merged["date"].to_numpy()                          # -> ndarray[datetime64] (T,)
    prices = merged["close"].to_numpy()                        # -> ndarray[float] (T,)
    entry_flags = merged["signal"].fillna(False).to_numpy(dtype=bool)  # -> ndarray[bool] (T,)

    message = ""  # -> str

    if verify:
        is_same, message = check_equivalence(dates, prices, entry_flags, holding_days)  # -> (bool, str)

        if not is_same:
            raise ValueError(f"루프/벡터 구현 불일치 — {message}")

    raw_trades = generate_trades_vectorized(dates, prices, entry_flags, holding_days)  # -> list[dict]

    frame = build_trade_frame(signal_id, holding_days, dates, prices, raw_trades)  # -> DataFrame

    return frame, message


def run_all(price_frame, signal_frame, holding_days=None, verify=True):
    """전 신호 × 전 보유기간.

    Returns
    -------
    (DataFrame, list[str])
        거래 로그 전체와 조합별 등가성 검사 메시지.
    """
    if holding_days is None:
        holding_days = config.HOLDING_DAYS  # -> list[int] (4,)

    prepared = prepare_price_frame(price_frame)  # -> DataFrame

    frames = []    # -> list[DataFrame]
    messages = []  # -> list[str]

    for signal_id, label, _condition in signals.SIGNAL_DEFINITIONS:
        for horizon in holding_days:
            frame, message = run_signal(
                prepared, signal_frame, signal_id, horizon, verify=verify
            )  # -> (DataFrame, str)

            frames.append(frame)
            messages.append(f"{signal_id} h={horizon}: {message}")

    combined = pd.concat(frames, ignore_index=True)  # -> DataFrame (총 거래 수, 10)

    return combined, messages
