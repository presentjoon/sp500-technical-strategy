"""D15 — 국면 × 신호 거래 성과 행렬.

    python scripts/d15_phase_matrix.py

입력은 `reports/day10_trades.csv`(D10 백테스트 산출물)뿐이다. 거래를 새로
생성하지 않고 이미 확정된 거래 기록을 국면별로 나눠 기술통계만 낸다.

무엇을 **하지 않는가** — 전부 의도된 것이다
-------------------------------------------
1. 누적수익률(``prod``)을 계산하지 않는다.
   같은 신호의 거래 구간이 서로 겹치기 때문이다. h=20이면 한 거래가 20거래일을
   점유하는데 그 안에서 다음 신호가 또 발생한다. 겹친 구간의 수익률을 곱하면
   같은 날의 움직임이 두 번 계상된다. 거래별 수익률의 **평균·중앙값**만 낸다.

2. CAGR·샤프·MDD를 계산하지 않는다.
   국면 길이가 695 ~ 2,756거래일로 4배 차이가 나고, 국면별 자본곡선이 없다.
   자본곡선 없이 낸 MDD는 거래 목록의 최저 수익률일 뿐 낙폭이 아니다.
   이 지표들이 필요하면 `src/metrics.py`가 자본곡선 위에서 계산한다 (D11).

3. `reports/day06_diag_regime.csv`의 신호 발생 수를 다시 세지 않는다.
   그 파일은 **신호일** 기준이고 이 파일은 **거래** 기준이다. 두 값이 다른
   것이 정상이며, 같아야 한다고 전제하면 D8과 같은 종류의 혼동이 된다.

4. 표본이 적은 칸도 행을 지우지 않는다.
   ``sufficient`` 열로 표시만 하고 남긴다. 지우면 "어느 국면에 거래가 없었나"가
   결과에서 사라진다.

국면 경계 처리
--------------
``config.MARKET_REGIMES``의 ``recent``는 ``end``가 ``None``이다. 그대로 두면
비교가 전부 False가 되어 **최근 국면 전체가 누락된다.** 원자료의 마지막
거래일로 채운다.
"""

import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()     # -> Path
PROJECT_ROOT = SCRIPT_PATH.parent.parent   # -> Path
ROOT_TEXT = str(PROJECT_ROOT)              # -> str

if ROOT_TEXT not in sys.path:
    sys.path.insert(0, ROOT_TEXT)

import numpy as np   # noqa: E402
import pandas as pd  # noqa: E402

from src import config  # noqa: E402
from src import data    # noqa: E402


TRADES_PATH = PROJECT_ROOT / "reports" / "day10_trades.csv"  # -> Path

# 표본이 이 미만이면 sufficient=False. 통계량은 그대로 내되 표시만 한다.
MIN_TRADES = 20  # -> int

# 경계 판정 폭. **달력일**이다 (거래일 아님).
BOUNDARY_DAYS = 1  # -> int

# 표준편차는 표본표준편차. 프로젝트 전체가 ddof=1로 통일돼 있다.
STD_DDOF = 1  # -> int

OUTPUTS = {20: "day15_phase_matrix.csv", 1: "day15_phase_matrix_h1.csv"}  # -> dict[int, str]


def force_utf8_console():
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)  # -> callable | None

        if reconfigure is not None:
            reconfigure(encoding="utf-8")


def resolve_regimes(last_trading_day):
    """``MARKET_REGIMES``를 (키, 이름, 시작, 끝) 목록으로 편다.

    ``end``가 ``None``인 국면은 `last_trading_day`로 닫는다. config는 읽기만
    하고 고치지 않는다 — 백테스트 가정은 config에만 있다는 규칙 때문이다.
    """
    rows = []  # -> list[dict]

    for key in config.MARKET_REGIMES:
        spec = config.MARKET_REGIMES[key]  # -> dict

        start = pd.Timestamp(spec["start"])  # -> Timestamp

        if spec["end"] is None:
            end = pd.Timestamp(last_trading_day)  # -> Timestamp, 진행 중인 국면
            open_ended = True                     # -> bool
        else:
            end = pd.Timestamp(spec["end"])  # -> Timestamp
            open_ended = False

        rows.append({
            "regime": key,
            "regime_name": spec["name"],
            "start": start,
            "end": end,
            "open_ended": open_ended,
        })

    return rows


def boundary_dates(regimes):
    """모든 국면의 시작일과 종료일을 모은다 (경계 판정용).

    국면이 서로 맞닿아 있으므로 앞 국면의 end와 뒤 국면의 start는 하루 차이다.
    둘 다 넣어도 판정 결과는 같지만, 어느 한쪽만 넣으면 국면 배정에 따라
    같은 거래가 경계로 잡히기도 하고 안 잡히기도 한다.
    """
    dates = []  # -> list[Timestamp]

    for regime in regimes:
        dates.append(regime["start"])
        dates.append(regime["end"])

    unique = sorted(set(dates))  # -> list[Timestamp]

    return unique


def assign_regime(entry_dates, regimes):
    """진입일을 국면 키로 바꾼다. 구간은 양끝 포함 ``[start, end]``."""
    assigned = pd.Series(index=entry_dates.index, dtype=object)  # -> Series[object]

    for regime in regimes:
        in_range = (entry_dates >= regime["start"]) & (entry_dates <= regime["end"])  # -> Series[bool]

        assigned = assigned.mask(in_range, regime["regime"])  # -> Series[object]

    return assigned


def near_boundary(entry_dates, boundaries, tolerance_days=None):
    """진입일이 어느 경계와도 `tolerance_days` **달력일** 이내인지."""
    if tolerance_days is None:
        tolerance_days = BOUNDARY_DAYS  # -> int

    limit = pd.Timedelta(days=tolerance_days)  # -> Timedelta

    flags = pd.Series(False, index=entry_dates.index)  # -> Series[bool]

    for boundary in boundaries:
        distance = (entry_dates - boundary).abs()  # -> Series[Timedelta]

        flags = flags | (distance <= limit)  # -> Series[bool]

    return flags


def build_matrix(trades, regimes, signal_ids):
    """(신호 × 국면) 전 조합에 대해 한 행씩. 빈 칸도 남긴다."""
    rows = []  # -> list[dict]

    for signal_id in signal_ids:
        for regime in regimes:
            match = (trades["signal_id"] == signal_id)          # -> Series[bool]
            match = match & (trades["regime"] == regime["regime"])  # -> Series[bool]

            cell = trades.loc[match]  # -> DataFrame

            n_trades = len(cell)  # -> int

            returns = cell["return_pct"].to_numpy()  # -> ndarray[float] (n_trades,)

            if n_trades == 0:
                mean_return = np.nan
                median_return = np.nan
                std_return = np.nan
                win_rate = np.nan
            else:
                mean_return = float(np.mean(returns))      # -> float, %
                median_return = float(np.median(returns))  # -> float, %

                # n=1이면 ddof=1 분산이 정의되지 않는다. 0으로 채우지 않는다 —
                # "산포가 0"과 "산포를 못 잰다"는 다른 말이다.
                if n_trades >= 2:
                    std_return = float(np.std(returns, ddof=STD_DDOF))  # -> float, %
                else:
                    std_return = np.nan

                wins = int((returns > 0).sum())  # -> int, 0은 승리로 세지 않는다
                win_rate = wins / n_trades       # -> float, 0~1 비율

            rows.append({
                "signal_id": signal_id,
                "regime": regime["regime"],
                "regime_name": regime["regime_name"],
                "n_trades": n_trades,
                "mean_return_pct": mean_return,
                "median_return_pct": median_return,
                "std_return_pct": std_return,
                "win_rate": win_rate,
                "n_boundary": int(cell["is_boundary"].sum()) if n_trades else 0,
                "sufficient": n_trades >= MIN_TRADES,
            })

    return pd.DataFrame(rows)  # -> DataFrame (신호 수 * 국면 수, 10)


def main():
    force_utf8_console()

    # --- 마지막 거래일 (recent 국면을 닫는 데만 쓴다) ------------------------
    price = data.load_parquet(config.RAW_OHLCV_PATH)  # -> DataFrame (18424, 8)

    ticker_mask = price["ticker"] == config.TICKERS["signal"]  # -> Series[bool]
    last_trading_day = price.loc[ticker_mask, "date"].max()    # -> Timestamp

    regimes = resolve_regimes(last_trading_day)  # -> list[dict] (6,)
    boundaries = boundary_dates(regimes)         # -> list[Timestamp]

    print("=" * 84)
    print("국면 정의 (config.MARKET_REGIMES)")
    print("=" * 84)
    for regime in regimes:
        note = "  ← end=None을 마지막 거래일로 닫음" if regime["open_ended"] else ""
        print(f"  {regime['regime']:<20} {regime['regime_name']:<8} "
              f"{regime['start'].date()} ~ {regime['end'].date()}{note}")

    print()
    print(f"경계 날짜 {len(boundaries)}개 (±{BOUNDARY_DAYS} 달력일로 판정):")
    print("  " + ", ".join(str(b.date()) for b in boundaries))

    # --- 거래 기록 -----------------------------------------------------------
    trades = pd.read_csv(TRADES_PATH, encoding="utf-8-sig")  # -> DataFrame (2766, 10)

    trades["entry_date"] = pd.to_datetime(trades["entry_date"])  # -> DataFrame

    signal_ids = sorted(trades["signal_id"].unique())  # -> list[str] (5,)

    trades["regime"] = assign_regime(trades["entry_date"], regimes)          # -> DataFrame (2766, 11)
    trades["is_boundary"] = near_boundary(trades["entry_date"], boundaries)  # -> DataFrame (2766, 12)

    unassigned = int(trades["regime"].isna().sum())  # -> int

    print()
    print(f"국면 미배정 거래: {unassigned}건")

    if unassigned:
        raise SystemExit("!! 국면에 배정되지 않은 거래가 있다. 국면 경계를 확인하라.")

    # --- 지평별 출력 ---------------------------------------------------------
    for horizon in sorted(OUTPUTS):
        subset = trades.loc[trades["holding_days"] == horizon]  # -> DataFrame

        matrix = build_matrix(subset, regimes, signal_ids)  # -> DataFrame (30, 10)

        output_path = PROJECT_ROOT / "reports" / OUTPUTS[horizon]  # -> Path
        matrix.to_csv(output_path, index=False, encoding="utf-8-sig")

        print()
        print("=" * 84)
        print(f"holding_days = {horizon}  —  거래 {len(subset)}건")
        print("=" * 84)
        print(matrix.to_string(index=False))

        total = int(matrix["n_trades"].sum())              # -> int
        enough = int(matrix["sufficient"].sum())           # -> int
        boundary_total = int(matrix["n_boundary"].sum())   # -> int

        print()
        print(f"  합계 거래 {total}건 (원본 {len(subset)}건, 일치 {total == len(subset)})")
        print(f"  sufficient=True 칸: {enough} / {len(matrix)}")
        print(f"  경계 ±{BOUNDARY_DAYS}일 거래: {boundary_total}건")
        print(f"  저장: reports/{OUTPUTS[horizon]}  ({matrix.shape[0]}행 × {matrix.shape[1]}열)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
