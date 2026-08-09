"""10일차: 무위험수익률(`^IRX`) 수집 + 스냅샷 저장.

프로젝트 루트에서 실행한다.

    python scripts/day10_collect_rf.py

왜 별도 스크립트인가
--------------------
`^IRX`를 백테스트 안에서 매번 내려받으면, 어제 돌린 백테스트와 오늘 돌린
백테스트가 다른 숫자를 낼 수 있다. `day01_collect.py`가 OHLCV를 parquet으로
굳혀둔 것과 같은 이유로 여기서도 스냅샷을 만든다.

`day01_collect.py`에 합치지 않은 이유는, 그 스크립트가 만든 `ohlcv_raw.parquet`은
D1 이후 모든 분석의 입력으로 **이미 굳어 있는 산출물**이기 때문이다. 거기에
티커를 추가하면 파일이 바뀌고, "D1 이후 원본은 그대로다"라는 전제가 깨진다.

무엇을 검사하는가
-----------------
1. 수집 성공 여부 (실패하면 종료 코드 1)
2. 분석구간 결측 — 보간하지 않는다. 몇 건인지 세어서 보고만 한다
3. `^GSPC` 거래일과의 정합 — 채권시장 휴일(콜럼버스 데이, 재향군인의 날)에
   주식시장은 열리므로 며칠이 어긋나는 것이 **정상**이다. 그 날짜를 출력해
   사람이 확인할 수 있게 한다
4. 음수 금리 — 2020년 3월에 실제로 있었다. 오류가 아니므로 세어서 보고만 한다

`config.RF_MISSING_POLICY`와 `config.RF_ALLOW_NEGATIVE`가 이 검사 결과를 어떻게
다룰지 정한다. 이 스크립트는 **판단하지 않고 기록만 한다.**
"""

import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()    # -> Path
PROJECT_ROOT = SCRIPT_PATH.parent.parent  # -> Path
ROOT_TEXT = str(PROJECT_ROOT)             # -> str

if ROOT_TEXT not in sys.path:
    sys.path.insert(0, ROOT_TEXT)

import pandas as pd  # noqa: E402

from src import config  # noqa: E402
from src import data    # noqa: E402


def force_utf8_console():
    """윈도우 콘솔 기본 인코딩(cp949)에서 한글 출력이 깨지는 것을 막는다."""
    streams = (sys.stdout, sys.stderr)  # -> tuple[TextIOWrapper] (2,)

    for stream in streams:
        reconfigure = getattr(stream, "reconfigure", None)  # -> callable | None

        if reconfigure is not None:
            reconfigure(encoding="utf-8")


def compare_calendar(rate_frame, price_frame):
    """`^IRX`와 `^GSPC`의 거래일 차이를 낸다.

    Returns
    -------
    (list[Timestamp], list[Timestamp])
        (주가에만 있는 날, 금리에만 있는 날)
    """
    price_dates = set(price_frame["date"])  # -> set[Timestamp]
    rate_dates = set(rate_frame["date"])    # -> set[Timestamp]

    price_only = sorted(price_dates - rate_dates)  # -> list[Timestamp]
    rate_only = sorted(rate_dates - price_dates)   # -> list[Timestamp]

    return price_only, rate_only


def main():
    force_utf8_console()

    ticker = config.RISK_FREE_SOURCE  # -> str ("^IRX")

    print(f"수집 대상: {ticker} (무위험수익률, 연율 %)")
    print(f"수집 구간: {config.COLLECT_START} ~ {config.COLLECT_END or '오늘'}")
    print()

    # ------------------------------------------------------------------
    # 1. 수집
    # ------------------------------------------------------------------
    frame, failures = data.download_ohlcv([ticker])  # -> (DataFrame, dict)

    if len(failures) > 0:
        print()
        print("수집 실패:")

        for failed_ticker in failures:
            reason = failures[failed_ticker]  # -> str
            print(f"  {failed_ticker}: {reason}")

        print()
        print("무위험수익률을 임의 값으로 대체하지 않는다 (CLAUDE.md 규칙 3, 5).")
        return 1

    if len(frame) == 0:
        print("수집된 행이 없다.")
        return 1

    saved_path = data.save_parquet(frame, config.RAW_IRX_PATH)  # -> Path

    # ------------------------------------------------------------------
    # 2. 분석구간 점검
    # ------------------------------------------------------------------
    analysis_start = pd.Timestamp(config.ANALYSIS_START)  # -> Timestamp

    date_mask = frame["date"] >= analysis_start  # -> Series[bool]
    analysis = frame.loc[date_mask]              # -> DataFrame
    analysis = analysis.reset_index(drop=True)   # -> DataFrame

    rate_column = analysis["close"]  # -> Series[float], 연율 %

    missing_count = int(rate_column.isna().sum())      # -> int
    negative_count = int((rate_column < 0).sum())      # -> int

    print()
    print("-" * 70)
    print(f"분석구간 점검 ({config.ANALYSIS_START} 이후)")
    print("-" * 70)
    print(f"행 수: {len(analysis):,}")
    print(f"결측: {missing_count}건")
    print(f"음수 금리: {negative_count}건 "
          f"(허용={config.RF_ALLOW_NEGATIVE})")
    print(f"금리 범위: {rate_column.min():.3f}% ~ {rate_column.max():.3f}%")

    if negative_count > 0:
        negative_rows = analysis.loc[rate_column < 0]  # -> DataFrame
        print()
        print("  음수 구간 (실제 있었던 일이므로 오류가 아니다):")
        print(negative_rows[["date", "close"]].to_string(index=False))

    # ------------------------------------------------------------------
    # 3. `^GSPC` 거래일과 정합
    # ------------------------------------------------------------------
    price_frame = data.load_parquet(config.RAW_OHLCV_PATH)  # -> DataFrame

    signal_ticker = config.TICKERS["signal"]                    # -> str
    price_mask = price_frame["ticker"] == signal_ticker         # -> Series[bool]
    price_frame = price_frame.loc[price_mask]                   # -> DataFrame
    price_frame = price_frame.loc[price_frame["date"] >= analysis_start]  # -> DataFrame

    price_only, rate_only = compare_calendar(analysis, price_frame)  # -> (list, list)

    print()
    print("-" * 70)
    print("거래일 정합")
    print("-" * 70)
    print(f"{signal_ticker}: {len(price_frame):,}일 / {ticker}: {len(analysis):,}일")
    print()
    print(f"주가에만 있는 날 {len(price_only)}건 — 채권시장 휴일로 예상되는 구간:")

    for date_value in price_only:
        print(f"  {date_value.date()}")

    print()
    print(f"  → 처리 방침: {config.RF_MISSING_POLICY} (config에 확정)")

    if len(rate_only) > 0:
        print()
        print(f"금리에만 있는 날 {len(rate_only)}건 — 주가 스냅샷 종료일 이후로 예상:")
        print(f"  {rate_only[0].date()} ~ {rate_only[-1].date()}")
        print("  → 백테스트는 주가 스냅샷 범위로 자르므로 영향 없다.")

    print()
    print(f"스냅샷 저장: {saved_path}")
    print()
    print("이 스크립트는 판단하지 않는다. 위 수치를 보고 config의 방침이")
    print("여전히 타당한지는 사람이 확인한다.")

    return 0


if __name__ == "__main__":
    exit_code = main()  # -> int
    sys.exit(exit_code)
