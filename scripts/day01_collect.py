"""1일차: S&P 500 지수 데이터 수집 + 감사 리포트 생성.

CLAUDE.md의 검증 명령 중 하나다. 프로젝트 루트에서 실행한다.

    python scripts/day01_collect.py

하는 일
--------
1. config.TICKERS의 티커를 수집한다 (^GSPC = 신호용, ^SP500TR = 벤치마크용)
2. 원본을 data/raw/ohlcv_raw.parquet 으로 굳힌다
3. audit()을 돌려 reports/day01_audit.txt 를 생성한다
4. 충격일 목록을 reports/day01_shock_days.csv 로 저장한다
5. 조사용 요약(하락 상위 10일 / 상승 상위 10일 / 연도별 충격일 수)을 출력한다
6. 반드시 통과해야 하는 검사(중복 0, OHLC 위반 0, 연평균 행 수 252 근처)를
   확인하고, 하나라도 실패하면 종료 코드 1로 죽는다

왜 실패 시 죽어야 하는가
------------------------
검증 스크립트가 조용히 성공하면 검증이 아니다. 데이터가 깨진 채로 다음 단계에
넘어가면, 나중에 백테스트 결과가 이상할 때 원인이 전략인지 데이터인지
구분할 수 없게 된다. 여기서 막는 것이 가장 싸다.

5번이 오늘의 핵심 관찰 대상이다
-------------------------------
연도별 충격일 개수를 보는 목적은 "큰 하락일과 큰 상승일이 같은 해에 몰려
있는가"를 확인하는 것이다. 만약 그렇다면, 폭락을 피하려고 시장에서 빠져나온
전략은 바로 뒤따라오는 급반등도 같이 놓친다는 뜻이 된다.
이것이 reports/why_it_fails.md 의 첫 번째 근거가 된다.
"""

import sys
from pathlib import Path

# 이 스크립트는 scripts/ 안에서 실행되므로 sys.path에 프로젝트 루트가 없다.
# src 패키지를 import하려면 루트를 직접 넣어줘야 한다.
SCRIPT_PATH = Path(__file__).resolve()   # -> Path (이 파일)
PROJECT_ROOT = SCRIPT_PATH.parent.parent  # -> Path (프로젝트 루트)
ROOT_TEXT = str(PROJECT_ROOT)             # -> str

if ROOT_TEXT not in sys.path:
    sys.path.insert(0, ROOT_TEXT)

from src import config  # noqa: E402 (sys.path 조작 뒤에 와야 한다)
from src import data    # noqa: E402


def force_utf8_console():
    """윈도우 콘솔 기본 인코딩(cp949)에서 한글 출력이 깨지는 것을 막는다.

    stdout과 stderr를 둘 다 바꿔야 한다. print()는 stdout으로 나가지만
    logging.StreamHandler는 기본값이 stderr라서, stdout만 고치면 로그 줄만
    깨진 채로 남는다.

    reconfigure()는 스트림 객체를 그 자리에서 바꾸므로, 이미 그 객체를
    붙들고 있는 로그 핸들러에도 소급 적용된다.
    """
    streams = (sys.stdout, sys.stderr)  # -> tuple[TextIOWrapper] (2,)

    for stream in streams:
        reconfigure = getattr(stream, "reconfigure", None)  # -> callable | None

        if reconfigure is not None:
            reconfigure(encoding="utf-8")


def check_audit_result(result):
    """CLAUDE.md가 "반드시"라고 규정한 항목만 골라 확인한다.

    Returns
    -------
    list[str]
        실패 메시지 목록. 전부 통과하면 빈 리스트.
    """
    problems = []  # -> list[str]

    # (1) (date, ticker) 중복은 반드시 0
    duplicate_count = result["duplicate_count"]  # -> int

    if duplicate_count != 0:
        problems.append(f"(date, ticker) 중복이 {duplicate_count}건 있다 (0이어야 함)")

    # (2) OHLC 정합성 위반도 반드시 0
    violation_count = result["ohlc_violation_count"]  # -> int

    if violation_count != 0:
        problems.append(f"OHLC 정합성 위반이 {violation_count}건 있다 (0이어야 함)")

    # (3) 티커별 연평균 행 수는 252 근처
    coverage = result["coverage"]                     # -> DataFrame (티커 수, 5)
    expected = config.TRADING_DAYS_PER_YEAR           # -> int
    tolerance = config.ROWS_PER_YEAR_TOLERANCE        # -> int
    lower_bound = expected - tolerance                # -> int
    upper_bound = expected + tolerance                # -> int

    for row_index in coverage.index:
        ticker = coverage.loc[row_index, "ticker"]              # -> str
        rows_per_year = coverage.loc[row_index, "rows_per_year"]  # -> float

        too_few = rows_per_year < lower_bound   # -> bool
        too_many = rows_per_year > upper_bound  # -> bool

        if too_few or too_many:
            problems.append(
                f"{ticker}의 연평균 행 수가 {rows_per_year} "
                f"({lower_bound}~{upper_bound} 밖)"
            )

    return problems


def build_shock_table(df):
    """신호용 티커의 분석 구간 충격일 목록을 만든다.

    계산 순서가 중요하다
    --------------------
    수익률을 먼저 계산하고(전체 수집 구간), 그 다음에 분석 구간을 자른다.
    순서를 뒤집으면 2000-01-03(분석 구간 첫 거래일)의 수익률이 NaN이 된다.
    비교할 전날(1999-12-31)이 잘려나가기 때문이다. 지표 워밍업과 완전히 같은
    논리다 — 계산은 긴 구간에서, 평가는 짧은 구간에서.

    신호용 티커만 쓰는 이유
    ----------------------
    충격일은 "신호가 무엇에 반응하는가"를 보려는 것이므로, 신호를 생성하는
    가격지수(^GSPC)를 기준으로 한다. 벤치마크(^SP500TR)는 성과 비교용이지
    신호 생성용이 아니다 (CLAUDE.md 규칙 5).

    Returns
    -------
    DataFrame
        (충격일 수, 10). daily_return_pct 컬럼이 추가된다.
    """
    signal_ticker = config.TICKERS["signal"]  # -> str ("^GSPC")

    # 1) 전체 수집 구간에서 수익률 계산 → 충격일 추출.
    #    shock_days()가 내부에서 groupby("ticker")를 쓰므로 티커 경계는 안전하다.
    all_shocks = data.shock_days(df, threshold=config.SHOCK_THRESHOLD)  # -> DataFrame (충격일 수, 9)

    # 2) 신호용 티커만 남긴다.
    shock_ticker_column = all_shocks["ticker"]              # -> Series[str] (충격일 수,)
    signal_mask = shock_ticker_column == signal_ticker      # -> Series[bool] (충격일 수,)
    signal_shocks = all_shocks.loc[signal_mask]             # -> DataFrame (신호 충격일 수, 9)
    signal_shocks = signal_shocks.reset_index(drop=True)    # -> DataFrame (신호 충격일 수, 9)

    # 3) 분석 구간만 남긴다 (수익률 계산이 끝난 뒤여야 한다).
    sliced = data.slice_analysis(signal_shocks)  # -> DataFrame (분석 구간 충격일 수, 9)

    # 4) 사람이 읽을 퍼센트 컬럼 추가. -0.0503 보다 -5.03 이 눈에 잘 들어온다.
    table = sliced.copy()                              # -> DataFrame (충격일 수, 9)
    return_column = table["daily_return"]              # -> Series[float] (충격일 수,)
    return_percent = return_column * 100               # -> Series[float] (충격일 수,)
    table["daily_return_pct"] = return_percent.round(2)  # -> DataFrame (충격일 수, 10)

    return table


def print_shock_summary(table):
    """충격일 조사용 요약을 화면에 출력한다.

    이 표는 전부 신호용 티커 하나로 이미 걸러진 데이터라서, 여기서 하는
    정렬·집계는 티커 경계를 넘을 수 없다 (CLAUDE.md 코드 규약).
    """
    top_rows = config.TOP_SHOCK_ROWS  # -> int
    show_columns = ["date", "close", "daily_return_pct"]  # -> list[str] (3,)

    print()
    print("-" * 70)
    print(f"충격일 조사 요약 — {config.TICKERS['signal']}, "
          f"{config.ANALYSIS_START} 이후, ±{config.SHOCK_THRESHOLD * 100:.0f}% 기준")
    print("-" * 70)
    print(f"총 충격일: {len(table)}일")

    # --- 하락 충격 상위 N일 (수익률이 가장 작은 = 가장 많이 떨어진 날) ---
    ascending_table = table.sort_values("daily_return_pct")  # -> DataFrame (충격일 수, 10)
    worst_days = ascending_table.head(top_rows)              # -> DataFrame (N, 10)
    worst_view = worst_days[show_columns]                    # -> DataFrame (N, 3)

    print()
    print(f"[하락 충격 상위 {top_rows}일]")
    print(worst_view.to_string(index=False))

    # --- 상승 충격 상위 N일 ---
    descending_table = table.sort_values("daily_return_pct", ascending=False)  # -> DataFrame (충격일 수, 10)
    best_days = descending_table.head(top_rows)                                # -> DataFrame (N, 10)
    best_view = best_days[show_columns]                                        # -> DataFrame (N, 3)

    print()
    print(f"[상승 충격 상위 {top_rows}일]")
    print(best_view.to_string(index=False))

    # --- 연도별 충격일 개수 (오늘의 핵심 관찰 대상) ---
    # 하락과 상승을 나눠서 세야 "같은 해에 몰려 있는가"가 보인다.
    # 합계만 보면 폭락의 해와 급등의 해가 구분되지 않는다.
    counted = table.copy()                       # -> DataFrame (충격일 수, 10)
    date_column = counted["date"]                # -> Series[datetime64] (충격일 수,)
    date_accessor = date_column.dt               # -> DatetimeProperties
    counted["year"] = date_accessor.year         # -> DataFrame (충격일 수, 11)

    percent_column = counted["daily_return_pct"]  # -> Series[float] (충격일 수,)
    counted["direction"] = "상승"                  # -> DataFrame (충격일 수, 12)

    down_mask = percent_column < 0                        # -> Series[bool] (충격일 수,)
    counted.loc[down_mask, "direction"] = "하락"

    pivot = counted.pivot_table(
        index="year",
        columns="direction",
        values="date",
        aggfunc="count",
        fill_value=0,
    )  # -> DataFrame (충격일이 있던 연도 수, 1~2)

    print()
    print("[연도별 충격일 개수]")
    print(pivot.to_string())
    print()
    print("  큰 하락일과 큰 상승일이 같은 해에 몰려 있는지 볼 것.")
    print("  몰려 있다면, 폭락을 피해 빠져나온 전략은 급반등도 함께 놓친다는 뜻이다.")
    print("  -> reports/why_it_fails.md 의 근거 자료.")


def main():
    """수집 → 저장 → 감사 → 충격일 분석 → 검증 순으로 진행한다."""
    force_utf8_console()

    ticker_values = config.TICKERS.values()  # -> dict_values[str] (2,)
    tickers = list(ticker_values)            # -> list[str] (2,)

    print(f"수집 대상: {tickers}")
    print(f"수집 구간: {config.COLLECT_START} ~ {config.COLLECT_END or '오늘'}")
    print(f"auto_adjust: {config.AUTO_ADJUST}")
    print()

    # ------------------------------------------------------------------
    # 1. 수집
    # ------------------------------------------------------------------
    df, failures = data.download_ohlcv(tickers)  # -> (DataFrame (행 수, 8), dict)

    if len(failures) > 0:
        # 실패를 조용히 넘기지 않는다. 어떤 티커가 왜 실패했는지 먼저 보여준다.
        print()
        print("수집 실패 티커:")
        for ticker in failures:
            reason = failures[ticker]  # -> str
            print(f"  {ticker}: {reason}")

    if len(df) == 0:
        print()
        print("수집된 데이터가 없다. 네트워크와 티커명을 확인할 것.")
        return 1

    # ------------------------------------------------------------------
    # 2. 원본 스냅샷 저장
    # ------------------------------------------------------------------
    # 수정주가는 오늘 기준으로 매번 재계산되므로, 지금 받은 원본을 굳혀둔다.
    # 이후 분석은 이 파일만 읽어야 결과가 날마다 흔들리지 않는다.
    saved_path = data.save_parquet(df, config.RAW_OHLCV_PATH)  # -> Path

    # ------------------------------------------------------------------
    # 3. 감사
    # ------------------------------------------------------------------
    print()
    result = data.audit(df, verbose=True)  # -> dict

    report_path = data.save_audit_report(result, failures)  # -> Path

    # ------------------------------------------------------------------
    # 4. 충격일 목록 저장 + 조사용 요약 출력
    # ------------------------------------------------------------------
    shock_table = build_shock_table(df)  # -> DataFrame (충격일 수, 10)

    shock_path = config.SHOCK_DAYS_CSV_PATH  # -> Path
    shock_table.to_csv(shock_path, index=False, encoding="utf-8-sig")
    # encoding="utf-8-sig": 엑셀이 BOM 없는 UTF-8 CSV의 한글을 깨뜨린다.

    print_shock_summary(shock_table)

    # ------------------------------------------------------------------
    # 5. 반드시 통과해야 하는 검사
    # ------------------------------------------------------------------
    problems = check_audit_result(result)  # -> list[str]

    print()
    print(f"원본 스냅샷: {saved_path}")
    print(f"감사 리포트: {report_path}")
    print(f"충격일 목록: {shock_path} ({len(shock_table)}행)")
    print()

    if len(problems) > 0:
        print("감사 실패 — 다음 문제를 해결하기 전에는 다음 단계로 넘어가지 말 것:")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    print("감사 통과 — 중복 0, OHLC 위반 0, 연평균 행 수 정상.")
    return 0


if __name__ == "__main__":
    exit_code = main()  # -> int
    sys.exit(exit_code)
