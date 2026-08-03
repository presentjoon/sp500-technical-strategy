"""데이터 수집 → 저장 → 감사(audit) 전담 모듈.

이 파일에는 분석이나 시각화 로직을 넣지 않는다 (CLAUDE.md: 로직은 src/,
실행·탐색은 notebooks/). 노트북은 여기 있는 함수를 호출해서 "믿을 수 있는
데이터프레임"을 얻는 것까지만 하고, 그 뒤의 지표 계산과 차트는 다른 모듈이
담당한다.

설계 원칙 세 가지
-----------------
1. 모든 함수는 티커가 여러 개일 수 있다고 가정한다.
   지수(^GSPC 하나)는 "티커가 1개인 특수 케이스"일 뿐이다. 지금 편하다고
   단일 티커 전용으로 짜두면, 나중에 S&P 500 개별 종목 500개로 확장할 때
   함수 시그니처와 반환 구조가 전부 바뀌고 노트북도 같이 깨진다.
   그래서 처음부터 tickers는 리스트로 받고, 시계열 연산은 반드시
   groupby("ticker") 또는 티커별 루프 안에서만 수행한다.

2. 데이터는 long format으로 저장한다.
   (date, ticker, open, high, low, close, adj_close, volume)
   가로로 넓은 wide format(티커마다 컬럼이 생기는 형태)은 티커가 늘어날 때마다
   컬럼이 늘어나서 스키마가 계속 변한다. long format은 티커가 몇 개든 스키마가
   그대로다. 나중에 뉴스 데이터를 (date, ticker, headline)으로 붙일 때
   조인 키 (date, ticker)가 그대로 맞아떨어지는 것도 같은 이유다.

3. 결측치를 절대 보간하지 않는다.
   보간(interpolate/ffill)은 "존재한 적 없는 가격"을 만들어내는 행위다.
   그렇게 만들어진 가격에서 나온 신호는 사실이 아니고, 그 신호로 계산한
   백테스트 수익률도 사실이 아니다. 결측은 버리거나 그대로 두고,
   "무엇을 얼마나 버렸는지"를 logs/ 와 reports/ 양쪽에 기록한다.
   판단은 사람이 한다.

사용 예 (프로젝트 루트에서 실행할 것)
------------------------------------
    from src import config
    from src import data

    tickers = list(config.TICKERS.values())     # -> list[str] (2,)
    df, failures = data.download_ohlcv(tickers) # -> (DataFrame, dict)
    result = data.audit(df)                     # -> dict
    data.save_audit_report(result, failures)    # -> Path
"""

import logging
from pathlib import Path

import pandas as pd
import yfinance as yf

from src import config


# ---------------------------------------------------------------------------
# 로거 설정 — 무엇을 버렸는지 기록하기 위한 통로 (설계 원칙 3)
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)  # -> logging.Logger


def setup_logger():
    """콘솔과 logs/data.log 양쪽에 기록하는 로거를 준비한다.

    노트북에서 모듈을 여러 번 import해도 핸들러가 중복으로 붙지 않도록 방어한다.

    Returns
    -------
    logging.Logger
    """
    if logger.handlers:  # 이미 설정됨 → 같은 줄이 두 번 찍히는 것 방지
        return logger

    logger.setLevel(logging.INFO)

    log_path = config.LOGS / "data.log"                          # -> Path
    file_handler = logging.FileHandler(log_path, encoding="utf-8")  # -> FileHandler
    stream_handler = logging.StreamHandler()                     # -> StreamHandler

    log_format = "%(asctime)s [%(levelname)s] %(message)s"  # -> str
    formatter = logging.Formatter(log_format)              # -> logging.Formatter

    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


setup_logger()


# ---------------------------------------------------------------------------
# 컬럼명 통일 규칙
# ---------------------------------------------------------------------------
# yfinance는 "Adj Close"처럼 공백과 대문자가 섞인 컬럼명을 준다.
# 대소문자/공백이 섞이면 df["Adj Close"]와 df["adj_close"]를 헷갈려서 KeyError가
# 나기 쉬우므로, 수집 직후 한 번만 소문자 스네이크케이스로 강제 통일한다.
COLUMN_RENAME = {
    "Date": "date",
    "Datetime": "date",
    "index": "date",
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Adj Close": "adj_close",
    "Adj_Close": "adj_close",
    "Volume": "volume",
}  # -> dict[str, str] (10,)

# long format의 표준 컬럼 순서. 저장/로드 시 항상 이 순서를 유지한다.
LONG_COLUMNS = [
    "date",
    "ticker",
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
]  # -> list[str] (8,)

# OHLC 정합성 검사와 0 이하 가격 검사에 쓰는 가격 컬럼 목록
PRICE_COLUMNS = ["open", "high", "low", "close"]  # -> list[str] (4,)


# ---------------------------------------------------------------------------
# 1. 수집
# ---------------------------------------------------------------------------
def download_ohlcv(
    tickers,
    start=config.COLLECT_START,
    end=config.COLLECT_END,
    auto_adjust=config.AUTO_ADJUST,
):
    """티커 목록을 하나씩 개별 다운로드해서 long format DataFrame으로 합친다.

    왜 한 번에 넘기지 않고 루프를 도는가
    ------------------------------------
    yf.download(["A", "B", "C"])처럼 여러 티커를 한 번에 넘기면, 그중 하나가
    실패해도 함수는 예외 없이 "성공"으로 돌아온다. 실패한 티커의 컬럼이
    통째로 NaN이 되거나 아예 빠진 채로 나오는데, 전체 shape만 보면 정상처럼
    보여서 알아채기 어렵다.
    지금은 티커가 2개뿐이라 눈으로 확인할 수 있지만, 개별 종목 500개를 받을 때
    "몇 개가 왜 실패했는지"는 reports/data_coverage.md(생존 편향 문서)의
    원재료가 된다.
    그래서 티커마다 따로 받고, 실패는 조용히 넘기지 않고 별도 딕셔너리에 담는다.

    Parameters
    ----------
    tickers : list[str]
        수집할 티커 목록. 문자열 하나를 넘겨도 리스트로 감싸서 처리한다.
    start, end : str | None
        수집 시작/종료일. end가 None이면 오늘까지.
    auto_adjust : bool
        False면 원본 가격과 adj_close를 둘 다 보존한다 (config.AUTO_ADJUST 참조).

    Returns
    -------
    (DataFrame, dict)
        DataFrame : long format (총 행 수, 8), 컬럼은 LONG_COLUMNS 순서
        dict      : {티커: 실패 사유 문자열}. 전부 성공하면 빈 딕셔너리.
    """
    if isinstance(tickers, str):  # 문자열 하나만 들어온 경우 방어
        tickers = [tickers]       # -> list[str] (1,)

    frames = []    # -> list[DataFrame], 티커별 결과를 모았다가 마지막에 concat
    failures = {}  # -> dict[str, str], {티커: 실패 사유}

    for ticker in tickers:
        logger.info("다운로드 시작: %s", ticker)

        try:
            raw = yf.download(
                ticker,
                start=start,
                end=end,
                auto_adjust=auto_adjust,  # 라이브러리 기본값에 의존하지 않는다
                progress=False,
            )  # -> DataFrame (거래일 수, 5~6) | 실패해도 예외 없이 빈 DataFrame 가능
        except Exception as error:  # 네트워크 오류, 티커명 오류 등
            error_name = type(error).__name__                            # -> str
            failures[ticker] = f"다운로드 예외: {error_name}: {error}"    # -> str
            logger.warning("다운로드 실패: %s — %s", ticker, failures[ticker])
            continue

        if raw is None or len(raw) == 0:
            # 예외 없이 빈 DataFrame이 오는 것이 실제로 가장 흔한 실패 형태다.
            failures[ticker] = "빈 응답 (티커명 오류 또는 해당 기간 데이터 없음)"
            logger.warning("다운로드 실패: %s — %s", ticker, failures[ticker])
            continue

        frame = _normalize_frame(raw, ticker)  # -> DataFrame (거래일 수, 8) | None

        if frame is None:
            failures[ticker] = "필수 컬럼(open/high/low/close) 누락"
            logger.warning("정규화 실패: %s — %s", ticker, failures[ticker])
            continue

        frames.append(frame)
        logger.info("다운로드 성공: %s — %d행", ticker, len(frame))

    if len(frames) == 0:
        # 전부 실패한 경우에도 "스키마가 있는" 빈 DataFrame을 돌려줘야
        # 뒤쪽 함수들이 KeyError로 죽지 않는다.
        empty = pd.DataFrame(columns=LONG_COLUMNS)  # -> DataFrame (0, 8)
        logger.warning("수집 완료: 성공 0개 / 실패 %d개", len(failures))
        return empty, failures

    combined = pd.concat(frames, ignore_index=True)      # -> DataFrame (총 행 수, 8)
    combined = combined.sort_values(["ticker", "date"])  # -> DataFrame (총 행 수, 8)
    combined = combined.reset_index(drop=True)           # -> DataFrame (총 행 수, 8)

    logger.info(
        "수집 완료: 티커 %d개 성공 / %d개 실패, 총 %d행",
        len(frames),
        len(failures),
        len(combined),
    )

    return combined, failures


def _normalize_frame(raw, ticker):
    """yfinance 원본 DataFrame 하나를 long format 한 조각으로 정규화한다.

    이 함수는 항상 티커 1개짜리 데이터만 받는다. 따라서 여기서 하는 연산은
    티커 경계를 넘을 수 없다 (설계 원칙 1). 노트북에서 직접 부르는 함수가
    아니라 download_ohlcv 내부 전용이라 이름 앞에 밑줄을 붙였다.

    Returns
    -------
    DataFrame | None
        정규화된 (행 수, 8) DataFrame. 필수 컬럼이 없으면 None.
    """
    frame = raw.copy()  # -> DataFrame (행 수, 5~6), 원본을 건드리지 않기 위해 복사

    # yfinance는 버전/옵션에 따라 컬럼이 MultiIndex((필드, 티커))로 오기도 한다.
    # 이 경우 첫 번째 레벨(필드명)만 남기고 티커 레벨은 버린다.
    # 티커 정보는 어차피 아래에서 ticker 컬럼으로 따로 넣는다.
    column_index = frame.columns  # -> Index | MultiIndex (컬럼 수,)

    if isinstance(column_index, pd.MultiIndex):
        flat_columns = column_index.get_level_values(0)  # -> Index (컬럼 수,), name="Price"
        frame.columns = flat_columns

    # 평탄화한 Index에는 원본 레벨 이름("Price")이 남아 있다. 이걸 그대로 두면
    # print나 to_string 출력에서 컬럼 헤더 위에 "Price"라는 유령 줄이 하나 더
    # 찍히고, parquet 메타데이터에도 들어간다. 데이터가 아니므로 지운다.
    frame.columns.name = None

    frame = frame.reset_index()                  # -> DataFrame (행 수, 6~7), 날짜를 컬럼으로
    frame = frame.rename(columns=COLUMN_RENAME)  # -> DataFrame (행 수, 6~7), 컬럼명 소문자화

    # 필수 컬럼 확인. 하나라도 없으면 이 티커는 쓸 수 없는 데이터다.
    required_columns = ["date"] + PRICE_COLUMNS  # -> list[str] (5,)

    for required in required_columns:
        if required not in frame.columns:
            return None

    # ------------------------------------------------------------------
    # 타임존 제거
    # ------------------------------------------------------------------
    # 타임존이 붙은 날짜(tz-aware)와 안 붙은 날짜(tz-naive)를 섞어서 merge하면
    # pandas가 TypeError를 내거나, 더 나쁘게는 같은 날짜인데 시각이 달라서
    # 조인이 조용히 0건으로 빠진다. 나중에 뉴스 데이터를 (date, ticker)로
    # 붙일 때 실제로 사고가 나는 지점이라, 수집 단계에서 미리 잘라둔다.
    date_series = frame["date"]                              # -> Series[datetime64] (행 수,)
    date_dtype = date_series.dtype                           # -> dtype
    timezone_info = getattr(date_dtype, "tz", None)          # -> tzinfo | None
    has_timezone = timezone_info is not None                 # -> bool

    if has_timezone:
        date_accessor = date_series.dt                       # -> DatetimeProperties
        date_series = date_accessor.tz_localize(None)        # -> Series[datetime64] (행 수,)

    date_accessor = date_series.dt                           # -> DatetimeProperties
    date_series = date_accessor.normalize()                  # -> Series[datetime64] (행 수,), 시각 00:00

    # dtype을 정규 단위로 고정한다. yfinance 버전에 따라 datetime64[ns]로도
    # datetime64[s]로도 오는데, 그 차이가 parquet 왕복 비교에서 잡음이 된다.
    date_series = date_series.astype(config.DATE_DTYPE)      # -> Series[datetime64[s]] (행 수,)
    frame["date"] = date_series

    # ------------------------------------------------------------------
    # 티커 컬럼 추가 (long format의 핵심 — 설계 원칙 1, 2)
    # ------------------------------------------------------------------
    frame["ticker"] = ticker  # -> Series[str] (행 수,), 모든 행이 같은 값

    # auto_adjust=True로 받으면 adj_close 컬럼이 아예 없다.
    # 이 경우 close가 이미 수정주가이므로 같은 값을 복사해 스키마를 맞춘다.
    if "adj_close" not in frame.columns:
        frame["adj_close"] = frame["close"]  # -> Series[float] (행 수,)

    if "volume" not in frame.columns:
        frame["volume"] = pd.NA  # -> Series[object] (행 수,)

    frame = frame[LONG_COLUMNS]  # -> DataFrame (행 수, 8), 컬럼 순서 고정

    # ------------------------------------------------------------------
    # OHLC가 전부 비어 있는 행 제거 (설계 원칙 3: 보간하지 않고 버리고 기록)
    # ------------------------------------------------------------------
    # yfinance가 가끔 값이 없는 날짜 행을 채워 넣는다. 이건 "가격이 없는 날"이지
    # "가격이 0인 날"이 아니므로 보간 대상이 아니라 삭제 대상이다.
    before_rows = len(frame)                                    # -> int
    frame = frame.dropna(subset=PRICE_COLUMNS, how="all")       # -> DataFrame (남은 행 수, 8)
    after_rows = len(frame)                                     # -> int
    dropped_rows = before_rows - after_rows                     # -> int

    if dropped_rows > 0:
        logger.warning(
            "%s: OHLC가 전부 비어 있는 행 %d개 삭제 (보간하지 않음)",
            ticker,
            dropped_rows,
        )

    frame = frame.reset_index(drop=True)  # -> DataFrame (남은 행 수, 8)

    return frame


# ---------------------------------------------------------------------------
# 2. 저장 / 로드
# ---------------------------------------------------------------------------
def save_parquet(df, path):
    """long format DataFrame을 parquet으로 저장한다.

    CSV가 아니라 parquet을 쓰는 이유: CSV는 타입 정보를 잃어버려서 다시 읽을 때
    date가 문자열이 되거나 volume이 float가 된다. parquet은 dtype을 그대로
    보존하므로 "저장했다 읽으면 값이 미묘하게 달라지는" 재현성 사고가 없다.

    Returns
    -------
    Path
        실제로 저장된 경로.
    """
    target = Path(path)          # -> Path
    parent_dir = target.parent   # -> Path

    parent_dir.mkdir(parents=True, exist_ok=True)  # 부모 폴더 없으면 생성

    df.to_parquet(target, index=False)  # index=False — 인덱스는 정보가 아니다

    logger.info("저장 완료: %s (%d행)", target, len(df))

    return target


def load_parquet(path):
    """parquet을 읽어 (ticker, date) 순으로 정렬해서 돌려준다.

    정렬을 로드 시점에 고정하는 이유: 시계열 계산(pct_change, rolling)은
    행 순서에 전적으로 의존한다. 순서가 보장되지 않은 데이터에 rolling을 걸면
    조용히 틀린 값이 나온다. "읽으면 항상 정렬돼 있다"를 규칙으로 만들어둔다.

    date의 dtype도 여기서 다시 고정한다. pyarrow가 저장 과정에서 datetime 단위를
    바꾸기 때문에, 그냥 읽으면 방금 수집한 DataFrame과 dtype이 달라진다
    (config.DATE_DTYPE 주석 참조).

    Returns
    -------
    DataFrame
        long format (행 수, 8), (ticker, date) 오름차순, 인덱스 0..n-1
    """
    df = pd.read_parquet(path)               # -> DataFrame (행 수, 8)
    df = df.sort_values(["ticker", "date"])  # -> DataFrame (행 수, 8)
    df = df.reset_index(drop=True)           # -> DataFrame (행 수, 8)

    date_column = df["date"]                                # -> Series[datetime64[ms]] (행 수,)
    df["date"] = date_column.astype(config.DATE_DTYPE)      # -> Series[datetime64[s]] (행 수,)

    logger.info("로드 완료: %s (%d행)", path, len(df))

    return df


# ---------------------------------------------------------------------------
# 3. 분석 구간 자르기
# ---------------------------------------------------------------------------
def slice_analysis(df, start=config.ANALYSIS_START):
    """수집 구간(1990~) 중 분석 구간(2000~)만 잘라낸다.

    왜 수집 구간과 분석 구간이 다른가
    ---------------------------------
    200일 이동평균 같은 지표는 앞의 200거래일이 있어야 첫 값이 나온다.
    분석 시작일인 2000-01-01부터 데이터를 받으면 2000년 한 해는 지표가 NaN이거나
    표본이 모자란 상태로 계산되어, 그 구간의 신호를 믿을 수 없다.
    그래서 1990년부터 받아 지표를 먼저 다 계산해두고(= 워밍업),
    성과 측정과 통계 검정은 지표가 완전히 익은 2000년 이후 구간에서만 한다.

    중요: 이 함수는 "지표 계산이 끝난 뒤"에 부르는 것이다.
    먼저 자르고 나서 지표를 계산하면 워밍업 구간을 확보한 의미가 사라진다.

    이 함수는 날짜 필터링만 하므로 티커 경계를 넘는 연산이 없다 (설계 원칙 1).

    Returns
    -------
    DataFrame
        start 이후 행만 남긴 long format DataFrame (분석 구간 행 수, 컬럼 수)
    """
    work = df.copy()  # -> DataFrame (행 수, 컬럼 수), 원본 보호

    start_timestamp = pd.Timestamp(start)  # -> Timestamp
    date_column = work["date"]             # -> Series[datetime64] (행 수,)
    mask = date_column >= start_timestamp  # -> Series[bool] (행 수,)

    sliced = work.loc[mask]                  # -> DataFrame (분석 구간 행 수, 컬럼 수)
    sliced = sliced.reset_index(drop=True)   # -> DataFrame (분석 구간 행 수, 컬럼 수)

    dropped_rows = len(work) - len(sliced)  # -> int

    logger.info(
        "분석 구간 절단: %s 이전 %d행 제외, %d행 남음",
        start,
        dropped_rows,
        len(sliced),
    )

    return sliced


# ---------------------------------------------------------------------------
# 4. 감사 (audit)
# ---------------------------------------------------------------------------
def audit(df, shock_threshold=config.SHOCK_THRESHOLD, verbose=True):
    """데이터를 믿기 전에 통과시켜야 하는 검사 목록.

    이 함수는 데이터를 고치지 않는다. 무엇이 이상한지 알려주기만 한다.
    고칠지 말지는 사람이 결과를 보고 판단한다 (설계 원칙 3).

    티커별 통계는 전부 "티커 하나로 필터링한 뒤" 계산한다. 따라서 티커 경계를
    넘는 연산이 발생하지 않는다 (설계 원칙 1).

    Returns
    -------
    dict
        각 검사 결과와, 사람이 읽을 수 있는 텍스트 리포트("report" 키).
    """
    work = df.copy()                          # -> DataFrame (행 수, 8), 원본 보호
    work = work.sort_values(["ticker", "date"])  # -> DataFrame (행 수, 8)
    work = work.reset_index(drop=True)           # -> DataFrame (행 수, 8)

    result = {}        # -> dict, 검사 결과를 담을 곳
    report_lines = []  # -> list[str], 사람이 읽을 리포트를 줄 단위로 모은다

    separator = "=" * 70  # -> str

    report_lines.append(separator)
    report_lines.append("데이터 감사 리포트")
    report_lines.append(separator)
    report_lines.append(f"전체 행 수: {len(work):,}")

    ticker_column = work["ticker"]             # -> Series[str] (행 수,)
    unique_tickers = ticker_column.unique()    # -> ndarray[str] (티커 수,)
    tickers = sorted(unique_tickers)           # -> list[str] (티커 수,)
    ticker_names = ", ".join(tickers)          # -> str

    report_lines.append(f"티커 수: {len(tickers)} — {ticker_names}")
    report_lines.append("")

    # ------------------------------------------------------------------
    # (1) 티커별 기간 / 행 수 / 연평균 행 수
    # ------------------------------------------------------------------
    # 연평균 행 수는 config.TRADING_DAYS_PER_YEAR(252) 근처가 정상이다.
    # 이 값이 240보다 훨씬 작으면 중간에 데이터가 빠진 것이고,
    # 260을 넘으면 중복 행이 있다는 뜻이다.
    coverage_rows = []  # -> list[dict]

    for ticker in tickers:
        mask = ticker_column == ticker  # -> Series[bool] (행 수,)
        subset = work.loc[mask]         # -> DataFrame (해당 티커 행 수, 8)

        subset_dates = subset["date"]   # -> Series[datetime64] (해당 티커 행 수,)
        first_date = subset_dates.min()  # -> Timestamp
        last_date = subset_dates.max()   # -> Timestamp
        row_count = len(subset)          # -> int

        span = last_date - first_date                          # -> Timedelta
        span_days = span.days                                  # -> int
        span_years = span_days / config.CALENDAR_DAYS_PER_YEAR  # -> float

        if span_years > 0:
            rows_per_year = row_count / span_years  # -> float
        else:
            rows_per_year = float("nan")            # -> float

        rounded_rows_per_year = round(rows_per_year, 1)  # -> float

        coverage_rows.append(
            {
                "ticker": ticker,
                "start": first_date,
                "end": last_date,
                "rows": row_count,
                "rows_per_year": rounded_rows_per_year,
            }
        )

    coverage = pd.DataFrame(coverage_rows)  # -> DataFrame (티커 수, 5)
    result["coverage"] = coverage

    coverage_text = coverage.to_string(index=False)  # -> str
    expected_rows = config.TRADING_DAYS_PER_YEAR     # -> int

    report_lines.append(f"[1] 티커별 커버리지 (rows_per_year는 {expected_rows} 근처가 정상)")
    report_lines.append(coverage_text)
    report_lines.append("")

    # ------------------------------------------------------------------
    # (2) (date, ticker) 중복
    # ------------------------------------------------------------------
    # 반드시 0이어야 한다. 같은 날 같은 티커가 두 번 있으면 그 날의 수익률이
    # 0으로 한 번 더 들어가서 변동성이 낮게, 샤프비율이 높게 왜곡된다.
    duplicate_mask = work.duplicated(subset=["date", "ticker"])  # -> Series[bool] (행 수,)
    duplicate_sum = duplicate_mask.sum()                         # -> numpy.int64
    duplicate_count = int(duplicate_sum)                         # -> int
    result["duplicate_count"] = duplicate_count

    if duplicate_count == 0:
        duplicate_note = "OK"  # -> str
    else:
        duplicate_note = "!! 반드시 0이어야 함 — 수집 로직 확인 필요"  # -> str

    report_lines.append(f"[2] (date, ticker) 중복: {duplicate_count} — {duplicate_note}")
    report_lines.append("")

    # ------------------------------------------------------------------
    # (3) 컬럼별 결측치
    # ------------------------------------------------------------------
    null_mask = work.isna()          # -> DataFrame (행 수, 8) bool
    null_counts = null_mask.sum()    # -> Series[int] (8,)
    missing = null_counts.to_dict()  # -> dict[str, int]
    result["missing"] = missing

    report_lines.append("[3] 컬럼별 결측치 (보간하지 않는다 — 기록만)")

    for column_name in LONG_COLUMNS:
        if column_name in missing:
            missing_count = missing[column_name]  # -> int
            report_lines.append(f"     {column_name:<12} {missing_count:,}")

    report_lines.append("")

    # ------------------------------------------------------------------
    # (4) 0 이하 가격
    # ------------------------------------------------------------------
    # 주가는 0 이하가 될 수 없다. 0이 있으면 수익률 계산에서 0나눗셈이 발생하고,
    # 로그수익률을 쓰면 -inf가 나와서 이후 통계가 전부 오염된다.
    nonpositive = {}  # -> dict[str, int]

    for column_name in PRICE_COLUMNS:
        price_series = work[column_name]  # -> Series[float] (행 수,)
        bad_mask = price_series <= 0      # -> Series[bool] (행 수,)
        bad_sum = bad_mask.sum()          # -> numpy.int64
        bad_count = int(bad_sum)          # -> int
        nonpositive[column_name] = bad_count

    result["nonpositive_prices"] = nonpositive

    nonpositive_values = nonpositive.values()    # -> dict_values[int] (4,)
    total_nonpositive = sum(nonpositive_values)  # -> int

    report_lines.append(f"[4] 0 이하 가격: 총 {total_nonpositive}건 {nonpositive}")
    report_lines.append("")

    # ------------------------------------------------------------------
    # (5) OHLC 정합성
    # ------------------------------------------------------------------
    # high는 그날 open/close/low보다 크거나 같아야 하고,
    # low는 그날 open/close/high보다 작거나 같아야 한다.
    # 이게 깨졌다는 건 데이터 소스 자체가 잘못됐다는 뜻이라, 이 데이터로 계산한
    # 어떤 지표도 믿을 수 없다.
    high_series = work["high"]    # -> Series[float] (행 수,)
    low_series = work["low"]      # -> Series[float] (행 수,)
    open_series = work["open"]    # -> Series[float] (행 수,)
    close_series = work["close"]  # -> Series[float] (행 수,)

    high_below_low = high_series < low_series      # -> Series[bool] (행 수,)
    high_below_open = high_series < open_series    # -> Series[bool] (행 수,)
    high_below_close = high_series < close_series  # -> Series[bool] (행 수,)
    low_above_open = low_series > open_series      # -> Series[bool] (행 수,)
    low_above_close = low_series > close_series    # -> Series[bool] (행 수,)

    violation_mask = high_below_low | high_below_open   # -> Series[bool] (행 수,)
    violation_mask = violation_mask | high_below_close  # -> Series[bool] (행 수,)
    violation_mask = violation_mask | low_above_open    # -> Series[bool] (행 수,)
    violation_mask = violation_mask | low_above_close   # -> Series[bool] (행 수,)

    violation_sum = violation_mask.sum()     # -> numpy.int64
    violation_count = int(violation_sum)     # -> int
    violation_rows = work.loc[violation_mask]  # -> DataFrame (위반 행 수, 8)

    result["ohlc_violation_count"] = violation_count
    result["ohlc_violation_rows"] = violation_rows

    if violation_count == 0:
        ohlc_note = "OK"  # -> str
    else:
        ohlc_note = "!! 데이터 소스 오류 의심 — 해당 행을 직접 확인할 것"  # -> str

    report_lines.append(f"[5] OHLC 정합성 위반: {violation_count}건 — {ohlc_note}")
    report_lines.append("")

    # ------------------------------------------------------------------
    # (6) close vs adj_close 괴리
    # ------------------------------------------------------------------
    # 지수(^GSPC, ^SP500TR)는 배당/분할 조정 개념이 없으므로 이 값이 0에 가까워야
    # 정상이다. 0이 아니면 티커를 잘못 받았거나 auto_adjust 설정이 의도와 다르다.
    #
    # 반대로 개별 종목이면 이 값이 크게 나오는 것이 정상이다. close는 그날 실제로
    # 거래된 가격이고 adj_close는 그 이후의 배당 지급과 액면분할을 소급 반영한
    # 가격이라, 둘의 차이가 곧 "누적 배당 + 분할 효과"다. 오래된 종목일수록,
    # 배당을 많이 준 종목일수록 이 괴리는 커진다.
    adj_diff = {}  # -> dict[str, float]

    for ticker in tickers:
        mask = ticker_column == ticker  # -> Series[bool] (행 수,)
        subset = work.loc[mask]         # -> DataFrame (해당 티커 행 수, 8)

        close_values = subset["close"]    # -> Series[float] (해당 티커 행 수,)
        adj_values = subset["adj_close"]  # -> Series[float] (해당 티커 행 수,)

        difference = close_values - adj_values  # -> Series[float] (해당 티커 행 수,)
        difference = difference.abs()           # -> Series[float] (해당 티커 행 수,)
        ratio = difference / close_values       # -> Series[float] (해당 티커 행 수,)

        mean_ratio_raw = ratio.mean()                             # -> numpy.float64
        mean_ratio = float(mean_ratio_raw)                        # -> float
        adj_diff[ticker] = round(mean_ratio, config.AUDIT_DECIMALS)  # -> float

    result["adj_close_diff_ratio"] = adj_diff

    report_lines.append("[6] close vs adj_close 평균 괴리율 (지수는 0에 가까워야 정상)")

    decimals = config.AUDIT_DECIMALS  # -> int, 반올림과 표시 자릿수를 일치시킨다

    for ticker in tickers:
        diff_value = adj_diff[ticker]  # -> float
        report_lines.append(f"     {ticker:<12} {diff_value:.{decimals}f}")

    report_lines.append("")

    # ------------------------------------------------------------------
    # (7) 달력일 공백
    # ------------------------------------------------------------------
    # 주말(금→월)은 3일이므로 정상이다. config.MAX_CALENDAR_GAP_DAYS 이상
    # 벌어졌다는 건 연휴가 겹친 클러스터거나(예: 크리스마스 주간),
    # 데이터가 통째로 빠진 것이다. 후자라면 그 구간의 이동평균과 수익률이
    # 전부 왜곡되므로 반드시 확인해야 한다.
    #
    # diff()는 바로 윗행과 비교하는 연산이라 티커 경계를 넘으면 안 된다.
    # 아래 루프는 티커 하나로 필터링한 subset 안에서만 diff()를 부르므로 안전하다.
    gap_rows = []  # -> list[dict]

    for ticker in tickers:
        mask = ticker_column == ticker       # -> Series[bool] (행 수,)
        subset = work.loc[mask]              # -> DataFrame (해당 티커 행 수, 8)
        subset = subset.sort_values("date")  # -> DataFrame (해당 티커 행 수, 8)

        subset_dates = subset["date"]     # -> Series[datetime64] (해당 티커 행 수,)
        date_diff = subset_dates.diff()   # -> Series[timedelta64], 첫 행은 NaT
        diff_accessor = date_diff.dt      # -> TimedeltaProperties
        gap_days = diff_accessor.days     # -> Series[float], 첫 행은 NaN

        gap_mask = gap_days >= config.MAX_CALENDAR_GAP_DAYS  # -> Series[bool]
        gap_subset = subset.loc[gap_mask]                    # -> DataFrame (공백 수, 8)
        gap_lengths = gap_days.loc[gap_mask]                 # -> Series[float] (공백 수,)

        for row_index in gap_subset.index:
            gap_end_date = gap_subset.loc[row_index, "date"]  # -> Timestamp
            gap_length_raw = gap_lengths.loc[row_index]       # -> numpy.float64
            gap_length = int(gap_length_raw)                  # -> int

            gap_rows.append(
                {
                    "ticker": ticker,
                    "gap_end_date": gap_end_date,
                    "gap_days": gap_length,
                }
            )

    if len(gap_rows) > 0:
        gaps = pd.DataFrame(gap_rows)                       # -> DataFrame (공백 수, 3)
        gaps = gaps.sort_values("gap_days", ascending=False)  # -> DataFrame (공백 수, 3)
        gaps = gaps.reset_index(drop=True)                  # -> DataFrame (공백 수, 3)
    else:
        gap_columns = ["ticker", "gap_end_date", "gap_days"]  # -> list[str] (3,)
        gaps = pd.DataFrame(columns=gap_columns)             # -> DataFrame (0, 3)

    result["calendar_gaps"] = gaps

    gap_threshold = config.MAX_CALENDAR_GAP_DAYS  # -> int
    report_lines.append(f"[7] {gap_threshold}일 이상 달력일 공백: {len(gaps)}건")

    if len(gaps) > 0:
        preview = gaps.head(config.AUDIT_PREVIEW_ROWS)  # -> DataFrame (최대 10, 3)
        preview_text = preview.to_string(index=False)   # -> str
        report_lines.append(preview_text)
        report_lines.append("     (연휴 클러스터면 정상, 아니면 데이터 누락)")

    report_lines.append("")

    # ------------------------------------------------------------------
    # (8) 충격일 개수
    # ------------------------------------------------------------------
    shocks = shock_days(work, threshold=shock_threshold)  # -> DataFrame (충격일 수, 9)

    shock_counts = {}  # -> dict[str, int]

    for ticker in tickers:
        shock_ticker_column = shocks["ticker"]           # -> Series[str] (충격일 수,)
        ticker_shock_mask = shock_ticker_column == ticker  # -> Series[bool] (충격일 수,)
        ticker_shock_sum = ticker_shock_mask.sum()       # -> numpy.int64
        shock_counts[ticker] = int(ticker_shock_sum)     # -> int

    result["shock_day_counts"] = shock_counts
    result["shock_days"] = shocks

    threshold_percent = shock_threshold * 100  # -> float
    report_lines.append(f"[8] 일간 ±{threshold_percent:.1f}% 이상 충격일")

    for ticker in tickers:
        shock_count = shock_counts[ticker]  # -> int
        report_lines.append(f"     {ticker:<12} {shock_count}일")

    report_lines.append("")
    report_lines.append(separator)

    report_text = "\n".join(report_lines)  # -> str
    result["report"] = report_text

    if verbose:
        print(report_text)

    return result


def save_audit_report(result, failures=None, path=config.AUDIT_REPORT_PATH):
    """감사 결과를 reports/day01_audit.txt 로 저장한다.

    CLAUDE.md 규칙 3은 "버리고, 무엇을 버렸는지 reports/에 기록한다"고 규정한다.
    logs/data.log 는 실행 흔적이라 매 실행마다 쌓이지만, reports/ 는
    리포트에서 인용할 수 있는 산출물이다. 둘은 용도가 다르므로 양쪽에 남긴다.

    파일명 주의: reports/data_coverage.md 는 개별 종목 확장 시 수집 성공·실패와
    생존 편향을 다루는 "사람이 쓰는" 문서로 예약돼 있다. 이 함수의 출력은
    매 실행마다 덮어쓰는 자동 생성물이므로 이름이 겹치면 안 된다.
    확장자를 .md가 아니라 .txt로 둔 것도 같은 이유다 — 이건 문서가 아니라 로그다.

    Parameters
    ----------
    result : dict
        audit()이 돌려준 딕셔너리.
    failures : dict | None
        download_ohlcv()가 돌려준 {티커: 실패 사유}. 개별 종목 500개로 확장할 때
        이 목록이 data_coverage.md의 원재료가 된다.
    path : str | Path

    Returns
    -------
    Path
    """
    target = Path(path)         # -> Path
    parent_dir = target.parent  # -> Path

    parent_dir.mkdir(parents=True, exist_ok=True)

    lines = []  # -> list[str]

    lines.append("src/data.py의 audit()가 자동 생성한 파일이다. 직접 편집하지 말 것.")
    lines.append("매 실행마다 덮어쓴다.")
    lines.append("")
    lines.append("[0] 수집 실패 티커")

    if failures is None:
        lines.append("     실패 정보가 전달되지 않았다 (failures 인자 없이 호출됨).")
    elif len(failures) == 0:
        lines.append("     없음 — 요청한 티커를 모두 수집했다.")
    else:
        for ticker in failures:
            reason = failures[ticker]  # -> str
            lines.append(f"     {ticker:<12} {reason}")

    lines.append("")
    lines.append(result["report"])
    lines.append("")

    document = "\n".join(lines)  # -> str

    target.write_text(document, encoding="utf-8")

    logger.info("감사 리포트 저장: %s", target)

    return target


# ---------------------------------------------------------------------------
# 5. 충격일 추출
# ---------------------------------------------------------------------------
def shock_days(df, threshold=config.SHOCK_THRESHOLD):
    """일간 수익률 절댓값이 threshold 이상인 날만 추출한다.

    왜 반드시 groupby("ticker")로 계산하는가 — long format 최다 빈발 버그
    -------------------------------------------------------------------
    long format은 티커가 세로로 쌓여 있다. 예를 들어 정렬 후 이런 순서가 된다.

        date        ticker    close
        2024-12-30  ^GSPC     5900     <- ^GSPC 마지막 행
        1990-01-02  ^SP500TR   350     <- ^SP500TR 첫 행

    여기서 df["close"].pct_change()를 그냥 걸면, pandas는 티커 경계를 모른 채
    바로 윗행과 비교한다. 즉 ^GSPC의 5900과 ^SP500TR의 350을 비교해서
    약 -94%라는 수익률을 만들어낸다. 이 하락은 **현실에 존재한 적이 없다.**
    그런데 -94%는 threshold를 가뿐히 넘으므로 충격일 목록에 그대로 섞여 들어가고,
    변동성·최대낙폭·샤프비율까지 전부 오염된다.

    groupby("ticker")를 거치면 각 티커의 첫 행 수익률이 NaN이 되어
    경계를 넘는 비교가 원천적으로 발생하지 않는다.

    Returns
    -------
    DataFrame
        충격일 행만 남긴 DataFrame (충격일 수, 9). daily_return 컬럼이 추가된다.
    """
    work = df.copy()  # -> DataFrame (행 수, 8), 원본 보호

    # 정렬은 선택이 아니라 필수다. pct_change는 행 순서를 그대로 믿는다.
    work = work.sort_values(["ticker", "date"])  # -> DataFrame (행 수, 8)
    work = work.reset_index(drop=True)           # -> DataFrame (행 수, 8)

    grouped = work.groupby("ticker")             # -> DataFrameGroupBy
    close_by_ticker = grouped["close"]           # -> SeriesGroupBy
    daily_return = close_by_ticker.pct_change()  # -> Series[float] (행 수,), 각 티커 첫 행 NaN

    work["daily_return"] = daily_return  # -> DataFrame (행 수, 9)

    return_column = work["daily_return"]        # -> Series[float] (행 수,)
    return_magnitude = return_column.abs()      # -> Series[float] (행 수,)
    shock_mask = return_magnitude >= threshold  # -> Series[bool] (행 수,), NaN은 False

    shocks = work.loc[shock_mask]                    # -> DataFrame (충격일 수, 9)
    shocks = shocks.sort_values(["ticker", "date"])  # -> DataFrame (충격일 수, 9)
    shocks = shocks.reset_index(drop=True)           # -> DataFrame (충격일 수, 9)

    return shocks
