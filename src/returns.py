"""로그수익률 계산과 시장 국면(market phase)별 분포 통계 전담 모듈.

이 파일에는 매매 신호나 기술적 지표(RSI, 볼린저밴드) 로직을 넣지 않는다.
여기서 하는 일은 "수익률이 국면마다 어떻게 생겼는지"를 계산해서 보여주는
것뿐이다 (CLAUDE.md: 로직은 src/, 실행·탐색은 notebooks/).

수집·감사는 src/data.py 전담이므로 이 파일에서 건드리지 않는다.

용어: 국면 = phase = regime
---------------------------
원본 정의는 config.MARKET_REGIMES 한 곳에만 있다. 이 모듈은 그것을 읽어
쓸 뿐 날짜를 다시 적지 않는다 (CLAUDE.md: 숫자를 다른 파일에 하드코딩하지
않는다). 날짜가 두 곳에 있으면 나중에 한쪽만 고쳐놓고 왜 결과가 안 바뀌는지
헤매게 된다.

국면 라벨은 사후적(post-hoc) 정보다
------------------------------------
tag_phase()가 붙이는 phase 컬럼은 이미 지나간 뒤에 사람이 나눈 구간이다.
CLAUDE.md 규칙 4: 이 라벨을 매매 진입/청산 조건 안에 쓰면 그 순간
미래 참조(look-ahead bias)가 된다. 2000년 1월에는 "지금이 닷컴 붕괴 국면"
이라는 사실을 알 방법이 없었기 때문이다. 여기서는 "이미 일어난 수익률
분포가 국면마다 어떻게 달랐나"를 묘사(descriptive)하는 용도로만 쓴다.

사용 예 (프로젝트 루트 기준)
----------------------------
    from src import config, data, returns

    df = data.load_parquet(config.RAW_OHLCV_PATH)  # -> DataFrame (18424, 8)
    df = returns.add_log_return(df)                 # -> DataFrame (18424, 9)
    stats = returns.phase_statistics(df)            # -> DataFrame (12, 10)
"""

import numpy as np
import pandas as pd

from src import config


def add_log_return(df, price_column="close"):
    """티커별 로그수익률 컬럼을 추가한다.

    정의: r_t = ln(P_t) - ln(P_{t-1})

    왜 반드시 groupby("ticker")를 거치는가 — 이 프로젝트 최대 함정
    ---------------------------------------------------------------
    이 데이터는 long format이라 티커가 세로로 쌓여 있다. (ticker, date)로
    정렬하면 경계가 이렇게 생긴다.

        date        ticker      close
        2026-07-31  ^GSPC     7489.72   <- ^GSPC 마지막 행
        1990-01-02  ^SP500TR   386.16   <- ^SP500TR 첫 행

    여기서 ticker를 무시하고 df["close"].shift(1)을 걸면, pandas는 티커
    경계를 모른 채 바로 윗행을 "전날"로 취급한다. 즉 ^SP500TR의 첫날
    수익률이 ln(386.16) - ln(7489.72) = -2.965 (단순수익률로 환산하면
    하루 만에 -94.8%)로 계산된다.
    이 하락은 **현실에 존재한 적이 없다.** 그런데도 숫자로는 멀쩡히 나오기
    때문에 눈으로 발견하기 어렵고, 이후의 변동성·왜도·첨도·MDD가 전부
    오염된다. groupby("ticker")를 거치면 각 티커의 첫 행이 NaN이 되어
    경계를 넘는 비교가 원천적으로 발생하지 않는다.

    왜 log()는 groupby 밖에서 거는가
    ---------------------------------
    log()는 각 행을 독립적으로 변환하는 원소별(element-wise) 연산이라 옆
    행을 보지 않는다. 티커 경계 문제가 생기는 연산은 shift/diff/pct_change
    처럼 "다른 행을 참조하는" 연산뿐이다. 그래서 log는 전체에 한 번 걸고,
    shift만 티커별로 한다.

    price_column을 매개변수로 둔 이유
    ----------------------------------
    지금은 지수(^GSPC, ^SP500TR)라 close와 adj_close가 동일하다
    (reports/day01_audit.txt [6]번: 괴리율 0.000000). 나중에 개별 종목으로
    확장하면 close는 배당락일마다 인위적으로 꺾이므로 그때는 adj_close를
    넘겨야 한다. 어떤 컬럼을 쓸지는 호출부가 정한다.

    Parameters
    ----------
    df : DataFrame
        long format (date, ticker, ...). price_column이 있어야 한다.
    price_column : str
        수익률을 계산할 가격 컬럼 이름.

    Returns
    -------
    DataFrame
        원본 + log_return 컬럼 (행 수, 원본 컬럼 수 + 1).
        각 티커의 첫 행은 비교할 전날이 없어 NaN이다. 이 NaN은 채우지도
        버리지도 않는다 — "그날의 수익률은 존재하지 않는다"가 사실이기 때문이다.
    """
    work = df.copy()  # -> DataFrame (행 수, 컬럼 수), 원본 보호

    # shift(1)은 행 순서를 그대로 믿는 연산이라 정렬이 선택이 아니라 필수다.
    # 정렬되지 않은 데이터에 shift를 걸면 조용히 틀린 값이 나온다.
    work = work.sort_values(["ticker", "date"])  # -> DataFrame (행 수, 컬럼 수)
    work = work.reset_index(drop=True)           # -> DataFrame (행 수, 컬럼 수)

    # 같은 (date, ticker)가 두 번 있으면 그날 수익률이 0으로 한 번 더 들어가
    # 변동성이 낮게 왜곡된다. data.audit()의 [2]번 검사와 같은 항목이지만,
    # 이 함수만 따로 불렀을 때도 막히도록 여기서 한 번 더 확인한다.
    duplicate_mask = work.duplicated(subset=["date", "ticker"])  # -> Series[bool] (행 수,)
    duplicate_count = int(duplicate_mask.sum())                  # -> int

    if duplicate_count > 0:
        raise ValueError(
            f"(date, ticker) 중복이 {duplicate_count}건 있다. "
            "shift(1)이 잘못된 전날을 가리키게 되므로 계산을 중단한다."
        )

    price_series = work[price_column]  # -> Series[float] (행 수,)
    log_price = np.log(price_series)   # -> Series[float] (행 수,), 원소별 연산 — 티커 경계 무관
    work["_log_price"] = log_price     # -> DataFrame (행 수, 컬럼 수 + 1), 중간 계산용

    grouped = work.groupby("ticker")             # -> DataFrameGroupBy
    log_price_by_ticker = grouped["_log_price"]  # -> SeriesGroupBy

    # LOOKAHEAD GUARD
    # shift(1)은 "한 칸 아래로 밀기" = 각 행이 자기보다 과거인 전날 값을 보게
    # 만드는 연산이다. 미래 방향(shift(-1))이 아니므로 t일 수익률 계산에
    # t+1일 정보가 섞이지 않는다. groupby("ticker") 안에서 호출하므로 티커
    # 경계를 넘지도 않는다 — 각 티커의 첫 행은 NaN이 된다.
    previous_log_price = log_price_by_ticker.shift(1)  # -> Series[float] (행 수,), 각 티커 첫 행 NaN

    current_log_price = work["_log_price"]                      # -> Series[float] (행 수,)
    work["log_return"] = current_log_price - previous_log_price  # -> DataFrame (행 수, 컬럼 수 + 2)

    work = work.drop(columns=["_log_price"])  # -> DataFrame (행 수, 컬럼 수 + 1)

    return work


def tag_phase(df, phases=None):
    """날짜를 기준으로 국면 라벨(phase 컬럼)을 붙인다.

    주의: 사후 라벨링 전용. 매매 로직에 이 컬럼을 쓰지 말 것
    (CLAUDE.md 규칙 4 — 국면 경계는 사후에 확정된 정보다).

    phases=None이 기본값인 이유 (가변 기본 인자 함정)
    ------------------------------------------------
    def tag_phase(df, phases=config.MARKET_REGIMES): 처럼 딕셔너리를 기본값
    으로 직접 쓰면, 그 딕셔너리는 함수가 정의되는 순간 딱 한 번만 만들어져
    모든 호출이 같은 객체를 공유한다. 누군가 실수로 그 안을 수정하면 이후
    모든 호출이 조용히 오염된다. None을 두고 함수 안에서 그때그때
    config.MARKET_REGIMES를 다시 읽는 것이 안전하다.

    Returns
    -------
    DataFrame
        원본 + phase 컬럼 (행 수, 원본 컬럼 수 + 1).
        어떤 국면에도 속하지 않는 날짜(2000년 이전 워밍업 구간)는 None.
    """
    if phases is None:
        phases = config.MARKET_REGIMES  # -> dict[str, dict] (6,)

    work = df.copy()      # -> DataFrame (행 수, 컬럼 수), 원본 보호
    work["phase"] = None  # -> DataFrame (행 수, 컬럼 수 + 1)

    date_column = work["date"]  # -> Series[datetime64] (행 수,)

    for phase_key in phases:
        phase_info = phases[phase_key]               # -> dict
        start_timestamp = pd.Timestamp(phase_info["start"])  # -> Timestamp
        end_value = phase_info["end"]                # -> str | None

        # config.MARKET_REGIMES는 닫힌구간 [start, end]로 정의돼 있다.
        # 국면끼리 겹치지 않도록 앞 국면 end 다음날이 뒤 국면 start다
        # (예: 닷컴 붕괴 ~2002-10-09, 회복·확장 2002-10-10~).
        after_start = date_column >= start_timestamp  # -> Series[bool] (행 수,)

        if end_value is None:
            # "최근 국면"처럼 아직 끝나지 않은 구간은 시작일 조건만 건다.
            in_phase = after_start  # -> Series[bool] (행 수,)
        else:
            end_timestamp = pd.Timestamp(end_value)     # -> Timestamp
            before_end = date_column <= end_timestamp   # -> Series[bool] (행 수,)
            in_phase = after_start & before_end         # -> Series[bool] (행 수,)

        work.loc[in_phase, "phase"] = phase_key

    return work


def phase_statistics(df, phases=None, verbose=True):
    """티커 x 국면별 로그수익률 분포 통계표를 만든다.

    df는 add_log_return()을 먼저 거친 결과여야 한다 (log_return 컬럼 필요).
    close 컬럼도 필요하다 — MDD는 로그수익률이 아니라 원가격 기준으로
    계산하기 때문이다 (아래 설명).

    계산하는 지표
    -------------
    - n_days         : 거래일 수
    - cum_return     : 누적수익률 = exp(로그수익률 합) - 1
                       로그수익률은 더할 수 있다는 성질 때문에 곱셈 없이
                       합만으로 누적수익률이 나온다. 이것이 로그수익률을
                       쓰는 가장 실용적인 이유다.
    - ann_log_return : 연율 로그수익률 = 일평균 로그수익률 x 252
                       **로그 공간의 값이다.** 이름에 log를 넣어둔 이유가
                       이것이다 — "연 14.7% 복리 성장"으로 읽으면 틀린다.
                       복리 기준 값이 필요하면 아래 cagr을 봐야 한다.
    - cagr           : 연복리 성장률. 누적 성장배수를 연 단위로 균등 분해한 값.
    - ann_vol        : 연율변동성 = 일간 표준편차 x sqrt(252)
                       252가 아니라 sqrt(252)를 곱하는 이유: 분산은 (독립이라는
                       가정 하에) 더해지므로 Var(연간) = 252 x Var(일간)이고,
                       표준편차는 그 제곱근이라 sqrt(252)가 곱해진다.
    - skew           : 왜도. 음수면 왼쪽(급락) 꼬리가 더 길다.
    - se_skew        : 왜도의 근사 표준오차 = sqrt(6 / n)
    - excess_kurt    : 초과첨도. 정규분포가 0. 양수면 극단값이 정규분포보다
                       자주 나온다 (fat tail).
    - se_kurt        : 초과첨도의 근사 표준오차 = sqrt(24 / n)
    - mdd            : 최대낙폭(Maximum Drawdown). 국면 시작 이후의 고점 대비
                       최대 하락률. 음수로 나온다.

    왜도·첨도에 표준오차를 같이 내는 이유
    --------------------------------------
    왜도와 첨도는 점추정치일 뿐이고, 3제곱·4제곱 연산이라 극단값 몇 개에
    크게 흔들린다. 표본이 작으면 "왜도 +0.15"가 그냥 잡음일 수 있다.
    se_skew/se_kurt는 **정규분포를 가정했을 때의 근사 표준오차**라서,
    추정치를 이 값으로 나누면 "몇 표준오차짜리 주장인가"를 어림할 수 있다.
    (엄밀한 검정은 아니다 — 애초에 수익률이 정규분포가 아니라는 것이 이
    표가 보여주는 결과이므로, 여기서는 크기 감각을 잡는 용도로만 쓴다.)

    252라는 숫자에 대하여
    ---------------------
    연율화와 CAGR 모두 config.TRADING_DAYS_PER_YEAR(=252)를 쓴다. 이는 미국
    증시의 관례값이고, D1 감사에서 실측한 값은 251.9행/년이었다
    (reports/day01_audit.txt [1]번). 차이가 0.04%라 결과에 실질적 영향은
    없지만, 실측값이 아니라 관례값을 쓰고 있다는 사실은 기록해둔다.

    왜 MDD만 close로 계산하는가
    ---------------------------
    MDD는 "고점 대비 얼마나 빠졌나"라서 실제 가격 수준을 봐야 한다.
    로그수익률에 cummax를 걸면 "가장 좋았던 하루" 대비 낙폭이 되어 전혀
    다른(의미 없는) 값이 나온다. 그래서 이 지표만 원가격 close를 쓴다.

    누적수익률과 국면 첫날에 대한 주의
    ----------------------------------
    각 국면의 첫 거래일 log_return은 "직전 국면 마지막 날 -> 이 국면 첫날"의
    수익률이다. 즉 국면 전환 당일의 움직임은 뒤쪽 국면에 귀속된다. 국면을
    빈틈없이 이어붙였을 때 전체 기간 누적수익률이 정확히 복원되려면 이렇게
    해야 한다 (노트북 검증 셀에서 실제로 확인한다).

    Parameters
    ----------
    df : DataFrame
        add_log_return()을 거친 long format DataFrame.
    phases : dict | None
        None이면 config.MARKET_REGIMES를 쓴다.
    verbose : bool
        True면 계산에서 제외한 행 수를 출력한다 (CLAUDE.md 규칙 3:
        버리되 무엇을 버렸는지 기록한다).

    Returns
    -------
    DataFrame
        (ticker, phase, n_days, cum_return, ann_log_return, cagr, ann_vol,
        skew, se_skew, excess_kurt, se_kurt, mdd, start, end).
        phase는 config에 정의된 연대순.
    """
    if phases is None:
        phases = config.MARKET_REGIMES  # -> dict[str, dict] (6,)

    work = tag_phase(df, phases=phases)          # -> DataFrame (행 수, 컬럼 수 + 1)
    work = work.sort_values(["ticker", "date"])  # -> DataFrame (행 수, 컬럼 수 + 1)
    work = work.reset_index(drop=True)           # -> DataFrame (행 수, 컬럼 수 + 1)

    # ------------------------------------------------------------------
    # 제외 대상 집계 — 보간하지 않고 버리되, 무엇을 버렸는지 남긴다
    # ------------------------------------------------------------------
    total_rows = len(work)  # -> int

    log_return_column = work["log_return"]      # -> Series[float] (행 수,)
    missing_return_mask = log_return_column.isna()  # -> Series[bool] (행 수,)
    missing_return_count = int(missing_return_mask.sum())  # -> int

    phase_column = work["phase"]                # -> Series[str | None] (행 수,)
    outside_phase_mask = phase_column.isna()    # -> Series[bool] (행 수,)
    outside_phase_count = int(outside_phase_mask.sum())  # -> int

    if verbose:
        print("[phase_statistics] 계산에서 제외한 행 (보간하지 않고 버림)")
        print(f"  전체 행 수                        : {total_rows:,}")
        print(f"  log_return이 NaN (각 티커 첫 거래일): {missing_return_count:,}")
        print(f"  국면 밖 날짜 (2000년 이전 워밍업)  : {outside_phase_count:,}")

    keep_mask = ~missing_return_mask & ~outside_phase_mask  # -> Series[bool] (행 수,)
    valid = work.loc[keep_mask]           # -> DataFrame (유효 행 수, 컬럼 수 + 1)
    valid = valid.reset_index(drop=True)  # -> DataFrame (유효 행 수, 컬럼 수 + 1)

    if verbose:
        print(f"  실제 계산에 쓴 행 수              : {len(valid):,}")
        print()

    # ------------------------------------------------------------------
    # 티커 x 국면 조합별로 하나씩 계산
    # ------------------------------------------------------------------
    # 이중 루프를 쓰는 이유: groupby().agg()에 함수를 여러 개 넘기면 한 줄에
    # 여러 연산이 압축되어 어느 단계에서 값이 틀어졌는지 보기 어렵다.
    # 한 조합씩 명시적으로 계산하면 중간값을 전부 눈으로 따라갈 수 있다.
    valid_ticker_column = valid["ticker"]        # -> Series[str] (유효 행 수,)
    unique_tickers = valid_ticker_column.unique()  # -> ndarray[str] (티커 수,)
    tickers = sorted(unique_tickers)             # -> list[str] (티커 수,)

    phase_keys = list(phases.keys())  # -> list[str] (6,), config.py에 연대순으로 정의됨

    trading_days = config.TRADING_DAYS_PER_YEAR  # -> int (252)
    trading_days_sqrt = np.sqrt(trading_days)    # -> numpy.float64

    stat_rows = []  # -> list[dict]

    for ticker in tickers:
        ticker_mask = valid_ticker_column == ticker  # -> Series[bool] (유효 행 수,)
        ticker_subset = valid.loc[ticker_mask]       # -> DataFrame (해당 티커 행 수, 컬럼 수 + 1)

        subset_phase_column = ticker_subset["phase"]  # -> Series[str] (해당 티커 행 수,)

        for phase_key in phase_keys:
            phase_mask = subset_phase_column == phase_key  # -> Series[bool] (해당 티커 행 수,)
            subset = ticker_subset.loc[phase_mask]         # -> DataFrame (국면 행 수, 컬럼 수 + 1)

            if len(subset) == 0:
                # 국면 정의가 데이터 범위 밖이면 빈 조합이 생길 수 있다.
                # 조용히 건너뛰지 않고 사람이 알 수 있게 알린다.
                if verbose:
                    print(f"  [주의] {ticker} x {phase_key}: 해당 구간에 데이터가 없다. 건너뛴다.")
                continue

            return_series = subset["log_return"]  # -> Series[float] (국면 행 수,)
            close_series = subset["close"]        # -> Series[float] (국면 행 수,)
            date_series = subset["date"]          # -> Series[datetime64] (국면 행 수,)

            n_days = len(subset)  # -> int

            # --- 누적수익률: 로그수익률은 더한 뒤 exp를 취하면 복원된다 ---
            return_sum = return_series.sum()          # -> numpy.float64
            total_growth = np.exp(return_sum)         # -> numpy.float64, 국면 전체 성장배수 (1.0 = 본전)
            cum_return = total_growth - 1             # -> numpy.float64

            # --- 연율 환산 (로그 공간) ---
            mean_daily = return_series.mean()            # -> numpy.float64
            std_daily = return_series.std()              # -> numpy.float64, 기본 ddof=1 (표본표준편차)
            ann_log_return = mean_daily * trading_days   # -> numpy.float64, 로그 공간 값
            ann_vol = std_daily * trading_days_sqrt      # -> numpy.float64

            # --- CAGR (복리 공간) ---
            # 정의 그대로 쓴다: 총 성장배수를 경과 연수로 균등 분해한다.
            # exp(mean_daily * 252) - 1 과 수학적으로 같은 값이지만, 이렇게
            # 쓰면 "총 성장배수의 연 단위 분해"라는 정의가 코드에 드러난다.
            n_years = n_days / trading_days              # -> float, 거래일 기준 경과 연수
            cagr = total_growth ** (1 / n_years) - 1     # -> numpy.float64

            # --- 분포 모양 ---
            skew_value = return_series.skew()   # -> numpy.float64
            excess_kurt = return_series.kurt()  # -> numpy.float64, pandas kurt()는 초과첨도(정규분포=0)

            # 정규분포 가정 하의 근사 표준오차. 추정치를 이 값으로 나누면
            # "몇 표준오차짜리 주장인가"가 나온다. n이 작을수록 커진다.
            se_skew = np.sqrt(6 / n_days)   # -> numpy.float64
            se_kurt = np.sqrt(24 / n_days)  # -> numpy.float64

            # --- MDD: 원가격 기준 (로그수익률이 아니라 close를 쓴다) ---
            running_max = close_series.cummax()            # -> Series[float] (국면 행 수,), 그 시점까지의 최고가
            drawdown = close_series / running_max - 1      # -> Series[float] (국면 행 수,), 0 이하
            mdd = drawdown.min()                           # -> numpy.float64, 가장 깊은 낙폭

            start_date = date_series.min()  # -> Timestamp
            end_date = date_series.max()    # -> Timestamp

            stat_rows.append(
                {
                    "ticker": ticker,
                    "phase": phase_key,
                    "n_days": n_days,
                    "cum_return": float(cum_return),
                    "ann_log_return": float(ann_log_return),
                    "cagr": float(cagr),
                    "ann_vol": float(ann_vol),
                    "skew": float(skew_value),
                    "se_skew": float(se_skew),
                    "excess_kurt": float(excess_kurt),
                    "se_kurt": float(se_kurt),
                    "mdd": float(mdd),
                    "start": start_date,
                    "end": end_date,
                }
            )

    stats = pd.DataFrame(stat_rows)  # -> DataFrame (조합 수, 14)

    # config.MARKET_REGIMES에 정의된 연대순으로 정렬한다.
    # 문자열 정렬을 그대로 두면 covid_crash가 dotcom_crash보다 앞에 오는 식으로
    # 시간 순서가 뒤섞여 표를 읽기 어려워진다.
    stats_phase_column = stats["phase"]  # -> Series[str] (조합 수,)
    ordered_phase = pd.Categorical(
        stats_phase_column,
        categories=phase_keys,
        ordered=True,
    )  # -> Categorical (조합 수,)
    stats["phase"] = ordered_phase

    stats = stats.sort_values(["ticker", "phase"])  # -> DataFrame (조합 수, 14)
    stats = stats.reset_index(drop=True)            # -> DataFrame (조합 수, 14)

    return stats


def to_markdown_table(stats, phases=None):
    """통계표를 마크다운 표 문자열로 바꾼다 (리포트에 붙여넣기용).

    pandas의 .to_markdown()은 tabulate 패키지를 추가로 요구하는데,
    표 하나 만들자고 의존성을 늘릴 이유가 없어서 직접 조립한다.

    국면 키(dotcom_crash)를 한국어 이름(닷컴 붕괴)으로 바꿔서 출력한다.
    한국어 이름도 config.MARKET_REGIMES 안에 이미 있다.

    왜도·초과첨도 칸에는 괄호로 "몇 SE짜리 추정치인가"를 병기한다.
    이 숫자가 없으면 표를 읽는 사람이 +0.15와 -0.69를 같은 신뢰도로
    비교하게 되는데, 표본 수가 다르면 그렇지 않다.

    표 아래에 가정을 캡션으로 붙인다. 리포트에 표만 잘라 붙여도 가정이
    같이 따라가게 하려는 것이다.

    Returns
    -------
    str
        마크다운 표 + 캡션 문자열.
    """
    if phases is None:
        phases = config.MARKET_REGIMES  # -> dict[str, dict] (6,)

    header_cells = [
        "티커", "국면", "거래일", "누적수익률", "연율로그수익률", "CAGR",
        "연율변동성", "왜도", "초과첨도", "MDD",
    ]  # -> list[str] (10,)

    lines = []  # -> list[str]

    header_line = "| " + " | ".join(header_cells) + " |"  # -> str
    lines.append(header_line)

    separator_cells = ["---"] * len(header_cells)           # -> list[str] (10,)
    separator_line = "|" + "|".join(separator_cells) + "|"  # -> str
    lines.append(separator_line)

    for row_index in stats.index:
        phase_key = stats.loc[row_index, "phase"]  # -> str
        phase_info = phases[phase_key]             # -> dict
        phase_name = phase_info["name"]            # -> str, 한국어 이름

        ticker = stats.loc[row_index, "ticker"]                      # -> str
        n_days = stats.loc[row_index, "n_days"]                      # -> int
        cum_return = stats.loc[row_index, "cum_return"]              # -> float
        ann_log_return = stats.loc[row_index, "ann_log_return"]      # -> float
        cagr = stats.loc[row_index, "cagr"]                          # -> float
        ann_vol = stats.loc[row_index, "ann_vol"]                    # -> float
        skew_value = stats.loc[row_index, "skew"]                    # -> float
        se_skew = stats.loc[row_index, "se_skew"]                    # -> float
        excess_kurt = stats.loc[row_index, "excess_kurt"]            # -> float
        se_kurt = stats.loc[row_index, "se_kurt"]                    # -> float
        mdd = stats.loc[row_index, "mdd"]                            # -> float

        # 추정치가 표준오차의 몇 배인가. 절댓값을 쓰는 이유는 방향이 아니라
        # "0에서 얼마나 떨어져 있나"를 보려는 것이기 때문이다.
        skew_se_multiple = abs(skew_value) / se_skew    # -> float
        kurt_se_multiple = abs(excess_kurt) / se_kurt   # -> float

        row_cells = [
            str(ticker),
            phase_name,
            f"{n_days:,}",
            f"{cum_return * 100:+.1f}%",
            f"{ann_log_return * 100:+.1f}%",
            f"{cagr * 100:+.1f}%",
            f"{ann_vol * 100:.1f}%",
            f"{skew_value:+.2f} ({skew_se_multiple:.1f} SE)",
            f"{excess_kurt:.2f} ({kurt_se_multiple:.1f} SE)",
            f"{mdd * 100:.1f}%",
        ]  # -> list[str] (10,)

        row_line = "| " + " | ".join(row_cells) + " |"  # -> str
        lines.append(row_line)

    trading_days = config.TRADING_DAYS_PER_YEAR  # -> int (252)

    caption_lines = [
        "",
        f"> **가정과 읽는 법**",
        f"> - 연 거래일 **{trading_days}일 가정**. D1 감사 실측은 251.9행/년이었으므로"
        f" 관례값을 쓰고 있다는 뜻이다 (`reports/day01_audit.txt` [1]번).",
        "> - **연율로그수익률**은 로그 공간의 값이라 복리 수익률이 아니다."
        " 복리 기준으로 읽어야 하면 **CAGR** 칸을 봐야 한다.",
        "> - 괄호 안 `SE`는 추정치를 정규분포 가정 하 근사 표준오차로 나눈 값이다"
        " (`se_skew = √(6/n)`, `se_kurt = √(24/n)`). 값이 작을수록 표본 잡음과"
        " 구분하기 어렵다는 뜻이다.",
        "> - **MDD만 원가격(`close`) 기준**이다. 나머지는 전부 로그수익률 기준.",
    ]  # -> list[str] (6,)

    all_lines = lines + caption_lines  # -> list[str]
    table = "\n".join(all_lines)       # -> str

    return table
