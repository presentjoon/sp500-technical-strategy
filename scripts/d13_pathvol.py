"""D13 실행 — RSI<30 신호 후 사후 경로변동성 (사전등록 `docs/prereg_day13b.md`).

    python scripts/d13_pathvol.py

이 스크립트는 **순서를 지키는 것**이 목적이다. 계산 로직은 전부 `src/pathvol.py`
에 있고 여기서는 호출 순서, 검증, 출력만 담당한다.

    1. σ_post 계산 (§1.4)          — 원리 버전 / 벡터 버전 등가성
    2. 검증                        — 숫자를 보기 전에. 실패하면 즉시 중단
    3. log 변환 후 층별 통계 (§1.5, §2.2)
    4. Welch SE (§2.3)
    5. 순열검정 (§2.4)
    6. 민감도 — 클러스터 축약 (§2.5)
    7. 진단 (§4)
    8. W=10 보조 (§1.2)

해석 문장은 쓰지 않는다. §5는 사용자가 작성한다.
"""

import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()     # -> Path
SCRIPT_DIR = SCRIPT_PATH.parent            # -> Path
PROJECT_ROOT = SCRIPT_DIR.parent           # -> Path
ROOT_TEXT = str(PROJECT_ROOT)              # -> str
SCRIPTS_TEXT = str(SCRIPT_DIR)             # -> str

if ROOT_TEXT not in sys.path:
    sys.path.insert(0, ROOT_TEXT)

# `d13_counts`를 import하기 위해서다. scripts/는 패키지가 아니므로 경로를 넣는다.
if SCRIPTS_TEXT not in sys.path:
    sys.path.insert(0, SCRIPTS_TEXT)

import numpy as np   # noqa: E402
import pandas as pd  # noqa: E402

from src import config      # noqa: E402
from src import data        # noqa: E402
from src import pathvol     # noqa: E402
from src import stratified  # noqa: E402

# 2단계 축약은 사전등록 §2.5의 확정 수치(34 / 281)를 만든 코드를 그대로 쓴다.
import d13_counts  # noqa: E402


REPORTS = PROJECT_ROOT / "reports"  # -> Path

HAND_CHECK_COUNT = 3  # -> int, 손계산 대조할 이벤트 수


def force_utf8_console():
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)  # -> callable | None

        if reconfigure is not None:
            reconfigure(encoding="utf-8")


def banner(text):
    print()
    print("=" * 88)
    print(text)
    print("=" * 88)


def halt(message):
    print()
    print("!! 중단 —", message)
    return 1


# ---------------------------------------------------------------------------
# 1. σ_post 계산 (§1.4)
# ---------------------------------------------------------------------------
def step1_estimator(price, ticker):
    """원리 버전과 벡터 버전을 둘 다 만들고 등가성을 검사한다."""
    banner("1. σ_post 계산 (§1.4) — 창 lr_{t+1} .. lr_{t+20}")

    with_returns = pathvol.add_log_return(price)  # -> DataFrame (18424, 9)

    ticker_mask = with_returns["ticker"] == ticker  # -> Series[bool] (18424,)
    one = with_returns.loc[ticker_mask]             # -> DataFrame (9212, 9)
    one = one.sort_values("date")                   # -> DataFrame
    one = one.reset_index(drop=True)                # -> DataFrame (9212, 9)

    log_returns = one["log_return"].to_numpy()  # -> ndarray[float] (9212,)

    loop_values = pathvol.post_path_volatility_loop(log_returns)  # -> ndarray[float] (9212,)

    with_vector = pathvol.add_post_path_volatility(with_returns)  # -> DataFrame (18424, 10)

    vector_mask = with_vector["ticker"] == ticker  # -> Series[bool] (18424,)
    vector_one = with_vector.loc[vector_mask]      # -> DataFrame (9212, 10)
    vector_one = vector_one.sort_values("date")    # -> DataFrame
    vector_values = vector_one["vol_post_20"].to_numpy()  # -> ndarray[float] (9212,)

    comparison = pathvol.compare_volatility_versions(loop_values, vector_values)  # -> dict

    print(f"  NaN 위치 일치      : {comparison['nan_positions_match']}")
    print(f"  양쪽 유효 행 수    : {comparison['n_both_valid']}")
    print(f"  최대 절대차        : {comparison['max_abs_diff']:.3e}")
    print(f"  등가 (tol 1e-12)   : {comparison['equivalent']}")

    return one, comparison


# ---------------------------------------------------------------------------
# 2. 검증 (§1.3, §2.1) — 숫자를 보기 전에
# ---------------------------------------------------------------------------
def step2_handcheck(price_frame, frame, ticker):
    """임의 이벤트 3건의 창 인덱스와 값을 원자료에서 직접 다시 만든다."""
    banner("2-A. 손계산 대조 — 창 인덱스와 값")

    signal_rows = frame.loc[frame["is_signal"]]  # -> DataFrame (65, 8)

    n_signal = len(signal_rows)  # -> int

    picks = [0, n_signal // 2, n_signal - 1]  # -> list[int], 첫 / 중간 / 마지막

    closes = price_frame["close"].to_numpy()          # -> ndarray[float] (9212,)
    log_returns = price_frame["log_return"].to_numpy()  # -> ndarray[float] (9212,)
    dates = price_frame["date"].to_numpy()            # -> ndarray[datetime64] (9212,)

    all_pass = True  # -> bool

    for pick in picks:
        row = signal_rows.iloc[pick]  # -> Series

        position = int(row["pos"])  # -> int, ^GSPC 전체 거래일 안에서의 t

        window_positions = pathvol.d13_window_positions(position)  # -> list[int] (20,)

        # (i) 창 첫 원소가 lr_{t+1}인가
        first_is_t_plus_1 = (window_positions[0] == position + 1)  # -> bool

        # (ii) lr_t 가 창에 들어가지 않았는가
        excludes_t = position not in window_positions  # -> bool

        # (iii) 길이 20인가
        length_ok = len(window_positions) == 20  # -> bool

        # 원자료 종가에서 로그수익률을 직접 다시 만든다 (log_return 컬럼 미사용)
        manual = []  # -> list[float]

        for index in window_positions:
            manual.append(float(np.log(closes[index] / closes[index - 1])))

        manual_array = np.asarray(manual, dtype=float)  # -> ndarray[float] (20,)

        manual_std = float(np.std(manual_array, ddof=pathvol.STD_DDOF))  # -> float

        stored = float(row["vol_post_20"])  # -> float

        difference = abs(manual_std - stored)  # -> float
        value_ok = difference < 1e-12          # -> bool

        # 창 첫 값이 log_return 컬럼의 lr_{t+1}과 같은지도 본다
        first_matches = abs(manual_array[0] - log_returns[position + 1]) < 1e-12  # -> bool

        passed = first_is_t_plus_1 and excludes_t and length_ok and value_ok and first_matches

        if not passed:
            all_pass = False

        signal_date = pd.Timestamp(row["date"]).date()            # -> date
        window_start = pd.Timestamp(dates[window_positions[0]]).date()   # -> date
        window_end = pd.Timestamp(dates[window_positions[-1]]).date()    # -> date

        print()
        print(f"  이벤트 {pick + 1}/{n_signal} — 신호일 {signal_date} (pos {position}, Q{int(row['quintile'])})")
        print(f"    창 인덱스        : [{window_positions[0]} .. {window_positions[-1]}]  (길이 {len(window_positions)})")
        print(f"    창 날짜          : {window_start} ~ {window_end}")
        print(f"    첫 원소 == lr_t+1: {first_is_t_plus_1}  (lr_t+1 = {log_returns[position + 1]:+.8f})")
        print(f"    lr_t 미포함      : {excludes_t}  (lr_t = {log_returns[position]:+.8f})")
        print(f"    손계산 σ_post    : {manual_std:.12f}")
        print(f"    저장된 σ_post    : {stored:.12f}")
        print(f"    절대차           : {difference:.3e}   통과: {passed}")

    return all_pass


def step2_counts(frame):
    banner("2-B. 표본 수 대조 (§1.3, §2.1) — 확정값과 일치해야 한다")

    check = pathvol.d13_check_frame(frame)  # -> dict

    for row in check["rows"]:
        print(f"  {row['항목']:<14} 실측 {row['실측']}")
        print(f"  {'':<14} 기대 {row['기대']}   통과: {row['통과']}")

    return check["all_pass"]


# ---------------------------------------------------------------------------
# 6. 민감도
# ---------------------------------------------------------------------------
def run_test(frame, label):
    """층별 통계 → Welch SE → 순열검정 한 묶음."""
    table = pathvol.d13_stratum_table(frame)          # -> DataFrame (층 수, 15)
    summary = pathvol.d13_delta_and_se(table)         # -> dict
    permutation = pathvol.d13_permutation(frame, table)  # -> dict

    # 관측 Δ·SE가 두 경로에서 같은지 확인한다 (표 경로 vs 순열 경로)
    delta_gap = abs(summary["delta"] - permutation["delta"])  # -> float
    se_gap = abs(summary["se"] - permutation["se"])           # -> float

    if delta_gap > 1e-12 or se_gap > 1e-12:
        raise ValueError(
            f"[{label}] 관측 Δ/SE가 두 경로에서 다르다: "
            f"Δ차 {delta_gap:.3e}, SE차 {se_gap:.3e}"
        )

    return table, summary, permutation


def print_test(label, table, summary, permutation):
    banner(label)

    columns = ["quintile", "n_sig", "n_ctl", "w_k", "mean_log_post_sig",
               "mean_log_post_ctl", "delta_k", "exp_delta_k", "se_k",
               "delta_pre_k", "included"]  # -> list[str] (11,)

    print(table[columns].to_string(index=False))

    print()
    print(f"  포함 층            : {summary['included']}   (K = {len(summary['included'])})")
    print(f"  유효 신호 수       : {summary['n_sig']}")
    print(f"  Δ                  : {summary['delta']:+.6f}")
    print(f"  exp(Δ)             : {summary['exp_delta']:.6f}")
    print(f"  SE(Δ)              : {summary['se']:.6f}")
    print(f"  Δ / SE             : {summary['t_stud']:+.6f}")
    print(f"  Δ_pre              : {summary['delta_pre']:+.6f}")
    print(f"  p_raw              : {permutation['p_raw']:.6f}  (초과 {permutation['extreme_raw']})")
    print(f"  p_stud             : {permutation['p_stud']:.6f}  (초과 {permutation['extreme_stud']})")
    print(f"  B / seed           : {permutation['iterations']} / {permutation['seed']}")
    print(f"  층 크기 보존       : {permutation['sizes_preserved']}")


# ---------------------------------------------------------------------------
def main():
    force_utf8_console()

    ticker = config.TICKERS["signal"]  # -> str ("^GSPC")

    price = data.load_parquet(config.RAW_OHLCV_PATH)  # -> DataFrame (18424, 8)

    # ---------------------------------------------------------------- 1
    price_frame, comparison = step1_estimator(price, ticker)  # -> (DataFrame, dict)

    if not comparison["equivalent"]:
        return halt("원리 버전과 벡터 버전이 다르다.")

    # ---------------------------------------------------------------- 층 배정
    bins = stratified.build_bin_frame(price)  # -> DataFrame (9212, 5)

    matched, counted = stratified.check_bin_assignment(bins)  # -> (bool, dict)

    print()
    print("  층 배정 재현 (D8b 집계표 대조):", counted, "| 일치:", matched)

    if not matched:
        return halt("층 배정이 D8b와 다르다.")

    frame = pathvol.d13_build_frame(price, bins, truncation="d8b")  # -> DataFrame (6663, 8)

    # ---------------------------------------------------------------- 2
    hand_ok = step2_handcheck(price_frame, frame, ticker)  # -> bool

    if not hand_ok:
        return halt("손계산 대조 실패.")

    counts_ok = step2_counts(frame)  # -> bool

    if not counts_ok:
        return halt("표본 수가 사전등록 확정값과 다르다.")

    # ---------------------------------------------------------------- 3~5
    table, summary, permutation = run_test(frame, "주검정")  # -> (DataFrame, dict, dict)

    print_test("3~5. 주검정 (W=20) — 층별 통계 / Welch SE / 순열검정", table, summary, permutation)

    # ---------------------------------------------------------------- 6
    banner("6. 민감도 — 2단계 클러스터 축약 (§2.5)")

    declustered, decluster_counts = pathvol.d13_decluster(
        frame, d13_counts.greedy_thin, d13_counts.drop_near_anchors
    )  # -> (DataFrame, dict)

    print(f"  축약 후 신호       : {decluster_counts['n_anchor']}  "
          f"(사전등록 {pathvol.D13_EXPECTED_DECLUSTERED_SIGNALS})")
    print(f"  축약 후 대조군     : {decluster_counts['n_control']}  "
          f"(사전등록 {pathvol.D13_EXPECTED_DECLUSTERED_CONTROLS})")

    signals_ok = decluster_counts["n_anchor"] == pathvol.D13_EXPECTED_DECLUSTERED_SIGNALS   # -> bool
    controls_ok = decluster_counts["n_control"] == pathvol.D13_EXPECTED_DECLUSTERED_CONTROLS  # -> bool

    if not (signals_ok and controls_ok):
        return halt("축약 후 표본 수가 사전등록 확정값과 다르다.")

    sens_table, sens_summary, sens_permutation = run_test(declustered, "민감도")

    print_test("6-B. 민감도 결과 (축약 후)", sens_table, sens_summary, sens_permutation)

    # ---------------------------------------------------------------- 7
    banner("7. 진단 (§4)")

    gap_table = pathvol.d13_gap_ratio(frame)              # -> DataFrame (10, 6)
    gap_table_thin = pathvol.d13_gap_ratio(declustered)   # -> DataFrame (10, 6)

    n_signal_raw = int(frame["is_signal"].sum())  # -> int

    diagnostics = pathvol.d13_diagnostics(
        table, summary, permutation, gap_table, n_signal_raw
    )  # -> dict

    print(f"  4-1 max_k w_k          : {diagnostics['d4_1_max_weight']:.6f}  "
          f">= {pathvol.D13_DOMINANCE_LIMIT} → flag {diagnostics['d4_1_flag']}")
    print(f"  4-2 부호 불일치 가중치 : {diagnostics['d4_2_opposite_weight']:.6f}  "
          f">= {pathvol.D13_OPPOSITE_WEIGHT_LIMIT} → flag {diagnostics['d4_2_flag']}")
    print(f"  4-3 신호군 중첩 초과 층: {diagnostics['d4_3_flagged_strata'] or '(없음)'}  "
          f"→ flag {diagnostics['d4_3_flag']}")
    print(f"  4-4 Δ_pre              : {diagnostics['d4_4_delta_pre']:+.6f}  "
          f"abs >= {pathvol.D13_PRE_GAP_LIMIT} → flag {diagnostics['d4_4_flag']}")
    print(f"  4-5 유의 교차          : {diagnostics['d4_5_crossing']}  "
          f"| abs(log(p_stud/p_raw)) {diagnostics['d4_5_abs_log_ratio']:.6f} "
          f">= log(3)={np.log(3):.6f} → {diagnostics['d4_5_ratio_exceeded']}  "
          f"→ flag {diagnostics['d4_5_flag']}")
    print(f"  4-6 원자료 유효 신호   : {diagnostics['d4_6_n_signal_raw']}  "
          f"< {pathvol.D13_MIN_SIGNALS} → flag {diagnostics['d4_6_flag']}")

    print()
    print("  축약 전 간격 (신호군은 판정, 대조군은 기술 항목)")
    print(gap_table.to_string(index=False))

    # ---------------------------------------------------------------- 8
    banner("8. W=10 보조 (§1.2) — 기술 전용, 유의성 판정 없음")

    frame_w10 = pathvol.d13_build_frame(
        price, bins, window=pathvol.D13_POST_WINDOW_AUX, truncation="own"
    )  # -> DataFrame

    table_w10 = pathvol.d13_stratum_table(frame_w10)   # -> DataFrame
    summary_w10 = pathvol.d13_delta_and_se(table_w10)  # -> dict

    print(f"  행 수              : {len(frame_w10)}  "
          f"(신호 {int(frame_w10['is_signal'].sum())} / "
          f"대조 {int((~frame_w10['is_signal']).sum())})")
    print(f"  포함 층            : {summary_w10['included']}")
    print(f"  Δ (W=10)           : {summary_w10['delta']:+.6f}")
    print(f"  exp(Δ) W=10        : {summary_w10['exp_delta']:.6f}")
    print(f"  exp(Δ) W=20        : {summary['exp_delta']:.6f}")

    # 절단 조건 해석 차이의 크기를 같이 남긴다 (§1.3 첫 문장 vs D8b valid_mask)
    frame_w10_alt = pathvol.d13_build_frame(
        price, bins, window=pathvol.D13_POST_WINDOW_AUX, truncation="d8b"
    )  # -> DataFrame
    table_w10_alt = pathvol.d13_stratum_table(frame_w10_alt)   # -> DataFrame
    summary_w10_alt = pathvol.d13_delta_and_se(table_w10_alt)  # -> dict

    print(f"  (참고) 절단을 D8b valid_mask로 두면 행 {len(frame_w10_alt)}, "
          f"exp(Δ) {summary_w10_alt['exp_delta']:.6f}")

    # ---------------------------------------------------------------- 산출
    banner("산출 파일")

    results_row = {
        "analysis": "main_W20",
        "W": pathvol.D13_POST_WINDOW,
        "n_signal_raw": n_signal_raw,
        "n_control_raw": int((~frame["is_signal"]).sum()),
        "n_signal_included": summary["n_sig"],
        "K_included": len(summary["included"]),
        "included_strata": ";".join(f"Q{q}" for q in summary["included"]),
        "delta": summary["delta"],
        "exp_delta": summary["exp_delta"],
        "se": summary["se"],
        "t_stud": summary["t_stud"],
        "p_raw": permutation["p_raw"],
        "p_stud": permutation["p_stud"],
        "extreme_raw": permutation["extreme_raw"],
        "extreme_stud": permutation["extreme_stud"],
        "iterations": permutation["iterations"],
        "seed": permutation["seed"],
        "alpha": pathvol.D13_ALPHA,
        "significant_stud": bool(permutation["p_stud"] < pathvol.D13_ALPHA),
        "delta_pre": summary["delta_pre"],
    }  # -> dict
    results_row.update(diagnostics)

    sens_row = {
        "analysis": "declustered_W20",
        "W": pathvol.D13_POST_WINDOW,
        "n_signal_raw": int(declustered["is_signal"].sum()),
        "n_control_raw": int((~declustered["is_signal"]).sum()),
        "n_signal_included": sens_summary["n_sig"],
        "K_included": len(sens_summary["included"]),
        "included_strata": ";".join(f"Q{q}" for q in sens_summary["included"]),
        "delta": sens_summary["delta"],
        "exp_delta": sens_summary["exp_delta"],
        "se": sens_summary["se"],
        "t_stud": sens_summary["t_stud"],
        "p_raw": sens_permutation["p_raw"],
        "p_stud": sens_permutation["p_stud"],
        "extreme_raw": sens_permutation["extreme_raw"],
        "extreme_stud": sens_permutation["extreme_stud"],
        "iterations": sens_permutation["iterations"],
        "seed": sens_permutation["seed"],
        "alpha": pathvol.D13_ALPHA,
        "significant_stud": bool(sens_permutation["p_stud"] < pathvol.D13_ALPHA),
        "delta_pre": sens_summary["delta_pre"],
    }  # -> dict

    aux_row = {
        "analysis": "aux_W10",
        "W": pathvol.D13_POST_WINDOW_AUX,
        "n_signal_raw": int(frame_w10["is_signal"].sum()),
        "n_control_raw": int((~frame_w10["is_signal"]).sum()),
        "n_signal_included": summary_w10["n_sig"],
        "K_included": len(summary_w10["included"]),
        "included_strata": ";".join(f"Q{q}" for q in summary_w10["included"]),
        "delta": summary_w10["delta"],
        "exp_delta": summary_w10["exp_delta"],
        "se": summary_w10["se"],
        "t_stud": summary_w10["t_stud"],
        "p_raw": np.nan,
        "p_stud": np.nan,
        "extreme_raw": np.nan,
        "extreme_stud": np.nan,
        "iterations": np.nan,
        "seed": np.nan,
        "alpha": np.nan,
        "significant_stud": None,
        "delta_pre": summary_w10["delta_pre"],
    }  # -> dict, 유의성 판정 대상이 아니므로 p 관련 항목은 공란

    results = pd.DataFrame([results_row, sens_row, aux_row])  # -> DataFrame (3, N)
    results.to_csv(REPORTS / "day13_pathvol_results.csv", index=False, encoding="utf-8-sig")

    table = table.copy()               # -> DataFrame
    sens_table = sens_table.copy()     # -> DataFrame
    table_w10 = table_w10.copy()       # -> DataFrame

    table["analysis"] = "main_W20"
    sens_table["analysis"] = "declustered_W20"
    table_w10["analysis"] = "aux_W10"

    strata_detail = pd.concat([table, sens_table, table_w10], ignore_index=True)  # -> DataFrame
    strata_detail.to_csv(REPORTS / "day13_strata_detail.csv", index=False, encoding="utf-8-sig")

    gap_table = gap_table.copy()            # -> DataFrame
    gap_table_thin = gap_table_thin.copy()  # -> DataFrame

    gap_table["stage"] = "before_decluster"
    gap_table_thin["stage"] = "after_decluster"

    gap_output = pd.concat([gap_table, gap_table_thin], ignore_index=True)  # -> DataFrame
    gap_output.to_csv(REPORTS / "day13_gap_ratio.csv", index=False, encoding="utf-8-sig")

    sensitivity = pd.DataFrame([results_row, sens_row])  # -> DataFrame (2, N)
    sensitivity.to_csv(REPORTS / "day13_sensitivity_declustered.csv",
                       index=False, encoding="utf-8-sig")

    for name, table_object in [
        ("day13_pathvol_results.csv", results),
        ("day13_strata_detail.csv", strata_detail),
        ("day13_gap_ratio.csv", gap_output),
        ("day13_sensitivity_declustered.csv", sensitivity),
    ]:
        print(f"  저장: reports/{name}  ({table_object.shape[0]}행 × {table_object.shape[1]}열)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
