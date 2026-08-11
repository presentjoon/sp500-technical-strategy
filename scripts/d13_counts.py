"""D13 사전등록 빈칸용 계수 전용 스크립트.

**계수만 한다.** 종속변수(사후 수익률·경로변동성)를 계산하거나 출력하지 않는다.
층 경계는 D12 코드(`src/stratified.py` → `src/pathvol.py`)를 import해 쓰고
새로 구현하지 않는다.

    python scripts/d13_counts.py

절단 조건
---------
`t+20`이 분석구간을 넘어가는 날은 제외한다. `signals.forward_returns()`가
`fwd_ret_20`을 NaN으로 두므로 **그 컬럼의 유효 여부**로 판정한다 — 인덱스를
직접 세지 않는 이유는 D12(`stratified.attach_returns`)가 쓴 것과 같은 기준을
유지하기 위해서다.

간격 단위
---------
거래일 인덱스 차이(정수)다. 달력일이 아니다.
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

from src import config      # noqa: E402
from src import data        # noqa: E402
from src import signals     # noqa: E402
from src import stratified  # noqa: E402


HORIZON = 20      # -> int, D13 절단 기준
MIN_GAP = 20      # -> int, 축약 시 요구하는 최소 거래일 간격
QUINTILES = [1, 2, 3, 4, 5]  # -> list[int]


def force_utf8_console():
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)  # -> callable | None

        if reconfigure is not None:
            reconfigure(encoding="utf-8")


# ---------------------------------------------------------------------------
# 2단계 축약
# ---------------------------------------------------------------------------
def greedy_thin(positions, min_gap=MIN_GAP):
    """시간순 정렬된 인덱스에서 간격 min_gap 미만인 후속 항목을 제거한다.

    가장 이른 항목을 첫 앵커로 두고, 현재 앵커와의 간격이 `min_gap` 미만이면
    버린다. `min_gap` 이상인 첫 항목이 새 앵커가 된다.

    Returns
    -------
    list[int]
        남은 앵커 인덱스.
    """
    ordered = sorted(positions)  # -> list[int]

    if len(ordered) == 0:
        return []

    anchors = [ordered[0]]  # -> list[int]

    for position in ordered[1:]:
        gap = position - anchors[-1]  # -> int

        if gap >= min_gap:
            anchors.append(position)

    return anchors


def drop_near_anchors(positions, anchors, min_gap=MIN_GAP):
    """앵커와의 거래일 간격이 min_gap 미만인 항목을 전부 제거한다."""
    if len(anchors) == 0:
        return sorted(positions)

    anchor_array = np.asarray(sorted(anchors))  # -> ndarray[int]

    kept = []  # -> list[int]

    for position in sorted(positions):
        distances = np.abs(anchor_array - position)  # -> ndarray[int]

        if distances.min() >= min_gap:
            kept.append(position)

    return kept


def gap_stats(positions):
    """연속 간격에 대한 (n_gap, count(g<20))."""
    ordered = sorted(positions)  # -> list[int]

    if len(ordered) < 2:
        return 0, 0

    gaps = np.diff(np.asarray(ordered))  # -> ndarray[int]

    return int(len(gaps)), int((gaps < MIN_GAP).sum())


def count_by_bin(frame, positions_column, subset_positions):
    """인덱스 집합을 층별로 센다."""
    lookup = dict(zip(frame[positions_column], frame["quintile"]))  # -> dict

    counts = {q: 0 for q in QUINTILES}  # -> dict[int, int]

    for position in subset_positions:
        bin_value = lookup.get(position)  # -> float | None

        if bin_value is None or (isinstance(bin_value, float) and np.isnan(bin_value)):
            continue

        counts[int(bin_value)] = counts[int(bin_value)] + 1

    return counts


def main():
    force_utf8_console()

    price = data.load_parquet(config.RAW_OHLCV_PATH)  # -> DataFrame (18424, 8)

    # 층 배정 — D12 코드 그대로 (재구현 없음)
    bins = stratified.build_bin_frame(price)  # -> DataFrame (9212, 5)

    matched, counted = stratified.check_bin_assignment(bins)  # -> (bool, dict)

    print("층 배정 재현 (D8b 집계표 대조):", counted, "| 일치:", matched)

    if not matched:
        print("!! 층 배정 불일치 — 중단")
        return 1

    # 사후 수익률 컬럼은 **절단 판정에만** 쓴다. 값은 읽지 않는다.
    forward = signals.forward_returns(price)  # -> DataFrame (18424, 13)

    ticker = config.TICKERS["signal"]                    # -> str
    forward = forward.loc[forward["ticker"] == ticker]   # -> DataFrame (9212, 13)
    forward = forward.sort_values("date")                # -> DataFrame

    column_name = f"fwd_ret_{HORIZON}"  # -> str, EX-POST ONLY (절단 판정 전용)

    truncation = forward[["date", column_name]]  # -> DataFrame (9212, 2)

    merged = bins.merge(truncation, on="date", how="left")  # -> DataFrame (9212, 6)

    if len(merged) != len(bins):
        raise ValueError(f"병합에서 행 수가 변했다: {len(bins)} → {len(merged)}")

    merged = merged.reset_index(drop=True)  # -> DataFrame
    merged["pos"] = merged.index            # -> DataFrame, 거래일 인덱스

    in_window = merged["in_analysis_period"]              # -> Series[bool]
    survives = merged[column_name].notna()                # -> Series[bool]
    has_bin = merged["quintile"].notna()                  # -> Series[bool]

    passes = in_window & survives & has_bin  # -> Series[bool]

    # ---------------------------------------------------------------- B1
    signal_all = merged.loc[in_window & merged["is_signal"]]        # -> DataFrame
    signal_valid = merged.loc[passes & merged["is_signal"]]         # -> DataFrame

    n_signal_all = len(signal_all)        # -> int
    n_signal_valid = len(signal_valid)    # -> int
    n_signal_dropped = n_signal_all - n_signal_valid  # -> int

    # ---------------------------------------------------------------- B2
    control_valid = merged.loc[passes & (~merged["is_signal"])]  # -> DataFrame
    n_control_valid = len(control_valid)                          # -> int

    # ---------------------------------------------------------------- B3
    signal_positions = signal_valid["pos"].tolist()    # -> list[int]
    control_positions = control_valid["pos"].tolist()  # -> list[int]

    anchors = greedy_thin(signal_positions)  # -> list[int], 1단계

    control_far = drop_near_anchors(control_positions, anchors)  # -> list[int], 2단계 전반
    control_thin = greedy_thin(control_far)                      # -> list[int], 2단계 후반

    # ---------------------------------------------------------------- B4 / B5 / B6
    sig_by_bin = count_by_bin(merged, "pos", signal_positions)   # -> dict
    ctl_by_bin = count_by_bin(merged, "pos", control_positions)  # -> dict

    sig_by_bin_thin = count_by_bin(merged, "pos", anchors)       # -> dict
    ctl_by_bin_thin = count_by_bin(merged, "pos", control_thin)  # -> dict

    # ---------------------------------------------------------------- B7
    bin_of = dict(zip(merged["pos"], merged["quintile"]))  # -> dict

    gap_rows = []  # -> list[dict]

    for quintile in QUINTILES:
        before = [p for p in signal_positions if bin_of.get(p) == quintile]  # -> list[int]
        after = [p for p in anchors if bin_of.get(p) == quintile]            # -> list[int]

        n_gap_before, lt_before = gap_stats(before)  # -> (int, int)
        n_gap_after, lt_after = gap_stats(after)     # -> (int, int)

        gap_rows.append({
            "quintile": f"Q{quintile}",
            "n_gap_before": n_gap_before,
            "cnt_lt20_before": lt_before,
            "ratio_before": lt_before / n_gap_before if n_gap_before else np.nan,
            "n_gap_after": n_gap_after,
            "cnt_lt20_after": lt_after,
            "ratio_after": lt_after / n_gap_after if n_gap_after else np.nan,
        })

    gap_table = pd.DataFrame(gap_rows)  # -> DataFrame (5, 7)

    # ---------------------------------------------------------------- 출력
    print()
    print("=" * 84)
    print("B1 — 절단 (t+20이 분석구간을 넘어가는 신호)")
    print("=" * 84)
    print(f"  분석구간 S1 신호      : {n_signal_all}")
    print(f"  절단으로 제외         : {n_signal_dropped}")
    print(f"  유효 신호             : {n_signal_valid}")

    print()
    print("=" * 84)
    print("B2 — 대조군 (신호 비발생일 중 동일 절단 조건 통과)")
    print("=" * 84)
    print(f"  대조군                : {n_control_valid}")
    print(f"  (합계 확인) 신호+대조 : {n_signal_valid + n_control_valid}")

    print()
    print("=" * 84)
    print("B3 — 2단계 축약 후")
    print("=" * 84)
    print(f"  1단계 신호 앵커                 : {len(anchors)}  (축약 전 {n_signal_valid})")
    print(f"  2단계-전반 앵커 근접 제거 후    : {len(control_far)}  (축약 전 {n_control_valid})")
    print(f"  2단계-후반 그리디 축약 후 대조군: {len(control_thin)}")

    print()
    print("=" * 84)
    print("B4 / B5 — 축약 전 층별 (n_k,sig / n_k,ctl)")
    print("=" * 84)
    before_table = pd.DataFrame({
        "quintile": [f"Q{q}" for q in QUINTILES],
        "n_sig": [sig_by_bin[q] for q in QUINTILES],
        "n_ctl": [ctl_by_bin[q] for q in QUINTILES],
    })  # -> DataFrame (5, 3)
    print(before_table.to_string(index=False))
    print(f"  합계: 신호 {before_table['n_sig'].sum()} / 대조 {before_table['n_ctl'].sum()}")

    print()
    print("=" * 84)
    print("B6 — 축약 후 층별 (n_k,sig / n_k,ctl)")
    print("=" * 84)
    after_table = pd.DataFrame({
        "quintile": [f"Q{q}" for q in QUINTILES],
        "n_sig": [sig_by_bin_thin[q] for q in QUINTILES],
        "n_ctl": [ctl_by_bin_thin[q] for q in QUINTILES],
    })  # -> DataFrame (5, 3)
    after_table["min5_pass"] = after_table["n_sig"] >= 5
    print(after_table.to_string(index=False))
    print(f"  합계: 신호 {after_table['n_sig'].sum()} / 대조 {after_table['n_ctl'].sum()}")

    print()
    print("=" * 84)
    print("B7 — 신호군 층별 연속 간격 count(g<20)/n_gap")
    print("=" * 84)
    print(gap_table.to_string(index=False))

    # ---------------------------------------------------------------- CSV
    rows = []  # -> list[dict]

    for quintile in QUINTILES:
        gap_row = gap_table.loc[gap_table["quintile"] == f"Q{quintile}"].iloc[0]  # -> Series

        rows.append({
            "quintile": f"Q{quintile}",
            "n_sig_before": sig_by_bin[quintile],
            "n_ctl_before": ctl_by_bin[quintile],
            "n_sig_after": sig_by_bin_thin[quintile],
            "n_ctl_after": ctl_by_bin_thin[quintile],
            "min5_pass_after": sig_by_bin_thin[quintile] >= 5,
            "n_gap_before": gap_row["n_gap_before"],
            "cnt_lt20_before": gap_row["cnt_lt20_before"],
            "ratio_before": gap_row["ratio_before"],
            "n_gap_after": gap_row["n_gap_after"],
            "cnt_lt20_after": gap_row["cnt_lt20_after"],
            "ratio_after": gap_row["ratio_after"],
        })

    rows.append({
        "quintile": "TOTAL",
        "n_sig_before": n_signal_valid,
        "n_ctl_before": n_control_valid,
        "n_sig_after": len(anchors),
        "n_ctl_after": len(control_thin),
        "min5_pass_after": None,
        "n_gap_before": None,
        "cnt_lt20_before": None,
        "ratio_before": None,
        "n_gap_after": None,
        "cnt_lt20_after": None,
        "ratio_after": None,
    })

    output = pd.DataFrame(rows)  # -> DataFrame (6, 12)
    output.to_csv(PROJECT_ROOT / "reports" / "day13_counts.csv",
                  index=False, encoding="utf-8-sig")

    print()
    print("저장: reports/day13_counts.csv", f"({len(output)}행)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
