# 산출물 인벤토리 — D1 ~ D8b

**작성:** 2026-08-09 (D9, 예비일)
**목적:** 8/15 최종 리포트와 8/16 보고 요약의 목차 재료.
**작성 방식:** 파일을 직접 열어 내용 기준으로 작성했다. 커밋 메시지만 보고
추측하지 않았다.

> **상태 열의 뜻**
> **유효** — 현재 문서에서 그대로 인용 가능
> **정정됨** — 값이나 서술이 뒤에 고쳐졌다. 원문은 정정 주석과 함께 보존
> **폐기** — 전제가 무너져 사용 금지. 방법론 기록으로만 보존

---

## 0. 날짜별 한 줄 요약 (복기 뼈대)

| D# | 날짜 | 무엇을 물었고 무엇이 나왔는가 |
|---|---|---|
| **D1** | 08-02 | **데이터를 믿을 수 있는가?** — 18,424행 수집, 중복 0·OHLC 위반 0 확인. ±5% 충격일 39건이 2008·2020에 몰려 있음을 발견(H1의 출발점) |
| **D2** | 08-03 | **국면을 나누면 수익률 성질이 달라지는가?** — 6국면 전부 초과첨도 > 0(변동성 군집), 국면 간 연율변동성 **3.30배**. 이 3.30이 이후 모든 지표 비교의 기준자가 됨 |
| **D3** | 08-04 | **내 RSI와 ta의 RSI는 왜 다른가?** — 평활이 아니라 **시딩 관례** 차이. 시드를 맞추면 비트 단위 일치, SMA 시드 차이는 1992-03-27에 정확히 0으로 흡수. "지표 이름만으로 신호가 특정되지 않는다" |
| **D4** | 08-05 | **어떤 지표가 국면 차이를 통과시키는가?** — 동차 차수 사다리 완성(ATR 원값 6.05배 → RSI 1.25배). H3′ 지지(구조가 지표 계열보다 중요), **H2는 역전**(볼린저 1.313599배 vs RSI 7.796056배) |
| **D5** | 08-06 | **결과를 보기 전에 무엇을 확정해야 하는가?** — 코드 없이 사전등록만. 확증 대상($m$=20)과 설계 파라미터를 분리, 임계값·지평·신호 목록을 결과 열람 전 고정 |
| **D6** | 08-07 | **신호는 무작위 진입보다 나은가?** — 20조합 중 Bonferroni 통과 **1건**(S1 h=1, p=0.0019). 단 **부호가 가설과 반대**(−0.502%p)이고, 사후 진단에서 `sd_ratio` 2.14·효과 3일 집중이 드러남 |
| **D7** | 08-08 | **그 1건은 진짜인가?** — 스튜던트화 순열검정에서 **1/20 → 0/20**. p가 0.0019 → 0.1244로 65배 커짐. 방향 규칙 20/20 일치. 남은 관측은 "변동성 탐지기" |
| **D8** | 08-08 | **(폐기)** 2.14가 직전 변동성의 재표현인가? — 대상 양을 잘못 전제(2.14는 h=1 횡단면 산포이지 실현변동성이 아님). **결과 산출 전** 발견하고 폐기 |
| **D8b** | 08-08 | **경로변동성으로 다시 물으면?** — $R_0$ = 1.8091, 5분위 층화 후 가중평균 **1.4120** → "잔여 정보 존재". 단 신호 80%가 상위 2분위 편중, Q5 제외 시 1.2652로 보류 구간 |

---

## 1-A. 데이터 · 문서 · 코드 산출물

### 데이터

| 파일 | D# | 커밋 | 주장/역할 | 의존 | 상태 |
|---|---|---|---|---|---|
| `data/raw/ohlcv_raw.parquet` | D1 | (gitignore) | **"수정주가가 매일 재계산되어도 분석 결과는 변하지 않는다"** — 원본을 스냅샷으로 굳혀 재현성을 보장한다 | — | 유효 |
| `data/processed/signals.parquet` | D6 | (gitignore) | **"같은 신호를 다른 엔진에 넣을 수 있다"** — 신호 생성과 백테스트 엔진을 분리한 증거. 2단계 프레임워크 대조의 전제 | `ohlcv_raw.parquet` | 유효 |

### 리포트 — D1 ~ D4

| 파일 | D# | 커밋 | 주장/역할 | 의존 | 상태 |
|---|---|---|---|---|---|
| `reports/day01_audit.txt` | D1 | `b2af1b0` | **"이 데이터셋은 검증을 통과했다"** — 중복 0, OHLC 위반 0, 연평균 행수 252 근처. 이후 모든 분석이 이 파일 없이는 "데이터가 맞는지 확인했나"에 답할 수 없다 | 원본 parquet | 유효 |
| `reports/day01_shock_days.csv` | D1 | `945e7bd` | **"±5% 충격일 39건이 2008·2020에 몰려 있다"** — H1(전략이 아니라 위기 탐지기)의 최초 근거 | `day01_audit.txt` | 유효 |
| `reports/day02_phase_stats.csv` | D2 | `c5e437a` | **"국면 간 연율변동성이 3.30배 차이 나고, 6국면 전부 초과첨도가 양수다"** — D4 지표 사다리의 기준자이자 D8b 층화 설계의 전제(변동성은 군집한다) | `day01_audit.txt` | 유효 |
| `reports/day03_indicator_diff.csv` | D3 | `94224cb` | **"내 구현과 ta의 차이는 평활 방식이 아니라 시딩 관례에서 온다"** — 시드를 맞추면 최대 절대차가 ULP 수준으로 떨어진다 | `day02_phase_stats.csv` (국면 라벨) | 유효 |
| `reports/day04_volatility_diff.csv` | D4 | `e0d71aa` | **"ATR은 ta가 warm-up을 0.0으로 채운다"** — 라이브러리 기본값을 믿으면 안 되는 두 번째 사례 | `day03_indicator_diff.csv` | 유효 |
| `reports/day04_phase_indicator_stats.csv` | D4 | `e0d71aa` (+ D9 컬럼 추가) | **"동차 차수가 국면 간 배율을 결정한다"** — ATR 원값 6.05배 ~ RSI 1.25배 사다리. 1차 동차 지표를 시기 간 비교에 쓰면 안 되는 이유. **D9에서 `임계규칙`·`판단가능일`·`발동일`·`발동률%` 4개 컬럼 추가** — 발동률이 CSV에 없고 문서에만 있었던 것이 7.85/7.80 불일치의 구조적 원인이었다 | `day02_phase_stats.csv` | 유효 |

### 리포트 — D6

| 파일 | D# | 커밋 | 주장/역할 | 의존 | 상태 |
|---|---|---|---|---|---|
| `reports/day06_signal_counts.csv` | D6 | `ddcd2d4` | **"사전 예측한 사건 수가 5개 모두 빗나갔고, 원인은 지속일수 과소평가다"** — 레벨 발동률은 맞았는데 사건 수가 틀렸다는 분해 | `signals.parquet` | 유효 |
| `reports/day06_event_study.csv` | D6 | `ddcd2d4` | **"20조합의 초과수익은 대부분 ±0.5%p 안에 있다"** — 무작위 진입 기준선(h=20에서 sd 4.60%)을 확정해 §7.3의 5.4% 가정이 17% 컸음을 드러냄 | `signals.parquet` | 유효 |
| `reports/day06_overlap_diagnostic.csv` | D6 | `ddcd2d4` | **"중첩 심각은 S5가 아니라 S2·S3였다"** — §6.6 사전 경고가 오적중했음의 증거. 예측이 틀리면 파생 진단도 함께 틀린다 | `day06_signal_counts.csv` | 유효 |
| `reports/day06_permutation.csv` | D6 | `ddcd2d4` | **"20조합 중 Bonferroni 통과는 1건뿐이다"** — 확증 검정의 1차 결과. 이후 D7이 이 1건을 해체한다 | `day06_event_study.csv` | **정정됨** — D7 스튜던트화로 통과가 0건이 됨. 원 결과는 유효한 sharp null 검정이므로 무효가 아니다 |
| `reports/day06_diag_variance.csv` | D6 | `3e25010` | **"S1 h=1의 `sd_ratio`가 2.1437로 20조합 중 최고다"** — 순열검정의 교환가능성 위반을 수치화. D7 스튜던트화와 D8·D8b 전체의 출발점 | `day06_permutation.csv` | 유효 |
| `reports/day06_diag_regime.csv` | D6 | `3e25010` | **"대안 2(국면 탐지기)는 기각된다"** — `rate_ratio` 최대 2.61 < 사전 기준 3, 국면 내부에서도 신호일이 더 나쁨 | `day06_permutation.csv` | 유효 |
| `reports/day06_diagnostics.md` | D6 | `06be1c1` (사전 약속) → `3e25010` (결과) | **"통과 1건을 축하하지 않고 의심했다"** — 진단 계획을 실행 **전에** 커밋한 기록. 사전등록의 실물 증거 | 위 진단 CSV 2종 | **정정됨** — "3일이 86.1%"의 분모 미명시. 2곳에 `[†]` 정정 주석(80.8%가 정본) |

### 리포트 — D7

| 파일 | D# | 커밋 | 주장/역할 | 의존 | 상태 |
|---|---|---|---|---|---|
| `reports/day07_studentized.csv` | D7 | `5a86f5e` | **"검정통계량을 바꾸면 통과가 1/20 → 0/20이 된다"** — 같은 데이터·같은 $B$·같은 시드에서 p가 65배 움직인 기록. `sd_ratio` > 1 ↔ p 상승이 20/20 일치 | `day06_diag_variance.csv`, `day06_permutation.csv` | 유효 |
| `reports/day07_signal_comparison.csv` | D7 | `684f9a5` | **"5신호를 11개 항목으로 한 화면에 놓으면 어느 것도 살아남지 못한다"** — 신규 계산 없이 D6·D7 산출물을 재구성한 요약표 | D6 CSV 5종 + `day07_studentized.csv` | 유효 |
| `reports/day07_robustness.md` | D7 | `b076423` (사전 약속) → `5a86f5e` (결과) → `1f63664` (정정) | **"강건성 검증 계획도 실행 전에 고정했다"** — 판독 기준 2개를 미리 적었고 실측이 둘 다 넘었다 | `day06_diagnostics.md` | **정정됨** — 판정 서술("1.495 유지" → 기준 미달), sharp/weak null 프레이밍, VR 분모를 Lo–MacKinlay로 교체 |
| `reports/week2.md` | D7 | `684f9a5` → `1f63664`, `c6a2cfd`, `c549a81` | **"2주차 전체를 하나의 논지로 묶는다"** — §0 논지를 백테스트 착수 **전에** 고정한 문서 | 2주차 CSV·MD 전부 | **정정됨** — §5·§6 해석은 아직 TODO(사용자 작성 대기) |

정정 2 (D15, 2026-08-14) — §5·§6 해석 문장 작성 완료. 근거 수치 표는 §5.1·§6.1로 보존.

### 리포트 — D8b

| 파일 | D# | 커밋 | 주장/역할 | 의존 | 상태 |
|---|---|---|---|---|---|
| `reports/day08b_pathvol.csv` | D8b | `1d15e3e` | **"신호일의 80%가 직전변동성 상위 2분위에 있다"** — 분위별 8개 컬럼. 층화가 비교 단위로 기능하지 못한 증거 | `signals.parquet`, `ohlcv_raw.parquet` | 유효 |
| `reports/day08b_pathvol.md` | D8b | `1d15e3e` → `c549a81` | **"경로변동성으로 물으면 층화 후에도 1.41배가 남지만, 그 판정은 세 가지 제약을 동반한다"** — $R_0$ 측정부터 판정·민감도·한계까지 | `day08b_pathvol.csv`, `day06_diag_variance.csv`(대비용) | 유효 |
| `reports/week2_artifacts.md` | D9 | (본 커밋) | **"2주차에 무엇을 만들었고 무엇이 폐기됐는지 한 화면에서 본다"** — 최종 리포트 목차 재료 | 위 전부 | 유효 |
| `reports/week2_key_numbers.md` | D9 | (본 커밋) | **"같은 수치가 문서마다 다르게 적히는 것을 막는다"** — 정의·출처·정정 이력의 단일 출처 | 위 전부 | 유효 |

### 문서 (`docs/`)

| 파일 | D# | 커밋 | 주장/역할 | 의존 | 상태 |
|---|---|---|---|---|---|
| `docs/glossary.md` | D1~D5 | `1919076` → `45c750a` | **"이해하지 못한 개념이 레포에 들어가지 않았다"** — 1,300여 줄, 한국어 정의 + 영어 병기. 랩실 평가에서 "이 코드 설명해보라"에 답하는 근거 | — | 유효 |
| `docs/signal_spec.md` | D5 | `c11e4a9` → `9eebfd0` | **"결과를 보기 전에 분석 계획 전체를 확정했다"** — §4~§7은 이후 절대 편집 금지. 확증 대상과 설계 파라미터의 분리가 핵심 | `day04_phase_indicator_stats.csv` | 유효 |
| `docs/prereg_d08_stratified.md` | D8 | `415e377` → `7c0b810` | **(사용 금지)** 층화 반증 테스트의 최초 설계 | `day06_diag_variance.csv` | **폐기** — §1·§2가 대상 양을 "사후 20일 실현변동성"으로 전제했으나 근거인 D7의 2.14는 (a) h=1 값이고 (b) 사건별 실현변동성이 아니라 사건 간 횡단면 표준편차였다. **결과 산출 전** 발견 |
| `docs/prereg_d08b_pathvol.md` | D8b | `f2f2947` | **"D8b의 예측과 판정 기준은 실행 전에 확정됐다"** — 판별력 하한 조항($R_0$ < 1.15면 판정 미수행)을 $R_0$ 확인 전에 넣은 기록 | `prereg_d08_stratified.md`(폐기 사유) | 유효 |

### 로그 (`logs/`)

| 파일 | D# | 커밋 | 주장/역할 | 상태 |
|---|---|---|---|---|
| `logs/2026-08-02.md` | D1 | `945e7bd` | O1~O3, H1 최초 제기. D8 일정을 "이벤트 스터디"로 적었으나 실제 D6로 앞당겨짐 | 유효 |
| `logs/2026-08-03.md` | D2 | `ac59cf3` → `75df6d3` | 국면 통계와 O4, H1 정량화 | 유효 |
| `logs/2026-08-04.md` | D3 | `bbd5cdb` → `4ba5c24` | 시딩 관례 발견(O5). 관찰 절은 사용자 직접 작성 | 유효 |
| `logs/2026-08-05.md` | D4 | `c73525c` → `4ba5c24` | H2 역전·H3′ 지지, 동차 사다리(O6·O7) | 유효 |
| `logs/2026-08-06.md` | D5 | `4a004de` → `ff5660e` | 사전등록 작성. **S5 SE 귀인 오류를 정정한 기록 포함** | **정정됨** |
| `logs/2026-08-07.md` | D6 | `e172c9b` → `a8717eb`, `4ba5c24` | 확증 결과와 사후 진단. **"몬테카를로 경계 사례" 최초 기입을 정정한 흔적 보존** | **정정됨** |
| `logs/2026-08-08.md` | D7·D8·D8b | `a8717eb` → `1f63664`, `c6a2cfd`, `c549a81` | 하루에 D7 정정 + D8 폐기 + D8b 완주. O8·O11~O13, H6 | **정정됨** |

### 노트북 (`notebooks/`)

| 파일 | D# | 커밋 | 주장/역할 | 상태 |
|---|---|---|---|---|
| `notebooks/day02_returns.ipynb` | D2 | `c5e437a` | 로그수익률과 6국면 통계 실행. 162 KB | 유효 |
| `notebooks/day03_indicators.ipynb` | D3 | `94224cb` | RSI·MACD 자체 검증 후 ta 대조. 495 KB | 유효 |
| `notebooks/day04_volatility_indicators.ipynb` | D4 | `e0d71aa` | ATR·볼린저와 H2/H3/H4 검정. **839 KB — CLAUDE.md의 1 MB 상한에 가장 근접** | 유효 |
| `notebooks/day06_event_study.ipynb` | D6 | `ddcd2d4` | 이벤트 스터디와 순열검정 실행. 690 KB | 유효 |

---

## 1-B. figures 매핑

**인용 판정 기준:** `reports/`, `docs/`, `logs/`, `notebooks/` 안에서 파일명이
언급되면 인용으로 본다. `week2.md` §3-2는 `event_study_*.png (5장)` 형태의
글롭으로 참조하므로, 그 5장은 **week2 인용으로 함께 센다.**

| 파일 | D# | 뒷받침하는 주장 (이 그림이 없으면 쓸 수 없는 문장) | 인용된 문서 |
|---|---|---|---|
| `day02_rolling_volatility.png` | D2 | **"변동성은 군집한다 — 조용한 시기와 격동기가 덩어리로 나타난다."** D8b 층화 설계 전체의 전제이자, "직전 변동성으로 사후 변동성을 예측할 수 있다"는 사슬의 첫 고리 | `glossary.md`, `logs/2026-08-03.md`, D2 노트북 |
| `day03_rsi_comparison.png` | D3 | **"두 RSI 구현의 차이는 눈으로는 보이지 않지만 임계선 근처에서 신호를 갈라놓는다."** 로그 스케일 전 구간(1990–) 대조 | `logs/2026-08-04.md`, D3 노트북 |
| `day03_seed_convergence.png` | D3 | **"시딩 차이는 지수적으로 소멸하며 1992-03-27에 정확히 0이 된다."** 초기값 차이가 영구적이지 않음을 보인 유일한 그림 | `logs/2026-08-04.md`, D3 노트북 |
| `day04_rsi_phase_hist.png` | D4 | **"RSI는 국면이 바뀌어도 분포가 거의 이동하지 않는다"** (배율 1.25배). 유계·0차 동차 지표의 시각적 증거 | `logs/2026-08-05.md`, D4 노트북 |
| `day04_atr_phase_hist.png` | D4 | **"ATR 원값은 국면에 따라 분포 자체가 통째로 이동한다"** (배율 6.05배). 위 RSI 그림과 짝을 이뤄야 사다리가 성립한다 | `logs/2026-08-05.md`, D4 노트북 |
| `day04_panel_calm.png` | D4 | **"조용한 구간에서는 밴드가 좁고 ATR이 낮다."** 아래 covid 패널과의 대비가 주장의 본체 | `logs/2026-08-05.md`, D4 노트북 |
| `day04_panel_covid.png` | D4 | **"같은 지표가 코로나 구간에서는 전혀 다른 스케일로 움직인다."** 단일 구간 성과를 믿지 않는다는 규칙의 시각적 근거 | `logs/2026-08-05.md`, D4 노트북 |
| `day06_horizon_curves.png` | D6 | **"지평을 늘려도 초과수익이 체계적으로 나타나지 않는다."** 20조합을 한 화면에 놓은 유일한 그림 | `week2.md` §3-2, D6 노트북 |
| `day06_s1_pre_event.png` | D6 | **"신호는 직전 5거래일 −5.3% 하락 뒤에 발동한다."** "떨어지는 칼날"과 D8b의 사슬(RSI<30 → 최근 변동성 높음)을 잇는 유일한 증거 | `day06_diagnostics.md`, `week2.md` §6 |
| `day06_diag_regime_rate.png` | D6 | **"발동률 편중이 사전 기준 3배에 못 미친다"** — 대안 2(국면 탐지기) 기각의 근거 | `day06_diagnostics.md` |
| `day06_diag_s1_timeline.png` | D6 | **"S1 신호가 특정 시기에만 몰려 있지는 않다."** H1(위기 탐지기)을 S1에 대해 부분 기각 | `day06_diagnostics.md` |
| `day07_p_mean_vs_stud.png` | D7 | **"`sd_ratio` > 1이면 p가 오르고 < 1이면 내린다 — 20/20 예외 없이."** 방향 규칙이 경험적 우연이 아님을 보이는 핵심 그림 | `day07_robustness.md`, `week2.md`, `logs/2026-08-08.md` |
| `day07_signal_overview.png` | D7 | **"5개 신호 어느 것도 조정 유의수준을 넘지 못한다."** 결론을 한 장으로 요약 | `week2.md`, `logs/2026-08-08.md` |
| `day08b_signal_concentration.png` | D8b | **"5분위 층화는 통제로 기능하지 못했다 — 신호의 80%가 상위 2분위에 있다."** 무작위 기대선 2개(사전등록 가정 13.0건 / 실제 점유율 반영 15.78건)를 함께 그려, 어느 분모를 써도 편중이 남음을 보인다 | `day08b_pathvol.md` §4-3, `week2.md` §6-1 |
| `event_study_S1_rsi_oversold.png` | D6 | **"S1의 사후 수익률 분포는 무작위 대비 넓고 왼쪽으로 치우쳐 있다."** `sd_ratio` 2.14의 시각적 대응물 | D6 노트북, `week2.md` §3-2 (글롭) |
| `event_study_S2_rsi_overbought.png` | D6 | **"S2는 분포가 오히려 좁다"** (`sd_ratio` 0.51) — 반보수 방향의 대조군 | D6 노트북, `week2.md` §3-2 (글롭) |
| `event_study_S3_bb_lower_break.png` | D6 | **"S3는 S1과 같은 방향이지만 폭이 작다"** (`sd_ratio` 1.53) | D6 노트북, `week2.md` §3-2 (글롭) |
| `event_study_S4_bb_upper_break.png` | D6 | **"S4는 분포가 좁아 스튜던트화 시 p가 내려간다"** (`sd_ratio` 0.53) | D6 노트북, `week2.md` §3-2 (글롭) |
| `event_study_S5_macd_cross.png` | D6 | **"S5는 무작위와 거의 구분되지 않는다"** (`sd_ratio` 0.80, 초과 −0.001%p) | D6 노트북, `week2.md` §3-2 (글롭) |
| `permutation_S1_rsi_oversold_h20.png` | D6 | **"관측값이 귀무분포의 어디에 놓이는가"** — S1 h=20 | D6 노트북 — **부록 후보** |
| `permutation_S2_rsi_overbought_h20.png` | D6 | 같음 — S2 h=20 | D6 노트북 — **부록 후보** |
| `permutation_S3_bb_lower_break_h20.png` | D6 | 같음 — S3 h=20 | D6 노트북 — **부록 후보** |
| `permutation_S4_bb_upper_break_h20.png` | D6 | 같음 — S4 h=20 | D6 노트북 — **부록 후보** |
| `permutation_S5_macd_cross_h20.png` | D6 | 같음 — S5 h=20 | D6 노트북 — **부록 후보** |

### 부록 후보 (버리라는 뜻이 아니라 선별 재료)

**순열 귀무분포 5장은 본문에 인용하지 않고 부록 후보로 둔다** (2026-08-09 결정).
노트북에는 있으나 리포트 문장이 이들을 근거로 쓰지 않는다.

**아직 빠져 있는 그림**

- **D8b는 `day08b_signal_concentration.png` 1장뿐이다.** 신호일 편중은
  이 그림이 받쳐주지만, 분위별 `ratio`(0.90 / 1.32 / 1.08 / 1.36 / 1.55)를
  보여주는 그림은 없다.
- **D7 강건성 그림이 `day07_p_mean_vs_stud.png` 하나뿐이다.** 3일 제거 대비나
  VR(20) 관련 그림은 없다.

---

## 1-C. `src` 모듈 매핑

| 모듈 | 주요 함수 | 만드는 산출물 |
|---|---|---|
| `src/config.py` | (상수만, 함수 없음) | 백테스트 가정의 **단일 출처**. 2단계에서 Backtrader로 그대로 옮길 값들 |
| `src/data.py` | `download_ohlcv()` → `_normalize_frame()` → `save_parquet()` | `data/raw/ohlcv_raw.parquet` |
| | `audit()` → `save_audit_report()` | `reports/day01_audit.txt` |
| | `shock_days()` | `reports/day01_shock_days.csv` |
| | `load_parquet()`, `slice_analysis()` | (이후 전 모듈의 입력) |
| `src/returns.py` | `add_log_return()` | 로그수익률 컬럼 (D2~D8b 전부의 기초) |
| | `tag_phase()` → `phase_statistics()` → `to_markdown_table()` | `reports/day02_phase_stats.csv` |
| `src/indicators.py` | `sma()`, `ema()`, `wilder_rma()` | 평활 원시 함수 (D3에서 추출, D4가 재사용) |
| | `rsi_simple()`, `rsi_wilder()`, `macd()` | D3 지표 |
| | `true_range()`, `atr()`, `bollinger()`, `ta_atr_masked()` | D4 지표 |
| | `precision_check()`, `compare_columns()`, `comparison_table()`, `classify_difference()` | `reports/day03_indicator_diff.csv`, `reports/day04_volatility_diff.csv` |
| | `phase_distribution()`, `phase_spread()`, `threshold_rate_by_phase()` | `reports/day04_phase_indicator_stats.csv` |
| | `threshold_signal()`, `decompose_disagreement()` | D3 불일치 분해 (신호 수준) |
| `src/signals.py` | `to_edge()`, `add_indicators()`, `make_signals()` | `data/processed/signals.parquet`, `reports/day06_signal_counts.csv` |
| | `forward_returns()`, `attach_forward_returns()` | `reports/day06_event_study.csv` |
| `src/diagnostics.py` | `prepare_frames()` | (D6·D7 진단 전부의 입력 — 재계산이 아니라 재현) |
| | `variance_diagnostic()`, `classify_variance()` | `reports/day06_diag_variance.csv` |
| | `regime_diagnostic()` | `reports/day06_diag_regime.csv` |
| | `pre_event_curve()` | `figures/day06_s1_pre_event.png` |
| | `permutation_pvalue()`, `studentized_permutation()` | `reports/day07_studentized.csv` |
| | `drop_days_recalculation()`, `concentration_shares()` | `day07_robustness.md` §8 (3일 제거·80.8%) |
| | `variance_ratio()` | `week2.md` 부록 A (VR(20), Lo–MacKinlay $z^*$) |
| `src/pathvol.py` | `add_log_return()`, `add_pre_volatility()`, `add_post_volatility()` | `pre_vol` / `post_vol` 컬럼 |
| | `expanding_quantile_bin()`, `build_frame()` | 분위 배정된 분석 프레임 |
| | `baseline_ratio()` | $R_0$ = 1.8091 |
| | `stratified_table()`, `weighted_ratio()`, `verdict()` | `reports/day08b_pathvol.csv`, 판정 |

**메모.** `src/pathvol.py`가 `add_log_return()`을 `returns.py`와 **중복 정의**한다.
정렬 순서와 컬럼명을 모듈이 스스로 통제하기 위한 의도된 중복이며, docstring에
사유를 적어뒀다. D12에서 통합할지는 열려 있다.

---

## 2. 다음 단계에서 이 문서를 쓰는 법

- **8/15 최종 리포트 목차** — §0의 날짜별 한 줄이 그대로 절 제목 후보다.
- **8/16 보고 요약** — "폐기" 항목(D8)을 숨기지 말고 넣는다. 결과 산출 전에
  전제 오류를 잡은 기록이 완벽한 결과보다 신뢰를 준다.
- **그림 선별** — 미인용 5장과 D8b 그림 부재를 함께 검토한다.
