# 왜 실패하는가 — 수익률 가설의 실패와 변동성 차이의 관측

> 골격 문서. 각 절의 판단 문장은 사람이 작성한다.
> 수치는 `reports/day14_inputs.md`(B1~B6)와 그 출처 CSV에서 옮긴 값이며
> 이 문서에서 새로 계산하지 않는다.

---

## §0. 용어 정의

이 문서에서 "실패"는 5개 신호 × 4개 지평의 수익률 검정에서 신호일의 사후 수익률이 무작위 대비 다르게 나타난다는 가설이 지지되지 않았다는 뜻이며, 신호가 다른 양에서도 아무런 정보를 갖지 않는다는 뜻은 아니다. "발견"은 D13에서 검정한 신호일과 대조군 사이의 사후 경로변동성 σ_post 차이를 가리키며, 그 차이가 통계적으로 관측되었다는 사실을 넘어 실용적 수익성이나 인과적 메커니즘까지 입증했다는 뜻은 아니다.

---

## §1. 수익률 결과 요약 — 무엇이 기각됐는가

수익률 분석에서는 5개 신호 × 4개 지평의 20개 조합 전체에서 무작위 대비 방향성 초과수익이 있다는 근거가 남지 않았다. D6의 일부 유의 결과는 D7에서 분산 차이를 고려한 스튜던트화 검정을 거치며 유지되지 않았고, D12에서도 직전 변동성을 층화한 검정에서 사전 설정한 판독 기준을 통과한 조합이 없었다. 따라서 여기서 말하는 "실패"는 이 20개 조합에서 수익률의 방향성을 예측한다는 가설의 실패를 뜻한다. S1은 이 중 이후 변동성 분석까지 이어진 대표 사례이며, §2 이후에서는 S1을 중심으로 그 관계를 따로 살핀다.

### 1-0. 신호별 발동 건수

이 표는 뒤의 수익률·변동성 결과를 해석할 때 각 신호가 실제로 얼마나 자주 발생했는지와 분석에 사용된 신호 사건의 규모를 확인하기 위해 둔다.

| 항목명 | 출처 파일 | 확정 수치 | B-ID |
|---|---|---|:---:|
| S1 RSI<30 | `reports/day06_signal_counts.csv` | 레벨 일수 126 / 레벨 발동률 1.8851 % / 사건 수 n 65 / 평균 지속일수 1.9385 | B6-1 |
| S2 RSI>70 | `reports/day06_signal_counts.csv` | 레벨 일수 426 / 레벨 발동률 6.3734 % / 사건 수 n 137 / 평균 지속일수 3.1095 | B6-1 |
| S3 볼린저 하단 | `reports/day06_signal_counts.csv` | 레벨 일수 368 / 레벨 발동률 5.5057 % / 사건 수 n 209 / 평균 지속일수 1.7608 | B6-1 |
| S4 볼린저 상단 | `reports/day06_signal_counts.csv` | 레벨 일수 321 / 레벨 발동률 4.8025 % / 사건 수 n 191 / 평균 지속일수 1.6806 | B6-1 |
| S5 MACD>Signal | `reports/day06_signal_counts.csv` | 레벨 일수 3360 / 레벨 발동률 50.2693 % / 사건 수 n 283 / 평균 지속일수 11.8728 | B6-1 |

### 1-1. D6 순열검정 20조합

이 표는 최초 분석에서 5개 신호와 4개 지평의 조합 중 어떤 결과가 유의하게 보였는지를 확인하고, 이후 D7에서 왜 강건성 검정이 필요했는지를 보여주기 위해 둔다.

| 항목명 | 출처 파일 | 확정 수치 | B-ID |
|---|---|---|:---:|
| S1 RSI<30 h=1 | `reports/day06_permutation.csv` | 차이 -0.502037 %p / p 0.001900 / Bonferroni True | B1 |
| S1 RSI<30 h=5 | `reports/day06_permutation.csv` | 차이 -0.197561 %p / p 0.510049 / Bonferroni False | B1 |
| S1 RSI<30 h=10 | `reports/day06_permutation.csv` | 차이 -0.026388 %p / p 0.948105 / Bonferroni False | B1 |
| S1 RSI<30 h=20 | `reports/day06_permutation.csv` | 차이 1.188019 %p / p 0.035996 / Bonferroni False | B1 |
| S2 RSI>70 h=1 | `reports/day06_permutation.csv` | 차이 -0.026255 %p / p 0.790221 / Bonferroni False | B1 |
| S2 RSI>70 h=5 | `reports/day06_permutation.csv` | 차이 -0.025273 %p / p 0.900210 / Bonferroni False | B1 |
| S2 RSI>70 h=10 | `reports/day06_permutation.csv` | 차이 0.091886 %p / p 0.739926 / Bonferroni False | B1 |
| S2 RSI>70 h=20 | `reports/day06_permutation.csv` | 차이 0.308302 %p / p 0.421758 / Bonferroni False | B1 |
| S3 볼린저 하단 h=1 | `reports/day06_permutation.csv` | 차이 -0.049185 %p / p 0.547245 / Bonferroni False | B1 |
| S3 볼린저 하단 h=5 | `reports/day06_permutation.csv` | 차이 0.061967 %p / p 0.710629 / Bonferroni False | B1 |
| S3 볼린저 하단 h=10 | `reports/day06_permutation.csv` | 차이 -0.142898 %p / p 0.528447 / Bonferroni False | B1 |
| S3 볼린저 하단 h=20 | `reports/day06_permutation.csv` | 차이 0.237417 %p / p 0.451955 / Bonferroni False | B1 |
| S4 볼린저 상단 h=1 | `reports/day06_permutation.csv` | 차이 -0.083017 %p / p 0.338766 / Bonferroni False | B1 |
| S4 볼린저 상단 h=5 | `reports/day06_permutation.csv` | 차이 -0.243943 %p / p 0.167583 / Bonferroni False | B1 |
| S4 볼린저 상단 h=10 | `reports/day06_permutation.csv` | 차이 -0.257886 %p / p 0.279372 / Bonferroni False | B1 |
| S4 볼린저 상단 h=20 | `reports/day06_permutation.csv` | 차이 -0.383456 %p / p 0.240976 / Bonferroni False | B1 |
| S5 MACD>Signal h=1 | `reports/day06_permutation.csv` | 차이 -0.000978 %p / p 0.988901 / Bonferroni False | B1 |
| S5 MACD>Signal h=5 | `reports/day06_permutation.csv` | 차이 -0.070741 %p / p 0.619038 / Bonferroni False | B1 |
| S5 MACD>Signal h=10 | `reports/day06_permutation.csv` | 차이 -0.090112 %p / p 0.649335 / Bonferroni False | B1 |
| S5 MACD>Signal h=20 | `reports/day06_permutation.csv` | 차이 -0.208135 %p / p 0.434657 / Bonferroni False | B1 |

### 1-2. D7 스튜던트화 20조합

이 표는 D6에서 관측된 수익률 차이가 분산 차이를 고려한 뒤에도 유지되는지 확인하기 위한 검증 결과를 보여주기 위해 둔다.

| 항목명 | 출처 파일 | 확정 수치 | B-ID |
|---|---|---|:---:|
| S1 RSI<30 h=1 | `reports/day07_studentized.csv` | sd_ratio 2.143736 / p_raw 0.001900 / p_stud 0.124388 / 판정_stud - | B2 |
| S1 RSI<30 h=5 | `reports/day07_studentized.csv` | sd_ratio 1.772629 / p_raw 0.510049 / p_stud 0.719828 / 판정_stud - | B2 |
| S1 RSI<30 h=10 | `reports/day07_studentized.csv` | sd_ratio 1.526071 / p_raw 0.948105 / p_stud 0.966203 / 판정_stud - | B2 |
| S1 RSI<30 h=20 | `reports/day07_studentized.csv` | sd_ratio 1.458782 / p_raw 0.035996 / p_stud 0.162184 / 판정_stud - | B2 |
| S2 RSI>70 h=1 | `reports/day07_studentized.csv` | sd_ratio 0.505852 / p_raw 0.790221 / p_stud 0.620838 / 판정_stud - | B2 |
| S2 RSI>70 h=5 | `reports/day07_studentized.csv` | sd_ratio 0.488359 / p_raw 0.900210 / p_stud 0.808719 / 판정_stud - | B2 |
| S2 RSI>70 h=10 | `reports/day07_studentized.csv` | sd_ratio 0.549628 / p_raw 0.739926 / p_stud 0.550345 / 판정_stud - | B2 |
| S2 RSI>70 h=20 | `reports/day07_studentized.csv` | sd_ratio 0.635613 / p_raw 0.421758 / p_stud 0.218578 / 판정_stud - | B2 |
| S3 볼린저 하단 h=1 | `reports/day07_studentized.csv` | sd_ratio 1.527536 / p_raw 0.547245 / p_stud 0.698830 / 판정_stud - | B2 |
| S3 볼린저 하단 h=5 | `reports/day07_studentized.csv` | sd_ratio 1.513921 / p_raw 0.710629 / p_stud 0.806619 / 판정_stud - | B2 |
| S3 볼린저 하단 h=10 | `reports/day07_studentized.csv` | sd_ratio 1.336287 / p_raw 0.528447 / p_stud 0.633037 / 판정_stud - | B2 |
| S3 볼린저 하단 h=20 | `reports/day07_studentized.csv` | sd_ratio 1.361763 / p_raw 0.451955 / p_stud 0.577142 / 판정_stud - | B2 |
| S4 볼린저 상단 h=1 | `reports/day07_studentized.csv` | sd_ratio 0.534852 / p_raw 0.338766 / p_stud 0.076092 / 판정_stud - | B2 |
| S4 볼린저 상단 h=5 | `reports/day07_studentized.csv` | sd_ratio 0.613524 / p_raw 0.167583 / p_stud 0.024598 / 판정_stud - | B2 |
| S4 볼린저 상단 h=10 | `reports/day07_studentized.csv` | sd_ratio 0.642488 / p_raw 0.279372 / p_stud 0.093991 / 판정_stud - | B2 |
| S4 볼린저 상단 h=20 | `reports/day07_studentized.csv` | sd_ratio 0.717936 / p_raw 0.240976 / p_stud 0.105289 / 판정_stud - | B2 |
| S5 MACD>Signal h=1 | `reports/day07_studentized.csv` | sd_ratio 0.802887 / p_raw 0.988901 / p_stud 0.983802 / 판정_stud - | B2 |
| S5 MACD>Signal h=5 | `reports/day07_studentized.csv` | sd_ratio 0.801913 / p_raw 0.619038 / p_stud 0.546645 / 판정_stud - | B2 |
| S5 MACD>Signal h=10 | `reports/day07_studentized.csv` | sd_ratio 0.964362 / p_raw 0.649335 / p_stud 0.634837 / 판정_stud - | B2 |
| S5 MACD>Signal h=20 | `reports/day07_studentized.csv` | sd_ratio 0.971529 / p_raw 0.434657 / p_stud 0.424358 / 판정_stud - | B2 |

### 1-3. D12 층화 순열검정 4지평

이 표는 직전 변동성을 비슷한 수준으로 맞춘 뒤에도 S1의 사후 수익률 차이가 남는지를 확인한 추가 검증 결과를 보여주기 위해 둔다.

| 항목명 | 출처 파일 | 확정 수치 | B-ID |
|---|---|---|:---:|
| S1 층화 h=1 | `reports/day12_results.csv` | 효과 -0.582053 %p / t -1.735666 / p 0.079792 / alpha_read 0.0125 / 유의 False | B4 |
| S1 층화 h=1 판정 항목 | `reports/day12_results.csv` | flag_dominance True (loo_max_p 0.303470) / flag_sign False / flag_overlap False (층 없음) | B4 |
| S1 층화 h=5 | `reports/day12_results.csv` | 효과 -0.316062 %p / t -0.562011 / p 0.572543 / alpha_read 0.0125 / 유의 False | B4 |
| S1 층화 h=5 판정 항목 | `reports/day12_results.csv` | flag_dominance True (loo_max_p 0.424558) / flag_sign False / flag_overlap False (층 없음) | B4 |
| S1 층화 h=10 | `reports/day12_results.csv` | 효과 -0.108463 %p / t -0.166170 / p 0.868913 / alpha_read 0.0125 / 유의 False | B4 |
| S1 층화 h=10 판정 항목 | `reports/day12_results.csv` | flag_dominance True (loo_max_p 0.582242) / flag_sign False / flag_overlap False (층 없음) | B4 |
| S1 층화 h=20 | `reports/day12_results.csv` | 효과 0.957472 %p / t 1.102715 / p 0.268573 / alpha_read 0.0125 / 유의 False | B4 |
| S1 층화 h=20 판정 항목 | `reports/day12_results.csv` | flag_dominance True (loo_max_p 0.646135) / flag_sign False / flag_overlap True (층 5.0) | B4 |

### 1-4. 국면별 분해 — 별개 항목

이 표는 수익률 검정 자체의 결과를 대신하는 것이 아니라, 각 신호가 특정 시장 국면에 얼마나 집중되어 있는지를 확인하기 위한 별도 진단으로 둔다.

`reports/day06_diag_regime.csv`는 국면(regime) 단위 분해이며 연도별 분포의
대체가 아니다. `reports/day14_inputs.md` B6-2에 같은 취지로 기록돼 있다.

| 항목명 | 출처 파일 | 확정 수치 | B-ID |
|---|---|---|:---:|
| S1 RSI<30 / 닷컴 붕괴 | `reports/day06_diag_regime.csv` | n_days 695 / n_sig 15 / rate 0.021583 / rate_ratio 2.2194 / share_sig 0.2308 / share_days 0.1040 | B6-2 |
| S1 RSI<30 / 회복·확장 | `reports/day06_diag_regime.csv` | n_days 1258 / n_sig 7 / rate 0.005564 / rate_ratio 0.5722 / share_sig 0.1077 / share_days 0.1882 | B6-2 |
| S1 RSI<30 / 금융위기 | `reports/day06_diag_regime.csv` | n_days 355 / n_sig 9 / rate 0.025352 / rate_ratio 2.6070 / share_sig 0.1385 / share_days 0.0531 | B6-2 |
| S1 RSI<30 / 장기 강세 | `reports/day06_diag_regime.csv` | n_days 2756 / n_sig 21 / rate 0.007620 / rate_ratio 0.7835 / share_sig 0.3231 / share_days 0.4123 | B6-2 |
| S1 RSI<30 / 코로나 | `reports/day06_diag_regime.csv` | n_days 135 / n_sig 3 / rate 0.022222 / rate_ratio 2.2851 / share_sig 0.0462 / share_days 0.0202 | B6-2 |
| S1 RSI<30 / 최근 국면 | `reports/day06_diag_regime.csv` | n_days 1485 / n_sig 10 / rate 0.006734 / rate_ratio 0.6925 / share_sig 0.1538 / share_days 0.2222 | B6-2 |
| S2 RSI>70 / 닷컴 붕괴 | `reports/day06_diag_regime.csv` | n_days 695 / n_sig 2 / rate 0.002878 / rate_ratio 0.1404 / share_sig 0.0146 / share_days 0.1040 | B6-2 |
| S2 RSI>70 / 회복·확장 | `reports/day06_diag_regime.csv` | n_days 1258 / n_sig 19 / rate 0.015103 / rate_ratio 0.7369 / share_sig 0.1387 / share_days 0.1882 | B6-2 |
| S2 RSI>70 / 금융위기 | `reports/day06_diag_regime.csv` | n_days 355 / n_sig 0 / rate 0.000000 / rate_ratio 0.0000 / share_sig 0.0000 / share_days 0.0531 | B6-2 |
| S2 RSI>70 / 장기 강세 | `reports/day06_diag_regime.csv` | n_days 2756 / n_sig 78 / rate 0.028302 / rate_ratio 1.3808 / share_sig 0.5693 / share_days 0.4123 | B6-2 |
| S2 RSI>70 / 코로나 | `reports/day06_diag_regime.csv` | n_days 135 / n_sig 2 / rate 0.014815 / rate_ratio 0.7228 / share_sig 0.0146 / share_days 0.0202 | B6-2 |
| S2 RSI>70 / 최근 국면 | `reports/day06_diag_regime.csv` | n_days 1485 / n_sig 36 / rate 0.024242 / rate_ratio 1.1827 / share_sig 0.2628 / share_days 0.2222 | B6-2 |
| S3 볼린저 하단 / 닷컴 붕괴 | `reports/day06_diag_regime.csv` | n_days 695 / n_sig 40 / rate 0.057554 / rate_ratio 1.8406 / share_sig 0.1914 / share_days 0.1040 | B6-2 |
| S3 볼린저 하단 / 회복·확장 | `reports/day06_diag_regime.csv` | n_days 1258 / n_sig 25 / rate 0.019873 / rate_ratio 0.6355 / share_sig 0.1196 / share_days 0.1882 | B6-2 |
| S3 볼린저 하단 / 금융위기 | `reports/day06_diag_regime.csv` | n_days 355 / n_sig 20 / rate 0.056338 / rate_ratio 1.8017 / share_sig 0.0957 / share_days 0.0531 | B6-2 |
| S3 볼린저 하단 / 장기 강세 | `reports/day06_diag_regime.csv` | n_days 2756 / n_sig 74 / rate 0.026851 / rate_ratio 0.8587 / share_sig 0.3541 / share_days 0.4123 | B6-2 |
| S3 볼린저 하단 / 코로나 | `reports/day06_diag_regime.csv` | n_days 135 / n_sig 4 / rate 0.029630 / rate_ratio 0.9476 / share_sig 0.0191 / share_days 0.0202 | B6-2 |
| S3 볼린저 하단 / 최근 국면 | `reports/day06_diag_regime.csv` | n_days 1485 / n_sig 46 / rate 0.030976 / rate_ratio 0.9907 / share_sig 0.2201 / share_days 0.2222 | B6-2 |
| S4 볼린저 상단 / 닷컴 붕괴 | `reports/day06_diag_regime.csv` | n_days 695 / n_sig 11 / rate 0.015827 / rate_ratio 0.5539 / share_sig 0.0576 / share_days 0.1040 | B6-2 |
| S4 볼린저 상단 / 회복·확장 | `reports/day06_diag_regime.csv` | n_days 1258 / n_sig 40 / rate 0.031797 / rate_ratio 1.1127 / share_sig 0.2094 / share_days 0.1882 | B6-2 |
| S4 볼린저 상단 / 금융위기 | `reports/day06_diag_regime.csv` | n_days 355 / n_sig 4 / rate 0.011268 / rate_ratio 0.3943 / share_sig 0.0209 / share_days 0.0531 | B6-2 |
| S4 볼린저 상단 / 장기 강세 | `reports/day06_diag_regime.csv` | n_days 2756 / n_sig 79 / rate 0.028665 / rate_ratio 1.0031 / share_sig 0.4136 / share_days 0.4123 | B6-2 |
| S4 볼린저 상단 / 코로나 | `reports/day06_diag_regime.csv` | n_days 135 / n_sig 5 / rate 0.037037 / rate_ratio 1.2961 / share_sig 0.0262 / share_days 0.0202 | B6-2 |
| S4 볼린저 상단 / 최근 국면 | `reports/day06_diag_regime.csv` | n_days 1485 / n_sig 52 / rate 0.035017 / rate_ratio 1.2254 / share_sig 0.2723 / share_days 0.2222 | B6-2 |
| S5 MACD>Signal / 닷컴 붕괴 | `reports/day06_diag_regime.csv` | n_days 695 / n_sig 29 / rate 0.041727 / rate_ratio 0.9855 / share_sig 0.1025 / share_days 0.1040 | B6-2 |
| S5 MACD>Signal / 회복·확장 | `reports/day06_diag_regime.csv` | n_days 1258 / n_sig 57 / rate 0.045310 / rate_ratio 1.0701 / share_sig 0.2014 / share_days 0.1882 | B6-2 |
| S5 MACD>Signal / 금융위기 | `reports/day06_diag_regime.csv` | n_days 355 / n_sig 11 / rate 0.030986 / rate_ratio 0.7318 / share_sig 0.0389 / share_days 0.0531 | B6-2 |
| S5 MACD>Signal / 장기 강세 | `reports/day06_diag_regime.csv` | n_days 2756 / n_sig 114 / rate 0.041364 / rate_ratio 0.9770 / share_sig 0.4028 / share_days 0.4123 | B6-2 |
| S5 MACD>Signal / 코로나 | `reports/day06_diag_regime.csv` | n_days 135 / n_sig 5 / rate 0.037037 / rate_ratio 0.8748 / share_sig 0.0177 / share_days 0.0202 | B6-2 |
| S5 MACD>Signal / 최근 국면 | `reports/day06_diag_regime.csv` | n_days 1485 / n_sig 67 / rate 0.045118 / rate_ratio 1.0656 / share_sig 0.2367 / share_days 0.2222 | B6-2 |

### 1-5. 성과 지표 대비

이 표는 통계적 검정 결과와 별개로 이 신호를 실제 전략으로 사용했을 때 시장 대비 의미 있는 성과가 나타나는지를 확인하기 위한 현실성 점검으로 둔다.

전략 20행은 CAGR %, 샤프, `cagr_nocash`, `cash_contrib_pp` 4개 열만 옮겼다.
벤치마크 3행은 `reports/day14_inputs.md` B3 표의 행 전체를 옮겼다.

| 항목명 | 출처 파일 | 확정 수치 | B-ID |
|---|---|---|:---:|
| S1_rsi_oversold h=1 | `reports/day11_metrics.csv` | CAGR 0.1390 % / 샤프 -0.4070 / cagr_nocash -1.7140 / cash_contrib_pp 1.8531 | B3 |
| S1_rsi_oversold h=5 | `reports/day11_metrics.csv` | CAGR 1.4183 % / 샤프 -0.0262 / cagr_nocash -0.4104 / cash_contrib_pp 1.8287 | B3 |
| S1_rsi_oversold h=10 | `reports/day11_metrics.csv` | CAGR 0.9150 % / 샤프 -0.0646 / cagr_nocash -0.8444 / cash_contrib_pp 1.7594 | B3 |
| S1_rsi_oversold h=20 | `reports/day11_metrics.csv` | CAGR 1.6543 % / 샤프 0.0257 / cagr_nocash -0.0568 / cash_contrib_pp 1.7111 | B3 |
| S2_rsi_overbought h=1 | `reports/day11_metrics.csv` | CAGR 0.8274 % / 샤프 -0.7563 / cagr_nocash -1.0060 / cash_contrib_pp 1.8335 | B3 |
| S2_rsi_overbought h=5 | `reports/day11_metrics.csv` | CAGR 1.0153 % / 샤프 -0.2835 / cagr_nocash -0.7238 / cash_contrib_pp 1.7391 | B3 |
| S2_rsi_overbought h=10 | `reports/day11_metrics.csv` | CAGR 2.0504 % / 샤프 0.0493 / cagr_nocash 0.3573 / cash_contrib_pp 1.6931 | B3 |
| S2_rsi_overbought h=20 | `reports/day11_metrics.csv` | CAGR 2.1463 % / 샤프 0.0661 / cagr_nocash 0.5512 / cash_contrib_pp 1.5951 | B3 |
| S3_bb_lower_break h=1 | `reports/day11_metrics.csv` | CAGR -0.0554 % / 샤프 -0.3516 / cagr_nocash -1.8288 / cash_contrib_pp 1.7734 | B3 |
| S3_bb_lower_break h=5 | `reports/day11_metrics.csv` | CAGR 2.9700 % / 샤프 0.1552 / cagr_nocash 1.2936 / cash_contrib_pp 1.6764 | B3 |
| S3_bb_lower_break h=10 | `reports/day11_metrics.csv` | CAGR 1.9974 % / 샤프 0.0642 / cagr_nocash 0.5054 / cash_contrib_pp 1.4920 | B3 |
| S3_bb_lower_break h=20 | `reports/day11_metrics.csv` | CAGR 5.4728 % / 샤프 0.3218 / cagr_nocash 4.1784 / cash_contrib_pp 1.2944 | B3 |
| S4_bb_upper_break h=1 | `reports/day11_metrics.csv` | CAGR -0.0251 % / 샤프 -1.0633 / cagr_nocash -1.8036 / cash_contrib_pp 1.7785 | B3 |
| S4_bb_upper_break h=5 | `reports/day11_metrics.csv` | CAGR 0.0266 % / 샤프 -0.4057 / cagr_nocash -1.5745 / cash_contrib_pp 1.6011 | B3 |
| S4_bb_upper_break h=10 | `reports/day11_metrics.csv` | CAGR -0.0297 % / 샤프 -0.2879 / cagr_nocash -1.4511 / cash_contrib_pp 1.4214 | B3 |
| S4_bb_upper_break h=20 | `reports/day11_metrics.csv` | CAGR 1.4465 % / 샤프 -0.0126 / cagr_nocash 0.2634 / cash_contrib_pp 1.1831 | B3 |
| S5_macd_cross h=1 | `reports/day11_metrics.csv` | CAGR -0.1091 % / 샤프 -0.6148 / cagr_nocash -1.8324 / cash_contrib_pp 1.7233 | B3 |
| S5_macd_cross h=5 | `reports/day11_metrics.csv` | CAGR -0.1684 % / 샤프 -0.2442 / cagr_nocash -1.5802 / cash_contrib_pp 1.4118 | B3 |
| S5_macd_cross h=10 | `reports/day11_metrics.csv` | CAGR -0.1041 % / 샤프 -0.1413 / cagr_nocash -1.2298 / cash_contrib_pp 1.1257 | B3 |
| S5_macd_cross h=20 | `reports/day11_metrics.csv` | CAGR 2.3241 % / 샤프 0.0950 / cagr_nocash 1.4787 / cash_contrib_pp 0.8454 | B3 |
| ^GSPC Buy&Hold (가격지수, 배당 제외) | `reports/day11_metrics.csv` | 노출도 — % / CAGR 6.3718 % / 샤프 0.3180 / MDD -56.7754 % / cagr_nocash — / cash_contrib_pp — | B3 |
| ^SP500TR Buy&Hold (총수익지수) — 기준선 | `reports/day11_metrics.csv` | 노출도 — % / CAGR 8.3413 % / 샤프 0.4133 / MDD -55.2502 % / cagr_nocash — / cash_contrib_pp — | B3 |
| ^IRX 전액 보유 (무위험수익률만) | `reports/day11_metrics.csv` | 노출도 — % / CAGR 1.9292 % / 샤프 -0.2069 / MDD -0.0015 % / cagr_nocash — / cash_contrib_pp — | B3 |

벤치마크 3행의 `cagr_nocash`와 `cash_contrib_pp`는 원본 CSV `reports/day11_metrics.csv`에서 공란이다.

---

## §2. 변동성 결과 요약 — 무엇이 확인됐는가

수익률에서는 5개 신호 × 4개 지평 전체에서 방향성 예측 가설을 지지하는 근거가 확인되지 않았지만, S1에 대해서는 이후 경로변동성이라는 다른 종속변수에서 차이가 관측되었다. D13에서는 신호일의 사후 경로변동성 σ_post가 대조군보다 큰 차이가 나타났으며, 이는 수익률 방향 예측 실패와 별개의 결과다.

### 2-1. D13 사후 경로변동성 3행

이 표는 D13의 주검정, 클러스터 축약 민감도, W=10 보조 분석을 함께 비교하여 변동성 차이가 각 분석에서 어떤 형태로 나타났는지를 확인하기 위해 둔다.

| 항목명 | 출처 파일 | 확정 수치 | B-ID |
|---|---|---|:---:|
| main_W20 (W=20) | `reports/day13_pathvol_results.csv` | delta 0.357981 / exp(delta) 1.430438 / SE 0.052810 / delta/SE 6.778623 / p_raw 0.000100 / p_stud 0.000100 / delta_pre 0.026917 / n_sig 원자료 65 · 포함 62 | B5 |
| declustered_W20 (W=20) | `reports/day13_pathvol_results.csv` | delta 0.444396 / exp(delta) 1.559547 / SE 0.074495 / delta/SE 5.965464 / p_raw 0.000100 / p_stud 0.000100 / delta_pre -0.008309 / n_sig 원자료 34 · 포함 31 | B5 |
| aux_W10 (W=10) | `reports/day13_pathvol_results.csv` | delta 0.387460 / exp(delta) 1.473234 / SE 0.054641 / delta/SE 7.091045 / p_raw — / p_stud — / delta_pre 0.026927 / n_sig 원자료 65 · 포함 62 | B5 |

### 2-2. D13 §4 진단 6종 (main_W20 행에만 저장)

이 표는 주효과의 크기만 확인하는 것이 아니라, 층 지배·신호군 창 중첩·층 내 사전 변동성 차이·스튜던트화 변화 등 결과를 해석할 때 함께 확인해야 할 구조적 진단을 확인하기 위해 둔다.

| 항목명 | 출처 파일 | 확정 수치 | B-ID |
|---|---|---|:---:|
| 4-1 층 지배도 max_k w_k | `reports/day13_pathvol_results.csv` | 0.516129 / flag True | B5 |
| 4-2 부호 불일치 가중치 | `reports/day13_pathvol_results.csv` | 0.000000 / flag False | B5 |
| 4-3 신호군 창 중첩 | `reports/day13_pathvol_results.csv` | Q3;Q4;Q5 / flag True | B5 |
| 4-4 층 내 delta_pre | `reports/day13_pathvol_results.csv` | 0.026917 / flag False | B5 |
| 4-5 스튜던트화 전후 p 변화 | `reports/day13_pathvol_results.csv` | abs(log(p_stud/p_raw)) 0.000000 / 교차 False / flag False | B5 |
| 4-6 원자료 유효 신호 수 | `reports/day13_pathvol_results.csv` | 65 / flag False | B5 |

---

## §3. 수익률 실패와 변동성 결과의 관계

D8에서 D8b로 간 과정은 결과가 좋지 않아서 종속변수를 임의로 바꾼 것이 아니라, D7에서 근거로 사용한 `sd_ratio` = 2.14가 원래 질문하려던 사건별 사후 실현변동성을 측정한 값이 아니었다는 정의 오류를 결과 산출 전에 발견하고 바로잡은 과정이다.

D7의 2.14는 S1 h=1에서 나온 사건 간 사후 수익률의 횡단면 표준편차 비율이었다. 따라서 이를 사건별 사후 실현변동성의 증거로 사용한 D8의 전제가 성립하지 않았고, D8은 결과를 확인하기 전에 폐기되었다.

D8b에서는 이 문제를 해결하기 위해 사건마다 실제 이후 수익률 경로에서 변동성을 계산하는 `post_vol`을 종속변수로 정의했다. 즉, 결과에 맞춰 유리한 종속변수를 선택한 것이 아니라, 처음부터 묻고 있던 "신호 뒤에 변동성이 커지는가?"라는 질문에 맞는 측정량으로 설계를 수정한 것이다.

D13에서는 여기서 한 단계 더 나아가 그 변동성 차이를 명시적인 추정량, 대조군, 층화, 스튜던트화 및 중첩 처리 절차로 정식화했다. 다만 D13은 D8b에서 이미 같은 방향의 변동성 차이를 관찰한 뒤 이루어진 분석이므로, 완전히 독립적인 확증적 발견으로 표현해서는 안 된다.

### 3-1. D8 → D8b → D13 경과

이 표는 D8부터 D13까지의 변경 사항을 단순한 버전 기록으로 남기는 것이 아니라, 왜 분석 대상과 설계가 단계적으로 수정되었는지를 시간순으로 추적하기 위해 둔다.

| 날짜 | 문서 / 커밋 | 무엇을 바꿨는가 |
|---|---|---|
| 2026-08-08 | `docs/prereg_d08_stratified.md` (`415e377`) | D8 사전등록. §1·§2가 대상 양을 "사후 20일 실현변동성"으로 전제 |
| 2026-08-08 | 같은 문서 폐기 블록 | 근거로 삼은 D7 `sd_ratio` 2.14가 (a) h=1의 값이며 (b) 사건별 실현변동성이 아니라 사건 간 사후 수익률의 횡단면 표준편차임을 확인. 결과 미확인 상태에서 폐기 |
| 2026-08-08 | `docs/prereg_d08b_pathvol.md` (`f2f2947`) | 대체 사전등록. 종속변수를 사건별 경로변동성 `post_vol`(창 `r_{t+2} … r_{t+21}`)로 재정의하고 대조군을 비신호일로 한정 |
| 2026-08-08 | `reports/day08b_pathvol.csv` (`1d15e3e`) | D8b 실행. 층화 전/후 배율 산출 |
| 2026-08-12 | `docs/prereg_day13.md` (`c10a81f`) | D13 사전등록. 추정량·스튜던트화·ATT형 층화 가중·중첩 처리·민감도를 명시 |
| 2026-08-12 | 같은 문서 폐기 블록 (`ad2cce1`) | §1.1 표와 §2.1이 σ_pre를 20일 창 [t−20, t−1]로 정의했으나 §2.1·§2.5·§3.4의 확정 수치는 14일 창 [t−13, t]에서 나온 값임을 확인. σ_post 미산출 상태에서 폐기 |
| 2026-08-12 | `docs/prereg_day13b.md` (`ad2cce1`) | 대체 사전등록. σ_pre를 14일 창 [t−13, t]로 통일하고 σ_post 창을 `r_{t+1} … r_{t+20}`으로 명시 |
| 2026-08-12 | `reports/day13_pathvol_results.csv` (`92bc93b`) | D13b 실행. 주검정 / 클러스터 축약 민감도 / W=10 보조 산출 |

D8의 폐기는 분석 결과를 보고 질문을 바꾼 것이 아니다. D8 사전등록 당시 근거로 삼은 D7의 `sd_ratio` = 2.14가 h=1의 사건 간 사후 수익률 횡단면 표준편차였으며, 사건별 사후 실현변동성이 아니었다는 점을 결과 산출 전에 확인했기 때문이다.

D8b는 이 정의 오류를 바로잡아 사건별 사후 경로변동성 자체를 측정하도록 설계를 바꾼 것이다. 이후 D13에서는 σ_pre와 σ_post의 정의, 층화 방법, 추정량, 스튜던트화 및 중첩 처리 등을 명시하여 이 질문을 더 엄격하게 검증했다.

따라서 D8 → D8b → D13은 실패한 결과를 수습하기 위해 종속변수를 바꾼 과정이라기보다, 처음 사용한 측정량이 질문과 맞지 않았음을 발견하고 질문에 맞는 측정량과 검정 설계를 단계적으로 정교화한 과정으로 기록한다.

§3의 2.14는 `reports/day07_studentized.csv`의 S1 h=1 행 `sd_ratio` 값(2.143736)이며, 본 문서 §1-2 표(B2)에 그대로 실려 있다.

---

## §4. 이월 항목

출처: `docs/prereg_day13b.md` §5.5.
아래 항목은 현재 결과에서 확인되었지만, 이를 확정적인 발견이나 새로운 가설의 증거로 확대하지 않는다.

### §4.1 표본 부족으로 크기/방향 불확실

이 항목들은 클러스터 축약으로 표본이 65건에서 31건으로 줄어든 뒤 관측된 효과 크기와 층별 패턴을 다루므로, 표본 감소에 따른 불확실성 문제로 묶는다. 따라서 아래 관찰값은 기록하되 축약 후 효과의 크기나 층별 방향을 확정적으로 해석하지 않는다.

- 축약 후 효과 증가 (1.430 → 1.560). 층당 7~12건이라 확정적으로 해석하지 않는다.
- 축약 후 Q5의 Δ_pre가 −0.044로 부호가 반전되었음에도 exp(Δ_k)는 1.781로 가장 컸다. 표본 12건이므로 확정적으로 해석하지 않는다.
- 대조군의 높은 창 중첩으로 인해 발생하는 유효표본 감소 문제.

### §4.2 설계 선택 미검증

현재 설계가 가능한 모든 설계 중 최선이라고 결론 내릴 수 없다는 점에서 하나의 그룹으로 묶는다. 아래 항목들은 현재 분석에서 확인된 사실이 아니라, 다른 설계 선택에서도 결과가 유지되는지를 확인해야 하는 미검증 사항이다.

- σ_pre 대안 모형(EWMA·GARCH) 대비 우위 미검증.
- 클러스터 축약의 단일 위상 사용에 따른 한계.
- D8b와 D13의 σ_post 창이 하루 어긋나는 설계 차이.

### §4.3 메커니즘 미규명

현재 결과에서 관측된 패턴의 존재와 그 패턴이 발생한 원인을 구분하기 위해 별도 항목으로 둔다. 현재 분석은 변동성 차이를 관측했지만 그 원인이 무엇인지는 검증하지 않았으므로, 아래 항목을 특정 메커니즘의 증거로 해석하지 않는다.

- W=10 1.473 / W=20 1.430으로 방향 일치. 창이 짧을수록 배율이 큰 이유는 미확인.

### §4.4 미분류 항목 — 사후 관찰과 미검증 설계는 별도로 유지

Δ_k의 단조 증가와 σ_pre_14의 경로 형태는 각각 사후 관찰과 향후 검증할 설계 질문이라는 성격이 달라 기존 세 분류에 억지로 넣지 않는다. 따라서 현재의 확정 결과나 사전 가설의 일부로 재분류하지 않고 별도 미분류 항목으로 유지한다.

- Δ_k의 층별 단조 증가 (1.173 / 1.418 / 1.530). 사후 관찰이며 사전 예측 항목이 아님. "발견"이라 쓰지 않는다.
- 향후 분석에서 σ_pre_14의 경로 형태를 추가로 통제할 필요성.

특히 결과를 확인한 뒤 발견한 패턴을 새로운 가설인 것처럼 소급하여 해석하지 않으며, 향후 검증이 필요하다면 새로운 분석 또는 별도의 사전등록 대상으로 취급한다.

---

## §5. 예상 질문 — 근거 위치 매핑

이 표는 결론을 읽는 사람이 제기할 가능성이 높은 질문에 대해 실제 근거가 문서의 어느 절에 있는지를 연결하기 위해 둔다.

| 예상 질문 | 근거 위치 |
|---|---|
| 1. D6에서는 유의했는데 왜 수익률 예측 실패라고 하는가? | §1-1 D6 순열검정, §1-2 D7 스튜던트화, §1-3 D12 층화 순열검정 |
| 2. D8에서 왜 변동성을 종속변수로 바꿨는가? 결과를 보고 유리한 방향으로 바꾼 것 아닌가? | §3, `logs/2026-08-08.md`, `docs/prereg_d08_stratified.md` 폐기 블록, `docs/prereg_d08b_pathvol.md` |
| 3. D13의 변동성 결과는 정말 새로운 발견인가? | §2-1, §3-1, `docs/prereg_day13b.md` §0. D8b에서 이미 변동성 차이를 관찰했으므로 완전히 독립적인 확증 결과로 표현하지 않는다. |
| 4. D13에서 변동성을 통제했는데도 어떤 한계가 남아 있는가? | §2-2, §4.1~§4.4. 층 지배, 창 중첩, 표본 부족, 설계 선택 미검증 및 메커니즘 미규명 등을 함께 확인한다. |
| 5. D13 결과를 어디까지 믿을 수 있는가? | §2-1~§2-2 및 §4.1~§4.4. 주검정에서 변동성 차이는 관측되었지만 그 크기와 일반화 가능성에는 별도의 한계가 남아 있다. |

---

## §6. 결론

현재까지의 결과에서 가장 안전하게 말할 수 있는 것은 5개 신호 × 4개 지평의 수익률 방향 예측 가설은 지지되지 않았다는 것이다. 반면 S1에서는 수익률이 아닌 사후 경로변동성 σ_post라는 다른 양에서 신호군과 대조군 사이의 차이가 관측되었다.

따라서 "RSI<30이 이후 수익률을 예측한다"고 결론 내릴 근거는 없지만, "RSI<30이 이후 변동성과 아무런 관계가 없다"고 결론 내릴 수도 없다. 현재 자료에서 확인된 것은 수익률 방향 예측의 실패와 변동성 차이의 관측이 동시에 존재한다는 것이다.

다만 D13의 변동성 결과는 D8b에서 이미 같은 방향의 현상을 관찰한 뒤 이루어진 분석이므로, 이를 완전히 독립적인 확증적 발견으로 표현하지 않는다. 또한 §4의 미검증 항목과 사후 관찰을 새로운 사전 가설의 증거로 소급하지 않는다.

다음 분석에서 필요한 것은 현재 결과를 더 강하게 주장하는 것이 아니라, 관측된 변동성 차이가 다른 설계에서도 유지되는지와 그 차이가 어떤 조건에서 발생하는지를 별도로 검증하는 것이다.
