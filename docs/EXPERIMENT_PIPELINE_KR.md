# Contact-EC 남은 실험 파이프라인

작성 기준: 2026-07-30 KST

## 최근 완료

### A. Foldseek/TM-score strictness sweep

목적: Fold-disjoint 기준을 바꿔도 Contact-EC fusion의 상대적 이득이 유지되는지 검증한다.

상태:

- TM-score 0.40 split 생성 완료
- TM-score 0.60 split 생성 완료
- TM-score 0.40 B1/B3/Fusion 학습/평가 완료
- TM-score 0.50 B1/B3/Fusion 기존 대표 split 결과 정리 완료
- TM-score 0.60 B1/B3/Fusion 학습/평가 완료
- Foldseek sweep split-composition audit 완료
- Main/Supplement에 결과 반영 완료

핵심 결과:

- TM-score 0.40: B1 0.0683, B3 0.2622, Fusion 0.2981
- TM-score 0.50: B1 0.0462, B3 0.0337, Fusion 0.0998
- TM-score 0.60: B1 0.0538, B3 0.3985, Fusion 0.4403
- 모든 split에서 fusion이 가장 높음
- 단, 0.50 split은 cluster-count assignment이고 unseen positive label 비율이 31.5%로 높아, 0.40/0.60과 직접적인 monotonic threshold-only 비교로 해석하면 안 됨

해석:

- Foldseek-disjoint 조건에서도 sequence-structure fusion의 상대적 이득은 유지됨
- Contact-only가 0.40/0.60에서 강하게 나와 structural topology signal의 근거가 강화됨
- 하지만 MMseqs2 homology-transfer baseline이 여전히 더 강하므로, 논문 framing은 homology replacement가 아니라 benchmark decomposition/fusion analysis로 유지해야 함

산출물:

- `outputs/results/*foldseek_tmscore40_cc*_hier_results.json`
- `outputs/results/*foldseek_tmscore60_cc*_hier_results.json`
- `outputs/audit/foldseek_tmscore_sweep_summary.csv`
- `outputs/audit/foldseek_tmscore_sweep_summary.md`
- `outputs/audit/foldseek_tmscore_sweep_split_composition.csv`
- `outputs/audit/foldseek_tmscore_sweep_split_composition.md`

재실행:

```bash
python scripts/collect_foldseek_tmscore_sweep.py
python scripts/audit_foldseek_sweep_split_composition.py
```

진행 상태 확인:

```bash
./scripts/run_high_quality_pipeline_status.sh
```

논문 반영:

- Main: Foldseek-disjoint 결과 문단에 0.40/0.50/0.60 robustness 및 split-composition caution 추가 완료
- Supplement: TM-score sweep neural table 및 split-composition table 추가 완료

## 1단계: 빠르게 완성도를 올리는 후처리 분석

### B. Fusion rescue case study

목적: Fusion만 맞힌 단백질을 실제 생물학적 사례로 설명한다.

현재 완료:

- Fusion-only rescue protein 후보 추출 완료
- EC Level-1/2 family 확인 완료
- training label frequency 확인 완료
- true EC가 seen/rare인지 확인 완료
- Supplement에 대표 사례 table 추가 완료

남은 작업:

- 필요 시 contact map 또는 Grad-CAM/attention figure 연결

예상 시간:

- figure까지 만들 경우 2-4시간
- table-only 원고 반영은 완료

현재 산출물:

- `outputs/audit/foldseek_tmscore50_cc_fusion_rescue_case_studies.md`
- `outputs/audit/foldseek_tmscore50_cc_fusion_rescue_case_studies.csv`
- `outputs/audit/foldseek_tmscore50_cc_fusion_rescue_case_studies.fasta`
- `outputs/audit/foldseek_tmscore50_cc_fusion_rescue_case_studies_family_summary.csv`

후보 추출 재실행:

```bash
python scripts/select_fusion_rescue_case_studies.py
```

논문 효과:

- “왜 structure fusion이 도움이 되는가”에 대한 biological interpretation 강화
- Bioinformatics Advances/BMC Bioinformatics 방어력 상승

### C. Abstention/open-set analysis 본문 연결

현재 완료:

- Fusion confidence로 unseen true Level-4 label 감지: AUROC 0.738, AP 0.590
- 낮은 confidence 50% 보류 시 retained Level-4 micro F1: 0.208 -> 0.305

남은 작업:

- 본문 Discussion에 1-2문장 연결
- Supplement table caption 다듬기

예상 시간:

- 1-2시간

## 2단계: 리뷰어 방어용 핵심 실험

### D0. Simple fusion architecture baselines

목적: 현재 Contact-EC flat FC의 gated additive cross-attention block이 단순한 feature combination보다 실제로 유리한지 검증한다.

현재 상태:

- 동일 contact encoder와 동일 flat FC head를 사용하는 세 가지 baseline 구현 완료
- `fusion_concat_flatfc`: ESM/contact projected feature concat 후 MLP
- `fusion_sum_flatfc`: ESM/contact projected feature sum
- `fusion_gated_mlp_flatfc`: ESM/contact projected feature 사이 feature-wise gate
- seed 42/43/44 반복 실행 스크립트와 결과 수집 스크립트 작성 완료
- seed 42/43/44 반복 실행 완료

핵심 결과:

- Contact-EC flat FC: 0.6241 ± 0.0170
- Concat flat FC: 0.5922 ± 0.0150
- Sum flat FC: 0.5505 ± 0.0230
- Gated MLP flat FC: 0.5543 ± 0.0525
- 해석: Contact-EC가 단순 concat/sum/gated MLP baseline보다 높지만, concat과의 차이는 크지 않으므로 architecture novelty claim은 보수적으로 서술해야 함

예상 시간:

- 완료

현재 산출물:

- `models/fusion_simple_baselines.py`
- `scripts/run_simple_fusion_baseline_seed_repeats.sh`
- `scripts/collect_simple_fusion_seed_repeats.py`
- `outputs/audit/simple_fusion_seed_repeats.csv`
- `outputs/audit/simple_fusion_seed_repeats.md`

진행/결과 확인:

```bash
python scripts/collect_simple_fusion_seed_repeats.py
tail -80 outputs/audit/simple_fusion_seed_repeats.md
```

논문 효과:

- fusion architecture claim 방어력 강화
- pooled-vector attention 구조가 과도한 모델링인지 확인 가능
- Bioinformatics Advances/BMC Bioinformatics reviewer가 요구할 가능성이 높은 simple baseline 비교를 선제적으로 제공

논문 반영 필요:

- Main temporal results 문단에 simple fusion baseline 결과 추가
- Supplement에 simple fusion baseline mean±std table 추가
- Methods에 세 baseline 정의 추가
- Discussion에서 Contact-EC의 architecture claim을 "complex attention novelty"가 아니라 "controlled learned fusion"으로 보수화

### D. Direct baseline 확장: BLAST/DIAMOND

목적: PLM/fusion 모델이 단순 homology baseline과 비교해 어떤 위치인지 명확히 한다.

현재 완료:

- MMseqs2가 설치되어 있어 DIAMOND 대신 MMseqs2 top-hit EC-transfer baseline을 먼저 수행
- Foldseek TM-score 0.40/0.50/0.60 train-test FASTA 생성 완료
- Foldseek TM-score 0.40/0.50/0.60 MMseqs2 search 완료
- top-hit EC transfer metric/table 생성 완료
- Foldseek identity-bin breakdown 완료
- temporal known-EC/Price-149 external homology baseline 완료
- Supplement에 homology baseline table 추가 완료

핵심 결과:

- TM-score 0.40: L4 micro F1 0.6936
- TM-score 0.50: L4 micro F1 0.5393
- TM-score 0.60: L4 micro F1 0.8176
- Swiss-Prot 2023 known-EC 124: L4 micro F1 0.5852
- Price-149 encoded 136: L4 micro F1 0.3446
- Price-149 raw 149: L4 micro F1 0.3119
- 해석: Foldseek split에서도 top-hit homology transfer가 neural closed-set models보다 강하므로, 본 논문은 homology replacement가 아니라 benchmark decomposition/fusion rescue 분석으로 framing해야 함
- temporal known-EC에서는 MMseqs2가 Contact-EC 0.6032와 근접하므로 temporal 성능은 homology coverage의 영향을 강하게 받는다고 해석해야 함
- Price-149에서는 MMseqs2도 lower-level hit는 높지만 Level-4 micro F1이 낮아, 외부 benchmark 실패가 단순 empty prediction 문제가 아니라 세밀한 EC specificity shift임을 보여줌

남은 작업:

- 필요하면 DIAMOND/BLAST를 추가해 MMseqs2 결과와 교차검증

예상 시간:

- Foldseek MMseqs2 baseline: 완료
- identity-bin breakdown: 완료
- temporal/Price-149 확장: 완료
- DIAMOND/BLAST 교차검증: 선택 사항, 2-6시간

현재 산출물:

- `outputs/baselines/homology/homology_baseline_manifest.csv`
- `outputs/baselines/homology/foldseek_tmscore40_cc_train.fasta`
- `outputs/baselines/homology/foldseek_tmscore40_cc_test.fasta`
- `outputs/baselines/homology/foldseek_tmscore50_cc_train.fasta`
- `outputs/baselines/homology/foldseek_tmscore50_cc_test.fasta`
- `outputs/baselines/homology/foldseek_tmscore60_cc_train.fasta`
- `outputs/baselines/homology/foldseek_tmscore60_cc_test.fasta`
- `outputs/baselines/homology/foldseek_tmscore40_cc_mmseqs_top10.tsv`
- `outputs/baselines/homology/foldseek_tmscore50_cc_mmseqs_top10.tsv`
- `outputs/baselines/homology/foldseek_tmscore60_cc_mmseqs_top10.tsv`
- `outputs/baselines/homology/sp2018_temporal_known124_mmseqs_top10.tsv`
- `outputs/baselines/homology/sp2018_price149_encoded136_mmseqs_top10.tsv`
- `outputs/baselines/homology/sp2018_price149_raw149_mmseqs_top10.tsv`
- `outputs/baselines/homology/mmseqs_homology_baseline_summary.csv`
- `outputs/baselines/homology/mmseqs_homology_baseline_summary.md`
- `outputs/baselines/homology/mmseqs_homology_identity_bins.csv`
- `outputs/baselines/homology/mmseqs_homology_identity_bins.md`
- `outputs/baselines/homology/external_homology_baseline_manifest.csv`

입력 재생성:

```bash
python scripts/prepare_homology_baseline_inputs.py
```

MMseqs2 baseline 재실행:

```bash
./scripts/run_mmseqs_homology_baseline.sh
```

참고:

- `diamond`와 `blastp`는 현재 PATH에 없음
- 현재는 설치 없이 사용 가능한 MMseqs2 baseline을 우선 채택

논문 효과:

- benchmark analysis paper로서 baseline 약점 완화
- reviewer의 “homology baseline과 비교했는가?” 질문 방어
- 다만 결과가 매우 강하므로 neural method claim은 보수적으로 조정해야 함

### E. Main temporal set seed repeat

목적: main temporal result가 single seed에 의존하지 않음을 보인다.

내용:

- B1/B3/Fusion main temporal setting 3 seeds
- mean ± std
- paired bootstrap CI는 기존 방식 재사용

현재 상태:

- `b1_esm2_fc`, `b3_contact`, `fusion_v2_flatfc` 대상으로 seed 42/43/44 반복 실험 스크립트 작성 완료
- seed 42/43/44 반복 실행 완료
- 평가 subset은 complete Level-4 Swiss-Prot 2023-01 temporal proteins N=124
- 평가 결과는 seed별 JSON으로 보존하고, 완료될 때마다 summary CSV/Markdown을 자동 갱신

핵심 결과:

- B1 ESM-2: 0.4508 ± 0.0203
- B3 contact: 0.4244 ± 0.0207
- Contact-EC flat FC: 0.6241 ± 0.0170
- 해석: Contact-EC의 temporal gain은 단일 seed 우연으로 보기 어렵고, B1/B3 대비 안정적인 개선을 보임

예상 시간:

- 완료

현재 산출물:

- `scripts/run_temporal_known_seed_repeats.sh`
- `scripts/collect_temporal_known_seed_repeats.py`
- `outputs/audit/temporal_known_seed_repeats.csv`
- `outputs/audit/temporal_known_seed_repeats.md`

진행/결과 확인:

```bash
python scripts/collect_temporal_known_seed_repeats.py
tail -80 outputs/audit/temporal_known_seed_repeats.md
```

논문 효과:

- statistical robustness 강화
- Bioinformatics Advances/BMC Bioinformatics에는 강한 보강

논문 반영 필요:

- Main Table의 B1/B3/Contact-EC temporal 숫자를 single-run 또는 mean±std 병기 방식으로 정리
- Bootstrap CI 문단과 seed-repeat 문단의 역할을 구분
- Supplement에 per-seed table 추가

## 3단계: Bioinformatics 본지 도전용 대형 실험

### F. Temporal recency decomposition

목적: “data recency effect”가 sample size, label vocabulary, homolog coverage와 섞여 있다는 약점을 분해한다.

권장 디자인:

1. SP-2018 vocabulary 고정 + newer samples 추가
2. sample size matched old/new corpus
3. vocabulary expansion only
4. accession/sequence hash/homolog exclusion audit
5. 2018 -> 2020 -> 2022 -> 2026 cutoff curve

예상 시간:

- 데이터가 이미 정리되어 있으면 4-7일
- UniProt release 재구성부터 해야 하면 1.5-3주
- 모든 cutoff에서 3-seed까지 하면 2-4주

논문 효과:

- Bioinformatics 본지 도전에서 가장 중요한 보강
- 현재 “recency가 backbone보다 크다”는 결론을 causal하게 만들 수 있음

현재 완료된 audit:

- `scripts/audit_recency_cutoff_decomposition.py`
- `scripts/build_recency_intersection_ids.py`
- `scripts/collect_recency_intersection_eval.py`
- `scripts/audit_recency_homology_coverage.py`
- 산출물:
  - `outputs/audit/recency_cutoff_decomposition.md`
  - `outputs/audit/recency_intersection_ids.md`
  - `outputs/audit/recency_intersection_eval.md`
  - `outputs/audit/recency_homology_coverage.md`
  - `outputs/audit/recency_homology_coverage_summary.csv`
  - `outputs/audit/recency_homology_coverage_per_protein.csv`

핵심 발견:

- SP-2018 temporal known set은 124개 모두 Level-4 평가 가능
- SP-2022 encoder도 124개 모두 Level-4 평가 가능
- ExpA/SP-2026 encoder는 124개 중 99개만 Level-4 평가 가능
- ExpA 전체 corpus에는 temporal protein 115개가 포함되지만, ExpA training split에는 accession 및 exact sequence overlap이 0개
- 따라서 ExpA 결과는 124개 결과가 아니라 99개 encoder-evaluable intersection 결과로 표기해야 함

수정된 공정 비교:

- Contact-EC SP-2018 on same N=99: micro F1 `0.6467 ± 0.0210`
- Contact-EC-ExpA SP-2026 on same N=99: micro F1 `0.7417 ± 0.0182` across 3 seeds
- fair-subset recent-corpus gain: `+9.5 pp`
- 기존의 `0.6032 -> 0.7209`, `+11.8 pp` 표현은 denominator가 달라서 main claim으로 쓰면 안 됨
- MMseqs2 nearest-neighbour audit에서 matched N=99 기준 median top-hit identity가 SP-2018 `0.556`에서 SP-2022/ExpA `0.619`로 증가
- 같은 matched N=99 기준 `>=0.60` identity training neighbour는 `38 -> 49`, top-hit exact L4 agreement는 `69 -> 78`로 증가
- 따라서 ExpA improvement는 pure calendar-date effect가 아니라 corpus expansion, homolog availability, label-frequency/coverage 변화가 섞인 recent-corpus effect로 해석해야 함

### G. Simple fusion architecture baselines

목적: 현재 fusion이 단순 concatenation/late fusion보다 좋은지 확인한다.

내용:

- late probability fusion: 후처리로 가능
- concat MLP / gated MLP: 추가 학습 필요
- 현재 fusion과 비교

예상 시간:

- late fusion: 1-3시간
- concat/gated retraining: 1-3일

논문 효과:

- architecture novelty 약점 보완
- ML reviewer의 “왜 이 fusion인가?” 질문 방어

## 추천 실행 순서

### 최소 저널 제출형

1. Foldseek TM-score sweep 완료 및 원고 반영
2. Fusion rescue case study
3. BLAST/DIAMOND baseline
4. Main temporal 3-seed repeat
5. 최종 paper polishing

예상 총 시간:

- 약 4-8일

타겟:

- Bioinformatics Advances
- BMC Bioinformatics
- ACM-BCB main 도전 가능

### Bioinformatics 본지 도전형

1. 최소 저널 제출형 전체 완료
2. Temporal recency decomposition
3. Simple fusion architecture baselines
4. 더 넓은 external benchmark 정리
5. 재현성 패키지 정리

예상 총 시간:

- 빠르면 2-3주
- 데이터 release 재구성까지 포함하면 3-5주

타겟:

- Bioinformatics 상향 도전
- ISMB/ECCB proceedings 상향 도전
- 실패 시 Bioinformatics Advances/BMC Bioinformatics로 전환

## 현재 판단

Foldseek sweep, fusion rescue case study, MMseqs2 homology baseline, external
homology baseline, main temporal 3-seed repeat, simple fusion architecture
baseline은 완료되었다. 현재 원고는 Bioinformatics Advances/BMC
Bioinformatics 수준의 핵심 reviewer 질문에는 상당히 더 잘 대응한다.

다음으로 가장 효율적인 작업은 다음 순서다.

1. 새 3-seed temporal 및 simple fusion baseline 결과를 Main/Supplement에 반영 완료
2. Main PDF를 8쪽 제출 길이로 재압축 완료
3. Open-set/abstention 결과를 본문 Discussion에 연결
4. Fusion rescue case study figure 또는 qualitative contact-map figure 추가 여부 결정
5. GitHub/reproducibility package를 새 실험 결과까지 포함해 갱신

Bioinformatics 본지를 목표로 계속 올리려면 이후에는 multi-cutoff temporal
recency curve와 더 넓은 baseline 재평가가 필요하다.
