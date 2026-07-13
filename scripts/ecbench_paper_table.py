"""
EC 예측 논문 비교 테이블 (논문용)

!! 주의: 각 논문은 데이터셋·메트릭·프로토콜이 달라 직접 비교 불가 !!
   → 같은 테스트셋/메트릭 기준으로 그룹화하여 비교

출처별 정리:
  1. CLEAN 비교 트랙   : New-392 / Price-149, F1 (per-protein avg)
  2. EC-Bench 비교 트랙: Swiss-Prot 2023-01, Weighted F1, temporal split
  3. HIT-EC 비교 트랙  : Swiss-Prot+PDB ~200k, Micro F1, 반복 hold-out

사용법:
  python scripts/ecbench_paper_table.py
"""
import json
from pathlib import Path

ROOT = Path("/home/user/Desktop/unlv")

# ── 트랙 1: CLEAN 비교 (New-392 / Price-149) ─────────────────
# 출처: Yang et al. (2024) Commun. Bio. — CLEAN-Contact 논문, Fig.2a/2b
#       Yu et al. (2023) Science — CLEAN 자체 보고
# 메트릭: F1 (P-value EC number selection algorithm)
# 평가셋: New-392 (392 Swiss-Prot proteins), Price-149 (149 OOD)
#
# !! 주의: CLEAN Science2023 자체보고 F1=0.865(New-392)는 다른 EC 선택 알고리즘 사용
#         CLEAN-Contact논문 에서 CLEAN을 재평가한 F1=0.504가 공정한 기준
BASELINES_NEW392 = {
    # ── 논문 본문에서 직접 확인된 수치 ────────────────────────────
    "CLEAN-Contact": {"f1": 0.566, "prec": 0.652, "rec": 0.555, "auroc": 0.777,
                      "note": "Commun.Bio 2024, p.2 text (P-value selection)"},
    "CLEAN":         {"f1": 0.504, "prec": 0.561, "rec": 0.509, "auroc": 0.753,
                      "note": "Commun.Bio 2024, p.2 text (P-value selection)"},
    # ProteInfer: CLEAN Science2023 본문 "ProteInfer...0.309" 확인
    "ProteInfer":    {"f1": 0.309,
                      "note": "CLEAN Science 2023, p.2 text (max-sep selection)"},
    # DeepEC: CLEAN Science2023 본문 "DeepEC had scores of...0.230" 확인
    # ※ CLEAN-Contact의 Fig.2a에서는 0.102로 다르게 나타남 (평가방법 상이)
    "DeepEC (CLEAN paper)":    {"f1": 0.230,
                                 "note": "CLEAN Science 2023, p.2 text (max-sep)"},
    # ── Figure에서만 읽은 수치 (본문 미확인, 참고용) ─────────────
    # DeepECtransformer: Commun.Bio 2024 Fig.2a 에서만 읽음
    "DeepECtransformer (fig)": {"f1": 0.465,
                                 "note": "Commun.Bio 2024, Fig.2a only — unverified"},
    # ── 참고: CLEAN 자체 Swiss-Prot split (다른 프로토콜) ──────────
    "CLEAN (Science2023 max-sep, own split)": {
        "f1": 0.865,
        "note": "Science 2023, p.1 text — own Swiss-Prot split <50% id, NOT New-392"},
}
BASELINES_PRICE149 = {
    # ── 논문 본문에서 직접 확인된 수치 ────────────────────────────
    "CLEAN-Contact": {"f1": 0.525, "prec": 0.621, "rec": 0.513, "auroc": 0.756,
                      "note": "Commun.Bio 2024, p.2 text (P-value selection)"},
    "CLEAN":         {"f1": 0.452, "prec": 0.531, "rec": 0.434, "auroc": 0.717,
                      "note": "Commun.Bio 2024, p.2 text (P-value selection)"},
    # ── Figure에서만 읽은 수치 (참고용) ──────────────────────────
    "DeepECtransformer (fig)": {"f1": 0.333,
                                 "note": "Commun.Bio 2024, Fig.2b only — unverified"},
    # ECPred: 제거 — 0.138 / 0.311은 잘못된 수치
    # CLEAN-Contact 본문: "ECPred merged F1=0.038, Recall=0.020" 와 모순
}

# ── 트랙 2: EC-Bench 비교 (Weighted F1, temporal split) ───────
# 출처: Davoudi et al. (2026) Bioinformatics Advances
# 메트릭: Weighted F1 (EC-Bench 공식)
# 평가셋: train=Swiss-Prot 2018-02 → test=Swiss-Prot 2023-01 (468개)
#
# 출처: Davoudi et al. (2026) Bioinformatics Advances — vbag004
# 모든 수치 논문 본문 p.7 text에서 직접 확인 ✅
# 메트릭: Weighted F1 at Level 4 ("learnt" = class-specific threshold)
# 테스트셋: Swiss-Prot 2023-01, 468 samples, 147 EC numbers
BASELINES_ECBENCH_100 = {   # 100% similarity threshold (test ↔ train 유사 허용)
    "Stacked":        {"weighted_f1": 0.639, "note": "EC-Bench vbag004 p.7 text ✅"},
    "BLASTp":         {"weighted_f1": 0.502, "note": "EC-Bench vbag004 p.7 text ✅"},
    "EnzBert-learnt": {"weighted_f1": 0.475, "note": "EC-Bench vbag004 p.7 text ✅"},
}
BASELINES_ECBENCH_30 = {    # 30% threshold (높은 신규성, 어려운 조건)
    "Stacked":        {"weighted_f1": 0.524, "note": "EC-Bench vbag004 p.7 text ✅"},
    "EnzBert-learnt": {"weighted_f1": 0.463, "note": "EC-Bench vbag004 p.7 text ✅"},
    "BLASTp":         {"weighted_f1": 0.431, "note": "EC-Bench vbag004 p.7 text ✅"},
}

# ── 트랙 3: HIT-EC 비교 (Micro F1, cross-val) ─────────────────
# 출처: Dumontet et al. (2026) Nature Commun. — Table 1, Fig.2
# 메트릭: Micro F1 (± std, n=10 반복 stratified hold-out)
# 프로토콜: Swiss-Prot+PDB ~200K, ≤1023aa, EC class N≥10, 반복 stratified hold-out
# ※ 우리 프로토콜과 다름 (temporal split) — 직접 비교 불가
BASELINES_HITEC = {
    # cross-validation (Fig. 2A / Table 1)
    "HIT-EC":  {"micro_f1": 0.932, "macro_f1": 0.836, "rare_ec_f1": 0.77,
                "l1":0.953, "l2":0.949, "l3":0.942, "l4":0.932},
    "CLEAN":   {"micro_f1": 0.875, "macro_f1": 0.802, "rare_ec_f1": 0.73,
                "l1":0.968, "l2":0.938, "l3":0.882, "l4":0.875},
    "ECPICK":  {"micro_f1": 0.818, "macro_f1": 0.752, "rare_ec_f1": 0.67,
                "l1":0.927, "l2":0.886, "l3":0.856, "l4":0.818},
    "DeepECT": {"micro_f1": 0.787, "macro_f1": 0.585, "rare_ec_f1": 0.47,
                "l1":0.784, "l2":0.773, "l3":0.768, "l4":0.787},
    # external val — New-28245 (pre-trained 공개모델, Fig. 4C)
    # HIT-EC=0.932, CLEAN=0.889, ECPICK=0.877, DeepECT=0.806 (different test set)
}

# ── 참고: ECPick 자체 평가 수치 ──────────────────────────────
# 출처: Han et al. (2024) Briefings in Bioinformatics
# 평가셋: 858 Swiss-Prot 신규 단백질 (Aug 2020 ~ Apr 2021)
# 메트릭: micro-avg F1 (예측 샘플만) = 0.876, 전체 F1 = 0.566
# → 다른 프로토콜이므로 별도 참고만
ECPICK_SELF = {
    "precision_predicted": 0.8759,  # FDR<0.05 예측된 407개 기준
    "f1_all_samples": 0.5660,       # 전체 858개 기준
    "note": "Briefings Bioinformatics 2024, Table S2"
}


def load_our_results():
    """우리 모델 결과 JSON 파일 로드."""
    result_dir = ROOT / "outputs/results"
    results = {"new392": {}, "price149": {}, "ecbench": {}, "sim_eval": {}}

    # New-392 / Price-149 결과
    for f in result_dir.glob("new392_eval_*.json"):
        with open(f) as fp:
            d = json.load(fp)
        tag = f.stem.replace("new392_eval_", "")
        if "new392"   in d.get("results", {}):
            results["new392"][tag]   = d["results"]["new392"]
        if "price149" in d.get("results", {}):
            results["price149"][tag] = d["results"]["price149"]

    # EC-Bench 결과
    for f in result_dir.glob("ecbench_eval_*.json"):
        with open(f) as fp:
            d = json.load(fp)
        tag = f.stem.replace("ecbench_eval_", "")
        if "swissprot_2023" in d.get("results", {}):
            results["ecbench"][tag] = d["results"]["swissprot_2023"]

    # Similarity-stratified 결과
    sim_file = result_dir / "sim_eval_swissprot.json"
    if sim_file.exists():
        with open(sim_file) as fp:
            results["sim_eval"] = json.load(fp)

    return results


def print_track1(our_results):
    print("\n" + "=" * 65)
    print("  트랙 1: New-392 / Price-149 비교 (CLEAN 논문 F1)")
    print("  ※ CLEAN 정의: per-protein F1 평균, 우리 모델: micro F1")
    print("=" * 65)

    print(f"\n  [New-392]  (392 Swiss-Prot proteins, 2022+ novel)")
    print(f"  {'방법':<22} {'F1':>8}  {'출처'}")
    print(f"  {'-'*22} {'-'*8}  {'-'*30}")
    for name, v in sorted(BASELINES_NEW392.items(), key=lambda x: -x[1]["f1"]):
        print(f"  {name:<22} {v['f1']:>8.3f}  {v['note']}")
    print(f"  {'-'*22}")
    for tag, v in sorted(our_results["new392"].items(),
                         key=lambda x: -x[1].get("micro_f1", 0)):
        f1 = v.get("micro_f1", 0)
        print(f"  Ours ({tag[:16]:<16}) {f1:>8.4f}  (N={v.get('n_samples',0)})")

    print(f"\n  [Price-149]  (149 OOD bacterial proteins)")
    print(f"  {'방법':<22} {'F1':>8}  {'출처'}")
    print(f"  {'-'*22} {'-'*8}  {'-'*30}")
    for name, v in sorted(BASELINES_PRICE149.items(), key=lambda x: -x[1]["f1"]):
        print(f"  {name:<22} {v['f1']:>8.3f}  {v['note']}")
    print(f"  {'-'*22}")
    for tag, v in sorted(our_results["price149"].items(),
                         key=lambda x: -x[1].get("micro_f1", 0)):
        f1 = v.get("micro_f1", 0)
        print(f"  Ours ({tag[:16]:<16}) {f1:>8.4f}  (N={v.get('n_samples',0)})")


def print_track2(our_results):
    print("\n" + "=" * 65)
    print("  트랙 2: EC-Bench Temporal Split (Weighted F1)")
    print("  train=Swiss-Prot 2018-02 → test=Swiss-Prot 2023-01")
    print("=" * 65)

    for label, baselines in [("100% similarity (전체)", BASELINES_ECBENCH_100),
                              ("30% similarity (신규)", BASELINES_ECBENCH_30)]:
        print(f"\n  [{label}]")
        print(f"  {'방법':<22} {'Weighted F1':>12}")
        print(f"  {'-'*22} {'-'*12}")
        for name, v in sorted(baselines.items(), key=lambda x: -x[1]["weighted_f1"]):
            print(f"  {name:<22} {v['weighted_f1']:>12.3f}")
        print(f"  {'-'*22}")
        for tag, v in sorted(our_results["ecbench"].items(),
                             key=lambda x: -x[1].get("weighted_f1", 0)):
            wf1 = v.get("weighted_f1", 0)
            print(f"  Ours ({tag[:16]:<16}) {wf1:>12.4f}")


def print_track3(our_results):
    print("\n" + "=" * 75)
    print("  트랙 3: HIT-EC Cross-Val 참고 수치")
    print("  ※ 프로토콜 다름(temporal split vs cross-val) — 직접 비교 불가")
    print("=" * 75)
    hdr = f"  {'방법':<14} {'Micro':>7} {'Macro':>7} {'Rare(≤25)':>10}  L1    L2    L3    L4"
    print(hdr)
    print(f"  {'-'*14} {'-'*7} {'-'*7} {'-'*10}  {'-'*4}  {'-'*4}  {'-'*4}  {'-'*4}")
    for name, v in sorted(BASELINES_HITEC.items(), key=lambda x: -x[1]["micro_f1"]):
        print(f"  {name:<14} {v['micro_f1']:>7.3f} {v['macro_f1']:>7.3f} "
              f"{v['rare_ec_f1']:>10.3f}  "
              f"{v['l1']:.3f}  {v['l2']:.3f}  {v['l3']:.3f}  {v['l4']:.3f}")
    print(f"  {'-'*14}")
    for tag, v in sorted(our_results["ecbench"].items(),
                         key=lambda x: -x[1].get("micro_f1", 0)):
        mf1  = v.get("micro_f1",    0)
        maf1 = v.get("macro_f1",    0)
        rrf1 = v.get("rare_ec_f1") or 0
        print(f"  {'Ours('+tag[:10]+')':.<14} {mf1:>7.4f} {maf1:>7.4f} "
              f"{rrf1:>10.4f}  (EC-Bench temporal, N={v.get('n_samples',0)})")


def print_sim_eval(our_results):
    sim = our_results.get("sim_eval", {})
    if not sim or "models" not in sim:
        print("\n  [Similarity-stratified 결과 없음 — ecbench_eval_by_similarity.py 실행 필요]")
        return

    print("\n" + "=" * 65)
    print(f"  Similarity-stratified Micro F1 ({sim.get('split','')})")
    print("=" * 65)

    models = sim["models"]
    header = f"  {'Threshold':<10}"
    for mname in models:
        header += f"  {mname[:18]:>18}"
    print(header)
    print(f"  {'-'*10}" + f"  {'-'*18}" * len(models))

    for lbl in ["≤10%", "≤30%", "≤50%", "≤70%", "≤100%"]:
        row = f"  {lbl:<10}"
        for mname in models:
            val = models[mname].get(lbl, {})
            f1  = val.get("micro_f1")
            row += f"  {f1:>18.4f}" if f1 is not None else f"  {'N/A':>18}"
        print(row)


def main():
    our_results = load_our_results()

    print_track1(our_results)
    print_track2(our_results)
    print_track3(our_results)
    print_sim_eval(our_results)

    print("\n\n  ── 참고 ──────────────────────────────────────────────")
    print(f"  ECPick (Briefings Bioinform. 2024) 자체 평가:")
    print(f"    예측 샘플(407/858) micro F1: {ECPICK_SELF['precision_predicted']:.4f}")
    print(f"    전체 샘플(858개)   F1:        {ECPICK_SELF['f1_all_samples']:.4f}")
    print(f"    평가 프로토콜: 다름 (Swiss-Prot Aug2020-Apr2021 신규 단백질)")


if __name__ == "__main__":
    main()
