"""
Ablation 결과 요약 테이블 생성.
outputs/results/에 있는 JSON들을 읽어 논문용 표 출력.
"""
import json, os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULT_DIR = ROOT / "outputs" / "results"

# 모델 이름 → 논문 이름
MODEL_LABELS = {
    "b0_cnn_test_hier_results":         "B0: 1D-CNN (서열 one-hot)",
    "b1_esm2_fc_test_hier_results":     "B1: ESM-2 + FC",
    "b2_esm2_hier_test_hier_results":   "B2: ESM-2 + Hier-FC",
    "b2_ml_test_hier_results":          "B2: ESM-2 + Hier-FC (multilabel)",
    "b3_contact_test_hier_results":     "B3: Contact Map + ResNet",
    "fusion_v2_ml_test_hier_results":   "V2: ESM-2 + Contact + Hier-FC (ML)",
    "fusion_v3_ml_test_hier_results":   "V3: ESM-2 + Pair-Contact + Hier-FC (ML)",
    "fusion_test_hier_results":         "Fusion: ESM-2 + Contact (original)",
}

PRICE149_LABELS = {
    "b2_esm2_hier_price149_hier_results":   "B2: ESM-2 + Hier-FC",
    "b2_ml_price149_hier_results":          "B2: ESM-2 + Hier-FC (multilabel)",
    "fusion_v2_ml_price149_hier_results":   "V2: ESM-2 + Contact + Hier-FC (ML)",
    "fusion_v3_ml_price149_hier_results":   "V3: ESM-2 + Pair-Contact + Hier-FC (ML)",
}

# CLEAN 기준점
CLEAN_BENCHMARKS = {
    "price149": {"CLEAN (Yu et al., 2023)": 0.499},
    "new392":   {"CLEAN (Yu et al., 2023)": 0.533},
}


def load_result(json_path):
    with open(json_path) as f:
        return json.load(f)


def print_table(title, rows, columns):
    """rows: list of (name, {col: val}) """
    col_w = max(len(c) for c in columns) + 2
    name_w = max(len(r[0]) for r in rows) + 2
    header = f"{'Model':<{name_w}}" + "".join(f"{c:>{col_w}}" for c in columns)
    print(f"\n{'='*len(header)}")
    print(f"  {title}")
    print(f"{'='*len(header)}")
    print(header)
    print("-" * len(header))
    for name, vals in rows:
        row = f"{name:<{name_w}}"
        for c in columns:
            v = vals.get(c, "—")
            row += f"{v:>{col_w}}"
        print(row)
    print("-" * len(header))


def main():
    print("\n" + "=" * 70)
    print("  논문 결과 요약 — EC Number Multilabel Prediction")
    print("=" * 70)

    # ── 내부 Test 셋 결과 ────────────────────────────────────
    test_rows = []
    for stem, label in MODEL_LABELS.items():
        p = RESULT_DIR / f"{stem}.json"
        if not p.exists():
            continue
        r = load_result(p)
        row = {}
        for lvl in ["level1", "level2", "level3"]:
            if lvl in r:
                row[f"L{lvl[-1]} Acc"]    = f"{r[lvl]['accuracy']:.4f}"
                row[f"L{lvl[-1]} Micro F1"] = f"{r[lvl]['micro_f1']:.4f}"
        if "level4" in r:
            r4 = r["level4"]
            row["L4 Micro F1"] = f"{r4.get('micro_f1', r4.get('f1', 0)):.4f}"
            row["L4 Macro F1"] = f"{r4.get('macro_f1', 0):.4f}"
            row["L4 Prec"]     = f"{r4['precision']:.4f}" if "precision" in r4 else "—"
            row["L4 Rec"]      = f"{r4['recall']:.4f}"    if "recall"    in r4 else "—"
        if "level4_underrepresented" in r:
            row["Rare F1"] = f"{r['level4_underrepresented']['micro_f1']:.4f}"
        test_rows.append((label, row))

    if test_rows:
        cols = ["L1 Acc", "L2 Acc", "L3 Acc", "L4 Micro F1", "L4 Macro F1", "Rare F1"]
        print_table("내부 Test 셋 결과 (random split)", test_rows, cols)

    # ── Price-149 결과 (CLEAN 공정 비교) ──────────────────────
    p149_rows = []
    for stem, label in PRICE149_LABELS.items():
        p = RESULT_DIR / f"{stem}.json"
        if not p.exists():
            continue
        r = load_result(p)
        row = {}
        if "level4" in r:
            row["L4 Micro F1"] = f"{r['level4']['micro_f1']:.4f}"
            row["L4 Macro F1"] = f"{r['level4']['macro_f1']:.4f}"
            row["Prec"]        = f"{r['level4']['precision']:.4f}"
            row["Recall"]      = f"{r['level4']['recall']:.4f}"
        p149_rows.append((label, row))

    # CLEAN 비교 기준점 추가
    p149_rows.append(("── CLEAN Baseline ──", {}))
    p149_rows.append(("CLEAN (Yu et al., Science 2023)", {
        "L4 Micro F1": "0.499",
        "L4 Macro F1": "—",
        "Prec": "—",
        "Recall": "—",
    }))

    if any(v for _, v in p149_rows if v):
        cols = ["L4 Micro F1", "L4 Macro F1", "Prec", "Recall"]
        print_table("Price-149 공정 평가 (0% 훈련 오버랩, CLEAN 비교)", p149_rows, cols)
        print("  ※ CLEAN F1 출처: Yu et al. Science 2023, max-sep threshold, Price 149 set")
    else:
        print("\n  [Price-149 결과 없음 — B2/V2/V3 훈련 후 재실행]")

    # ── 아직 없는 결과 파일 목록 ───────────────────────────────
    all_expected = list(MODEL_LABELS) + list(PRICE149_LABELS)
    missing = [s for s in all_expected if not (RESULT_DIR / f"{s}.json").exists()]
    if missing:
        print(f"\n  [미완성 결과 파일 ({len(missing)}개) — 훈련 후 생성 예정]")
        for m in missing:
            print(f"    - {m}.json")


if __name__ == "__main__":
    main()
