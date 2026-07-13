"""
Price-149 평가 데이터 준비:
  1. ESM-2 임베딩 추출 → data/processed/embeddings/{id}.npy
  2. ESMFold 구조 예측 → contact map → data/processed/contact_maps/{id}.npy
  3. dataset_meta.csv에 Price-149 행 추가 (별도 파일로 저장)
"""
import os, sys, json, pickle, argparse
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def step1_embeddings(price_df, device="cpu"):
    from transformers import AutoTokenizer, EsmModel
    import torch

    print(f"[Step 1] ESM-2 임베딩 추출 ({device}) ...")
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    model = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D").to(device)
    model.eval()

    embed_dir = ROOT / "data" / "processed" / "embeddings"
    done, skip = 0, 0
    with torch.no_grad():
        for _, row in price_df.iterrows():
            out_path = embed_dir / f"{row['Entry']}.npy"
            if out_path.exists():
                skip += 1
                continue
            enc = tokenizer(row["Sequence"], return_tensors="pt",
                            truncation=True, max_length=1024).to(device)
            emb = model(**enc).last_hidden_state[:, 0, :].squeeze().cpu().numpy()
            np.save(str(out_path), emb.astype(np.float32))
            done += 1
            if done % 10 == 0:
                print(f"  {done + skip}/{len(price_df)}", flush=True)
    print(f"  완료: {done}개 추출, {skip}개 스킵")


def step2_esmfold(price_df, device="cuda:1"):
    """ESMFold로 구조 예측 → contact map 생성."""
    from transformers import EsmForProteinFolding, AutoTokenizer as AT
    from scipy.ndimage import zoom
    import torch

    print(f"[Step 2] ESMFold 구조 예측 ({device}) ...")
    tokenizer = AT.from_pretrained("facebook/esmfold_v1")
    fold_model = EsmForProteinFolding.from_pretrained(
        "facebook/esmfold_v1", low_cpu_mem_usage=True
    ).to(device)
    fold_model.eval()
    fold_model.trunk.set_chunk_size(64)

    cmap_dir = ROOT / "data" / "processed" / "contact_maps"
    done, skip = 0, 0

    with torch.no_grad():
        for _, row in price_df.iterrows():
            out_path = cmap_dir / f"{row['Entry']}.npy"
            if out_path.exists():
                skip += 1
                continue
            seq = row["Sequence"][:1024]
            try:
                tok = tokenizer([seq], return_tensors="pt",
                                add_special_tokens=False).to(device)
                out = fold_model(**tok)
                # plddt에서 pseudo-contact map (pairwise distance < 8Å)
                pos = out.positions[-1].squeeze(0).cpu().numpy()  # (L, 37, 3) or (L,3)
                if pos.ndim == 3:
                    pos = pos[:, 1, :]   # CA atom
                diff = pos[:, None, :] - pos[None, :, :]
                dist = np.sqrt((diff**2).sum(-1))
                cmap = (dist < 8.0).astype(np.float32)
                # 256×256으로 resize
                L = cmap.shape[0]
                if L != 256:
                    scale = 256 / L
                    cmap = zoom(cmap, (scale, scale), order=1)
                cmap = cmap[:256, :256].astype(np.float32)
                np.save(str(out_path), cmap)
                done += 1
                if done % 10 == 0:
                    print(f"  {done + skip}/{len(price_df)}", flush=True)
            except Exception as e:
                print(f"  [{row['Entry']}] 실패: {e}")
    print(f"  완료: {done}개 예측, {skip}개 스킵")


def step3_meta(price_df, label_enc_pkl):
    """Price-149용 meta CSV 생성.
    EC number가 ';'로 구분된 다중 EC도 지원 (멀티레이블).
    우리 vocab에 없는 EC는 제외 (mask=0)."""
    print("[Step 3] Price-149 메타데이터 생성 ...")
    with open(label_enc_pkl, "rb") as f:
        encoders = pickle.load(f)

    l4_cls_list = list(encoders["level4"].classes_)
    l4_cls_set  = {c: i for i, c in enumerate(l4_cls_list)}
    l1_cls_set  = {c: i for i, c in enumerate(encoders["level1"].classes_)}
    l2_cls_set  = {c: i for i, c in enumerate(encoders["level2"].classes_)}
    l3_cls_set  = {c: i for i, c in enumerate(encoders["level3"].classes_)}

    rows = []
    missing_ec = []
    for _, row in price_df.iterrows():
        ec_raw = str(row["EC number"]).strip()
        # ';' 구분 다중 EC 처리
        ec_list = [e.strip() for e in ec_raw.split(";") if e.strip()]

        # 유효한 L4 인덱스 수집
        l4_valid = []
        for ec in ec_list:
            parts = ec.split(".")
            if len(parts) == 4 and ec in l4_cls_set:
                l4_valid.append(ec)

        if not l4_valid:
            missing_ec.append((row["Entry"], ec_raw))
            continue

        # 대표 EC (첫 번째 유효 EC)
        ec_rep   = l4_valid[0]
        p        = ec_rep.split(".")
        l1s = p[0]
        l2s = f"{p[0]}.{p[1]}"
        l3s = f"{p[0]}.{p[1]}.{p[2]}"

        l4_idxs = [l4_cls_set[e] for e in l4_valid]

        rows.append({
            "accession":   row["Entry"],
            "sequence":    row["Sequence"],
            "seq_len":     len(row["Sequence"]),
            "ec_chosen":   ec_rep,
            "l1_str":      l1s, "l2_str": l2s, "l3_str": l3s, "l4_str": ec_rep,
            "m1": int(l1s in l1_cls_set),
            "m2": int(l2s in l2_cls_set),
            "m3": int(l3s in l3_cls_set),
            "m4": 1,
            "l1_idx": l1_cls_set.get(l1s, -1),
            "l2_idx": l2_cls_set.get(l2s, -1),
            "l3_idx": l3_cls_set.get(l3s, -1),
            "l4_idx": l4_idxs[0],
            "l4_all_idxs": ",".join(str(i) for i in l4_idxs),
        })

    meta_df = pd.DataFrame(rows)
    out_path = ROOT / "data" / "processed" / "price149_meta.csv"
    meta_df.to_csv(out_path, index=False)

    # split 파일
    ids = meta_df["accession"].tolist()
    (ROOT / "data" / "splits" / "price149_ids.txt").write_text("\n".join(ids))

    print(f"  저장: {out_path}  ({len(meta_df)}개)")
    if missing_ec:
        print(f"  EC 인식 실패: {missing_ec}")
    return meta_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", choices=["embed", "fold", "meta", "all"],
                        default="embed")
    parser.add_argument("--gpu", type=int, default=1)
    args = parser.parse_args()

    price_df     = pd.read_csv("/tmp/price149.csv")
    label_enc    = ROOT / "data" / "label_encoders.pkl"

    if args.step in ("embed", "all"):
        step1_embeddings(price_df, device="cpu")

    if args.step in ("meta", "all"):
        step3_meta(price_df, label_enc)

    if args.step in ("fold", "all"):
        step2_esmfold(price_df, device=f"cuda:{args.gpu}")
        # fold 후 meta 재생성 (contact map 있는 버전)
        step3_meta(price_df, label_enc)

    print("\n=== 준비 완료 ===")
