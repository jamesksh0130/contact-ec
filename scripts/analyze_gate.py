"""
Contact Map 기여도 분석
- Gate 값 분포: contact map이 실제로 fusion에 기여하는지
- Attention weight: ESM-2가 contact feature를 얼마나 참조하는지
- Contact encoder attention: 어떤 공간 위치를 주목하는지
- EC 레벨별 gate 값: 어떤 클래스에서 구조 정보가 더 중요한지
"""
import argparse, pickle, yaml, sys
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
with open(ROOT / "configs" / "config.yaml") as f:
    CFG = yaml.safe_load(f)

from models.dataset import ProteinDataset, collate_fn
from models.fusion_v2 import FusionModelV2, AttentionContactEncoder

DEVICE = "cuda:1"


# ── 중간값 추출용 모델 래퍼 ──────────────────────────────────
class FusionV2Inspector(FusionModelV2):
    """forward 시 gate, attn_weight, spatial_attn, contact_feat 반환."""
    def forward_inspect(self, esm_emb, cmap):
        # Contact encoder
        feat = self.contact_encoder.features(cmap)        # (B, 2048, 8, 8)
        B, C, H, W = feat.shape

        spatial_attn_logit = self.contact_encoder.attn(feat)           # (B, 1, 8, 8)
        spatial_attn = torch.softmax(
            spatial_attn_logit.view(B, 1, H*W), dim=-1)               # (B, 1, 64)
        feat_flat   = feat.view(B, C, H*W)
        pooled      = (feat_flat * spatial_attn).sum(dim=-1)           # (B, 2048)
        contact_feat = self.contact_encoder.proj(pooled)               # (B, 512)

        # GCA
        esm_proj    = self.esm_proj(esm_emb)                          # (B, 1024)
        contact_kv  = self.contact_kv_proj(contact_feat)              # (B, 1024)
        attn_out, attn_weight = self.cross_attn(
            query=esm_proj.unsqueeze(1),
            key=contact_kv.unsqueeze(1),
            value=contact_kv.unsqueeze(1),
        )                                                               # attn_weight (B,1,1)
        attn_out = attn_out.squeeze(1)
        gate     = torch.sigmoid(self.gate_fc(contact_feat))           # (B, 1024)
        fused    = esm_proj + gate * attn_out

        logits   = self.head(fused)
        l4_pred  = torch.sigmoid(logits[3]).argmax(dim=1)

        return {
            "gate":         gate.cpu(),                  # (B, 1024)
            "attn_weight":  attn_weight.squeeze().cpu(), # (B,) or scalar
            "spatial_attn": spatial_attn.squeeze(1).cpu(), # (B, 64)
            "contact_feat": contact_feat.cpu(),          # (B, 512)
            "esm_proj":     esm_proj.cpu(),              # (B, 1024)
            "attn_out":     attn_out.cpu(),              # (B, 1024)
            "l4_pred":      l4_pred.cpu(),
            "logits":       [l.cpu() for l in logits],
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="outputs/checkpoints/fusion_v2_best.pt")
    parser.add_argument("--split", default="val")
    parser.add_argument("--n_batches", type=int, default=50,
                        help="분석할 배치 수 (전체=0)")
    args = parser.parse_args()

    # 모델 로드
    with open(ROOT / CFG["paths"]["label_enc"], "rb") as f:
        encoders = pickle.load(f)
    n_classes = [len(encoders[f"level{i}"].classes_) for i in range(1, 5)]

    ckpt  = torch.load(ROOT / args.checkpoint, map_location=DEVICE)
    model = FusionV2Inspector(n_classes, dropout=0.0).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"체크포인트 로드 완료 (val micro_f1={ckpt.get('micro_f1',0):.4f})")

    # 데이터
    ds = ProteinDataset(
        ids_file      = ROOT / CFG["paths"]["splits_dir"] / f"{args.split}_ids.txt",
        meta_csv      = ROOT / CFG["paths"]["meta_csv"],
        embed_dir     = ROOT / CFG["paths"]["embed_dir"],
        cmap_dir      = ROOT / CFG["paths"]["cmap_dir"],
        label_enc_pkl = ROOT / CFG["paths"]["label_enc"],
    )
    loader = DataLoader(ds, batch_size=256, shuffle=False,
                        num_workers=4, collate_fn=collate_fn, pin_memory=True)

    # 중간값 수집
    all_gate, all_attn_w, all_spatial, all_l4_pred, all_l4_label = [], [], [], [], []
    all_esm_proj, all_attn_out = [], []

    with torch.no_grad():
        for i, (esm_emb, cmap, _, labels, masks, _, _) in enumerate(loader):
            if args.n_batches > 0 and i >= args.n_batches:
                break
            esm_emb = esm_emb.to(DEVICE)
            cmap    = cmap.to(DEVICE)

            out = model.forward_inspect(esm_emb, cmap)

            all_gate.append(out["gate"])
            all_attn_w.append(out["attn_weight"].reshape(-1))
            all_spatial.append(out["spatial_attn"])
            all_l4_pred.append(out["l4_pred"])
            all_l4_label.append(labels[:, 3])
            all_esm_proj.append(out["esm_proj"])
            all_attn_out.append(out["attn_out"])

            if (i+1) % 10 == 0:
                print(f"  배치 {i+1} 처리 중...")

    gate       = torch.cat(all_gate)        # (N, 1024)
    attn_w     = torch.cat(all_attn_w)      # (N,)
    spatial    = torch.cat(all_spatial)     # (N, 64)
    l4_pred    = torch.cat(all_l4_pred)     # (N,)
    l4_label   = torch.cat(all_l4_label)    # (N,)
    esm_proj   = torch.cat(all_esm_proj)    # (N, 1024)
    attn_out   = torch.cat(all_attn_out)    # (N, 1024)

    N = gate.shape[0]
    print(f"\n총 {N:,}개 샘플 분석")

    # ── 분석 1: Gate 값 전체 통계 ────────────────────────────
    gate_mean_per_sample = gate.mean(dim=1)   # (N,) 샘플별 평균 gate
    gate_mean_overall    = gate_mean_per_sample.mean().item()
    gate_std             = gate_mean_per_sample.std().item()

    print("\n" + "="*55)
    print("  [분석 1] Gate 값 (contact map 기여도)")
    print("="*55)
    print(f"  전체 평균 gate: {gate_mean_overall:.4f}  (std={gate_std:.4f})")
    print(f"  최솟값: {gate_mean_per_sample.min():.4f}")
    print(f"  최댓값: {gate_mean_per_sample.max():.4f}")
    print(f"  gate > 0.5인 샘플 비율: {(gate_mean_per_sample > 0.5).float().mean()*100:.1f}%")
    print(f"  gate > 0.3인 샘플 비율: {(gate_mean_per_sample > 0.3).float().mean()*100:.1f}%")
    print(f"  gate < 0.1인 샘플 비율: {(gate_mean_per_sample < 0.1).float().mean()*100:.1f}%")

    # gate 해석
    if gate_mean_overall < 0.1:
        print("\n  ⚠️  gate 거의 0 → 모델이 contact map을 사실상 무시")
    elif gate_mean_overall < 0.3:
        print("\n  ⚡  gate 낮음 → contact map 기여 제한적")
    else:
        print("\n  ✓  gate 활성 → contact map이 실제로 기여 중")

    # ── 분석 2: ESM-proj vs gate*attn_out 크기 비교 ─────────
    esm_norm  = esm_proj.norm(dim=1).mean().item()
    attn_norm = (gate * attn_out).norm(dim=1).mean().item()
    ratio     = attn_norm / (esm_norm + 1e-8)

    print("\n" + "="*55)
    print("  [분석 2] ESM vs Contact 기여 크기 비교")
    print("="*55)
    print(f"  ESM-proj 평균 L2 norm:       {esm_norm:.4f}")
    print(f"  gate*attn_out 평균 L2 norm:  {attn_norm:.4f}")
    print(f"  Contact/ESM 비율:             {ratio:.4f}  ({ratio*100:.1f}%)")

    if ratio < 0.05:
        print("  ⚠️  contact 기여가 ESM의 5% 미만 → 사실상 ESM만 사용")
    elif ratio < 0.2:
        print("  ⚡  contact 기여가 ESM의 20% 미만 → 보조적 역할")
    else:
        print("  ✓  contact가 ESM과 비슷한 수준으로 기여")

    # ── 분석 3: EC 레벨 1별 gate 값 (어떤 효소 클래스에서 구조가 중요한가) ─
    print("\n" + "="*55)
    print("  [분석 3] EC Level-1별 Gate 값")
    print("="*55)
    l1_classes = encoders["level1"].classes_
    # l4_label로 l1 추출
    # l4 → l1 매핑: 라벨의 앞자리
    l4_classes = encoders["level4"].classes_

    ec_gate_by_l1 = {i: [] for i in range(len(l1_classes))}
    for idx in range(N):
        l4_idx = l4_label[idx].item()
        if l4_idx < 0:
            continue
        ec_str = l4_classes[l4_idx]          # e.g. "1.1.1.1"
        l1_num = int(ec_str.split(".")[0]) - 1
        if 0 <= l1_num < len(l1_classes):
            ec_gate_by_l1[l1_num].append(gate_mean_per_sample[idx].item())

    ec_labels = ["1. Oxidoreductase", "2. Transferase", "3. Hydrolase",
                 "4. Lyase", "5. Isomerase", "6. Ligase", "7. Translocase"]
    l1_gate_means = []
    for i, ec_name in enumerate(ec_labels):
        vals = ec_gate_by_l1[i]
        if vals:
            m = np.mean(vals)
            l1_gate_means.append(m)
            print(f"  EC {ec_name:<22}: gate={m:.4f}  (n={len(vals):,})")
        else:
            l1_gate_means.append(0)

    # ── 분석 4: Spatial Attention 분포 (8x8 → 어디를 보는가) ──
    spatial_mean = spatial.mean(dim=0).reshape(8, 8).numpy()  # (8,8)

    print("\n" + "="*55)
    print("  [분석 4] Contact Map Spatial Attention (8×8)")
    print("="*55)
    print(f"  가장 주목받는 위치 (top-5):")
    flat = spatial.mean(dim=0).numpy()
    top5 = np.argsort(flat)[::-1][:5]
    for rank, pos in enumerate(top5):
        r, c = divmod(pos, 8)
        print(f"    {rank+1}위: 위치({r},{c}) = {flat[pos]:.4f}")

    uniform = 1/64
    print(f"  균일 분포 기댓값: {uniform:.4f}")
    print(f"  최고 attention: {flat.max():.4f}  ({flat.max()/uniform:.1f}x 집중)")

    # ── 시각화 ────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle("FusionV2 Contact Map 기여도 분석", fontsize=16, fontweight="bold")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # 1. Gate 분포 히스토그램
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(gate_mean_per_sample.numpy(), bins=50, color="#4C72B0", edgecolor="white", alpha=0.85)
    ax1.axvline(gate_mean_overall, color="red", linestyle="--", linewidth=2,
                label=f"mean={gate_mean_overall:.3f}")
    ax1.axvline(0.5, color="gray", linestyle=":", linewidth=1.5, label="0.5 기준선")
    ax1.set_xlabel("Sample-wise Mean Gate Value", fontsize=11)
    ax1.set_ylabel("Count", fontsize=11)
    ax1.set_title("Gate 값 분포\n(contact map 기여도)", fontsize=12)
    ax1.legend(fontsize=10)

    # 2. ESM vs Contact 기여 비교 (bar)
    ax2 = fig.add_subplot(gs[0, 1])
    bars = ax2.bar(["ESM-proj\n(서열)", "gate×attn\n(구조)"],
                   [esm_norm, attn_norm],
                   color=["#4C72B0", "#DD8452"], width=0.5, edgecolor="white")
    for bar, val in zip(bars, [esm_norm, attn_norm]):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f"{val:.2f}", ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax2.set_ylabel("평균 L2 Norm", fontsize=11)
    ax2.set_title(f"서열 vs 구조 기여 크기\n(Contact/ESM = {ratio*100:.1f}%)", fontsize=12)

    # 3. EC Level-1별 Gate
    ax3 = fig.add_subplot(gs[0, 2])
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52",
              "#8172B3", "#937860", "#DA8BC3"]
    short_labels = ["1.Oxido", "2.Trans", "3.Hydro", "4.Lyase",
                    "5.Isom", "6.Ligase", "7.Transl"]
    bars3 = ax3.bar(short_labels, l1_gate_means, color=colors, edgecolor="white")
    ax3.axhline(gate_mean_overall, color="red", linestyle="--", linewidth=1.5,
                label=f"전체 평균 {gate_mean_overall:.3f}")
    ax3.set_ylabel("평균 Gate 값", fontsize=11)
    ax3.set_title("EC Level-1별 Gate 값\n(어떤 효소에서 구조가 중요한가)", fontsize=12)
    ax3.legend(fontsize=9)
    ax3.tick_params(axis="x", labelsize=9)

    # 4. Spatial Attention 히트맵 (8×8)
    ax4 = fig.add_subplot(gs[1, 0])
    im = ax4.imshow(spatial_mean, cmap="hot", interpolation="nearest")
    plt.colorbar(im, ax=ax4)
    ax4.set_title("Contact Map Spatial Attention\n(8×8 격자, 어떤 위치를 주목하는가)", fontsize=12)
    ax4.set_xlabel("Contact Map 열 위치 (x8)", fontsize=10)
    ax4.set_ylabel("Contact Map 행 위치 (x32)", fontsize=10)

    # 5. Gate 누적분포
    ax5 = fig.add_subplot(gs[1, 1])
    sorted_gate = np.sort(gate_mean_per_sample.numpy())
    cdf = np.arange(1, len(sorted_gate)+1) / len(sorted_gate)
    ax5.plot(sorted_gate, cdf, color="#4C72B0", linewidth=2)
    ax5.axvline(gate_mean_overall, color="red", linestyle="--", linewidth=1.5,
                label=f"mean={gate_mean_overall:.3f}")
    ax5.axvline(0.5, color="gray", linestyle=":", linewidth=1.5, label="0.5")
    ax5.axvline(0.1, color="orange", linestyle=":", linewidth=1.5, label="0.1")
    ax5.set_xlabel("Gate 값", fontsize=11)
    ax5.set_ylabel("누적 비율", fontsize=11)
    ax5.set_title("Gate CDF\n(contact map 활성 비율)", fontsize=12)
    ax5.legend(fontsize=9)
    ax5.grid(alpha=0.3)

    # 6. Gate dim별 분포 (1024 차원 각각)
    ax6 = fig.add_subplot(gs[1, 2])
    gate_per_dim = gate.mean(dim=0).numpy()  # (1024,)
    ax6.plot(np.sort(gate_per_dim), color="#55A868", linewidth=1.2)
    ax6.axhline(0.5, color="gray", linestyle=":", linewidth=1.5)
    ax6.axhline(gate_per_dim.mean(), color="red", linestyle="--", linewidth=1.5,
                label=f"mean={gate_per_dim.mean():.3f}")
    ax6.fill_between(range(1024), np.sort(gate_per_dim), alpha=0.3, color="#55A868")
    ax6.set_xlabel("차원 (정렬)", fontsize=11)
    ax6.set_ylabel("평균 Gate 값", fontsize=11)
    ax6.set_title("1024 차원별 Gate 값 분포\n(0=구조 무시, 1=구조 완전 반영)", fontsize=12)
    ax6.legend(fontsize=9)

    out_dir = ROOT / "outputs" / "results"
    out_path = out_dir / "fusion_v2_gate_analysis.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n시각화 저장: {out_path}")

    # 최종 요약
    print("\n" + "="*55)
    print("  [최종 요약]")
    print("="*55)
    print(f"  Gate 평균:          {gate_mean_overall:.4f}")
    print(f"  Contact/ESM 비율:   {ratio*100:.1f}%")
    if gate_mean_overall > 0.3 and ratio > 0.15:
        print("  → contact map이 실제로 기여하고 있음")
        print("  → 논문 근거: 구조 정보가 모델에 의미있게 통합됨")
    elif gate_mean_overall > 0.1:
        print("  → contact map 기여 있으나 제한적")
        print("  → 특정 EC 클래스 분석으로 차별점 찾기 필요")
    else:
        print("  → contact map 거의 무시됨")
        print("  → 융합 방법 재설계 필요")


if __name__ == "__main__":
    main()
