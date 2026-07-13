"""
통합 해석 가능성 모듈 — Grad-CAM + Integrated Gradients

두 가지 해석 방법:
  1. Grad-CAM       : ResNet-50 layer4 → 어느 contact 패턴이 활성화되는가
  2. Integrated Gradients (IG) : contact map 픽셀 수준 기여도 → 어느 잔기 쌍이 EC 결정에 기여하는가

우리만 할 수 있는 분석:
  - Ch1(all) vs Ch2(short-range) vs Ch3(long-range) 채널별 IG 비교
  - Level 1→4로 갈수록 long-range contact 기여가 증가하는지 정량화

사용법:
  python visualize.py --checkpoint outputs/checkpoints/fusion_phase2_best.pt \\
                      --accession P00330 --level 4

  python visualize.py --checkpoint ... --accession P00330 --analyze_channels
    → Ch2 vs Ch3 기여도 분석 (논문용 핵심 분석)

출력:
  outputs/results/interp_{accession}_L{level}.png   — 개별 시각화
  outputs/results/channel_analysis.png              -- 채널별 기여도 분석
"""
import argparse, pickle, yaml
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from captum.attr import IntegratedGradients, NoiseTunnel, LayerGradCam

ROOT   = Path(__file__).resolve().parent
OUT_DIR = ROOT / "outputs" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

with open(ROOT / "configs" / "config.yaml") as f:
    CFG = yaml.safe_load(f)


# ── 모델 래퍼 (captum은 단일 텐서 입력 선호) ────────────────
class ContactOnlyWrapper(nn.Module):
    """contact map만 입력으로 받는 wrapper — IG 계산용."""
    def __init__(self, fusion_model, esm_emb_fixed, level_idx):
        super().__init__()
        self.model      = fusion_model
        self.esm_fixed  = esm_emb_fixed   # (1, 1280) 고정
        self.level_idx  = level_idx

    def forward(self, cmap):
        logits = self.model(self.esm_fixed.expand(cmap.shape[0], -1), cmap)
        return logits[self.level_idx]      # (B, n_classes)


# ── Integrated Gradients ─────────────────────────────────────
def compute_ig(wrapper, cmap_input, target_class, n_steps=50):
    """
    contact map → IG attribution (3, 256, 256)
    baseline: 0 (접촉 없음)
    """
    ig = IntegratedGradients(wrapper)
    baseline = torch.zeros_like(cmap_input)
    attr, delta = ig.attribute(
        cmap_input,
        baselines=baseline,
        target=target_class,
        n_steps=n_steps,
        return_convergence_delta=True,
    )
    return attr.squeeze(0).detach().cpu().numpy(), delta.item()


# ── Grad-CAM ─────────────────────────────────────────────────
class GradCAM:
    def __init__(self, model, target_layer):
        self.model     = model
        self.grads     = None
        self.acts      = None
        self._hooks    = []
        self._hooks.append(target_layer.register_forward_hook(
            lambda m, i, o: setattr(self, 'acts', o.detach())))
        self._hooks.append(target_layer.register_full_backward_hook(
            lambda m, gi, go: setattr(self, 'grads', go[0].detach())))

    def remove(self):
        for h in self._hooks: h.remove()

    def __call__(self, esm_emb, cmap, level_idx, class_idx=None):
        self.model.eval()
        logits = self.model(esm_emb, cmap)
        if class_idx is None:
            class_idx = logits[level_idx].argmax(dim=1).item()
        self.model.zero_grad()
        logits[level_idx][0, class_idx].backward(retain_graph=True)
        weights = self.grads.mean(dim=(2, 3), keepdim=True)
        cam     = F.relu((weights * self.acts).sum(dim=1)).squeeze()
        cam     = cam.cpu().numpy()
        if cam.max() > 0: cam /= cam.max()
        cam_256 = F.interpolate(
            torch.tensor(cam).unsqueeze(0).unsqueeze(0),
            size=(256, 256), mode="bilinear", align_corners=False
        ).squeeze().numpy()
        return cam_256, class_idx


# ── 채널별 기여도 정량화 ─────────────────────────────────────
def channel_contribution(ig_attr):
    """
    IG attribution (3, 256, 256) → 채널별 절댓값 합 비율
    Returns: dict {ch1_all, ch2_short, ch3_long}
    """
    ch_totals = np.abs(ig_attr).sum(axis=(1, 2))   # (3,)
    total = ch_totals.sum() + 1e-8
    return {
        "ch1_all":   ch_totals[0] / total,
        "ch2_short": ch_totals[1] / total,
        "ch3_long":  ch_totals[2] / total,
    }


# ── 시각화 ────────────────────────────────────────────────────
def plot_interpretation(cmap_3ch, cam, ig_attr, accession, ec_str, level, out_path):
    """
    6패널 그림:
    Row 1: 3채널 contact map
    Row 2: Grad-CAM 오버레이 | IG Attribution | Ch2 vs Ch3 비교
    """
    fig = plt.figure(figsize=(18, 10))
    gs  = gridspec.GridSpec(2, 4, hspace=0.45, wspace=0.35, figure=fig)

    ch_labels = ["Ch1: All contacts", "Ch2: Short-range (|i-j|<12)", "Ch3: Long-range (|i-j|≥12)"]
    ch_cmaps  = ["Blues", "Greens", "Reds"]

    # ── Row 1: 3채널 contact map ──
    for c in range(3):
        ax = fig.add_subplot(gs[0, c])
        ax.imshow(cmap_3ch[c], cmap=ch_cmaps[c], vmin=0, vmax=1, origin="upper")
        ax.set_title(ch_labels[c], fontsize=9)
        ax.set_xlabel("Residue j", fontsize=8)
        ax.set_ylabel("Residue i", fontsize=8)
        ax.tick_params(labelsize=7)

    # ── Row 1, Col 4: 채널 기여도 파이차트 ──
    contrib = channel_contribution(ig_attr)
    ax_pie = fig.add_subplot(gs[0, 3])
    sizes  = [contrib["ch1_all"], contrib["ch2_short"], contrib["ch3_long"]]
    colors = ["#3498db", "#2ecc71", "#e74c3c"]
    wedges, texts, autotexts = ax_pie.pie(
        sizes, labels=["All", "Short", "Long"],
        colors=colors, autopct="%1.1f%%", startangle=90,
        textprops={"fontsize": 9}
    )
    ax_pie.set_title(f"IG Channel Contribution\n(Level {level})", fontsize=9, fontweight="bold")

    # ── Row 2, Col 1: Grad-CAM 오버레이 ──
    ax_cam = fig.add_subplot(gs[1, 0])
    ax_cam.imshow(cmap_3ch[0], cmap="Blues", vmin=0, vmax=1, origin="upper", alpha=0.55)
    im_cam = ax_cam.imshow(cam, cmap="jet", vmin=0, vmax=1, origin="upper", alpha=0.55)
    plt.colorbar(im_cam, ax=ax_cam, fraction=0.046, pad=0.04)
    ax_cam.set_title("Grad-CAM\n(Layer4 Activation)", fontsize=9)
    ax_cam.set_xlabel("Residue j", fontsize=8)
    ax_cam.set_ylabel("Residue i", fontsize=8)

    # ── Row 2, Col 2: IG Attribution (All contacts 채널) ──
    ax_ig = fig.add_subplot(gs[1, 1])
    ig_all = ig_attr[0]
    vmax   = np.percentile(np.abs(ig_all), 99)
    im_ig  = ax_ig.imshow(ig_all, cmap="RdBu_r", vmin=-vmax, vmax=vmax, origin="upper")
    plt.colorbar(im_ig, ax=ax_ig, fraction=0.046, pad=0.04)
    ax_ig.set_title("IG Attribution\n(Ch1: All contacts)", fontsize=9)
    ax_ig.set_xlabel("Residue j", fontsize=8)
    ax_ig.set_ylabel("Residue i", fontsize=8)

    # ── Row 2, Col 3: Short vs Long IG 비교 ──
    ax_sl = fig.add_subplot(gs[1, 2])
    ig_diff = ig_attr[2] - ig_attr[1]   # long - short (양수=long 기여 더 큼)
    vmax_d  = np.percentile(np.abs(ig_diff), 99)
    im_sl   = ax_sl.imshow(ig_diff, cmap="RdBu", vmin=-vmax_d, vmax=vmax_d, origin="upper")
    plt.colorbar(im_sl, ax=ax_sl, fraction=0.046, pad=0.04)
    ax_sl.set_title("IG: Long - Short Range\n(+red=long, -blue=short)", fontsize=9)
    ax_sl.set_xlabel("Residue j", fontsize=8)
    ax_sl.set_ylabel("Residue i", fontsize=8)

    # ── Row 2, Col 4: Grad-CAM × IG (GradCAM++ 유사) ──
    ax_comb = fig.add_subplot(gs[1, 3])
    ig_mag  = np.abs(ig_attr).mean(axis=0)   # 3채널 평균 절댓값
    if ig_mag.max() > 0: ig_mag /= ig_mag.max()
    combined = cam * ig_mag
    if combined.max() > 0: combined /= combined.max()
    im_c = ax_comb.imshow(combined, cmap="hot", vmin=0, vmax=1, origin="upper")
    plt.colorbar(im_c, ax=ax_comb, fraction=0.046, pad=0.04)
    ax_comb.set_title("Grad-CAM × IG\n(Combined Attribution)", fontsize=9)
    ax_comb.set_xlabel("Residue j", fontsize=8)
    ax_comb.set_ylabel("Residue i", fontsize=8)

    fig.suptitle(
        f"{accession}  —  Predicted EC (Level {level}): {ec_str}\n"
        f"Short-range contribution: {contrib['ch2_short']*100:.1f}%  |  "
        f"Long-range contribution: {contrib['ch3_long']*100:.1f}%",
        fontsize=12, fontweight="bold"
    )
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"해석 결과 저장: {out_path}")
    print(f"  Short-range: {contrib['ch2_short']*100:.1f}%  "
          f"Long-range: {contrib['ch3_long']*100:.1f}%")


def plot_channel_analysis(results_by_level, out_path):
    """
    Level 1~4에 걸쳐 short-range vs long-range 기여도 변화 그래프.
    핵심 분석: "Level 4로 갈수록 long-range 기여가 증가하는가?"
    """
    levels = sorted(results_by_level.keys())
    shorts = [results_by_level[l]["ch2_short"] * 100 for l in levels]
    longs  = [results_by_level[l]["ch3_long"]  * 100 for l in levels]
    alls   = [results_by_level[l]["ch1_all"]   * 100 for l in levels]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.array(levels)
    ax.plot(x, shorts, "-o", color="#2ecc71", linewidth=2.5, markersize=8, label="Short-range (2° structure)")
    ax.plot(x, longs,  "-s", color="#e74c3c", linewidth=2.5, markersize=8, label="Long-range (3° contacts)")
    ax.plot(x, alls,   "-^", color="#3498db", linewidth=2.5, markersize=8, label="All contacts", alpha=0.6)

    ax.set_xlabel("EC Level (1→4, coarse→fine)", fontsize=12)
    ax.set_ylabel("IG Attribution Share (%)", fontsize=12)
    ax.set_title("Contact Type Contribution by EC Level\n"
                 "(Our unique analysis: meaningful 3-channel encoding)",
                 fontsize=12, fontweight="bold")
    ax.set_xticks(levels)
    ax.set_xticklabels([f"Level {l}" for l in levels])
    ax.legend(fontsize=11)
    ax.grid(alpha=0.35)
    ax.set_ylim(0, 70)

    # 해석 텍스트
    if longs[-1] > longs[0]:
        ax.annotate("↑ Long-range gains importance\nat finer EC levels",
                    xy=(levels[-1], longs[-1]),
                    xytext=(levels[-1]-0.6, longs[-1]+8),
                    fontsize=9, color="#e74c3c",
                    arrowprops=dict(arrowstyle="->", color="#e74c3c"))

    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"채널 분석 그래프 저장: {out_path}")


# ── 메인 ─────────────────────────────────────────────────────
def build_model(n_classes):
    from models.fusion_model import FusionModel
    return FusionModel(n_classes, esm_dim=CFG["model"]["esm2_dim"],
                       contact_dim=CFG["model"]["resnet_out_dim"],
                       fusion_dim=CFG["model"]["fusion_dim"], dropout=0.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",       required=True)
    parser.add_argument("--accession",        required=True)
    parser.add_argument("--level",            type=int, default=4, choices=[1,2,3,4])
    parser.add_argument("--analyze_channels", action="store_true",
                        help="Level 1~4 전체 채널 기여도 분석 (논문용)")
    parser.add_argument("--ig_steps",         type=int, default=50)
    parser.add_argument("--gpu",              type=int, default=0)
    args = parser.parse_args()

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"

    with open(ROOT / CFG["paths"]["label_enc"], "rb") as f:
        encoders = pickle.load(f)
    n_classes = [len(encoders[f"level{i}"].classes_) for i in range(1, 5)]

    ckpt  = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model = build_model(n_classes).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    from models.dataset import _make_3ch_cmap
    emb_path  = ROOT / CFG["paths"]["embed_dir"] / f"{args.accession}.npy"
    cmap_path = ROOT / CFG["paths"]["cmap_dir"]  / f"{args.accession}.npy"

    if not emb_path.exists() or not cmap_path.exists():
        print(f"파일 없음: {args.accession}")
        return

    esm_np   = np.load(emb_path).astype(np.float32)
    esm_emb  = torch.tensor(esm_np).unsqueeze(0).to(device)   # (1,1280)
    raw_cmap = np.load(cmap_path).astype(np.float32)
    cmap_3ch = _make_3ch_cmap(raw_cmap)                        # (3,256,256)
    cmap_t   = torch.tensor(cmap_3ch).unsqueeze(0).to(device)  # (1,3,256,256)

    if args.analyze_channels:
        # ── Level 1~4 전체 채널 기여도 분석 ──
        results = {}
        for lv in [1, 2, 3, 4]:
            lv_idx  = lv - 1
            wrapper = ContactOnlyWrapper(model, esm_emb, lv_idx).to(device)
            cmap_req = cmap_t.clone().requires_grad_(True)
            with torch.no_grad():
                pred_idx = model(esm_emb, cmap_t)[lv_idx].argmax(dim=1).item()
            ig_attr, _ = compute_ig(wrapper, cmap_req, pred_idx, n_steps=args.ig_steps)
            results[lv] = channel_contribution(ig_attr)
            print(f"Level {lv}: short={results[lv]['ch2_short']*100:.1f}%  "
                  f"long={results[lv]['ch3_long']*100:.1f}%")
        plot_channel_analysis(results, OUT_DIR / "channel_analysis.png")
        return

    # ── 단일 레벨 해석 ──
    level_idx = args.level - 1

    # Grad-CAM
    target_layer = model.contact_encoder.backbone.layer4[-1]
    cam_extractor = GradCAM(model, target_layer)
    cmap_cam = cmap_t.clone()
    cam, pred_idx = cam_extractor(esm_emb, cmap_cam, level_idx)
    cam_extractor.remove()

    # Integrated Gradients
    wrapper  = ContactOnlyWrapper(model, esm_emb, level_idx).to(device)
    cmap_ig  = cmap_t.clone().requires_grad_(True)
    ig_attr, delta = compute_ig(wrapper, cmap_ig, pred_idx, n_steps=args.ig_steps)
    print(f"IG convergence delta: {delta:.6f}")

    ec_str = encoders[f"level{args.level}"].classes_[pred_idx]
    out_path = OUT_DIR / f"interp_{args.accession}_L{args.level}.png"
    plot_interpretation(cmap_3ch, cam, ig_attr, args.accession, ec_str, args.level, out_path)


if __name__ == "__main__":
    main()
