"""
Grad-CAM Visualization for Contact-EC (B4)

Generates:
  1. ResNet Grad-CAM heatmap on 256×256 contact map
  2. Spatial attention weights from AttentionContactEncoder
  3. Side-by-side: contact map | Grad-CAM overlay | attn weights

Usage:
  CUDA_VISIBLE_DEVICES=0 python scripts/gradcam_visualize.py \
      --n_samples 6 --split test --output outputs/results/gradcam/
"""
import sys, pickle, argparse, json
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import yaml
with open(ROOT / "configs" / "config_ecbench.yaml") as f:
    CFG = yaml.safe_load(f)

from models.dataset import ProteinDataset, collate_fn
from models.fusion_v2_flatfc import FusionV2FlatFC

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


# ── Grad-CAM hook infrastructure ────────────────────────────────────────────

class GradCAM:
    """Grad-CAM on the last ResNet conv block (layer4) of AttentionContactEncoder."""

    def __init__(self, model: FusionV2FlatFC):
        self.model = model
        self._activations = None
        self._gradients   = None
        self._attn_weights = None
        self._hooks = []

        target_layer = model.contact_encoder.features[-1]  # ResNet layer4

        self._hooks.append(
            target_layer.register_forward_hook(self._save_activation)
        )
        self._hooks.append(
            target_layer.register_full_backward_hook(self._save_gradient)
        )

        # Also capture spatial attention weights
        def _attn_hook(module, inp, out):
            B = out.shape[0]
            # attn out is (B, 1, H, W) before softmax in the model
            # We re-run softmax here for cleaner weights
            w = out.view(B, -1)
            w = F.softmax(w, dim=-1)
            self._attn_weights = w.detach().cpu()

        self._hooks.append(
            model.contact_encoder.attn.register_forward_hook(_attn_hook)
        )

    def _save_activation(self, module, inp, out):
        self._activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self._gradients = grad_out[0].detach()

    def __call__(self, esm_emb, cmap, target_level=3, target_class=None):
        """
        Returns:
            cam      : np.ndarray (256, 256)  Grad-CAM heatmap
            attn_map : np.ndarray (8, 8)      spatial attention weights
            pred_classes : list[int]          top predicted L4 classes
            pred_probs   : list[float]        corresponding probs
        """
        self.model.zero_grad()
        logits = self.model(esm_emb.to(DEVICE), cmap.to(DEVICE))
        l4_logit = logits[target_level]  # (1, n_classes)

        if target_class is None:
            target_class = int(l4_logit.argmax(dim=1).item())

        score = l4_logit[0, target_class]
        score.backward(retain_graph=True)

        # Grad-CAM
        grads = self._gradients[0]        # (C, H, W)
        acts  = self._activations[0]      # (C, H, W)
        weights = grads.mean(dim=(1, 2))  # (C,) global avg pooling
        cam = (weights[:, None, None] * acts).sum(0)  # (H, W)
        cam = F.relu(cam)
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        cam_up = F.interpolate(
            cam.unsqueeze(0).unsqueeze(0),
            size=(256, 256), mode="bilinear", align_corners=False
        ).squeeze().cpu().numpy()

        # Attention map
        attn_8x8 = self._attn_weights[0].view(8, 8).numpy()

        # Top-5 predictions
        probs = torch.sigmoid(l4_logit[0]).detach().cpu().numpy()
        top_idx  = np.argsort(probs)[::-1][:5]
        top_prob = probs[top_idx].tolist()

        return cam_up, attn_8x8, list(top_idx), top_prob, target_class

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()


# ── Visualization ────────────────────────────────────────────────────────────

def visualize_sample(cmap_3ch, cam, attn_8x8, uid, true_ec, pred_classes,
                     pred_probs, target_class, label_enc, out_path):
    """
    3-panel figure:
      (a) contact map (binary, all-contacts channel)
      (b) Grad-CAM overlay on contact map
      (c) spatial attention weights (8×8)
    """
    fig = plt.figure(figsize=(15, 4.5))
    gs  = gridspec.GridSpec(1, 4, figure=fig, wspace=0.3)

    cmap_bin = cmap_3ch[0]  # binary all-contacts channel (256×256)

    # ── (a) Contact Map ──────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.imshow(cmap_bin, cmap="Blues", interpolation="nearest", vmin=0, vmax=1)
    ax1.set_title(f"(a) Contact Map\n{uid}", fontsize=9)
    ax1.set_xlabel("Residue j"); ax1.set_ylabel("Residue i")
    ax1.tick_params(labelsize=7)

    # ── (b) Grad-CAM overlay ────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    ax2.imshow(cmap_bin, cmap="Greys", alpha=0.5, vmin=0, vmax=1)
    im2 = ax2.imshow(cam, cmap="jet", alpha=0.6, vmin=0, vmax=1)
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    ec_str = label_enc.inverse_transform([target_class])[0] \
             if hasattr(label_enc, "inverse_transform") else str(target_class)
    ax2.set_title(f"(b) Grad-CAM\nTarget: {ec_str}", fontsize=9)
    ax2.set_xlabel("Residue j"); ax2.tick_params(labelsize=7)

    # ── (c) Spatial attention ────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2])
    im3 = ax3.imshow(attn_8x8, cmap="YlOrRd", vmin=0)
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    ax3.set_title("(c) ResNet Spatial\nAttention (8×8)", fontsize=9)
    ax3.set_xlabel("Patch j"); ax3.set_ylabel("Patch i")
    ax3.tick_params(labelsize=7)

    # ── (d) Prediction bar chart ─────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[3])
    colors = ["#d62728" if i == target_class else "#1f77b4" for i in pred_classes]
    bars = ax4.barh(range(5), pred_probs[::-1], color=colors[::-1])
    try:
        labels = [label_enc.inverse_transform([c])[0] for c in pred_classes[::-1]]
    except Exception:
        labels = [str(c) for c in pred_classes[::-1]]
    ax4.set_yticks(range(5)); ax4.set_yticklabels(labels, fontsize=7)
    ax4.set_xlim(0, 1); ax4.set_xlabel("Sigmoid prob")
    ax4.axvline(0.5, color="gray", linestyle="--", alpha=0.5)
    true_str = true_ec if true_ec else "unknown"
    ax4.set_title(f"(d) Top-5 Predictions\nTrue EC: {true_str}", fontsize=9)

    fig.suptitle(f"Contact-EC Grad-CAM Analysis  |  {uid}", fontsize=10, y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=6,
                        help="Number of samples to visualize")
    parser.add_argument("--split", default="test",
                        choices=["test", "val_hard", "price149"],
                        help="Which split to sample from")
    parser.add_argument("--correct_only", action="store_true", default=True,
                        help="Only visualize correctly predicted samples")
    parser.add_argument("--output", default="outputs/results/gradcam/")
    args = parser.parse_args()

    out_dir = ROOT / args.output
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load label encoder
    with open(ROOT / CFG["paths"]["label_enc"], "rb") as f:
        encoders = pickle.load(f)
    n_classes = [len(encoders[f"level{i}"].classes_) for i in range(1, 5)]
    l4_enc = encoders["level4"]

    # Load model
    ckpt  = torch.load(ROOT / "outputs/checkpoints/ecbench_b4_flatfc_best.pt",
                       map_location=DEVICE, weights_only=False)
    model = FusionV2FlatFC(n_classes,
                           esm_dim=CFG["model"]["esm2_dim"],
                           contact_dim=CFG["model"]["resnet_out_dim"],
                           fusion_dim=CFG["model"]["fusion_dim"],
                           dropout=0.0).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Dataset
    splits = ROOT / CFG["paths"]["splits_dir"]
    proc   = ROOT / "data" / "ecbench" / "processed"

    split_map = {
        "test":      (splits / "test_ids_full.txt",  proc / "test_meta_full.csv"),
        "val_hard":  (splits / "val_hard_ids.txt",   proc / "val_hard_meta.csv"),
        "price149":  (splits / "price149_ids.txt",   proc / "price149_meta.csv"),
    }
    ids_file, meta_csv = split_map[args.split]

    ds = ProteinDataset(
        ids_file=str(ids_file), meta_csv=str(meta_csv),
        embed_dir=ROOT / CFG["paths"]["embed_dir"],
        cmap_dir=ROOT / CFG["paths"]["cmap_dir"],
        label_enc_pkl=ROOT / CFG["paths"]["label_enc"],
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False,
                        num_workers=0, collate_fn=collate_fn)

    gradcam = GradCAM(model)

    collected = 0
    summary   = []

    for batch_idx, batch in enumerate(loader):
        if collected >= args.n_samples:
            break

        esm_emb, cmap, sequences, labels, masks, l4_mh, uids = batch
        uid = uids[0]

        # Skip if no valid L4 label
        if l4_mh.sum() == 0:
            continue

        # Get true EC
        true_indices = l4_mh[0].nonzero(as_tuple=True)[0].tolist()
        true_ec = l4_enc.inverse_transform([true_indices[0]])[0] \
                  if true_indices else None

        # Run Grad-CAM
        cam, attn_8x8, pred_classes, pred_probs, target_class = gradcam(
            esm_emb, cmap, target_level=3, target_class=true_indices[0]
        )

        # Check if correctly predicted (top-1 = true)
        is_correct = (pred_classes[0] == true_indices[0])
        if args.correct_only and not is_correct:
            continue

        # Get contact map (3-channel numpy)
        cmap_np = cmap[0].cpu().numpy()  # (3, 256, 256)

        out_path = out_dir / f"gradcam_{uid}_{true_ec.replace('.','_')}.png"
        visualize_sample(
            cmap_np, cam, attn_8x8, uid, true_ec,
            pred_classes, pred_probs, target_class,
            l4_enc, out_path
        )
        summary.append({
            "uid": uid, "true_ec": true_ec,
            "top1_pred": l4_enc.inverse_transform([pred_classes[0]])[0],
            "correct": bool(is_correct),
            "top1_prob": round(float(pred_probs[0]), 4),
        })
        collected += 1

    gradcam.remove_hooks()

    # Save summary
    summary_path = out_dir / "gradcam_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nVisualized {collected} samples → {out_dir}")
    print(f"Summary: {summary_path}")

    # Print case study summary
    print("\n=== Case Study Summary ===")
    for s in summary:
        status = "✓" if s["correct"] else "✗"
        print(f"  {status} {s['uid']}  true={s['true_ec']}  pred={s['top1_pred']}  p={s['top1_prob']}")


if __name__ == "__main__":
    main()
