"""
통합 학습 스크립트
사용법:
  python train.py --model b0_cnn      --epochs 30
  python train.py --model b1_esm2_fc  --epochs 30
  python train.py --model b2_esm2_hier --epochs 30
  python train.py --model b3_contact  --epochs 30
  python train.py --model fusion --phase 1 --epochs 30
  python train.py --model fusion --phase 2 --epochs 20 \
                  --resume outputs/checkpoints/fusion_phase1_best.pt
"""
import os, argparse, pickle, yaml, time, random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import f1_score
import numpy as np

ROOT = Path(__file__).resolve().parent

import sys as _sys
_config_file = "config.yaml"
for _i, _a in enumerate(_sys.argv):
    if _a == "--config" and _i + 1 < len(_sys.argv):
        _config_file = _sys.argv[_i + 1]
        break
with open(ROOT / "configs" / _config_file) as f:
    CFG = yaml.safe_load(f)

from models.dataset import ProteinDataset, collate_fn, ContactPairDataset, collate_fn_v3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# ── 모델 팩토리 ──────────────────────────────────────────────
def build_model(model_name: str, n_classes: list[int]):
    if model_name == "b0_cnn":
        from models.baseline_cnn import BaselineCNN
        return BaselineCNN(n_classes, dropout=CFG["model"]["dropout"])

    elif model_name == "b1_esm2_fc":
        from models.esm2_fc import ESM2FC
        return ESM2FC(n_classes, esm_dim=CFG["model"]["esm2_dim"],
                      dropout=CFG["model"]["dropout"])

    elif model_name == "b2_esm2_hier":
        from models.esm2_hierarchical import ESM2Hierarchical
        return ESM2Hierarchical(n_classes, esm_dim=CFG["model"]["esm2_dim"],
                                dropout=CFG["model"]["dropout"])

    elif model_name == "b3_contact":
        from models.contact_resnet import ContactResNet
        return ContactResNet(n_classes, dropout=CFG["model"]["dropout"])

    elif model_name == "fusion":
        from models.fusion_model import FusionModel
        return FusionModel(n_classes, esm_dim=CFG["model"]["esm2_dim"],
                           contact_dim=CFG["model"]["resnet_out_dim"],
                           fusion_dim=CFG["model"]["fusion_dim"],
                           dropout=CFG["model"]["dropout"])

    elif model_name == "fusion_esm_ft":
        from models.fusion_esm_ft import FusionESMFinetune
        unfreeze = CFG["model"].get("esm_unfreeze_layers", 4)
        return FusionESMFinetune(n_classes,
                                 contact_dim=CFG["model"]["resnet_out_dim"],
                                 fusion_dim=CFG["model"]["fusion_dim"],
                                 dropout=CFG["model"]["dropout"],
                                 unfreeze_layers=unfreeze)

    elif model_name == "fusion_v2":
        from models.fusion_v2 import FusionModelV2
        return FusionModelV2(n_classes,
                             esm_dim=CFG["model"]["esm2_dim"],
                             contact_dim=CFG["model"]["resnet_out_dim"],
                             fusion_dim=CFG["model"]["fusion_dim"],
                             dropout=CFG["model"]["dropout"])

    elif model_name == "fusion_v3":
        from models.fusion_v3 import FusionModelV3
        return FusionModelV3(n_classes,
                             esm_dim=CFG["model"]["esm2_dim"],
                             contact_dim=CFG["model"]["resnet_out_dim"],
                             fusion_dim=CFG["model"]["fusion_dim"],
                             dropout=CFG["model"]["dropout"])

    elif model_name == "fusion_v3_esm_ft":
        from models.fusion_v3_esm_ft import FusionV3ESMFt
        unfreeze = CFG["model"].get("esm_unfreeze_layers", 2)
        return FusionV3ESMFt(n_classes,
                             contact_dim=CFG["model"]["resnet_out_dim"],
                             fusion_dim=CFG["model"]["fusion_dim"],
                             dropout=CFG["model"]["dropout"],
                             unfreeze_layers=unfreeze)

    elif model_name == "fusion_v2_flatfc":
        from models.fusion_v2_flatfc import FusionV2FlatFC
        return FusionV2FlatFC(n_classes,
                              esm_dim=CFG["model"]["esm2_dim"],
                              contact_dim=CFG["model"]["resnet_out_dim"],
                              fusion_dim=CFG["model"]["fusion_dim"],
                              dropout=CFG["model"]["dropout"])

    elif model_name == "fusion_concat_flatfc":
        from models.fusion_simple_baselines import FusionConcatFlatFC
        return FusionConcatFlatFC(n_classes,
                                  esm_dim=CFG["model"]["esm2_dim"],
                                  contact_dim=CFG["model"]["resnet_out_dim"],
                                  fusion_dim=CFG["model"]["fusion_dim"],
                                  dropout=CFG["model"]["dropout"])

    elif model_name == "fusion_sum_flatfc":
        from models.fusion_simple_baselines import FusionSumFlatFC
        return FusionSumFlatFC(n_classes,
                               esm_dim=CFG["model"]["esm2_dim"],
                               contact_dim=CFG["model"]["resnet_out_dim"],
                               fusion_dim=CFG["model"]["fusion_dim"],
                               dropout=CFG["model"]["dropout"])

    elif model_name == "fusion_gated_mlp_flatfc":
        from models.fusion_simple_baselines import FusionGatedMLPFlatFC
        return FusionGatedMLPFlatFC(n_classes,
                                    esm_dim=CFG["model"]["esm2_dim"],
                                    contact_dim=CFG["model"]["resnet_out_dim"],
                                    fusion_dim=CFG["model"]["fusion_dim"],
                                    dropout=CFG["model"]["dropout"])

    elif model_name == "fusion_v2_3b":
        from models.fusion_v2 import FusionModelV2
        return FusionModelV2(n_classes,
                             esm_dim=CFG["model"].get("esm2_dim_3b", 2560),
                             contact_dim=CFG["model"]["resnet_out_dim"],
                             fusion_dim=CFG["model"]["fusion_dim"],
                             dropout=CFG["model"]["dropout"])

    else:
        raise ValueError(f"알 수 없는 모델: {model_name}")


# ── 부모-자식 매핑 행렬 (Hierarchical Consistency Loss용) ────
def build_parent_child_matrix(encoders, parent_level: int, child_level: int) -> torch.Tensor:
    """
    EC prefix 구조 활용: l(child_level) 클래스가 어느 l(parent_level) 클래스의 자식인지 binary matrix.
    반환: (n_parent, n_child) float32 tensor
    """
    parent_cls = encoders[f"level{parent_level}"].classes_
    child_cls  = encoders[f"level{child_level}"].classes_
    p_dict = {c: i for i, c in enumerate(parent_cls)}

    M = torch.zeros(len(parent_cls), len(child_cls), dtype=torch.float32)
    for j, cc in enumerate(child_cls):
        parts = cc.split(".")
        prefix = ".".join(parts[:parent_level])
        if prefix in p_dict:
            M[p_dict[prefix], j] = 1.0
    return M


def consistency_loss(logits_list, M12, M23, M34) -> torch.Tensor:
    """
    멀티레이블 계층 일관성: sigmoid + MSE.
    자식 레벨 예측을 부모로 집계했을 때 부모 예측과 일치하도록.
    두 논문(HIT-EC, CLEAN) 모두 미포함 — 우리의 독자적 기여.
    """
    l1 = torch.sigmoid(logits_list[0])   # (B, n1)
    l2 = torch.sigmoid(logits_list[1])   # (B, n2)
    l3 = torch.sigmoid(logits_list[2])   # (B, n3)
    l4 = torch.sigmoid(logits_list[3])   # (B, n4)

    # 자식 → 부모 집계: (B, n_child) @ M.T → (B, n_parent)
    l2g = (l2 @ M12.T).clamp(0, 1)   # (B, n1)
    l3g = (l3 @ M23.T).clamp(0, 1)   # (B, n2)
    l4g = (l4 @ M34.T).clamp(0, 1)   # (B, n3)

    loss = (F.mse_loss(l2g, l1.detach()) +
            F.mse_loss(l3g, l2.detach()) +
            F.mse_loss(l4g, l3.detach())) / 3.0
    return loss


# ── Focal Loss ──────────────────────────────────────────────────
class FocalLoss(nn.Module):
    """Focal Loss for class imbalance in rare EC classes."""
    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


# ── Masked Hierarchical Loss (전 레벨 BCE 멀티레이블) ─────────────────────
def hierarchical_loss(logits_list, labels, masks, weights,
                      l4_multihot=None, use_focal: bool = False,
                      M12=None, M23=None, M34=None):
    """
    L1~L4 전 레벨: BCEWithLogitsLoss (멀티레이블)
    L1~L3 멀티핫은 l4_multihot + M 행렬로 자동 유도.
    M 행렬 없을 시 L1~L3은 CrossEntropy fallback.
    """
    bce = nn.BCEWithLogitsLoss(reduction="mean")
    ce  = nn.CrossEntropyLoss(reduction="none")

    # L4 → L3 → L2 → L1 멀티핫 유도
    if l4_multihot is not None and M12 is not None:
        l3_mh = (l4_multihot @ M34.T).clamp(0, 1)   # (B, n3)
        l2_mh = (l3_mh       @ M23.T).clamp(0, 1)   # (B, n2)
        l1_mh = (l2_mh       @ M12.T).clamp(0, 1)   # (B, n1)
        all_mh = [l1_mh, l2_mh, l3_mh, l4_multihot]
    else:
        all_mh = [None, None, None, l4_multihot]

    total_loss = 0.0
    for i, (logits, w) in enumerate(zip(logits_list, weights)):
        valid = masks[:, i].bool()
        if valid.sum() == 0:
            continue

        if all_mh[i] is not None:
            loss_i = bce(logits[valid], all_mh[i][valid])
        else:
            lbl    = labels[valid, i]
            loss_i = ce(logits[valid], lbl).mean()

        total_loss = total_loss + w * loss_i

    return total_loss


# ── 메트릭 계산 (멀티레이블 F1) ─────────────────────────────
def compute_metrics_ml(all_l4_probs, all_l4_mh, threshold: float = 0.5):
    """Level 4 멀티레이블 Micro/Macro/Weighted F1 (sigmoid threshold 기반).
    - Micro   : aggregate TP/FP/FN across all classes (HIT-EC 비교용)
    - Macro   : unweighted per-class average
    - Weighted: support-weighted per-class average (EC-Bench 비교용)
    """
    pred_bin = (all_l4_probs >= threshold).astype(np.int32)
    target   = all_l4_mh.astype(np.int32)
    micro    = f1_score(target, pred_bin, average="micro",    zero_division=0)
    macro    = f1_score(target, pred_bin, average="macro",    zero_division=0)
    weighted = f1_score(target, pred_bin, average="weighted", zero_division=0)
    return micro, macro, weighted


# ── 한 에폭 학습 ─────────────────────────────────────────────
_CONS_WEIGHT = 0.05   # hierarchical consistency loss 가중치

def train_epoch(model, loader, optimizer, weights, model_name,
                M12=None, M23=None, M34=None, use_focal: bool = False,
                grad_accum: int = 1, scaler=None):
    model.train()
    total_loss = 0.0
    n_batch    = 0
    use_amp    = scaler is not None

    optimizer.zero_grad()
    for step, batch in enumerate(loader):
        if model_name == "fusion_v3":
            esm_emb, cmap, sequences, labels, masks, l4_mh, _, pair_emb = batch
            pair_emb = pair_emb.to(DEVICE)
        else:
            esm_emb, cmap, sequences, labels, masks, l4_mh, _ = batch

        esm_emb  = esm_emb.to(DEVICE)
        cmap     = cmap.to(DEVICE)
        labels   = labels.to(DEVICE)
        masks    = masks.to(DEVICE)
        l4_mh    = l4_mh.to(DEVICE)

        with autocast(enabled=use_amp):
            if model_name == "b0_cnn":
                logits = model(sequences, device=DEVICE)
            elif model_name in ("fusion_esm_ft", "fusion_v3_esm_ft"):
                logits = model(sequences, cmap)
            elif model_name == "fusion_v3":
                logits = model(esm_emb, pair_emb)
            else:
                logits = model(esm_emb, cmap)

            loss = hierarchical_loss(logits, labels, masks, weights,
                                     l4_multihot=l4_mh, use_focal=use_focal,
                                     M12=M12, M23=M23, M34=M34)
            if model_name.startswith("fusion") and M12 is not None:
                loss = loss + _CONS_WEIGHT * consistency_loss(logits, M12, M23, M34)

        if use_amp:
            scaler.scale(loss / grad_accum).backward()
        else:
            (loss / grad_accum).backward()

        if (step + 1) % grad_accum == 0 or (step + 1) == len(loader):
            if use_amp:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item()
        n_batch    += 1

        total_loss += loss.item()
        n_batch    += 1

    return total_loss / max(n_batch, 1)


# ── 검증 ─────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, weights, model_name, M12=None, M23=None, M34=None):
    model.eval()
    total_loss   = 0.0
    all_l4_probs = []   # sigmoid 확률 (B, n_l4)
    all_l4_mh    = []   # 정답 멀티핫 (B, n_l4)

    for batch in loader:
        if model_name == "fusion_v3":
            esm_emb, cmap, sequences, labels, masks, l4_mh, _, pair_emb = batch
            pair_emb = pair_emb.to(DEVICE)
        else:
            esm_emb, cmap, sequences, labels, masks, l4_mh, _ = batch

        esm_emb = esm_emb.to(DEVICE)
        cmap    = cmap.to(DEVICE)
        labels  = labels.to(DEVICE)
        masks   = masks.to(DEVICE)
        l4_mh   = l4_mh.to(DEVICE)

        if model_name == "b0_cnn":
            logits = model(sequences, device=DEVICE)
        elif model_name in ("fusion_esm_ft", "fusion_v3_esm_ft"):
            logits = model(sequences, cmap)
        elif model_name == "fusion_v3":
            logits = model(esm_emb, pair_emb)
        else:
            logits = model(esm_emb, cmap)

        loss = hierarchical_loss(logits, labels, masks, weights,
                                 l4_multihot=l4_mh, M12=M12, M23=M23, M34=M34)
        total_loss += loss.item()

        all_l4_probs.append(torch.sigmoid(logits[3]).cpu().numpy())
        all_l4_mh.append(l4_mh.cpu().numpy())

    all_l4_probs = np.concatenate(all_l4_probs, axis=0)   # (N, n_l4)
    all_l4_mh    = np.concatenate(all_l4_mh,    axis=0)   # (N, n_l4)
    micro, macro, weighted = compute_metrics_ml(all_l4_probs, all_l4_mh)
    return total_loss / len(loader), micro, macro, weighted


# ── 메인 ─────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   default="fusion",
                        choices=["b0_cnn", "b1_esm2_fc", "b2_esm2_hier",
                                 "b3_contact", "fusion", "fusion_esm_ft",
                                 "fusion_v2", "fusion_v2_flatfc", "fusion_v2_3b",
                                 "fusion_concat_flatfc", "fusion_sum_flatfc",
                                 "fusion_gated_mlp_flatfc",
                                 "fusion_v3", "fusion_v3_esm_ft"])
    parser.add_argument("--phase",   type=int, default=1,
                        help="fusion 모델: 1=ESM frozen, 2=partial unfreeze")
    parser.add_argument("--epochs",  type=int, default=30)
    parser.add_argument("--resume",  default=None,
                        help="이전 체크포인트 경로 (Phase 2 시작용)")
    parser.add_argument("--gpu",     type=int, default=0)
    parser.add_argument("--tag",     default=None,
                        help="체크포인트 저장 태그 (기본: {model}_phase{phase})")
    parser.add_argument("--split_prefix", default="",
                        help="split 파일 prefix (기본: '', cluster split이면 'cluster_')")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="배치 크기 오버라이드 (기본: config.yaml 값)")
    parser.add_argument("--focal_loss", action="store_true",
                        help="L1~L3에 Focal Loss 사용 (희귀 EC 클래스 개선)")
    parser.add_argument("--lr_esm",   type=float, default=1e-5,
                        help="fusion_esm_ft Phase 2: ESM-2 fine-tuning LR")
    parser.add_argument("--config",   default="config.yaml",
                        help="config 파일명 (configs/ 아래, 기본: config.yaml)")
    parser.add_argument("--val_file", default=None,
                        help="val split 파일명 override (예: val_hard_ids.txt)")
    parser.add_argument("--grad_accum", type=int, default=1,
                        help="gradient accumulation steps (effective_bs = batch_size * grad_accum)")
    parser.add_argument("--fp16", action="store_true",
                        help="fp16 mixed precision training (AMP)")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for reproducible repeated runs")
    args = parser.parse_args()

    seed_everything(args.seed)

    global DEVICE
    DEVICE = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    print(f"디바이스: {DEVICE}  |  모델: {args.model}  |  Phase: {args.phase}  |  Seed: {args.seed}", flush=True)

    # ── 라벨 인코더에서 클래스 수 로드 ──
    with open(ROOT / CFG["paths"]["label_enc"], "rb") as f:
        encoders = pickle.load(f)
    n_classes = [len(encoders[f"level{i}"].classes_) for i in range(1, 5)]
    print(f"클래스 수: {n_classes}", flush=True)

    # ── 계층 일관성 매핑 행렬 (GPU에 올려둠) ──
    M12 = build_parent_child_matrix(encoders, 1, 2).to(DEVICE)
    M23 = build_parent_child_matrix(encoders, 2, 3).to(DEVICE)
    M34 = build_parent_child_matrix(encoders, 3, 4).to(DEVICE)
    print(f"일관성 매핑: M12{tuple(M12.shape)} M23{tuple(M23.shape)} M34{tuple(M34.shape)}", flush=True)

    def seed_worker(worker_id):
        worker_seed = args.seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def loader_generator():
        generator = torch.Generator()
        generator.manual_seed(args.seed)
        return generator

    # ── 데이터셋 ──
    def make_loader(split, shuffle):
        bs = args.batch_size if args.batch_size else CFG["train"]["batch_size"]

        if args.model == "fusion_v3":
            ds = ContactPairDataset(
                ids_file      = ROOT / CFG["paths"]["splits_dir"] / f"{split}_ids.txt",
                meta_csv      = ROOT / CFG["paths"]["meta_csv"],
                embed_dir     = ROOT / CFG["paths"]["embed_dir"],
                cmap_dir      = ROOT / CFG["paths"]["cmap_dir"],
                label_enc_pkl = ROOT / CFG["paths"]["label_enc"],
                pair_emb_dir  = ROOT / "data" / "processed" / "contact_pair_embs",
                k_pairs       = 32,
            )
            return DataLoader(ds, batch_size=bs, shuffle=shuffle,
                              num_workers=4, collate_fn=collate_fn_v3, pin_memory=True,
                              worker_init_fn=seed_worker, generator=loader_generator())
        else:
            ds = ProteinDataset(
                ids_file      = ROOT / CFG["paths"]["splits_dir"] / f"{split}_ids.txt",
                meta_csv      = ROOT / CFG["paths"]["meta_csv"],
                embed_dir     = ROOT / CFG["paths"]["embed_dir"],
                cmap_dir      = ROOT / CFG["paths"]["cmap_dir"],
                label_enc_pkl = ROOT / CFG["paths"]["label_enc"],
            )
            return DataLoader(ds, batch_size=bs, shuffle=shuffle,
                              num_workers=4, collate_fn=collate_fn, pin_memory=True,
                              worker_init_fn=seed_worker, generator=loader_generator())

    pfx = args.split_prefix  # e.g. "" or "cluster_"
    train_loader = make_loader(f"{pfx}train", shuffle=True)
    # val_file 오버라이드 지원 (hard val 사용 시)
    if args.val_file:
        bs = args.batch_size if args.batch_size else CFG["train"]["batch_size"]
        val_ds = ProteinDataset(
            ids_file      = ROOT / CFG["paths"]["splits_dir"] / args.val_file,
            meta_csv      = ROOT / CFG["paths"]["meta_csv"],
            embed_dir     = ROOT / CFG["paths"]["embed_dir"],
            cmap_dir      = ROOT / CFG["paths"]["cmap_dir"],
            label_enc_pkl = ROOT / CFG["paths"]["label_enc"],
        )
        val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False,
                                num_workers=4, collate_fn=collate_fn, pin_memory=True,
                                worker_init_fn=seed_worker,
                                generator=loader_generator())
    else:
        val_loader = make_loader(f"{pfx}val", shuffle=False)
    print(f"Train: {len(train_loader.dataset):,}  Val: {len(val_loader.dataset):,}", flush=True)

    # ── 모델 ──
    model = build_model(args.model, n_classes).to(DEVICE)

    # Phase 2: 이전 체크포인트 로드
    if args.resume:
        ckpt = torch.load(args.resume, map_location=DEVICE)
        # fusion_v3_esm_ft: ESM-2 파라미터는 HuggingFace에서 로드되므로 strict=False
        strict = (args.model != "fusion_v3_esm_ft")
        missing, unexpected = model.load_state_dict(ckpt["model"], strict=strict)
        if missing:
            print(f"  로드 제외(ESM-2 등): {len(missing)}개 키")
        print(f"체크포인트 로드: {args.resume}")

    lr = (CFG["train"]["lr_phase1"] if args.phase == 1
          else CFG["train"]["lr_phase2"])
    epochs = args.epochs

    # ESM-2 마지막 N층 unfreeze + 차등 LR (fusion_esm_ft / fusion_v3_esm_ft Phase 2)
    if args.model in ("fusion_esm_ft", "fusion_v3_esm_ft") and args.phase == 2:
        model.unfreeze_esm_last()
        param_groups = model.get_param_groups(lr_esm=args.lr_esm, lr_rest=lr)
        optimizer = AdamW(param_groups, weight_decay=CFG["train"]["weight_decay"])
        print(f"ESM-2 fine-tuning: lr_esm={args.lr_esm}, lr_rest={lr}")
    else:
        optimizer = AdamW(model.parameters(), lr=lr,
                          weight_decay=CFG["train"]["weight_decay"])

    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    weights   = CFG["train"]["loss_weights"]
    use_focal = args.focal_loss
    if use_focal:
        print("Focal Loss 활성화 (gamma=2.0, L1~L3 적용)")

    scaler = GradScaler() if (args.fp16 and torch.cuda.is_available()) else None
    if scaler:
        print("fp16 mixed precision (AMP) 활성화")
    if args.grad_accum > 1:
        bs = args.batch_size if args.batch_size else CFG["train"]["batch_size"]
        print(f"Gradient accumulation: {args.grad_accum} steps (effective batch={bs * args.grad_accum})")

    # ── 학습 루프 ──
    ckpt_dir = ROOT / CFG["paths"]["ckpt_dir"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_micro = 0.0
    tag = args.tag if args.tag else f"{args.model}_phase{args.phase}"

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, weights, args.model,
                                 M12=M12, M23=M23, M34=M34, use_focal=use_focal,
                                 grad_accum=args.grad_accum, scaler=scaler)
        val_loss, val_micro, val_macro, val_weighted = evaluate(
            model, val_loader, weights, args.model, M12=M12, M23=M23, M34=M34)
        scheduler.step()

        elapsed = time.time() - t0
        print(f"[{epoch:03d}/{epochs}]  "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"micro_f1={val_micro:.4f}  weighted_f1={val_weighted:.4f}  macro_f1={val_macro:.4f}  "
              f"({elapsed:.0f}s)", flush=True)

        # 베스트 모델 저장 (micro_f1 기준)
        if val_micro > best_micro:
            best_micro = val_micro
            ckpt_path = ckpt_dir / f"{tag}_best.pt"
            torch.save({
                "epoch":      epoch,
                "model":      model.state_dict(),
                "optim":      optimizer.state_dict(),
                "micro_f1":   val_micro,
                "weighted_f1": val_weighted,
                "macro_f1":   val_macro,
                "n_classes":  n_classes,
            }, ckpt_path)
            print(f"  ✓ 베스트 저장: {ckpt_path}  "
                  f"(micro={val_micro:.4f}  weighted={val_weighted:.4f})", flush=True)

    print(f"\n학습 완료! 베스트 Micro F1: {best_micro:.4f}", flush=True)


if __name__ == "__main__":
    main()
