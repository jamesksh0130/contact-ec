"""
자동 파이프라인 실행기
Step 3 (PDB) 완료 → Step 4 (Contact Map) → ESM-2 완료 대기
→ B0 → B1 → B2 → B3 → Fusion Phase1 → Fusion Phase2 순서로 자동 학습

사용법:
  python scripts/run_pipeline.py \
      --pdb_pid 948984 --esm_pid 986785 --gpu 1
"""
import os, sys, time, argparse, subprocess, logging
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
LOG_FILE = ROOT / "outputs" / "pipeline.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE),
    ],
)
log = logging.getLogger()


def is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def wait_for_pid(pid: int, label: str, check_interval: int = 60):
    log.info(f"대기 중: {label} (PID {pid})")
    while is_running(pid):
        n_pdb  = len(list((ROOT / "data/raw/pdb").glob("*.pdb")))
        n_emb  = len(list((ROOT / "data/processed/embeddings").glob("*.npy")))
        n_cmap = len(list((ROOT / "data/processed/contact_maps").glob("*.npy")))
        log.info(f"  진행 중 | PDB={n_pdb:,}  EMB={n_emb:,}  CMAP={n_cmap:,}")
        time.sleep(check_interval)
    log.info(f"완료: {label}")


def run_step(cmd: list[str], label: str) -> int:
    log.info(f"\n{'='*50}")
    log.info(f"시작: {label}")
    log.info(f"명령: {' '.join(cmd)}")
    log.info(f"{'='*50}")
    t0 = time.time()
    ret = subprocess.run(cmd, cwd=ROOT).returncode
    elapsed = time.time() - t0
    status = "완료" if ret == 0 else f"오류(code={ret})"
    log.info(f"{label} {status} — {elapsed/60:.1f}분 소요")
    return ret


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb_pid", type=int, required=True,
                        help="Step 3 PDB 다운로드 PID")
    parser.add_argument("--esm_pid", type=int, required=True,
                        help="Step 5 ESM-2 추출 PID")
    parser.add_argument("--gpu",      type=int, default=1,
                        help="학습에 사용할 GPU 번호")
    parser.add_argument("--workers",  type=int, default=8,
                        help="Contact Map 생성 worker 수")
    parser.add_argument("--skip_baselines", action="store_true",
                        help="B0~B3 건너뛰고 Fusion만 학습")
    args = parser.parse_args()

    log.info("="*60)
    log.info("자동 파이프라인 시작")
    log.info(f"  PDB PID : {args.pdb_pid}")
    log.info(f"  ESM PID : {args.esm_pid}")
    log.info(f"  GPU     : {args.gpu}")
    log.info(f"  시작 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info("="*60)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # ── Step 3 완료 대기 → Step 4 실행 ──────────────────────
    if is_running(args.pdb_pid):
        wait_for_pid(args.pdb_pid, "PDB 다운로드(Step 3)", check_interval=60)
    else:
        log.info("PDB 다운로드 이미 완료")

    n_pdb = len(list((ROOT / "data/raw/pdb").glob("*.pdb")))
    log.info(f"수집된 PDB 파일: {n_pdb:,}개")

    # Step 4: Contact Map 생성
    n_cmap = len(list((ROOT / "data/processed/contact_maps").glob("*.npy")))
    if n_cmap < n_pdb * 0.9:
        ret = run_step(
            ["python", "scripts/04_build_contact_maps.py",
             "--workers", str(args.workers)],
            "Step 4: Contact Map 생성"
        )
        if ret != 0:
            log.error("Contact Map 생성 실패. 계속 진행...")
    else:
        log.info(f"Contact Map 이미 존재 ({n_cmap:,}개) — 건너뜀")

    # ── Step 5 완료 대기 ─────────────────────────────────────
    if is_running(args.esm_pid):
        wait_for_pid(args.esm_pid, "ESM-2 임베딩 추출(Step 5)", check_interval=120)
    else:
        log.info("ESM-2 임베딩 추출 이미 완료")

    n_emb  = len(list((ROOT / "data/processed/embeddings").glob("*.npy")))
    n_cmap = len(list((ROOT / "data/processed/contact_maps").glob("*.npy")))
    log.info(f"\n데이터 준비 완료!")
    log.info(f"  ESM-2 임베딩 : {n_emb:,}개")
    log.info(f"  Contact Map  : {n_cmap:,}개")

    # ── 학습 시작 ─────────────────────────────────────────────
    train_cmd = lambda model, phase=1, resume=None, ep=None: (
        ["python", "train.py",
         "--model", model,
         "--phase", str(phase),
         "--gpu",   str(args.gpu),
         "--epochs", str(ep or (
             30 if phase == 1 else 20
         ))]
        + (["--resume", resume] if resume else [])
    )

    results = {}

    if not args.skip_baselines:
        # B0: 1D CNN
        ret = run_step(train_cmd("b0_cnn", ep=30), "B0: 1D CNN 학습")
        results["b0"] = "ok" if ret == 0 else "fail"

        # B1: ESM-2 + Flat FC
        ret = run_step(train_cmd("b1_esm2_fc", ep=30), "B1: ESM-2 + FC 학습")
        results["b1"] = "ok" if ret == 0 else "fail"

        # B2: ESM-2 + Hierarchical
        ret = run_step(train_cmd("b2_esm2_hier", ep=30), "B2: ESM-2 + Hierarchical 학습")
        results["b2"] = "ok" if ret == 0 else "fail"

        # B3: Contact Map + ResNet
        ret = run_step(train_cmd("b3_contact", ep=30), "B3: Contact Map + ResNet 학습")
        results["b3"] = "ok" if ret == 0 else "fail"

    # Fusion Phase 1 (ESM-2 frozen 개념 — lr=1e-3)
    ret = run_step(train_cmd("fusion", phase=1, ep=30), "Fusion Phase 1 학습")
    results["fusion_p1"] = "ok" if ret == 0 else "fail"

    # Fusion Phase 2 (lr=1e-4, 이전 체크포인트 이어받기)
    best_p1 = ROOT / "outputs/checkpoints/fusion_phase1_best.pt"
    if best_p1.exists():
        ret = run_step(
            train_cmd("fusion", phase=2, resume=str(best_p1), ep=20),
            "Fusion Phase 2 학습"
        )
        results["fusion_p2"] = "ok" if ret == 0 else "fail"

    # ── 최종 평가 ─────────────────────────────────────────────
    log.info("\n" + "="*60)
    log.info("전체 학습 완료! 테스트 셋 평가 시작...")
    log.info("="*60)

    ckpt_dir = ROOT / "outputs/checkpoints"
    for model_tag, ckpt_name in [
        ("b0_cnn",      "b0_cnn_phase1_best.pt"),
        ("b1_esm2_fc",  "b1_esm2_fc_phase1_best.pt"),
        ("b2_esm2_hier","b2_esm2_hier_phase1_best.pt"),
        ("b3_contact",  "b3_contact_phase1_best.pt"),
        ("fusion",      "fusion_phase2_best.pt"),
    ]:
        ckpt_path = ckpt_dir / ckpt_name
        if not ckpt_path.exists():
            ckpt_path = ckpt_dir / ckpt_name.replace("phase2", "phase1")
        if ckpt_path.exists():
            run_step(
                ["python", "evaluate.py",
                 "--model", model_tag,
                 "--checkpoint", str(ckpt_path),
                 "--split", "test",
                 "--gpu", str(args.gpu)],
                f"{model_tag} 테스트 평가"
            )

    log.info("\n" + "="*60)
    log.info("파이프라인 전체 완료!")
    log.info(f"결과 위치: {ROOT / 'outputs/results/'}")
    log.info("="*60)
    for k, v in results.items():
        log.info(f"  {k}: {v}")


if __name__ == "__main__":
    main()
