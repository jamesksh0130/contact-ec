"""
EC-Bench Step 5: 새 단백질 Contact Map 생성

data/ecbench/new_proteins.txt 중 기존 cmap 없는 것만 처리.
PDB가 없으면 zeros로 저장하지 않음 (ProteinDataset에서 zeros 처리).
"""
import numpy as np
from pathlib import Path
from scipy.ndimage import zoom
from Bio import PDB
from tqdm import tqdm

ROOT     = Path("/home/user/Desktop/unlv")
PDB_DIR  = ROOT / "data/raw/pdb"
CMAP_DIR = ROOT / "data/processed/contact_maps"
ECBENCH  = ROOT / "data/ecbench"

THRESHOLD = 8.0
SIZE      = 256


def pdb_to_contact_map(pdb_path: Path, threshold=THRESHOLD, size=SIZE) -> np.ndarray | None:
    try:
        parser = PDB.PDBParser(QUIET=True)
        struct = parser.get_structure("p", str(pdb_path))
        ca_atoms = [r["CA"] for r in struct.get_residues() if r.has_id("CA")]
        if len(ca_atoms) < 5:
            return None
        coords = np.array([a.get_vector().get_array() for a in ca_atoms])
        diff   = coords[:, None, :] - coords[None, :, :]
        dist   = np.sqrt((diff ** 2).sum(-1))
        cmap   = (dist < threshold).astype(np.float32)
        n      = len(ca_atoms)
        if n != size:
            scale       = size / n
            cmap_resized = zoom(cmap, (scale, scale), order=1)
            cmap_resized = cmap_resized[:size, :size]
            if cmap_resized.shape != (size, size):
                pad = np.zeros((size, size), dtype=np.float32)
                h, w = cmap_resized.shape
                pad[:h, :w] = cmap_resized
                cmap_resized = pad
        else:
            cmap_resized = cmap
        return cmap_resized.astype(np.float32)
    except Exception:
        return None


def main():
    new_ids = set((ECBENCH / "new_proteins.txt").read_text().strip().split("\n"))
    existing = set(p.stem for p in CMAP_DIR.glob("*.npy"))
    to_process = [uid for uid in sorted(new_ids) if uid not in existing]
    print(f"새 contact map 필요: {len(to_process):,}개")

    ok, no_pdb, fail = 0, 0, 0
    for uid in tqdm(to_process, desc="contact map 생성"):
        pdb_path = PDB_DIR / f"{uid}.pdb"
        if not pdb_path.exists():
            no_pdb += 1
            continue
        cmap = pdb_to_contact_map(pdb_path)
        if cmap is None:
            fail += 1
            continue
        np.save(CMAP_DIR / f"{uid}.npy", cmap)
        ok += 1

    print(f"\n완료: 생성={ok}, PDB없음={no_pdb}, 실패={fail}")


if __name__ == "__main__":
    main()
