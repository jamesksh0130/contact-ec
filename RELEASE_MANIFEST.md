# GitHub Upload Manifest

Prepared on: 2026-07-14

Package:

- Folder: `github_release/contact-ec/`
- Zip: `github_release/contact-ec-github-upload.zip`
- Approximate folder size before compression: 4.6 MB
- Files in package: 168

Contents:

- Source code: `train.py`, `evaluate.py`, `visualize.py`, `models/`, `scripts/`
- Configurations: `configs/`
- Manuscript PDFs and LaTeX source: `paper/`
- Selected results: `outputs/results/`
- HIT-EC vs Contact-EC case-wise comparison: `outputs/results/casewise_hitec_contactec.*`
- Contact-EC threshold sensitivity analysis: `outputs/results/contactec_threshold_sensitivity.*`
- Reliability audit artifacts: `outputs/audit/`
- Manuscript figures: `outputs/figures/`
- GitHub documentation: `README.md`, `docs/`, `CITATION.cff`, `requirements.txt`, `.gitignore`

Excluded by design:

- Raw datasets
- Structure files
- ESM-2 embedding caches
- Contact-map caches
- Model checkpoints
- Third-party baseline repositories
- Logs and temporary files

Required updates before public release:

- MIT license included.
- Author/email/repository metadata have been populated; update only if the public repository path changes.
- Add DOI or archive links for data and checkpoints.
- Run a fresh-clone smoke test after external data paths are configured.
