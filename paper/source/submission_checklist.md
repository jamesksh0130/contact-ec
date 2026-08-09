# Submission Checklist

Regenerated 2026-08-09 from the current `paper/main.tex` / `paper/supplementary.tex`
(BMC Bioinformatics-targeted manuscript, currently built with the existing LaTeX template;
13 + 23 pages). The previous package in this
folder (`main_submission.tex`, dated 2026-07-17) was built from an older manuscript
draft that still used the pre-A/B/C `fig1_pipeline.png` overview figure and a
different abstract — it predates the SP-2024 reversal finding, the fig1/fig3/fig4/fig6
figure-accuracy fixes, and the "gated cross-attention" -> "structure-gated additive
fusion" terminology correction, all from 2026-08-09. It has been removed; do not
resurrect it as the submission source.

- Main manuscript: `main.pdf` (source: `main.tex`)
- Supplementary information: `supplementary.pdf` (source: `supplementary.tex`)
- Source archive: `bmc_bioinformatics_submission_source.zip`
- PDF archive: `bmc_bioinformatics_submission_pdfs.zip`
- Article type: Research article
- Suggested category: Sequence analysis, protein function prediction, or structural bioinformatics
- Review mode: submit `main.tex` / `main.pdf` as the author-visible manuscript unless the submission system explicitly requests a blinded file. `main_blind.tex` exists only as a backup for venues/preprint workflows that require blinding.
- Completed: author name, affiliation, corresponding email, ORCID, funding statement, acknowledgements, conflict of interest, and author contribution statement have been added.
- **Repository policy note 2026-08-09:** use the direct public GitHub URL rather than an anonymous mirror.
- **Done 2026-08-09:** `github.com/jamesksh0130/contact-ec` is now **public** (was private; created 2026-08-06). It is tracked separately at `github_release/contact-ec/` (its own clone, `origin` -> this repo). Synced the 2026-08-09 manuscript/package fixes through commit `74cd196`; BMC declaration files were added and pushed afterward. Confirmed externally reachable post-push.
- **Done 2026-08-09:** the manuscript's Abstract and Data Availability statement now cite the public repository directly: `https://github.com/jamesksh0130/contact-ec`.
- Required before upload: provide ORCID iD `0009-0002-0708-748X` for the submitting author in the submission system.
- Required before upload: archive code/data release in a stable repository such as GitHub plus Zenodo if the manuscript is presented as software/tooling.
- Optional before upload: add a Zenodo DOI if an archived release is created before submission.
- BMC format note checked 2026-08-09: BMC Bioinformatics accepts TeX/LaTeX and encourages the Springer Nature LaTeX template. The current review PDF/source can be uploaded, but convert to the Springer/BMC template if the submission system or editor requests it.
- BMC figure note checked 2026-08-09: figures should be numbered in first-citation order, uploaded in that order, and multi-panel figures should be single composite files. Current figure placement follows first-use narrative order.
- BMC cost note checked 2026-08-09: BMC Bioinformatics is fully open access and charges an APC only for accepted articles. Current listed APC is GBP 2290 / USD 3090 / EUR 2590, subject to tax and possible institutional/country-tiered pricing, discount, or waiver handling.
- Submission-system declarations prepared in `declarations_for_submission.md`.
- Cost-aware fallback journal notes prepared in `journal_cost_options.md`.
