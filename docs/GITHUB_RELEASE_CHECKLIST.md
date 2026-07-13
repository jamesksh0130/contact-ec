# GitHub Release Checklist

Before making the repository public:

- Confirm author, affiliation, email, and repository metadata in `paper/source/`.
- Add a final `LICENSE` file.
- Update `CITATION.cff` with final author names, repository URL, DOI, and license.
- Add external links for data, checkpoints, ESM-2 embedding cache, and contact-map cache.
- Confirm that no raw private data, tokens, or oversized binaries are committed.
- Run a fresh-clone smoke test for at least import, evaluation script help, and result-table generation.
- Add a GitHub release tag matching the manuscript submission version.
