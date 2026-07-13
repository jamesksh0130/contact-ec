# Data and Checkpoints

This GitHub upload package does not include large files.

## Excluded from GitHub

- Raw Swiss-Prot/UniProt downloads.
- AlphaFold/PDB structure files.
- Contact-map arrays.
- ESM-2 embedding caches.
- Trained PyTorch checkpoints.
- Third-party baseline repositories and model weights.

The local workspace contains hundreds of GB of data, so these files should be hosted separately through Zenodo, Figshare, institutional storage, Hugging Face, or another long-term archive.

## Recommended public-release structure

Before making the repository public, add links for:

| Artifact | Suggested hosting | Required for |
|---|---|---|
| Processed split metadata | GitHub or Zenodo | Evaluation reproducibility |
| Contact-map cache | Zenodo/Hugging Face | Contact and fusion models |
| ESM-2 embedding cache | Zenodo/Hugging Face | Fast reproduction |
| Trained checkpoints | Zenodo/Hugging Face | Inference and exact metric reproduction |
| Raw data accession list | GitHub | Dataset reconstruction |

## Minimum reproducibility expectation

At minimum, the public repository should provide:

1. Accession IDs and EC labels for each evaluated split.
2. Exact model checkpoint hashes.
3. Exact ESM-2 model identifiers.
4. Scripts used to reconstruct contact maps and embeddings.
5. The JSON result files and audit reports already included in this package.
