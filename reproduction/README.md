# Reproduction workflow

This directory contains the deterministic baseline wrapper and the TIGRIS
execution configuration used to validate the paper repository.

- `run_art_average_baseline.py` calls the repository's original regression and
  comparative experiment functions, adds fixed seeds, persists partial
  results, and compares them with the tracked result tables.
- `BASELINE_REPORT.md` records the completed baseline and its interpretation.
- `cluster/` contains the Git-backed TIGRIS synchronization, smoke test, and
  full Slurm job.

Generated outputs are intentionally ignored by Git. Code and configuration
move through GitHub; cluster logs, checkpoints, datasets not licensed for
redistribution, and experiment outputs remain in cluster storage or an
appropriate artifact store.

See [`cluster/README.md`](cluster/README.md) for TIGRIS commands.
