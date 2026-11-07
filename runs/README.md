# Runs

Logged experiment outputs.

**Committed**
- `*/metrics.json` — training curves / selection metrics
- `*/FROZEN` — checkpoint freeze markers
- `eval/*.json` — attack, calibration, and audit summaries
- `paired_training_summary.json`

**Not committed** (see root `.gitignore`)
- `*.pt` checkpoints (~43MB each). Keep locally if you need reliability diagrams or re-eval.
