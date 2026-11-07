# Runs

Logged experiment outputs.

**Committed**
- `seed{0,1,2}/{st,pgd_at}/metrics.json` — training curves / selection metrics
- `eval/*.json` — attack, calibration, and audit summaries
- `paired_training_summary.json`

**Not committed** (see root `.gitignore`)
- `*.pt` checkpoints (~43MB each). Keep locally if you need reliability diagrams or re-eval.
- Pilot folders (`st_pilot/`, `pgd_at_pilot/`) and `FROZEN` markers.
