# TriAudit

Controlled empirical audit of **PGD adversarial training** vs **standard training** on CIFAR-10 (CIFAR-adapted ResNet-18).

TriAudit measures three things together:

1. **Clean utility** — clean accuracy cost of PGD-AT vs seed-matched ST  
2. **White-box robustness / attack adequacy** — FGSM, PGD-20, multi-restart PGD  
3. **Confidence calibration** — ECE (15 bins), NLL, Brier on clean / FGSM / PGD-20  

This is an empirical study under a defined threat model — **not** deployment certification. Zero audit flags does **not** mean a model is safe to deploy.

## Repository layout

```text
TriAudit/
├── README.md
├── requirements.txt
├── pyproject.toml
├── TriAudit_Final_Project_Architecture.md   # authoritative blueprint
├── SERVER_AUTO_IMPLEMENTATION.md            # optional server auto-run notes
├── configs/                                 # YAML experiment configs
├── dataset/
│   ├── raw/                                 # CIFAR-10 download (gitignored)
│   └── splits/                              # persisted 45k/5k indices
├── src/triaudit/                            # package: data, model, train, attacks, metrics
├── scripts/
│   ├── prepare_data.py
│   ├── make_report_figures.py
│   ├── make_reliability_diagrams.py
│   └── auto/                                # phased train/eval orchestration
├── tests/
├── runs/                                    # metrics + eval JSON (checkpoints ignored)
├── reports/
│   ├── figures/                             # presentation figures
│   ├── tables/
│   └── TriAudit_research_report.md
└── project_monitor/                         # phase gates (logs ignored)
```

**Tracked for GitHub:** code, configs, split indices, eval summaries, figures, report, phase-verify JSON.  
**Not uploaded:** CIFAR-10 raw files (~340MB), `*.pt` checkpoints (~43MB each), orchestrator logs, `.cursor/`.

## Setup

```bash
# from repo root
python -m venv .venv && source .venv/bin/activate   # or use your conda env
pip install -r requirements.txt
pip install -e .

# download CIFAR-10 into dataset/raw
python scripts/prepare_data.py
```

On the shared TriAudit server, use the `ridoy` conda env only (see `.cursor/rules` / architecture locks).

## Reproduce (high level)

Phases follow `TriAudit_Final_Project_Architecture.md`:

| Phase | What |
|------|------|
| 0–1 | Architecture verify, dataset + ResNet-18 setup |
| 2–3 | ST / PGD-AT pilots |
| 4 | Paired training seeds `{0,1,2}` |
| 5–6 | Attack + calibration evaluation |
| 7–9 | Audit aggregation, stats/figures, research report |

Unattended server path (optional): see `SERVER_AUTO_IMPLEMENTATION.md`.

Regenerate presentation figures from logged metrics:

```bash
PYTHONPATH=src python scripts/make_report_figures.py
# optional (needs GPU + checkpoints present locally):
PYTHONPATH=src python scripts/make_reliability_diagrams.py
```

## Headline results (3 seeds, mean ± std)

| Metric | Value |
|--------|------:|
| Clean ST | 94.05 ± 0.23% |
| Clean PGD-AT | 82.28 ± 0.38% |
| Δ clean (AT − ST) | −11.77 ± 0.50 pp |
| PGD-20 robust (AT), ε=8/255 | 48.40 ± 0.08% |
| AttackGap (AT) | 0.19 ± 0.16 pp |
| ECE clean (AT) | 0.098 ± 0.007 |

Audit status: seeds 0 & 2 → **Major concerns**; seed 1 → **Caveat** (driven mainly by clean-accuracy cost / PGD-20 ECE).

Figures: `reports/figures/` (see `manifest.json` for slide order).

## Claim boundaries

- CIFAR-10 + ResNet-18 only; white-box ℓ∞ primary ε=`8/255`
- Alert thresholds are **project-defined**, not community standards
- Missing metrics must never be treated as passing
- Full limitations: architecture doc §10 and `reports/TriAudit_research_report.md`

## Citation / artifacts

Primary eval artifacts:

- `runs/eval/attacks_summary.json`
- `runs/eval/calibration_summary.json`
- `runs/eval/audit_status.json`
- `reports/tables/summary_mean_std.json`
