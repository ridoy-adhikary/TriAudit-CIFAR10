# TriAudit Research Report

Generated: 2026-11-02T07:04:46

## Methods

- Dataset: CIFAR-10 with 45k/5k split; official test held out until final eval.
- Model: CIFAR-adapted ResNet-18.
- Conditions: Standard Training (ST) and PGD-AT (PGD-10, ε=8/255, α=2/255, random start).
- Seeds: paired {0,1,2} with shared initialization.
- Attacks: FGSM, PGD-20, multi-restart PGD (50 steps / 10 restarts) at primary ε=8/255; sensitivity at 4/255 and 12/255.
- Calibration: ECE (15 equal-width bins), NLL, multiclass Brier on clean/FGSM/PGD-20.

## Results (from runs/ artifacts)

Mean ± std across seeds:

- **clean_st**: 94.0500 ± 0.2307 (values=[94.03, 94.29, 93.83])
- **clean_at**: 82.2767 ± 0.3787 (values=[82.7, 81.97, 82.16])
- **delta_clean**: -11.7733 ± 0.5030 (values=[-11.329999999999998, -12.320000000000007, -11.670000000000002])
- **pgd20_at**: 48.4033 ± 0.0839 (values=[48.5, 48.35, 48.36])
- **attack_gap_at**: 0.1872 ± 0.1624 (values=[0.00390625, 0.24453125000000142, 0.31312499999999943])
- **ece_clean_at**: 0.0984 ± 0.0069 (values=[0.09072872996330261, 0.10398115962743759, 0.10046122968196869])

### Per-seed TriAudit Audit Status

- Seed 0: **Major concerns**
  - measurements: `{'clean_st': 94.03, 'clean_at': 82.7, 'pgd20_at': 48.5, 'mr_at': 48.49609375, 'ece': {'clean': 0.09072872996330261, 'fgsm': 0.09009166061878204, 'pgd20': 0.17065085470676422}}`
- Seed 1: **Caveat**
  - measurements: `{'clean_st': 94.29, 'clean_at': 81.97, 'pgd20_at': 48.35, 'mr_at': 48.10546875, 'ece': {'clean': 0.10398115962743759, 'fgsm': 0.07701993733644485, 'pgd20': 0.14793436229228973}}`
- Seed 2: **Major concerns**
  - measurements: `{'clean_st': 93.83, 'clean_at': 82.16, 'pgd20_at': 48.36, 'mr_at': 48.046875, 'ece': {'clean': 0.10046122968196869, 'fgsm': 0.08097626268863678, 'pgd20': 0.15570448338985443}}`

## Limitations and claim boundaries

- CIFAR-10 only; single architecture; limited seeds.
- White-box ℓ∞ threat model only.
- Alert thresholds are project-defined, not universal standards.
- Zero TriAudit flags does **not** mean the model is safe to deploy.
- Missing metrics must never be treated as passing.

## Artifact index

- `runs/eval/attacks_summary.json`
- `runs/eval/calibration_summary.json`
- `runs/eval/audit_status.json`
- `reports/tables/summary_mean_std.json`
