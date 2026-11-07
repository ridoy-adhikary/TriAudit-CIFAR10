# TriAudit — Server-Side Auto Implementation Instruction

**Purpose:** Run the full TriAudit implementation on the shared server **without needing your laptop attached**.  
**Authority:** Follow `TriAudit_Final_Project_Architecture.md` exactly.  
**Environment lock:** **`ridoy` conda env only** — never modify shared `/home/sda/Anaconda` or other envs.  
**Data path lock:** use project folder `dataset/` (architecture `data/` maps here).

---

## 0. Architecture verification (DONE before any coding phase)

Before Phase 1+, the orchestrator **must** pass Phase 0:

```bash
/home/sda/Users/temp_user/.conda/envs/ridoy/bin/python \
  scripts/auto/verify_architecture.py
```

Gate artifact: `project_monitor/phase0_architecture_verify.json` with `"passed": true`.

### Architecture locks confirmed by verifier

| Lock | Required value |
|---|---|
| Dataset | CIFAR-10, 45k/5k split, official test held out |
| Model | CIFAR-adapted ResNet-18 |
| Training | ST + PGD-AT (PGD-10, ε=8/255, α=2/255, random start) |
| Eval attacks | FGSM, PGD-20, multi-restart PGD (50/10 proposed); AutoAttack optional |
| Calibration | ECE 15 bins + NLL + Brier |
| Pairing | Seed-paired ST↔PGD-AT; no second vanilla branch |
| Order | Train → freeze checkpoint → attack/calibration |
| Out of core | No EOT-PGD, no autoencoder blocker |
| Alerts | Project-defined; missing ≠ pass; 0 flags ≠ deploy-safe |
| Env | ridoy only |

**Rule:** If Phase 0 fails, **do not start implementation**.

---

## 1. Hard phase gate rule (non-negotiable)

```text
FOR phase in 0..9:
  IF phase > 0 AND previous phase gate passed != true:
      STOP  # do not start this phase
  RUN implementation for phase
  RUN verification for phase
  IF verification failed:
      STOP  # do not continue
```

Previous-phase verification is mandatory. No skipping in auto mode.

Phase state file: `project_monitor/phase_state.json`  
Per-phase gates: `project_monitor/phaseN_*.json`

---

## 2. Phase map (architecture milestones)

| Phase | Name | Implements | Gate file |
|---|---|---|---|
| 0 | Architecture verify | Consistency + locks | `phase0_architecture_verify.json` |
| 1 | Dataset + model setup | Split 45k/5k, ResNet-18, package scaffold | `phase1_verify.json` |
| 2 | ST pilot | SGD ST + scheduler lock | `phase2_verify.json` |
| 3 | PGD-AT pilot | PGD-10 AT + constraint tests | `phase3_verify.json` |
| 4 | Paired training | Seeds {0,1,2} ST+AT, freeze ckpts | `phase4_verify.json` |
| 5 | Attack evaluation | FGSM / PGD-20 / multi-restart (± AA) | `phase5_verify.json` |
| 6 | Calibration | ECE-15 / NLL / Brier | `phase6_verify.json` |
| 7 | Audit aggregation | Layers + TriAudit Audit Status | `phase7_verify.json` |
| 8 | Stats + figures | mean±std, plots | `phase8_verify.json` |
| 9 | Research report | Methods/results/limitations | `phase9_verify.json` |

Each phase verifier must encode the architecture **acceptance criteria** for that milestone.

---

## 3. How to run unattended (laptop OFF)

### One-time checks

```bash
cd /home/sda/Users/temp_user/Ridoy/Other_Projects/TriAudit

# 1) ridoy only
/home/sda/Users/temp_user/.conda/envs/ridoy/bin/python -c "import sys; print(sys.executable)"

# 2) CUDA for training phases (2–4). If false, fix GPU driver before full training.
/home/sda/Users/temp_user/.conda/envs/ridoy/bin/python -c "import torch; print('cuda', torch.cuda.is_available())"
```

### Start detached

```bash
bash scripts/auto/start_detached.sh
```

This uses **tmux** if available, otherwise **nohup**. Either way, closing the laptop/SSH does not stop the job.

### Monitor from anywhere later

```bash
# status
cat project_monitor/phase_state.json
tail -f project_monitor/orchestrator.log
tail -f project_monitor/project_activity.log
tail -f project_monitor/code_changes.log

# if tmux:
tmux attach -t triaudit_auto
```

### Stop (only if needed)

```bash
# nohup:
kill "$(cat project_monitor/orchestrator.pid)"
# tmux:
tmux kill-session -t triaudit_auto
```

---

## 4. Per-phase verification checklist (must pass before next)

### Phase 0 — Architecture
- [ ] Architecture file present
- [ ] Required methods/thresholds present
- [ ] Milestones 1–9 present
- [ ] CIFAR-10 present under `dataset/raw`
- [ ] ridoy python exists

### Phase 1 — Dataset + model
- [ ] `dataset/splits` has deterministic 45k/5k indices
- [ ] Train/val/test disjoint; test = official 10k
- [ ] Package under `src/triaudit/`
- [ ] CIFAR ResNet-18 forward `B×3×32×32 → B×10`
- [ ] Configs created under `configs/`

### Phase 2 — ST pilot
- [ ] ST trains without NaNs
- [ ] Val metrics logged; checkpoint reloadable
- [ ] Epochs + LR schedule locked in `configs/train/st.yaml`
- [ ] **No test-set tuning**

### Phase 3 — PGD-AT pilot
- [ ] PGD-10, ε=8/255, α=2/255, random start, clip [0,1]
- [ ] Constraint unit tests pass
- [ ] Pilot epochs complete

### Phase 4 — Paired training
- [ ] Seeds paired (target 3)
- [ ] Frozen checkpoints for ST and PGD-AT per seed
- [ ] Metadata recorded under `runs/`

### Phase 5 — Attacks (frozen only)
- [ ] FGSM, PGD-20, multi-restart PGD done
- [ ] ε ∈ {4,8,12}/255 sensitivity
- [ ] Constraint validation pass
- [ ] AttackGap computed
- [ ] No weight updates during eval

### Phase 6 — Calibration
- [ ] ECE with **15** equal-width bins
- [ ] NLL + Brier always present
- [ ] Clean / FGSM / PGD-20 conditions

### Phase 7 — Audit status
- [ ] Layer flags use project-defined thresholds
- [ ] `UNEVALUATED` ≠ pass
- [ ] TriAudit Audit Status emitted with raw metrics

### Phase 8 — Analysis
- [ ] mean ± std across seeds
- [ ] Figures/tables under `reports/` from logged metrics only

### Phase 9 — Report
- [ ] Report cites `runs/` artifacts
- [ ] Limitations + claim boundaries included
- [ ] No fabricated results / deploy-safety claims

---

## 5. Environment & safety rules

1. **Python:** `/home/sda/Users/temp_user/.conda/envs/ridoy/bin/python` only.  
2. **Installs:** only into `ridoy` (`.../envs/ridoy/bin/pip`).  
3. **Never** modify shared Anaconda base or other users’ envs.  
4. **Writes:** prefer only under this TriAudit project tree.  
5. **GPU:** Phases 2–4 require CUDA unless `TRIAUDIT_ALLOW_CPU=1` (smoke only).  
6. **Logs:** all progress goes to `project_monitor/project_activity.log` and code edits to `code_changes.log`.  
7. **On failure:** orchestrator **stops**; fix, re-verify failed phase, then resume with `TRIAUDIT_START_PHASE=<failed>`.

---

## 6. Resume after failure or reboot

```bash
# Inspect which phase failed
cat project_monitor/phase_state.json

# Resume from a specific phase (previous gate must already be green)
TRIAUDIT_START_PHASE=3 TRIAUDIT_END_PHASE=9 bash scripts/auto/start_detached.sh
```

If the server rebooted, start_detached again; completed phase gates remain on disk and block illegal skips.

---

## 7. Current known server constraints (check every start)

| Item | Implication |
|---|---|
| CUDA / nvidia driver must work | Otherwise training phases halt by design |
| tmux may be missing | Fallback is nohup (still survives logout) |
| Shared disk | Keep artifacts in project `runs/` + `project_monitor/` |

---

## 8. Operator quick start (copy/paste)

```bash
cd /home/sda/Users/temp_user/Ridoy/Other_Projects/TriAudit

# Phase 0 only (architecture gate)
/home/sda/Users/temp_user/.conda/envs/ridoy/bin/python scripts/auto/verify_architecture.py

# Full auto 0→9 detached
bash scripts/auto/start_detached.sh

# Watch
tail -f project_monitor/orchestrator.log
```

---

## 9. What “done” means

Auto mode is complete only when:

1. `project_monitor/phase_state.json` shows phases 0–9 `verified: true`
2. `project_monitor/orchestrator.status` contains `completed`
3. Research outputs exist under `reports/` and metrics under `runs/`
4. Activity log contains a final `ALL REQUESTED PHASES COMPLETE` line

Until then, treat the project as **in progress**, not finished.
