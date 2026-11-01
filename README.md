# TriAudit

### A Three-Dimensional Audit of Adversarially Trained Image Classifiers

**TriAudit** is an ongoing research project investigating adversarially trained image classifiers through three dimensions:

1. **Clean Utility** – Measuring the clean-accuracy trade-off introduced by adversarial training.
2. **Adversarial Robustness** – Evaluating model performance under white-box adversarial attacks.
3. **Confidence Calibration** – Examining prediction confidence on clean and adversarial inputs.

### Experimental Setup

- **Dataset:** CIFAR-10
- **Architecture:** CIFAR-adapted ResNet-18
- **Training Methods:** Standard Training (ST) and PGD Adversarial Training (PGD-AT)
- **Attacks:** FGSM, PGD-20, and Multi-restart PGD
- **Calibration Metrics:** ECE, NLL, and Brier Score
- **Framework:** Python and PyTorch

### Research Objective

The goal is to develop a reproducible evaluation framework that examines clean performance, adversarial robustness, and confidence calibration together rather than relying solely on adversarial accuracy.

### Development Status

<badge color="warning">Under Development</badge>

The project is currently in the architecture and implementation stage. Experimental results, analysis, and visualizations will be added as development progresses.

### Repository

```text
triaudit/
├── configs/
├── data/
├── src/
├── scripts/
├── tests/
├── runs/
├── reports/
└── README.md
```

### Disclaimer

This project is an empirical research study on CIFAR-10. It does not claim universal adversarial robustness or real-world deployment safety.

### This Project is Under Developement

---

**TriAudit — Evaluating More Than Robust Accuracy.**
