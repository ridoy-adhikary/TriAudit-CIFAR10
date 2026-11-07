"""TriAudit Layer 1–3 flags and status aggregation."""

from __future__ import annotations


def layer1_clean_gap(clean_at: float, clean_st: float) -> dict:
    delta = clean_at - clean_st  # percentage points if inputs are in %
    flag = delta < -5.0
    return {"delta_clean": delta, "flag": flag, "threshold": -5.0}


def layer2_attack_gap(rob_pgd20: float, rob_mr: float) -> dict:
    gap = rob_pgd20 - rob_mr
    flag = gap > 5.0
    return {"attack_gap": gap, "flag": flag, "threshold": 5.0}


def layer3_calibration(ece_values: dict[str, float | None]) -> dict:
    """Flag if any evaluated ECE > 0.15. Missing keys -> UNEVALUATED handling upstream."""
    triggered = []
    for k, v in ece_values.items():
        if v is None:
            continue
        if v > 0.15:
            triggered.append(k)
    return {
        "ece_by_condition": ece_values,
        "flag": len(triggered) > 0,
        "triggered_conditions": triggered,
        "threshold": 0.15,
    }


def audit_status(flags: list[bool | None]) -> dict:
    """flags: True=flagged, False=clear, None=UNEVALUATED (does not count as pass or flag)."""
    unevaluated = [i for i, f in enumerate(flags, 1) if f is None]
    evaluated_flags = [f for f in flags if f is not None]
    n_flags = sum(1 for f in evaluated_flags if f)
    labels = {
        0: "No flags under tested scope",
        1: "Caveat",
        2: "Major concerns",
        3: "Critical concerns",
    }
    # Only map among evaluated count of True flags; if any UNEVALUATED, annotate.
    status = labels.get(n_flags, "Critical concerns")
    if unevaluated:
        status = f"{status} (UNEVALUATED layers: {unevaluated})"
    return {
        "n_flags": n_flags,
        "status": status,
        "unevaluated_layers": unevaluated,
        "note": "Zero flags does not mean safe to deploy.",
    }
