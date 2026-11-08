"""
Evaluation and Metrics Benchmark Suite for CallTrust Ensemble.

Computes:
  - Equal Error Rate (EER) via ROC curve threshold sweep
  - Minimum Detection Cost Function (minDCF) per ASVspoof protocol
  - Accuracy, Precision, Recall, F1, and False-Positive Rate (FPR)
  - Expected Calibration Error (ECE)
  - Real-Time Factor (RTF) and inference latency distribution (p50, p95, p99)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class EvaluationReport:
    eer: float
    min_dcf: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    fpr: float
    ece: float
    latency_ms_p50: float
    latency_ms_p95: float
    latency_ms_p99: float
    rtf: float  # Real-Time Factor (< 1.0 means faster than real-time)

    def print_summary(self) -> None:
        print("\n" + "═" * 56)
        print("   CallTrust Engine ML Benchmark & Validation Report   ")
        print("═" * 56)
        print(f"  Equal Error Rate (EER):           {self.eer * 100:.2f}%")
        print(f"  Minimum DCF (minDCF):             {self.min_dcf:.4f}")
        print(f"  Accuracy:                         {self.accuracy * 100:.2f}%")
        print(f"  Precision:                        {self.precision * 100:.2f}%")
        print(f"  Recall:                           {self.recall * 100:.2f}%")
        print(f"  F1 Score:                         {self.f1:.4f}")
        print(f"  False Positive Rate (FPR):        {self.fpr * 100:.2f}% (Target: < 2.0%)")
        print(f"  Expected Calibration Error (ECE): {self.ece * 100:.2f}%")
        print("-" * 56)
        print(f"  Inference Latency (p50):          {self.latency_ms_p50:.2f} ms")
        print(f"  Inference Latency (p95):          {self.latency_ms_p95:.2f} ms")
        print(f"  Inference Latency (p99):          {self.latency_ms_p99:.2f} ms")
        print(f"  Real-Time Factor (RTF):           {self.rtf:.4f} (Streaming budget: < 0.20)")
        print("═" * 56 + "\n")


def compute_eer(y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[float, float]:
    """
    Compute Equal Error Rate (EER) and operating threshold.
    y_true: 0 for bona fide, 1 for spoof.
    y_scores: probability or logit of being spoof.
    """
    thresholds = np.sort(np.unique(y_scores))
    if len(thresholds) == 0:
        return 0.0, 0.5

    fnr_list = []
    fpr_list = []

    positives = np.sum(y_true == 1)
    negatives = np.sum(y_true == 0)

    if positives == 0 or negatives == 0:
        return 0.0, 0.5

    for th in thresholds:
        preds = (y_scores >= th).astype(int)
        fn = np.sum((preds == 0) & (y_true == 1))
        fp = np.sum((preds == 1) & (y_true == 0))
        fnr_list.append(fn / positives)
        fpr_list.append(fp / negatives)

    fnr = np.array(fnr_list)
    fpr = np.array(fpr_list)

    # Find the point where FNR and FPR are closest
    diff = np.abs(fnr - fpr)
    idx = np.argmin(diff)
    eer = float((fnr[idx] + fpr[idx]) / 2.0)
    best_th = float(thresholds[idx])

    return eer, best_th


def compute_min_dcf(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    p_target: float = 0.05,
    c_miss: float = 1.0,
    c_fa: float = 10.0,
) -> float:
    """
    Compute ASVspoof-style Minimum Detection Cost Function (minDCF).
    Penalizes false alarms heavily (c_fa=10) per product rule of avoiding false accusations.
    """
    thresholds = np.sort(np.unique(y_scores))
    costs = []

    positives = np.sum(y_true == 1)
    negatives = np.sum(y_true == 0)

    if positives == 0 or negatives == 0:
        return 0.0

    c_default = min(c_miss * p_target, c_fa * (1 - p_target))

    for th in thresholds:
        preds = (y_scores >= th).astype(int)
        p_miss = np.sum((preds == 0) & (y_true == 1)) / positives
        p_fa = np.sum((preds == 1) & (y_true == 0)) / negatives

        cost = c_miss * p_target * p_miss + c_fa * (1 - p_target) * p_fa
        norm_cost = cost / c_default
        costs.append(norm_cost)

    return float(np.min(costs)) if costs else 0.0


def compute_ece(y_true: np.ndarray, y_probs: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error (ECE)."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(y_true)

    for i in range(n_bins):
        b_low, b_high = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (y_probs >= b_low) & (y_probs < b_high)
        bin_count = np.sum(mask)

        if bin_count > 0:
            bin_acc = np.mean(y_true[mask] == (y_probs[mask] >= 0.5).astype(int))
            bin_conf = np.mean(y_probs[mask])
            ece += (bin_count / total) * abs(bin_acc - bin_conf)

    return float(ece)


def evaluate_predictions(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    latencies_ms: List[float],
    audio_duration_sec: float = 3.0,
    threshold: float = 0.5,
) -> EvaluationReport:
    """Compute all standard evaluation metrics from predictions and latency timings."""
    preds = (y_scores >= threshold).astype(int)

    tp = np.sum((preds == 1) & (y_true == 1))
    fp = np.sum((preds == 1) & (y_true == 0))
    tn = np.sum((preds == 0) & (y_true == 0))
    fn = np.sum((preds == 0) & (y_true == 1))

    acc = (tp + tn) / max(1, len(y_true))
    prec = tp / max(1, (tp + fp))
    rec = tp / max(1, (tp + fn))
    f1 = 2 * (prec * rec) / max(1e-9, (prec + rec))
    fpr = fp / max(1, (fp + tn))

    eer, _ = compute_eer(y_true, y_scores)
    min_dcf = compute_min_dcf(y_true, y_scores)
    ece = compute_ece(y_true, y_scores)

    lat_sorted = np.sort(latencies_ms)
    p50 = float(np.percentile(lat_sorted, 50))
    p95 = float(np.percentile(lat_sorted, 95))
    p99 = float(np.percentile(lat_sorted, 99))

    # RTF = processing_time / audio_duration
    rtf = (p50 / 1000.0) / audio_duration_sec

    return EvaluationReport(
        eer=eer,
        min_dcf=min_dcf,
        accuracy=acc,
        precision=prec,
        recall=rec,
        f1=f1,
        fpr=fpr,
        ece=ece,
        latency_ms_p50=p50,
        latency_ms_p95=p95,
        latency_ms_p99=p99,
        rtf=rtf,
    )


if __name__ == "__main__":
    # Self-test benchmark with simulated sample data
    np.random.seed(42)
    n = 1000
    y_true = np.random.binomial(1, 0.4, n)

    # Simulated calibrated predictions (low false positive rate)
    y_scores = np.zeros(n)
    for i in range(n):
        if y_true[i] == 1:
            y_scores[i] = np.random.beta(7, 2)
        else:
            y_scores[i] = np.random.beta(1.5, 8)

    latencies = np.random.normal(18.5, 3.0, n).tolist()
    report = evaluate_predictions(y_true, y_scores, latencies)
    report.print_summary()
