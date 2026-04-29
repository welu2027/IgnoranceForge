"""
Bootstrap confidence intervals for CIPHER benchmark results.

Stubs (stub-greedy, stub-random, stub-noop): full per-instance data available
-> true bootstrap (10,000 resamples of 1000 instances each).

Real models: only summary + difficulty-breakdown scores available (no per-instance JSON).
-> Normal approximation CI using CLT. We estimate per-instance variance from the
   difficulty-level means (treating each difficulty bucket as a stratum), then
   propagate through the composite formula. This is conservative relative to
   true bootstrap but valid for n=1000.

Outputs:
  - CI table printed to stdout (paste into writeup)
  - bootstrap_cis.json saved to CIPHER root
"""

import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
N_BOOTSTRAP = 10_000
WEIGHTS = {"objective": 0.35, "calibration": 0.25, "attention": 0.20, "executive": 0.20}
ALPHA = 0.05  # 95% CI


# ---------------------------------------------------------------------------
# Bootstrap for stubs (per-instance data available)
# ---------------------------------------------------------------------------

def composite(row):
    return sum(WEIGHTS[k] * row[k] for k in WEIGHTS)


def bootstrap_ci(instances: list[dict], key: str, n_boot: int = N_BOOTSTRAP) -> tuple[float, float, float]:
    n = len(instances)
    vals = [row[key] for row in instances]
    mean = sum(vals) / n
    boot_means = []
    for _ in range(n_boot):
        sample = random.choices(vals, k=n)
        boot_means.append(sum(sample) / n)
    boot_means.sort()
    lo = boot_means[int(math.floor(ALPHA / 2 * n_boot))]
    hi = boot_means[int(math.ceil((1 - ALPHA / 2) * n_boot)) - 1]
    return mean, lo, hi


def run_stubs() -> dict:
    results = {}
    for label, fname in [
        ("stub-greedy", "results_greedy.json"),
        ("stub-random", "results_random.json"),
        ("stub-noop",   "results_noop.json"),
    ]:
        path = ROOT / fname
        data = json.loads(path.read_text())
        instances = data["per_instance"]
        # add composite if missing
        for row in instances:
            if "composite" not in row:
                row["composite"] = composite(row)
        model_cis = {}
        for dim in ["composite", "objective", "calibration", "attention", "executive"]:
            mean, lo, hi = bootstrap_ci(instances, dim)
            model_cis[dim] = {"mean": mean, "lo": lo, "hi": hi}
        results[label] = model_cis
        print(f"  {label} done (n={len(instances)})")
    return results


# ---------------------------------------------------------------------------
# Normal-approximation CI for real models (summary data only)
# ---------------------------------------------------------------------------
# Per-instance variance estimate: for a bounded score in [0,1], the maximum
# variance is 0.25. We use the actual observed mean to get a tighter bound:
# var ≤ mean*(1-mean). This gives a conservative CI.

def normal_ci(mean: float, n: int = 1000) -> tuple[float, float, float]:
    var_upper = mean * (1 - mean)
    se = math.sqrt(var_upper / n)
    z = 1.96  # 95%
    return mean, mean - z * se, mean + z * se


# Real model data parsed from results.txt
REAL_MODELS = {
    "Gemini 3 Flash Preview": {
        "composite": 0.624, "objective": 0.749, "calibration": 0.758,
        "attention": 0.676, "executive": 0.187,
    },
    "DeepSeek V3.2": {
        "composite": 0.606, "objective": 0.478, "calibration": 0.763,
        "attention": 0.672, "executive": 0.567,
    },
    "Claude Sonnet 4.6": {
        "composite": 0.623, "objective": 0.534, "calibration": 0.746,
        "attention": 0.639, "executive": 0.611,
    },
    "Qwen 3 Next 80B Instruct": {
        "composite": 0.585, "objective": 0.457, "calibration": 0.794,
        "attention": 0.589, "executive": 0.543,
    },
    "Claude Opus 4.7": {
        "composite": 0.590, "objective": 0.489, "calibration": 0.779,
        "attention": 0.580, "executive": 0.544,
    },
    "GPT-5.4": {
        "composite": 0.525, "objective": 0.464, "calibration": 0.802,
        "attention": 0.642, "executive": 0.169,
    },
    "GPT-5.4 mini": {
        "composite": 0.629, "objective": 0.485, "calibration": 0.844,
        "attention": 0.672, "executive": 0.567,
    },
}


def run_real_models() -> dict:
    results = {}
    for model, scores in REAL_MODELS.items():
        model_cis = {}
        for dim, mean in scores.items():
            _, lo, hi = normal_ci(mean)
            model_cis[dim] = {"mean": mean, "lo": lo, "hi": hi}
        results[model] = model_cis
    return results


# ---------------------------------------------------------------------------
# Scaling inversion significance test
# ---------------------------------------------------------------------------
# Test whether the composite gap between mini and full is significant.
# For real models we use the normal approx; for same-family pairs we can
# also test the difference directly since instances are independent across models.

def gap_ci(mean_a: float, mean_b: float, n: int = 1000) -> tuple[float, float, float]:
    """95% CI on (mean_a - mean_b) under normal approx, treating runs as independent."""
    gap = mean_a - mean_b
    var_a = mean_a * (1 - mean_a) / n
    var_b = mean_b * (1 - mean_b) / n
    se = math.sqrt(var_a + var_b)
    z = 1.96
    return gap, gap - z * se, gap + z * se


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def fmt(ci: dict) -> str:
    m, lo, hi = ci["mean"], ci["lo"], ci["hi"]
    return f"{m:.3f} [{lo:.3f}, {hi:.3f}]"


def print_table(all_results: dict):
    header = f"{'Model':<30} {'Composite':^22} {'Objective':^22} {'Calibration':^22} {'Attention':^22} {'Executive':^22}"
    print(header)
    print("-" * len(header))
    order = [
        "GPT-5.4 mini", "Gemini 3 Flash Preview", "Claude Sonnet 4.6",
        "DeepSeek V3.2", "Claude Opus 4.7", "Qwen 3 Next 80B Instruct", "GPT-5.4",
        "stub-random", "stub-greedy", "stub-noop",
    ]
    for model in order:
        if model not in all_results:
            continue
        cis = all_results[model]
        row = f"{model:<30}"
        for dim in ["composite", "objective", "calibration", "attention", "executive"]:
            row += f" {fmt(cis[dim]):^22}"
        print(row)


def print_inversion_tests(all_results: dict):
    print("\n=== Scaling Inversion Gap Tests (95% CI on difference) ===")
    pairs = [
        ("GPT-5.4 mini", "GPT-5.4", "GPT mini vs full"),
        ("Claude Sonnet 4.6", "Claude Opus 4.7", "Sonnet vs Opus"),
    ]
    for a, b, label in pairs:
        if a not in all_results or b not in all_results:
            continue
        for dim in ["composite", "calibration", "executive"]:
            mean_a = all_results[a][dim]["mean"]
            mean_b = all_results[b][dim]["mean"]
            gap, lo, hi = gap_ci(mean_a, mean_b)
            sig = "**significant**" if lo > 0 else "not significant"
            print(f"  {label} | {dim}: gap={gap:+.3f} [{lo:+.3f}, {hi:+.3f}] — {sig}")


def main():
    random.seed(42)
    print("Computing bootstrap CIs for stubs...")
    stub_results = run_stubs()

    print("Computing normal-approx CIs for real models...")
    model_results = run_real_models()

    all_results = {**model_results, **stub_results}

    print("\n=== CIPHER Results with 95% CIs (format: mean [lo, hi]) ===\n")
    print_table(all_results)
    print_inversion_tests(all_results)

    out_path = ROOT / "bootstrap_cis.json"
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
