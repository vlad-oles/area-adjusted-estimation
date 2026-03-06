#!/usr/bin/env python3
"""
Batch-based Adaptive Stratified Sampling for Rare Class Area Estimation
========================================================================
Implements Algorithm 1 from:
  "Adaptive area estimation via batch-based stratified sampling"

Usage:
    python estimate_area.py --N1 5000 --N2 95000 --delta 0.05 --alpha 0.05 --batch 50

The algorithm iteratively samples units from two classes (Class 1 = rare/target,
Class 2 = background), collects misclassification counts from the user, and
refines the area estimate until a target precision is achieved.
"""

import argparse
import math
import sys
from scipy import stats
import numpy as np


def credible_interval(x1, n1, x2, n2, N1, N2, alpha):
    """
    Return the equal-tail (1-alpha) credible interval [N_L, N_U] for N_{•1}.
    Uses Monte-Carlo sampling from the joint Beta posterior.
    """
    N = N1 + N2
    rng = np.random.default_rng(0)
    n_samples = 300_000

    a1, b1 = x1 + 1, n1 - x1 + 1
    a2, b2 = x2 + 1, n2 - x2 + 1

    p1_s = rng.beta(a1, b1, n_samples)
    p2_s = rng.beta(a2, b2, n_samples)
    N_hat_s = (1 - p1_s) * N1 + p2_s * N2

    N_L = float(np.quantile(N_hat_s, alpha / 2))
    N_U = float(np.quantile(N_hat_s, 1 - alpha / 2))
    return N_L, N_U


# ---------------------------------------------------------------------------
# Area estimate
# ---------------------------------------------------------------------------

def point_estimate(x1, n1, x2, n2, N1, N2):
    """MAP area-adjusted estimate: N̂_{•1} = (1 - x1/n1)*N1• + (x2/n2)*N2•"""
    p1_hat = x1 / n1 if n1 > 0 else 0.0
    p2_hat = x2 / n2 if n2 > 0 else 0.0
    return (1 - p1_hat) * N1 + p2_hat * N2


# ---------------------------------------------------------------------------
# Optimal allocation ratio (λ_opt)
# ---------------------------------------------------------------------------

def optimal_lambda(x1, n1, x2, n2, N1, N2):
    """
    Compute λ_opt using posterior means p̃1, p̃2.
    λ_opt = (Ñ_{•1} * sqrt(p̃1*(1-p̃1))) / ((N - Ñ_{•1}) * sqrt(p̃2*(1-p̃2)))
    """
    p1_tilde = (x1 + 1) / (n1 + 2)
    p2_tilde = (x2 + 1) / (n2 + 2)

    N = N1 + N2
    N_tilde = (1 - p1_tilde) * N1 + p2_tilde * N2

    num = N_tilde * math.sqrt(p1_tilde * (1 - p1_tilde))
    den = (N - N_tilde) * math.sqrt(p2_tilde * (1 - p2_tilde))

    if den == 0:
        return 1.0
    return num / den


# ---------------------------------------------------------------------------
# Precision check
# ---------------------------------------------------------------------------

def precision_achieved(N_L, N_U, N_hat, delta):
    """
    Check if [N_L, N_U] ⊆ [N̂/(1+δ), N̂/(1-δ)].
    """
    if N_hat <= 0:
        return False
    lower_bound = N_hat / (1 + delta)
    upper_bound = N_hat / (1 - delta)
    return N_L >= lower_bound and N_U <= upper_bound


# ---------------------------------------------------------------------------
# Interactive sampling prompt
# ---------------------------------------------------------------------------

def prompt_sample(n1_batch, n2_batch, iteration):
    """
    Instruct the user to sample units and report misclassification counts.
    Returns (x1_batch, x2_batch).
    """
    print()
    print("=" * 60)
    print(f"  ITERATION {iteration}  —  Sampling Instructions")
    print("=" * 60)
    print(f"  Please sample and label the following units:")
    print(f"    • {n1_batch:>6} units from CLASS 1  (rare / target class)")
    print(f"    • {n2_batch:>6} units from CLASS 2  (background class)")
    print()
    print("  After labelling, count misclassifications:")
    print("    x1 = number of Class-1 units that are actually Class 2")
    print("    x2 = number of Class-2 units that are actually Class 1")
    print("-" * 60)

    while True:
        try:
            x1 = int(input(f"  Enter x1 (misclassified from Class 1, 0–{n1_batch}): ").strip())
            if not (0 <= x1 <= n1_batch):
                print(f"  ⚠  x1 must be between 0 and {n1_batch}. Try again.")
                continue
            break
        except ValueError:
            print("  ⚠  Please enter an integer.")

    while True:
        try:
            x2 = int(input(f"  Enter x2 (misclassified from Class 2, 0–{n2_batch}): ").strip())
            if not (0 <= x2 <= n2_batch):
                print(f"  ⚠  x2 must be between 0 and {n2_batch}. Try again.")
                continue
            break
        except ValueError:
            print("  ⚠  Please enter an integer.")

    return x1, x2


# ---------------------------------------------------------------------------
# Main algorithm  (Algorithm 1)
# ---------------------------------------------------------------------------

def estimate_area(N1, N2, delta, alpha, b, simulate=None):
    """
    ESTIMATEAREA(N1•, N2•, δ, α, b)

    Parameters
    ----------
    N1, N2  : mapped areas (unit counts) for Class 1 and Class 2
    delta   : relative precision target  (δ)
    alpha   : credible-interval significance level  (α)
    b       : batch size
    simulate: optional callable(n1_batch, n2_batch) -> (x1, x2) for
              non-interactive / simulation mode
    """
    N = N1 + N2

    # Initialise accumulators
    n1_total, n2_total = 0, 0
    x1_total, x2_total = 0, 0
    t = 0

    # First batch allocated proportionally to mapped areas (line 5)
    n1_batch = math.floor((N1 / N) * b)
    n1_batch = max(1, min(n1_batch, b - 1))   # ensure at least 1 per class
    n2_batch = b - n1_batch

    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   Rare-Class Area Estimation — Adaptive Stratified       ║")
    print("║   Sampling (Algorithm 1)                                 ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"  N1• = {N1:,}   N2• = {N2:,}   N = {N:,}")
    print(f"  δ = {delta}   α = {alpha}   batch size b = {b}")
    print()

    history = []

    while True:
        t += 1

        # ── Sample and collect misclassification counts ──────────────────
        if simulate is not None:
            x1_batch, x2_batch = simulate(n1_batch, n2_batch)
            print(f"\n[Iter {t}] Sampling {n1_batch} from Class 1, "
                  f"{n2_batch} from Class 2  →  x1={x1_batch}, x2={x2_batch}")
        else:
            x1_batch, x2_batch = prompt_sample(n1_batch, n2_batch, t)

        # ── Accumulate (lines 9–10) ──────────────────────────────────────
        n1_total += n1_batch
        n2_total += n2_batch
        x1_total += x1_batch
        x2_total += x2_batch

        # Guard: avoid division by zero
        if n1_total == 0 or n2_total == 0:
            print("  ⚠  Accumulated sample size is zero for one class. Skipping.")
            continue

        # ── Point estimate (line 11) ─────────────────────────────────────
        N_hat = point_estimate(x1_total, n1_total, x2_total, n2_total, N1, N2)

        # ── Credible interval (line 12) ──────────────────────────────────
        N_L, N_U = credible_interval(x1_total, n1_total,
                                     x2_total, n2_total,
                                     N1, N2, alpha)

        # ── Report to user ───────────────────────────────────────────────
        print()
        print(f"  ┌─ Iteration {t} Results {'─'*35}")
        print(f"  │  Cumulative samples : n1={n1_total}  n2={n2_total}")
        print(f"  │  Misclassifications : x1={x1_total}  x2={x2_total}")
        print(f"  │  Estimate  N̂_{{•1}}  : {N_hat:,.1f}")
        print(f"  │  {int((1-alpha)*100)}% credible interval : [{N_L:,.1f},  {N_U:,.1f}]")

        # Precision bounds
        if N_hat > 0:
            prec_lo = N_hat / (1 + delta)
            prec_hi = N_hat / (1 - delta)
            print(f"  │  Precision target  : [{prec_lo:,.1f},  {prec_hi:,.1f}]")

        achieved = precision_achieved(N_L, N_U, N_hat, delta)
        print(f"  │  Target achieved?  : {'✓ YES — stopping.' if achieved else '✗ No — continuing.'}")
        print(f"  └{'─'*50}")

        history.append({
            "iteration": t,
            "n1": n1_total, "n2": n2_total,
            "x1": x1_total, "x2": x2_total,
            "N_hat": N_hat, "N_L": N_L, "N_U": N_U,
            "achieved": achieved,
        })

        # ── Stopping criterion (line 18) ──────────────────────────────────
        if achieved:
            break

        # ── Optimal allocation for next batch (lines 13–17) ───────────────
        lam = optimal_lambda(x1_total, n1_total, x2_total, n2_total, N1, N2)

        n1_next_raw = (n2_total + b) * lam / (1 + lam) - n1_total
        n1_batch = max(1, math.floor(n1_next_raw))
        n1_batch = min(n1_batch, b - 1)
        n2_batch = b - n1_batch



    print()
    print("══════════════════════════════════════════════════════════")
    print(f"  FINAL ESTIMATE:  N̂_{{•1}} = {N_hat:,.1f}")
    print(f"  {int((1-alpha)*100)}% credible interval: [{N_L:,.1f},  {N_U:,.1f}]")
    print(f"  Total samples used: n1={n1_total}, n2={n2_total}  "
          f"(total={n1_total+n2_total})")
    print("══════════════════════════════════════════════════════════")

    return N_hat, N_L, N_U, history


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Batch-based adaptive stratified sampling for rare-class area estimation.\n\n"
            "The algorithm iteratively instructs you to label batches of units from\n"
            "Class 1 (rare/target) and Class 2 (background), collects your\n"
            "misclassification counts, and refines the area estimate until the\n"
            "requested precision is achieved.\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Interactive mode:\n"
            "  python estimate_area.py --N1 5000 --N2 95000 --delta 0.05 --alpha 0.05 --batch 50\n\n"
            "  # Simulation mode (random oracle, for testing):\n"
            "  python estimate_area.py --N1 5000 --N2 95000 --delta 0.1 --alpha 0.1 --batch 100 "
            "--simulate --true-p1 0.05 --true-p2 0.02\n"
        ),
    )
    parser.add_argument("--N1", type=int, required=True,
                        help="Mapped area of Class 1 (rare class), in map units (e.g. pixels).")
    parser.add_argument("--N2", type=int, required=True,
                        help="Mapped area of Class 2 (background class), in map units.")
    parser.add_argument("--delta", type=float, default=0.05,
                        help="Relative precision target δ  (default: 0.05).")
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="Significance level α for the credible interval (default: 0.05).")
    parser.add_argument("--batch", type=int, default=50,
                        help="Batch size b — number of units to label per iteration (default: 50).")

    # Simulation / testing mode
    sim_group = parser.add_argument_group("simulation / testing (non-interactive)")
    sim_group.add_argument("--simulate", action="store_true",
                           help="Run in simulation mode: a random oracle answers sampling queries.")
    sim_group.add_argument("--true-p1", type=float, default=0.1,
                           help="True misclassification rate for Class 1 (simulation only).")
    sim_group.add_argument("--true-p2", type=float, default=0.05,
                           help="True misclassification rate for Class 2 (simulation only).")
    sim_group.add_argument("--seed", type=int, default=42,
                           help="Random seed for simulation (default: 42).")

    return parser.parse_args()


def main():
    args = parse_args()

    # Validate inputs
    if args.N1 <= 0 or args.N2 <= 0:
        print("Error: N1 and N2 must be positive integers.", file=sys.stderr)
        sys.exit(1)
    if not (0 < args.delta < 1):
        print("Error: delta must be in (0, 1).", file=sys.stderr)
        sys.exit(1)
    if not (0 < args.alpha < 1):
        print("Error: alpha must be in (0, 1).", file=sys.stderr)
        sys.exit(1)
    if args.batch < 2:
        print("Error: batch size must be at least 2.", file=sys.stderr)
        sys.exit(1)

    simulate_fn = None
    if args.simulate:
        rng = np.random.default_rng(args.seed)
        p1_true = args.true_p1
        p2_true = args.true_p2

        def simulate_fn(n1_b, n2_b):
            x1 = int(rng.binomial(n1_b, p1_true))
            x2 = int(rng.binomial(n2_b, p2_true))
            return x1, x2

        print(f"\n[Simulation mode]  true p1={p1_true}, true p2={p2_true}, seed={args.seed}")
        true_area = (1 - p1_true) * args.N1 + p2_true * args.N2
        print(f"[Simulation mode]  True N_{{•1}} = {true_area:,.1f}")

    try:
        estimate_area(
            N1=args.N1,
            N2=args.N2,
            delta=args.delta,
            alpha=args.alpha,
            b=args.batch,
            simulate=simulate_fn,
        )
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(0)


if __name__ == "__main__":
    main()
