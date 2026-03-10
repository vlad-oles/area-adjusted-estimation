#!/usr/bin/env python3
"""
Batch-based Adaptive Stratified Sampling for Rare Class Area Estimation
========================================================================
Implements Algorithm 1 from:
  "Adaptive area estimation via batch-based stratified sampling"

Usage:
    python estimate_area.py --N1 5000 --N2 95000 --delta 0.1 --alpha 0.05 --batch 100

The algorithm iteratively samples units from two classes (Class 1 = rare/target,
Class 2 = background), collects misclassification counts from the user, and
refines the area estimate until a target precision is achieved.
"""

import argparse
import math
import sys
import scipy
import numpy as np


def beta_cdf_cached(a, b):
    def cdf(u):
        if u <= 0:
            return 0.
        elif u >= 1:
            return 1.

        # Regularized incomplete beta I_v(a,b).
        return scipy.special.betainc(a, b, u)
    
    return cdf

def beta_pdf_cached(a, b):
    # Precompute normalization once.
    norm = 1 / scipy.special.beta(a, b)
    am1, bm1 = a - 1, b - 1
    def pdf(v):
        return norm * (v**am1) * ((1-v)**bm1)

    return pdf

    
def Ndot1_cdf(u, a1, b1, a2, b2, N1, N2):
    """
    Find Pr[Ndot1 < u] assuming p1 ~ Beta(a1, b1), p2 ~ Beta(a2, b2)
    """
    if u <= 0:
        return 0.
    elif u >= N1 + N2:
        return 1.

    pdf_p1 = beta_pdf_cached(a1, b1)
    cdf_p2 = beta_cdf_cached(a2, b2)

    def calc_p2_thresh(u, p1):
        return (u - (1 - p1)*N1) / N2
    
    def integrand(p1):
#        p1 = float(p1)
        p2_thresh = calc_p2_thresh(u, p1)

        return pdf_p1(p1) * cdf_p2(p2_thresh)
    
    # Integrate once over p1 in [0,1].
    cdf, _ = scipy.integrate.quad(integrand, 0, 1)
    
    return cdf

def Ndot1_quantile(q, a1, b1, a2, b2, N1, N2):
    # q in (0,1).
    def func(u):
        return Ndot1_cdf(u, a1, b1, a2, b2, N1, N2) - q

    res = scipy.optimize.root_scalar(func, bracket=[0., N1 + N2], method="brentq")
    return res.root

    
def credible_interval(x1, n1, x2, n2, N1, N2, alpha):
    """
    Return the equal-tail (1-alpha) credible interval for N_{•1}.
    """
    a1, b1 = x1 + 1, n1 - x1 + 1
    a2, b2 = x2 + 1, n2 - x2 + 1

    Ndot1_L = Ndot1_quantile(alpha/2, a1, b1, a2, b2, N1, N2)
    Ndot1_U = Ndot1_quantile(1 - alpha/2, a1, b1, a2, b2, N1, N2)

    return Ndot1_L, Ndot1_U

# ---------------------------------------------------------------------------
# Area estimate
# ---------------------------------------------------------------------------

def point_estimate(x1, n1, x2, n2, N1, N2):
    """MAP area-adjusted estimate: N̂_{•1} = (1 - x1/n1)*N1• + (x2/n2)*N2•"""
    p1_hat = x1 / n1 if n1 > 0 else 0.
    p2_hat = x2 / n2 if n2 > 0 else 0.
    return (1 - p1_hat) * N1 + p2_hat * N2


# ---------------------------------------------------------------------------
# Optimal allocation ratio (λ_opt)
# ---------------------------------------------------------------------------

def optimal_lambda(x1, n1, x2, n2, N1, N2):
    """
    Compute λ_opt using posterior means p̃1, p̃2.
    λ_opt = (Ñ_{•1} * sqrt(p̃1*(1-p̃1))) / ((N - Ñ_{•1}) * sqrt(p̃2*(1-p̃2)))
    """
    N = N1 + N2
    p1_tilde = (x1 + 1) / (n1 + 2)
    p2_tilde = (x2 + 1) / (n2 + 2)
    Ndot1_tilde = (1 - p1_tilde) * N1 + p2_tilde * N2

    num = Ndot1_tilde * (p1_tilde * (1 - p1_tilde))**.5
    den = (N - Ndot1_tilde) * (p2_tilde * (1 - p2_tilde))**.5

    return num / den if den > 0 else 1.

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
    if n1_batch > 0:
        print(f"    • {n1_batch:>6} units from CLASS 1  (rare / target class)")
    if n2_batch > 0:
        print(f"    • {n2_batch:>6} units from CLASS 2  (background class)")
    print()
    print("  After labelling, count misclassifications:")
    if n1_batch > 0:
        print("    x1 = number of Class-1 units that are actually Class 2")
    if n2_batch > 0:
        print("    x2 = number of Class-2 units that are actually Class 1")
    print("-" * 60)

    if n1_batch > 0:
        while True:
            try:
                x1 = int(input(f"  Enter x1 (misclassified from Class 1, 0–{n1_batch}): ").strip())
                if not (0 <= x1 <= n1_batch):
                    print(f"  ⚠  x1 must be between 0 and {n1_batch}. Try again.")
                    continue
                break
            except ValueError:
                print("  ⚠  Please enter an integer.")
    else:
        x1 = 0

    if n2_batch > 0:
        while True:
            try:
                x2 = int(input(f"  Enter x2 (misclassified from Class 2, 0–{n2_batch}): ").strip())
                if not (0 <= x2 <= n2_batch):
                    print(f"  ⚠  x2 must be between 0 and {n2_batch}. Try again.")
                    continue
                break
            except ValueError:
                print("  ⚠  Please enter an integer.")
    else:
        x2 = 0

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
              simulation mode
    """
    N = N1 + N2

    # Initialise accumulators
    n1_total, n2_total = 0, 0
    x1_total, x2_total = 0, 0
    t = 0

    # First batch allocated proportionally to mapped areas (line 5)
    n1_batch = round(N1 / N * b)
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

        # ── Point estimate (line 11) ─────────────────────────────────────
        Ndot1_hat = point_estimate(x1_total, n1_total, x2_total, n2_total, N1, N2)

        # ── Credible interval (line 12) ──────────────────────────────────
        Ndot1_L, Ndot1_U = credible_interval(x1_total, n1_total, x2_total, n2_total, N1, N2, alpha)

        # ── Report to user ───────────────────────────────────────────────
        print()
        print(f"  ┌─ Iteration {t} Results {'─'*35}")
        print(f"  │  Cumulative samples : n1={n1_total}  n2={n2_total}")
        print(f"  │  Misclassifications : x1={x1_total}  x2={x2_total}")
        print(f"  │  Estimate  N̂_{{•1}}  : {Ndot1_hat:,.1f}")
        print(f"  │  {int((1-alpha)*100)}% credible interval : [{Ndot1_L:,.1f},  {Ndot1_U:,.1f}]")

        # Precision bounds
        prec_lo = Ndot1_hat / (1 + delta)
        prec_hi = Ndot1_hat / (1 - delta)
        print(f"  │  Precision target  : [{prec_lo:,.1f},  {prec_hi:,.1f}]")

        achieved = Ndot1_L >= prec_lo and Ndot1_U <= prec_hi
        print(f"  │  Target achieved?  : {'✓ YES — stopping.' if achieved else '✗ No — continuing.'}")
        print(f"  └{'─'*50}")

        history.append({
            "iteration": t,
            "n1": n1_total, "n2": n2_total,
            "x1": x1_total, "x2": x2_total,
            "N̂_{{•1}}": Ndot1_hat, "N̂_{{•1}}^L": Ndot1_L, "N̂_{{•1}}^U": Ndot1_U,
            "achieved": achieved,
        })

        # ── Stopping criterion (line 18) ──────────────────────────────────
        if achieved:
            break

        # ── Optimal allocation for next batch (lines 13–17) ───────────────
        lam = optimal_lambda(x1_total, n1_total, x2_total, n2_total, N1, N2)
        n1_batch_raw = round(((n2_total + b) * lam - n1_total) / (1 + lam))
        n1_batch = max(min(n1_batch_raw, b), 0)
        n2_batch = b - n1_batch



    print()
    print("══════════════════════════════════════════════════════════")
    print(f"  FINAL ESTIMATE:  N̂_{{•1}} = {Ndot1_hat:,.1f}")
    print(f"  {int((1-alpha)*100)}% credible interval: [{Ndot1_L:,.1f},  {Ndot1_U:,.1f}]")
    print(f"  Total samples used: n1={n1_total}, n2={n2_total}  "
          f"(total={n1_total+n2_total})")
    print("══════════════════════════════════════════════════════════")

    return Ndot1_hat, Ndot1_L, Ndot1_U, history


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
            "  python estimate_area.py --N1 5000 --N2 95000 --delta 0.1 --alpha 0.05 --batch 100\n\n"
            "  # Simulation mode (random oracle, for testing):\n"
            "  python estimate_area.py --N1 5000 --N2 95000 --delta 0.1 --alpha 0.1 --batch 100 "
            "--simulate --true-p1 0.05 --true-p2 0.02\n"
        ),
    )
    parser.add_argument("--N1", type=int, required=True,
                        help="Mapped area of Class 1 (rare class), in map units (e.g. pixels).")
    parser.add_argument("--N2", type=int, required=True,
                        help="Mapped area of Class 2 (background class), in map units.")
    parser.add_argument("--delta", type=float, default=0.1,
                        help="Relative precision target δ  (default: 0.1).")
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="Significance level α for the credible interval (default: 0.05).")
    parser.add_argument("--batch", type=int, default=100,
                        help="Batch size b — number of units to label per iteration (default: 100).")

    # Simulation / testing mode
    sim_group = parser.add_argument_group("simulation / testing (non-interactive)")
    sim_group.add_argument("--simulate", action="store_true",
                           help="Run in simulation mode: a random oracle answers sampling queries.")
    sim_group.add_argument("--true-p1", type=float, default=0.1,
                           help="True misclassification rate for Class 1 (simulation only).")
    sim_group.add_argument("--true-p2", type=float, default=0.05,
                           help="True misclassification rate for Class 2 (simulation only).")
    sim_group.add_argument("--seed", type=int, default=666,
                           help="Random seed for simulation (default: 666).")

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
    if args.batch < 1:
        print("Error: batch size must be at least 1.", file=sys.stderr)
        sys.exit(1)

    simulate_fn = None
    if args.simulate:
        if not (0 <= args.true_p1 <= 1):
            print("Error: true-p1 must be in [0, 1].", file=sys.stderr)
            sys.exit(1)
        if not (0 <= args.true_p2 <= 1):
            print("Error: true-p2 must be in [0, 1].", file=sys.stderr)
            sys.exit(1)

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
