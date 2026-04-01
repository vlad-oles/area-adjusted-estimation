import numpy as np
import mpmath as mp
import scipy
from scipy.optimize import root_scalar
import pandas as pd
import os
import json
import time
from mpi4py import MPI


def sample(x, n, p, rng):
    x += rng.binomial(1, p)
    n += 1
    
    return x, n
    
def beta_cdf_cached(a, b):
    def cdf(u):
        if u <= 0:
            return 0.
        elif u >= 1:
            return 1.

        # regularized incomplete beta I_v(a,b)
        return scipy.special.betainc(a, b, u)
    
    return cdf

def beta_pdf_cached(a, b):
    # precompute normalization once
    norm = 1 / scipy.special.beta(a, b)
    am1, bm1 = a - 1, b - 1
    def pdf(v):
        return norm * (v**am1) * ((1-v)**bm1)

    return pdf

def Ndot1_cdf(u, as_, bs, Ns): # find Pr[Ndot1 < u] assuming p1 ~ Beta(a1, b1), p2 ~ Beta(a2, b2)
    if u <= 0:
        return mp.mpf('0')
    elif u >= Ns[1] + Ns[2]:
        return mp.mpf('1')

    pdf_p1 = beta_pdf_cached(as_[1], bs[1])
    cdf_p2 = beta_cdf_cached(as_[2], bs[2])

    def calc_p2_thresh(u, p1):
        return (u - (1 - p1)*Ns[1]) / Ns[2]
    
    def integrand(p1):
        p1 = float(p1)
        p2_thresh = calc_p2_thresh(u, p1)

        return pdf_p1(p1) * cdf_p2(p2_thresh)
    
    # integrate once over p1 in [0,1]
    return mp.quad(integrand, [0, 1])

def Ndot1_quantile(q, as_, bs, Ns):
    # q in (0,1)
    def func(u):
        return float(Ndot1_cdf(u, as_, bs, Ns) - q)

    res = root_scalar(func, bracket=[0., Ns[1] + Ns[2]], method="brentq")
    return res.root

def sample_and_update(ps, Ns, eps, alpha, i, xs, ns, as_, bs, p_means, iter_data, rng):
    # Sample a new point and update the sampling ratio.
    xs[i], ns[i] = sample(xs[i], ns[i], ps[i], rng)
    try:
        sampling_ratio = ns[1]/ns[2]
    except ZeroDivisionError:
        sampling_ratio = float('nan')

    # Update the posterior (from the inductive formula starting with a = b = 1).
    as_[i] = xs[i] + 1
    bs[i] = ns[i] - xs[i] + 1
    
    # Update the estimates and relative error bound violation probability.
    p_means[i] = as_[i] / (as_[i] + bs[i]) # mean of Beta(a, b)
    try:
        Ndot1_hat = Ns[1]*(1 - xs[1]/ns[1]) + Ns[2]*xs[2]/ns[2] # MLE estimate
    except ZeroDivisionError:
        Ndot1_hat = violation_prob = float('nan')
    else:
        # Find equal-tailed (1-α)-credible interval for Ndot1.
        Ndot1_lower = Ndot1_quantile(alpha/2, as_, bs, Ns)
        Ndot1_upper = Ndot1_quantile(1 - alpha/2, as_, bs, Ns)
        # Find if it's contained in the target interval.
        is_violation = Ndot1_lower < Ndot1_hat / (1 + eps) or Ndot1_upper > Ndot1_hat / (1 - eps)
        violation_prob = int(is_violation)

    # Record sampling ratio, p1, p2, estimated area.
    iter_data.append([sampling_ratio, p_means[1], p_means[2], Ndot1_hat])
    
    return Ndot1_hat, violation_prob

def estimate_area(eps, alpha, ps, Ns, rng):
    # Initialize.
    xs = {1: 0, 2: 0}
    ns = {1: 0, 2: 0}
    as_ = {1: .5, 2: .5} # Jeffry priors Beta(1, 1)
    bs = {1: .5, 2: .5} # Uniform priors Beta(1, 1)
    p_means = {1: float('nan'), 2: float('nan')}
    p_means = {1: .5, 2: .5}
    iter_data = []

    # Sample 1 point from each class.
    for i in [1, 2]:
        Ndot1_hat, violation_prob = sample_and_update(ps, Ns, eps, alpha, i, xs, ns, as_, bs, p_means, iter_data, rng)

    # Sample until the probability of relative error bound violation is ≤ alpha.
    while violation_prob > alpha or np.isnan(violation_prob):
        # Use posterior distribution means for the target sampling ratio.
        target_ratio = Ns[1] / Ns[2] * (p_means[1] * (1 - p_means[1]) / p_means[2] / (1 - p_means[2]))**.5
        Ndot1_tilde = Ns[1]*(1 - p_means[1]) + Ns[2]*p_means[2] # means-based estimate
        target_ratio = Ndot1_tilde / (sum(Ns.values()) - Ndot1_tilde) * (p_means[1] * (1 - p_means[1]) / p_means[2] / (1 - p_means[2]))**.5

        # Choose the class to sample from.
        i = 1 if ns[1] / ns[2] < target_ratio else 2

        # Sample from Class i and update the estimates.
        Ndot1_hat, violation_prob = sample_and_update(ps, Ns, eps, alpha, i, xs, ns, as_, bs, p_means, iter_data, rng)

    return Ndot1_hat, iter_data

# =============================================== DO WORK
RATIO_COL = 'sampling_ratio'
P1_HAT_COL = 'p1_hat'
P2_HAT_COL = 'p2_hat'
AREA_COL = 'N_dot1'

root_dir = ''
params_path = os.path.join(root_dir, 'map_params.json')
params = json.load(open(params_path))

# Read map parameters.
eps = params['eps']
alpha = params['alpha']
ps = {1: params['p1'], 2: params['p2']} # misclassification rates
Ns = {1: params['N1dot']*params['px_per_km2'], 2: params['N2dot']*params['px_per_km2']} # mapped areas (pixels)

# Ensure data dir exists.
sim_dir = os.path.join(root_dir, f'eps={eps},alpha={alpha},p1={ps[1]:.4f},p2={ps[2]:.4f},N1={Ns[1]:.0f},N2={Ns[2]:.0f}')
os.makedirs(sim_dir, exist_ok=True)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
rng = np.random.default_rng(seed=rank)

if rank == 0:
    print(f'{time.ctime()} {size=}')

n_simulations = 10000
for replication in range(n_simulations):
    if replication % size == rank:
        print(f'{time.ctime()} {rank=} starting {replication=}')
        _, iter_data = estimate_area(eps, alpha, ps, Ns, rng)

        # Store experiment data.
        sim_path = os.path.join(sim_dir, f'replication_{replication}.csv')
        df = pd.DataFrame(
            data=iter_data,
            columns=[RATIO_COL, P1_HAT_COL, P2_HAT_COL, AREA_COL])
        df.to_csv(sim_path, index=False)

        print(f'{time.ctime()} {rank=} finished {replication=}')

