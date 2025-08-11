#!/usr/bin/env python3
"""
Analyze a single bond from bondlengths.dump:
- Verify bond types (no unfolding) by counting type==2.
- Overlay BOTH:
    (1) Exact normalized single-bond PDF: p(r)=r^2 exp[-alpha(r-R0)^2]/Z, alpha=k_phys/(2T)
    (2) Gaussian surrogate used in your gnuplot:
        k_r = 2*K_r_lmp, R0=1, mu = ((3*R0/k_r)+R0**3)/((1/k_r)+R0**2), sigma = sqrt(1/k_r)

Saves: histogram_overlays.png
"""

import os, re
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate, optimize

# -------------------- user knobs --------------------
T_LJ = 1.0
R0   = 1.0

# Choose which bond to analyze from the 2xT dump (0 or 1)
BOND_IDX = 1

# Requested LAMMPS knobs; we’ll snap to nearest present in CORR/
REQ_K_LMP      = 22.0
REQ_KTHETA_LMP = 1.0

# Histogram look (to compare with gnuplot)
BIN_WIDTH = 0.005
XRANGE    = (0.0, 2.0)

# Run location: run this from scripts/, go up once to repo root
ROOT_STEP_UP = 1

OUT_FIG  = "histogram_overlays.png"

# -------------------- helpers --------------------
def project_corr_path():
    path = os.getcwd()
    for _ in range(ROOT_STEP_UP):
        path = os.path.dirname(path)
    return os.path.join(path, "Fibrin-Monomer", "output", "CORR")

def nearest(arr, val):
    arr = np.asarray(arr, float)
    return arr[np.argmin(np.abs(arr - val))]

# Exact model pieces
def Z_alpha(alpha, r0=R0):
    return integrate.quad(lambda x: x**2*np.exp(-alpha*(x-r0)**2), 0, np.inf, limit=500)[0]

def pdf_single_exact(r, k_phys, T=T_LJ, r0=R0):
    a = k_phys/(2*T)
    Z = Z_alpha(a, r0)
    return (r**2)*np.exp(-a*(r-r0)**2)/Z

def fit_alpha_mle(data, r0=R0, T=T_LJ):
    d = np.asarray(data, float)
    d = d[np.isfinite(d) & (d > 0)]
    if d.size < 50:
        raise RuntimeError("Not enough valid samples to fit alpha (need ~50+).")
    def nll(a):
        if a <= 0: return np.inf
        Z = Z_alpha(a, r0)
        return -(2*np.log(d) - a*(d-r0)**2 - np.log(Z)).sum()
    res = optimize.minimize_scalar(nll, bounds=(1e-6, 1e5), method='bounded')
    return float(res.x)

# Gaussian surrogate (gnuplot)
def gaussian_params_from_Kr_lmp(K_r_lmp, R0=R0):
    k_r = 2.0 * K_r_lmp
    mu  = ((3.0*R0/k_r) + R0**3) / ((1.0/k_r) + R0**2)
    sigma = np.sqrt(1.0/k_r)
    return mu, sigma, k_r

def gaussian_pdf(x, mu, sigma):
    return (1.0/np.sqrt(2.0*np.pi)/sigma) * np.exp(-0.5*((x-mu)/sigma)**2)

# -------------------- I/O --------------------
def scan_index(corr_path):
    recs = []
    for folder in sorted(os.listdir(corr_path)):
        m_seed = re.search(r'Seed(\d+)', folder)
        m_Kr   = re.search(r'k_r(\d+(?:\.\d+)?)', folder)
        m_Kt   = re.search(r'k_theta(\d+(?:\.\d+)?)', folder)
        if not (m_seed and m_Kr and m_Kt):
            continue
        recs.append({
            "folder": folder,
            "seed": int(m_seed.group(1)),
            "K_r_lmp": float(m_Kr.group(1)),
            "K_t_lmp": float(m_Kt.group(1)),
        })
    if not recs:
        raise RuntimeError("No matching folders under CORR/")
    return recs

def pick_folder(corr_path, req_kr_lmp, req_kt_lmp):
    recs = scan_index(corr_path)
    kr_vals = sorted(set(r["K_r_lmp"] for r in recs))
    kt_vals = sorted(set(r["K_t_lmp"] for r in recs))
    kr_sel  = nearest(kr_vals, req_kr_lmp)
    kt_sel  = nearest(kt_vals, req_kt_lmp)
    cand = [r for r in recs if np.isclose(r["K_r_lmp"], kr_sel) and np.isclose(r["K_t_lmp"], kt_sel)]
    if not cand:
        raise RuntimeError("No folder matches selection.")
    sel = cand[0]
    return os.path.join(corr_path, sel["folder"]), sel

def read_dump_types_and_lengths(path_dump):
    """
    Returns: types (2,T) int, lengths (2,T) float
    Assumes each 'ITEM: ENTRIES c_btype c_bondlen' block has exactly two lines (two bonds).
    """
    types = [[], []]
    blen  = [[], []]
    with open(path_dump) as f:
        in_block = False
        got = 0
        for line in f:
            if 'ITEM: ENTRIES c_btype c_bondlen' in line:
                in_block = True; got = 0
                continue
            if not in_block:
                continue
            if got == 2:
                in_block = False; got = 0
                continue
            toks = line.split()
            if len(toks) < 2:
                continue
            try:
                t = int(toks[0]); d = float(toks[1])
            except ValueError:
                continue
            types[got].append(t)
            blen[got].append(d)
            got += 1
    t_arr = np.array(types, dtype=int)
    r_arr = np.array(blen,  dtype=float)
    if r_arr.size == 0:
        raise RuntimeError(f"No bond data parsed from {path_dump}")
    return t_arr, r_arr

# -------------------- main --------------------
def main():
    corr = project_corr_path()
    print("Reading from:", corr)

    folder_path, meta = pick_folder(corr, REQ_K_LMP, REQ_KTHETA_LMP)
    print(f"[folder] {meta['folder']}")
    print(f"[LAMMPS] K_r_lmp={meta['K_r_lmp']}, K_theta_lmp={meta['K_t_lmp']}")

    dump_path = os.path.join(folder_path, "bondlengths.dump")
    t_2xT, r_2xT = read_dump_types_and_lengths(dump_path)

    # verify no type-2 occurrences
    type_counts = {1: int((t_2xT==1).sum()), 2: int((t_2xT==2).sum())}
    print(f"[types] counts: type1={type_counts[1]}, type2={type_counts[2]}")
    if type_counts[2] > 0:
        print("WARNING: Found type-2 bonds in dump → reactions happened at least once.")

    # pick one bond series to analyze
    if BOND_IDX not in (0,1):
        raise ValueError("BOND_IDX must be 0 or 1")
    series = r_2xT[BOND_IDX]
    types_series = t_2xT[BOND_IDX]
    print(f"[bond] using bond index {BOND_IDX}: N={series.size}, type2_in_series={int((types_series==2).sum())}")

    # empirical stats
    r_mean = float(np.mean(series))
    r_std  = float(np.std(series))
    print(f"[empirical] mean={r_mean:.6f}, std={r_std:.6f}")

    # exact-model fit and overlay
    k_phys_theory = 2.0 * meta["K_r_lmp"]
    alpha_theory  = k_phys_theory/(2.0*T_LJ)
    alpha_hat     = fit_alpha_mle(series)
    k_hat         = 2.0*T_LJ*alpha_hat
    print(f"[exact] k_phys(theory)={k_phys_theory:.6f}, alpha_theory={alpha_theory:.6f}")
    print(f"[fit]   alpha_hat={alpha_hat:.6f}, k_hat={k_hat:.6f}, ratio a_hat/a_theory={alpha_hat/alpha_theory:.3f}")

    # gaussian surrogate params
    mu_g, sigma_g, k_r_sur = gaussian_params_from_Kr_lmp(meta["K_r_lmp"], R0)
    print(f"[gaussian] k_r=2*K_lmp={k_r_sur:.6f}, mu={mu_g:.6f}, sigma={sigma_g:.6f}")

    # plot histogram + overlays
    lo, hi = XRANGE
    bins = int(round((hi - lo)/BIN_WIDTH))
    x = np.linspace(lo, hi, 1200)

    plt.figure(figsize=(7.6, 5.2))
    plt.xlim(lo, hi)
    plt.hist(series, bins=bins, range=(lo, hi), density=True, alpha=0.35, color='gray',
             label=f"data (bond {BOND_IDX})")
    # exact (solid)
    plt.plot(x, pdf_single_exact(x, k_phys_theory), lw=2.4, color='black',
             label=f"exact theory k={k_phys_theory:g}")
    # gaussian surrogate (dash-dot)
    plt.plot(x, gaussian_pdf(x, mu_g, sigma_g), lw=2.0, ls='-.',
             label=f"Gaussian surrogate (mu={mu_g:.3f}, sigma={sigma_g:.3f})")

    # vertical lines
    plt.axvline(r_mean,  color='C1', lw=1.2, alpha=0.9, label='empirical mean')
    plt.axvline(mu_g,    color='C2', lw=1.0, alpha=0.9, label='Gaussian μ')

    plt.xlabel("bond length r")
    plt.ylabel("density")
    plt.title(f"{meta['folder']} | bond {BOND_IDX}")
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(OUT_FIG, dpi=220)
    print(f"[saved] {OUT_FIG}")
    plt.show()

if __name__ == "__main__":
    main()
