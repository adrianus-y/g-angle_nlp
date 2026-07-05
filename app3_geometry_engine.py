"""
APP 3 — Geometry Engine
=======================
Core directional geometry dan teori matematika utama.

Analisis yang dilakukan:
  • Δθ (delta-theta) computation     — A_g(u→v) − A_g(v→u)
  • p-norm geometry                  — functional_g dengan berbagai p
  • antisymmetry test                — apakah Δθ < 0 secara konsisten
  • normalization                    — L2-norm check per pair
  • signed directional score         — −Δθ sebagai ranking score
  • geometric diagnostics            — skewness, kurtosis, entropy, bimodality
  • degeneracy detection             — p=2 ⟹ Δθ ≡ 0 (Euclidean adalah simetris)
  • symmetry diagnostics             — P(|Δθ| < τ) untuk relasi simetris

Input:
  dataset       .csv   — word1,word2,relation,is_symmetric
  embedding     .pkl   — cached dict word→np.array (dari APP 2)
  config        .json  — {"p_values":[2,3,4,5,10], "n_permutations":500, ...}

Output:
  geometry_scores.csv  — per-pair: word1,word2,relation,p,delta_theta,signed_score,magnitude
  diagnostics.json     — per-relation × per-p: ringkasan statistik + degeneracy flag
  degeneracy_report.txt — relasi & p yang degenerate beserta alasannya

Requirements:
  pip install numpy scipy
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading, os, json, csv, pickle, time
import numpy as np
from scipy.stats import (ttest_1samp, mannwhitneyu,
                         skew as sp_skew, kurtosis as sp_kurtosis,
                         t as t_dist)

# =============================================================================
# COLOUR PALETTE
# =============================================================================
BG, BG2  = "#1a1a2e", "#16213e"
FG, FG2  = "#e0e0e0", "#90caf9"
FG3      = "#b0bec5"
ACCENT   = "#0d47a1"
ACC2     = "#1565c0"
GREEN    = "#2e7d32"
ORANGE   = "#e65100"
PURPLE   = "#4a148c"
RED      = "#c62828"

RANDOM_SEED = 42
DIR_EPS     = 1e-4    # pairs with |Δθ| < DIR_EPS are "uncertain"
SYM_TAU     = 0.05    # rad (~2.9°) — threshold for symmetric correctness
NEAR_ZERO   = 0.02    # rad — below this → "near-zero ambiguity" in error analysis

# =============================================================================
# GEOMETRY ENGINE  (Gunawan et al., 2018)
# =============================================================================

class GeometryEngine:
    """
    Core geometry computations for the g-angle metric.

    functional_g(x, y, p):
        G_p(x, y) = ‖x‖_p^{2-p} · Σ_i |x_i|^{p-1} sign(x_i) · y_i
                  = inner product in the dual space of ℓ^p.

    g_similarity(u, v, p):
        sim_g(u,v) = G_p(v, u) / (‖u‖_p · ‖v‖_p)   ∈ [−1, 1]
        (Note: v is in the "weight" role; u is the reference ball.)

    g_angle(u, v, p):
        A_g(u→v) = arccos(sim_g(u, v, p))            ∈ [0, π]

    delta_theta(u, v, p):
        Δθ(u,v) = A_g(u→v) − A_g(v→u)
        Δθ < 0  ⟺  u is more "contained in" v than v in u
                 ⟺  u is the hyponym/part/capital (expected direction)
        Δθ = 0  for all pairs when p=2 (Euclidean — symmetric geometry)
    """

    def __init__(self, eps: float = 1e-15):
        self.eps = eps

    def functional_g(self, x, y, p: float = 2.0) -> float:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        norm_x = np.linalg.norm(x, ord=p)
        if norm_x < self.eps:
            return 0.0
        abs_x  = np.abs(x)
        mask   = abs_x > 0
        term_x = np.zeros_like(x)
        term_x[mask] = (abs_x[mask] ** (p - 1)) * np.sign(x[mask])
        return float((norm_x ** (2 - p)) * np.sum(term_x * y))

    def g_similarity(self, u, v, p: float = 2.0) -> float:
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        nu = np.linalg.norm(u, ord=p)
        nv = np.linalg.norm(v, ord=p)
        if nu < self.eps or nv < self.eps:
            return 0.0
        g_vu = self.functional_g(v, u, p)
        return float(np.clip(g_vu / (nu * nv), -1.0, 1.0))

    def g_angle(self, u, v, p: float = 2.0) -> float:
        return float(np.arccos(np.clip(self.g_similarity(u, v, p), -1.0, 1.0)))

    def delta_theta(self, u, v, p: float = 2.0) -> float:
        return self.g_angle(u, v, p) - self.g_angle(v, u, p)

    def cosine_sim(self, u, v) -> float:
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        nu = np.linalg.norm(u)
        nv = np.linalg.norm(v)
        if nu < self.eps or nv < self.eps:
            return 0.0
        return float(np.dot(u, v) / (nu * nv))


ENGINE = GeometryEngine()

# =============================================================================
# BH FDR CORRECTION
# =============================================================================

def bh_correct(p_values: np.ndarray) -> np.ndarray:
    """Benjamini–Hochberg (1995) step-up FDR correction → q-values."""
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    if n == 0:
        return p.copy()
    order      = np.argsort(p)
    ranks      = np.empty(n, dtype=float)
    ranks[order] = np.arange(1, n + 1)
    q          = p * n / ranks
    q_sorted   = q[order]
    for i in range(n - 2, -1, -1):
        q_sorted[i] = min(q_sorted[i], q_sorted[i + 1])
    q[order] = np.clip(q_sorted, 0.0, 1.0)
    return q

# =============================================================================
# CORE: evaluate_pairs
# =============================================================================

def evaluate_pairs(pairs, get_vec, p: float, expected: str,
                   log, label: str = "",
                   n_permutations: int = 500,
                   progress_cb=None) -> dict:
    """
    Full geometry evaluation for one relation × one p value.

    Parameters
    ----------
    pairs       : list of (word1, word2, relation)
    get_vec     : callable word → np.ndarray | None
    p           : float  — p-norm parameter
    expected    : 'asymmetric' or 'symmetric'
    log         : callable str → None
    label       : relation name (for logging)
    n_permutations : int — permutation test iterations
    progress_cb : callable int(0-100) → None

    Returns
    -------
    results dict with keys: stats, deltas, cosines, pairs_ok,
                             error_pairs, is_degenerate, perm_*, ranking_acc, …
    """
    def _prog(v):
        if progress_cb:
            progress_cb(int(v))

    results = {
        "label"       : label,
        "expected"    : expected,
        "deltas"      : [],
        "cosines"     : [],
        "g_angles_uv" : [],
        "g_angles_vu" : [],
        "pairs_ok"    : [],
        "skipped"     : 0,
    }

    n   = len(pairs)
    vecs = []   # (eu, ev) tuples for permutation / ranking

    # ── 1. Per-pair Δθ computation ───────────────────────────────────────────
    for i, row in enumerate(pairs):
        w1, w2 = row[0], row[1]
        if i % 200 == 0:
            log(f"  {label}: {i}/{n} …")
            _prog(10 + int(i / max(n, 1) * 40))
        eu = get_vec(w1)
        ev = get_vec(w2)
        if eu is None or ev is None:
            results["skipped"] += 1
            continue
        dt   = ENGINE.delta_theta(eu, ev, p)
        cosv = ENGINE.cosine_sim(eu, ev)
        a_uv = ENGINE.g_angle(eu, ev, p)
        a_vu = ENGINE.g_angle(ev, eu, p)
        results["deltas"].append(dt)
        results["cosines"].append(cosv)
        results["g_angles_uv"].append(a_uv)
        results["g_angles_vu"].append(a_vu)
        results["pairs_ok"].append((w1, w2))
        vecs.append((eu, ev))

    _prog(50)

    for k in ("deltas", "cosines", "g_angles_uv", "g_angles_vu"):
        results[k] = np.array(results[k])

    d       = results["deltas"]
    c       = results["cosines"]
    n_eval  = len(d)
    results["n"] = n_eval

    if n_eval == 0:
        results["stats"]        = _empty_stats()
        results["is_degenerate"] = False
        return results

    # ── 2. Direction accuracy ────────────────────────────────────────────────
    n_uncertain = int(np.sum(np.abs(d) < DIR_EPS))
    if expected == "asymmetric":
        d_cert  = d[np.abs(d) >= DIR_EPS]
        dir_acc = float(np.mean(d_cert < 0)) if len(d_cert) > 0 else 0.0
    else:
        dir_acc = float(np.mean(np.abs(d) < SYM_TAU))

    obs_dir_acc = dir_acc
    obs_mean    = float(np.mean(d))
    cos_mean    = float(np.mean(c))
    cos_std     = float(np.std(c))

    # ── 3. Effect size: Cohen's d + 95% CI ──────────────────────────────────
    std_d    = float(np.std(d, ddof=1)) if n_eval > 1 else 1.0
    cohens_d = float(obs_mean / std_d)  if std_d > 1e-15 else 0.0
    se_d     = std_d / np.sqrt(n_eval)  if n_eval > 1 else float("inf")
    t_crit   = t_dist.ppf(0.975, df=max(n_eval - 1, 1))
    ci_lo    = float(obs_mean - t_crit * se_d)
    ci_hi    = float(obs_mean + t_crit * se_d)
    t_stat, t_p = ttest_1samp(d, 0) if n_eval > 1 else (0.0, 1.0)

    # ── 4. Distribution shape metrics ───────────────────────────────────────
    d_sk  = float(sp_skew(d))
    d_ku  = float(sp_kurtosis(d, fisher=True))
    bc    = (d_sk**2 + 1) / (d_ku + 3) if (d_ku + 3) != 0 else 0.0
    hist_c, _ = np.histogram(d, bins=max(10, n_eval // 20))
    probs = hist_c / hist_c.sum()
    probs = probs[probs > 0]
    d_ent = float(-np.sum(probs * np.log(probs)))

    # ── 5. Bootstrap CI (B=2000) for dir_acc & mean_dt ──────────────────────
    N_BOOT = 2000
    rng_b  = np.random.default_rng(RANDOM_SEED + 999)
    boot_dir, boot_mdt = [], []
    for _ in range(N_BOOT):
        idx_b = rng_b.integers(0, n_eval, size=n_eval)
        d_b   = d[idx_b]
        boot_mdt.append(float(np.mean(d_b)))
        if expected == "asymmetric":
            d_bc = d_b[np.abs(d_b) >= DIR_EPS]
            boot_dir.append(float(np.mean(d_bc < 0)) if len(d_bc) > 0 else 0.0)
        else:
            boot_dir.append(float(np.mean(np.abs(d_b) < SYM_TAU)))
    boot_dir = np.array(boot_dir)
    boot_mdt = np.array(boot_mdt)
    dir_ci_lo  = float(np.percentile(boot_dir, 2.5))
    dir_ci_hi  = float(np.percentile(boot_dir, 97.5))
    mdt_ci_lo  = float(np.percentile(boot_mdt, 2.5))
    mdt_ci_hi  = float(np.percentile(boot_mdt, 97.5))
    log(f"    Bootstrap dir_acc 95%CI = [{dir_ci_lo:.1%}, {dir_ci_hi:.1%}]")
    log(f"    Bootstrap mean_Δθ 95%CI = [{mdt_ci_lo:+.4f}, {mdt_ci_hi:+.4f}]")

    # ── 6. Degeneracy detection ──────────────────────────────────────────────
    is_degen = bool(np.allclose(d, 0.0, atol=1e-9))
    results["is_degenerate"] = is_degen

    eu_list = [eu for eu, _ in vecs]
    ev_list = [ev for _, ev in vecs]

    if is_degen:
        log(f"  [{label}] p={p} DEGENERATE — Δθ ≡ 0 (Euclidean geometry is symmetric). "
            f"Permutation & ranking skipped.")
        _prog(95)
        perm_p_orient = perm_p_pair = perm_p = float("nan")
        ranking_acc   = float("nan")
        rank_ci_lo = rank_ci_hi = float("nan")
        perm_orient_accs = perm_pair_accs = np.array([])
        results.update({
            "perm_means"     : perm_orient_accs,
            "perm_pair_accs" : perm_pair_accs,
            "perm_p"         : perm_p,
            "perm_p_orient"  : perm_p_orient,
            "perm_p_pair"    : perm_p_pair,
            "perm_obs_stat"  : obs_dir_acc,
            "ranking_acc"    : ranking_acc,
            "rand_abs_dt"    : np.array([]),
        })

    else:
        # ── 7a. Permutation A — orientation flip (asymmetric only) ───────────
        if expected == "asymmetric":
            log(f"  {label}: Perm A (orientation) n={n_permutations} …")
            _prog(60)
            rng_a = np.random.default_rng(RANDOM_SEED)
            perm_orient_accs = []
            for _ in range(n_permutations):
                flip = rng_a.random(n_eval) < 0.5
                perm_dt = np.array([
                    ENGINE.delta_theta(ev_list[i], eu_list[i], p) if flip[i]
                    else ENGINE.delta_theta(eu_list[i], ev_list[i], p)
                    for i in range(n_eval)
                ])
                perm_orient_accs.append(float(np.mean(perm_dt < 0)))
            perm_orient_accs = np.array(perm_orient_accs)
            perm_p_orient    = float(np.mean(perm_orient_accs >= obs_dir_acc))
            log(f"    [Orient] obs={obs_dir_acc:.3f}  "
                f"null={np.mean(perm_orient_accs):.3f}±{np.std(perm_orient_accs):.3f}  "
                f"p={perm_p_orient:.4f}")
        else:
            log(f"  {label}: Perm A SKIPPED (symmetric) — |Δθ| invariant to u↔v flip.")
            perm_orient_accs = np.array([])
            perm_p_orient    = float("nan")
            _prog(60)

        # ── 7b. Permutation B — pair shuffle ─────────────────────────────────
        log(f"  {label}: Perm B (pair shuffle) n={n_permutations} …")
        _prog(72)
        rng_pb = np.random.default_rng(RANDOM_SEED + 100)

        if expected == "asymmetric":
            perm_pair_accs = []
            for _ in range(n_permutations):
                shuf = rng_pb.permutation(n_eval)
                perm_dt = np.array([
                    ENGINE.delta_theta(eu_list[i], ev_list[shuf[i]], p)
                    for i in range(n_eval)
                ])
                perm_pair_accs.append(float(np.mean(perm_dt < 0)))
            perm_pair_accs = np.array(perm_pair_accs)
            perm_p_pair    = float(np.mean(perm_pair_accs >= obs_dir_acc))
            log(f"    [Pair]   obs={obs_dir_acc:.3f}  "
                f"null={np.mean(perm_pair_accs):.3f}±{np.std(perm_pair_accs):.3f}  "
                f"p={perm_p_pair:.4f}")
        else:
            # Cross-pool: even-idx u paired with odd-idx v (genuinely unrelated)
            obs_sym = float(np.mean(np.abs(d)))
            cross_u = [eu_list[i] for i in range(0, n_eval, 2)]
            cross_v = [ev_list[i] for i in range(1, n_eval, 2)]
            n_cross = min(len(cross_u), len(cross_v))
            perm_sym_stats = []
            if n_cross < 10:
                log(f"    [Sym cross-pool] fallback to intra-pool (n_cross={n_cross})")
                for _ in range(n_permutations):
                    shuf = rng_pb.permutation(n_eval)
                    perm_dt = np.array([
                        ENGINE.delta_theta(eu_list[i], ev_list[shuf[i]], p)
                        for i in range(n_eval)
                    ])
                    perm_sym_stats.append(float(np.mean(np.abs(perm_dt))))
            else:
                for _ in range(n_permutations):
                    idx = rng_pb.integers(0, n_cross, size=n_cross)
                    perm_dt = np.array([
                        ENGINE.delta_theta(cross_u[i], cross_v[i], p) for i in idx
                    ])
                    perm_sym_stats.append(float(np.mean(np.abs(perm_dt))))
            perm_pair_accs = np.array(perm_sym_stats)
            perm_p_pair    = float(np.mean(perm_pair_accs <= obs_sym))
            results["perm_obs_stat_sym"] = obs_sym
            log(f"    [Sym]    obs_mean|Δθ|={obs_sym:.4f}  "
                f"null={np.mean(perm_pair_accs):.4f}±{np.std(perm_pair_accs):.4f}  "
                f"p={perm_p_pair:.4f}")

        perm_p = (max(perm_p_orient, perm_p_pair)
                  if expected == "asymmetric" else perm_p_pair)

        results.update({
            "perm_means"     : perm_orient_accs,
            "perm_pair_accs" : perm_pair_accs,
            "perm_p"         : perm_p,
            "perm_p_orient"  : perm_p_orient,
            "perm_p_pair"    : perm_p_pair,
            "perm_obs_stat"  : results.get("perm_obs_stat_sym", obs_dir_acc)
                               if expected == "symmetric" else obs_dir_acc,
        })

        # ── 8. Ranking accuracy (AUC / Mann-Whitney) ─────────────────────────
        log(f"  {label}: Ranking accuracy (AUC) …")
        _prog(80)
        rng2  = np.random.default_rng(RANDOM_SEED + 1)
        n_rand = min(n_eval, 500)
        idx_u  = rng2.choice(n_eval, size=n_rand, replace=True)
        idx_v  = rng2.choice(n_eval, size=n_rand, replace=True)
        for i in range(n_rand):
            while idx_v[i] == idx_u[i]:
                idx_v[i] = rng2.integers(0, n_eval)
        rand_dt = np.array([
            ENGINE.delta_theta(eu_list[idx_u[i]], ev_list[idx_v[i]], p)
            for i in range(n_rand)
        ])
        if expected == "asymmetric":
            real_auc = -d;          rand_auc = -rand_dt
        else:
            real_auc = -np.abs(d);  rand_auc = -np.abs(rand_dt)
        mw_stat, _ = mannwhitneyu(real_auc, rand_auc, alternative="greater")
        ranking_acc = float(mw_stat / (len(real_auc) * len(rand_auc)))
        results["ranking_acc"] = ranking_acc
        results["rand_abs_dt"] = np.abs(rand_dt)
        log(f"    Ranking AUC = {ranking_acc:.1%}  (real n={len(real_auc)}, rand n={n_rand})")

        # Bootstrap CI for ranking_acc
        rng_rb = np.random.default_rng(RANDOM_SEED + 777)
        boot_rank = []
        for _ in range(N_BOOT):
            ib  = rng_rb.integers(0, len(real_auc),  size=len(real_auc))
            irb = rng_rb.integers(0, n_rand, size=n_rand)
            mw_b, _ = mannwhitneyu(real_auc[ib], rand_auc[irb], alternative="greater")
            boot_rank.append(float(mw_b / (len(ib) * len(irb))))
        boot_rank = np.array(boot_rank)
        rank_ci_lo = float(np.percentile(boot_rank, 2.5))
        rank_ci_hi = float(np.percentile(boot_rank, 97.5))
        log(f"    Bootstrap rank_acc 95%CI = [{rank_ci_lo:.1%}, {rank_ci_hi:.1%}]")

    # ── 9. Error analysis ────────────────────────────────────────────────────
    _prog(90)
    abs_d = np.abs(d)
    mean_abs = float(np.mean(abs_d))
    std_abs  = float(np.std(abs_d, ddof=1)) if n_eval > 1 else 0.0
    outlier_tau = mean_abs + 2.0 * std_abs

    def _classify(dt_val):
        a = abs(dt_val)
        if a < NEAR_ZERO:
            return "near_zero_ambiguity"
        elif a > outlier_tau:
            return "extreme_outlier"
        return "false_direction"

    error_pairs = []
    if expected == "asymmetric":
        for i, ((w1, w2), dt) in enumerate(zip(results["pairs_ok"], d)):
            if dt >= 0:
                eu_i, ev_i = eu_list[i], ev_list[i]
                nr = (np.linalg.norm(eu_i, ord=p) /
                      max(np.linalg.norm(ev_i, ord=p), 1e-12))
                error_pairs.append({
                    "word1": w1, "word2": w2,
                    "delta_theta"    : float(dt),
                    "cosine"         : float(c[i]),
                    "norm_ratio"     : float(nr),
                    "sub_class"      : _classify(dt),
                })
        error_pairs.sort(key=lambda x: x["delta_theta"], reverse=True)
    else:
        med = float(np.median(abs_d))
        for i, ((w1, w2), dt) in enumerate(zip(results["pairs_ok"], d)):
            if abs(dt) > med:
                eu_i, ev_i = eu_list[i], ev_list[i]
                nr = (np.linalg.norm(eu_i, ord=p) /
                      max(np.linalg.norm(ev_i, ord=p), 1e-12))
                error_pairs.append({
                    "word1": w1, "word2": w2,
                    "delta_theta"    : float(dt),
                    "cosine"         : float(c[i]),
                    "norm_ratio"     : float(nr),
                    "sub_class"      : _classify(dt),
                })
        error_pairs.sort(key=lambda x: abs(x["delta_theta"]), reverse=True)

    results["error_pairs"] = error_pairs
    n_fd = sum(1 for e in error_pairs if e["sub_class"] == "false_direction")
    n_nz = sum(1 for e in error_pairs if e["sub_class"] == "near_zero_ambiguity")
    n_ol = sum(1 for e in error_pairs if e["sub_class"] == "extreme_outlier")
    error_breakdown = {
        "false_direction"     : n_fd,
        "near_zero_ambiguity" : n_nz,
        "extreme_outlier"     : n_ol,
        "near_zero_tau"       : NEAR_ZERO,
        "outlier_tau"         : outlier_tau,
    }
    results["error_breakdown"] = error_breakdown
    log(f"    Errors: {len(error_pairs)} total  "
        f"false_dir={n_fd}  near_zero={n_nz}  outlier={n_ol}")

    # ── 10. Verdict & stats dict ─────────────────────────────────────────────
    if expected == "asymmetric":
        verdict = "G-Angle ✔" if dir_acc > 0.5 else "Not significant"
    else:
        verdict = "Symmetric ✔" if abs(obs_mean) < 0.05 else "Unexpected asymmetry"

    degen_str = "N/A (degenerate)"
    results["stats"] = {
        # Core
        "n"                   : n_eval,
        "skipped"             : results["skipped"],
        "mean_dt"             : obs_mean,
        "std_dt"              : float(np.std(d)),
        "median_dt"           : float(np.median(d)),
        "mean_abs_dt"         : mean_abs,
        "pct_neg"             : float(np.mean(d < 0)),
        # Direction
        "dir_acc"             : dir_acc,
        "dir_acc_display"     : degen_str if is_degen else f"{dir_acc:.1%}",
        "n_uncertain"         : n_uncertain,
        # Ranking
        "ranking_acc"         : ranking_acc if not is_degen else float("nan"),
        "ranking_acc_display" : degen_str if is_degen else
                                f"{ranking_acc:.1%}",
        # Degeneracy
        "is_degenerate"       : is_degen,
        # Stats tests
        "t_stat"              : float(t_stat),
        "t_p"                 : float(t_p),
        "t_q"                 : float("nan"),   # filled by apply_bh_correction()
        # Permutation
        "perm_p"              : perm_p,
        "perm_p_orient"       : perm_p_orient,
        "perm_p_pair"         : perm_p_pair,
        "perm_q_orient"       : float("nan"),
        "perm_q_pair"         : float("nan"),
        "perm_mean_base"      : float(np.mean(perm_orient_accs))
                                if len(perm_orient_accs) > 0 else float("nan"),
        # Effect size
        "cohens_d"            : cohens_d,
        "ci_lo"               : ci_lo,
        "ci_hi"               : ci_hi,
        # Bootstrap CIs
        "dir_acc_ci_lo"       : dir_ci_lo,
        "dir_acc_ci_hi"       : dir_ci_hi,
        "rank_acc_ci_lo"      : rank_ci_lo if not is_degen else float("nan"),
        "rank_acc_ci_hi"      : rank_ci_hi if not is_degen else float("nan"),
        "mean_dt_ci_lo"       : mdt_ci_lo,
        "mean_dt_ci_hi"       : mdt_ci_hi,
        # Distribution shape
        "skewness"            : d_sk,
        "kurtosis"            : d_ku,
        "entropy"             : d_ent,
        "bimodality_coeff"    : bc,
        # Cosine
        "cos_mean"            : cos_mean,
        "cos_std"             : cos_std,
        # Error analysis
        "n_errors"            : len(error_pairs),
        "error_breakdown"     : error_breakdown,
        # Verdict
        "verdict"             : verdict,
        # Symmetry tau
        "sym_tau"             : SYM_TAU if expected == "symmetric" else float("nan"),
    }
    _prog(100)
    return results


def _empty_stats() -> dict:
    nan = float("nan")
    return {
        "n": 0, "skipped": 0, "mean_dt": 0, "std_dt": 0,
        "median_dt": 0, "mean_abs_dt": 0, "pct_neg": 0,
        "dir_acc": 0, "dir_acc_display": "N/A", "n_uncertain": 0,
        "ranking_acc": nan, "ranking_acc_display": "N/A",
        "is_degenerate": False,
        "t_stat": 0, "t_p": 1.0, "t_q": nan,
        "perm_p": 1.0, "perm_p_orient": 1.0, "perm_p_pair": 1.0,
        "perm_q_orient": nan, "perm_q_pair": nan, "perm_mean_base": 0.0,
        "cohens_d": 0.0, "ci_lo": 0.0, "ci_hi": 0.0,
        "dir_acc_ci_lo": 0.0, "dir_acc_ci_hi": 0.0,
        "rank_acc_ci_lo": 0.0, "rank_acc_ci_hi": 0.0,
        "mean_dt_ci_lo": 0.0, "mean_dt_ci_hi": 0.0,
        "skewness": 0.0, "kurtosis": 0.0, "entropy": 0.0, "bimodality_coeff": 0.0,
        "cos_mean": 0.0, "cos_std": 0.0,
        "n_errors": 0, "verdict": "No data",
        "error_breakdown": {"false_direction": 0, "near_zero_ambiguity": 0,
                            "extreme_outlier": 0, "near_zero_tau": NEAR_ZERO, "outlier_tau": 0},
        "sym_tau": nan,
    }


# =============================================================================
# BH FDR — apply post-hoc across all results
# =============================================================================

def apply_bh_correction(results_by_rel: dict, log=None):
    """Apply BH FDR to all t_p / perm_p_orient / perm_p_pair values in results_by_rel."""
    keys, labels, pvals = [], [], []
    for rk, res in results_by_rel.items():
        st = res.get("stats", {})
        if st.get("n", 0) == 0:
            continue
        for field in ("t_p", "perm_p_orient", "perm_p_pair"):
            v = st.get(field, float("nan"))
            if not np.isnan(v):
                keys.append(rk); labels.append(field); pvals.append(v)
    if not pvals:
        return
    qv = bh_correct(np.array(pvals))
    qmap = {"t_p": "t_q", "perm_p_orient": "perm_q_orient",
            "perm_p_pair": "perm_q_pair"}
    for rk, field, q in zip(keys, labels, qv):
        results_by_rel[rk]["stats"][qmap[field]] = float(q)
    if log:
        log(f"\n  [BH FDR]  m={len(pvals)} hypotheses")
        for rk, field, pv, q in zip(keys, labels, pvals, qv):
            log(f"    {rk:<12} {field:<16}  p={pv:.4f}  q={q:.4f}  "
                f"{'✔ sig' if q < 0.05 else '  ns '}")


# =============================================================================
# ANALYTICAL BEST-p SELECTION  (v16)
# =============================================================================

def select_best_p(sweep_results: dict, log=None):
    """
    Analytically select optimal p from a sweep dict {p_value → {rel → results}}.

    Score(p) = mean_asym_dir_acc + 0.3 * mean_asym_abs_dt_norm
    Stable region = all p with score ≥ peak − 0.02.
    Best p = smallest p in stable region (parsimony).
    """
    ASYM_KEYS   = ["hyponymy", "meronymy", "capital"]
    PLATEAU_TOL = 0.02
    p_vals = sorted(sweep_results.keys())

    dir_accs, abs_dts = {}, {}
    for pv in p_vals:
        da_v, adt_v, all_degen = [], [], True
        for key in ASYM_KEYS:
            r = sweep_results[pv].get(key)
            if not r or r.get("n", 0) == 0:
                continue
            st = r["stats"]
            if st.get("is_degenerate", False):
                continue
            all_degen = False
            da_v.append(st["dir_acc"]); adt_v.append(st["mean_abs_dt"])
        if all_degen or not da_v:
            dir_accs[pv] = abs_dts[pv] = float("nan")
        else:
            dir_accs[pv] = float(np.mean(da_v))
            abs_dts[pv]  = float(np.mean(adt_v))

    valid_adt = [v for v in abs_dts.values() if not np.isnan(v)]
    adt_max   = max(valid_adt) if valid_adt else 1.0
    adt_max   = adt_max if adt_max > 1e-9 else 1.0

    scores = {}
    for pv in p_vals:
        da, adt = dir_accs[pv], abs_dts[pv]
        scores[pv] = (float("nan") if np.isnan(da)
                      else da + 0.3 * (adt / adt_max if not np.isnan(adt) else 0.0))

    valid = {pv: s for pv, s in scores.items() if not np.isnan(s)}
    if not valid:
        non_d = [pv for pv in p_vals if not np.isnan(dir_accs.get(pv, float("nan")))]
        best_p = non_d[-1] if non_d else p_vals[-1]
        return best_p, scores, [], [f"No valid p — fallback to p={best_p}"]

    peak = max(valid.values())
    stable = sorted([pv for pv, s in valid.items() if s >= peak - PLATEAU_TOL])
    best_p = stable[0] if stable else max(valid, key=valid.get)

    lines = [
        "  [v16] ANALYTICAL p SELECTION",
        f"  {'─'*54}",
        f"  {'p':>6}  {'dir_acc':>9}  {'mean|Δθ|':>10}  {'score':>8}  status",
        f"  {'─'*54}",
    ]
    for pv in p_vals:
        da  = dir_accs.get(pv, float("nan"))
        adt = abs_dts.get(pv,  float("nan"))
        sc  = scores.get(pv,   float("nan"))
        if np.isnan(da):   status = "DEGENERATE (skip)"
        elif pv == best_p: status = "◀ BEST (parsimony)"
        elif pv in stable: status = "  stable plateau"
        else:              status = ""
        lines.append(
            f"  {pv:>6.1f}  "
            f"{(f'{da:.1%}' if not np.isnan(da) else 'N/A'):>9}  "
            f"{(f'{adt:.4f}' if not np.isnan(adt) else 'N/A'):>10}  "
            f"{(f'{sc:.4f}' if not np.isnan(sc) else 'N/A'):>8}  {status}"
        )
    lines += [
        f"  {'─'*54}",
        f"  Plateau (score ≥ {peak:.4f} − {PLATEAU_TOL}): p ∈ {stable}",
        f"  → Best p = {best_p}  "
        f"(dir_acc={dir_accs.get(best_p, float('nan')):.1%}  "
        f"score={valid.get(best_p, float('nan')):.4f})",
    ]
    if log:
        for ln in lines:
            log(ln)
    return best_p, scores, stable, lines


# =============================================================================
# I/O HELPERS
# =============================================================================

def load_dataset_csv(path: str) -> list:
    """Load dataset CSV → list of (word1, word2, relation, is_symmetric)."""
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append((
                row["word1"].strip().lower(),
                row["word2"].strip().lower(),
                row["relation"].strip(),
                int(row.get("is_symmetric", 0)),
            ))
    return rows


def load_pkl(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_geometry_csv(all_rows: list, path: str):
    """Save per-pair geometry scores."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["word1", "word2", "relation", "p",
                    "delta_theta", "signed_score", "magnitude"])
        w.writerows(all_rows)


def save_diagnostics_json(diag: list, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(diag, f, indent=2, ensure_ascii=False,
                  default=lambda x: None if (isinstance(x, float) and np.isnan(x)) else x)


def save_degeneracy_report(degen_entries: list, path: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write("APP 3 — Geometry Engine  Degeneracy Report\n")
        f.write("=" * 60 + "\n\n")
        if not degen_entries:
            f.write("No degenerate (p, relation) combinations found.\n")
        else:
            for e in degen_entries:
                f.write(f"p={e['p']:>5}  relation={e['relation']:<16}  "
                        f"n_pairs={e['n_pairs']:>5}  "
                        f"reason={e['reason']}\n")


# =============================================================================
# FULL PIPELINE
# =============================================================================

def run_pipeline(cfg: dict, log) -> dict:
    """
    cfg keys:
      dataset_path   str
      embedding_path str
      p_values       list[float]   default [2,3,4,5,10]
      n_permutations int           default 500
      out_dir        str
    """
    out = cfg["out_dir"]
    os.makedirs(out, exist_ok=True)
    log("=" * 60)
    log("APP 3 — Geometry Engine")
    log("=" * 60)

    # ── Load inputs ──────────────────────────────────────────────────────────
    log("\n[1] Loading dataset …")
    rows = load_dataset_csv(cfg["dataset_path"])
    log(f"  {len(rows):,} pairs loaded")

    log("\n[2] Loading embedding cache …")
    vocab = load_pkl(cfg["embedding_path"])
    get_vec = lambda w: vocab.get(w.lower())
    log(f"  {len(vocab):,} words in embedding")

    p_values       = cfg.get("p_values", [2, 3, 4, 5, 10])
    n_permutations = cfg.get("n_permutations", 500)

    # ── Group pairs by relation ──────────────────────────────────────────────
    from collections import defaultdict
    by_rel = defaultdict(list)
    for w1, w2, rel, sym in rows:
        by_rel[rel].append((w1, w2, rel))
    expected_map = {rel: ("symmetric" if any(sym for *_, sym in
                          [(w1, w2, r, s) for w1, w2, r, s in rows if r == rel])
                          else "asymmetric")
                    for rel in by_rel}

    # ── Sweep over p values ──────────────────────────────────────────────────
    sweep      = {}   # p → {rel → results}
    all_rows   = []   # geometry_scores.csv rows
    diag       = []   # diagnostics.json entries
    degen      = []   # degeneracy_report.txt entries

    for pv in p_values:
        log(f"\n{'─'*60}")
        log(f"[p={pv}]")
        sweep[pv] = {}
        for rel, pairs in by_rel.items():
            exp = expected_map[rel]
            log(f"\n  Relation: {rel}  expected={exp}  n={len(pairs)}")
            res = evaluate_pairs(
                pairs, get_vec, pv, exp, log,
                label=rel, n_permutations=n_permutations,
            )
            sweep[pv][rel] = res

            # Per-pair rows for geometry CSV
            for i, (w1, w2) in enumerate(res.get("pairs_ok", [])):
                dt  = float(res["deltas"][i])
                all_rows.append((
                    w1, w2, rel, pv,
                    round(dt, 6),
                    round(-dt, 6),        # signed_score = −Δθ
                    round(abs(dt), 6),    # magnitude
                ))

            # Diagnostics entry
            st = res.get("stats", {})
            diag.append({
                "p"               : pv,
                "relation"        : rel,
                "expected"        : exp,
                "antisymmetric"   : bool(st.get("dir_acc", 0) > 0.5),
                "degenerate"      : bool(st.get("is_degenerate", False)),
                "n_pairs"         : st.get("n", 0),
                "mean_abs_delta"  : st.get("mean_abs_dt"),
                "dir_acc"         : st.get("dir_acc"),
                "ranking_acc"     : st.get("ranking_acc"),
                "t_p"             : st.get("t_p"),
                "perm_p"          : st.get("perm_p"),
                "cohens_d"        : st.get("cohens_d"),
                "skewness"        : st.get("skewness"),
                "bimodality_coeff": st.get("bimodality_coeff"),
                "verdict"         : st.get("verdict"),
            })

            if st.get("is_degenerate"):
                degen.append({
                    "p"       : pv,
                    "relation": rel,
                    "n_pairs" : st.get("n", 0),
                    "reason"  : "p=2 Euclidean geometry — Δθ ≡ 0 (symmetric by construction)",
                })

        # BH FDR for this p-sweep
        apply_bh_correction(sweep[pv], log=log)

    # ── Best-p selection ─────────────────────────────────────────────────────
    log("\n" + "=" * 60)
    log("[Best-p Selection]")
    best_p, scores, stable, _ = select_best_p(sweep, log=log)

    # ── Save outputs ─────────────────────────────────────────────────────────
    log("\n[Saving outputs]")
    geo_path  = os.path.join(out, "geometry_scores.csv")
    diag_path = os.path.join(out, "diagnostics.json")
    degen_path= os.path.join(out, "degeneracy_report.txt")

    save_geometry_csv(all_rows,  geo_path)
    save_diagnostics_json(diag,  diag_path)
    save_degeneracy_report(degen, degen_path)

    log(f"  geometry_scores.csv   → {geo_path}  ({len(all_rows):,} rows)")
    log(f"  diagnostics.json      → {diag_path}  ({len(diag)} entries)")
    log(f"  degeneracy_report.txt → {degen_path}  ({len(degen)} degenerate)")
    log(f"\n  Best p selected: p = {best_p}")
    log("\n" + "=" * 60)
    log("DONE")
    log("=" * 60)

    return {"sweep": sweep, "best_p": best_p, "scores": scores,
            "diag": diag, "degen": degen}


# =============================================================================
# GUI helpers
# =============================================================================

def _lbl(parent, text="", fg=FG3, font=("Consolas", 9), **kw):
    return tk.Label(parent, text=text, bg=BG, fg=fg, font=font, **kw)

def _entry(parent, var, width=40):
    return tk.Entry(parent, textvariable=var, width=width,
                    bg=BG2, fg=FG, insertbackground="#fff",
                    relief="flat", font=("Consolas", 9))

def _btn(parent, text, cmd, color=ACCENT, font=("Consolas", 9, "bold"), **kw):
    return tk.Button(parent, text=text, command=cmd,
                     bg=color, fg="#fff", relief="flat", font=font,
                     activebackground="#42a5f5", cursor="hand2", **kw)


# =============================================================================
# GUI — App
# =============================================================================

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("APP 3 — Geometry Engine")
        self.geometry("860x780")
        self.configure(bg=BG)
        self.resizable(True, True)
        self._last_result = None
        self._build_style()
        self._build_ui()

    def _build_style(self):
        s = ttk.Style(); s.theme_use("clam")
        s.configure("TFrame",      background=BG)
        s.configure("TNotebook",   background=BG, borderwidth=0)
        s.configure("TNotebook.Tab", background=BG2, foreground=FG3,
                    padding=[10, 4], font=("Consolas", 9))
        s.map("TNotebook.Tab", background=[("selected", ACCENT)],
              foreground=[("selected", "#fff")])
        s.configure("TLabelframe",       background=BG, foreground=FG2,
                    bordercolor="#0f3460")
        s.configure("TLabelframe.Label", background=BG, foreground=FG2,
                    font=("Consolas", 9, "bold"))
        s.configure("Treeview",         background=BG2, foreground=FG,
                    fieldbackground=BG2, rowheight=22, font=("Consolas", 9))
        s.configure("Treeview.Heading", background="#0f3460", foreground=FG2,
                    font=("Consolas", 8, "bold"))
        s.configure("TProgressbar",     troughcolor="#0f3460", background="#42a5f5")

    def _build_ui(self):
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=6, pady=6)
        tabs = [(" ⚙  Config & Run ", "_t_cfg"),
                (" 📊  Results Table ", "_t_res"),
                (" 🔬  Diagnostics ", "_t_diag"),
                (" 📝  Log ", "_t_log")]
        for label, attr in tabs:
            f = ttk.Frame(nb); setattr(self, attr, f); nb.add(f, text=label)
        self._build_cfg()
        self._build_res()
        self._build_diag()
        self._build_log()

    # =========================================================
    # TAB 1 — Config & Run
    # =========================================================
    def _build_cfg(self):
        P = dict(padx=10, pady=5)

        # Dataset
        lf1 = ttk.LabelFrame(self._t_cfg, text="1.  Dataset  (.csv)", padding=8)
        lf1.pack(fill="x", **P)
        r = tk.Frame(lf1, bg=BG); r.pack(fill="x")
        _lbl(r, "File:").pack(side="left")
        self._ds_path = tk.StringVar()
        _entry(r, self._ds_path, 54).pack(side="left", padx=6)
        _btn(r, "Browse…", self._browse_ds, ACC2).pack(side="left")
        self._ds_lbl = _lbl(lf1, "● Not loaded", fg="#78909c")
        self._ds_lbl.pack(anchor="w", pady=(2, 0))

        # Embedding
        lf2 = ttk.LabelFrame(self._t_cfg, text="2.  Embedding Cache  (.pkl)", padding=8)
        lf2.pack(fill="x", **P)
        r2 = tk.Frame(lf2, bg=BG); r2.pack(fill="x")
        _lbl(r2, "File:").pack(side="left")
        self._emb_path = tk.StringVar()
        _entry(r2, self._emb_path, 54).pack(side="left", padx=6)
        _btn(r2, "Browse…", self._browse_emb, ACC2).pack(side="left")
        self._emb_lbl = _lbl(lf2, "● Not loaded", fg="#78909c")
        self._emb_lbl.pack(anchor="w", pady=(2, 0))

        # Config
        lf3 = ttk.LabelFrame(self._t_cfg, text="3.  Parameters", padding=8)
        lf3.pack(fill="x", **P)

        r3a = tk.Frame(lf3, bg=BG); r3a.pack(fill="x", pady=2)
        _lbl(r3a, "p values (space-separated):").pack(side="left")
        self._p_vals = tk.StringVar(value="2 3 4 5 10")
        _entry(r3a, self._p_vals, 30).pack(side="left", padx=6)

        r3b = tk.Frame(lf3, bg=BG); r3b.pack(fill="x", pady=2)
        _lbl(r3b, "n_permutations:").pack(side="left")
        self._n_perm = tk.StringVar(value="500")
        _entry(r3b, self._n_perm, 8).pack(side="left", padx=6)
        _lbl(r3b, "  (≥200 recommended)", fg="#546e7a").pack(side="left")

        # Config JSON
        r3c = tk.Frame(lf3, bg=BG); r3c.pack(fill="x", pady=2)
        _lbl(r3c, "Config .json (optional):").pack(side="left")
        self._cfg_path = tk.StringVar()
        _entry(r3c, self._cfg_path, 38).pack(side="left", padx=6)
        _btn(r3c, "Browse…", self._browse_cfg, ACC2).pack(side="left")
        _btn(r3c, "Load", self._load_cfg, ACC2).pack(side="left", padx=4)

        # Output
        lf4 = ttk.LabelFrame(self._t_cfg, text="4.  Output", padding=8)
        lf4.pack(fill="x", **P)
        r4 = tk.Frame(lf4, bg=BG); r4.pack(fill="x")
        _lbl(r4, "Directory:").pack(side="left")
        self._out_dir = tk.StringVar(
            value=os.path.join(os.path.expanduser("~"), "geometry_output"))
        _entry(r4, self._out_dir, 46).pack(side="left", padx=6)
        _btn(r4, "Browse…", self._browse_out, ACC2).pack(side="left")

        # Run
        lf5 = ttk.LabelFrame(self._t_cfg, text="5.  Run", padding=8)
        lf5.pack(fill="x", **P)
        bf = tk.Frame(lf5, bg=BG); bf.pack(fill="x")
        self._run_btn = _btn(bf, "▶  Run Geometry Engine", self._start,
                             PURPLE, pady=5, padx=18)
        self._run_btn.pack(side="left", padx=4)
        self._best_p_lbl = _lbl(bf, "  best_p: —", fg=FG2,
                                 font=("Consolas", 10, "bold"))
        self._best_p_lbl.pack(side="left", padx=10)
        self._pbar = ttk.Progressbar(lf5, mode="indeterminate", length=360)
        self._pbar.pack(fill="x", pady=(6, 0))

    # =========================================================
    # TAB 2 — Results Table
    # =========================================================
    def _build_res(self):
        P = dict(padx=10, pady=5)

        ctrl = tk.Frame(self._t_res, bg=BG); ctrl.pack(fill="x", **P)
        _lbl(ctrl, "Filter p:").pack(side="left")
        self._filter_p = tk.StringVar(value="all")
        self._filter_p_om = tk.OptionMenu(ctrl, self._filter_p, "all")
        self._filter_p_om.config(bg=BG2, fg=FG, activebackground=ACCENT,
                                  highlightbackground=BG, relief="flat",
                                  font=("Consolas", 9), width=8)
        self._filter_p_om["menu"].config(bg=BG2, fg=FG)
        self._filter_p_om.pack(side="left", padx=6)
        _lbl(ctrl, "  Relation:").pack(side="left")
        self._filter_rel = tk.StringVar(value="all")
        self._filter_rel_om = tk.OptionMenu(ctrl, self._filter_rel, "all")
        self._filter_rel_om.config(bg=BG2, fg=FG, activebackground=ACCENT,
                                    highlightbackground=BG, relief="flat",
                                    font=("Consolas", 9), width=14)
        self._filter_rel_om["menu"].config(bg=BG2, fg=FG)
        self._filter_rel_om.pack(side="left", padx=6)
        _btn(ctrl, "🔄  Refresh", self._refresh_res, ACC2).pack(side="left", padx=6)

        lf = ttk.LabelFrame(self._t_res, text="Geometry Scores (per relation × per p)", padding=6)
        lf.pack(fill="both", expand=True, **P)
        cols = ("relation", "p", "n", "mean_Δθ", "mean|Δθ|",
                "dir_acc", "rank_acc", "perm_p", "cohen_d", "verdict")
        widths = [110, 40, 55, 80, 80, 75, 75, 70, 75, 140]
        self._res_tree = ttk.Treeview(lf, columns=cols, show="headings", height=18)
        for c, w in zip(cols, widths):
            self._res_tree.heading(c, text=c)
            self._res_tree.column(c, width=w, anchor="center")
        xsb = ttk.Scrollbar(lf, orient="horizontal", command=self._res_tree.xview)
        vsb = ttk.Scrollbar(lf, orient="vertical",   command=self._res_tree.yview)
        self._res_tree.configure(xscrollcommand=xsb.set, yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        self._res_tree.pack(fill="both", expand=True)
        xsb.pack(fill="x")

    # =========================================================
    # TAB 3 — Diagnostics
    # =========================================================
    def _build_diag(self):
        P = dict(padx=10, pady=5)

        lf1 = ttk.LabelFrame(self._t_diag, text="Degeneracy Report", padding=6)
        lf1.pack(fill="x", **P)
        cols = ("p", "relation", "n_pairs", "reason")
        self._degen_tree = ttk.Treeview(lf1, columns=cols, show="headings", height=4)
        for c, w in zip(cols, [50, 120, 70, 400]):
            self._degen_tree.heading(c, text=c)
            self._degen_tree.column(c, width=w, anchor="w")
        self._degen_tree.pack(fill="x")

        lf2 = ttk.LabelFrame(self._t_diag, text="Distribution Shape Diagnostics", padding=6)
        lf2.pack(fill="both", expand=True, **P)
        cols2 = ("relation", "p", "skewness", "kurtosis",
                 "entropy", "bimodality", "degenerate")
        widths2 = [120, 50, 90, 90, 90, 90, 90]
        self._diag_tree = ttk.Treeview(lf2, columns=cols2, show="headings", height=14)
        for c, w in zip(cols2, widths2):
            self._diag_tree.heading(c, text=c)
            self._diag_tree.column(c, width=w, anchor="center")
        vsb2 = ttk.Scrollbar(lf2, orient="vertical", command=self._diag_tree.yview)
        self._diag_tree.configure(yscrollcommand=vsb2.set)
        vsb2.pack(side="right", fill="y")
        self._diag_tree.pack(fill="both", expand=True)

    # =========================================================
    # TAB 4 — Log
    # =========================================================
    def _build_log(self):
        bf = tk.Frame(self._t_log, bg=BG); bf.pack(fill="x", padx=8, pady=4)
        _btn(bf, "Clear", self._clear_log, "#37474f").pack(side="left")
        self._log_box = scrolledtext.ScrolledText(
            self._t_log, state="disabled",
            bg="#0d1117", fg="#c9d1d9",
            font=("Consolas", 9), insertbackground="#fff")
        self._log_box.pack(fill="both", expand=True, padx=8, pady=4)

    # ── Log ──────────────────────────────────────────────────────────────────
    def _log(self, msg):
        self._log_box.config(state="normal")
        self._log_box.insert("end", msg + "\n")
        self._log_box.see("end")
        self._log_box.config(state="disabled")
        self.update_idletasks()

    def _clear_log(self):
        self._log_box.config(state="normal")
        self._log_box.delete("1.0", "end")
        self._log_box.config(state="disabled")

    # ── Browse ───────────────────────────────────────────────────────────────
    def _browse_ds(self):
        p = filedialog.askopenfilename(filetypes=[("CSV", "*.csv"), ("All", "*.*")])
        if p:
            self._ds_path.set(p)
            self._ds_lbl.config(text=f"● {os.path.basename(p)}", fg="#66bb6a")

    def _browse_emb(self):
        p = filedialog.askopenfilename(filetypes=[("Pickle", "*.pkl"), ("All", "*.*")])
        if p:
            self._emb_path.set(p)
            self._emb_lbl.config(text=f"● {os.path.basename(p)}", fg="#66bb6a")

    def _browse_cfg(self):
        p = filedialog.askopenfilename(filetypes=[("JSON", "*.json"), ("All", "*.*")])
        if p: self._cfg_path.set(p)

    def _browse_out(self):
        d = filedialog.askdirectory()
        if d: self._out_dir.set(d)

    def _load_cfg(self):
        p = self._cfg_path.get().strip()
        if not p or not os.path.isfile(p):
            messagebox.showerror("Error", "Config JSON not found."); return
        try:
            cfg = load_config(p)
            if "p_values" in cfg:
                self._p_vals.set(" ".join(str(v) for v in cfg["p_values"]))
            if "n_permutations" in cfg:
                self._n_perm.set(str(cfg["n_permutations"]))
            messagebox.showinfo("Loaded", f"Config loaded from {os.path.basename(p)}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    # ── Run ──────────────────────────────────────────────────────────────────
    def _start(self):
        ds  = self._ds_path.get().strip()
        emb = self._emb_path.get().strip()
        if not ds or not os.path.isfile(ds):
            messagebox.showerror("Error", "Dataset CSV not found."); return
        if not emb or not os.path.isfile(emb):
            messagebox.showerror("Error", "Embedding .pkl not found."); return
        try:
            p_vals = [float(x) for x in self._p_vals.get().split()]
        except ValueError:
            messagebox.showerror("Error", "p values must be numbers."); return

        cfg = {
            "dataset_path"   : ds,
            "embedding_path" : emb,
            "p_values"       : p_vals,
            "n_permutations" : int(self._n_perm.get() or 500),
            "out_dir"        : self._out_dir.get().strip(),
        }
        self._run_btn.config(state="disabled")
        self._pbar.start(12)

        def _run():
            try:
                result = run_pipeline(cfg, self._log)
                self._last_result = result
                self.after(0, lambda: (
                    self._best_p_lbl.config(
                        text=f"  best_p = {result['best_p']}"),
                    self._refresh_res(),
                    self._refresh_diag(),
                    messagebox.showinfo("Done",
                        f"Geometry Engine complete!\n"
                        f"Best p = {result['best_p']}\n"
                        f"Output: {cfg['out_dir']}"),
                ))
            except Exception as e:
                import traceback
                self._log(f"\n[ERROR] {e}\n{traceback.format_exc()}")
                self.after(0, lambda: messagebox.showerror("Error", str(e)))
            finally:
                self.after(0, lambda: (self._pbar.stop(),
                                       self._run_btn.config(state="normal")))

        threading.Thread(target=_run, daemon=True).start()

    # ── Refresh results table ─────────────────────────────────────────────────
    def _refresh_res(self):
        if not self._last_result:
            return
        sweep   = self._last_result["sweep"]
        fp      = self._filter_p.get()
        fr      = self._filter_rel.get()
        p_list  = sorted(sweep.keys())
        rel_list = sorted({rel for pv in sweep for rel in sweep[pv]})

        # Update dropdowns
        for var, om, opts in [
            (self._filter_p,   self._filter_p_om,   ["all"] + [str(pv) for pv in p_list]),
            (self._filter_rel, self._filter_rel_om, ["all"] + rel_list),
        ]:
            m = om["menu"]; m.delete(0, "end")
            for opt in opts:
                m.add_command(label=opt, command=lambda v=opt, sv=var: sv.set(v))

        self._res_tree.delete(*self._res_tree.get_children())

        def _fmt(v, fmt=".4f"):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return "—"
            return format(v, fmt)

        for pv in p_list:
            if fp != "all" and str(pv) != fp:
                continue
            for rel, res in sweep[pv].items():
                if fr != "all" and rel != fr:
                    continue
                st = res.get("stats", {})
                self._res_tree.insert("", "end", values=(
                    rel, pv,
                    st.get("n", 0),
                    _fmt(st.get("mean_dt")),
                    _fmt(st.get("mean_abs_dt")),
                    st.get("dir_acc_display", "—"),
                    st.get("ranking_acc_display", "—"),
                    _fmt(st.get("perm_p")),
                    _fmt(st.get("cohens_d")),
                    st.get("verdict", "—"),
                ))

    # ── Refresh diagnostics ───────────────────────────────────────────────────
    def _refresh_diag(self):
        if not self._last_result:
            return
        diag  = self._last_result["diag"]
        degen = self._last_result["degen"]

        self._degen_tree.delete(*self._degen_tree.get_children())
        for e in degen:
            self._degen_tree.insert("", "end", values=(
                e["p"], e["relation"], e["n_pairs"], e["reason"]))

        self._diag_tree.delete(*self._diag_tree.get_children())
        for e in diag:
            def _f(v):
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    return "—"
                return f"{v:.4f}"
            self._diag_tree.insert("", "end", values=(
                e["relation"], e["p"],
                _f(e.get("skewness")),
                _f(e.get("bimodality_coeff")),
                _f(e.get("mean_abs_delta")),
                _f(e.get("cohens_d")),
                "YES ⚠" if e.get("degenerate") else "no",
            ))


# =============================================================================
if __name__ == "__main__":
    App().mainloop()
