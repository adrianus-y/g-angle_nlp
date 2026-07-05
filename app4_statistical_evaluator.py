"""
APP 4 — Statistical Evaluator
==============================
Inference statistik terhadap hasil geometry.

Input  (dari APP 3 — Geometry Engine):
  geometry_scores.csv   — word1,word2,relation,p,delta_theta,signed_score,magnitude
  relation_metadata.json — {"hyponymy": {"expected": "asymmetric"}, ...}
  config.json            — {"p_values":[3], "n_permutations":500, "n_boot":2000, ...}

Output:
  evaluation_result.csv  — relation,p,dir_acc,rank_acc,t_p,q_value,...
  significance_report.json
  bootstrap_samples.csv
  permutation_summary.json
  error_analysis.csv

Analisis yang dilakukan:
  • directional accuracy   — P(Δθ < 0) untuk asymmetric; P(|Δθ| < τ) untuk symmetric
  • ranking accuracy       — AUC via Mann-Whitney U (real vs random mismatch pairs)
  • bootstrap CI           — percentile bootstrap B=2000 untuk dir_acc, rank_acc, mean_Δθ
  • permutation test A     — orientation flip null (asymmetric only)
  • permutation test B     — pair shuffle null (intra-pool/cross-pool)
  • one-sample t-test      — H₀: E[Δθ] = 0
  • BH FDR correction      — Benjamini–Hochberg across all hypotheses
  • symmetric evaluation   — P(|Δθ| < τ), τ=0.05 rad (~2.9°)
  • error classification   — false_direction / near_zero_ambiguity / extreme_outlier

Requirements:
  pip install numpy scipy
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading, os, json, csv, time
import numpy as np
from scipy.stats import (ttest_1samp, mannwhitneyu,
                         skew as sp_skew, kurtosis as sp_kurtosis,
                         t as t_dist)
from collections import defaultdict

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

# =============================================================================
# CONSTANTS
# =============================================================================
RANDOM_SEED = 42
DIR_EPS     = 1e-4    # |Δθ| < DIR_EPS → "uncertain" (excluded from dir_acc)
SYM_TAU     = 0.05    # rad — symmetric correctness threshold (~2.9°)
NEAR_ZERO   = 0.02    # rad — near-zero ambiguity classification
N_BOOT_DEF  = 2000    # default bootstrap iterations
N_PERM_DEF  = 500     # default permutation iterations

# =============================================================================
# I/O HELPERS
# =============================================================================

def load_geometry_csv(path: str) -> dict:
    """
    Load geometry_scores.csv (output of APP 3).
    Returns dict: (relation, p) → list of dicts with delta_theta, signed_score, magnitude.
    Also returns raw rows for mismatch-pair sampling.
    """
    rows_by_key = defaultdict(list)
    raw_rows    = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["relation"].strip(), float(row["p"]))
            entry = {
                "word1"       : row["word1"].strip().lower(),
                "word2"       : row["word2"].strip().lower(),
                "delta_theta" : float(row["delta_theta"]),
                "signed_score": float(row["signed_score"]),
                "magnitude"   : float(row["magnitude"]),
            }
            rows_by_key[key].append(entry)
            raw_rows.append(entry | {"relation": row["relation"], "p": float(row["p"])})
    return dict(rows_by_key), raw_rows


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _safe_json(obj):
    """JSON-serialisable: convert NaN → None."""
    if isinstance(obj, float) and np.isnan(obj):
        return None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def save_json(data, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=_safe_json)


def save_csv(rows: list, path: str, fieldnames: list):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


# =============================================================================
# BH FDR CORRECTION
# =============================================================================

def bh_correct(p_values: np.ndarray) -> np.ndarray:
    """Benjamini–Hochberg (1995) step-up FDR. Returns q-values in input order."""
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    if n == 0:
        return p.copy()
    order = np.argsort(p)
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(1, n + 1)
    q = p * n / ranks
    q_s = q[order]
    for i in range(n - 2, -1, -1):
        q_s[i] = min(q_s[i], q_s[i + 1])
    q[order] = np.clip(q_s, 0.0, 1.0)
    return q


# =============================================================================
# CORE STATISTICAL FUNCTIONS
# =============================================================================

def compute_direction_accuracy(deltas: np.ndarray, expected: str) -> dict:
    """
    Directional accuracy with epsilon margin.

    Asymmetric: dir_acc = P(Δθ < 0) among pairs with |Δθ| ≥ DIR_EPS.
    Symmetric:  dir_acc = P(|Δθ| < SYM_TAU)  [absolute threshold, not median split].

    Returns dict with dir_acc, n_uncertain, n_certain.
    """
    n_uncertain = int(np.sum(np.abs(deltas) < DIR_EPS))
    if expected == "asymmetric":
        d_cert  = deltas[np.abs(deltas) >= DIR_EPS]
        dir_acc = float(np.mean(d_cert < 0)) if len(d_cert) > 0 else 0.0
        n_certain = len(d_cert)
    else:
        dir_acc   = float(np.mean(np.abs(deltas) < SYM_TAU))
        n_certain = len(deltas)
    return {"dir_acc": dir_acc, "n_uncertain": n_uncertain, "n_certain": n_certain}


def compute_bootstrap_ci(deltas: np.ndarray, expected: str,
                         n_boot: int = N_BOOT_DEF, seed: int = RANDOM_SEED) -> dict:
    """
    Percentile bootstrap (B=n_boot) for:
      • dir_acc          95% CI
      • mean_delta_theta 95% CI

    Returns dict with all CI bounds + boot arrays.
    """
    rng    = np.random.default_rng(seed + 999)
    n_eval = len(deltas)
    boot_dir, boot_mdt = [], []

    for _ in range(n_boot):
        idx  = rng.integers(0, n_eval, size=n_eval)
        d_b  = deltas[idx]
        boot_mdt.append(float(np.mean(d_b)))
        if expected == "asymmetric":
            d_bc = d_b[np.abs(d_b) >= DIR_EPS]
            boot_dir.append(float(np.mean(d_bc < 0)) if len(d_bc) > 0 else 0.0)
        else:
            boot_dir.append(float(np.mean(np.abs(d_b) < SYM_TAU)))

    boot_dir = np.array(boot_dir)
    boot_mdt = np.array(boot_mdt)
    return {
        "dir_acc_ci_lo"  : float(np.percentile(boot_dir, 2.5)),
        "dir_acc_ci_hi"  : float(np.percentile(boot_dir, 97.5)),
        "mean_dt_ci_lo"  : float(np.percentile(boot_mdt, 2.5)),
        "mean_dt_ci_hi"  : float(np.percentile(boot_mdt, 97.5)),
        "boot_dir_accs"  : boot_dir,
        "boot_mean_dts"  : boot_mdt,
    }


def compute_ranking_accuracy(deltas: np.ndarray, expected: str,
                              n_rand: int = 500, n_boot: int = N_BOOT_DEF,
                              seed: int = RANDOM_SEED) -> dict:
    """
    AUC-style ranking accuracy via Mann-Whitney U.
    real_scores vs mismatch (randomly shuffled) scores.
    Asymmetric: score = −Δθ. Symmetric: score = −|Δθ|.

    Returns ranking_acc (AUC), CI bounds, rand_deltas.
    """
    n_eval = len(deltas)
    rng    = np.random.default_rng(seed + 1)
    n_rand = min(n_eval, n_rand)

    # Generate mismatch-pair delta_theta by shuffling indices
    idx_u = rng.choice(n_eval, size=n_rand, replace=True)
    idx_v = rng.choice(n_eval, size=n_rand, replace=True)
    for i in range(n_rand):
        while idx_v[i] == idx_u[i]:
            idx_v[i] = rng.integers(0, n_eval)

    # Use signed_scores directly from deltas (approximation for mismatch)
    rand_dt = deltas[idx_v]  # mismatch: pick a v from a different pair

    if expected == "asymmetric":
        real_s = -deltas;          rand_s = -rand_dt
    else:
        real_s = -np.abs(deltas);  rand_s = -np.abs(rand_dt)

    mw, _ = mannwhitneyu(real_s, rand_s, alternative="greater")
    rank_acc = float(mw / (len(real_s) * len(rand_s)))

    # Bootstrap CI for ranking_acc
    rng_rb = np.random.default_rng(seed + 777)
    boot_rank = []
    for _ in range(n_boot):
        ib  = rng_rb.integers(0, len(real_s), size=len(real_s))
        irb = rng_rb.integers(0, n_rand,      size=n_rand)
        mw_b, _ = mannwhitneyu(real_s[ib], rand_s[irb], alternative="greater")
        boot_rank.append(float(mw_b / (len(ib) * len(irb))))
    boot_rank = np.array(boot_rank)

    return {
        "ranking_acc"    : rank_acc,
        "rank_acc_ci_lo" : float(np.percentile(boot_rank, 2.5)),
        "rank_acc_ci_hi" : float(np.percentile(boot_rank, 97.5)),
        "rand_deltas"    : rand_dt,
    }


def compute_ttest(deltas: np.ndarray) -> dict:
    """One-sample t-test H₀: E[Δθ] = 0. Returns t_stat, t_p."""
    if len(deltas) > 1:
        t_stat, t_p = ttest_1samp(deltas, 0)
    else:
        t_stat, t_p = 0.0, 1.0
    return {"t_stat": float(t_stat), "t_p": float(t_p)}


def compute_cohens_d(deltas: np.ndarray) -> dict:
    """Cohen's d + 95% CI via t-distribution."""
    n     = len(deltas)
    mu    = float(np.mean(deltas))
    std_d = float(np.std(deltas, ddof=1)) if n > 1 else 1.0
    cd    = mu / std_d if std_d > 1e-15 else 0.0
    se    = std_d / np.sqrt(n) if n > 1 else float("inf")
    tc    = t_dist.ppf(0.975, df=max(n - 1, 1))
    return {
        "cohens_d" : float(cd),
        "ci_lo"    : float(mu - tc * se),
        "ci_hi"    : float(mu + tc * se),
    }


def compute_distribution_shape(deltas: np.ndarray) -> dict:
    """Skewness, excess kurtosis, differential entropy, bimodality coefficient."""
    sk  = float(sp_skew(deltas))
    ku  = float(sp_kurtosis(deltas, fisher=True))
    bc  = (sk**2 + 1) / (ku + 3) if (ku + 3) != 0 else 0.0
    hc, _ = np.histogram(deltas, bins=max(10, len(deltas) // 20))
    probs  = hc / hc.sum()
    probs  = probs[probs > 0]
    ent    = float(-np.sum(probs * np.log(probs)))
    return {"skewness": sk, "kurtosis": ku,
            "bimodality_coeff": float(bc), "entropy": ent}


def compute_permutation_tests(deltas: np.ndarray, expected: str,
                               n_perm: int = N_PERM_DEF,
                               seed: int = RANDOM_SEED) -> dict:
    """
    Permutation Test A — orientation flip (asymmetric only).
      Null: flip (u,v) ↔ (v,u) per pair randomly.
      Δθ(v,u) = −Δθ(u,v) is a mathematical identity → negate deltas.
      Statistic: P(Δθ < 0).

    Permutation Test B — pair shuffle.
      Asymmetric: intra-pool shuffle.
      Symmetric:  cross-pool (even-idx u × odd-idx v).
      Statistic: P(Δθ < 0) for asym; mean|Δθ| for sym.

    Returns perm_p_orient, perm_p_pair, perm_p (conservative), arrays.
    """
    n_eval   = len(deltas)
    obs_dir  = compute_direction_accuracy(deltas, expected)["dir_acc"]

    # ── Perm A ──────────────────────────────────────────────────────────────
    if expected == "asymmetric":
        rng_a = np.random.default_rng(seed)
        perm_a_stats = []
        for _ in range(n_perm):
            flip    = rng_a.random(n_eval) < 0.5
            perm_dt = np.where(flip, -deltas, deltas)
            d_cert  = perm_dt[np.abs(perm_dt) >= DIR_EPS]
            perm_a_stats.append(float(np.mean(d_cert < 0)) if len(d_cert) > 0 else 0.0)
        perm_a_stats  = np.array(perm_a_stats)
        perm_p_orient = float(np.mean(perm_a_stats >= obs_dir))
    else:
        perm_a_stats  = np.array([])
        perm_p_orient = float("nan")

    # ── Perm B ──────────────────────────────────────────────────────────────
    rng_b = np.random.default_rng(seed + 100)
    if expected == "asymmetric":
        perm_b_stats = []
        for _ in range(n_perm):
            shuf    = rng_b.permutation(n_eval)
            perm_dt = deltas[shuf]     # u_i paired with v_shuf[i]
            d_cert  = perm_dt[np.abs(perm_dt) >= DIR_EPS]
            perm_b_stats.append(float(np.mean(d_cert < 0)) if len(d_cert) > 0 else 0.0)
        perm_b_stats = np.array(perm_b_stats)
        perm_p_pair  = float(np.mean(perm_b_stats >= obs_dir))
    else:
        # Cross-pool: even-indexed u × odd-indexed v
        obs_sym   = float(np.mean(np.abs(deltas)))
        cross_u   = deltas[0:n_eval:2]
        cross_v   = deltas[1:n_eval:2]
        n_cross   = min(len(cross_u), len(cross_v))
        perm_b_stats = []
        if n_cross < 10:
            for _ in range(n_perm):
                shuf = rng_b.permutation(n_eval)
                perm_b_stats.append(float(np.mean(np.abs(deltas[shuf]))))
        else:
            for _ in range(n_perm):
                idx = rng_b.integers(0, n_cross, size=n_cross)
                perm_dt = np.abs(cross_u[idx] - cross_v[idx])
                perm_b_stats.append(float(np.mean(perm_dt)))
        perm_b_stats = np.array(perm_b_stats)
        perm_p_pair  = float(np.mean(perm_b_stats <= obs_sym))

    perm_p = (max(perm_p_orient, perm_p_pair)
              if expected == "asymmetric" else perm_p_pair)

    return {
        "perm_p_orient"   : perm_p_orient,
        "perm_p_pair"     : perm_p_pair,
        "perm_p"          : perm_p,
        "perm_a_stats"    : perm_a_stats,
        "perm_b_stats"    : perm_b_stats,
        "obs_stat"        : obs_dir,
    }


def classify_errors(entries: list, expected: str) -> list:
    """
    Classify errors per pair.

    Asymmetric error: Δθ ≥ 0  (should be negative).
    Symmetric error:  |Δθ| > median(|Δθ|).

    Error sub-classes:
      near_zero_ambiguity — |Δθ| < NEAR_ZERO
      extreme_outlier     — |Δθ| > mean + 2σ
      false_direction     — everything else
    """
    deltas   = np.array([e["delta_theta"] for e in entries])
    abs_d    = np.abs(deltas)
    mean_abs = float(np.mean(abs_d))
    std_abs  = float(np.std(abs_d, ddof=1)) if len(abs_d) > 1 else 0.0
    outlier_tau = mean_abs + 2.0 * std_abs

    def _cls(dt):
        a = abs(dt)
        if a < NEAR_ZERO:     return "near_zero_ambiguity"
        if a > outlier_tau:   return "extreme_outlier"
        return "false_direction"

    errors = []
    if expected == "asymmetric":
        for e in entries:
            if e["delta_theta"] >= 0:
                errors.append({
                    "word1"       : e["word1"],
                    "word2"       : e["word2"],
                    "relation"    : e.get("relation", ""),
                    "p"           : e.get("p", ""),
                    "delta_theta" : e["delta_theta"],
                    "magnitude"   : e["magnitude"],
                    "error_type"  : _cls(e["delta_theta"]),
                })
    else:
        med = float(np.median(abs_d))
        for e in entries:
            if abs(e["delta_theta"]) > med:
                errors.append({
                    "word1"       : e["word1"],
                    "word2"       : e["word2"],
                    "relation"    : e.get("relation", ""),
                    "p"           : e.get("p", ""),
                    "delta_theta" : e["delta_theta"],
                    "magnitude"   : e["magnitude"],
                    "error_type"  : _cls(e["delta_theta"]),
                })

    if expected == "asymmetric":
        errors.sort(key=lambda x: x["delta_theta"], reverse=True)
    else:
        errors.sort(key=lambda x: abs(x["delta_theta"]), reverse=True)
    return errors


# =============================================================================
# FULL PIPELINE
# =============================================================================

def run_pipeline(cfg: dict, log) -> dict:
    """
    cfg keys:
      geometry_csv      str   — path to geometry_scores.csv (from APP 3)
      metadata_json     str   — path to relation metadata .json (optional)
      p_values          list  — subset of p values to evaluate (None = all)
      n_permutations    int
      n_boot            int
      out_dir           str
    """
    out = cfg["out_dir"]
    os.makedirs(out, exist_ok=True)
    log("=" * 60)
    log("APP 4 — Statistical Evaluator")
    log("=" * 60)

    # ── 1. Load geometry scores ──────────────────────────────────────────────
    log(f"\n[1] Loading geometry scores: {cfg['geometry_csv']}")
    rows_by_key, raw_rows = load_geometry_csv(cfg["geometry_csv"])
    all_rels = sorted({k[0] for k in rows_by_key})
    all_ps   = sorted({k[1] for k in rows_by_key})
    log(f"  Relations: {all_rels}")
    log(f"  p values : {all_ps}")
    log(f"  Total pair×p combinations: {len(rows_by_key)}")

    # ── 2. Load metadata (expected direction per relation) ───────────────────
    meta = {}
    if cfg.get("metadata_json") and os.path.isfile(cfg["metadata_json"]):
        log(f"\n[2] Loading relation metadata: {cfg['metadata_json']}")
        raw_meta = load_json(cfg["metadata_json"])
        # Accept both list-of-dicts (from APP 3 diagnostics) or dict
        if isinstance(raw_meta, list):
            for entry in raw_meta:
                rel = entry.get("relation", "")
                if rel and rel not in meta:
                    meta[rel] = entry.get("expected", "asymmetric")
        else:
            meta = raw_meta
    # Default fallback
    for rel in all_rels:
        if rel not in meta:
            meta[rel] = ("symmetric" if rel in ("sibling", "coordinate")
                         else "asymmetric")
    log(f"  Relation types: {meta}")

    # ── 3. Filter p values ───────────────────────────────────────────────────
    p_filter = cfg.get("p_values") or all_ps
    log(f"\n[3] Evaluating p values: {p_filter}")

    n_perm = cfg.get("n_permutations", N_PERM_DEF)
    n_boot = cfg.get("n_boot",         N_BOOT_DEF)

    # ── 4. Per-relation × per-p evaluation ──────────────────────────────────
    eval_rows     = []   # evaluation_result.csv
    boot_rows     = []   # bootstrap_samples.csv
    perm_summary  = []   # permutation_summary.json
    error_rows    = []   # error_analysis.csv
    sig_entries   = []   # significance_report.json

    # Collect raw p-values for BH correction
    all_t_p_info     = []   # (rel, pv, t_p)
    all_perm_p_info  = []   # (rel, pv, perm_p_orient), (rel, pv, perm_p_pair)

    for pv in p_filter:
        log(f"\n{'─'*60}")
        log(f"[p = {pv}]")
        for rel in all_rels:
            key      = (rel, pv)
            entries  = rows_by_key.get(key, [])
            expected = meta.get(rel, "asymmetric")
            n_eval   = len(entries)
            log(f"\n  {rel}  expected={expected}  n={n_eval}")

            if n_eval == 0:
                log(f"  SKIP — no pairs")
                continue

            deltas = np.array([e["delta_theta"] for e in entries])

            # Degeneracy check (inherit from geometry layer)
            is_degen = bool(np.allclose(deltas, 0.0, atol=1e-9))
            if is_degen:
                log(f"  DEGENERATE — Δθ ≡ 0 (p=2 Euclidean). Stats skipped.")

            # 4a. Direction accuracy
            da_res = compute_direction_accuracy(deltas, expected)
            log(f"    dir_acc={da_res['dir_acc']:.1%}  uncertain={da_res['n_uncertain']}")

            # 4b. t-test
            tt_res = compute_ttest(deltas)
            log(f"    t_stat={tt_res['t_stat']:+.4f}  t_p={tt_res['t_p']:.4f}")

            # 4c. Cohen's d
            cd_res = compute_cohens_d(deltas)
            log(f"    cohens_d={cd_res['cohens_d']:+.4f}  "
                f"95%CI=[{cd_res['ci_lo']:+.4f}, {cd_res['ci_hi']:+.4f}]")

            # 4d. Distribution shape
            sh_res = compute_distribution_shape(deltas)
            log(f"    skew={sh_res['skewness']:+.4f}  "
                f"kurt={sh_res['kurtosis']:+.4f}  "
                f"bimod={sh_res['bimodality_coeff']:.4f}")

            # 4e. Bootstrap CI
            boot_res = compute_bootstrap_ci(deltas, expected, n_boot=n_boot)
            log(f"    dir_acc 95%CI=[{boot_res['dir_acc_ci_lo']:.1%}, "
                f"{boot_res['dir_acc_ci_hi']:.1%}]")
            log(f"    mean_Δθ 95%CI=[{boot_res['mean_dt_ci_lo']:+.4f}, "
                f"{boot_res['mean_dt_ci_hi']:+.4f}]")

            # 4f. Ranking accuracy
            if is_degen:
                rk_res = {"ranking_acc": float("nan"),
                          "rank_acc_ci_lo": float("nan"),
                          "rank_acc_ci_hi": float("nan"),
                          "rand_deltas": np.array([])}
            else:
                rk_res = compute_ranking_accuracy(
                    deltas, expected, n_rand=500, n_boot=n_boot)
                log(f"    rank_acc={rk_res['ranking_acc']:.1%}  "
                    f"95%CI=[{rk_res['rank_acc_ci_lo']:.1%}, "
                    f"{rk_res['rank_acc_ci_hi']:.1%}]")

            # 4g. Permutation tests
            if is_degen:
                pm_res = {
                    "perm_p_orient": float("nan"),
                    "perm_p_pair":   float("nan"),
                    "perm_p":        float("nan"),
                    "perm_a_stats":  np.array([]),
                    "perm_b_stats":  np.array([]),
                    "obs_stat":      da_res["dir_acc"],
                }
            else:
                log(f"    Running permutation tests n_perm={n_perm} …")
                pm_res = compute_permutation_tests(
                    deltas, expected, n_perm=n_perm)
                log(f"    perm_p_orient={pm_res['perm_p_orient']}  "
                    f"perm_p_pair={pm_res['perm_p_pair']}  "
                    f"perm_p={pm_res['perm_p']}")

            # 4h. Error classification
            for e in entries:
                e["relation"] = rel
                e["p"]        = pv
            errs = classify_errors(entries, expected)
            n_fd = sum(1 for e in errs if e["error_type"] == "false_direction")
            n_nz = sum(1 for e in errs if e["error_type"] == "near_zero_ambiguity")
            n_ol = sum(1 for e in errs if e["error_type"] == "extreme_outlier")
            log(f"    errors={len(errs)}  false_dir={n_fd}  near_zero={n_nz}  outlier={n_ol}")
            error_rows.extend(errs)

            # ── Collect for BH correction ────────────────────────────────────
            all_t_p_info.append((rel, pv, tt_res["t_p"]))
            if not np.isnan(pm_res["perm_p_orient"]):
                all_perm_p_info.append((rel, pv, "orient", pm_res["perm_p_orient"]))
            if not np.isnan(pm_res["perm_p_pair"]):
                all_perm_p_info.append((rel, pv, "pair",   pm_res["perm_p_pair"]))

            # ── Accumulate rows ──────────────────────────────────────────────
            eval_rows.append({
                "relation"       : rel,
                "p"              : pv,
                "expected"       : expected,
                "n"              : n_eval,
                "degenerate"     : is_degen,
                "mean_dt"        : float(np.mean(deltas)),
                "std_dt"         : float(np.std(deltas)),
                "mean_abs_dt"    : float(np.mean(np.abs(deltas))),
                "dir_acc"        : da_res["dir_acc"],
                "n_uncertain"    : da_res["n_uncertain"],
                "dir_acc_ci_lo"  : boot_res["dir_acc_ci_lo"],
                "dir_acc_ci_hi"  : boot_res["dir_acc_ci_hi"],
                "ranking_acc"    : rk_res["ranking_acc"],
                "rank_acc_ci_lo" : rk_res["rank_acc_ci_lo"],
                "rank_acc_ci_hi" : rk_res["rank_acc_ci_hi"],
                "t_stat"         : tt_res["t_stat"],
                "t_p"            : tt_res["t_p"],
                "t_q"            : float("nan"),   # filled after BH
                "perm_p_orient"  : pm_res["perm_p_orient"],
                "perm_p_pair"    : pm_res["perm_p_pair"],
                "perm_p"         : pm_res["perm_p"],
                "perm_q_orient"  : float("nan"),
                "perm_q_pair"    : float("nan"),
                "cohens_d"       : cd_res["cohens_d"],
                "ci_lo"          : cd_res["ci_lo"],
                "ci_hi"          : cd_res["ci_hi"],
                "skewness"       : sh_res["skewness"],
                "kurtosis"       : sh_res["kurtosis"],
                "bimodality"     : sh_res["bimodality_coeff"],
                "entropy"        : sh_res["entropy"],
                "n_errors"       : len(errs),
                "n_false_dir"    : n_fd,
                "n_near_zero"    : n_nz,
                "n_outlier"      : n_ol,
                "verdict"        : ("G-Angle ✔" if (not is_degen and da_res["dir_acc"] > 0.5
                                                    and expected == "asymmetric")
                                    else "Symmetric ✔" if (not is_degen and expected == "symmetric"
                                                           and abs(float(np.mean(deltas))) < 0.05)
                                    else "N/A (degenerate)" if is_degen else "Not significant"),
            })

            for bv in boot_res["boot_dir_accs"]:
                boot_rows.append({"relation": rel, "p": pv,
                                  "metric": "dir_acc", "value": bv})
            for bv in boot_res["boot_mean_dts"]:
                boot_rows.append({"relation": rel, "p": pv,
                                  "metric": "mean_dt", "value": bv})

            perm_summary.append({
                "relation"         : rel,
                "p"                : pv,
                "expected"         : expected,
                "n_permutations"   : n_perm,
                "obs_stat"         : pm_res["obs_stat"],
                "perm_p_orient"    : pm_res["perm_p_orient"],
                "perm_p_pair"      : pm_res["perm_p_pair"],
                "perm_p"           : pm_res["perm_p"],
                "null_mean_orient" : float(np.mean(pm_res["perm_a_stats"]))
                                     if len(pm_res["perm_a_stats"]) > 0 else None,
                "null_std_orient"  : float(np.std(pm_res["perm_a_stats"]))
                                     if len(pm_res["perm_a_stats"]) > 0 else None,
                "null_mean_pair"   : float(np.mean(pm_res["perm_b_stats"]))
                                     if len(pm_res["perm_b_stats"]) > 0 else None,
                "null_std_pair"    : float(np.std(pm_res["perm_b_stats"]))
                                     if len(pm_res["perm_b_stats"]) > 0 else None,
            })

    # ── 5. BH FDR correction ─────────────────────────────────────────────────
    log(f"\n[5] BH FDR correction  m_t={len(all_t_p_info)}  "
        f"m_perm={len(all_perm_p_info)}")

    # t-test correction
    if all_t_p_info:
        t_pvals = np.array([x[2] for x in all_t_p_info])
        t_qvals = bh_correct(t_pvals)
        # Map back to eval_rows
        lookup_t = {(rel, pv): q for (rel, pv, _), q in zip(all_t_p_info, t_qvals)}
        for row in eval_rows:
            key = (row["relation"], row["p"])
            if key in lookup_t:
                row["t_q"] = float(lookup_t[key])
                log(f"    {row['relation']:<12} p={row['p']}  "
                    f"t_p={row['t_p']:.4f}  t_q={row['t_q']:.4f}  "
                    f"{'✔ sig' if row['t_q'] < 0.05 else '  ns '}")

    # Permutation correction
    if all_perm_p_info:
        pm_pvals = np.array([x[3] for x in all_perm_p_info])
        pm_qvals = bh_correct(pm_pvals)
        lookup_orient = {}; lookup_pair = {}
        for (rel, pv, kind, _), q in zip(all_perm_p_info, pm_qvals):
            if kind == "orient": lookup_orient[(rel, pv)] = q
            else:                lookup_pair[(rel, pv)]   = q
        for row in eval_rows:
            key = (row["relation"], row["p"])
            if key in lookup_orient: row["perm_q_orient"] = float(lookup_orient[key])
            if key in lookup_pair:   row["perm_q_pair"]   = float(lookup_pair[key])

    # Build significance report
    for row in eval_rows:
        sig_entries.append({
            "relation"      : row["relation"],
            "p"             : row["p"],
            "expected"      : row["expected"],
            "n"             : row["n"],
            "dir_acc"       : row["dir_acc"],
            "t_p"           : row["t_p"],
            "t_q"           : row["t_q"],
            "perm_p"        : row["perm_p"],
            "perm_q_orient" : row.get("perm_q_orient", float("nan")),
            "perm_q_pair"   : row.get("perm_q_pair",   float("nan")),
            "cohens_d"      : row["cohens_d"],
            "significant"   : (not row["degenerate"]
                                and not np.isnan(row["t_q"])
                                and row["t_q"] < 0.05),
            "verdict"       : row["verdict"],
        })

    # ── 6. Save outputs ───────────────────────────────────────────────────────
    log("\n[6] Saving outputs …")

    eval_fields = [
        "relation","p","expected","n","degenerate",
        "mean_dt","std_dt","mean_abs_dt",
        "dir_acc","n_uncertain","dir_acc_ci_lo","dir_acc_ci_hi",
        "ranking_acc","rank_acc_ci_lo","rank_acc_ci_hi",
        "t_stat","t_p","t_q",
        "perm_p_orient","perm_p_pair","perm_p","perm_q_orient","perm_q_pair",
        "cohens_d","ci_lo","ci_hi",
        "skewness","kurtosis","bimodality","entropy",
        "n_errors","n_false_dir","n_near_zero","n_outlier","verdict",
    ]
    err_fields = ["word1","word2","relation","p","delta_theta","magnitude","error_type"]
    boot_fields = ["relation","p","metric","value"]
    perm_fields = [
        "relation","p","expected","n_permutations","obs_stat",
        "perm_p_orient","perm_p_pair","perm_p",
        "null_mean_orient","null_std_orient",
        "null_mean_pair","null_std_pair",
    ]

    p_eval = os.path.join(out, "evaluation_result.csv")
    p_sig  = os.path.join(out, "significance_report.json")
    p_boot = os.path.join(out, "bootstrap_samples.csv")
    p_perm = os.path.join(out, "permutation_summary.json")
    p_err  = os.path.join(out, "error_analysis.csv")

    save_csv(eval_rows,    p_eval, eval_fields)
    save_json(sig_entries, p_sig)
    save_csv(boot_rows,    p_boot, boot_fields)
    save_json(perm_summary, p_perm)
    save_csv(error_rows,   p_err,  err_fields)

    log(f"  evaluation_result.csv   → {p_eval}  ({len(eval_rows)} rows)")
    log(f"  significance_report.json → {p_sig}  ({len(sig_entries)} entries)")
    log(f"  bootstrap_samples.csv   → {p_boot}  ({len(boot_rows)} rows)")
    log(f"  permutation_summary.json → {p_perm}")
    log(f"  error_analysis.csv      → {p_err}  ({len(error_rows)} errors)")

    log("\n" + "=" * 60)
    log("DONE")
    log("=" * 60)

    return {
        "eval_rows"    : eval_rows,
        "sig_entries"  : sig_entries,
        "perm_summary" : perm_summary,
        "error_rows"   : error_rows,
    }


# =============================================================================
# GUI HELPERS
# =============================================================================

def _lbl(parent, text="", fg=FG3, font=("Consolas", 9), **kw):
    return tk.Label(parent, text=text, bg=BG, fg=fg, font=font, **kw)

def _entry(parent, var, width=40):
    return tk.Entry(parent, textvariable=var, width=width,
                    bg=BG2, fg=FG, insertbackground="#fff",
                    relief="flat", font=("Consolas", 9))

def _btn(parent, text, cmd, color=ACCENT, font=("Consolas", 9, "bold"), **kw):
    return tk.Button(parent, text=text, command=cmd, bg=color, fg="#fff",
                     relief="flat", font=font,
                     activebackground="#42a5f5", cursor="hand2", **kw)


# =============================================================================
# GUI — App
# =============================================================================

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("APP 4 — Statistical Evaluator")
        self.geometry("900x800")
        self.configure(bg=BG)
        self.resizable(True, True)
        self._result = None
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
                (" 📊  Evaluation ", "_t_eval"),
                (" 🔬  Significance ", "_t_sig"),
                (" ⚠  Errors ", "_t_err"),
                (" 📝  Log ", "_t_log")]
        for label, attr in tabs:
            f = ttk.Frame(nb); setattr(self, attr, f); nb.add(f, text=label)
        self._build_cfg()
        self._build_eval()
        self._build_sig()
        self._build_err()
        self._build_log()

    # =========================================================
    # TAB 1 — Config & Run
    # =========================================================
    def _build_cfg(self):
        P = dict(padx=10, pady=5)

        # Input files
        lf1 = ttk.LabelFrame(self._t_cfg, text="1.  Input Files", padding=8)
        lf1.pack(fill="x", **P)

        for label, attr, ext in [
            ("geometry_scores.csv", "_geo_path", "*.csv"),
            ("relation_metadata.json (optional)", "_meta_path", "*.json"),
            ("config.json (optional)", "_cfg_json_path", "*.json"),
        ]:
            row = tk.Frame(lf1, bg=BG); row.pack(fill="x", pady=2)
            _lbl(row, f"{label}:").pack(side="left")
            var = tk.StringVar(); setattr(self, attr, var)
            _entry(row, var, 46).pack(side="left", padx=6)
            _btn(row, "Browse…",
                 lambda e=ext, v=var: self._browse_file(e, v), ACC2).pack(side="left")

        # Parameters
        lf2 = ttk.LabelFrame(self._t_cfg, text="2.  Parameters", padding=8)
        lf2.pack(fill="x", **P)

        r1 = tk.Frame(lf2, bg=BG); r1.pack(fill="x", pady=2)
        _lbl(r1, "p values (space-sep, blank = all from CSV):").pack(side="left")
        self._p_str = tk.StringVar(value="")
        _entry(r1, self._p_str, 28).pack(side="left", padx=6)

        r2 = tk.Frame(lf2, bg=BG); r2.pack(fill="x", pady=2)
        _lbl(r2, "n_permutations:").pack(side="left")
        self._n_perm = tk.StringVar(value="500")
        _entry(r2, self._n_perm, 8).pack(side="left", padx=6)
        _lbl(r2, "   n_boot:").pack(side="left")
        self._n_boot = tk.StringVar(value="2000")
        _entry(r2, self._n_boot, 8).pack(side="left", padx=6)

        r3 = tk.Frame(lf2, bg=BG); r3.pack(fill="x", pady=2)
        _lbl(r3, "Output directory:").pack(side="left")
        self._out_dir = tk.StringVar(
            value=os.path.join(os.path.expanduser("~"), "stat_eval_output"))
        _entry(r3, self._out_dir, 42).pack(side="left", padx=6)
        _btn(r3, "Browse…", lambda: (
            d := filedialog.askdirectory()) and self._out_dir.set(d),
             ACC2).pack(side="left")

        # Load config JSON button
        r4 = tk.Frame(lf2, bg=BG); r4.pack(fill="x", pady=2)
        _btn(r4, "Load config.json into fields", self._load_cfg_json, ACC2).pack(side="left")

        # Run
        lf3 = ttk.LabelFrame(self._t_cfg, text="3.  Run", padding=8)
        lf3.pack(fill="x", **P)
        bf = tk.Frame(lf3, bg=BG); bf.pack(fill="x")
        self._run_btn = _btn(bf, "▶  Run Statistical Evaluator",
                             self._start, PURPLE, pady=5, padx=18)
        self._run_btn.pack(side="left", padx=4)
        self._pbar = ttk.Progressbar(lf3, mode="indeterminate", length=380)
        self._pbar.pack(fill="x", pady=(6, 0))

    # =========================================================
    # TAB 2 — Evaluation Results
    # =========================================================
    def _build_eval(self):
        P = dict(padx=10, pady=5)
        ctrl = tk.Frame(self._t_eval, bg=BG); ctrl.pack(fill="x", **P)
        _lbl(ctrl, "Filter p:").pack(side="left")
        self._fp = tk.StringVar(value="all")
        self._fp_om = tk.OptionMenu(ctrl, self._fp, "all")
        self._fp_om.config(bg=BG2, fg=FG, activebackground=ACCENT,
                           highlightbackground=BG, relief="flat",
                           font=("Consolas", 9), width=8)
        self._fp_om["menu"].config(bg=BG2, fg=FG)
        self._fp_om.pack(side="left", padx=4)
        _lbl(ctrl, "  Relation:").pack(side="left")
        self._fr = tk.StringVar(value="all")
        self._fr_om = tk.OptionMenu(ctrl, self._fr, "all")
        self._fr_om.config(bg=BG2, fg=FG, activebackground=ACCENT,
                           highlightbackground=BG, relief="flat",
                           font=("Consolas", 9), width=14)
        self._fr_om["menu"].config(bg=BG2, fg=FG)
        self._fr_om.pack(side="left", padx=4)
        _btn(ctrl, "🔄 Refresh", self._refresh_eval, ACC2).pack(side="left", padx=6)

        lf = ttk.LabelFrame(self._t_eval, text="Evaluation Results", padding=6)
        lf.pack(fill="both", expand=True, **P)
        cols = ("relation","p","n","dir_acc","rank_acc",
                "t_p","t_q","perm_p","cohens_d","verdict")
        widths = [110,40,55,80,80,75,75,75,80,140]
        self._eval_tree = ttk.Treeview(lf, columns=cols, show="headings", height=18)
        for c, w in zip(cols, widths):
            self._eval_tree.heading(c, text=c)
            self._eval_tree.column(c, width=w, anchor="center")
        xsb = ttk.Scrollbar(lf, orient="horizontal", command=self._eval_tree.xview)
        vsb = ttk.Scrollbar(lf, orient="vertical",   command=self._eval_tree.yview)
        self._eval_tree.configure(xscrollcommand=xsb.set, yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        self._eval_tree.pack(fill="both", expand=True)
        xsb.pack(fill="x")

    # =========================================================
    # TAB 3 — Significance Report
    # =========================================================
    def _build_sig(self):
        P = dict(padx=10, pady=5)
        lf = ttk.LabelFrame(self._t_sig, text="Significance Report (BH FDR corrected)", padding=6)
        lf.pack(fill="both", expand=True, **P)
        cols = ("relation","p","dir_acc","t_p","t_q",
                "perm_p","cohens_d","significant","verdict")
        widths = [110,40,80,75,75,75,80,80,140]
        self._sig_tree = ttk.Treeview(lf, columns=cols, show="headings", height=20)
        for c, w in zip(cols, widths):
            self._sig_tree.heading(c, text=c)
            self._sig_tree.column(c, width=w, anchor="center")
        xsb = ttk.Scrollbar(lf, orient="horizontal", command=self._sig_tree.xview)
        vsb = ttk.Scrollbar(lf, orient="vertical",   command=self._sig_tree.yview)
        self._sig_tree.configure(xscrollcommand=xsb.set, yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        self._sig_tree.pack(fill="both", expand=True)
        xsb.pack(fill="x")

    # =========================================================
    # TAB 4 — Error Analysis
    # =========================================================
    def _build_err(self):
        P = dict(padx=10, pady=5)

        ctrl = tk.Frame(self._t_err, bg=BG); ctrl.pack(fill="x", **P)
        _lbl(ctrl, "Filter type:").pack(side="left")
        self._fe = tk.StringVar(value="all")
        self._fe_om = tk.OptionMenu(ctrl, self._fe,
                                    "all", "false_direction",
                                    "near_zero_ambiguity", "extreme_outlier")
        self._fe_om.config(bg=BG2, fg=FG, activebackground=ACCENT,
                           highlightbackground=BG, relief="flat",
                           font=("Consolas", 9), width=22)
        self._fe_om["menu"].config(bg=BG2, fg=FG)
        self._fe_om.pack(side="left", padx=6)
        _btn(ctrl, "🔄 Refresh", self._refresh_err, ACC2).pack(side="left", padx=4)

        lf = ttk.LabelFrame(self._t_err, text="Error Analysis", padding=6)
        lf.pack(fill="both", expand=True, **P)
        cols = ("word1","word2","relation","p","delta_theta","magnitude","error_type")
        widths = [120,120,100,40,100,90,180]
        self._err_tree = ttk.Treeview(lf, columns=cols, show="headings", height=20)
        for c, w in zip(cols, widths):
            self._err_tree.heading(c, text=c)
            self._err_tree.column(c, width=w, anchor="center")
        vsb = ttk.Scrollbar(lf, orient="vertical", command=self._err_tree.yview)
        self._err_tree.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        self._err_tree.pack(fill="both", expand=True)

        self._err_summary = _lbl(self._t_err, "", fg=FG2,
                                  font=("Consolas", 9, "bold"))
        self._err_summary.pack(pady=4)

    # =========================================================
    # TAB 5 — Log
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

    # ── Browse helpers ────────────────────────────────────────────────────────
    def _browse_file(self, ext, var):
        p = filedialog.askopenfilename(
            filetypes=[(ext.replace("*.", "").upper(), ext), ("All", "*.*")])
        if p: var.set(p)

    def _load_cfg_json(self):
        p = self._cfg_json_path.get().strip()
        if not p or not os.path.isfile(p):
            messagebox.showerror("Error", "Select a config.json first."); return
        try:
            cfg = load_json(p)
            if "p_values"       in cfg: self._p_str.set(" ".join(str(v) for v in cfg["p_values"]))
            if "n_permutations" in cfg: self._n_perm.set(str(cfg["n_permutations"]))
            if "n_boot"         in cfg: self._n_boot.set(str(cfg["n_boot"]))
            messagebox.showinfo("Loaded", "Config loaded.")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    # ── Run ──────────────────────────────────────────────────────────────────
    def _start(self):
        geo = self._geo_path.get().strip()
        if not geo or not os.path.isfile(geo):
            messagebox.showerror("Error", "geometry_scores.csv not found."); return

        p_str = self._p_str.get().strip()
        try:
            p_vals = [float(x) for x in p_str.split()] if p_str else None
        except ValueError:
            messagebox.showerror("Error", "p values must be numbers."); return

        cfg = {
            "geometry_csv"  : geo,
            "metadata_json" : self._meta_path.get().strip() or None,
            "p_values"      : p_vals,
            "n_permutations": int(self._n_perm.get() or N_PERM_DEF),
            "n_boot"        : int(self._n_boot.get() or N_BOOT_DEF),
            "out_dir"       : self._out_dir.get().strip(),
        }
        self._run_btn.config(state="disabled")
        self._pbar.start(12)

        def _run():
            try:
                result = run_pipeline(cfg, self._log)
                self._result = result
                self.after(0, lambda: (
                    self._refresh_eval(),
                    self._refresh_sig(),
                    self._refresh_err(),
                    messagebox.showinfo("Done",
                        f"Statistical evaluation complete!\n"
                        f"{len(result['eval_rows'])} relation×p combinations\n"
                        f"{len(result['error_rows'])} error pairs\n"
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

    # ── Refresh tables ────────────────────────────────────────────────────────
    def _fmt(self, v, fmt=".4f"):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "—"
        if isinstance(v, bool): return str(v)
        return format(v, fmt)

    def _refresh_eval(self):
        if not self._result: return
        rows  = self._result["eval_rows"]
        fp    = self._fp.get(); fr = self._fr.get()
        p_list  = sorted({r["p"]        for r in rows})
        rel_list= sorted({r["relation"] for r in rows})

        for var, om, opts in [
            (self._fp, self._fp_om, ["all"] + [str(p) for p in p_list]),
            (self._fr, self._fr_om, ["all"] + rel_list),
        ]:
            m = om["menu"]; m.delete(0, "end")
            for o in opts:
                m.add_command(label=o, command=lambda v=o, sv=var: sv.set(v))

        self._eval_tree.delete(*self._eval_tree.get_children())
        for r in rows:
            if fp != "all" and str(r["p"]) != fp: continue
            if fr != "all" and r["relation"]  != fr: continue
            self._eval_tree.insert("", "end", values=(
                r["relation"], r["p"], r["n"],
                self._fmt(r["dir_acc"],     ".1%"),
                self._fmt(r["ranking_acc"], ".1%"),
                self._fmt(r["t_p"]),
                self._fmt(r["t_q"]),
                self._fmt(r["perm_p"]),
                self._fmt(r["cohens_d"]),
                r["verdict"],
            ))

    def _refresh_sig(self):
        if not self._result: return
        self._sig_tree.delete(*self._sig_tree.get_children())
        for e in self._result["sig_entries"]:
            self._sig_tree.insert("", "end", values=(
                e["relation"], e["p"],
                self._fmt(e["dir_acc"],  ".1%"),
                self._fmt(e["t_p"]),
                self._fmt(e["t_q"]),
                self._fmt(e["perm_p"]),
                self._fmt(e["cohens_d"]),
                "✔" if e["significant"] else "✗",
                e["verdict"],
            ))

    def _refresh_err(self):
        if not self._result: return
        fe   = self._fe.get()
        errs = self._result["error_rows"]
        self._err_tree.delete(*self._err_tree.get_children())
        shown = 0
        counts = defaultdict(int)
        for e in errs:
            counts[e["error_type"]] += 1
            if fe != "all" and e["error_type"] != fe:
                continue
            self._err_tree.insert("", "end", values=(
                e["word1"], e["word2"],
                e.get("relation", ""), e.get("p", ""),
                f"{e['delta_theta']:+.6f}",
                f"{e['magnitude']:.6f}",
                e["error_type"],
            ))
            shown += 1
        self._err_summary.config(
            text=f"Total errors: {len(errs)}   "
                 f"false_direction={counts['false_direction']}   "
                 f"near_zero={counts['near_zero_ambiguity']}   "
                 f"outlier={counts['extreme_outlier']}   "
                 f"(showing {shown})")


# =============================================================================
if __name__ == "__main__":
    App().mainloop()
