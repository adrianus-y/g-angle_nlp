"""
APP 5 — Parameter Sweep Lab
============================
Eksplorasi parameter p dan stabilitas sistem.

Analisis yang dilakukan:
  • multi-p sweep       — evaluasi Δθ untuk setiap p ∈ p_values
  • stability detection — identifikasi plateau performa (score ≥ peak − 2%)
  • plateau analysis    — region p mana yang stabil
  • auto-selection      — pilih p terkecil di plateau (parsimoni)
  • sensitivity analysis — seberapa sensitif dir_acc / mean|Δθ| terhadap perubahan p

Input (dari APP 3 geometry_scores.csv atau langsung dari pipeline):
  dataset.csv          — word1,word2,relation,is_symmetric  (dari APP 1)
  embedding_cache.pkl  — dict word→np.array                 (dari APP 2)
  sweep_config.json    — {"p_values":[1.5,2,3,4,5,10], "asym_relations":["hyponymy",...]}

Output:
  sweep_result.csv     — p,relation,dir_acc,rank_acc,mean_abs_delta,score
  stable_region.json   — {stable_region:[3,4,5], best_p:3, scores:{...}}
  best_p_report.txt    — human-readable selection report
  sensitivity_metrics.json — δ(dir_acc)/δp, CV, saturation point per relation

Requirements:
  pip install numpy scipy matplotlib
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading, os, json, csv, pickle, time
import numpy as np
from scipy.stats import ttest_1samp, mannwhitneyu
from collections import defaultdict
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# =============================================================================
# COLOUR PALETTE
# =============================================================================
BG, BG2   = "#1a1a2e", "#16213e"
FG, FG2   = "#e0e0e0", "#90caf9"
FG3       = "#b0bec5"
ACCENT    = "#0d47a1"
ACC2      = "#1565c0"
GREEN     = "#2e7d32"
ORANGE    = "#e65100"
PURPLE    = "#4a148c"

PALETTE = {
    "hyponymy"  : "#d32f2f",
    "meronymy"  : "#e65100",
    "capital"   : "#2e7d32",
    "sibling"   : "#1565c0",
    "coordinate": "#6a1b9a",
}

RANDOM_SEED = 42
DIR_EPS     = 1e-4
SYM_TAU     = 0.05
PLATEAU_TOL = 0.02   # 2% tolerance for stable plateau

# =============================================================================
# GEOMETRY — minimal inline (no dep on APP 3 file)
# =============================================================================

class _Geo:
    eps = 1e-15
    def functional_g(self, x, y, p):
        nx = np.linalg.norm(x, ord=p)
        if nx < self.eps: return 0.0
        ax = np.abs(x); m = ax > 0; t = np.zeros_like(x)
        t[m] = (ax[m]**(p-1)) * np.sign(x[m])
        return float((nx**(2-p)) * np.sum(t * y))
    def g_sim(self, u, v, p):
        nu = np.linalg.norm(u, ord=p); nv = np.linalg.norm(v, ord=p)
        if nu < self.eps or nv < self.eps: return 0.0
        return float(np.clip(self.functional_g(v, u, p) / (nu * nv), -1, 1))
    def g_angle(self, u, v, p):
        return float(np.arccos(np.clip(self.g_sim(u, v, p), -1, 1)))
    def delta_theta(self, u, v, p):
        return self.g_angle(u, v, p) - self.g_angle(v, u, p)

GEO = _Geo()

# =============================================================================
# I/O
# =============================================================================

def load_dataset(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append((r["word1"].strip().lower(),
                         r["word2"].strip().lower(),
                         r["relation"].strip(),
                         int(r.get("is_symmetric", 0))))
    return rows

def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(data, path):
    def _fix(o):
        if isinstance(o, float) and np.isnan(o): return None
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return o
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=_fix)

def save_csv(rows, path, fields):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)

def save_txt(lines, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

# =============================================================================
# CORE: single-p evaluation (returns compact stats dict)
# =============================================================================

def eval_one_p(pairs_by_rel, get_vec, pv, expected_map, log):
    """
    Evaluate all relations for one p value.
    Returns dict: rel → stats_dict
    """
    results = {}
    for rel, pairs in pairs_by_rel.items():
        exp     = expected_map.get(rel, "asymmetric")
        deltas  = []
        for w1, w2, *_ in pairs:
            eu = get_vec(w1); ev = get_vec(w2)
            if eu is None or ev is None: continue
            deltas.append(GEO.delta_theta(eu, ev, pv))
        if not deltas:
            results[rel] = {"n": 0, "dir_acc": float("nan"),
                            "rank_acc": float("nan"),
                            "mean_abs_dt": float("nan"),
                            "degenerate": False, "expected": exp}
            continue

        d = np.array(deltas)
        n = len(d)
        is_degen = bool(np.allclose(d, 0.0, atol=1e-9))

        # direction accuracy
        if exp == "asymmetric":
            d_cert  = d[np.abs(d) >= DIR_EPS]
            dir_acc = float(np.mean(d_cert < 0)) if len(d_cert) > 0 else 0.0
        else:
            dir_acc = float(np.mean(np.abs(d) < SYM_TAU))

        # ranking accuracy (AUC, mismatch pairs from shuffled pool)
        if is_degen or n < 4:
            rank_acc = float("nan")
        else:
            rng     = np.random.default_rng(RANDOM_SEED + 1)
            n_rand  = min(n, 300)
            idx_u   = rng.choice(n, size=n_rand, replace=True)
            idx_v   = rng.choice(n, size=n_rand, replace=True)
            for i in range(n_rand):
                while idx_v[i] == idx_u[i]:
                    idx_v[i] = rng.integers(0, n)
            rand_d  = d[idx_v]
            if exp == "asymmetric":
                rs = -d; rr = -rand_d
            else:
                rs = -np.abs(d); rr = -np.abs(rand_d)
            mw, _   = mannwhitneyu(rs, rr, alternative="greater")
            rank_acc = float(mw / (len(rs) * len(rr)))

        results[rel] = {
            "n"          : n,
            "dir_acc"    : dir_acc,
            "rank_acc"   : rank_acc,
            "mean_abs_dt": float(np.mean(np.abs(d))),
            "mean_dt"    : float(np.mean(d)),
            "std_dt"     : float(np.std(d)),
            "degenerate" : is_degen,
            "expected"   : exp,
        }
        log(f"    {rel:<14} n={n:>4}  dir_acc={dir_acc:.1%}  "
            f"rank_acc={'N/A' if np.isnan(rank_acc) else f'{rank_acc:.1%}'}  "
            f"mean|Δθ|={float(np.mean(np.abs(d))):.4f}"
            f"{'  DEGENERATE' if is_degen else ''}")
    return results


# =============================================================================
# COMPOSITE SCORE & PLATEAU ANALYSIS
# =============================================================================

def compute_composite_scores(sweep, asym_rels):
    """
    score(p) = mean_asym_dir_acc + 0.3 * mean_asym_abs_dt_norm

    Returns:
      dir_accs  dict p → float
      abs_dts   dict p → float
      scores    dict p → float
    """
    p_vals   = sorted(sweep.keys())
    dir_accs = {}
    abs_dts  = {}

    for pv in p_vals:
        da_v, adt_v, any_valid = [], [], False
        for rel in asym_rels:
            r = sweep[pv].get(rel)
            if not r or r["n"] == 0 or r["degenerate"]: continue
            any_valid = True
            da_v.append(r["dir_acc"])
            adt_v.append(r["mean_abs_dt"])
        if not any_valid:
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
    return dir_accs, abs_dts, scores


def select_best_p(scores, dir_accs):
    """
    Stable region: all p with score ≥ peak − PLATEAU_TOL.
    Best p: smallest p in stable region (parsimony).
    """
    p_vals  = sorted(scores.keys())
    valid   = {pv: s for pv, s in scores.items() if not np.isnan(s)}
    if not valid:
        non_d = [pv for pv in p_vals if not np.isnan(dir_accs.get(pv, float("nan")))]
        best  = non_d[-1] if non_d else p_vals[-1]
        return best, [], float("nan")
    peak   = max(valid.values())
    stable = sorted([pv for pv, s in valid.items() if s >= peak - PLATEAU_TOL])
    best   = stable[0] if stable else max(valid, key=valid.get)
    return best, stable, peak


def build_report(p_vals, dir_accs, abs_dts, scores, best_p, stable, peak):
    """Build human-readable best-p report lines."""
    lines = [
        "APP 5 — Parameter Sweep Lab  Best-p Report",
        "=" * 58,
        "",
        "Score(p) = mean_asym_dir_acc + 0.3 × mean_asym_|Δθ|_norm",
        "Stable plateau: score ≥ peak − 2%",
        "Best p: smallest p in stable plateau (parsimony)",
        "",
        f"  {'p':>6}  {'dir_acc':>9}  {'mean|Δθ|':>10}  {'score':>8}  status",
        "  " + "─" * 54,
    ]
    for pv in sorted(p_vals):
        da  = dir_accs.get(pv, float("nan"))
        adt = abs_dts.get(pv,  float("nan"))
        sc  = scores.get(pv,   float("nan"))
        if np.isnan(da):       status = "DEGENERATE"
        elif pv == best_p:     status = "◀ BEST"
        elif pv in stable:     status = "stable plateau"
        else:                  status = ""
        lines.append(
            f"  {pv:>6.1f}  "
            f"{(f'{da:.1%}' if not np.isnan(da) else 'N/A'):>9}  "
            f"{(f'{adt:.4f}' if not np.isnan(adt) else 'N/A'):>10}  "
            f"{(f'{sc:.4f}' if not np.isnan(sc) else 'N/A'):>8}  {status}"
        )
    lines += [
        "  " + "─" * 54,
        f"  Peak score       : {peak:.4f}" if not np.isnan(peak) else "  Peak score: N/A",
        f"  Stable plateau   : p ∈ {stable}",
        f"  Best p selected  : p = {best_p}",
    ]
    return lines


# =============================================================================
# SENSITIVITY ANALYSIS
# =============================================================================

def compute_sensitivity(sweep, p_vals, relations):
    """
    For each relation:
      • slope (δ dir_acc / δ p)  — via finite differences on valid p pairs
      • CV    (std / mean)        — of dir_acc across p values
      • saturation_p              — first p where |Δ score| < 0.01 (plateau onset)

    Returns list of dicts.
    """
    result = []
    for rel in relations:
        da_series  = []
        adt_series = []
        pv_valid   = []
        for pv in p_vals:
            r = sweep[pv].get(rel)
            if not r or r["n"] == 0 or r["degenerate"]: continue
            da_series.append(r["dir_acc"])
            adt_series.append(r["mean_abs_dt"])
            pv_valid.append(pv)

        if len(da_series) < 2:
            result.append({"relation": rel, "slope_dir_acc": float("nan"),
                           "cv_dir_acc": float("nan"),
                           "saturation_p": float("nan"),
                           "mean_dir_acc": float("nan"),
                           "range_dir_acc": float("nan")})
            continue

        da_arr = np.array(da_series)
        pv_arr = np.array(pv_valid, dtype=float)

        # Finite-difference slope (mean of consecutive differences)
        diffs = np.diff(da_arr) / np.diff(pv_arr)
        slope = float(np.mean(diffs))

        # Coefficient of variation
        mu  = float(np.mean(da_arr))
        std = float(np.std(da_arr, ddof=1))
        cv  = std / mu if abs(mu) > 1e-9 else float("nan")

        # Saturation: first p where |Δ dir_acc| < 0.01
        sat_p = float("nan")
        for i in range(len(diffs)):
            if abs(diffs[i]) < 0.01:
                sat_p = float(pv_valid[i])
                break

        result.append({
            "relation"     : rel,
            "slope_dir_acc": slope,
            "cv_dir_acc"   : cv,
            "saturation_p" : sat_p,
            "mean_dir_acc" : mu,
            "range_dir_acc": float(np.max(da_arr) - np.min(da_arr)),
        })
    return result


# =============================================================================
# PLOTTING
# =============================================================================

def _mpl_style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor"  : "#f8f9fa",
        "axes.edgecolor"  : "#aaaaaa",
        "axes.labelsize"  : 9,
        "axes.titlesize"  : 9,
        "axes.grid"       : True,
        "grid.color"      : "#dddddd",
        "grid.alpha"      : 0.7,
        "xtick.labelsize" : 8,
        "ytick.labelsize" : 8,
        "legend.facecolor": "white",
        "legend.fontsize" : 7.5,
        "font.family"     : "DejaVu Sans",
    })


def plot_sweep(sweep, scores, dir_accs, abs_dts,
               best_p, stable, asym_rels, all_rels):
    """4-panel sweep plot: Dir Acc / Rank Acc / Mean|Δθ| / Composite Score."""
    _mpl_style()
    p_vals = sorted(sweep.keys())
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2), constrained_layout=True)

    def _shade(ax, pct=False):
        if stable and len(stable) >= 2:
            lo, hi = min(stable), max(stable)
            ax.axvspan(lo, hi, alpha=0.08, color="#43a047",
                       label=f"Stable p ∈ {stable}")
        if best_p is not None:
            ax.axvline(best_p, color="#e53935", lw=1.8,
                       linestyle="-.", label=f"Best p={best_p}", zorder=5)

    # ── Panel 1: Direction Accuracy ──────────────────────────────────────────
    ax = axes[0]
    for rel in all_rels:
        exp = sweep[p_vals[0]].get(rel, {}).get("expected", "asymmetric")
        xs, ys = [], []
        for pv in p_vals:
            r = sweep[pv].get(rel)
            if r and r["n"] > 0 and not r["degenerate"]:
                xs.append(pv); ys.append(r["dir_acc"] * 100)
        if xs:
            ls = "-" if exp == "asymmetric" else "--"
            ax.plot(xs, ys, marker="o", label=rel,
                    color=PALETTE.get(rel, "#555"), lw=1.8, linestyle=ls)
    ax.axhline(50, color="#aaa", lw=1, linestyle=":", label="Chance (50%)")
    _shade(ax, pct=True)
    ax.set_xlabel("p"); ax.set_ylabel("Direction Accuracy (%)")
    ax.set_title(f"Direction Accuracy vs p\n(best p={best_p}, dash-dot red)")
    ax.set_xticks(p_vals); ax.set_ylim(0, 110)
    ax.legend(fontsize=7); ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ── Panel 2: Ranking Accuracy ─────────────────────────────────────────────
    ax = axes[1]
    for rel in all_rels:
        exp = sweep[p_vals[0]].get(rel, {}).get("expected", "asymmetric")
        xs, ys = [], []
        for pv in p_vals:
            r = sweep[pv].get(rel)
            if r and r["n"] > 0 and not r["degenerate"] \
                    and not np.isnan(r["rank_acc"]):
                xs.append(pv); ys.append(r["rank_acc"] * 100)
        if xs:
            ls = "-" if exp == "asymmetric" else "--"
            ax.plot(xs, ys, marker="s", label=rel,
                    color=PALETTE.get(rel, "#555"), lw=1.8, linestyle=ls)
    ax.axhline(50, color="#aaa", lw=1, linestyle=":", label="Chance (50%)")
    _shade(ax)
    ax.set_xlabel("p"); ax.set_ylabel("Ranking Accuracy / AUC (%)")
    ax.set_title("Ranking Accuracy vs p")
    ax.set_xticks(p_vals); ax.set_ylim(0, 110)
    ax.legend(fontsize=7); ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ── Panel 3: Mean |Δθ| ────────────────────────────────────────────────────
    ax = axes[2]
    for rel in all_rels:
        exp = sweep[p_vals[0]].get(rel, {}).get("expected", "asymmetric")
        xs, ys = [], []
        for pv in p_vals:
            r = sweep[pv].get(rel)
            if r and r["n"] > 0 and not r["degenerate"]:
                xs.append(pv); ys.append(r["mean_abs_dt"])
        if xs:
            ls = "-" if exp == "asymmetric" else ":"
            ax.plot(xs, ys, marker="^", label=rel,
                    color=PALETTE.get(rel, "#555"), lw=1.8, linestyle=ls)
    ax.axvline(2.0, color="#aaa", lw=1.2, linestyle="--",
               label="p=2 (degenerate)")
    _shade(ax)
    ax.set_xlabel("p"); ax.set_ylabel("Mean |Δθ|  [rad]")
    ax.set_title("Mean |Δθ| vs p\n(asymm=solid, symm=dotted)")
    ax.set_xticks(p_vals)
    ax.legend(fontsize=7); ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ── Panel 4: Composite Score ──────────────────────────────────────────────
    ax = axes[3]
    sc_ps = [pv for pv in p_vals if not np.isnan(scores.get(pv, float("nan")))]
    sc_ys = [scores[pv] for pv in sc_ps]
    if sc_ps:
        colors = []
        for pv in sc_ps:
            if pv == best_p:       colors.append("#e53935")
            elif pv in stable:     colors.append("#43a047")
            else:                  colors.append("#78909c")
        bars = ax.bar(sc_ps, sc_ys, color=colors, edgecolor="white",
                      lw=0.6, zorder=3, width=0.35)
        for bar, pv, sc in zip(bars, sc_ps, sc_ys):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(sc_ys, default=0) * 0.02,
                    f"{sc:.3f}", ha="center", va="bottom", fontsize=8,
                    fontweight="bold",
                    color="#e53935" if pv == best_p else "#333")
        degen_ps = [pv for pv in p_vals if np.isnan(scores.get(pv, float("nan")))]
        for pv in degen_ps:
            ax.text(pv, max(sc_ys, default=0) * 0.05, "✗\ndegen.",
                    ha="center", va="bottom", fontsize=7, color="#aaa")
        if stable and len(stable) >= 2:
            ax.axvspan(min(stable), max(stable), alpha=0.10, color="#43a047")
        if best_p is not None:
            ax.axvline(best_p, color="#e53935", lw=2.0, linestyle="-.", zorder=5)
    ax.set_xlabel("p")
    ax.set_ylabel("Composite Score\ndir_acc + 0.3·|Δθ|_norm")
    ax.set_title(f"Auto-selected Best p = {best_p}\nScore(p) = dir_acc + 0.3·|Δθ|_norm")
    ax.set_xticks(p_vals)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    stable_str = f"p ∈ {stable}" if stable else "no plateau"
    fig.suptitle(
        f"Parameter Sweep — Auto-selected: p = {best_p}  |  Stable region: {stable_str}",
        fontsize=10, fontweight="bold", y=1.01)
    return fig


def plot_sensitivity(sens_list):
    """Bar chart: sensitivity metrics per relation."""
    _mpl_style()
    rels = [s["relation"] for s in sens_list]
    slopes = [s["slope_dir_acc"]  for s in sens_list]
    cvs    = [s["cv_dir_acc"]     for s in sens_list]
    ranges = [s["range_dir_acc"]  for s in sens_list]

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True)
    colors = [PALETTE.get(r, "#78909c") for r in rels]

    def _bar(ax, vals, title, ylabel):
        clean = [v if not (v is None or (isinstance(v, float) and np.isnan(v))) else 0
                 for v in vals]
        bars = ax.bar(rels, clean, color=colors, edgecolor="white", lw=0.5)
        for bar, v in zip(bars, clean):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + max(abs(c) for c in clean) * 0.02,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)
        ax.set_title(title, fontsize=9); ax.set_ylabel(ylabel, fontsize=8)
        ax.set_xticklabels(rels, rotation=18, ha="right", fontsize=8)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    _bar(axes[0], slopes, "Mean δ(dir_acc)/δp", "slope (Δdir_acc / Δp)")
    _bar(axes[1], cvs,    "CV of dir_acc across p", "CV = std/mean")
    _bar(axes[2], ranges, "Range of dir_acc", "max − min dir_acc")
    fig.suptitle("Sensitivity Analysis — per relation", fontsize=10,
                 fontweight="bold")
    return fig


# =============================================================================
# FULL PIPELINE
# =============================================================================

def run_pipeline(cfg, log):
    """
    cfg keys:
      dataset_path     str
      embedding_path   str
      p_values         list[float]
      asym_relations   list[str]   default ["hyponymy","meronymy","capital"]
      out_dir          str
    """
    out = cfg["out_dir"]
    os.makedirs(out, exist_ok=True)
    log("=" * 60)
    log("APP 5 — Parameter Sweep Lab")
    log("=" * 60)

    # ── 1. Load data ─────────────────────────────────────────────────────────
    log(f"\n[1] Loading dataset: {cfg['dataset_path']}")
    rows = load_dataset(cfg["dataset_path"])
    by_rel = defaultdict(list)
    for w1, w2, rel, sym in rows:
        by_rel[rel].append((w1, w2, rel, sym))
    all_rels    = sorted(by_rel.keys())
    asym_rels   = cfg.get("asym_relations") or [
        r for r in all_rels if r not in ("sibling", "coordinate")]
    expected_map = {rel: ("symmetric" if any(s for *_, s in by_rel[rel])
                          else "asymmetric") for rel in all_rels}
    log(f"  Relations: {all_rels}")
    log(f"  Asym key relations: {asym_rels}")

    log(f"\n[2] Loading embedding: {cfg['embedding_path']}")
    vocab   = load_pkl(cfg["embedding_path"])
    get_vec = lambda w: vocab.get(w.lower())
    log(f"  {len(vocab):,} words")

    p_values = cfg.get("p_values") or [1.5, 2, 3, 4, 5, 10]
    log(f"\n[3] Sweep over p ∈ {p_values}")

    # ── 2. Sweep ──────────────────────────────────────────────────────────────
    sweep = {}
    for pv in p_values:
        log(f"\n  ── p = {pv} ──")
        sweep[pv] = eval_one_p(by_rel, get_vec, pv, expected_map, log)

    # ── 3. Composite scores & plateau ────────────────────────────────────────
    log("\n[4] Computing composite scores & plateau …")
    dir_accs, abs_dts, scores = compute_composite_scores(sweep, asym_rels)
    best_p, stable, peak = select_best_p(scores, dir_accs)
    log(f"  Best p = {best_p}   Stable: {stable}   Peak score: {peak:.4f}" if not np.isnan(peak)
        else f"  Best p = {best_p}   Stable: {stable}")

    # ── 4. Sensitivity analysis ───────────────────────────────────────────────
    log("\n[5] Sensitivity analysis …")
    sens = compute_sensitivity(sweep, p_values, all_rels)
    for s in sens:
        log(f"  {s['relation']:<14}  slope={s['slope_dir_acc']:.4f}  "
            f"CV={s['cv_dir_acc']:.4f}  "
            f"sat_p={s['saturation_p']}")

    # ── 5. Build report lines ─────────────────────────────────────────────────
    report_lines = build_report(p_values, dir_accs, abs_dts, scores,
                                best_p, stable, peak)
    for ln in report_lines:
        log(ln)

    # ── 6. Save outputs ───────────────────────────────────────────────────────
    log("\n[6] Saving outputs …")

    # sweep_result.csv
    sweep_rows = []
    for pv in p_values:
        for rel in all_rels:
            r = sweep[pv].get(rel, {})
            sc = scores.get(pv, float("nan"))
            sweep_rows.append({
                "p"             : pv,
                "relation"      : rel,
                "expected"      : r.get("expected", "—"),
                "n"             : r.get("n", 0),
                "dir_acc"       : r.get("dir_acc",     float("nan")),
                "rank_acc"      : r.get("rank_acc",    float("nan")),
                "mean_abs_delta": r.get("mean_abs_dt", float("nan")),
                "mean_delta"    : r.get("mean_dt",     float("nan")),
                "degenerate"    : r.get("degenerate",  False),
                "composite_score" : sc,
            })
    sweep_path = os.path.join(out, "sweep_result.csv")
    save_csv(sweep_rows, sweep_path,
             ["p","relation","expected","n","dir_acc","rank_acc",
              "mean_abs_delta","mean_delta","degenerate","composite_score"])
    log(f"  sweep_result.csv       → {sweep_path}  ({len(sweep_rows)} rows)")

    # stable_region.json
    stable_path = os.path.join(out, "stable_region.json")
    save_json({
        "best_p"       : best_p,
        "stable_region": stable,
        "peak_score"   : peak,
        "plateau_tol"  : PLATEAU_TOL,
        "scores"       : scores,
        "dir_accs"     : dir_accs,
        "abs_dts"      : abs_dts,
    }, stable_path)
    log(f"  stable_region.json     → {stable_path}")

    # best_p_report.txt
    report_path = os.path.join(out, "best_p_report.txt")
    save_txt(report_lines, report_path)
    log(f"  best_p_report.txt      → {report_path}")

    # sensitivity_metrics.json
    sens_path = os.path.join(out, "sensitivity_metrics.json")
    save_json(sens, sens_path)
    log(f"  sensitivity_metrics.json → {sens_path}")

    log("\n" + "=" * 60)
    log(f"DONE  Best p = {best_p}")
    log("=" * 60)

    return {
        "sweep"    : sweep,
        "scores"   : scores,
        "dir_accs" : dir_accs,
        "abs_dts"  : abs_dts,
        "best_p"   : best_p,
        "stable"   : stable,
        "peak"     : peak,
        "sens"     : sens,
        "all_rels" : all_rels,
        "asym_rels": asym_rels,
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
        self.title("APP 5 — Parameter Sweep Lab")
        self.geometry("960x820")
        self.configure(bg=BG)
        self.resizable(True, True)
        self._result  = None
        self._fig_sw  = None   # sweep figure
        self._fig_sen = None   # sensitivity figure
        self._build_style()
        self._build_ui()

    def _build_style(self):
        s = ttk.Style(); s.theme_use("clam")
        for k, v in [
            ("TFrame",      {"background": BG}),
            ("TNotebook",   {"background": BG, "borderwidth": 0}),
            ("TLabelframe", {"background": BG, "foreground": FG2,
                             "bordercolor": "#0f3460"}),
            ("TLabelframe.Label", {"background": BG, "foreground": FG2,
                                   "font": ("Consolas", 9, "bold")}),
            ("Treeview",         {"background": BG2, "foreground": FG,
                                  "fieldbackground": BG2, "rowheight": 22,
                                  "font": ("Consolas", 9)}),
            ("Treeview.Heading", {"background": "#0f3460", "foreground": FG2,
                                  "font": ("Consolas", 8, "bold")}),
            ("TProgressbar",     {"troughcolor": "#0f3460",
                                  "background": "#42a5f5"}),
        ]:
            s.configure(k, **v)
        s.configure("TNotebook.Tab", background=BG2, foreground=FG3,
                    padding=[10, 4], font=("Consolas", 9))
        s.map("TNotebook.Tab",
              background=[("selected", ACCENT)],
              foreground=[("selected", "#fff")])

    def _build_ui(self):
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=6, pady=6)
        for label, attr in [
            (" ⚙  Config & Run ",   "_t_cfg"),
            (" 📊  Sweep Table ",   "_t_tbl"),
            (" 📈  Sweep Chart ",   "_t_chart"),
            (" 🔬  Sensitivity ",   "_t_sens"),
            (" 📝  Log ",           "_t_log"),
        ]:
            f = ttk.Frame(nb); setattr(self, attr, f); nb.add(f, text=label)
        self._build_cfg()
        self._build_tbl()
        self._build_chart()
        self._build_sens()
        self._build_log()

    # =========================================================
    # TAB 1 — Config & Run
    # =========================================================
    def _build_cfg(self):
        P = dict(padx=10, pady=5)

        lf1 = ttk.LabelFrame(self._t_cfg, text="1.  Input Files", padding=8)
        lf1.pack(fill="x", **P)
        for label, attr, ext in [
            ("Dataset  (.csv)",       "_ds",  "*.csv"),
            ("Embedding  (.pkl)",     "_emb", "*.pkl"),
            ("Sweep config  (.json)", "_cfg", "*.json"),
        ]:
            r = tk.Frame(lf1, bg=BG); r.pack(fill="x", pady=2)
            _lbl(r, f"{label}:").pack(side="left")
            var = tk.StringVar(); setattr(self, attr + "_path", var)
            _entry(r, var, 50).pack(side="left", padx=6)
            _btn(r, "Browse…",
                 lambda e=ext, v=var: self._browse(e, v), ACC2).pack(side="left")

        lf2 = ttk.LabelFrame(self._t_cfg, text="2.  Parameters", padding=8)
        lf2.pack(fill="x", **P)

        r1 = tk.Frame(lf2, bg=BG); r1.pack(fill="x", pady=2)
        _lbl(r1, "p values (space-sep):").pack(side="left")
        self._pv = tk.StringVar(value="1.5 2 3 4 5 10")
        _entry(r1, self._pv, 28).pack(side="left", padx=6)

        r2 = tk.Frame(lf2, bg=BG); r2.pack(fill="x", pady=2)
        _lbl(r2, "Asymmetric relations (space-sep, blank=auto):").pack(side="left")
        self._ar = tk.StringVar(value="")
        _entry(r2, self._ar, 40).pack(side="left", padx=6)

        r3 = tk.Frame(lf2, bg=BG); r3.pack(fill="x", pady=2)
        _lbl(r3, "Output directory:").pack(side="left")
        self._out = tk.StringVar(
            value=os.path.join(os.path.expanduser("~"), "sweep_output"))
        _entry(r3, self._out, 44).pack(side="left", padx=6)
        _btn(r3, "Browse…",
             lambda: (d := filedialog.askdirectory()) and self._out.set(d),
             ACC2).pack(side="left")

        r4 = tk.Frame(lf2, bg=BG); r4.pack(fill="x", pady=2)
        _btn(r4, "Load config.json into fields",
             self._load_cfg_json, ACC2).pack(side="left")

        lf3 = ttk.LabelFrame(self._t_cfg, text="3.  Run", padding=8)
        lf3.pack(fill="x", **P)
        bf = tk.Frame(lf3, bg=BG); bf.pack(fill="x")
        self._run_btn = _btn(bf, "▶  Run Sweep", self._start,
                             PURPLE, pady=5, padx=20)
        self._run_btn.pack(side="left", padx=4)
        self._best_lbl = _lbl(bf, "  best_p: —", fg=FG2,
                               font=("Consolas", 10, "bold"))
        self._best_lbl.pack(side="left", padx=10)
        self._pbar = ttk.Progressbar(lf3, mode="indeterminate", length=380)
        self._pbar.pack(fill="x", pady=(6, 0))

    # =========================================================
    # TAB 2 — Sweep Table
    # =========================================================
    def _build_tbl(self):
        P = dict(padx=10, pady=5)
        ctrl = tk.Frame(self._t_tbl, bg=BG); ctrl.pack(fill="x", **P)
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
        _btn(ctrl, "🔄 Refresh", self._refresh_tbl, ACC2).pack(side="left", padx=6)

        lf = ttk.LabelFrame(self._t_tbl,
                             text="Sweep Results per relation × p", padding=6)
        lf.pack(fill="both", expand=True, **P)
        cols = ("p","relation","n","dir_acc","rank_acc",
                "mean|Δθ|","score","degenerate","stable","best")
        widths = [45,110,55,80,80,90,80,80,60,50]
        self._tbl = ttk.Treeview(lf, columns=cols, show="headings", height=20)
        for c, w in zip(cols, widths):
            self._tbl.heading(c, text=c)
            self._tbl.column(c, width=w, anchor="center")
        xsb = ttk.Scrollbar(lf, orient="horizontal", command=self._tbl.xview)
        vsb = ttk.Scrollbar(lf, orient="vertical",   command=self._tbl.yview)
        self._tbl.configure(xscrollcommand=xsb.set, yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        self._tbl.pack(fill="both", expand=True)
        xsb.pack(fill="x")

    # =========================================================
    # TAB 3 — Sweep Chart
    # =========================================================
    def _build_chart(self):
        P = dict(padx=10, pady=5)
        bf = tk.Frame(self._t_chart, bg=BG); bf.pack(fill="x", **P)
        _btn(bf, "📈  Show Sweep Plot", self._show_sweep_plot, PURPLE).pack(
            side="left", padx=4)
        _btn(bf, "💾  Save PNG", self._save_sweep_png, ACC2).pack(side="left", padx=4)
        self._chart_frame = tk.Frame(self._t_chart, bg=BG)
        self._chart_frame.pack(fill="both", expand=True, **P)
        self._canvas_widget = None

    # =========================================================
    # TAB 4 — Sensitivity
    # =========================================================
    def _build_sens(self):
        P = dict(padx=10, pady=5)
        lf = ttk.LabelFrame(self._t_sens,
                             text="Sensitivity Metrics per Relation", padding=6)
        lf.pack(fill="x", **P)
        cols = ("relation","slope_dir_acc","cv_dir_acc",
                "saturation_p","mean_dir_acc","range_dir_acc")
        widths = [120,110,100,110,110,110]
        self._sens_tbl = ttk.Treeview(lf, columns=cols, show="headings", height=8)
        for c, w in zip(cols, widths):
            self._sens_tbl.heading(c, text=c)
            self._sens_tbl.column(c, width=w, anchor="center")
        self._sens_tbl.pack(fill="x")

        bf2 = tk.Frame(self._t_sens, bg=BG); bf2.pack(fill="x", **P)
        _btn(bf2, "📈  Sensitivity Chart", self._show_sens_plot, PURPLE).pack(
            side="left", padx=4)
        _btn(bf2, "💾  Save PNG", self._save_sens_png, ACC2).pack(side="left", padx=4)
        self._sens_frame = tk.Frame(self._t_sens, bg=BG)
        self._sens_frame.pack(fill="both", expand=True, **P)
        self._sens_canvas = None

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

    # ── Helpers ──────────────────────────────────────────────────────────────
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

    def _browse(self, ext, var):
        p = filedialog.askopenfilename(
            filetypes=[(ext.replace("*.", "").upper(), ext), ("All", "*.*")])
        if p: var.set(p)

    def _load_cfg_json(self):
        p = self._cfg_path.get().strip()
        if not p or not os.path.isfile(p):
            messagebox.showerror("Error", "Select a config.json first."); return
        try:
            cfg = load_json(p)
            if "p_values"       in cfg: self._pv.set(" ".join(str(v) for v in cfg["p_values"]))
            if "asym_relations" in cfg: self._ar.set(" ".join(cfg["asym_relations"]))
            messagebox.showinfo("Loaded", "Config loaded.")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _fmt(self, v, fmt=".4f"):
        if v is None or (isinstance(v, float) and np.isnan(v)): return "—"
        if isinstance(v, bool): return "YES" if v else "no"
        return format(v, fmt)

    # ── Run ──────────────────────────────────────────────────────────────────
    def _start(self):
        ds  = self._ds_path.get().strip()
        emb = self._emb_path.get().strip()
        if not ds or not os.path.isfile(ds):
            messagebox.showerror("Error", "Dataset CSV not found."); return
        if not emb or not os.path.isfile(emb):
            messagebox.showerror("Error", "Embedding .pkl not found."); return
        try:
            p_vals = [float(x) for x in self._pv.get().split()]
        except ValueError:
            messagebox.showerror("Error", "p values must be numbers."); return
        ar_str = self._ar.get().strip()
        asym   = ar_str.split() if ar_str else None

        cfg = {
            "dataset_path"   : ds,
            "embedding_path" : emb,
            "p_values"       : p_vals,
            "asym_relations" : asym,
            "out_dir"        : self._out.get().strip(),
        }
        self._run_btn.config(state="disabled")
        self._pbar.start(12)

        def _run():
            try:
                result = run_pipeline(cfg, self._log)
                self._result = result
                self.after(0, lambda: (
                    self._best_lbl.config(text=f"  best_p = {result['best_p']}"),
                    self._refresh_tbl(),
                    self._refresh_sens(),
                    messagebox.showinfo("Done",
                        f"Sweep complete!\n"
                        f"Best p = {result['best_p']}\n"
                        f"Stable region: {result['stable']}\n"
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

    # ── Refresh table ─────────────────────────────────────────────────────────
    def _refresh_tbl(self):
        if not self._result: return
        res   = self._result
        sweep = res["sweep"]
        fp    = self._fp.get(); fr = self._fr.get()
        p_list   = sorted(sweep.keys())
        rel_list = sorted({rel for pv in sweep for rel in sweep[pv]})

        for var, om, opts in [
            (self._fp, self._fp_om, ["all"] + [str(p) for p in p_list]),
            (self._fr, self._fr_om, ["all"] + rel_list),
        ]:
            m = om["menu"]; m.delete(0, "end")
            for o in opts:
                m.add_command(label=o, command=lambda v=o, sv=var: sv.set(v))

        self._tbl.delete(*self._tbl.get_children())
        for pv in p_list:
            if fp != "all" and str(pv) != fp: continue
            for rel in rel_list:
                if fr != "all" and rel != fr: continue
                r  = sweep[pv].get(rel, {})
                sc = res["scores"].get(pv, float("nan"))
                self._tbl.insert("", "end", values=(
                    pv, rel, r.get("n", 0),
                    self._fmt(r.get("dir_acc"),     ".1%"),
                    self._fmt(r.get("rank_acc"),    ".1%"),
                    self._fmt(r.get("mean_abs_dt"), ".4f"),
                    self._fmt(sc,                   ".4f"),
                    "YES ⚠" if r.get("degenerate") else "no",
                    "✔" if pv in res["stable"] else "",
                    "★" if pv == res["best_p"] else "",
                ))

    # ── Refresh sensitivity table ─────────────────────────────────────────────
    def _refresh_sens(self):
        if not self._result: return
        self._sens_tbl.delete(*self._sens_tbl.get_children())
        for s in self._result["sens"]:
            self._sens_tbl.insert("", "end", values=(
                s["relation"],
                self._fmt(s["slope_dir_acc"]),
                self._fmt(s["cv_dir_acc"]),
                self._fmt(s["saturation_p"],  ".1f"),
                self._fmt(s["mean_dir_acc"],  ".1%"),
                self._fmt(s["range_dir_acc"], ".4f"),
            ))

    # ── Plot helpers ──────────────────────────────────────────────────────────
    def _embed_fig(self, fig, frame, canvas_attr):
        old = getattr(self, canvas_attr)
        if old:
            old.get_tk_widget().destroy()
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        setattr(self, canvas_attr, canvas)

    def _show_sweep_plot(self):
        if not self._result:
            messagebox.showinfo("Info", "Run sweep first."); return
        r = self._result
        try:
            if self._fig_sw: plt.close(self._fig_sw)
            self._fig_sw = plot_sweep(
                r["sweep"], r["scores"], r["dir_accs"], r["abs_dts"],
                r["best_p"], r["stable"], r["asym_rels"], r["all_rels"])
            self._embed_fig(self._fig_sw, self._chart_frame, "_canvas_widget")
        except Exception as e:
            messagebox.showerror("Plot Error", str(e))

    def _show_sens_plot(self):
        if not self._result:
            messagebox.showinfo("Info", "Run sweep first."); return
        try:
            if self._fig_sen: plt.close(self._fig_sen)
            self._fig_sen = plot_sensitivity(self._result["sens"])
            self._embed_fig(self._fig_sen, self._sens_frame, "_sens_canvas")
        except Exception as e:
            messagebox.showerror("Plot Error", str(e))

    def _save_sweep_png(self):
        if not self._fig_sw:
            messagebox.showinfo("Info", "Show plot first."); return
        p = filedialog.asksaveasfilename(defaultextension=".png",
            filetypes=[("PNG", "*.png")])
        if p:
            self._fig_sw.savefig(p, dpi=150, bbox_inches="tight")
            self._log(f"Sweep chart saved → {p}")

    def _save_sens_png(self):
        if not self._fig_sen:
            messagebox.showinfo("Info", "Show sensitivity plot first."); return
        p = filedialog.asksaveasfilename(defaultextension=".png",
            filetypes=[("PNG", "*.png")])
        if p:
            self._fig_sen.savefig(p, dpi=150, bbox_inches="tight")
            self._log(f"Sensitivity chart saved → {p}")


# =============================================================================
if __name__ == "__main__":
    App().mainloop()
