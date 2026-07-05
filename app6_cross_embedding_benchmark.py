"""
APP 6 — Cross-Embedding Benchmark
===================================
Membandingkan embedding dan baseline.

Analisis yang dilakukan:
  • FastText vs GloVe          — dir_acc / rank_acc / mean|Δθ| per relasi
  • real vs randomized         — apakah signal semantik atau geometrik semata
  • robustness comparison      — seberapa konsisten hasil lintas embedding
  • cross-model consistency    — Cohen's κ (pair-level agreement) antar embedding

Input:
  dataset.csv      — word1,word2,relation,is_symmetric  (dari APP 1)
  embedding_A.pkl  — dict word→np.array  (e.g. FastText, dari APP 2)
  embedding_B.pkl  — dict word→np.array  (e.g. GloVe,    dari APP 2)
  random_base.pkl  — dict word→np.array  (Gaussian baseline, dari APP 2; opsional)

Output:
  benchmark_table.csv     — embedding,relation,p,dir_acc,rank_acc,mean_abs_dt,...
  robustness_report.json  — per relation: consistency, delta_dir_acc, verdict
  embedding_comparison.csv — per relation: side-by-side A vs B vs Random

Requirements:
  pip install numpy scipy matplotlib
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading, os, json, csv, pickle, time
import numpy as np
from scipy.stats import mannwhitneyu, spearmanr
from collections import defaultdict
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# =============================================================================
# COLOUR PALETTE
# =============================================================================
BG, BG2 = "#1a1a2e", "#16213e"
FG, FG2 = "#e0e0e0", "#90caf9"
FG3     = "#b0bec5"
ACCENT  = "#0d47a1"
ACC2    = "#1565c0"
GREEN   = "#2e7d32"
ORANGE  = "#e65100"
PURPLE  = "#4a148c"

EMB_COLORS = {
    "A"      : "#1565c0",
    "B"      : "#2e7d32",
    "Random" : "#7b1fa2",
}
REL_COLORS = {
    "hyponymy"  : "#d32f2f",
    "meronymy"  : "#e65100",
    "capital"   : "#2e7d32",
    "sibling"   : "#1565c0",
    "coordinate": "#6a1b9a",
}

RANDOM_SEED = 42
DIR_EPS     = 1e-4
SYM_TAU     = 0.05

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

def save_json(data, path):
    def _fix(o):
        if isinstance(o, float) and np.isnan(o): return None
        if isinstance(o, np.integer): return int(o)
        if isinstance(o, np.floating): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return o
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=_fix)

def save_csv(rows, path, fields):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)

# =============================================================================
# GEOMETRY — inline minimal
# =============================================================================

class _Geo:
    eps = 1e-15
    def fg(self, x, y, p):
        nx = np.linalg.norm(x, ord=p)
        if nx < self.eps: return 0.0
        ax = np.abs(x); m = ax > 0; t = np.zeros_like(x)
        t[m] = (ax[m]**(p-1)) * np.sign(x[m])
        return float((nx**(2-p)) * np.sum(t*y))
    def gsim(self, u, v, p):
        nu = np.linalg.norm(u, ord=p); nv = np.linalg.norm(v, ord=p)
        if nu < self.eps or nv < self.eps: return 0.0
        return float(np.clip(self.fg(v, u, p)/(nu*nv), -1, 1))
    def ga(self, u, v, p):
        return float(np.arccos(np.clip(self.gsim(u, v, p), -1, 1)))
    def dt(self, u, v, p):
        return self.ga(u, v, p) - self.ga(v, u, p)
    def cos(self, u, v):
        nu = np.linalg.norm(u); nv = np.linalg.norm(v)
        if nu < self.eps or nv < self.eps: return 0.0
        return float(np.dot(u, v)/(nu*nv))

GEO = _Geo()

# =============================================================================
# CORE: evaluate one embedding × one relation × one p
# =============================================================================

def eval_emb_rel_p(pairs, get_vec, pv, expected, log=None):
    """
    Compute Δθ for all pairs, return compact stats dict.
    """
    deltas, cosines, angles_uv, angles_vu = [], [], [], []
    skip = 0
    for w1, w2, *_ in pairs:
        eu = get_vec(w1); ev = get_vec(w2)
        if eu is None or ev is None:
            skip += 1; continue
        deltas.append(GEO.dt(eu, ev, pv))
        cosines.append(GEO.cos(eu, ev))
        angles_uv.append(GEO.ga(eu, ev, pv))
        angles_vu.append(GEO.ga(ev, eu, pv))

    n = len(deltas)
    if n == 0:
        return {"n": 0, "skipped": skip,
                "dir_acc": float("nan"), "rank_acc": float("nan"),
                "mean_abs_dt": float("nan"), "mean_dt": float("nan"),
                "std_dt": float("nan"), "cos_mean": float("nan"),
                "degenerate": False, "deltas": np.array([]),
                "cosines": np.array([]), "angles_uv": np.array([]),
                "angles_vu": np.array([])}

    d  = np.array(deltas)
    c  = np.array(cosines)
    is_degen = bool(np.allclose(d, 0.0, atol=1e-9))

    # direction accuracy
    if expected == "asymmetric":
        d_cert  = d[np.abs(d) >= DIR_EPS]
        dir_acc = float(np.mean(d_cert < 0)) if len(d_cert) > 0 else 0.0
    else:
        dir_acc = float(np.mean(np.abs(d) < SYM_TAU))

    # ranking AUC
    if is_degen or n < 4:
        rank_acc = float("nan")
    else:
        rng    = np.random.default_rng(RANDOM_SEED + 1)
        n_rand = min(n, 300)
        idx_u  = rng.choice(n, size=n_rand, replace=True)
        idx_v  = rng.choice(n, size=n_rand, replace=True)
        for i in range(n_rand):
            while idx_v[i] == idx_u[i]:
                idx_v[i] = rng.integers(0, n)
        rand_d = d[idx_v]
        if expected == "asymmetric":
            rs = -d; rr = -rand_d
        else:
            rs = -np.abs(d); rr = -np.abs(rand_d)
        mw, _ = mannwhitneyu(rs, rr, alternative="greater")
        rank_acc = float(mw / (len(rs)*len(rr)))

    return {
        "n"          : n,
        "skipped"    : skip,
        "dir_acc"    : dir_acc,
        "rank_acc"   : rank_acc,
        "mean_abs_dt": float(np.mean(np.abs(d))),
        "mean_dt"    : float(np.mean(d)),
        "std_dt"     : float(np.std(d)),
        "cos_mean"   : float(np.mean(c)),
        "degenerate" : is_degen,
        "deltas"     : d,
        "cosines"    : c,
        "angles_uv"  : np.array(angles_uv),
        "angles_vu"  : np.array(angles_vu),
    }


# =============================================================================
# CROSS-MODEL CONSISTENCY (Cohen's κ approximation)
# =============================================================================

def cross_model_agreement(res_a, res_b, expected):
    """
    Pair-level agreement: both predict correct direction (or both wrong).
    Returns agreement rate and Cohen's κ.
    Only uses pairs present in both (by index, assumes same pair order).
    """
    d_a = res_a["deltas"]; d_b = res_b["deltas"]
    n = min(len(d_a), len(d_b))
    if n == 0:
        return {"agreement": float("nan"), "kappa": float("nan"), "n_common": 0}

    d_a = d_a[:n]; d_b = d_b[:n]

    if expected == "asymmetric":
        pred_a = (d_a < 0).astype(int)
        pred_b = (d_b < 0).astype(int)
    else:
        pred_a = (np.abs(d_a) < SYM_TAU).astype(int)
        pred_b = (np.abs(d_b) < SYM_TAU).astype(int)

    agree = float(np.mean(pred_a == pred_b))

    # Cohen's κ
    p_yes_a = float(np.mean(pred_a)); p_yes_b = float(np.mean(pred_b))
    p_e     = p_yes_a*p_yes_b + (1-p_yes_a)*(1-p_yes_b)
    kappa   = (agree - p_e) / (1 - p_e) if abs(1 - p_e) > 1e-9 else float("nan")

    return {"agreement": agree, "kappa": float(kappa), "n_common": n}


# =============================================================================
# ROBUSTNESS ANALYSIS
# =============================================================================

def robustness_analysis(results_by_emb, rel, expected, log):
    """
    Compare dir_acc across all non-Random embeddings.
    Robustness = 1 − CV(dir_acc).
    Signal hypothesis:
      real dir_acc >> 50%  AND  rand dir_acc ≈ 50%  → SEMANTIC signal
      rand mean|Δθ| > 0                              → GEOMETRIC signal too
    """
    real_accs = []
    rand_acc  = float("nan")
    rand_madt = float("nan")
    real_madts = []

    for emb_name, res in results_by_emb.items():
        r = res.get(rel)
        if not r or r["n"] == 0 or r["degenerate"]:
            continue
        if "Random" in emb_name:
            rand_acc  = r["dir_acc"]
            rand_madt = r["mean_abs_dt"]
        else:
            real_accs.append(r["dir_acc"])
            real_madts.append(r["mean_abs_dt"])

    if len(real_accs) == 0:
        return {"relation": rel, "robustness": float("nan"),
                "delta_dir_acc": float("nan"),
                "rand_dir_acc": rand_acc, "rand_mean_abs_dt": rand_madt,
                "n_embeddings": 0, "verdict": "No data"}

    mean_acc = float(np.mean(real_accs))
    std_acc  = float(np.std(real_accs, ddof=1)) if len(real_accs) > 1 else 0.0
    cv       = std_acc / mean_acc if abs(mean_acc) > 1e-9 else float("nan")
    robustness = 1.0 - cv if not np.isnan(cv) else float("nan")
    delta_dir  = (mean_acc - rand_acc) if not np.isnan(rand_acc) else float("nan")

    # Hypothesis verdict
    if expected == "asymmetric":
        if (not np.isnan(delta_dir) and delta_dir > 0.05 and
                not np.isnan(rand_madt) and rand_madt > 0.005):
            verdict = "Semantic + Geometric signal"
        elif not np.isnan(delta_dir) and delta_dir > 0.05:
            verdict = "Semantic signal only"
        elif not np.isnan(rand_madt) and rand_madt > 0.005:
            verdict = "Geometric signal only"
        else:
            verdict = "No clear signal"
    else:
        verdict = "Symmetric (no direction expected)"

    return {
        "relation"       : rel,
        "n_embeddings"   : len(real_accs),
        "mean_dir_acc"   : mean_acc,
        "std_dir_acc"    : std_acc,
        "cv_dir_acc"     : cv,
        "robustness"     : robustness,
        "delta_dir_acc"  : delta_dir,
        "rand_dir_acc"   : rand_acc,
        "rand_mean_abs_dt": rand_madt,
        "real_mean_abs_dt": float(np.mean(real_madts)) if real_madts else float("nan"),
        "verdict"        : verdict,
    }


# =============================================================================
# FULL PIPELINE
# =============================================================================

def run_pipeline(cfg, log):
    """
    cfg keys:
      dataset_path   str
      emb_a_path     str   — required
      emb_a_name     str   — label for embedding A (e.g. "FastText")
      emb_b_path     str   — optional
      emb_b_name     str
      rand_path      str   — optional  (random baseline .pkl)
      p_value        float — single p to evaluate (best_p from APP 5)
      out_dir        str
    """
    out = cfg["out_dir"]
    os.makedirs(out, exist_ok=True)
    log("=" * 60)
    log("APP 6 — Cross-Embedding Benchmark")
    log("=" * 60)

    # ── 1. Load dataset ──────────────────────────────────────────────────────
    log(f"\n[1] Loading dataset: {cfg['dataset_path']}")
    rows = load_dataset(cfg["dataset_path"])
    by_rel = defaultdict(list)
    for w1, w2, rel, sym in rows:
        by_rel[rel].append((w1, w2, rel, sym))
    all_rels     = sorted(by_rel.keys())
    expected_map = {rel: ("symmetric"
                          if any(s for *_, s in by_rel[rel])
                          else "asymmetric") for rel in all_rels}
    log(f"  Relations: {all_rels}")

    # ── 2. Load embeddings ───────────────────────────────────────────────────
    log(f"\n[2] Loading embeddings …")
    embeddings = {}   # name → get_vec callable

    name_a = cfg.get("emb_a_name") or "Embedding_A"
    log(f"  {name_a}: {cfg['emb_a_path']}")
    va = load_pkl(cfg["emb_a_path"])
    embeddings[name_a] = lambda w, _v=va: _v.get(w.lower())
    log(f"    {len(va):,} words")

    if cfg.get("emb_b_path") and os.path.isfile(cfg["emb_b_path"]):
        name_b = cfg.get("emb_b_name") or "Embedding_B"
        log(f"  {name_b}: {cfg['emb_b_path']}")
        vb = load_pkl(cfg["emb_b_path"])
        embeddings[name_b] = lambda w, _v=vb: _v.get(w.lower())
        log(f"    {len(vb):,} words")
    else:
        name_b = None

    if cfg.get("rand_path") and os.path.isfile(cfg["rand_path"]):
        log(f"  Random baseline: {cfg['rand_path']}")
        vr = load_pkl(cfg["rand_path"])
        rand_name = f"Random [{name_a}]"
        embeddings[rand_name] = lambda w, _v=vr: _v.get(w.lower())
        log(f"    {len(vr):,} words")
    else:
        rand_name = None

    pv  = cfg.get("p_value", 3.0)
    log(f"\n[3] Evaluating at p = {pv}")

    # ── 3. Evaluate all embeddings × all relations ───────────────────────────
    results_by_emb = {}   # emb_name → {rel → stats_dict}

    for emb_name, get_vec in embeddings.items():
        log(f"\n  [{emb_name}]")
        results_by_emb[emb_name] = {}
        for rel, pairs in by_rel.items():
            exp = expected_map[rel]
            res = eval_emb_rel_p(pairs, get_vec, pv, exp, log)
            results_by_emb[emb_name][rel] = res
            log(f"    {rel:<14} n={res['n']:>4}  "
                f"dir_acc={res['dir_acc']:.1%}" if not np.isnan(res["dir_acc"]) else
                f"    {rel:<14} n={res['n']:>4}  dir_acc=N/A"
                + (f"  rank_acc={res['rank_acc']:.1%}"
                   if not np.isnan(res.get("rank_acc", float("nan"))) else ""))

    # ── 4. Cross-model consistency ────────────────────────────────────────────
    log("\n[4] Cross-model consistency …")
    consistency = []
    emb_names = list(results_by_emb.keys())
    non_rand  = [n for n in emb_names if "Random" not in n]
    if len(non_rand) >= 2:
        for rel in all_rels:
            exp  = expected_map[rel]
            ra   = results_by_emb[non_rand[0]].get(rel, {})
            rb   = results_by_emb[non_rand[1]].get(rel, {})
            if ra.get("n", 0) > 0 and rb.get("n", 0) > 0:
                agr = cross_model_agreement(ra, rb, exp)
                log(f"  {rel:<14} agreement={agr['agreement']:.1%}  κ={agr['kappa']:.3f}")
                consistency.append({
                    "relation"  : rel,
                    "emb_a"     : non_rand[0],
                    "emb_b"     : non_rand[1],
                    **agr,
                })

    # ── 5. Robustness analysis ────────────────────────────────────────────────
    log("\n[5] Robustness analysis …")
    robustness = []
    for rel in all_rels:
        exp = expected_map[rel]
        rob = robustness_analysis(results_by_emb, rel, exp, log)
        log(f"  {rel:<14} robustness={rob['robustness']:.3f}  "
            f"Δdir_acc={rob['delta_dir_acc']:+.3f}"
            if not np.isnan(rob['robustness']) else
            f"  {rel:<14} robustness=N/A")
        log(f"    verdict: {rob['verdict']}")
        robustness.append(rob)

    # ── 6. Build output tables ────────────────────────────────────────────────
    log("\n[6] Saving outputs …")

    # benchmark_table.csv
    bench_rows = []
    for emb_name, rel_res in results_by_emb.items():
        for rel, res in rel_res.items():
            bench_rows.append({
                "embedding"   : emb_name,
                "relation"    : rel,
                "p"           : pv,
                "expected"    : expected_map[rel],
                "n"           : res["n"],
                "degenerate"  : res["degenerate"],
                "dir_acc"     : res["dir_acc"],
                "rank_acc"    : res["rank_acc"],
                "mean_abs_dt" : res["mean_abs_dt"],
                "mean_dt"     : res["mean_dt"],
                "std_dt"      : res["std_dt"],
                "cos_mean"    : res["cos_mean"],
            })

    bench_fields = ["embedding","relation","p","expected","n","degenerate",
                    "dir_acc","rank_acc","mean_abs_dt","mean_dt","std_dt","cos_mean"]
    bench_path = os.path.join(out, "benchmark_table.csv")
    save_csv(bench_rows, bench_path, bench_fields)
    log(f"  benchmark_table.csv    → {bench_path}  ({len(bench_rows)} rows)")

    # embedding_comparison.csv  (wide: one row per relation)
    comp_rows = []
    for rel in all_rels:
        row = {"relation": rel, "expected": expected_map[rel]}
        for emb_name in emb_names:
            res = results_by_emb[emb_name].get(rel, {})
            safe = emb_name.replace(" ", "_").replace("[", "").replace("]", "")
            row[f"{safe}_dir_acc"]     = res.get("dir_acc",     float("nan"))
            row[f"{safe}_rank_acc"]    = res.get("rank_acc",    float("nan"))
            row[f"{safe}_mean_abs_dt"] = res.get("mean_abs_dt", float("nan"))
        for c in consistency:
            if c["relation"] == rel:
                row["agreement"]  = c.get("agreement",  float("nan"))
                row["kappa"]      = c.get("kappa",       float("nan"))
        comp_rows.append(row)

    comp_fields = (["relation","expected"]
                   + [f"{n.replace(' ','_').replace('[','').replace(']','')}_{m}"
                      for n in emb_names for m in ("dir_acc","rank_acc","mean_abs_dt")]
                   + ["agreement","kappa"])
    comp_path = os.path.join(out, "embedding_comparison.csv")
    save_csv(comp_rows, comp_path, comp_fields)
    log(f"  embedding_comparison.csv → {comp_path}  ({len(comp_rows)} rows)")

    # robustness_report.json
    rob_path = os.path.join(out, "robustness_report.json")
    save_json({
        "p"                  : pv,
        "embeddings_compared": emb_names,
        "consistency"        : consistency,
        "robustness"         : robustness,
    }, rob_path)
    log(f"  robustness_report.json → {rob_path}")

    log("\n" + "=" * 60)
    log("DONE")
    log("=" * 60)

    return {
        "results_by_emb": results_by_emb,
        "consistency"   : consistency,
        "robustness"    : robustness,
        "emb_names"     : emb_names,
        "all_rels"      : all_rels,
        "expected_map"  : expected_map,
        "bench_rows"    : bench_rows,
        "comp_rows"     : comp_rows,
        "p_value"       : pv,
    }


# =============================================================================
# PLOTTING
# =============================================================================

def _mpl_style():
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "#f8f9fa",
        "axes.edgecolor": "#aaaaaa", "axes.grid": True,
        "grid.color": "#dddddd", "grid.alpha": 0.7,
        "axes.labelsize": 9, "axes.titlesize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 7.5, "legend.facecolor": "white",
        "font.family": "DejaVu Sans",
    })


def plot_benchmark(result):
    """
    4-panel figure:
      1. Dir Acc per relation (grouped bars per embedding)
      2. Rank Acc per relation
      3. Mean|Δθ| per relation
      4. Robustness bar (1−CV)
    """
    _mpl_style()
    emb_names = result["emb_names"]
    all_rels  = result["all_rels"]
    rbe       = result["results_by_emb"]
    rob_list  = result["robustness"]
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5), constrained_layout=True)
    x = np.arange(len(all_rels))
    n_emb = len(emb_names)
    w     = 0.7 / max(n_emb, 1)

    def _emb_color(name):
        if "Random" in name: return EMB_COLORS["Random"]
        idx = emb_names.index(name)
        return EMB_COLORS.get(chr(65+idx), "#78909c")

    def _bars(ax, metric, ylabel, title, pct=True, chance=None):
        for i, emb in enumerate(emb_names):
            vals  = [rbe[emb].get(r, {}).get(metric, float("nan")) for r in all_rels]
            offsets = (i - (n_emb-1)/2) * w
            ys = [v*100 if pct and not np.isnan(v) else (v if not np.isnan(v) else 0)
                  for v in vals]
            bars = ax.bar(x + offsets, ys, w*0.92,
                          label=emb, color=_emb_color(emb),
                          edgecolor="white", lw=0.4, zorder=3,
                          alpha=0.9 if "Random" not in emb else 0.65)
            for bar, v, ov in zip(bars, ys, vals):
                if not np.isnan(ov):
                    ax.text(bar.get_x()+bar.get_width()/2,
                            bar.get_height() + (1 if pct else 0.002),
                            f"{v:.1f}{'%' if pct else ''}" if pct
                            else f"{v:.3f}",
                            ha="center", va="bottom", fontsize=6.5)
        if chance is not None:
            ax.axhline(chance, color="#555", lw=1, linestyle="--",
                       label=f"Chance ({chance}{'%' if pct else ''})")
        ax.set_xticks(x)
        ax.set_xticklabels(all_rels, rotation=18, ha="right", fontsize=8)
        ax.set_ylabel(ylabel); ax.set_title(title)
        ax.legend(fontsize=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    _bars(axes[0], "dir_acc",     "Direction Accuracy (%)",
          "Direction Accuracy\nper Embedding × Relation", pct=True, chance=50)
    _bars(axes[1], "rank_acc",    "Ranking Accuracy / AUC (%)",
          "Ranking Accuracy\nper Embedding × Relation", pct=True, chance=50)
    _bars(axes[2], "mean_abs_dt", "Mean |Δθ|  [rad]",
          "Mean |Δθ|\nper Embedding × Relation", pct=False)

    # Panel 4: Robustness bars
    ax = axes[3]
    rob_map = {r["relation"]: r["robustness"] for r in rob_list}
    rob_vals = [rob_map.get(rel, float("nan")) for rel in all_rels]
    colors   = [REL_COLORS.get(rel, "#78909c") for rel in all_rels]
    bars4 = ax.bar(x, [v if not np.isnan(v) else 0 for v in rob_vals],
                   color=colors, edgecolor="white", lw=0.5, zorder=3)
    for bar, v in zip(bars4, rob_vals):
        if not np.isnan(v):
            ax.text(bar.get_x()+bar.get_width()/2,
                    bar.get_height() + 0.01,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax.axhline(0.8, color="#43a047", lw=1.2, linestyle="--", label="Good (0.8)")
    ax.set_xticks(x)
    ax.set_xticklabels(all_rels, rotation=18, ha="right", fontsize=8)
    ax.set_ylabel("Robustness  (1 − CV)"); ax.set_title("Robustness\n1 − CV(dir_acc)")
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=7)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    pv = result["p_value"]
    embs_str = " vs ".join(result["emb_names"])
    fig.suptitle(f"Cross-Embedding Benchmark  |  p = {pv}  |  {embs_str}",
                 fontsize=10, fontweight="bold")
    return fig


def plot_scatter_comparison(result, rel):
    """
    Scatter A_g(u→v) vs A_g(v→u) for each embedding side-by-side.
    + Δθ distribution overlay.
    """
    _mpl_style()
    emb_names = result["emb_names"]
    rbe       = result["results_by_emb"]
    n_emb     = len(emb_names)
    if n_emb == 0: return None

    fig, axes = plt.subplots(2, n_emb,
                              figsize=(5.5*n_emb, 9),
                              constrained_layout=True)
    if n_emb == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    for col, emb in enumerate(emb_names):
        res = rbe[emb].get(rel)
        if not res or res["n"] == 0:
            for row in range(2):
                axes[row][col].text(0.5, 0.5, "No data",
                                    ha="center", va="center",
                                    transform=axes[row][col].transAxes)
            continue

        uv = res["angles_uv"]; vu = res["angles_vu"]
        d  = res["deltas"];    c  = res["cosines"]
        color = _emb_color_local(emb, emb_names)

        # Row 0: scatter A_g(u→v) vs A_g(v→u)
        ax = axes[0][col]
        sc = ax.scatter(uv, vu, alpha=0.2, s=5, c=c,
                        cmap="RdYlGn", vmin=0, vmax=1, rasterized=True)
        mn = min(uv.min(), vu.min()) - 0.01
        mx = max(uv.max(), vu.max()) + 0.01
        ax.plot([mn,mx], [mn,mx], "--", color="#555", lw=1,
                label="Symmetry (Δθ=0)")
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="cos(u,v)")
        ax.set_xlabel("A_g(u→v) [rad]"); ax.set_ylabel("A_g(v→u) [rad]")
        da = res["dir_acc"]
        ax.set_title(f"{emb}\n{rel}  dir_acc={da:.1%}"
                     if not np.isnan(da) else f"{emb}\n{rel}")
        ax.legend(fontsize=7)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

        # Row 1: Δθ histogram
        ax2 = axes[1][col]
        ax2.hist(d, bins=35, color=color, edgecolor="white",
                 lw=0.3, alpha=0.85)
        ax2.axvline(0, color="#333", lw=1.4, linestyle="--", label="Δθ=0")
        ax2.axvline(float(np.mean(d)), color="#c0392b", lw=1.4,
                    label=f"μ={float(np.mean(d)):+.4f}")
        ax2.set_xlabel("Δθ [rad]"); ax2.set_ylabel("Frequency")
        ax2.set_title(f"Δθ distribution\n{emb} | {rel}")
        ax2.legend(fontsize=7)
        ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)

    fig.suptitle(f"Scatter & Δθ Distribution — {rel}  (p={result['p_value']})",
                 fontsize=10, fontweight="bold")
    return fig


def _emb_color_local(name, emb_names):
    if "Random" in name: return EMB_COLORS["Random"]
    idx = emb_names.index(name)
    return list(EMB_COLORS.values())[min(idx, len(EMB_COLORS)-1)]


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
# GUI
# =============================================================================

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("APP 6 — Cross-Embedding Benchmark")
        self.geometry("960x820")
        self.configure(bg=BG)
        self.resizable(True, True)
        self._result   = None
        self._fig_bm   = None
        self._fig_sc   = None
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
            ("TProgressbar",     {"troughcolor": "#0f3460", "background": "#42a5f5"}),
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
            (" ⚙  Config & Run ", "_t_cfg"),
            (" 📊  Benchmark Table ", "_t_tbl"),
            (" 📈  Benchmark Chart ", "_t_chart"),
            (" 🔬  Robustness ", "_t_rob"),
            (" 🔍  Scatter Detail ", "_t_sc"),
            (" 📝  Log ", "_t_log"),
        ]:
            f = ttk.Frame(nb); setattr(self, attr, f); nb.add(f, text=label)
        self._build_cfg()
        self._build_tbl()
        self._build_chart()
        self._build_rob()
        self._build_sc()
        self._build_log()

    # =========================================================
    # TAB 1 — Config & Run
    # =========================================================
    def _build_cfg(self):
        P = dict(padx=10, pady=5)

        lf1 = ttk.LabelFrame(self._t_cfg, text="1.  Input Files", padding=8)
        lf1.pack(fill="x", **P)

        inputs = [
            ("Dataset  (.csv)",          "_ds",   "*.csv"),
            ("Embedding A  (.pkl)",      "_emb_a","*.pkl"),
            ("Embedding B  (.pkl, opt)", "_emb_b","*.pkl"),
            ("Random Baseline  (.pkl, opt)", "_rand","*.pkl"),
        ]
        for label, attr, ext in inputs:
            r = tk.Frame(lf1, bg=BG); r.pack(fill="x", pady=2)
            _lbl(r, f"{label}:").pack(side="left")
            var = tk.StringVar(); setattr(self, attr+"_path", var)
            _entry(r, var, 48).pack(side="left", padx=6)
            _btn(r, "Browse…",
                 lambda e=ext, v=var: self._browse(e, v), ACC2).pack(side="left")

        lf_names = ttk.LabelFrame(self._t_cfg, text="   Embedding Labels", padding=8)
        lf_names.pack(fill="x", **P)
        rn = tk.Frame(lf_names, bg=BG); rn.pack(fill="x")
        _lbl(rn, "Name A:").pack(side="left")
        self._name_a = tk.StringVar(value="FastText")
        _entry(rn, self._name_a, 18).pack(side="left", padx=6)
        _lbl(rn, "  Name B:").pack(side="left")
        self._name_b = tk.StringVar(value="GloVe")
        _entry(rn, self._name_b, 18).pack(side="left", padx=6)

        lf2 = ttk.LabelFrame(self._t_cfg, text="2.  Parameters", padding=8)
        lf2.pack(fill="x", **P)
        rp = tk.Frame(lf2, bg=BG); rp.pack(fill="x", pady=2)
        _lbl(rp, "p value (single, from APP 5 best_p):").pack(side="left")
        self._pv = tk.StringVar(value="3")
        _entry(rp, self._pv, 8).pack(side="left", padx=6)

        ro = tk.Frame(lf2, bg=BG); ro.pack(fill="x", pady=2)
        _lbl(ro, "Output directory:").pack(side="left")
        self._out = tk.StringVar(
            value=os.path.join(os.path.expanduser("~"), "benchmark_output"))
        _entry(ro, self._out, 44).pack(side="left", padx=6)
        _btn(ro, "Browse…",
             lambda: (d := filedialog.askdirectory()) and self._out.set(d),
             ACC2).pack(side="left")

        lf3 = ttk.LabelFrame(self._t_cfg, text="3.  Run", padding=8)
        lf3.pack(fill="x", **P)
        bf = tk.Frame(lf3, bg=BG); bf.pack(fill="x")
        self._run_btn = _btn(bf, "▶  Run Benchmark", self._start,
                             PURPLE, pady=5, padx=20)
        self._run_btn.pack(side="left", padx=4)
        self._pbar = ttk.Progressbar(lf3, mode="indeterminate", length=380)
        self._pbar.pack(fill="x", pady=(6, 0))

    # =========================================================
    # TAB 2 — Benchmark Table
    # =========================================================
    def _build_tbl(self):
        P = dict(padx=10, pady=5)
        ctrl = tk.Frame(self._t_tbl, bg=BG); ctrl.pack(fill="x", **P)
        _lbl(ctrl, "Filter embedding:").pack(side="left")
        self._fe = tk.StringVar(value="all")
        self._fe_om = tk.OptionMenu(ctrl, self._fe, "all")
        self._fe_om.config(bg=BG2, fg=FG, activebackground=ACCENT,
                           highlightbackground=BG, relief="flat",
                           font=("Consolas", 9), width=20)
        self._fe_om["menu"].config(bg=BG2, fg=FG)
        self._fe_om.pack(side="left", padx=4)
        _lbl(ctrl, "  Relation:").pack(side="left")
        self._fr = tk.StringVar(value="all")
        self._fr_om = tk.OptionMenu(ctrl, self._fr, "all")
        self._fr_om.config(bg=BG2, fg=FG, activebackground=ACCENT,
                           highlightbackground=BG, relief="flat",
                           font=("Consolas", 9), width=14)
        self._fr_om["menu"].config(bg=BG2, fg=FG)
        self._fr_om.pack(side="left", padx=4)
        _btn(ctrl, "🔄 Refresh", self._refresh_tbl, ACC2).pack(side="left", padx=6)

        lf = ttk.LabelFrame(self._t_tbl, text="Benchmark Results", padding=6)
        lf.pack(fill="both", expand=True, **P)
        cols = ("embedding","relation","p","n","dir_acc","rank_acc",
                "mean|Δθ|","cos_mean","degenerate")
        widths = [160,110,45,55,80,80,90,80,80]
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
    # TAB 3 — Benchmark Chart
    # =========================================================
    def _build_chart(self):
        P = dict(padx=10, pady=5)
        bf = tk.Frame(self._t_chart, bg=BG); bf.pack(fill="x", **P)
        _btn(bf, "📈  Show Chart", self._show_bm_plot, PURPLE).pack(side="left", padx=4)
        _btn(bf, "💾  Save PNG",   self._save_bm_png,  ACC2).pack(side="left", padx=4)
        self._chart_frame  = tk.Frame(self._t_chart, bg=BG)
        self._chart_frame.pack(fill="both", expand=True, **P)
        self._canvas_bm = None

    # =========================================================
    # TAB 4 — Robustness
    # =========================================================
    def _build_rob(self):
        P = dict(padx=10, pady=5)
        lf = ttk.LabelFrame(self._t_rob, text="Robustness Report", padding=6)
        lf.pack(fill="x", **P)
        cols = ("relation","n_emb","mean_dir_acc","robustness",
                "Δdir_acc","rand_dir_acc","verdict")
        widths = [110,60,100,90,90,100,220]
        self._rob_tbl = ttk.Treeview(lf, columns=cols, show="headings", height=7)
        for c, w in zip(cols, widths):
            self._rob_tbl.heading(c, text=c)
            self._rob_tbl.column(c, width=w, anchor="center")
        self._rob_tbl.pack(fill="x")

        lf2 = ttk.LabelFrame(self._t_rob, text="Cross-Model Consistency (Cohen's κ)", padding=6)
        lf2.pack(fill="x", **P)
        cols2 = ("relation","emb_a","emb_b","n_common","agreement","kappa")
        widths2 = [110,160,160,80,90,80]
        self._cons_tbl = ttk.Treeview(lf2, columns=cols2, show="headings", height=6)
        for c, w in zip(cols2, widths2):
            self._cons_tbl.heading(c, text=c)
            self._cons_tbl.column(c, width=w, anchor="center")
        self._cons_tbl.pack(fill="x")

    # =========================================================
    # TAB 5 — Scatter Detail
    # =========================================================
    def _build_sc(self):
        P = dict(padx=10, pady=5)
        ctrl = tk.Frame(self._t_sc, bg=BG); ctrl.pack(fill="x", **P)
        _lbl(ctrl, "Relation:").pack(side="left")
        self._sc_rel = tk.StringVar(value="hyponymy")
        self._sc_rel_om = tk.OptionMenu(ctrl, self._sc_rel, "hyponymy")
        self._sc_rel_om.config(bg=BG2, fg=FG, activebackground=ACCENT,
                               highlightbackground=BG, relief="flat",
                               font=("Consolas", 9), width=14)
        self._sc_rel_om["menu"].config(bg=BG2, fg=FG)
        self._sc_rel_om.pack(side="left", padx=6)
        _btn(ctrl, "📈  Show Scatter", self._show_sc_plot, PURPLE).pack(side="left", padx=4)
        _btn(ctrl, "💾  Save PNG",     self._save_sc_png,  ACC2).pack(side="left", padx=4)
        self._sc_frame  = tk.Frame(self._t_sc, bg=BG)
        self._sc_frame.pack(fill="both", expand=True, **P)
        self._canvas_sc = None

    # =========================================================
    # TAB 6 — Log
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

    def _fmt(self, v, fmt=".4f"):
        if v is None or (isinstance(v, float) and np.isnan(v)): return "—"
        if isinstance(v, bool): return "YES" if v else "no"
        return format(v, fmt)

    # ── Run ──────────────────────────────────────────────────────────────────
    def _start(self):
        ds  = self._ds_path.get().strip()
        ea  = self._emb_a_path.get().strip()
        if not ds or not os.path.isfile(ds):
            messagebox.showerror("Error", "Dataset CSV not found."); return
        if not ea or not os.path.isfile(ea):
            messagebox.showerror("Error", "Embedding A .pkl not found."); return
        try:
            pv = float(self._pv.get())
        except ValueError:
            messagebox.showerror("Error", "p value must be a number."); return

        cfg = {
            "dataset_path": ds,
            "emb_a_path"  : ea,
            "emb_a_name"  : self._name_a.get().strip() or "Embedding_A",
            "emb_b_path"  : self._emb_b_path.get().strip() or None,
            "emb_b_name"  : self._name_b.get().strip() or "Embedding_B",
            "rand_path"   : self._rand_path.get().strip() or None,
            "p_value"     : pv,
            "out_dir"     : self._out.get().strip(),
        }
        self._run_btn.config(state="disabled")
        self._pbar.start(12)

        def _run():
            try:
                result = run_pipeline(cfg, self._log)
                self._result = result
                self.after(0, lambda: (
                    self._refresh_tbl(),
                    self._refresh_rob(),
                    self._update_sc_dropdown(),
                    messagebox.showinfo("Done",
                        f"Benchmark complete!\n"
                        f"Embeddings: {result['emb_names']}\n"
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
    def _refresh_tbl(self):
        if not self._result: return
        rows = self._result["bench_rows"]
        fe   = self._fe.get(); fr = self._fr.get()
        emb_list = sorted({r["embedding"] for r in rows})
        rel_list = sorted({r["relation"]  for r in rows})

        for var, om, opts in [
            (self._fe, self._fe_om, ["all"] + emb_list),
            (self._fr, self._fr_om, ["all"] + rel_list),
        ]:
            m = om["menu"]; m.delete(0, "end")
            for o in opts:
                m.add_command(label=o, command=lambda v=o, sv=var: sv.set(v))

        self._tbl.delete(*self._tbl.get_children())
        for r in rows:
            if fe != "all" and r["embedding"] != fe: continue
            if fr != "all" and r["relation"]  != fr: continue
            self._tbl.insert("", "end", values=(
                r["embedding"], r["relation"], r["p"], r["n"],
                self._fmt(r["dir_acc"],     ".1%"),
                self._fmt(r["rank_acc"],    ".1%"),
                self._fmt(r["mean_abs_dt"], ".4f"),
                self._fmt(r["cos_mean"],    ".4f"),
                "YES ⚠" if r["degenerate"] else "no",
            ))

    def _refresh_rob(self):
        if not self._result: return
        self._rob_tbl.delete(*self._rob_tbl.get_children())
        for r in self._result["robustness"]:
            self._rob_tbl.insert("", "end", values=(
                r["relation"], r["n_embeddings"],
                self._fmt(r.get("mean_dir_acc"),  ".1%"),
                self._fmt(r.get("robustness"),    ".3f"),
                self._fmt(r.get("delta_dir_acc"), "+.3f"),
                self._fmt(r.get("rand_dir_acc"),  ".1%"),
                r["verdict"],
            ))
        self._cons_tbl.delete(*self._cons_tbl.get_children())
        for c in self._result["consistency"]:
            self._cons_tbl.insert("", "end", values=(
                c["relation"], c["emb_a"], c["emb_b"],
                c.get("n_common", 0),
                self._fmt(c.get("agreement"), ".1%"),
                self._fmt(c.get("kappa"),     ".3f"),
            ))

    def _update_sc_dropdown(self):
        if not self._result: return
        rels = self._result["all_rels"]
        m = self._sc_rel_om["menu"]; m.delete(0, "end")
        for r in rels:
            m.add_command(label=r,
                          command=lambda v=r: self._sc_rel.set(v))
        if rels: self._sc_rel.set(rels[0])

    # ── Charts ────────────────────────────────────────────────────────────────
    def _embed_fig(self, fig, frame, canvas_attr):
        old = getattr(self, canvas_attr)
        if old:
            old.get_tk_widget().destroy()
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        setattr(self, canvas_attr, canvas)

    def _show_bm_plot(self):
        if not self._result:
            messagebox.showinfo("Info", "Run benchmark first."); return
        try:
            if self._fig_bm: plt.close(self._fig_bm)
            self._fig_bm = plot_benchmark(self._result)
            self._embed_fig(self._fig_bm, self._chart_frame, "_canvas_bm")
        except Exception as e:
            messagebox.showerror("Plot Error", str(e))

    def _save_bm_png(self):
        if not self._fig_bm:
            messagebox.showinfo("Info", "Show chart first."); return
        p = filedialog.asksaveasfilename(defaultextension=".png",
            filetypes=[("PNG", "*.png")])
        if p:
            self._fig_bm.savefig(p, dpi=150, bbox_inches="tight")
            self._log(f"Benchmark chart saved → {p}")

    def _show_sc_plot(self):
        if not self._result:
            messagebox.showinfo("Info", "Run benchmark first."); return
        rel = self._sc_rel.get()
        try:
            if self._fig_sc: plt.close(self._fig_sc)
            self._fig_sc = plot_scatter_comparison(self._result, rel)
            if self._fig_sc:
                self._embed_fig(self._fig_sc, self._sc_frame, "_canvas_sc")
        except Exception as e:
            messagebox.showerror("Plot Error", str(e))

    def _save_sc_png(self):
        if not self._fig_sc:
            messagebox.showinfo("Info", "Show scatter first."); return
        p = filedialog.asksaveasfilename(defaultextension=".png",
            filetypes=[("PNG", "*.png")])
        if p:
            self._fig_sc.savefig(p, dpi=150, bbox_inches="tight")
            self._log(f"Scatter chart saved → {p}")


# =============================================================================
if __name__ == "__main__":
    App().mainloop()
