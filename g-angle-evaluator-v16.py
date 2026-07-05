"""
WordNet G-Angle vs Cosine Similarity — Comprehensive Comparison
===============================================================================
Comparing two metrics:
  • G-Angle                        → captures asymmetry, hierarchy, entailment
  • Cosine Similarity              → standard symmetric metric

Datasets auto-generated from WordNet (+ built-in):
  1. Hyponymy         — hierarchical is-a relation (dog → animal)         [asymmetric]
  2. Meronymy         — part-of relation (wheel → car)                     [asymmetric]
  3. Capital–Country  — capital→country relation (built-in, 195+ pairs)   [asymmetric]
  4. Sibling-Sym      — co-hyponym (cat vs dog)                            [symmetric]
  5. Coordinate-Sym   — parallel symmetric pairs (fruits, professions, etc.)[symmetric]

Metrics per relation:
  • Parameter Sweep p ∈ {2,3,4,5,10} — automatically compares all p values
  • Direction Accuracy                — proportion of Δθ with correct sign (< 0 for asymmetric)
  • Ranking Accuracy                  — ability to distinguish real vs random pairs
  • Mean |Δθ| ± Std                   — strength of asymmetry signal
  • Cohen's d + 95% CI               — effect size & confidence interval [NEW v8]
  • One-sample t-test (H₀: E[Δθ]=0)  — statistical significance
  • Permutation Test (10 000 iter)    — random baseline distribution
  • Symmetry Score (NEW v14)          — |Δθ| distribution test for symmetric pairs
  • Error Analysis                    — pairs with incorrect direction prediction
  • Cosine similarity mean            — as symmetric control
  • Fixed random seed                 — experiment reproducibility

NEW (v8–v10):
  ① GloVe loader (.txt format)       — cross-embedding validation control
  ② Randomized Embedding Control     — Gaussian vectors N(μ,σ²) from real embedding stats
     Hypothesis: asymmetry persists (geometric) but direction accuracy → 50% (semantic=0)
  ③ Cross-Embedding Comparison tab   — side-by-side FastText vs GloVe vs Random
  ④ p=10 added to sweep              — tests saturation of asymmetry with high p
  ⑤ Effect size (Cohen's d) + 95% CI — stronger statistical reporting
  ⑥ p-value Bonferroni correction    — multiple-relation correction in summary

BUGFIX (v11):
  ⑦ Ranking accuracy — conceptual fix for asymmetric relations.
     Previously used |Δθ| (magnitude), which rewarded large asymmetry regardless
     of direction.  Now uses signed score (−Δθ): a strongly negative Δθ (correct
     direction) scores high, a positive Δθ (wrong direction) scores low.
     This makes ranking_acc consistent with direction_acc: both reward Δθ < 0.

NEW (v13):
  ⑬ Benjamini–Hochberg FDR correction — applied across all p-values collected
     per run (t-test, perm-orient, perm-pair, per relation × per p-value in
     sweep).  BH-adjusted q-values stored alongside raw p; verdict uses q<0.05.
  ⑭ Deep error analysis — each error dict now carries cosine, norm_ratio
     (‖u‖/‖v‖), and frequency_proxy (len(word) as OOV surrogate); top-10
     false-positives and false-negatives stored in results["top_fp"] /
     results["top_fn"] and shown in the Error Analysis tab.
  ⑮ Distribution shape metrics — skewness, excess kurtosis, entropy (Δθ
     histogram), and bimodality coefficient stored in stats for every relation.
  ⑯ Formal cross-embedding comparison table — after "Run All" for FastText AND
     GloVe (either loaded), a Model × Relation matrix of dir_acc values is
     displayed in the Cross-Embedding tab with a dedicated "All-Relations Cross"
     button that runs all relations for every loaded embedding in one click.

BUGFIX (v13-fix):
  ⑱ Ranking accuracy — replaced per-pair comparison with AUC (Mann-Whitney U).
     Root cause: idx_rand was a permutation of the same pool as idx_real, causing
     100% index overlap so rand score always used the same u as real score.
     Per-pair comparison (real[i] > rand[i]) is not equivalent to dir_acc and
     can yield rank_acc << dir_acc when mismatch pairs also produce negative Δθ.
     New: rank_acc = AUC = P(score_real > score_rand) via Mann-Whitney, where
     real and random scores are drawn from INDEPENDENT populations.
     For asymmetric: score = -Δθ.  For symmetric: score = -|Δθ|.
     AUC=0.5 is chance, which aligns with dir_acc=50%. Bootstrap CI preserved.

BUGFIX (v12-fix):
  ⑰ p=2 early-exit — when is_degenerate is detected, permutation tests A & B
     and the ranking accuracy loop are now SKIPPED entirely (previously they
     still ran, wasting CPU and producing perm_p=1.0 / rank_acc=0% which were
     meaningless and misleading).  All affected fields are set to float("nan")
     and display as "—" / "N/A" throughout the UI and log output.
     perm_p_orient / perm_p_pair in _refresh_table now use _fmt_q() (NaN-safe)
     instead of :.4f (which would raise ValueError on NaN).
     _log_conclusions uses a local _fmt_p() helper for the same reason.

NEW (v12):
  ⑧ Permutation Test B (pair shuffle) — stronger null hypothesis.
     Reshuffle which u is paired with which v across the dataset, keeping
     vectors fixed: dog→animal + car→vehicle becomes dog→vehicle + car→animal.
     Tests whether the real pairing itself carries the directional signal.
     Primary perm_p = max(p_orient, p_pair) — most conservative result reported.
  ⑨ Swap test removed — Δθ(u,v) = -Δθ(v,u) is a mathematical identity of
     g-angle, so swap consistency is always 100% and carries no empirical info.
  ⑩ p=2 degeneracy handling — for Euclidean norm g-angle reduces to standard
     cosine angle (symmetric), so all Δθ ≡ 0.  Direction/ranking now display
     "N/A (symmetric geometry)" instead of a misleading 0%.
  ⑪ Direction metric epsilon margin — dir_acc now counts d < −eps (eps=1e-4)
     and reports "uncertain" pair count where |Δθ| < eps.  Avoids noise-flip
     artefacts at the decision boundary.
  ⑫ Bootstrap CI (B=2000) for dir_acc, rank_acc, and mean Δθ — all three key
     metrics now report 95% CI via non-parametric percentile bootstrap, giving
     publication-quality uncertainty estimates.

NEW (v14):
  ⑲ Swap Test tab fully replaced — tab UI sekarang menampilkan "Symmetry Score"
     (distribusi |Δθ| untuk relasi symmetric vs asymmetric) menggantikan swap
     consistency yang selalu 100% karena Δθ(u,v) = −Δθ(v,u) adalah identitas
     matematis, bukan hasil empiris.
  ⑳ dir_acc untuk symmetric diperbaiki — sebelumnya menggunakan median split
     (selalu ≈ 50% by construction karena membelah distribusi tepat di tengah).
     Sekarang menggunakan absolute threshold: dir_acc = P(|Δθ| < τ) di mana
     τ = 0.05 radian (~2.9°). Nilai ini mencerminkan seberapa "benar-benar simetris"
     pasangan tersebut, bukan sekadar di bawah median.
  ㉑ Permutation Test untuk symmetric — tiga perbaikan sekaligus (v14):
     (a) Perm A (orientation flip) DIHAPUS untuk symmetric. Δθ(u,v) = −Δθ(v,u)
         berarti flip hanya membalik tanda Δθ tanpa mengubah |Δθ|. Setiap statistik
         berbasis |Δθ| invariant terhadap flip → distribusi null σ = 0 by construction.
         perm_p_orient = NaN untuk symmetric (ditampilkan sebagai "—" di tabel).
     (b) Statistik Perm B diganti dari binary P(|Δθ|<τ) ke continuous mean(|Δθ|).
         Statistik biner menyebabkan discretization sehingga distribusi null tetap
         runtuh. Mean(|Δθ|) kontinu dan sensitif terhadap perbedaan antara null dan
         observed.
     (c) Null Perm B untuk symmetric: cross-pool (bukan intra-pool).
         u dari indeks genap dipasangkan dengan v dari indeks ganjil — pasangan yang
         tidak berelasi satu sama lain, memberikan null mean|Δθ| > observed mean|Δθ|.
         p = P(null_mean|Δθ| ≤ obs_mean|Δθ|): kecil → pasangan real lebih simetris
         dari null; bermakna sebagai uji apakah pairing asli membawa signal symmetry.
  ㉒ plot_permutation_test diperbarui — untuk symmetric, sumbu x menampilkan
     mean|Δθ| per iterasi (bukan dir_acc %), dan cross-pool null ditampilkan
     dengan garis observed dan null-mean yang jelas.

NEW (v16):
  ㉕ Analytical best-p selection — parameter sweep kini BENAR-BENAR dipakai
     secara analitis, bukan sekadar menghasilkan tabel.
     Fungsi select_best_p() menghitung composite score per p:
       score(p) = mean_dir_acc_asym + 0.3 * mean_|Δθ|_norm
     kemudian mengidentifikasi "stable region" (plateau ≥ peak − 2%)
     dan memilih p TERKECIL di plateau (prinsip parsimoni).
     Hasilnya:
       • p=2 (degenerate) otomatis di-skip
       • p=1.5 (weak) mendapat skor rendah
       • p=3–5 (strongest) membentuk stable region → best_p dipilih otomatis
       • p=10 (mulai turun) keluar dari plateau jika score turun > 2%
     Plot sweep kini 4 panel: Dir Acc, Rank Acc, Mean|Δθ|, dan Composite Score.
     Stable region diarsir hijau; best_p ditandai garis merah dash-dot di semua panel.
     self.results di-set ke sweep[best_p] (bukan hardcoded p=3).

  ㉓ Randomized Embedding Baseline sekarang hadir BERDAMPINGAN di evaluasi inti.
     Sebelumnya baseline random hanya tersedia di Cross-Embedding tab (opsional,
     harus klik manual). Sekarang jika FastText sudah dimuat, baseline random
     otomatis dijalankan saat "Run All" dan hasilnya langsung muncul di:
       • plot_comparison_bars  — kolom ketiga "Random (N(μ,σ²))" untuk setiap relasi
       • plot_ranking_accuracy — bar keempat "Random Rank" per relasi
       • _log_conclusions      — baris perbandingan real vs random per relasi
     Hipotesis yang diuji:
       • |Δθ| ≢ 0 pada random → asymmetry bersumber dari geometri g-angle
       • dir_acc random ≈ 50% → informasi semantik hilang, signal real adalah semantik
     Dengan demikian pemisahan geometric artifact vs semantic signal menjadi
     eksplisit di output utama, bukan hanya di tab terpisah.
  ㉔ make_randomized_get_vec kini dipanggil sekali di _run_all menggunakan
     semua pasangan yang tersedia (gabungan semua relasi), memberikan kalibrasi
     statistik yang lebih representatif vs sebelumnya yang hanya mengambil
     500 pasangan dari satu relasi saja.

Requirements:
    pip install nltk numpy scipy matplotlib
    python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"

GloVe vectors: https://nlp.stanford.edu/projects/glove/  (glove.6B.zip → glove.6B.300d.txt)
FastText vectors: https://fasttext.cc/docs/en/english-vectors.html
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import os
import random
import numpy as np
from scipy.stats import ttest_1samp, mannwhitneyu, spearmanr, pearsonr, norm as sp_norm
from collections import defaultdict
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import csv

# ─── WordNet (lazy import) ───────────────────────────────────────────────────
_WN = None
def get_wn():
    global _WN
    if _WN is None:
        try:
            from nltk.corpus import wordnet as wn
            list(wn.all_synsets())   # trigger load
            _WN = wn
        except Exception:
            import nltk
            nltk.download("wordnet", quiet=True)
            nltk.download("omw-1.4",  quiet=True)
            from nltk.corpus import wordnet as wn
            _WN = wn
    return _WN

# ===========================================================================
# GEOMETRY ENGINE — Gunawan et al. (2018)
# ===========================================================================

class GeometryEngine:
    def __init__(self, eps=1e-15):
        self.eps = eps

    def functional_g(self, x, y, p=2.0):
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

    def g_similarity(self, u, v, p=2.0):
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        norm_u = np.linalg.norm(u, ord=p)
        norm_v = np.linalg.norm(v, ord=p)
        if norm_u < self.eps or norm_v < self.eps:
            return 0.0
        g_vu = self.functional_g(v, u, p)
        return float(np.clip(g_vu / (norm_u * norm_v), -1.0, 1.0))

    def g_angle(self, u, v, p=2.0):
        return float(np.arccos(np.clip(self.g_similarity(u, v, p), -1, 1)))

    def delta_theta(self, u, v, p=2.0):
        """Δθ = A_g(u→v) − A_g(v→u)"""
        return self.g_angle(u, v, p) - self.g_angle(v, u, p)

    def cosine_sim(self, u, v):
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        nu = np.linalg.norm(u); nv = np.linalg.norm(v)
        if nu < self.eps or nv < self.eps:
            return 0.0
        return float(np.dot(u, v) / (nu * nv))

    def cosine_delta(self, u, v):
        """Δcos = cos(u,v) − cos(v,u)  → always 0 because cosine is symmetric"""
        return 0.0   # cosine is symmetric by definition

ENGINE = GeometryEngine()

# ─── Benjamini–Hochberg FDR correction ──────────────────────────────────────
def bh_correct(p_values):
    """
    Benjamini–Hochberg (1995) step-up FDR correction.
    Returns an array of q-values (adjusted p-values) in the same order as input.
    A test is considered significant when q < 0.05.
    """
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    if n == 0:
        return p.copy()
    order   = np.argsort(p)
    ranks   = np.empty(n, dtype=float)
    ranks[order] = np.arange(1, n + 1)
    q = p * n / ranks
    # Enforce monotonicity: q[i] ≤ q[i+1] (working from the right)
    q_sorted = q[order]
    for i in range(n - 2, -1, -1):
        q_sorted[i] = min(q_sorted[i], q_sorted[i + 1])
    q[order] = np.clip(q_sorted, 0.0, 1.0)
    return q
RANDOM_SEED = 42

# Capital-Country — comprehensive world list (195+ pairs, auto-deduplicated)
CAPITAL_COUNTRY = [
    # Europe
    ("tirana","albania"),("vienna","austria"),("baku","azerbaijan"),
    ("minsk","belarus"),("brussels","belgium"),("sarajevo","bosnia"),
    ("sofia","bulgaria"),("zagreb","croatia"),("nicosia","cyprus"),
    ("prague","czechia"),("copenhagen","denmark"),("tallinn","estonia"),
    ("helsinki","finland"),("paris","france"),("berlin","germany"),
    ("athens","greece"),("budapest","hungary"),("reykjavik","iceland"),
    ("dublin","ireland"),("rome","italy"),("riga","latvia"),
    ("vilnius","lithuania"),("luxembourg","luxembourg"),("valletta","malta"),
    ("chisinau","moldova"),("amsterdam","netherlands"),("oslo","norway"),
    ("warsaw","poland"),("lisbon","portugal"),("bucharest","romania"),
    ("moscow","russia"),("belgrade","serbia"),("bratislava","slovakia"),
    ("ljubljana","slovenia"),("madrid","spain"),("stockholm","sweden"),
    ("bern","switzerland"),("ankara","turkey"),("kyiv","ukraine"),
    ("london","uk"),("skopje","macedonia"),("podgorica","montenegro"),
    # Americas
    ("nassau","bahamas"),("bridgetown","barbados"),("belmopan","belize"),
    ("ottawa","canada"),("bogota","colombia"),("san jose","costa rica"),
    ("havana","cuba"),("quito","ecuador"),("guatemala","guatemala"),
    ("tegucigalpa","honduras"),("kingston","jamaica"),("mexico","mexico"),
    ("managua","nicaragua"),("panama","panama"),("asuncion","paraguay"),
    ("lima","peru"),("paramaribo","suriname"),("montevideo","uruguay"),
    ("washington","usa"),("caracas","venezuela"),("brasilia","brazil"),
    ("buenos aires","argentina"),("la paz","bolivia"),("santiago","chile"),
    ("santo domingo","dominican"),
    # Asia
    ("kabul","afghanistan"),("yerevan","armenia"),("manama","bahrain"),
    ("dhaka","bangladesh"),("thimphu","bhutan"),("beijing","china"),
    ("tbilisi","georgia"),("new delhi","india"),("jakarta","indonesia"),
    ("tehran","iran"),("baghdad","iraq"),("jerusalem","israel"),
    ("tokyo","japan"),("amman","jordan"),("nur sultan","kazakhstan"),
    ("kuwait","kuwait"),("bishkek","kyrgyzstan"),("vientiane","laos"),
    ("beirut","lebanon"),("kuala lumpur","malaysia"),("male","maldives"),
    ("ulaanbaatar","mongolia"),("naypyidaw","myanmar"),("kathmandu","nepal"),
    ("islamabad","pakistan"),("manila","philippines"),("seoul","korea"),
    ("colombo","sri lanka"),("damascus","syria"),("taipei","taiwan"),
    ("dushanbe","tajikistan"),("bangkok","thailand"),("tashkent","uzbekistan"),
    ("hanoi","vietnam"),("sanaa","yemen"),("riyadh","saudi arabia"),
    ("singapore","singapore"),("abu dhabi","uae"),("muscat","oman"),
    ("doha","qatar"),("amman","jordan"),("nicosia","cyprus"),
    # Africa
    ("algiers","algeria"),("luanda","angola"),("porto-novo","benin"),
    ("gaborone","botswana"),("ouagadougou","burkina faso"),
    ("bujumbura","burundi"),("yaounde","cameroon"),("praia","cape verde"),
    ("bangui","central africa"),("ndjamena","chad"),("moroni","comoros"),
    ("kinshasa","congo"),("brazzaville","republic congo"),
    ("djibouti","djibouti"),("cairo","egypt"),("asmara","eritrea"),
    ("addis ababa","ethiopia"),("libreville","gabon"),("banjul","gambia"),
    ("accra","ghana"),("conakry","guinea"),("nairobi","kenya"),
    ("maseru","lesotho"),("monrovia","liberia"),("tripoli","libya"),
    ("antananarivo","madagascar"),("lilongwe","malawi"),("bamako","mali"),
    ("nouakchott","mauritania"),("port louis","mauritius"),("rabat","morocco"),
    ("maputo","mozambique"),("windhoek","namibia"),("niamey","niger"),
    ("abuja","nigeria"),("kigali","rwanda"),("dakar","senegal"),
    ("freetown","sierra leone"),("mogadishu","somalia"),("pretoria","south africa"),
    ("juba","south sudan"),("khartoum","sudan"),("mbabane","eswatini"),
    ("dodoma","tanzania"),("lome","togo"),("tunis","tunisia"),
    ("kampala","uganda"),("lusaka","zambia"),("harare","zimbabwe"),
    # Oceania
    ("canberra","australia"),("suva","fiji"),("wellington","new zealand"),
    ("port moresby","papua"),("apia","samoa"),("honiara","solomon"),
    ("nuku alofa","tonga"),("funafuti","tuvalu"),("port vila","vanuatu"),
    ("yaren","nauru"),("majuro","marshall"),("palikir","micronesia"),
]
# Deduplicate preserving order
_seen_cap, _deduped = set(), []
for _c, _k in CAPITAL_COUNTRY:
    _ck = (_c, _k)
    if _ck not in _seen_cap and _c != _k:
        _seen_cap.add(_ck)
        _deduped.append(_ck)
CAPITAL_COUNTRY = _deduped

# ===========================================================================
# WORDNET DATASET GENERATORS
# ===========================================================================

def gen_hyponymy(max_pairs=2000, log=None):
    """
    Hyponymy: (hyponym, hypernym)  → u is more specific than v
    Example: dog → animal
    G-Angle expectation: Δθ < 0  (u→v < v→u in g-angle)
    """
    wn = get_wn()
    pairs = []
    for syn in wn.all_synsets("n"):
        for hypo in syn.hyponyms():
            for l1 in hypo.lemmas():
                for l2 in syn.lemmas():
                    u = l1.name().replace("_", " ").lower()
                    v = l2.name().replace("_", " ").lower()
                    if u != v and len(u.split()) == 1 and len(v.split()) == 1:
                        pairs.append((u, v, "hyponymy"))
                    if len(pairs) >= max_pairs:
                        return pairs
    return pairs

def gen_meronymy(max_pairs=2000, log=None):
    """
    Meronymy: (meronym, holonym)  → u is a part of v
    Example: wheel → car
    G-Angle expectation: Δθ < 0  (part is more specific than whole)
    """
    wn = get_wn()
    pairs = []
    for syn in wn.all_synsets("n"):
        for mero in syn.part_meronyms() + syn.member_meronyms() + syn.substance_meronyms():
            for l1 in mero.lemmas():
                for l2 in syn.lemmas():
                    u = l1.name().replace("_", " ").lower()
                    v = l2.name().replace("_", " ").lower()
                    if u != v and len(u.split()) == 1 and len(v.split()) == 1:
                        pairs.append((u, v, "meronymy"))
                    if len(pairs) >= max_pairs:
                        return pairs
    return pairs

def gen_capital_country(log=None):
    """Capital-Country (built-in): capital → country"""
    return [(c, k, "capital") for c, k in CAPITAL_COUNTRY]

def gen_sibling_symmetric(max_pairs=1500, log=None):
    """
    Sibling-Symmetric: two co-hyponyms of the same hypernym.
    Example: cat ↔ dog (both are hyponyms of 'animal')
    G-Angle expectation: |Δθ| ≈ 0  (symmetric, no hierarchy)
    """
    wn = get_wn()
    pairs = []
    seen  = set()
    for syn in wn.all_synsets("n"):
        children = syn.hyponyms()
        if len(children) < 2:
            continue
        words = []
        for child in children[:6]:
            for l in child.lemmas():
                w = l.name().replace("_", " ").lower()
                if len(w.split()) == 1:
                    words.append(w)
        for i in range(len(words)):
            for j in range(i+1, len(words)):
                u, v = words[i], words[j]
                key  = (min(u,v), max(u,v))
                if key not in seen and u != v:
                    seen.add(key)
                    pairs.append((u, v, "sibling"))
                if len(pairs) >= max_pairs:
                    return pairs
    return pairs

def gen_coordinate_symmetric(max_pairs=1500, log=None):
    """
    Coordinate-Symmetric: pairs from the same thematic category.
    Drawn from synsets 'food', 'animal', 'profession', 'fruit', etc.
    G-Angle expectation: |Δθ| ≈ 0
    """
    wn = get_wn()
    SEEDS = [
        "fruit.n.01", "animal.n.01", "profession.n.01",
        "vegetable.n.01", "color.n.01", "tool.n.01",
        "vehicle.n.01", "sport.n.01",
    ]
    pairs = []
    seen  = set()
    for seed_name in SEEDS:
        try:
            seed = wn.synset(seed_name)
        except Exception:
            continue
        # Collect all hyponyms at levels 1 and 2
        words = []
        for hypo in seed.hyponyms():
            for l in hypo.lemmas():
                w = l.name().replace("_", " ").lower()
                if len(w.split()) == 1:
                    words.append(w)
            for hypo2 in hypo.hyponyms()[:3]:
                for l in hypo2.lemmas():
                    w = l.name().replace("_", " ").lower()
                    if len(w.split()) == 1:
                        words.append(w)
        words = list(set(words))[:30]
        for i in range(len(words)):
            for j in range(i+1, len(words)):
                u, v = words[i], words[j]
                key  = (min(u,v), max(u,v))
                if key not in seen and u != v:
                    seen.add(key)
                    pairs.append((u, v, "coordinate"))
                if len(pairs) >= max_pairs:
                    return pairs
    return pairs

# ===========================================================================
# LOAD FASTTEXT
# ===========================================================================

def load_fasttext_vec(path, log):
    log("Loading FastText .vec...")
    vocab = {}
    with open(path, "r", encoding="utf-8") as f:
        f.readline()
        for i, line in enumerate(f):
            p = line.rstrip().split(" ")
            if len(p) < 2: continue
            vocab[p[0].lower()] = np.array(p[1:], dtype=np.float32)
            if (i+1) % 100_000 == 0:
                log(f"  {i+1:,} words...")
    log(f"Done: {len(vocab):,} words.")
    return lambda w: vocab.get(w.lower(), None)

def load_fasttext_bin(path, log):
    try: import fasttext
    except ImportError: raise ImportError("pip install fasttext")
    log("Loading FastText .bin...")
    m = fasttext.load_model(path)
    log("Model .bin ready.")
    return lambda w: np.array(m.get_word_vector(w.lower()), dtype=np.float32)

def load_fasttext(path, log):
    ext = os.path.splitext(path)[-1].lower()
    if ext == ".vec": return load_fasttext_vec(path, log)
    if ext == ".bin": return load_fasttext_bin(path, log)
    raise ValueError(f"Unrecognized format: {ext}")

# ===========================================================================
# LOAD GLOVE  [NEW v8]
# ===========================================================================

def load_glove(path, log):
    """
    Load GloVe .txt vectors (e.g. glove.6B.300d.txt).
    Format: word f1 f2 ... fd  (space-separated, no header line)
    Download: https://nlp.stanford.edu/projects/glove/
    """
    log("Loading GloVe .txt ...")
    vocab = {}
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            parts = line.rstrip().split(" ")
            if len(parts) < 2:
                continue
            word = parts[0].lower()
            try:
                vec = np.array(parts[1:], dtype=np.float32)
                vocab[word] = vec
            except ValueError:
                continue
            if (i + 1) % 100_000 == 0:
                log(f"  {i+1:,} GloVe words ...")
    log(f"GloVe done: {len(vocab):,} words.")
    return lambda w: vocab.get(w.lower(), None)


# ===========================================================================
# RANDOMIZED EMBEDDING CONTROL  [NEW v8]
# ===========================================================================

def make_randomized_get_vec(real_get_vec, pairs, log, seed=42):
    """
    Build a get_vec that returns random Gaussian vectors N(mu, sigma^2)
    calibrated to the same statistics as the real embedding.

    Scientific rationale:
      - If asymmetry still appears  -> geometry of g-angle is the source
      - If direction accuracy -> ~50% -> semantic directional info is gone
    This separates geometric asymmetry from semantic directional bias.
    """
    log("Randomized control: sampling embedding stats from real vectors ...")

    sample_vecs = []
    for u, v, _ in pairs[:500]:
        eu = real_get_vec(u)
        ev = real_get_vec(v)
        if eu is not None:
            sample_vecs.append(eu)
        if ev is not None:
            sample_vecs.append(ev)
    if not sample_vecs:
        log("  [WARN] No real vectors found for stats -- using N(0,1) dim=300")
        dim = 300
        mu  = np.zeros(dim)
        std = np.ones(dim)
    else:
        arr = np.stack(sample_vecs, axis=0)
        mu  = arr.mean(axis=0)
        std = arr.std(axis=0)
        std[std < 1e-9] = 1e-9
        dim = arr.shape[1]
        log(f"  Calibrated from {len(sample_vecs)} vectors  dim={dim}")

    cache = {}
    def get_rand_vec(word):
        w = word.lower()
        if w not in cache:
            sub_seed = abs(hash(w)) % (2**31)
            rng2 = np.random.default_rng(sub_seed + seed)
            cache[w] = (rng2.standard_normal(dim) * std + mu).astype(np.float32)
        return cache[w]

    log("  Randomized embedding ready.")
    return get_rand_vec


# ===========================================================================
# EVALUATION — G-ANGLE AND COSINE
# ===========================================================================

def evaluate_pairs(pairs, get_vec, p, expected, log, label="",
                   n_permutations=500, top_k_error=20, progress_cb=None):
    """
    Evaluate pairs with BOTH metrics:
    - G-Angle: Δθ = A_g(u,v) - A_g(v,u)
    - Cosine:  cos(u,v)  [symmetric, has no Δ]

    expected: 'asymmetric' → Δθ expected to be negative
              'symmetric'  → |Δθ| expected to be small

    Additional features:
    - Permutation test  : random baseline distribution (n_permutations times)
    - Symmetry score    : P(|Δθ| < τ) for symmetric relations (replaces swap test)
    - Ranking accuracy  : ability to distinguish real vs random pairs
    - Error analysis    : pairs with incorrect direction prediction

    progress_cb(value): optional callback called with int 0-100 to update progress bar
    """
    def _prog(v):
        if progress_cb:
            progress_cb(int(v))

    results = {
        "label"      : label,
        "expected"   : expected,
        "deltas"     : [],   # G-Angle Δθ
        "cosines"    : [],   # cosine similarity
        "g_angles_uv": [],   # A_g(u→v)
        "g_angles_vu": [],   # A_g(v→u)
        "pairs_ok"   : [],   # successfully evaluated pairs
        "skipped"    : 0,
    }
    n = len(pairs)
    vecs = []   # store (eu, ev) for permutation/ranking
    for i, (u, v, _) in enumerate(pairs):
        if i % 200 == 0:
            log(f"  {label}: {i}/{n} ...")
            _prog(10 + int(i / max(n, 1) * 40))  # 10%–50%
        eu = get_vec(u)
        ev = get_vec(v)
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
        results["pairs_ok"].append((u, v))
        vecs.append((eu, ev))

    _prog(50)  # selesai hitung vektor

    # Convert to arrays
    for k in ["deltas","cosines","g_angles_uv","g_angles_vu"]:
        results[k] = np.array(results[k])

    # Compute basic statistics
    d = results["deltas"]
    c = results["cosines"]
    n_eval = len(d)
    results["n"] = n_eval

    if n_eval == 0:
        results["stats"] = _empty_stats()
        return results

    # ── G-Angle stats ─────────────────────────────────────────────────────────
    t_stat, t_p = ttest_1samp(d, 0) if n_eval > 1 else (0.0, 1.0)

    # Direction accuracy with epsilon margin  [NEW v12]
    # Pairs where |Δθ| < DIR_EPS are near the decision boundary (noise-prone);
    # they are excluded from dir_acc to avoid noise-flip artefacts.
    DIR_EPS = 1e-4
    # SYM_TAU: absolute threshold (radians). |Δθ| < τ → "correctly symmetric".
    # Unlike the old median split (always ≈50% by construction), this gives a
    # meaningful score: how many pairs are genuinely near-zero asymmetry.
    SYM_TAU = 0.05   # ~2.9°; tunable

    if expected == "asymmetric":
        n_uncertain = int(np.sum(np.abs(d) < DIR_EPS))
        d_certain   = d[np.abs(d) >= DIR_EPS]
        dir_acc     = float(np.mean(d_certain < 0)) if len(d_certain) > 0 else 0.0
    else:
        n_uncertain = int(np.sum(np.abs(d) < DIR_EPS))
        # Use absolute threshold: proportion of pairs with |Δθ| < SYM_TAU.
        # This is not forced to 50% — if SYM_TAU is well-calibrated it reflects
        # true embedding symmetry (high → pairs are genuinely symmetric).
        dir_acc     = float(np.mean(np.abs(d) < SYM_TAU))

    cos_mean = float(np.mean(c))
    cos_std  = float(np.std(c))

    # ── EFFECT SIZE: Cohen's d + 95% CI  [NEW v8] ────────────────────────────
    # Cohen's d = mean(Δθ) / std(Δθ)
    # 95% CI via t-distribution: mean ± t(0.975, df=n-1) * SE
    std_d = float(np.std(d, ddof=1)) if n_eval > 1 else 1.0
    cohens_d = float(np.mean(d) / std_d) if std_d > 1e-15 else 0.0
    se_d = std_d / np.sqrt(n_eval) if n_eval > 1 else float("inf")
    from scipy.stats import t as t_dist
    t_crit = t_dist.ppf(0.975, df=max(n_eval - 1, 1))
    ci_lo  = float(np.mean(d) - t_crit * se_d)
    ci_hi  = float(np.mean(d) + t_crit * se_d)


    # ── DISTRIBUTION SHAPE METRICS  [NEW v13] ────────────────────────────────
    # Skewness, excess kurtosis, differential entropy (histogram), bimodality.
    # Bimodality coefficient BC = (skew²+1) / kurtosis  (Pfister et al. 2013):
    #   BC > 0.555 (= 5/9) suggests bimodality.
    from scipy.stats import skew as sp_skew, kurtosis as sp_kurtosis
    d_sk  = float(sp_skew(d))                          # Fisher skewness
    d_ku  = float(sp_kurtosis(d, fisher=True))         # excess kurtosis
    bc    = (d_sk**2 + 1) / (d_ku + 3) if (d_ku + 3) != 0 else 0.0
    # Differential entropy via histogram (10 bins)
    hist_counts, _ = np.histogram(d, bins=max(10, n_eval // 20))
    probs = hist_counts / hist_counts.sum()
    probs = probs[probs > 0]
    d_ent = float(-np.sum(probs * np.log(probs)))      # in nats

    # ── BOOTSTRAP CI for dir_acc, mean_dt  [NEW v12] ────────────────────────
    # Non-parametric percentile bootstrap (B=2000 resamples).
    # Covers dir_acc and mean_dt here; rank_acc bootstrap done after ranking.
    N_BOOT = 2000
    rng_boot = np.random.default_rng(RANDOM_SEED + 999)
    boot_dir_accs   = []
    boot_mean_dts   = []
    d_arr = np.asarray(d)
    for _ in range(N_BOOT):
        idx_b = rng_boot.integers(0, n_eval, size=n_eval)
        d_b   = d_arr[idx_b]
        boot_mean_dts.append(float(np.mean(d_b)))
        if expected == "asymmetric":
            d_b_cert = d_b[np.abs(d_b) >= DIR_EPS]
            boot_dir_accs.append(float(np.mean(d_b_cert < 0)) if len(d_b_cert) > 0 else 0.0)
        else:
            boot_dir_accs.append(float(np.mean(np.abs(d_b) < SYM_TAU)))
    boot_dir_accs  = np.array(boot_dir_accs)
    boot_mean_dts  = np.array(boot_mean_dts)
    dir_acc_ci_lo  = float(np.percentile(boot_dir_accs, 2.5))
    dir_acc_ci_hi  = float(np.percentile(boot_dir_accs, 97.5))
    mean_dt_ci_lo  = float(np.percentile(boot_mean_dts, 2.5))
    mean_dt_ci_hi  = float(np.percentile(boot_mean_dts, 97.5))
    log(f"    Bootstrap dir_acc 95%CI  = [{dir_acc_ci_lo:.1%}, {dir_acc_ci_hi:.1%}]")
    log(f"    Bootstrap mean_dt 95%CI  = [{mean_dt_ci_lo:+.4f}, {mean_dt_ci_hi:+.4f}]")

    # ── GEOMETRIC DEGENERACY CHECK (p=2 / Euclidean) ─────────────────────────
    # For p=2, g-angle reduces to the standard angle (arccos of cosine similarity),
    # which is SYMMETRIC: A_g(u,v) ≡ A_g(v,u)  →  Δθ ≡ 0 for all pairs.
    # This is not a failure — it is expected mathematical behaviour.
    # We detect it and skip all permutation / ranking tests (which would be
    # meaningless and waste CPU), then mark all affected metrics as NaN / N/A.
    is_degenerate = bool(np.allclose(d, 0.0, atol=1e-9))
    results["is_degenerate"] = is_degenerate

    eu_list = [eu for eu, _ in vecs]
    ev_list = [ev for _, ev in vecs]

    obs_dir_acc = dir_acc
    obs_mean    = float(np.mean(d))

    if is_degenerate:
        # ── DEGENERATE PATH (p=2): skip all permutation & ranking loops ───────
        log(f"  [{label}] p=2 symmetric geometry — Δθ ≡ 0 for all pairs. "
            f"Permutation and ranking tests skipped (N/A).")
        _prog(95)

        perm_p_orient    = float("nan")
        perm_p_pair      = float("nan")
        perm_p           = float("nan")
        ranking_acc      = float("nan")
        rand_abs_dt      = np.array([])
        rank_acc_ci_lo   = float("nan")
        rank_acc_ci_hi   = float("nan")

        # Empty arrays for plot compatibility
        perm_orient_accs = np.array([])
        perm_pair_accs   = np.array([])

        results["perm_means"]     = perm_orient_accs
        results["perm_pair_accs"] = perm_pair_accs
        results["perm_p"]         = perm_p
        results["perm_p_orient"]  = perm_p_orient
        results["perm_p_pair"]    = perm_p_pair
        results["perm_obs_stat"]  = obs_dir_acc
        results["ranking_acc"]    = ranking_acc
        results["rand_abs_dt"]    = rand_abs_dt

    else:
        # ── NORMAL PATH: run permutation tests & ranking ───────────────────────

        # ── PERMUTATION TEST A — ORIENTATION PERMUTATION (asymmetric only) ──
        # Null: direction accuracy does not differ from chance (50%).
        # Method: independently flip (u,v) <-> (v,u) per pair with prob 0.5,
        # then recompute dir_acc = P(Δθ < 0).
        #
        # *** SYMMETRIC RELATIONS: Perm A SKIPPED (v14 fix) ***
        # Δθ(u,v) = −Δθ(v,u) is a mathematical identity, so flipping a pair
        # only negates Δθ but never changes |Δθ|.  Any statistic of |Δθ|
        # (P(|Δθ|<τ), mean|Δθ|, …) is invariant to the flip → null σ = 0.
        # Only Perm B with cross-pool null is informative for symmetric.
        if expected == "asymmetric":
            log(f"  {label}: Permutation A (orientation) n={n_permutations} ...")
            _prog(60)
            rng = np.random.default_rng(RANDOM_SEED)
            perm_orient_accs = []
            for _ in range(n_permutations):
                flip_mask = rng.random(n_eval) < 0.5
                perm_dt = np.empty(n_eval)
                for i in range(n_eval):
                    if flip_mask[i]:
                        perm_dt[i] = ENGINE.delta_theta(ev_list[i], eu_list[i], p)
                    else:
                        perm_dt[i] = ENGINE.delta_theta(eu_list[i], ev_list[i], p)
                perm_orient_accs.append(float(np.mean(perm_dt < 0)))
            perm_orient_accs = np.array(perm_orient_accs)
            perm_p_orient = float(np.mean(perm_orient_accs >= obs_dir_acc))
            log(f"    [Orientation] obs={obs_dir_acc:.3f}  "
                f"null={np.mean(perm_orient_accs):.3f}±{np.std(perm_orient_accs):.3f}  "
                f"p={perm_p_orient:.4f}")
        else:
            log(f"  {label}: Perm A skipped (symmetric) — "
                f"|Δθ| invariant to u↔v flip by identity Δθ(u,v)=−Δθ(v,u). "
                f"Cross-pool Perm B used instead.")
            perm_orient_accs = np.array([])
            perm_p_orient    = float("nan")
            _prog(60)

        # ── PERMUTATION TEST B — PAIR PERMUTATION ────────────────────────────
        # Null: the PAIRING itself carries no directional signal.
        #
        # ASYMMETRIC: reshuffle which u is matched with which v (intra-pool).
        #   Statistic: P(Δθ < 0).  Tests whether real pairings are needed.
        #
        # SYMMETRIC (v14 redesign):
        #   Statistic: mean(|Δθ|)  — continuous, not binary.
        #     Rationale: binary P(|Δθ|<τ) collapses perm distribution to σ≈0
        #     because most or all permuted pairs also satisfy |Δθ|<τ (the
        #     embedding vectors in a symmetric dataset all occupy a similar
        #     region, so random pairings look nearly identical to real ones).
        #   Null: CROSS-POOL shuffle — pair each u from this relation with a
        #     v drawn from the OTHER symmetric pool (sibling ↔ coordinate).
        #     This creates genuinely different pairs (different topical clusters)
        #     and gives the null distribution nonzero variance.
        #     p = P(mean_null_|Δθ| ≤ obs_mean_|Δθ|): real symmetric pairs
        #     should have SMALLER |Δθ| than cross-pool pairs → small p is good.
        log(f"  {label}: Permutation B (pair shuffle) n={n_permutations} ...")
        _prog(72)
        rng_b = np.random.default_rng(RANDOM_SEED + 100)

        if expected == "asymmetric":
            # ── Asymmetric: intra-pool shuffle, statistic = P(Δθ < 0) ──────
            perm_pair_accs = []
            for _ in range(n_permutations):
                shuffled_v = rng_b.permutation(n_eval)
                perm_dt = np.array([
                    ENGINE.delta_theta(eu_list[i], ev_list[shuffled_v[i]], p)
                    for i in range(n_eval)
                ])
                perm_pair_accs.append(float(np.mean(perm_dt < 0)))
            perm_pair_accs = np.array(perm_pair_accs)
            perm_p_pair = float(np.mean(perm_pair_accs >= obs_dir_acc))
            log(f"    [Pair shuffle] obs={obs_dir_acc:.3f}  "
                f"null={np.mean(perm_pair_accs):.3f}±{np.std(perm_pair_accs):.3f}  "
                f"p={perm_p_pair:.4f}")

        else:
            # ── Symmetric: cross-pool shuffle, statistic = mean |Δθ| ────────
            # Observed statistic: mean |Δθ| of the real pairs (lower = more symmetric).
            obs_sym_stat = float(np.mean(np.abs(d)))

            # Build cross-pool: collect vectors from ALL OTHER evaluated results
            # that we can reach through the shared `vecs` list.  Since evaluate_pairs
            # is called per relation, we cannot access other relations' vectors here
            # directly.  Instead, we approximate the cross-pool by using the OPPOSITE
            # half of THIS relation's own pair pool (split by index parity):
            #   • even-indexed pairs provide u vectors
            #   • odd-indexed pairs provide v vectors for the null
            # This creates pairs that were not in the original dataset and share no
            # semantic co-hyponym relationship, giving nonzero null variance.
            #
            # Note: caller may inject cross_vecs via the `_cross_pool_vecs` key in
            # the results dict (set before calling evaluate_pairs) for a stronger null.
            cross_u = [eu_list[i] for i in range(0,   n_eval, 2)]
            cross_v = [ev_list[i] for i in range(1,   n_eval, 2)]
            n_cross = min(len(cross_u), len(cross_v))

            perm_sym_stats = []
            if n_cross < 10:
                # Fallback: intra-pool shuffle with mean|Δθ| statistic
                log(f"    [Sym cross-pool] too few pairs ({n_cross}) — "
                    f"falling back to intra-pool mean|Δθ|.")
                for _ in range(n_permutations):
                    shuf = rng_b.permutation(n_eval)
                    perm_dt = np.array([
                        ENGINE.delta_theta(eu_list[i], ev_list[shuf[i]], p)
                        for i in range(n_eval)
                    ])
                    perm_sym_stats.append(float(np.mean(np.abs(perm_dt))))
            else:
                for _ in range(n_permutations):
                    # Sample n_cross pairs from the cross-pool with replacement
                    idx = rng_b.integers(0, n_cross, size=n_cross)
                    perm_dt = np.array([
                        ENGINE.delta_theta(cross_u[i], cross_v[i], p)
                        for i in idx
                    ])
                    perm_sym_stats.append(float(np.mean(np.abs(perm_dt))))

            perm_pair_accs = np.array(perm_sym_stats)   # contains mean|Δθ| values

            # p-value: fraction of null iterations where mean|Δθ|_null ≤ obs_sym_stat
            # (real pairs should have SMALLER mean|Δθ| than random cross-pool pairs)
            perm_p_pair = float(np.mean(perm_pair_accs <= obs_sym_stat))
            # Store obs_sym_stat for plot (override obs_dir_acc for symmetric)
            results["perm_obs_stat_sym"] = obs_sym_stat
            log(f"    [Sym cross-pool] obs_mean|Δθ|={obs_sym_stat:.4f}  "
                f"null_mean|Δθ|={np.mean(perm_pair_accs):.4f}±"
                f"{np.std(perm_pair_accs):.4f}  p={perm_p_pair:.4f}")

        # Primary perm_p:
        #   Asymmetric: max(orient, pair) — most conservative
        #   Symmetric:  perm_p_pair only (orient is NaN — skipped)
        if expected == "asymmetric":
            perm_p = max(perm_p_orient, perm_p_pair)
        else:
            perm_p = perm_p_pair   # orient is nan for symmetric

        results["perm_means"]     = perm_orient_accs
        results["perm_pair_accs"] = perm_pair_accs
        results["perm_p"]         = perm_p
        results["perm_p_orient"]  = perm_p_orient
        results["perm_p_pair"]    = perm_p_pair
        # perm_obs_stat for plot: dir_acc for asymmetric, mean|Δθ| for symmetric
        results["perm_obs_stat"]  = (
            results.get("perm_obs_stat_sym", obs_dir_acc)
            if expected == "symmetric" else obs_dir_acc
        )

        # ── RANKING ACCURACY (AUC / Mann-Whitney) ────────────────────────────
        # Measures: P(score_real > score_random) over independently sampled pairs.
        #
        # FIX (v13): replaced per-pair comparison with AUC-style Mann-Whitney U.
        #
        # Previous bug: real_score[i] was compared against rand_score[i] where
        # both used the SAME u vector (eu_list[idx_real[i]]).  This is not a fair
        # "real vs random" test — it asks "is (u,v_real) better than (u,v_rand)?",
        # a fundamentally different question from dir_acc.  Moreover, idx_rand was
        # drawn from the same permutation as idx_real, causing 100% overlap of u
        # indices.  This could yield rank_acc well below dir_acc (observed ~44%
        # when dir_acc=68%) because a mismatched v can also produce a negative Δθ,
        # making rand_score unexpectedly high.
        #
        # Correct formulation (AUC):
        #   • real_scores  = -Δθ for ALL real pairs  (higher = more directional)
        #   • rand_scores  = -Δθ for n_sample INDEPENDENT mismatch pairs,
        #                    where u_idx and v_idx are drawn separately
        #   • ranking_acc  = Mann-Whitney U / (n_real × n_rand)
        #                  = P(score_real > score_rand)
        #
        # Asymmetric: real score = -Δθ  (negative Δθ → high score)
        # Symmetric:  real score = -|Δθ| (small |Δθ| → high score, i.e. less asymmetric)
        #
        # AUC = 0.5 is chance, > 0.5 is signal — consistent with dir_acc > 50%.
        log(f"  {label}: Ranking accuracy (AUC) ...")
        _prog(80)
        rng2 = np.random.default_rng(RANDOM_SEED + 1)
        n_rand = min(n_eval, 500)

        # Draw u and v indices INDEPENDENTLY for mismatch pairs
        idx_u = rng2.choice(n_eval, size=n_rand, replace=True)
        idx_v = rng2.choice(n_eval, size=n_rand, replace=True)
        # Avoid accidental self-pairs (same index → identical vector → Δθ=0)
        for i in range(n_rand):
            while idx_v[i] == idx_u[i]:
                idx_v[i] = rng2.integers(0, n_eval)

        rand_score_list = []
        for i in range(n_rand):
            rd = ENGINE.delta_theta(eu_list[idx_u[i]], ev_list[idx_v[i]], p)
            rand_score_list.append(rd)
        rand_dt = np.array(rand_score_list)

        if expected == "asymmetric":
            real_scores_auc = -d              # higher = more negative Δθ = correct
            rand_scores_auc = -rand_dt
        else:
            real_scores_auc = -np.abs(d)     # higher = smaller |Δθ| = more symmetric
            rand_scores_auc = -np.abs(rand_dt)

        from scipy.stats import mannwhitneyu
        mw_stat, _ = mannwhitneyu(real_scores_auc, rand_scores_auc, alternative="greater")
        ranking_acc = float(mw_stat / (len(real_scores_auc) * len(rand_scores_auc)))

        rand_abs_dt = np.abs(rand_dt)   # kept for plot compatibility
        results["ranking_acc"] = ranking_acc
        results["rand_abs_dt"] = rand_abs_dt
        log(f"    Ranking AUC = {ranking_acc:.1%}  "
            f"(real n={len(real_scores_auc)}, rand n={n_rand})")

        # Bootstrap CI for ranking_acc (percentile bootstrap over real_scores_auc)
        rng_rank_boot = np.random.default_rng(RANDOM_SEED + 777)
        boot_rank_accs = []
        for _ in range(N_BOOT):
            idx_b      = rng_rank_boot.integers(0, len(real_scores_auc), size=len(real_scores_auc))
            idx_rb     = rng_rank_boot.integers(0, n_rand, size=n_rand)
            mw_b, _    = mannwhitneyu(real_scores_auc[idx_b], rand_scores_auc[idx_rb],
                                      alternative="greater")
            boot_rank_accs.append(float(mw_b / (len(idx_b) * len(idx_rb))))
        boot_rank_accs = np.array(boot_rank_accs)
        rank_acc_ci_lo = float(np.percentile(boot_rank_accs, 2.5))
        rank_acc_ci_hi = float(np.percentile(boot_rank_accs, 97.5))
        log(f"    Bootstrap rank_acc 95%CI = [{rank_acc_ci_lo:.1%}, {rank_acc_ci_hi:.1%}]")

    # ── ERROR ANALYSIS  [enriched v13, 3-way breakdown v14] ─────────────────
    # Asymmetric pairs where direction was predicted wrong (Δθ ≥ 0, should be < 0)
    # Symmetric pairs where |Δθ| is large (larger than median)
    #
    # v14: each error is further sub-classified into one of three buckets:
    #   near_zero_ambiguity  — |Δθ| < NEAR_ZERO_TAU  (signal too weak to decide)
    #   extreme_outlier      — |Δθ| > OUTLIER_TAU     (gross misprediction)
    #   false_direction      — all other errors        (clear wrong-direction)
    #
    # Thresholds (radians):
    #   NEAR_ZERO_TAU = DIR_EPS already defined (1e-4) would be too tight for
    #   the "ambiguous" concept; use 0.02 rad (~1.1°) — pairs this close to 0
    #   are genuinely ambiguous regardless of sign.
    #   OUTLIER_TAU   = mean_abs_dt + 2 * std_dt  (dataset-adaptive)
    #
    # norm_ratio      = ‖u‖p / ‖v‖p   — magnitude imbalance
    # frequency_proxy = 1/(len(word)+1) — crude OOV/frequency surrogate
    _prog(90)

    NEAR_ZERO_TAU = 0.02          # rad — below this |Δθ| is "near-zero ambiguity"
    abs_d_arr     = np.abs(d)
    mean_abs      = float(np.mean(abs_d_arr))
    std_abs       = float(np.std(abs_d_arr, ddof=1)) if n_eval > 1 else 0.0
    OUTLIER_TAU   = mean_abs + 2.0 * std_abs   # dataset-adaptive outlier threshold

    def _classify_error(dt_val):
        """Return (category, sub_class, error_type_str) for an error pair."""
        abs_dt = abs(dt_val)
        if abs_dt < NEAR_ZERO_TAU:
            return ("near_zero_ambiguity",
                    f"|Δθ|={abs_dt:.4f} < {NEAR_ZERO_TAU:.3f} rad (ambiguous)")
        elif abs_dt > OUTLIER_TAU:
            return ("extreme_outlier",
                    f"|Δθ|={abs_dt:.4f} > outlier thresh {OUTLIER_TAU:.4f}")
        else:
            return ("false_direction",
                    f"Δθ={dt_val:+.4f} wrong direction")

    error_pairs = []
    if expected == "asymmetric":
        for i, ((u, v), dt) in enumerate(zip(results["pairs_ok"], d)):
            if dt >= 0:   # wrong prediction: should be negative
                eu_i = eu_list[i]; ev_i = ev_list[i]
                norm_u = float(np.linalg.norm(eu_i, ord=p))
                norm_v = float(np.linalg.norm(ev_i, ord=p))
                nr = norm_u / norm_v if norm_v > 1e-12 else float("inf")
                sub_class, err_str = _classify_error(dt)
                error_pairs.append({
                    "word1"           : u,
                    "word2"           : v,
                    "delta_theta"     : float(dt),
                    "cosine"          : float(c[i]),
                    "error_type"      : err_str,
                    "norm_ratio"      : nr,
                    "frequency_proxy" : 1.0 / (len(u) + 1),
                    "category"        : "false_positive",
                    "sub_class"       : sub_class,
                })
    else:
        med_abs = float(np.median(abs_d_arr))
        for i, ((u, v), dt) in enumerate(zip(results["pairs_ok"], d)):
            if abs(dt) > med_abs:   # unexpected asymmetry
                eu_i = eu_list[i]; ev_i = ev_list[i]
                norm_u = float(np.linalg.norm(eu_i, ord=p))
                norm_v = float(np.linalg.norm(ev_i, ord=p))
                nr = norm_u / norm_v if norm_v > 1e-12 else float("inf")
                sub_class, err_str = _classify_error(dt)
                error_pairs.append({
                    "word1"           : u,
                    "word2"           : v,
                    "delta_theta"     : float(dt),
                    "cosine"          : float(c[i]),
                    "error_type"      : err_str,
                    "norm_ratio"      : nr,
                    "frequency_proxy" : 1.0 / (len(u) + 1),
                    "category"        : "false_positive",
                    "sub_class"       : sub_class,
                })

    # Sort by "how wrong" (distance from ideal)
    if expected == "asymmetric":
        error_pairs.sort(key=lambda x: x["delta_theta"], reverse=True)
    else:
        error_pairs.sort(key=lambda x: abs(x["delta_theta"]), reverse=True)
    results["error_pairs"] = error_pairs   # all pairs stored (for CSV export)

    # ── 3-way breakdown counts ────────────────────────────────────────────────
    n_false_dir  = sum(1 for e in error_pairs if e["sub_class"] == "false_direction")
    n_near_zero  = sum(1 for e in error_pairs if e["sub_class"] == "near_zero_ambiguity")
    n_outlier    = sum(1 for e in error_pairs if e["sub_class"] == "extreme_outlier")
    error_breakdown = {
        "false_direction"    : n_false_dir,
        "near_zero_ambiguity": n_near_zero,
        "extreme_outlier"    : n_outlier,
        "near_zero_tau"      : NEAR_ZERO_TAU,
        "outlier_tau"        : OUTLIER_TAU,
    }
    results["error_breakdown"] = error_breakdown

    # Top-10 false positives (worst errors) and false negatives
    # For asymmetric: false positive = predicted wrong (Δθ≥0); no true FN concept
    # For asymmetric: false negative = correctly predicted but just barely (smallest |Δθ| negatives)
    results["top_fp"] = error_pairs[:10]
    if expected == "asymmetric":
        correct_pairs = []
        for i, ((u, v), dt) in enumerate(zip(results["pairs_ok"], d)):
            if dt < 0:
                eu_i = eu_list[i]; ev_i = ev_list[i]
                norm_u = float(np.linalg.norm(eu_i, ord=p))
                norm_v = float(np.linalg.norm(ev_i, ord=p))
                nr = norm_u / norm_v if norm_v > 1e-12 else float("inf")
                correct_pairs.append({
                    "word1": u, "word2": v,
                    "delta_theta"     : float(dt),
                    "cosine"          : float(c[i]),
                    "error_type"      : "marginal correct (smallest |Δθ|<0)",
                    "norm_ratio"      : nr,
                    "frequency_proxy" : 1.0 / (len(u) + 1),
                    "category"        : "false_negative",
                    "sub_class"       : "marginal_correct",
                })
        # sort by least-negative Δθ (closest to flip)
        correct_pairs.sort(key=lambda x: x["delta_theta"], reverse=True)
        results["top_fn"] = correct_pairs[:10]
    else:
        results["top_fn"] = []

    log(f"    Error pairs: {len(error_pairs)} total  "
        f"|  false_direction={n_false_dir}  "
        f"near_zero={n_near_zero}  "
        f"outlier={n_outlier}  "
        f"|  top_fp={len(results['top_fp'])}  top_fn={len(results['top_fn'])}")

    # ── Verdict ───────────────────────────────────────────────────────────────
    if expected == "asymmetric":
        ga_wins = dir_acc > 0.5
        verdict = "G-Angle ✔" if ga_wins else "Tie / Not significant"
    else:
        ga_wins = abs(obs_mean) < 0.05
        verdict = "G-Angle symmetric ✔" if ga_wins else "G-Angle unexpected asymmetry"

    # Degeneracy-aware display strings
    dir_acc_display     = "N/A (symmetric geometry)" if is_degenerate else f"{dir_acc:.1%}"
    ranking_acc_display = "N/A (symmetric geometry)" if is_degenerate else f"{ranking_acc:.1%}"

    results["stats"] = {
        "n"                  : n_eval,
        "skipped"            : results["skipped"],
        "mean_dt"            : obs_mean,
        "std_dt"             : float(np.std(d)),
        "median_dt"          : float(np.median(d)),
        "pct_neg"            : float(np.mean(d < 0)),
        "dir_acc"            : dir_acc,
        "dir_acc_display"    : dir_acc_display,
        "ranking_acc_display": ranking_acc_display,
        "is_degenerate"      : is_degenerate,
        "t_stat"             : float(t_stat),
        "t_p"                : float(t_p),
        "cos_mean"           : cos_mean,
        "cos_std"            : cos_std,
        "ga_wins"            : ga_wins,
        "verdict"            : verdict,
        "mean_abs_dt"        : float(np.mean(np.abs(d))),
        # Permutation A + B
        "perm_p"             : perm_p,
        "perm_p_orient"      : perm_p_orient,
        "perm_p_pair"        : perm_p_pair,
        "perm_mean_base"     : float(np.mean(perm_orient_accs)) if len(perm_orient_accs) > 0 else float("nan"),
        # Ranking
        "ranking_acc"        : ranking_acc,
        "n_errors"           : len(error_pairs),
        # Error 3-way breakdown [v14]
        "error_breakdown"    : error_breakdown,
        # Effect size [v8]
        "cohens_d"           : cohens_d,
        "ci_lo"              : ci_lo,
        "ci_hi"              : ci_hi,
        # Bootstrap CIs [v12]
        "n_uncertain"        : n_uncertain,
        "dir_acc_ci_lo"      : dir_acc_ci_lo,
        "dir_acc_ci_hi"      : dir_acc_ci_hi,
        "rank_acc_ci_lo"     : rank_acc_ci_lo,
        "rank_acc_ci_hi"     : rank_acc_ci_hi,
        "mean_dt_ci_lo"      : mean_dt_ci_lo,
        "mean_dt_ci_hi"      : mean_dt_ci_hi,
        # Distribution shape [v13]
        "skewness"           : d_sk,
        "kurtosis"           : d_ku,
        "entropy"            : d_ent,
        "bimodality_coeff"   : bc,
        # BH-corrected q-values filled in post-hoc by apply_bh_correction()
        "t_q"                : float("nan"),
        "perm_q_orient"      : float("nan"),
        "perm_q_pair"        : float("nan"),
        # swap_consistency removed (v12): Δθ(u,v) = -Δθ(v,u) is mathematically
        # guaranteed — always 100%, carries no empirical information.
        # v14: replaced by symmetry_score = P(|Δθ| < SYM_TAU) stored in dir_acc.
        "sym_tau"            : SYM_TAU if expected == "symmetric" else float("nan"),
    }
    _prog(100)
    return results

def _empty_stats():
    return {
        "n":0,"skipped":0,"mean_dt":0,"std_dt":0,"median_dt":0,
        "pct_neg":0,"dir_acc":0,"dir_acc_display":"N/A","ranking_acc_display":"N/A",
        "is_degenerate":False,"t_stat":0,"t_p":1,"cos_mean":0,
        "cos_std":0,"ga_wins":False,"verdict":"No data","mean_abs_dt":0,
        "perm_p":1.0,"perm_p_orient":1.0,"perm_p_pair":1.0,"perm_mean_base":0.0,
        "ranking_acc":0.0,"n_errors":0,"cohens_d":0.0,"ci_lo":0.0,"ci_hi":0.0,
        "error_breakdown":{"false_direction":0,"near_zero_ambiguity":0,"extreme_outlier":0,
                           "near_zero_tau":0.02,"outlier_tau":0.0},
        # v12 bootstrap fields
        "n_uncertain":0,
        "dir_acc_ci_lo":0.0,"dir_acc_ci_hi":0.0,
        "rank_acc_ci_lo":0.0,"rank_acc_ci_hi":0.0,
        "mean_dt_ci_lo":0.0,"mean_dt_ci_hi":0.0,
        # v13 shape + BH fields
        "skewness":0.0,"kurtosis":0.0,"entropy":0.0,"bimodality_coeff":0.0,
        "t_q":float("nan"),"perm_q_orient":float("nan"),"perm_q_pair":float("nan"),
    }

def apply_bh_correction(results_dict, log=None):
    """
    Collect all raw p-values from a results dict (key → evaluate_pairs output),
    apply Benjamini–Hochberg FDR correction, and write q-values back into each
    stats dict.  Call this once after all relations for a given p have been run.

    p-values collected: t_p, perm_p_orient, perm_p_pair  × n_relations
    Total hypotheses = 3 × n_relations  (e.g. 15 for 5 relations).
    """
    keys     = []
    p_labels = []
    p_vals   = []
    for rel_key, res in results_dict.items():
        st = res.get("stats", {})
        if st.get("n", 0) == 0:
            continue
        for field in ("t_p", "perm_p_orient", "perm_p_pair"):
            v = st.get(field, float("nan"))
            if not np.isnan(v):
                keys.append(rel_key)
                p_labels.append(field)
                p_vals.append(v)

    if not p_vals:
        return

    q_vals = bh_correct(np.array(p_vals))

    # Map field → q-field name
    q_field_map = {"t_p": "t_q", "perm_p_orient": "perm_q_orient",
                   "perm_p_pair": "perm_q_pair"}

    for rel_key, field, q in zip(keys, p_labels, q_vals):
        results_dict[rel_key]["stats"][q_field_map[field]] = float(q)

    if log:
        log(f"\n  [BH FDR correction]  m={len(p_vals)} hypotheses")
        for rel_key, field, pv, qv in zip(keys, p_labels, p_vals, q_vals):
            sig = "✔ sig" if qv < 0.05 else "  ns "
            log(f"    {rel_key:<12} {field:<16}  p={pv:.4f}  q={qv:.4f}  {sig}")


# ===========================================================================
# ANALYTICAL BEST-p SELECTION  [NEW v16]
# ===========================================================================

def select_best_p(sweep_results, log=None):
    """
    Analytically select the optimal p from sweep results.

    Strategy (in order of priority):
      1. Skip degenerate p values (p=2 → Euclidean, Δθ≡0 for all pairs).
      2. Compute a composite score per p:
           score(p) = mean_asym_dir_acc  +  0.3 * mean_asym_abs_dt_norm
         where:
           • mean_asym_dir_acc   = mean direction accuracy across asymmetric relations
           • mean_asym_abs_dt_norm = mean |Δθ| across asymmetric relations,
             normalised by the maximum observed across all p (range [0,1])
         Direction accuracy is the primary signal; |Δθ| is a tiebreaker.
      3. Identify the "stable region": contiguous p values within 2% of the peak score.
      4. Select the SMALLEST p inside the stable region (parsimony principle:
         prefer lower p when performance is equivalent).

    Returns
    -------
    best_p        : float — the selected p value
    scores        : dict[p] → float — composite score per p
    stable_region : list[float] — p values in stable plateau
    report_lines  : list[str]  — human-readable explanation (for log)
    """
    ASYM_KEYS = ["hyponymy", "meronymy", "capital"]
    p_vals = sorted(sweep_results.keys())

    # ── Collect raw metrics per p ─────────────────────────────────────────────
    dir_accs  = {}   # p → mean dir_acc over asym relations (NaN if degenerate)
    abs_dts   = {}   # p → mean |Δθ|  over asym relations

    for pv in p_vals:
        da_vals = []
        adt_vals = []
        all_degen = True
        for key in ASYM_KEYS:
            r = sweep_results[pv].get(key)
            if r is None or r.get("n", 0) == 0:
                continue
            st = r["stats"]
            if st.get("is_degenerate", False):
                continue
            all_degen = False
            da_vals.append(st["dir_acc"])
            adt_vals.append(st["mean_abs_dt"])
        if all_degen or not da_vals:
            dir_accs[pv] = float("nan")
            abs_dts[pv]  = float("nan")
        else:
            dir_accs[pv] = float(np.mean(da_vals))
            abs_dts[pv]  = float(np.mean(adt_vals))

    # ── Normalise |Δθ| to [0,1] for tiebreaking ──────────────────────────────
    valid_adt = [v for v in abs_dts.values() if not np.isnan(v)]
    adt_max   = max(valid_adt) if valid_adt else 1.0
    adt_max   = adt_max if adt_max > 1e-9 else 1.0

    # ── Composite score ───────────────────────────────────────────────────────
    scores = {}
    for pv in p_vals:
        da  = dir_accs[pv]
        adt = abs_dts[pv]
        if np.isnan(da):
            scores[pv] = float("nan")
        else:
            adt_norm   = (adt / adt_max) if not np.isnan(adt) else 0.0
            scores[pv] = da + 0.3 * adt_norm

    valid_scores = {pv: s for pv, s in scores.items() if not np.isnan(s)}
    if not valid_scores:
        # Fallback: pick largest non-degenerate p
        non_degen = [pv for pv in p_vals if not np.isnan(dir_accs.get(pv, float("nan")))]
        best_p = non_degen[-1] if non_degen else p_vals[-1]
        return best_p, scores, [], [f"No valid p found — fallback to p={best_p}"]

    peak_score = max(valid_scores.values())
    PLATEAU_TOL = 0.02   # 2% of dir_acc scale

    # ── Stable region: all p with score ≥ peak - tolerance ───────────────────
    stable_region = sorted([pv for pv, s in valid_scores.items()
                            if s >= peak_score - PLATEAU_TOL])

    # ── Best p = smallest p in stable region (parsimony) ─────────────────────
    best_p = stable_region[0] if stable_region else max(valid_scores, key=valid_scores.get)

    # ── Human-readable report ─────────────────────────────────────────────────
    lines = []
    lines.append(f"  [v16] ANALYTICAL p SELECTION")
    lines.append(f"  {'─'*54}")
    lines.append(f"  {'p':>6}  {'dir_acc':>9}  {'mean|Δθ|':>10}  {'score':>8}  {'status'}")
    lines.append(f"  {'─'*54}")
    for pv in p_vals:
        da  = dir_accs.get(pv, float("nan"))
        adt = abs_dts.get(pv,  float("nan"))
        sc  = scores.get(pv,   float("nan"))
        if np.isnan(da):
            status = "DEGENERATE (skip)"
        elif pv == best_p:
            status = "◀ BEST (smallest in stable plateau)"
        elif pv in stable_region:
            status = "  stable plateau"
        else:
            status = ""
        da_s  = f"{da:.1%}"  if not np.isnan(da)  else "  N/A   "
        adt_s = f"{adt:.4f}" if not np.isnan(adt) else "  N/A  "
        sc_s  = f"{sc:.4f}"  if not np.isnan(sc)  else "  N/A  "
        lines.append(f"  {pv:>6.1f}  {da_s:>9}  {adt_s:>10}  {sc_s:>8}  {status}")
    lines.append(f"  {'─'*54}")
    lines.append(f"  Stable plateau (score ≥ {peak_score:.4f} − {PLATEAU_TOL}): "
                 f"p ∈ {stable_region}")
    lines.append(f"  → Best p selected: p = {best_p}  "
                 f"(dir_acc={dir_accs.get(best_p,float('nan')):.1%}  "
                 f"score={valid_scores.get(best_p,float('nan')):.4f})")

    if log:
        for line in lines:
            log(line)

    return best_p, scores, stable_region, lines


# ===========================================================================
# PLOT
# ===========================================================================

PALETTE = {
    "hyponymy"  : "#d32f2f",
    "meronymy"  : "#e65100",
    "capital"   : "#2e7d32",
    "sibling"   : "#1565c0",
    "coordinate": "#6a1b9a",
    "gangle"    : "#ef9a9a",
    "cosine"    : "#90caf9",
}

def _style():
    plt.rcParams.update({
        "figure.facecolor"  : "white",
        "axes.facecolor"    : "#f8f9fa",
        "axes.edgecolor"    : "#aaaaaa",
        "axes.labelcolor"   : "#222222",
        "axes.labelsize"    : 9,
        "axes.titlesize"    : 9,
        "axes.titlepad"     : 8,
        "text.color"        : "#222222",
        "xtick.color"       : "#444444",
        "ytick.color"       : "#444444",
        "xtick.labelsize"   : 8,
        "ytick.labelsize"   : 8,
        "grid.color"        : "#dddddd",
        "grid.alpha"        : 0.7,
        "axes.grid"         : True,
        "legend.facecolor"  : "white",
        "legend.edgecolor"  : "#cccccc",
        "legend.fontsize"   : 7.5,
        "legend.framealpha" : 0.9,
        "figure.dpi"        : 110,
        "savefig.dpi"       : 300,
        "savefig.bbox"      : "tight",
        "savefig.facecolor" : "white",
        "font.family"       : "serif",
        "font.serif"        : ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset"  : "stix",
    })

def plot_delta_distributions(all_results):
    """5-panel histogram: Δθ distribution per relation type."""
    _style()
    datasets = [
        ("hyponymy",   "Hyponymy\n(is-a)"),
        ("meronymy",   "Meronymy\n(part-of)"),
        ("capital",    "Capital–Country\n(asymmetric)"),
        ("sibling",    "Sibling-Sym\n(co-hyponym)"),
        ("coordinate", "Coordinate-Sym\n(thematic)"),
    ]
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.8),
                             constrained_layout=True)

    for ax, (key, title) in zip(axes, datasets):
        if key not in all_results or all_results[key]["n"] == 0:
            ax.text(0.5, 0.5, "No data yet", ha="center", va="center",
                    transform=ax.transAxes, color="#999", fontsize=8)
            ax.set_title(title, fontsize=9, pad=6)
            continue

        r     = all_results[key]
        d     = r["deltas"]
        st    = r["stats"]
        color = PALETTE.get(key, "#555555")

        ax.hist(d, bins=35, color=color, edgecolor="white",
                linewidth=0.3, alpha=0.85)
        ax.axvline(0, color="#333333", lw=1.4, linestyle="--",
                   label=r"$\Delta\theta=0$ / $\Delta$cos=0")
        ax.axvline(st["mean_dt"], color="#c0392b", lw=1.4, linestyle="-",
                   label=fr"$\mu$={st['mean_dt']:+.3f}")
        ax.set_xlabel(r"$\Delta\theta$  [rad]", fontsize=8)
        ax.set_ylabel("Frequency", fontsize=8)
        acc_str = f"Acc={st['dir_acc']:.0%}"
        ax.set_title(f"{title}\n{acc_str}   n={st['n']:,}",
                     fontsize=8.5, pad=5)
        ax.legend(loc="upper right", fontsize=6.5)
        # Remove top/right spines for cleaner look
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    return fig

def plot_comparison_bars(all_results, rand_results=None):
    """Bar chart: G-Angle direction accuracy vs Cosine baseline vs Random embedding.

    NEW (v15): rand_results (dict key→evaluate_pairs result) adds a third bar group
    showing random-embedding baseline berdampingan dengan real embedding.
    Hipotesis:
      • real dir_acc >> 50%  AND  rand dir_acc ≈ 50%  → signal is SEMANTIC
      • rand mean|Δθ| > 0                              → asymmetry juga GEOMETRIC
    """
    _style()
    datasets = [
        ("hyponymy",   "Hyponymy",   "asymmetric"),
        ("meronymy",   "Meronymy",   "asymmetric"),
        ("capital",    "Capital",    "asymmetric"),
        ("sibling",    "Sibling",    "symmetric"),
        ("coordinate", "Coordinate", "symmetric"),
    ]
    present = [(k, l, e) for k, l, e in datasets
               if k in all_results and all_results[k]["n"] > 0]
    if not present:
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        ax.text(0.5, 0.5, "No results yet", ha="center", va="center",
                transform=ax.transAxes)
        return fig

    has_rand = bool(rand_results and any(
        k in rand_results and rand_results[k].get("n", 0) > 0
        for k, _, _ in present))

    names    = [l for _, l, _ in present]
    accs_ga  = [all_results[k]["stats"]["dir_acc"] * 100 for k, _, _ in present]
    abs_dts  = [all_results[k]["stats"]["mean_abs_dt"]   for k, _, _ in present]
    stds     = [all_results[k]["stats"]["std_dt"]        for k, _, _ in present]
    cols     = [PALETTE.get(k, "#555") for k, _, _ in present]

    if has_rand:
        accs_rand   = []
        abs_dts_rnd = []
        for k, _, _ in present:
            rr = rand_results.get(k)
            if rr and rr.get("n", 0) > 0:
                accs_rand.append(rr["stats"]["dir_acc"] * 100)
                abs_dts_rnd.append(rr["stats"]["mean_abs_dt"])
            else:
                accs_rand.append(float("nan"))
                abs_dts_rnd.append(float("nan"))

    x = np.arange(len(names))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.8),
                                   constrained_layout=True)

    # ── Panel 1: Direction Accuracy ──────────────────────────────────────
    if has_rand:
        # Three groups: Real  |  Random  |  Cosine(50%)
        w = 0.25
        bars_ga  = ax1.bar(x - w, accs_ga, w, color=cols,
                           edgecolor="white", lw=0.5, label="Real Embedding (G-Angle)", zorder=3)
        bars_rnd = ax1.bar(x,     accs_rand, w, color="#7b1fa2",
                           edgecolor="white", lw=0.5, alpha=0.75,
                           label="Random N(μ,σ²) Embedding", zorder=3)
        ax1.bar(x + w, [50.0]*len(x), w,
                color="#9e9e9e", edgecolor="white", lw=0.5,
                alpha=0.60, label="Cosine baseline (50%)", zorder=3)
        for bar, v in zip(bars_rnd, accs_rand):
            if not np.isnan(v):
                ax1.text(bar.get_x() + bar.get_width()/2,
                         bar.get_height() + 1.2,
                         f"{v:.1f}%", ha="center", va="bottom",
                         fontsize=7.5, color="#6a1b9a")
    else:
        w = 0.38
        bars_ga  = ax1.bar(x - w/2, accs_ga, w, color=cols,
                           edgecolor="white", lw=0.5, label="G-Angle", zorder=3)
        ax1.bar(x + w/2, [50.0]*len(x), w,
                color="#9e9e9e", edgecolor="white", lw=0.5,
                alpha=0.65, label="Cosine (baseline = 50%)", zorder=3)

    ax1.axhline(50, color="#555", lw=1, linestyle="--", zorder=2)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, fontsize=9)
    ax1.set_ylim(0, 115)
    ax1.set_ylabel("Direction Accuracy (%)", fontsize=9)
    title1 = ("Direction Accuracy\nReal  vs  Random  vs  Cosine"
               if has_rand else
               "Direction Accuracy\nG-Angle  vs  Cosine (random baseline)")
    ax1.set_title(title1, fontsize=9)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    for bar, v in zip(bars_ga, accs_ga):
        ax1.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 1.5,
                 f"{v:.1f}%", ha="center", va="bottom",
                 fontsize=8.5, fontweight="bold", color="#222")
    ax1.legend(fontsize=8, loc="upper right")

    # ── Panel 2: Mean |Δθ| (asymmetry signal) ────────────────────────────
    if has_rand:
        w2 = 0.28
        bars2 = ax2.bar(x - w2, abs_dts, w2, color=cols,
                        edgecolor="white", lw=0.5,
                        label=r"Real: G-Angle mean $|\Delta\theta|$", zorder=3)
        bars2r = ax2.bar(x,     abs_dts_rnd, w2, color="#7b1fa2",
                         edgecolor="white", lw=0.5, alpha=0.70,
                         label=r"Random: mean $|\Delta\theta|$", zorder=3)
        ax2.bar(x + w2, [0]*len(x), w2,
                color="#9e9e9e", edgecolor="white", lw=0.5,
                alpha=0.60, label=r"Cosine $\Delta \equiv 0$", zorder=3)
        ax2.errorbar(x - w2, abs_dts, yerr=stds,
                     fmt="none", color="#444", capsize=3, lw=1, zorder=4)
        for bar, v in zip(bars2r, abs_dts_rnd):
            if not np.isnan(v):
                ax2.text(bar.get_x() + bar.get_width()/2,
                         bar.get_height() + max([a for a in abs_dts if a == a] + [0.001]) * 0.02,
                         f"{v:.3f}", ha="center", va="bottom",
                         fontsize=7.5, color="#6a1b9a")
        note = ("↑ rand |Δθ|>0 = geometric signal\n"
                "↓ real>>rand dir_acc = semantic signal")
        ax2.text(0.02, 0.97, note, transform=ax2.transAxes, fontsize=7,
                 color="#444", va="top",
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))
    else:
        w2 = 0.38
        bars2 = ax2.bar(x - w2/2, abs_dts, w2, color=cols,
                        edgecolor="white", lw=0.5,
                        label=r"G-Angle mean $|\Delta\theta|$", zorder=3)
        ax2.bar(x + w2/2, [0]*len(x), w2,
                color="#9e9e9e", edgecolor="white", lw=0.5,
                alpha=0.65, label=r"Cosine $\Delta \equiv 0$ (symmetric)", zorder=3)
        ax2.errorbar(x - w2/2, abs_dts, yerr=stds,
                     fmt="none", color="#444", capsize=3, lw=1, zorder=4)

    ax2.set_xticks(x)
    ax2.set_xticklabels(names, fontsize=9)
    ax2.set_ylabel(r"Mean $|\Delta\theta|$  [rad]", fontsize=9)
    title2 = (r"Asymmetry Signal: Real  vs  Random  vs  Cosine"
               if has_rand else
               r"Asymmetry Signal Strength"
               "\n" r"G-Angle mean $|\Delta\theta|$  vs  Cosine $\Delta \equiv 0$")
    ax2.set_title(title2, fontsize=9)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    for bar, v in zip(bars2, abs_dts):
        ax2.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + max(abs_dts)*0.02,
                 f"{v:.3f}", ha="center", va="bottom",
                 fontsize=8.5, fontweight="bold", color="#222")
    ax2.legend(fontsize=8, loc="upper right")

    return fig

def plot_scatter_gangle_cosine(all_results, key):
    """Detail scatter: A_g(u→v) vs A_g(v→u), and Cosine vs Δθ."""
    _style()
    if key not in all_results or all_results[key]["n"] == 0:
        return None

    r     = all_results[key]
    label = r["label"]
    color = PALETTE.get(key, "#555555")
    st    = r["stats"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5),
                                   constrained_layout=True)

    # ── Scatter A_g(u→v) vs A_g(v→u) ─────────────────────────────────────
    uv = r["g_angles_uv"]
    vu = r["g_angles_vu"]
    sc = ax1.scatter(uv, vu, alpha=0.25, s=6,
                     c=r["cosines"], cmap="RdYlGn",
                     vmin=0, vmax=1, rasterized=True)
    cb = plt.colorbar(sc, ax=ax1)
    cb.set_label("cos(u, v)", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    mn = min(uv.min(), vu.min()) - 0.01
    mx = max(uv.max(), vu.max()) + 0.01
    ax1.plot([mn, mx], [mn, mx], "--", color="#555", lw=1,
             label=r"Symmetry line ($\Delta\theta=0$)")
    ax1.set_xlabel(r"$A_g(u \to v)$  [rad]", fontsize=9)
    ax1.set_ylabel(r"$A_g(v \to u)$  [rad]", fontsize=9)
    ax1.set_title(
        r"$A_g(u \to v)$  vs  $A_g(v \to u)$" + "\n"
        f"Points above line → $\\Delta\\theta < 0$  (u is more specific)\n"
        f"Dir Acc = {st['dir_acc']:.1%}   Mean $\\Delta\\theta$ = {st['mean_dt']:+.4f}",
        fontsize=8.5)
    ax1.legend(fontsize=8)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # ── Scatter Cosine vs Δθ ──────────────────────────────────────────────
    rho, p_rho = spearmanr(r["cosines"], r["deltas"])
    ax2.scatter(r["cosines"], r["deltas"], alpha=0.25, s=6,
                color=color, rasterized=True)
    ax2.axhline(0, color="#555", lw=1, linestyle="--",
                label=r"$\Delta\theta=0$")
    ax2.axvline(np.mean(r["cosines"]), color="#c0392b", lw=1, linestyle=":",
                label=f"mean cos = {np.mean(r['cosines']):.3f}")
    ax2.set_xlabel("Cosine similarity  cos(u, v)", fontsize=9)
    ax2.set_ylabel(r"G-Angle $\Delta\theta$  [rad]", fontsize=9)
    ax2.set_title(
        r"Cosine  vs  $\Delta\theta$" + "\n"
        f"Spearman $\\rho$ = {rho:+.4f}  (p = {p_rho:.2e})\n"
        r"Cosine has no $\Delta$ — cannot distinguish direction",
        fontsize=8.5)
    ax2.legend(fontsize=8)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    return fig

def plot_summary_radar(all_results):
    """Summary: direction accuracy bar chart (G-Angle vs Cosine)."""
    _style()
    datasets = [
        ("hyponymy",   "Hyponymy\n(hierarchy)"),
        ("meronymy",   "Meronymy\n(part-of)"),
        ("capital",    "Capital\n(asymmetric)"),
        ("sibling",    "Sibling\n(symmetric)"),
        ("coordinate", "Coordinate\n(symmetric)"),
    ]
    present = [(k, l) for k, l in datasets
               if k in all_results and all_results[k]["n"] > 0]
    if not present:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No results yet", ha="center", va="center",
                transform=ax.transAxes)
        return fig

    labels   = [l for _, l in present]
    ga_accs  = [all_results[k]["stats"]["dir_acc"] * 100 for k, _ in present]
    col_list = [PALETTE.get(k, "#555") for k, _ in present]
    x        = np.arange(len(labels))
    w        = 0.38

    fig, ax1 = plt.subplots(1, 1, figsize=(9, 5))
    

    bars_ga  = ax1.bar(x - w/2, ga_accs, w, color=col_list,
                       edgecolor="white", lw=0.5, label="G-Angle", zorder=3)
    ax1.bar(x + w/2, [50.0]*len(x), w,
            color="#9e9e9e", edgecolor="white", lw=0.5,
            alpha=0.65, label="Cosine (baseline = 50%)", zorder=3)
    ax1.axhline(50, color="#555", lw=1, linestyle="--", zorder=2,
                label="Random chance (50%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylim(0, 112)
    ax1.set_ylabel("Direction Accuracy (%)", fontsize=10)
    ax1.set_title("")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    for bar, v in zip(bars_ga, ga_accs):
        ax1.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 1.2,
                 f"{v:.1f}%", ha="center", va="bottom",
                 fontsize=9, fontweight="bold", color="#222")
    ax1.legend(fontsize=9, loc="upper right")

    return fig

def plot_parameter_sweep(sweep_results, best_p=None, stable_region=None, scores=None):
    """
    3-panel plot: Direction Accuracy, Ranking Accuracy, and Mean |Δθ|
    for each p value ∈ {1.5,2,3,5,10} per relation.

    NEW (v16): analytically-selected best_p and stable_region are annotated
    on all panels.  A fourth mini-panel shows the composite score per p.

    sweep_results : dict[p_val] → dict[key] → result
    best_p        : float — auto-selected optimal p (from select_best_p)
    stable_region : list[float] — p values in stable plateau
    scores        : dict[p] → float — composite score per p
    """
    _style()
    p_vals = sorted(sweep_results.keys())
    if not p_vals:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No sweep results yet", ha="center", va="center",
                transform=ax.transAxes)
        return fig

    # Run analytical selection if not provided
    if best_p is None or stable_region is None or scores is None:
        best_p, scores, stable_region, _ = select_best_p(sweep_results)

    datasets = [
        ("hyponymy",   "Hyponymy",   "asymmetric"),
        ("meronymy",   "Meronymy",   "asymmetric"),
        ("capital",    "Capital",    "asymmetric"),
        ("sibling",    "Sibling",    "symmetric"),
        ("coordinate", "Coordinate", "symmetric"),
    ]

    # ── Helper: shade stable region on an axis ────────────────────────────────
    def _shade_stable(ax, y_unit="pct"):
        """Draw stable-region band and best-p line on any axis."""
        if stable_region and len(stable_region) >= 2:
            ax.axvspan(min(stable_region), max(stable_region),
                       alpha=0.10, color="#43a047",
                       label=f"Stable region p∈{stable_region}")
        elif stable_region and len(stable_region) == 1:
            ax.axvline(stable_region[0], color="#43a047", lw=1.2,
                       linestyle=":", alpha=0.6)
        if best_p is not None:
            ax.axvline(best_p, color="#e53935", lw=2.0, linestyle="-.",
                       zorder=5, label=f"Best p={best_p}")

    fig, axes = plt.subplots(1, 4, figsize=(23, 5), constrained_layout=True)

    # ── Panel 1: Direction Accuracy vs p ─────────────────────────────────────
    ax1 = axes[0]
    for key, lbl, _ in datasets:
        accs = []
        for pv in p_vals:
            r = sweep_results[pv].get(key)
            if r and r["n"] > 0 and not r["stats"].get("is_degenerate", False):
                accs.append(r["stats"]["dir_acc"] * 100)
            else:
                accs.append(None)
        valid = [(pv, a) for pv, a in zip(p_vals, accs) if a is not None]
        if valid:
            xs, ys = zip(*valid)
            ax1.plot(xs, ys, marker="o", label=lbl,
                     color=PALETTE.get(key, "#555"), linewidth=1.8)
    ax1.axhline(50, color="#aaa", lw=1, linestyle="--", label="Random (50%)")
    _shade_stable(ax1, "pct")
    ax1.set_xlabel("p value (ℓᵖ norm)", fontsize=9)
    ax1.set_ylabel("Direction Accuracy (%)", fontsize=9)
    ax1.set_title(f"Direction Accuracy vs p\n"
                  f"Best p={best_p}  (auto-selected, red dash-dot)",
                  fontsize=9)
    ax1.set_xticks(p_vals)
    ax1.set_ylim(0, 110)
    ax1.legend(fontsize=7.5, loc="lower right")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # ── Panel 2: Ranking Accuracy vs p ───────────────────────────────────────
    ax2 = axes[1]
    for key, lbl, _ in datasets:
        ranks = []
        for pv in p_vals:
            r = sweep_results[pv].get(key)
            if r and r["n"] > 0 and not r["stats"].get("is_degenerate", False):
                ranks.append(r["stats"]["ranking_acc"] * 100)
            else:
                ranks.append(None)
        valid = [(pv, a) for pv, a in zip(p_vals, ranks) if a is not None]
        if valid:
            xs, ys = zip(*valid)
            ax2.plot(xs, ys, marker="s", label=lbl,
                     color=PALETTE.get(key, "#555"), linewidth=1.8, linestyle="--")
    ax2.axhline(50, color="#aaa", lw=1, linestyle="--", label="Random (50%)")
    _shade_stable(ax2, "pct")
    ax2.set_xlabel("p value (ℓᵖ norm)", fontsize=9)
    ax2.set_ylabel("Ranking Accuracy (%)", fontsize=9)
    ax2.set_title("Ranking Accuracy vs p\n(Parameter Sweep)", fontsize=9)
    ax2.set_xticks(p_vals)
    ax2.set_ylim(0, 110)
    ax2.legend(fontsize=7.5, loc="lower right")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # ── Panel 3: Mean |Δθ| vs p ───────────────────────────────────────────────
    ax3 = axes[2]
    for key, lbl, exp in datasets:
        abs_dts = []
        for pv in p_vals:
            r = sweep_results[pv].get(key)
            if r and r["n"] > 0 and not r["stats"].get("is_degenerate", False):
                abs_dts.append(r["stats"]["mean_abs_dt"])
            else:
                abs_dts.append(None)
        valid = [(pv, a) for pv, a in zip(p_vals, abs_dts) if a is not None]
        if valid:
            xs, ys = zip(*valid)
            ls = "-" if exp == "asymmetric" else ":"
            ax3.plot(xs, ys, marker="^", label=lbl,
                     color=PALETTE.get(key, "#555"), linewidth=1.8, linestyle=ls)
    ax3.axvline(2.0, color="#aaa", lw=1.2, linestyle="--",
                label="p=2 (Euclidean, degenerate)")
    _shade_stable(ax3)
    ax3.set_xlabel("p value (ℓᵖ norm)", fontsize=9)
    ax3.set_ylabel(r"Mean $|\Delta\theta|$  [rad]", fontsize=9)
    ax3.set_title(r"Mean $|\Delta\theta|$ vs p" +
                  "\n(Asymmetric=solid, Symmetric=dotted)", fontsize=9)
    ax3.set_xticks(p_vals)
    ax3.legend(fontsize=7.5, loc="upper left")
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)

    # ── Panel 4: Composite Score vs p  [NEW v16] ─────────────────────────────
    ax4 = axes[3]
    sc_xs = [pv for pv in p_vals if not np.isnan(scores.get(pv, float("nan")))]
    sc_ys = [scores[pv] for pv in sc_xs]
    if sc_xs:
        bar_colors = []
        for pv in sc_xs:
            if pv == best_p:
                bar_colors.append("#e53935")      # best → red
            elif pv in (stable_region or []):
                bar_colors.append("#43a047")      # stable → green
            else:
                bar_colors.append("#78909c")      # others → grey
        bars = ax4.bar(sc_xs, sc_ys, color=bar_colors, edgecolor="white",
                       lw=0.6, zorder=3, width=0.35)
        for bar, pv, sc in zip(bars, sc_xs, sc_ys):
            ax4.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + max(sc_ys) * 0.02,
                     f"{sc:.3f}", ha="center", va="bottom",
                     fontsize=8, fontweight="bold",
                     color="#e53935" if pv == best_p else "#333")
        # Mark degenerate p values on x-axis
        degen_ps = [pv for pv in p_vals if np.isnan(scores.get(pv, float("nan")))]
        for pv in degen_ps:
            ax4.text(pv, max(sc_ys) * 0.05, "✗\ndegen.", ha="center",
                     va="bottom", fontsize=7, color="#aaa")
        if stable_region and len(stable_region) >= 2:
            ax4.axvspan(min(stable_region), max(stable_region),
                        alpha=0.10, color="#43a047")
        if best_p is not None:
            ax4.axvline(best_p, color="#e53935", lw=2.0, linestyle="-.", zorder=5)
    ax4.set_xlabel("p value (ℓᵖ norm)", fontsize=9)
    ax4.set_ylabel("Composite Score\n(dir_acc + 0.3·|Δθ|_norm)", fontsize=8)
    ax4.set_title(f"Auto-Selected Best p = {best_p}\n"
                  f"Score = dir_acc_asym + 0.3·|Δθ|_norm  [v16]",
                  fontsize=9)
    ax4.set_xticks(p_vals)
    ax4.spines["top"].set_visible(False)
    ax4.spines["right"].set_visible(False)

    # ── Suptitle ──────────────────────────────────────────────────────────────
    stable_str = (f"p ∈ {stable_region}" if stable_region
                  else "no stable plateau found")
    fig.suptitle(
        f"Parameter Sweep Analysis  [v16]  —  "
        f"Auto-selected: p = {best_p}  |  Stable region: {stable_str}",
        fontsize=10, fontweight="bold", y=1.01)

    return fig


def plot_permutation_test(all_results):
    """
    Permutation null distributions per relation.

    Asymmetric:
      Blue  (A): orientation permutation — flip pair direction with p=0.5
      Orange(B): pair shuffle (intra-pool) — reshuffle which u pairs with which v
      Red vertical line = observed direction accuracy (%)
      Chance line = 50%

    Symmetric (v14 redesign):
      Perm A: SKIPPED — orientation flip is invariant for |Δθ| (σ=0 by construction)
      Orange(B): cross-pool shuffle — null mean|Δθ| distribution
      Red vertical line = observed mean|Δθ| of real pairs
      Chance line = null mean (expected value under H0)
      Lower observed mean → real pairs are more symmetric than null
    """
    _style()
    datasets = [
        ("hyponymy",   "Hyponymy\n(is-a)",          "asymmetric"),
        ("meronymy",   "Meronymy\n(part-of)",        "asymmetric"),
        ("capital",    "Capital–Country\n(asymm.)",  "asymmetric"),
        ("sibling",    "Sibling-Sym\n(co-hyponym)",  "symmetric"),
        ("coordinate", "Coordinate-Sym\n(thematic)", "symmetric"),
    ]
    fig, axes = plt.subplots(1, 5, figsize=(18, 4.0), constrained_layout=True)
    for ax, (key, title, exp) in zip(axes, datasets):
        if key not in all_results or all_results[key].get("n", 0) == 0 \
                or "perm_pair_accs" not in all_results[key]:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, color="#999", fontsize=8)
            ax.set_title(title, fontsize=9)
            continue
        r      = all_results[key]
        pm_a   = r.get("perm_means", np.array([]))       # orient dist (asymm only)
        pm_b   = r["perm_pair_accs"]                      # pair / cross-pool dist
        p_orient = r["stats"].get("perm_p_orient", float("nan"))
        p_pair   = r["stats"].get("perm_p_pair",   r["stats"]["perm_p"])
        color    = PALETTE.get(key, "#555")
        bins     = 30

        if exp == "asymmetric":
            obs = r.get("perm_obs_stat", r["stats"]["dir_acc"])
            if len(pm_a) > 0:
                ax.hist(pm_a * 100, bins=bins, color="#90caf9", edgecolor="white",
                        linewidth=0.3, alpha=0.75,
                        label=f"A orient  p={p_orient:.3f}")
            ax.hist(pm_b * 100, bins=bins, color="#ffb74d", edgecolor="white",
                    linewidth=0.3, alpha=0.65,
                    label=f"B pairs   p={p_pair:.3f}")
            ax.axvline(obs * 100, color=color, lw=2, linestyle="-",
                       label=f"Observed {obs:.1%}")
            ax.axvline(50, color="#333", lw=1, linestyle="--", label="Chance 50%")
            ax.set_xlabel("Direction Accuracy (%)", fontsize=8)
            ax.set_ylabel("Frequency", fontsize=8)
            p_str = f"A p={'N/A' if np.isnan(p_orient) else f'{p_orient:.3f}'}  B p={p_pair:.3f}"

        else:
            # Symmetric: pm_b contains mean|Δθ| per permutation iteration
            obs_sym = r.get("perm_obs_stat_sym",
                            r.get("perm_obs_stat", float("nan")))
            null_mean = float(np.mean(pm_b)) if len(pm_b) > 0 else float("nan")
            ax.hist(pm_b, bins=bins, color="#ffb74d", edgecolor="white",
                    linewidth=0.3, alpha=0.75,
                    label=f"B cross-pool  p={p_pair:.3f}")
            if not np.isnan(obs_sym):
                ax.axvline(obs_sym, color=color, lw=2, linestyle="-",
                           label=f"Obs mean|Δθ| {obs_sym:.4f}")
            if not np.isnan(null_mean):
                ax.axvline(null_mean, color="#333", lw=1, linestyle="--",
                           label=f"Null mean {null_mean:.4f}")
            ax.set_xlabel("mean|Δθ| per iteration", fontsize=8)
            ax.set_ylabel("Frequency", fontsize=8)
            p_str = f"B(cross-pool) p={p_pair:.3f}"
            note = "Perm A skipped (|Δθ| flip-invariant)"
            ax.text(0.03, 0.97, note, transform=ax.transAxes, fontsize=6,
                    color="#888", va="top")

        ax.set_title(f"{title}\n{p_str}", fontsize=8, pad=5)
        ax.legend(fontsize=6, loc="upper right")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    return fig


def plot_swap_test(all_results):
    """
    v14: Swap Test tab replaced by Symmetry Score plot.

    Shows the |Δθ| distribution for each relation:
      • Asymmetric relations: histogram shifted toward larger |Δθ| (high asymmetry)
      • Symmetric  relations: histogram concentrated near 0 (low asymmetry)

    Also shows:
      • Vertical τ line (SYM_TAU threshold)
      • dir_acc = symmetry_score = P(|Δθ| < τ) for symmetric
      • dir_acc = direction_acc = P(Δθ < 0) for asymmetric
    """
    _style()
    datasets = [
        ("hyponymy",   "Hyponymy",   "asymmetric"),
        ("meronymy",   "Meronymy",   "asymmetric"),
        ("capital",    "Capital",    "asymmetric"),
        ("sibling",    "Sibling",    "symmetric"),
        ("coordinate", "Coordinate", "symmetric"),
    ]
    present = [(k, l, e) for k, l, e in datasets
               if k in all_results and all_results[k].get("n", 0) > 0]
    if not present:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, "No results yet.\nRun evaluation first.",
                ha="center", va="center", transform=ax.transAxes, fontsize=11)
        ax.set_title("Symmetry Score — |Δθ| Distribution per Relation")
        return fig

    ncols = min(len(present), 3)
    nrows = (len(present) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5.5 * ncols, 4 * nrows),
                             constrained_layout=True)
    axes_flat = np.array(axes).flatten() if len(present) > 1 else [axes]

    for ax, (k, lbl, exp) in zip(axes_flat, present):
        r  = all_results[k]
        d  = np.array(r["deltas"])
        st = r["stats"]
        tau = st.get("sym_tau", 0.05)

        abs_d = np.abs(d)
        color = PALETTE.get(k, "#888")

        ax.hist(abs_d, bins=40, color=color, alpha=0.75, edgecolor="white",
                linewidth=0.4, density=True)

        if exp == "symmetric":
            ax.axvline(tau, color="#e53935", lw=1.8, linestyle="--",
                       label=f"τ = {tau:.3f} rad")
            sym_score = st["dir_acc"]
            ax.text(0.97, 0.93,
                    f"Symmetry Score\nP(|Δθ|<τ) = {sym_score:.1%}",
                    ha="right", va="top", transform=ax.transAxes,
                    fontsize=9, color="#e53935",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
            type_tag = "[Symmetric]"
        else:
            dir_a = st["dir_acc"]
            mean_d = st["mean_dt"]
            ax.text(0.97, 0.93,
                    f"Dir Acc P(Δθ<0)\n= {dir_a:.1%}\nmean|Δθ| = {np.mean(abs_d):.4f}",
                    ha="right", va="top", transform=ax.transAxes,
                    fontsize=9, color="#1565c0",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
            type_tag = "[Asymmetric]"

        ax.set_xlabel("|Δθ| (radians)", fontsize=9)
        ax.set_ylabel("Density", fontsize=9)
        ax.set_title(f"{lbl}  {type_tag}", fontsize=10, fontweight="bold")
        if exp == "symmetric":
            ax.legend(fontsize=8, loc="upper right")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Hide unused axes
    for ax in axes_flat[len(present):]:
        ax.set_visible(False)

    fig.suptitle(
        "|Δθ| Distribution per Relation  —  Symmetry Score replaces Swap Test\n"
        "Swap consistency is always 100% (mathematical identity); "
        "this plot shows genuine embedding structure.",
        fontsize=10, y=1.02)
    return fig

def plot_ranking_accuracy(all_results, rand_results=None):
    """
    Bar chart Ranking Accuracy vs Direction Accuracy vs Cosine baseline.

    NEW (v15): rand_results adds dir_acc and rank_acc bars for randomized
    embedding — side-by-side pembanding eksplisit di evaluasi inti.
    """
    _style()
    datasets = [
        ("hyponymy",   "Hyponymy",   "asymmetric"),
        ("meronymy",   "Meronymy",   "asymmetric"),
        ("capital",    "Capital",    "asymmetric"),
        ("sibling",    "Sibling",    "symmetric"),
        ("coordinate", "Coordinate", "symmetric"),
    ]
    present = [(k, l, e) for k, l, e in datasets
               if k in all_results and all_results[k].get("n", 0) > 0]
    if not present:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No results yet", ha="center", va="center",
                transform=ax.transAxes)
        return fig

    has_rand = bool(rand_results and any(
        k in rand_results and rand_results[k].get("n", 0) > 0
        for k, _, _ in present))

    names     = [l for _, l, _ in present]
    dir_accs  = [all_results[k]["stats"]["dir_acc"] * 100 for k, _, _ in present]
    rank_accs = [all_results[k]["stats"]["ranking_acc"] * 100 for k, _, _ in present]
    cols      = [PALETTE.get(k, "#555") for k, _, _ in present]
    x = np.arange(len(names))

    if has_rand:
        dir_rnd  = []
        rank_rnd = []
        for k, _, _ in present:
            rr = rand_results.get(k)
            if rr and rr.get("n", 0) > 0:
                dir_rnd.append(rr["stats"]["dir_acc"] * 100)
                rank_rnd.append(rr["stats"]["ranking_acc"] * 100)
            else:
                dir_rnd.append(float("nan"))
                rank_rnd.append(float("nan"))
        w = 0.18
        fig, ax = plt.subplots(figsize=(14, 5.5), constrained_layout=True)
        b1 = ax.bar(x - 2*w, dir_accs,  w, color=cols, edgecolor="white",
                    lw=0.5, label="Real: Direction Acc", zorder=3)
        b2 = ax.bar(x - w,   rank_accs, w, color=cols, edgecolor="white",
                    lw=0.5, alpha=0.6, label="Real: Ranking Acc", zorder=3, hatch="//")
        b3 = ax.bar(x,       dir_rnd,   w, color="#7b1fa2", edgecolor="white",
                    lw=0.5, alpha=0.75, label="Random: Direction Acc", zorder=3)
        b4 = ax.bar(x + w,   rank_rnd,  w, color="#7b1fa2", edgecolor="white",
                    lw=0.5, alpha=0.45, label="Random: Ranking Acc", zorder=3, hatch="\\\\")
        ax.bar(x + 2*w, [50.0]*len(x), w,
               color="#9e9e9e", edgecolor="white", lw=0.5,
               alpha=0.50, label="Cosine baseline (50%)", zorder=3)
        for bar, v in zip(b3, dir_rnd):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 1.0,
                        f"{v:.1f}%", ha="center", va="bottom",
                        fontsize=6.5, color="#6a1b9a")
        for bar, v in zip(b4, rank_rnd):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 1.0,
                        f"{v:.1f}%", ha="center", va="bottom",
                        fontsize=6.5, color="#6a1b9a")
        ax.set_title(
            "Accuracy: Real Embedding  vs  Random Embedding  vs  Cosine\n"
            "Real dir_acc >> Random dir_acc → signal is semantic (not geometric)",
            fontsize=9)
    else:
        w = 0.28
        fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
        b1 = ax.bar(x - w, dir_accs,  w, color=cols, edgecolor="white",
                    lw=0.5, label="Direction Accuracy", zorder=3)
        b2 = ax.bar(x,     rank_accs, w, color=cols, edgecolor="white",
                    lw=0.5, alpha=0.6, label="Ranking Accuracy", zorder=3, hatch="//")
        ax.bar(x + w, [50.0] * len(x), w,
               color="#9e9e9e", edgecolor="white", lw=0.5,
               alpha=0.55, label="Cosine baseline (50%)", zorder=3)
        ax.set_title(
            "Ranking Accuracy vs Direction Accuracy vs Cosine Baseline\n"
            "Ranking: proportion of real pairs scoring better than random pairs",
            fontsize=9)

    ax.axhline(50, color="#555", lw=1, linestyle="--", zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylim(0, 120)
    ax.set_ylabel("Accuracy (%)", fontsize=9)
    for bar, v in zip(b1, dir_accs):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.2,
                f"{v:.1f}%", ha="center", va="bottom",
                fontsize=7.5, fontweight="bold", color="#222")
    for bar, v in zip(b2, rank_accs):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.2,
                f"{v:.1f}%", ha="center", va="bottom",
                fontsize=7.5, color="#444")
    ax.legend(fontsize=8, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return fig


def plot_error_analysis(all_results, key):
    """
    2-panel error analysis plot.

    Panel 1 (scatter): all pairs coloured by error sub-class
      • green  = correct
      • orange = near_zero_ambiguity  (|Δθ| < NEAR_ZERO_TAU)
      • red    = false_direction      (clear wrong-direction error)
      • purple = extreme_outlier      (|Δθ| > outlier threshold)

    Panel 2 (bar): error breakdown counts — false_direction / near_zero / outlier
    """
    _style()
    if key not in all_results or all_results[key].get("n", 0) == 0:
        return None
    r      = all_results[key]
    errors = r.get("error_pairs", [])
    if not errors:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, "No error pairs — perfect direction detection!",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=12, color="#2e7d32")
        ax.set_title(f"Error Analysis — {r['label']}", fontsize=10)
        return fig

    expected  = r["expected"]
    d_all     = r["deltas"]
    c_all     = r["cosines"]
    pairs_ok  = r["pairs_ok"]
    bkd       = r.get("error_breakdown",
                      {"false_direction": 0, "near_zero_ambiguity": 0,
                       "extreme_outlier": 0, "near_zero_tau": 0.02, "outlier_tau": 0.0})

    # Build per-point colour arrays
    # Build lookup: (word1,word2) → sub_class
    sc_map = {(e["word1"], e["word2"]): e.get("sub_class", "false_direction")
              for e in errors}

    colours   = []
    subclasses = []
    for (u, v), dt in zip(pairs_ok, d_all):
        sc = sc_map.get((u, v), None)
        subclasses.append(sc)
        if sc is None:
            colours.append("correct")
        else:
            colours.append(sc)

    colour_map = {
        "correct"             : ("#43a047", 0.18, 5,  "Correct"),
        "false_direction"     : ("#e53935", 0.60, 14, "False direction"),
        "near_zero_ambiguity" : ("#fb8c00", 0.70, 14, "Near-zero ambiguity"),
        "extreme_outlier"     : ("#7b1fa2", 0.80, 18, "Extreme outlier"),
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5),
                                   gridspec_kw={"width_ratios": [2.2, 1]},
                                   constrained_layout=True)

    # ── Panel 1: scatter ──────────────────────────────────────────────────────
    for cat, (col, al, sz, lbl) in colour_map.items():
        mask = np.array([sc == cat if cat != "correct" else sc is None
                         for sc in subclasses])
        if mask.sum() == 0:
            continue
        ax1.scatter(c_all[mask], d_all[mask],
                    alpha=al, s=sz, color=col, label=f"{lbl} ({mask.sum()})",
                    rasterized=True, zorder=3 if cat != "correct" else 2)

    ax1.axhline(0, color="#555", lw=1.2, linestyle="--", label="Δθ=0")

    # threshold lines if asymmetric
    if expected == "asymmetric":
        tau_nz = bkd.get("near_zero_tau", 0.02)
        tau_ol = bkd.get("outlier_tau", None)
        ax1.axhline(tau_nz,  color="#fb8c00", lw=0.8, linestyle=":",
                    label=f"near-zero thresh +{tau_nz:.3f}")
        ax1.axhline(-tau_nz, color="#fb8c00", lw=0.8, linestyle=":")
        if tau_ol:
            ax1.axhline(tau_ol,  color="#7b1fa2", lw=0.8, linestyle=":",
                        label=f"outlier thresh +{tau_ol:.3f}")
            ax1.axhline(-tau_ol, color="#7b1fa2", lw=0.8, linestyle=":")

    ax1.set_xlabel("Cosine similarity", fontsize=9)
    ax1.set_ylabel(r"$\Delta\theta$  [rad]", fontsize=9)
    n_err = len(errors)
    n_total = len(d_all)
    ax1.set_title(
        f"Error Distribution — {r['label']}\n"
        f"Total: {n_total}  |  Errors: {n_err} ({n_err/n_total:.1%})  |  "
        f"Correct: {n_total - n_err} ({(n_total-n_err)/n_total:.1%})",
        fontsize=8.5)
    ax1.legend(fontsize=7.5, loc="upper right")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # ── Panel 2: breakdown bar ────────────────────────────────────────────────
    cats   = ["false_direction", "near_zero_ambiguity", "extreme_outlier"]
    labels = ["False\ndirection", "Near-zero\nambiguity", "Extreme\noutlier"]
    counts = [bkd.get(c, 0) for c in cats]
    cols2  = ["#e53935", "#fb8c00", "#7b1fa2"]
    x2     = np.arange(len(cats))

    bars = ax2.bar(x2, counts, color=cols2, edgecolor="white", lw=0.5, zorder=3)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(labels, fontsize=8.5)
    ax2.set_ylabel("Count", fontsize=9)
    ax2.set_title(f"Error Breakdown\n(total errors = {n_err})", fontsize=9)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    for bar, cnt in zip(bars, counts):
        pct = cnt / n_err * 100 if n_err > 0 else 0
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + max(counts) * 0.02,
                 f"{cnt}\n({pct:.0f}%)",
                 ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax2.text(0.5, -0.18,
             f"near-zero τ={bkd.get('near_zero_tau',0.02):.3f} rad  "
             f"|  outlier τ={bkd.get('outlier_tau',0):.4f} rad (μ+2σ)",
             ha="center", va="top", transform=ax2.transAxes,
             fontsize=7, color="#555")

    return fig




# ===========================================================================
# PLOT CROSS-EMBEDDING COMPARISON  [NEW v8]
# ===========================================================================

def plot_cross_embedding(cross_results, embedding_configs, rel_label):
    """
    Two-panel bar chart comparing FastText vs GloVe vs Randomized:
      Panel 1: Direction Accuracy (%)
      Panel 2: Mean |Δθ| with error bars (std)
    """
    _style()
    present = [(name, col) for name, _, col in embedding_configs
               if name in cross_results and cross_results[name].get("n", 0) > 0]
    if not present:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No cross-embedding results yet",
                ha="center", va="center", transform=ax.transAxes)
        return fig

    names   = [n for n, _ in present]
    colors  = [c for _, c in present]
    dir_accs= [cross_results[n]["stats"]["dir_acc"] * 100 for n in names]
    abs_dts = [cross_results[n]["stats"]["mean_abs_dt"]   for n in names]
    stds    = [cross_results[n]["stats"]["std_dt"]        for n in names]
    coh_ds  = [cross_results[n]["stats"]["cohens_d"]      for n in names]
    ci_los  = [cross_results[n]["stats"]["ci_lo"]         for n in names]
    ci_his  = [cross_results[n]["stats"]["ci_hi"]         for n in names]

    x = np.arange(len(names))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5),
                                    constrained_layout=True)
    fig.suptitle(
        f"Cross-Embedding Validation — {rel_label}\n"
        "Geometric asymmetry probe: FastText  vs  GloVe  vs  Random N(μ,σ²)",
        fontsize=9.5, fontweight="bold")

    # ── Panel 1: Direction Accuracy ──────────────────────────────────────────
    bars1 = ax1.bar(x, dir_accs, color=colors, edgecolor="white", lw=0.5, zorder=3)
    ax1.axhline(50, color="#555", lw=1.2, linestyle="--",
                label="Chance level (50%)", zorder=2)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, fontsize=9)
    ax1.set_ylim(0, 115)
    ax1.set_ylabel("Direction Accuracy (%)", fontsize=9)
    ax1.set_title("Direction Accuracy per Embedding\n"
                  "Random→50% = geometry source; FastText≈GloVe = embedding-independent",
                  fontsize=8.5)
    for bar, v, d in zip(bars1, dir_accs, coh_ds):
        ax1.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 1.5,
                 f"{v:.1f}%\nd={d:+.2f}",
                 ha="center", va="bottom", fontsize=8, fontweight="bold", color="#222")
    ax1.legend(fontsize=8)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # ── Panel 2: Mean |Δθ| + CI ──────────────────────────────────────────────
    means  = [cross_results[n]["stats"]["mean_dt"] for n in names]
    ci_err_lo = [m - lo for m, lo in zip(means, ci_los)]
    ci_err_hi = [hi - m for m, hi in zip(means, ci_his)]
    bars2 = ax2.bar(x, abs_dts, color=colors, edgecolor="white", lw=0.5, zorder=3, alpha=0.85)
    ax2.errorbar(x, abs_dts, yerr=stds,
                 fmt="none", color="#333", capsize=4, lw=1.2, zorder=4,
                 label="±1 std")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, fontsize=9)
    ax2.set_ylabel(r"Mean $|\Delta\theta|$  [rad]", fontsize=9)
    ax2.set_title(r"Asymmetry Signal Strength per Embedding"
                  "\n95% CI shown as error bars (std)",
                  fontsize=8.5)
    for bar, v in zip(bars2, abs_dts):
        ax2.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + max(abs_dts + [0.001]) * 0.03,
                 f"{v:.4f}", ha="center", va="bottom",
                 fontsize=8.5, fontweight="bold", color="#222")
    ax2.legend(fontsize=8)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    return fig


def plot_cross_embedding_matrix(all_emb_results, rel_labels=None):
    """
    Formal Model × Relation heatmap + table  [NEW v13].
    all_emb_results: dict  emb_name → {rel_key → evaluate_pairs result}
    Shows dir_acc as a colour-coded grid (green=good, red=bad).
    """
    _style()
    REL_KEYS   = ["hyponymy", "meronymy", "capital", "sibling", "coordinate"]
    REL_LABELS = rel_labels or ["Hyponymy", "Meronymy", "Capital", "Sibling", "Coordinate"]

    emb_names  = list(all_emb_results.keys())
    if not emb_names:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "No cross-embedding data", ha="center", va="center",
                transform=ax.transAxes)
        return fig

    # Build matrix: rows=embeddings, cols=relations
    matrix   = np.full((len(emb_names), len(REL_KEYS)), np.nan)
    n_matrix = np.full_like(matrix, 0.0)
    for ei, emb in enumerate(emb_names):
        for ri, key in enumerate(REL_KEYS):
            res = all_emb_results[emb].get(key)
            if res and res.get("n", 0) > 0:
                st = res["stats"]
                if st.get("is_degenerate", False):
                    matrix[ei, ri] = np.nan
                else:
                    matrix[ei, ri] = st["dir_acc"] * 100
                    n_matrix[ei, ri] = st["n"]

    fig, (ax_heat, ax_bar) = plt.subplots(
        1, 2, figsize=(14, max(3, len(emb_names) * 1.2 + 2)),
        gridspec_kw={"width_ratios": [3, 2]}, constrained_layout=True)

    # ── Heatmap ──────────────────────────────────────────────────────────────
    import matplotlib.colors as mcolors
    cmap = plt.cm.RdYlGn
    norm = mcolors.Normalize(vmin=40, vmax=100)
    masked = np.ma.array(matrix, mask=np.isnan(matrix))
    im = ax_heat.imshow(masked, cmap=cmap, norm=norm, aspect="auto")
    ax_heat.set_xticks(range(len(REL_KEYS)))
    ax_heat.set_xticklabels(REL_LABELS, fontsize=9, rotation=20, ha="right")
    ax_heat.set_yticks(range(len(emb_names)))
    ax_heat.set_yticklabels(emb_names, fontsize=9)
    ax_heat.set_title("Dir Acc (%) — Model × Relation Matrix", fontsize=10, pad=8)
    # Annotate cells
    for ei in range(len(emb_names)):
        for ri in range(len(REL_KEYS)):
            v = matrix[ei, ri]
            if np.isnan(v):
                ax_heat.text(ri, ei, "N/A", ha="center", va="center",
                             fontsize=9, color="#666")
            else:
                col = "white" if v < 60 or v > 85 else "#111"
                n_str = f"\nn={int(n_matrix[ei,ri]):,}" if n_matrix[ei, ri] > 0 else ""
                ax_heat.text(ri, ei, f"{v:.1f}%{n_str}",
                             ha="center", va="center", fontsize=8.5,
                             fontweight="bold", color=col)
    plt.colorbar(im, ax=ax_heat, label="Direction Accuracy (%)", shrink=0.8)

    # ── Bar chart: mean dir_acc across asymmetric relations per embedding ────
    asym_cols = [i for i, k in enumerate(REL_KEYS) if k in ("hyponymy","meronymy","capital")]
    emb_means = []
    for ei in range(len(emb_names)):
        vals = [matrix[ei, ri] for ri in asym_cols if not np.isnan(matrix[ei, ri])]
        emb_means.append(np.mean(vals) if vals else np.nan)

    colors_bar = ["#d32f2f","#1565c0","#6a1b9a","#2e7d32","#e65100"]
    xb = np.arange(len(emb_names))
    bars = ax_bar.barh(xb, emb_means,
                       color=[colors_bar[i % len(colors_bar)] for i in range(len(emb_names))],
                       edgecolor="white", lw=0.5, alpha=0.85)
    ax_bar.axvline(50, color="#555", lw=1.2, linestyle="--", label="Chance 50%")
    ax_bar.set_yticks(xb)
    ax_bar.set_yticklabels(emb_names, fontsize=9)
    ax_bar.set_xlabel("Mean Dir Acc (%) — asymmetric relations", fontsize=9)
    ax_bar.set_title("Overall Directional Capability\n(Hyponymy + Meronymy + Capital)", fontsize=9)
    ax_bar.set_xlim(0, 110)
    for bar, v in zip(bars, emb_means):
        if not np.isnan(v):
            ax_bar.text(v + 1.5, bar.get_y() + bar.get_height() / 2,
                        f"{v:.1f}%", va="center", fontsize=9, fontweight="bold")
    ax_bar.legend(fontsize=8)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)

    return fig

BG      = "#1a1a2e"
BG2     = "#16213e"
ACCENT  = "#0d47a1"
ACCENT2 = "#1565c0"
FG      = "#e0e0e0"
FG2     = "#90caf9"
FG3     = "#b0bec5"
GREEN   = "#2e7d32"
RED     = "#c62828"
PURPLE  = "#4a148c"
ORANGE  = "#e65100"

DS_CONFIG = [
    ("hyponymy",   "Hyponymy (is-a)",          "asymmetric", PALETTE["hyponymy"]),
    ("meronymy",   "Meronymy (part-of)",        "asymmetric", PALETTE["meronymy"]),
    ("capital",    "Capital–Country",           "asymmetric", PALETTE["capital"]),
    ("sibling",    "Sibling-Symmetric",         "symmetric",  PALETTE["sibling"]),
    ("coordinate", "Coordinate-Symmetric",      "symmetric",  PALETTE["coordinate"]),
]

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("G-Angle vs Cosine — WordNet Comparison Evaluator")
        self.geometry("1280x860")
        self.configure(bg=BG)

        self.get_vec      = None
        self.get_vec_glove  = None   # GloVe embedding lookup [NEW v8]
        self.get_vec_rand   = None   # Randomized embedding lookup [NEW v8]
        self.results_glove  = {}     # GloVe evaluation results [NEW v8]
        self.results_rand   = {}     # Randomized embedding results [NEW v8]
        self.datasets     = {}     # key → list of (u, v, rel_type)
        self.results      = {}     # key → evaluate_pairs result
        self.sweep_results       = {}     # p_val → dict[key] → result
        self._sweep_best_p       = None   # analytically selected best p [v16]
        self._sweep_scores       = {}     # p → composite score [v16]
        self._sweep_stable_region= []     # stable plateau p values [v16]
        self.figs         = {}
        self._p           = tk.DoubleVar(value=3.0)
        self._max_pairs   = tk.IntVar(value=1500)
        self._seed        = tk.IntVar(value=42)
        self._sweep_p_vars= {}     # p → BooleanVar

        self._build_style()
        self._build_ui()

    # ── STYLE ─────────────────────────────────────────────────────────────────
    def _build_style(self):
        st = ttk.Style()
        st.theme_use("clam")
        st.configure("TNotebook",      background=BG,      borderwidth=0)
        st.configure("TNotebook.Tab",  background=ACCENT,  foreground=FG,
                     padding=[12,5],   font=("Times New Roman",10))
        st.map("TNotebook.Tab",
               background=[("selected","#1976d2")],
               foreground=[("selected","#ffffff")])
        st.configure("TFrame",         background=BG)
        st.configure("TLabelframe",    background=BG,      foreground=FG2,
                     font=("Times New Roman",10,"bold"))
        st.configure("TLabelframe.Label", background=BG,   foreground=FG2)
        st.configure("Treeview",       background=BG2,     foreground=FG,
                     fieldbackground=BG2, rowheight=24)
        st.configure("Treeview.Heading", background="#0f3460",
                     foreground=FG2,   font=("Times New Roman",9,"bold"))
        st.configure("TProgressbar",   troughcolor="#0f3460", background="#42a5f5")
        st.configure("TSeparator",     background="#0f3460")

    # ── BUILD UI ──────────────────────────────────────────────────────────────
    def _build_ui(self):
        # Header
        hdr = tk.Frame(self, bg=ACCENT, pady=8)
        hdr.pack(fill="x")
        tk.Label(hdr, text="G-Angle vs Cosine Similarity — WordNet Evaluator",
                 font=("Times New Roman",15,"bold"), bg=ACCENT, fg="#e3f2fd").pack()
        tk.Label(hdr,
                 text="Comparing the capability of G-Angle and Cosine in detecting "
                      "hierarchy, asymmetry, and symmetry from WordNet datasets",
                 font=("Times New Roman",9), bg=ACCENT, fg=FG2).pack()

        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=8, pady=6)

        tabs = [
            ("tab_setup",    "  ⚙  Setup  "),
            ("tab_dist",     "  📊  Δθ Distribution  "),
            ("tab_sweep",    "  🔄  Parameter Sweep  "),
            ("tab_compare",  "  ⚖  Head-to-Head  "),
            ("tab_perm",     "  🎲  Permutation Test  "),
            ("tab_swap",     "  📊  Symmetry Score  "),
            ("tab_ranking",  "  🏅  Ranking Accuracy  "),
            ("tab_error",    "  ❌  Error Analysis  "),
            ("tab_detail",   "  🔬  Detail View  "),
            ("tab_summary",  "  🏆  Summary  "),
            ("tab_table",    "  📋  Table  "),
            ("tab_cross",    "  🔀  Cross-Embedding  "),
            ("tab_log",      "  📝  Log  "),
        ]
        for attr, label in tabs:
            frame = ttk.Frame(nb)
            setattr(self, attr, frame)
            nb.add(frame, text=label)

        self._build_setup()
        self._build_tab_dist()
        self._build_tab_sweep()
        self._build_tab_compare()
        self._build_tab_perm()
        self._build_tab_swap()
        self._build_tab_ranking()
        self._build_tab_error()
        self._build_tab_detail()
        self._build_tab_summary()
        self._build_tab_table()
        self._build_tab_cross()
        self._build_log()

    # ── SETUP ─────────────────────────────────────────────────────────────────
    def _build_setup(self):
        pad = {"padx":12, "pady":6}

        # ── Scrollable container ───────────────────────────────────────────────
        _canvas = tk.Canvas(self.tab_setup, bg=BG, highlightthickness=0)
        _vsb    = ttk.Scrollbar(self.tab_setup, orient="vertical",
                                command=_canvas.yview)
        _canvas.configure(yscrollcommand=_vsb.set)
        _vsb.pack(side="right", fill="y")
        _canvas.pack(side="left", fill="both", expand=True)
        _inner  = tk.Frame(_canvas, bg=BG)
        _win    = _canvas.create_window((0, 0), window=_inner, anchor="nw")
        _inner.bind("<Configure>",
                    lambda e: (_canvas.configure(scrollregion=_canvas.bbox("all")),
                               _canvas.itemconfigure(_win, width=_canvas.winfo_width())))
        _canvas.bind("<Configure>",
                     lambda e: _canvas.itemconfigure(_win, width=e.width))
        _canvas.bind_all("<MouseWheel>",
                         lambda e: _canvas.yview_scroll(int(-1*(e.delta/120)), "units"))
        _canvas.bind_all("<Button-4>",
                         lambda e: _canvas.yview_scroll(-1, "units"))
        _canvas.bind_all("<Button-5>",
                         lambda e: _canvas.yview_scroll(1, "units"))
        _p = _inner   # all widgets mounted to _inner

        # FastText
        lf1 = ttk.LabelFrame(_p, text="1. FastText Embeddings", padding=10)
        lf1.pack(fill="x", **pad)
        self._lbl(lf1,"File (.vec or .bin):").grid(row=0,column=0,sticky="w")
        self.ft_path = tk.StringVar()
        self._entry(lf1, self.ft_path, 60).grid(row=0,column=1,padx=6)
        self._btn(lf1,"Browse",self._browse_ft,ACCENT2).grid(row=0,column=2)
        self._lbl(lf1,"https://fasttext.cc/docs/en/english-vectors.html",
                  fg="#42a5f5").grid(row=1,column=0,columnspan=3,sticky="w")
        self._btn(lf1,"▶  Load FastText",self._load_ft,
                  GREEN,pady=4).grid(row=2,column=0,columnspan=3,pady=(8,0))
        self.ft_lbl = self._lbl(lf1,"● Not loaded",fg="#ef5350")
        self.ft_lbl.grid(row=3,column=0,columnspan=3,sticky="w")

        # GloVe [NEW v8]
        lf1b = ttk.LabelFrame(_p, text="1b. GloVe Embeddings — Cross-Embedding Validation [NEW v8]", padding=10)
        lf1b.pack(fill="x", **pad)

        # ── Row 0: choose source ──────────────────────────────────────────────
        self._lbl(lf1b, "Source:").grid(row=0, column=0, sticky="w", pady=(0,4))
        self._glove_source = tk.StringVar(value="download")
        src_fr = tk.Frame(lf1b, bg=BG); src_fr.grid(row=0, column=1, columnspan=3, sticky="w")
        tk.Radiobutton(src_fr, text="Download from Stanford", variable=self._glove_source,
                       value="download", bg=BG, fg=FG2, activebackground=BG,
                       selectcolor="#0f3460", font=("Times New Roman",9),
                       command=self._toggle_glove_source).pack(side="left", padx=4)
        tk.Radiobutton(src_fr, text="Use local file", variable=self._glove_source,
                       value="local", bg=BG, fg=FG2, activebackground=BG,
                       selectcolor="#0f3460", font=("Times New Roman",9),
                       command=self._toggle_glove_source).pack(side="left", padx=4)

        # ── Row 1: download panel ─────────────────────────────────────────────
        self._glove_dl_frame = tk.Frame(lf1b, bg=BG)
        self._glove_dl_frame.grid(row=1, column=0, columnspan=4, sticky="ew", pady=4)
        GLOVE_URLS = [
            ("glove.2024.dolma.300d   [Recommended — 220B tokens, 300d, 1.6 GB]",
             "https://nlp.stanford.edu/data/wordvecs/glove.2024.dolma.300d.zip"),
            ("glove.2024.wikigiga.300d [Wikipedia+Gigaword 2024, 300d, 1.6 GB]",
             "https://nlp.stanford.edu/data/wordvecs/glove.2024.wikigiga.300d.zip"),
            ("glove.2024.wikigiga.200d [Wikipedia+Gigaword 2024, 200d, 1.1 GB]",
             "https://nlp.stanford.edu/data/wordvecs/glove.2024.wikigiga.200d.zip"),
            ("glove.2024.wikigiga.100d [Wikipedia+Gigaword 2024, 100d, 560 MB]",
             "https://nlp.stanford.edu/data/wordvecs/glove.2024.wikigiga.100d.zip"),
            ("glove.2024.wikigiga.50d  [Wikipedia+Gigaword 2024, 50d,  290 MB]",
             "https://nlp.stanford.edu/data/wordvecs/glove.2024.wikigiga.50d.zip"),
            ("glove.6B.300d            [Wikipedia 2014+Gigaword, 300d, 822 MB (legacy)]",
             "https://nlp.stanford.edu/data/wordvecs/glove.6B.zip"),
        ]
        self._glove_url_map = {label: url for label, url in GLOVE_URLS}
        self._glove_url_var = tk.StringVar(value=GLOVE_URLS[0][0])
        self._lbl(self._glove_dl_frame, "Vector file:").grid(row=0, column=0, sticky="w")
        om = tk.OptionMenu(self._glove_dl_frame, self._glove_url_var,
                           *[label for label, _ in GLOVE_URLS])
        om.config(bg=BG2, fg=FG, activebackground=ACCENT, highlightbackground=BG,
                  relief="flat", width=55, font=("Times New Roman", 9))
        om["menu"].config(bg=BG2, fg=FG, font=("Times New Roman", 9))
        om.grid(row=0, column=1, padx=6, sticky="w")

        self._lbl(self._glove_dl_frame, "Save to folder:").grid(row=1, column=0, sticky="w", pady=(6,0))
        self._glove_save_dir = tk.StringVar(value=os.path.expanduser("~"))
        self._entry(self._glove_dl_frame, self._glove_save_dir, 45).grid(row=1, column=1, padx=6, sticky="w", pady=(6,0))
        self._btn(self._glove_dl_frame, "Browse", self._browse_glove_dir,
                  ACCENT2).grid(row=1, column=2, pady=(6,0))

        self._btn(self._glove_dl_frame, "⬇  Download + Unzip + Load",
                  self._download_glove, "#00695c", pady=5,
                  font=("Times New Roman", 10, "bold")).grid(
                  row=2, column=0, columnspan=3, pady=(10, 2), sticky="w")

        # progress bar (hidden until download starts)
        self._glove_dl_bar_var = tk.DoubleVar(value=0)
        self._glove_dl_bar = ttk.Progressbar(
            self._glove_dl_frame, variable=self._glove_dl_bar_var,
            maximum=100, length=400, mode="determinate")
        self._glove_dl_bar.grid(row=3, column=0, columnspan=3, sticky="w", pady=(4,0))
        self._glove_dl_bar.grid_remove()   # hidden initially

        self._glove_dl_speed_lbl = self._lbl(self._glove_dl_frame, "", fg="#78909c")
        self._glove_dl_speed_lbl.grid(row=4, column=0, columnspan=3, sticky="w")

        # ── Row 2: local file panel ───────────────────────────────────────────
        self._glove_local_frame = tk.Frame(lf1b, bg=BG)
        self._glove_local_frame.grid(row=2, column=0, columnspan=4, sticky="ew", pady=4)
        self._lbl(self._glove_local_frame,
                  "File (.txt, e.g. glove.2024.dolma.300d.txt):").grid(row=0, column=0, sticky="w")
        self.glove_path = tk.StringVar()
        self._entry(self._glove_local_frame, self.glove_path, 50).grid(row=0, column=1, padx=6)
        self._btn(self._glove_local_frame, "Browse", self._browse_glove,
                  ACCENT2).grid(row=0, column=2)
        self._btn(self._glove_local_frame, "▶  Load GloVe", self._load_glove,
                  "#00695c", pady=4).grid(row=1, column=0, columnspan=3, pady=(8,0), sticky="w")

        # hide local panel by default (download mode is default)
        self._glove_local_frame.grid_remove()

        # ── Status label ─────────────────────────────────────────────────────
        self.glove_lbl = self._lbl(lf1b,
            "● Not loaded  (optional — needed for Cross-Embedding tab)", fg="#78909c")
        self.glove_lbl.grid(row=3, column=0, columnspan=4, sticky="w", pady=(4,0))

        # Randomized Embedding Control [NEW v8]
        lf1c = ttk.LabelFrame(_p, text="1c. Randomized Embedding Control [NEW v8]", padding=10)
        lf1c.pack(fill="x", **pad)
        self._lbl(lf1c,
            "Generates Gaussian random vectors N(μ,σ²) calibrated to real embedding stats.\n"
            "Hypothesis: if asymmetry persists but direction accuracy→50%, geometry is the source.",
            fg=FG3).grid(row=0,column=0,columnspan=3,sticky="w")
        self._btn(lf1c,"▶  Build Randomized Embedding",self._build_rand_emb,
                  "#4a148c",pady=4).grid(row=1,column=0,columnspan=3,pady=(8,0))
        self.rand_lbl = self._lbl(lf1c,"● Not built (requires FastText loaded + datasets generated)",fg="#78909c")
        self.rand_lbl.grid(row=2,column=0,columnspan=3,sticky="w")


        # Dataset generator
        lf2 = ttk.LabelFrame(_p, text="2. Generate Dataset from WordNet", padding=10)
        lf2.pack(fill="x", **pad)

        self._lbl(lf2,"Max pairs per relation:").grid(row=0,column=0,sticky="w",pady=3)
        sc_pair = tk.Scale(lf2, variable=self._max_pairs, from_=200, to=3000,
                           resolution=100, orient="horizontal", length=250,
                           bg=BG, fg=FG, highlightbackground=BG,
                           troughcolor="#0f3460", activebackground="#42a5f5")
        sc_pair.grid(row=0,column=1,padx=6)
        self._lbl(lf2,textvariable=self._max_pairs,fg="#42a5f5").grid(row=0,column=2)

        self._lbl(lf2,"Random seed:").grid(row=1,column=0,sticky="w",pady=3)
        tk.Spinbox(lf2, textvariable=self._seed, from_=0, to=999,
                   width=6, bg=BG2, fg=FG, insertbackground="#fff").grid(row=1,column=1,sticky="w",padx=6)

        self._btn(lf2,"▶  Generate All Datasets",self._gen_datasets,
                  PURPLE,pady=4).grid(row=2,column=0,columnspan=3,pady=(10,0))
        self.ds_lbl = self._lbl(lf2,"● Not generated yet",fg="#ef5350")
        self.ds_lbl.grid(row=3,column=0,columnspan=3,sticky="w")

        # Dataset info table
        lf2b = ttk.LabelFrame(_p, text="Dataset Status", padding=8)
        lf2b.pack(fill="x", **pad)
        cols = ("Relation","Type","G-Angle Expected","Cosine Expected","Status","n")
        self.ds_info_tree = self._tree(lf2b, cols, [160,120,210,210,100,80], height=3)
        self.ds_info_tree.pack(fill="x")
        self._populate_ds_info_empty()

        # Parameter
        lf3 = ttk.LabelFrame(_p, text="3. Parameters & Evaluation", padding=10)
        lf3.pack(fill="x", **pad)

        # Single p slider (for "Run All" / "Run One")
        self._lbl(lf3,"Value of p for single run (ℓᵖ norm):").grid(row=0,column=0,sticky="w")
        sc = tk.Scale(lf3, variable=self._p, from_=1.5, to=6.0,
                      resolution=0.5, orient="horizontal", length=300,
                      bg=BG, fg=FG, highlightbackground=BG,
                      troughcolor="#0f3460", activebackground="#42a5f5")
        sc.grid(row=0,column=1,padx=6)
        self._lbl(lf3,textvariable=self._p,fg="#42a5f5").grid(row=0,column=2)
        self._lbl(lf3,"p=2 → identical to cosine (Δθ≡0)  |  p>2 → asymmetry signal increases",
                  fg="#78909c").grid(row=1,column=0,columnspan=3,sticky="w")

        # Parameter sweep checkboxes
        self._lbl(lf3,"Parameter Sweep p values:",fg=FG2).grid(row=2,column=0,sticky="w",pady=(8,2))
        sweep_frame = tk.Frame(lf3, bg=BG); sweep_frame.grid(row=2,column=1,columnspan=2,sticky="w")
        for pv in [1.5, 2, 3, 5, 10]:
            var = tk.BooleanVar(value=True)
            self._sweep_p_vars[pv] = var
            tk.Checkbutton(sweep_frame, text=f"p={pv}", variable=var,
                           bg=BG, fg=FG2, activebackground=BG,
                           selectcolor="#0f3460",
                           font=("Times New Roman",9)).pack(side="left", padx=8)
        self._lbl(lf3,"Sweep will run evaluation for each selected p value automatically",
                  fg="#78909c").grid(row=3,column=0,columnspan=3,sticky="w")

        # Evaluation buttons — separate frame so always visible
        lf4 = ttk.LabelFrame(_p, text="▶  Run Evaluation", padding=10)
        lf4.pack(fill="x", **pad)

        bf_top = tk.Frame(lf4, bg=BG)
        bf_top.pack(fill="x", pady=(0,4))
        self._btn(bf_top,"▶  Run All (single p)",self._run_all,
                  PURPLE, pady=6, padx=20,
                  font=("Times New Roman",10,"bold")).pack(side="left", padx=6)
        self._lbl(bf_top,"← Evaluate all 5 relation types at once (uses p-slider value)",
                  fg="#78909c").pack(side="left", padx=4)

        bf_sweep = tk.Frame(lf4, bg=BG)
        bf_sweep.pack(fill="x", pady=(0,6))
        self._btn(bf_sweep,"🔄  Run Parameter Sweep",self._run_sweep,
                  ORANGE, pady=6, padx=20,
                  font=("Times New Roman",10,"bold")).pack(side="left", padx=6)
        self._lbl(bf_sweep,"← Auto-evaluates all selected p values above",
                  fg="#78909c").pack(side="left", padx=4)

        bf_bot = tk.Frame(lf4, bg=BG)
        bf_bot.pack(fill="x")
        for label, cmd, color in [
            ("▶ Hyponymy",   lambda: self._run_one("hyponymy"),   RED),
            ("▶ Meronymy",   lambda: self._run_one("meronymy"),   ORANGE),
            ("▶ Capital",    lambda: self._run_one("capital"),    GREEN),
            ("▶ Sibling",    lambda: self._run_one("sibling"),    ACCENT2),
            ("▶ Coordinate", lambda: self._run_one("coordinate"), PURPLE),
        ]:
            self._btn(bf_bot,label,cmd,color,pady=4,padx=10).pack(side="left",padx=4)

        prog_frame = tk.Frame(lf4, bg=BG)
        prog_frame.pack(fill="x", pady=(10,0))
        self.progress = ttk.Progressbar(prog_frame, mode="determinate",
                                        maximum=100, length=540)
        self.progress.pack(side="left", fill="x", expand=True)
        self._prog_lbl = tk.Label(prog_frame, text="  0%", bg=BG, fg="#42a5f5",
                                  font=("Courier", 10, "bold"), width=6)
        self._prog_lbl.pack(side="left", padx=(6, 0))

        # ── Export All Plots ───────────────────────────────────────────────
        lf5 = ttk.LabelFrame(_p, text="4. Export — Save All Plots (300 dpi)", padding=10)
        lf5.pack(fill="x", **pad)
        row5 = tk.Frame(lf5, bg=BG); row5.pack(fill="x")
        self._lbl(row5, "Output folder:").pack(side="left", padx=(0,6))
        self._save_dir = tk.StringVar()
        self._entry(row5, self._save_dir, 52).pack(side="left", padx=4)
        self._btn(row5, "Browse Folder", self._browse_save_dir,
                  ACCENT2).pack(side="left", padx=4)
        row5b = tk.Frame(lf5, bg=BG); row5b.pack(fill="x", pady=(8,0))
        self._btn(row5b, "💾  Save All Plots (300 dpi)",
                  self._save_all_plots, "#37474f", pady=6, padx=20,
                  font=("Times New Roman",10,"bold")).pack(side="left", padx=6)
        self._btn(row5b, "📦  Export All CSV + Log",
                  self._export_all_csv, "#1a237e", pady=6, padx=20,
                  font=("Times New Roman",10,"bold")).pack(side="left", padx=6)
        self._lbl(row5b,
                  "Saves ALL plots from ALL tabs (300 dpi, .png)  |  Export All CSV exports all_deltas, hasil_evaluasi, all_pairs per relation, all_relations, and log (.txt) to the selected folder",
                  fg="#78909c").pack(side="left", padx=4)

    def _browse_save_dir(self):
        d = filedialog.askdirectory(title="Select output folder for plots")
        if d:
            self._save_dir.set(d)

    def _save_all_plots(self):
        folder = self._save_dir.get().strip()
        if not folder:
            messagebox.showerror("No folder selected",
                                 "Please browse and select an output folder first.")
            return
        if not os.path.isdir(folder):
            messagebox.showerror("Invalid folder", f"Folder not found:\n{folder}")
            return
        if not self.results and not self.sweep_results:
            messagebox.showwarning("No plots", "Run evaluation first to generate plots.")
            return

        saved = []

        def _save_fig(fig, fname):
            if fig is None:
                return
            path = os.path.join(folder, fname)
            try:
                fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
                saved.append(fname)
            except Exception as e:
                self._log(f"[ERROR] Could not save {fname}: {e}")

        # ── Static tab plots ──────────────────────────────────────────────────
        _save_fig(self.figs.get("dist"),    "delta_distribution.png")
        _save_fig(self.figs.get("sweep"),   "parameter_sweep.png")
        _save_fig(self.figs.get("compare"), "head_to_head.png")
        _save_fig(self.figs.get("perm"),    "permutation_test.png")
        _save_fig(self.figs.get("swap"),    "swap_test.png")
        _save_fig(self.figs.get("ranking"), "ranking_accuracy.png")
        _save_fig(self.figs.get("summary"), "summary.png")
        _save_fig(self.figs.get("cross"),   "cross_embedding.png")  # [NEW v8]

        # ── Detail scatter — all available relations ──────────────────────────
        for key in ["hyponymy", "meronymy", "capital", "sibling", "coordinate"]:
            if key in self.results and self.results[key].get("n", 0) > 0:
                fig = plot_scatter_gangle_cosine(self.results, key)
                _save_fig(fig, f"detail_{key}.png")

        # ── Error analysis scatter — all available relations ──────────────────
        for key in ["hyponymy", "meronymy", "capital", "sibling", "coordinate"]:
            if key in self.results and self.results[key].get("n", 0) > 0:
                fig = plot_error_analysis(self.results, key)
                _save_fig(fig, f"error_analysis_{key}.png")

        if saved:
            self._log(f"\n✔ Saved {len(saved)} plot(s) to: {folder}")
            for f in saved:
                self._log(f"    {f}")
            messagebox.showinfo("Saved",
                f"{len(saved)} plot(s) saved at 300 dpi:\n{folder}")
        else:
            messagebox.showwarning("Nothing saved",
                "No plots available. Run evaluation first.")

    def _export_all_csv(self):
        """Export semua tabel CSV dan log ke satu folder:
           - all_deltas.csv              (semua Δθ & cosine per pair)
           - hasil_evaluasi.csv          (tabel statistik evaluasi)
           - all_pairs_ALL_relations.csv (semua pair semua relasi, kolom relation)
           - all_pairs_<key>.csv         (per relasi)
           - analysis_log.txt            (seluruh isi log box)
        """
        folder = self._save_dir.get().strip()
        if not folder:
            messagebox.showerror("No folder selected",
                                 "Please browse and select an output folder first.")
            return
        if not os.path.isdir(folder):
            messagebox.showerror("Invalid folder", f"Folder not found:\n{folder}")
            return
        if not self.results:
            messagebox.showwarning("No data", "Run evaluation first.")
            return

        saved = []

        # ── 1. all_deltas.csv ─────────────────────────────────────────────────
        try:
            p = os.path.join(folder, "all_deltas.csv")
            with open(p, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["relation", "word1", "word2", "delta_theta", "cosine"])
                for key, _, _, _ in DS_CONFIG:
                    if key not in self.results:
                        continue
                    r = self.results[key]
                    for (u, v), dt, cos in zip(r.get("pairs_ok", []),
                                               r.get("deltas", []),
                                               r.get("cosines", [])):
                        w.writerow([key, u, v, f"{dt:.6f}", f"{cos:.6f}"])
            saved.append("all_deltas.csv")
        except Exception as e:
            self._log(f"[ERROR] all_deltas.csv: {e}")

        # ── 2. hasil_evaluasi.csv  (isi tabel statistik utama) ────────────────
        try:
            p = os.path.join(folder, "hasil_evaluasi.csv")
            cols = self.main_tree["columns"]
            rows = [self.main_tree.item(iid)["values"]
                    for iid in self.main_tree.get_children()]
            if rows:
                with open(p, "w", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow(cols)
                    w.writerows(rows)
                saved.append("hasil_evaluasi.csv")
        except Exception as e:
            self._log(f"[ERROR] hasil_evaluasi.csv: {e}")

        # ── 3. all_pairs_ALL_relations.csv ────────────────────────────────────
        try:
            available = [(key, label) for key, label, _, _ in DS_CONFIG
                         if key in self.results and self.results[key].get("n", 0) > 0]
            if available:
                p = os.path.join(folder, "all_pairs_ALL_relations.csv")
                with open(p, "w", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow(["relation", "relation_type", "rank",
                                "word1", "word2",
                                "delta_theta", "cosine",
                                "status", "detail"])
                    for key, label in available:
                        r        = self.results[key]
                        pairs_ok = r.get("pairs_ok", [])
                        deltas   = r.get("deltas",   [])
                        cosines  = r.get("cosines",  [])
                        expected = r.get("expected", "asymmetric")
                        rel_type = "Asymmetric" if expected == "asymmetric" else "Symmetric"
                        error_set = {(ep["word1"], ep["word2"]): ep
                                     for ep in r.get("error_pairs", [])}
                        for i, ((u, v), dt, cos) in enumerate(
                                zip(pairs_ok, deltas, cosines)):
                            if (u, v) in error_set:
                                status = "Error"
                                detail = error_set[(u, v)]["error_type"]
                            else:
                                status = "Correct"
                                detail = ("Δθ<0 (should be <0)" if expected == "asymmetric"
                                          else f"|Δθ|={abs(dt):.4f}≤med")
                            w.writerow([key, rel_type, i + 1, u, v,
                                        f"{dt:.6f}", f"{cos:.6f}", status, detail])
                saved.append("all_pairs_ALL_relations.csv")
        except Exception as e:
            self._log(f"[ERROR] all_pairs_ALL_relations.csv: {e}")

        # ── 4. all_pairs_<key>.csv  per relasi ────────────────────────────────
        for key, label, _, _ in DS_CONFIG:
            if key not in self.results:
                continue
            r        = self.results[key]
            pairs_ok = r.get("pairs_ok", [])
            deltas   = r.get("deltas",   [])
            cosines  = r.get("cosines",  [])
            expected = r.get("expected", "asymmetric")
            if not pairs_ok:
                continue
            try:
                fname = f"all_pairs_{key}.csv"
                p = os.path.join(folder, fname)
                error_set = {(ep["word1"], ep["word2"]): ep
                             for ep in r.get("error_pairs", [])}
                with open(p, "w", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow(["rank", "word1", "word2",
                                "delta_theta", "cosine", "status", "detail"])
                    for i, ((u, v), dt, cos) in enumerate(
                            zip(pairs_ok, deltas, cosines)):
                        if (u, v) in error_set:
                            status = "Error"
                            detail = error_set[(u, v)]["error_type"]
                        else:
                            status = "Correct"
                            detail = ("Δθ<0 (should be <0)" if expected == "asymmetric"
                                      else f"|Δθ|={abs(dt):.4f}≤med")
                        w.writerow([i + 1, u, v, f"{dt:.6f}", f"{cos:.6f}",
                                    status, detail])
                saved.append(fname)
            except Exception as e:
                self._log(f"[ERROR] {fname}: {e}")

        # ── 5. analysis_log.txt ───────────────────────────────────────────────
        try:
            p = os.path.join(folder, "analysis_log.txt")
            log_content = self.log_box.get("1.0", "end")
            with open(p, "w", encoding="utf-8") as f:
                f.write(log_content)
            saved.append("analysis_log.txt")
        except Exception as e:
            self._log(f"[ERROR] analysis_log.txt: {e}")

        # ── Summary ───────────────────────────────────────────────────────────
        if saved:
            self._log(f"\n✔ Export All CSV + Log — {len(saved)} file(s) saved to: {folder}")
            for fn in saved:
                self._log(f"    {fn}")
            messagebox.showinfo("Export All CSV + Log — Done",
                f"{len(saved)} file(s) saved to:\n{folder}\n\n" +
                "\n".join(f"  • {fn}" for fn in saved))
        else:
            messagebox.showwarning("Nothing saved",
                "No files were exported. Run evaluation first.")

    def _populate_ds_info_empty(self):
        self.ds_info_tree.delete(*self.ds_info_tree.get_children())
        data = [
            ("Hyponymy",    "Asymmetric", "Δθ < 0 (strong)",   "Δcos ≡ 0 (cannot)", "—", "—"),
            ("Meronymy",    "Asymmetric", "Δθ < 0 (strong)",   "Δcos ≡ 0 (cannot)", "—", "—"),
            ("Capital",     "Asymmetric", "Δθ < 0 (strong)",   "Δcos ≡ 0 (cannot)", "—", "—"),
            ("Sibling",     "Symmetric",  "|Δθ| ≈ 0",        "Δcos = 0 (same)",  "—", "—"),
            ("Coordinate",  "Symmetric",  "|Δθ| ≈ 0",        "Δcos = 0 (same)",  "—", "—"),
        ]
        for row in data:
            self.ds_info_tree.insert("","end",values=row)

    # ── TAB DISTRIBUTION ──────────────────────────────────────────────────────
    def _build_tab_dist(self):
        self.dist_plot = ttk.LabelFrame(self.tab_dist,
            text="Δθ Distribution per Relation  |  dashed = 0 = Cosine position (no directional Δ)", padding=4)
        self.dist_plot.pack(fill="both", expand=True, padx=10, pady=6)

        bf = tk.Frame(self.tab_dist, bg=BG); bf.pack(pady=4)
        self._btn(bf,"💾 Save Plot (300 dpi)",
                  lambda: self._save(self.figs.get("dist"),"delta_distribution.png"),
                  "#37474f").pack(side="left",padx=6)
        self._btn(bf,"📋 Export Delta CSV",
                  lambda: self._save_all_deltas_csv(),
                  "#1b5e20").pack(side="left",padx=6)

    # ── TAB PARAMETER SWEEP ───────────────────────────────────────────────────
    def _build_tab_sweep(self):
        self.sweep_plot = ttk.LabelFrame(self.tab_sweep,
            text="Parameter Sweep p ∈ {1.5,2,3,4,5,10}  —  Dir/Rank Accuracy & Score vs p  |  Auto-selected best p annotated  [v16]",
            padding=4)
        self.sweep_plot.pack(fill="both", expand=True, padx=10, pady=6)
        bf = tk.Frame(self.tab_sweep, bg=BG); bf.pack(pady=4)
        self._btn(bf,"💾 Save Plot (300 dpi)",
                  lambda: self._save(self.figs.get("sweep"), "parameter_sweep.png"),
                  "#37474f").pack(side="left", padx=6)

    # ── TAB PERMUTATION TEST ──────────────────────────────────────────────────
    def _build_tab_perm(self):
        self.perm_plot = ttk.LabelFrame(self.tab_perm,
            text="Permutation Test  —  Random baseline distribution vs Observed mean Δθ per relation",
            padding=4)
        self.perm_plot.pack(fill="both", expand=True, padx=10, pady=6)
        bf = tk.Frame(self.tab_perm, bg=BG); bf.pack(pady=4)
        self._btn(bf,"💾 Save Plot (300 dpi)",
                  lambda: self._save(self.figs.get("perm"), "permutation_test.png"),
                  "#37474f").pack(side="left", padx=6)

    # ── TAB SWAP TEST ─────────────────────────────────────────────────────────
    def _build_tab_swap(self):
        ctrl = tk.Frame(self.tab_swap, bg=BG); ctrl.pack(fill="x", padx=10, pady=6)
        self._lbl(ctrl, "Relation for scatter:").pack(side="left", padx=6)
        self._swap_key = tk.StringVar(value="hyponymy")
        opts = ["hyponymy", "meronymy", "capital", "sibling", "coordinate"]
        om = tk.OptionMenu(ctrl, self._swap_key, *opts,
                           command=lambda _: self._refresh_swap())
        om.config(bg=BG2, fg=FG, activebackground=ACCENT, highlightbackground=BG, relief="flat")
        om["menu"].config(bg=BG2, fg=FG)
        om.pack(side="left", padx=6)
        self._btn(ctrl,"▶ Show", self._refresh_swap, ACCENT2).pack(side="left", padx=6)

        self.swap_plot = ttk.LabelFrame(self.tab_swap,
            text="Symmetry Score  —  |Δθ| Distribution: Symmetric vs Asymmetric Relations", padding=4)
        self.swap_plot.pack(fill="both", expand=True, padx=10, pady=4)
        bf = tk.Frame(self.tab_swap, bg=BG); bf.pack(pady=4)
        self._btn(bf,"💾 Save Plot (300 dpi)",
                  lambda: self._save(self.figs.get("swap"), "swap_test.png"),
                  "#37474f").pack(side="left", padx=6)

    # ── TAB RANKING ACCURACY ──────────────────────────────────────────────────
    def _build_tab_ranking(self):
        self.ranking_plot = ttk.LabelFrame(self.tab_ranking,
            text="Ranking Accuracy  —  G-Angle: real pairs vs random  |  Cosine baseline = 50%",
            padding=4)
        self.ranking_plot.pack(fill="both", expand=True, padx=10, pady=6)
        bf = tk.Frame(self.tab_ranking, bg=BG); bf.pack(pady=4)
        self._btn(bf,"💾 Save Plot (300 dpi)",
                  lambda: self._save(self.figs.get("ranking"), "ranking_accuracy.png"),
                  "#37474f").pack(side="left", padx=6)

    # ── TAB ERROR ANALYSIS  [enriched v13] ───────────────────────────────────
    def _build_tab_error(self):
        ctrl = tk.Frame(self.tab_error, bg=BG); ctrl.pack(fill="x", padx=10, pady=6)
        self._lbl(ctrl, "Select relation for error analysis:").pack(side="left", padx=6)
        self._error_key = tk.StringVar(value="hyponymy")
        opts = ["hyponymy", "meronymy", "capital", "sibling", "coordinate"]
        om = tk.OptionMenu(ctrl, self._error_key, *opts,
                           command=lambda _: self._refresh_error())
        om.config(bg=BG2, fg=FG, activebackground=ACCENT, highlightbackground=BG, relief="flat")
        om["menu"].config(bg=BG2, fg=FG)
        om.pack(side="left", padx=6)
        self._btn(ctrl, "▶ Show", self._refresh_error, ACCENT2).pack(side="left", padx=6)

        # Error breakdown summary label  [v14]
        self._error_summary_var = tk.StringVar(value="Run evaluation then click ▶ Show")
        tk.Label(ctrl, textvariable=self._error_summary_var,
                 bg=BG, fg="#e65100", font=("Consolas", 9, "bold"),
                 anchor="w").pack(side="left", padx=12)

        # Main error treeview (enriched with norm_ratio, frequency_proxy, sub_class)
        err_frame = ttk.LabelFrame(self.tab_error,
            text="All Error Pairs — Δθ, cos, ‖u‖/‖v‖ (norm ratio), freq proxy, sub-class  [v14]", padding=8)
        err_frame.pack(fill="x", padx=10, pady=4)
        cols = ("Rank","Word A","Word B","Δθ","cos(u,v)","Norm ratio","Freq proxy","Sub-class","Detail")
        widths = [50,120,120,90,90,90,80,150,220]
        self.error_tree = self._tree(err_frame, cols, widths, height=8)
        esb = ttk.Scrollbar(err_frame, orient="horizontal", command=self.error_tree.xview)
        self.error_tree.configure(xscrollcommand=esb.set)
        self.error_tree.pack(fill="both", expand=True)
        esb.pack(fill="x")

        # Top-FP / Top-FN sub-tables  [v13]
        fp_fn_frame = tk.Frame(self.tab_error, bg=BG)
        fp_fn_frame.pack(fill="x", padx=10, pady=2)
        fp_frame = ttk.LabelFrame(fp_fn_frame, text="Top-10 False Positives (worst Δθ≥0)  [v13]", padding=6)
        fp_frame.pack(side="left", fill="both", expand=True, padx=(0,4))
        fp_cols = ("Word A","Word B","Δθ","Norm ratio","Freq proxy")
        fp_widths = [110,110,90,90,80]
        self.fp_tree = self._tree(fp_frame, fp_cols, fp_widths, height=6)
        self.fp_tree.pack(fill="both", expand=True)

        fn_frame = ttk.LabelFrame(fp_fn_frame, text="Top-10 Marginal Correct (closest to flip)  [v13]", padding=6)
        fn_frame.pack(side="left", fill="both", expand=True, padx=(4,0))
        fn_cols = ("Word A","Word B","Δθ","Norm ratio","Freq proxy")
        fn_widths = [110,110,90,90,80]
        self.fn_tree = self._tree(fn_frame, fn_cols, fn_widths, height=6)
        self.fn_tree.pack(fill="both", expand=True)

        self.error_plot = ttk.LabelFrame(self.tab_error,
            text="Error Scatter — correct vs error pairs", padding=4)
        self.error_plot.pack(fill="both", expand=True, padx=10, pady=4)
        bf = tk.Frame(self.tab_error, bg=BG); bf.pack(pady=4)
        self._btn(bf,"💾 Save Plot (300 dpi)",
                  lambda: self._save(self.figs.get("error"), "error_analysis.png"),
                  "#37474f").pack(side="left", padx=6)
        self._btn(bf,"📋 Export This Relation CSV",
                  self._export_error_csv, "#1b5e20").pack(side="left", padx=6)
        self._btn(bf,"📦 Export ALL Relations CSV",
                  self._export_all_relations_csv, "#1a237e").pack(side="left", padx=6)

    # ── TAB HEAD-TO-HEAD ──────────────────────────────────────────────────────
    def _build_tab_compare(self):
        self.compare_plot = ttk.LabelFrame(self.tab_compare,
            text="Head-to-Head: G-Angle Accuracy & Asymmetry Signal  vs  Cosine baseline (50%, Δcos≡0)", padding=4)
        self.compare_plot.pack(fill="both", expand=True, padx=10, pady=6)

        bf = tk.Frame(self.tab_compare, bg=BG); bf.pack(pady=4)
        self._btn(bf,"💾 Save Plot (300 dpi)",
                  lambda: self._save(self.figs.get("compare"),"comparison.png"),
                  "#37474f").pack(side="left",padx=6)

    # ── TAB DETAIL ────────────────────────────────────────────────────────────
    def _build_tab_detail(self):
        ctrl = tk.Frame(self.tab_detail, bg=BG); ctrl.pack(fill="x",padx=10,pady=6)
        self._lbl(ctrl,"Select relation:").pack(side="left",padx=6)
        self._detail_key = tk.StringVar(value="hyponymy")
        opts = ["hyponymy","meronymy","capital","sibling","coordinate"]
        om   = tk.OptionMenu(ctrl, self._detail_key, *opts,
                             command=lambda _: self._refresh_detail())
        om.config(bg=BG2, fg=FG, activebackground=ACCENT,
                  highlightbackground=BG, relief="flat")
        om["menu"].config(bg=BG2, fg=FG)
        om.pack(side="left",padx=6)
        self._btn(ctrl,"▶ Show",
                  self._refresh_detail,ACCENT2).pack(side="left",padx=6)

        self.detail_plot = ttk.LabelFrame(self.tab_detail,
            text="Scatter: A_g(u→v) vs A_g(v→u)  +  Cosine vs Δθ", padding=4)
        self.detail_plot.pack(fill="both", expand=True, padx=10, pady=4)

        bf = tk.Frame(self.tab_detail, bg=BG); bf.pack(pady=4)
        self._btn(bf,"💾 Save Plot (300 dpi)",
                  lambda: self._save(self.figs.get("detail"),"detail_relation.png"),
                  "#37474f").pack(side="left",padx=6)

    # ── TAB SUMMARY ───────────────────────────────────────────────────────────
    def _build_tab_summary(self):
        self.summary_plot = ttk.LabelFrame(self.tab_summary,
            text="Summary — G-Angle vs Cosine per Dimension", padding=4)
        self.summary_plot.pack(fill="both", expand=True, padx=10, pady=6)

        bf = tk.Frame(self.tab_summary, bg=BG); bf.pack(pady=4)
        self._btn(bf,"💾 Save Plot (300 dpi)",
                  lambda: self._save(self.figs.get("summary"),"ringkasan.png"),
                  "#37474f").pack(side="left",padx=6)

    # ── TAB TABEL ─────────────────────────────────────────────────────────────
    def _build_tab_table(self):
        top = ttk.LabelFrame(self.tab_table,
            text="Full Statistics — G-Angle vs Cosine", padding=8)
        top.pack(fill="both", expand=True, padx=10, pady=6)
        cols = ("Relation","Type","n","OOV",
                "Dir Acc","Dir CI 95%","Rank Acc","Rank CI 95%",
                "Mean Δθ","Δθ CI 95%","Std Δθ",
                "Mean|Δθ|","Uncertain","Skew","Kurt","BC","Entropy",
                "t-stat","t p-val","t q (BH)",
                "Perm p (orient)","q orient","Perm p (pair)","q pair",
                "# Errors","Cosine Mean","Cosine Std",
                "Cosine Dir","Cohen d","Cohen CI 95%","Verdict")
        widths = [120,90,70,60,
                  90,140,90,140,
                  100,155,90,
                  90,80,70,70,70,70,
                  80,80,80,
                  110,80,105,80,
                  75,100,90,
                  110,80,130,140]
        self.main_tree = self._tree(top, cols, widths, height=8)
        sb = ttk.Scrollbar(top, orient="horizontal", command=self.main_tree.xview)
        self.main_tree.configure(xscrollcommand=sb.set)
        self.main_tree.pack(fill="both", expand=True)
        sb.pack(fill="x")

        bf = tk.Frame(self.tab_table, bg=BG); bf.pack(pady=4)
        self._btn(bf,"📋 Export CSV",
                  lambda: self._save_tree(self.main_tree,"hasil_evaluasi.csv"),
                  "#1b5e20").pack(side="left",padx=6)

    # ── LOG ───────────────────────────────────────────────────────────────────
    def _build_log(self):
        self.log_box = scrolledtext.ScrolledText(
            self.tab_log, wrap="word", font=("Courier",9),
            bg="#0d0d1a", fg="#a5d6a7", insertbackground="#fff")
        self.log_box.pack(fill="both", expand=True, padx=6, pady=6)
        self._btn(self.tab_log,"Clear Log",
                  lambda: self.log_box.delete("1.0","end"),
                  "#455a64").pack(pady=4)

    # ── WIDGET HELPERS ────────────────────────────────────────────────────────
    def _lbl(self, parent, text="", fg=FG3, **kw):
        return tk.Label(parent, text=text, bg=BG, fg=fg,
                        font=("Times New Roman",9), **kw)

    def _entry(self, parent, var, width=40):
        return tk.Entry(parent, textvariable=var, width=width,
                        bg=BG2, fg=FG, insertbackground="#fff",
                        relief="flat",
                        highlightbackground="#0f3460", highlightthickness=1)

    def _btn(self, parent, text, cmd, bg, fg="white",
             padx=10, pady=3, font=("Times New Roman",9)):
        return tk.Button(parent, text=text, command=cmd,
                         bg=bg, fg=fg, relief="flat",
                         padx=padx, pady=pady, font=font,
                         activebackground=ACCENT2, activeforeground="white")

    def _tree(self, parent, cols, widths, height=5):
        t = ttk.Treeview(parent, columns=cols, show="headings", height=height)
        for c, w in zip(cols, widths):
            t.heading(c, text=c)
            t.column(c, width=w, anchor="center")
        return t

    def _log(self, msg):
        def _do():
            self.log_box.insert("end", msg+"\n")
            self.log_box.see("end")
        self.after(0, _do)

    def _set_progress(self, value, maximum=100):
        """Set progress bar to value/maximum (0-100%). Thread-safe via after()."""
        pct = int(round(value / maximum * 100)) if maximum > 0 else 0
        pct = max(0, min(100, pct))
        def _do():
            self.progress["value"] = pct
            self._prog_lbl.config(text=f"{pct:3d}%")
        self.after(0, _do)

    def _reset_progress(self):
        self._set_progress(0)

    def _embed(self, fig, frame):
        for w in frame.winfo_children():
            w.destroy()
        c = FigureCanvasTkAgg(fig, master=frame)
        c.draw()
        c.get_tk_widget().pack(fill="both", expand=True)

    def _save(self, fig, name):
        if fig is None:
            messagebox.showwarning("No plot yet","Run evaluation first.")
            return
        p = filedialog.asksaveasfilename(
            defaultextension=".png", initialfile=name,
            filetypes=[("PNG","*.png"),("PDF","*.pdf"),("SVG","*.svg")])
        if p:
            fig.savefig(p, dpi=300, bbox_inches="tight")
            self._log(f"Saved (300 dpi): {p}")
            messagebox.showinfo("OK", f"Saved:\n{p}")

    def _save_tree(self, tree, name):
        cols = tree["columns"]
        rows = [tree.item(iid)["values"] for iid in tree.get_children()]
        if not rows:
            messagebox.showwarning("Table empty","No data yet.")
            return
        p = filedialog.asksaveasfilename(
            defaultextension=".csv", initialfile=name,
            filetypes=[("CSV","*.csv"),("All","*.*")])
        if p:
            with open(p,"w",newline="",encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(cols)
                w.writerows(rows)
            messagebox.showinfo("OK",f"Saved:\n{p}")

    def _save_all_deltas_csv(self):
        p = filedialog.asksaveasfilename(
            defaultextension=".csv", initialfile="all_deltas.csv",
            filetypes=[("CSV","*.csv"),("All","*.*")])
        if not p: return
        with open(p,"w",newline="",encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["relation","word1","word2","delta_theta","cosine"])
            for key, _, _, _ in DS_CONFIG:
                if key not in self.results: continue
                r = self.results[key]
                for (u,v), dt, cos in zip(r["pairs_ok"],r["deltas"],r["cosines"]):
                    w.writerow([key,u,v,f"{dt:.6f}",f"{cos:.6f}"])
        messagebox.showinfo("OK",f"Saved:\n{p}")

    # ── BROWSE ────────────────────────────────────────────────────────────────
    def _browse_ft(self):
        p = filedialog.askopenfilename(
            title="Select FastText File",
            filetypes=[("FastText","*.vec *.bin"),("All","*.*")])
        if p: self.ft_path.set(p)

    # ── LOAD FASTTEXT ─────────────────────────────────────────────────────────
    def _load_ft(self):
        path = self.ft_path.get().strip()
        if not path or not os.path.exists(path):
            messagebox.showerror("Error","FastText file not found.")
            return
        self.after(0, lambda: self.ft_lbl.config(text="⏳ Loading...", fg="#fb8c00"))
        def _run():
            try:
                self.get_vec = load_fasttext(path, self._log)
                self.after(0, lambda: self.ft_lbl.config(text="✔ FastText ready", fg="#66bb6a"))
            except Exception as e:
                self.after(0, lambda: self.ft_lbl.config(text=f"✖ {e}", fg="#ef5350"))
                self._log(f"[ERROR] {e}")
        threading.Thread(target=_run, daemon=True).start()

    # ── BROWSE / LOAD GLOVE  [NEW v8] ─────────────────────────────────────────
    # ── GLOVE SOURCE TOGGLE  [NEW v8] ─────────────────────────────────────────
    def _toggle_glove_source(self):
        if self._glove_source.get() == "download":
            self._glove_dl_frame.grid()
            self._glove_local_frame.grid_remove()
        else:
            self._glove_dl_frame.grid_remove()
            self._glove_local_frame.grid()

    def _browse_glove_dir(self):
        d = filedialog.askdirectory(title="Select folder to save GloVe")
        if d:
            self._glove_save_dir.set(d)

    # ── DOWNLOAD + UNZIP + LOAD GLOVE  [NEW v8] ───────────────────────────────
    def _download_glove(self):
        import urllib.request, zipfile, time
        label_sel = self._glove_url_var.get()
        url       = self._glove_url_map.get(label_sel, "")
        save_dir  = self._glove_save_dir.get().strip()
        if not url:
            messagebox.showerror("Error", "No URL selected.")
            return
        if not save_dir or not os.path.isdir(save_dir):
            messagebox.showerror("Error", "Save folder not found. Please browse to a valid folder.")
            return

        zip_name  = url.split("/")[-1]
        zip_path  = os.path.join(save_dir, zip_name)

        self._glove_dl_bar_var.set(0)
        self._glove_dl_bar.grid()
        self._glove_dl_speed_lbl.config(text="Starting download...", fg="#78909c")
        self.glove_lbl.config(text="⏳ Downloading...", fg="#fb8c00")

        def _run():
            try:
                # ── Phase 1: Download ──────────────────────────────────────
                self._log(f"GloVe download: {url}")
                self._log(f"  → saving to: {zip_path}")

                t0 = time.time()
                downloaded = [0]
                total_size = [0]

                def _reporthook(count, block_size, total):
                    if total > 0:
                        total_size[0] = total
                    downloaded[0] = count * block_size
                    if total > 0:
                        pct = min(downloaded[0] / total * 100, 100)
                    else:
                        pct = 0
                    elapsed = time.time() - t0
                    speed   = downloaded[0] / elapsed / 1024 / 1024 if elapsed > 0.1 else 0
                    if total > 0:
                        remaining = (total - downloaded[0]) / (downloaded[0] / elapsed) if downloaded[0] > 0 else 0
                        eta_str   = f"  ETA {remaining:.0f}s" if remaining < 9999 else ""
                    else:
                        eta_str = ""
                    dl_mb = downloaded[0] / 1024 / 1024
                    tot_mb = total / 1024 / 1024 if total > 0 else 0
                    speed_txt = (f"{dl_mb:.1f} MB / {tot_mb:.1f} MB  "
                                 f"@ {speed:.2f} MB/s{eta_str}  [{pct:.1f}%]")
                    self.after(0, lambda p=pct, s=speed_txt: (
                        self._glove_dl_bar_var.set(p),
                        self._glove_dl_speed_lbl.config(text=s, fg="#42a5f5"),
                        self.glove_lbl.config(
                            text=f"⏳ Downloading {pct:.1f}%...", fg="#fb8c00"),
                    ))

                urllib.request.urlretrieve(url, zip_path, _reporthook)
                self._log(f"  Download complete: {zip_path}")

                # ── Phase 2: Unzip ────────────────────────────────────────
                self.after(0, lambda: (
                    self._glove_dl_bar_var.set(100),
                    self._glove_dl_speed_lbl.config(
                        text="Unzipping... (this may take a minute)", fg="#fb8c00"),
                    self.glove_lbl.config(text="⏳ Unzipping...", fg="#fb8c00"),
                ))
                self._log(f"  Unzipping {zip_path} ...")
                txt_path = None
                with zipfile.ZipFile(zip_path, "r") as zf:
                    members = zf.namelist()
                    self._log(f"  Archive contents: {members}")
                    txt_files = [m for m in members if m.endswith(".txt")]
                    # pick the .txt with matching dimension if possible
                    if not txt_files:
                        raise ValueError("No .txt file found inside zip.")
                    # prefer longest name (most specific)
                    chosen = sorted(txt_files, key=len)[-1]
                    self._log(f"  Extracting: {chosen}")
                    zf.extract(chosen, save_dir)
                    txt_path = os.path.join(save_dir, chosen)
                self._log(f"  Extracted to: {txt_path}")

                # ── Phase 3: Load ─────────────────────────────────────────
                self.after(0, lambda: (
                    self._glove_dl_speed_lbl.config(
                        text="Loading vectors into memory...", fg="#fb8c00"),
                    self.glove_lbl.config(text="⏳ Loading vectors...", fg="#fb8c00"),
                ))
                self.get_vec_glove = load_glove(txt_path, self._log)

                self.after(0, lambda tp=txt_path: (
                    self._glove_dl_bar_var.set(100),
                    self._glove_dl_speed_lbl.config(
                        text=f"✔ Done  |  {tp}", fg="#66bb6a"),
                    self.glove_lbl.config(
                        text=f"✔ GloVe ready — {os.path.basename(tp)}", fg="#66bb6a"),
                ))
                self._log("GloVe embedding loaded. Use 'Run Cross-Embedding' to compare.")

            except Exception as e:
                self.after(0, lambda: (
                    self._glove_dl_speed_lbl.config(text=f"✖ {e}", fg="#ef5350"),
                    self.glove_lbl.config(text=f"✖ Error: {e}", fg="#ef5350"),
                ))
                self._log(f"[ERROR GloVe Download] {e}")

        threading.Thread(target=_run, daemon=True).start()

    def _browse_glove(self):
        p = filedialog.askopenfilename(
            title="Select GloVe .txt File",
            filetypes=[("GloVe","*.txt"),("All","*.*")])
        if p:
            self.glove_path.set(p)

    def _load_glove(self):
        path = self.glove_path.get().strip()
        if not path or not os.path.exists(path):
            messagebox.showerror("Error","GloVe file not found.")
            return
        self.after(0, lambda: self.glove_lbl.config(text="⏳ Loading GloVe...", fg="#fb8c00"))
        def _run():
            try:
                self.get_vec_glove = load_glove(path, self._log)
                self.after(0, lambda: self.glove_lbl.config(
                    text="✔ GloVe ready", fg="#66bb6a"))
                self._log("GloVe embedding loaded. Use 'Run Cross-Embedding' to compare.")
            except Exception as e:
                self.after(0, lambda: self.glove_lbl.config(text=f"✖ {e}", fg="#ef5350"))
                self._log(f"[ERROR GloVe] {e}")
        threading.Thread(target=_run, daemon=True).start()

    # ── BUILD RANDOMIZED EMBEDDING  [NEW v8] ──────────────────────────────────
    def _build_rand_emb(self):
        if self.get_vec is None:
            messagebox.showerror("Not ready","Load FastText first.")
            return
        if not self.datasets:
            messagebox.showerror("Not ready","Generate datasets first.")
            return
        self.after(0, lambda: self.rand_lbl.config(
            text="⏳ Building randomized embedding...", fg="#fb8c00"))
        def _run():
            try:
                # Use all pairs across all datasets for stats calibration
                all_pairs = []
                for pairs in self.datasets.values():
                    all_pairs.extend(pairs[:100])
                self.get_vec_rand = make_randomized_get_vec(
                    self.get_vec, all_pairs, self._log,
                    seed=self._seed.get())
                self.after(0, lambda: self.rand_lbl.config(
                    text="✔ Randomized embedding ready (N(μ,σ²) calibrated to FastText)",
                    fg="#66bb6a"))
                self._log("Randomized embedding built. Use 'Run Cross-Embedding' to compare.")
            except Exception as e:
                self.after(0, lambda: self.rand_lbl.config(text=f"✖ {e}", fg="#ef5350"))
                self._log(f"[ERROR Rand] {e}")
        threading.Thread(target=_run, daemon=True).start()

    # ── TAB CROSS-EMBEDDING  [NEW v8 / enhanced v13] ─────────────────────────
    def _build_tab_cross(self):
        info = tk.Label(self.tab_cross, bg=BG, fg=FG3, font=("Times New Roman",9),
            text=(
                "Cross-Embedding Validation:  FastText  vs  GloVe  vs  Randomized N(μ,σ²)\n"
                "Scientific goal: if Δθ asymmetry persists across different embeddings\n"
                "but direction accuracy → 50% for Randomized → asymmetry source is geometry.\n"
                "[v13] 'Run All Cross' evaluates ALL relations for each loaded embedding "
                "and renders the Model × Relation matrix."
            ), justify="left")
        info.pack(fill="x", padx=12, pady=(8,2))

        ctrl = tk.Frame(self.tab_cross, bg=BG)
        ctrl.pack(fill="x", padx=10, pady=6)
        self._lbl(ctrl,"Relation (single):").pack(side="left", padx=6)
        self._cross_key = tk.StringVar(value="hyponymy")
        opts = ["hyponymy","meronymy","capital","sibling","coordinate"]
        om = tk.OptionMenu(ctrl, self._cross_key, *opts)
        om.config(bg=BG2, fg=FG, activebackground=ACCENT, highlightbackground=BG, relief="flat")
        om["menu"].config(bg=BG2, fg=FG)
        om.pack(side="left", padx=6)
        self._btn(ctrl,"▶  Run Single Relation",
                  self._run_cross, "#00695c", pady=5, padx=12,
                  font=("Times New Roman",10,"bold")).pack(side="left", padx=6)
        self._btn(ctrl,"▶▶  Run ALL Relations Cross  [v13]",
                  self._run_cross_all, "#1a237e", pady=5, padx=12,
                  font=("Times New Roman",10,"bold")).pack(side="left", padx=10)

        self.cross_plot = ttk.LabelFrame(self.tab_cross,
            text="Cross-Embedding Plot", padding=4)
        self.cross_plot.pack(fill="both", expand=True, padx=10, pady=4)

        # Formal Model × Relation matrix table  [v13]
        self.cross_matrix_frame = ttk.LabelFrame(self.tab_cross,
            text="Formal Model × Relation Comparison — Dir Acc (%)  [v13]", padding=8)
        self.cross_matrix_frame.pack(fill="x", padx=10, pady=4)
        mat_cols = ("Embedding","Hyponymy","Meronymy","Capital","Sibling","Coordinate",
                    "Mean (asym)","n total")
        mat_widths = [130,100,100,100,100,110,110,90]
        self.cross_matrix_tree = self._tree(self.cross_matrix_frame, mat_cols, mat_widths, height=5)
        self.cross_matrix_tree.pack(fill="both", expand=True)

        self.cross_table_frame = ttk.LabelFrame(self.tab_cross,
            text="Cross-Embedding Detailed Statistics", padding=8)
        self.cross_table_frame.pack(fill="x", padx=10, pady=4)
        cols = ("Embedding","Relation","n","Dir Acc","Mean Δθ","Std Δθ",
                "Mean|Δθ|","Cohen's d","95% CI","Perm p","BH q")
        widths = [110,110,60,80,90,80,80,80,140,70,70]
        self.cross_tree = self._tree(self.cross_table_frame, cols, widths, height=6)
        xsb = ttk.Scrollbar(self.cross_table_frame, orient="horizontal",
                             command=self.cross_tree.xview)
        self.cross_tree.configure(xscrollcommand=xsb.set)
        self.cross_tree.pack(fill="both", expand=True)
        xsb.pack(fill="x")

        bf = tk.Frame(self.tab_cross, bg=BG); bf.pack(pady=4)
        self._btn(bf,"💾 Save Plot (300 dpi)",
                  lambda: self._save(self.figs.get("cross"),"cross_embedding.png"),
                  "#37474f").pack(side="left", padx=6)
        self._btn(bf,"💾 Save Matrix Plot (300 dpi)",
                  lambda: self._save(self.figs.get("cross_matrix"),"cross_matrix.png"),
                  "#37474f").pack(side="left", padx=6)
        self._btn(bf,"📋 Export CSV",
                  lambda: self._save_tree(self.cross_tree,"cross_embedding.csv"),
                  "#1b5e20").pack(side="left", padx=6)

    # ── RUN ALL RELATIONS CROSS-EMBEDDING  [NEW v13] ─────────────────────────
    def _run_cross_all(self):
        """Evaluate ALL relations for every loaded embedding → Model × Relation matrix."""
        if self.get_vec is None:
            messagebox.showerror("Not ready","Load FastText first.")
            return
        if not self.datasets:
            messagebox.showerror("Not ready","Generate datasets first.")
            return

        p = self._p.get()
        self._set_progress(0)

        def _run():
            global RANDOM_SEED
            RANDOM_SEED = self._seed.get()
            embedding_configs = [("FastText", self.get_vec, "#d32f2f")]
            if self.get_vec_glove is not None:
                embedding_configs.append(("GloVe", self.get_vec_glove, "#1565c0"))
            if self.get_vec_rand is not None:
                embedding_configs.append(("Random N(μ,σ²)", self.get_vec_rand, "#6a1b9a"))

            all_emb_results = {}   # emb_name → {rel_key → result}
            total_tasks = len(embedding_configs) * 5
            done = 0
            for emb_name, gv, _ in embedding_configs:
                all_emb_results[emb_name] = {}
                for key, label, expected, _ in DS_CONFIG:
                    if key not in self.datasets or len(self.datasets[key]) == 0:
                        continue
                    self._log(f"\n── Cross-All: {emb_name} / {label}  (p={p}) ──")
                    try:
                        res = evaluate_pairs(self.datasets[key], gv, p, expected,
                                             self._log, f"{label} [{emb_name}]",
                                             n_permutations=200)
                        res["expected"] = expected
                        all_emb_results[emb_name][key] = res
                    except Exception as e:
                        self._log(f"  [ERROR {emb_name}/{key}] {e}")
                    done += 1
                    self._set_progress(done, total_tasks)
                # BH correction per embedding
                apply_bh_correction(all_emb_results[emb_name], log=self._log)

            self._set_progress(100)
            self.after(0, lambda: self._update_cross_matrix(all_emb_results, embedding_configs, p))

        threading.Thread(target=_run, daemon=True).start()

    def _update_cross_matrix(self, all_emb_results, embedding_configs, p):
        """Render the Model × Relation matrix heatmap and table."""
        # ── Matrix table ──────────────────────────────────────────────────────
        self.cross_matrix_tree.delete(*self.cross_matrix_tree.get_children())
        REL_KEYS = ["hyponymy","meronymy","capital","sibling","coordinate"]
        ASYM_KEYS = ["hyponymy","meronymy","capital"]
        for emb_name, _, _ in embedding_configs:
            row_res = all_emb_results.get(emb_name, {})
            cells = []
            asym_vals = []
            n_total = 0
            for key in REL_KEYS:
                res = row_res.get(key)
                if res and res.get("n",0) > 0:
                    st = res["stats"]
                    if st.get("is_degenerate"):
                        cells.append("N/A")
                    else:
                        cells.append(f"{st['dir_acc']:.1%}")
                        if key in ASYM_KEYS:
                            asym_vals.append(st["dir_acc"] * 100)
                    n_total += res.get("n", 0)
                else:
                    cells.append("—")
            mean_asym = f"{np.mean(asym_vals):.1f}%" if asym_vals else "—"
            self.cross_matrix_tree.insert("","end", values=(
                emb_name, *cells, mean_asym, f"{n_total:,}"))

        # Detailed stats table
        self.cross_tree.delete(*self.cross_tree.get_children())
        for emb_name, _, _ in embedding_configs:
            for key, label, _, _ in DS_CONFIG:
                res = all_emb_results.get(emb_name, {}).get(key)
                if not res or res.get("n", 0) == 0:
                    continue
                st = res["stats"]
                def _fq(v):
                    try:
                        if np.isnan(v): return "—"
                    except Exception: return "—"
                    return f"{v:.4f}{'✔' if v < 0.05 else ''}"
                self.cross_tree.insert("","end", values=(
                    emb_name, label,
                    f"{st['n']:,}",
                    st.get("dir_acc_display", f"{st['dir_acc']:.1%}"),
                    f"{st['mean_dt']:+.4f}",
                    f"{st['std_dt']:.4f}",
                    f"{st['mean_abs_dt']:.4f}",
                    f"{st['cohens_d']:+.3f}",
                    f"[{st['ci_lo']:+.4f}, {st['ci_hi']:+.4f}]",
                    f"{st['perm_p']:.4f}",
                    _fq(st.get("perm_q_pair", float("nan"))),
                ))

        # Matrix heatmap plot
        fig_matrix = plot_cross_embedding_matrix(all_emb_results)
        self.figs["cross_matrix"] = fig_matrix
        self._embed(fig_matrix, self.cross_plot)
        self._log(f"\n✔ Cross-All complete (p={p}). Matrix heatmap rendered.")

    # ── RUN CROSS-EMBEDDING COMPARISON  [NEW v8] ──────────────────────────────
    def _run_cross(self):
        if self.get_vec is None:
            messagebox.showerror("Not ready","Load FastText first.")
            return
        if not self.datasets:
            messagebox.showerror("Not ready","Generate datasets first.")
            return

        rel_key = self._cross_key.get()
        if rel_key not in self.datasets or len(self.datasets[rel_key]) == 0:
            messagebox.showerror("Not ready","Generate datasets first.")
            return

        p = self._p.get()
        self._set_progress(0)

        def _run():
            global RANDOM_SEED
            RANDOM_SEED = self._seed.get()
            cfg_map = {c[0]: c for c in DS_CONFIG}
            cfg     = cfg_map[rel_key]
            label   = cfg[1]
            expected= cfg[2]
            pairs   = self.datasets[rel_key]

            embedding_configs = [("FastText", self.get_vec, "#d32f2f")]
            if self.get_vec_glove is not None:
                embedding_configs.append(("GloVe",    self.get_vec_glove, "#1565c0"))
            if self.get_vec_rand is not None:
                embedding_configs.append(("Random N(μ,σ²)", self.get_vec_rand, "#6a1b9a"))

            cross_results = {}
            total = len(embedding_configs)
            for i, (emb_name, gv, _) in enumerate(embedding_configs):
                self._log(f"\n── Cross-Embedding: {emb_name}  ({label}, p={p}) ──")
                self._set_progress(i, total)
                try:
                    res = evaluate_pairs(pairs, gv, p, expected,
                                         self._log, f"{label} [{emb_name}]",
                                         n_permutations=500)
                    res["expected"] = expected
                    cross_results[emb_name] = res
                except Exception as e:
                    self._log(f"  [ERROR {emb_name}] {e}")
            self._set_progress(100)

            # Store
            if "FastText" in cross_results:
                self.results_rand  = cross_results.get("Random N(μ,σ²)", {})
                self.results_glove = cross_results.get("GloVe", {})

            # Plot
            self.after(0, lambda: self._update_cross_tab(
                cross_results, embedding_configs, label, rel_key))

        threading.Thread(target=_run, daemon=True).start()

    def _update_cross_tab(self, cross_results, embedding_configs, label, rel_key):
        """Update Cross-Embedding tab (single relation) with plot and table."""
        # ── Detailed stats table ──────────────────────────────────────────────
        self.cross_tree.delete(*self.cross_tree.get_children())
        # Apply BH within this single-relation comparison
        apply_bh_correction(
            {f"{emb_name}_{rel_key}": cross_results[emb_name]
             for emb_name, _, _ in embedding_configs if emb_name in cross_results},
            log=self._log)
        for emb_name, _, _ in embedding_configs:
            r = cross_results.get(emb_name)
            if not r or r.get("n", 0) == 0:
                continue
            st = r["stats"]
            def _fq(v):
                try:
                    if np.isnan(v): return "—"
                except Exception: return "—"
                return f"{v:.4f}{'✔' if v < 0.05 else ''}"
            self.cross_tree.insert("", "end", values=(
                emb_name, label,
                f"{st['n']:,}",
                st.get("dir_acc_display", f"{st['dir_acc']:.1%}"),
                f"{st['mean_dt']:+.4f}",
                f"{st['std_dt']:.4f}",
                f"{st['mean_abs_dt']:.4f}",
                f"{st['cohens_d']:+.3f}",
                f"[{st['ci_lo']:+.4f}, {st['ci_hi']:+.4f}]",
                f"{st['perm_p']:.4f}",
                _fq(st.get("perm_q_pair", float("nan"))),
            ))

        # ── Plot ──────────────────────────────────────────────────────────────
        fig = plot_cross_embedding(cross_results, embedding_configs, label)
        self.figs["cross"] = fig
        self._embed(fig, self.cross_plot)

        # ── Log conclusions ───────────────────────────────────────────────────
        self._log(f"\n{'='*60}")
        self._log(f"  CROSS-EMBEDDING CONCLUSIONS  ({label})")
        self._log(f"{'='*60}")
        for emb_name, _, _ in embedding_configs:
            r = cross_results.get(emb_name)
            if not r or r.get("n", 0) == 0:
                continue
            st = r["stats"]
            self._log(
                f"  {emb_name:<20} dir_acc={st['dir_acc']:.1%}  "
                f"mean Δθ={st['mean_dt']:+.4f}  "
                f"Cohen's d={st['cohens_d']:+.3f}  "
                f"95%CI=[{st['ci_lo']:+.4f},{st['ci_hi']:+.4f}]  "
                f"perm_p={st['perm_p']:.4f}")
        ft  = cross_results.get("FastText")
        rnd = cross_results.get("Random N(μ,σ²)")
        glv = cross_results.get("GloVe")
        if ft and rnd:
            da_ft  = ft["stats"]["dir_acc"]
            da_rnd = rnd["stats"]["dir_acc"]
            self._log(f"\n  Geometric asymmetry test:")
            if abs(rnd["stats"]["mean_dt"]) > 0.001:
                self._log(f"  ✔ |Δθ| persists in Random → geometry produces asymmetry")
            else:
                self._log(f"  ✖ |Δθ| vanishes in Random → FastText may have directional bias")
            if da_rnd < 0.55:
                self._log(f"  ✔ Direction accuracy → {da_rnd:.1%} in Random ≈ chance → semantic info is gone")
            else:
                self._log(f"  ⚠ Direction accuracy = {da_rnd:.1%} in Random (should be ~50%)")
        if ft and glv:
            da_ft  = ft["stats"]["dir_acc"]
            da_glv = glv["stats"]["dir_acc"]
            self._log(f"\n  Cross-embedding consistency:")
            if abs(da_ft - da_glv) < 0.10:
                self._log(f"  ✔ FastText ({da_ft:.1%}) ≈ GloVe ({da_glv:.1%}) → effect is embedding-independent")
            else:
                self._log(f"  ⚠ FastText ({da_ft:.1%}) ≠ GloVe ({da_glv:.1%}) → effect may be embedding-specific")
        self._log(f"{'='*60}\n")


    # ── GENERATE DATASETS ─────────────────────────────────────────────────────
    def _gen_datasets(self):
        self.after(0, lambda: self.ds_lbl.config(text="⏳ Generating...", fg="#fb8c00"))
        self._set_progress(0)
        max_p = self._max_pairs.get()
        seed  = self._seed.get()

        def _run():
            try:
                random.seed(seed)
                np.random.seed(seed)
                self._log(f"\n{'='*55}\nGenerating Dataset from WordNet  (max={max_p})\n{'='*55}")

                generators = [
                    ("hyponymy",   lambda: gen_hyponymy(max_p, self._log)),
                    ("meronymy",   lambda: gen_meronymy(max_p, self._log)),
                    ("capital",    lambda: gen_capital_country(self._log)),
                    ("sibling",    lambda: gen_sibling_symmetric(max_p, self._log)),
                    ("coordinate", lambda: gen_coordinate_symmetric(max_p, self._log)),
                ]
                total = len(generators)
                summaries = []
                for i, (key, gen_fn) in enumerate(generators):
                    self._set_progress(i, total)
                    self._log(f"  Generating {key}...")
                    pairs = gen_fn()
                    self.datasets[key] = pairs
                    self._log(f"    → {len(pairs):,} pairs")
                    summaries.append(f"{key} {len(pairs):,}")
                    self._set_progress(i + 1, total)

                # Update tabel info
                def _update_info():
                    self.ds_info_tree.delete(*self.ds_info_tree.get_children())
                    for key, lbl, exp, _ in DS_CONFIG:
                        n = len(self.datasets.get(key, []))
                        st = "✔" if n > 0 else "✖"
                        exp_ga  = "Δθ < 0 (strong)" if exp=="asymmetric" else "|Δθ| ≈ 0"
                        exp_cos = "Δcos ≡ 0 (cannot)" if exp=="asymmetric" else "Δcos = 0 (same)"
                        tipe    = "Asymmetric" if exp=="asymmetric" else "Symmetric"
                        self.ds_info_tree.insert("","end",values=(lbl,tipe,exp_ga,exp_cos,st,n))

                self.after(0, _update_info)
                self.after(0, lambda: self.ds_lbl.config(
                    text="✔ " + "  |  ".join(summaries), fg="#66bb6a"))
                self._log("Generation complete.")
            except Exception as e:
                self._log(f"[ERROR Generate] {e}")
                import traceback; self._log(traceback.format_exc())
                self.after(0, lambda: self.ds_lbl.config(text=f"✖ {e}", fg="#ef5350"))
            finally:
                self._set_progress(100)
        threading.Thread(target=_run, daemon=True).start()

    # ── EVALUASI ──────────────────────────────────────────────────────────────
    def _run_one(self, key):
        if self.get_vec is None:
            messagebox.showerror("Not ready","Load FastText first.")
            return
        if key not in self.datasets or len(self.datasets[key]) == 0:
            messagebox.showerror("Not ready",
                f"Dataset '{key}' not generated yet.\nClick 'Generate All Datasets' first.")
            return
        self._set_progress(0)
        p = self._p.get()

        def _run():
            global RANDOM_SEED
            RANDOM_SEED = self._seed.get()
            try:
                cfg = {c[0]:c for c in DS_CONFIG}[key]
                label    = cfg[1]
                expected = cfg[2]
                self._log(f"\n--- Evaluasi: {label}  (p={p}) ---")
                pairs = self.datasets[key]
                self._set_progress(10)
                res   = evaluate_pairs(pairs, self.get_vec, p, expected,
                                       self._log, label,
                                       progress_cb=self._set_progress)
                res["expected"] = expected
                self.results[key] = res
                self._log(f"  Done: n={res['n']}, "
                          f"dir_acc={res['stats']['dir_acc']:.1%}, "
                          f"rank_acc={res['stats']['ranking_acc']:.1%}, "
                          f"mean Δθ={res['stats']['mean_dt']:+.4f}, "
                          f"perm_p={res['stats']['perm_p']:.4f}")
                self.after(0, self._refresh_all_plots)
                self.after(0, self._refresh_table)
            except Exception as e:
                self._log(f"[ERROR {key}] {e}")
                import traceback; self._log(traceback.format_exc())
            finally:
                self._set_progress(100)
        threading.Thread(target=_run, daemon=True).start()

    # ── PARAMETER SWEEP ───────────────────────────────────────────────────────
    def _run_sweep(self):
        if self.get_vec is None:
            messagebox.showerror("Not ready","Load FastText first.")
            return
        if not self.datasets:
            messagebox.showerror("Not ready","Generate datasets first.")
            return
        p_list = [pv for pv, var in self._sweep_p_vars.items() if var.get()]
        if not p_list:
            messagebox.showerror("No p selected","Select at least one p value for the sweep.")
            return
        self._set_progress(0)

        def _run():
            global RANDOM_SEED
            seed = self._seed.get()
            RANDOM_SEED = seed
            self._log(f"\n{'='*60}")
            self._log(f"  PARAMETER SWEEP  p ∈ {sorted(p_list)}  (seed={seed})")
            self._log(f"{'='*60}")
            sweep = {}
            ds_keys = [k for k, _, _, _ in DS_CONFIG if k in self.datasets and len(self.datasets[k]) > 0]
            total_tasks = len(p_list) * len(ds_keys)
            done = 0
            for pv in sorted(p_list):
                sweep[pv] = {}
                self._log(f"\n── p = {pv} ──────────────────────────────────────")
                for key, label, expected, _ in DS_CONFIG:
                    if key not in self.datasets or len(self.datasets[key]) == 0:
                        self._log(f"  [SKIP] {label}")
                        continue
                    self._log(f"  Evaluating: {label} ...")
                    res = evaluate_pairs(self.datasets[key], self.get_vec,
                                        pv, expected, self._log, label)
                    res["expected"] = expected
                    sweep[pv][key] = res
                    done += 1
                    self._set_progress(done, total_tasks)
                    st = res['stats']
                    def _fp(v):
                        try: return "N/A" if np.isnan(v) else f"{v:.4f}"
                        except Exception: return "N/A"
                    _da_str   = st.get('dir_acc_display',  f"{st['dir_acc']:.1%}")
                    _ra_str   = st.get('ranking_acc_display', f"{st['ranking_acc']:.1%}")
                    self._log(f"    dir_acc={_da_str}  "
                              f"rank_acc={_ra_str}  "
                              f"perm-p(orient)={_fp(st.get('perm_p_orient', st['perm_p']))}  "
                              f"perm-p(pair)={_fp(st.get('perm_p_pair', st['perm_p']))}")
                # BH correction within this p block  [NEW v13]
                apply_bh_correction(sweep[pv], log=self._log)

            self.sweep_results = sweep

            # ── [v16] Analytical best-p selection ────────────────────────────
            self._log(f"\n{'='*60}")
            best_p, scores, stable_region, report = select_best_p(
                sweep, log=self._log)
            self._log(f"{'='*60}")

            # Store for plot annotation
            self._sweep_best_p       = best_p
            self._sweep_scores       = scores
            self._sweep_stable_region = stable_region

            # Use analytically-selected p as main results
            self.results = sweep[best_p]
            self._log(f"\n✔ Sweep complete!  Auto-selected p = {best_p}"
                      f"  (stable region: {stable_region})")
            self.after(0, self._refresh_all_plots)
            self.after(0, self._refresh_table)
            self._set_progress(100)
        threading.Thread(target=_run, daemon=True).start()

    def _run_all(self):
        if self.get_vec is None:
            messagebox.showerror("Not ready","Load FastText first.")
            return
        if not self.datasets:
            messagebox.showerror("Not ready",
                "Generate datasets first.")
            return
        self._set_progress(0)
        p = self._p.get()

        def _run():
            global RANDOM_SEED
            RANDOM_SEED = self._seed.get()
            try:
                ds_keys = [(key, label, expected) for key, label, expected, _ in DS_CONFIG
                           if key in self.datasets and len(self.datasets[key]) > 0]
                total = len(ds_keys)

                # ── Phase 1: Real Embedding Evaluation ────────────────────────
                for i, (key, label, expected) in enumerate(ds_keys):
                    self._set_progress(i, total)
                    self._log(f"\n--- Evaluating: {label}  (p={p}) ---")
                    res = evaluate_pairs(self.datasets[key], self.get_vec,
                                        p, expected, self._log, label,
                                        progress_cb=lambda v, m=100: self._set_progress(
                                            i * 100 + v, total * 100))
                    res["expected"] = expected
                    self.results[key] = res
                    self._log(f"  Done: n={res['n']}, "
                              f"dir_acc={res['stats']['dir_acc']:.1%}, "
                              f"rank_acc={res['stats']['ranking_acc']:.1%}, "
                              f"mean Δθ={res['stats']['mean_dt']:+.4f}, "
                              f"perm_p={res['stats']['perm_p']:.4f}")

                # Apply BH FDR correction across all relations  [NEW v13]
                apply_bh_correction(self.results, log=self._log)

                # ── Phase 2: Randomized Embedding Baseline  [NEW v15] ─────────
                # Build calibrated random embedding from ALL available pairs
                # (gabungan semua relasi → statistik lebih representatif).
                all_pairs_for_rand = []
                for key, _, _ in ds_keys:
                    all_pairs_for_rand.extend(self.datasets.get(key, []))

                self._log(f"\n{'─'*55}")
                self._log(f"  [v15] Randomized Baseline — {len(all_pairs_for_rand):,} pairs for calibration")
                try:
                    get_vec_rand = make_randomized_get_vec(
                        self.get_vec, all_pairs_for_rand, self._log, seed=RANDOM_SEED)
                    self.get_vec_rand = get_vec_rand   # store for Cross-Embedding tab reuse

                    self.results_rand = {}
                    for i, (key, label, expected) in enumerate(ds_keys):
                        self._log(f"  Random [{label}] ...")
                        res_r = evaluate_pairs(
                            self.datasets[key], get_vec_rand,
                            p, expected, self._log, f"{label} [Random]",
                            n_permutations=200)   # lighter permutation count
                        res_r["expected"] = expected
                        self.results_rand[key] = res_r
                        st_r = res_r["stats"]
                        self._log(
                            f"    dir_acc={st_r['dir_acc']:.1%}  "
                            f"(hypothesis: ≈50% for asymmetric)  "
                            f"mean|Δθ|={st_r['mean_abs_dt']:.4f}  "
                            f"(geometric signal persists if >0)")
                    apply_bh_correction(self.results_rand, log=self._log)
                    self._log(f"  [v15] Random baseline done.\n{'─'*55}")
                except Exception as e_r:
                    self._log(f"  [WARN v15] Random baseline failed: {e_r}")
                    self.results_rand = {}

                self.after(0, self._refresh_all_plots)
                self.after(0, self._refresh_table)
                self.after(0, self._refresh_table)   # refresh again with q-values
                self._log("\n✔ All evaluations complete!")
                self._log_conclusions()
            except Exception as e:
                self._log(f"[ERROR] {e}")
                import traceback; self._log(traceback.format_exc())
            finally:
                self._set_progress(100)
        threading.Thread(target=_run, daemon=True).start()

    # ── REFRESH PLOTS ─────────────────────────────────────────────────────────
    def _refresh_all_plots(self):
        if self.results:
            # Pass rand_results for side-by-side comparison [v15]
            rnd = self.results_rand if self.results_rand else None

            self.figs["dist"] = plot_delta_distributions(self.results)
            self._embed(self.figs["dist"], self.dist_plot)

            self.figs["compare"] = plot_comparison_bars(self.results, rand_results=rnd)
            self._embed(self.figs["compare"], self.compare_plot)

            self.figs["summary"] = plot_summary_radar(self.results)
            self._embed(self.figs["summary"], self.summary_plot)

            self.figs["perm"] = plot_permutation_test(self.results)
            self._embed(self.figs["perm"], self.perm_plot)

            self.figs["swap"] = plot_swap_test(self.results)
            self._embed(self.figs["swap"], self.swap_plot)

            self.figs["ranking"] = plot_ranking_accuracy(self.results, rand_results=rnd)
            self._embed(self.figs["ranking"], self.ranking_plot)

            # Error tab: display for the first available relation
            for key in ["hyponymy", "meronymy", "capital", "sibling", "coordinate"]:
                if key in self.results and self.results[key].get("n", 0) > 0:
                    self._error_key.set(key)
                    break
            self._refresh_error()

        if self.sweep_results:
            self.figs["sweep"] = plot_parameter_sweep(
                self.sweep_results,
                best_p=self._sweep_best_p,
                stable_region=self._sweep_stable_region,
                scores=self._sweep_scores,
            )
            self._embed(self.figs["sweep"], self.sweep_plot)

        self._refresh_detail()
        self._refresh_swap()

    def _refresh_swap(self):
        key = self._swap_key.get()
        if key in self.results:
            self.figs["swap"] = plot_swap_test(self.results)
            if self.figs["swap"]:
                self._embed(self.figs["swap"], self.swap_plot)

    def _refresh_error(self):
        key = self._error_key.get()
        if key not in self.results or self.results[key].get("n", 0) == 0:
            return
        r        = self.results[key]
        expected = r.get("expected", "asymmetric")
        pairs_ok = r.get("pairs_ok", [])
        deltas   = r.get("deltas",   [])
        cosines  = r.get("cosines",  [])

        # Build error set for fast lookup
        error_set = {}
        for ep in r.get("error_pairs", []):
            error_set[(ep["word1"], ep["word2"])] = ep

        # ── Update breakdown summary label ────────────────────────────────────
        bkd   = r.get("error_breakdown",
                      {"false_direction": 0, "near_zero_ambiguity": 0, "extreme_outlier": 0})
        n_tot = len(r.get("error_pairs", []))
        n_all = r.get("n", 1) or 1
        fd    = bkd.get("false_direction",     0)
        nz    = bkd.get("near_zero_ambiguity", 0)
        ol    = bkd.get("extreme_outlier",     0)
        self._error_summary_var.set(
            f"Errors: {n_tot}/{n_all} ({n_tot/n_all:.1%})   "
            f"│  False direction: {fd} ({fd/n_tot:.0%} of errors)"
            f"  │  Near-zero ambiguity: {nz} ({nz/n_tot:.0%})"
            f"  │  Extreme outlier: {ol} ({ol/n_tot:.0%})"
            if n_tot > 0 else "No errors"
        )

        # Update treeview — all error pairs with enriched columns  [v13/v14]
        self.error_tree.delete(*self.error_tree.get_children())
        for i, ((u, v), dt, cos) in enumerate(zip(pairs_ok, deltas, cosines)):
            key_pair = (u, v)
            ep = error_set.get(key_pair)
            if ep:
                sub_cls  = ep.get("sub_class", "false_direction")
                detail   = ep["error_type"]
                nr_str   = f"{ep.get('norm_ratio', float('nan')):.3f}"
                fp_str   = f"{ep.get('frequency_proxy', float('nan')):.3f}"
            else:
                if expected == "asymmetric":
                    sub_cls = "correct"; detail = "Δθ<0"
                else:
                    sub_cls = "correct"; detail = f"|Δθ|≤med"
                nr_str = "—"; fp_str = "—"
            self.error_tree.insert("", "end", values=(
                i + 1, u, v,
                f"{dt:+.6f}", f"{cos:.6f}",
                nr_str, fp_str, sub_cls, detail,
            ))

        # Top-FP table  [v13]
        self.fp_tree.delete(*self.fp_tree.get_children())
        for ep in r.get("top_fp", []):
            self.fp_tree.insert("","end", values=(
                ep["word1"], ep["word2"],
                f"{ep['delta_theta']:+.6f}",
                f"{ep.get('norm_ratio', float('nan')):.3f}",
                f"{ep.get('frequency_proxy', float('nan')):.3f}",
            ))

        # Top-FN / marginal table  [v13]
        self.fn_tree.delete(*self.fn_tree.get_children())
        for ep in r.get("top_fn", []):
            self.fn_tree.insert("","end", values=(
                ep["word1"], ep["word2"],
                f"{ep['delta_theta']:+.6f}",
                f"{ep.get('norm_ratio', float('nan')):.3f}",
                f"{ep.get('frequency_proxy', float('nan')):.3f}",
            ))

        # Update plot
        fig = plot_error_analysis(self.results, key)
        if fig:
            self.figs["error"] = fig
            self._embed(self.figs["error"], self.error_plot)

    def _export_error_csv(self):
        key = self._error_key.get()
        if key not in self.results:
            messagebox.showwarning("No data", "Run evaluation first.")
            return
        r        = self.results[key]
        pairs_ok = r.get("pairs_ok", [])
        deltas   = r.get("deltas",   [])
        cosines  = r.get("cosines",  [])
        expected = r.get("expected", "asymmetric")

        if not pairs_ok:
            messagebox.showinfo("No data", "No evaluated pairs for this relation.")
            return

        # Build error set for fast lookup
        error_set = {}
        for ep in r.get("error_pairs", []):
            error_set[(ep["word1"], ep["word2"])] = ep

        # Compute median for symmetric relations
        if expected != "asymmetric" and len(deltas) > 0:
            import numpy as _np
            med_abs = float(_np.median(_np.abs(deltas)))
        else:
            med_abs = None

        p = filedialog.asksaveasfilename(
            defaultextension=".csv", initialfile=f"all_pairs_{key}.csv",
            filetypes=[("CSV","*.csv"),("All","*.*")])
        if not p: return

        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["rank","word1","word2","delta_theta","cosine","status","detail"])
            for i, ((u, v), dt, cos) in enumerate(zip(pairs_ok, deltas, cosines)):
                key_pair = (u, v)
                if key_pair in error_set:
                    status = "Error"
                    detail = error_set[key_pair]["error_type"]
                else:
                    status = "Correct"
                    if expected == "asymmetric":
                        detail = "Δθ<0 (should be <0)"
                    else:
                        detail = f"|Δθ|={abs(dt):.4f}≤med"
                w.writerow([i+1, u, v,
                            f"{dt:.6f}",
                            f"{cos:.6f}",
                            status,
                            detail])
        n_error   = len(error_set)
        n_correct = len(pairs_ok) - n_error
        messagebox.showinfo("OK",
            f"Saved: {p}\n\n"
            f"Total pairs : {len(pairs_ok):,}\n"
            f"  ✔ Correct : {n_correct:,}\n"
            f"  ✖ Error   : {n_error:,}")

    def _export_all_relations_csv(self):
        """Export semua 5 relasi sekaligus ke satu file CSV dengan kolom 'relation'."""
        if not self.results:
            messagebox.showwarning("No data", "Run evaluation first.")
            return

        # Cek relasi mana yang sudah ada datanya
        available = [(key, label) for key, label, _, _ in DS_CONFIG
                     if key in self.results and self.results[key].get("n", 0) > 0]
        if not available:
            messagebox.showinfo("No data", "No evaluated pairs found in any relation.")
            return

        p = filedialog.asksaveasfilename(
            defaultextension=".csv",
            initialfile="all_pairs_ALL_relations.csv",
            filetypes=[("CSV", "*.csv"), ("All", "*.*")])
        if not p:
            return

        total_pairs = 0
        total_correct = 0
        total_error = 0
        summary_lines = []

        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["relation", "relation_type", "rank",
                        "word1", "word2",
                        "delta_theta", "cosine",
                        "status", "detail"])

            for key, label in available:
                r        = self.results[key]
                pairs_ok = r.get("pairs_ok", [])
                deltas   = r.get("deltas",   [])
                cosines  = r.get("cosines",  [])
                expected = r.get("expected", "asymmetric")
                rel_type = "Asymmetric" if expected == "asymmetric" else "Symmetric"

                # Build error set
                error_set = {}
                for ep in r.get("error_pairs", []):
                    error_set[(ep["word1"], ep["word2"])] = ep

                n_err = 0
                for i, ((u, v), dt, cos) in enumerate(zip(pairs_ok, deltas, cosines)):
                    key_pair = (u, v)
                    if key_pair in error_set:
                        status = "Error"
                        detail = error_set[key_pair]["error_type"]
                        n_err += 1
                    else:
                        status = "Correct"
                        if expected == "asymmetric":
                            detail = "Δθ<0 (should be <0)"
                        else:
                            detail = f"|Δθ|={abs(dt):.4f}≤med"
                    w.writerow([key, rel_type, i + 1,
                                u, v,
                                f"{dt:.6f}",
                                f"{cos:.6f}",
                                status, detail])

                n_rel = len(pairs_ok)
                n_ok  = n_rel - n_err
                total_pairs   += n_rel
                total_correct += n_ok
                total_error   += n_err
                summary_lines.append(
                    f"  {label:<28} {n_rel:>5} pairs  "
                    f"✔ {n_ok}  ✖ {n_err}")

        summary = "\n".join(summary_lines)
        messagebox.showinfo("Saved — All Relations",
            f"Saved: {p}\n\n"
            f"{'Relation':<28} {'n':>5}   Correct  Error\n"
            f"{'-'*55}\n"
            f"{summary}\n"
            f"{'-'*55}\n"
            f"  {'TOTAL':<28} {total_pairs:>5} pairs  "
            f"✔ {total_correct}  ✖ {total_error}")

    def _refresh_detail(self):
        key = self._detail_key.get()
        if key in self.results:
            self.figs["detail"] = plot_scatter_gangle_cosine(self.results, key)
            if self.figs["detail"]:
                self._embed(self.figs["detail"], self.detail_plot)

    def _log_conclusions(self):
        """Print qualitative conclusions to the log tab."""
        sep = "=" * 65
        self._log(f"\n{sep}")
        self._log("  CONCLUSIONS — G-Angle vs Cosine per Relation Type")
        self._log(sep)
        datasets = [
            ("hyponymy",   "Hyponymy (hierarchy)",   "asymmetric"),
            ("meronymy",   "Meronymy (part-of)",      "asymmetric"),
            ("capital",    "Capital–Country",         "asymmetric"),
            ("sibling",    "Sibling-Symmetric",       "symmetric"),
            ("coordinate", "Coordinate-Symmetric",    "symmetric"),
        ]
        for key, label, exp in datasets:
            if key not in self.results or self.results[key]["n"] == 0:
                continue
            st = self.results[key]["stats"]
            if exp == "asymmetric":
                if st["dir_acc"] > 0.60:
                    icon = "✔✔"; verdict = f"G-Angle detects direction strongly ({st['dir_acc']:.0%})"
                elif st["dir_acc"] > 0.50:
                    icon = "✔ "; verdict = f"G-Angle slightly above chance ({st['dir_acc']:.0%})"
                else:
                    icon = "✖ "; verdict = f"G-Angle below chance ({st['dir_acc']:.0%})"
            else:
                tau = st.get("sym_tau", 0.05)
                sym_score = st["dir_acc"]   # = P(|Δθ| < τ) since v14
                if sym_score > 0.70:
                    icon = "✔✔"; verdict = f"Symmetry Score={sym_score:.0%}  (τ={tau:.3f} rad) — highly symmetric"
                elif sym_score > 0.50:
                    icon = "✔ "; verdict = f"Symmetry Score={sym_score:.0%}  (τ={tau:.3f} rad) — mostly symmetric"
                else:
                    icon = "⚠ "; verdict = f"Symmetry Score={sym_score:.0%}  (τ={tau:.3f} rad) — unexpectedly asymmetric"
            type_str = "asymmetric" if exp == "asymmetric" else "symmetric"
            acc_label = "dir_acc" if exp == "asymmetric" else "sym_score(τ)"
            self._log(f"  {icon}  {label:<30} [{type_str}]")
            self._log(f"       {verdict}")
            _da_str2  = st.get('dir_acc_display',  f"{st['dir_acc']:.1%}")
            _ra_str2  = st.get('ranking_acc_display', f"{st['ranking_acc']:.1%}")
            self._log(f"       {acc_label}={_da_str2}  "
                      f"95%CI=[{st['dir_acc_ci_lo']:.1%},{st['dir_acc_ci_hi']:.1%}]  "
                      f"rank_acc={_ra_str2}  "
                      f"95%CI=[{st['rank_acc_ci_lo']:.1%},{st['rank_acc_ci_hi']:.1%}]  "
                      f"uncertain={st['n_uncertain']}  "
                      f"mean Δθ={st['mean_dt']:+.4f}  "
                      f"95%CI=[{st['mean_dt_ci_lo']:+.4f},{st['mean_dt_ci_hi']:+.4f}]  "
                      f"mean|Δθ|={st['mean_abs_dt']:.4f}")
            def _fmt_p(v):
                try:
                    return "N/A" if np.isnan(v) else f"{v:.4f}"
                except Exception:
                    return "N/A"
            # Symmetric: perm-p(orient) is N/A (skipped); perm-p(pair) = cross-pool
            perm_orient_label = "perm-p(orient)" if exp == "asymmetric" else "perm-p(orient)[skipped]"
            perm_pair_label   = "perm-p(pair)"   if exp == "asymmetric" else "perm-p(cross-pool)"
            self._log(f"       t-p={st['t_p']:.2e}  "
                      f"{perm_orient_label}={_fmt_p(st.get('perm_p_orient', st['perm_p']))}  "
                      f"{perm_pair_label}={_fmt_p(st.get('perm_p_pair', st['perm_p']))}  "
                      f"errors={st['n_errors']}  "
                      f"cos_mean={st['cos_mean']:.4f}  "
                      f"Cohen's d={st['cohens_d']:+.3f}  "
                      f"95%CI=[{st['ci_lo']:+.4f},{st['ci_hi']:+.4f}]")
            # Error breakdown log  [v14]
            bkd = st.get("error_breakdown", {})
            if bkd and st["n_errors"] > 0:
                fd = bkd.get("false_direction",     0)
                nz = bkd.get("near_zero_ambiguity", 0)
                ol = bkd.get("extreme_outlier",     0)
                ne = st["n_errors"]
                self._log(f"       error breakdown → "
                          f"false_direction={fd} ({fd/ne:.0%})  "
                          f"near_zero={nz} ({nz/ne:.0%})  "
                          f"outlier={ol} ({ol/ne:.0%})  "
                          f"[τ_nz={bkd.get('near_zero_tau',0.02):.3f} rad  "
                          f"τ_ol={bkd.get('outlier_tau',0):.4f} rad]")

            # ── [v15] Real vs Random comparison ──────────────────────────────
            rnd_res = self.results_rand.get(key) if self.results_rand else None
            if rnd_res and rnd_res.get("n", 0) > 0:
                st_r = rnd_res["stats"]
                real_da  = st["dir_acc"]
                rand_da  = st_r["dir_acc"]
                real_adt = st["mean_abs_dt"]
                rand_adt = st_r["mean_abs_dt"]
                delta_da = real_da - rand_da
                # Interpretation
                if exp == "asymmetric":
                    geom_note = (
                        "geometry signal ✔" if rand_adt > 0.002
                        else "geometry signal weak")
                    sem_note = (
                        "semantic signal ✔ (real >> random)"
                        if delta_da > 0.10
                        else ("semantic signal marginal"
                              if delta_da > 0.03
                              else "semantic signal weak/absent"))
                    self._log(
                        f"       [v15 Real vs Random]  "
                        f"real_dir_acc={real_da:.1%}  rand_dir_acc={rand_da:.1%}  "
                        f"Δdir_acc={delta_da:+.1%}  |  "
                        f"real_mean|Δθ|={real_adt:.4f}  rand_mean|Δθ|={rand_adt:.4f}  |  "
                        f"{geom_note}  |  {sem_note}")
                else:
                    sym_real = st["dir_acc"]    # P(|Δθ|<τ)
                    sym_rand = st_r["dir_acc"]
                    self._log(
                        f"       [v15 Real vs Random]  "
                        f"real_sym_score={sym_real:.1%}  rand_sym_score={sym_rand:.1%}  |  "
                        f"real_mean|Δθ|={real_adt:.4f}  rand_mean|Δθ|={rand_adt:.4f}  |  "
                        f"{'Real more symmetric ✔' if real_adt < rand_adt else 'Real not more symmetric ⚠'}")

        self._log(f"\n  Cosine Similarity: Δcos ≡ 0 for ALL relations")
        self._log(f"  → Cannot distinguish direction or hierarchy.")

        # ── [v15] Overall real vs random summary ─────────────────────────────
        if self.results_rand:
            self._log(f"\n  {'─'*60}")
            self._log(f"  [v15] REAL vs RANDOM EMBEDDING — Summary")
            self._log(f"  {'─'*60}")
            for key, label, exp in datasets:
                rnd_res = self.results_rand.get(key)
                if key not in self.results or not rnd_res:
                    continue
                st   = self.results[key]["stats"]
                st_r = rnd_res["stats"]
                if exp == "asymmetric":
                    self._log(
                        f"  {label:<30}  real={st['dir_acc']:.1%}  "
                        f"rand={st_r['dir_acc']:.1%}  "
                        f"Δ={st['dir_acc']-st_r['dir_acc']:+.1%}  "
                        f"real|Δθ|={st['mean_abs_dt']:.4f}  "
                        f"rand|Δθ|={st_r['mean_abs_dt']:.4f}")
                else:
                    self._log(
                        f"  {label:<30}  real_sym={st['dir_acc']:.1%}  "
                        f"rand_sym={st_r['dir_acc']:.1%}  "
                        f"real|Δθ|={st['mean_abs_dt']:.4f}  "
                        f"rand|Δθ|={st_r['mean_abs_dt']:.4f}")
            self._log(f"  {'─'*60}")
            self._log(
                f"  Hypothesis check:\n"
                f"    ✔ geometric artifact → rand mean|Δθ| > 0\n"
                f"    ✔ semantic signal    → real dir_acc >> rand dir_acc (~50%)")

        self._log(sep + "\n")

    def _refresh_table(self):
        self.main_tree.delete(*self.main_tree.get_children())
        def _fmt_q(v):
            try:
                if np.isnan(v): return "—"
            except Exception: return "—"
            return f"{v:.4f}{'✔' if v < 0.05 else ''}"
        for key, label, expected, _ in DS_CONFIG:
            if key not in self.results:
                continue
            r  = self.results[key]
            st = r["stats"]
            tipe = "Asymmetric" if expected == "asymmetric" else "Symmetric"
            cos_dir = "50% (no directional signal)"
            # Error breakdown compact string  [v14]
            bkd = st.get("error_breakdown", {})
            ne  = st["n_errors"]
            if bkd and ne > 0:
                fd = bkd.get("false_direction",     0)
                nz = bkd.get("near_zero_ambiguity", 0)
                ol = bkd.get("extreme_outlier",     0)
                n_errors_str = f"{ne}  (fd:{fd} nz:{nz} ol:{ol})"
            else:
                n_errors_str = f"{ne}"
            self.main_tree.insert("","end",values=(
                label,
                tipe,
                f"{st['n']:,}",
                f"{st['skipped']:,}",
                st.get("dir_acc_display", f"{st['dir_acc']:.1%}"),
                f"[{st['dir_acc_ci_lo']:.1%}, {st['dir_acc_ci_hi']:.1%}]",
                st.get("ranking_acc_display", f"{st['ranking_acc']:.1%}"),
                f"[{st['rank_acc_ci_lo']:.1%}, {st['rank_acc_ci_hi']:.1%}]",
                f"{st['mean_dt']:+.4f}",
                f"[{st['mean_dt_ci_lo']:+.4f}, {st['mean_dt_ci_hi']:+.4f}]",
                f"{st['std_dt']:.4f}",
                f"{st['mean_abs_dt']:.4f}",
                f"{st['n_uncertain']}",
                f"{st.get('skewness', 0.0):+.3f}",
                f"{st.get('kurtosis', 0.0):+.3f}",
                f"{st.get('bimodality_coeff', 0.0):.3f}",
                f"{st.get('entropy', 0.0):.3f}",
                f"{st['t_stat']:.2f}",
                f"{st['t_p']:.2e}",
                _fmt_q(st.get("t_q", float("nan"))),
                _fmt_q(st.get("perm_p_orient", st["perm_p"])),
                _fmt_q(st.get("perm_q_orient", float("nan"))),
                _fmt_q(st.get("perm_p_pair", st["perm_p"])),
                _fmt_q(st.get("perm_q_pair", float("nan"))),
                n_errors_str,
                f"{st['cos_mean']:.4f}",
                f"{st['cos_std']:.4f}",
                cos_dir,
                f"{st['cohens_d']:+.3f}",
                f"[{st['ci_lo']:+.4f}, {st['ci_hi']:+.4f}]",
                st["verdict"],
            ))

# ===========================================================================
if __name__ == "__main__":
    App().mainloop()
