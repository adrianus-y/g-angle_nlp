"""
APP 2 — Embedding Manager
=========================
Mengelola semua embedding dan caching vector.

Analisis yang dilakukan:
  • load FastText (.vec / .bin)   — word-level + subword
  • load GloVe (.txt)             — word-level pre-trained
  • load word2vec (.vec/.txt)     — format identik dengan FastText .vec
  • vector normalization          — L2-normalize setiap vector sebelum disimpan
  • vocabulary checking           — ringkasan vocab & coverage vs kata-kata target
  • vector caching (.pkl)         — simpan/load dict word→np.array cepat
  • random baseline generation    — N(μ, σ²) dikalibrasi dari statistik embedding asli
  • embedding statistics (.json)  — dimensi, vocab size, norm stats, coverage

Input:
  - FastText model  .bin / .vec
  - GloVe embedding .txt
  - word2vec        .vec / .txt
  - optional cache  .pkl   (skip reload jika sudah ada)

Output:
  - cached_vectors.pkl       — dict word → np.float32 array
  - vocab_list.txt           — satu kata per baris
  - randomized_embedding.pkl — dict word → Gaussian vector (kalibrasi)
  - embedding_stats.json     — metadata embedding

Requirements:
  pip install numpy
  # Untuk FastText .bin:
  pip install fasttext-wheel   (atau fasttext)
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import os
import json
import pickle
import time
import numpy as np

# ===========================================================================
# CORE LOADERS
# ===========================================================================

def load_fasttext_vec(path: str, log, normalize: bool = False) -> dict:
    """
    Load FastText / word2vec .vec format.
    Baris pertama: vocab_size dim
    Baris berikutnya: word f1 f2 … fd
    """
    log("Loading .vec file…")
    vocab: dict[str, np.ndarray] = {}
    t0 = time.time()
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().split()
        n_words = int(header[0]) if header else 0
        dim_hdr = int(header[1]) if len(header) > 1 else 0
        log(f"  Header: {n_words:,} words × {dim_hdr} dims")
        for i, line in enumerate(f):
            parts = line.rstrip().split(" ")
            if len(parts) < 2:
                continue
            word = parts[0].lower()
            try:
                vec = np.array(parts[1:], dtype=np.float32)
                if normalize:
                    nrm = np.linalg.norm(vec)
                    if nrm > 1e-9:
                        vec = vec / nrm
                vocab[word] = vec
            except ValueError:
                continue
            if (i + 1) % 200_000 == 0:
                elapsed = time.time() - t0
                log(f"  {i+1:,} words … ({elapsed:.1f}s)")
    elapsed = time.time() - t0
    log(f"  Done: {len(vocab):,} words in {elapsed:.1f}s")
    return vocab


def load_fasttext_bin(path: str, log, normalize: bool = False) -> dict:
    """Load FastText .bin model (requires fasttext-wheel or fasttext)."""
    try:
        import fasttext
    except ImportError:
        raise ImportError(
            "fasttext package required for .bin files.\n"
            "Install with:  pip install fasttext-wheel"
        )
    log("Loading FastText .bin…")
    t0 = time.time()
    m = fasttext.load_model(path)
    words = m.get_words()
    vocab: dict[str, np.ndarray] = {}
    for word in words:
        w = word.lower()
        vec = np.array(m.get_word_vector(w), dtype=np.float32)
        if normalize:
            nrm = np.linalg.norm(vec)
            if nrm > 1e-9:
                vec = vec / nrm
        vocab[w] = vec
    elapsed = time.time() - t0
    log(f"  Done: {len(vocab):,} words in {elapsed:.1f}s")
    return vocab


def load_glove(path: str, log, normalize: bool = False) -> dict:
    """
    Load GloVe .txt vectors.
    Format: word f1 f2 … fd  (no header line)
    Download: https://nlp.stanford.edu/projects/glove/
    """
    log("Loading GloVe .txt…")
    t0 = time.time()
    vocab: dict[str, np.ndarray] = {}
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            parts = line.rstrip().split(" ")
            if len(parts) < 2:
                continue
            word = parts[0].lower()
            try:
                vec = np.array(parts[1:], dtype=np.float32)
                if normalize:
                    nrm = np.linalg.norm(vec)
                    if nrm > 1e-9:
                        vec = vec / nrm
                vocab[word] = vec
            except ValueError:
                continue
            if (i + 1) % 200_000 == 0:
                elapsed = time.time() - t0
                log(f"  {i+1:,} words … ({elapsed:.1f}s)")
    elapsed = time.time() - t0
    log(f"  Done: {len(vocab):,} words in {elapsed:.1f}s")
    return vocab


def load_any_embedding(path: str, log, normalize: bool = False) -> dict:
    """Auto-detect format from extension and load."""
    ext = os.path.splitext(path)[-1].lower()
    if ext == ".bin":
        return load_fasttext_bin(path, log, normalize)
    elif ext in (".vec", ".txt"):
        # Both GloVe .txt and FastText .vec use the same line format
        # (difference: .vec has a header line, .txt may not)
        # Strategy: peek at line 1 — if it's "N D" treat as .vec, else .txt
        with open(path, "r", encoding="utf-8") as f:
            first = f.readline().strip().split()
        if len(first) == 2 and all(t.isdigit() for t in first):
            log("Detected FastText/word2vec .vec format (has header)")
            return load_fasttext_vec(path, log, normalize)
        else:
            log("Detected GloVe .txt format (no header)")
            return load_glove(path, log, normalize)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")


# ===========================================================================
# NORMALIZATION
# ===========================================================================

def normalize_vocab(vocab: dict, log) -> dict:
    """L2-normalize all vectors in-place (returns same dict)."""
    log(f"Normalizing {len(vocab):,} vectors…")
    t0 = time.time()
    for word in vocab:
        v = vocab[word]
        nrm = np.linalg.norm(v)
        if nrm > 1e-9:
            vocab[word] = v / nrm
    log(f"  Done in {time.time()-t0:.1f}s")
    return vocab


# ===========================================================================
# VOCABULARY UTILITIES
# ===========================================================================

def get_vocab_set(vocab: dict) -> set:
    return set(vocab.keys())


def check_coverage(vocab: dict, target_words: list[str], log) -> dict:
    """Return coverage statistics for a list of target words."""
    vocab_set = get_vocab_set(vocab)
    in_vocab   = [w for w in target_words if w in vocab_set]
    oov        = [w for w in target_words if w not in vocab_set]
    coverage   = len(in_vocab) / len(target_words) if target_words else 0.0
    result = {
        "total_target": len(target_words),
        "in_vocab":     len(in_vocab),
        "oov":          len(oov),
        "coverage_pct": coverage * 100,
        "oov_words":    oov[:50],  # first 50 for reporting
    }
    log(f"  Coverage: {len(in_vocab):,}/{len(target_words):,} "
        f"({coverage:.1%})   OOV: {len(oov):,}")
    return result


# ===========================================================================
# CACHING
# ===========================================================================

def save_cache(vocab: dict, path: str, log) -> None:
    log(f"Saving cache → {path}…")
    t0 = time.time()
    with open(path, "wb") as f:
        pickle.dump(vocab, f, protocol=4)
    size_mb = os.path.getsize(path) / 1024 / 1024
    log(f"  Saved {len(vocab):,} vectors  ({size_mb:.1f} MB)  in {time.time()-t0:.1f}s")


def load_cache(path: str, log) -> dict:
    log(f"Loading cache ← {path}…")
    t0 = time.time()
    with open(path, "rb") as f:
        vocab = pickle.load(f)
    log(f"  Loaded {len(vocab):,} vectors in {time.time()-t0:.1f}s")
    return vocab


def save_vocab_list(vocab: dict, path: str, log) -> None:
    log(f"Saving vocab list → {path}…")
    with open(path, "w", encoding="utf-8") as f:
        for word in sorted(vocab.keys()):
            f.write(word + "\n")
    log(f"  {len(vocab):,} words written")


# ===========================================================================
# RANDOMIZED BASELINE
# ===========================================================================

def build_randomized_embedding(
    vocab: dict,
    seed: int = 42,
    sample_n: int = 2000,
    log = None,
) -> dict:
    """
    Build a randomized embedding N(μ, σ²) calibrated to the real embedding.

    Scientific rationale (from evaluasi-16):
      • If asymmetry persists  → geometry of g-angle is the source
      • If direction accuracy → ~50% → semantic directional info is absent
    Separates geometric artifact from semantic directional bias.

    Returns a NEW dict with the same keys but Gaussian vectors.
    """
    if log: log(f"Building randomized embedding  (seed={seed}, sample={sample_n})…")

    words   = list(vocab.keys())
    sample  = words[:sample_n]
    vecs    = np.stack([vocab[w] for w in sample], axis=0)
    mu      = vecs.mean(axis=0)
    std     = vecs.std(axis=0)
    std[std < 1e-9] = 1e-9
    dim     = vecs.shape[1]
    if log: log(f"  Calibrated from {len(sample):,} vectors  dim={dim}")

    rand_vocab: dict[str, np.ndarray] = {}
    for word in words:
        sub_seed = abs(hash(word)) % (2**31)
        rng = np.random.default_rng(sub_seed + seed)
        rand_vocab[word] = (rng.standard_normal(dim) * std + mu).astype(np.float32)

    if log: log(f"  Done: {len(rand_vocab):,} random vectors")
    return rand_vocab


# ===========================================================================
# EMBEDDING STATISTICS
# ===========================================================================

def compute_stats(
    name: str,
    vocab: dict,
    normalized: bool,
    log = None,
) -> dict:
    """Compute and return embedding_stats dict (also saves to JSON)."""
    if not vocab:
        return {"error": "empty vocab"}

    sample_vecs = np.stack(list(vocab.values())[:5000], axis=0)
    norms       = np.linalg.norm(sample_vecs, axis=1)
    dim         = sample_vecs.shape[1]

    stats = {
        "embedding":   name,
        "dimension":   int(dim),
        "vocab_size":  len(vocab),
        "normalized":  normalized,
        "norm_mean":   float(norms.mean()),
        "norm_std":    float(norms.std()),
        "norm_min":    float(norms.min()),
        "norm_max":    float(norms.max()),
        "dtype":       str(sample_vecs.dtype),
    }
    if log:
        log(f"  Stats: dim={dim}  vocab={len(vocab):,}  "
            f"norm_mean={norms.mean():.4f}  norm_std={norms.std():.4f}  "
            f"normalized={normalized}")
    return stats


def save_stats_json(stats: dict, path: str, log) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    log(f"  Stats saved → {path}")


# ===========================================================================
# FULL PIPELINE
# ===========================================================================

def run_pipeline(cfg: dict, log) -> dict:
    """
    cfg keys:
      embedding_name  str   — label (FastText / GloVe / word2vec / …)
      file_path       str   — path to .vec / .txt / .bin
      cache_path      str | None — load from .pkl if exists, else save to this path
      normalize       bool
      build_rand      bool
      rand_seed       int
      rand_sample_n   int
      out_dir         str   — where to write outputs
    Returns dict with loaded vocab (get_vec callable) + stats.
    """
    out = cfg["out_dir"]
    os.makedirs(out, exist_ok=True)
    log("=" * 60)
    log(f"APP 2 — Embedding Manager  [{cfg['embedding_name']}]")
    log("=" * 60)

    # ── 1. Load or restore cache ─────────────────────────────────
    cache_path = cfg.get("cache_path", "").strip() or None
    vocab      = None

    if cache_path and os.path.isfile(cache_path):
        log(f"\n[1] Cache found — loading from {cache_path}")
        try:
            vocab = load_cache(cache_path, log)
        except Exception as e:
            log(f"  [WARN] Cache load failed: {e} — falling back to file load")
            vocab = None

    if vocab is None:
        file_path = cfg["file_path"].strip()
        if not file_path or not os.path.isfile(file_path):
            raise FileNotFoundError(f"Embedding file not found: {file_path!r}")
        log(f"\n[1] Loading from file: {file_path}")
        vocab = load_any_embedding(file_path, log, normalize=False)

    # ── 2. Normalization ─────────────────────────────────────────
    normalized = cfg.get("normalize", False)
    if normalized:
        log("\n[2] Normalizing vectors")
        vocab = normalize_vocab(vocab, log)
    else:
        log("\n[2] Normalization — SKIPPED")

    # ── 3. Statistics ────────────────────────────────────────────
    log("\n[3] Computing embedding statistics")
    stats = compute_stats(cfg["embedding_name"], vocab, normalized, log)

    stats_path = os.path.join(out, "embedding_stats.json")
    save_stats_json(stats, stats_path, log)

    # ── 4. Save cache ────────────────────────────────────────────
    log("\n[4] Saving cache")
    out_cache = os.path.join(out, "cached_vectors.pkl")
    save_cache(vocab, out_cache, log)

    # ── 5. Save vocab list ───────────────────────────────────────
    log("\n[5] Saving vocab list")
    vocab_path = os.path.join(out, "vocab_list.txt")
    save_vocab_list(vocab, vocab_path, log)

    # ── 6. Randomized baseline ───────────────────────────────────
    rand_vocab = None
    if cfg.get("build_rand", False):
        log("\n[6] Building randomized embedding baseline")
        rand_vocab = build_randomized_embedding(
            vocab,
            seed=cfg.get("rand_seed", 42),
            sample_n=cfg.get("rand_sample_n", 2000),
            log=log,
        )
        rand_path = os.path.join(out, "randomized_embedding.pkl")
        save_cache(rand_vocab, rand_path, log)
    else:
        log("\n[6] Randomized baseline — SKIPPED")

    # ── Done ─────────────────────────────────────────────────────
    log("\n" + "=" * 60)
    log(f"DONE  —  {stats['vocab_size']:,} words  |  dim={stats['dimension']}  "
        f"|  normalized={normalized}")
    log("=" * 60)

    return {
        "vocab":       vocab,
        "rand_vocab":  rand_vocab,
        "stats":       stats,
        "out_cache":   out_cache,
        "vocab_path":  vocab_path,
        "stats_path":  stats_path,
    }


# ===========================================================================
# HELPER — make_get_vec
# ===========================================================================

def make_get_vec(vocab: dict):
    """Return a get_vec(word) callable from a vocab dict."""
    return lambda w: vocab.get(w.lower(), None)


# ===========================================================================
# GUI
# ===========================================================================

BG     = "#1a1a2e"
BG2    = "#16213e"
FG     = "#e0e0e0"
FG2    = "#90caf9"
FG3    = "#b0bec5"
ACCENT = "#0d47a1"
ACC2   = "#1565c0"
GREEN  = "#2e7d32"
RED    = "#c62828"
ORANGE = "#e65100"
PURPLE = "#4a148c"



class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("APP 2 — Embedding Manager")
        self.geometry("860x780")
        self.configure(bg=BG)
        self.resizable(True, True)

        # State
        self.loaded_vocabs: dict[str, dict] = {}   # name → vocab dict
        self.loaded_stats:  dict[str, dict] = {}   # name → stats dict
        self._active_run = False

        self._build_style()
        self._build_ui()

    # ── Style ────────────────────────────────────────────────────
    def _build_style(self):
        st = ttk.Style()
        st.theme_use("clam")
        st.configure("TFrame",     background=BG)
        st.configure("TLabelframe", background=BG, foreground=FG2,
                     font=("Consolas", 9, "bold"))
        st.configure("TLabelframe.Label", background=BG, foreground=FG2)
        st.configure("Treeview",   background=BG2, foreground=FG,
                     fieldbackground=BG2, rowheight=22)
        st.configure("Treeview.Heading", background="#0f3460", foreground=FG2,
                     font=("Consolas", 8, "bold"))
        st.configure("TProgressbar", troughcolor="#0f3460", background="#42a5f5")

    def _lbl(self, parent, text="", fg=None, **kw):
        fg = fg or FG3
        defaults = dict(bg=BG, fg=fg, font=("Consolas", 9))
        defaults.update(kw)
        return tk.Label(parent, text=text, **defaults)

    def _entry(self, parent, textvariable, width=40):
        return tk.Entry(parent, textvariable=textvariable, width=width,
                        bg=BG2, fg=FG, insertbackground="#fff",
                        relief="flat", font=("Consolas", 9))

    def _btn(self, parent, text, cmd, color=None, **kw):
        color = color or ACCENT
        defaults = dict(bg=color, fg="#fff", relief="flat",
                        font=("Consolas", 9, "bold"),
                        activebackground="#42a5f5", cursor="hand2")
        defaults.update(kw)   # caller's kw overrides defaults (e.g. custom font)
        return tk.Button(parent, text=text, command=cmd, **defaults)

    def _tree(self, parent, cols, widths, height=5):
        t = ttk.Treeview(parent, columns=cols, show="headings", height=height)
        for c, w in zip(cols, widths):
            t.heading(c, text=c)
            t.column(c, width=w, anchor="center", minwidth=w)
        return t

    # ── Build UI ─────────────────────────────────────────────────
    def _build_ui(self):
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=6, pady=6)

        tabs = [("tab_load",  " ⚙  Load & Config "),
                ("tab_vocab", " 📚  Vocab & Coverage "),
                ("tab_stats", " 📊  Embedding Stats "),
                ("tab_rand",  " 🎲  Random Baseline "),
                ("tab_log",   " 📝  Log ")]
        for attr, label in tabs:
            f = ttk.Frame(nb)
            setattr(self, attr, f)
            nb.add(f, text=label)

        self._build_tab_load()
        self._build_tab_vocab()
        self._build_tab_stats()
        self._build_tab_rand()
        self._build_tab_log()

    # ── Tab: Load & Config ───────────────────────────────────────
    def _build_tab_load(self):
        pad = dict(padx=10, pady=6)

        # FastText / word2vec
        lf1 = ttk.LabelFrame(self.tab_load, text="1.  FastText / word2vec  (.vec  .bin)", padding=8)
        lf1.pack(fill="x", **pad)
        self._lbl(lf1, "File:").grid(row=0, column=0, sticky="w")
        self.ft_path = tk.StringVar()
        self._entry(lf1, self.ft_path, 52).grid(row=0, column=1, padx=6)
        self._btn(lf1, "Browse…", self._browse_ft, ACC2).grid(row=0, column=2)
        self.ft_status = self._lbl(lf1, "● Not loaded", fg="#ef5350")
        self.ft_status.grid(row=1, column=0, columnspan=3, sticky="w", pady=(2, 0))
        self._btn(lf1, "▶  Load FastText / word2vec",
                  self._load_ft, GREEN, pady=3).grid(row=2, column=0, columnspan=3,
                                                     pady=(6, 0), sticky="w")

        # GloVe
        lf2 = ttk.LabelFrame(self.tab_load, text="2.  GloVe  (.txt)", padding=8)
        lf2.pack(fill="x", **pad)
        self._lbl(lf2, "File:").grid(row=0, column=0, sticky="w")
        self.glove_path = tk.StringVar()
        self._entry(lf2, self.glove_path, 52).grid(row=0, column=1, padx=6)
        self._btn(lf2, "Browse…", self._browse_glove, ACC2).grid(row=0, column=2)
        self._lbl(lf2, "  ",
                  fg="#546e7a").grid(row=1, column=0, columnspan=3, sticky="w", pady=(2, 0))
        self._btn(lf2, "▶  Load GloVe",
                  self._load_glove, "#00695c", pady=3).grid(row=2, column=0,
                                                             columnspan=3, pady=(6, 0), sticky="w")
        self.glove_status = self._lbl(lf2, "● Not loaded", fg="#78909c")
        self.glove_status.grid(row=3, column=0, columnspan=3, sticky="w", pady=(4, 0))

        # Options
        lf3 = ttk.LabelFrame(self.tab_load, text="3.  Options", padding=8)
        lf3.pack(fill="x", **pad)

        # Normalize checkbox
        self._normalize = tk.BooleanVar(value=False)
        tk.Checkbutton(lf3, text="L2-normalize all vectors after loading",
                       variable=self._normalize, bg=BG, fg=FG2,
                       activebackground=BG, selectcolor="#0f3460",
                       font=("Consolas", 9)).grid(row=0, column=0, sticky="w")

        # Cache
        cf = tk.Frame(lf3, bg=BG); cf.grid(row=1, column=0, sticky="ew", pady=(4, 0))
        self._lbl(cf, "Cache .pkl (load from / save to):").pack(side="left")
        self.cache_path = tk.StringVar()
        self._entry(cf, self.cache_path, 36).pack(side="left", padx=6)
        self._btn(cf, "Browse…", self._browse_cache, ACC2).pack(side="left")

        # Output dir
        of = tk.Frame(lf3, bg=BG); of.grid(row=2, column=0, sticky="ew", pady=(4, 0))
        self._lbl(of, "Output directory:").pack(side="left")
        self.out_dir = tk.StringVar(value=os.path.join(
            os.path.expanduser("~"), "embedding_manager_output"))
        self._entry(of, self.out_dir, 40).pack(side="left", padx=6)
        self._btn(of, "Browse…", self._browse_out, ACC2).pack(side="left")

        # Random baseline options
        rbf = tk.Frame(lf3, bg=BG); rbf.grid(row=3, column=0, sticky="ew", pady=(6, 0))
        self._build_rand = tk.BooleanVar(value=True)
        tk.Checkbutton(rbf, text="Build randomized baseline embedding",
                       variable=self._build_rand, bg=BG, fg=FG2,
                       activebackground=BG, selectcolor="#0f3460",
                       font=("Consolas", 9)).pack(side="left")
        self._lbl(rbf, "  seed:").pack(side="left")
        self._rand_seed = tk.StringVar(value="42")
        tk.Entry(rbf, textvariable=self._rand_seed, width=6,
                 bg=BG2, fg=FG, insertbackground="#fff",
                 relief="flat", font=("Consolas", 9)).pack(side="left", padx=4)
        self._lbl(rbf, "  sample_n:").pack(side="left")
        self._rand_sample = tk.StringVar(value="2000")
        tk.Entry(rbf, textvariable=self._rand_sample, width=8,
                 bg=BG2, fg=FG, insertbackground="#fff",
                 relief="flat", font=("Consolas", 9)).pack(side="left", padx=4)

        # Run pipeline
        lf4 = ttk.LabelFrame(self.tab_load, text="4.  Run Full Pipeline", padding=8)
        lf4.pack(fill="x", **pad)
        bf = tk.Frame(lf4, bg=BG); bf.pack(fill="x")
        self._run_btn = self._btn(bf, "▶  Run Full Pipeline", self._start_pipeline,
                                  PURPLE, pady=5, padx=20,
                                  font=("Consolas", 10, "bold"))
        self._run_btn.pack(side="left", padx=4)
        self._lbl(bf, "  ← loads embedding, normalizes, saves cache, vocab, stats, rand").pack(
            side="left", padx=4)
        self._pbar = ttk.Progressbar(lf4, mode="indeterminate", length=300)
        self._pbar.pack(fill="x", pady=(6, 0))

    # ── Tab: Vocab & Coverage ─────────────────────────────────────
    def _build_tab_vocab(self):
        pad = dict(padx=10, pady=6)

        ctrl = tk.Frame(self.tab_vocab, bg=BG); ctrl.pack(fill="x", **pad)
        self._lbl(ctrl, "Embedding:").pack(side="left")
        self._vocab_emb = tk.StringVar(value="")
        self._vocab_emb_om = tk.OptionMenu(ctrl, self._vocab_emb, "")
        self._vocab_emb_om.config(bg=BG2, fg=FG, activebackground=ACCENT,
                                   highlightbackground=BG, relief="flat",
                                   font=("Consolas", 9), width=20)
        self._vocab_emb_om["menu"].config(bg=BG2, fg=FG, font=("Consolas", 9))
        self._vocab_emb_om.pack(side="left", padx=6)

        self._lbl(ctrl, "  Target words (one per line):").pack(side="left")
        self._btn(ctrl, "Load .txt…", self._browse_target_words, ACC2).pack(side="left", padx=4)
        self._btn(ctrl, "▶  Check Coverage", self._check_coverage, GREEN).pack(side="left", padx=4)

        self._target_words_path = tk.StringVar()
        self._target_lbl = self._lbl(ctrl, "No target file loaded", fg="#78909c")
        self._target_lbl.pack(side="left", padx=8)

        lf = ttk.LabelFrame(self.tab_vocab, text="Coverage Report", padding=8)
        lf.pack(fill="both", expand=True, **pad)
        cols = ("Word", "In Vocab", "Note")
        widths = [200, 80, 400]
        self.vocab_tree = self._tree(lf, cols, widths, height=16)
        vsb = ttk.Scrollbar(lf, orient="vertical", command=self.vocab_tree.yview)
        self.vocab_tree.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        self.vocab_tree.pack(fill="both", expand=True)

        self.coverage_lbl = self._lbl(self.tab_vocab, "", fg="#42a5f5",
                                       font=("Consolas", 10, "bold"))
        self.coverage_lbl.pack(pady=4)

    # ── Tab: Embedding Stats ──────────────────────────────────────
    def _build_tab_stats(self):
        pad = dict(padx=10, pady=6)

        lf = ttk.LabelFrame(self.tab_stats, text="Loaded Embeddings", padding=8)
        lf.pack(fill="x", **pad)
        cols = ("Name", "Dimension", "Vocab Size", "Normalized", "norm_mean",
                "norm_std", "dtype", "Cache Path")
        widths = [130, 80, 100, 90, 90, 80, 80, 280]
        self.stats_tree = self._tree(lf, cols, widths, height=6)
        xsb = ttk.Scrollbar(lf, orient="horizontal", command=self.stats_tree.xview)
        self.stats_tree.configure(xscrollcommand=xsb.set)
        self.stats_tree.pack(fill="both", expand=True)
        xsb.pack(fill="x")

        bf = tk.Frame(self.tab_stats, bg=BG); bf.pack(pady=4)
        self._btn(bf, "📋  Export stats.json", self._export_stats, "#1b5e20").pack(side="left", padx=6)
        self._btn(bf, "🔄  Refresh", self._refresh_stats, ACC2).pack(side="left", padx=6)

    # ── Tab: Random Baseline ─────────────────────────────────────
    def _build_tab_rand(self):
        pad = dict(padx=10, pady=6)

        info = tk.Label(self.tab_rand, bg=BG, fg=FG3, font=("Consolas", 9),
            text=(
                "Randomized baseline embedding:  N(μ, σ²) calibrated from real embedding statistics.\n"
                "Hypothesis (Gunawan et al.): if asymmetry persists → geometry is the source;\n"
                "if direction accuracy → 50% → semantic directional information is absent."
            ), justify="left")
        info.pack(fill="x", **pad)

        ctrl = tk.Frame(self.tab_rand, bg=BG); ctrl.pack(fill="x", **pad)
        self._lbl(ctrl, "Source embedding:").pack(side="left")
        self._rand_emb_sel = tk.StringVar(value="")
        self._rand_emb_om = tk.OptionMenu(ctrl, self._rand_emb_sel, "")
        self._rand_emb_om.config(bg=BG2, fg=FG, activebackground=ACCENT,
                                  highlightbackground=BG, relief="flat",
                                  font=("Consolas", 9), width=20)
        self._rand_emb_om["menu"].config(bg=BG2, fg=FG, font=("Consolas", 9))
        self._rand_emb_om.pack(side="left", padx=6)

        self._lbl(ctrl, "  seed:").pack(side="left")
        self._rand_seed2 = tk.StringVar(value="42")
        tk.Entry(ctrl, textvariable=self._rand_seed2, width=6,
                 bg=BG2, fg=FG, insertbackground="#fff",
                 relief="flat", font=("Consolas", 9)).pack(side="left", padx=4)
        self._lbl(ctrl, "  sample_n:").pack(side="left")
        self._rand_sample2 = tk.StringVar(value="2000")
        tk.Entry(ctrl, textvariable=self._rand_sample2, width=8,
                 bg=BG2, fg=FG, insertbackground="#fff",
                 relief="flat", font=("Consolas", 9)).pack(side="left", padx=4)
        self._btn(ctrl, "▶  Build Randomized Embedding",
                  self._build_rand_now, PURPLE, pady=4).pack(side="left", padx=10)

        lf = ttk.LabelFrame(self.tab_rand, text="Randomized Embedding Status", padding=8)
        lf.pack(fill="x", **pad)
        self.rand_status_lbl = self._lbl(lf,
            "● Not built  (load an embedding first)", fg="#78909c")
        self.rand_status_lbl.pack(anchor="w")

        # Stats comparison
        lf2 = ttk.LabelFrame(self.tab_rand, text="Stats Comparison: Real vs Random", padding=8)
        lf2.pack(fill="both", expand=True, **pad)
        cols2 = ("Property", "Real Embedding", "Random Baseline")
        widths2 = [160, 200, 200]
        self.rand_cmp_tree = self._tree(lf2, cols2, widths2, height=8)
        self.rand_cmp_tree.pack(fill="both", expand=True)

    # ── Tab: Log ─────────────────────────────────────────────────
    def _build_tab_log(self):
        bf = tk.Frame(self.tab_log, bg=BG); bf.pack(fill="x", padx=8, pady=4)
        self._btn(bf, "Clear", self._clear_log, "#37474f").pack(side="left", padx=4)
        self.log_box = scrolledtext.ScrolledText(
            self.tab_log, state="disabled",
            bg="#0d1117", fg="#c9d1d9", font=("Consolas", 9),
            insertbackground="#fff")
        self.log_box.pack(fill="both", expand=True, padx=8, pady=4)

    # ── Browse helpers ───────────────────────────────────────────
    def _browse_ft(self):
        p = filedialog.askopenfilename(
            title="Select FastText / word2vec File",
            filetypes=[("FastText/w2v", "*.vec *.bin *.txt"), ("All", "*.*")])
        if p: self.ft_path.set(p)

    def _browse_glove(self):
        p = filedialog.askopenfilename(
            title="Select GloVe .txt File",
            filetypes=[("GloVe", "*.txt"), ("All", "*.*")])
        if p: self.glove_path.set(p)


    def _browse_cache(self):
        p = filedialog.askopenfilename(
            title="Select cache .pkl", filetypes=[("Pickle", "*.pkl"), ("All", "*.*")])
        if p: self.cache_path.set(p)

    def _browse_out(self):
        d = filedialog.askdirectory()
        if d: self.out_dir.set(d)

    def _browse_target_words(self):
        p = filedialog.askopenfilename(
            title="Select target words .txt",
            filetypes=[("Text", "*.txt"), ("All", "*.*")])
        if p:
            self._target_words_path.set(p)
            self._target_lbl.config(text=os.path.basename(p), fg="#42a5f5")

    # ── Log ─────────────────────────────────────────────────────
    def _log(self, msg: str):
        self.log_box.config(state="normal")
        self.log_box.insert("end", msg + "\n")
        self.log_box.see("end")
        self.log_box.config(state="disabled")
        self.update_idletasks()

    def _clear_log(self):
        self.log_box.config(state="normal")
        self.log_box.delete("1.0", "end")
        self.log_box.config(state="disabled")

    def _load_ft(self):
        path = self.ft_path.get().strip()
        if not path or not os.path.isfile(path):
            messagebox.showerror("Error", "FastText file not found.")
            return
        norm = self._normalize.get()
        self.ft_status.config(text="⏳ Loading…", fg="#fb8c00")
        def _run():
            try:
                vocab = load_any_embedding(path, self._log, normalize=norm)
                name  = f"FastText ({os.path.basename(path)})"
                stats = compute_stats(name, vocab, norm, self._log)
                self.loaded_vocabs[name] = vocab
                self.loaded_stats[name]  = stats
                self.after(0, lambda: (
                    self.ft_status.config(
                        text=f"✔ {name} — {len(vocab):,} words", fg="#66bb6a"),
                    self._update_emb_dropdowns(),
                    self._refresh_stats(),
                ))
            except Exception as e:
                self.after(0, lambda: self.ft_status.config(text=f"✖ {e}", fg="#ef5350"))
                self._log(f"[ERROR] {e}")
        threading.Thread(target=_run, daemon=True).start()

    # ── Load GloVe local ──────────────────────────────────────────
    def _load_glove(self):
        path = self.glove_path.get().strip()
        if not path or not os.path.isfile(path):
            messagebox.showerror("Error", "GloVe file not found.")
            return
        norm = self._normalize.get()
        self.glove_status.config(text="⏳ Loading…", fg="#fb8c00")
        def _run():
            try:
                vocab = load_glove(path, self._log, normalize=norm)
                name  = f"GloVe ({os.path.basename(path)})"
                stats = compute_stats(name, vocab, norm, self._log)
                self.loaded_vocabs[name] = vocab
                self.loaded_stats[name]  = stats
                self.after(0, lambda: (
                    self.glove_status.config(
                        text=f"✔ {name} — {len(vocab):,} words", fg="#66bb6a"),
                    self._update_emb_dropdowns(),
                    self._refresh_stats(),
                ))
            except Exception as e:
                self.after(0, lambda: self.glove_status.config(text=f"✖ {e}", fg="#ef5350"))
                self._log(f"[ERROR GloVe] {e}")
        threading.Thread(target=_run, daemon=True).start()

    def _start_pipeline(self):
        file_path = self.ft_path.get().strip()
        # Allow GloVe path fallback
        if not file_path:
            file_path = self.glove_path.get().strip()
        if not file_path:
            messagebox.showerror("Error", "Set a file path in FastText or GloVe section.")
            return
        ext  = os.path.splitext(file_path)[-1].lower()
        name = ("GloVe" if ext == ".txt" and "glove" in file_path.lower()
                else "FastText" if ext in (".vec", ".bin")
                else "word2vec")
        cfg = {
            "embedding_name": name,
            "file_path":      file_path,
            "cache_path":     self.cache_path.get().strip() or None,
            "normalize":      self._normalize.get(),
            "build_rand":     self._build_rand.get(),
            "rand_seed":      int(self._rand_seed.get() or 42),
            "rand_sample_n":  int(self._rand_sample.get() or 2000),
            "out_dir":        self.out_dir.get().strip(),
        }
        self._run_btn.config(state="disabled")
        self._pbar.start(12)
        def _run():
            try:
                result = run_pipeline(cfg, self._log)
                self.loaded_vocabs[name]              = result["vocab"]
                self.loaded_stats[name]               = result["stats"]
                if result.get("rand_vocab"):
                    rname = f"Random N(μ,σ²) [{name}]"
                    self.loaded_vocabs[rname]          = result["rand_vocab"]
                    rand_stats = compute_stats(rname, result["rand_vocab"],
                                               normalized=False, log=self._log)
                    self.loaded_stats[rname]           = rand_stats
                self.after(0, lambda: (
                    self._update_emb_dropdowns(),
                    self._refresh_stats(),
                    messagebox.showinfo("Done",
                        f"Pipeline complete!\n"
                        f"{result['stats']['vocab_size']:,} words  |  "
                        f"dim={result['stats']['dimension']}\n"
                        f"Outputs: {cfg['out_dir']}"),
                ))
            except Exception as e:
                import traceback
                self.after(0, lambda: messagebox.showerror("Error", str(e)))
                self._log(f"\n[ERROR] {e}\n{traceback.format_exc()}")
            finally:
                self.after(0, self._pipeline_done)
        threading.Thread(target=_run, daemon=True).start()

    def _pipeline_done(self):
        self._pbar.stop()
        self._run_btn.config(state="normal")

    # ── Check coverage ────────────────────────────────────────────
    def _check_coverage(self):
        emb_name = self._vocab_emb.get()
        if emb_name not in self.loaded_vocabs:
            messagebox.showerror("Error", "Load an embedding first.")
            return
        tw_path = self._target_words_path.get().strip()
        if not tw_path or not os.path.isfile(tw_path):
            messagebox.showerror("Error", "Load a target words .txt first.")
            return
        vocab = self.loaded_vocabs[emb_name]
        with open(tw_path, "r", encoding="utf-8") as f:
            words = [l.strip().lower() for l in f if l.strip()]
        self._log(f"\nCoverage check: {len(words):,} target words vs {len(vocab):,} vocab")
        result = check_coverage(vocab, words, self._log)

        self.vocab_tree.delete(*self.vocab_tree.get_children())
        for w in words:
            in_v = w in vocab
            self.vocab_tree.insert("", "end", values=(
                w,
                "✔" if in_v else "✖",
                "" if in_v else "OOV",
            ))
        self.coverage_lbl.config(
            text=f"Coverage: {result['in_vocab']:,}/{result['total_target']:,} "
                 f"({result['coverage_pct']:.1f}%)   OOV: {result['oov']:,}")

    # ── Build random baseline (from Rand tab) ─────────────────────
    def _build_rand_now(self):
        emb_name = self._rand_emb_sel.get()
        if emb_name not in self.loaded_vocabs:
            messagebox.showerror("Error", "Load a source embedding first.")
            return
        vocab = self.loaded_vocabs[emb_name]
        seed  = int(self._rand_seed2.get() or 42)
        samp  = int(self._rand_sample2.get() or 2000)
        self.rand_status_lbl.config(text="⏳ Building…", fg="#fb8c00")
        def _run():
            try:
                rand_vocab = build_randomized_embedding(vocab, seed=seed,
                                                         sample_n=samp, log=self._log)
                rname = f"Random N(μ,σ²) [{emb_name}]"
                self.loaded_vocabs[rname] = rand_vocab
                real_s = compute_stats(emb_name,  vocab,      False, None)
                rand_s = compute_stats(rname,      rand_vocab, False, None)
                self.loaded_stats[rname] = rand_s

                rows = [
                    ("Vocab size",   f"{real_s['vocab_size']:,}",  f"{rand_s['vocab_size']:,}"),
                    ("Dimension",    str(real_s["dimension"]),      str(rand_s["dimension"])),
                    ("norm_mean",    f"{real_s['norm_mean']:.4f}",  f"{rand_s['norm_mean']:.4f}"),
                    ("norm_std",     f"{real_s['norm_std']:.4f}",   f"{rand_s['norm_std']:.4f}"),
                    ("norm_min",     f"{real_s['norm_min']:.4f}",   f"{rand_s['norm_min']:.4f}"),
                    ("norm_max",     f"{real_s['norm_max']:.4f}",   f"{rand_s['norm_max']:.4f}"),
                    ("Normalized",   str(real_s["normalized"]),     "False"),
                    ("dtype",        real_s["dtype"],               rand_s["dtype"]),
                ]
                self.after(0, lambda: (
                    self.rand_status_lbl.config(
                        text=f"✔ {rname} ready — {len(rand_vocab):,} words",
                        fg="#66bb6a"),
                    self._fill_rand_cmp(rows),
                    self._update_emb_dropdowns(),
                    self._refresh_stats(),
                ))

                # Save .pkl to out_dir if set
                out = self.out_dir.get().strip()
                if out:
                    os.makedirs(out, exist_ok=True)
                    p = os.path.join(out, "randomized_embedding.pkl")
                    save_cache(rand_vocab, p, self._log)

            except Exception as e:
                self.after(0, lambda: self.rand_status_lbl.config(
                    text=f"✖ {e}", fg="#ef5350"))
                self._log(f"[ERROR Rand] {e}")
        threading.Thread(target=_run, daemon=True).start()

    def _fill_rand_cmp(self, rows):
        self.rand_cmp_tree.delete(*self.rand_cmp_tree.get_children())
        for row in rows:
            self.rand_cmp_tree.insert("", "end", values=row)

    # ── Stats refresh ─────────────────────────────────────────────
    def _refresh_stats(self):
        self.stats_tree.delete(*self.stats_tree.get_children())
        for name, st in self.loaded_stats.items():
            cache_p = ""
            out = self.out_dir.get().strip()
            if out:
                cache_p = os.path.join(out, "cached_vectors.pkl")
            self.stats_tree.insert("", "end", values=(
                name,
                st.get("dimension",  "—"),
                f"{st.get('vocab_size', 0):,}",
                str(st.get("normalized", "—")),
                f"{st.get('norm_mean', 0):.4f}",
                f"{st.get('norm_std',  0):.4f}",
                st.get("dtype", "—"),
                cache_p,
            ))

    def _export_stats(self):
        out = self.out_dir.get().strip()
        if not out:
            messagebox.showerror("Error", "Set output directory first.")
            return
        os.makedirs(out, exist_ok=True)
        all_stats = list(self.loaded_stats.values())
        path = os.path.join(out, "embedding_stats.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(all_stats, f, indent=2, ensure_ascii=False)
        self._log(f"Stats exported → {path}")
        messagebox.showinfo("Saved", f"Saved to:\n{path}")

    # ── Dropdown refresh ──────────────────────────────────────────
    def _update_emb_dropdowns(self):
        names = list(self.loaded_vocabs.keys())
        if not names:
            return
        for var, om in [(self._vocab_emb,    self._vocab_emb_om),
                        (self._rand_emb_sel, self._rand_emb_om)]:
            menu = om["menu"]
            menu.delete(0, "end")
            for n in names:
                menu.add_command(label=n, command=lambda v=n, sv=var: sv.set(v))
            if not var.get() or var.get() not in names:
                var.set(names[0])


# ===========================================================================
if __name__ == "__main__":
    App().mainloop()