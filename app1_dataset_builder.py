"""
APP 1 — Dataset Builder
=======================
Membangun dataset relasi linguistik dari WordNet dan sumber eksternal.

Analisis yang dilakukan:
  • hyponymy extraction        — relasi is-a hierarkis (dog → animal)
  • meronymy extraction        — relasi part-of (wheel → car)
  • sibling extraction         — co-hyponym (cat ↔ dog)
  • coordinate extraction      — pasangan tematik simetris (fruit, vehicle, …)
  • capital-country extraction — relasi kapital → negara (built-in 195+ pasang)
  • balancing                  — subsample relasi dominan agar distribusi seimbang
  • duplicate removal          — hapus pasangan (w1,w2) identik lintas relasi
  • OOV filtering              — buang pasangan yang tidak ada di embedding vocab
  • dataset validation         — cek integritas CSV output

Input:
  - WordNet (NLTK built-in)
  - daftar negara/kapital (opsional, built-in tersedia)
  - embedding vocabulary .txt  (opsional, untuk OOV filter)

Output:
  - dataset.csv          — pasangan kata + label
  - metadata.json        — statistik per relasi
  - oov_report.csv       — pasangan yang dibuang (OOV)
  - duplicate_log.csv    — pasangan duplikat
  - balance_summary.json — distribusi relasi setelah balancing

Requirements:
  pip install nltk
  python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import os
import csv
import json
import random
from collections import defaultdict, Counter

# ─── WordNet (lazy import) ───────────────────────────────────────────────────
_WN = None
def get_wn():
    global _WN
    if _WN is None:
        try:
            from nltk.corpus import wordnet as wn
            list(wn.all_synsets())
            _WN = wn
        except Exception:
            import nltk
            nltk.download("wordnet", quiet=True)
            nltk.download("omw-1.4",  quiet=True)
            from nltk.corpus import wordnet as wn
            _WN = wn
    return _WN

# ===========================================================================
# CAPITAL-COUNTRY DATA (built-in, 195+ pairs)
# ===========================================================================
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
    ("doha","qatar"),
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
# EXTRACTION FUNCTIONS
# ===========================================================================

def extract_hyponymy(max_pairs=2000, log=None):
    """Hyponymy: (hyponym, hypernym) — dog → animal  [asymmetric]"""
    if log: log("Extracting hyponymy pairs from WordNet...")
    wn = get_wn()
    pairs = []
    for syn in wn.all_synsets("n"):
        for hypo in syn.hyponyms():
            for l1 in hypo.lemmas():
                for l2 in syn.lemmas():
                    u = l1.name().replace("_", " ").lower()
                    v = l2.name().replace("_", " ").lower()
                    if u != v and len(u.split()) == 1 and len(v.split()) == 1:
                        pairs.append((u, v, "hyponymy", 0))
                    if len(pairs) >= max_pairs:
                        if log: log(f"  hyponymy: {len(pairs):,} pairs")
                        return pairs
    if log: log(f"  hyponymy: {len(pairs):,} pairs")
    return pairs

def extract_meronymy(max_pairs=2000, log=None):
    """Meronymy: (meronym, holonym) — wheel → car  [asymmetric]"""
    if log: log("Extracting meronymy pairs from WordNet...")
    wn = get_wn()
    pairs = []
    for syn in wn.all_synsets("n"):
        for mero in (syn.part_meronyms() +
                     syn.member_meronyms() +
                     syn.substance_meronyms()):
            for l1 in mero.lemmas():
                for l2 in syn.lemmas():
                    u = l1.name().replace("_", " ").lower()
                    v = l2.name().replace("_", " ").lower()
                    if u != v and len(u.split()) == 1 and len(v.split()) == 1:
                        pairs.append((u, v, "meronymy", 0))
                    if len(pairs) >= max_pairs:
                        if log: log(f"  meronymy: {len(pairs):,} pairs")
                        return pairs
    if log: log(f"  meronymy: {len(pairs):,} pairs")
    return pairs

def extract_sibling(max_pairs=1500, log=None):
    """Sibling: co-hyponyms of the same hypernym — cat ↔ dog  [symmetric]"""
    if log: log("Extracting sibling pairs from WordNet...")
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
            for j in range(i + 1, len(words)):
                u, v = words[i], words[j]
                key = (min(u, v), max(u, v))
                if key not in seen and u != v:
                    seen.add(key)
                    pairs.append((u, v, "sibling", 1))
                if len(pairs) >= max_pairs:
                    if log: log(f"  sibling: {len(pairs):,} pairs")
                    return pairs
    if log: log(f"  sibling: {len(pairs):,} pairs")
    return pairs

def extract_coordinate(max_pairs=1500, log=None):
    """Coordinate: thematic peers — fruit, vehicle, sport, …  [symmetric]"""
    if log: log("Extracting coordinate pairs from WordNet...")
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
            for j in range(i + 1, len(words)):
                u, v = words[i], words[j]
                key = (min(u, v), max(u, v))
                if key not in seen and u != v:
                    seen.add(key)
                    pairs.append((u, v, "coordinate", 1))
                if len(pairs) >= max_pairs:
                    if log: log(f"  coordinate: {len(pairs):,} pairs")
                    return pairs
    if log: log(f"  coordinate: {len(pairs):,} pairs")
    return pairs

def extract_capital_country(csv_path=None, log=None):
    """Capital-Country: built-in list or user CSV  [asymmetric]"""
    if log: log("Extracting capital-country pairs...")
    if csv_path and os.path.isfile(csv_path):
        pairs = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2:
                    c = row[0].strip().lower()
                    k = row[1].strip().lower()
                    if c and k and c != k:
                        pairs.append((c, k, "capital", 0))
        if log: log(f"  capital (from CSV): {len(pairs):,} pairs")
        return pairs
    # built-in
    pairs = [(c, k, "capital", 0) for c, k in CAPITAL_COUNTRY]
    if log: log(f"  capital (built-in): {len(pairs):,} pairs")
    return pairs

# ===========================================================================
# POST-PROCESSING
# ===========================================================================

def remove_duplicates(all_pairs, log=None):
    """
    Hapus duplikat berdasarkan (word1, word2, relation).
    Duplikat = pasangan identik muncul lebih dari satu kali.
    Return (unique_pairs, duplicate_log)
    """
    seen    = set()
    unique  = []
    dupes   = []
    for row in all_pairs:
        w1, w2, rel, sym = row
        key = (w1, w2, rel)
        if key in seen:
            dupes.append(row)
        else:
            seen.add(key)
            unique.append(row)
    if log:
        log(f"Duplicate removal: {len(dupes):,} duplicates removed → {len(unique):,} unique pairs")
    return unique, dupes

def filter_oov(pairs, vocab_set, log=None):
    """
    Buang pasangan di mana salah satu kata tidak ada di vocab.
    Return (in_vocab_pairs, oov_pairs)
    """
    in_vocab = []
    oov      = []
    for row in pairs:
        w1, w2, rel, sym = row
        if w1 in vocab_set and w2 in vocab_set:
            in_vocab.append(row)
        else:
            oov.append(row)
    if log:
        log(f"OOV filtering: {len(oov):,} pairs removed → {len(in_vocab):,} kept")
    return in_vocab, oov

def balance_dataset(pairs, target_per_relation=None, seed=42, log=None):
    """
    Subsample setiap relasi ke target_per_relation (default = min count).
    Return balanced list.
    """
    by_rel = defaultdict(list)
    for row in pairs:
        by_rel[row[2]].append(row)

    counts = {r: len(v) for r, v in by_rel.items()}
    if target_per_relation is None:
        target_per_relation = min(counts.values())

    rng = random.Random(seed)
    balanced = []
    for rel, rows in by_rel.items():
        n = min(len(rows), target_per_relation)
        balanced.extend(rng.sample(rows, n))

    if log:
        log(f"Balancing: target={target_per_relation:,} per relation")
        for rel, rows in by_rel.items():
            n = min(len(rows), target_per_relation)
            log(f"  {rel:<20} {len(rows):>5} → {n:>5}")
    return balanced, target_per_relation

def validate_dataset(pairs, log=None):
    """Cek integritas: no empty words, no self-pairs, valid relations."""
    VALID_REL = {"hyponymy", "meronymy", "capital", "sibling", "coordinate"}
    errors = []
    for i, (w1, w2, rel, sym) in enumerate(pairs):
        if not w1 or not w2:
            errors.append(f"Row {i}: empty word")
        if w1 == w2:
            errors.append(f"Row {i}: self-pair '{w1}'")
        if rel not in VALID_REL:
            errors.append(f"Row {i}: unknown relation '{rel}'")
        if sym not in (0, 1):
            errors.append(f"Row {i}: invalid is_symmetric value '{sym}'")
    if log:
        if errors:
            log(f"Validation: {len(errors)} issue(s) found")
            for e in errors[:10]:
                log(f"  {e}")
        else:
            log(f"Validation: ✔ all {len(pairs):,} rows OK")
    return errors

# ===========================================================================
# I/O HELPERS
# ===========================================================================

def load_vocab(path, log=None):
    """Load embedding vocab dari .txt (satu kata per baris, atau format FastText)."""
    vocab = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            w = line.split()[0].lower()
            vocab.add(w)
    if log: log(f"Vocab loaded: {len(vocab):,} words from {os.path.basename(path)}")
    return vocab

def save_dataset_csv(pairs, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["word1", "word2", "relation", "is_symmetric"])
        writer.writerows(pairs)

def save_oov_csv(oov_pairs, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["word1", "word2", "relation", "is_symmetric", "reason"])
        for row in oov_pairs:
            writer.writerow(list(row) + ["OOV"])

def save_duplicate_csv(dupes, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["word1", "word2", "relation", "is_symmetric"])
        writer.writerows(dupes)

def save_metadata_json(stats_list, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stats_list, f, indent=2, ensure_ascii=False)

def save_balance_summary_json(pairs, target, path):
    counter = Counter(r[2] for r in pairs)
    summary = {
        "target_per_relation": target,
        "total_pairs": len(pairs),
        "distribution": dict(counter),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

# ===========================================================================
# MAIN PIPELINE
# ===========================================================================

def run_pipeline(cfg, log):
    """
    Jalankan full pipeline.
    cfg dict:
      max_hypo, max_mero, max_sib, max_coord  — int batas pasang per relasi
      capital_csv      — path CSV kapital/negara (str | None)
      vocab_path       — path vocab .txt (str | None)
      do_balance       — bool
      target_per_rel   — int | None (None = min count)
      out_dir          — output directory
    """
    out  = cfg["out_dir"]
    os.makedirs(out, exist_ok=True)
    log("=" * 60)
    log("APP 1 — Dataset Builder")
    log("=" * 60)

    # ── 1. Extraction ────────────────────────────────────────────
    log("\n[1] Extraction")
    all_pairs = []
    all_pairs.extend(extract_hyponymy(cfg["max_hypo"], log))
    all_pairs.extend(extract_meronymy(cfg["max_mero"], log))
    all_pairs.extend(extract_sibling(cfg["max_sib"], log))
    all_pairs.extend(extract_coordinate(cfg["max_coord"], log))
    all_pairs.extend(extract_capital_country(cfg.get("capital_csv"), log))
    log(f"  Total raw pairs: {len(all_pairs):,}")

    # ── 2. Duplicate removal ─────────────────────────────────────
    log("\n[2] Duplicate Removal")
    unique_pairs, dupes = remove_duplicates(all_pairs, log)

    # ── 3. OOV filtering ─────────────────────────────────────────
    oov_pairs = []
    if cfg.get("vocab_path"):
        log("\n[3] OOV Filtering")
        try:
            vocab = load_vocab(cfg["vocab_path"], log)
            unique_pairs, oov_pairs = filter_oov(unique_pairs, vocab, log)
        except Exception as e:
            log(f"  [WARN] vocab load failed: {e} — skipping OOV filter")
    else:
        log("\n[3] OOV Filtering — SKIPPED (no vocab file)")

    # ── 4. Balancing ─────────────────────────────────────────────
    target_used = None
    if cfg["do_balance"]:
        log("\n[4] Balancing")
        unique_pairs, target_used = balance_dataset(
            unique_pairs,
            target_per_relation=cfg.get("target_per_rel") or None,
            log=log,
        )
    else:
        log("\n[4] Balancing — SKIPPED")

    # ── 5. Validation ────────────────────────────────────────────
    log("\n[5] Validation")
    errors = validate_dataset(unique_pairs, log)

    # ── 6. Save outputs ──────────────────────────────────────────
    log("\n[6] Saving outputs")

    # dataset.csv
    ds_path = os.path.join(out, "dataset.csv")
    save_dataset_csv(unique_pairs, ds_path)
    log(f"  dataset.csv      → {ds_path}  ({len(unique_pairs):,} rows)")

    # oov_report.csv
    oov_path = os.path.join(out, "oov_report.csv")
    save_oov_csv(oov_pairs, oov_path)
    log(f"  oov_report.csv   → {oov_path}  ({len(oov_pairs):,} rows)")

    # duplicate_log.csv
    dup_path = os.path.join(out, "duplicate_log.csv")
    save_duplicate_csv(dupes, dup_path)
    log(f"  duplicate_log.csv → {dup_path}  ({len(dupes):,} rows)")

    # metadata.json
    by_rel = defaultdict(list)
    for row in unique_pairs:
        by_rel[row[2]].append(row)

    # Count raw extracted & oov removed per relation
    raw_counts = defaultdict(int)
    for row in all_pairs:
        raw_counts[row[2]] += 1
    oov_counts = defaultdict(int)
    for row in oov_pairs:
        oov_counts[row[2]] += 1
    dup_counts = defaultdict(int)
    for row in dupes:
        dup_counts[row[2]] += 1

    meta = []
    for rel, rows in by_rel.items():
        meta.append({
            "relation":           rel,
            "n_pairs":            len(rows),
            "n_raw":              raw_counts[rel],
            "oov_removed":        oov_counts[rel],
            "duplicates_removed": dup_counts[rel],
            "balanced":           cfg["do_balance"],
        })
    meta_path = os.path.join(out, "metadata.json")
    save_metadata_json(meta, meta_path)
    log(f"  metadata.json    → {meta_path}")

    # balance_summary.json
    bal_path = os.path.join(out, "balance_summary.json")
    save_balance_summary_json(unique_pairs, target_used or 0, bal_path)
    log(f"  balance_summary.json → {bal_path}")

    log("\n" + "=" * 60)
    log(f"DONE  —  {len(unique_pairs):,} pairs  |  {len(errors)} validation issue(s)")
    log("=" * 60)
    return unique_pairs, errors

# ===========================================================================
# GUI
# ===========================================================================

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("APP 1 — Dataset Builder")
        self.geometry("780x680")
        self.resizable(True, True)
        self._build_ui()

    # ── UI ──────────────────────────────────────────────────────
    def _build_ui(self):
        pad = dict(padx=8, pady=4)

        # ── Config frame ────────────────────────────────────────
        cf = ttk.LabelFrame(self, text="Configuration", padding=8)
        cf.pack(fill="x", padx=10, pady=8)

        # Row limits
        limits_f = ttk.Frame(cf)
        limits_f.pack(fill="x", pady=2)
        for col, (label, attr, default) in enumerate([
            ("Max Hyponymy",  "max_hypo",  "2000"),
            ("Max Meronymy",  "max_mero",  "2000"),
            ("Max Sibling",   "max_sib",   "1500"),
            ("Max Coordinate","max_coord", "1500"),
        ]):
            ttk.Label(limits_f, text=label+":").grid(row=0, column=col*2,   sticky="e", padx=4)
            var = tk.StringVar(value=default)
            setattr(self, attr, var)
            ttk.Entry(limits_f, textvariable=var, width=7).grid(row=0, column=col*2+1, padx=4)

        # Capital CSV
        cap_f = ttk.Frame(cf)
        cap_f.pack(fill="x", pady=2)
        ttk.Label(cap_f, text="Capital CSV (opsional):").pack(side="left")
        self.capital_csv = tk.StringVar()
        ttk.Entry(cap_f, textvariable=self.capital_csv, width=40).pack(side="left", padx=4)
        ttk.Button(cap_f, text="Browse…", command=self._browse_cap).pack(side="left")

        # Vocab file
        voc_f = ttk.Frame(cf)
        voc_f.pack(fill="x", pady=2)
        ttk.Label(voc_f, text="Vocab .txt (OOV filter):").pack(side="left")
        self.vocab_path = tk.StringVar()
        ttk.Entry(voc_f, textvariable=self.vocab_path, width=40).pack(side="left", padx=4)
        ttk.Button(voc_f, text="Browse…", command=self._browse_vocab).pack(side="left")

        # Balance options
        bal_f = ttk.Frame(cf)
        bal_f.pack(fill="x", pady=2)
        self.do_balance = tk.BooleanVar(value=True)
        ttk.Checkbutton(bal_f, text="Balance dataset", variable=self.do_balance,
                        command=self._toggle_target).pack(side="left")
        ttk.Label(bal_f, text="  Target per relation:").pack(side="left")
        self.target_per_rel = tk.StringVar(value="")
        self.target_entry = ttk.Entry(bal_f, textvariable=self.target_per_rel, width=8)
        self.target_entry.pack(side="left", padx=4)
        ttk.Label(bal_f, text="(blank = auto min)").pack(side="left")

        # Output dir
        out_f = ttk.Frame(cf)
        out_f.pack(fill="x", pady=2)
        ttk.Label(out_f, text="Output directory:").pack(side="left")
        self.out_dir = tk.StringVar(value=os.path.join(os.path.expanduser("~"), "dataset_builder_output"))
        ttk.Entry(out_f, textvariable=self.out_dir, width=44).pack(side="left", padx=4)
        ttk.Button(out_f, text="Browse…", command=self._browse_out).pack(side="left")

        # ── Buttons ─────────────────────────────────────────────
        btn_f = ttk.Frame(self)
        btn_f.pack(fill="x", padx=10, pady=4)
        self.run_btn  = ttk.Button(btn_f, text="▶  Build Dataset", command=self._start)
        self.run_btn.pack(side="left", padx=4)
        ttk.Button(btn_f, text="Clear Log", command=self._clear_log).pack(side="left", padx=4)
        self.pbar = ttk.Progressbar(btn_f, mode="indeterminate", length=200)
        self.pbar.pack(side="right", padx=8)

        # ── Stats frame ─────────────────────────────────────────
        st_f = ttk.LabelFrame(self, text="Last Run — Relation Statistics", padding=6)
        st_f.pack(fill="x", padx=10, pady=4)
        cols = ("Relation", "Raw", "OOV Removed", "Dup Removed", "Final", "Symmetric")
        self.tree = ttk.Treeview(st_f, columns=cols, show="headings", height=6)
        for c in cols:
            self.tree.heading(c, text=c)
            self.tree.column(c, width=100, anchor="center")
        self.tree.pack(fill="x")

        # ── Log ─────────────────────────────────────────────────
        log_f = ttk.LabelFrame(self, text="Log", padding=4)
        log_f.pack(fill="both", expand=True, padx=10, pady=4)
        self.log_box = scrolledtext.ScrolledText(log_f, height=12, state="disabled",
                                                 font=("Courier", 9))
        self.log_box.pack(fill="both", expand=True)

    # ── Helpers ─────────────────────────────────────────────────
    def _browse_cap(self):
        p = filedialog.askopenfilename(filetypes=[("CSV", "*.csv"), ("All", "*.*")])
        if p: self.capital_csv.set(p)

    def _browse_vocab(self):
        p = filedialog.askopenfilename(filetypes=[("Text", "*.txt"), ("All", "*.*")])
        if p: self.vocab_path.set(p)

    def _browse_out(self):
        p = filedialog.askdirectory()
        if p: self.out_dir.set(p)

    def _toggle_target(self):
        self.target_entry.config(state="normal" if self.do_balance.get() else "disabled")

    def _clear_log(self):
        self.log_box.config(state="normal")
        self.log_box.delete("1.0", "end")
        self.log_box.config(state="disabled")

    def _log(self, msg):
        self.log_box.config(state="normal")
        self.log_box.insert("end", msg + "\n")
        self.log_box.see("end")
        self.log_box.config(state="disabled")
        self.update_idletasks()

    # ── Run ─────────────────────────────────────────────────────
    def _start(self):
        self.run_btn.config(state="disabled")
        self.pbar.start(12)
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        try:
            target_str = self.target_per_rel.get().strip()
            target_val = int(target_str) if target_str.isdigit() else None

            cfg = {
                "max_hypo":     int(self.max_hypo.get()  or 2000),
                "max_mero":     int(self.max_mero.get()  or 2000),
                "max_sib":      int(self.max_sib.get()   or 1500),
                "max_coord":    int(self.max_coord.get() or 1500),
                "capital_csv":  self.capital_csv.get().strip() or None,
                "vocab_path":   self.vocab_path.get().strip()  or None,
                "do_balance":   self.do_balance.get(),
                "target_per_rel": target_val,
                "out_dir":      self.out_dir.get().strip(),
            }

            pairs, errors = run_pipeline(cfg, self._log)
            self.after(0, lambda: self._refresh_table(pairs, cfg, errors))

        except Exception as ex:
            import traceback
            self.after(0, lambda: self._log(f"\n[ERROR] {ex}\n{traceback.format_exc()}"))
            self.after(0, lambda: messagebox.showerror("Error", str(ex)))
        finally:
            self.after(0, self._done)

    def _done(self):
        self.pbar.stop()
        self.run_btn.config(state="normal")

    def _refresh_table(self, pairs, cfg, errors):
        """Isi treeview dengan statistik per relasi."""
        self.tree.delete(*self.tree.get_children())

        # Recompute counts
        raw_all = []
        raw_all += extract_hyponymy.__wrapped__(cfg["max_hypo"]) if hasattr(extract_hyponymy, "__wrapped__") else []
        # Simple recount from pairs (already processed)
        from collections import Counter
        rel_count  = Counter(r[2] for r in pairs)
        sym_count  = Counter(r[2] for r in pairs if r[3] == 1)

        for rel, n in sorted(rel_count.items()):
            self.tree.insert("", "end", values=(
                rel,
                "—",   # raw (would need re-extraction; skip for speed)
                "—",
                "—",
                f"{n:,}",
                "Yes" if sym_count[rel] > 0 else "No",
            ))

        if errors:
            messagebox.showwarning(
                "Validation Issues",
                f"{len(errors)} issue(s) found.\nCheck log for details.",
            )
        else:
            messagebox.showinfo("Done", f"Dataset built: {len(pairs):,} pairs\nSaved to: {cfg['out_dir']}")


# ===========================================================================
if __name__ == "__main__":
    App().mainloop()
