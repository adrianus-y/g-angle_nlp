import tkinter as tk
from tkinter import ttk, messagebox
import threading
import nltk
import random
from nltk.corpus import wordnet as wn

# =========================
# SETUP
# =========================
nltk.download('wordnet')
random.seed(42)


# =========================
# CORE FUNCTIONS (SAMA)
# =========================
def get_hypernym_pairs(max_pairs=5000):
    pairs = []

    for syn in wn.all_synsets('n'):
        for hypo in syn.hyponyms():
            for lemma1 in hypo.lemmas():
                for lemma2 in syn.lemmas():
                    u = lemma1.name().lower()
                    v = lemma2.name().lower()

                    if u != v:
                        pairs.append((u, v))

                    if len(pairs) >= max_pairs:
                        return pairs
    return pairs


def build_direction_dataset(pairs):
    dataset = []
    vocab = list(set([w for p in pairs for w in p]))

    for u, v in pairs:
        dataset.append((u, v, 1))
        dataset.append((v, u, 0))
        dataset.append((u, random.choice(vocab), 0))

    return dataset


def build_ranking_dataset(max_samples=3000):
    dataset = []

    for syn in wn.all_synsets('n'):
        hyper = syn.hypernyms()
        if not hyper:
            continue

        for h in hyper:
            hyper2 = h.hypernyms()
            if not hyper2:
                continue

            for lemma_u in syn.lemmas():
                for lemma_v1 in h.lemmas():
                    for lemma_v2 in hyper2[0].lemmas():
                        u = lemma_u.name().lower()
                        v1 = lemma_v1.name().lower()
                        v2 = lemma_v2.name().lower()

                        if u != v1 and v1 != v2:
                            dataset.append((u, v1, v2))

                        if len(dataset) >= max_samples:
                            return dataset
    return dataset


def build_asymmetry_dataset(pairs):
    return pairs


def save_direction(data, filename):
    with open(filename, "w") as f:
        for u, v, label in data:
            f.write(f"{u},{v},{label}\n")


def save_ranking(data, filename):
    with open(filename, "w") as f:
        for u, v1, v2 in data:
            f.write(f"{u},{v1},{v2}\n")


def save_asymmetry(data, filename):
    with open(filename, "w") as f:
        for u, v in data:
            f.write(f"{u},{v}\n")


# =========================
# GUI
# =========================
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("WordNet Dataset Generator (g-angle)")

        # Input
        ttk.Label(root, text="Max Hypernym Pairs:").pack(pady=5)
        self.max_pairs_entry = ttk.Entry(root)
        self.max_pairs_entry.insert(0, "3000")
        self.max_pairs_entry.pack()

        ttk.Label(root, text="Ranking Samples:").pack(pady=5)
        self.rank_entry = ttk.Entry(root)
        self.rank_entry.insert(0, "2000")
        self.rank_entry.pack()

        # Button
        self.generate_btn = ttk.Button(root, text="Generate Dataset", command=self.start_generation)
        self.generate_btn.pack(pady=10)

        # Log box
        self.log = tk.Text(root, height=15, width=60)
        self.log.pack(pady=10)

    def log_msg(self, msg):
        self.log.insert(tk.END, msg + "\n")
        self.log.see(tk.END)

    def start_generation(self):
        thread = threading.Thread(target=self.generate)
        thread.start()

    def generate(self):
        try:
            max_pairs = int(self.max_pairs_entry.get())
            rank_samples = int(self.rank_entry.get())

            self.log_msg("Mengambil hypernym pairs...")
            pairs = get_hypernym_pairs(max_pairs)
            self.log_msg(f"Pairs: {len(pairs)}")

            self.log_msg("Membuat direction dataset...")
            direction = build_direction_dataset(pairs)
            save_direction(direction, "direction.txt")
            self.log_msg(f"Saved direction.txt ({len(direction)})")

            self.log_msg("Membuat ranking dataset...")
            ranking = build_ranking_dataset(rank_samples)
            save_ranking(ranking, "ranking.txt")
            self.log_msg(f"Saved ranking.txt ({len(ranking)})")

            self.log_msg("Membuat asymmetry dataset...")
            asym = build_asymmetry_dataset(pairs)
            save_asymmetry(asym, "asymmetry.txt")
            self.log_msg(f"Saved asymmetry.txt ({len(asym)})")

            self.log_msg("SELESAI ✔")

        except Exception as e:
            messagebox.showerror("Error", str(e))


# =========================
# RUN
# =========================
if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()