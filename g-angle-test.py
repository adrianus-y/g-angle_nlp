import tkinter as tk
from tkinter import filedialog, ttk
import threading
import numpy as np

# =========================
# GEOMETRY ENGINE
# =========================
class GeometryEngine:
    def __init__(self, eps=1e-15):
        self.eps = eps

    def functional_g(self, x, y, p=3.0):
        norm_x = np.linalg.norm(x, ord=p)
        if norm_x < self.eps:
            return 0.0

        abs_x = np.abs(x)
        mask = abs_x > 0
        term_x = np.zeros_like(x)
        term_x[mask] = (abs_x[mask] ** (p - 1)) * np.sign(x[mask])

        inner_sum = np.sum(term_x * y)
        return (norm_x ** (2 - p)) * inner_sum

    def g_similarity(self, u, v, p=3.0):
        norm_u = np.linalg.norm(u, ord=p)
        norm_v = np.linalg.norm(v, ord=p)

        if norm_u < self.eps or norm_v < self.eps:
            return 0.0

        g_vu = self.functional_g(v, u, p)
        return np.clip(g_vu / (norm_u * norm_v), -1.0, 1.0)


engine = GeometryEngine()


# =========================
# LOAD FASTTEXT (FIXED)
# =========================
def load_fasttext(path, log):
    log("Loading FastText model... (ini bisa lama)")

    from gensim.models.fasttext import load_facebook_model
    from gensim.models import KeyedVectors

    try:
        if path.endswith(".bin"):
            model = load_facebook_model(path)
            log("Loaded as FastText Facebook model ✔")
            return model.wv

        elif path.endswith(".vec"):
            model = KeyedVectors.load_word2vec_format(path, binary=False)
            log("Loaded as .vec ✔")
            return model

        else:
            raise ValueError("Format tidak dikenali (.bin / .vec saja)")

    except Exception as e:
        log(f"ERROR loading model: {e}")
        return None


# =========================
# LOAD DATASET
# =========================
def load_direction(file):
    data = []
    with open(file) as f:
        for line in f:
            u, v, label = line.strip().split(",")
            data.append((u, v, int(label)))
    return data


def load_ranking(file):
    data = []
    with open(file) as f:
        for line in f:
            u, v1, v2 = line.strip().split(",")
            data.append((u, v1, v2))
    return data


def load_asym(file):
    data = []
    with open(file) as f:
        for line in f:
            u, v = line.strip().split(",")
            data.append((u, v))
    return data


# =========================
# EVALUATION
# =========================
def eval_direction(data, model):
    correct, total, oov = 0, 0, 0

    for u, v, label in data:
        if u not in model or v not in model:
            oov += 1
            continue

        vec_u = model[u]
        vec_v = model[v]

        score = engine.g_similarity(vec_u, vec_v) - \
                engine.g_similarity(vec_v, vec_u)

        pred = 1 if score > 0 else 0

        if pred == label:
            correct += 1
        total += 1

    acc = correct / total if total > 0 else 0
    return acc, oov


def eval_ranking(data, model):
    correct, total, oov = 0, 0, 0

    for u, v1, v2 in data:
        if u not in model or v1 not in model or v2 not in model:
            oov += 1
            continue

        s1 = engine.g_similarity(model[u], model[v1])
        s2 = engine.g_similarity(model[u], model[v2])

        if s1 > s2:
            correct += 1
        total += 1

    acc = correct / total if total > 0 else 0
    return acc, oov


def eval_asymmetry(data, model):
    deltas = []
    oov = 0

    for u, v in data:
        if u not in model or v not in model:
            oov += 1
            continue

        d = engine.g_similarity(model[u], model[v]) - \
            engine.g_similarity(model[v], model[u])

        deltas.append(abs(d))

    mean_delta = np.mean(deltas) if deltas else 0
    return mean_delta, oov


# =========================
# GUI
# =========================
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("g-Angle Evaluator (FastText Fixed)")

        self.model = None
        self.dir_data = None
        self.rank_data = None
        self.asym_data = None

        ttk.Button(root, text="Load FastText (.bin/.vec)", command=self.load_model).pack(pady=5)
        ttk.Button(root, text="Load Direction", command=self.load_dir).pack(pady=5)
        ttk.Button(root, text="Load Ranking", command=self.load_rank).pack(pady=5)
        ttk.Button(root, text="Load Asymmetry", command=self.load_asym).pack(pady=5)

        ttk.Button(root, text="Evaluate", command=self.run_eval).pack(pady=10)

        self.output = tk.Text(root, height=18, width=70)
        self.output.pack()

    def log(self, msg):
        self.output.insert(tk.END, msg + "\n")
        self.output.see(tk.END)

    def load_model(self):
        path = filedialog.askopenfilename()
        if not path:
            return

        def task():
            self.model = load_fasttext(path, self.log)
            if self.model:
                self.log("Model siap digunakan ✔\n")

        threading.Thread(target=task).start()

    def load_dir(self):
        path = filedialog.askopenfilename()
        self.dir_data = load_direction(path)
        self.log(f"Direction loaded ({len(self.dir_data)})")

    def load_rank(self):
        path = filedialog.askopenfilename()
        self.rank_data = load_ranking(path)
        self.log(f"Ranking loaded ({len(self.rank_data)})")

    def load_asym(self):
        path = filedialog.askopenfilename()
        self.asym_data = load_asym(path)
        self.log(f"Asymmetry loaded ({len(self.asym_data)})")

    def run_eval(self):
        threading.Thread(target=self.evaluate).start()

    def evaluate(self):
        if self.model is None:
            self.log("❌ Load model dulu!")
            return

        self.log("\n=== MULAI EVALUASI ===")

        if self.dir_data:
            acc, oov = eval_direction(self.dir_data, self.model)
            self.log(f"Direction Accuracy: {acc:.4f} | OOV: {oov}")

        if self.rank_data:
            acc, oov = eval_ranking(self.rank_data, self.model)
            self.log(f"Ranking Accuracy: {acc:.4f} | OOV: {oov}")

        if self.asym_data:
            val, oov = eval_asymmetry(self.asym_data, self.model)
            self.log(f"Mean Asymmetry |Δ|: {val:.6f} | OOV: {oov}")

        self.log("=== SELESAI ✔ ===\n")


# =========================
# RUN
# =========================
if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()