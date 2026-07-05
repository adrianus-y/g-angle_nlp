import numpy as np


class GeometryEngine:
    """
    SCA v5.7 Engine: Pure l^p Geometry Edition.
    Konsisten dengan konvensi Gunawan et al. (2018):

        A_g(x, y) := arccos( g(y, x) / (||x|| * ||y||) )

    Artinya cos A_g(u, v) = g(v, u) / (||u|| * ||v||)
    bukan g(u, v) / (||u|| * ||v||).

    Perubahan dari v5.6:
        g_similarity(u, v) sekarang menggunakan functional_g(v, u)
        bukan functional_g(u, v).
    """

    def __init__(self, eps=1e-15):
        # eps hanya untuk pengecekan zero-division,
        # tidak memodifikasi struktur formula p-norm.
        self.eps = eps

    def functional_g(self, x, y, p=2.0):
        """
        Fungsional g(x, y) murni pada ruang l^p.

        Definisi:
            g(x, y) = ||x||_p^{2-p} * sum_k( |x_k|^{p-1} * sgn(x_k) * y_k )

        Catatan:
            Untuk x_k = 0, suku kontribusinya = 0 (ditangani via mask).
            Tidak ada regularisasi (+ eps) di dalam |x_k|.
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        norm_x = np.linalg.norm(x, ord=p)

        if norm_x < self.eps:
            return 0.0

        abs_x = np.abs(x)
        mask = abs_x > 0
        term_x = np.zeros_like(x)
        term_x[mask] = (abs_x[mask] ** (p - 1)) * np.sign(x[mask])

        inner_sum = np.sum(term_x * y)

        return (norm_x ** (2 - p)) * inner_sum

    def g_similarity(self, u, v, p=2.0):
        """
        Kosinus g-angle dari u ke v, sesuai konvensi Gunawan:

            cos A_g(u, v) = g(v, u) / (||u||_p * ||v||_p)

        Interpretasi:
            Seberapa besar komponen v yang "sejalan" dengan arah ruang u.
        """
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)

        norm_u = np.linalg.norm(u, ord=p)
        norm_v = np.linalg.norm(v, ord=p)

        if norm_u < self.eps or norm_v < self.eps:
            return 0.0

        # Konvensi Gunawan: g(v, u) — argumen dibalik dari urutan sudut
        g_vu = self.functional_g(v, u, p)

        cos_val = g_vu / (norm_u * norm_v)
        return np.clip(cos_val, -1.0, 1.0)

    def calculate_g_metrics(self, u, v, p=2.0):
        """
        Main API untuk GUI SCA.

        Input:
            u = semantic anchor (vektor subtitle)
            v = target (vektor komentar)

        Output:
            dist_asym  : 1 - cos A_g(u, v)  — jarak direktional u→v
            dist_sym   : 1 - 0.5*(cos A_g(u,v) + cos A_g(v,u))  — jarak simetris
            asym_score : cos A_g(u,v) - cos A_g(v,u)  — skor asimetri bertanda

        Interpretasi asym_score:
            > 0 : A_g(u,v) < A_g(v,u) → v lebih "terkandung" dalam arah u
                  (komentar secara semantik lebih dekat ke anchor dari sisi anchor)
            < 0 : A_g(u,v) > A_g(v,u) → u lebih "terkandung" dalam arah v
                  (anchor lebih dekat ke komentar dari sisi komentar)
            = 0 : relasi simetris
        """
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)

        # cos A_g(u, v) = g(v, u) / (||u|| ||v||)
        cos_uv = self.g_similarity(u, v, p)

        # cos A_g(v, u) = g(u, v) / (||u|| ||v||)
        cos_vu = self.g_similarity(v, u, p)

        dist_asym = 1.0 - cos_uv
        dist_sym  = 1.0 - 0.5 * (cos_uv + cos_vu)
        asym_score = cos_uv - cos_vu

        return dist_asym, dist_sym, asym_score

    def g_angle(self, u, v, p=2.0):
        """
        Menghitung A_g(u, v) dalam radian.
        Wrapper eksplisit untuk keperluan scatter plot asimetri.
        """
        cos_val = self.g_similarity(u, v, p)
        return np.arccos(cos_val)

    def cosine_similarity(self, u, v):
        """
        Baseline Euclidean (p=2) tanpa g-angle.
        Digunakan sebagai pembanding simetris.
        """
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        norm_u = np.linalg.norm(u)
        norm_v = np.linalg.norm(v)
        if norm_u < self.eps or norm_v < self.eps:
            return 0.0
        return float(np.dot(u, v) / (norm_u * norm_v))


# ------------------------------------------------------------------------------
# UNIT TEST SEDERHANA
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    engine = GeometryEngine()
    np.random.seed(0)

    u = np.random.randn(300)
    v = np.random.randn(300)
    p = 3.0

    cos_uv = engine.g_similarity(u, v, p)
    cos_vu = engine.g_similarity(v, u, p)
    a_uv   = engine.g_angle(u, v, p)
    a_vu   = engine.g_angle(v, u, p)
    _, _, delta = engine.calculate_g_metrics(u, v, p)

    print(f"cos A_g(u,v) = {cos_uv:.6f}  →  A_g(u,v) = {np.degrees(a_uv):.4f}°")
    print(f"cos A_g(v,u) = {cos_vu:.6f}  →  A_g(v,u) = {np.degrees(a_vu):.4f}°")
    print(f"Δ (asym_score) = {delta:.6f}")
    print(f"Asimetri terdeteksi: {abs(delta) > 1e-6}")

    # Verifikasi: pada p=2, g-angle harus mendekati cosine similarity
    cos_standard = engine.cosine_similarity(u, v)
    cos_g_p2     = engine.g_similarity(u, v, p=2.0)
    print(f"\nVerifikasi p=2:")
    print(f"  cosine_similarity  = {cos_standard:.6f}")
    print(f"  g_similarity (p=2) = {cos_g_p2:.6f}")
    print(f"  Selisih            = {abs(cos_standard - cos_g_p2):.2e}")
