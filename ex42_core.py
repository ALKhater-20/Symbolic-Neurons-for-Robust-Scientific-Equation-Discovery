"""ex42_core.py -- shared engine for the pre-specified experiments (A1, E4-E12).

Builds on ex41.py (Study C code): same library, same targets, same data generator, same BIC.
Differences are documented in the paper (Appendix: implementation notes):
  * best-subset search uses Gram-matrix batched solves (same optimum as ex41.sn_exh up to
    numerical tolerance on near-singular subsets) so that K_max up to 5 is tractable;
  * new selectors: stepwise (forward-backward), ridge-STLSQ, stability-selection LASSO.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import itertools, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
import ex41
from ex41 import LIB, T

# ---------------------------------------------------------------- numerics
def bic_val(rss, n, k):
    return n * np.log(rss / n + 1e-10) + k * np.log(n)

class Gram:
    """Centered/standardised design Z (n x p) and response y; intercept is implicit (centering)."""
    def __init__(self, Z, y):
        self.Z, self.y = Z, y
        self.n, self.p = Z.shape
        self.G = Z.T @ Z
        self.b = Z.T @ y
        self.yy = float(y @ y)

    def rss_batch(self, idx):                       # idx: (C,k) int array
        G, b = self.G, self.b
        Gs = G[idx[:, :, None], idx[:, None, :]] + 1e-13 * self.n * np.eye(idx.shape[1])
        bs = b[idx]
        try:
            sol = np.linalg.solve(Gs, bs[..., None])[..., 0]
        except np.linalg.LinAlgError:
            sol = np.stack([np.linalg.lstsq(g, v, rcond=None)[0] for g, v in zip(Gs, bs)])
        return np.maximum(self.yy - (bs * sol).sum(1), 1e-12 * self.yy)

    def coef(self, idx):
        """intercept + coefficients for a subset (least squares)."""
        idx = list(idx)
        if not idx:
            return np.array([float(self.y.mean())])
        A = np.column_stack([np.ones(self.n)] + [self.Z[:, j] for j in idx])
        return np.linalg.lstsq(A, self.y, rcond=None)[0]

def best_subset(gr, kmax, allowed=None, keep_all=False, chunk=150000):
    """Exact best-subset by BIC over all subsets of at most kmax allowed columns."""
    n = gr.n
    cols = np.array(list(range(gr.p)) if allowed is None else list(allowed))
    best_b, best_s = bic_val(gr.yy, n, 0), ()
    all_s, all_b = [()], [bic_val(gr.yy, n, 0)]
    cand = 1
    for k in range(1, kmax + 1):
        it = itertools.combinations(range(len(cols)), k)
        while True:
            blk = list(itertools.islice(it, chunk))
            if not blk:
                break
            ii = cols[np.array(blk)]
            b = bic_val(gr.rss_batch(ii), n, k)
            cand += len(blk)
            j = int(np.argmin(b))
            if b[j] < best_b:
                best_b, best_s = float(b[j]), tuple(int(t) for t in ii[j])
            if keep_all:
                all_s.extend(tuple(int(t) for t in r) for r in ii); all_b.extend(b.tolist())
    out = (list(best_s), cand)
    if keep_all:
        out = out + (all_s, np.array(all_b))
    return out

def stepwise(gr, kmax=None, allowed=None, max_iter=60):
    """Forward-backward (add/drop/swap-free) BIC stepwise; kmax=None means uncapped."""
    n = gr.n
    cols = list(range(gr.p)) if allowed is None else list(allowed)
    kmax = kmax or len(cols)
    sel = []; cur = bic_val(gr.yy, n, 0); cand = 1
    for _ in range(max_iter):
        improved = False
        if len(sel) < kmax:                                    # forward
            rest = [j for j in cols if j not in sel]
            if rest:
                ii = np.array([sel + [j] for j in rest]); r = gr.rss_batch(ii); cand += len(rest)
                b = bic_val(r, n, len(sel) + 1); j = int(np.argmin(b))
                if b[j] < cur - 1e-9:
                    sel = sel + [rest[j]]; cur = float(b[j]); improved = True
        if len(sel) >= 2:                                      # backward
            ii = np.array([[t for t in sel if t != s] for s in sel]); r = gr.rss_batch(ii); cand += len(sel)
            b = bic_val(r, n, len(sel) - 1); j = int(np.argmin(b))
            if b[j] < cur - 1e-9:
                sel = [t for t in sel if t != sel[j]]; cur = float(b[j]); improved = True
        if not improved:
            break
    return sel, cand

def posterior(all_s, all_b, p):
    w = np.exp(-0.5 * (all_b - all_b.min())); w /= w.sum()
    pi = np.zeros(p)
    for s, wi in zip(all_s, w):
        for j in s:
            pi[j] += wi
    k = int(np.argmin(all_b))
    ent = float(-(w[w > 0] * np.log(w[w > 0])).sum())
    return pi, float(w[k]), ent

def stlsq_ridge(gr, grid=(.005, .01, .02, .05, .1, .2, .4), lam=1.0, mode="bic", seed=0):
    """STLSQ with a ridge-regularised fit; threshold by BIC (mode='bic') or 5-fold CV (mode='cv')."""
    Z, y, n, p = gr.Z, gr.y, gr.n, gr.p
    def ridge(act, Zs=None, ys=None):
        Zs = Z if Zs is None else Zs; ys = y if ys is None else ys
        A = Zs[:, act]; return np.linalg.solve(A.T @ A + lam * np.eye(len(act)), A.T @ ys)
    supports = []
    for thr in grid:
        act = list(range(p))
        for _ in range(10):
            c = ridge(act); keep = [j for j, cj in zip(act, c) if abs(cj) >= thr]
            if len(keep) == len(act) or not keep:
                act = keep if keep else act; break
            act = keep
        supports.append(act)
    if mode == "bic":
        sc = []
        for act in supports:
            c = ridge(act); r = y - Z[:, act] @ c; sc.append(bic_val(float(r @ r), n, len(act)))
    else:
        rng = np.random.default_rng(seed); fold = rng.permutation(n) % 5; sc = []
        for act in supports:
            e = 0.0
            for f in range(5):
                tr, te = fold != f, fold == f
                c = ridge(act, Z[tr], y[tr]); e += float(((y[te] - Z[te][:, act] @ c) ** 2).sum())
            sc.append(e)
    act = supports[int(np.argmin(sc))]
    return act, len(grid) * 10, ridge(act)

def lasso_stability(gr, seed=0, B=30, thr=0.6, frac=0.5):
    from sklearn.linear_model import LassoCV, Lasso
    m = LassoCV(cv=5, max_iter=50000, random_state=seed).fit(gr.Z, gr.y)
    rng = np.random.default_rng(seed); cnt = np.zeros(gr.p)
    for _ in range(B):
        ii = rng.choice(gr.n, int(frac * gr.n), replace=False)
        l = Lasso(alpha=m.alpha_ * 1.5, max_iter=20000).fit(gr.Z[ii], gr.y[ii]); cnt += (np.abs(l.coef_) > 1e-8)
    sel = [int(j) for j in np.flatnonzero(cnt / B >= thr)]
    return sel, B

# ---------------------------------------------------------------- Study-C style problem
class Prob:
    pass

def make_problem(fid, sigma, seed, N=100, a=1.0, lib=None, vocab=3):
    """Same data generator as ex41.run_config (identical seeds => identical data)."""
    fam, f, coefs, sing = T[fid]
    rng = np.random.default_rng([list(T).index(fid), int(round(sigma * 100)), seed])
    lo, hi = -a, a
    nlo = max(-a - 2, -1.8) if sing else -a - 2
    xtr = rng.uniform(lo, hi, N); clean = f(xtr)
    ytr = clean + sigma * clean.std() * rng.standard_normal(N)
    P = Prob()
    P.fid, P.f, P.coefs, P.fam, P.sigma, P.seed = fid, f, coefs, fam, sigma, seed
    P.xtr, P.ytr, P.clean = xtr, ytr, clean
    P.xi, P.xp, P.xn = np.linspace(lo, hi, 1000), np.linspace(hi, hi + 2, 1000), np.linspace(nlo, lo, 1000)
    with np.errstate(all="ignore"):
        P.fi, P.fp, P.fn = f(P.xi), f(P.xp), f(P.xn)
    allf = np.concatenate([P.fi, P.fp, P.fn])
    P.scale = float(np.sqrt(np.mean(allf ** 2))) + 1e-12; P.fmax = float(np.max(np.abs(allf)))
    if lib is None:
        lib = {n: v[0] for n, v in LIB.items() if v[1] <= vocab}
    P.lib = lib
    return P

def design(P, lib=None):
    lib = P.lib if lib is None else lib
    names = list(lib)
    with np.errstate(all="ignore"):
        Phi = np.column_stack([lib[n](P.xtr) for n in names])
    ok = np.isfinite(Phi).all(0) & (Phi.std(0) > 1e-12)
    names = [n for n, k in zip(names, ok) if k]; Phi = Phi[:, ok]
    D = Prob(); D.names, D.mu, D.sd = names, Phi.mean(0), Phi.std(0) + 1e-12
    D.Z = (Phi - D.mu) / D.sd
    D.ym, D.ys = P.ytr.mean(), P.ytr.std() + 1e-12; D.yz = (P.ytr - D.ym) / D.ys
    D.lib = lib; D.gr = Gram(D.Z, D.yz)
    return D

def evaluate(P, D, idx, c=None):
    """Metrics for a selected support idx (indices into D.names) with LS (or given) coefficients."""
    idx = list(idx)
    if c is None:
        c = D.gr.coef(idx)
    def pred(z):
        out = np.full(len(z), c[0], float)
        with np.errstate(all="ignore"):
            for cj, j in zip(c[1:], idx):
                out += cj * (D.lib[D.names[j]](z) - D.mu[j]) / D.sd[j]
        return D.ys * out + D.ym
    with np.errstate(all="ignore"):
        pi, pp, pn = [np.nan_to_num(pred(z), nan=1e9, posinf=1e9, neginf=-1e9) for z in (P.xi, P.xp, P.xn)]
    rm = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))
    ri, re = rm(pi, P.fi), rm(np.r_[pp, pn], np.r_[P.fp, P.fn])
    got = sorted(D.names[j] for j in idx)
    r = dict(nrmse_int=ri / P.scale, nrmse_ext=re / P.scale, k=len(idx),
             relmax=float(max(np.abs(pi - P.fi).max(), np.abs(pp - P.fp).max(), np.abs(pn - P.fn).max()) / P.fmax))
    r["srr_sem"] = float(r["relmax"] <= 0.05)
    r["explosive"] = float(r["nrmse_ext"] > 100)
    r["covered"] = float(P.coefs is not None and set(P.coefs) <= set(D.names))
    if P.coefs is not None:
        r["srr_struct"] = float(set(got) == set(P.coefs))
        ex = 0.0
        if r["srr_struct"]:
            cm = {D.names[j]: D.ys * cj / D.sd[j] for j, cj in zip(idx, c[1:])}
            ex = float(all(abs(cm[k] - v) / abs(v) < 1e-6 for k, v in P.coefs.items()))
        r["srr_exact"] = ex
    else:
        r["srr_struct"] = r["srr_exact"] = np.nan
    r["terms"] = "+".join(got)
    return r

# ---------------------------------------------------------------- method dispatcher (Study C library)
def run_method(m, D, seed, kmax=3):
    """returns idx, candidates, coefficient-or-None"""
    gr = D.gr
    if m.startswith("SN-Exh"):
        idx, cand = best_subset(gr, kmax); return idx, cand, None
    if m.startswith("SN-Step"):
        idx, cand = stepwise(gr, None if kmax in (None, 0) else kmax); return idx, cand, None
    if m.startswith("SN-Greedy"):
        idx, cand = ex41.sn_greedy(D.Z, D.yz, kmax)[:2]; return idx, cand, None
    if m.startswith("OMP"):
        idx, cand = ex41.omp_bic(D.Z, D.yz, kmax)[:2]; return idx, cand, None
    if m == "STLSQ":
        idx, cand = ex41.stlsq(D.Z, D.yz)[:2]; return idx, cand, None
    if m == "STLSQ-ridge":
        idx, cand, c = stlsq_ridge(gr, mode="bic"); return idx, cand, np.r_[0.0, c]
    if m == "STLSQ-ridge-cv":
        idx, cand, c = stlsq_ridge(gr, mode="cv", seed=seed); return idx, cand, np.r_[0.0, c]
    if m == "LASSO":
        s = ex41.lasso_sel(D.Z, D.yz, seed); return s[0], np.nan, s[3]
    if m == "LASSO-stab":
        idx, cand = lasso_stability(gr, seed); return idx, cand, None
    raise ValueError(m)

# ================================================================ A1 : re-score with three criteria
def exp_A1(seeds=10, noises=(0, .01, .05, .10, .20), out="a1.csv"):
    rows = []
    for fid in T:
        for s in noises:
            for sd in range(seeds):
                P = make_problem(fid, s, sd); D = design(P)
                for m in ("SN-Exh", "SN-Greedy"):
                    idx, cand, c = run_method(m, D, sd, 3)
                    r = evaluate(P, D, idx, c); r.update(exp="A1", function=fid, family=P.fam, noise=s, seed=sd, model=m)
                    rows.append(r)
    return rows

# ================================================================ E4 : sparsity-prior sensitivity
E4_METHODS = [("SN-Exh", 2), ("SN-Exh", 3), ("SN-Exh", 4), ("SN-Exh", 5), ("SN-Exh", 8),
              ("SN-Step", 8), ("SN-Step", 0),
              ("SN-Greedy", 3), ("SN-Greedy", 8), ("SN-Greedy", 0),
              ("OMP", 3), ("OMP", 8),
              ("STLSQ", None), ("STLSQ-ridge", None), ("STLSQ-ridge-cv", None),
              ("LASSO", None), ("LASSO-stab", None)]

def exp_E4(seeds=6, noises=(0.01, .05, .20), out="e4.csv"):
    rows = []
    for fid in T:
        for s in noises:
            for sd in range(seeds):
                P = make_problem(fid, s, sd); D = design(P)
                for m, k in E4_METHODS:
                    if k == 8 and m == "SN-Exh" and sd >= 2:      # exact K=8 costs ~4 s/run: first 2 seeds only
                        continue
                    t0 = time.perf_counter()
                    idx, cand, c = run_method(m, D, sd, k if k is not None else 3)
                    el = time.perf_counter() - t0
                    r = evaluate(P, D, idx, c)
                    r.update(exp="E4", function=fid, family=P.fam, noise=s, seed=sd, model=m,
                             kmax=("inf" if k == 0 else (k if k is not None else "-")), time=el, candidates=cand)
                    rows.append(r)
    return rows

# ================================================================ E5 : library learning
UNARY = {"sin": np.sin, "cos": np.cos, "exp": np.exp, "exp-": lambda u: np.exp(-u),
         "log1p2": lambda u: np.log1p(u ** 2), "inv1p2": lambda u: 1 / (1 + u ** 2)}
def uname(u, t):
    return {"sin": f"sin({t})", "cos": f"cos({t})", "exp": f"exp({t})", "exp-": f"exp(-{t})",
            "log1p2": f"log(1+{t}^2)", "inv1p2": f"1/(1+{t}^2)"}[u]

def grow_library(P, base_lib, rounds=2, top=2, kmax=3):
    """L1: growth by composition of high-inclusion terms with a small unary set; data-driven."""
    lib = dict(base_lib)
    log = []
    for _ in range(rounds):
        D = design(P, lib)
        idx, cand, all_s, all_b = best_subset(D.gr, kmax, keep_all=True)
        pi, mass, ent = posterior(all_s, all_b, len(D.names))
        order = [D.names[j] for j in np.argsort(-pi)[:top]]
        seeds_t = list(dict.fromkeys(order + ["x"]))
        new = {}
        for t in seeds_t:
            ft = lib[t]
            for u, fu in UNARY.items():
                nm = uname(u, t)
                if nm not in lib:
                    new[nm] = (lambda fu, ft: (lambda x: fu(ft(x))))(fu, ft)
            if t != "x":
                nm = f"{t}*x"
                if nm not in lib: new[nm] = (lambda ft: (lambda x: ft(x) * x))(ft)
        lib.update(new); log.append(len(lib))
    return lib

def prune_library(P, lib, thr=0.995):
    """L2: drop terms whose |corr| with an already kept (cheaper) term exceeds thr on the training domain."""
    names = list(lib)
    with np.errstate(all="ignore"):
        Phi = np.column_stack([lib[n](P.xtr) for n in names])
    ok = np.isfinite(Phi).all(0) & (Phi.std(0) > 1e-12); names = [n for n, k in zip(names, ok) if k]; Phi = Phi[:, ok]
    Z = (Phi - Phi.mean(0)) / Phi.std(0); C = np.abs(Z.T @ Z) / len(Z)
    cost = {n: len(n) for n in names}; order = sorted(range(len(names)), key=lambda j: cost[names[j]])
    keep = []
    for j in order:
        if all(C[j, i] < thr for i in keep):
            keep.append(j)
    return {names[j]: lib[names[j]] for j in sorted(keep)}

def exp_E5(seeds=6, noise=.05, out="e5.csv"):
    rows = []
    full = {n: v[0] for n, v in LIB.items()}
    small = {n: v[0] for n, v in LIB.items() if v[1] <= 2}
    tier1 = {n: v[0] for n, v in LIB.items() if v[1] <= 1}
    for fid in T:
        for sd in range(seeds):
            P = make_problem(fid, noise, sd)
            libs = {"a_fixed_small(13)": small, "b_fixed_large(26)": full}
            libs["c_growth"] = grow_library(P, tier1)
            libs["d_prune(26)"] = prune_library(P, full)
            libs["e_growth+prune"] = prune_library(P, libs["c_growth"])
            for ln, lib in libs.items():
                t0 = time.perf_counter()
                D = design(P, lib)
                idx, cand = best_subset(D.gr, 3)
                el = time.perf_counter() - t0
                r = evaluate(P, D, idx)
                r.update(exp="E5", function=fid, family=P.fam, noise=noise, seed=sd, model="SN-Exh(K=3)", library=ln,
                         lib_size=len(D.names), cond=float(np.linalg.cond(D.Z)), time=el, candidates=cand)
                rows.append(r)
    return rows

# ================================================================ E6 : structural uncertainty
def exp_E6(seeds=8, noises=(.01, .05, .10, .20), B=30, out="e6.csv"):
    rows, calib = [], []
    for fid in T:
        for s in noises:
            for sd in range(seeds):
                P = make_problem(fid, s, sd); D = design(P)
                idx, cand, all_s, all_b = best_subset(D.gr, 3, keep_all=True)
                pi, mass, ent = posterior(all_s, all_b, len(D.names))
                # bootstrap stability: MAP of each resample
                rng = np.random.default_rng([sd, 77, list(T).index(fid)]); freq = np.zeros(len(D.names))
                for _ in range(B):
                    ii = rng.choice(len(P.xtr), len(P.xtr))
                    Zb, yb = D.Z[ii], D.yz[ii]; Zb = Zb - Zb.mean(0); yb = yb - yb.mean()
                    ib, _ = best_subset(Gram(Zb, yb), 3)
                    for j in ib: freq[j] += 1
                freq /= B
                r = evaluate(P, D, idx)
                r.update(exp="E6", function=fid, family=P.fam, noise=s, seed=sd, model="SN-Exh(K=3)",
                         mass=mass, entropy=ent, mean_pi_sel=float(np.mean(pi[idx])) if idx else np.nan,
                         min_pi_sel=float(np.min(pi[idx])) if idx else np.nan,
                         min_freq_sel=float(np.min(freq[idx])) if idx else np.nan)
                rows.append(r)
                if P.coefs is not None:
                    tv = set(P.coefs)
                    for j, nm in enumerate(D.names):
                        calib.append(dict(function=fid, noise=s, seed=sd, term=nm, pi_bic=float(pi[j]),
                                          pi_boot=float(freq[j]), truth=float(nm in tv)))
    import pandas as pd
    pd.DataFrame(calib).to_csv(out.replace(".csv", "_calib.csv"), index=False)
    return rows
