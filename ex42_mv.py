"""ex42_mv.py -- multivariate library experiments: E7, E8, E9, E12."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import itertools, time, warnings, zlib
import numpy as np
warnings.filterwarnings("ignore")
import ex42_core as core
from ex42_core import Gram, best_subset, stepwise, run_method

EXPS = (-1, 0, 1, 2)
class Var:
    def __init__(s, name, lo, hi, dim=(0, 0, 0), kind="pos", ext=None):
        s.name, s.lo, s.hi, s.dim, s.kind, s.ext = name, lo, hi, np.array(dim, float), kind, ext
class Tgt:
    def __init__(s, name, vars, f, true, ydim, flips=(), note=""):
        s.name, s.vars, s.f, s.true, s.ydim, s.flips, s.note = name, vars, f, true, np.array(ydim, float), list(flips), note
        s.exps = EXPS

def enum_terms(vars, exps=EXPS):
    opts = [exps if v.kind == "pos" else (0, "sin", "cos") for v in vars]
    return [c for c in itertools.product(*opts) if not all(o == 0 for o in c)]
def tname(vars, combo):
    fs = []
    for v, o in zip(vars, combo):
        if o == 0: continue
        fs.append(f"{o}({v.name})" if v.kind == "angle" else (v.name if o == 1 else f"{v.name}^{o}"))
    return "*".join(fs)
def tval(vars, combo, X):
    out = np.ones(len(X))
    with np.errstate(all="ignore"):
        for j, (v, o) in enumerate(zip(vars, combo)):
            if o == 0: continue
            out = out * (np.sin(X[:, j]) if o == "sin" else np.cos(X[:, j]) if o == "cos" else X[:, j] ** o)
    return out
def tdim(vars, combo):
    return sum((o * v.dim for v, o in zip(vars, combo) if v.kind == "pos"), np.zeros(3))
def dim_ok(t, combo, extra=None):
    d = t.ydim - tdim(t.vars, combo)
    if extra is None:
        return bool(np.allclose(d, 0))
    return any(np.allclose(d, e) for e in extra)
def sym_ok(t, combo):
    idx = [i for i, v in enumerate(t.vars) if v.name in t.flips]
    return sum(combo[i] for i in idx if isinstance(combo[i], (int, float))) % 2 == 0

def sample(t, n, rng, region):
    cols = []
    for v in t.vars:
        if region == "ext" and v.kind == "pos":
            lo, hi = v.ext if v.ext else (v.hi, 2 * v.hi)
        else:
            lo, hi = v.lo, v.hi
        cols.append(rng.uniform(lo, hi, n))
    return np.column_stack(cols)

class MP:
    pass
def mv_problem(t, sigma, seed, N=100, nt=500):
    rng = np.random.default_rng([zlib.crc32(t.name.encode()) % 10**6, int(round(sigma * 100)), seed])
    P = MP(); P.t = t
    P.Xtr = sample(t, N, rng, "train"); P.Xi = sample(t, nt, rng, "int"); P.Xe = sample(t, nt, rng, "ext")
    clean = t.f(P.Xtr); P.ytr = clean + sigma * clean.std() * rng.standard_normal(N)
    P.yi, P.ye = t.f(P.Xi), t.f(P.Xe)
    P.scale = float(np.sqrt(np.mean(np.r_[P.yi, P.ye] ** 2))); P.ymax = float(np.abs(np.r_[P.yi, P.ye]).max())
    return P

def mv_design(P, mask_fn=None, extra_dims=None):
    t = P.t; combos = enum_terms(t.vars, t.exps)
    if mask_fn is not None:
        combos = [c for c in combos if mask_fn(c)]
    if not combos:
        return None
    Phi = np.column_stack([tval(t.vars, c, P.Xtr) for c in combos])
    Pi = np.column_stack([tval(t.vars, c, P.Xi) for c in combos]); Pe = np.column_stack([tval(t.vars, c, P.Xe) for c in combos])
    ok = np.isfinite(Phi).all(0) & (Phi.std(0) > 1e-12) & np.isfinite(Pi).all(0) & np.isfinite(Pe).all(0)
    combos = [c for c, k in zip(combos, ok) if k]; Phi, Pi, Pe = Phi[:, ok], Pi[:, ok], Pe[:, ok]
    D = MP(); D.combos = combos; D.names = [tname(t.vars, c) for c in combos]
    mu, sd = Phi.mean(0), Phi.std(0) + 1e-12
    D.Z = (Phi - mu) / sd; D.Zi = (Pi - mu) / sd; D.Ze = (Pe - mu) / sd
    D.ym, D.ys = P.ytr.mean(), P.ytr.std() + 1e-12; D.yz = (P.ytr - D.ym) / D.ys
    D.gr = Gram(D.Z, D.yz); D.mu, D.sd = mu, sd
    D.true_names = None if t.true is None else {tname(t.vars, c): v for c, v in t.true.items()}
    return D

def mv_eval(P, D, idx, c=None):
    idx = list(idx)
    if c is None: c = D.gr.coef(idx)
    def pred(Z):
        return D.ys * (c[0] + (Z[:, idx] @ np.asarray(c[1:]) if idx else 0)) + D.ym
    pi = np.nan_to_num(pred(D.Zi), nan=1e9, posinf=1e9, neginf=-1e9); pe = np.nan_to_num(pred(D.Ze), nan=1e9, posinf=1e9, neginf=-1e9)
    ri = float(np.sqrt(np.mean((pi - P.yi) ** 2))); re = float(np.sqrt(np.mean((pe - P.ye) ** 2)))
    got = sorted(D.names[j] for j in idx)
    r = dict(nrmse_int=ri / P.scale, nrmse_ext=re / P.scale, k=len(idx),
             relmax=float(max(np.abs(pi - P.yi).max(), np.abs(pe - P.ye).max()) / P.ymax))
    r["srr_sem"] = float(r["relmax"] <= 0.05); r["explosive"] = float(r["nrmse_ext"] > 100)
    if D.true_names is not None:
        r["srr_struct"] = float(set(got) == set(D.true_names)); r["covered"] = float(set(D.true_names) <= set(D.names))
    else:
        r["srr_struct"] = np.nan; r["covered"] = 0.0
    r["terms"] = "+".join(got); r["lib_size"] = len(D.names)
    return r

# ----------------------------------------------------------------- targets
def V(n, lo=1, hi=3, dim=(0, 0, 0), **k): return Var(n, lo, hi, dim, **k)
M_, L_, T_ = (1, 0, 0), (0, 1, 0), (0, 0, 1)
def dm(M=0, L=0, T=0): return (M, L, T)
def combo(vars, **p):
    return tuple(p.get(v.name, 0) for v in vars)

def build_E7():
    out = []
    V = lambda n, dim=(0, 0, 0): Var(n, 1.0, 1.6, dim)          # narrow training range => collinear monomials
    vs = [V("m", dim=dm(M=1)), V("v", dim=dm(L=1, T=-1)), V("g", dim=dm(L=1, T=-2)), V("h", dim=dm(L=1))]
    out.append(Tgt("E7-energy", vs, lambda X: .5 * X[:, 0] * X[:, 1] ** 2 + X[:, 0] * X[:, 2] * X[:, 3],
                   {combo(vs, m=1, v=2): .5, combo(vs, m=1, g=1, h=1): 1.0}, dm(1, 2, -2), flips=["v"]))
    vs = [V("rho", dim=dm(M=1, L=-3)), V("v", dim=dm(L=1, T=-1)), V("g", dim=dm(L=1, T=-2)), V("h", dim=dm(L=1))]
    out.append(Tgt("E7-bernoulli", vs, lambda X: .5 * X[:, 0] * X[:, 1] ** 2 + X[:, 0] * X[:, 2] * X[:, 3],
                   {combo(vs, rho=1, v=2): .5, combo(vs, rho=1, g=1, h=1): 1.0}, dm(1, -1, -2), flips=["v"]))
    vs = [V("v", dim=dm(L=1, T=-1)), V("a", dim=dm(L=1, T=-2)), V("t", dim=dm(T=1)), V("m", dim=dm(M=1))]
    out.append(Tgt("E7-kinematics", vs, lambda X: X[:, 0] * X[:, 2] + .5 * X[:, 1] * X[:, 2] ** 2,
                   {combo(vs, v=1, t=1): 1.0, combo(vs, a=1, t=2): .5}, dm(L=1), flips=["v", "t"]))
    return out

def build_E9():
    out = []
    def add(name, vs, f, true, ydim):
        tg = Tgt(name, vs, f, true, ydim); tg.exps = (-2, -1, 0, 1, 2); out.append(tg)
    vs = [V("m", dim=dm(M=1)), V("g", dim=dm(L=1, T=-2)), V("z", dim=dm(L=1))]
    add("I.14.3 U=mgz", vs, lambda X: X[:, 0] * X[:, 1] * X[:, 2], {combo(vs, m=1, g=1, z=1): 1.0}, dm(1, 2, -2))
    vs = [V("k", dim=dm(M=1, T=-2)), V("x", dim=dm(L=1))]
    add("I.14.4 U=kx^2/2", vs, lambda X: .5 * X[:, 0] * X[:, 1] ** 2, {combo(vs, k=1, x=2): .5}, dm(1, 2, -2))
    vs = [V("p", dim=dm(M=1, L=-1, T=-2)), V("V", dim=dm(L=3))]
    add("I.39.1 E=3pV/2", vs, lambda X: 1.5 * X[:, 0] * X[:, 1], {combo(vs, p=1, V=1): 1.5}, dm(1, 2, -2))
    vs = [V("P", dim=dm(M=1, L=2, T=-3)), V("r", dim=dm(L=1))]
    add("II.3.24 Ef=P/(4pi r^2)", vs, lambda X: X[:, 0] / (4 * np.pi * X[:, 1] ** 2), {combo(vs, P=1, r=-2): 1 / (4 * np.pi)}, dm(1, 0, -3))
    vs = [V("r", dim=dm(L=1)), V("F", dim=dm(M=1, L=1, T=-2)), V("th", .2, 1.2, kind="angle")]
    add("I.18.12 tau=rF sin(th)", vs, lambda X: X[:, 0] * X[:, 1] * np.sin(X[:, 2]), {combo(vs, r=1, F=1, th="sin"): 1.0}, dm(1, 2, -2))
    vs = [V("w", dim=dm(T=-1)), V("c", dim=dm(L=1, T=-1))]
    add("I.29.4 k=w/c", vs, lambda X: X[:, 0] / X[:, 1], {combo(vs, w=1, c=-1): 1.0}, dm(L=-1))
    vs = [V("m", dim=dm(M=1)), V("w", dim=dm(T=-1)), V("w0", dim=dm(T=-1)), V("x", dim=dm(L=1))]
    add("I.24.6 E=m(w^2+w0^2)x^2/4", vs, lambda X: .25 * X[:, 0] * (X[:, 1] ** 2 + X[:, 2] ** 2) * X[:, 3] ** 2,
        {combo(vs, m=1, w=2, x=2): .25, combo(vs, m=1, w0=2, x=2): .25}, dm(1, 2, -2))
    vs = [V("m0", dim=dm(M=1)), V("b", .1, .7, ext=(.7, .95))]
    add("I.10.7 m=m0/sqrt(1-b^2) [out of lib]", vs, lambda X: X[:, 0] / np.sqrt(1 - X[:, 1] ** 2), None, dm(M=1))
    vs = [V("th", .2, 1.2, ext=(1.2, 2.5))]
    add("I.6.20a gauss [out of lib]", vs, lambda X: np.exp(-X[:, 0] ** 2 / 2) / np.sqrt(2 * np.pi), None, dm())
    return out

# ----------------------------------------------------------------- E7
def exp_E7(seeds=15, noises=(.05, .10, .20), N=50):
    rows = []
    for t in build_E7():
        for s in noises:
            for sd in range(seeds):
                P = mv_problem(t, s, sd, N=N)
                for cfg in ("none", "dim", "sym", "dim+sym"):
                    def mk(c):
                        ok = True
                        if "dim" in cfg: ok = ok and dim_ok(t, c)
                        if "sym" in cfg: ok = ok and sym_ok(t, c)
                        return ok
                    D = mv_design(P, mk)
                    t0 = time.perf_counter(); idx, cand = best_subset(D.gr, 2); el = time.perf_counter() - t0
                    r = mv_eval(P, D, idx); r.update(exp="E7", target=t.name, noise=s, seed=sd, config=cfg, model="SN-Exh(K=2)", time=el, candidates=cand)
                    rows.append(r)
    return rows

# ----------------------------------------------------------------- E9
def mlp_mv(X, y, Xte_list, seed):
    from sklearn.neural_network import MLPRegressor
    mx, sx = X.mean(0), X.std(0) + 1e-12; my, sy = y.mean(), y.std() + 1e-12
    Xs, t = (X - mx) / sx, (y - my) / sy
    perm = np.random.default_rng(seed).permutation(len(X)); nv = max(2, len(X) // 5); va, tr = perm[:nv], perm[nv:]
    best = None
    for lr, a in [(3e-3, 0.), (1e-3, 1e-3)]:
        m = MLPRegressor(hidden_layer_sizes=(64, 64), activation="relu", solver="adam", learning_rate_init=lr, alpha=a,
                         max_iter=600, tol=1e-7, n_iter_no_change=40, random_state=seed).fit(Xs[tr], t[tr])
        v = np.mean((m.predict(Xs[va]) - t[va]) ** 2)
        if best is None or v < best[0]: best = (v, lr, a)
    m = MLPRegressor(hidden_layer_sizes=(64, 64), activation="relu", solver="adam", learning_rate_init=best[1], alpha=best[2],
                     max_iter=600, tol=1e-7, n_iter_no_change=40, random_state=seed).fit(Xs, t)
    return [m.predict((Z - mx) / sx) * sy + my for Z in Xte_list]

def gp_mv(X, y, Xte_list, seed):
    from gplearn.genetic import SymbolicRegressor
    m = SymbolicRegressor(population_size=600, generations=12, tournament_size=20, function_set=("add", "sub", "mul", "div", "sqrt"),
                          parsimony_coefficient=1e-3, const_range=(-3, 3), random_state=seed, n_jobs=1, verbose=0).fit(X, y)
    return [m.predict(Z) for Z in Xte_list], str(m._program)

def _ext_metrics(P, pi, pe):
    pi = np.nan_to_num(pi, nan=1e9, posinf=1e9, neginf=-1e9); pe = np.nan_to_num(pe, nan=1e9, posinf=1e9, neginf=-1e9)
    ri = float(np.sqrt(np.mean((pi - P.yi) ** 2))); re = float(np.sqrt(np.mean((pe - P.ye) ** 2)))
    rel = float(max(np.abs(pi - P.yi).max(), np.abs(pe - P.ye).max()) / P.ymax)
    return dict(nrmse_int=ri / P.scale, nrmse_ext=re / P.scale, relmax=rel, srr_sem=float(rel <= .05), explosive=float(re / P.scale > 100))

E9_SYM = [("SN-Exh", 2, False), ("SN-Exh+dim", 2, True), ("SN-Greedy", 3, False), ("OMP", 2, False), ("STLSQ", None, False),
          ("STLSQ-ridge", None, False), ("LASSO", None, False), ("LASSO-stab", None, False)]
def exp_E9(seeds=6, noises=(0, .01, .05), gp_seeds=2):
    rows = []
    for t in build_E9():
        for s in noises:
            for sd in range(seeds):
                P = mv_problem(t, s, sd)
                for m, k, usedim in E9_SYM:
                    D = mv_design(P, (lambda c, t=t: dim_ok(t, c)) if usedim else None)
                    if D is None: continue            # no library term has the target dimension: method not applicable
                    t0 = time.perf_counter(); idx, cand, c = run_method(m.replace("+dim", ""), D, sd, k or 3); el = time.perf_counter() - t0
                    r = mv_eval(P, D, idx, c); r.update(exp="E9", target=t.name, noise=s, seed=sd, model=m, time=el, candidates=cand,
                                                        in_lib=int(t.true is not None), lib_size=len(D.names), kind="library-based")
                    rows.append(r)
                t0 = time.perf_counter(); pi, pe = mlp_mv(P.Xtr, P.ytr, [P.Xi, P.Xe], sd); el = time.perf_counter() - t0
                r = _ext_metrics(P, pi, pe); r.update(exp="E9", target=t.name, noise=s, seed=sd, model="MLP-ReLU", time=el, in_lib=int(t.true is not None),
                                                      srr_struct=np.nan, kind="library-free")
                rows.append(r)
                if sd < gp_seeds:
                    try:
                        t0 = time.perf_counter(); (pi, pe), prog = gp_mv(P.Xtr, P.ytr, [P.Xi, P.Xe], sd); el = time.perf_counter() - t0
                        r = _ext_metrics(P, pi, pe); r.update(exp="E9", target=t.name, noise=s, seed=sd, model="GP (gplearn)", time=el,
                                                              in_lib=int(t.true is not None), srr_struct=np.nan, kind="library-free", terms=prog)
                        rows.append(r)
                    except Exception as e:
                        print("gp fail", e)
    return rows

# ----------------------------------------------------------------- E8 implicit neurons
IMP_TERMS = [(i, j) for d in range(0, 5) for i in range(d, -1, -1) for j in [d - i]]     # x^i y^j, degree<=4 (15 terms)
def _pn(v, p): return "" if p == 0 else (v if p == 1 else f"{v}^{p}")
IMP_NAMES = [("1" if (i, j) == (0, 0) else "*".join([q for q in (_pn("x", i), _pn("y", j)) if q])) for i, j in IMP_TERMS]
def imp_theta(x, y):
    return np.column_stack([x ** i * y ** j for i, j in IMP_TERMS])
def imp_grad(c, x, y):
    gx = sum(cj * i * x ** max(i - 1, 0) * y ** j for cj, (i, j) in zip(c, IMP_TERMS))
    gy = sum(cj * j * x ** i * y ** max(j - 1, 0) for cj, (i, j) in zip(c, IMP_TERMS))
    return gx, gy

IMP_T = {
    "circle": dict(param=lambda t: (np.cos(t), np.sin(t)), tr=(0.3, 2.3), te=(0, 2 * np.pi), true={"x^2": 1, "y^2": 1, "1": -1}),
    "ellipse": dict(param=lambda t: (2 * np.cos(t), np.sin(t)), tr=(0.3, 2.3), te=(0, 2 * np.pi), true={"x^2": .25, "y^2": 1, "1": -1}),
    "hyperbola": dict(param=lambda t: (np.cosh(t), np.sinh(t)), tr=(-1, 1), te=(-2, 2), true={"x^2": 1, "y^2": -1, "1": -1}),
    "parabola": dict(param=lambda t: (t, t ** 2), tr=(-1, 1), te=(-2, 2), true={"y": 1, "x^2": -1}),
}
def _msub(Th, sub, w=None):
    A = Th[:, sub]; A = A * np.sqrt(w)[:, None] if w is not None else A
    M = A.T @ A / len(A); ev, U = np.linalg.eigh(M); return ev[0], U[:, 0]
def imp_alg_exh(x, y, kmax=3, geo=False, iters=3):
    Th = imp_theta(x, y); n = len(x); p = Th.shape[1]; best = (np.inf, None, None)
    cn = Th.std(0) + 1e-9; Thn = Th / np.maximum(np.abs(Th).max(0), 1e-9)               # column scaling for numerical balance
    for k in range(1, kmax + 1):
        for sub in itertools.combinations(range(p), k):
            sub = list(sub); lam, v = _msub(Thn, sub)
            if geo:
                for _ in range(iters):
                    c = np.zeros(p); c[sub] = v / np.maximum(np.abs(Th).max(0)[sub], 1e-9)
                    gx, gy = imp_grad(c, x, y); w = 1 / (gx ** 2 + gy ** 2 + 1e-8)
                    lam, v = _msub(Thn, sub, w / w.mean())
                c = np.zeros(p); c[sub] = v / np.maximum(np.abs(Th).max(0)[sub], 1e-9)
                gx, gy = imp_grad(c, x, y); F = Th @ c; S = np.mean(F ** 2 / (gx ** 2 + gy ** 2 + 1e-8))
                crit = n * np.log(S + 1e-10) + k * np.log(n)
            else:
                c = np.zeros(p); c[sub] = v / np.maximum(np.abs(Th).max(0)[sub], 1e-9)
                crit = n * np.log(lam + 1e-10) + k * np.log(n)
            if crit < best[0]: best = (crit, sub, c / np.linalg.norm(c))
    return best[2]
def imp_svd_stlsq(x, y, thr=0.1, iters=8):
    Th = imp_theta(x, y); sc = np.maximum(np.abs(Th).max(0), 1e-9); Thn = Th / sc; act = list(range(Th.shape[1]))
    for _ in range(iters):
        lam, v = _msub(Thn, act); c = np.zeros(Th.shape[1]); c[act] = v; c /= np.abs(c).max()
        keep = [j for j in act if abs(c[j]) >= thr]
        if len(keep) == len(act) or len(keep) < 2: break
        act = keep
    c = np.zeros(Th.shape[1]); lam, v = _msub(Thn, act); c[act] = v / sc[act]
    return c / np.linalg.norm(c)

def curve_dist(pts, curves):
    from scipy.spatial import cKDTree
    if not curves: return np.full(len(pts), 10.0)
    C = np.vstack(curves); d, _ = cKDTree(C).query(pts); return np.minimum(d, 10.0)
def implicit_curves(c, R):
    import contourpy
    g = np.linspace(-R, R, 300); X, Y = np.meshgrid(g, g); F = imp_theta(X.ravel(), Y.ravel()) @ c
    F = np.nan_to_num(F.reshape(X.shape), nan=1e6)
    gen = contourpy.contour_generator(g, g, F)
    return [l for l in gen.lines(0.0)]

def exp_E8(seeds=20, noises=(0, .01, .02, .05), N=100):
    rows = []
    for name, spec in IMP_T.items():
        for s in noises:
            for sd in range(seeds):
                rng = np.random.default_rng([zlib.crc32(name.encode()) % 10**6, int(round(s * 100)), sd])
                t = rng.uniform(*spec["tr"], N); x0, y0 = spec["param"](t)
                x = x0 + s * x0.std() * rng.standard_normal(N); y = y0 + s * y0.std() * rng.standard_normal(N)
                tt = np.linspace(*spec["te"], 400); xt, yt = spec["param"](tt); pts = np.column_stack([xt, yt]); R = 1.6 * max(np.abs(xt).max(), np.abs(yt).max())
                ctrue = np.zeros(len(IMP_TERMS))
                for k, v in spec["true"].items(): ctrue[IMP_NAMES.index(k)] = v
                ctrue /= np.linalg.norm(ctrue)
                for m in ("Implicit-SVD+STLSQ", "SN-Implicit (algebraic)", "SN-Implicit (geometric)"):
                    t0 = time.perf_counter()
                    c = imp_svd_stlsq(x, y) if m.startswith("Implicit-SVD") else imp_alg_exh(x, y, 3, geo=m.endswith("(geometric)"))
                    el = time.perf_counter() - t0
                    sup = {IMP_NAMES[j] for j in np.flatnonzero(np.abs(c) > 1e-6)}
                    ang = float(np.degrees(np.arccos(min(1.0, abs(c @ ctrue)))))
                    d = curve_dist(pts, implicit_curves(c, R))
                    rows.append(dict(exp="E8", curve=name, noise=s, seed=sd, model=m, srr_struct=float(sup == set(spec["true"])), angle_deg=ang,
                                     dist_med=float(np.median(d)), dist_mean=float(d.mean()), time=el))
                # explicit neuron y=f(x), flexible Study-C library
                t0 = time.perf_counter()
                lib = {n: v[0] for n, v in core.LIB.items()}; names = list(lib)
                Phi = np.column_stack([lib[n](x) for n in names]); ok = np.isfinite(Phi).all(0) & (Phi.std(0) > 1e-12)
                Z = (Phi[:, ok] - Phi[:, ok].mean(0)) / Phi[:, ok].std(0); ym, ys = y.mean(), y.std() + 1e-12; gr = Gram(Z, (y - ym) / ys)
                idx, _ = best_subset(gr, 3); cc = gr.coef(idx); nm = [n for n, k in zip(names, ok) if k]
                mu, sdv = Phi[:, ok].mean(0), Phi[:, ok].std(0)
                xs = np.linspace(min(xt.min(), x.min()) - .2, max(xt.max(), x.max()) + .2, 800)
                with np.errstate(all="ignore"):
                    out = np.full(len(xs), cc[0]);
                    for cj, j in zip(cc[1:], idx): out += cj * (lib[nm[j]](xs) - mu[j]) / sdv[j]
                ys_p = ys * out + ym; good = np.isfinite(ys_p) & (np.abs(ys_p) < 1e3)
                d = curve_dist(pts, [np.column_stack([xs[good], ys_p[good]])] if good.any() else [])
                rows.append(dict(exp="E8", curve=name, noise=s, seed=sd, model="Explicit SN y=f(x)", srr_struct=np.nan, angle_deg=np.nan,
                                 dist_med=float(np.median(d)), dist_mean=float(d.mean()), time=time.perf_counter() - t0))
    return rows

# ----------------------------------------------------------------- E12 synthetic EV
EV_G, EV_RHO, EV_CRR, EV_CDA, EV_ETA = 9.81, 1.2, 0.009, 0.667, 0.9
EV_VARS = [V("m", 1300, 1900, dm(M=1)), V("v", 5, 20, dm(L=1, T=-1)), V("a", -1, 1, dm(L=1, T=-2)),
           V("s", -.052, .052), V("c", .9986, 1.0), V("w", -3, 3, dm(L=1, T=-1))]
def ev_power(m, v, a, s, c, w, Crr=EV_CRR, CdA=EV_CDA, eta=EV_ETA):
    return (m * EV_G * Crr * c + m * EV_G * s + .5 * EV_RHO * CdA * (v + w) ** 2 + m * a) * v / eta
def ev_terms(maxdeg=4):
    names = [v.name for v in EV_VARS]; out = []
    for e in itertools.product(range(maxdeg + 1), repeat=len(names)):
        if 1 <= sum(e) <= maxdeg: out.append(e)
    return out
EV_TRUE = {(1, 1, 0, 0, 1, 0): EV_G * EV_CRR / EV_ETA, (1, 1, 0, 1, 0, 0): EV_G / EV_ETA, (0, 3, 0, 0, 0, 0): .5 * EV_RHO * EV_CDA / EV_ETA,
           (0, 2, 0, 0, 0, 1): EV_RHO * EV_CDA / EV_ETA, (0, 1, 0, 0, 0, 2): .5 * EV_RHO * EV_CDA / EV_ETA, (1, 1, 1, 0, 0, 0): 1 / EV_ETA}
def ev_name(e):
    return "*".join((v.name if p == 1 else f"{v.name}^{p}") for v, p in zip(EV_VARS, e) if p)
def ev_data(n, rng, region="train"):
    m = rng.uniform(1300, 1900, n); a = rng.uniform(-1, 1, n); w = rng.uniform(-3, 3, n)
    v = rng.uniform(20, 35, n) if region == "speed" else rng.uniform(5, 20, n)
    th = np.radians(rng.choice([-1, 1], n) * rng.uniform(5, 8, n)) if region == "slope" else np.radians(rng.uniform(-3, 3, n))
    return np.column_stack([m, v, a, np.sin(th), np.cos(th), w])
def ev_dim_ok(e):
    dv = sum((p * v.dim for v, p in zip(EV_VARS, e)), np.zeros(3)); yd = np.array([1, 2, -3.0])
    gd, rd, Ad = np.array([0, 1, -2.0]), np.array([1, -3, 0.0]), np.array([0, 2, 0.0])
    for ng, nr, na in itertools.product((0, 1), repeat=3):
        if np.allclose(yd - dv, ng * gd + nr * rd + na * Ad): return True
    return False
def exp_E12(seeds=10, noises=(.02, .05), n=600):
    from sklearn.ensemble import GradientBoostingRegressor
    rows = []; terms = ev_terms(); names = [ev_name(e) for e in terms]; okd = np.array([ev_dim_ok(e) for e in terms])
    def phi(X, cols): return np.column_stack([np.prod(X ** np.array(terms[j]), axis=1) for j in cols])
    for s in noises:
        for sd in range(seeds):
            rng = np.random.default_rng([12, int(s * 100), sd]); X = ev_data(n, rng); y0 = ev_power(*X.T)
            y = y0 + s * y0.std() * rng.standard_normal(n)
            tests = {"interp": ev_data(400, rng), "ood_speed": ev_data(400, rng, "speed"), "ood_slope": ev_data(400, rng, "slope")}
            ty = {k: ev_power(*v.T) for k, v in tests.items()}; sc = {k: float(np.sqrt(np.mean(ty[k] ** 2))) for k in ty}
            def score(model, pred_fn, **extra):
                r = dict(exp="E12", noise=s, seed=sd, model=model, **extra)
                for k in tests:
                    p = np.nan_to_num(pred_fn(tests[k]), nan=1e9, posinf=1e9, neginf=-1e9); r["nrmse_" + k] = float(np.sqrt(np.mean((p - ty[k]) ** 2)) / sc[k])
                rows.append(r)
            for lab, cols in (("SN-Step (no filter)", list(range(len(terms)))), ("SN-Step + dimensional filter", list(np.flatnonzero(okd)))):
                Phi = phi(X, cols); mu, sdv = Phi.mean(0), Phi.std(0) + 1e-12; Z = (Phi - mu) / sdv; ym, ys = y.mean(), y.std(); gr = Gram(Z, (y - ym) / ys)
                t0 = time.perf_counter(); idx, cand = stepwise(gr, None); el = time.perf_counter() - t0; c = gr.coef(idx)
                sel = {names[cols[j]] for j in idx}; tr = {ev_name(e) for e in EV_TRUE}; tp = len(sel & tr)
                f1 = 2 * tp / (len(sel) + len(tr)) if (sel or tr) else 0
                def pf(Xt, cols=cols, idx=idx, c=c, mu=mu, sdv=sdv, ym=ym, ys=ys):
                    Zt = (phi(Xt, cols) - mu) / sdv; return ys * (c[0] + Zt[:, idx] @ np.asarray(c[1:])) + ym
                score(lab, pf, support_f1=f1, k=len(idx), time=el, lib_size=len(cols))
            Phi = phi(X, list(range(len(terms)))); mu, sdv = Phi.mean(0), Phi.std(0) + 1e-12; Z = (Phi - mu) / sdv; ym, ys = y.mean(), y.std(); gr = Gram(Z, (y - ym) / ys)
            ci, _, cr = core.stlsq_ridge(gr, mode="cv", seed=sd)
            sel = {names[j] for j in ci}; tp = len(sel & {ev_name(e) for e in EV_TRUE})
            def pf2(Xt, ci=ci, cr=cr): Zt = (phi(Xt, list(range(len(terms)))) - mu) / sdv; return ys * (Zt[:, ci] @ cr) + ym
            score("STLSQ-ridge (CV)", pf2, support_f1=2 * tp / (len(sel) + len(EV_TRUE)), k=len(ci), lib_size=len(terms))
            gb = GradientBoostingRegressor(random_state=sd).fit(X, y); score("Gradient boosting", gb.predict)
            outs = mlp_mv(X, y, [tests["interp"], tests["ood_speed"], tests["ood_slope"]], sd)
            r = dict(exp="E12", noise=s, seed=sd, model="MLP-ReLU")
            for k, p in zip(tests, outs): r["nrmse_" + k] = float(np.sqrt(np.mean((p - ty[k]) ** 2)) / sc[k])
            rows.append(r)
            score("Nominal physics (biased params)", lambda Xt: ev_power(*Xt.T, Crr=.012, CdA=.55, eta=.95))
    return rows
