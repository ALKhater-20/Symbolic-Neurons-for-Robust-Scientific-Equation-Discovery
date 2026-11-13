"""ex42_dyn.py -- E10 (dynamical systems) and E11 (PDEs), weak-form identification, no-intercept regression."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import itertools, time, warnings, zlib
import numpy as np
warnings.filterwarnings("ignore")
from scipy.integrate import solve_ivp
from scipy.signal import savgol_filter
import ex42_core as core
from ex42_core import Gram, best_subset, bic_val

# ------------------------------------------------------------ no-intercept selectors
class NG(Gram):
    """Gram without intercept (columns scaled to unit norm, y scaled)."""
    def __init__(self, Th, b):
        self.sc = np.maximum(np.linalg.norm(Th, axis=0), 1e-12) / np.sqrt(len(Th)); self.ys = np.linalg.norm(b) / np.sqrt(len(b)) + 1e-12
        Z, y = Th / self.sc, b / self.ys
        super().__init__(Z, y)
    def coef_sub(self, idx):
        c = np.zeros(self.p)
        if idx:
            c[list(idx)] = np.linalg.lstsq(self.Z[:, list(idx)], self.y, rcond=None)[0]
        return c * self.ys / self.sc                                        # back to original units

def fwd_greedy(g, kmax):
    sel = []; cur = bic_val(g.yy, g.n, 0); cand = 1
    while len(sel) < kmax:
        rest = [j for j in range(g.p) if j not in sel]
        ii = np.array([sel + [j] for j in rest]); b = bic_val(g.rss_batch(ii), g.n, len(sel) + 1); cand += len(rest); j = int(np.argmin(b))
        if b[j] >= cur - 1e-9: break
        sel.append(rest[j]); cur = float(b[j])
    return sel, cand

def stlsq_ni(g, ridge=0.0, grid=(.005, .01, .02, .05, .1, .2, .4)):
    best = None
    for thr in grid:
        act = list(range(g.p))
        for _ in range(10):
            A = g.Z[:, act]; c = np.linalg.solve(A.T @ A + ridge * g.n * np.eye(len(act)), A.T @ g.y)
            keep = [j for j, cj in zip(act, c) if abs(cj) >= thr]
            if len(keep) == len(act) or not keep:
                act = keep if keep else act; break
            act = keep
        A = g.Z[:, act]; c = np.linalg.solve(A.T @ A + ridge * g.n * np.eye(len(act)), A.T @ g.y); r = g.y - A @ c
        bb = bic_val(float(r @ r), g.n, len(act))
        if best is None or bb < best[0]: best = (bb, act, c)
    return best[1], c if False else best[2], len(grid) * 10

def select_ni(m, g, kmax):
    """returns (support, coef in original units)"""
    if m == "SN-Exh": idx, _ = best_subset(g, kmax); return idx, g.coef_sub(idx)
    if m == "SN-Greedy": idx, _ = fwd_greedy(g, kmax); return idx, g.coef_sub(idx)
    if m == "STLSQ":
        act, c, _ = stlsq_ni(g); cc = np.zeros(g.p); cc[act] = c; return act, cc * g.ys / g.sc
    if m == "STLSQ-ridge":
        act, c, _ = stlsq_ni(g, ridge=1e-2); cc = np.zeros(g.p); cc[act] = c; return act, cc * g.ys / g.sc
    if m == "LASSO":
        from sklearn.linear_model import LassoCV
        l = LassoCV(cv=5, fit_intercept=False, max_iter=50000, random_state=0).fit(g.Z, g.y)
        idx = [int(j) for j in np.flatnonzero(np.abs(l.coef_) > 1e-8)]; return idx, l.coef_ * g.ys / g.sc
    if m == "OMP":
        from sklearn.linear_model import OrthogonalMatchingPursuit
        best = (bic_val(g.yy, g.n, 0), [])
        for k in range(1, kmax + 1):
            o = OrthogonalMatchingPursuit(n_nonzero_coefs=min(k, g.p), fit_intercept=False).fit(g.Z, g.y)
            idx = [int(j) for j in np.flatnonzero(o.coef_)]; c = g.coef_sub(idx)
            r = g.y - g.Z[:, idx] @ np.linalg.lstsq(g.Z[:, idx], g.y, rcond=None)[0] if idx else g.y
            bb = bic_val(float(r @ r), g.n, len(idx))
            if bb < best[0]: best = (bb, idx)
        return best[1], g.coef_sub(best[1])
    raise ValueError(m)

# ============================================================ E10 : ODE systems
def mono_names_vals(vars_, deg, extra=()):
    n = len(vars_); exps = [e for e in itertools.product(range(deg + 1), repeat=n) if 1 <= sum(e) <= deg]
    names = ["*".join((v if p == 1 else f"{v}^{p}") for v, p in zip(vars_, e) if p) for e in exps]
    fns = [(lambda e: (lambda X: np.prod(X ** np.array(e), axis=1)))(e) for e in exps]
    for nm, fn in extra: names.append(nm); fns.append(fn)
    return names, fns

SYS = {}
SYS["LotkaVolterra"] = dict(vars=["x", "y"], f=lambda t, z: [1.0 * z[0] - .5 * z[0] * z[1], -.75 * z[1] + .25 * z[0] * z[1]], x0=[4., 2.],
                            T=15., dt=.05, deg=2, extra=[], true=[{"x": 1, "x*y": -.5}, {"y": -.75, "x*y": .25}], x0_test=[3., 1.], horizon=15.)
SYS["Pendulum"] = dict(vars=["th", "om"], f=lambda t, z: [z[1], -np.sin(z[0]) - .1 * z[1]], x0=[2.0, 0.], T=20., dt=.05, deg=3,
                       extra=[("sin(th)", lambda X: np.sin(X[:, 0])), ("cos(th)", lambda X: np.cos(X[:, 0]))],
                       true=[{"om": 1}, {"sin(th)": -1, "om": -.1}], x0_test=[1.2, .5], horizon=20.)
SYS["VanDerPol"] = dict(vars=["x", "y"], f=lambda t, z: [z[1], 1.5 * (1 - z[0] ** 2) * z[1] - z[0]], x0=[2., 0.], T=20., dt=.02, deg=3, extra=[],
                        true=[{"y": 1}, {"y": 1.5, "x^2*y": -1.5, "x": -1}], x0_test=[.5, 1.], horizon=10.)
SYS["Lorenz"] = dict(vars=["x", "y", "z"], f=lambda t, z: [10 * (z[1] - z[0]), z[0] * (28 - z[2]) - z[1], z[0] * z[1] - 8 / 3 * z[2]], x0=[-8., 7., 27.],
                     T=10., dt=.01, deg=2, extra=[], true=[{"x": -10, "y": 10}, {"x": 28, "x*z": -1, "y": -1}, {"x*y": 1, "z": -8 / 3}],
                     x0_test=[1., 1.5, 20.], horizon=3.)

def simulate(f, x0, T, dt):
    t = np.arange(0, T + dt / 2, dt)
    s = solve_ivp(f, (0, T), x0, t_eval=t, method="LSODA", rtol=1e-9, atol=1e-11); return t, s.y.T

def weak_system(t, X, fns, L=40, stride=4, p=4):
    """weak-form data: b_k = -int w' x_k, Theta_w = int w Theta ; test function w=(1-s^2)^p on windows of L samples."""
    dt = t[1] - t[0]; s = np.linspace(-1, 1, L + 1); w = (1 - s ** 2) ** p; dw = (-2 * p * s * (1 - s ** 2) ** (p - 1)) * (2 / (L * dt))
    starts = np.arange(0, len(t) - L, stride)
    Th = np.column_stack([fn(X) for fn in fns])
    A = np.array([(Th[i:i + L + 1] * w[:, None]).sum(0) * dt for i in starts]); B = np.array([-(X[i:i + L + 1] * dw[:, None]).sum(0) * dt for i in starts])
    return A, B

def strong_system(t, X, fns):
    dt = t[1] - t[0]; Xs = savgol_filter(X, 11, 3, axis=0); dX = savgol_filter(X, 11, 3, deriv=1, delta=dt, axis=0)
    return np.column_stack([fn(Xs) for fn in fns]), dX

def exp_E10(seeds=8, noises=(0, .01, .05), kmax=3):
    rows = []
    for name, sp in SYS.items():
        names, fns = mono_names_vals(sp["vars"], sp["deg"], sp["extra"]); nv = len(sp["vars"])
        trueC = np.zeros((len(names), nv))
        for k, d in enumerate(sp["true"]):
            for nm, v in d.items(): trueC[names.index(nm), k] = v
        t, Xc = simulate(sp["f"], sp["x0"], sp["T"], sp["dt"]); tt, Xt = simulate(sp["f"], sp["x0_test"], sp["horizon"], sp["dt"])
        for s in noises:
            for sd in range(seeds):
                rng = np.random.default_rng([zlib.crc32(name.encode()) % 10**6, int(s * 100), sd])
                X = Xc + s * Xc.std(0) * rng.standard_normal(Xc.shape)
                for form in ("weak", "strong"):
                    A, B = weak_system(t, X, fns) if form == "weak" else strong_system(t, X, fns)
                    for m in ("SN-Exh", "SN-Greedy", "STLSQ", "STLSQ-ridge", "LASSO", "OMP"):
                        if form == "strong" and m not in ("SN-Exh", "STLSQ"): continue
                        t0 = time.perf_counter(); C = np.zeros((len(names), nv))
                        for k in range(nv):
                            g = NG(A, B[:, k]); idx, c = select_ni(m, g, kmax); C[:, k] = c
                        el = time.perf_counter() - t0
                        supp = np.abs(C) > 1e-8; tsup = np.abs(trueC) > 0
                        tp = (supp & tsup).sum(); fp = (supp & ~tsup).sum(); fn_ = (~supp & tsup).sum(); f1 = 2 * tp / (2 * tp + fp + fn_)
                        cerr = float(np.linalg.norm(C - trueC) / np.linalg.norm(trueC))
                        def rhs(_, z, C=C):
                            return np.column_stack([fn(np.atleast_2d(z)) for fn in fns])[0] @ C
                        try:
                            sol = solve_ivp(rhs, (0, sp["horizon"]), sp["x0_test"], t_eval=tt, method="LSODA", rtol=1e-7, atol=1e-9)
                            if sol.y.shape[1] == len(tt) and np.isfinite(sol.y).all():
                                tr_err = float(np.sqrt(np.mean((sol.y.T - Xt) ** 2)) / (Xt.std() + 1e-12))
                            else: tr_err = 1e3
                        except Exception:
                            tr_err = 1e3
                        rows.append(dict(exp="E10", system=name, noise=s, seed=sd, model=m, form=form, support_f1=f1, srr_struct=float(((supp == tsup).all())),
                                         coef_err=cerr, traj_err=min(tr_err, 1e3), explosive=float(tr_err > 100), time=el))
    return rows

# ============================================================ E11 : PDEs (periodic 1-D, pseudo-spectral IF-RK4)
def spectral_setup(N, Lx):
    q = 2 * np.pi * np.fft.fftfreq(N, d=Lx / N); dealias = (np.abs(np.fft.fftfreq(N) * N) < N / 3).astype(float)
    return q, dealias

def pde_sim(lin_c, nonlin_c, u0, Lx, T, dt, save_every, const=0.0):
    """u_t = sum_k a_k d^k u + sum_{m>=2,k} c_{k,m} d^k(u^m) ;  lin_c: {k: a_k}, nonlin_c: {(k,m): c}. IF-RK4. returns snapshots (nt,N)."""
    N = len(u0); q, da = spectral_setup(N, Lx); Lop = sum(a * (1j * q) ** k for k, a in lin_c.items()) if lin_c else np.zeros(N, complex)
    nsteps = int(round(T / dt)); E = np.exp(Lop * dt); E2 = np.exp(Lop * dt / 2)
    def Nf(uh):
        u = np.fft.ifft(uh).real; out = const * np.zeros(N, complex) + const * N * (np.arange(N) == 0)
        for (k, m), c in nonlin_c.items():
            out = out + c * (1j * q) ** k * np.fft.fft(u ** m) * da
        return out
    uh = np.fft.fft(u0); snaps = [u0.copy()]
    with np.errstate(all="ignore"):
        for n in range(nsteps):
            k1 = Nf(uh); k2 = Nf(E2 * (uh + dt / 2 * k1)); k3 = Nf(E2 * uh + dt / 2 * k2); k4 = Nf(E * uh + dt * E2 * k3)
            uh = E * uh + dt / 6 * (E * k1 + 2 * E2 * (k2 + k3) + k4)
            if (n + 1) % save_every == 0:
                u = np.fft.ifft(uh).real
                if not np.isfinite(u).all() or np.abs(u).max() > 1e4: return None
                snaps.append(u)
    return np.array(snaps)

PDES = {
    "Burgers": dict(Lx=2 * np.pi, N=128, T=2.0, dt=2e-3, every=10, lin={2: .1}, nl={(1, 2): -.5}, true={"d2(u)": .1, "d1(u^2)": -.5}, hor=1.0,
                    ic=lambda x, r: sum(r.normal(0, 1) / (k) * np.sin(k * x + r.uniform(0, 6.28)) for k in (1, 2, 3))),
    "KdV": dict(Lx=2 * np.pi, N=128, T=.6, dt=1e-4, every=25, lin={3: -1.0}, nl={(1, 2): -3.0}, true={"d3(u)": -1.0, "d1(u^2)": -3.0}, hor=.2,
                ic=lambda x, r: 3 * np.cos(x + r.uniform(0, 6.28)) + 1.5 * np.cos(2 * x + r.uniform(0, 6.28))),
    "KuramotoSivashinsky": dict(Lx=32.0, N=128, T=60., dt=.01, every=10, lin={2: -1.0, 4: -1.0}, nl={(1, 2): -.5},
                                true={"d2(u)": -1.0, "d4(u)": -1.0, "d1(u^2)": -.5}, hor=5.0, spin=40.0,
                                ic=lambda x, r: .1 * (np.cos(2 * np.pi * x / 32 + r.uniform(0, 6.28)) + np.cos(4 * np.pi * x / 32 + r.uniform(0, 6.28)))),
    "FisherKPP": dict(Lx=2 * np.pi, N=128, T=3.0, dt=2e-3, every=10, lin={0: 1.0, 2: .1}, nl={(0, 2): -1.0}, true={"d0(u)": 1.0, "d2(u)": .1, "d0(u^2)": -1.0}, hor=2.0,
                      ic=lambda x, r: .3 + .25 * np.sin(x + r.uniform(0, 6.28)) * np.sin(2 * x + r.uniform(0, 6.28))),
}
PD_TERMS = [(k, m) for m in (1, 2, 3) for k in range(5)]
PD_NAMES = [f"d{k}(u)" if m == 1 else f"d{k}(u^{m})" for k, m in PD_TERMS]

def gen_pde(name, seed, ic_seed_offset=0):
    sp = PDES[name]; rng = np.random.default_rng([zlib.crc32(name.encode()) % 10**6, seed + ic_seed_offset]); x = np.arange(sp["N"]) * sp["Lx"] / sp["N"]
    u0 = sp["ic"](x, rng); lin = {k: a for k, a in sp["lin"].items()}; nl = dict(sp["nl"])
    spin = sp.get("spin", 0.0)
    if spin:
        s0 = pde_sim(lin, nl, u0, sp["Lx"], spin, sp["dt"], int(round(spin / sp["dt"]))); u0 = s0[-1]
    S = pde_sim(lin, nl, u0, sp["Lx"], sp["T"] if not spin else sp["T"] - spin, sp["dt"], sp["every"])
    return x, sp["dt"] * sp["every"], S, u0

def test_fn_derivs(p, kmax):
    """derivatives wrt X of (1-X^2)^p, k=0..kmax, as numpy Polynomial objects"""
    from numpy.polynomial import Polynomial as Pn
    base = (Pn([1, 0, -1])) ** p; return [base.deriv(k) if k else base for k in range(kmax + 1)]

def weak_pde(S, dx, dt, nbox=300, hx=20, ht=20, px=6, pt=3, seed=0):
    nt, N = S.shape; rng = np.random.default_rng(seed)
    X = np.linspace(-1, 1, 2 * hx + 1); Tt = np.linspace(-1, 1, 2 * ht + 1)
    wx = test_fn_derivs(px, 4); wt = test_fn_derivs(pt, 1)
    WX = [np.polyval(pn.coef[::-1], X) / (hx * dx) ** k for k, pn in enumerate(wx)]; WT0 = np.polyval(wt[0].coef[::-1], Tt); WT1 = np.polyval(wt[1].coef[::-1], Tt) / (ht * dt)
    A = []; b = []
    pw = {m: S ** m for m in (1, 2, 3)}
    for _ in range(nbox):
        t0 = rng.integers(0, nt - 2 * ht - 1); x0 = rng.integers(0, N); cols = (x0 + np.arange(-hx, hx + 1)) % N
        blk = {m: pw[m][t0:t0 + 2 * ht + 1][:, cols] for m in pw}
        b.append(-(WT1[:, None] * WX[0][None, :] * blk[1]).sum() * dx * dt)
        A.append([(-1) ** k * (WT0[:, None] * WX[k][None, :] * blk[m]).sum() * dx * dt for k, m in PD_TERMS])
    return np.array(A), np.array(b)

def strong_pde(S, dx, dt):
    from scipy.ndimage import gaussian_filter
    Ss = gaussian_filter(S, sigma=(1.5, 1.5), mode="wrap"); ut = np.gradient(Ss, dt, axis=0, edge_order=2)
    def dxk(F, k):
        for _ in range(k): F = (np.roll(F, -1, 1) - np.roll(F, 1, 1)) / (2 * dx)
        return F
    cols = [dxk(Ss ** m, k)[2:-2, 2:-2].ravel() for k, m in PD_TERMS]; return np.column_stack(cols), ut[2:-2, 2:-2].ravel()

def exp_E11(seeds=5, noises=(0, .01, .05, .10), kmax=4, systems=None):
    rows = []
    for name in (systems or PDES):
        sp = PDES[name]; dx = sp["Lx"] / sp["N"]
        for sd in range(seeds):
            x, dts, S, _ = gen_pde(name, sd)
            if S is None: continue
            _, _, Stest, u0t = None, None, None, None
            xt = x; rng_t = np.random.default_rng([zlib.crc32(name.encode()) % 10**6, 9000 + sd]); u0t = sp["ic"](xt, rng_t)
            if sp.get("spin"):
                u0t = pde_sim(dict(sp["lin"]), dict(sp["nl"]), u0t, sp["Lx"], sp["spin"], sp["dt"], int(round(sp["spin"] / sp["dt"])))[-1]
            nh = int(round(sp["hor"] / sp["dt"])); Ttrue = pde_sim(dict(sp["lin"]), dict(sp["nl"]), u0t, sp["Lx"], sp["hor"], sp["dt"], nh)
            utrue = Ttrue[-1]
            for s in noises:
                rng = np.random.default_rng([zlib.crc32(name.encode()) % 10**6, int(s * 100), sd]); Sn = S + s * S.std() * rng.standard_normal(S.shape)
                for form in ("weak", "strong"):
                    A, b = weak_pde(Sn, dx, dts, seed=sd) if form == "weak" else strong_pde(Sn, dx, dts)
                    for m in ("SN-Exh", "SN-Greedy", "STLSQ", "STLSQ-ridge", "LASSO", "OMP"):
                        if form == "strong" and m not in ("SN-Exh", "STLSQ"): continue
                        t0 = time.perf_counter(); g = NG(A, b); idx, c = select_ni(m, g, kmax); el = time.perf_counter() - t0
                        got = {PD_NAMES[j] for j in np.flatnonzero(np.abs(c) > 1e-10)}; tr = set(sp["true"])
                        tp = len(got & tr); f1 = 2 * tp / (len(got) + len(tr)) if (got or tr) else 0
                        ctrue = np.zeros(len(PD_NAMES))
                        for k_, v in sp["true"].items(): ctrue[PD_NAMES.index(k_)] = v
                        cerr = float(np.linalg.norm(c - ctrue) / np.linalg.norm(ctrue))
                        lin = {k: c[PD_TERMS.index((k, 1))] for k in range(5) if abs(c[PD_TERMS.index((k, 1))]) > 1e-10}
                        nl = {(k, mm): c[j] for j, (k, mm) in enumerate(PD_TERMS) if mm > 1 and abs(c[j]) > 1e-10}
                        try:
                            R = pde_sim(lin, nl, u0t, sp["Lx"], sp["hor"], sp["dt"], nh)
                            err = 1e3 if R is None else float(np.linalg.norm(R[-1] - utrue) / np.linalg.norm(utrue))
                        except Exception:
                            err = 1e3
                        rows.append(dict(exp="E11", system=name, noise=s, seed=sd, model=m, form=form, support_f1=f1, srr_struct=float(got == tr),
                                         coef_err=cerr, forecast_err=min(err, 1e3), explosive=float(err > 100), time=el))
    return rows
