import os
os.environ.setdefault("OMP_NUM_THREADS", "1")     # avant numpy : timings comparables
import time, itertools, argparse, warnings
import numpy as np, pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LassoCV, ElasticNetCV, OrthogonalMatchingPursuit
warnings.filterwarnings("ignore")

# ======================= Dictionnaire généré par composition =======================
LIB = {}                                   # nom -> (fonction, palier, nb de noeuds)
def reg(n, f, tier, nodes): LIB[n] = (f, tier, nodes)
for k in range(1, 6):                      # palier 1 : polynômes
    reg("x" if k == 1 else f"x^{k}", (lambda k: lambda x: x**k)(k), 1, 1 if k == 1 else 3)
BASES = {"x": (lambda x: x, 1), "2x": (lambda x: 2*x, 3),
         "x^2": (lambda x: x**2, 3), "x^3": (lambda x: x**3, 3)}
for b, (g, nb) in BASES.items():           # u(b) : 4 unaires x 4 bases
    reg(f"sin({b})",  (lambda g: lambda x: np.sin(g(x)))(g),  2, nb+1)   # palier 2
    reg(f"cos({b})",  (lambda g: lambda x: np.cos(g(x)))(g),  2, nb+1)
    reg(f"exp({b})",  (lambda g: lambda x: np.exp(g(x)))(g),  3, nb+1)   # palier 3
    reg(f"exp(-{b})", (lambda g: lambda x: np.exp(-g(x)))(g), 3, nb+2)
reg("log(1+x^2)", lambda x: np.log1p(x**2), 3, 6)
reg("1/(1+x^2)",  lambda x: 1/(1+x**2),     3, 7)
reg("x/(1+x^2)",  lambda x: x/(1+x**2),     3, 8)
# FIX 1 : opérateurs protégés (identiques à l'original sur x+2 >= 0.2, mais finis partout, y compris en x = -2)
reg("log(x+2)",   lambda x: np.log(np.abs(x+2)+1e-6),     3, 4)
reg("1/(x+2)",    lambda x: (x+2)/((x+2)**2+1e-6),        3, 5)     # 26 termes + intercept libre

# ======================= Cibles : id -> (famille, f, coefficients vrais ou None, singulière) =====
T = {}
def add(i, fam, f, coefs, sing=False): T[i] = (fam, f, coefs, sing)
add("A1","Poly",lambda x:x**2,{"x^2":1})
add("A2","Poly",lambda x:x**3+2*x+1,{"x^3":1,"x":2})
add("A3","Poly",lambda x:2*x**4-3*x**2+x+1,{"x^4":2,"x^2":-3,"x":1})
add("A4","Poly",lambda x:.5*x**5-2*x**3+x**2+3,{"x^5":.5,"x^3":-2,"x^2":1})
add("B1","Trig",np.sin,{"sin(x)":1})
add("B2","Trig",lambda x:np.cos(x)+x,{"cos(x)":1,"x":1})
add("B3","Trig",lambda x:x**2+np.sin(x),{"x^2":1,"sin(x)":1})
add("B4","Trig",lambda x:np.sin(x)+.5*np.cos(2*x),{"sin(x)":1,"cos(2x)":.5})
add("C1","ExpLog",lambda x:np.exp(-x**2),{"exp(-x^2)":1})
add("C2","ExpLog",lambda x:np.exp(.5*x),None)                      # HORS dictionnaire
add("C3","ExpLog",lambda x:np.log(x+2),{"log(x+2)":1},True)
add("C4","ExpLog",lambda x:np.exp(-x**2)+.5*x,{"exp(-x^2)":1,"x":.5})
add("D1","Rat",lambda x:1/(1+x**2),{"1/(1+x^2)":1})
add("D2","Rat",lambda x:x/(1+x**2),{"x/(1+x^2)":1})
add("D3","Rat",lambda x:(x**2+1)/(x+2),{"x":1,"1/(x+2)":5},True)   # = x - 2 + 5/(x+2)
add("D4","Rat",lambda x:np.sin(x)/(1+x**2),None)                  # HORS dictionnaire
add("E1","Comp",lambda x:np.sin(x**2),{"sin(x^2)":1})             # NB : DANS le dictionnaire ici
add("E2","Comp",lambda x:np.exp(-x**2)+np.sin(x),{"exp(-x^2)":1,"sin(x)":1})
add("E3","Comp",lambda x:np.log(1+x**2),{"log(1+x^2)":1})
add("E4","Comp",lambda x:np.sin(x)+np.exp(-x**2)+x**2,{"sin(x)":1,"exp(-x^2)":1,"x^2":1})

# ======================= Moteur de sélection (espace standardisé) =======================
def fit_sub(Z, y, idx):
    A = np.column_stack([np.ones(len(y))] + [Z[:, j] for j in idx]) if idx else np.ones((len(y), 1))
    c = np.linalg.lstsq(A, y, rcond=None)[0]; r = y - A @ c
    return c, float(r @ r)
def bic(rss, n, k): return n*np.log(rss/n + 1e-10) + k*np.log(n)   # y standardisé : var = 1

def sn_exh(Z, y, kmax=3):                  # best subset exact, |S| <= kmax
    n, p = Z.shape; best = (bic(fit_sub(Z, y, [])[1], n, 0), []); cand = 1
    for k in range(1, kmax+1):
        for idx in itertools.combinations(range(p), k):
            _, rss = fit_sub(Z, y, list(idx)); cand += 1; b = bic(rss, n, k)
            if b < best[0]: best = (b, list(idx))
    return best[1], cand, ""

def sn_greedy(Z, y, kmax=3):               # forward selection + arrêt BIC
    n, p = Z.shape; sel = []; best = bic(fit_sub(Z, y, [])[1], n, 0); cand = 1; path = []
    while len(sel) < kmax:
        tr = None
        for j in range(p):
            if j in sel: continue
            _, rss = fit_sub(Z, y, sel+[j]); cand += 1
            if tr is None or rss < tr[0]: tr = (rss, j)
        b = bic(tr[0], n, len(sel)+1)
        if b >= best - 1e-9: break
        best = b; sel.append(tr[1]); path.append(f"{tr[1]}:{b:.2f}")
    return sel, cand, "|".join(path)

def stlsq(Z, y, grid=(.005,.01,.02,.05,.1,.2,.4)):   # SINDy ; seuil choisi par BIC
    n, p = Z.shape; best = None; cand = 0
    for thr in grid:
        act = list(range(p))
        for _ in range(10):
            c, _ = fit_sub(Z, y, act); cand += 1
            keep = [j for j, cj in zip(act, c[1:]) if abs(cj) >= thr]
            if len(keep) == len(act): break
            act = keep
        _, rss = fit_sub(Z, y, act); cand += 1; b = bic(rss, n, len(act))
        if best is None or b < best[0]: best = (b, act)
    return best[1], cand, ""

def omp_bic(Z, y, kmax=3):
    n = len(y); best = (bic(fit_sub(Z, y, [])[1], n, 0), []); cand = 1
    for k in range(1, kmax+1):
        m = OrthogonalMatchingPursuit(n_nonzero_coefs=k).fit(Z, y)
        idx = [int(j) for j in np.flatnonzero(m.coef_)]
        _, rss = fit_sub(Z, y, idx); cand += 1; b = bic(rss, n, len(idx))
        if b < best[0]: best = (b, idx)
    return best[1], cand, ""

def lasso_sel(Z, y, seed, enet=False):     # coefficients rétrécis conservés
    m = (ElasticNetCV(l1_ratio=[.5,.9,1.0], cv=5, max_iter=50000, random_state=seed) if enet
         else LassoCV(cv=5, max_iter=50000, random_state=seed)).fit(Z, y)
    idx = [int(j) for j in np.flatnonzero(np.abs(m.coef_) > 1e-8)]
    return idx, np.nan, "", np.r_[m.intercept_, m.coef_[idx]]

def select(m, Z, y, seed):
    if m == "SN-Exh":    return sn_exh(Z, y)
    if m == "SN-Greedy": return sn_greedy(Z, y)
    if m == "STLSQ":     return stlsq(Z, y)
    if m == "OMP":       return omp_bic(Z, y)
    if m == "LASSO":     return lasso_sel(Z, y, seed)
    if m == "ENet":      return lasso_sel(Z, y, seed, enet=True)

def make_pred(names, idx, c, mu, sd, ym, ys):
    def pred(z):
        out = np.full(len(z), c[0], float)
        for cj, j in zip(c[1:], idx): out += cj*(LIB[names[j]][0](z) - mu[j])/sd[j]
        return ys*out + ym
    return pred

# ======================= Baselines neuronales / symboliques lourdes =======================
def mlp_fit(act, x, y, seed, tune=True):
    mu, sd = y.mean(), y.std() + 1e-12; t = (y - mu)/sd
    grid = [(3e-3,0.),(3e-3,1e-3),(1e-3,0.),(1e-3,1e-3)] if tune else [(3e-3,0.)]
    mk = lambda lr, a: MLPRegressor(hidden_layer_sizes=(64,64), activation=act,   # FIX 4 : argument nomme (le 1er positionnel est 'loss' dans scikit-learn recent)
                                     solver="adam", learning_rate_init=lr,
                                    alpha=a, max_iter=1500, tol=1e-7, n_iter_no_change=50, random_state=seed)
    perm = np.random.default_rng(seed).permutation(len(x)); nv = max(2, len(x)//5)
    va, tr = perm[:nv], perm[nv:]; best = None
    if len(grid) > 1:
        for lr, a in grid:
            m = mk(lr, a).fit(x[tr, None], t[tr]); v = np.mean((m.predict(x[va, None]) - t[va])**2)
            if best is None or v < best[0]: best = (v, lr, a)
        lr, a = best[1], best[2]
    else: lr, a = grid[0]
    m = mk(lr, a).fit(x[:, None], t)                       # refit sur tout le train
    return lambda z: m.predict(z[:, None])*sd + mu

def eql_fit(x, y, seed, h=8, E=3000, l1=1e-4):             # EQL sans division (Martius & Lampert)
    import torch, torch.nn as nn
    torch.manual_seed(seed); mu, sd = y.mean(), y.std() + 1e-12
    X = torch.tensor(x[:, None], dtype=torch.float32)
    Y = torch.tensor(((y - mu)/sd)[:, None], dtype=torch.float32)
    class L(nn.Module):
        def __init__(s, i): super().__init__(); s.lin = nn.Linear(i, 5*h)
        def forward(s, z):
            a, b, c, d, e = s.lin(z).split(h, 1)
            return torch.cat([a, torch.sin(b), torch.cos(c), d*e], 1)
    net = nn.Sequential(L(1), L(4*h), nn.Linear(4*h, 1)); opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    for ep in range(E):
        opt.zero_grad(); loss = ((net(X) - Y)**2).mean()
        if ep < 0.7*E: loss = loss + l1*sum(p.abs().sum() for n_, p in net.named_parameters() if "weight" in n_)
        loss.backward(); opt.step()
    def pred(z):
        with torch.no_grad(): return net(torch.tensor(z[:, None], dtype=torch.float32)).numpy()[:, 0]*sd + mu
    return pred, np.nan, ""

def gp_fit(x, y, seed):                                    # gplearn : pas d'exp/log
    from gplearn.genetic import SymbolicRegressor
    m = SymbolicRegressor(population_size=1000, generations=20, tournament_size=20,
        function_set=("add","sub","mul","div","sin","cos"), parsimony_coefficient=1e-3,
        const_range=(-5, 5), random_state=seed, n_jobs=1, verbose=0).fit(x[:, None], y)
    return (lambda z: m.predict(z[:, None])), m._program.length_, str(m._program)

def pysr_fit(x, y, seed, budget=20):
    from pysr import PySRRegressor                          # vérifier l'API de votre version
    m = PySRRegressor(niterations=10**6, timeout_in_seconds=budget,
        binary_operators=["+","-","*","/"], unary_operators=["sin","cos","exp","log","square","cube"],
        maxsize=25, deterministic=True, parallelism="serial", random_state=seed,
        verbosity=0, progress=False, temp_equation_file=True).fit(x[:, None], y)
    b = m.get_best()
    return (lambda z: m.predict(z[:, None])), int(b["complexity"]), str(b["equation"])

FAST = ("SN-Exh","SN-Greedy","STLSQ","LASSO","ENet","OMP","MLP-tanh","MLP-ReLU")
SLOW = ("EQL","GP","PySR")
SYM  = {"SN-Exh","SN-Greedy","STLSQ","LASSO","ENet","OMP"}

# ======================= Évaluation =======================
def rmse(a, b): return float(np.sqrt(np.mean((a-b)**2))) if len(a) else np.nan
def run_config(exp, fid, sigma, seed, N=100, a=1.0, vocab=3, models=FAST, tune=True):
    fam, f, coefs, sing = T[fid]
    rng = np.random.default_rng([list(T).index(fid), int(round(sigma*100)), seed])
    lo, hi = -a, a
    # FIX 2 : région d'extrapolation négative conforme au protocole : [-(a+2), -a] ; limitée à [-1.8, -a] pour les
    #         cibles singulières en x=-2 (C3, D3) ; la version d'origine appliquait -1.8 à TOUTES les fonctions
    #         et le drapeau `sing` n'était jamais utilisé.
    nlo = max(-a-2, -1.8) if sing else -a-2
    xtr = rng.uniform(lo, hi, N); clean = f(xtr)
    ytr = clean + sigma*clean.std()*rng.standard_normal(N)      # bruit relatif à std(y_clean)
    xi, xp, xn = np.linspace(lo, hi, 1000), np.linspace(hi, hi+2, 1000), np.linspace(nlo, lo, 1000)
    with np.errstate(all="ignore"): fi, fp, fn = f(xi), f(xp), f(xn)
    allf = np.concatenate([fi, fp, fn])
    scale = float(np.sqrt(np.mean(allf**2))) + 1e-12; fmax = np.max(np.abs(allf))
    names = [n for n in LIB if LIB[n][1] <= vocab]
    with np.errstate(all="ignore"): Phi = np.column_stack([LIB[n][0](xtr) for n in names])
    ok = np.isfinite(Phi).all(0) & (Phi.std(0) > 1e-12)         # termes indéfinis sur le train retirés
    names = [n for n, k in zip(names, ok) if k]; Phi = Phi[:, ok]
    mu, sd = Phi.mean(0), Phi.std(0) + 1e-12; Z = (Phi - mu)/sd
    ym, ys = ytr.mean(), ytr.std() + 1e-12; yz = (ytr - ym)/ys
    rows = []
    for m in models:
        cand = comp = s_struct = cerr = np.nan; terms = expr = path = ""
        try:
            t0 = time.perf_counter()
            if m in SYM:
                sel = select(m, Z, yz, seed); idx, cand, path = sel[0], sel[1], sel[2]
                c = sel[3] if len(sel) > 3 else fit_sub(Z, yz, idx)[0]
                pred = make_pred(names, idx, c, mu, sd, ym, ys)
            elif m.startswith("MLP"):
                pred = mlp_fit("tanh" if m == "MLP-tanh" else "relu", xtr, ytr, seed, tune)
            elif m == "EQL": pred, comp, expr = eql_fit(xtr, ytr, seed)
            elif m == "GP":  pred, comp, expr = gp_fit(xtr, ytr, seed)
            elif m == "PySR": pred, comp, expr = pysr_fit(xtr, ytr, seed)
            el = time.perf_counter() - t0
            if m in SYM:
                got = sorted(names[j] for j in idx); terms = "+".join(got)
                comp = sum(LIB[names[j]][2] for j in idx) + len(idx) + max(len(idx)-1, 0)
                expr = " + ".join(f"{ys*cj/sd[j]:.4g}*{names[j]}" for j, cj in zip(idx, c[1:]))
                if coefs is not None:
                    s_struct = float(set(got) == set(coefs))
                    if s_struct: cerr = max(abs(ys*cj/sd[j]-coefs[names[j]])/abs(coefs[names[j]])
                                            for j, cj in zip(idx, c[1:]))
            with np.errstate(all="ignore"):
                P = [np.nan_to_num(pred(z), nan=1e9, posinf=1e9, neginf=-1e9) for z in (xi, xp, xn, xtr)]
            pi, pp, pn, ptr = P
            ri, re = rmse(pi, fi), rmse(np.r_[pp, pn], np.r_[fp, fn])
            rows.append(dict(exp=exp, function=fid, family=fam, in_lib=int(coefs is not None),
                noise=sigma, seed=seed, N=N, a=a, vocab=vocab, model=m, time=el, candidates=cand,
                complexity=comp, train_rmse_noisy=rmse(ptr, ytr), train_rmse_clean=rmse(ptr, clean),
                rmse_int=ri, rmse_ext=re, rmse_ext_pos=rmse(pp, fp), rmse_ext_neg=rmse(pn, fn),
                nrmse_int=ri/scale, nrmse_ext=re/scale,
                # FIX 3 : l'epsilon 1e-12 rendait ED astronomique (1e12 et plus) dès que l'interpolation est exacte
                ED=(re/scale)/(ri/scale + 1e-3),
                relmax=float(max(np.abs(pi-fi).max(), np.abs(pp-fp).max(), np.abs(pn-fn).max())/fmax),
                srr_struct=s_struct, coef_err=cerr, terms=terms, expr=expr, path=path))
        except Exception as e:
            print(f"[skip] {m} {fid} s={sigma} seed={seed}: {type(e).__name__}: {e}", flush=True)
    return rows

# ======================= Campagnes =======================
OUT = "results_experiment2_v2.csv"
NOISE = [0, .01, .05, .10, .20]
SLOW_F = ["A2","A4","B3","C1","C2","D4","E1","E4"]            # E1 est DANS le dictionnaire (corrigé dans le commentaire)
ABL_F  = ["A2","A3","B3","B4","C1","D2","E2","E4"]            # non singulières
ABL_SYM = ("SN-Exh","SN-Greedy","STLSQ","LASSO")

def jobs(phase, tune):
    if phase in ("main","all"):
        for fid in T:
            for s in NOISE:
                for sd in range(30): yield dict(exp="main", fid=fid, sigma=s, seed=sd, tune=tune)
    if phase in ("slow","all"):
        for fid in SLOW_F:
            for s in (0, .05, .20):
                for sd in range(10): yield dict(exp="slow", fid=fid, sigma=s, seed=sd, models=SLOW)
    if phase in ("abl","all"):
        for fid in ABL_F:
            for sd in range(30):
                for v in (1,2,3): yield dict(exp="abl_vocab", fid=fid, sigma=.05, seed=sd, vocab=v, models=ABL_SYM)
                for N in (10,25,50,100,250): yield dict(exp="abl_N", fid=fid, sigma=.05, seed=sd, N=N, models=ABL_SYM+("MLP-ReLU",), tune=tune)
                for a in (.5,1.,2.): yield dict(exp="abl_domain", fid=fid, sigma=.05, seed=sd, a=a, models=ABL_SYM+("MLP-ReLU",), tune=tune)

def key(j): return (j["exp"], j["fid"], round(j["sigma"], 4), j["seed"], j.get("N",100), j.get("a",1.0), j.get("vocab",3))
def run_job(j): return run_config(**j)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="all", choices=["main","slow","abl","all"])
    ap.add_argument("--workers", type=int, default=1); ap.add_argument("--no-tune", action="store_true")
    ap.add_argument("--smoke", action="store_true"); ap.add_argument("--out", default=OUT); args = ap.parse_args()
    OUT = args.out
    if args.smoke:
        pd.set_option("display.width", 250, "display.max_colwidth", 60)
        for s in (0, .01, .05):
            r = pd.DataFrame(run_config("smoke", "A2", s, 0, tune=False))
            print(r[["noise","model","time","candidates","srr_struct","nrmse_int","nrmse_ext","relmax","terms"]].to_string(max_colwidth=45))
        raise SystemExit
    done = set()
    if os.path.exists(OUT):
        d = pd.read_csv(OUT)
        done = {(r.exp, r.function, round(r.noise, 4), r.seed, r.N, r.a, r.vocab) for r in d.itertuples()}
    todo = [j for j in jobs(args.phase, not args.no_tune) if key(j) not in done]
    first = not os.path.exists(OUT)
    def sink(rows):
        global first
        if rows: pd.DataFrame(rows).to_csv(OUT, mode="a", header=first, index=False); first = False
    if args.workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(args.workers) as ex:
            for rows in ex.map(run_job, todo, chunksize=2): sink(rows)
    else:
        for j in todo: sink(run_job(j))
