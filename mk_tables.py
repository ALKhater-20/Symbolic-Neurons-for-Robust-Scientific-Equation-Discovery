import os, sys, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
R, OUT = "results", "tables"; os.makedirs(OUT, exist_ok=True)
def ld(n): return pd.read_csv(f"{R}/{n}.csv")
def num(x, p=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)): return "--"
    x = float(x)
    if x == 0: return "0"
    if abs(x) >= 1000 or abs(x) < 1e-3:
        m, e = f"{x:.1e}".split("e"); return f"${m}\\times10^{{{int(e)}}}$"
    return f"{x:.{p}g}" if abs(x) < 10 else f"{x:.1f}"
def pct(x): return "--" if not np.isfinite(x) else f"{100*x:.1f}"
def esc(s): return str(s).replace("_", "\\_").replace("%", "\\%").replace("&", "\\&").replace("^", "\\^{}")
def write(name, header, rows, colfmt, small=True, resize=False):
    L = []
    if resize: L.append("\\resizebox{\\textwidth}{!}{%")
    L.append(f"\\begin{{tabular}}{{{colfmt}}}\\toprule"); L.append(" & ".join(header) + " \\\\ \\midrule")
    for r in rows:
        L.append("\\midrule" if r == "MID" else " & ".join(r) + " \\\\")
    L.append("\\bottomrule\\end{tabular}" + ("}" if resize else ""))
    open(f"{OUT}/{name}.tex", "w").write("\n".join(L)); print("table", name)

def a1():
    d = ld("a1"); nz = sorted(d.noise.unique()); rows = []
    for n in nz:
        r = [f"{int(n*100)}"]
        for m in ("SN-Exh", "SN-Greedy"):
            x = d[(d.model == m) & (d.noise == n)]
            r += [pct(x.srr_exact.mean()), pct(x.srr_struct.mean()), pct(x.srr_sem.mean())]
        rows.append(r)
    write("a1", ["noise (\\%)", "exact", "struct", "sem", "exact", "struct", "sem"], rows, "rrrrrrr")

CFG = [("SN-Exh", 2), ("SN-Exh", 3), ("SN-Exh", 4), ("SN-Exh", 5), ("SN-Exh", 8), ("SN-Step", 8), ("SN-Step", "inf"), ("SN-Greedy", 3), ("SN-Greedy", 8), ("SN-Greedy", "inf"),
       ("OMP", 3), ("OMP", 8), ("STLSQ", "-"), ("STLSQ-ridge", "-"), ("STLSQ-ridge-cv", "-"), ("LASSO", "-"), ("LASSO-stab", "-")]
def e4():
    d = ld("e4"); d["kmax"] = d.kmax.astype(str); rows = []
    for m, k in CFG:
        x = d[(d.model == m) & (d.kmax == str(k))]
        if x.empty: continue
        lab = esc(m) + ("" if k == "-" else f" ($K{{=}}{'\\infty' if k=='inf' else k}$)")
        r = [lab]
        for n in (.01, .05, .20):
            y = x[x.noise == n]
            r += [pct(y.srr_struct.mean()) if len(y) else "--", num(y.nrmse_ext.median()) if len(y) else "--", pct(y.explosive.mean()) if len(y) else "--"] if len(y) else ["--"] * 3
        r.append(num(x.k.mean(), 2)); r.append(num(x.time.median())); rows.append(r)
    write("e4", ["method", "SRR", "med.", "expl.", "SRR", "med.", "expl.", "SRR", "med.", "expl.", "terms", "time (s)"], rows, "lrrrrrrrrrrr", resize=True)
    # paired comparison on identical runs (seeds 0-1, all noise) for K=8 vs K=3,5
    q = d[(d.model == "SN-Exh") & (d.seed < 2)]; return None

def e5():
    d = ld("e5"); inl = d[d.srr_struct.notna()]; rows = []; ool = d[d.function.isin(["C2", "D4"])]
    for l in list(dict.fromkeys(d.library)):
        rows.append([esc(l), f"{int(d[d.library==l].lib_size.median())}", pct(inl[inl.library == l].covered.mean()), pct(inl[inl.library == l].srr_struct.mean()),
                     num(d[d.library == l].nrmse_ext.median()), num(ool[ool.library == l].nrmse_ext.median()), pct(d[d.library == l].explosive.mean()), num(d[d.library == l].cond.median())])
    write("e5", ["library", "size", "coverage", "SRR$_{struct}$", "med. NRMSE$_{ext}$", "(2 out-of-lib.)", "expl. (\\%)", "cond$(Z)$"], rows, "lrrrrrrr", resize=True)

def ece(p, y, nb=10):
    b = np.minimum((p * nb).astype(int), nb - 1); return sum((b == k).mean() * abs(p[b == k].mean() - y[b == k].mean()) for k in range(nb) if (b == k).any())
def e6():
    e = ld("e6"); c = pd.read_csv(f"{R}/e6_calib.csv"); rows = []
    ev = {"explosive": (e.explosive > .5).values, "NRMSE$_{ext}>1$": (e.nrmse_ext > 1).values, "wrong structure": (e.srr_struct.fillna(0) < .5).values}
    sc = {"1 $-$ posterior mass of $S^{\\ast}$": 1 - e.mass, "entropy of $p(S|D)$": e.entropy, "1 $-$ min $\\pi_j$ over selected": 1 - e.min_pi_sel.fillna(0), "1 $-$ min bootstrap freq.": 1 - e.min_freq_sel.fillna(0)}
    for sn, s in sc.items():
        r = [sn]
        for en, y in ev.items():
            m = (e.srr_struct.notna().values if en == "wrong structure" else np.ones(len(e), bool)) & np.isfinite(s.values)
            r.append(f"{roc_auc_score(y[m], s.values[m]):.2f}")
        rows.append(r)
    write("e6_auroc", ["score", "explosive", "NRMSE$_{ext}>1$", "wrong structure"], rows, "lrrr")
    base = [f"{100*ev['explosive'].mean():.1f}", f"{100*ev['NRMSE$_{ext}>1$'].mean():.1f}", f"{100*ev['wrong structure'][e.srr_struct.notna().values].mean():.1f}"]
    open(f"{OUT}/e6_info.tex", "w").write(f"ECE$_{{BIC}}$={ece(c.pi_bic.values, c.truth.values):.3f}; ECE$_{{boot}}$={ece(c.pi_boot.values, c.truth.values):.3f}; pairs={len(c)}; base rates {base}")
    print("e6 ECE", ece(c.pi_bic.values, c.truth.values), ece(c.pi_boot.values, c.truth.values), len(c), base)

def e7():
    d = ld("e7"); rows = []
    for cfg in ("none", "dim", "sym", "dim+sym"):
        r = [esc(cfg)]
        for n in sorted(d.noise.unique()):
            x = d[(d.config == cfg) & (d.noise == n)]; r += [pct(x.srr_struct.mean()), num(x.nrmse_ext.median()), pct(x.explosive.mean())]
        r.append(f"{int(d[d.config==cfg].lib_size.median())}"); rows.append(r)
    write("e7", ["filter", "SRR", "med.", "expl.", "SRR", "med.", "expl.", "SRR", "med.", "expl.", "lib."], rows, "lrrrrrrrrrr", resize=True)

def e8():
    d = ld("e8"); rows = []
    for m in list(dict.fromkeys(d.model)):
        r = [esc(m)]
        for n in sorted(d.noise.unique()):
            x = d[(d.model == m) & (d.noise == n)]; r += [pct(x.srr_struct.mean()), num(x.dist_med.median())]
        rows.append(r)
    write("e8", ["method", "SRR", "dist.", "SRR", "dist.", "SRR", "dist.", "SRR", "dist."], rows, "lrrrrrrrr", resize=True)
    rows = []
    x = d[d.noise == .02]
    for m in list(dict.fromkeys(d.model)):
        rows.append([esc(m)] + [num(x[(x.model == m) & (x.curve == c)].dist_med.median()) for c in dict.fromkeys(d.curve)])
    write("e8_curves", ["method"] + list(dict.fromkeys(d.curve)), rows, "lrrrr")

def e9():
    p = f"{R}/e9.csv"
    if not os.path.exists(p): return
    d = ld("e9"); rows = []; nz = sorted(d.noise.unique())
    for m in list(dict.fromkeys(d.model)):
        r = [esc(m), d[d.model == m].kind.iloc[0].replace("library-", "lib.-")]
        for n in nz:
            x = d[(d.model == m) & (d.noise == n)]; r += [pct(x[x.in_lib == 1].srr_struct.mean()) if x.srr_struct.notna().any() else "--", pct(x.srr_sem.mean()), num(x.nrmse_ext.median())]
        r.append(pct(d[d.model == m].explosive.mean())); rows.append(r)
    hdr = ["method", "type"] + sum([[f"SRR$_s$", "SRR$_{sem}$", "med."] for _ in nz], []) + ["expl."]
    write("e9", hdr, rows, "ll" + "r" * (3 * len(nz) + 1), resize=True)

def dyn(name, err, nzsel):
    d = ld(name); d["mm"] = d.model + "/" + d.form.str[0]; rows = []
    order = ["SN-Exh/w", "SN-Greedy/w", "STLSQ/w", "STLSQ-ridge/w", "LASSO/w", "OMP/w", "SN-Exh/s", "STLSQ/s"]; nz = sorted(d.noise.unique())
    for m in order:
        r = [esc(m)]
        for n in nz:
            x = d[(d.mm == m) & (d.noise == n)]; r.append(f"{x.support_f1.mean():.2f} / {100*x.srr_struct.mean():.0f}")
        x = d[(d.mm == m) & (d.noise == max(nz))]; r += [num(x[err].median()), pct(x.explosive.mean())]; rows.append(r)
    write(name, ["method"] + [f"{int(n*100)}\\%" for n in nz] + [f"med. err. ({int(max(nz)*100)}\\%)", "expl."], rows, "l" + "r" * len(nz) + "rr", resize=True)

def e12():
    d = ld("e12"); rows = []
    for m in list(dict.fromkeys(d.model)):
        x = d[d.model == m]; rows.append([esc(m), num(x.nrmse_interp.median()), num(x.nrmse_ood_speed.median()), num(x.nrmse_ood_slope.median()),
                                          num(x.support_f1.mean(), 2) if x.support_f1.notna().any() else "--", num(x.k.mean(), 3) if x.k.notna().any() else "--"])
    write("e12", ["model", "in-dist.", "OOD speed", "OOD slope", "support F1", "terms"], rows, "lrrrrr")

if __name__ == "__main__":
    a1(); e4(); e5(); e6(); e7(); e8(); e9(); dyn("e10", "traj_err", None); dyn("e11", "forecast_err", None); e12()
