import os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
plt.rcParams.update({"font.size": 8, "axes.grid": True, "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "legend.fontsize": 7, "axes.titlesize": 8.5})
R, F = "results", "figures"
os.makedirs(F, exist_ok=True)
COL = {"SN-Exh": "#d62728", "SN-Exhaustive": "#d62728", "SN-Greedy": "#ff7f0e", "STLSQ": "#2ca02c", "LASSO": "#9467bd", "ENet": "#8c564b", "Elastic Net": "#8c564b",
       "OMP": "#bcbd22", "MLP-tanh": "#1f77b4", "MLP-ReLU": "#17becf", "MLP-Tanh": "#1f77b4"}
def save(fig, name): fig.savefig(f"{F}/{name}.pdf", bbox_inches="tight"); fig.savefig(f"{F}/{name}.png", dpi=200, bbox_inches="tight"); plt.close(fig); print("saved", name)
def ld(n): return pd.read_csv(f"{R}/{n}.csv")
def med(s): return float(np.median(s))

# ------------------------------------------------ reported-value figures (Studies B, C and ablations; numbers from the manuscript tables)
def fig_reported():
    nz = [0, 1, 5, 10, 20]
    B = {"SN-Exhaustive": ([8.8e-15, 1.5409, 6.4261, 10.0327, 15.8956], [2.4e-16, .0012, .0062, .0123, .0245]),
         "SN-Greedy": ([5.3083, 5.5182, 6.3502, 7.7012, 10.3289], [.0103, .0109, .0140, .0181, .0273]),
         "MLP-ReLU": ([5.1525, 5.1531, 5.1611, 5.1756, 5.2175], [.0076, .0080, .0126, .0199, .0346]),
         "MLP-Tanh": ([5.4602, 5.4626, 5.4644, 5.4817, 5.5194], [.0157, .0161, .0186, .0249, .0399])}
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.3))
    for m, (e, i) in B.items():
        ax[0].plot(nz, np.maximum(i, 1e-6), "o-", c=COL[m], label=m, ms=3); ax[1].plot(nz, np.maximum(e, 1e-6), "o-", c=COL[m], ms=3)
    ax[0].set_yscale("log"); ax[1].set_yscale("log"); ax[0].set_title("Interpolation RMSE (mean)"); ax[1].set_title("Extrapolation RMSE (mean)")
    ax[2].plot(nz, [95, 0, 0, 0, 0], "o-", c=COL["SN-Exh"], label="SN-Exhaustive", ms=3); ax[2].plot(nz, [56.5, .17, 0, 0, 0], "o-", c=COL["SN-Greedy"], label="SN-Greedy", ms=3)
    ax[2].set_title(r"SRR$_{exact}$ (%)")
    for a in ax: a.set_xlabel("noise (%)")
    ax[0].legend(); ax[2].legend(); save(fig, "figB_noise")
    C = {"SN-Exh": ([100, 70, 54.6, 47.4, 35.6], [90, 67.8, 50, 43.3, 31.3], [5.4e-16, 2.7e-3, .050, .458, .831]),
         "SN-Greedy": ([50, 41.5, 40.9, 39.1, 32.2], [45, 40.3, 37.5, 35.7, 28.7], [.461, .583, .628, .640, .927]),
         "STLSQ": ([100, 0, 0, 0, 0], [96.7, 0, 0, 0, 0], [5.4e-16, 2.6e16, 1.5e17, 2.7e17, 5.0e17]),
         "LASSO": ([0, .7, .4, .7, 1.5], [10.7, 9.7, 4.7, 2.8, 2.8], [.553, .828, 1.41, 1.69, 2.44]),
         "ENet": ([0, .4, .4, .7, 1.5], [6.7, 5.0, 3.7, 2.8, 2.5], [5.79, 5.93, 1.77, 2.53, 5.01]),
         "OMP": ([44.4, 38.9, 40.7, 38.1, 31.9], [40, 37.2, 38, 35, 28.5], [2.06, 2.02, 1.86, 1.84, 1.83]),
         "MLP-tanh": (None, [.2, .3, .2, .2, 0], [.707, .691, .695, .677, .741]), "MLP-ReLU": (None, [0] * 5, [.778, .775, .775, .782, .793])}
    fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.4))
    for m, (a, b, c) in C.items():
        if a: ax[0].plot(nz, a, "o-", c=COL[m], ms=3, label=m)
        ax[1].plot(nz, b, "o-", c=COL[m], ms=3)
        ax[2].plot(nz, np.maximum(c, 1e-17), "o-", c=COL[m], ms=3)
    ax[0].set_title(r"SRR$_{struct}$ (%)"); ax[1].set_title(r"SRR$_{sem}$ (%)"); ax[2].set_title("median extrapolation NRMSE"); ax[2].set_yscale("log")
    for a in ax: a.set_xlabel("noise (%)")
    ax[0].legend(ncol=2); save(fig, "figC_srr")
    fig, ax = plt.subplots(1, 4, figsize=(7.6, 2.2))
    ax[0].plot([5, 13, 26], [93.3, 44.2, 35.8], "o-", c=COL["SN-Exh"]); ax[0].plot([5, 13, 26], [93.3, 31.7, 26.2], "o-", c=COL["SN-Greedy"]); ax[0].set_title("library size"); ax[0].set_ylabel(r"SRR$_{struct}$ (%)")
    N = [10, 25, 50, 100, 250]
    ax[1].plot(N, [7.5, 21.7, 31.2, 35.8, 54.6], "o-", c=COL["SN-Exh"], label="SN-Exh"); ax[1].plot(N, [14.6, 17.9, 27.5, 26.2, 30.4], "o-", c=COL["SN-Greedy"], label="SN-Greedy"); ax[1].set_xscale("log"); ax[1].set_title("training size N"); ax[1].legend()
    ax[2].plot(N, [9.04, 1.17, .847, .675, .013], "o-", c=COL["SN-Exh"]); ax[2].plot(N, [4.26, 4.46, 4.13, 5.74, 5.41], "o-", c=COL["SN-Greedy"]); ax[2].plot(N, [.909, .855, .836, .817, .812], "o--", c=COL["MLP-ReLU"], label="MLP-ReLU")
    ax[2].set_xscale("log"); ax[2].set_yscale("log"); ax[2].set_title("median NRMSE$_{ext}$ vs N"); ax[2].legend()
    a_ = [.5, 1, 2]; ax[3].plot(a_, [20, 35.8, 62.9], "o-", c=COL["SN-Exh"]); ax[3].plot(a_, [17.1, 26.2, 31.7], "o-", c=COL["SN-Greedy"]); ax[3].set_title("training half-width a")
    save(fig, "figC_ablation")

# ------------------------------------------------ A1
def fig_A1():
    d = ld("a1"); nz = sorted(d.noise.unique()); fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.3)); M = {"SN-Exh": "-", "SN-Greedy": "--"}
    for m, ls in M.items():
        x = d[d.model == m]
        for a, col, ttl, sub in ((ax[0], "srr_exact", r"SRR$_{exact}$", x[x.srr_struct.notna()]), (ax[1], "srr_struct", r"SRR$_{struct}$", x[x.srr_struct.notna()]), (ax[2], "srr_sem", r"SRR$_{sem}$", x)):
            a.plot([n * 100 for n in nz], [100 * sub[sub.noise == n][col].mean() for n in nz], "o" + ls, c=COL[m], ms=3, label=m); a.set_title(ttl + " (%)"); a.set_xlabel("noise (%)")
    ax[0].legend(); save(fig, "figA1_criteria")

# ------------------------------------------------ E4
def lab(r):
    k = r.kmax; return f"{r.model}" + (f"\nK={k}" if str(k) not in ("-", "nan") else "")
def fig_E4():
    d = ld("e4"); d["cfg"] = d.apply(lab, axis=1)
    pref = ["SN-Exh", "SN-Step", "SN-Greedy", "OMP", "STLSQ", "STLSQ-ridge", "STLSQ-ridge-cv", "LASSO", "LASSO-stab"]
    order = sorted(dict.fromkeys(d.cfg), key=lambda c: (pref.index(c.split("\n")[0]), c)); nzs = sorted(d.noise.unique())
    fig, ax = plt.subplots(3, 1, figsize=(7.4, 6.2), sharex=True); w = .8 / len(nzs)
    for i, n in enumerate(nzs):
        x = d[d.noise == n]; pos = np.arange(len(order)) + i * w
        srr = [100 * x[x.cfg == c].srr_struct.mean() for c in order]; med_ = [np.median(x[x.cfg == c].nrmse_ext) for c in order]; ex = [100 * x[x.cfg == c].explosive.mean() for c in order]
        ax[0].bar(pos, srr, w, label=f"noise {int(n*100)}%"); ax[1].bar(pos, np.maximum(med_, 1e-4), w); ax[2].bar(pos, ex, w)
    ax[0].set_ylabel(r"SRR$_{struct}$ (%)"); ax[1].set_ylabel("median NRMSE$_{ext}$"); ax[1].set_yscale("log"); ax[2].set_ylabel("explosive share (%)")
    ax[2].set_xticks(np.arange(len(order)) + w); ax[2].set_xticklabels(order, rotation=55, ha="right", fontsize=6.5); ax[0].legend(); save(fig, "figE4_kmax")

# ------------------------------------------------ E5
def fig_E5():
    d = ld("e5"); lib = list(dict.fromkeys(d.library)); fig, ax = plt.subplots(1, 4, figsize=(7.8, 2.5)); x = np.arange(len(lib))
    inl = d[d.srr_struct.notna()]
    ax[0].bar(x - .2, [100 * inl[inl.library == l].covered.mean() for l in lib], .4, label="coverage"); ax[0].bar(x + .2, [100 * inl[inl.library == l].srr_struct.mean() for l in lib], .4, label=r"SRR$_{struct}$"); ax[0].legend(); ax[0].set_title("in-library targets (%)")
    ax[1].bar(x, [np.median(d[d.library == l].nrmse_ext) for l in lib], color="#d62728"); ax[1].set_yscale("log"); ax[1].set_title("median NRMSE$_{ext}$ (all 20)")
    ax[2].bar(x, [100 * d[d.library == l].explosive.mean() for l in lib], color="#ff7f0e"); ax[2].set_title("explosive share (%)")
    ax[3].bar(x, [np.median(d[d.library == l].cond) for l in lib], color="#2ca02c"); ax[3].set_yscale("log"); ax[3].set_title("median cond(Z)")
    for a in ax: a.set_xticks(x); a.set_xticklabels([l.split("(")[0] for l in lib], rotation=60, ha="right", fontsize=6.5)
    save(fig, "figE5_library")

# ------------------------------------------------ E6
def ece(p, y, nb=10):
    b = np.minimum((p * nb).astype(int), nb - 1); e = 0
    for k in range(nb):
        m = b == k
        if m.any(): e += m.mean() * abs(p[m].mean() - y[m].mean())
    return e
def fig_E6():
    d = ld("e6"); c = pd.read_csv(f"{R}/e6_calib.csv"); fig, ax = plt.subplots(1, 3, figsize=(9.2, 2.7)); fig.subplots_adjust(wspace=.42)
    for col, lab_, cc in (("pi_bic", r"$\pi_j$ (BIC posterior)", "#d62728"), ("pi_boot", r"bootstrap frequency", "#1f77b4")):
        p, y = c[col].values, c.truth.values; nb = 10; b = np.minimum((p * nb).astype(int), nb - 1)
        xs = [p[b == k].mean() for k in range(nb) if (b == k).any()]; ys = [y[b == k].mean() for k in range(nb) if (b == k).any()]
        ax[0].plot(xs, ys, "o-", c=cc, ms=3, label=f"{lab_} ECE={ece(p, y):.3f}")
    ax[0].plot([0, 1], [0, 1], "k:"); ax[0].set_xlabel("predicted inclusion"); ax[0].set_ylabel("empirical frequency"); ax[0].legend(fontsize=6); ax[0].set_title("calibration (term inclusion)")
    ev = {"explosive (>100)": d.explosive.values > .5, "NRMSE$_{ext}$>1": d.nrmse_ext.values > 1, "wrong structure": d.srr_struct.fillna(0).values < .5}
    sc = {"1-posterior mass": 1 - d.mass.values, "entropy": d.entropy.values, "1-min $\\pi$ (sel.)": 1 - d.min_pi_sel.fillna(0).values, "1-min boot. freq.": 1 - d.min_freq_sel.fillna(0).values}
    ina = d.srr_struct.notna().values; w = .2
    for i, (sn, s) in enumerate(sc.items()):
        au = []
        for en, e in ev.items():
            m = ina if en == "wrong structure" else np.ones(len(d), bool)
            m = m & np.isfinite(s)
            au.append(roc_auc_score(e[m], s[m]) if 0 < e[m].sum() < m.sum() else np.nan)
        ax[1].bar(np.arange(3) + i * w, au, w, label=sn)
    ax[1].axhline(.5, c="k", ls=":"); ax[1].set_xticks(np.arange(3) + 1.5 * w); ax[1].set_xticklabels(["explosive\n(>100)", "NRMSE$_{ext}$>1", "wrong\nstructure"], fontsize=6.5); ax[1].set_ylim(.3, 1); ax[1].legend(fontsize=5.5, loc="lower right"); ax[1].set_title("AUROC of uncertainty scores")
    for n in sorted(d.noise.unique()):
        x = d[d.noise == n]; ax[2].scatter(x.mass, np.log10(np.maximum(x.nrmse_ext, 1e-8)), s=3, alpha=.4, label=f"{int(n*100)}%")
    ax[2].set_xlabel("posterior mass of selected structure"); ax[2].set_ylabel("log10 NRMSE$_{ext}$"); ax[2].legend(title="noise", fontsize=6); ax[2].set_title("mass vs extrapolation error")
    save(fig, "figE6_uncertainty")
    return {k: roc_auc_score(ev["explosive (>100)"], v) for k, v in sc.items() if np.isfinite(v).all()}

# ------------------------------------------------ E7
def fig_E7():
    d = ld("e7"); cf = ["none", "dim", "sym", "dim+sym"]; nz = sorted(d.noise.unique()); fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.4)); w = .8 / len(nz)
    for i, n in enumerate(nz):
        x = d[d.noise == n]; pos = np.arange(4) + i * w
        ax[0].bar(pos, [100 * x[x.config == c].srr_struct.mean() for c in cf], w, label=f"noise {int(n*100)}%")
        ax[1].bar(pos, [np.median(x[x.config == c].nrmse_ext) for c in cf], w); ax[2].bar(pos, [100 * x[x.config == c].explosive.mean() for c in cf], w)
    ax[0].set_title(r"SRR$_{struct}$ (%)"); ax[1].set_title("median NRMSE$_{ext}$"); ax[1].set_yscale("log"); ax[2].set_title("explosive share (%)")
    for a in ax: a.set_xticks(np.arange(4) + .4); a.set_xticklabels(cf, fontsize=7)
    ax[0].legend(fontsize=6); save(fig, "figE7_physics")

# ------------------------------------------------ E8
def fig_E8():
    d = ld("e8"); cur = list(dict.fromkeys(d.curve)); mods = list(dict.fromkeys(d.model)); fig, ax = plt.subplots(2, 4, figsize=(8, 3.8), sharex=True)
    cm = dict(zip(mods, ["#7f7f7f", "#1f77b4", "#2ca02c", "#d62728"]))
    for j, cname in enumerate(cur):
        x = d[d.curve == cname]
        for m in mods:
            g = x[x.model == m].groupby("noise")
            ax[0, j].plot(g.dist_med.median().index * 100, np.maximum(g.dist_med.median().values, 1e-4), "o-", c=cm[m], ms=3, label=m)
            if x[x.model == m].srr_struct.notna().any(): ax[1, j].plot(g.srr_struct.mean().index * 100, 100 * g.srr_struct.mean().values, "o-", c=cm[m], ms=3)
        ax[0, j].set_yscale("log"); ax[0, j].set_title(cname); ax[1, j].set_xlabel("noise on x,y (%)")
    ax[0, 0].set_ylabel("median dist. to true curve\n(full curve, incl. unseen arc)"); ax[1, 0].set_ylabel(r"SRR$_{struct}$ (%)"); ax[0, 0].legend(fontsize=5.5); save(fig, "figE8_implicit")

# ------------------------------------------------ heatmap helper
def heat(ax, M, rows, cols, title, fmt="{:.2f}", log=False, cmap="viridis_r"):
    im = ax.imshow(np.log10(np.maximum(M, 1e-8)) if log else M, aspect="auto", cmap=cmap)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=60, ha="right", fontsize=6.5); ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows, fontsize=6.5); ax.set_title(title); ax.grid(False)
    for i in range(len(rows)):
        for j in range(len(cols)):
            v = M[i, j]
            if np.isfinite(v): ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=5.5, color="w")
    return im

# ------------------------------------------------ E9
def fig_E9():
    d = ld("e9"); mods = list(dict.fromkeys(d.model)); eqs = list(dict.fromkeys(d.target)); s = .05; x = d[d.noise == s]
    fig, ax = plt.subplots(1, 3, figsize=(8.4, 3.0), gridspec_kw=dict(width_ratios=[1.5, 1, 1]))
    M = np.array([[np.median(x[(x.target == e) & (x.model == m)].nrmse_ext) if ((x.target == e) & (x.model == m)).any() else np.nan for m in mods] for e in eqs])
    heat(ax[0], M, [e[:22] for e in eqs], mods, "median NRMSE$_{ext}$ (log10), noise 5%", fmt="{:.1f}", log=True)
    il = d[d.in_lib == 1]; nz = sorted(d.noise.unique())
    for m in mods:
        y = il[il.model == m]
        if y.srr_struct.notna().any(): ax[1].plot(np.array(nz) * 100, [100 * y[y.noise == n].srr_struct.mean() for n in nz], "o-", ms=3, label=m)
        ax[2].plot(np.array(nz) * 100, [100 * d[(d.model == m) & (d.noise == n)].srr_sem.mean() for n in nz], "o-", ms=3)
    ax[1].set_title(r"SRR$_{struct}$ (%), in-library eqs."); ax[2].set_title(r"SRR$_{sem}$ (%), all eqs."); ax[1].set_xlabel("noise (%)"); ax[2].set_xlabel("noise (%)"); ax[1].legend(fontsize=5.5)
    save(fig, "figE9_feynman")

# ------------------------------------------------ E10/E11
def fig_dyn(name, title, ykey, tag):
    d = ld(name); d["mm"] = d.model + "/" + d.form.str[:1]; mods = ["SN-Exh/w", "SN-Greedy/w", "STLSQ/w", "STLSQ-ridge/w", "LASSO/w", "OMP/w", "SN-Exh/s", "STLSQ/s"]
    sysn = list(dict.fromkeys(d.system)); fig, ax = plt.subplots(1, 3, figsize=(8.4, 2.8)); x = d[d.noise == .05]
    heat(ax[0], np.array([[x[(x.system == s) & (x.mm == m)].support_f1.mean() if ((x.system == s) & (x.mm == m)).any() else np.nan for m in mods] for s in sysn]), sysn, mods, "support F1, noise 5%", cmap="viridis")
    heat(ax[1], np.array([[np.median(x[(x.system == s) & (x.mm == m)][ykey]) if ((x.system == s) & (x.mm == m)).any() else np.nan for m in mods] for s in sysn]), sysn, mods, f"median {tag}, noise 5% (log10)", fmt="{:.1f}", log=True)
    ax[1].set_yticks([]); fig.subplots_adjust(wspace=.35)
    nz = sorted(d.noise.unique())
    for m in mods:
        y = d[d.mm == m]; ax[2].plot(np.array(nz) * 100, [100 * y[y.noise == n].srr_struct.mean() for n in nz], "o-" if m.endswith("w") else "o--", ms=3, label=m)
    ax[2].set_title(r"exact support recovery (%)"); ax[2].set_xlabel("noise (%)"); ax[2].legend(fontsize=5, ncol=2)
    fig.suptitle(title, fontsize=8, y=1.02); save(fig, name.replace("e", "fig_E") if False else f"figE{name[1:]}_{'dynamics' if name=='e10' else 'pde'}")

# ------------------------------------------------ E12
def fig_E12():
    d = ld("e12"); mods = list(dict.fromkeys(d.model)); fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.6), gridspec_kw=dict(width_ratios=[1.6, 1])); w = .27
    for i, (k, lab_) in enumerate((("nrmse_interp", "in-distribution"), ("nrmse_ood_speed", "OOD speed 20-35 m/s"), ("nrmse_ood_slope", "OOD slope 5-8 deg"))):
        ax[0].bar(np.arange(len(mods)) + i * w, [np.median(d[d.model == m][k]) for m in mods], w, label=lab_)
    ax[0].set_yscale("log"); ax[0].set_xticks(np.arange(len(mods)) + w); ax[0].set_xticklabels(mods, rotation=40, ha="right", fontsize=6.5); ax[0].legend(fontsize=6); ax[0].set_title("median NRMSE (synthetic EV power)")
    s = d[d.support_f1.notna()]; sm = list(dict.fromkeys(s.model)); ax[1].bar(range(len(sm)), [s[s.model == m].support_f1.mean() for m in sm], color="#d62728"); ax[1].set_xticks(range(len(sm))); ax[1].set_xticklabels(sm, rotation=40, ha="right", fontsize=6.5); ax[1].set_title("support F1 vs true 6 terms")
    save(fig, "figE12_ev")

if __name__ == "__main__":
    fig_reported()
    for f in sys.argv[1:] or ["a1", "e4", "e5", "e6", "e7", "e8", "e9", "e10", "e11", "e12"]:
        if not os.path.exists(f"{R}/{f}.csv"): print("missing", f); continue
        {"a1": fig_A1, "e4": fig_E4, "e5": fig_E5, "e6": fig_E6, "e7": fig_E7, "e8": fig_E8, "e9": fig_E9, "e12": fig_E12,
         "e10": lambda: fig_dyn("e10", "E10: dynamical systems (/w weak form, /s strong form)", "traj_err", "trajectory error"),
         "e11": lambda: fig_dyn("e11", "E11: PDEs (/w weak form, /s strong form)", "forecast_err", "forecast error")}[f]()
