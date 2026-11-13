import pandas as pd, numpy as np, time
import ex42_core as core
from ex42_core import T
d = pd.read_csv("results/e4.csv")
d = d[~((d.model == "SN-Greedy") & (d.kmax.astype(str) == "inf"))]
rows = []
for fid in T:
    for s in (0.01, .05, .20):
        for sd in range(6):
            P = core.make_problem(fid, s, sd); D = core.design(P)
            t0 = time.perf_counter(); idx, cand = core.ex41.sn_greedy(D.Z, D.yz, D.Z.shape[1])[:2]; el = time.perf_counter() - t0
            r = core.evaluate(P, D, idx); r.update(exp="E4", function=fid, family=P.fam, noise=s, seed=sd, model="SN-Greedy", kmax="inf", time=el, candidates=cand); rows.append(r)
            if s == .05 and 2 <= sd < 6:                       # extend exact K=8 to seeds 2-5 at 5% noise
                t0 = time.perf_counter(); idx, cand = core.best_subset(D.gr, 8); el = time.perf_counter() - t0
                r = core.evaluate(P, D, idx); r.update(exp="E4", function=fid, family=P.fam, noise=s, seed=sd, model="SN-Exh", kmax=8, time=el, candidates=cand); rows.append(r)
pd.concat([d, pd.DataFrame(rows)], ignore_index=True).to_csv("results/e4.csv", index=False); print("done", len(rows))
