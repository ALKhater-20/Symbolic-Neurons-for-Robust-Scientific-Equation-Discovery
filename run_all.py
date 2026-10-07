import sys, time, os, pandas as pd
import ex42_core as core, ex42_mv as mv, ex42_dyn as dyn
os.makedirs("results", exist_ok=True)
JOBS = {
 "a1":  lambda: core.exp_A1(seeds=10),
 "e4":  lambda: core.exp_E4(seeds=6),
 "e5":  lambda: core.exp_E5(seeds=6),
 "e6":  lambda: core.exp_E6(seeds=8),
 "e7":  lambda: mv.exp_E7(seeds=15),
 "e8":  lambda: mv.exp_E8(seeds=20),
 "e10": lambda: dyn.exp_E10(seeds=8),
 "e11": lambda: dyn.exp_E11(seeds=5),
 "e12": lambda: mv.exp_E12(seeds=10),
 "e9":  lambda: mv.exp_E9(seeds=6, gp_seeds=2),
}
want = sys.argv[1:] or list(JOBS)
for k in want:
    f = f"results/{k}.csv"
    if os.path.exists(f): print("skip", k, flush=True); continue
    t = time.time(); rows = JOBS[k]()
    out = f if k != "e6" else f
    pd.DataFrame(rows).to_csv(f, index=False)
    if k == "e6" and os.path.exists("e6_calib.csv"): os.replace("e6_calib.csv", "results/e6_calib.csv")
    print(k, len(rows), f"{time.time()-t:.0f}s", flush=True)
