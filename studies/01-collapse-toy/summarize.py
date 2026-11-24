import json, sys, collections, numpy as np
rows = [json.loads(l) for l in open(sys.argv[1])]
gens = sorted({r["gen"] for r in rows}); show = [g for g in gens if g in (0, 1, 2, 5, 10, 15, 20, 30, 50)]
grp = collections.defaultdict(list)
for r in rows: grp[(r["topp"], r["gamma"], r["gen"])].append(r)
for metric in ["tail_survive", "distinct_valid", "invalid_rate", "tail_mass", "kl"]:
    print(f"\n{metric} (mean over seeds; tail_n={rows[0]['tail_n']}, true_tail_mass={rows[0]['true_tail_mass']:.4f})")
    print("| top_p | real frac | " + " | ".join(f"gen {g}" for g in show) + " |"); print("|---" * (len(show) + 2) + "|")
    for topp in sorted({r["topp"] for r in rows}, reverse=True):
        for gamma in (0.0, 0.1):
            vals = [np.mean([r[metric] for r in grp[(topp, gamma, g)]]) for g in show]
            print(f"| {topp} | {gamma} | " + " | ".join(f"{v:.3g}" for v in vals) + " |")
