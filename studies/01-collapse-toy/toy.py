"""D4 pilot: model collapse of a toy agent language under repeated self-training.
Language: messages = (ACTION, OBJECT, QUALIFIER) token triples. 8x10x10 = 800 combos, of which 400 are
"valid intents" (random 50%). True intent frequencies are Zipf(s=1.1) over the 400 valid intents.
Generator: bigram LM over tokens  P(A|BOS) P(O|A) P(Q|O) P(EOS|Q), MLE with tiny add-k smoothing.
  (It can generalize = emit unseen combos; combos outside the valid set count as 'invalid' = propagation errors.)
Each generation: train on M messages = (1-g)*M sampled from the previous generation's model (top-p sampling)
+ g*M sampled from a fixed pool of real data. Gen 0 trains on real data only.
Tracked per generation: survival of the rarest intents, distinct valid intents emitted, invalid rate, KL(true||model).
"""
import numpy as np, json, sys, argparse
p = argparse.ArgumentParser()
p.add_argument("--gens", type=int, default=20); p.add_argument("--M", type=int, default=5000)
p.add_argument("--seeds", type=int, default=5); p.add_argument("--topp", type=float, default=0.95)
p.add_argument("--k", type=float, default=1e-3); p.add_argument("--gen_type", default="bigram", choices=["bigram","table"]); p.add_argument("--out", default="toy_log.jsonl")
a = p.parse_args()
NA, NO, NQ = 8, 10, 10

def run(seed, gamma, topp):
    rng = np.random.default_rng(seed)
    combos = np.array([(i, j, k) for i in range(NA) for j in range(NO) for k in range(NQ)])
    valid_idx = rng.choice(len(combos), 400, replace=False)
    valid = combos[valid_idx]
    ranks = rng.permutation(400) + 1
    ptrue = ranks ** -1.1; ptrue /= ptrue.sum()
    validset = {tuple(v) for v in valid}
    tail = set(np.argsort(ptrue)[:40])        # 40 rarest intents (10%)
    real_pool = valid[rng.choice(400, size=a.M, p=ptrue)]   # fixed real data (what a human corpus would be)

    def fit(data):
        if a.gen_type == "table":
            c = np.full(NA * NO * NQ, a.k); np.add.at(c, data[:, 0] * NO * NQ + data[:, 1] * NQ + data[:, 2], 1)
            return (c / c.sum(),)
        cA = np.bincount(data[:, 0], minlength=NA) + a.k
        cO = np.full((NA, NO), a.k); np.add.at(cO, (data[:, 0], data[:, 1]), 1)
        cQ = np.full((NO, NQ), a.k); np.add.at(cQ, (data[:, 1], data[:, 2]), 1)
        return cA / cA.sum(), cO / cO.sum(1, keepdims=True), cQ / cQ.sum(1, keepdims=True)

    def trunc(pr):  # top-p truncation per step (like decoding with nucleus sampling)
        if topp >= 1: return pr
        o = np.argsort(-pr); c = np.cumsum(pr[o]); keep = o[: np.searchsorted(c, topp) + 1]
        q = np.zeros_like(pr); q[keep] = pr[keep]; return q / q.sum()

    def sample(model, n):
        if a.gen_type == "table":
            idx = rng.choice(NA * NO * NQ, n, p=trunc(model[0])); return combos[idx]
        pA, pO, pQ = model
        pA = trunc(pA); pO = np.array([trunc(r) for r in pO]); pQ = np.array([trunc(r) for r in pQ])
        A = rng.choice(NA, n, p=pA)
        O = np.array([rng.choice(NO, p=pO[x]) for x in A])
        Q = np.array([rng.choice(NQ, p=pQ[x]) for x in O])
        return np.stack([A, O, Q], 1)

    def model_prob(model, v):
        if a.gen_type == "table": return model[0][v[0] * NO * NQ + v[1] * NQ + v[2]]
        pA, pO, pQ = model; return pA[v[0]] * pO[v[0], v[1]] * pQ[v[1], v[2]]

    model = fit(real_pool); rows = []
    for g in range(a.gens + 1):
        gen = sample(model, a.M)            # what this generation's model emits
        emitted = {tuple(x) for x in gen}
        val_mask = np.array([tuple(x) in validset for x in gen])
        surv_tail = sum(tuple(valid[i]) in emitted for i in tail)
        pm = np.array([model_prob(model, v) for v in valid])
        kl = float(np.sum(ptrue * np.log(ptrue / np.maximum(pm, 1e-12))))
        rows.append(dict(gen_type=a.gen_type, seed=seed, gamma=gamma, topp=topp, gen=g, tail_survive=surv_tail, tail_n=len(tail),
                         distinct_valid=len(emitted & validset), invalid_rate=round(1 - val_mask.mean(), 4),
                         tail_mass=float(pm[list(tail)].sum()), true_tail_mass=float(ptrue[list(tail)].sum()), kl=round(kl, 4)))
        n_real = int(round(gamma * a.M))
        train = np.concatenate([gen[: a.M - n_real], real_pool[rng.choice(a.M, n_real, replace=False)]]) if n_real else gen
        model = fit(train)
    return rows

with open(a.out, "w") as f:
    for topp in sorted({a.topp, 1.0}, reverse=True):
        for gamma in [0.0, 0.1]:
            for s in range(a.seeds):
                for r in run(s, gamma, topp): f.write(json.dumps(r) + "\n")
print("wrote", a.out)
