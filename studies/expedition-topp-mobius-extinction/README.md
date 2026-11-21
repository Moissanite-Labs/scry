# Expedition Study: Exact Nonlinear Top-$p$ Support-Truncation Extinction, 1D $PGL(2, \mathbb{Q})$ Möbius Collapse & Hysteresis Band Theorem

- **Target:** `Moissanite-Labs/scry` PR `#1` (`study/01-collapse-toy`, head commit `790fa32d`, `studies/01-collapse-toy/toy.py`, `SHA-256: f1dffb494726e538b42254753bf69b6ecdc4cb4badddf5cfbde2856ae3b1b27b`).
- **Status:** `analysis-verified` (independent adversarial verification audit `AUDIT-032`: `VERIFIED`, confirmed `N2`; 100% pure Python 3 stdlib, zero external dependencies, zero bare `assert`s, bit-identical across all 4 isolated `-I` interpreter modes).
- **Standalone Data Bundle:** Read-only copies of `toy.py`, `results_table.md`, and `results_bigram.md` from PR `#1` (~`10 KB` total) are bundled under `data/scry_pr1/` so this study runs standalone prior to PR `#1` merging.
- **Disclosure:** Scripts, exact rational theorems, and verification runs were produced and independently audited with AI assistance as part of the Foundation-0 / Moissanite-Labs cross-repo verification sprint.

---

## 1. Exact Algebraic Theorems (100% Pure `fractions.Fraction`, Zero Floats)

Consider iterative synthetic self-training with top-$p$ nucleus truncation $\mathcal{N}_p$ and real-data mixing fraction $\alpha \in (0, 1]$ on a strictly decreasing base distribution $q_1 > q_2 > \dots > q_V > 0$ with prefix masses $Q(K) = \sum_{i=1}^K q_i$:
$$P_{t+1} = F_{p,\alpha}(P_t) := (1 - \alpha)\,\mathcal{N}_p(P_t) + \alpha\,q.$$

### Theorem E1 (Exact Hysteresis Band $[K_{\min}, K_{\max}]$) & Corollary E1.1 (Exact 1D $PGL(2, \mathbb{Q})$ Möbius Collapse)
Define the exact rational 1D Möbius transformation
$$\mu_{p,\alpha}(x) := \frac{p\,x}{1 - \alpha(1 - x)}, \qquad x^* = 1 - \frac{1 - p}{\alpha}.$$

1. **Exact Fixed-Point Characterization:** A prefix $S_K = \{1, \dots, K\}$ supports an exact fixed point $P^*_K$ of $F_{p,\alpha}$ **if and only if**
   $$Q(K - 1) < \mu_{p,\alpha}\bigl(Q(K)\bigr) \le Q(K),$$
   equivalently **(C1)** $Q(K) \ge 1 - \frac{1 - p}{\alpha}$ and **(C2)** $\frac{1 - \alpha(1 - Q(K))}{Q(K)}\,Q(K - 1) < p$.
2. **Closed-Form Fixed Point & Divergence:** Every $K \in [K_{\min}, K_{\max}]$ is an exact absorbing fixed point with
   $$P^*_K(i) = \begin{cases} \dfrac{1 - \alpha(1 - Q(K))}{Q(K)}\,q_i & i \le K, \\[6pt] \alpha\,q_i & i > K, \end{cases}$$
   and exact closed-form divergence $D_{\mathrm{KL}}(q \| P^*_K) = Q(K)\ln\frac{Q(K)}{1 - \alpha(1 - Q(K))} + (1 - Q(K))\ln\frac{1}{\alpha}$. Both boundary controls $K_{\min} - 1$ and $K_{\max} + 1$ fail to be fixed points identically over $\mathbb{Q}$.
3. **Exact 1D Möbius Reduction (Corollary E1.1):** Starting from $P_0 = q$, whenever $K_t \le K_{t-1}$ every state $i \le K_t$ shares the exact same scalar prefactor multiplying $q_i$, which cancels identically inside $\mathcal{N}_p(P_t)$. Consequently, the entire $V$-dimensional nonlinear map collapses **with exact zero error** onto the monotone 1D Möbius recurrence
   $$K_0 = \min\{k : Q(k) \ge p\}, \qquad K_{t+1} = \min\bigl\{k \ge 1 : Q(k) \ge \mu_{p,\alpha}(Q(K_t))\bigr\},$$
   which arrests at $K_{\max}$ in finite steps.

### Theorem E2 (Four-Phase Rare-Intent Extinction Diagram) & Minority-Portal Floor Cure
For a rare tail $T = \{V - r + 1, \dots, V\}$ of true mass $T_0 = 1 - Q(V - r)$, exact rational thresholds $0 < \alpha_{\mathrm{ext}} < \alpha_{\min} = \frac{1 - p}{T_0}$ partition $(p, \alpha)$ into four exact phases:
1. **Unconditional Extinction ($1 - p \ge T_0$):** Clipped at gen 0 for all $\alpha$; model tail mass locks to $\alpha T_0$ (plus add-$k$ smoothing floor).
2. **Cascade-Induced Extinction ($1 - p < T_0,\ \alpha < \alpha_{\mathrm{ext}}$):** Tail **survives gen 0**, yet the Möbius cascade drives emitted tail support to exact `0` in finite generations.
3. **Metastable Hysteresis Window ($\alpha_{\mathrm{ext}} < \alpha < \alpha_{\min}$):** Deterministic cascade arrests at $K_{\max} > V - r$ (tail alive), yet $K_{\min} \le V - r$ creates an irreversible finite-sample tunneling extinction trap (**Anti-Pattern `CS-09`**).
4. **Tunnel-Proof Permanent Survival ($\alpha > \alpha_{\min} = (1 - p)/T_0$):** Even $K_{\min} > V - r$, guaranteeing survival against both deterministic cascade and finite-sample downward tunneling.

**Minority-Portal Floor Cure:** Mixing a non-truncated protected floor $\eta = 1/50$ into the emitter yields an exact zero-residual closed-form rational fixed point (`K = 6` on $\mathbb{Q}^{12}$, exact tail mass $(1 - \alpha)\eta + \alpha T_0 \approx 0.0205$ vs unprotected $\alpha T_0 \approx 0.0025$) that prevents extinction even inside Phase 1 ($p = 0.95,\ \alpha = 0.10$).

---

## 2. Live Anchor Results on `scry` PR `#1` (`Zipf(1.1)`, $V = 400$, $T_0 = 0.011408$)

| Quantity | Exact Formula / Prediction | Live `scry#1` (`results_table.md` / `results_bigram.md`) | Independent Pure-Stdlib MC (`M = 5000`) |
|---|---:|---:|---:|
| **`p = 0.95, alpha = 0.10` hysteresis band `[K_min, K_max]`** | **`[9, 19]`** | **`distinct_valid` gen 50 = `14.4`** (inside `[9, 19]`) | gen 30 = **`16.33`** (`tail_g5 = 0.0`) |
| **`p = 0.95, alpha = 0.10` tail-mass floor** | `alpha * T_0 = 0.001141` | `0.00101` (gen 50) | — |
| **`p = 0.95, alpha = 0.00` add-$k$ smoothing floor** | `40 * 1e-3 / 5000.8 = 7.999e-6` | `8e-06` (gen 50) | — |
| **`p = 0.995` exact phase thresholds** | **`alpha_ext = 0.422839`**, **`alpha_min = 0.438283`** (`[360, 361]` metastable at `alpha = 0.43`) | — | — |
| **`p = 0.995` deterministic cascade (exact `Fraction`)** | `alpha=0.1`: `22 -> 7 -> 0` (gen 2); `alpha=0.3`: `22 -> 10 -> 2 -> 0` (gen 3); `alpha=0.5`: `6` permanent | — | `M = 5000`: `1.67` (`alpha=0.1`) vs `8.0` (`alpha=0.5`) — see finite-$M$ nuance below |
| **Product-quotient `bigram` vs `table` (`p = 0.95, alpha = 0.10`)** | Factorized `8x10x10` marginals lift tail above `top-p` at cost of quotient leakage | `bigram`: `5.2` tail alive, **`16.1%` invalid** vs `table`: `0.0` tail | — |

### Honest Finite-$M$ Count-Discretization Nuance (`CS-09`)
At $p = 0.995$ and finite $M = 5000$, only $(1 - p)M = 25$ tokens of empirical mass are trimmed per generation while $\alpha M = 500$ fresh real draws inject rare Poisson count-$\ge 2$ spikes. Those discrete count spikes transiently rescue ~`1–3` tail tokens at $\alpha = 0.10$ (`1.67` vs `8.0` at $\alpha = 0.50$, and `2.67` vs `9.33` on independent unseen seeds in `AUDIT-032`) even while total support collapses below $K_{\min} = 255$ (`192.3`) and the exact $M \to \infty$ deterministic cascade hits `0/40` by generation 2. Both regimes are reported side by side without conflation.

---

## 3. How to Rerun (4-Mode Isolated R4 Parity)

From the repository root (pure Python 3 stdlib, zero external dependencies, ~`1.9s` per run):

```bash
for flags in "-I -B" "-I -O -B" "-I -B -W error" "-I -O -B -W error"; do
  python3 $flags studies/expedition-topp-mobius-extinction/probe_s4_scry_topp_extinction.py
done
```

Expected deterministic artifact SHA-256 digest:
- `probe_s4_scry_topp_extinction_results.json`: `672afa5a391d39c484113e28f384ed04b31f92a4899a4f8b1356047c3f1ee51a`
- `probe_s4_scry_topp_extinction.py`: `9bf27796def2d419b1d05489d13c3e7e61608b50b57f71a59b716e19b57f0fc4`

---

## 4. Limits

- **Unigram / strictly decreasing assumption:** Theorem E1 and Corollary E1.1 apply exactly to unigram/tabular top-$p$ truncation on strictly decreasing base distributions $q$. Factorized/autoregressive generators (such as PR `#1`'s `bigram` model) couple marginal truncation across positions, trading tail survival (`5.2`) against invalid-combination leakage (`16.1%`).
- **Finite-$M$ tunneling:** Exact deterministic dynamics arrest at $K_{\max}$; downward tunneling into $[K_{\min}, K_{\max}]$ and discrete Poisson count-$\ge 2$ rescue near $(1 - p)M = O(10)$ are finite-sample phenomena captured by our MC replay rather than by the $M \to \infty$ map alone.
