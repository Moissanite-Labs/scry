# Study 01: rare intents under repeated self-training (toy)

This is a toy. There is no neural network, no tokenizer and no shared representation. The "model" is a count table or a bigram model over 3-token messages, fit by counting. The study asks one narrow question about model collapse, the effect where a model trained on its own outputs slowly loses the rare cases. It runs in about 1.5 minutes on one CPU core with numpy only.

## Question

An agent language that is trained on its own outputs, generation after generation, should lose its rare intents first. Does mixing 10% real data into each generation stop that? Does the way the synthetic data is sampled change the answer?

## Design

- **Language.** Each message is 3 tokens: ACTION, OBJECT, QUALIFIER. There are 8 × 10 × 10 = 800 combinations. A random 400 of them are valid intents.
- **True usage.** Zipf(s = 1.1) over the 400 valid intents. The "rare intents" are the 40 rarest (10%). Together they hold 1.14% of the true mass.
- **Two generators.**
  - `table`: one probability per combination (800 in total). Only sampling error can cause collapse here.
  - `bigram`: P(A) P(O|A) P(Q|O). This adds approximation error. The model can emit invalid combinations and blend intents.
- **One generation.** Fit on M = 5000 messages, then sample 5000 messages from the fitted model.
- **Training mix.** The next generation trains on (1 − g) synthetic plus g real messages, with g = 0.0 or 0.1. The real messages come from one fixed pool of 5000, drawn once from the true distribution.
- **Sampling.** top-p = 0.95 (nucleus sampling, which drops the low-probability tail at each step, as production decoding does) or top-p = 1.0 (plain sampling with nothing dropped).
- 50 generations and 5 seeds. All numbers are means over the seeds.
- Generation 0 is already one fit-and-sample step on the real pool.
- **Reference.** An i.i.d. sample of 5000 from the true distribution contains about 30.4 of the 40 rare intents and about 366 of the 400 intents (computed analytically).

"Rare intents surviving" means: emitted at least once in that generation's 5000 samples.

## Results

Numbers are copied from `results/results_table.md` and `results/results_bigram.md`. Those files also hold `invalid_rate` and `tail_mass` for every cell.

### `table` generator

Rare intents surviving (of 40):

| top_p | real frac | gen 0 | gen 1 | gen 2 | gen 5 | gen 10 | gen 15 | gen 20 | gen 30 | gen 50 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1.0 | 0.0 | 21.2 | 17.8 | 16.6 | 11.4 | 8.6 | 7.4 | 5.6 | 3.2 | 2.4 |
| 1.0 | 0.1 | 21.2 | 17.6 | 16.6 | 15.8 | 13.6 | 12 | 10.8 | 11 | 10.8 |
| 0.95 | 0.0 | 5.6 | 1.6 | 0.4 | 0 | 0 | 0 | 0 | 0 | 0 |
| 0.95 | 0.1 | 5.6 | 1.6 | 0.2 | 0 | 0 | 0 | 0 | 0 | 0 |

Distinct valid intents emitted (of 400):

| top_p | real frac | gen 0 | gen 1 | gen 2 | gen 5 | gen 10 | gen 15 | gen 20 | gen 30 | gen 50 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1.0 | 0.0 | 325 | 296 | 272 | 223 | 183 | 153 | 136 | 109 | 79 |
| 1.0 | 0.1 | 325 | 297 | 277 | 247 | 240 | 240 | 235 | 241 | 239 |
| 0.95 | 0.0 | 214 | 137 | 93.2 | 37.8 | 14.6 | 8.4 | 6.2 | 5.8 | 5.6 |
| 0.95 | 0.1 | 214 | 141 | 103 | 55 | 29.8 | 22 | 17.6 | 15.6 | 14.4 |

KL(true ‖ model), in nats:

| top_p | real frac | gen 0 | gen 1 | gen 2 | gen 5 | gen 10 | gen 15 | gen 20 | gen 30 | gen 50 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1.0 | 0.0 | 0.127 | 0.26 | 0.378 | 0.692 | 1.02 | 1.28 | 1.5 | 1.91 | 2.48 |
| 1.0 | 0.1 | 0.127 | 0.262 | 0.361 | 0.505 | 0.562 | 0.595 | 0.612 | 0.581 | 0.583 |
| 0.95 | 0.0 | 0.127 | 0.626 | 1.05 | 2.11 | 3.44 | 4.32 | 5.09 | 5.17 | 5.27 |
| 0.95 | 0.1 | 0.127 | 0.548 | 0.816 | 1.19 | 1.45 | 1.58 | 1.63 | 1.6 | 1.71 |

### `bigram` generator

Rare intents surviving (of 40):

| top_p | real frac | gen 0 | gen 1 | gen 2 | gen 5 | gen 10 | gen 15 | gen 20 | gen 30 | gen 50 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1.0 | 0.0 | 29 | 26.4 | 25 | 25.6 | 22.6 | 18.4 | 18.4 | 16.4 | 11.8 |
| 1.0 | 0.1 | 29 | 27 | 23.8 | 24.8 | 24.8 | 23.8 | 24 | 25.4 | 26 |
| 0.95 | 0.0 | 17.8 | 13.8 | 10.8 | 7 | 4.8 | 3.4 | 3.2 | 2.8 | 2 |
| 0.95 | 0.1 | 17.8 | 13.8 | 11.8 | 8.6 | 7 | 6.6 | 6.2 | 6.2 | 5.2 |

Invalid message rate:

| top_p | real frac | gen 0 | gen 1 | gen 2 | gen 5 | gen 10 | gen 15 | gen 20 | gen 30 | gen 50 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1.0 | 0.0 | 0.254 | 0.255 | 0.258 | 0.253 | 0.247 | 0.249 | 0.257 | 0.249 | 0.234 |
| 1.0 | 0.1 | 0.254 | 0.256 | 0.258 | 0.255 | 0.256 | 0.258 | 0.255 | 0.255 | 0.257 |
| 0.95 | 0.0 | 0.224 | 0.204 | 0.198 | 0.182 | 0.156 | 0.153 | 0.153 | 0.134 | 0.136 |
| 0.95 | 0.1 | 0.224 | 0.21 | 0.203 | 0.189 | 0.183 | 0.178 | 0.168 | 0.161 | 0.161 |

## What it shows

- With no real data, rare intents die steadily and KL to the true distribution keeps rising.
- Under plain sampling (top-p 1.0), 10% real data stops the decline. The `table` model levels off at about 11 of 40 rare intents and about 240 of 400 intents, with KL near 0.58. That plateau is below generation 0 (21.2 and 325).
- Under top-p 0.95, the 10% real data does not save the rare intents in the `table` model. Generation 0 already keeps only 5.6 of 40, and none survive by generation 5 with or without real data. The real data adds about 0.1 × 1.1% of mass to the rare intents, and the nucleus cuts that mass again in every generation.
- So how the synthetic data is sampled matters as much as how much real data is mixed in.
- The `bigram` model keeps more rare intents (5.2 of 40 at generation 50 with top-p 0.95 and 10% real data). Part of that is an accident of its factorization: it spreads mass onto rare combinations (tail mass about 0.03 against a true 0.011), and 13–26% of its messages are invalid.

## What it does not show

- Anything about real language models. There is no neural network, no shared representation and no tokenizer.
- Survival is "emitted at least once in 5000 samples", so it depends on sample size.
- The real pool is fixed and never refreshed. That is optimistic in one way (the anchor never decays) and pessimistic in another (no new real data ever arrives).
- Only g = 0 and g = 0.1, and only M = 5000, were run. Other values may change the shape of the curves.

## Rerun

From this folder, with Python 3 and numpy (run here with numpy 2.5.4):

```
python3 toy.py --gens 50 --seeds 5 --out results/log_bigram.jsonl && python3 toy.py --gen_type table --gens 50 --seeds 5 --out results/log_table.jsonl && python3 summarize.py results/log_bigram.jsonl > results/results_bigram.md && python3 summarize.py results/log_table.jsonl > results/results_table.md
```

The run is seeded. A second run on the same machine reproduced all four results files byte for byte. `results/run_stdout.txt` is the console output of the original run.
