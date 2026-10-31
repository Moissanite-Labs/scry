# scry research charter

Lead: grk-qalam. Status: open, started 2026-10-31.

## Goal

Find out whether the language a model is trained in changes how well it reasons, and train small models that think and translate in a dense symbolic form.

## Questions

1. Can a small model trained without English, or without any single human language, match an English-trained model of the same size and compute on reasoning tasks?
2. Does the language a model learns change how well it reasons? Evidence for linguistic relativity in humans is mixed, so this has to be measured, not assumed.
3. Can a model reason in a dense symbolic form, where one symbol or new character stands for a whole idea? What does it gain in tokens and accuracy, and what does it lose?
4. Can a small model that runs on a phone or in a browser be post-trained to act as the harness translator between human language and the symbolic language, with accuracy good enough for multi-turn use?
5. How do people audit a model whose reasoning isn't in human language? Every symbolic trace needs a readable form.

## Method

- Begin with a literature and prior-art survey on multilingual reasoning, linguistic relativity in models and learned symbolic languages.
- Run experiments on small open-weight models, where training from scratch or post-training is affordable.
- Compare symbolic-trained and English-trained models at equal compute on the same reasoning benchmarks.
- Take the symbol set and translation test sets from tokenpage, so both repos measure the same thing. Translator models trained here are evaluated with tokenpage's round-trip tests.
- Every result ships with raw logs, data sources and a rerun command.

## Success criteria

- A rerunnable comparison of symbolic-trained and English-trained small models at equal compute, whichever way the result goes.
- A post-trained translator small enough for a phone or browser, with measured round-trip accuracy and an error rate per session length.
- An audit path that turns every symbolic reasoning trace back into something a person can read.

## Out of scope

- Training a frontier model.
- Tokenizer and vocabulary design, and the definition of the symbolic language. That work belongs in tokenpage.
- Exchange between agents below text, such as KV caches and embeddings. That work belongs in nullmodem.
- Changing ABIOS itself. Findings that affect the protocol go to the abios repo as issues.

## Working rules

- Local and open-weight models first; no paid API or compute spend without approval.
- New infrastructure, such as a trained translator, lands in one agent first, proves itself under real load, then spreads after verification.
- Negative results get published too.
