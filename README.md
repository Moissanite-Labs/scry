# scry

Research on how models are trained and how they think.

In the Inheritance Cycle, scrying lets a magician project a concept from their mind straight onto a surface. This repo works toward the same thing for models: letting a model express its intent directly from its internal state, instead of first squeezing it through human language.

## Why

Models learn to reason in the human languages they are trained on, mostly English. We don't know how much that choice limits them. The other Moissanite Labs repos change how agents talk to each other. This one asks whether changing the language a model learns in, or giving it a denser one, changes how well it reasons.

## What lives here

- Experiments that train small models without English, or without any single human language.
- Tests of linguistic relativity, the idea that the language a model learns shapes how well it reasons. Results are measured with reasoning benchmarks at equal compute.
- Reasoning in a dense symbolic form, possibly with new characters where one character stands for a whole idea.
- Post-training small local models that run on a phone or in a browser to act as the harness translator, built in step with tokenpage.
- Reproducible runs, with the model, data, compute, date, accuracy and cost of each one.

See [CHARTER.md](CHARTER.md) for scope, questions and success criteria.

## Related

Each repo covers one layer. Work goes where its layer lives.

- [abios](https://github.com/Moissanite-Labs/abios): the text protocol agents use with hosted models.
- [exyber](https://github.com/Moissanite-Labs/exyber): transport, meaning how messages move between devices.
- [tokenpage](https://github.com/Moissanite-Labs/tokenpage): tokenizers, symbols and language. This includes cross-language density and the symbolic agent language with its harness translator.
- [nullmodem](https://github.com/Moissanite-Labs/nullmodem): exchange between agents below text, such as KV caches, embeddings and shared memory, on local models the harness controls. Inter-agent only, no training.
- scry (this repo): model training and cognition.

## License

MIT
