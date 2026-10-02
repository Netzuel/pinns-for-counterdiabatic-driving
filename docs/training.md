# Training

A configuration and an output directory are all that training requires. Below
are the bundled configurations, what each one costs to run, and the way to add
your own.

## Named configurations

There are 52 named configurations in
[`datasets/experiments.json`](../datasets/experiments.json). Any field a
configuration leaves out falls back to the `RunConfig` default: width 30,
learning rate $`10^{-5}`$, seed 7, single precision, and CPU.

| Name | Settings | Purpose |
| --- | --- | --- |
| `smoke` | 3 × 16 hidden units, 32 points, 30 updates, learning rate $`10^{-4}`$ | workflow check in about a second |
| `paper/h2-d1.0` | 6 × 30, 2048 points, 500000 updates, `paper` weights, snapshots at 15000, 25000, 75000 | the article's main H₂ run, including the stages of its Figure 3 |
| `retained/h2-d{1.0,1.5,2.0,2.5}` | 7 × 30, 2048 points, `retained` weights, 500001 or 300001 updates | the settings that produced the bundled trajectories |
| `scalability/h2-d{1.0…3.0}-n{00128…16384}` | 7 × 30, $`2^7`$–$`2^{14}`$ points, 500000 updates | H₂ part of the article's Figure 8 (40 runs) |
| `scalability/lih-d1.0-n{00128…04096}` | 7 × 30, $`2^7`$–$`2^{12}`$ points, 500000 updates | LiH part of Figure 8 (6 runs) |

```bash
quantum-pinn train --experiment paper/h2-d1.0 --output datasets/runs/h2-paper
```

Command-line options take precedence over the configuration. They are
`--system`, `--distance`, `--steps`, `--collocation`, `--seed`, `--weights`,
`--device`, and `--snapshot-steps`. With `--smoke`, any configuration shrinks
to three updates on 64 points, a quick way to test it before committing to a
long run:

```bash
quantum-pinn train --experiment scalability/lih-d1.0-n02048 --smoke --output datasets/runs/lih-test
```

## Cost

In every update the network and its time derivative are evaluated at all
collocation points, and gradients flow back through both. Cost therefore rises
with the number of points and with the $`4^n`$ size of the output. On the
original hardware, the stored H₂ runs of 300001 updates took 2.3–8.4 h and
the LiH runs took 33–71 h ([results](results.md#scalability-article-figure-8)).
With PyTorch 2.11 on an Apple M5 CPU, `paper/h2-d1.0` reached about 80 updates
per second, or roughly 1.7 h for 500000 updates, while
`scalability/lih-d1.0-n02048` managed about 7 updates per second, roughly 20 h.
Before a long run, time a short one on your own machine with `--steps 500`.

Progress goes to standard error after every 5% of the updates. If a run is
interrupted, no `model.pt` is written, but any snapshots saved up to that point
are kept.

## Custom settings

```python
from quantum_pinn import RunConfig, train

config = RunConfig(
    system="lih",            # "h2" (2 qubits) or "lih" (4 qubits)
    distance=1.5,            # bundled: 0.5 to 3.5 angstrom in steps of 0.5
    width=30,
    hidden_layers=6,
    collocation=1024,
    steps=50_000,
    learning_rate=1e-5,
    weights="paper",         # or "retained"
    seed=3,
    dtype="float64",
    device="cpu",
    snapshot_steps=(10_000, 25_000),
)
model, summary = train(config, "datasets/runs/lih-d1.5")
```

Loss weights other than the two presets need your own loop that calls
`quantum_pinn.loss_terms` with a `LossWeights` instance.
[`training.py`](../src/quantum_pinn/training.py) is a short loop you can start
from.

Other Hamiltonians work in the same way. Create a `QuantumPINN` with the right
number of qubits and give `loss_terms` any pair of Hermitian $`2^n\times2^n`$
matrices.

## Reproducibility

Both the weight initialisation and the scrambled Sobol sample are set by the
seed. Repeating a configuration on the same machine with the same library
versions reproduces the weights exactly, and a test checks this. On other
hardware, other PyTorch versions, or with a different thread count, results may
change.
