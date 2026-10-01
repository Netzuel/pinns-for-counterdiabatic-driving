# Quickstart

Once [installation](installation.md) is done, the steps below train a small
model, load it back, and evaluate one of the bundled models. On a CPU the whole
sequence needs only a few seconds.

## Command line

```bash
quantum-pinn train --experiment smoke --output datasets/runs/example
quantum-pinn evaluate datasets/runs/example/model.pt
```

The `smoke` configuration has three hidden layers of 16 units and runs 30 Adam
updates on 32 collocation points. Inside the run directory you will find

| File | Contents |
| --- | --- |
| `config.json` | the complete `RunConfig` |
| `history.npz` | every loss term at every update and the training time in seconds |
| `model.pt` | configuration and weights, loadable with `weights_only=True` |

Evaluation prints $\lambda$ at both endpoints and the largest deviation of
$H(0)$ and $H(1)$ from their target Hamiltonians. With only 30 updates the
values are nowhere near converged, so this run tests the workflow and nothing
more. Give every run its own output directory; Git ignores `datasets/runs/`.

Evaluate the model trained for the article:

```bash
quantum-pinn evaluate h2-d1.0
```

```json
{
  "system": "h2",
  "distance": 1.0,
  "schedule_start": 6.9e-09,
  "schedule_end": 1.0,
  "initial_hamiltonian_error": 2.9e-05,
  "final_hamiltonian_error": 0.0
}
```

(Rounded values.) To see every named configuration and bundled model, run
`quantum-pinn list`.

## Python

```python
import torch
from quantum_pinn import RunConfig, load_checkpoint, load_pretrained, train

config = RunConfig(system="h2", distance=1.0, collocation=256, steps=3000, learning_rate=1e-3)
model, summary = train(config, "datasets/runs/python-example")
print(summary)

model, config = load_checkpoint("datasets/runs/python-example/model.pt")
t = torch.linspace(0, 1, 11).reshape(-1, 1)
out = model.predict(t)
print(out["schedule"].ravel(), out["potential"].shape)   # (11, 4, 4)

pretrained, pretrained_config = load_pretrained("h2-d1.0")
```

Where to go from here: the [Jupyter tutorials](examples/README.md), the
[training guide](training.md) for full runs, and the
[inference guide](inference.md) for physical checks.
