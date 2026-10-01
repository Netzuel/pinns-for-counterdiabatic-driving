# Inference and physical checks

## Bundled models

| Name | System | Bond distance (Å) | Network | Training |
| --- | --- | ---: | --- | --- |
| `h2-d1.0`, `h2-d1.5`, `h2-d2.0`, `h2-d2.5`, `h2-d3.0` | H$_2$, 2 qubits | 1.0–3.0 | 7 × 30 tanh | 2048 points, `retained` weights |
| `lih-d1.0` | LiH, 4 qubits | 1.0 | 7 × 30 tanh | 2048 points, `retained` weights |

```python
import torch
from quantum_pinn import load_pretrained, evaluate

model, config = load_pretrained("h2-d1.5")
out = model.predict(torch.linspace(0, 1, 101).reshape(-1, 1))
data = evaluate(model, config.system, config.distance, points=1001)   # NumPy arrays
```

The output of `predict` is a set of PyTorch tensors: `time`, `schedule`, and
`schedule_rate`, each $(N,1)$; the complex `potential`, $(N,d,d)$; and
`coefficients`, $(N,4^n)$. With `create_graph=True` the autograd graph is kept,
which you need, for instance, to differentiate some quantity with respect to
the weights. `evaluate` also returns the total `hamiltonian`, $(N,d,d)$, and
gives everything as flattened NumPy arrays on a uniform grid.

On the command line, `quantum-pinn evaluate NAME_OR_PATH` reports the schedule
at its endpoints together with the endpoint errors of $H$.

## Checks worth running

A low training loss is required, but on its own it proves little. Check a
trained model on a grid that does not share points with the collocation set:

1. Endpoints: check that $\lambda(0)\approx0$, $\lambda(1)\approx1$,
   $H(0)\approx H_{\mathrm{initial}}$, and $H(1)\approx H_{\mathrm{final}}$.
2. Hermiticity: since the matrix $A_{\mathrm{CD}}$ is Hermitian only
   approximately, report $\max|A_{\mathrm{CD}}-A_{\mathrm{CD}}^\dagger|$. By
   construction, the Pauli reconstruction $\sum_\mu C_\mu P_\mu$ is exactly
   Hermitian.
3. Gauge potential: compare against
   `quantum_pinn.physics.exact_gauge_potential(schedule, h_initial, h_final)`,
   bearing in mind that the loss leaves terms that commute with
   $H_{\mathrm{AD}}$ undetermined.
4. Dynamics: use `quantum_pinn.physics.propagate` to evolve the ground state of
   $H_{\mathrm{initial}}$, then measure the population of the instantaneous
   ground state with `instantaneous_populations`. If $H$ is not exactly
   Hermitian, divide by the norm of the state.

For a trajectory dictionary, these checks are implemented in
`quantum_pinn.analysis.physics_checks` and `quantum_pinn.analysis.dynamics`,
and the [inference tutorial](examples/inference.ipynb) goes through them step
by step.

## Stored trajectories

```python
from quantum_pinn.analysis import load_trajectory, load_history, load_scalability

trajectory = load_trajectory(2.0)   # 1.0, 1.5, 2.0 or 2.5
history = load_history()            # per-update losses of the 1.0 angstrom run
rows = load_scalability()           # 46 rows of scalability.csv
```

These files are the training outputs from the article, and every figure in
[results](results.md) is drawn from them. Their provenance is documented in
[reference data](reference-data.md).
