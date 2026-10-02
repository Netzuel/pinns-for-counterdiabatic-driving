# Repository map

| Location | Role | Used by |
| --- | --- | --- |
| `src/quantum_pinn/model.py` | `QuantumPINN`: network and `predict` | training, inference |
| `src/quantum_pinn/physics.py` | Pauli strings, loss terms and weights, exact gauge potential, propagation | model, training, analysis, tests |
| `src/quantum_pinn/training.py` | data access, `RunConfig`, training loop, checkpoints, `quantum-pinn` command | users, tutorials |
| `src/quantum_pinn/analysis.py` | stored trajectories, dynamics, Table 1, physics checks | figures, tutorials, tests |
| `src/figures.py` | all result figures and `evaluation.json` | `python src/figures.py` |
| `src/diagrams/` | TikZ architecture diagram and renderer | `bash src/diagrams/render.sh` |
| `src/verify.py` | privacy audit and data checksums | `python src/verify.py` |
| `datasets/hamiltonians.h5` | H₂ and LiH Hamiltonians | training, analysis |
| `datasets/pretrained/` | six trained models | `load_pretrained` |
| `datasets/results/` | stored H₂ trajectories, loss history, scalability table | analysis, figures |
| `datasets/experiments.json` | 52 named training configurations | `quantum-pinn train --experiment` |
| `datasets/figures/` | light and dark SVG figures, `evaluation.json` | README, results guide |
| `datasets/runs/` | new training runs, ignored by Git | users |
| `docs/` | method, installation, quickstart, training, inference, results, data | README |
| `docs/examples/` | three Jupyter tutorials | users, tests |
| `tests/` | tests and fixtures | `python -m pytest` |
| `summary/` | verification guide, this map, record of removed material | maintainers |

`pip install .` installs the package and copies `datasets/` into
`quantum_pinn/data`, so the command line and tutorials also work outside the
checkout. Without installing, it is enough to put `src/` on `PYTHONPATH`, and
the code then reads `datasets/` directly.
