# Installation

The package requires Python 3.10 or newer. Tests run on Python 3.10 against the
versions pinned in [`requirements.txt`](../requirements.txt) and
[`pyproject.toml`](../pyproject.toml): PyTorch 2.11.0, NumPy 2.2.5, SciPy
1.15.3, h5py 3.16.0, plus Matplotlib 3.10.8 for the figures.

From the repository root, create an environment and install:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

You get the `quantum_pinn` package, the `quantum-pinn` command, and a copy of
the bundled Hamiltonians, models, and results. To check the installation, run

```bash
quantum-pinn list
quantum-pinn evaluate h2-d1.0
```

On Linux the default PyTorch wheel ships with CUDA. If you want a smaller
CPU-only install, take PyTorch from its CPU index before anything else:

```bash
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

## Optional extras

| Extra | Installs | Needed for |
| --- | --- | --- |
| `figures` | Matplotlib | `python src/figures.py` |
| `notebooks` | JupyterLab, ipykernel, Matplotlib | the [tutorials](examples/README.md) |
| `test` | pytest, nbclient, nbformat, ipykernel, Matplotlib | `python -m pytest` |

```bash
python -m pip install '.[test]'
```

To rebuild the architecture diagram you also need
[Tectonic](https://tectonic-typesetting.github.io/) and Poppler's `pdftocairo`
on `PATH`; [`src/diagrams/render.sh`](../src/diagrams/render.sh) shows how they
are used.

## Devices

By default, training uses the CPU. For a GPU, set `device` to an available CUDA
device, such as `--device cuda:0`. Apple MPS is untested. Unless you ask for
another device, checkpoints are always loaded onto the CPU.
