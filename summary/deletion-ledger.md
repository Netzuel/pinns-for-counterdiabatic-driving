# Removed material

For the public release, only the code and data needed to train, evaluate, and
reproduce the article's results were kept. The research history was removed or
condensed as listed below, and the release depends on none of the removed
items.

| Removed | Replacement |
| --- | --- |
| Nine research variants of the model class and plotting utilities (`models.py`, `utils.py`, about 3900 lines) | `quantum_pinn.model` and `quantum_pinn.physics`; the loss is tested against the original class on a bundled model |
| Historical training scripts for each distance and system, including a six-qubit H₂ variant not used in the article | `datasets/experiments.json` and `quantum-pinn train` |
| Analysis and data-generation notebooks with hard-coded matrices | `quantum_pinn.analysis`, three tutorials, and `datasets/hamiltonians.h5` |
| 745 MB of result files: per-run HDF5 outputs at 10000 times, checkpoints for every collocation size, duplicated runs, and short runs with 16–64 collocation points | 10 MB: four trajectories, the full loss history of the 1.0 Å run, a 46-row scalability table, and six checkpoints |
| Raster figures and figure drafts | regenerated vector figures in light and dark versions |
| Six-qubit H₂ (6-31G) Hamiltonians and a pickled input file | not used in the article; pickles are unsafe to load |
| Earlier packaging, audit, and reproduction modules | `src/figures.py`, `src/verify.py` |

During conversion, checkpoint parameters were renamed (`params_hidden.*` became
`layers.*`), and arrays were stored under descriptive names in single precision
with lossless compression. No numerical value changed.
