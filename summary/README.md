# Verification

Bundled with the repository are 14 Hamiltonian pairs for H₂ and LiH, six
trained models, the article's stored H₂ trajectories and loss history, the
46-run scalability table, 52 training configurations, and three tutorials. The
scientific comparison is in [results](../docs/results.md), and the role of each
file is in the [repository map](repository-map.md).

## Run the checks

```bash
python -m pip install -e '.[test]' ruff==0.12.0 pyright==1.1.414
python -m pytest -q
ruff check src tests
ruff format --check src tests
pyright
python src/verify.py
```

What the tests check:

- the Pauli basis, the exact gauge potential (required to satisfy the
  least-action condition within $`10^{-10}`$), and propagation that preserves the
  norm;
- the loss, compared on a bundled model with values from the original research
  implementation;
- shapes and dtypes of the model, and its autograd schedule rate compared with
  finite differences;
- short training runs, determinism, reloading of checkpoints, the command line,
  and the catalogue of configurations;
- the stored results: a counterdiabatic term that vanishes at both endpoints,
  agreement with the exact gauge potential, a ground-state population above
  0.9999 at 1.0 Å, and agreement between `evaluation.json` and a fresh
  recomputation;
- running the three notebooks and checking that documentation links are valid.

`src/verify.py` verifies the SHA-256 hashes of the bundled data. It also
searches every public file, HDF5 attributes and the contents of checkpoint
archives included, for home-directory paths, e-mail addresses, host names,
private keys, and access tokens. Only the working tree is scanned, not the Git
history.

Together, these checks show that the software does what the documentation says
and that the stored results back the figures. The 500000-update training runs
are not repeated here. To repeat them, use the configurations in the
[training guide](../docs/training.md).
