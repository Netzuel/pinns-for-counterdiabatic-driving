# Jupyter tutorials

| Notebook | What it does | Time on CPU |
| --- | --- | --- |
| [1. Train and test](quickstart.ipynb) | Trains an H$_2$ model for 3000 updates, plots the loss terms, compares the learned counterdiabatic term with the exact one, propagates the ground state with and without it, and reloads the checkpoint | about 10 s |
| [2. Evaluate a pretrained model](inference.ipynb) | Loads `h2-d1.0`, checks the endpoints, plots the schedule and counterdiabatic coefficients, and propagates the ground state with and without the counterdiabatic term | about 5 s |
| [3. Reproduce the article's results](reproduce-results.ipynb) | Recomputes Table 1, the physics checks, and the scalability table from the stored trajectories | about 10 s |

Once [installation](../installation.md) is complete, run from the repository
root:

```bash
python -m pip install '.[notebooks]'
python -m jupyterlab docs/examples
```

Each notebook is saved with its outputs, so GitHub shows the results without
anyone running it. To run one again, open it and select **Run → Run All
Cells**. The notebooks rely only on bundled data, and any temporary files they
write go outside the repository. All three are executed by the test suite.
