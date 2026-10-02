# Reference data

Every file sits in [`datasets/`](../datasets/) and is installed along with the
package. Their SHA-256 hashes are recorded in
[`datasets/checksums.json`](../datasets/checksums.json), and
`python src/verify.py` verifies them.

## Hamiltonians

Real symmetric qubit Hamiltonians, in hartree and single precision, are stored
in [`hamiltonians.h5`](../datasets/hamiltonians.h5) under
`/{system}/{distance}/{initial,final}`:

| Group | Qubits | Matrix size | Bond distances (Å) |
| --- | ---: | --- | --- |
| `h2` | 2 | 4 × 4 | 0.5, 1.0, …, 3.5 |
| `lih` | 4 | 16 × 16 | 0.5, 1.0, …, 3.5 |

`final` is the molecular Hamiltonian in the STO-3G basis. `initial` is its
diagonal in the computational basis, which the article names the Hartree–Fock
Hamiltonian. It retains the $`I`$ and $`Z`$ strings and removes the off-diagonal
couplings ($`XX`$ for H₂), so its ground state is a computational basis state.
According to the article (Appendix C), both matrices were generated with
Qiskit. The two-qubit H₂ form, built from the strings $`II, IZ, ZI, ZZ, XX`$,
agrees with the parity mapping with two-qubit reduction, and its ground
energies agree with STO-3G full-configuration-interaction energies that include
nuclear repulsion. Neither the original generation script nor the LiH
active-space choice that gives four qubits was kept, so the stored matrices are
used here as they are.

## Trained models

Six PyTorch checkpoints are kept in [`pretrained/`](../datasets/pretrained/).
The original research code wrote them, and they were converted to the
`quantum_pinn.training.save_checkpoint` format, a dictionary with `format`,
`config`, `step` and `state_dict`. `load_pretrained(name)` loads one, with
`weights_only=True`. Each network has seven hidden layers of 30 units and was
trained on 2048 collocation points with the `retained` loss weights. The step
count on record, 300001, comes from the historical training script and has not
been independently confirmed.

## Stored results

The predictions of the article's H₂ runs, on 10000 uniform times, are in
[`results/h2-trajectories.h5`](../datasets/results/h2-trajectories.h5):

| Path | Shape | Meaning |
| --- | --- | --- |
| `/{1.0,1.5,2.0,2.5}/time` | (10000,) | $`t\in[0,1]`$ |
| `…/schedule`, `…/schedule_rate` | (10000,) | $`\lambda`$ and $`\dot\lambda`$ |
| `…/potential` | (10000, 4, 4) complex | $`A_{\mathrm{CD}}`$ |
| `…/coefficients` | (10000, 16) | $`C_\mu`$ in the order of `pauli_strings(2)` |
| `…` attribute `training_seconds` | scalar | recorded wall time |
| `/history/{total,initial,final,least_action,schedule_rate,pauli,hermiticity}` | (500001,) | loss before each update of the 1.0 Å run |

The 1.0 Å group is taken from the 500001-update run, and the other groups from
runs with 2048 collocation points. As in the output of the original code,
arrays are kept in single precision with lossless compression.

For each of the 46 runs in the collocation study,
[`results/scalability.csv`](../datasets/results/scalability.csv) gives the
system, distance, number of collocation points, final total loss with its
individual terms, and recorded wall time in seconds.

## Configurations and figures

The named training configurations covered in the [training guide](training.md)
are in [`experiments.json`](../datasets/experiments.json). In
[`figures/`](../datasets/figures/) are the light and dark SVG figures together
with [`evaluation.json`](../datasets/figures/evaluation.json), all produced by
`python src/figures.py` and `bash src/diagrams/render.sh`.

## Terms of use

The terms are in [`datasets/LICENSE.txt`](../datasets/LICENSE.txt). Please cite
the article if you use these data.
