# Method

What the code computes is set out here. Equation numbers follow the
[article](https://doi.org/10.1088/2632-2153/ad450f), and the implementation
lives in [`model.py`](../src/quantum_pinn/model.py) and
[`physics.py`](../src/quantum_pinn/physics.py).

## Counterdiabatic driving

In an adiabatic protocol the Hamiltonian is interpolated from an initial one,
with a ground state that is easy to prepare, to a final one, whose ground state
is the target:

```math
H_{\mathrm{AD}}(t)=\bigl(1-\lambda(t)\bigr)H_{\mathrm{initial}}+\lambda(t)H_{\mathrm{final}},
\qquad \lambda(0)=0,\quad \lambda(1)=1.
```

Fast evolution causes transitions to excited states. Counterdiabatic driving
suppresses these transitions with the added term $`\dot\lambda A_\lambda`$. In
the instantaneous eigenbasis $`|n\rangle`$ of $`H_{\mathrm{AD}}`$, the adiabatic
gauge potential is (eq. B.7)

```math
\langle m|A_\lambda|n\rangle=-i\,\frac{\langle m|\partial_\lambda H_{\mathrm{AD}}|n\rangle}{E_m-E_n},
\qquad m\neq n .
```

Evaluating this expression requires the full spectrum at every time, and it
becomes ill-conditioned close to level crossings. Sels and Polkovnikov
([PNAS 114 (2017)](https://doi.org/10.1073/pnas.1619826114)) proved that
$`A_\lambda`$ is also the minimiser of the action
$`S=\mathrm{Tr}\,G_\lambda^2`$, where
$`G_\lambda=\partial_\lambda H_{\mathrm{AD}}+i[A_\lambda,H_{\mathrm{AD}}]`$. The
corresponding Euler–Lagrange equation is (eq. 4)

```math
\bigl[\,i\,\partial_\lambda H_{\mathrm{AD}}-[A_{\mathrm{CD}},H_{\mathrm{AD}}],\;H_{\mathrm{AD}}\bigr]=0,
\qquad \partial_\lambda H_{\mathrm{AD}}=H_{\mathrm{final}}-H_{\mathrm{initial}} .
```

Since only matrix products enter this condition, a PINN can impose it as a
residual. The driven Hamiltonian reads (eq. 3)

```math
H(t)=H_{\mathrm{AD}}(t)+\dot\lambda(t)\,A_{\mathrm{CD}}(t).
```

Units: $`\hbar=1`$, energies are in hartree, and $`t\in[0,1]`$. The total evolution
time is thus $`1\,\hbar/E_{\mathrm{h}}`$.

## Network

`QuantumPINN(n_qubits, width, hidden_layers)` is a fully connected network. It
has a single input $`t`$, `hidden_layers` layers of `width` tanh units with
Glorot uniform weights, and a linear output layer. With $`n`$ qubits and $`d=2^n`$,
there are $`1+3d^2`$ output channels (eq. 6):

| Channels | Quantity | Transformation |
| --- | --- | --- |
| 1 | schedule $`\lambda(t)`$ | sigmoid, so $`0\lt \lambda\lt 1`$ |
| $`d^2`$ | real part of $`A_{\mathrm{CD}}(t)`$, row-major | none |
| $`d^2`$ | imaginary part of $`A_{\mathrm{CD}}(t)`$ | none |
| $`4^n=d^2`$ | real coefficients $`C_\mu(t)`$ of the Pauli strings $`P_\mu`$ | none |

Automatic differentiation with respect to the input gives the schedule rate
$`\dot\lambda`$. `predict(t)` returns `schedule`, `schedule_rate`, `potential` of
shape $`(N,d,d)`$, and `coefficients` of shape $`(N,4^n)`$. The ordering of Pauli
strings is that of `itertools.product("IXYZ", repeat=n)`. For two qubits this
gives `II, IX, IY, IZ, XI, …, ZZ`, and the first letter acts on the most
significant qubit of the Kronecker product.

That amounts to 49 network outputs for H₂ ($`n=2`$) and 769 for LiH ($`n=4`$).

## Loss

Given a matrix residual $`R`$, write $`\langle\lVert R\rVert^2\rangle`$ for the
squared Frobenius norm averaged over collocation times. The norm sums over
matrix entries and is not divided by $`d^2`$. `loss_terms` evaluates (eqs. 7–15)

```math
\begin{aligned}
\mathcal L_{\mathrm{IC}}&=\omega_{\mathrm{IC}}\bigl(|\lambda(0)|^2+\lVert H(0)-H_{\mathrm{initial}}\rVert^2\bigr),\\
\mathcal L_{\mathrm{FC}}&=\omega_{\mathrm{FC}}\bigl(|\lambda(1)-1|^2+\lVert H(1)-H_{\mathrm{final}}\rVert^2\bigr),\\
\mathcal L_{\mathrm{LA}}&=\omega_{\mathrm{LA}}\bigl\langle\lVert[\,i\,\partial_\lambda H_{\mathrm{AD}}-[A_{\mathrm{CD}},H_{\mathrm{AD}}],H_{\mathrm{AD}}]\rVert^2\bigr\rangle,\\
\mathcal L_{\mathrm{Ad}}&=\omega_{\mathrm{Ad}}\bigl\langle\dot\lambda^2\bigr\rangle,\\
\mathcal L_{C}&=\omega_{C}\Bigl\langle\Bigl\lVert A_{\mathrm{CD}}-\sum_\mu C_\mu P_\mu\Bigr\rVert^2\Bigr\rangle,\\
\mathcal L&=\mathcal L_{\mathrm{IC}}+\mathcal L_{\mathrm{FC}}+\mathcal L_{\mathrm{LA}}+\mathcal L_{\mathrm{Ad}}+\mathcal L_{C}.
\end{aligned}
```

Because the endpoint terms are built from the total Hamiltonian, they force
$`\dot\lambda A_{\mathrm{CD}}`$ to zero at $`t=0`$ and $`t=1`$. The code provides two
weight sets:

| Preset | $`\omega_{\mathrm{IC}}`$ | $`\omega_{\mathrm{FC}}`$ | $`\omega_{\mathrm{LA}}`$ | $`\omega_{\mathrm{Ad}}`$ | $`\omega_C`$ | Use |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `paper` | $`10^3`$ | $`10^3`$ | $`10^2`$ | 0.5 | 250 | eq. (16) of the article; default |
| `retained` | $`10^3`$ | $`10^3`$ | $`10^2`$ | 0.1 | 100 | training code that produced the bundled models and trajectories |

For monitoring, the Hermiticity error
$`\langle\lVert A_{\mathrm{CD}}-A_{\mathrm{CD}}^\dagger\rVert^2\rangle`$ is
returned as `hermiticity`. Following the article, it is left out of the
objective. Real coefficients already make $`\sum_\mu C_\mu P_\mu`$ Hermitian, and
$`\mathcal L_C`$ draws $`A_{\mathrm{CD}}`$ towards that sum.
[`test_loss_matches_original_research_implementation`](../tests/test_physics.py)
compares these terms with values from the original research code.

## Training

`train(RunConfig(...), output)` takes a scrambled one-dimensional Sobol sample
of `collocation` times in $`[0,1]`$, seeded by `seed`. It evaluates the endpoint
terms at $`t=0`$ and $`t=1`$ and then runs full-batch Adam at a constant learning
rate. The `RunConfig` defaults reproduce the article's settings: six hidden
layers of 30 units, 2048 collocation points, learning rate $`10^{-5}`$, 500000
updates, and the `paper` weights. Seven hidden layers were used for the bundled
models instead ([results](results.md#differences-from-the-article)).

## What the outputs mean

From the least-action condition, $`A_{\mathrm{CD}}`$ is fixed only up to terms
that commute with $`H_{\mathrm{AD}}`$, the identity among them. In the
zero-diagonal gauge the exact potential of the two-qubit H₂ Hamiltonian is
$`A_\lambda=a(\lambda)\,(XY-YX)`$. Here $`H_{\mathrm{final}}-H_{\mathrm{initial}}`$
is proportional to $`XX`$. Within the $`\{|00\rangle,|11\rangle\}`$ block both
diagonal entries coincide, the eigenvectors are independent of $`\lambda`$, and
only the $`\{|01\rangle,|10\rangle\}`$ block contributes. The learned $`C_{XY}`$
and $`C_{YX}`$ track $`\pm a(\lambda)`$. Any other learned coefficient, $`C_{II}`$
for example, is either gauge freedom or residual error. To compare learned and
exact potentials, use `quantum_pinn.physics.exact_gauge_potential`.

With fixed endpoints, the linear schedule $`\lambda=t`$ is the exact minimiser of
the $`\dot\lambda^2`$ penalty, and there
$`\mathcal L_{\mathrm{Ad}}=\omega_{\mathrm{Ad}}`$. The learned schedules are
nearly linear, so this floor accounts for most of the final total loss.

A small loss alone does not prove that the state is prepared. For that reason
the [results](results.md) propagate the Schrödinger equation under the learned
$`H(t)`$ and report the final energy and the ground-state population.

There are $`4^n`$ strings in the complete Pauli basis and $`4^n`$ entries in each
dense matrix. Memory and the cost per update thus grow exponentially with the
number of qubits. Restricting $`C(t)`$ to a subset compatible with hardware is
discussed in the article as future work.
