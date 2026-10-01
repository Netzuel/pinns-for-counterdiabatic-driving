"""Hamiltonians, physics-informed residuals, and post-training diagnostics.

Conventions follow the article (hbar = 1, time t in [0, 1]):

    H_AD(t) = (1 - lambda(t)) H_initial + lambda(t) H_final
    H(t)    = H_AD(t) + d(lambda)/dt A_CD(t)

The least-action residual is the Euler-Lagrange condition of the adiabatic
gauge potential, eq. (4) of the article:

    [ i dH_AD/dlambda - [A_CD, H_AD], H_AD ] = 0,  dH_AD/dlambda = H_final - H_initial.

Functions that take ``torch.Tensor`` arguments are differentiable and used
during training; the NumPy functions in the second half are used to analyse
trained models.
"""

from dataclasses import asdict, dataclass
from functools import lru_cache
from itertools import product

import numpy as np
import torch

# Single-qubit Pauli matrices; multi-qubit strings are Kronecker products of these.
PAULI = {
    "I": np.eye(2, dtype=np.complex128),
    "X": np.array([[0, 1], [1, 0]], dtype=np.complex128),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
    "Z": np.array([[1, 0], [0, -1]], dtype=np.complex128),
}


@lru_cache(maxsize=None)
def pauli_strings(n_qubits: int) -> tuple[str, ...]:
    """Names of all 4**n Pauli strings, ordered as itertools.product("IXYZ").

    For two qubits: ("II", "IX", "IY", "IZ", "XI", ..., "ZZ"). The first letter
    acts on the most significant qubit of the Kronecker product.
    """
    return tuple("".join(name) for name in product("IXYZ", repeat=n_qubits))


@lru_cache(maxsize=None)
def _pauli_matrices(n_qubits: int) -> np.ndarray:
    # Build each string P = sigma_1 (x) sigma_2 (x) ... left to right, so the
    # matrix order matches the names returned by pauli_strings.
    matrices = []
    for name in pauli_strings(n_qubits):
        matrix = np.ones((1, 1), dtype=np.complex128)
        for letter in name:
            matrix = np.kron(matrix, PAULI[letter])
        matrices.append(matrix)
    return np.stack(matrices)


def pauli_basis(n_qubits: int) -> np.ndarray:
    """Return the Pauli strings as an array of shape (4**n, 2**n, 2**n)."""
    # Copy so that callers cannot modify the cached array.
    return _pauli_matrices(n_qubits).copy()


def pauli_coefficients(matrices: np.ndarray) -> np.ndarray:
    """Coefficients c_mu = Tr(P_mu A) / 2**n of matrices A with shape (..., d, d).

    Pauli strings are orthogonal, Tr(P_mu P_nu) = d delta_mu,nu, so
    A = sum_mu c_mu P_mu. The coefficients are real when A is Hermitian.
    """
    matrices = np.asarray(matrices)
    dimension = matrices.shape[-1]
    n_qubits = dimension.bit_length() - 1
    basis = _pauli_matrices(n_qubits)
    # einsum computes Tr(P_mu A) = sum_ij (P_mu)_ij A_ji for every leading index.
    return np.einsum("pij,...ji->...p", basis, matrices) / dimension


def commutator(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """[a, b] = ab - ba, batched over leading dimensions."""
    return a @ b - b @ a


@dataclass(frozen=True)
class LossWeights:
    """Weights of the five loss terms, eq. (16) of the article.

    ``initial`` and ``final`` weight the boundary conditions at t = 0 and t = 1,
    ``least_action`` the gauge-potential residual, ``schedule_rate`` the
    penalty on (d lambda/dt)^2, and ``pauli`` the agreement between A_CD and
    its Pauli reconstruction.
    """

    initial: float = 1e3
    final: float = 1e3
    least_action: float = 1e2
    schedule_rate: float = 0.5
    pauli: float = 2.5e2

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


PAPER_WEIGHTS = LossWeights()
"""Weights stated in the article."""

RETAINED_WEIGHTS = LossWeights(schedule_rate=0.1, pauli=1e2)
"""Weights used by the training code that produced the bundled checkpoints."""

# Names accepted by RunConfig.weights.
WEIGHT_PRESETS = {"paper": PAPER_WEIGHTS, "retained": RETAINED_WEIGHTS}


def total_hamiltonian(schedule, schedule_rate, potential, h_initial, h_final):
    """H(t) for arrays of shape (N, 1), (N, 1), (N, d, d), and (d, d).

    Works for both NumPy arrays and PyTorch tensors.
    """
    # Reshape the scalar fields to (N, 1, 1) so they broadcast over matrices.
    s = schedule.reshape(-1, 1, 1)
    rate = schedule_rate.reshape(-1, 1, 1)
    return (1 - s) * h_initial + s * h_final + rate * potential


def _squared_norm(residual: torch.Tensor) -> torch.Tensor:
    """Mean over samples of the squared Frobenius norm |R|_F^2.

    Entries are summed (not averaged) within each matrix, as in the original
    research code, so the scale grows with the matrix size.
    """
    return residual.abs().square().flatten(1).sum(dim=1).mean()


def loss_terms(model, times, h_initial, h_final, weights: LossWeights = PAPER_WEIGHTS):
    """Weighted loss terms of eqs. (7), (8), (10), (11) and (13).

    ``times`` holds the interior collocation points, shape (N, 1). The endpoint
    terms are evaluated at t = 0 and t = 1. The Hermiticity diagnostic is
    returned for monitoring only; as in the article, it does not enter the
    objective. All returned values are scalar tensors; ``total`` can be
    backpropagated.
    """
    h_initial = h_initial.to(model.complex_dtype)
    h_final = h_final.to(model.complex_dtype)

    # Boundary conditions, eqs. (7) and (8): lambda(0) = 0, lambda(1) = 1, and
    # the total Hamiltonian (including the CD term) must equal H_initial and
    # H_final at the two ends. This forces d(lambda)/dt * A_CD to vanish there.
    endpoints = torch.tensor([[0.0], [1.0]], dtype=times.dtype, device=times.device)
    edge = model.predict(endpoints, create_graph=True)
    edge_h = total_hamiltonian(
        edge["schedule"], edge["schedule_rate"], edge["potential"], h_initial, h_final
    )
    initial = weights.initial * (
        edge["schedule"][0].square().sum() + (edge_h[0] - h_initial).abs().square().sum()
    )
    final = weights.final * (
        (edge["schedule"][1] - 1).square().sum() + (edge_h[1] - h_final).abs().square().sum()
    )

    # Interior predictions with the graph kept, so that the loss can be
    # differentiated with respect to the network weights (including through
    # d(lambda)/dt, which is itself an autograd derivative).
    out = model.predict(times, create_graph=True)
    s = out["schedule"].reshape(-1, 1, 1).to(model.complex_dtype)
    adiabatic = (1 - s) * h_initial + s * h_final
    potential = out["potential"]

    # Least-action residual, eq. (10): [i dH_AD/dlambda - [A, H_AD], H_AD].
    change = 1j * (h_final - h_initial)
    residual = commutator(change - commutator(potential, adiabatic), adiabatic)
    least_action = weights.least_action * _squared_norm(residual)

    # Adiabaticity penalty, eq. (11): mean of (d lambda/dt)^2.
    schedule_rate = weights.schedule_rate * out["schedule_rate"].square().mean()

    # Pauli consistency, eq. (13): A_CD should equal sum_mu C_mu P_mu.
    basis = model.pauli_basis.to(potential)
    reconstruction = torch.einsum("np,pij->nij", out["coefficients"].to(potential), basis)
    pauli = weights.pauli * _squared_norm(reconstruction - potential)

    # Monitoring only: distance of A_CD from a Hermitian matrix.
    hermiticity = _squared_norm(potential - potential.mH)

    total = initial + final + least_action + schedule_rate + pauli
    return {
        "total": total,
        "initial": initial,
        "final": final,
        "least_action": least_action,
        "schedule_rate": schedule_rate,
        "pauli": pauli,
        "hermiticity": hermiticity,
    }


# --- Diagnostics on NumPy arrays -------------------------------------------------


def adiabatic_hamiltonian(schedule, h_initial, h_final):
    """H_AD = (1 - lambda) H_initial + lambda H_final for a schedule of shape (N,)."""
    s = np.asarray(schedule).reshape(-1, 1, 1)
    return (1 - s) * h_initial + s * h_final


def exact_gauge_potential(schedule, h_initial, h_final, tolerance=1e-9):
    """Exact adiabatic gauge potential in the zero-diagonal gauge, eq. (B.7).

    <m|A|n> = -i <m|dH/dlambda|n> / (E_m - E_n) for m != n, evaluated in the
    instantaneous eigenbasis of H_AD(lambda). Pairs with |E_m - E_n| below
    ``tolerance`` are set to zero. Returns an array of shape (N, d, d).
    """
    hamiltonians = adiabatic_hamiltonian(schedule, h_initial, h_final)
    energies, vectors = np.linalg.eigh(hamiltonians)
    change = np.asarray(h_final - h_initial, dtype=np.complex128)
    # Matrix elements <m|dH/dlambda|n> in the instantaneous eigenbasis.
    elements = vectors.conj().transpose(0, 2, 1) @ change @ vectors
    # Energy differences E_m - E_n; degenerate pairs (including the diagonal
    # m = n) are divided by infinity, which sets those elements to zero.
    gaps = energies[:, :, None] - energies[:, None, :]
    safe = np.where(np.abs(gaps) > tolerance, gaps, np.inf)
    eigenbasis = -1j * elements / safe
    # Rotate back to the computational basis.
    return vectors @ eigenbasis @ vectors.conj().transpose(0, 2, 1)


def propagate(times, hamiltonians, initial_state, rtol=1e-10, atol=1e-12):
    """Integrate i d|psi>/dt = H(t)|psi> with piecewise-linear H between samples.

    ``hamiltonians`` has one (d, d) matrix per entry of ``times``. Returns the
    states at every sample time, shape (N, d). The norm is not renormalised,
    so a non-Hermitian H shows up as norm drift.
    """
    from scipy.integrate import solve_ivp

    times = np.asarray(times, dtype=float).reshape(-1)
    hamiltonians = np.asarray(hamiltonians, dtype=np.complex128)
    if len(times) < 2 or np.any(np.diff(times) <= 0):
        raise ValueError("times must be strictly increasing")
    if hamiltonians.shape[0] != len(times):
        raise ValueError("one Hamiltonian per sample time is required")

    def rhs(t, state):
        # Locate the sampling interval containing t and interpolate H linearly.
        index = int(np.clip(np.searchsorted(times, t) - 1, 0, len(times) - 2))
        fraction = (t - times[index]) / (times[index + 1] - times[index])
        matrix = (1 - fraction) * hamiltonians[index] + fraction * hamiltonians[index + 1]
        return -1j * (matrix @ state)

    # DOP853 is an explicit 8th-order Runge-Kutta method; tight tolerances keep
    # the integration error far below the effects being measured.
    solution = solve_ivp(
        rhs,
        (times[0], times[-1]),
        np.asarray(initial_state, dtype=np.complex128),
        method="DOP853",
        t_eval=times,
        rtol=rtol,
        atol=atol,
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    return solution.y.T


def instantaneous_populations(states, hamiltonians):
    """|<n(t)|psi(t)>|^2 for the eigenstates n(t) of Hermitian ``hamiltonians``.

    Column 0 is the population of the instantaneous ground state.
    """
    _, vectors = np.linalg.eigh(hamiltonians)
    # amplitudes[t, n] = <n(t)|psi(t)>, with eigenvectors stored as columns.
    amplitudes = np.einsum("nij,ni->nj", vectors.conj(), states)
    return np.abs(amplitudes) ** 2
