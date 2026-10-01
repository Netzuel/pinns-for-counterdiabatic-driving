import json
from pathlib import Path

import numpy as np
import pytest
import torch

from quantum_pinn import LossWeights, load_hamiltonians, load_pretrained, loss_terms
from quantum_pinn.physics import (
    exact_gauge_potential,
    pauli_basis,
    pauli_coefficients,
    pauli_strings,
    propagate,
)

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.mark.parametrize("n_qubits", [1, 2, 3])
def test_pauli_basis_is_hermitian_and_orthogonal(n_qubits):
    basis = pauli_basis(n_qubits)
    d = 2**n_qubits
    assert basis.shape == (4**n_qubits, d, d)
    assert np.allclose(basis, basis.conj().transpose(0, 2, 1))
    gram = np.einsum("aij,bji->ab", basis, basis)
    assert np.allclose(gram, d * np.eye(4**n_qubits))
    assert pauli_strings(n_qubits)[0] == "I" * n_qubits


def test_pauli_coefficients_reconstruct_matrices():
    rng = np.random.default_rng(1)
    matrix = rng.normal(size=(5, 4, 4)) + 1j * rng.normal(size=(5, 4, 4))
    hermitian = matrix + matrix.conj().transpose(0, 2, 1)
    coefficients = pauli_coefficients(hermitian)
    assert np.allclose(coefficients.imag, 0)
    assert np.allclose(np.einsum("np,pij->nij", coefficients, pauli_basis(2)), hermitian)


@pytest.mark.parametrize("distance", [1.0, 2.5])
def test_exact_gauge_potential_solves_least_action_condition(distance):
    h_initial, h_final = load_hamiltonians("h2", distance)
    schedule = np.linspace(0.01, 1.0, 50)
    potential = exact_gauge_potential(schedule, h_initial, h_final)
    adiabatic = (1 - schedule)[:, None, None] * h_initial + schedule[:, None, None] * h_final
    inner = 1j * (h_final - h_initial) - (potential @ adiabatic - adiabatic @ potential)
    residual = inner @ adiabatic - adiabatic @ inner
    assert np.abs(residual).max() < 1e-10
    assert np.allclose(potential, potential.conj().transpose(0, 2, 1))
    names = pauli_strings(2)
    coefficients = pauli_coefficients(potential).real
    xy, yx = names.index("XY"), names.index("YX")
    assert np.allclose(coefficients[:, xy], -coefficients[:, yx])
    others = np.delete(coefficients, [xy, yx], axis=1)
    assert np.abs(others).max() < 1e-12


def test_loss_matches_original_research_implementation():
    fixture = json.loads((FIXTURES / "legacy-loss.json").read_text())
    model, _ = load_pretrained(fixture["pretrained"])
    h_initial, h_final = (
        torch.as_tensor(h, dtype=torch.complex64) for h in load_hamiltonians("h2", 1.0)
    )
    sobol = torch.quasirandom.SobolEngine(1, scramble=True, seed=fixture["sobol_seed"])
    times = sobol.draw(fixture["collocation"])
    weights = LossWeights(1e3, 1e3, 1e2, 0.5, 250.0)
    terms = loss_terms(model, times, h_initial, h_final, weights)
    for name, expected in fixture["terms"].items():
        assert terms[name].item() == pytest.approx(expected, rel=1e-4, abs=1e-9), name


def test_propagation_conserves_norm_for_hermitian_hamiltonians():
    h_initial, h_final = load_hamiltonians("h2", 1.5)
    times = np.linspace(0, 1, 201)
    hamiltonians = (1 - times)[:, None, None] * h_initial + times[:, None, None] * h_final
    states = propagate(times, hamiltonians, np.eye(4)[1])
    assert np.allclose(np.linalg.norm(states, axis=1), 1, atol=1e-9)
