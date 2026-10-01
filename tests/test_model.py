import pytest
import torch

from quantum_pinn import QuantumPINN


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_predictions_have_documented_shapes(dtype):
    model = QuantumPINN(n_qubits=2, width=8, hidden_layers=2).to(dtype)
    out = model.predict(torch.linspace(0, 1, 7).reshape(-1, 1))
    assert out["schedule"].shape == (7, 1)
    assert out["schedule_rate"].shape == (7, 1)
    assert out["potential"].shape == (7, 4, 4)
    assert out["potential"].dtype == model.complex_dtype
    assert out["coefficients"].shape == (7, 16)
    assert ((out["schedule"] > 0) & (out["schedule"] < 1)).all()
    assert len(model.layers) == 3
    assert model.layers[-1].out_features == 1 + 3 * 16


def test_pauli_basis_keeps_imaginary_parts_after_dtype_change():
    model = QuantumPINN(n_qubits=1, width=4, hidden_layers=1).to(torch.float64)
    basis = model.pauli_basis
    assert basis.dtype == torch.complex128
    assert torch.allclose(basis[2], torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex128))


def test_schedule_rate_matches_finite_differences():
    torch.manual_seed(0)
    model = QuantumPINN(n_qubits=1, width=16, hidden_layers=3).to(torch.float64)
    t = torch.linspace(0.1, 0.9, 9, dtype=torch.float64).reshape(-1, 1)
    h = 1e-6
    rate = model.predict(t)["schedule_rate"]
    numeric = (model.predict(t + h)["schedule"] - model.predict(t - h)["schedule"]) / (2 * h)
    assert torch.allclose(rate, numeric, atol=1e-8)


def test_predict_rejects_wrong_shape():
    with pytest.raises(ValueError):
        QuantumPINN().predict(torch.zeros(3))
