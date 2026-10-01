"""Fully connected PINN for the schedule, gauge potential, and Pauli coefficients."""

import torch
from torch import nn

from .physics import pauli_basis


class QuantumPINN(nn.Module):
    """Map time t to (lambda, A_CD, C), eq. (6) of the article.

    The network has one input, ``hidden_layers`` tanh layers of ``width`` units,
    and a linear output layer with 1 + 3 d**2 channels for d = 2**n_qubits:

    * one schedule logit, mapped to lambda in (0, 1) by a sigmoid;
    * d**2 real and d**2 imaginary entries of the gauge potential A_CD;
    * 4**n_qubits = d**2 real coefficients C of the Pauli strings.

    The matrix A_CD and its Pauli reconstruction sum_mu C_mu P_mu are separate
    outputs; the Pauli loss term couples them.
    """

    def __init__(self, n_qubits: int = 2, width: int = 30, hidden_layers: int = 6):
        super().__init__()
        if n_qubits < 1 or width < 1 or hidden_layers < 1:
            raise ValueError("n_qubits, width and hidden_layers must be positive")
        self.n_qubits = n_qubits
        self.dimension = 2**n_qubits
        # Layer sizes: 1 input (time) -> hidden_layers x width -> 1 + 3 d^2 outputs.
        widths = [1] + [width] * hidden_layers + [1 + 3 * self.dimension**2]
        self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(widths[:-1], widths[1:]))
        # Glorot (Xavier) uniform weights, as in the article; biases keep the
        # PyTorch default initialisation used by the original code.
        for layer in self.layers:
            assert isinstance(layer, nn.Linear)
            nn.init.xavier_uniform_(layer.weight)
        # Pauli strings used by the Pauli loss term. Kept as a plain tensor, not
        # a registered buffer: Module.to(float32) would otherwise cast the complex
        # matrices to real and drop the imaginary part of Y.
        self._pauli_basis = torch.from_numpy(pauli_basis(n_qubits))

    @property
    def input_layer(self) -> nn.Linear:
        """First linear layer; its weight sets the model's device and real dtype."""
        layer = self.layers[0]
        assert isinstance(layer, nn.Linear)
        return layer

    @property
    def pauli_basis(self) -> torch.Tensor:
        """Pauli strings, shape (4**n, d, d), in the model's complex dtype and device."""
        weight = self.input_layer.weight
        # Convert lazily after the model has been moved with .to(device, dtype).
        if (
            self._pauli_basis.device != weight.device
            or self._pauli_basis.dtype != self.complex_dtype
        ):
            self._pauli_basis = self._pauli_basis.to(weight.device, self.complex_dtype)
        return self._pauli_basis

    @property
    def complex_dtype(self) -> torch.dtype:
        """complex64 for a float32 model, complex128 for a float64 model."""
        real = self.input_layer.weight.dtype
        return torch.complex128 if real == torch.float64 else torch.complex64

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Raw network output of shape (N, 1 + 3 d**2) for times of shape (N, 1)."""
        x = t
        # tanh on every hidden layer; the output layer is linear.
        for layer in self.layers[:-1]:
            x = torch.tanh(layer(x))
        return self.layers[-1](x)

    def predict(self, t, create_graph: bool = False) -> dict[str, torch.Tensor]:
        """Evaluate the physical outputs at times ``t`` of shape (N, 1).

        Returns ``schedule`` (N, 1), ``schedule_rate`` d(lambda)/dt (N, 1),
        complex ``potential`` (N, d, d), and real ``coefficients`` (N, 4**n).
        With ``create_graph=False`` the results are detached.
        """
        weight = self.input_layer.weight
        t = torch.as_tensor(t, dtype=weight.dtype, device=weight.device)
        if t.ndim != 2 or t.shape[1] != 1:
            raise ValueError("times must have shape (N, 1)")
        # Gradients are needed for d(lambda)/dt even when the caller is inside
        # torch.no_grad(), so enable them locally.
        with torch.enable_grad():
            t = t.detach().requires_grad_(True)
            raw = self(t)
            # Channel 0 is the schedule logit; the sigmoid keeps lambda in (0, 1).
            schedule = torch.sigmoid(raw[:, :1])
            # d(lambda)/dt by automatic differentiation. Each output row depends
            # only on its own input time, so the vector-Jacobian product with a
            # vector of ones gives the per-sample derivative. create_graph=True
            # keeps this derivative differentiable for training.
            (rate,) = torch.autograd.grad(
                schedule, t, torch.ones_like(schedule), create_graph=create_graph
            )
        # Remaining channels: d^2 real parts and d^2 imaginary parts of A_CD
        # (row-major), followed by the 4^n = d^2 Pauli coefficients.
        n = self.dimension**2
        potential = torch.complex(raw[:, 1 : 1 + n], raw[:, 1 + n : 1 + 2 * n])
        result = {
            "time": t,
            "schedule": schedule,
            "schedule_rate": rate,
            "potential": potential.reshape(-1, self.dimension, self.dimension),
            "coefficients": raw[:, 1 + 2 * n :],
        }
        if not create_graph:
            result = {key: value.detach() for key, value in result.items()}
        return result
