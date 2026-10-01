"""Data access, training loop, checkpoints, and the ``quantum-pinn`` command line.

Typical use::

    from quantum_pinn import RunConfig, train
    model, summary = train(RunConfig(steps=1000), "datasets/runs/example")

or, from a shell, ``quantum-pinn train --experiment smoke --output datasets/runs/example``.
"""

import argparse
from dataclasses import asdict, dataclass, field, replace
import json
from pathlib import Path
import sys
from time import perf_counter
from typing import Any

import h5py
import numpy as np
import torch

from .model import QuantumPINN
from .physics import WEIGHT_PRESETS, loss_terms, total_hamiltonian

# Bundled molecules and their number of qubits (groups of hamiltonians.h5).
SYSTEMS = {"h2": 2, "lih": 4}
# Tag stored in every checkpoint so that unrelated .pt files are rejected.
CHECKPOINT_FORMAT = "quantum-pinn/1"


def data_root() -> Path:
    """Bundled data: the installed package copy, or ``datasets/`` in a checkout."""
    # After ``pip install .`` the datasets/ folder is copied to quantum_pinn/data;
    # in a source checkout (src/ on PYTHONPATH) it is two levels above this file.
    package = Path(__file__).resolve().parent / "data"
    source = Path(__file__).resolve().parents[2] / "datasets"
    for candidate in (package, source):
        if (candidate / "hamiltonians.h5").is_file():
            return candidate
    raise FileNotFoundError("Bundled data not found; reinstall the package or use a checkout")


def distance_key(distance: float) -> str:
    """HDF5 group name for a bond distance in angstrom, e.g. 1.0 -> "1.0"."""
    return f"{float(distance):.1f}"


def load_hamiltonians(system: str = "h2", distance: float = 1.0):
    """Return (H_initial, H_final) as complex128 arrays for a bundled molecule."""
    if system not in SYSTEMS:
        raise ValueError(f"system must be one of {sorted(SYSTEMS)}")
    # Layout: /{system}/{distance}/{initial,final}, real symmetric matrices in hartree.
    with h5py.File(data_root() / "hamiltonians.h5", "r") as handle:
        group = handle[system]
        assert isinstance(group, h5py.Group)
        key = distance_key(distance)
        if key not in group:
            raise ValueError(f"No bundled {system} Hamiltonians at {key} angstrom")
        pair = group[key]
        assert isinstance(pair, h5py.Group)
        matrices = []
        for name in ("initial", "final"):
            dataset = pair[name]
            assert isinstance(dataset, h5py.Dataset)
            matrices.append(np.asarray(dataset[()], dtype=np.complex128))
    return matrices[0], matrices[1]


def available_distances(system: str = "h2") -> list[float]:
    """Bond distances (angstrom) with bundled Hamiltonians for ``system``."""
    with h5py.File(data_root() / "hamiltonians.h5", "r") as handle:
        group = handle[system]
        assert isinstance(group, h5py.Group)
        return sorted(float(str(key)) for key in group.keys())


def experiments() -> dict[str, dict]:
    """Named training configurations from ``experiments.json``.

    Each entry maps a name such as ``paper/h2-d1.0`` to keyword arguments of
    :class:`RunConfig`; omitted fields take the class defaults.
    """
    return json.loads((data_root() / "experiments.json").read_text(encoding="utf-8"))["experiments"]


@dataclass(frozen=True)
class RunConfig:
    """Training settings. Defaults reproduce the setup stated in the article.

    system         "h2" (2 qubits) or "lih" (4 qubits)
    distance       bond distance in angstrom; must exist in hamiltonians.h5
    width          units per hidden layer
    hidden_layers  number of tanh hidden layers
    collocation    number of Sobol collocation times in [0, 1]
    steps          number of full-batch Adam updates
    learning_rate  constant Adam learning rate
    weights        loss-weight preset, "paper" or "retained" (see physics.py)
    seed           seed for the weight initialisation and the Sobol sample
    dtype          "float32" or "float64" (complex64 or complex128 matrices)
    device         "cpu" or a CUDA device such as "cuda:0"
    snapshot_steps updates after which an extra checkpoint is written
    """

    system: str = "h2"
    distance: float = 1.0
    width: int = 30
    hidden_layers: int = 6
    collocation: int = 2048
    steps: int = 500_000
    learning_rate: float = 1e-5
    weights: str = "paper"
    seed: int = 7
    dtype: str = "float32"
    device: str = "cpu"
    snapshot_steps: tuple[int, ...] = field(default=())

    def __post_init__(self):
        # Validate early so that a long run does not fail after hours of work.
        if self.system not in SYSTEMS:
            raise ValueError(f"system must be one of {sorted(SYSTEMS)}")
        for name in ("width", "hidden_layers", "collocation", "steps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.weights not in WEIGHT_PRESETS:
            raise ValueError(f"weights must be one of {sorted(WEIGHT_PRESETS)}")
        if self.dtype not in ("float32", "float64"):
            raise ValueError("dtype must be float32 or float64")
        snapshots = tuple(int(step) for step in self.snapshot_steps)
        if any(not 1 <= step <= self.steps for step in snapshots) or list(snapshots) != sorted(
            set(snapshots)
        ):
            raise ValueError("snapshot_steps must be increasing and within 1..steps")
        # The dataclass is frozen; normalise fields (e.g. JSON lists to tuples)
        # through object.__setattr__.
        object.__setattr__(self, "snapshot_steps", snapshots)
        object.__setattr__(self, "distance", float(self.distance))

    @property
    def n_qubits(self) -> int:
        """Number of qubits of the selected molecule."""
        return SYSTEMS[self.system]


def build(config: RunConfig):
    """Create the model, collocation times, and Hamiltonians for a configuration."""
    # Seed before creating the layers so that the initial weights are reproducible.
    torch.manual_seed(config.seed)
    dtype = getattr(torch, config.dtype)
    device = torch.device(config.device)
    model = QuantumPINN(config.n_qubits, config.width, config.hidden_layers).to(device, dtype)
    # Scrambled Sobol points cover [0, 1] more evenly than uniform random samples.
    # They are drawn once and reused for every update (full-batch training).
    sobol = torch.quasirandom.SobolEngine(1, scramble=True, seed=config.seed)
    times = sobol.draw(config.collocation).to(device=device, dtype=dtype)
    complex_dtype = model.complex_dtype
    h_initial, h_final = (
        torch.as_tensor(h, dtype=complex_dtype, device=device)
        for h in load_hamiltonians(config.system, config.distance)
    )
    return model, times, h_initial, h_final


def save_checkpoint(model: QuantumPINN, config: RunConfig, path, step: int):
    """Write the configuration and weights; readable with ``weights_only=True``."""
    torch.save(
        {
            "format": CHECKPOINT_FORMAT,
            "config": asdict(config),
            "step": step,
            "state_dict": model.state_dict(),
        },
        path,
    )


def load_checkpoint(path, device: str = "cpu"):
    """Load a checkpoint written by :func:`train` or a bundled pretrained model."""
    # weights_only=True refuses arbitrary pickled objects, so loading an
    # untrusted file cannot execute code.
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"{path} is not a {CHECKPOINT_FORMAT} checkpoint")
    options = dict(payload["config"])
    # Rebuild the exact architecture from the stored configuration; the
    # requested device replaces the one used during training.
    options["device"] = device
    config = RunConfig(**options)
    model = QuantumPINN(config.n_qubits, config.width, config.hidden_layers)
    model = model.to(torch.device(device), getattr(torch, config.dtype))
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return model, config


def pretrained_names() -> list[str]:
    """Names of the bundled models in ``datasets/pretrained``."""
    return sorted(path.stem for path in (data_root() / "pretrained").glob("*.pt"))


def load_pretrained(name: str = "h2-d1.0", device: str = "cpu"):
    """Load a bundled model, for example ``h2-d1.0`` or ``lih-d1.0``."""
    path = data_root() / "pretrained" / f"{name}.pt"
    if not path.is_file():
        raise ValueError(f"Unknown pretrained model {name!r}; choose from {pretrained_names()}")
    return load_checkpoint(path, device)


def train(config: RunConfig, output, progress: bool = False):
    """Train a model and write ``config.json``, ``history.npz`` and ``model.pt``.

    The output directory must not exist. Loss histories are recorded before
    each Adam update.
    """
    # Refuse to overwrite an existing run.
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")
    model, times, h_initial, h_final = build(config)
    weights = WEIGHT_PRESETS[config.weights]
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    # One value per update for every loss term (see physics.loss_terms).
    names = ("total", "initial", "final", "least_action", "schedule_rate", "pauli", "hermiticity")
    history = {name: np.empty(config.steps) for name in names}
    report = max(1, config.steps // 20)  # progress message every 5% of the run
    started = perf_counter()
    for step in range(config.steps):
        # One full-batch Adam update on all collocation points and both endpoints.
        optimizer.zero_grad(set_to_none=True)
        terms = loss_terms(model, times, h_initial, h_final, weights)
        terms["total"].backward()
        optimizer.step()
        # The recorded losses belong to the weights *before* this update.
        for name in names:
            history[name][step] = terms[name].item()
        if not np.isfinite(history["total"][step]):
            raise FloatingPointError(f"Non-finite loss at step {step + 1}")
        if step + 1 in config.snapshot_steps:
            (directory / "snapshots").mkdir(exist_ok=True)
            save_checkpoint(
                model, config, directory / f"snapshots/step-{step + 1:07d}.pt", step + 1
            )
        if progress and ((step + 1) % report == 0 or step + 1 == config.steps):
            print(
                f"step {step + 1}/{config.steps}  loss {history['total'][step]:.4e}",
                file=sys.stderr,
            )
    seconds = perf_counter() - started
    arrays: dict[str, Any] = {"seconds": np.asarray(seconds), **history}
    np.savez_compressed(directory / "history.npz", **arrays)
    save_checkpoint(model, config, directory / "model.pt", config.steps)
    return model, {"seconds": seconds, "final_loss": float(history["total"][-1])}


def evaluate(model: QuantumPINN, system: str, distance: float, points: int = 1001):
    """Evaluate a model on a uniform grid and return NumPy arrays."""
    # Uniform grid in the model's dtype and device.
    weight = model.input_layer.weight
    t = torch.linspace(0, 1, points, dtype=weight.dtype, device=weight.device).reshape(-1, 1)
    out = model.predict(t)
    h_initial, h_final = load_hamiltonians(system, distance)
    # Convert to NumPy and flatten the (N, 1) scalar fields to (N,).
    arrays = {key: value.cpu().numpy() for key, value in out.items()}
    arrays["time"] = arrays["time"].ravel()
    arrays["schedule"] = arrays["schedule"].ravel()
    arrays["schedule_rate"] = arrays["schedule_rate"].ravel()
    arrays["hamiltonian"] = total_hamiltonian(
        arrays["schedule"], arrays["schedule_rate"], arrays["potential"], h_initial, h_final
    )
    return arrays


def _config_from_args(args) -> RunConfig:
    """Merge a named experiment with command-line overrides."""
    options = {}
    if args.experiment:
        catalogue = experiments()
        if args.experiment not in catalogue:
            raise ValueError(f"Unknown experiment {args.experiment!r}; see 'quantum-pinn list'")
        options.update(catalogue[args.experiment])
    for key in ("system", "distance", "steps", "collocation", "seed", "device", "weights"):
        value = getattr(args, key)
        if value is not None:
            options[key] = value
    if args.snapshot_steps:
        options["snapshot_steps"] = tuple(args.snapshot_steps)
    config = RunConfig(**options)
    if args.smoke:
        config = replace(config, steps=3, collocation=64, snapshot_steps=())
    return config


def main(argv=None):
    """Entry point of the ``quantum-pinn`` command; prints a JSON summary."""
    parser = argparse.ArgumentParser(prog="quantum-pinn", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="list named experiments and pretrained models")
    run = commands.add_parser("train", help="train a model")
    run.add_argument("--experiment", help="named configuration from experiments.json")
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--system", choices=sorted(SYSTEMS))
    run.add_argument("--distance", type=float)
    run.add_argument("--steps", type=int)
    run.add_argument("--collocation", type=int)
    run.add_argument("--seed", type=int)
    run.add_argument("--weights", choices=sorted(WEIGHT_PRESETS))
    run.add_argument("--device")
    run.add_argument("--snapshot-steps", type=int, nargs="+")
    run.add_argument("--smoke", action="store_true", help="3 updates on 64 points")
    check = commands.add_parser("evaluate", help="evaluate a checkpoint or pretrained model")
    check.add_argument("model", help="checkpoint path or pretrained name such as h2-d1.0")
    check.add_argument("--points", type=int, default=1001)
    args = parser.parse_args(argv)
    try:
        if args.command == "list":
            result = {"experiments": sorted(experiments()), "pretrained": pretrained_names()}
        elif args.command == "train":
            config = _config_from_args(args)
            _, summary = train(config, args.output, progress=True)
            result = {"output": str(args.output), **summary}
        else:
            # Accept either a checkpoint file or the name of a bundled model.
            path = Path(args.model)
            model, config = load_checkpoint(path) if path.is_file() else load_pretrained(args.model)
            data = evaluate(model, config.system, config.distance, args.points)
            h_initial, h_final = load_hamiltonians(config.system, config.distance)
            result = {
                "system": config.system,
                "distance": config.distance,
                "schedule_start": float(data["schedule"][0]),
                "schedule_end": float(data["schedule"][-1]),
                "initial_hamiltonian_error": float(
                    np.abs(data["hamiltonian"][0] - h_initial).max()
                ),
                "final_hamiltonian_error": float(np.abs(data["hamiltonian"][-1] - h_final).max()),
            }
    except (ValueError, FileExistsError, FileNotFoundError, FloatingPointError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
