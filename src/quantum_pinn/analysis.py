"""Post-training analysis of stored or newly evaluated H2 trajectories.

These functions reproduce the quantitative content of the article's figures
and Table 1 from arrays: schedules, gauge potentials, Pauli coefficients,
spectra, and state propagation with and without the counterdiabatic term.
"""

import csv

import h5py
import numpy as np

from .physics import (
    adiabatic_hamiltonian,
    exact_gauge_potential,
    instantaneous_populations,
    pauli_coefficients,
    pauli_strings,
    propagate,
    total_hamiltonian,
)
from .training import data_root, distance_key, load_hamiltonians

# Bond distances (angstrom) of the H2 results shown in the article.
PAPER_DISTANCES = (1.0, 1.5, 2.0, 2.5)


def load_trajectory(distance: float = 1.0) -> dict:
    """Stored H2 prediction from the article's training runs at one distance.

    Returns a dictionary with ``time`` (10000,), ``schedule``, ``schedule_rate``,
    ``potential`` (10000, 4, 4) and ``coefficients`` (10000, 16), plus the
    derived matrices added by :func:`complete`.
    """
    with h5py.File(data_root() / "results/h2-trajectories.h5", "r") as handle:
        group = handle[distance_key(distance)]
        assert isinstance(group, h5py.Group)
        data: dict = {
            "distance": float(distance),
            "training_seconds": group.attrs["training_seconds"],
        }
        for name in ("time", "schedule", "schedule_rate", "potential", "coefficients"):
            dataset = group[name]
            assert isinstance(dataset, h5py.Dataset)
            # Stored in single precision; analyse in double precision.
            value = np.asarray(dataset[()])
            data[name] = value.astype(np.complex128 if np.iscomplexobj(value) else np.float64)
    return complete(data, "h2", distance)


def load_history() -> dict[str, np.ndarray]:
    """Per-update loss terms of the stored 500001-update H2 run at 1.0 angstrom."""
    with h5py.File(data_root() / "results/h2-trajectories.h5", "r") as handle:
        group = handle["history"]
        assert isinstance(group, h5py.Group)
        return {name: np.asarray(dataset[()], dtype=np.float64) for name, dataset in group.items()}


def load_scalability() -> list[dict]:
    """Rows of ``scalability.csv``: final losses and wall time of 46 training runs."""
    with (data_root() / "results/scalability.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    # csv returns strings; convert every column except the system name.
    for row in rows:
        for key, value in row.items():
            if key != "system":
                row[key] = int(value) if key == "collocation" else float(value)
    return rows


def complete(data: dict, system: str, distance: float) -> dict:
    """Add Hamiltonians and derived matrices to a trajectory dictionary.

    Adds ``h_initial``, ``h_final``, the adiabatic Hamiltonian ``adiabatic``,
    the total ``hamiltonian``, the counterdiabatic term ``counterdiabatic``
    (d lambda/dt times A_CD), and the ``exact_potential`` for the same schedule.
    """
    h_initial, h_final = load_hamiltonians(system, distance)
    data["h_initial"], data["h_final"] = h_initial, h_final
    data["adiabatic"] = adiabatic_hamiltonian(data["schedule"], h_initial, h_final)
    data["hamiltonian"] = total_hamiltonian(
        data["schedule"], data["schedule_rate"], data["potential"], h_initial, h_final
    )
    data["counterdiabatic"] = data["schedule_rate"][:, None, None] * data["potential"]
    data["exact_potential"] = exact_gauge_potential(data["schedule"], h_initial, h_final)
    return data


def counterdiabatic_coefficients(data: dict) -> dict[str, np.ndarray]:
    """Pauli coefficients of H_CD = d(lambda)/dt A_CD.

    ``learned`` uses the network's real coefficients C; ``exact`` projects the
    exact gauge potential of eq. (B.7) with the same schedule.
    """
    rate = data["schedule_rate"][:, None]
    return {
        "learned": rate * data["coefficients"],
        "exact": rate * pauli_coefficients(data["exact_potential"]).real,
    }


def dynamics(data: dict, stride: int = 5) -> dict[str, dict[str, np.ndarray]]:
    """Propagate the ground state of H_initial with and without the CD term.

    Returns, for each protocol, the times, instantaneous-ground-state
    population, and energy expectation <H_AD(t)>.
    """
    # Every ``stride``-th stored time: 2000 points for the default stride of 5.
    times = data["time"][::stride]
    adiabatic = data["adiabatic"][::stride]
    # Start in the ground state of H_initial (eigh sorts eigenvalues ascending).
    initial_state = np.linalg.eigh(data["h_initial"])[1][:, 0]
    result = {}
    for name, matrices in (
        ("counterdiabatic", data["hamiltonian"][::stride]),
        ("adiabatic", adiabatic),
    ):
        states = propagate(times, matrices, initial_state)
        # The learned A_CD is only approximately Hermitian, so the norm can drift
        # slightly; normalise expectation values and populations by it.
        norms = np.einsum("ni,ni->n", states.conj(), states).real
        energy = np.einsum("ni,nij,nj->n", states.conj(), adiabatic, states).real / norms
        populations = instantaneous_populations(states, adiabatic)
        result[name] = {
            "time": times,
            "ground_population": populations[:, 0] / norms,
            "energy": energy,
            "final_energy": float(
                np.vdot(states[-1], data["h_final"] @ states[-1]).real / norms[-1]
            ),
            "final_norm": float(norms[-1]),
        }
    return result


def energy_table(distances=PAPER_DISTANCES) -> list[dict]:
    """Recomputed Table 1: final spectrum of H(t=1) and the propagated energy."""
    rows = []
    for distance in distances:
        data = load_trajectory(distance)
        # General eigensolver: H(t) need not be exactly Hermitian.
        spectrum = np.sort(np.linalg.eigvals(data["hamiltonian"][-1]).real)
        exact = float(np.linalg.eigvalsh(data["h_final"])[0])
        runs = dynamics(data)
        cd, bare = runs["counterdiabatic"], runs["adiabatic"]
        rows.append(
            {
                "distance": distance,
                "spectrum": spectrum.tolist(),
                "exact_ground_energy": exact,
                "final_energy": cd["final_energy"],
                "relative_error_percent": 100 * abs((cd["final_energy"] - exact) / exact),
                "ground_population": float(cd["ground_population"][-1]),
                "adiabatic_final_energy": bare["final_energy"],
                "adiabatic_ground_population": float(bare["ground_population"][-1]),
            }
        )
    return rows


def physics_checks(data: dict) -> dict[str, float]:
    """Endpoint, Hermiticity, and gauge-potential diagnostics for one trajectory.

    ``xy_yx_relative_error`` compares the learned and exact XY and YX
    coefficients of H_CD over the interior 80% of the evolution, where the
    boundary layers at t = 0 and t = 1 do not dominate.
    """
    potential, hamiltonian = data["potential"], data["hamiltonian"]
    names = pauli_strings(2)
    pair = [names.index("XY"), names.index("YX")]
    coefficients = counterdiabatic_coefficients(data)
    interior = slice(len(data["time"]) // 10, -len(data["time"]) // 10)
    learned, exact = coefficients["learned"][interior, pair], coefficients["exact"][interior, pair]
    return {
        "schedule_start": float(data["schedule"][0]),
        "schedule_end": float(data["schedule"][-1]),
        "initial_hamiltonian_error": float(np.abs(hamiltonian[0] - data["h_initial"]).max()),
        "final_hamiltonian_error": float(np.abs(hamiltonian[-1] - data["h_final"]).max()),
        "counterdiabatic_start": float(np.abs(data["counterdiabatic"][0]).max()),
        "counterdiabatic_end": float(np.abs(data["counterdiabatic"][-1]).max()),
        "counterdiabatic_peak": float(np.abs(data["counterdiabatic"]).max()),
        "hermiticity_error": float(np.abs(potential - potential.conj().transpose(0, 2, 1)).max()),
        "xy_yx_relative_error": float(np.linalg.norm(learned - exact) / np.linalg.norm(exact)),
    }
