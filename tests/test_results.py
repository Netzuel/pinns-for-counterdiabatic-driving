import json

import numpy as np
import pytest

from quantum_pinn.analysis import (
    counterdiabatic_coefficients,
    dynamics,
    load_history,
    load_scalability,
    load_trajectory,
    physics_checks,
)
from quantum_pinn.training import data_root


@pytest.fixture(scope="module")
def trajectory():
    return load_trajectory(1.0)


def test_counterdiabatic_term_vanishes_at_both_endpoints(trajectory):
    checks = physics_checks(trajectory)
    assert checks["counterdiabatic_start"] < 1e-3 * checks["counterdiabatic_peak"]
    assert checks["counterdiabatic_end"] < 1e-3 * checks["counterdiabatic_peak"]
    learned = counterdiabatic_coefficients(trajectory)["learned"]
    assert np.abs(learned[[0, -1]]).max() < 1e-4


def test_learned_coefficients_follow_exact_gauge_potential(trajectory):
    assert physics_checks(trajectory)["xy_yx_relative_error"] < 0.02


def test_counterdiabatic_driving_keeps_ground_state(trajectory):
    runs = dynamics(trajectory, stride=10)
    exact = np.linalg.eigvalsh(trajectory["h_final"])[0]
    assert runs["counterdiabatic"]["ground_population"].min() > 0.9999
    assert runs["counterdiabatic"]["final_energy"] == pytest.approx(exact, abs=1e-6)
    assert runs["adiabatic"]["ground_population"][-1] < 0.98


def test_stored_history_and_scalability_tables():
    history = load_history()
    assert len(history["total"]) == 500_001
    assert history["total"][-1] < 1e-3 * history["total"][0]
    rows = load_scalability()
    assert len(rows) == 46
    assert {r["system"] for r in rows} == {"h2", "lih"}


def test_evaluation_record_matches_recomputation(trajectory):
    record = json.loads((data_root() / "figures/evaluation.json").read_text())
    stored = record["distances"]["1.0"]
    assert stored["checks"] == pytest.approx(physics_checks(trajectory))
    spectrum = np.sort(np.linalg.eigvals(trajectory["hamiltonian"][-1]).real)
    assert stored["final_spectrum_real"] == pytest.approx(spectrum.tolist(), abs=1e-6)
