import json
from typing import Any

import numpy as np
import pytest
import torch

from quantum_pinn import RunConfig, load_checkpoint, load_pretrained, train
from quantum_pinn.training import (
    available_distances,
    data_root,
    evaluate,
    experiments,
    main,
    pretrained_names,
)

SMALL: dict[str, Any] = dict(width=8, hidden_layers=2, collocation=16, steps=4, learning_rate=1e-3)


def test_short_training_decreases_loss_and_reloads(tmp_path):
    config = RunConfig(**SMALL, snapshot_steps=(2,))
    model, summary = train(config, tmp_path / "run")
    history = np.load(tmp_path / "run/history.npz")
    assert history["total"].shape == (4,)
    assert history["total"][-1] < history["total"][0]
    assert (tmp_path / "run/snapshots/step-0000002.pt").is_file()
    reloaded, loaded = load_checkpoint(tmp_path / "run/model.pt")
    assert loaded == config
    t = torch.linspace(0, 1, 5).reshape(-1, 1)
    assert torch.equal(model.predict(t)["potential"], reloaded.predict(t)["potential"])
    assert json.loads((tmp_path / "run/config.json").read_text())["steps"] == 4
    with pytest.raises(FileExistsError):
        train(config, tmp_path / "run")


def test_training_is_deterministic(tmp_path):
    config = RunConfig(**SMALL)
    first, _ = train(config, tmp_path / "a")
    second, _ = train(config, tmp_path / "b")
    for a, b in zip(first.parameters(), second.parameters()):
        assert torch.equal(a, b)


@pytest.mark.parametrize("name", ["h2-d1.0", "h2-d2.5", "lih-d1.0"])
def test_pretrained_models_meet_endpoint_conditions(name):
    model, config = load_pretrained(name)
    data = evaluate(model, config.system, config.distance, points=101)
    assert data["schedule"][0] < 1e-3 and data["schedule"][-1] > 1 - 1e-3
    from quantum_pinn import load_hamiltonians

    h_initial, h_final = load_hamiltonians(config.system, config.distance)
    assert np.abs(data["hamiltonian"][0] - h_initial).max() < 1e-3
    assert np.abs(data["hamiltonian"][-1] - h_final).max() < 1e-3


def test_catalogue_and_bundled_data_are_consistent():
    catalogue = experiments()
    assert len([k for k in catalogue if k.startswith("scalability/")]) == 46
    for options in catalogue.values():
        config = RunConfig(**options)
        assert config.distance in available_distances(config.system)
    assert set(pretrained_names()) == {
        "h2-d1.0",
        "h2-d1.5",
        "h2-d2.0",
        "h2-d2.5",
        "h2-d3.0",
        "lih-d1.0",
    }
    assert (data_root() / "results/h2-trajectories.h5").is_file()


def test_invalid_configuration_is_rejected():
    with pytest.raises(ValueError):
        RunConfig(system="he")
    with pytest.raises(ValueError):
        RunConfig(steps=3, snapshot_steps=(4,))
    with pytest.raises(ValueError):
        RunConfig(weights="custom")


def test_command_line_smoke_and_evaluate(tmp_path, capsys):
    assert main(["train", "--experiment", "smoke", "--smoke", "--output", str(tmp_path / "r")]) == 0
    assert json.loads(capsys.readouterr().out)["output"].endswith("r")
    assert main(["evaluate", str(tmp_path / "r/model.pt"), "--points", "11"]) == 0
    assert json.loads(capsys.readouterr().out)["system"] == "h2"
    assert main(["list"]) == 0
    assert "h2-d1.0" in json.loads(capsys.readouterr().out)["pretrained"]
