"""The tutorials are stored with their outputs; check them and re-execute them."""

from pathlib import Path

import nbformat
from nbclient import NotebookClient
import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "docs" / "examples"
NAMES = ["quickstart", "inference", "reproduce-results"]


@pytest.mark.parametrize("name", NAMES)
def test_stored_outputs_are_complete_and_error_free(name):
    notebook = nbformat.read(EXAMPLES / f"{name}.ipynb", as_version=4)
    code = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert all(cell.get("execution_count") for cell in code), "notebook was not fully executed"
    outputs = [output for cell in code for output in cell.get("outputs", [])]
    assert outputs and all(output.output_type != "error" for output in outputs)


@pytest.mark.parametrize("name", NAMES)
def test_tutorial_runs(name, tmp_path, monkeypatch):
    # Make the source package importable in the kernel without an installation.
    monkeypatch.setenv("PYTHONPATH", str(EXAMPLES.parents[1] / "src"))
    monkeypatch.setenv("MPLBACKEND", "Agg")
    notebook = nbformat.read(EXAMPLES / f"{name}.ipynb", as_version=4)
    NotebookClient(
        notebook,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": str(tmp_path)}},
    ).execute()
