"""Reproduce the article's figures and numerical results from the bundled data.

Writes light and dark SVG figures and ``evaluation.json`` to ``datasets/figures``:

    python src/figures.py

Every figure is computed from ``datasets/`` without training. The loss history
(500001 updates) is drawn at log-spaced updates; all other curves use every
stored sample.
"""

import argparse
import json
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import LogLocator, NullFormatter  # noqa: E402
import numpy as np  # noqa: E402

from quantum_pinn.analysis import (  # noqa: E402
    PAPER_DISTANCES,
    counterdiabatic_coefficients,
    dynamics,
    load_history,
    load_scalability,
    load_trajectory,
    physics_checks,
)
from quantum_pinn.physics import pauli_coefficients, pauli_strings  # noqa: E402
from quantum_pinn.training import data_root, load_hamiltonians  # noqa: E402

# Colour tokens matched to GitHub's light and dark page backgrounds. Series
# colours are the Okabe-Ito palette, lightened for the dark theme.
THEMES = {
    "light": {
        "background": "#ffffff",
        "ink": "#1f2328",
        "muted": "#59636e",
        "grid": "#d1d9e0",
        "series": ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"],
        "level": "#8c959f",
    },
    "dark": {
        "background": "#0d1117",
        "ink": "#e6edf3",
        "muted": "#9198a1",
        "grid": "#30363d",
        "series": ["#56B4E9", "#F28B50", "#3CC79A", "#E59CCB", "#F0C24B"],
        "level": "#6e7681",
    },
}
WIDTH = 7.2  # figure width in inches, sized for the README column
NAMES = pauli_strings(2)
XY, YX = NAMES.index("XY"), NAMES.index("YX")


def style(theme):
    """Matplotlib rc parameters for a theme: STIX fonts, colours, vector text."""
    tokens = THEMES[theme]
    return {
        "font.family": "STIXGeneral",
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.facecolor": tokens["background"],
        "axes.facecolor": tokens["background"],
        "savefig.facecolor": tokens["background"],
        "text.color": tokens["ink"],
        "axes.labelcolor": tokens["ink"],
        "axes.edgecolor": tokens["muted"],
        "xtick.color": tokens["muted"],
        "ytick.color": tokens["muted"],
        "xtick.labelcolor": tokens["ink"],
        "ytick.labelcolor": tokens["ink"],
        "grid.color": tokens["grid"],
        "grid.linewidth": 0.5,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.linewidth": 0.6,
        "lines.linewidth": 1.2,
        "legend.frameon": False,
        "svg.fonttype": "path",
        "svg.hashsalt": "quantum-pinn",
        "path.simplify": True,
        "path.simplify_threshold": 0.05,
    }


def legend_row(fig, area, handles, labels, columns):
    """Place a legend inside a dedicated, axis-free row of the layout."""
    ax = fig.add_subplot(area)
    ax.axis("off")
    return ax.legend(
        handles,
        labels,
        loc="center",
        ncol=columns,
        handlelength=2.2,
        columnspacing=1.4,
        borderaxespad=0,
    )


def check_layout(fig, data_axes, legends):
    """Fail if a legend, padded by 3 pt, touches any data Axes or its labels.

    Exports use a tight bounding box, so legends near the edge are never clipped.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()  # type: ignore[attr-defined]
    pad = renderer.points_to_pixels(3)
    for legend in legends:
        extent = legend.get_window_extent(renderer)
        box = extent.padded(pad)
        for ax in data_axes:
            if box.overlaps(ax.get_tightbbox(renderer)):
                raise RuntimeError("legend overlaps a data panel")


def audit_svg(path):
    """Reject an SVG that contains raster images or embedded foreign content."""
    root = ET.parse(path).getroot()
    for element in root.iter():
        tag = element.tag.rsplit("}", 1)[-1]
        if tag in {"image", "foreignObject"}:
            raise RuntimeError(f"{path} contains raster or foreign content")


# Leave out the RDF <metadata> block (creator, date, type, format) that Matplotlib adds by default.
SVG_METADATA = {"Creator": None, "Date": None, "Format": None, "Type": None}


def save(fig, output, name, theme, data_axes, legends):
    """Check the layout, write ``{name}-{theme}.svg``, and audit the file."""
    check_layout(fig, data_axes, legends)
    path = Path(output) / f"{name}-{theme}.svg"
    fig.savefig(path, bbox_inches="tight", pad_inches=0.04, metadata=SVG_METADATA)
    plt.close(fig)
    audit_svg(path)
    return path


def label_panel(ax, text):
    """Panel label such as "(a) d = 1.0 A" at the top left of an Axes."""
    ax.set_title(text, loc="left", fontsize=9, pad=3)


# --- Figures -----------------------------------------------------------------------


def figure_counterdiabatic(trajectories, output, theme):
    """Learned and exact XY/YX coefficients of H_CD, with endpoint zooms (README)."""
    tokens = THEMES[theme]
    blue, red = tokens["series"][:2]
    fig = plt.figure(figsize=(WIDTH, 3.55))
    grid = GridSpec(
        2,
        4,
        figure=fig,
        height_ratios=[0.12, 1],
        hspace=0.25,
        wspace=0.32,
        left=0.07,
        right=0.995,
        top=0.995,
        bottom=0.11,
    )
    data_axes = []
    for column, data in enumerate(trajectories):
        inner = GridSpecFromSubplotSpec(
            2, 2, grid[1, column], height_ratios=[2.1, 1], hspace=0.5, wspace=0.18
        )
        main = fig.add_subplot(inner[0, :])
        start = fig.add_subplot(inner[1, 0])
        end = fig.add_subplot(inner[1, 1], sharey=start)
        coefficients = counterdiabatic_coefficients(data)
        t = data["time"]
        windows = (slice(None), t <= 0.0105, t >= 0.9895)
        for ax, window in zip((main, start, end), windows):
            for index, color in ((XY, blue), (YX, red)):
                ax.plot(
                    t[window],
                    coefficients["exact"][window, index],
                    color=tokens["muted"],
                    lw=1.0,
                    ls=(0, (3, 2)),
                    zorder=2,
                )
                ax.plot(
                    t[window], coefficients["learned"][window, index], color=color, lw=1.2, zorder=3
                )
            ax.axhline(0, color=tokens["muted"], lw=0.5, zorder=1)
        main.set_xlim(0, 1)
        main.set_xticks([0, 0.5, 1])
        start.set_xlim(0, 0.01)
        start.set_xticks([0, 0.01])
        start.set_xticklabels(["0", "0.01"], fontsize=7)
        start.get_xticklabels()[1].set_horizontalalignment("right")
        end.set_xlim(0.99, 1)
        end.set_xticks([0.99, 1])
        end.set_xticklabels(["0.99", "1"], fontsize=7)
        end.get_xticklabels()[0].set_horizontalalignment("left")
        end.tick_params(labelleft=False)
        label_panel(main, f"({'abcd'[column]}) $d={data['distance']:.1f}$ Å")
        if column == 0:
            main.set_ylabel(r"$\dot\lambda\,C_\mu$ (Ha)")
            start.set_ylabel("zoom")
        start.set_xlabel("start")
        end.set_xlabel("end")
        main.set_xlabel("$t$", labelpad=1)
        data_axes += [main, start, end]
    handles = [
        Line2D([], [], color=blue),
        Line2D([], [], color=red),
        Line2D([], [], color=tokens["muted"], ls=(0, (3, 2))),
    ]
    labels = [
        r"PINN $\dot\lambda C_{XY}$",
        r"PINN $\dot\lambda C_{YX}$",
        r"exact $\dot\lambda\,\mathrm{Tr}(P_\mu A_\lambda)/4$",
    ]
    legend = legend_row(fig, grid[0, :], handles, labels, 3)
    return save(fig, output, "counterdiabatic", theme, data_axes, [legend])


def figure_energies(trajectories, runs, output, theme):
    """Adiabatic spectrum, energy, and ground population with and without CD."""
    tokens = THEMES[theme]
    blue, red = tokens["series"][:2]
    fig = plt.figure(figsize=(WIDTH, 3.9))
    grid = GridSpec(
        3,
        4,
        figure=fig,
        height_ratios=[0.12, 1.25, 0.8],
        hspace=0.38,
        wspace=0.3,
        left=0.08,
        right=0.995,
        top=0.995,
        bottom=0.1,
    )
    data_axes = []
    for column, (data, run) in enumerate(zip(trajectories, runs)):
        top = fig.add_subplot(grid[1, column])
        bottom = fig.add_subplot(grid[2, column], sharex=top)
        levels = np.linalg.eigvalsh(data["adiabatic"])
        top.plot(data["time"], levels, color=tokens["level"], lw=0.9, zorder=2)
        for name, color, dash in (("adiabatic", red, (0, (4, 2))), ("counterdiabatic", blue, "-")):
            top.plot(run[name]["time"], run[name]["energy"], color=color, ls=dash, zorder=3)
            bottom.plot(
                run[name]["time"], run[name]["ground_population"], color=color, ls=dash, zorder=3
            )
        top.set_xlim(0, 1)
        top.tick_params(labelbottom=False)
        bottom.set_ylim(0.55, 1.02)
        bottom.set_xticks([0, 0.5, 1])
        bottom.set_xlabel("$t$", labelpad=1)
        label_panel(top, f"({'abcd'[column]}) $d={data['distance']:.1f}$ Å")
        if column == 0:
            top.set_ylabel("energy (Ha)")
            bottom.set_ylabel("ground-state\npopulation")
        data_axes += [top, bottom]
    handles = [
        Line2D([], [], color=tokens["level"]),
        Line2D([], [], color=blue),
        Line2D([], [], color=red, ls=(0, (4, 2))),
    ]
    labels = [
        r"eigenvalues of $H_{\mathrm{AD}}(t)$",
        r"with $\dot\lambda A_{\mathrm{CD}}$ (PINN)",
        r"without $\dot\lambda A_{\mathrm{CD}}$",
    ]
    legend = legend_row(fig, grid[0, :], handles, labels, 3)
    return save(fig, output, "energies", theme, data_axes, [legend])


def figure_training(history, trajectories, output, theme):
    """Loss history of the 1.0 A run and learned schedules (article Figs. 2-3)."""
    tokens = THEMES[theme]
    colors = tokens["series"]
    fig = plt.figure(figsize=(WIDTH, 3.0))
    grid = GridSpec(
        2,
        3,
        figure=fig,
        height_ratios=[0.3, 1],
        width_ratios=[1.3, 1, 1],
        hspace=0.2,
        wspace=0.42,
        left=0.08,
        right=0.995,
        top=0.995,
        bottom=0.15,
    )
    # 3000 log-spaced update indices keep the SVG small; the stored history has
    # every update and is used unchanged.
    steps = np.unique(np.geomspace(1, len(history["total"]), 3000).astype(int)) - 1
    loss = fig.add_subplot(grid[1, 0])
    series = [
        ("total", "total", tokens["ink"], "-"),
        ("schedule_rate", r"$\mathcal{L}_{\mathrm{Ad}}$", colors[0], "-"),
        ("least_action", r"$\mathcal{L}_{\mathrm{LA}}$", colors[1], "-"),
        ("pauli", r"$\mathcal{L}_{C}$", colors[2], "-"),
        ("endpoints", r"$\mathcal{L}_{\mathrm{IC}}+\mathcal{L}_{\mathrm{FC}}$", colors[3], "-"),
        ("hermiticity", r"$|A-A^\dagger|^2$ (monitor)", tokens["muted"], (0, (3, 2))),
    ]
    values = dict(history, endpoints=history["initial"] + history["final"])
    handles = []
    for key, label, color, dash in series:
        y = np.where(values[key][steps] > 0, values[key][steps], np.nan)
        (line,) = loss.plot(steps + 1, y, color=color, ls=dash, lw=1.0, label=label)
        handles.append(line)
    loss.set_xscale("log")
    loss.set_yscale("log")
    loss.set_xlim(1, len(history["total"]))
    loss.yaxis.set_major_locator(LogLocator(numticks=8))
    loss.yaxis.set_minor_formatter(NullFormatter())
    loss.set_xlabel("Adam update")
    loss.set_ylabel("loss")
    label_panel(loss, r"(a) losses, $d=1.0$ Å")
    legend_a = legend_row(fig, grid[0, 0], handles, [h.get_label() for h in handles], 3)

    schedule = fig.add_subplot(grid[1, 1])
    rate = fig.add_subplot(grid[1, 2], sharex=schedule)
    lines = []
    for index, data in enumerate(trajectories):
        (line,) = schedule.plot(data["time"], data["schedule"], color=colors[index], lw=1.0)
        rate.plot(data["time"], data["schedule_rate"], color=colors[index], lw=1.0)
        lines.append(line)
    for ax in (schedule, rate):
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.5, 1])
        ax.set_xlabel("$t$")
    schedule.plot([0, 1], [0, 1], color=tokens["muted"], lw=0.8, ls=(0, (1, 2)), zorder=1)
    schedule.set_ylabel(r"$\lambda(t)$")
    rate.set_ylabel(r"$\dot\lambda(t)$")
    label_panel(schedule, r"(b) schedule")
    label_panel(rate, r"(c) schedule rate")
    labels = [f"{data['distance']:.1f} Å" for data in trajectories]
    legend_b = legend_row(fig, grid[0, 1:], lines, labels, 2)
    return save(fig, output, "training", theme, [loss, schedule, rate], [legend_a, legend_b])


def figure_pauli(trajectories, output, theme):
    """Time-averaged Pauli coefficients against the exact potential (Fig. 7)."""
    tokens = THEMES[theme]
    colors = tokens["series"]
    means = np.array([data["coefficients"].mean(axis=0) for data in trajectories])
    exact = np.array(
        [pauli_coefficients(data["exact_potential"]).real.mean(axis=0) for data in trajectories]
    )
    # Sort Pauli strings by their mean magnitude over the four distances.
    order = np.argsort(-np.abs(means).mean(axis=0))
    fig = plt.figure(figsize=(WIDTH, 2.6))
    grid = GridSpec(
        2,
        1,
        figure=fig,
        height_ratios=[0.14, 1],
        hspace=0.12,
        left=0.08,
        right=0.995,
        top=0.995,
        bottom=0.16,
    )
    ax = fig.add_subplot(grid[1])
    width = 0.2
    x = np.arange(len(NAMES))
    handles = []
    for index, data in enumerate(trajectories):
        offset = (index - 1.5) * width
        bars = ax.bar(
            x + offset,
            means[index, order],
            width,
            color=colors[index],
            zorder=3,
            label=f"PINN, {data['distance']:.1f} Å",
        )
        ax.scatter(
            x + offset,
            exact[index, order],
            marker="_",
            s=40,
            color=tokens["ink"],
            linewidths=1.2,
            zorder=4,
        )
        handles.append(bars)
    handles.append(Line2D([], [], color=tokens["ink"], marker="_", ls="none", ms=8, mew=1.2))
    ax.axhline(0, color=tokens["muted"], lw=0.6, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels([NAMES[i] for i in order], fontfamily="DejaVu Sans Mono", fontsize=7.5)
    ax.set_xlim(-0.6, len(NAMES) - 0.4)
    ax.set_ylabel(r"time average of $C_\mu$")
    ax.grid(axis="x", visible=False)
    labels = [h.get_label() for h in handles[:-1]] + ["exact gauge potential"]
    legend = legend_row(fig, grid[0], handles, labels, 5)
    return save(fig, output, "pauli-coefficients", theme, [ax], [legend])


def figure_spectra(trajectories, output, theme):
    """Eigenvalues of H(t) and of H_AD(t) at four distances (Fig. 5)."""
    tokens = THEMES[theme]
    blue, red = tokens["series"][:2]
    fig = plt.figure(figsize=(WIDTH, 3.7))
    grid = GridSpec(
        3,
        4,
        figure=fig,
        height_ratios=[0.12, 1.2, 0.8],
        hspace=0.38,
        wspace=0.32,
        left=0.085,
        right=0.995,
        top=0.995,
        bottom=0.1,
    )
    data_axes = []
    for column, data in enumerate(trajectories):
        top = fig.add_subplot(grid[1, column])
        bottom = fig.add_subplot(grid[2, column], sharex=top)
        values = np.linalg.eigvals(data["hamiltonian"])
        values = np.take_along_axis(values, np.argsort(np.real(values), axis=1), axis=1)
        top.plot(
            data["time"],
            np.linalg.eigvalsh(data["adiabatic"]),
            color=red,
            lw=1.0,
            ls=(0, (4, 2)),
            zorder=2,
        )
        top.plot(data["time"], np.real(values), color=blue, lw=1.1, zorder=3)
        bottom.plot(data["time"], np.imag(values) * 1e3, color=blue, lw=0.9, zorder=3)
        top.set_xlim(0, 1)
        top.tick_params(labelbottom=False)
        bottom.set_xticks([0, 0.5, 1])
        bottom.set_xlabel("$t$", labelpad=1)
        label_panel(top, f"({'abcd'[column]}) $d={data['distance']:.1f}$ Å")
        if column == 0:
            top.set_ylabel(r"Re $E_n$ (Ha)")
            bottom.set_ylabel(r"Im $E_n$ ($10^{-3}$ Ha)")
        data_axes += [top, bottom]
    handles = [Line2D([], [], color=blue), Line2D([], [], color=red, ls=(0, (4, 2)))]
    labels = [
        r"eigenvalues of $H(t)=H_{\mathrm{AD}}+\dot\lambda A_{\mathrm{CD}}$",
        r"eigenvalues of $H_{\mathrm{AD}}(t)$",
    ]
    legend = legend_row(fig, grid[0, :], handles, labels, 2)
    return save(fig, output, "spectra", theme, data_axes, [legend])


def figure_matrix(data, output, theme):
    """Real and imaginary parts of each element of H_CD at 1.0 A (Figs. 4, 6)."""
    tokens = THEMES[theme]
    blue, red = tokens["series"][:2]
    matrix = data["counterdiabatic"]
    limit = 1.08 * np.abs(matrix).max()
    fig = plt.figure(figsize=(WIDTH, 4.8))
    grid = GridSpec(
        5,
        4,
        figure=fig,
        height_ratios=[0.16, 1, 1, 1, 1],
        hspace=0.42,
        wspace=0.14,
        left=0.08,
        right=0.995,
        top=0.995,
        bottom=0.08,
    )
    data_axes = []
    for row in range(4):
        for column in range(4):
            ax = fig.add_subplot(grid[row + 1, column])
            ax.plot(data["time"], matrix[:, row, column].real, color=blue, lw=1.1, zorder=3)
            ax.plot(
                data["time"],
                matrix[:, row, column].imag,
                color=red,
                lw=1.0,
                ls=(0, (4, 2)),
                zorder=3,
            )
            ax.set_xlim(0, 1)
            ax.set_ylim(-limit, limit)
            ax.set_xticks([0, 0.5, 1])
            ax.set_xticklabels(["0", "0.5", "1"])
            ax.set_title(f"$j={row},\\ k={column}$", fontsize=8, pad=2, color=tokens["muted"])
            ax.tick_params(labelbottom=row == 3, labelleft=column == 0)
            if row == 3:
                ax.set_xlabel("$t$", labelpad=1)
            if column == 0:
                ax.set_ylabel("Ha")
            data_axes.append(ax)
    handles = [Line2D([], [], color=blue), Line2D([], [], color=red, ls=(0, (4, 2)))]
    labels = [
        r"Re $[\dot\lambda A_{\mathrm{CD}}]_{jk}$",
        r"Im $[\dot\lambda A_{\mathrm{CD}}]_{jk}$",
    ]
    legend = legend_row(fig, grid[0, :], handles, labels, 2)
    return save(fig, output, "counterdiabatic-matrix", theme, data_axes, [legend])


def figure_scalability(rows, output, theme):
    """Final loss and wall time against collocation points (Fig. 8)."""
    tokens = THEMES[theme]
    colors = tokens["series"]
    groups = sorted({(row["system"], row["distance"]) for row in rows})
    fig = plt.figure(figsize=(WIDTH, 2.7))
    grid = GridSpec(
        2,
        2,
        figure=fig,
        height_ratios=[0.2, 1],
        hspace=0.16,
        wspace=0.27,
        left=0.085,
        right=0.995,
        top=0.995,
        bottom=0.16,
    )
    loss = fig.add_subplot(grid[1, 0])
    time = fig.add_subplot(grid[1, 1], sharex=loss)
    markers = "osD^vP"
    handles = []
    for index, (system, distance) in enumerate(groups):
        part = sorted(
            (r for r in rows if (r["system"], r["distance"]) == (system, distance)),
            key=lambda r: r["collocation"],
        )
        n = [r["collocation"] for r in part]
        color = colors[index % len(colors)] if system == "h2" else tokens["ink"]
        label = f"H$_2$, {distance:.1f} Å" if system == "h2" else f"LiH, {distance:.1f} Å"
        kwargs: dict[str, Any] = dict(
            color=color, marker=markers[index], ms=3.5, lw=1.0, label=label
        )
        (line,) = loss.plot(n, [r["total_loss"] for r in part], **kwargs)
        time.plot(n, [r["training_seconds"] / 3600 for r in part], **kwargs)
        handles.append(line)
    for ax in (loss, time):
        ax.set_xscale("log", base=2)
        ax.set_xlabel(r"collocation points $N_F$")
    time.set_yscale("log")
    time.set_yticks([2, 5, 10, 20, 50])
    time.set_yticklabels(["2", "5", "10", "20", "50"])
    time.yaxis.set_minor_formatter(NullFormatter())
    loss.set_ylabel("final total loss")
    time.set_ylabel("recorded training time (h)")
    label_panel(loss, "(a) final loss")
    label_panel(time, "(b) wall time")
    legend = legend_row(fig, grid[0, :], handles, [h.get_label() for h in handles], 6)
    return save(fig, output, "scalability", theme, [loss, time], [legend])


# --- Numerical record -------------------------------------------------------------


def evaluation_record(trajectories, runs):
    """Numbers behind the figures and Table 1, written to evaluation.json."""
    record = {"distances": {}, "notes": []}
    for data, run in zip(trajectories, runs):
        spectrum = np.linalg.eigvals(data["hamiltonian"][-1])
        exact = float(np.linalg.eigvalsh(data["h_final"])[0])
        cd, bare = run["counterdiabatic"], run["adiabatic"]
        record["distances"][f"{data['distance']:.1f}"] = {
            "checks": physics_checks(data),
            "final_spectrum_real": np.sort(np.real(spectrum)).tolist(),
            "final_spectrum_imag_max": float(np.abs(np.imag(spectrum)).max()),
            "exact_ground_energy": exact,
            "final_energy": cd["final_energy"],
            "relative_error_percent": 100 * abs((cd["final_energy"] - exact) / exact),
            "ground_population": float(cd["ground_population"][-1]),
            "minimum_ground_population": float(cd["ground_population"].min()),
            "without_cd_final_energy": bare["final_energy"],
            "without_cd_ground_population": float(bare["ground_population"][-1]),
            "time_averaged_coefficients": dict(
                zip(NAMES, np.round(data["coefficients"].mean(axis=0), 6).tolist())
            ),
        }
    data = trajectories[0]
    values, vectors = np.linalg.eig(data["hamiltonian"][-1])
    order = np.argsort(np.real(values))
    vectors = vectors[:, order]
    # Fix the arbitrary phase of each eigenvector: largest component real, positive.
    vectors = vectors * np.exp(
        -1j * np.angle(vectors[np.abs(vectors).argmax(axis=0), np.arange(4)])
    )
    record["final_eigenvectors_d1.0"] = {
        "real": np.round(np.real(vectors).T, 6).tolist(),
        "imag_max": float(np.abs(np.imag(vectors)).max()),
    }
    record["notes"] = [
        "Energies in hartree; hbar = 1; t in [0, 1].",
        "Propagation: DOP853, rtol 1e-10, atol 1e-12, piecewise-linear H(t) on every fifth "
        "stored sample (2000 points), initial state = ground state of H_initial.",
        "relative_error_percent compares <H_final> at t = 1 with the exact ground energy.",
        "Eigenvectors are columns of H(t = 1) sorted by real eigenvalue, phase-fixed so the "
        "largest component is real and positive.",
    ]
    return record


def build(output, themes=("light", "dark")):
    """Compute every result once and draw all figures in each theme."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    trajectories = [load_trajectory(d) for d in PAPER_DISTANCES]
    runs = [dynamics(data) for data in trajectories]
    history = load_history()
    rows = load_scalability()
    written = []
    for theme in themes:
        with plt.rc_context(style(theme)):
            written += [
                figure_counterdiabatic(trajectories, output, theme),
                figure_energies(trajectories, runs, output, theme),
                figure_training(history, trajectories, output, theme),
                figure_pauli(trajectories, output, theme),
                figure_spectra(trajectories, output, theme),
                figure_matrix(trajectories[0], output, theme),
                figure_scalability(rows, output, theme),
            ]
    record = evaluation_record(trajectories, runs)
    (output / "evaluation.json").write_text(json.dumps(record, indent=2) + "\n")
    return written, record


def main():
    """Command-line entry point: ``python src/figures.py [--output DIR]``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=data_root() / "figures")
    parser.add_argument("--themes", nargs="+", choices=sorted(THEMES), default=["light", "dark"])
    args = parser.parse_args()
    load_hamiltonians()  # Fail early if the bundled data are absent
    written, _ = build(args.output, tuple(args.themes))
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
