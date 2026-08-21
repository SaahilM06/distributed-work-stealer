#!/usr/bin/env python3
"""Render the benchmark charts used by README.md.

Numbers are transcribed from results/*.txt rather than recomputed, so a chart can
never disagree with the report it illustrates. Re-run the benchmarks, update the
tables below, then re-run this.

Each chart is written twice — light and dark — because a README is read in both
themes and GitHub can serve the right one via <picture>. The dark variant is a
selected palette stepped for the dark surface, not an automatic inversion.

    python3 scripts/make_charts.py
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

OUT_DIR = "docs/img"

# ── Palette ──────────────────────────────────────────────────────────────────
# Validated with the dataviz skill's checker in both modes: categorical pair
# (blue/orange) and the ordinal percentile ramp (one hue, monotone lightness).
LIGHT = dict(
    surface="#fcfcfb",
    text="#0b0b0b",
    text_secondary="#52514e",
    grid="#e3e2df",
    accent="#2a78d6",       # categorical slot 1
    accent_2="#eb6834",     # categorical slot 2
    muted="#b8b7b2",        # de-emphasis gray for the emphasis form
    ramp=["#86b6ef", "#2a78d6", "#184f95"],   # p50 → p95 → p99
)
DARK = dict(
    surface="#1a1a19",
    text="#ffffff",
    text_secondary="#c3c2b7",
    grid="#3a3a37",
    accent="#3987e5",
    accent_2="#d95926",
    muted="#6b6a65",
    ramp=["#9ec5f4", "#3987e5", "#184f95"],
)

# ── Data (from results/phase12_onnx_bench.txt) ───────────────────────────────
POLICIES = ["none\n(static)", "random", "load-aware", "adaptive"]
ONNX_THROUGHPUT = [204, 483, 500, 484]              # req/s
ONNX_P50 = [985.8, 441.1, 412.3, 427.9]             # ms
ONNX_P95 = [1884.4, 786.8, 761.0, 783.7]
ONNX_P99 = [1938.8, 811.0, 780.3, 808.1]

# ── Data (from results/phase11_inference_bench.txt, simulated cost) ──────────
SIM_TIME = {"none": 168.4, "random": 154.1, "load-aware": 81.2, "adaptive": 82.2}
ONNX_TIME = {"none": 1959.7, "random": 827.8, "load-aware": 799.4, "adaptive": 826.2}

# ── Data (from results/phase6_bench.txt, single node, tracing off) ───────────
WORKERS = [1, 2, 4, 8]
WORKER_THROUGHPUT = [498503, 957190, 1817191, 2531298]   # tasks/s


def style_axes(ax, c, ylabel=None, xlabel=None):
    """Recessive grid and axes; the data should be the only assertive thing."""
    ax.set_facecolor(c["surface"])
    ax.figure.patch.set_facecolor(c["surface"])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(c["grid"])
    ax.tick_params(colors=c["text_secondary"], length=0, labelsize=10)
    ax.grid(axis="y", color=c["grid"], linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    if ylabel:
        ax.set_ylabel(ylabel, color=c["text_secondary"], fontsize=10, labelpad=10)
    if xlabel:
        ax.set_xlabel(xlabel, color=c["text_secondary"], fontsize=10, labelpad=10)


def title(ax, c, head, sub=None):
    # The subtitle sits just above the axes and the title is padded clear of it;
    # a smaller pad lets the two overlap at these font sizes.
    ax.set_title(head, color=c["text"], fontsize=13, fontweight="600",
                 loc="left", pad=34 if sub else 12)
    if sub:
        ax.text(0, 1.015, sub, transform=ax.transAxes, color=c["text_secondary"],
                fontsize=10, va="bottom", ha="left")


def save(fig, name, c):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=c["surface"])
    plt.close(fig)
    print("wrote", path)


# ── 1. Throughput by policy (the headline) ───────────────────────────────────
def chart_throughput(c, suffix):
    fig, ax = plt.subplots(figsize=(7.2, 4.0))

    # Emphasis form: the winner carries the accent hue, the rest recede. The story
    # is "stealing wins", not "here are four equal categories".
    best = ONNX_THROUGHPUT.index(max(ONNX_THROUGHPUT))
    colors = [c["muted"]] * len(POLICIES)
    colors[best] = c["accent"]
    colors[0] = c["accent_2"]   # the baseline is the thing being beaten

    bars = ax.bar(POLICIES, ONNX_THROUGHPUT, color=colors, width=0.62,
                  linewidth=0, zorder=3)
    for b, v in zip(bars, ONNX_THROUGHPUT):
        ax.text(b.get_x() + b.get_width() / 2, v + 12, f"{v:,}",
                ha="center", va="bottom", color=c["text"], fontsize=11, fontweight="600")

    ax.set_ylim(0, max(ONNX_THROUGHPUT) * 1.18)
    style_axes(ax, c, ylabel="requests / second")
    title(ax, c, "Work stealing raises inference throughput 2.45×",
          "400 requests through a real MobileNetV2 · 3 nodes · higher is better")
    save(fig, f"throughput{suffix}.png", c)


# ── 2. Latency percentiles by policy ─────────────────────────────────────────
def chart_latency(c, suffix):
    fig, ax = plt.subplots(figsize=(7.6, 4.2))

    x = range(len(POLICIES))
    w = 0.26
    series = [("p50", ONNX_P50, c["ramp"][0]),
              ("p95", ONNX_P95, c["ramp"][1]),
              ("p99", ONNX_P99, c["ramp"][2])]

    for i, (label, vals, col) in enumerate(series):
        off = (i - 1) * w
        # 2px-equivalent gap between adjacent bars comes from w vs the 0.28 pitch.
        ax.bar([xi + off for xi in x], vals, width=w - 0.02, label=label,
               color=col, linewidth=0, zorder=3)

    # Direct-label only the tail: a number on every bar is noise.
    for i, v in enumerate(ONNX_P99):
        ax.text(i + w, v + 30, f"{v:.0f}", ha="center", va="bottom",
                color=c["text"], fontsize=9.5, fontweight="600")

    ax.set_xticks(list(x))
    ax.set_xticklabels(POLICIES)
    ax.set_ylim(0, max(ONNX_P99) * 1.16)
    style_axes(ax, c, ylabel="end-to-end latency (ms)")
    title(ax, c, "The tail improves most: p99 drops 60%",
          "per-request latency across all four pipeline stages · lower is better")

    leg = ax.legend(frameon=False, loc="upper right", fontsize=10, ncol=3,
                    handlelength=1.2, columnspacing=1.4)
    for t in leg.get_texts():
        t.set_color(c["text_secondary"])
    save(fig, f"latency{suffix}.png", c)


# ── 3. Task granularity changes which policy wins ────────────────────────────
def chart_granularity(c, suffix):
    fig, ax = plt.subplots(figsize=(7.2, 4.0))

    names = ["random", "load-aware", "adaptive"]
    # Speedup against the static baseline. A ratio puts two workloads of very
    # different absolute cost on one axis honestly — the alternative would be a
    # second y-scale, which is never acceptable.
    sim = [SIM_TIME["none"] / SIM_TIME[n] for n in names]
    onnx = [ONNX_TIME["none"] / ONNX_TIME[n] for n in names]

    x = range(len(names))
    w = 0.34
    b1 = ax.bar([xi - w / 2 for xi in x], sim, width=w - 0.02,
                label="simulated cost (~0.3 ms/task)", color=c["accent"],
                linewidth=0, zorder=3)
    b2 = ax.bar([xi + w / 2 for xi in x], onnx, width=w - 0.02,
                label="real MobileNetV2 (~20 ms/task)", color=c["accent_2"],
                linewidth=0, zorder=3)

    for bars, vals in ((b1, sim), (b2, onnx)):
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.03, f"{v:.2f}×",
                    ha="center", va="bottom", color=c["text"], fontsize=10)

    # Reference line drawn ABOVE the bars: at zorder 2 the bars hid it exactly where
    # it mattered.
    ax.axhline(1.0, color=c["text_secondary"], linewidth=1.1,
               linestyle=(0, (5, 4)), zorder=5)

    ax.set_xticks(list(x))
    ax.set_xticklabels(names)
    ax.set_xlim(-0.55, len(names) - 0.45)
    top = max(sim + onnx) * 1.28
    ax.set_ylim(0, top)

    # Label the reference line just outside the plot: every in-plot position either
    # sat on a bar or on a value label, since all six bars clear the baseline.
    ax.text(1.012, 1.0 / top, "static\nbaseline", transform=ax.transAxes,
            color=c["text_secondary"], fontsize=8.5, va="center", ha="left",
            linespacing=1.35, clip_on=False)
    style_axes(ax, c, ylabel="speedup vs static assignment")
    title(ax, c, "Smart placement only pays off on fine-grained tasks",
          "with coarse tasks any steal moves a lot of work, so the policies converge")

    leg = ax.legend(frameon=False, loc="upper left", fontsize=9.5,
                    handlelength=1.2)
    for t in leg.get_texts():
        t.set_color(c["text_secondary"])
    save(fig, f"granularity{suffix}.png", c)


# ── 4. Single-node scaling ───────────────────────────────────────────────────
def chart_scaling(c, suffix):
    fig, ax = plt.subplots(figsize=(7.2, 4.0))

    ideal = [WORKER_THROUGHPUT[0] * w for w in WORKERS]
    ax.plot(WORKERS, ideal, color=c["muted"], linewidth=2, linestyle=(0, (4, 3)),
            zorder=3, label="linear scaling")
    ax.plot(WORKERS, WORKER_THROUGHPUT, color=c["accent"], linewidth=2,
            marker="o", markersize=8, markerfacecolor=c["accent"],
            markeredgecolor=c["surface"], markeredgewidth=2, zorder=4,
            label="measured")

    # Selective labelling: only the endpoint, which is the headline number. A value
    # on every point is noise, and at 1-2 workers the two series coincide, so a label
    # there would sit on top of the other line.
    ax.annotate(f"{WORKER_THROUGHPUT[-1]/1e6:.2f}M", (WORKERS[-1], WORKER_THROUGHPUT[-1]),
                textcoords="offset points", xytext=(-6, -20), ha="right",
                color=c["text"], fontsize=10, fontweight="600")

    ax.set_xticks(WORKERS)
    ax.set_xlim(0.6, 8.6)
    ax.set_ylim(0, max(ideal) * 1.06)
    # Explicit ticks: the default locator picked spacings that all rounded to the
    # same "M" label, so the axis read 0M 0M 1M 1M 2M 2M.
    ax.set_yticks([0, 1e6, 2e6, 3e6, 4e6])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v/1e6:.0f}M"))
    style_axes(ax, c, ylabel="tasks / second", xlabel="worker threads")
    title(ax, c, "Lock-free deque scales to 2.5M tasks/sec on 8 cores",
          "single node, 10k synthetic tasks · gap to linear is memory bandwidth and E-cores")

    leg = ax.legend(frameon=False, loc="upper left", fontsize=9.5, handlelength=1.8)
    for t in leg.get_texts():
        t.set_color(c["text_secondary"])
    save(fig, f"scaling{suffix}.png", c)


def main():
    for palette, suffix in ((LIGHT, ""), (DARK, "-dark")):
        plt.rcParams["font.family"] = "sans-serif"
        chart_throughput(palette, suffix)
        chart_latency(palette, suffix)
        chart_granularity(palette, suffix)
        chart_scaling(palette, suffix)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
