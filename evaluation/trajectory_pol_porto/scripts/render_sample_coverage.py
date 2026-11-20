"""Render the rasters of sample_coverage_map.py: one panel per sample size, plus the full data.

Each panel shows the segment density of that sample (log colour scale), and in
orange the cells that carry segments in the full data but none in the sample, so
missing parts of the network stand out. ``--zoom`` crops all panels to the box that
holds the given share of the full data's segment mass (for Porto, whose window
domain is mostly empty countryside).

    python evaluation/trajectory_pol_porto/scripts/render_sample_coverage.py \\
        --input pol_coverage.npz --title "POL" --output pol_coverage.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np

MISSED = (0.93, 0.45, 0.13, 1.0)   # orange: used in the full data, absent from the sample
EMPTY = (1.0, 1.0, 1.0, 1.0)


def zoom_box(full: np.ndarray, share: float) -> Tuple[slice, slice]:
    """Rows and columns that keep ``share`` of the mass (equal tails cut on each axis)."""

    if share >= 1.0:
        return slice(0, full.shape[0]), slice(0, full.shape[1])
    tail = (1.0 - share) / 4.0
    out = []
    for axis in (1, 0):
        profile = full.sum(axis=axis).astype(float)
        cumulative = np.cumsum(profile) / profile.sum()
        low = int(np.searchsorted(cumulative, tail))
        high = int(np.searchsorted(cumulative, 1.0 - tail)) + 1
        out.append(slice(low, high))
    return out[0], out[1]


def panel_image(counts: np.ndarray, full: np.ndarray, vmax: float):
    import matplotlib

    rgba = np.empty(counts.shape + (4,))
    rgba[:] = EMPTY
    used = counts > 0
    level = np.log10(counts[used]) / max(np.log10(vmax), 1e-9)
    rgba[used] = matplotlib.colormaps["viridis_r"](0.15 + 0.85 * np.clip(level, 0.0, 1.0))
    rgba[(full > 0) & ~used] = MISSED
    return rgba


def render(arrays, summary, title: str, output: Path, zoom: float, dpi: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    full = arrays["full"].astype(np.int64)
    sizes = [int(s) for s in arrays["sizes"]]
    rows, cols = zoom_box(full, zoom)
    panels = [(f"n{n}", s) for n, s in zip(sizes, summary["samples"])] + [("full", None)]
    height, width = full[rows, cols].shape
    panel_w = 3.4
    figure, axes = plt.subplots(1, len(panels), figsize=(panel_w * len(panels), panel_w * height / width + 1.1),
                                squeeze=False)
    cell = summary["cell_m"]
    for axis, (key, stats) in zip(axes[0], panels):
        counts = arrays[key].astype(np.int64)[rows, cols]
        axis.imshow(panel_image(counts, full[rows, cols], max(2, counts.max())), origin="lower",
                    interpolation="nearest")
        axis.set_xticks([]); axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_color("#bbbbbb")
        if stats is None:
            axis.set_title(f"All {summary['population_trips']:,} trips\n"
                           f"{summary['population_segments']:,} segments", fontsize=9)
        else:
            axis.set_title(f"{stats['sample_trips']:,} trips ({100 * stats['sample_fraction']:.2g} %)\n"
                           f"cells hit {100 * stats['cells_hit']:.1f} %, traffic covered "
                           f"{100 * stats['segment_mass_in_hit_cells']:.1f} %", fontsize=9)
    figure.suptitle(f"{title}: trajectory-sampling samples (seed {summary['seed']}), "
                    f"{cell:g} m cells", fontsize=11)
    figure.legend(handles=[Patch(color=MISSED, label="segments in the full data, none in the sample"),
                           Patch(color=matplotlib.colormaps["viridis_r"](0.6),
                                 label="sampled segments (darker = denser, log scale)")],
                  loc="lower center", ncol=2, frameon=False, fontsize=8)
    figure.tight_layout(rect=(0, 0.06, 1, 0.94))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi)
    plt.close(figure)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--title", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--zoom", type=float, default=1.0, help="share of the segment mass to keep in view")
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args(argv)
    arrays = np.load(args.input)
    summary = json.loads(args.input.with_suffix(".json").read_text())
    render(arrays, summary, args.title, args.output, args.zoom, args.dpi)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
