#!/usr/bin/env python3
"""Generate LaTeX result tables for JOB-light and JOB-light-ranges.

Reads the final ``comparison.json`` (q-error and inference latency) and, when
available, the per-run ``build_metrics.json`` files (training time and model
size).  For every workload it writes five tables:

    <workload>_qerror_smoothed.tex
    <workload>_qerror_raw.tex
    <workload>_inference.tex
    <workload>_training.tex
    <workload>_model_size.tex

Data and layout are kept apart:

    joblight_data.tex   every number shown in any table, as \\JLdef{<key>}{<value>},
                        plus \\JLbest{<key>} for the cells marked as best
    <table>.tex         layout only: captions, labels, row names, notes; each
                        cell is \\JLcell{<key>}, each number in a note \\JLval{<key>}
    table_macros.tex    the macros above; inputs joblight_data.tex

So a layout change cannot alter a number, and a change in the data shows up as
a diff of joblight_data.tex alone.  Every formatted number is also parsed back
and checked against its source value before it is written; generation aborts if
a value does not round to what is printed.

The best value per column is wrapped in ``\\best{...}`` (bold by default).  Rows
whose coverage is incomplete, and rows listed as out of competition, are shown
but never marked as best.

Required LaTeX packages: booktabs, adjustbox.

Usage
-----
    python make_latex_tables.py \\
        --comparison results/joblight_phase1/final_comparison/comparison.json \\
        --results-root results/joblight_phase1 \\
        --output results/joblight_phase1/final_comparison/latex

``--results-root`` is only needed for the training-time and model-size tables.
Without per-run build metrics those tables are still written, with "--" cells.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import tempfile
from pathlib import Path
from typing import Any, Callable, Iterable

WORKLOADS = {
    "job_light": "JOB-light",
    "job_light_ranges": "JOB-light-ranges",
}

# Row grouping: a \midrule is placed between groups.
GROUP_OF_METHOD = {
    "postgresql": 0,
    "mscn": 1,
    "deepdb": 1,
    "neurocard": 1,
    "distjoin": 1,
    "foj_sampling": 2,
    "own_model": 3,
}

DASH = "--"

# These variants are deliberately selected single-checkpoint reports rather
# than a generic "own_model" row.  Keeping this mapping keyed by variant makes
# the selection visible in the rendered tables and leaves future ablations
# unambiguous.
OWN_MODEL_VARIANTS = {
    "duet_neurocard_distjoin_rare_aux_best_median_step_18000_corrected_model_code": {
        "label": r"\OwnModelName{} (best median)",
        "selection_order": 0,
        "checkpoint_paths": (
            "/work_beegfs/sunip956/master_thesis_trajectories/"
            "Master_thesis_cardest/model/runs/"
            "job_light_duet_binary_native_anpm_40k_rare_auxiliary/"
            "checkpoint_step_18000.pt",
        ),
        "training_summaries": (
            "/work_beegfs/sunip956/master_thesis_trajectories/"
            "Master_thesis_cardest/model/runs/"
            "job_light_duet_binary_native_anpm_40k_rare_auxiliary/"
            "training_summary.json",
        ),
        "training_time_mode": "checkpoint",
    },
    "duet_neurocard_distjoin_rare_aux_step_11000_corrected_model_code": {
        "label": r"\OwnModelName{} (best p99)",
        "selection_order": 1,
        "qerror_only": True,
        "checkpoint_paths": (
            "/work_beegfs/sunip956/master_thesis_trajectories/"
            "Master_thesis_cardest/model/runs/"
            "job_light_duet_binary_native_anpm_40k_rare_auxiliary/"
            "checkpoint_step_11000.pt",
            "/work_beegfs/sunip956/master_thesis_trajectories/"
            "joblight_evaluation/own_model_rare_aux_40k/seed_1/"
            "checkpoint_step_11000.pt",
            "/work_beegfs/sunip956/master_thesis_trajectories/"
            "joblight_evaluation/own_model_rare_aux_40k/seed_2/"
            "checkpoint_step_11000.pt",
        ),
        "training_summaries": (
            "/work_beegfs/sunip956/master_thesis_trajectories/"
            "Master_thesis_cardest/model/runs/"
            "job_light_duet_binary_native_anpm_40k_rare_auxiliary/"
            "training_summary.json",
            "/work_beegfs/sunip956/master_thesis_trajectories/"
            "joblight_evaluation/own_model_rare_aux_40k/seed_1/"
            "training_summary.json",
            "/work_beegfs/sunip956/master_thesis_trajectories/"
            "joblight_evaluation/own_model_rare_aux_40k/seed_2/"
            "training_summary.json",
        ),
        "training_time_mode": "checkpoint",
    },
    "duet_neurocard_distjoin_rare_aux_best_median_step_40000_corrected_model_code": {
        "label": r"\OwnModelName{} (best median)",
        "selection_order": 0,
        "checkpoint_paths": (
            "/work_beegfs/sunip956/master_thesis_trajectories/"
            "Master_thesis_cardest/model/runs/"
            "job_light_ranges_duet_binary_native_anpm_40k_no_rare_support_live_dropout_0_0_seed_0/"
            "checkpoint_step_40000.pt",
        ),
        "training_summaries": (
            "/work_beegfs/sunip956/master_thesis_trajectories/"
            "Master_thesis_cardest/model/runs/"
            "job_light_ranges_duet_binary_native_anpm_40k_no_rare_support_live_dropout_0_0_seed_0/"
            "training_summary.json",
        ),
        "training_time_mode": "full_run",
    },
    "duet_neurocard_distjoin_rare_aux_best_p99_step_19500_corrected_model_code": {
        "label": r"\OwnModelName{} (best p99)",
        "selection_order": 1,
        "qerror_only": True,
        "checkpoint_paths": (
            "/work_beegfs/sunip956/master_thesis_trajectories/"
            "joblight_evaluation/own_model_job_light_ranges_rare_aux_20k/seed_1/"
            "checkpoint_step_19500.pt",
        ),
        "training_summaries": (
            "/work_beegfs/sunip956/master_thesis_trajectories/"
            "joblight_evaluation/own_model_job_light_ranges_rare_aux_20k/seed_1/"
            "training_summary.json",
        ),
        "training_time_mode": "checkpoint",
    },
}

# Methods that are shown in the inference table but excluded from its best-value
# marking (see the table note for the reason).
INFERENCE_OUT_OF_COMPETITION = {"mscn"}

_CHECKPOINT_LOAD_STATUSES = {
    "pretrained_checkpoint",
    "reused_checkpoint",
    "checkpoint_reused",
}


# --------------------------------------------------------------------------- #
# Labels
# --------------------------------------------------------------------------- #
def _thousands(n: int) -> str:
    return f"{n:,}".replace(",", "{,}")


def row_label(result: dict[str, Any]) -> str:
    method = result["method_id"]
    variant = result["variant_id"]
    if method == "postgresql":
        return "PostgreSQL 16.10"
    if method == "mscn":
        return "MSCN (adapted)" if variant == "mscn_job_light_ranges_adapted" else "MSCN"
    if method == "deepdb":
        if result.get("protocol") == "adapted":
            return "DeepDB (adapted)"
        return "DeepDB"
    if method == "neurocard":
        return "NeuroCard"
    if method == "distjoin":
        return "DistJoin"
    if method == "foj_sampling":
        match = re.fullmatch(r"n(\d+)", variant)
        size = _thousands(int(match.group(1))) if match else variant
        return f"FOJ sampling ($N={size}$)"
    if method == "own_model":
        return OWN_MODEL_VARIANTS.get(variant, {}).get("label", r"\OwnModelName{}")
    return result.get("display_name", f"{method}/{variant}")


def is_complete(result: dict[str, Any]) -> bool:
    return bool(result.get("coverage_complete_all_seeds", True))


# --------------------------------------------------------------------------- #
# Number formatting
# --------------------------------------------------------------------------- #
def _is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def fmt_number(value: float, sig: int = 3) -> str:
    """Math-mode body (no $) for a non-negative number."""
    if value == 0:
        return "0"
    magnitude = abs(value)
    if magnitude >= 1e5:
        exponent = int(math.floor(math.log10(magnitude)))
        mantissa = value / 10**exponent
        if round(mantissa, 2) >= 10:
            mantissa /= 10
            exponent += 1
        return f"{mantissa:.2f}\\cdot 10^{{{exponent}}}"
    if magnitude >= 1000:
        return _thousands(int(round(value)))
    # Round to the requested precision before choosing the final decimal
    # places: 9.997 at three significant digits is 10.0, not 10.00.
    digits = sig - 1 - int(math.floor(math.log10(magnitude)))
    rounded = round(value, digits)
    rounded_magnitude = abs(rounded)
    if rounded_magnitude >= 1000:
        return _thousands(int(round(rounded)))
    digits = max(sig - 1 - int(math.floor(math.log10(rounded_magnitude))), 0)
    return f"{rounded:.{digits}f}"


_NUMBER_RE = re.compile(r"(\d+(?:\.\d+)?)(?:\\cdot 10\^\{(-?\d+)\})?")
_SHARED_EXPONENT_RE = re.compile(
    r"\$\((\d+(?:\.\d+)?)\\,\{\\scriptstyle\\pm (\d+(?:\.\d+)?)\}\)\\cdot 10\^\{(-?\d+)\}\$"
)


def _half_unit(digits: str) -> float:
    """Half a unit of the last significant printed digit.

    '1.23' -> 0.005 and '1234' -> 0.5; trailing zeros of an integer count as
    rounding ('240' -> 5), because values are rounded to significant digits.
    """
    if "." in digits:
        return 0.5 * 10.0 ** -len(digits.split(".", 1)[1])
    stripped = digits.rstrip("0")
    trailing = len(digits) - len(stripped) if stripped else 0
    return 0.5 * 10.0 ** trailing


def parse_printed_numbers(text: str) -> list[tuple[float, float]]:
    """Return (value, rounding tolerance) for every number in a formatted cell, in order."""

    plain = text.replace("{,}", "")
    shared = _SHARED_EXPONENT_RE.search(plain)
    if shared:
        scale = 10.0 ** int(shared.group(3))
        return [
            (float(shared.group(1)) * scale, _half_unit(shared.group(1)) * scale),
            (float(shared.group(2)) * scale, _half_unit(shared.group(2)) * scale),
        ]
    numbers = []
    for match in _NUMBER_RE.finditer(plain):
        scale = 10.0 ** int(match.group(2) or 0)
        numbers.append((float(match.group(1)) * scale, _half_unit(match.group(1)) * scale))
    return numbers


def verify_printed(text: str, expected: list[float], *, unit_scale: float = 1.0) -> str:
    """Raise unless every printed number rounds back to its source value.

    ``unit_scale`` converts printed numbers back to the source unit (60 for a value
    printed in minutes, 1e6 for "M", ...).  Returns ``text`` unchanged.
    """

    printed = parse_printed_numbers(text)
    if len(printed) != len(expected):
        raise ValueError(f"printed {text!r} has {len(printed)} numbers, expected {len(expected)}")
    for (value, tolerance), source in zip(printed, expected):
        source = float(source)
        slack = tolerance * unit_scale + 1e-9 * max(1.0, abs(source))
        if abs(value * unit_scale - source) > slack:
            raise ValueError(f"printed {text!r} does not round to source value {source!r}")
    return text


def fmt_cell(mean: Any, std: Any = None, sig: int = 3, show_std: bool = True,
             suffix: str = "") -> str:
    if _is_missing(mean):
        return DASH
    mean = float(mean)
    body = f"${fmt_number(mean, sig)}$"
    if show_std and not _is_missing(std) and float(std) > 0:
        std = float(std)
        if mean >= 1e5:
            # share the mean's exponent: (1.33 +- 0.78) * 10^5
            mantissa, exponent = fmt_number(mean).split("\\cdot 10^{")
            exponent = exponent.rstrip("}")
            std_mantissa = std / 10 ** int(exponent)
            text = (f"$({mantissa}\\,{{\\scriptstyle\\pm {std_mantissa:.2f}}})"
                    f"\\cdot 10^{{{exponent}}}$")
            return verify_printed(text, [mean, std]) + suffix
        body += f"\\,{{\\scriptsize$\\pm {fmt_number(std, 2)}$}}"
        return verify_printed(body, [mean, std]) + suffix
    return verify_printed(body, [mean]) + suffix


def fmt_count(mean: Any, std: Any, show_std: bool = True) -> str:
    if _is_missing(mean):
        return DASH
    mean = float(mean)
    text = f"${mean:.0f}$" if abs(mean - round(mean)) < 1e-9 else f"${mean:.1f}$"
    expected = [mean]
    if show_std and not _is_missing(std) and float(std) > 0:
        text += f"\\,{{\\scriptsize$\\pm {float(std):.1f}$}}"
        expected.append(float(std))
    return verify_printed(text, expected)


def fmt_duration(seconds_mean: Any, seconds_std: Any, show_std: bool = True) -> str:
    """Pick s / min / h per cell, std in the same unit."""
    if _is_missing(seconds_mean):
        return DASH
    value = float(seconds_mean)
    if value < 120:
        scale, unit = 1.0, "s"
    elif value < 2 * 3600:
        scale, unit = 60.0, "min"
    else:
        scale, unit = 3600.0, "h"
    std = None if _is_missing(seconds_std) else float(seconds_std) / scale
    text = fmt_cell(value / scale, std, show_std=show_std, suffix=f"\\,{unit}")
    expected = [value] + ([float(seconds_std)] if len(parse_printed_numbers(text)) == 2 else [])
    return verify_printed(text, expected, unit_scale=scale)


def fmt_int(value: Any) -> str:
    if _is_missing(value):
        return DASH
    value = float(value)
    if value >= 1e6:
        return verify_printed(f"${value / 1e6:.2f}$\\,M", [value], unit_scale=1e6)
    if value >= 1e3:
        return verify_printed(f"${value / 1e3:.1f}$\\,k", [value], unit_scale=1e3)
    return verify_printed(f"${value:.0f}$", [value])


def fmt_percent(fraction: float) -> str:
    return verify_printed(f"{100 * fraction:.1f}\\,\\%", [100 * fraction])


class DataStore:
    """Every value shown in the tables, keyed; written to joblight_data.tex."""

    _KEY_RE = re.compile(r"^[A-Za-z0-9_/.\-]+$")

    def __init__(self) -> None:
        self.values: dict[str, str] = {}
        self.best: set[str] = set()

    def add(self, key: str, text: str) -> str:
        if not self._KEY_RE.match(key):
            raise ValueError(f"invalid data key {key!r}")
        if key in self.values and self.values[key] != text:
            raise ValueError(f"data key {key!r} defined twice with different values")
        self.values[key] = text
        return key

    def cell(self, key: str, text: str) -> str:
        return f"\\JLcell{{{self.add(key, text)}}}"

    def inline(self, key: str, text: str) -> str:
        return f"\\JLval{{{self.add(key, text)}}}"

    def render(self, sources: list[str]) -> str:
        lines = [
            "% Generated by make_latex_tables.py -- do not edit by hand.",
            "% All numbers shown in the JOB-light tables. Sources:",
            *[f"%   {source}" for source in sources],
            "",
        ]
        lines += [f"\\JLdef{{{key}}}{{{self.values[key]}}}" for key in sorted(self.values)]
        lines += ["", "% Cells marked as best value of their column."]
        lines += [f"\\JLbest{{{key}}}" for key in sorted(self.best)]
        return "\n".join(lines) + "\n"


def row_key(row: dict[str, Any]) -> str:
    return f"{row['method_id']}/{row['variant_id']}"


# --------------------------------------------------------------------------- #
# Generic table writer
# --------------------------------------------------------------------------- #
class Column:
    def __init__(self, header: str, value: Callable[[dict], Any],
                 render: Callable[[dict], str], lower_is_better: bool = True,
                 mark_best: bool = True, key: str | None = None) -> None:
        self.header = header
        self.value = value
        self.render = render
        self.lower_is_better = lower_is_better
        self.mark_best = mark_best
        self.key = key or re.sub(r"[^A-Za-z0-9]+", "_", header).strip("_").lower()


def best_rows(rows: list[dict], column: Column) -> set[int]:
    candidates = []
    for index, row in enumerate(rows):
        if not row["_eligible"]:
            continue
        value = column.value(row)
        if _is_missing(value):
            continue
        candidates.append((float(value), index))
    if not candidates or not column.mark_best:
        return set()
    target = min(v for v, _ in candidates) if column.lower_is_better else max(
        v for v, _ in candidates
    )
    tolerance = 1e-12 * max(1.0, abs(target))
    return {i for v, i in candidates if abs(v - target) <= tolerance}


def render_table(rows: list[dict], columns: list[Column], *, caption: str,
                 label: str, data: DataStore, key_prefix: str,
                 header_prefix: str = "", notes: Iterable[str] = (),
                 col_spec: str | None = None) -> str:
    best = [best_rows(rows, column) for column in columns]
    keys = [column.key for column in columns]
    if len(set(keys)) != len(keys):
        raise ValueError(f"duplicate column keys in {key_prefix}: {keys}")
    spec = col_spec or ("l" + "r" * len(columns))
    lines = [
        "% Generated by make_latex_tables.py -- do not edit by hand.",
        "% Layout only; every number is a key into joblight_data.tex.",
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \small",
        r"  \setlength{\tabcolsep}{4pt}",
        f"  \\caption{{{caption}}}",
        f"  \\label{{{label}}}",
        r"  \begin{adjustbox}{max width=\linewidth}",
        f"  \\begin{{tabular}}{{{spec}}}",
        r"    \toprule",
    ]
    if header_prefix:
        lines.append("    " + header_prefix)
    lines.append(
        "    Method & " + " & ".join(column.header for column in columns) + r" \\"
    )
    lines.append(r"    \midrule")
    previous_group = None
    for index, row in enumerate(rows):
        group = GROUP_OF_METHOD.get(row["method_id"], 99)
        if previous_group is not None and group != previous_group:
            lines.append(r"    \midrule")
        previous_group = group
        cells = []
        for column_index, column in enumerate(columns):
            text = column.render(row)
            key = f"{key_prefix}/{row_key(row)}/{column.key}"
            cells.append(data.cell(key, text))
            if index in best[column_index] and text != DASH:
                data.best.add(key)
        lines.append(f"    {row['_label']} & " + " & ".join(cells) + r" \\")
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  \end{adjustbox}",
    ]
    notes = [note for note in notes if note]
    if notes:
        lines.append(r"  \par\smallskip")
        lines.append(r"  \begin{minipage}{\linewidth}\footnotesize")
        lines.append("    " + " ".join(notes))
        lines.append(r"  \end{minipage}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Rows
# --------------------------------------------------------------------------- #
def prepare_rows(results: list[dict], workload: str) -> list[dict]:
    rows = []
    for result in results:
        if result["workload"] != workload:
            continue
        row = dict(result)
        label = row_label(row)
        if not is_complete(row):
            label += r"$^{\dagger}$"
        variant = OWN_MODEL_VARIANTS.get(row.get("variant_id"), {})
        row["_qerror_only"] = bool(variant.get("qerror_only"))
        if row.get("method_id") == "own_model" and int(row.get("seed_count", 1)) == 1:
            label += r"$^{\S}$"
        row["_label"] = label
        row["_eligible"] = is_complete(row)
        rows.append(row)
    rows.sort(key=lambda r: (
        GROUP_OF_METHOD.get(r["method_id"], 99),
        OWN_MODEL_VARIANTS.get(r.get("variant_id"), {}).get("selection_order", 0),
    ))
    return rows


def coverage_note(rows: list[dict], data: DataStore) -> str:
    partial = [r for r in rows if not is_complete(r)]
    if not partial:
        return ""
    parts = []
    for row in partial:
        fraction = row.get("coverage_fraction.mean")
        share = (
            data.inline(f"{row['workload']}/coverage/{row_key(row)}", fmt_percent(fraction))
            if fraction is not None else "part"
        )
        parts.append(f"{row_label(row)} supports only {share} of the queries")
    return (
        r"$^{\dagger}$" + "; ".join(parts)
        + "; its values are computed on that subset and are not considered for the best value."
    )


def seeds_note(rows: list[dict]) -> str:
    return "Values are mean\\,$\\pm$\\,standard deviation over three seeds (PostgreSQL: a single deterministic run)."


def own_model_single_run_note(rows: list[dict]) -> str:
    if any(r.get("method_id") == "own_model" and int(r.get("seed_count", 1)) == 1
           for r in rows):
        return r"$^{\S}$This selected checkpoint was evaluated for one seed; no same-step three-seed aggregate is available."
    return ""


def own_model_selection_note(rows: list[dict]) -> str:
    selected = [row for row in rows if row.get("variant_id") in OWN_MODEL_VARIANTS]
    if len(selected) == 1:
        return (r"The \OwnModelName{} row was selected on this benchmark's own queries "
                r"for both median (A) and p99 (B).")
    if len(selected) > 1:
        return (r"The two \OwnModelName{} rows are the best-median (A) and best-p99 (B) "
                r"selections on this benchmark's own queries.")
    return ""


# --------------------------------------------------------------------------- #
# Q-error tables
# --------------------------------------------------------------------------- #
BEST_NOTE = r"Bold marks the best value per column."


def qerror_table(rows: list[dict], workload: str, family: str, show_std: bool,
                 data: DataStore) -> str:
    rows = [row for row in rows if not row.get("_qerror_only") or row.get("method_id") == "own_model"]
    kind = "Smoothed" if family == "smoothed_q_error" else "Raw"
    stats = [("p50", "Median"), ("p90", "90th"), ("p95", "95th"),
             ("p99", "99th"), ("max", "Max")]
    columns = []
    for key, header in stats:
        columns.append(Column(
            header,
            value=lambda r, k=key: r.get(f"{family}.{k}.mean"),
            render=lambda r, k=key: fmt_cell(
                r.get(f"{family}.{k}.mean"), r.get(f"{family}.{k}.std"),
                show_std=show_std),
            key=key,
        ))
    has_sub1 = any(not _is_missing(r.get("estimate_lt_1_count.mean")) for r in rows)
    if has_sub1:
        columns.append(Column(
            r"Est.\,$<1$",
            value=lambda r: r.get("estimate_lt_1_count.mean"),
            render=lambda r: fmt_count(
                r.get("estimate_lt_1_count.mean"), r.get("estimate_lt_1_count.std"),
                show_std=show_std),
            key="estimates_below_one",
        ))
    header_prefix = (
        r"& \multicolumn{5}{c}{q-error percentile}"
        + (r" & \\" if has_sub1 else r" \\")
        + r" \cmidrule(lr){2-6}"
    )
    description = (
        "Estimates and true cardinalities are clamped to at least one before the q-error is computed."
        if kind == "Smoothed"
        else "Q-error on the unclamped estimates."
    )
    notes = [
        description,
        seeds_note(rows),
        BEST_NOTE,
        own_model_single_run_note(rows),
        own_model_selection_note(rows),
        (r"Est.\,$<1$ is the number of queries with an estimate below one."
         if has_sub1 else ""),
        coverage_note(rows, data),
    ]
    workload_slug = workload.replace("_", "-")
    return render_table(
        rows, columns,
        caption=f"{kind} q-error on {WORKLOADS[workload]}",
        label=f"tab:{workload_slug}-qerror-{kind.lower()}",
        data=data,
        key_prefix=f"{workload}/qerror_{kind.lower()}",
        header_prefix=header_prefix,
        notes=notes,
        col_spec="l" + "r" * 5 + ("r" if has_sub1 else ""),
    )


# --------------------------------------------------------------------------- #
# Inference timing table
# --------------------------------------------------------------------------- #
def _timing(row: dict, profile: str, metric: str) -> tuple[Any, Any]:
    return row.get(f"timing.{profile}.{metric}.mean"), row.get(
        f"timing.{profile}.{metric}.std")


def _cpu_profile(row: dict) -> str | None:
    if not _is_missing(row.get("timing.cpu_1core.mean_ms.mean")):
        return "cpu_1core"
    if not _is_missing(row.get("timing.cpu_postgres_2core.mean_ms.mean")):
        return "cpu_postgres_2core"
    return None


def _names(value: Any) -> list[str]:
    if isinstance(value, str):
        return [part.strip() for part in value.split(";") if part.strip()]
    if isinstance(value, (list, tuple)):
        return [str(part) for part in value if part]
    return []


def _append_footnote_marker(label: str, marker: str) -> str:
    if label.endswith(r"$^{\S}$"):
        return label[:-2] + marker + "}$"
    return label + f"$^{{{marker}}}$"


def _missing_timing_profiles(row: dict) -> list[str]:
    missing = []
    if _cpu_profile(row) is None:
        missing.append("CPU")
    if _is_missing(row.get("timing.gpu_single_query.mean_ms.mean")):
        missing.append("GPU")
    return missing


def inference_table(rows: list[dict], workload: str, show_std: bool, data: DataStore) -> str:
    rows = [row for row in rows if not row.get("_qerror_only")]
    for row in rows:
        if _cpu_profile(row) == "cpu_postgres_2core":
            row["_label"] += r"$^{\ast}$"
        missing_profiles = _missing_timing_profiles(row)
        if len(missing_profiles) == 2:
            row["_label"] = _append_footnote_marker(row["_label"], r"\ddagger")
        elif missing_profiles:
            row["_label"] = _append_footnote_marker(row["_label"], r"\P")
        if row["method_id"] in INFERENCE_OUT_OF_COMPETITION:
            row["_eligible"] = False
            row["_label"] = _append_footnote_marker(row["_label"], r"\circ")
    metrics = [("mean_ms", "Mean"), ("p50_ms", "p50"), ("p95_ms", "p95"), ("p99_ms", "p99")]
    columns = []
    for metric, header in metrics:
        def cpu_value(r, m=metric):
            profile = _cpu_profile(r)
            return _timing(r, profile, m)[0] if profile else None

        def cpu_render(r, m=metric):
            profile = _cpu_profile(r)
            if not profile:
                return DASH
            mean, std = _timing(r, profile, m)
            return fmt_cell(mean, std, show_std=show_std)

        columns.append(Column(header, cpu_value, cpu_render, key=f"cpu_{metric}"))
    for metric, header in metrics:
        columns.append(Column(
            header,
            value=lambda r, m=metric: _timing(r, "gpu_single_query", m)[0],
            render=lambda r, m=metric: fmt_cell(
                *_timing(r, "gpu_single_query", m), show_std=show_std),
            key=f"gpu_{metric}",
        ))
    header_prefix = (
        r"& \multicolumn{4}{c}{CPU, 1 core (ms)} & \multicolumn{4}{c}{GPU, single query (ms)} \\"
        r" \cmidrule(lr){2-5} \cmidrule(lr){6-9}"
    )
    hardware = sorted({
        name
        for row in rows
        for key in ("timing.cpu_1core.cpu_models", "timing.gpu_single_query.gpu_names")
        for name in _names(row.get(key))
    })
    hardware_text = (
        data.inline(f"{workload}/inference/hardware", ", ".join(hardware)) if hardware else ""
    )
    notes = [
        "Per-query end-to-end estimation latency under the standardized timing profiles"
        + (f" ({hardware_text})" if hardware else "") + ".",
        seeds_note(rows),
    ]
    out_of_competition = [r for r in rows if r["method_id"] in INFERENCE_OUT_OF_COMPETITION]
    if out_of_competition:
        names = ", ".join(sorted({row_label(r) for r in out_of_competition}))
        notes.append(
            rf"$^{{\circ}}${names} is shown out of competition: as a query-driven model it "
            r"evaluates one small set-convolution network per query, whereas all other methods "
            r"estimate from a model or sample of the data. Bold marks the best value per column "
            r"among the remaining methods."
        )
    else:
        notes.append(BEST_NOTE)
    notes.append(own_model_single_run_note(rows))
    if any(_cpu_profile(r) == "cpu_postgres_2core" for r in rows):
        notes.append(r"$^{\ast}$Measured with client and PostgreSQL backend pinned to one core each.")
    for row in rows:
        missing_profiles = _missing_timing_profiles(row)
        if not missing_profiles:
            continue
        mean = row.get("evaluate_stage_inference.mean_ms.mean")
        device = row.get("evaluate_stage_device")
        if _is_missing(mean):
            mean = row.get("inference.mean_ms.mean")
            device = row.get("inference_device")
        device = {"cuda": "GPU", "cpu": "CPU"}.get(device or "", device or "")
        extra = ""
        if not _is_missing(mean):
            value = data.inline(
                f"{workload}/inference/{row_key(row)}/evaluate_stage_mean_ms",
                verify_printed(f"${fmt_number(float(mean))}$", [float(mean)]),
            )
            extra = f"; the non-profile-controlled evaluate-stage mean is {value}\\,ms ({device})"
        if len(missing_profiles) == 2:
            notes.append(
                rf"$^{{\ddagger}}${row_label(row)} was not measured under a standardized profile{extra}."
            )
        else:
            profiles = " and ".join(missing_profiles)
            notes.append(
                rf"$^{{\P}}${row_label(row)} is missing its standardized {profiles} timing profile{extra}."
            )
    notes.append(coverage_note(rows, data))
    workload_slug = workload.replace("_", "-")
    return render_table(
        rows, columns,
        caption=f"Inference latency per query on {WORKLOADS[workload]}",
        label=f"tab:{workload_slug}-inference",
        data=data,
        key_prefix=f"{workload}/inference",
        header_prefix=header_prefix,
        notes=notes,
        col_spec="l" + "r" * 8,
    )


# --------------------------------------------------------------------------- #
# Build metrics (training time / model size)
# --------------------------------------------------------------------------- #
RUN_DIR_RE = re.compile(r"^\d{8}T\d{6}Z-")


def _remap(path: str, results_root: Path) -> Path:
    marker = f"/results/{results_root.name}/"
    if marker in path:
        return results_root / path.split(marker, 1)[1]
    return Path(path)


def _run_dirs_from_timing(result: dict, results_root: Path) -> list[Path]:
    runs: dict[str, Path] = {}
    for profile in (result.get("standardized_timing") or {}).values():
        for directory in profile.get("timing_directories") or []:
            path = _remap(directory, results_root)
            parts = path.parts
            if "timing" in parts:
                run = Path(*parts[: parts.index("timing")])
                seed = run.parent.name
                runs.setdefault(seed, run)
    return [p for p in runs.values() if (p / "build_metrics.json").exists()]


def _run_dirs_from_scan(result: dict, results_root: Path) -> list[Path]:
    variant_dir = results_root / result["method_id"] / result["variant_id"]
    chosen = []
    for seed_dir in sorted(variant_dir.glob("seed_*")):
        candidates = []
        for run in sorted(seed_dir.iterdir()):
            # Accuracy runs may use descriptive IDs instead of the default
            # timestamp prefix.  Their run manifest and build artifact are the
            # reliable identity, not the directory spelling.
            if not run.is_dir():
                continue
            if not (run / "build_metrics.json").exists():
                continue
            summary_path = run / "summary.json"
            if summary_path.exists():
                try:
                    summary = json.loads(summary_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    continue
                workload = summary.get("workload") or (summary.get("config") or {}).get("workload")
                if workload and workload != result["workload"]:
                    continue
            candidates.append(run)
        if candidates:
            chosen.append(max(candidates, key=lambda path: path.stat().st_mtime))
    return chosen


def _mean_std(values: list[float]) -> tuple[float | None, float | None]:
    values = [v for v in values if not _is_missing(v)]
    if not values:
        return None, None
    return statistics.fmean(values), (statistics.stdev(values) if len(values) > 1 else 0.0)


def _build_status(metrics: dict[str, Any]) -> str | None:
    detail = (metrics.get("build") or {}).get("detail") or {}
    bridge = detail.get("bridge_metrics") or {}
    return bridge.get("status") or detail.get("status")


def _selected_own_model_training_seconds(row: dict[str, Any]) -> tuple[float, float] | None:
    variant = OWN_MODEL_VARIANTS.get(row.get("variant_id"), {})
    summaries = tuple(variant.get("training_summaries") or ())
    checkpoints = tuple(variant.get("checkpoint_paths") or ())
    if not summaries:
        return None
    if variant.get("training_time_mode") == "checkpoint" and len(summaries) != len(checkpoints):
        raise ValueError(f"checkpoint/summary mismatch for {row['variant_id']}")
    values = []
    for index, record in enumerate(summaries):
        try:
            summary_path = Path(record)
            value = json.loads(summary_path.read_text(encoding="utf-8")).get("training_seconds")
        except (OSError, json.JSONDecodeError, KeyError):
            return None
        if value is None:
            return None
        elapsed = float(value)
        if variant.get("training_time_mode") == "checkpoint":
            checkpoint_path = Path(checkpoints[index])
            try:
                training_start = summary_path.stat().st_mtime - elapsed
                elapsed = checkpoint_path.stat().st_mtime - training_start
            except OSError:
                return None
            if elapsed < 0 or elapsed > float(value) + 60:
                raise ValueError(f"invalid selected-checkpoint time for {checkpoint_path}")
        values.append(elapsed)
    return _mean_std(values)


def _inference_state_dict_metadata(checkpoint_path: Path) -> dict[str, float | int | str]:
    """Measure the deployable model state without optimizer or training metadata."""
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError(f"{checkpoint_path} has no model_state_dict")
    tensors = [value for value in state_dict.values() if isinstance(value, torch.Tensor)]
    raw_bytes = sum(value.numel() * value.element_size() for value in tensors)
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as handle:
        temporary_path = Path(handle.name)
    try:
        torch.save(state_dict, temporary_path)
        serialized_bytes = temporary_path.stat().st_size
    finally:
        temporary_path.unlink(missing_ok=True)
    return {
        "inference_state_dict_mb": serialized_bytes / 1_000_000,
        "inference_state_dict_raw_mb": raw_bytes / 1_000_000,
        "inference_state_dict_tensor_count": sum(value.numel() for value in tensors),
        "inference_state_dict_tensor_bytes": raw_bytes,
        "full_checkpoint_mb": checkpoint_path.stat().st_size / 1_000_000,
        "model_size_source": "torch.save(model_state_dict)",
    }


def _selected_own_model_metadata(row: dict[str, Any]) -> list[dict[str, Any]] | None:
    variant = OWN_MODEL_VARIANTS.get(row.get("variant_id"), {})
    checkpoints = tuple(variant.get("checkpoint_paths") or ())
    if not checkpoints:
        return None
    try:
        return [_inference_state_dict_metadata(Path(path)) for path in checkpoints]
    except (ImportError, OSError, ValueError):
        return None


def refresh_selected_own_model_build_records(rows: list[dict], results_root: Path) -> int:
    """Persist selected-checkpoint deployment and elapsed-training metadata.

    The evaluation bridge only knows checkpoint loading time.  This refresh is
    deliberately run where the original checkpoints are mounted, then the
    portable build records remain sufficient for later table generation.
    """
    updated = 0
    for row in rows:
        if row.get("method_id") != "own_model":
            continue
        metadata = _selected_own_model_metadata(row)
        training = _selected_own_model_training_seconds(row)
        if not metadata:
            continue
        runs = _run_dirs_from_timing(row, results_root) or _run_dirs_from_scan(row, results_root)
        if not runs:
            continue
        if len(metadata) not in (1, len(runs)):
            raise ValueError(f"selected metadata/run mismatch for {row['variant_id']}")
        for index, run in enumerate(runs):
            path = run / "build_metrics.json"
            try:
                document = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            artifact = document.setdefault("artifacts", {})
            artifact.update(metadata[index if len(metadata) > 1 else 0])
            if training is not None:
                artifact["selected_checkpoint_training_seconds"] = training[0]
                artifact["selected_checkpoint_training_seconds_std"] = training[1]
                artifact["selected_checkpoint_training_time_source"] = (
                    "checkpoint_mtime - (training_summary_mtime - training_seconds)"
                    if OWN_MODEL_VARIANTS[row["variant_id"]].get("training_time_mode") == "checkpoint"
                    else "training_summary.training_seconds (selected final checkpoint)"
                )
            path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            updated += 1
    return updated


def attach_build_metrics(rows: list[dict], results_root: Path | None) -> int:
    found = 0
    for row in rows:
        runs: list[Path] = []
        if results_root is not None and results_root.exists():
            runs = _run_dirs_from_timing(row, results_root) or _run_dirs_from_scan(
                row, results_root)
        metrics = []
        for run in runs:
            try:
                metrics.append(json.loads((run / "build_metrics.json").read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                continue
        row["_build_runs"] = len(metrics)
        found += bool(metrics)

        def collect(getter: Callable[[dict], Any]) -> list[float]:
            out = []
            for m in metrics:
                try:
                    value = getter(m)
                except (KeyError, TypeError):
                    value = None
                if value is not None:
                    out.append(float(value))
            return out

        artifacts = lambda m: m.get("artifacts") or {}
        for key, getter in {
            "preprocessing_s": lambda m: m.get("preprocessing_seconds"),
            "training_s": lambda m: m.get("training_or_statistics_seconds"),
            "total_s": lambda m: m.get("total_build_seconds"),
            "params": lambda m: artifacts(m).get("parameter_count"),
            "size_mb": lambda m: artifacts(m).get("serialized_model_mb"),
        }.items():
            mean, std = _mean_std(collect(getter))
            row[f"_build.{key}.mean"] = mean
            row[f"_build.{key}.std"] = std

        if row.get("method_id") == "foj_sampling":
            # Evaluation commonly reuses a completed reservoir, which makes the
            # stage-local build timer zero.  The adapter preserves the actual
            # cumulative draw time to each nested prefix in the artifact.
            sampling_time = _mean_std(collect(
                lambda m: artifacts(m).get("cumulative_sampling_seconds")
            ))
            if sampling_time[0] is not None:
                row["_build.training_s.mean"], row["_build.training_s.std"] = sampling_time
                row["_build.total_s.mean"], row["_build.total_s.std"] = sampling_time
                row["_build.foj_cumulative_sampling"] = True

        statuses = {_build_status(metric) for metric in metrics}
        checkpoint_load_only = bool(statuses & _CHECKPOINT_LOAD_STATUSES)
        row["_build.checkpoint_load_only"] = checkpoint_load_only
        row["_build.checkpoint_statuses"] = sorted(status for status in statuses if status)
        recorded_selected_training = _mean_std(collect(
            lambda m: artifacts(m).get("selected_checkpoint_training_seconds")
        ))
        selected_training = recorded_selected_training
        if selected_training[0] is None:
            selected_training = _selected_own_model_training_seconds(row)
        has_selected_training = selected_training is not None and selected_training[0] is not None
        row["_build.has_selected_checkpoint_training_seconds"] = (
            recorded_selected_training[0] is not None
        )
        if has_selected_training:
            row["_build.training_s.mean"], row["_build.training_s.std"] = selected_training
            row["_build.total_s.mean"], row["_build.total_s.std"] = selected_training
            row["_build.training_record_is_selected_checkpoint"] = True
        selected_size = _mean_std(collect(
            lambda m: artifacts(m).get("inference_state_dict_mb")
        ))
        if selected_size[0] is not None:
            row["_build.size_mb.mean"], row["_build.size_mb.std"] = selected_size
            row["_build.size_is_inference_state_dict"] = True
        row["_build.has_inference_state_dict_mb"] = selected_size[0] is not None
        if checkpoint_load_only and not has_selected_training:
            # A bridge load is not training/statistics construction when no
            # selected-checkpoint record supplies the real training duration.
            row["_build.training_s.mean"] = None
            row["_build.training_s.std"] = None
    return found


def validate_selected_own_model_build_metrics(rows: list[dict]) -> None:
    """Reject selected own-model rows that would silently fall back in tables."""
    missing = []
    for row in rows:
        variant = OWN_MODEL_VARIANTS.get(row.get("variant_id"), {})
        if row.get("method_id") != "own_model" or not variant or variant.get("qerror_only"):
            continue
        fields = []
        if not row.get("_build.has_inference_state_dict_mb"):
            fields.append("inference_state_dict_mb")
        if not row.get("_build.has_selected_checkpoint_training_seconds"):
            fields.append("selected_checkpoint_training_seconds")
        if fields:
            missing.append(f"{row['workload']}/{row['variant_id']}: {', '.join(fields)}")
    if missing:
        raise ValueError(
            "selected own-model build records are incomplete; rerun "
            "make_latex_tables.py --refresh-own-model-build-records: "
            + "; ".join(missing)
        )


def _missing_build_note(rows: list[dict]) -> str:
    if all(row.get("_build_runs") for row in rows):
        return ""
    if not any(row.get("_build_runs") for row in rows):
        return (r"\textbf{TODO:} no per-run \texttt{build\_metrics.json} found; "
                r"re-run \texttt{make\_latex\_tables.py} with \texttt{--results-root}.")
    missing = ", ".join(row_label(r) for r in rows if not r.get("_build_runs"))
    return f"No build metrics found for: {missing}."


def training_table(rows: list[dict], workload: str, show_std: bool, data: DataStore) -> str:
    rows = [row for row in rows if not row.get("_qerror_only")]
    columns = [
        Column("Preprocessing",
               value=lambda r: r.get("_build.preprocessing_s.mean"),
               render=lambda r: fmt_duration(r.get("_build.preprocessing_s.mean"),
                                             r.get("_build.preprocessing_s.std"), show_std),
               mark_best=False, key="preprocessing"),
        Column("Training / statistics",
               value=lambda r: r.get("_build.training_s.mean"),
               render=lambda r: fmt_duration(r.get("_build.training_s.mean"),
                                             r.get("_build.training_s.std"), show_std),
               key="training"),
        Column("Total build",
               value=lambda r: r.get("_build.total_s.mean"),
               render=lambda r: fmt_duration(r.get("_build.total_s.mean"),
                                             r.get("_build.total_s.std"), show_std),
               key="total"),
    ]
    notes = [
        "Wall-clock time of the preprocessing stage, the training (or statistics-collection) "
        "stage, and their sum.",
        seeds_note(rows),
        "Bold marks the best value in the training and total-build columns.",
        own_model_single_run_note(rows),
        ("For selected own-model checkpoints, training and total-build time is measured to "
         "the selected checkpoint; final-step selections use the recorded full-run time."
         if any(r.get("_build.training_record_is_selected_checkpoint") for r in rows) else ""),
        ("-- denotes a pretrained/reused checkpoint load rather than a training/statistics time."
         if any(r.get("_build.checkpoint_load_only") and not r.get("_build.training_record_is_selected_checkpoint")
                for r in rows) else ""),
        ("FOJ sampling uses the recorded cumulative reservoir draw time at each sample prefix; "
         "the zero-valued cache-hit build timer is not reported."
         if any(r.get("_build.foj_cumulative_sampling") for r in rows) else ""),
        coverage_note(rows, data),
        _missing_build_note(rows),
    ]
    workload_slug = workload.replace("_", "-")
    return render_table(
        rows, columns,
        caption=f"Training and build time on {WORKLOADS[workload]}",
        label=f"tab:{workload_slug}-training",
        data=data,
        key_prefix=f"{workload}/training",
        notes=notes,
    )


def model_size_table(rows: list[dict], workload: str, show_std: bool, data: DataStore) -> str:
    rows = [row for row in rows if not row.get("_qerror_only")]
    columns = [
        Column("Parameters",
               value=lambda r: r.get("_build.params.mean"),
               render=lambda r: fmt_int(r.get("_build.params.mean")),
               key="parameters"),
        Column("Serialized size (MB)",
               value=lambda r: r.get("_build.size_mb.mean"),
               render=lambda r: fmt_cell(r.get("_build.size_mb.mean"),
                                         r.get("_build.size_mb.std"), show_std=show_std),
               key="serialized_mb"),
    ]
    notes = [
        "Trainable parameter count and serialized model or summary size in decimal MB "
        "($10^6$ bytes); -- where not applicable.",
        (r"For \OwnModelName{}, serialized size is the inference-only \texttt{state\_dict}."
         if any(r.get("_build.size_is_inference_state_dict") for r in rows) else ""),
        BEST_NOTE,
        own_model_single_run_note(rows),
        coverage_note(rows, data),
        _missing_build_note(rows),
    ]
    workload_slug = workload.replace("_", "-")
    return render_table(
        rows, columns,
        caption=f"Model size on {WORKLOADS[workload]}",
        label=f"tab:{workload_slug}-model-size",
        data=data,
        key_prefix=f"{workload}/model_size",
        notes=notes,
    )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
PREAMBLE = r"""% Generated by make_latex_tables.py -- macros used by the JOB-light tables.
% Requires: booktabs, adjustbox. Input once in the preamble; it loads the data file.
\providecommand{\JobLightTableDir}{.}
\providecommand{\best}[1]{{\boldmath\bfseries #1}}
\providecommand{\OwnModelName}{Own model}
\makeatletter
% \JLdef{key}{value} stores a value; \JLbest{key} marks that cell as best.
\providecommand{\JLdef}[2]{\@namedef{JL@v@#1}{#2}}
\providecommand{\JLbest}[1]{\@namedef{JL@b@#1}{}}
% \JLval{key} prints a value; \JLcell{key} prints it and applies \best if marked.
\providecommand{\JLval}[1]{%
  \@ifundefined{JL@v@#1}%
    {\textbf{??}\PackageWarning{joblight-tables}{No value for key #1}}%
    {\@nameuse{JL@v@#1}}}
\providecommand{\JLcell}[1]{%
  \@ifundefined{JL@b@#1}{\JLval{#1}}{\best{\JLval{#1}}}}
\makeatother
\input{\JobLightTableDir/joblight_data}
"""


def main() -> None:
    here = Path(__file__).resolve().parent
    default_root = here.parent / "results" / "joblight_phase1"
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--comparison", type=Path,
                        default=default_root / "final_comparison" / "comparison.json")
    parser.add_argument("--results-root", type=Path, default=default_root,
                        help="experiment results directory holding <method>/<variant>/seed_*/")
    parser.add_argument("--output", type=Path,
                        default=default_root / "final_comparison" / "latex")
    parser.add_argument("--no-std", action="store_true", help="omit standard deviations")
    parser.add_argument(
        "--refresh-own-model-build-records",
        action="store_true",
        help="measure selected checkpoints and persist deployable-size/training-time metadata",
    )
    args = parser.parse_args()

    data = json.loads(args.comparison.read_text(encoding="utf-8"))
    results = data["results"]
    show_std = not args.no_std
    args.output.mkdir(parents=True, exist_ok=True)
    if args.refresh_own_model_build_records:
        refreshed = 0
        for workload in WORKLOADS:
            refreshed += refresh_selected_own_model_build_records(
                prepare_rows(results, workload), args.results_root
            )
        print(f"refreshed {refreshed} selected own-model build records")

    written = []
    store = DataStore()
    (args.output / "table_macros.tex").write_text(PREAMBLE, encoding="utf-8")
    written.append("table_macros.tex")
    for workload in WORKLOADS:
        builders = {
            "qerror_smoothed": lambda rows: qerror_table(rows, workload, "smoothed_q_error", show_std, store),
            "qerror_raw": lambda rows: qerror_table(rows, workload, "raw_q_error", show_std, store),
            "inference": lambda rows: inference_table(rows, workload, show_std, store),
            "training": lambda rows: training_table(rows, workload, show_std, store),
            "model_size": lambda rows: model_size_table(rows, workload, show_std, store),
        }
        build_found = None
        for name, builder in builders.items():
            rows = prepare_rows(results, workload)  # fresh copy: labels get markers
            if name in ("training", "model_size"):
                found = attach_build_metrics(rows, args.results_root)
                validate_selected_own_model_build_metrics(rows)
                build_found = found if build_found is None else build_found
            path = args.output / f"{workload}_{name}.tex"
            path.write_text(builder(rows), encoding="utf-8")
            written.append(path.name)
        print(f"{workload}: build metrics found for {build_found} of "
              f"{len(prepare_rows(results, workload))} rows")

    (args.output / "joblight_data.tex").write_text(
        store.render([f"comparison: {args.comparison}", f"build metrics: {args.results_root}"]),
        encoding="utf-8",
    )
    written.append("joblight_data.tex")
    print(f"joblight_data.tex: {len(store.values)} values, {len(store.best)} marked best")

    all_lines = [
        "% Generated by make_latex_tables.py",
        "% Set \\JobLightTableDir to this folder relative to your main .tex file.",
        r"\providecommand{\JobLightTableDir}{.}",
        r"\input{\JobLightTableDir/table_macros}",
    ]
    all_lines += [f"\\input{{\\JobLightTableDir/{Path(name).stem}}}"
                  for name in written if name not in ("table_macros.tex", "joblight_data.tex")]
    (args.output / "all_tables.tex").write_text("\n".join(all_lines) + "\n", encoding="utf-8")
    print("wrote", len(written) + 1, "files to", args.output)


if __name__ == "__main__":
    main()
