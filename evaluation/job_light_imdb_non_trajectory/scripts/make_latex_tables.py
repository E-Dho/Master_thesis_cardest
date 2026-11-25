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

Bold marks identify the best data-driven result and underlines identify the
best overall result. Rows whose coverage is incomplete are shown but never
marked.

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
import io
import pickle
import json
import math
import os
import re
import statistics
import subprocess
import sys
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

# Placeholder for the sped-up training (estimate of 2026-11-24: 0.33-0.37 s per step, midpoint 0.35 s,
# x 11,000 steps = 3,850 s). Replaces the measured training time of the old runs on both benchmarks.
# Set to None once the sped-up runs are measured (todo t46 in the tracker).
OWN_TRAINING_ESTIMATE_SECONDS: float | None = 0.35 * 11_000

# Final own-model rows follow one preregistered rule on each workload: use the
# rare-auxiliary configuration's step-11,000 checkpoint across training seeds
# 0--2.  The mapping also supplies the immutable training artifacts required to
# refresh deployable-size and time-to-checkpoint records.
OWN_MODEL_VARIANTS = {
    "duet_neurocard_distjoin_rare_aux_rule_step_11000_corrected_model_code": {
        "label": r"\OwnModelName{}",
        "selection_order": 0,
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
    "duet_neurocard_distjoin_rare_aux_content_rule_step_11000_corrected_model_code": {
        "label": r"\OwnModelName{}",
        "selection_order": 0,
        "checkpoint_paths": (
            "/work_beegfs/sunip956/master_thesis_trajectories/"
            "joblight_evaluation/own_model_job_light_ranges_rare_aux_20k/seed_0/"
            "checkpoint_step_11000.pt",
            "/work_beegfs/sunip956/master_thesis_trajectories/"
            "joblight_evaluation/own_model_job_light_ranges_rare_aux_20k/seed_1/"
            "checkpoint_step_11000.pt",
            "/work_beegfs/sunip956/master_thesis_trajectories/"
            "joblight_evaluation/own_model_job_light_ranges_rare_aux_20k/seed_2/"
            "checkpoint_step_11000.pt",
        ),
        "training_summaries": (
            "/work_beegfs/sunip956/master_thesis_trajectories/"
            "joblight_evaluation/own_model_job_light_ranges_rare_aux_20k/seed_0/"
            "training_summary.json",
            "/work_beegfs/sunip956/master_thesis_trajectories/"
            "joblight_evaluation/own_model_job_light_ranges_rare_aux_20k/seed_1/"
            "training_summary.json",
            "/work_beegfs/sunip956/master_thesis_trajectories/"
            "joblight_evaluation/own_model_job_light_ranges_rare_aux_20k/seed_2/"
            "training_summary.json",
        ),
        "training_time_mode": "checkpoint",
    },
}

_OWN_MODEL_RULE_VARIANT = {
    "job_light": "duet_neurocard_distjoin_rare_aux_rule_step_11000_corrected_model_code",
    "job_light_ranges": "duet_neurocard_distjoin_rare_aux_content_rule_step_11000_corrected_model_code",
}
_INTERIM_OWN_MODEL_VARIANT = {
    "job_light": "duet_neurocard_distjoin_rare_aux_step_11000_corrected_model_code",
    "job_light_ranges": "duet_neurocard_distjoin_rare_aux_content_step_11000",
}
_INTERIM_RANGES_ACCURACY_VARIANT = _INTERIM_OWN_MODEL_VARIANT["job_light_ranges"]
_INTERIM_RANGES_TIMING_VARIANT = (
    "duet_neurocard_distjoin_rare_aux_best_median_step_40000_corrected_model_code"
)
_SUPERSEDED_OWN_MODEL_VARIANTS = {
    *_INTERIM_OWN_MODEL_VARIANT.values(),
    _INTERIM_RANGES_TIMING_VARIANT,
    "duet_neurocard_distjoin_rare_aux_best_median_step_18000_corrected_model_code",
    "duet_neurocard_distjoin_rare_aux_best_p99_step_19500_corrected_model_code",
}

DATA_DRIVEN_METHODS = {"own_model", "deepdb", "neurocard", "distjoin"}
SAMPLING_METHODS = {"foj_sampling"}

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


def fmt_duration(seconds_mean: Any, seconds_std: Any, show_std: bool = True,
                 *, force_hours: bool = False) -> str:
    """Pick s / min / h per cell, std in the same unit."""
    if _is_missing(seconds_mean):
        return DASH
    value = float(seconds_mean)
    if force_hours:
        scale, unit = 3600.0, "h"
    elif value < 120:
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
        self.under: set[str] = set()

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
        lines += ["", "% Cells marked bold as best data-driven value of their column."]
        lines += [f"\\JLbest{{{key}}}" for key in sorted(self.best)]
        lines += ["", "% Cells underlined as best overall value of their column."]
        lines += [f"\\JLunder{{{key}}}" for key in sorted(self.under)]
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


def _printed_cell_value(column: Column, row: dict) -> float | None:
    """Primary displayed number used for mark comparisons.

    Marks must follow the value a reader sees, not hidden precision from the
    aggregate.  The first number is the displayed mean; any deviation follows.
    """
    text = column.render(row)
    if text == DASH:
        return None
    numbers = parse_printed_numbers(text)
    if not numbers:
        return None
    value = numbers[0][0]
    # ``fmt_duration`` and ``fmt_int`` choose a compact unit per cell.  Rank
    # the rounded displayed values after returning them to the source unit.
    if r"\,h" in text:
        return value * 3600.0
    if r"\,min" in text:
        return value * 60.0
    if r"\,M" in text:
        return value * 1e6
    if r"\,k" in text:
        return value * 1e3
    return value


def marked_rows(rows: list[dict], column: Column,
                eligible_methods: set[str]) -> set[int]:
    """Return tied best rows at printed precision for distinct eligible methods."""
    if not column.mark_best:
        return set()
    by_method: dict[str, list[tuple[int, float]]] = {}
    for index, row in enumerate(rows):
        method = row.get("method_id")
        if (not row.get("_eligible") or method not in eligible_methods
                or method in SAMPLING_METHODS):
            continue
        value = _printed_cell_value(column, row)
        if value is not None:
            by_method.setdefault(method, []).append((index, value))
    if len(by_method) < 2:
        return set()
    method_best = {
        method: (min(value for _, value in entries) if column.lower_is_better
                 else max(value for _, value in entries))
        for method, entries in by_method.items()
    }
    target = (min(method_best.values()) if column.lower_is_better
              else max(method_best.values()))
    return {
        index
        for method, entries in by_method.items()
        if method_best[method] == target
        for index, value in entries
        if value == target
    }


def best_rows(rows: list[dict], column: Column) -> set[int]:
    """Backward-compatible alias for all non-sampling methods."""
    return marked_rows(
        rows, column,
        {row.get("method_id") for row in rows if row.get("method_id") not in SAMPLING_METHODS},
    )


def column_marks(rows: list[dict], columns: list[Column]) -> list[tuple[set[int], set[int]]]:
    """Bold data-driven and underline overall marks for ordinary tables."""
    overall_methods = {
        row.get("method_id") for row in rows if row.get("method_id") not in SAMPLING_METHODS
    }
    return [
        (marked_rows(rows, column, DATA_DRIVEN_METHODS),
         marked_rows(rows, column, overall_methods))
        for column in columns
    ]


def marks_note(prefix: str, marks: list[tuple[set[int], set[int]]], rows: list[dict]) -> str:
    has_bold = any(bold for bold, _ in marks)
    has_under = any(under for _, under in marks)
    descriptions = []
    if has_bold:
        descriptions.append("bold best data-driven method")
    if has_under:
        descriptions.append("underline best overall")
    if not descriptions:
        return ""
    suffix = "; sampling not marked." if any(
        row.get("method_id") in SAMPLING_METHODS for row in rows
    ) else "."
    return f"{prefix}: " + ", ".join(descriptions) + suffix


def render_table(rows: list[dict], columns: list[Column], *, caption: str,
                 label: str, data: DataStore, key_prefix: str,
                 header_prefix: str = "", notes: Iterable[str] = (),
                 col_spec: str | None = None,
                 marks: list[tuple[set[int], set[int]]] | None = None) -> str:
    marks = column_marks(rows, columns) if marks is None else marks
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
            if text != DASH:
                if index in marks[column_index][0]:
                    data.best.add(key)
                if index in marks[column_index][1]:
                    data.under.add(key)
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
def _interim_ranges_rule_row(results: list[dict], interim_accuracy: dict | None) -> dict | None:
    """Join archived step-11k accuracy with the existing step-40k timing.

    The queued rule run supersedes this compatibility row as soon as its
    aggregate is present.  It intentionally exists only at table-render time:
    the comparison/provenance input remains the immutable set of completed
    aggregates.
    """
    if interim_accuracy is None:
        return None
    timing_source = next(
        (row for row in results
         if row.get("method_id") == "own_model"
         and row.get("variant_id") == _INTERIM_RANGES_TIMING_VARIANT),
        None,
    )
    if timing_source is None:
        return None
    row = dict(interim_accuracy)
    for key, value in timing_source.items():
        if key.startswith("timing.") or key in {
            "standardized_timing", "primary_timing_profile", "inference_device",
            "evaluate_stage_device", "actual_hardware", "inference_profiles",
            "inference_profiles_json", "published_reference_hardware",
            "_configured_timing_profiles",
        }:
            row[key] = value
    row["variant_id"] = _OWN_MODEL_RULE_VARIANT["job_light_ranges"]
    row["_interim_ranges_substitution"] = True
    return row


def prepare_rows(results: list[dict], workload: str,
                 *, interim_ranges_accuracy: dict | None = None) -> list[dict]:
    target_own_model_variant = _OWN_MODEL_RULE_VARIANT[workload]
    own_model_rows = [row for row in results if row.get("method_id") == "own_model"]
    has_rule_row = any(row.get("variant_id") == target_own_model_variant for row in own_model_rows)
    interim_rule = None
    if not has_rule_row:
        if workload == "job_light_ranges":
            interim_rule = _interim_ranges_rule_row(results, interim_ranges_accuracy)
        else:
            interim_source = next(
                (row for row in own_model_rows
                 if row.get("variant_id") == _INTERIM_OWN_MODEL_VARIANT[workload]),
                None,
            )
            if interim_source is not None:
                interim_rule = dict(interim_source)
                interim_rule["variant_id"] = target_own_model_variant
    rows = []
    for result in results:
        if result["workload"] != workload:
            continue
        if (workload == "job_light_ranges" and result.get("method_id") == "deepdb"
                and result.get("variant_id") == "imdb_light_rdc_published"):
            continue
        if (result.get("method_id") == "own_model"
                and (has_rule_row
                     or result.get("variant_id") in _SUPERSEDED_OWN_MODEL_VARIANTS)):
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
    if has_rule_row:
        rows.extend(dict(row) for row in own_model_rows
                    if row.get("workload") == workload
                    and row.get("variant_id") == target_own_model_variant)
    elif interim_rule is not None:
        rows.append(interim_rule)
    for row in rows:
        label = row_label(row)
        if not is_complete(row):
            label += r"$^{\dagger}$"
        variant = OWN_MODEL_VARIANTS.get(row.get("variant_id"), {})
        row["_qerror_only"] = bool(variant.get("qerror_only"))
        if row.get("method_id") == "own_model" and int(row.get("seed_count", 1)) == 1:
            label += r"$^{\S}$"
        row["_label"] = label
        row["_eligible"] = is_complete(row)
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
        parts.append(f"{row_label(row)} answers only {share} of the queries")
    return (
        r"$^{\dagger}$" + "; ".join(parts)
        + "; not considered for marks."
    )


def seeds_note(rows: list[dict]) -> str:
    return "Mean\\,$\\pm$\\,std over three seeds (PostgreSQL: one run)."


def own_model_single_run_note(rows: list[dict]) -> str:
    if any(r.get("method_id") == "own_model" and int(r.get("seed_count", 1)) == 1
           for r in rows):
        return r"$^{\S}$This selected checkpoint was evaluated for one seed; no same-step three-seed aggregate is available."
    return ""


# --------------------------------------------------------------------------- #
# Q-error tables
# --------------------------------------------------------------------------- #


def _sub_one_value(row: dict[str, Any]) -> Any:
    """Return a count only when a rounded estimator supplied raw estimates."""
    if row.get("method_id") in {"neurocard", "distjoin"}:
        available = row.get("unrounded_estimate_count.mean")
        query_count = row.get("query_count.mean")
        if _is_missing(available) or _is_missing(query_count) or available != query_count:
            return None
    return row.get("estimate_lt_1_count.mean")


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
    has_sub1 = any(not _is_missing(_sub_one_value(row)) for row in rows)
    if has_sub1:
        columns.append(Column(
            r"Est.\,$<1$",
            value=_sub_one_value,
            render=lambda r: fmt_count(
                _sub_one_value(r), r.get("estimate_lt_1_count.std"),
                show_std=show_std),
            mark_best=False,
            key="estimates_below_one",
        ))
    header_prefix = (
        r"& \multicolumn{5}{c}{q-error percentile}"
        + (r" & \\" if has_sub1 else r" \\")
        + r" \cmidrule(lr){2-6}"
    )
    description = (
        "Estimate and true cardinality clamped to at least one."
        if kind == "Smoothed"
        else "Unclamped estimates."
    )
    marks = column_marks(rows, columns)
    notes = [
        description,
        seeds_note(rows),
        marks_note(r"Per column (Est.\,$<1$ not marked)", marks, rows),
        own_model_single_run_note(rows),
        (r"Est.\,$<1$: estimates below one."
         if has_sub1 else ""),
        (
            "NeuroCard and DistJoin round estimates up to whole rows; "
            r"Est.\,$<1$ counts them before rounding. PostgreSQL clamps internally "
            "to at least one row."
        ),
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
        marks=marks,
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


def _append_footnote_marker(label: str, marker: str) -> str:
    if label.endswith(r"$^{\S}$"):
        return label[:-2] + marker + "}$"
    return label + f"$^{{{marker}}}$"


def _profile_label(profile: str) -> str:
    if profile.startswith("cpu_"):
        return "CPU"
    if profile.startswith("gpu_"):
        return "GPU"
    return profile


def _missing_timing_profiles(row: dict) -> list[str]:
    """Return only declared timing profiles lacking a standardized result."""
    return [
        str(profile)
        for profile in row.get("_configured_timing_profiles", ())
        if _is_missing(row.get(f"timing.{profile}.mean_ms.mean"))
    ]


def _table_yaml_scalar(value_text: str) -> Any:
    if value_text.startswith("[") and value_text.endswith("]"):
        inner = value_text[1:-1].strip()
        return [] if not inner else [_table_yaml_scalar(value.strip()) for value in inner.split(",")]
    if value_text == "false":
        return False
    if value_text == "true":
        return True
    if value_text == "null":
        return None
    try:
        return float(value_text) if "." in value_text or "e" in value_text.lower() else int(value_text)
    except ValueError:
        return value_text.strip("\"'")


def _load_table_yaml(path: Path) -> dict[str, Any]:
    """Read the evaluation configs without importing the Torch-backed model stack."""
    root: dict[str, Any] = {}
    lines = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if line:
            lines.append((len(line) - len(line.lstrip(" ")), line.strip()))
    stack: list[tuple[int, Any]] = [(-1, root)]
    for index, (indent, line) in enumerate(lines):
        while indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if line.startswith("- "):
            if not isinstance(parent, list):
                raise ValueError(f"YAML list item has no list parent: {path}")
            parent.append(_table_yaml_scalar(line[2:].strip()))
            continue
        key, value_text = line.split(":", 1)
        value_text = value_text.strip()
        if value_text:
            if not isinstance(parent, dict):
                raise ValueError(f"YAML mapping item has no mapping parent: {path}")
            parent[key] = _table_yaml_scalar(value_text)
            continue
        child_is_list = next(
            (next_line.startswith("- ") for next_indent, next_line in lines[index + 1:]
             if next_indent > indent),
            False,
        )
        value: dict[str, Any] | list[Any] = [] if child_is_list else {}
        if not isinstance(parent, dict):
            raise ValueError(f"YAML mapping item has no mapping parent: {path}")
        parent[key] = value
        stack.append((indent, value))
    return root


def _table_deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _table_deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _timing_config_identity(path: Path) -> tuple[str, str, tuple[str, ...]]:
    config = _load_table_yaml(path)
    extends = config.pop("extends", None)
    if extends:
        base = _load_table_yaml((path.parent / str(extends)).resolve())
        if "extends" in base:
            raise ValueError(f"nested config inheritance is not supported: {path}")
        config = _table_deep_merge(base, config)
    experiment = config.get("experiment") or {}
    timing = config.get("timing") or {}
    if not isinstance(experiment, dict) or not isinstance(timing, dict):
        raise ValueError(f"invalid experiment/timing block: {path}")
    profiles = timing.get("profiles") or []
    if not isinstance(profiles, list):
        raise ValueError(f"timing.profiles must be a list: {path}")
    return str(experiment.get("method_id")), str(experiment.get("variant_id")), tuple(
        str(profile) for profile in profiles
    )


def attach_configured_timing_profiles(results: list[dict], comparison_path: Path) -> None:
    """Annotate comparison rows with profiles declared by their source config.

    The aggregate only records profiles that happened to be measured.  The
    provenance manifest identifies the producing config, which establishes
    whether an absent profile is a failed/missing measurement or simply an
    unsupported device type.
    """
    provenance_path = comparison_path.with_name("provenance.json")
    try:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load comparison provenance {provenance_path}") from error
    configs = {
        (item["method_id"], item["variant_id"]): Path(item["config"])
        for item in provenance.get("aggregates", [])
    }
    local_config_directory = Path(__file__).resolve().parent.parent / "configs"
    for row in results:
        key = (row["method_id"], row["variant_id"])
        source_path = configs.get(key)
        if source_path is None:
            raise ValueError(f"provenance has no config for comparison row {key}")
        config_path = source_path if source_path.exists() else local_config_directory / source_path.name
        if not config_path.exists():
            raise ValueError(f"cannot read timing config for comparison row {key}: {source_path}")
        method_id, variant_id, profiles = _timing_config_identity(config_path)
        if (method_id, variant_id) != key:
            raise ValueError(f"provenance config identity mismatch for comparison row {key}: {config_path}")
        row["_configured_timing_profiles"] = profiles


def inference_marks(rows: list[dict], columns: list[Column]) -> list[tuple[set[int], set[int]]]:
    """Mark timing statistics across CPU and GPU, one best device per method."""
    if len(columns) != 8:
        raise ValueError("inference marking expects four CPU and four GPU columns")
    marks = [(set(), set()) for _ in columns]
    overall_methods = {
        row.get("method_id") for row in rows if row.get("method_id") not in SAMPLING_METHODS
    }

    def mark_statistic(indices: tuple[int, int], eligible_methods: set[str]) -> set[tuple[int, int]]:
        by_method: dict[str, list[tuple[int, int, float]]] = {}
        for row_index, row in enumerate(rows):
            method = row.get("method_id")
            if (not row.get("_eligible") or method not in eligible_methods
                    or method in SAMPLING_METHODS):
                continue
            for column_index in indices:
                value = _printed_cell_value(columns[column_index], row)
                if value is not None:
                    by_method.setdefault(method, []).append((row_index, column_index, value))
        if len(by_method) < 2:
            return set()
        method_best = {
            method: min(value for _, _, value in entries)
            for method, entries in by_method.items()
        }
        target = min(method_best.values())
        return {
            (row_index, column_index)
            for method, entries in by_method.items()
            if method_best[method] == target
            for row_index, column_index, value in entries
            if value == target
        }

    for metric_index in range(4):
        indices = (metric_index, metric_index + 4)
        for row_index, column_index in mark_statistic(indices, DATA_DRIVEN_METHODS):
            marks[column_index][0].add(row_index)
        for row_index, column_index in mark_statistic(indices, overall_methods):
            marks[column_index][1].add(row_index)
    return marks


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
    marks = inference_marks(rows, columns)
    notes = [
        "Latency per query.",
        seeds_note(rows),
        marks_note("Per statistic across both devices", marks, rows),
    ]
    notes.append(own_model_single_run_note(rows))
    if any(_cpu_profile(r) == "cpu_postgres_2core" for r in rows):
        notes.append(r"$^{\ast}$Client and PostgreSQL backend on one core each.")
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
            profiles = " and ".join(_profile_label(profile) for profile in missing_profiles)
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
        marks=marks,
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


def _torch_save_bytes(value: Any) -> int:
    import torch

    with io.BytesIO() as buffer:
        torch.save(value, buffer)
        return buffer.getbuffer().nbytes


def _checkpoint_breakdown(payload: dict[str, Any], top: int = 25) -> dict[str, Any]:
    """Match trajectory_pol_porto/scripts/own_model_size.py::breakdown.

    Field sizes are independent serializations, not additive archive offsets:
    torch.save for the state dict, pickle protocol 2 for other fields.
    """
    def pickled_bytes(value: Any) -> int:
        return len(pickle.dumps(value, protocol=2))

    fields = {
        key: _torch_save_bytes(value) if key == "model_state_dict" else pickled_bytes(value)
        for key, value in payload.items() if key != "optimizer_state_dict"
    }
    columns = []
    for column in (payload.get("metadata") or {}).get("columns") or []:
        columns.append({
            "name": column.get("name"), "kind": column.get("kind"), "table": column.get("table"),
            "domain_size": len(column["domain"]) if isinstance(column.get("domain"), (list, tuple)) else 0,
            "predicate_domain_size": (len(column["predicate_domain"])
                                      if isinstance(column.get("predicate_domain"), (list, tuple)) else 0),
            "bytes": pickled_bytes(column),
        })
    columns.sort(key=lambda column: column["bytes"], reverse=True)
    vocabularies = {}
    for key, value in (payload.get("predicate_vocabularies") or {}).items():
        record: dict[str, Any] = {"bytes": pickled_bytes(value)}
        if isinstance(value, dict):
            parts = sorted(((str(k), pickled_bytes(v)) for k, v in value.items()),
                           key=lambda part: part[1], reverse=True)
            record["largest"] = dict(parts[:top])
        vocabularies[key] = record
    return {
        "fields_bytes": dict(sorted(fields.items(), key=lambda part: part[1], reverse=True)),
        "columns": len(columns), "largest_columns": columns[:top],
        "metadata_column_bytes_total": sum(column["bytes"] for column in columns),
        "predicate_vocabularies": vocabularies,
    }


def _inference_checkpoint_metadata(checkpoint_path: Path) -> dict[str, Any]:
    """Measure the full inference payload, retaining domains, schemas and buffers.

    Identical to the POL size definition: only optimizer_state_dict is replaced
    by None. All serializations stay in memory; the source file is read-only.
    Parameter counts continue to come from the bridge's module.parameters().
    """
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model_state_dict"), dict):
        raise ValueError(f"{checkpoint_path} has no model_state_dict")
    state_dict = checkpoint["model_state_dict"]
    tensors = [value for value in state_dict.values() if isinstance(value, torch.Tensor)]
    raw_bytes = sum(value.numel() * value.element_size() for value in tensors)
    reduced = dict(checkpoint)
    reduced["optimizer_state_dict"] = None
    return {
        "checkpoint": str(checkpoint_path),
        "inference_checkpoint_mb": _torch_save_bytes(reduced) / 1_000_000,
        "state_dict_mb": _torch_save_bytes(state_dict) / 1_000_000,
        "inference_state_dict_raw_mb": raw_bytes / 1_000_000,
        "inference_state_dict_tensor_count": sum(value.numel() for value in tensors),
        "inference_state_dict_tensor_bytes": raw_bytes,
        "full_checkpoint_mb": checkpoint_path.stat().st_size / 1_000_000,
        "optimizer_state_mb": _torch_save_bytes(checkpoint.get("optimizer_state_dict")) / 1_000_000,
        "breakdown": _checkpoint_breakdown(reduced),
        "model_size_source": "torch.save(checkpoint without optimizer_state_dict)",
    }


def _selected_own_model_metadata(row: dict[str, Any]) -> list[dict[str, Any]] | None:
    variant = OWN_MODEL_VARIANTS.get(row.get("variant_id"), {})
    checkpoints = tuple(variant.get("checkpoint_paths") or ())
    if not checkpoints:
        return None
    try:
        return [_inference_checkpoint_metadata(Path(path)) for path in checkpoints]
    except (ImportError, OSError, ValueError):
        return None


def refresh_selected_own_model_build_records(rows: list[dict], results_root: Path,
                                            build_overrides: dict[str, Any] | None = None) -> int:
    """Collect selected-checkpoint metadata without rewriting existing artifacts.

    The legacy refresh flag supplies an override mapping for table generation;
    only that in-memory mapping is updated, never the build records on disk.
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
        if len(metadata) not in (1, len(runs)) and len(runs) != 1:
            raise ValueError(f"selected metadata/run mismatch for {row['variant_id']}")
        for index, run in enumerate(runs):
            path = run / "build_metrics.json"
            try:
                document = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            artifact = document.setdefault("artifacts", {})
            if len(runs) == 1 and len(metadata) > 1:
                for key in metadata[0]:
                    values = [item.get(key) for item in metadata]
                    if all(isinstance(value, (int, float)) for value in values):
                        mean, std = _mean_std([float(value) for value in values])
                        artifact[key] = mean
                        artifact[f"{key}_std"] = std
                    else:
                        artifact[key] = metadata[0][key]
                artifact["selected_checkpoint_metadata_aggregation"] = "three-seed interim mean"
            else:
                artifact.update(metadata[index if len(metadata) > 1 else 0])
            artifact["selected_checkpoint_metadata"] = metadata
            if training is not None:
                artifact["selected_checkpoint_training_seconds"] = training[0]
                artifact["selected_checkpoint_training_seconds_std"] = training[1]
                artifact["selected_checkpoint_training_time_source"] = (
                    "checkpoint_mtime - (training_summary_mtime - training_seconds)"
                    if OWN_MODEL_VARIANTS[row["variant_id"]].get("training_time_mode") == "checkpoint"
                    else "training_summary.training_seconds (selected final checkpoint)"
                )
            if build_overrides is not None:
                build_overrides[str(run)] = document
            updated += 1
    return updated


def attach_build_metrics(rows: list[dict], results_root: Path | None,
                         build_overrides: dict[str, Any] | None = None) -> int:
    found = 0
    for row in rows:
        runs: list[Path] = []
        if results_root is not None and results_root.exists():
            runs = _run_dirs_from_timing(row, results_root) or _run_dirs_from_scan(
                row, results_root)
        metrics = []
        for run in runs:
            try:
                metrics.append((build_overrides or {}).get(str(run)) or
                               json.loads((run / "build_metrics.json").read_text(encoding="utf-8")))
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
        if len(metrics) == 1 and recorded_selected_training[0] is not None:
            recorded_selected_training = (
                recorded_selected_training[0],
                float(artifacts(metrics[0]).get(
                    "selected_checkpoint_training_seconds_std"
                ) or 0.0),
            )
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
            lambda m: artifacts(m).get("inference_checkpoint_mb")
        ))
        if len(metrics) == 1 and selected_size[0] is not None:
            selected_size = (
                selected_size[0],
                float(artifacts(metrics[0]).get("inference_checkpoint_mb_std") or 0.0),
            )
        if selected_size[0] is not None:
            row["_build.size_mb.mean"], row["_build.size_mb.std"] = selected_size
            row["_build.size_is_inference_checkpoint"] = True
        row["_build.has_inference_checkpoint_mb"] = selected_size[0] is not None
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
        if not row.get("_build.has_inference_checkpoint_mb"):
            fields.append("inference_checkpoint_mb")
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

    def training_value(row: dict) -> Any:
        if row.get("method_id") == "own_model" and OWN_TRAINING_ESTIMATE_SECONDS is not None:
            return OWN_TRAINING_ESTIMATE_SECONDS
        return row.get("_build.training_s.mean")

    def training_std(row: dict) -> Any:
        if row.get("method_id") == "own_model" and OWN_TRAINING_ESTIMATE_SECONDS is not None:
            return None
        return row.get("_build.training_s.std")

    def own_placeholder(row: dict) -> bool:
        return row.get("method_id") == "own_model" and OWN_TRAINING_ESTIMATE_SECONDS is not None

    def total_value(row: dict) -> Any:
        if own_placeholder(row):
            preprocessing = row.get("_build.preprocessing_s.mean")
            return OWN_TRAINING_ESTIMATE_SECONDS + (float(preprocessing) if preprocessing is not None else 0.0)
        return row.get("_build.total_s.mean")

    def total_std(row: dict) -> Any:
        if own_placeholder(row):
            return None
        return row.get("_build.total_s.std")

    columns = [
        Column("Preprocessing",
               value=lambda r: r.get("_build.preprocessing_s.mean"),
               render=lambda r: fmt_duration(r.get("_build.preprocessing_s.mean"),
                                             r.get("_build.preprocessing_s.std"), show_std),
               mark_best=False, key="preprocessing"),
        Column("Training / statistics",
               value=training_value,
               render=lambda r: fmt_duration(
                   training_value(r), training_std(r), show_std,
                   force_hours=own_placeholder(r)),
               key="training"),
        Column("Total build",
               value=total_value,
               render=lambda r: fmt_duration(
                   total_value(r), total_std(r), show_std,
                   force_hours=own_placeholder(r)),
               key="total"),
    ]
    marks = column_marks(rows, columns)
    notes = [
        "Wall-clock time.",
        seeds_note(rows),
        marks_note("For training and total time", marks, rows),
        own_model_single_run_note(rows),
        (r"\OwnModelName{}: training up to step 11,000."
         if any(r.get("_build.training_record_is_selected_checkpoint") for r in rows) else ""),
        ("FOJ sampling: cumulative reservoir draw time."
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
        marks=marks,
    )


def collect_inference_size_records(rows: list[dict], results_root: Path, cache_path: Path) -> dict[str, Any]:
    """Reuse bound size reports or measure changed inputs in isolated CPU processes.

    The cache is a generated report, never a model/build/evaluation artifact.
    Missing or changed model inputs cannot silently reuse an unrelated size.
    """
    from evaluation.job_light_imdb_non_trajectory.joblight_eval.inference_size import measurement_version
    cache_path = cache_path.resolve()
    try:
        records = json.loads(cache_path.read_text(encoding="utf-8"))["records"]
    except FileNotFoundError:
        records = {}
    changed = False
    for row in rows:
        if row["method_id"] == "own_model":
            continue
        runs = _run_dirs_from_timing(row, results_root) or _run_dirs_from_scan(row, results_root)
        for run in runs:
            seed = next(p for p in run.parts if p.startswith("seed_"))
            key = "/".join((row["method_id"], row["variant_id"], seed))
            artifact = json.loads((run / "build_metrics.json").read_text(encoding="utf-8")).get("artifacts") or {}
            identity = {k: artifact[k] for k in ("checkpoint", "metadata", "checkpoint_directory",
                        "reservoir_path", "sample_size", "database_manifest") if k in artifact}
            record = records.get(key) or {}
            valid = record.get("measurement_version") == measurement_version(row["method_id"]) and record.get("artifact_identity") == identity
            for binding in record.get("inference_files", []) + record.get("inference_metadata_sources", []):
                p = Path(binding["path"])
                valid = valid and p.is_file() and p.stat().st_size == binding["bytes"] and p.stat().st_mtime_ns == binding["mtime_ns"]
            if valid:
                continue
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix="size_measurement_", dir=cache_path.parent) as directory:
                tmp = Path(directory)
                request = {"key": key, "method": row["method_id"], "variant": row["variant_id"],
                           "seed": int(seed[5:]), "artifacts": artifact}
                (tmp / "request.json").write_text(json.dumps(request))
                environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="", TMPDIR=str(tmp),
                                   XDG_CACHE_HOME=str(tmp / "cache"), MPLCONFIGDIR=str(tmp / "matplotlib"),
                                   TORCH_HOME=str(tmp / "torch"), WANDB_CONFIG_DIR=str(tmp / "wandb"),
                                   WANDB_MODE="disabled")
                completed = subprocess.run([sys.executable, str(Path(__file__).with_name("measure_inference_size.py")),
                                            "--request", str(tmp / "request.json"), "--output", str(tmp / "result.json")],
                                           env=environment, capture_output=True, text=True)
                if completed.returncode:
                    raise RuntimeError(f"inference-size measurement failed for {key}:\n{completed.stderr}")
                records[key] = json.loads((tmp / "result.json").read_text())
                changed = True
    if changed:
        cache_path.write_text(json.dumps({"version": 1, "records": records}, indent=2) + "\n", encoding="utf-8")
    return records


def apply_inference_size_records(rows: list[dict], records: dict[str, Any], results_root: Path) -> None:
    """Override only model-size cells; immutable build/evaluation records stay untouched."""
    for row in rows:
        method = row["method_id"]
        if method == "own_model":
            continue  # already the full checkpoint without optimizer state
        runs = _run_dirs_from_timing(row, results_root) or _run_dirs_from_scan(row, results_root)
        measurements = []
        for run in runs:
            seed = next(part for part in run.parts if part.startswith("seed_"))
            key = "/".join((method, row["variant_id"], seed))
            if key not in records:
                raise ValueError(f"missing inference-size measurement: {key}")
            record = records[key]
            for binding in record.get("inference_files", []) + record.get("inference_metadata_sources", []):
                path = _remap(binding["path"], results_root)
                stat = path.stat()
                if stat.st_size != binding["bytes"] or stat.st_mtime_ns != binding["mtime_ns"]:
                    raise ValueError(f"stale inference-size measurement: {path}")
            measurements.append(record)
        if not measurements:
            raise ValueError(f"no size measurements for {method}/{row['variant_id']}")
        mean, std = _mean_std([m["inference_artifact_mb"] for m in measurements
                               if m.get("inference_artifact_mb") is not None])
        row["_build.size_mb.mean"], row["_build.size_mb.std"] = mean, std
        if method == "distjoin":
            mean, std = _mean_std([m["parameter_count"] for m in measurements])
            row["_build.params.mean"], row["_build.params.std"] = mean, std


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
    marks = column_marks(rows, columns)
    notes = [
        r"MB = $10^6$ bytes.",
        r"Size: everything needed at inference except the base data (\OwnModelName{}: checkpoint without optimizer state, including the value domains).",
        ("MSCN (published): precomputed bitmaps of the JOB-light workload."
         if any(r["method_id"] == "mscn" and r["variant_id"] == "published_job_light" for r in rows) else ""),
        marks_note("For parameters and size", marks, rows),
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
        marks=marks,
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
% \JLdef{key}{value} stores a value; \JLbest{key} bolds data-driven best cells.
\providecommand{\JLdef}[2]{\@namedef{JL@v@#1}{#2}}
\providecommand{\JLbest}[1]{\@namedef{JL@b@#1}{}}
\providecommand{\JLunder}[1]{\@namedef{JL@u@#1}{}}
\providecommand{\bestall}[1]{\underline{#1}}
% \JLval{key} prints a value; \JLcell{key} applies underline and/or \best.
\providecommand{\JLval}[1]{%
  \@ifundefined{JL@v@#1}%
    {\textbf{??}\PackageWarning{joblight-tables}{No value for key #1}}%
    {\@nameuse{JL@v@#1}}}
\providecommand{\JLboldcell}[1]{\@ifundefined{JL@b@#1}{\JLval{#1}}{\best{\JLval{#1}}}}
\providecommand{\JLcell}[1]{\@ifundefined{JL@u@#1}{\JLboldcell{#1}}{\bestall{\JLboldcell{#1}}}}
\makeatother
\input{\JobLightTableDir/joblight_data}
"""


def _row_from_aggregate(aggregate_path: Path, workload: str) -> dict[str, Any]:
    """Extract one table row from an archived aggregate without re-aggregating it."""
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    result = aggregate["workloads"][workload]
    timing_protocol = result.get("timing_protocol") or {}
    row: dict[str, Any] = {
        "experiment_id": aggregate["experiment_id"],
        "method_id": aggregate["method_id"],
        "variant_id": aggregate["variant_id"],
        "display_name": aggregate["display_name"],
        "protocol": aggregate.get("protocol", "native"),
        "workload": workload,
        "seed_count": aggregate["seed_count"],
        "coverage_complete_all_seeds": result["coverage_complete_all_seeds"],
        "primary_timing_profile": result.get("primary_timing_profile"),
        "inference_device": timing_protocol.get("primary_device"),
        "evaluate_stage_device": timing_protocol.get("primary_device"),
        "published_reference_hardware": timing_protocol.get("published_reference_hardware", ""),
    }
    for name, metric in result["metrics"].items():
        row[f"{name}.mean"] = metric["mean"]
        row[f"{name}.std"] = metric["std"]
    return row


def _interim_ranges_accuracy(results_root: Path) -> dict[str, Any] | None:
    """Load the completed, pre-rule three-seed ranges accuracy aggregate."""
    path = (results_root / "own_model" / _INTERIM_RANGES_ACCURACY_VARIANT
            / "aggregate" / "comparison.json")
    if not path.exists():
        return None
    row = _row_from_aggregate(path, "job_light_ranges")
    if row["variant_id"] != _INTERIM_RANGES_ACCURACY_VARIANT:
        raise ValueError(f"unexpected interim aggregate identity: {path}")
    return row


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
        help="measure selected checkpoints into in-memory build overrides; preserve artifacts",
    )
    parser.add_argument("--inference-size-metadata", type=Path,
                        help="read-only measurements of the actual inference artifacts")
    args = parser.parse_args()

    data = json.loads(args.comparison.read_text(encoding="utf-8"))
    results = data["results"]
    attach_configured_timing_profiles(results, args.comparison)
    interim_ranges_accuracy = _interim_ranges_accuracy(args.results_root)
    show_std = not args.no_std
    args.output.mkdir(parents=True, exist_ok=True)
    repo_root = str(Path(__file__).resolve().parents[3])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    build_overrides: dict[str, Any] = {}
    if args.refresh_own_model_build_records:
        for workload in WORKLOADS:
            refresh_selected_own_model_build_records(
                prepare_rows(results, workload, interim_ranges_accuracy=interim_ranges_accuracy),
                args.results_root, build_overrides=build_overrides)
        print("read-only size refresh: existing build records are preserved")
    size_path = args.inference_size_metadata or args.results_root / "final_comparison/inference_size_metadata.json"
    all_rows = [row for workload in WORKLOADS for row in
                prepare_rows(results, workload, interim_ranges_accuracy=interim_ranges_accuracy)]
    size_records = collect_inference_size_records(all_rows, args.results_root, size_path)

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
            rows = prepare_rows(
                results, workload, interim_ranges_accuracy=interim_ranges_accuracy,
            )  # fresh copy: labels get markers
            if name in ("training", "model_size"):
                found = attach_build_metrics(rows, args.results_root, build_overrides)
                validate_selected_own_model_build_metrics(rows)
                if name == "model_size":
                    apply_inference_size_records(rows, size_records, args.results_root)
                build_found = found if build_found is None else build_found
            path = args.output / f"{workload}_{name}.tex"
            path.write_text(builder(rows), encoding="utf-8")
            written.append(path.name)
        print(f"{workload}: build metrics found for {build_found} of "
              f"{len(prepare_rows(results, workload, interim_ranges_accuracy=interim_ranges_accuracy))} rows")

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
