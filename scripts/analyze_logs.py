#!/usr/bin/env python3
"""
analyze_logs.py — Post-processing and analysis tools for GalgameBench game logs.

Operations
----------
1. Extract structured step data from raw game logs (type_help / no_case_should_remain_unsolved)
   - type_help: outputs {session_id}_full_steps.json + _full_steps.xlsx
   - no_case_should_remain_unsolved: outputs {session_id}_steps.json + _steps.xlsx

2. Generate Gantt-style unlock timeline charts
   - type_help: stem-plot style, blue = self-unlocked, orange = hint
   - no_case_should_remain_unsolved: swim-lane event markers, 4-color coded

3. Build recall-based adjacency matrices from type_help full_steps JSON
   - Edges: recall_file → newly_unlocked_file per step
   - Values: 1 = self-unlock, 2 = hint-unlock

4. Compute DAG similarity metrics against human-annotated adjacency matrix
   - Modes: all nodes / unlocked-nodes subgraph / custom node-ID range
   - Metrics: Precision, Recall, F1, Jaccard

Usage
-----
    python scripts/analyze_logs.py

Output is written to <project_root>/analysis/.
Reference data (human adjacency matrix, node ID map) is read from
dataset/type_help-en/.
"""

from __future__ import annotations

import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ─────────────────────────── Directory configuration ─────────────────────────

# Root of the GalgameBench project (two levels up from scripts/)
GALGAME_ROOT = Path(__file__).resolve().parent.parent

# Directory where memground_agent.py writes game log JSON files
LOG_DIR = GALGAME_ROOT / "logs"

# Output directory for all analysis artefacts (created on demand)
ANALYSIS_DIR = GALGAME_ROOT / "analysis"

# Reference data directory for type_help game
DATASET_DIR = GALGAME_ROOT / "dataset" / "type_help-en"


# ─────────────────────────── Matplotlib style ────────────────────────────────

_PLOT_STYLE = {
    "font.family":          "DejaVu Sans",
    "font.size":            11,
    "axes.titlesize":       13,
    "axes.labelsize":       11,
    "xtick.labelsize":      10,
    "ytick.labelsize":      10,
    "axes.titleweight":     "bold",
    "axes.spines.top":      False,
    "axes.spines.right":    False,
    "axes.grid":            False,
    "figure.facecolor":     "white",
    "axes.facecolor":       "white",
    "axes.edgecolor":       "#333333",
    "xtick.color":          "#333333",
    "ytick.color":          "#333333",
    "axes.labelcolor":      "#333333",
    "text.color":           "#222222",
    "lines.linewidth":      2.2,
    "lines.solid_capstyle": "round",
    "savefig.dpi":          200,
    "savefig.facecolor":    "white",
    "savefig.bbox":         "tight",
}


# ─────────────────────────── Operation 1: data extraction ────────────────────

def extract_typehelp_steps(actions: list) -> list:
    """Extract full-step records from type_help log actions.

    Reads ``choices.decision_rationale`` directly from the log; no checkpoint
    needed.  All rows have ``is_reconstructed = False``.

    Fields returned per step
    ------------------------
    step, node_id, unlocked_files_count, hint_unlocked_files_count,
    unlocked_files, hint_unlocked_files, recall, reason,
    steps_no_new_self_unlock, is_reconstructed
    """
    rows: list[dict] = []
    steps_no_new_self_unlock = 0
    prev_self_unlock_count   = 0

    for action in actions:
        step    = action.get("step")
        node_id = action.get("node_id", "")

        unlocked_files      = set(action.get("unlocked_files")      or [])
        hint_unlocked_files = set(action.get("hint_unlocked_files") or [])

        choices = action.get("choices") or {}
        recall  = choices.get("recall") or []
        reason  = choices.get("decision_rationale", "")

        cur_self_unlock_count = len(unlocked_files - hint_unlocked_files)
        if cur_self_unlock_count > prev_self_unlock_count:
            steps_no_new_self_unlock = 0
        else:
            steps_no_new_self_unlock += 1
        prev_self_unlock_count = cur_self_unlock_count

        rows.append({
            "step":                      step,
            "node_id":                   node_id,
            "unlocked_files_count":      len(unlocked_files),
            "hint_unlocked_files_count": len(hint_unlocked_files),
            "unlocked_files":            sorted(unlocked_files),
            "hint_unlocked_files":       sorted(hint_unlocked_files),
            "recall":                    recall,
            "reason":                    reason,
            "steps_no_new_self_unlock":  steps_no_new_self_unlock,
            "is_reconstructed":          False,
        })

    return rows


def extract_no_case_steps(actions: list) -> tuple[list, list]:
    """Extract per-step score and unlock counts from no_case_should_remain_unsolved log actions.

    Returns
    -------
    steps_data : list[dict]
        Full records (saved as JSON).
    excel_rows : list[dict]
        Five-column records suitable for Excel export.
    """
    steps_data: list[dict] = []
    excel_rows: list[dict] = []

    ever_seen: dict[str, set] = {"pink": set(), "purple": set(), "yellow": set()}

    for action in actions:
        step          = action.get("step")
        score         = action.get("score")
        locked_events = action.get("locked_events") or {}

        for color in ("pink", "purple", "yellow"):
            ever_seen[color].update(locked_events.get(color) or [])

        unlocked_pink   = len(ever_seen["pink"])   - len(locked_events.get("pink")   or [])
        unlocked_purple = len(ever_seen["purple"]) - len(locked_events.get("purple") or [])
        unlocked_yellow = len(ever_seen["yellow"]) - len(locked_events.get("yellow") or [])

        steps_data.append({
            "step":             step,
            "score":            score,
            "locked_events":    locked_events,
            "unlocked_pink":    unlocked_pink,
            "unlocked_purple":  unlocked_purple,
            "unlocked_yellow":  unlocked_yellow,
        })
        excel_rows.append({
            "step":             step,
            "score":            score,
            "unlocked_pink":    unlocked_pink,
            "unlocked_purple":  unlocked_purple,
            "unlocked_yellow":  unlocked_yellow,
        })

    return steps_data, excel_rows


def process_log_file(log_file: Path, model_label: str) -> None:
    """Process a single game log JSON file and write analysis artefacts.

    Output location: ``analysis/{model_label}_{game_type}/``

    - type_help → ``{session_id}_full_steps.json`` + ``_full_steps.xlsx``
    - no_case_should_remain_unsolved → ``{session_id}_steps.json`` + ``_steps.xlsx``
    """
    print(f"Processing: {log_file}")

    with open(log_file, encoding="utf-8") as f:
        data = json.load(f)

    game_type  = data.get("game_type", "unknown")
    session_id = data.get("session_id", log_file.stem)
    actions    = data.get("actions", [])

    clean_game_type = game_type.replace("_", "")
    out_dir = ANALYSIS_DIR / f"{model_label}_{clean_game_type}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if game_type == "type_help":
        rows = extract_typehelp_steps(actions)

        json_out = out_dir / f"{session_id}_full_steps.json"
        with open(json_out, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
        print(f"  Saved JSON : {json_out}")

        df = pd.DataFrame(rows)
        for col in ("unlocked_files", "hint_unlocked_files", "recall"):
            df[col] = df[col].apply(lambda x: ", ".join(x) if x else "")
        excel_out = out_dir / f"{session_id}_full_steps.xlsx"
        df.to_excel(excel_out, index=False)
        print(f"  Saved Excel: {excel_out}")

    elif game_type == "no_case_should_remain_unsolved":
        steps_data, excel_rows = extract_no_case_steps(actions)

        json_out = out_dir / f"{session_id}_steps.json"
        with open(json_out, "w", encoding="utf-8") as f:
            json.dump(steps_data, f, ensure_ascii=False, indent=2)
        print(f"  Saved JSON : {json_out}")

        df = pd.DataFrame(excel_rows)
        excel_out = out_dir / f"{session_id}_steps.xlsx"
        df.to_excel(excel_out, index=False)
        print(f"  Saved Excel: {excel_out}")

    else:
        print(f"  [skip] Unsupported game type: {game_type!r}")


# ─────────────────────────── File collection helpers ─────────────────────────

def collect_log_files() -> list[tuple[Path, str]]:
    """Scan LOG_DIR for GalgameBench game log JSON files.

    Each log file is expected to contain a top-level ``model`` field written
    by ``GameLogger``.  The model name is used as the label for output folders.

    Returns
    -------
    list of (file_path, model_label) tuples, sorted by file name.
    """
    results: list[tuple[Path, str]] = []
    if not LOG_DIR.exists():
        return results
    for f in sorted(LOG_DIR.glob("*.json")):
        if f.stem.endswith("_summary"):
            continue
        try:
            meta = json.loads(f.read_text(encoding="utf-8"))
            # Sanitise model name for use in directory names
            model_label = (meta.get("model") or f.stem).replace("/", "_")
            results.append((f, model_label))
        except Exception:
            continue
    return results


def collect_analysis_files() -> list[Path]:
    """Collect processed JSON files from ANALYSIS_DIR (used by Gantt plotter).

    - Folders whose name contains ``typehelp``:  collects ``*_full_steps.json``
    - Folders whose name contains ``no_case_should_remain_unsolved``: collects ``*_steps.json``
      (excludes ``*_full_steps.json``)
    """
    all_json: list[Path] = []
    if not ANALYSIS_DIR.exists():
        return all_json
    for folder in sorted(ANALYSIS_DIR.iterdir()):
        if not folder.is_dir() or folder.name.startswith("."):
            continue
        if "typehelp" in folder.name or folder.name.startswith("type_"):
            all_json.extend(sorted(folder.glob("*_full_steps.json")))
        elif "no_case_should_remain_unsolved" in folder.name:
            for p in sorted(folder.glob("*_steps.json")):
                if not p.stem.endswith("_full_steps"):
                    all_json.append(p)
    return all_json


def show_file_list(all_files: list[tuple[Path, str]]) -> None:
    print("\nAvailable log files:")
    print("-" * 60)
    for i, (log_file, model_label) in enumerate(all_files):
        print(f"  [{i + 1:>2}] {model_label} / {log_file.name}")
    print("-" * 60)


def show_json_list(all_json: list[Path]) -> None:
    print("\nAvailable JSON files (processed):")
    print("-" * 70)
    for i, p in enumerate(all_json):
        print(f"  [{i + 1:>2}] {p.parent.name} / {p.name}")
    print("-" * 70)


def parse_selection(raw: str, total: int) -> list[int]:
    """Parse a user selection string into zero-based indices.

    Supported formats: ``all``, ``3``, ``1,3,5``, ``2-5``, combinations.
    """
    raw = raw.strip().lower()
    if raw == "all":
        return list(range(total))
    indices: set[int] = set()
    for part in raw.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            indices.update(range(int(start) - 1, int(end)))
        elif part.isdigit():
            indices.add(int(part) - 1)
    return sorted(i for i in indices if 0 <= i < total)


# ─────────────────────────── Operation 2: Gantt charts ───────────────────────

def plot_typehelp_gantt(json_file: Path, model_label: str = "") -> None:
    """Plot a stem-style unlock timeline for a type_help full_steps JSON.

    Blue stems = Self-Unlocked, orange stems = Hint.
    Saved as ``{session_id}_gantt.png`` alongside the JSON, and copied to
    ``analysis/plots/``.
    """
    from matplotlib.lines import Line2D

    with open(json_file, encoding="utf-8") as f:
        steps = json.load(f)
    if not steps:
        print(f"  [skip] Empty data: {json_file.name}")
        return

    session_id = json_file.stem.replace("_full_steps", "").replace("_steps", "")
    out_dir    = json_file.parent
    label      = model_label or session_id
    max_step   = steps[-1].get("step", len(steps))

    seen_self: set = set()
    seen_hint: set = set()
    step_delta_self: dict = defaultdict(int)
    step_delta_hint: dict = defaultdict(int)
    total_self = total_hint = 0

    for row in steps:
        step          = row.get("step", 0)
        all_unlocked  = set(row.get("unlocked_files")      or [])
        hint_unlocked = set(row.get("hint_unlocked_files") or [])

        for fname in hint_unlocked - seen_hint:
            seen_hint.add(fname)
            step_delta_hint[step] += 1
            total_hint += 1

        for fname in (all_unlocked - seen_self - hint_unlocked):
            seen_self.add(fname)
            step_delta_self[step] += 1
            total_self += 1

        seen_self |= hint_unlocked

    if total_self == 0 and total_hint == 0:
        print(f"  [skip] No unlock records: {json_file.name}")
        return

    all_events: list[tuple[int, str, int]] = []
    for s in sorted(set(list(step_delta_self.keys()) + list(step_delta_hint.keys()))):
        if step_delta_hint[s] > 0:
            all_events.append((s, "hint", step_delta_hint[s]))
        if step_delta_self[s] > 0:
            all_events.append((s, "self", step_delta_self[s]))

    C         = {"self": "#1e88e5", "hint": "#f5a000"}
    LABEL_MAP = {"self": "Self-Unlocked", "hint": "Hint"}
    GRAY_DARK = "#b0b0b0"
    STEM_SEQ  = [0.32, -0.20, 0.56, -0.34]

    step_count: dict[int, int] = Counter(s for s, _, _ in all_events)
    step_cursor: dict[int, int] = {}
    JITTER = max(max_step * 0.010, 1.5)
    raw_xs: list[float] = []
    for s, typ, cnt in all_events:
        k = step_cursor.get(s, 0)
        n = step_count[s]
        offset = (k - (n - 1) / 2) * JITTER
        raw_xs.append(float(s) + offset)
        step_cursor[s] = k + 1

    MIN_GAP = max(max_step * 0.015, 2.5)
    disp_xs = list(raw_xs)
    for i in range(1, len(disp_xs)):
        if disp_xs[i] < disp_xs[i - 1] + MIN_GAP:
            disp_xs[i] = disp_xs[i - 1] + MIN_GAP
    for i in range(len(disp_xs) - 2, -1, -1):
        if disp_xs[i] > disp_xs[i + 1] - MIN_GAP:
            disp_xs[i] = disp_xs[i + 1] - MIN_GAP
        disp_xs[i] = max(disp_xs[i], raw_xs[i] - MIN_GAP * 3)

    Y_TOP = max(h for h in STEM_SEQ if h > 0)
    Y_BOT = max(abs(h) for h in STEM_SEQ if h < 0)
    fig_h  = max(3.2, Y_TOP + Y_BOT + 1.0)
    x_right = max(float(max_step), disp_xs[-1] if disp_xs else 0)

    with plt.rc_context(_PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(20, fig_h))
        ax.set_facecolor("white")
        ax.axhspan(0, Y_TOP + 0.2,    color="#f9f9f9", zorder=0)
        ax.axhspan(-(Y_BOT + 0.2), 0, color="#f4f4f4", zorder=0)
        ax.hlines(0, 0, x_right, colors="#999999", linewidth=1.6, zorder=2)

        for i, ((s, typ, cnt), dx) in enumerate(zip(all_events, disp_xs)):
            color = C[typ]
            h     = STEM_SEQ[i % len(STEM_SEQ)]
            sign  = 1 if h > 0 else -1
            ax.plot([dx, dx], [0, h], color=color, linewidth=1.3, alpha=0.60, zorder=3)
            ax.scatter(dx, h, s=60, color=color, zorder=5,
                       edgecolors="white", linewidths=1.1)
            va  = "bottom" if h > 0 else "top"
            pad = 0.028 * sign
            ax.text(dx, h + pad, str(s),
                    ha="center", va=va, fontsize=8, color=color,
                    fontweight="semibold", zorder=6)

        ax.set_yticks([0])
        ax.set_yticklabels([label], fontsize=12, fontweight="bold", color="#222222")
        ax.set_ylim(-(Y_BOT + 0.50), Y_TOP + 0.52)
        ax.set_xlim(-x_right * 0.01, x_right * 1.03)
        ax.set_xlabel("Step", labelpad=28, fontsize=11)
        ax.set_title(
            f"Type Help  \u00b7  Unlock Timeline  \u00b7  {session_id}",
            pad=10, fontsize=12, fontweight="bold", color="#1a1a1a", loc="left",
        )

        _w = max(len(str(total_self)), len(str(total_hint)), len(str(max_step + 1)))
        stats = "\n".join([
            f"  Self    {str(total_self).rjust(_w)}",
            f"  Hint    {str(total_hint).rjust(_w)}",
            f"  Steps   {str(max_step + 1).rjust(_w)}",
        ])
        ax.text(0.995, 0.90, stats, transform=ax.transAxes,
                fontsize=9, va="bottom", ha="right",
                fontfamily="monospace", color="#333333", linespacing=1.7,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#f5f5f5",
                          edgecolor=GRAY_DARK, linewidth=0.8),
                zorder=6, clip_on=False)

        legend_handles = [
            Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=C[k], markersize=9,
                   markeredgecolor="white", label=LABEL_MAP[k])
            for k in ["self", "hint"]
        ]
        ax.legend(handles=legend_handles, loc="lower center",
                  bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=9,
                  frameon=True, framealpha=0.95, edgecolor=GRAY_DARK,
                  handlelength=1.0, columnspacing=2.0)

        for spine in ("left", "top", "right", "bottom"):
            ax.spines[spine].set_visible(False)
        ax.tick_params(axis="y", length=0, labelsize=12)
        ax.tick_params(axis="x", direction="in", length=4,
                       width=0.8, color="#aaaaaa", labelsize=9.5, pad=4)
        ax.xaxis.set_ticks_position("bottom")
        ax.spines["bottom"].set_position(("data", 0))

        plot_out = out_dir / f"{session_id}_gantt.png"
        fig.savefig(plot_out, bbox_inches="tight", dpi=220)
        plt.close(fig)

    print(f"  Saved Gantt : {plot_out}")
    plots_dir = ANALYSIS_DIR / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(plot_out, plots_dir / plot_out.name)
    print(f"  Copied  to  : {plots_dir / plot_out.name}")


def _build_no_case_events(steps: list) -> tuple[int, int, dict, list]:
    """Parse a no_case_should_remain_unsolved _steps.json into event sequences for Gantt plotting."""
    ever_seen: dict[str, set] = {"pink": set(), "purple": set(), "yellow": set()}
    event_unlock: dict[str, tuple] = {}
    score_events: list = []
    prev_score = None
    max_step = 0
    final_score = 0

    for row in steps:
        step   = row.get("step", 0)
        score  = row.get("score", 0) or 0
        locked = row.get("locked_events") or {}

        max_step    = max(max_step, step)
        final_score = score

        if prev_score is not None and score > prev_score:
            score_events.append((step, score))
        prev_score = score

        for color in ("pink", "purple", "yellow"):
            cur_locked = set(locked.get(color) or [])
            for ev in cur_locked:
                ever_seen[color].add(ev)
            for ev in ever_seen[color]:
                if ev not in cur_locked and ev not in event_unlock:
                    event_unlock[ev] = (step, color)

    return max_step, final_score, event_unlock, score_events


def plot_no_case_gantt(json_file: Path, model_label: str = "") -> None:
    """Plot a swim-lane event marker chart for a no_case_should_remain_unsolved _steps.json.

    Four event types are colour-coded: pink lock, purple lock, yellow lock,
    and score increase.  Saved as ``{session_id}_gantt.png``.
    """
    from matplotlib.lines import Line2D

    with open(json_file, encoding="utf-8") as f:
        steps = json.load(f)
    if not steps:
        print(f"  [skip] Empty data: {json_file.name}")
        return

    session_id = json_file.stem.replace("_steps", "")
    out_dir    = json_file.parent
    label      = model_label or session_id

    max_step, final_score, event_unlock, score_events = _build_no_case_events(steps)
    if not event_unlock and not score_events:
        print(f"  [skip] No event data: {json_file.name}")
        return

    C = {
        "pink":   "#f0305a",
        "purple": "#8c35d6",
        "yellow": "#f5a000",
        "score":  "#00b386",
    }
    LABEL = {
        "pink":   "Pink lock",
        "purple": "Purple lock",
        "yellow": "Yellow lock",
        "score":  "Score ↑",
    }
    GRAY_DARK = "#b0b0b0"
    ORDER     = ["pink", "purple", "yellow", "score"]
    STEM_SEQ  = [0.32, -0.26, 0.56, -0.46]

    step_events: dict[int, list] = defaultdict(list)
    for ev, (s, col) in event_unlock.items():
        step_events[s].append(col)
    for s, _ in score_events:
        if "score" not in step_events[s]:
            step_events[s].append("score")
    for s in step_events:
        step_events[s].sort(key=lambda c: ORDER.index(c) if c in ORDER else 9)

    all_events: list[tuple[int, str]] = []
    for s in sorted(step_events.keys()):
        for col in step_events[s]:
            all_events.append((s, col))

    ev_stem: dict[tuple, float] = {
        (s, col): STEM_SEQ[i % len(STEM_SEQ)]
        for i, (s, col) in enumerate(all_events)
    }

    Y_TOP = max(abs(h) for h in STEM_SEQ if h > 0)
    Y_BOT = max(abs(h) for h in STEM_SEQ if h < 0)
    fig_h  = max(3.2, Y_TOP + Y_BOT + 1.0)

    with plt.rc_context(_PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(20, fig_h))
        ax.set_facecolor("white")
        ax.axhspan(0, Y_TOP + 0.2,    color="#f9f9f9", zorder=0)
        ax.axhspan(-(Y_BOT + 0.2), 0, color="#f4f4f4", zorder=0)
        ax.hlines(0, 0, max_step, colors="#999999", linewidth=1.6, zorder=2)

        for (s, col), h in ev_stem.items():
            color = C.get(col, "#888")
            sign  = 1 if h > 0 else -1
            ax.plot([s, s], [0, h], color=color, linewidth=1.3, alpha=0.60, zorder=3)
            ax.scatter(s, h, s=60, color=color, zorder=5,
                       edgecolors="white", linewidths=1.1)
            va  = "bottom" if h > 0 else "top"
            pad = 0.028 * sign
            ax.text(s, h + pad, str(s),
                    ha="center", va=va, fontsize=8, color=color,
                    fontweight="semibold", zorder=6)

        ax.set_yticks([0])
        ax.set_yticklabels([label], fontsize=12, fontweight="bold", color="#222222")
        ax.set_ylim(-(Y_BOT + 0.50), Y_TOP + 0.52)
        ax.set_xlim(-max_step * 0.01, max_step * 1.02)
        ax.set_xlabel("Step", labelpad=28, fontsize=11)
        ax.set_title(
            f"No Case Should Remain Unsolved  \u00b7  Event Unlock Timeline  \u00b7  {session_id}",
            pad=10, fontsize=12, fontweight="bold", color="#1a1a1a", loc="left",
        )

        total_ev = len(event_unlock)
        _w = max(len(str(final_score)), len(str(total_ev)), len(str(max_step + 1)))
        stats = "\n".join([
            f"  Score     {str(final_score).rjust(_w)}",
            f"  Unlocked  {str(total_ev).rjust(_w)}",
            f"  Steps     {str(max_step + 1).rjust(_w)}",
        ])
        ax.text(0.995, 0.95, stats, transform=ax.transAxes,
                fontsize=9, va="bottom", ha="right",
                fontfamily="monospace", color="#333333", linespacing=1.7,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#f5f5f5",
                          edgecolor=GRAY_DARK, linewidth=0.8),
                zorder=6, clip_on=False)

        legend_handles = [
            Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=C[k], markersize=9,
                   markeredgecolor="white", label=LABEL[k])
            for k in ORDER
        ]
        ax.legend(handles=legend_handles, loc="lower center",
                  bbox_to_anchor=(0.5, -0.22), ncol=4, fontsize=9,
                  frameon=True, framealpha=0.95, edgecolor=GRAY_DARK,
                  handlelength=1.0, columnspacing=2.0)

        for spine in ("left", "top", "right", "bottom"):
            ax.spines[spine].set_visible(False)
        ax.tick_params(axis="y", length=0, labelsize=12)
        ax.tick_params(axis="x", direction="in", length=4,
                       width=0.8, color="#aaaaaa", labelsize=9.5, pad=4)
        ax.xaxis.set_ticks_position("bottom")
        ax.spines["bottom"].set_position(("data", 0))

        plot_out = out_dir / f"{session_id}_gantt.png"
        fig.savefig(plot_out, bbox_inches="tight", dpi=220)
        plt.close(fig)

    print(f"  Saved Gantt : {plot_out}")
    plots_dir = ANALYSIS_DIR / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(plot_out, plots_dir / plot_out.name)
    print(f"  Copied  to  : {plots_dir / plot_out.name}")


def _plot_json_file(json_file: Path) -> None:
    """Auto-detect game type from the parent folder name and call the appropriate plotter."""
    folder_name = json_file.parent.name
    model_label = folder_name  # folder is already named {model}_{game_type}

    is_typehelp = "typehelp" in folder_name or folder_name.startswith("type_")
    is_no_case  = "no_case_should_remain_unsolved" in folder_name

    if is_typehelp:
        plot_typehelp_gantt(json_file, model_label=model_label)
    elif is_no_case:
        plot_no_case_gantt(json_file, model_label=model_label)
    else:
        print(f"  [skip] Cannot determine game type from folder: {folder_name!r}")


# ─────────────────────────── Operation 3: recall adjacency matrix ─────────────

def build_recall_adj(json_file: Path, human_adj_csv: Path) -> pd.DataFrame:
    """Build a recall-based adjacency matrix from a type_help full_steps JSON.

    Edge definition
    ---------------
    For each step N: every file in ``recall`` → every file newly unlocked at
    that step.  Semantics: the model recalled ``recall_file`` and thereby
    deduced / unlocked ``new_file``.

    The node set and ordering match ``human_adj_csv`` for easy comparison.

    Matrix values
    -------------
    0 = no edge
    1 = self-unlock edge
    2 = hint-unlock edge  (overrides 1 when both apply)

    When ``recall`` is empty the previous step's ``node_id`` is used as a
    fallback source.
    """
    human_df = pd.read_csv(human_adj_csv, index_col=0)
    nodes    = list(human_df.columns)
    node_set = set(nodes)

    adj = pd.DataFrame(0, index=nodes, columns=nodes)

    with open(json_file, encoding="utf-8") as f:
        steps = json.load(f)

    for n in range(1, len(steps)):
        prev_unlocked = set(steps[n - 1].get("unlocked_files", []))
        curr_unlocked = set(steps[n].get("unlocked_files", []))
        new_files = curr_unlocked - prev_unlocked
        if not new_files:
            continue

        prev_hint      = set(steps[n - 1].get("hint_unlocked_files", []))
        curr_hint      = set(steps[n].get("hint_unlocked_files", []))
        hint_new_files = (curr_hint - prev_hint) & new_files
        self_new_files = new_files - hint_new_files

        recall = steps[n].get("recall", [])
        valid_recall = [r for r in recall if r in node_set]

        if not valid_recall:
            prev_node = steps[n - 1].get("node_id", "")
            if prev_node and prev_node in node_set:
                valid_recall = [prev_node]

        if not valid_recall:
            continue

        for new_f in self_new_files:
            if new_f not in node_set:
                continue
            for rec_f in valid_recall:
                adj.loc[rec_f, new_f] = 1

        for new_f in hint_new_files:
            if new_f not in node_set:
                continue
            for rec_f in valid_recall:
                adj.loc[rec_f, new_f] = 2  # hint overrides self

    return adj


def run_recall_graph_flow() -> None:
    """Interactive: scan ANALYSIS_DIR for full_steps JSON and build adjacency matrices.

    Human adjacency matrix is read from ``dataset/type_help-en/qa/human_adj.csv``.
    Output is written to ``analysis/dag/{model}/full_adj.csv``.
    """
    human_adj_csv = DATASET_DIR / "qa" / "human_adj.csv"
    if not human_adj_csv.exists():
        print(f"[error] Human adjacency matrix not found: {human_adj_csv}")
        return

    available: list[tuple[str, Path]] = []
    if ANALYSIS_DIR.exists():
        for folder in sorted(ANALYSIS_DIR.iterdir()):
            if not folder.is_dir():
                continue
            if "typehelp" not in folder.name and not folder.name.startswith("type_"):
                continue
            for json_file in sorted(folder.glob("*_full_steps.json")):
                model_label = folder.name.replace("_typehelp", "").replace("typehelp", "")
                available.append((model_label, json_file))

    if not available:
        print("No *_full_steps.json files found. Run operation 1 first.")
        return

    print("\nAvailable full_steps JSON files:")
    print("-" * 70)
    for i, (model, p) in enumerate(available):
        print(f"  [{i + 1:>2}] {model:20s}  {p.parent.name} / {p.name}")
    print("-" * 70)
    print("Enter file numbers to process (comma-separated, range like 1-3, all, q to go back):")
    raw = input("> ").strip()
    if raw.lower() == "q":
        return

    selected = parse_selection(raw, len(available))
    if not selected:
        print("No valid files selected.")
        return

    for i in selected:
        model, json_file = available[i]
        print(f"\n[{model}] Processing {json_file.name} ...")
        adj = build_recall_adj(json_file, human_adj_csv)

        out_dir = ANALYSIS_DIR / "dag" / model
        out_dir.mkdir(parents=True, exist_ok=True)

        out_path = out_dir / "full_adj.csv"
        adj.to_csv(out_path)

        # Record source path for use by the similarity computation
        (out_dir / "full_steps_source.txt").write_text(str(json_file), encoding="utf-8")

        self_edges = int((adj.values == 1).sum())
        hint_edges = int((adj.values == 2).sum())
        print(f"[{model}] Saved → {out_path}")
        print(f"[{model}] Self edges: {self_edges}  Hint edges: {hint_edges}  "
              f"Total: {self_edges + hint_edges}")

    print("\nProcessing complete.")


# ─────────────────────────── Operation 4: DAG similarity ─────────────────────

def compute_dag_similarity(
    human_adj: pd.DataFrame,
    model_adj: pd.DataFrame,
    restrict_nodes: list | None = None,
) -> dict:
    """Compute similarity metrics between a model DAG and the human DAG.

    Rules
    -----
    - human matrix: NaN/0 → no edge; 1 → edge
    - model matrix: 0/2   → no edge; 1 → edge
    Metrics are computed on the common-node submatrix.
    If ``restrict_nodes`` is given, only those nodes are considered.

    Returns
    -------
    dict with keys: common_nodes, human_edges, model_edges,
    TP, FP, FN, precision, recall, f1, jaccard.
    """
    def _dedup(df: pd.DataFrame) -> pd.DataFrame:
        """Merge duplicate rows/columns (take max) to ensure a square matrix."""
        df = df.fillna(0)
        df = df.groupby(df.index).max()
        df = df.T.groupby(df.T.index).max().T
        return df

    human_adj = _dedup(human_adj)
    model_adj = _dedup(model_adj)

    common_nodes = sorted(
        set(human_adj.index) & set(human_adj.columns) &
        set(model_adj.index) & set(model_adj.columns)
    )
    if restrict_nodes:
        restrict_set = set(restrict_nodes)
        common_nodes = [n for n in common_nodes if n in restrict_set]

    h = human_adj.loc[common_nodes, common_nodes].values
    m = model_adj.loc[common_nodes, common_nodes].values

    h_bin = (h == 1).astype(int).flatten()
    m_bin = (m == 1).astype(int).flatten()

    tp = int((h_bin & m_bin).sum())
    fp = int((m_bin & ~h_bin.astype(bool)).sum())
    fn = int((h_bin.astype(bool) & ~m_bin.astype(bool)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    jaccard   = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

    return {
        "common_nodes": len(common_nodes),
        "human_edges":  int(h_bin.sum()),
        "model_edges":  int(m_bin.sum()),
        "TP": tp, "FP": fp, "FN": fn,
        "precision": round(precision, 4),
        "recall":    round(recall,    4),
        "f1":        round(f1,        4),
        "jaccard":   round(jaccard,   4),
    }


def _get_unlocked_nodes(full_steps_path: Path) -> list:
    """Return the final unlocked node list from a full_steps JSON."""
    data = json.loads(full_steps_path.read_text(encoding="utf-8"))
    for step in reversed(data):
        unlocked = step.get("unlocked_files")
        if unlocked:
            return list(unlocked)
    return []


def run_similarity_flow() -> None:
    """Interactive: compute similarity between model DAGs and the human DAG.

    Comparison modes
    ----------------
    1. All nodes (full matrix comparison)
    2. Unlocked-nodes subgraph (nodes the model actually reached)
    3. Custom node-ID range (looked up in ``dataset/type_help-en/qa/node_id_map.csv``)

    Results are saved to ``analysis/dag/dag_similarity{suffix}.xlsx``.
    """
    dag_dir     = ANALYSIS_DIR / "dag"
    human_path  = DATASET_DIR / "qa" / "human_adj.csv"
    id_map_path = DATASET_DIR / "qa" / "node_id_map.csv"

    if not human_path.exists():
        print(f"[error] Human adjacency matrix not found: {human_path}")
        return

    print("\n  Similarity comparison mode:")
    print("  [1] All nodes (full matrix comparison)")
    print("  [2] Unlocked-nodes subgraph only")
    print("  [3] Specify node ID range (partial comparison)")
    sub = input("  Select > ").strip()
    if sub not in ("1", "2", "3"):
        print("Invalid input")
        return

    partial_nodes: list | None = None
    if sub == "3":
        print("  Enter node ID range, supported formats:")
        print("    Single: 5")
        print("    Range: 13-20 (inclusive)")
        print("    Combined: 1,3,5-10,15")
        raw = input("  Node ID > ").strip()
        partial_nodes = []
        for part in raw.split(","):
            part = part.strip()
            if "-" in part:
                try:
                    lo, hi = part.split("-", 1)
                    partial_nodes.extend(range(int(lo), int(hi) + 1))
                except ValueError:
                    print(f"  [warning] Cannot parse {part!r}, skipped")
            else:
                try:
                    partial_nodes.append(int(part))
                except ValueError:
                    print(f"  [warning] Cannot parse {part!r}, skipped")
        partial_ids = sorted(set(partial_nodes))
        if not partial_ids:
            print("No valid nodes specified, cancelling")
            return

        if not id_map_path.exists():
            print(f"[error] Node ID map not found: {id_map_path}")
            return
        id_map = pd.read_csv(id_map_path).set_index("id")["name"].to_dict()
        partial_nodes = []
        for nid in partial_ids:
            if nid in id_map:
                partial_nodes.append(id_map[nid])
            else:
                print(f"  [warning] ID {nid} not found in node_id_map.csv, skipped")
        if not partial_nodes:
            print("All specified IDs have no corresponding nodes, cancelling")
            return
        print(f"  Will compute for nodes ({len(partial_nodes)}): {partial_nodes}")

    human_adj = pd.read_csv(human_path, index_col=0)

    if not dag_dir.exists():
        print("[error] analysis/dag/ not found. Run operation 3 first.")
        return
    models = [d.name for d in sorted(dag_dir.iterdir())
              if d.is_dir() and (d / "full_adj.csv").exists()]
    if not models:
        print("[error] No model full_adj.csv found. Run operation 3 first.")
        return

    rows: list[dict] = []
    for model in models:
        model_path = dag_dir / model / "full_adj.csv"
        model_adj  = pd.read_csv(model_path, index_col=0)

        if sub == "2":
            source_txt = dag_dir / model / "full_steps_source.txt"
            if source_txt.exists():
                chosen = Path(source_txt.read_text(encoding="utf-8").strip())
                if not chosen.exists():
                    print(f"[{model}] Source file missing: {chosen}, skipped")
                    continue
            else:
                print(f"[{model}] full_steps_source.txt not found. Run operation 3 first, skipped")
                continue
            unlocked_nodes = _get_unlocked_nodes(chosen)
            print(f"[{model}] Using {chosen.name}, unlocked nodes = {len(unlocked_nodes)}")
            metrics = compute_dag_similarity(human_adj, model_adj,
                                             restrict_nodes=unlocked_nodes)
        elif sub == "3":
            metrics = compute_dag_similarity(human_adj, model_adj,
                                             restrict_nodes=partial_nodes)
        else:
            metrics = compute_dag_similarity(human_adj, model_adj)

        metrics["model"] = model
        rows.append(metrics)
        print(f"[{model}] nodes={metrics['common_nodes']}  "
              f"human_edges={metrics['human_edges']}  model_edges={metrics['model_edges']}  "
              f"TP={metrics['TP']} FP={metrics['FP']} FN={metrics['FN']}  "
              f"P={metrics['precision']}  R={metrics['recall']}  "
              f"F1={metrics['f1']}  Jaccard={metrics['jaccard']}")

    if not rows:
        print("[No results]")
        return

    result_df = pd.DataFrame(rows)[
        ["model", "common_nodes", "human_edges", "model_edges",
         "TP", "FP", "FN", "precision", "recall", "f1", "jaccard"]
    ]
    if sub == "2":
        suffix = "_unlocked"
    elif sub == "3":
        suffix = "_nodes" + "_".join(str(n) for n in partial_nodes[:5])
        if len(partial_nodes) > 5:
            suffix += f"_etc{len(partial_nodes)}"
    else:
        suffix = ""
    dag_dir.mkdir(parents=True, exist_ok=True)
    out_path = dag_dir / f"dag_similarity{suffix}.xlsx"
    result_df.to_excel(out_path, index=False)
    print(f"\n[Saved] {out_path}")


# ─────────────────────────── Interactive flows ────────────────────────────────

def run_extract_flow() -> None:
    """Interactive: select log files and extract structured step data."""
    all_files = collect_log_files()
    if not all_files:
        print(f"No log files found (search directory: {LOG_DIR}).")
        return

    show_file_list(all_files)
    print("Enter file numbers to process (comma-separated, range like 1-3, all, q to go back):")
    raw = input("> ").strip()
    if raw.lower() == "q":
        return

    selected = parse_selection(raw, len(all_files))
    if not selected:
        print("No valid files selected.")
        return

    print(f"\nWill process the following {len(selected)} files:")
    for i in selected:
        log_file, model_label = all_files[i]
        print(f"  [{i + 1}] {model_label} / {log_file.name}")

    if input("Confirm processing? (y/n) > ").strip().lower() != "y":
        print("Cancelled.")
        return

    for i in selected:
        log_file, model_label = all_files[i]
        process_log_file(log_file, model_label)

    print("\nProcessing complete.")


def run_gantt_flow() -> None:
    """Interactive: select processed JSON files and generate Gantt charts."""
    all_json = collect_analysis_files()
    if not all_json:
        print("No processed JSON files found. Run operation 1 first.")
        return

    show_json_list(all_json)
    print("Enter file numbers to plot (comma-separated, range like 1-3, all, q to go back):")
    raw = input("> ").strip()
    if raw.lower() == "q":
        return

    selected = parse_selection(raw, len(all_json))
    if not selected:
        print("No valid files selected.")
        return

    print(f"\nWill plot the following {len(selected)} files:")
    for i in selected:
        print(f"  [{i + 1}] {all_json[i].parent.name} / {all_json[i].name}")

    if input("Confirm plotting? (y/n) > ").strip().lower() != "y":
        print("Cancelled.")
        return

    for i in selected:
        _plot_json_file(all_json[i])

    print("\nPlotting complete.")


# ─────────────────────────── Entry point ─────────────────────────────────────

def main() -> None:
    while True:
        print("\n" + "=" * 40)
        print("  Select an operation:")
        print("  [1] Process log files (extract data and save JSON/Excel)")
        print("  [2] Plot Gantt charts (unlock timeline)")
        print("  [3] Build adjacency matrices from recall")
        print("  [4] Compute DAG similarity (compare with human)")
        print("  [q] Quit")
        print("=" * 40)
        choice = input("> ").strip().lower()

        if choice == "q":
            print("Exited.")
            break
        elif choice == "1":
            run_extract_flow()
        elif choice == "2":
            run_gantt_flow()
        elif choice == "3":
            run_recall_graph_flow()
        elif choice == "4":
            run_similarity_flow()
        else:
            print("Invalid input, please try again.")


if __name__ == "__main__":
    main()
