"""
summarize.py — Generate six-dimensional evaluation summary table from results.json
Usage:
    python3 summarize.py logs/trpg/20260309_133157/results.json
    python3 summarize.py logs/trpg/*/results.json   # compare multiple models
"""
import io
import json
import sys
from pathlib import Path


# ── Scoring rules ────────────────────────────────────────────────────────────────
# Per-dimension max weight (total weight = 10)
_WEIGHTS = {
    "Acc":   3.0,   # answer accuracy, most critical
    "Comp.": 1.5,   # comparison reasoning
    "Depth": 1.5,   # reasoning depth (D1/D2/D3 average)
    "Inst.": 0.5,   # format adherence (simpler task, lower weight)
    "Cit":   0.5,   # evidence grounding (current judge is lenient, lower weight)
    "Read.": 3.0,   # evidence coverage rate, independent of correctness
}

_TOTAL_WEIGHT = sum(_WEIGHTS.values())  # = 10.0

# Grade thresholds (percentage)
_GRADE = [(90, "A+"), (80, "A"), (70, "B+"), (60, "B"),
          (50, "C+"), (40, "C"), (30, "D"), (0, "F")]


def _grade(pct) -> str:
    if not isinstance(pct, (int, float)):
        return "-"
    for threshold, label in _GRADE:
        if pct >= threshold:
            return label
    return "F"


def compute_metrics(d: dict) -> dict:
    acc      = d.get("accuracy", {})
    comp     = d.get("comp_accuracy", {})
    inst     = d.get("inst_stats", {})
    cit      = d.get("cit_score_distribution", {})
    by_depth = d.get("accuracy_by_depth", {})
    read_cov = d.get("read_coverage", {})

    # Depth: three-level independent accuracy
    depth_map = {}
    for key, val in by_depth.items():
        try:
            level = int(str(key).split("_")[0])
        except ValueError:
            continue
        depth_map[level] = round(val["correct"] / val["total"] * 100, 2) if val.get("total") else 0.0

    d1 = depth_map.get(1, None)
    d2 = depth_map.get(2, None)
    d3 = depth_map.get(3, None)

    # Depth composite = simple average of D1/D2/D3 (included in Overall)
    depth_vals = [v for v in [d1, d2, d3] if v is not None]
    depth_avg  = round(sum(depth_vals) / len(depth_vals), 2) if depth_vals else None

    c_acc    = acc.get("pct", 0)
    comp_pct = comp.get("pct", 0)
    inst_pct = inst.get("pass_rate", 0)
    ecit_pct = cit.get("high_rate", 0)
    read_pct = read_cov.get("pct", None)

    # Overall = weighted average (max 100)
    dims = {
        "Acc":   c_acc,
        "Comp.": comp_pct,
        "Depth": depth_avg if depth_avg is not None else 0.0,
        "Inst.": inst_pct,
        "Cit":   ecit_pct,
        "Read.": read_pct if read_pct is not None else 0.0,
    }
    overall = sum(_WEIGHTS[k] * v for k, v in dims.items()) / _TOTAL_WEIGHT

    return {
        "model":   d.get("model", "unknown"),
        "story":   d.get("story", ""),
        # summary columns
        "Overall": round(overall, 2),
        "Acc":     round(c_acc, 2),
        "Comp.":   round(comp_pct, 2),
        "Depth":   depth_avg if depth_avg is not None else "-",
        "D1(S)":   round(d1, 2) if d1 is not None else "-",
        "D2(C)":   round(d2, 2) if d2 is not None else "-",
        "D3(X)":   round(d3, 2) if d3 is not None else "-",
        "Inst.":   round(inst_pct, 2),
        "Cit":     round(ecit_pct, 2),
        "Read.":   round(read_pct, 2) if read_pct is not None else "-",
        "n_qa":    acc.get("total", 0),
        # internal use for score breakdown table
        "_dims":   dims,
    }


def print_table(rows: list[dict]) -> None:
    cols = ["model", "story", "Overall", "Acc", "Comp.", "Depth",
            "D1(S)", "D2(C)", "D3(X)", "Inst.", "Cit", "Read.", "n_qa"]
    widths = {c: max(len(c), max(len(str(r[c])) for r in rows)) for c in cols}

    header = "  ".join(c.ljust(widths[c]) for c in cols)
    sep    = "  ".join("-" * widths[c] for c in cols)
    print(header)
    print(sep)
    for r in rows:
        print("  ".join(str(r[c]).ljust(widths[c]) for c in cols))


def print_score_table(rows: list[dict]) -> None:
    """Print per-model score breakdown table (raw % → weighted score → grade)"""
    score_cols = list(_WEIGHTS.keys())
    col_w = 9  # column width

    for r in rows:
        dims = r["_dims"]
        print(f"\n  {'─'*55}")
        print(f"  Score breakdown  model={r['model']}  story={r['story']}")
        print(f"  {'─'*55}")
        print(f"  {'Dimension':<10}  {'Wt':>4}  {'Raw (%)':>10}  {'Weighted':>8}  {'Grade':>5}")
        print(f"  {'-'*10}  {'-'*4}  {'-'*10}  {'-'*8}  {'-'*5}")
        total_weighted = 0.0
        for dim in score_cols:
            w     = _WEIGHTS[dim]
            raw   = dims.get(dim, 0.0) or 0.0
            wscore = w * raw / 100  # weighted score (max = weight)
            total_weighted += wscore
            grade = _grade(raw)
            print(f"  {dim:<10}  {w:>4.1f}  {raw:>10.2f}  {wscore:>8.4f}  {grade:>5}")
        print(f"  {'─'*55}")
        pct_overall = total_weighted / _TOTAL_WEIGHT * 100
        print(f"  {'Overall':<10}  {_TOTAL_WEIGHT:>4.1f}  "
              f"{'':>10}  {total_weighted:>8.4f}  "
              f"→ {pct_overall:.2f} / 100  {_grade(pct_overall)}")


def main():
    paths = sys.argv[1:]
    if not paths:
        print("Usage: python3 summarize.py <results.json> [results2.json ...]")
        sys.exit(1)

    rows = []
    first_dir = None
    for p in paths:
        path = Path(p)
        if not path.exists():
            print(f"[skip] file not found: {p}")
            continue
        d = json.loads(path.read_text())
        if not d.get("qa_results"):
            print(f"[skip] no QA results: {p}")
            continue
        rows.append(compute_metrics(d))
        if first_dir is None:
            first_dir = path.parent

    if not rows:
        print("No results files available")
        sys.exit(1)

    # ── Generate output content ─────────────────────────────────────────────────
    buf = io.StringIO()

    def out(*args, **kwargs):
        print(*args, **kwargs)           # print to terminal
        print(*args, **kwargs, file=buf) # also write to buffer

    out("\n=== Six-Dimensional Evaluation Summary ===\n")

    # manually build table rows (output to both terminal and buf)
    cols = ["model", "story", "Overall", "Acc", "Comp.", "Depth",
            "D1(S)", "D2(C)", "D3(X)", "Inst.", "Cit", "Read.", "n_qa"]
    widths = {c: max(len(c), max(len(str(r[c])) for r in rows)) for c in cols}
    header = "  ".join(c.ljust(widths[c]) for c in cols)
    sep    = "  ".join("-" * widths[c] for c in cols)
    out(header)
    out(sep)
    for r in rows:
        out("  ".join(str(r[c]).ljust(widths[c]) for c in cols))

    out()
    out("Note: Depth = average of D1(S)/D2(C)/D3(X) (included in Overall)")
    out("    D1(S)=Surface  D2(C)=Character  D3(X)=Cross-section")
    out("    Read. = evidence grounding coverage (independent of answer correctness)")
    out("    E.Cit = HIGH fraction; Inst = format pass rate")

    out("\n=== Overall Weighted Score Breakdown ===")
    out(f"  Weights: Acc×{_WEIGHTS['Acc']}  Read.×{_WEIGHTS['Read.']}  "
        f"Comp.×{_WEIGHTS['Comp.']}  Depth×{_WEIGHTS['Depth']}  "
        f"Inst.×{_WEIGHTS['Inst.']}  Cit×{_WEIGHTS['Cit']}  "
        f"(total weight={_TOTAL_WEIGHT})")

    score_cols = list(_WEIGHTS.keys())
    for r in rows:
        dims = r["_dims"]
        out(f"\n  {'─'*55}")
        out(f"  Score breakdown  model={r['model']}  story={r['story']}")
        out(f"  {'─'*55}")
        out(f"  {'Dimension':<10}  {'Wt':>4}  {'Raw (%)':>10}  {'Weighted':>8}  {'Grade':>5}")
        out(f"  {'-'*10}  {'-'*4}  {'-'*10}  {'-'*8}  {'-'*5}")
        total_weighted = 0.0
        for dim in score_cols:
            w      = _WEIGHTS[dim]
            raw    = dims.get(dim, 0.0) or 0.0
            wscore = w * raw / 100
            total_weighted += wscore
            grade  = _grade(raw)
            out(f"  {dim:<10}  {w:>4.1f}  {raw:>10.2f}  {wscore:>8.4f}  {grade:>5}")
        out(f"  {'─'*55}")
        pct_overall = total_weighted / _TOTAL_WEIGHT * 100
        out(f"  {'Overall':<10}  {_TOTAL_WEIGHT:>4.1f}  "
            f"{'':>10}  {total_weighted:>8.4f}  "
            f"→ {pct_overall:.2f} / 100  {_grade(pct_overall)}")

    out()
    out("Grade: A+(≥90) A(≥80) B+(≥70) B(≥60) C+(≥50) C(≥40) D(≥30) F(<30)")

    # ── Save to corresponding log directory ───────────────────────────────────────
    if first_dir is not None:
        out_path = first_dir / "summary.txt"
        out_path.write_text(buf.getvalue(), encoding="utf-8")
        print(f"\n[OK] Saved to: {out_path}")


if __name__ == "__main__":
    main()
