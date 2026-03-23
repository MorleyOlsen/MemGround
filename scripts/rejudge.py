"""
rejudge.py — Re-run judge evaluation on an existing results.json using a specified model.
Computes all six evaluation dimensions: Acc / Cit / Inst / Read / Comp / Depth.

Usage:
    python rejudge.py                          # uses _TARGET config below
    python rejudge.py --results logs/trpg/xxx/results.json  # override via CLI

CLI arguments take priority over _TARGET config.
"""

# ════════════════════════════════════════════════════════════════════════════
# Quick config — edit here, then run: python rejudge.py
# ════════════════════════════════════════════════════════════════════════════

_TARGET = {
    # path to target results.json
    "results":     "logs/trpg/20260309_133157/results.json",

    # output path (None = auto: results_rejudged.json in same directory)
    "output":      None,

    # story data directory
    "stories_dir": "dataset/trpg_en/stories",

    # skip certain judges (True = keep existing value, skip LLM call)
    "skip_acc":  False,
    "skip_inst": False,
    "skip_cit":  False,
    "skip_read": False,
}

# ════════════════════════════════════════════════════════════════════════════

import argparse
import json
import re
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

from openai import OpenAI

# ── Judge Prompts ────────────────────────────────────────────────────────────────



_JUDGE_SYSTEM_EN = """\
You are an objective and strict answer consistency evaluator.
Determine whether the "predicted answer" and the "gold answer" are semantically equivalent. Rules:
- CONSISTENT      : Both convey the same core meaning (different wording but referring to the same entity/relation/fact)
- INCONSISTENT    : They point to different entities or contain substantive contradictions
- UNDETERMINABLE  : The gold answer itself is ambiguous, or consistency cannot be determined

Note TRPG data specifics: player labels (playerX) are interchangeable with in-game character names; if the core fact matches, it is CONSISTENT.

Output ONLY JSON, nothing else:
{"result": "CONSISTENT"|"INCONSISTENT"|"UNDETERMINABLE", "reason": "brief reason (≤30 words)"}\
"""



_CIT_JUDGE_SYSTEM_EN = """\
You are an objective evidence-grounding evaluator.
Given the key evidence IDs for a question and the model's predicted answer, assess whether the answer correctly utilizes the information from those key story moments.

Rules:
- HIGH   : The predicted answer clearly reflects the information from the key events/dialogues indicated by the evidence (explicit ID citation is NOT required)
- MEDIUM : Partially reflects the evidence — uses some relevant information but misses major evidence points
- LOW    : Does not reflect the evidence, is based on incorrect information, or is unrelated to the evidenced events

Output ONLY JSON, nothing else:
{"cit_score": "HIGH"|"MEDIUM"|"LOW", "cit_reason": "brief reason (≤30 words)"}\
"""



_INST_JUDGE_SYSTEM_EN = """\
You are a strict format and reasoning quality evaluator.
Assess whether the model's answer meets the following instruction requirements:
1. Contains explicit reasoning with logical connectives (e.g., "because", "therefore", "this suggests", "taken together", "thus", "hence", "which means")
2. Has a clear structure with a distinct reasoning part and a conclusion
3. Appropriate length (not a single sentence, not excessively long)

Judgment:
- PASS : Fully satisfies all three criteria
- FAIL : Fails to meet any one criterion

Output ONLY JSON, nothing else:
{"inst_score": "PASS"|"FAIL", "inst_reason": "brief reason (≤20 words)"}\
"""



_READ_JUDGE_SYSTEM_EN = """\
You are a strict reading comprehension and evidence grounding evaluator.
You are given a question, the model's predicted answer (including reasoning), and the actual text of key evidence passages.
Judge whether the model's reasoning/answer genuinely uses the key information from the provided evidence.

Rules:
- HIGH   : The reasoning or answer clearly reflects core events, dialogues, or details from the evidence text (different wording is fine)
- MEDIUM : Partially reflects the evidence — mentions some related information but misses key evidence content
- LOW    : Largely ignores the evidence; the answer is based on guesses or unrelated information

Note: Verbatim quotation is NOT required. Judge whether the model semantically "read and understood" the evidence.

Output ONLY JSON, nothing else:
{"read_score": "HIGH"|"MEDIUM"|"LOW", "read_reason": "brief reason (≤30 words)"}\
"""

_DEPTH_MAP   = {5: 1, 1: 2, 2: 2, 3: 3, 4: 3}
_DEPTH_LABEL = {1: "Surface", 2: "Character", 3: "Cross-section"}


# ── Helpers ──────────────────────────────────────────────────────────────────────

def normalize(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r'[\s\u3000，。！？、；：""''（）【】《》,.!?;:\'"()\[\]{}\-—]+', "", text)
    return text


def parse_json(s: str) -> dict:
    s = (s or "").strip()
    if s.startswith("```"):
        lines = s.split("\n")
        s = "\n".join(lines[1:-1]) if len(lines) > 2 else s.strip("`")
    if "{" in s and "}" in s:
        s = s[s.find("{"):s.rfind("}") + 1]
    try:
        return json.loads(s)
    except Exception:
        return {}


def parse_qa_output(raw: str, is_en: bool = True) -> tuple:
    r_marker, a_marker = "Reasoning:", "Answer:"
    raw = (raw or "").strip()
    r_idx = raw.find(r_marker)
    a_idx = raw.find(a_marker)
    if r_idx != -1 and a_idx != -1 and a_idx > r_idx:
        return raw[r_idx + len(r_marker):a_idx].strip(), raw[a_idx + len(a_marker):].strip()
    if a_idx != -1:
        return "", raw[a_idx + len(a_marker):].strip()
    return "", raw


def call_judge(client: OpenAI, model: str, messages: list, max_retries: int = 3) -> str:
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model, messages=messages, temperature=0.0, max_tokens=200,
            )
            return resp.choices[0].message.content or ""
        except Exception as exc:
            print(f"  [error] attempt {attempt+1}: {exc}")
            if attempt < max_retries - 1:
                time.sleep(5)
    return ""


def build_evidence_lookup(stories_dir: str, story_name: str) -> Dict[str, str]:
    story_path = Path(stories_dir) / story_name
    lookup: Dict[str, str] = {}
    if not story_path.exists():
        print(f"[warning] story directory not found: {story_path}", file=sys.stderr)
        return lookup
    for fn in sorted(story_path.glob("*.json")):
        m = re.match(r'^(\d+)_', fn.name)
        if not m:
            continue
        sec_num = int(m.group(1))
        try:
            data = json.loads(fn.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[warning] failed to read {fn.name}: {e}", file=sys.stderr)
            continue
        for idx, conv in enumerate(data.get("conversation", [])):
            key = f"D{sec_num:02d}:{idx}"
            lookup[key] = conv.get("text", "")
    print(f"[OK] built evidence lookup: {len(lookup)} entries")
    return lookup


# ── Judge Functions ──────────────────────────────────────────────────────────────

def judge_acc(client, model, question, gold, predicted, is_en) -> Tuple[str, str]:
    if not predicted.strip():
        return "INCONSISTENT", "predicted answer is empty"
    if normalize(gold) == normalize(predicted):
        return "CONSISTENT", "exact match after normalization"
    sys_prompt = _JUDGE_SYSTEM_EN
    user_msg = f"Question: {question}\nGold answer: {gold}\nPredicted answer: {predicted}\n\nJudge:"
    raw    = call_judge(client, model, [{"role": "system", "content": sys_prompt},
                                        {"role": "user",   "content": user_msg}])
    parsed = parse_json(raw)
    result = parsed.get("result", "")
    reason = parsed.get("reason", "")
    if result in ("CONSISTENT", "INCONSISTENT", "UNDETERMINABLE"):
        return result, reason
    return "UNDETERMINABLE", "judge parse failed"


def judge_cit(client, model, question, evidence, predicted, reasoning, is_en) -> Tuple[str, str]:
    if not predicted.strip():
        return "LOW", "predicted answer is empty"
    evidence_str = ", ".join(evidence) if evidence else "(no evidence)"
    combined     = (reasoning + "\n" + predicted).strip() if reasoning else predicted
    sys_prompt = _CIT_JUDGE_SYSTEM_EN
    user_msg = (f"Question: {question}\n"
                f"Key evidence IDs: {evidence_str}\n"
                f"Predicted answer: {combined}\n\nEvaluate:")
    raw    = call_judge(client, model, [{"role": "system", "content": sys_prompt},
                                        {"role": "user",   "content": user_msg}])
    parsed = parse_json(raw)
    score  = parsed.get("cit_score", "")
    reason = parsed.get("cit_reason", "")
    if score in ("HIGH", "MEDIUM", "LOW"):
        return score, reason
    return "LOW", "citation judge parse failed"


def judge_inst(client, model, reasoning, answer, is_en) -> Dict:
    if not (reasoning or answer).strip():
        return {"_inst_pass": False, "_inst_reason": "empty output"}
    combined   = (reasoning + "\n" + answer).strip()
    sys_prompt = _INST_JUDGE_SYSTEM_EN
    user_msg = f"Model answer:\n{combined}\n\nEvaluate:"
    raw    = call_judge(client, model, [{"role": "system", "content": sys_prompt},
                                        {"role": "user",   "content": user_msg}])
    parsed = parse_json(raw)
    score  = parsed.get("inst_score", "")
    reason = parsed.get("inst_reason", "")
    return {
        "_inst_pass":   score == "PASS",
        "_inst_reason": reason or "inst judge parse failed",
    }


def judge_read(client, model, question, reasoning, answer, evidence_ids, lookup, is_en) -> Tuple[str, str]:
    if not evidence_ids or not lookup:
        return "LOW", "no evidence"
    ev_texts = [f"[{eid}] {lookup[eid]}" for eid in evidence_ids if lookup.get(eid)]
    if not ev_texts:
        return "LOW", "evidence text not found"
    evidence_block = "\n".join(ev_texts)
    combined       = (reasoning + "\n" + answer).strip() if reasoning else answer
    sys_prompt = _READ_JUDGE_SYSTEM_EN
    user_msg = (f"Question: {question}\n\n"
                f"Evidence passages:\n{evidence_block}\n\n"
                f"Model's reasoning and answer:\n{combined}\n\nEvaluate:")
    raw    = call_judge(client, model, [{"role": "system", "content": sys_prompt},
                                        {"role": "user",   "content": user_msg}])
    parsed = parse_json(raw)
    score  = parsed.get("read_score", "")
    reason = parsed.get("read_reason", "")
    if score in ("HIGH", "MEDIUM", "LOW"):
        return score, reason
    return "LOW", "read judge parse failed"


# ── Summary Stats ────────────────────────────────────────────────────────────────

def compute_summary_stats(results: list) -> dict:
    n = len(results)

    # Acc
    correct        = sum(1 for r in results if r.get("_judge_result") == "CONSISTENT")
    incorrect      = sum(1 for r in results if r.get("_judge_result") == "INCONSISTENT")
    undeterminable = sum(1 for r in results if r.get("_judge_result") == "UNDETERMINABLE")

    by_cat: Dict = {}
    for r in results:
        cat = r.get("category", "?")
        by_cat.setdefault(cat, []).append(r["_judge_result"])
    acc_by_cat = {
        str(cat): {"correct": sum(1 for x in rs if x == "CONSISTENT"), "total": len(rs),
                   "pct": round(sum(1 for x in rs if x == "CONSISTENT") / len(rs) * 100, 2)}
        for cat, rs in sorted(by_cat.items(), key=lambda x: str(x[0]))
    }

    # Depth
    by_depth: Dict = {}
    for r in results:
        cat = r.get("category")
        d   = _DEPTH_MAP.get(cat, 2)
        r.setdefault("depth", d)
        by_depth.setdefault(d, []).append(r["_judge_result"])
    acc_by_depth = {}
    for d, rs in sorted(by_depth.items()):
        c = sum(1 for x in rs if x == "CONSISTENT")
        acc_by_depth[f"{d}_{_DEPTH_LABEL.get(d, str(d))}"] = {
            "correct": c, "total": len(rs),
            "pct": round(c / len(rs) * 100, 2) if rs else 0.0,
        }

    # Read complexity (kept for backward compatibility)
    by_read: Dict = {}
    for r in results:
        ev = len(r.get("evidence", []))
        bucket = "low(≤2)" if ev <= 2 else ("mid(3-4)" if ev <= 4 else "high(≥5)")
        by_read.setdefault(bucket, []).append(r["_judge_result"])
    acc_by_read = {
        b: {"correct": sum(1 for x in rs if x == "CONSISTENT"), "total": len(rs),
            "pct": round(sum(1 for x in rs if x == "CONSISTENT") / len(rs) * 100, 2)}
        for b, rs in by_read.items()
    }

    # Comp
    comp_r   = [r for r in results if r.get("category") in (2, 5)]
    comp_cor = sum(1 for r in comp_r if r.get("_judge_result") == "CONSISTENT")

    # Inst
    inst_r    = [r for r in results if "_inst_pass" in r]
    inst_pass = sum(1 for r in inst_r if r.get("_inst_pass"))

    # Cit
    cit_r    = [r for r in results if "_cit_score" in r]
    cit_dist: Dict[str, int] = {}
    for r in cit_r:
        s = r["_cit_score"]
        cit_dist[s] = cit_dist.get(s, 0) + 1

    # Read coverage (LLM judge: HIGH/MEDIUM/LOW)
    read_r    = [r for r in results if "_read_score" in r and r["_read_score"] in ("HIGH", "MEDIUM", "LOW")]
    read_dist: Dict[str, int] = {}
    for r in read_r:
        s = r["_read_score"]
        read_dist[s] = read_dist.get(s, 0) + 1
    if read_r:
        read_avg = (read_dist.get("HIGH", 0) * 1.0 + read_dist.get("MEDIUM", 0) * 0.5) / len(read_r)
    else:
        read_avg = 0.0

    return {
        "accuracy": {
            "total": n, "correct": correct, "incorrect": incorrect,
            "undeterminable": undeterminable,
            "pct": round(correct / n * 100, 2) if n else 0.0,
        },
        "accuracy_by_category":       acc_by_cat,
        "accuracy_by_depth":          acc_by_depth,
        "accuracy_by_read_complexity": acc_by_read,
        "comp_accuracy": {
            "total": len(comp_r), "correct": comp_cor,
            "pct": round(comp_cor / len(comp_r) * 100, 2) if comp_r else 0.0,
            "note": "category 2 (contradiction) + category 5 (skill comparison)",
        },
        "inst_stats": {
            "total": len(inst_r), "pass": inst_pass, "fail": len(inst_r) - inst_pass,
            "pass_rate": round(inst_pass / len(inst_r) * 100, 2) if inst_r else 0.0,
        },
        "cit_score_distribution": {
            **{k: cit_dist.get(k, 0) for k in ("HIGH", "MEDIUM", "LOW")},
            "total": len(cit_r),
            "high_rate": round(cit_dist.get("HIGH", 0) / len(cit_r) * 100, 2) if cit_r else 0.0,
        },
        "read_coverage": {
            **{k: read_dist.get(k, 0) for k in ("HIGH", "MEDIUM", "LOW")},
            "total":      len(read_r),
            "avg_score":  round(read_avg, 4),
            "pct":        round(read_avg * 100, 2),
            "high_rate":  round(read_dist.get("HIGH", 0) / len(read_r) * 100, 2) if read_r else 0.0,
        },
    }


# ── Main ─────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results",     default=None, help="path to results.json (overrides _TARGET)")
    parser.add_argument("--api_key",     default=None)
    parser.add_argument("--base_url",    default=None)
    parser.add_argument("--model",       default=None)
    parser.add_argument("--output",      default=None, help="output path (overrides _TARGET)")
    parser.add_argument("--stories-dir", default=None, help="story data directory (overrides _TARGET)")
    parser.add_argument("--skip_acc",  action="store_true")
    parser.add_argument("--skip_cit",  action="store_true")
    parser.add_argument("--skip_inst", action="store_true")
    parser.add_argument("--skip_read", action="store_true")
    args = parser.parse_args()

    # CLI args take priority, fall back to _TARGET config
    results_str  = args.results      or _TARGET["results"]
    output_str   = args.output       or _TARGET["output"]
    stories_str  = args.stories_dir  or _TARGET["stories_dir"]
    skip_acc     = args.skip_acc  or _TARGET["skip_acc"]
    skip_inst    = args.skip_inst or _TARGET["skip_inst"]
    skip_cit     = args.skip_cit  or _TARGET["skip_cit"]
    skip_read    = args.skip_read or _TARGET["skip_read"]

    results_path = Path(results_str)
    if not results_path.exists():
        print(f"file not found: {results_path}")
        sys.exit(1)

    # read config.yaml
    config_path = Path(__file__).parent.parent / "config.yaml"
    api_key, base_url, model = args.api_key, args.base_url, args.model
    if not all([api_key, base_url, model]):
        try:
            import yaml
            cfg  = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            j    = cfg.get("judge_llm") or cfg.get("llm", {})
            api_key  = api_key  or j.get("api_key")
            base_url = base_url or j.get("base_url")
            model    = model    or j.get("model")
        except Exception as e:
            print(f"failed to read config.yaml: {e}")
            sys.exit(1)

    if not all([api_key, base_url, model]):
        print("missing api_key / base_url / model")
        sys.exit(1)

    print(f"Judge model: {model}  ({base_url})")

    client = OpenAI(api_key=api_key, base_url=base_url, timeout=180.0)

    with open(results_path, encoding="utf-8") as f:
        data = json.load(f)

    is_en      = True  # English-only benchmark
    story_name = data.get("story", "")
    qa_results = data.get("qa_results", [])
    total      = len(qa_results)
    print(f"Total {total} items, story={story_name}\n")

    # build evidence lookup (for Read judge)
    lookup: Dict[str, str] = {}
    if not skip_read and story_name:
        stories_dir = Path(stories_str)
        if not stories_dir.is_absolute():
            root = Path(__file__).resolve().parent.parent
            if (root / stories_dir).exists():
                stories_dir = root / stories_dir
        lookup = build_evidence_lookup(str(stories_dir), story_name)

    new_results = deepcopy(qa_results)

    for i, qa in enumerate(new_results):
        question = qa.get("question", "")
        gold     = qa.get("answer", "")
        evidence = qa.get("evidence", [])

        # use existing structured output
        reasoning = qa.get("_predicted_reasoning", "")
        predicted = qa.get("_predicted_answer", "")
        if not reasoning:
            reasoning, predicted = parse_qa_output(predicted, is_en)

        # Acc
        if not skip_acc:
            result, reason = judge_acc(client, model, question, gold, predicted, is_en)
            qa["_judge_result"] = result
            qa["_judge_reason"] = reason
        else:
            result = qa.get("_judge_result", "UNDETERMINABLE")

        # Inst
        if not skip_inst:
            inst_info = judge_inst(client, model, reasoning, predicted, is_en)
            qa.update(inst_info)

        # Cit
        if not skip_cit:
            cit_score, cit_reason = judge_cit(
                client, model, question, evidence, predicted, reasoning, is_en
            )
            qa["_cit_score"]  = cit_score
            qa["_cit_reason"] = cit_reason
        else:
            cit_score = qa.get("_cit_score", "—")

        # Read
        if not skip_read:
            read_score, read_reason = judge_read(
                client, model, question, reasoning, predicted, evidence, lookup, is_en
            )
            qa["_read_score"]  = read_score
            qa["_read_reason"] = read_reason
        else:
            read_score = qa.get("_read_score", "—")

        mark      = "✓" if result == "CONSISTENT" else ("?" if result == "UNDETERMINABLE" else "✗")
        inst_mark = "✓" if qa.get("_inst_pass") else "✗"
        print(f"[{i+1:3d}/{total}] {mark} {result:<16}  "
              f"cit={cit_score}  read={read_score}  inst={inst_mark}")

    # aggregate stats
    stats = compute_summary_stats(new_results)

    out_data = deepcopy(data)
    out_data["model"]     = model
    out_data["timestamp"] = datetime.now().isoformat()
    out_data.update(stats)
    out_data["qa_results"] = new_results

    out_path = Path(output_str) if output_str else results_path.parent / "results_rejudged.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False, indent=2)

    acc  = stats["accuracy"]
    inst = stats["inst_stats"]
    cit  = stats["cit_score_distribution"]
    read = stats["read_coverage"]
    print(f"\n{'='*55}")
    print(f"Acc   : {acc['correct']}/{acc['total']} = {acc['pct']}%")
    print(f"Comp  : {stats['comp_accuracy']['correct']}/{stats['comp_accuracy']['total']} = {stats['comp_accuracy']['pct']}%")
    print(f"Depth : {stats['accuracy_by_depth']}")
    print(f"Inst  : pass_rate={inst['pass_rate']}%  ({inst['pass']}/{inst['total']})")
    print(f"Cit   : HIGH={cit['HIGH']}  MEDIUM={cit['MEDIUM']}  LOW={cit['LOW']}  high_rate={cit['high_rate']}%")
    print(f"Read  : HIGH={read['HIGH']}  MEDIUM={read['MEDIUM']}  LOW={read['LOW']}  pct={read['pct']}%")
    print(f"\nResults saved to: {out_path}")

    # auto-generate summary.txt (same directory as results_rejudged.json)
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from summarize import compute_metrics, _WEIGHTS, _TOTAL_WEIGHT, _grade
        import io
        row  = compute_metrics(out_data)
        buf  = io.StringIO()
        def out(*args, **kwargs):
            print(*args, **kwargs)
            print(*args, **kwargs, file=buf)
        cols = ["model", "story", "Overall", "Acc", "Comp.", "Depth",
                "D1(S)", "D2(C)", "D3(X)", "Inst.", "Cit", "Read.", "n_qa"]
        widths = {c: max(len(c), len(str(row[c]))) for c in cols}
        out("\n=== Six-Dimensional Evaluation Summary ===\n")
        out("  ".join(c.ljust(widths[c]) for c in cols))
        out("  ".join("-" * widths[c] for c in cols))
        out("  ".join(str(row[c]).ljust(widths[c]) for c in cols))
        out("\n=== Overall Weighted Score Breakdown ===")
        out(f"  Weights: Acc×{_WEIGHTS['Acc']}  Read.×{_WEIGHTS['Read.']}  "
            f"Comp.×{_WEIGHTS['Comp.']}  Depth×{_WEIGHTS['Depth']}  "
            f"Inst.×{_WEIGHTS['Inst.']}  Cit×{_WEIGHTS['Cit']}  (total weight={_TOTAL_WEIGHT})")
        dims = row["_dims"]
        out(f"\n  {'─'*55}")
        out(f"  Score breakdown  model={row['model']}  story={row['story']}")
        out(f"  {'─'*55}")
        out(f"  {'Dimension':<10}  {'Wt':>4}  {'Raw (%)':>10}  {'Weighted':>8}  {'Grade':>5}")
        out(f"  {'-'*10}  {'-'*4}  {'-'*10}  {'-'*8}  {'-'*5}")
        total_w = 0.0
        for dim in list(_WEIGHTS.keys()):
            w = _WEIGHTS[dim]; raw = dims.get(dim, 0.0) or 0.0
            ws = w * raw / 100; total_w += ws
            out(f"  {dim:<10}  {w:>4.1f}  {raw:>10.2f}  {ws:>8.4f}  {_grade(raw):>5}")
        out(f"  {'─'*55}")
        pct_ov = total_w / _TOTAL_WEIGHT * 100
        out(f"  {'Overall':<10}  {_TOTAL_WEIGHT:>4.1f}  {'':>10}  {total_w:>8.4f}  "
            f"→ {pct_ov:.2f} / 100  {_grade(pct_ov)}")
        out("\nGrade: A+(≥90) A(≥80) B+(≥70) B(≥60) C+(≥50) C(≥40) D(≥30) F(<30)")
        summary_path = out_path.parent / "summary.txt"
        summary_path.write_text(buf.getvalue(), encoding="utf-8")
        print(f"[Summary] saved to: {summary_path}")
    except Exception as e:
        print(f"[warning] failed to generate summary: {e}")


if __name__ == "__main__":
    main()
