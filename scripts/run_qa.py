#!/usr/bin/env python3
"""
run_qa.py — QA evaluation script for Dust / TypeHelp games

Two calling modes:
  1. Standalone CLI  : python scripts/run_qa.py [--game_type dust|type_help]
     Loads memory from a checkpoint file (path configured at top of file).
  2. Inline (auto)   : called from galAgent.py after game ends via run_qa_from_store().
     Uses the live MemoryStore directly — no checkpoint file needed.

Pipeline:
  1. Load QA items for the selected game
  2. Build story context (from store + optional JSONL log)
  3. For each QA: answer with llm → score with judge_llm on 4 dimensions
  4. Save logs/{game_type}/{session_id}/results.json + print summary
"""

from __future__ import annotations

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ── Standalone mode config (edit here for CLI usage) ────────────────────────
GAME_TYPE = "type_help"   # "dust" or "type_help"

CHECKPOINT_PATH = {
    "dust":      "logs/dust/dust_20260224_210044/checkpoints/step_000599.json",
    "type_help": "logs/type_help/type_help_20260224_210044/checkpoints/step_000199.json",
}
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from openai import OpenAI

# ── Add project root to path for galagent imports ──────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from galagent.memory.store import MemoryStore

# ── Game routing config ─────────────────────────────────────────────────────────
GAME_CONFIG: Dict[str, Optional[Dict]] = {
    "dust": {
        "qa_path":    "dataset/The_dust_settles-en/dust_qa_eval.json",
        "stories_dir": "dataset/The_dust_settles-en/stories",
        "story_name": "The_dust_settles_en",
        "language":   "en",
    },
    "type_help": {
        "qa_path":    "dataset/type_help-en/type_help_qa_eval.json",
        "stories_dir": "dataset/type_help-en/stories",
        "story_name": "type_help_en",
        "language":   "en",
    },
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ── Prompt constants ────────────────────────────────────────────────────────────
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_QA_SYSTEM_EN = """\
You are an expert in deep reading comprehension of detective/mystery game records.
You have read the interrogation logs of a case in order (earlier content may have been \
summarized due to context limits).
Based on your memory and current context, respond using the following fixed format \
(do NOT omit either label):

Reasoning: [2-4 sentences of analytical reasoning, using logical connectives such as \
"because... therefore...", "this suggests...", "taken together...", to show your reasoning chain]
Answer: [1-2 sentence concise conclusion that directly answers the question]

Answer in English. All content must be grounded in the story.\
"""

_QA_SYSTEM_TYPE_HELP_EN = """\
You are an expert in deep reading comprehension of mystery puzzle game records.
You have explored a series of files in a mansion investigation game in order (earlier content \
may have been summarized due to context limits). Each file reveals information about characters, \
locations, and events at specific times.
Based on your memory and current context, respond using the following fixed format \
(do NOT omit either label):

Reasoning: [2-4 sentences of analytical reasoning, using logical connectives such as \
"because... therefore...", "this suggests...", "taken together...", to show your reasoning chain]
Answer: [1-2 sentence concise conclusion that directly answers the question]

Answer in English. All content must be grounded in the files you have read.\
"""

_JUDGE_SYSTEM_EN = """\
You are an objective and strict answer consistency evaluator.
Determine whether the "predicted answer" and the "gold answer" are semantically equivalent. Rules:
- CONSISTENT      : Both convey the same core meaning (different wording but referring to the same entity/relation/fact)
- INCONSISTENT    : They point to different entities or contain substantive contradictions
- UNDETERMINABLE  : The gold answer itself is ambiguous, or consistency cannot be determined

Output ONLY JSON, nothing else:
{"result": "CONSISTENT"|"INCONSISTENT"|"UNDETERMINABLE", "reason": "brief reason (≤30 words)"}\
"""

_CIT_JUDGE_SYSTEM_EN = """\
You are an objective evidence-grounding evaluator.
Given the key evidence IDs for a question and the model's predicted answer, assess whether \
the answer correctly utilizes the information from those key story moments.

Rules:
- HIGH   : The predicted answer clearly reflects the information from the key events/dialogues \
indicated by the evidence (explicit ID citation is NOT required)
- MEDIUM : Partially reflects the evidence — uses some relevant information but misses major evidence points
- LOW    : Does not reflect the evidence, is based on incorrect information, or is unrelated to \
the evidenced events

Note: Evaluate content grounding, not explicit citation of IDs.

Output ONLY JSON, nothing else:
{"cit_score": "HIGH"|"MEDIUM"|"LOW", "cit_reason": "brief reason (≤30 words)"}\
"""

_INST_JUDGE_SYSTEM_EN = """\
You are a strict format and reasoning quality evaluator.
Assess whether the model's answer meets the following instruction requirements:
1. Contains explicit reasoning with logical connectives (e.g., "because", "therefore", \
"this suggests", "taken together", "thus", "hence", "which means")
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
You are given a question, the model's predicted answer (including reasoning), and the actual \
text of key evidence passages.
Judge whether the model's reasoning/answer genuinely uses the key information from the \
provided evidence.

Rules:
- HIGH   : The reasoning or answer clearly reflects core events, dialogues, or details from the \
evidence text (different wording is fine)
- MEDIUM : Partially reflects the evidence — mentions some related information but misses key \
evidence content
- LOW    : Largely ignores the evidence; the answer is based on guesses or unrelated information

Note: Verbatim quotation is NOT required. Judge whether the model semantically "read and \
understood" the evidence.

Output ONLY JSON, nothing else:
{"read_score": "HIGH"|"MEDIUM"|"LOW", "read_reason": "brief reason (≤30 words)"}\
"""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ── Utility functions ────────────────────────────────────────────────────────────
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _parse_json(s: str) -> Dict[str, str]:
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


def _parse_qa_output(raw: str) -> Tuple[str, str]:
    """Parse Reasoning: ... Answer: ... format"""
    raw = (raw or "").strip()
    r_marker, a_marker = "Reasoning:", "Answer:"
    r_idx = raw.find(r_marker)
    a_idx = raw.find(a_marker)
    if r_idx != -1 and a_idx != -1 and a_idx > r_idx:
        reasoning = raw[r_idx + len(r_marker):a_idx].strip()
        answer    = raw[a_idx + len(a_marker):].strip()
        return reasoning, answer
    if a_idx != -1:
        return "", raw[a_idx + len(a_marker):].strip()
    return "", raw


def build_evidence_lookup(stories_dir: Path, story_name: str) -> Dict[str, str]:
    """Build D{n:02d}:{idx} → text lookup table"""
    story_path = stories_dir / story_name
    lookup: Dict[str, str] = {}
    if not story_path.exists():
        return lookup
    for fn in sorted(story_path.glob("*.json")):
        m = re.match(r'^(T?)(\d+)_', fn.name)
        if not m:
            continue
        prefix  = m.group(1) or "D"  # T or empty string (empty → dust uses D)
        sec_num = int(m.group(2))    # number part moved to group 2
        try:
            data = json.loads(fn.read_text(encoding="utf-8"))
        except Exception:
            continue
        for idx, conv in enumerate(data.get("conversation", [])):
            key = f"{prefix}{sec_num:02d}:{idx}"  # dust→D01:0，type_help→T01:0
            lookup[key] = conv.get("text", "")
    return lookup


def _find_memory_jsonl(ckpt_path: Path) -> Optional[Path]:
    """Find memory_*.jsonl files in same directory as checkpoint, return the latest one"""
    candidates = sorted(ckpt_path.parent.glob("memory_*.jsonl"),
                        key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def _load_jsonl_messages(jsonl_path: Path, last_n_steps: int = 100) -> List[Dict[str, str]]:
    """Read JSONL memory file, keep only the latest last_n_steps messages"""
    entries = []
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
            if entry.get("content", ""):
                entries.append(entry)
        except json.JSONDecodeError:
            continue

    # find max step, keep only the latest last_n_steps steps
    max_step = max((e.get("meta", {}).get("step", 0) for e in entries), default=0)
    min_step = max(0, max_step - last_n_steps + 1)
    print(f"[Memory JSONL] total_entries={len(entries)}  max_step={max_step}  "
          f"keeping step {min_step}~{max_step} (last {last_n_steps} steps)")

    return [
        {"role": e.get("role", "user"), "content": e["content"]}
        for e in entries
        if e.get("meta", {}).get("step", 0) >= min_step
    ]


def _call_llm(
    client: OpenAI,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.2,
    max_retries: int = 3,
) -> str:
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
            )
            return resp.choices[0].message.content or ""
        except Exception as exc:
            print(f"  [LLM] attempt {attempt+1}/{max_retries} failed: {exc}")
            if attempt < max_retries - 1:
                time.sleep(5)
    return ""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ── Shared helpers ──────────────────────────────────────────────────────────────
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _build_story_messages(
    store: MemoryStore,
    memory_log_file: Optional[str],
) -> List[Dict[str, str]]:
    """Build LLM context messages from a MemoryStore + optional JSONL log file.

    Used by both standalone run_qa() and inline run_qa_from_store().
    """
    ckpt_messages = store.to_chat_messages()
    print(f"[Memory] store items: {len(store.items)}")

    if memory_log_file and Path(memory_log_file).exists():
        jsonl_messages = _load_jsonl_messages(Path(memory_log_file))
        ckpt_contents  = {m["content"] for m in ckpt_messages}
        unique_jsonl   = [m for m in jsonl_messages if m["content"] not in ckpt_contents]
        print(f"[Memory JSONL] {Path(memory_log_file).name}  "
              f"total={len(jsonl_messages)}  unique={len(unique_jsonl)}")
        return (
            [{"role": "system", "content":
              "[Compressed Memory State] The following is the agent's condensed memory "
              "from the game session:"}]
            + ckpt_messages
            + [{"role": "system", "content":
                "[Supplementary Game History] The following are additional game records "
                "(deduplicated):"}]
            + unique_jsonl
        )

    if memory_log_file:
        print("[Memory JSONL] NOT FOUND — using store memory only")
    return ckpt_messages


def _load_qa_data(game_type: str) -> Tuple[List[Dict[str, Any]], Dict[str, str], str, str]:
    """Load QA list, evidence lookup, story_name, and language for a game type.

    Returns: (qa_list, ev_lookup, story_name, language)
    """
    gcfg        = GAME_CONFIG[game_type]
    qa_path     = _ROOT / gcfg["qa_path"]
    stories_dir = _ROOT / gcfg["stories_dir"]
    story_name  = gcfg["story_name"]
    language    = gcfg["language"]

    raw_data = json.loads(qa_path.read_text(encoding="utf-8"))
    qa_list: List[Dict[str, Any]] = raw_data[0]["qa"]
    ev_lookup = build_evidence_lookup(stories_dir, story_name)

    print(f"[QA] loaded {len(qa_list)} QA items from {qa_path.name}")
    print(f"[Evidence] lookup: {len(ev_lookup)} entries")
    return qa_list, ev_lookup, story_name, language


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ── Judge functions ─────────────────────────────────────────────────────────────
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def judge_answer(
    client: OpenAI, model: str,
    question: str, gold: str, predicted: str,
) -> Tuple[str, str]:
    if not predicted.strip():
        return "INCONSISTENT", "predicted answer is empty"
    # fast normalized comparison
    def _norm(t):
        t = t.strip().lower()
        return re.sub(r'[\s\u3000,.!?;:\'"()\[\]{}\-—]+', "", t)
    if _norm(gold) == _norm(predicted):
        return "CONSISTENT", "exact match after normalization"

    msgs = [
        {"role": "system", "content": _JUDGE_SYSTEM_EN},
        {"role": "user", "content":
            f"Question: {question}\nGold answer: {gold}\nPredicted answer: {predicted}\n\nJudge:"},
    ]
    raw    = _call_llm(client, model, msgs, temperature=0.0)
    parsed = _parse_json(raw)
    result = parsed.get("result", "")
    reason = parsed.get("reason", "")
    if result in ("CONSISTENT", "INCONSISTENT", "UNDETERMINABLE"):
        return result, reason
    return "UNDETERMINABLE", "judge parse failed"


def judge_inst(
    client: OpenAI, model: str,
    reasoning: str, answer: str,
) -> Dict[str, Any]:
    if not (reasoning or answer).strip():
        return {"_inst_pass": False, "_inst_reason": "empty output"}
    combined = (reasoning + "\n" + answer).strip()
    msgs = [
        {"role": "system", "content": _INST_JUDGE_SYSTEM_EN},
        {"role": "user", "content": f"Model answer:\n{combined}\n\nEvaluate:"},
    ]
    raw    = _call_llm(client, model, msgs, temperature=0.0)
    parsed = _parse_json(raw)
    score  = parsed.get("inst_score", "")
    reason = parsed.get("inst_reason", "")
    return {
        "_inst_pass":   score == "PASS",
        "_inst_reason": reason or "inst judge parse failed",
    }


def judge_citation(
    client: OpenAI, model: str,
    question: str, evidence: List[str], predicted_answer: str, predicted_reasoning: str,
) -> Tuple[str, str]:
    if not predicted_answer.strip():
        return "LOW", "predicted answer is empty"
    evidence_str = ", ".join(evidence) if evidence else "(no evidence)"
    combined = (predicted_reasoning + "\n" + predicted_answer).strip() if predicted_reasoning else predicted_answer
    msgs = [
        {"role": "system", "content": _CIT_JUDGE_SYSTEM_EN},
        {"role": "user", "content":
            f"Question: {question}\n"
            f"Key evidence IDs: {evidence_str}\n"
            f"Predicted answer: {combined}\n\nEvaluate:"},
    ]
    raw    = _call_llm(client, model, msgs, temperature=0.0)
    parsed = _parse_json(raw)
    score  = parsed.get("cit_score", "")
    reason = parsed.get("cit_reason", "")
    if score in ("HIGH", "MEDIUM", "LOW"):
        return score, reason
    return "LOW", "citation judge parse failed"


def judge_read(
    client: OpenAI, model: str,
    question: str, reasoning: str, answer: str,
    evidence_ids: List[str], lookup: Dict[str, str],
) -> Tuple[str, str]:
    if not evidence_ids or not lookup:
        return "LOW", "no evidence"
    ev_texts = []
    for eid in evidence_ids:
        text = lookup.get(eid, "")
        if text:
            ev_texts.append(f"[{eid}] {text}")
    if not ev_texts:
        return "LOW", "evidence text not found in lookup"
    evidence_block = "\n".join(ev_texts)
    combined = (reasoning + "\n" + answer).strip() if reasoning else answer
    msgs = [
        {"role": "system", "content": _READ_JUDGE_SYSTEM_EN},
        {"role": "user", "content":
            f"Question: {question}\n\n"
            f"Evidence passages:\n{evidence_block}\n\n"
            f"Model's reasoning and answer:\n{combined}\n\nEvaluate:"},
    ]
    raw    = _call_llm(client, model, msgs, temperature=0.0)
    parsed = _parse_json(raw)
    score  = parsed.get("read_score", "")
    reason = parsed.get("read_reason", "")
    if score in ("HIGH", "MEDIUM", "LOW"):
        return score, reason
    return "LOW", "read judge parse failed"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ── Config loading ──────────────────────────────────────────────────────────────
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def load_config(config_path: Path) -> Dict[str, Any]:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_client(cfg: Dict[str, Any]) -> Tuple[OpenAI, str, float]:
    """Return (client, model_name, temperature)"""
    client = OpenAI(
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url", "https://api.openai.com/v1"),
    )
    model  = cfg["model"]
    temp   = float(cfg.get("temperature", 0.2))
    return client, model, temp


def _init_mem_agent(mem_cfg: Dict[str, Any], game_type: str, model_name: str = "") -> Any:
    """Initialize memory agent based on mem_agent config"""
    mem_name = mem_cfg.get("mem_name", "mem_0")
    verbose  = mem_cfg.get("verbose", False)
    if mem_name == "mem_0":
        from galagent.memory.mem0 import Mem0Agent
        api_key = mem_cfg.get("mem0_api_key", "")
        if not api_key:
            raise ValueError("mem_agent.mem0_api_key is required when mem_name='mem_0'")
        return Mem0Agent(api_key=api_key, game_name=game_type, model_name=model_name, verbose=verbose)
    elif mem_name == "a_mem":
        from galagent.memory.amem import AMemAgent
        return AMemAgent(
            game_name=game_type,
            embedding_model=mem_cfg.get("amem_embedding_model", "all-MiniLM-L6-v2"),
            llm_backend=mem_cfg.get("amem_llm_backend", "openai"),
            llm_model=mem_cfg.get("amem_llm_model", "gpt-4o-mini"),
            verbose=verbose,
        )
    else:
        raise ValueError(f"Unknown mem_name: {mem_name!r}. Supported: 'mem_0', 'a_mem'")


def _mem_agent_to_messages(memories: List[Dict[str, Any]], max_memories: int = 100) -> List[Dict[str, str]]:
    """Convert mem_agent.get_all_memories() results to chat messages list (sorted by created_at ascending, latest max_memories items)"""
    sorted_mems = sorted(memories, key=lambda m: m.get("created_at", ""))
    sorted_mems = sorted_mems[-max_memories:]
    messages = []
    for mem in sorted_mems:
        role    = mem.get("metadata", {}).get("role", "user")
        content = mem.get("text", "")
        if content:
            messages.append({"role": role, "content": content})
    return messages


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ── Result summary ──────────────────────────────────────────────────────────────
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_WEIGHTS = {"Acc": 3.0, "Read.": 3.0, "Comp.": 1.5, "Depth": 1.5, "Inst.": 0.5, "Cit": 0.5}
_TOTAL_WEIGHT = sum(_WEIGHTS.values())


def _grade(pct: float) -> str:
    for threshold, grade in [(90, "A+"), (80, "A"), (70, "B+"), (60, "B"),
                              (50, "C+"), (40, "C"), (30, "D")]:
        if pct >= threshold:
            return grade
    return "F"


def compute_metrics(output: Dict[str, Any]) -> Dict[str, Any]:
    n      = output["accuracy"]["total"]
    acc    = output["accuracy"]["pct"]
    inst_r = output["inst_stats"]["total"]
    inst_p = output["inst_stats"]["pass_rate"]
    cit_r  = output["cit_score_distribution"]["total"]
    cit_h  = output["cit_score_distribution"]["high_rate"]
    cit_m  = output["cit_score_distribution"].get("MEDIUM", 0)
    cit_l  = output["cit_score_distribution"].get("LOW", 0)
    read_r = output["read_coverage"]["total"]
    read_h = output["read_coverage"].get("HIGH", 0)
    read_m = output["read_coverage"].get("MEDIUM", 0)
    read_avg = output["read_coverage"]["avg_score"] * 100

    comp = output["comp_accuracy"]["pct"]

    # depth breakdown — weighted average over D1/D2/D3
    by_depth = output.get("accuracy_by_depth", {})
    depth_vals: List[float] = []
    for key in sorted(by_depth.keys()):
        s = by_depth[key]
        depth_vals.append(s["pct"])
    depth_avg = sum(depth_vals) / len(depth_vals) if depth_vals else 0.0

    # depth sub-scores
    def _depth_pct(label_substr: str) -> float:
        for k, v in by_depth.items():
            if label_substr in str(k):
                return v.get("pct", 0.0)
        return 0.0

    d1 = _depth_pct("Surface")
    d2 = _depth_pct("Character")
    d3 = _depth_pct("Cross")

    dims = {
        "Acc":   acc,
        "Read.": read_avg,
        "Comp.": comp,
        "Depth": depth_avg,
        "Inst.": inst_p,
        "Cit":   cit_h,
    }
    overall = sum(_WEIGHTS[d] * (dims[d] / 100) for d in _WEIGHTS) / _TOTAL_WEIGHT * 100

    return {
        "model":   output.get("model", "?"),
        "story":   output.get("story", "?"),
        "Overall": round(overall, 2),
        "Acc":     round(acc, 2),
        "Comp.":   round(comp, 2),
        "Depth":   round(depth_avg, 2),
        "D1(S)":   round(d1, 2),
        "D2(C)":   round(d2, 2),
        "D3(X)":   round(d3, 2),
        "Inst.":   round(inst_p, 2),
        "Cit":     round(cit_h, 2),
        "Read.":   round(read_avg, 2),
        "n_qa":    n,
        "_dims":   dims,
    }


def print_summary(output: Dict[str, Any], log_dir: Optional[Path] = None) -> None:
    row  = compute_metrics(output)
    cols = ["model", "story", "Overall", "Acc", "Comp.", "Depth",
            "D1(S)", "D2(C)", "D3(X)", "Inst.", "Cit", "Read.", "n_qa"]
    widths = {c: max(len(c), len(str(row[c]))) for c in cols}

    print("\n=== Six-Dimensional Evaluation Summary ===\n")
    print("  ".join(c.ljust(widths[c]) for c in cols))
    print("  ".join("-" * widths[c] for c in cols))
    print("  ".join(str(row[c]).ljust(widths[c]) for c in cols))

    print("\n=== Overall Weighted Score Breakdown ===")
    print(f"  Weights: Acc×{_WEIGHTS['Acc']}  Read.×{_WEIGHTS['Read.']}  "
          f"Comp.×{_WEIGHTS['Comp.']}  Depth×{_WEIGHTS['Depth']}  "
          f"Inst.×{_WEIGHTS['Inst.']}  Cit×{_WEIGHTS['Cit']}  (total weight={_TOTAL_WEIGHT})")
    dims = row["_dims"]
    print(f"\n  {'─'*55}")
    print(f"  Score breakdown  model={row['model']}  story={row['story']}")
    print(f"  {'─'*55}")
    print(f"  {'Dimension':<10}  {'Wt':>4}  {'Raw (%)':>10}  {'Weighted':>8}  {'Grade':>5}")
    print(f"  {'-'*10}  {'-'*4}  {'-'*10}  {'-'*8}  {'-'*5}")
    total_w = 0.0
    for dim in list(_WEIGHTS.keys()):
        w = _WEIGHTS[dim]
        raw = dims.get(dim, 0.0) or 0.0
        ws = w * raw / 100
        total_w += ws
        print(f"  {dim:<10}  {w:>4.1f}  {raw:>10.2f}  {ws:>8.4f}  {_grade(raw):>5}")
    print(f"  {'─'*55}")
    pct_ov = total_w / _TOTAL_WEIGHT * 100
    print(f"  {'Overall':<10}  {_TOTAL_WEIGHT:>4.1f}  {'':>10}  {total_w:>8.4f}  "
          f"→ {pct_ov:.2f} / 100  {_grade(pct_ov)}")
    print("\nGrade: A+(≥90) A(≥80) B+(≥70) B(≥60) C+(≥50) C(≥40) D(≥30) F(<30)")

    if log_dir is not None:
        import io
        buf = io.StringIO()
        row2  = compute_metrics(output)
        dims2 = row2["_dims"]
        buf.write("  ".join(c.ljust(widths[c]) for c in cols) + "\n")
        buf.write("  ".join("-" * widths[c] for c in cols) + "\n")
        buf.write("  ".join(str(row2[c]).ljust(widths[c]) for c in cols) + "\n")
        buf.write(f"\n=== Overall Weighted Score Breakdown ===\n")
        buf.write(f"  Weights: Acc×{_WEIGHTS['Acc']}  Read.×{_WEIGHTS['Read.']}  "
                  f"Comp.×{_WEIGHTS['Comp.']}  Depth×{_WEIGHTS['Depth']}  "
                  f"Inst.×{_WEIGHTS['Inst.']}  Cit×{_WEIGHTS['Cit']}  (total weight={_TOTAL_WEIGHT})\n")
        buf.write(f"\n  {'─'*55}\n")
        buf.write(f"  Score breakdown  model={row2['model']}  story={row2['story']}\n")
        buf.write(f"  {'─'*55}\n")
        buf.write(f"  {'Dimension':<10}  {'Wt':>4}  {'Raw (%)':>10}  {'Weighted':>8}  {'Grade':>5}\n")
        buf.write(f"  {'-'*10}  {'-'*4}  {'-'*10}  {'-'*8}  {'-'*5}\n")
        total_w2 = 0.0
        for dim in list(_WEIGHTS.keys()):
            w = _WEIGHTS[dim]; raw = dims2.get(dim, 0.0) or 0.0
            ws = w * raw / 100; total_w2 += ws
            buf.write(f"  {dim:<10}  {w:>4.1f}  {raw:>10.2f}  {ws:>8.4f}  {_grade(raw):>5}\n")
        buf.write(f"  {'─'*55}\n")
        pct_ov2 = total_w2 / _TOTAL_WEIGHT * 100
        buf.write(f"  {'Overall':<10}  {_TOTAL_WEIGHT:>4.1f}  {'':>10}  {total_w2:>8.4f}  "
                  f"→ {pct_ov2:.2f} / 100  {_grade(pct_ov2)}\n")
        buf.write("\nGrade: A+(≥90) A(≥80) B+(≥70) B(≥60) C+(≥50) C(≥40) D(≥30) F(<30)\n")
        summary_path = log_dir / "summary.txt"
        summary_path.write_text(buf.getvalue(), encoding="utf-8")
        print(f"  [Summary] saved to: {summary_path}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ── Core QA evaluation loop ─────────────────────────────────────────────────────
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _run_qa_loop(
    game_type: str,
    qa_list: List[Dict[str, Any]],
    ev_lookup: Dict[str, str],
    story_name: str,
    language: str,
    story_messages: List[Dict[str, str]],
    client_llm: OpenAI,
    model_llm: str,
    temp_llm: float,
    client_judge: OpenAI,
    model_judge: str,
    log_dir: Path,
    session_id: str,
    checkpoint_ref: str = "inline",
    mem_agent: Any = None,
) -> None:
    """Core QA evaluation loop.

    Answers all QA items, judges them on 4 dimensions, saves results.json,
    and prints a six-dimensional summary.

    Args:
        mem_agent: Optional mem_agent for per-question memory retrieval (standalone mode).
                   Pass None (default) for inline mode.
        checkpoint_ref: Value stored in the output JSON's "checkpoint" field.
    """
    results: List[Dict[str, Any]] = []
    correct = 0
    undet   = 0
    n       = len(qa_list)

    print(f"\n{'='*60}")
    print(f"QA Evaluation  story={story_name}  n={n}")
    print(f"{'='*60}\n")

    qa_system_prompt = (_QA_SYSTEM_TYPE_HELP_EN if game_type == "type_help"
                        else _QA_SYSTEM_EN)

    for i, qa in enumerate(qa_list):
        question = qa.get("question", "")
        gold     = qa.get("answer", "")
        evidence = qa.get("evidence", [])

        # ── Answer ─────────────────────────────────────────────────────────────
        # Optional per-question memory retrieval (mem_agent mode only)
        retrieved_mem_context = ""
        if mem_agent is not None:
            retrieved = mem_agent.search_memories(question, top_k=5)
            retrieved = retrieved[:5]
            if retrieved:
                mem_lines = [f"{idx}. {m.get('text', '')}" for idx, m in enumerate(retrieved, 1)]
                retrieved_mem_context = "\n".join(mem_lines)

        qa_user_msg = qa_system_prompt
        if retrieved_mem_context:
            qa_user_msg += f"\n\n[Retrieved relevant memories]\n{retrieved_mem_context}"
        qa_user_msg += f"\n\nQuestion: {question}\n\nResponse:"

        messages   = story_messages + [{"role": "user", "content": qa_user_msg}]
        raw_output = _call_llm(client_llm, model_llm, messages, temperature=temp_llm).strip()
        reasoning, predicted = _parse_qa_output(raw_output)

        # ── Acc ───────────────────────────────────────────────────────────────
        judge_result, judge_reason = judge_answer(
            client_judge, model_judge, question, gold, predicted
        )
        if judge_result == "CONSISTENT":
            correct += 1
        elif judge_result == "UNDETERMINABLE":
            undet += 1

        # ── Inst ──────────────────────────────────────────────────────────────
        inst_info = judge_inst(client_judge, model_judge, reasoning, predicted)

        # ── Cit ───────────────────────────────────────────────────────────────
        cit_score, cit_reason = judge_citation(
            client_judge, model_judge,
            question, evidence, predicted, reasoning,
        )

        # ── Read ──────────────────────────────────────────────────────────────
        read_score, read_reason = judge_read(
            client_judge, model_judge,
            question, reasoning, predicted, evidence, ev_lookup,
        )

        mark = "✓" if judge_result == "CONSISTENT" else ("?" if judge_result == "UNDETERMINABLE" else "✗")
        print(f"  [{i+1:3d}/{n}] {mark} {judge_result:<16} "
              f"({correct}/{i+1})  cat={qa.get('category','?')}  depth={qa.get('depth','?')}  "
              f"cit={cit_score}  read={read_score}  "
              f"inst={'✓' if inst_info['_inst_pass'] else '✗'}  "
              f"pred={predicted[:40]!r}")

        results.append({
            **qa,
            "_predicted_reasoning": reasoning,
            "_predicted_answer":    predicted,
            "_judge_result":        judge_result,
            "_judge_reason":        judge_reason,
            "_cit_score":           cit_score,
            "_cit_reason":          cit_reason,
            "_read_score":          read_score,
            "_read_reason":         read_reason,
            **inst_info,
        })

    pct = correct / n * 100 if n else 0.0
    print(f"\n  QA complete: {n} questions  correct {correct}  accuracy {pct:.1f}%"
          + (f"  (incl. {undet} UNDETERMINABLE)" if undet else "") + "\n")

    # ── Aggregate statistics ──────────────────────────────────────────────────
    def _cat_stats() -> Dict[str, Any]:
        by_cat: Dict[Any, Dict[str, int]] = {}
        for r in results:
            cat = r.get("category", "?")
            s   = by_cat.setdefault(cat, {"total": 0, "correct": 0, "undeterminable": 0})
            s["total"] += 1
            if r.get("_judge_result") == "CONSISTENT":
                s["correct"] += 1
            elif r.get("_judge_result") == "UNDETERMINABLE":
                s["undeterminable"] += 1
        return {
            str(cat): {**s, "pct": round(s["correct"] / s["total"] * 100, 2) if s["total"] else 0.0}
            for cat, s in sorted(by_cat.items(), key=lambda x: str(x[0]))
        }

    _DEPTH_LABEL = {1: "Surface", 2: "Character", 3: "Cross-section"}

    def _depth_stats() -> Dict[str, Any]:
        by_depth: Dict[Any, Dict[str, int]] = {}
        for r in results:
            d = r.get("depth", "?")
            s = by_depth.setdefault(d, {"total": 0, "correct": 0})
            s["total"] += 1
            if r.get("_judge_result") == "CONSISTENT":
                s["correct"] += 1
        return {
            f"{d}_{_DEPTH_LABEL.get(d, str(d))}": {
                **s, "pct": round(s["correct"] / s["total"] * 100, 2) if s["total"] else 0.0
            }
            for d, s in sorted(by_depth.items(), key=lambda x: str(x[0]))
        }

    def _read_complexity() -> Dict[str, Any]:
        by_read: Dict[str, Dict[str, int]] = {}
        for r in results:
            ev_count = len(r.get("evidence", []))
            if ev_count <= 2:
                bucket = "low(≤2)"
            elif ev_count <= 4:
                bucket = "mid(3-4)"
            else:
                bucket = "high(≥5)"
            s = by_read.setdefault(bucket, {"total": 0, "correct": 0})
            s["total"] += 1
            if r.get("_judge_result") == "CONSISTENT":
                s["correct"] += 1
        return {
            bucket: {**s, "pct": round(s["correct"] / s["total"] * 100, 2) if s["total"] else 0.0}
            for bucket, s in by_read.items()
        }

    comp_r   = [r for r in results if r.get("category") in (2, 5)]
    comp_cor = sum(1 for r in comp_r if r.get("_judge_result") == "CONSISTENT")

    inst_r    = [r for r in results if "_inst_pass" in r]
    inst_pass = sum(1 for r in inst_r if r.get("_inst_pass"))

    cit_dist: Dict[str, int] = {}
    for r in results:
        s = r.get("_cit_score", "LOW")
        cit_dist[s] = cit_dist.get(s, 0) + 1

    read_dist: Dict[str, int] = {}
    for r in results:
        s = r.get("_read_score", "LOW")
        read_dist[s] = read_dist.get(s, 0) + 1
    if results:
        read_avg_score = (
            read_dist.get("HIGH", 0) * 1.0 + read_dist.get("MEDIUM", 0) * 0.5
        ) / len(results)
    else:
        read_avg_score = 0.0

    output: Dict[str, Any] = {
        "story":    story_name,
        "language": language,
        "model":    model_llm,
        "judge_model": model_judge,
        "session_id": session_id,
        "checkpoint": checkpoint_ref,
        "timestamp":  datetime.now().isoformat(),
        # ── Acc ────────────────────────────────────────────────────────────────
        "accuracy": {
            "total":          n,
            "correct":        correct,
            "incorrect":      n - correct - undet,
            "undeterminable": undet,
            "pct":            round(pct, 2),
        },
        "accuracy_by_category": _cat_stats(),
        "accuracy_by_depth":    _depth_stats(),
        "accuracy_by_read_complexity": _read_complexity(),
        # ── Comp ───────────────────────────────────────────────────────────────
        "comp_accuracy": {
            "total":   len(comp_r),
            "correct": comp_cor,
            "pct":     round(comp_cor / len(comp_r) * 100, 2) if comp_r else 0.0,
            "note":    "category 2 (contradiction) + category 5 (skill comparison)",
        },
        # ── Inst ───────────────────────────────────────────────────────────────
        "inst_stats": {
            "total":     len(inst_r),
            "pass":      inst_pass,
            "fail":      len(inst_r) - inst_pass,
            "pass_rate": round(inst_pass / len(inst_r) * 100, 2) if inst_r else 0.0,
        },
        # ── Cit ────────────────────────────────────────────────────────────────
        "cit_score_distribution": {
            **{k: cit_dist.get(k, 0) for k in ("HIGH", "MEDIUM", "LOW")},
            "total":     len(results),
            "high_rate": round(cit_dist.get("HIGH", 0) / len(results) * 100, 2) if results else 0.0,
        },
        # ── Read ───────────────────────────────────────────────────────────────
        "read_coverage": {
            **{k: read_dist.get(k, 0) for k in ("HIGH", "MEDIUM", "LOW")},
            "total":      len(results),
            "avg_score":  round(read_avg_score, 4),
            "pct":        round(read_avg_score * 100, 2),
            "high_rate":  round(read_dist.get("HIGH", 0) / len(results) * 100, 2) if results else 0.0,
        },
        "qa_results": results,
    }

    # ── Save results.json ──────────────────────────────────────────────────────
    out_file = log_dir / "results.json"
    out_file.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  Results saved to: {out_file}")

    # ── Print summary ──────────────────────────────────────────────────────────
    print_summary(output, log_dir=log_dir)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ── Mode 1: Standalone (loads from checkpoint file) ─────────────────────────────
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_qa(game_type: str) -> None:
    """Standalone QA evaluation: loads memory from a checkpoint file.

    Configure CHECKPOINT_PATH at the top of this file, then run:
        python scripts/run_qa.py --game_type dust
    """
    cfg_path = _ROOT / "config.yaml"
    cfg      = load_config(cfg_path)

    # ── LLM client ──────────────────────────────────────────────────────────
    llm_cfg   = cfg["llm"]
    judge_cfg = cfg.get("judge_llm", llm_cfg)

    client_llm,   model_llm,   temp_llm   = make_client(llm_cfg)
    client_judge, model_judge, _          = make_client(judge_cfg)
    print(f"[Config] llm_model={model_llm}  judge_model={model_judge}")

    # ── Game routing ─────────────────────────────────────────────────────────
    gcfg = GAME_CONFIG.get(game_type)
    if gcfg is None:
        raise ValueError(f"game_type={game_type!r} not supported yet. "
                         f"Supported: {[k for k,v in GAME_CONFIG.items() if v]}")

    qa_list, ev_lookup, story_name, language = _load_qa_data(game_type)

    # ── Load checkpoint, restore memory ──────────────────────────────────────
    ckpt_path = _ROOT / (CHECKPOINT_PATH if isinstance(CHECKPOINT_PATH, str)
                         else CHECKPOINT_PATH[game_type])
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt_data = json.loads(ckpt_path.read_text(encoding="utf-8"))

    session_id = ckpt_data.get("logger_session_id", datetime.now().strftime("%Y%m%d_%H%M%S"))
    ckpt_step  = ckpt_data.get("step", 0)

    # ── Decide whether to use mem_agent ──────────────────────────────────────
    mem_agent_cfg = cfg.get("mem_agent", {})
    use_mem_agent = bool(mem_agent_cfg.get("use_mem", False))
    mem_agent_obj = None

    if use_mem_agent:
        mem_agent_obj = _init_mem_agent(mem_agent_cfg, game_type, model_name=model_llm)
        mem_agent_state = ckpt_data["memory_state"].get("mem_agent_state", {})
        if mem_agent_state:
            is_amem = "memories" in mem_agent_state
            mem_agent_obj.restore_state(mem_agent_state)
            if is_amem:
                n_restored = mem_agent_state.get("total_memories", len(mem_agent_state.get("memories", [])))
                print(f"[Checkpoint] step={ckpt_step}  amem state restored, {n_restored} memories  "
                      f"session_id={session_id}")
            else:
                print(f"[Checkpoint] step={ckpt_step}  mem0 config restored (memories in cloud)  "
                      f"session_id={session_id}")
        else:
            print(f"[Checkpoint] step={ckpt_step}  no mem_agent_state in checkpoint  "
                  f"session_id={session_id}")
            if mem_agent_cfg.get("mem_name") == "a_mem":
                print("[Checkpoint] warning: amem mode but mem_agent_state is empty, memories not restored!")
        all_memories = mem_agent_obj.get_all_memories()
        print(f"[MemAgent] mem_name={mem_agent_cfg['mem_name']}  "
              f"total_memories={len(all_memories)}")
        if all_memories:
            first = all_memories[0]
            print(f"[MemAgent] first memory preview: role={first.get('metadata', {}).get('role', 'N/A')}  "
                  f"created_at={first.get('created_at', 'N/A')}  "
                  f"text={str(first.get('text', ''))[:80]!r}")
        else:
            print("[MemAgent] warning: no memories retrieved, check user_id / checkpoint")
        mem_messages = _mem_agent_to_messages(all_memories)
        print(f"[MemAgent] story_messages count={len(mem_messages)}"
              f"  (latest 100 items)")
        story_messages = (
            [{"role": "system", "content":
              "[Memory Agent State] The following memories were retrieved from the memory agent:"}]
            + mem_messages
        )
    else:
        store = MemoryStore(verbose=False)
        store.restore_state(ckpt_data["memory_state"])
        print(f"[Checkpoint] step={ckpt_step}  memory_items={len(store.items)}  "
              f"session_id={session_id}")

        jsonl_path = _find_memory_jsonl(ckpt_path)
        story_messages = _build_story_messages(store, str(jsonl_path) if jsonl_path else None)

    # ── Log directory ─────────────────────────────────────────────────────────
    mem_tag = mem_agent_cfg.get("mem_name", "nomem") if use_mem_agent else "nomem"
    log_dir = _ROOT / "qa_result" / game_type / model_llm / mem_tag
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Log] output directory: {log_dir}")

    _run_qa_loop(
        game_type, qa_list, ev_lookup, story_name, language,
        story_messages, client_llm, model_llm, temp_llm,
        client_judge, model_judge, log_dir, session_id,
        checkpoint_ref=str(CHECKPOINT_PATH),
        mem_agent=mem_agent_obj,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ── Mode 2: Inline (called from galAgent.py after game ends) ────────────────────
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_qa_from_store(
    game_type: str,
    store: MemoryStore,
    memory_log_file: str,
    log_dir: Path,
    llm_config: Any,
    judge_llm_config: Any,
    session_id: str,
) -> None:
    """Inline QA evaluation: uses live MemoryStore, no checkpoint file needed.

    Called automatically from galAgent.py after type_help / dust game ends.

    Args:
        game_type:        "type_help" or "dust"
        store:            The live MemoryStore from galAgent.py
        memory_log_file:  Path to logs/memory_{session_id}.jsonl
        log_dir:          Output directory, e.g. logs/{game_type}/{session_id}/
        llm_config:       LLMConfig dataclass (api_key, base_url, model, temperature)
        judge_llm_config: Optional LLMConfig for judge; falls back to llm_config
        session_id:       Session ID string for the output JSON
    """
    gcfg = GAME_CONFIG.get(game_type)
    if gcfg is None:
        print(f"[QA] game_type={game_type!r} not supported for QA, skipping")
        return

    qa_list, ev_lookup, story_name, language = _load_qa_data(game_type)

    judge_cfg    = judge_llm_config or llm_config
    client_llm   = OpenAI(api_key=llm_config.api_key,   base_url=llm_config.base_url)
    client_judge = OpenAI(api_key=judge_cfg.api_key,     base_url=judge_cfg.base_url)
    print(f"[QA] llm_model={llm_config.model}  judge_model={judge_cfg.model}")

    story_messages = _build_story_messages(store, memory_log_file)

    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Log] output directory: {log_dir}")

    _run_qa_loop(
        game_type, qa_list, ev_lookup, story_name, language,
        story_messages, client_llm, llm_config.model, llm_config.temperature,
        client_judge, judge_cfg.model, log_dir, session_id,
        checkpoint_ref="inline",
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ── CLI ──────────────────────────────────────────────────────────────────────
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GalgameBench QA Evaluation")
    parser.add_argument(
        "--game_type", default=GAME_TYPE,
        choices=list(GAME_CONFIG.keys()),
        help=f"game type (default: {GAME_TYPE}; can also set GAME_TYPE at top of file)",
    )
    args = parser.parse_args()
    run_qa(args.game_type)
