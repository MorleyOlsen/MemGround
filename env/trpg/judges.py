# env/trpg/judges.py
"""Judge functions and prompt templates for TRPG game evaluation"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple


# ── Judge prompt templates ────────────────────────────────────────────────────

JUDGE_SYSTEM_CH = """\
You are an objective and strict answer consistency evaluator.
Determine whether the "predicted answer" and the "gold answer" are semantically equivalent. Rules:
- CONSISTENT      : Both convey the same core meaning (different wording but referring to the same entity/relation/fact)
- INCONSISTENT    : They point to different entities or contain substantive contradictions
- UNDETERMINABLE  : The gold answer itself is ambiguous, or consistency cannot be determined

Note TRPG data specifics: player labels (playerX) are interchangeable with in-game character names (e.g. player2 = a character name); if the core fact matches, it is CONSISTENT.

Output ONLY JSON, nothing else:
{"result": "CONSISTENT"|"INCONSISTENT"|"UNDETERMINABLE", "reason": "brief reason (≤30 words)"}\
"""

JUDGE_SYSTEM_EN = """\
You are an objective and strict answer consistency evaluator.
Determine whether the "predicted answer" and the "gold answer" are semantically equivalent. Rules:
- CONSISTENT      : Both convey the same core meaning (different wording but referring to the same entity/relation/fact)
- INCONSISTENT    : They point to different entities or contain substantive contradictions
- UNDETERMINABLE  : The gold answer itself is ambiguous, or consistency cannot be determined

Note TRPG data specifics: player labels (playerX) are interchangeable with in-game character names; if the core fact matches, it is CONSISTENT.

Output ONLY JSON, nothing else:
{"result": "CONSISTENT"|"INCONSISTENT"|"UNDETERMINABLE", "reason": "brief reason (≤30 words)"}\
"""

CIT_JUDGE_SYSTEM_CH = """\
You are an objective evidence-grounding evaluator.
Given the key evidence IDs for a question and the model's predicted answer, assess whether the answer correctly utilizes the information from those key story moments.

Rules:
- HIGH   : The predicted answer clearly reflects the information from the key events/dialogues indicated by the evidence (explicit ID citation is NOT required)
- MEDIUM : Partially reflects the evidence — uses some relevant information but misses major evidence points
- LOW    : Does not reflect the evidence, is based on incorrect information, or is unrelated to the evidenced events

Note: Evaluate content grounding, not explicit citation of IDs.

Output ONLY JSON, nothing else:
{"cit_score": "HIGH"|"MEDIUM"|"LOW", "cit_reason": "brief reason (≤30 words)"}\
"""

CIT_JUDGE_SYSTEM_EN = """\
You are an objective evidence-grounding evaluator.
Given the key evidence IDs for a question and the model's predicted answer, assess whether the answer correctly utilizes the information from those key story moments.

Rules:
- HIGH   : The predicted answer clearly reflects the information from the key events/dialogues indicated by the evidence (explicit ID citation is NOT required)
- MEDIUM : Partially reflects the evidence — uses some relevant information but misses major evidence points
- LOW    : Does not reflect the evidence, is based on incorrect information, or is unrelated to the evidenced events

Note: Evaluate content grounding, not explicit citation of IDs.

Output ONLY JSON, nothing else:
{"cit_score": "HIGH"|"MEDIUM"|"LOW", "cit_reason": "brief reason (≤30 words)"}\
"""

INST_JUDGE_SYSTEM_CH = """\
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

INST_JUDGE_SYSTEM_EN = """\
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

READ_JUDGE_SYSTEM_CH = """\
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

READ_JUDGE_SYSTEM_EN = """\
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


# ── Judge functions ─────────────────────────────────────────────────────────


def judge_answer(
    question: str,
    gold: str,
    predicted: str,
    call_llm: Callable,
    env_config: Any,
) -> Tuple[str, str]:
    """Determine whether the predicted answer is consistent with the gold answer

    Args:
        question: Question text
        gold: Gold answer
        predicted: Model-predicted answer
        call_llm: LLM call function with signature: call_llm(messages, temperature, use_judge_client) -> str
        env_config: Environment configuration containing the test_language field

    Returns:
        (result, reason) tuple
        - result: "CONSISTENT" | "INCONSISTENT" | "UNDETERMINABLE"
        - reason: Reason for the judgement
    """
    if not predicted.strip():
        return "INCONSISTENT", "predicted answer is empty"

    if _normalize(gold) == _normalize(predicted):
        return "CONSISTENT", "exact match after normalization"

    is_en = getattr(env_config, "test_language", "en") == "en"

    if is_en:
        judge_sys = JUDGE_SYSTEM_EN
        user_content = (
            f"Question: {question}\nGold answer: {gold}\nPredicted answer: {predicted}\n\nJudge:"
        )
    else:
        judge_sys = JUDGE_SYSTEM_CH
        user_content = (
            f"Question: {question}\nGold answer: {gold}\nPredicted answer: {predicted}\n\nJudge:"
        )
    msgs = [
        {"role": "system", "content": judge_sys},
        {"role": "user", "content": user_content},
    ]
    raw = call_llm(msgs, temperature=0.0, use_judge_client=True)
    parsed = _parse_json(raw)
    result = parsed.get("result", "")
    reason = parsed.get("reason", "")

    if result in ("CONSISTENT", "INCONSISTENT", "UNDETERMINABLE"):
        return result, reason
    return "UNDETERMINABLE", "judge parse failed"


def judge_citation(
    question: str,
    evidence: List[str],
    predicted_answer: str,
    predicted_reasoning: str,
    is_en: bool,
    call_llm: Callable,
) -> Tuple[str, str]:
    """Cit metric: assess whether the predicted answer is grounded in the correct evidence events

    Args:
        question: Question text
        evidence: List of evidence IDs
        predicted_answer: The predicted answer portion
        predicted_reasoning: The predicted reasoning portion
        is_en: Whether to use English prompts
        call_llm: LLM call function

    Returns:
        (cit_score, cit_reason) tuple
        - cit_score: "HIGH" | "MEDIUM" | "LOW"
        - cit_reason: Reason for the score
    """
    if not predicted_answer.strip():
        return "LOW", "predicted answer is empty"

    evidence_str = ", ".join(evidence) if evidence else "(no evidence)"
    combined = (predicted_reasoning + "\n" + predicted_answer).strip() if predicted_reasoning else predicted_answer

    if is_en:
        cit_sys = CIT_JUDGE_SYSTEM_EN
        user_msg = (
            f"Question: {question}\n"
            f"Key evidence IDs: {evidence_str}\n"
            f"Predicted answer: {combined}\n\nEvaluate:"
        )
    else:
        cit_sys = CIT_JUDGE_SYSTEM_CH
        user_msg = (
            f"Question: {question}\n"
            f"Key evidence IDs: {evidence_str}\n"
            f"Predicted answer: {combined}\n\nEvaluate:"
        )

    raw = call_llm(
        [{"role": "system", "content": cit_sys}, {"role": "user", "content": user_msg}],
        temperature=0.0,
        use_judge_client=True,
    )
    parsed = _parse_json(raw)
    score = parsed.get("cit_score", "")
    reason = parsed.get("cit_reason", "")

    if score in ("HIGH", "MEDIUM", "LOW"):
        return score, reason
    return "LOW", "citation judge parse failed"


def judge_inst(
    reasoning: str,
    answer: str,
    is_en: bool,
    call_llm: Callable,
) -> Dict[str, Any]:
    """Inst metric: use LLM judge to evaluate instruction following (reasoning structure + length)

    Args:
        reasoning: Reasoning process text
        answer: Answer text
        is_en: Whether to use English prompts
        call_llm: LLM call function

    Returns:
        Dictionary containing _inst_pass and _inst_reason
    """
    if not (reasoning or answer).strip():
        return {"_inst_pass": False, "_inst_reason": "empty output"}

    combined = (reasoning + "\n" + answer).strip()
    if is_en:
        inst_sys = INST_JUDGE_SYSTEM_EN
        user_msg = f"Model answer:\n{combined}\n\nEvaluate:"
    else:
        inst_sys = INST_JUDGE_SYSTEM_CH
        user_msg = f"Model answer:\n{combined}\n\nEvaluate:"

    raw = call_llm(
        [{"role": "system", "content": inst_sys}, {"role": "user", "content": user_msg}],
        temperature=0.0,
        use_judge_client=True,
    )
    parsed = _parse_json(raw)
    score = parsed.get("inst_score", "")
    reason = parsed.get("inst_reason", "")

    return {
        "_inst_pass": score == "PASS",
        "_inst_reason": reason or "inst judge parse failed",
    }


def judge_read(
    question: str,
    reasoning: str,
    answer: str,
    evidence_ids: List[str],
    lookup: Dict[str, str],
    is_en: bool,
    call_llm: Callable,
) -> Tuple[str, str]:
    """Read metric: use LLM judge to evaluate whether the answer genuinely references the key content of gold evidence

    Args:
        question: Question text
        reasoning: Reasoning process text
        answer: Answer text
        evidence_ids: List of evidence IDs
        lookup: Mapping from evidence ID to original text
        is_en: Whether to use English prompts
        call_llm: LLM call function

    Returns:
        (read_score, read_reason) tuple
        - read_score: "HIGH" | "MEDIUM" | "LOW"
        - read_reason: Reason for the score
    """
    if not evidence_ids or not lookup:
        return "LOW", "no evidence"

    ev_texts = []
    for eid in evidence_ids:
        text = lookup.get(eid, "")
        if text:
            ev_texts.append(f"[{eid}] {text}")
    if not ev_texts:
        return "LOW", "evidence text not found"

    evidence_block = "\n".join(ev_texts)
    combined = (reasoning + "\n" + answer).strip() if reasoning else answer

    if is_en:
        read_sys = READ_JUDGE_SYSTEM_EN
        user_msg = (
            f"Question: {question}\n\n"
            f"Evidence passages:\n{evidence_block}\n\n"
            f"Model's reasoning and answer:\n{combined}\n\nEvaluate:"
        )
    else:
        read_sys = READ_JUDGE_SYSTEM_CH
        user_msg = (
            f"Question: {question}\n\n"
            f"Evidence passages:\n{evidence_block}\n\n"
            f"Model's reasoning and answer:\n{combined}\n\nEvaluate:"
        )

    raw = call_llm(
        [{"role": "system", "content": read_sys}, {"role": "user", "content": user_msg}],
        temperature=0.0,
        use_judge_client=True,
    )
    parsed = _parse_json(raw)
    score = parsed.get("read_score", "")
    reason = parsed.get("read_reason", "")

    if score in ("HIGH", "MEDIUM", "LOW"):
        return score, reason
    return "LOW", "read judge parse failed"


# ── Helper functions ──────────────────────────────────────────────────────────


def parse_qa_output(raw: str, is_en: bool) -> Tuple[str, str]:
    """Parse reasoning and answer from structured output

    English format: Reasoning: ... Answer: ...
    If parsing fails, reasoning returns empty string and answer falls back to the full text.

    Args:
        raw: Raw text output from the model
        is_en: Whether the text is in English

    Returns:
        (reasoning, answer) tuple
    """
    if is_en:
        r_marker, a_marker = "Reasoning:", "Answer:"
    else:
        r_marker, a_marker = "Reasoning:", "Answer:"

    raw = (raw or "").strip()
    r_idx = raw.find(r_marker)
    a_idx = raw.find(a_marker)

    if r_idx != -1 and a_idx != -1 and a_idx > r_idx:
        reasoning = raw[r_idx + len(r_marker):a_idx].strip()
        answer = raw[a_idx + len(a_marker):].strip()
        return reasoning, answer

    # Fallback: single-label, only look for Answer
    if a_idx != -1:
        return "", raw[a_idx + len(a_marker):].strip()

    # Complete failsafe: return full text as answer
    return "", raw


def build_evidence_lookup(data_path: Path, story_name: str) -> Dict[str, str]:
    """Build evidence ID -> original text mapping

    Evidence ID format: D{sec:02d}:{idx}, e.g. D01:20.
    Scans all NN_*.json files under data_path/story_name/.

    Args:
        data_path: Dataset path
        story_name: Story name

    Returns:
        Dictionary mapping evidence ID to original text
    """
    story_path = Path(data_path) / story_name
    lookup: Dict[str, str] = {}
    if not story_path.exists():
        return lookup

    for fn in sorted(story_path.glob("*.json")):
        m = re.match(r'^(\d+)_', fn.name)
        if not m:
            continue
        sec_num = int(m.group(1))
        try:
            data = json.loads(fn.read_text(encoding="utf-8"))
        except Exception:
            continue
        for idx, conv in enumerate(data.get("conversation", [])):
            key = f"D{sec_num:02d}:{idx}"
            lookup[key] = conv.get("text", "")
    return lookup


def _normalize(text: str) -> str:
    """Normalize text for comparison (strip whitespace, punctuation, and case)"""
    text = text.strip().lower()
    text = re.sub(r'[\s\u3000，。！？、；：""''（）【】《》,.!?;:\'"()\[\]{}\-—]+', "", text)
    return text


def _parse_json(s: str) -> Dict[str, str]:
    """Parse JSON from LLM output

    Handles possible markdown code block wrapping and other text noise
    """
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
