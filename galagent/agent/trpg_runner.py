# galagent/agent/trpg_runner.py
"""TRPG 跑团评测专用运行器"""
from __future__ import annotations

import io
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

try:
    from summarize import compute_metrics, _WEIGHTS, _TOTAL_WEIGHT, _grade
    _SUMMARIZE_AVAILABLE = True
except ImportError:
    _SUMMARIZE_AVAILABLE = False
from galagent.common.config import AgentConfig, LLMConfig, EnvConfig
from galagent.memory.store import MemoryStore
from env.trpg.prompt_builder import (
    QA_SYSTEM_EN,
    SUMMARY_SYSTEM_EN,
    RETRY_HIGH_EN,
    RETRY_LOW_EN,
)
from env.trpg.judges import (
    judge_answer, judge_citation, judge_inst, judge_read,
    parse_qa_output, build_evidence_lookup,
)


class TRPGRunner:
    """TRPG 评测运行器：负责阅读阶段和QA阶段的完整流程"""

    def __init__(
        self,
        env: Any,  # TRPGEnv
        store: MemoryStore,
        config: AgentConfig,
        llm_config: LLMConfig,
        judge_llm_config: Optional[LLMConfig] = None,
        log_dir: Optional[Path] = None,
        session_id: str = "",
        ckpt_resume_from: Optional[str] = None,
        mem_agent: Optional[Any] = None,
    ):
        """初始化 TRPG Runner

        Args:
            env: TRPG 游戏环境
            store: 记忆存储
            config: Agent 配置
            llm_config: 答题模型配置
            judge_llm_config: 判题模型配置（可选，未配置时回退到答题模型）
            log_dir: 日志目录
            session_id: 会话ID
            ckpt_resume_from: checkpoint 恢复路径
            mem_agent: A-mem 记忆代理（可选，配置后 checkpoint 会包含其状态）
        """
        self.env = env
        self.store = store
        self.config = config
        self.session_id = session_id
        self.log_dir = log_dir
        self.ckpt_resume_from = ckpt_resume_from
        self.mem_agent = mem_agent

        # checkpoint 目录
        self.ckpt_dir = log_dir / "checkpoints" if log_dir else None
        if self.ckpt_dir:
            self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        # 答题模型配置
        self._trpg_client = OpenAI(
            api_key=llm_config.api_key,
            base_url=llm_config.base_url,
            timeout=180.0
        )
        self._trpg_model = llm_config.model
        self._trpg_temperature = llm_config.temperature

        # 判题模型配置（未配置时回退到答题模型）
        if judge_llm_config:
            self._judge_client = OpenAI(
                api_key=judge_llm_config.api_key,
                base_url=judge_llm_config.base_url,
                timeout=180.0
            )
            self._judge_model = judge_llm_config.model
        else:
            self._judge_client = self._trpg_client
            self._judge_model = self._trpg_model

        # 阅读阶段状态
        self._trpg_total_turns = 0
        self._trpg_compressions = 0
        self._trpg_sec_idx = -1
        self._trpg_turn_idx = -2

        # 证据查找表（用于 Read 指标）
        self._evidence_lookup: Dict[str, str] = {}

    def run(self) -> None:
        """执行 TRPG 两阶段评测"""
        verbose = getattr(self.config, 'verbose', True)
        print(f"\n{'='*60}")
        print(f"TRPG Evaluation  Story: {self.env.config.story_name}")
        print(f"  Sections: {len(self.env.sections)}  "
              f"Total turns: {self.env.total_turns()}  "
              f"QA: {len(self.env.qa_list)} questions")
        print(f"  Compression threshold: {self.config.compression_threshold}  "
              f"Per compression: {self.config.compression_count}")
        print(f"{'='*60}\n")

        # ── 尝试从 checkpoint 恢复 ──────────────────────────────────────────
        ckpt = None
        if self.ckpt_resume_from:
            # 直接使用指定的文件路径
            ckpt_path = Path(self.ckpt_resume_from)
            if ckpt_path.exists():
                with open(ckpt_path, encoding="utf-8") as _f:
                    ckpt = json.load(_f)
                print(f"[Checkpoint] Resuming from file: {ckpt_path.name}, "
                      f"phase={ckpt['phase']}, compressions={ckpt.get('compressions', 0)}")
            else:
                print(f"[Checkpoint] Warning: file not found: {ckpt_path}")
        elif self.ckpt_dir and self.ckpt_dir.exists():
            ckpt_files = sorted(
                self.ckpt_dir.glob("trpg_ckpt_*.json"),
                key=lambda p: p.stat().st_mtime,
            )
            if ckpt_files:
                ckpt_path = ckpt_files[-1]  # 最新的 checkpoint
                with open(ckpt_path, encoding="utf-8") as _f:
                    ckpt = json.load(_f)
                print(f"[Checkpoint] Found checkpoint: {ckpt_path.name}, "
                      f"phase={ckpt['phase']}, compressions={ckpt.get('compressions', 0)}")

        if ckpt:
            self.store.reset()
            for _item in ckpt.get("store_items", []):
                self.store.add_message(
                    _item["text"],
                    role=_item.get("role", "user"),
                    step=_item.get("step", ""),
                )
            # 注：amem 模式下 store_items 为空（reading 阶段消息直接写入 mem_agent），
            # 实际记忆在 amem_state.memories 中，通过下方 restore_state 恢复；
            # 非 amem 模式下 store_items 保存压缩后的滑动窗口，是有效数据。
            self._trpg_total_turns = ckpt.get("total_turns", 0)
            self._trpg_compressions = ckpt.get("compressions", 0)
            self._trpg_sec_idx = ckpt.get("sec_idx", 0)
            self._trpg_turn_idx = ckpt.get("turn_idx", -2)

            # 恢复记忆代理状态：
            # - amem_state 含 "memories" 字段 → amem（本地存储）：restore_state 重建全部 MemoryNote 和
            #   ChromaDB 索引；恢复后 store.to_chat_messages() 将通过 store.mem_agent.get_all_memories()
            #   返回这些记忆（store.use_mem=True 分支）
            # - amem_state 不含 "memories" 字段 → mem0（云端存储）：restore_state 只重建 user_id，
            #   实际记忆无需恢复
            amem_state = ckpt.get("amem_state")
            if amem_state and self.mem_agent is not None:
                if "memories" in amem_state:
                    # amem：从 checkpoint 恢复完整本地记忆
                    self.mem_agent.restore_state(amem_state)
                    print(f"[Checkpoint] amem state restored, {amem_state.get('total_memories', 0)} memories")
                else:
                    # mem0: only restore game_name/model_name to rebuild user_id; actual memories are in the cloud
                    self.mem_agent.restore_state(amem_state)
                    print(f"[Checkpoint] mem0 config restored (memories stored in cloud, no local restore needed)")

        # ── Phase 1: 阅读 ─────────────────────────────────────────────────────────
        if not ckpt or ckpt["phase"] == "reading":
            self._reading_phase(resume_ckpt=ckpt)
        else:
            print("─── Phase 1: skipped via checkpoint ───\n")

        # ── Phase 2: QA ───────────────────────────────────────────────────────────
        qa_start = 0
        if ckpt and ckpt["phase"] == "qa":
            self.env.results = ckpt.get("results", [])
            qa_start = ckpt.get("qa_idx", -1) + 1
            print(f"[Checkpoint] QA resuming from question {qa_start + 1}\n")

        # 构建证据文本查找表（用于 Read 指标）
        self._evidence_lookup = build_evidence_lookup(
            self.env.config.data_path, self.env.config.story_name
        )
        self._qa_phase(start_idx=qa_start)
        self._save_trpg_results()

    # ── Phase 1: 阅读 ─────────────────────────────────────────────────────────

    def _reading_phase(self, resume_ckpt=None) -> None:
        """阅读阶段：逐步读取故事并进行记忆压缩"""
        verbose = getattr(self.config, 'verbose', True)
        print("─── Phase 1: Reading story ───")
        threshold = self.config.compression_threshold
        compress_n = self.config.compression_count
        story_name = self.env.config.story_name

        # checkpoint 恢复位置：skip_sec/skip_turn 之前（含）的内容已处理
        skip_sec = resume_ckpt.get("sec_idx", -1) if resume_ckpt else -1
        skip_turn = resume_ckpt.get("turn_idx", -2) if resume_ckpt else -2

        for sec_idx, section in enumerate(self.env.sections):
            if sec_idx < skip_sec:
                continue  # 整个 section 已处理，跳过

            section_title = section.get("section", f"Section {sec_idx+1}")
            description = section.get("description", "")
            conversation = section.get("conversation", [])
            tag = f"{story_name} | {section_title}"

            header = f"=== {section_title} ==="
            if description:
                header += f"\nDescription: {description}"

            # header 已添加的条件：sec_idx == skip_sec 且 skip_turn >= -1
            if not (sec_idx == skip_sec and skip_turn >= -1):
                self.store.add_message(header, role="user", step=tag)
                self._trpg_total_turns += 1
                self._trpg_sec_idx = sec_idx
                self._trpg_turn_idx = -1
                self._maybe_compress(threshold, compress_n)

            for turn_idx, turn in enumerate(conversation):
                if sec_idx == skip_sec and turn_idx <= skip_turn:
                    continue  # 该 turn 已处理，跳过
                text = (turn.get("text") or "").replace("\n", " ").strip()
                if not text:
                    continue
                speaker = turn.get("speaker", "")
                self.store.add_message(f"[{speaker}]: {text}", role="user", step=tag)
                self._trpg_total_turns += 1
                self._trpg_sec_idx = sec_idx
                self._trpg_turn_idx = turn_idx
                self._maybe_compress(threshold, compress_n)

            if verbose:
                print(f"  [{sec_idx+1}/{len(self.env.sections)}] {section_title} "
                      f"({len(conversation)} turns)  "
                      f"window={len(self.store.items)}  "
                      f"compressions={self._trpg_compressions}")

        print(f"\n  Reading complete: {self._trpg_total_turns} turns total, "
              f"{self._trpg_compressions} compressions, "
              f"current window={len(self.store.items)} items\n")
        self._save_trpg_checkpoint("reading_complete", name="reading_complete")

    def _maybe_compress(self, threshold: int, compress_n: int) -> None:
        """根据阈值判断是否需要压缩"""
        # use_mem=True 时 mem_agent 自行管理记忆，无需手动压缩
        if hasattr(self.store, 'use_mem') and self.store.use_mem:
            return
        if len(self.store.items) >= threshold:
            self._compress_with_summary(compress_n)

    def _compress_with_summary(self, compress_n: int) -> None:
        """使用 LLM 生成摘要压缩最早的 N 条记忆"""
        verbose = getattr(self.config, 'verbose', True)
        items = self.store.items
        if len(items) < compress_n:
            compress_n = len(items)

        oldest = items[:compress_n]
        remaining = [(item.text, item.meta.get("role", "user")) for item in items[compress_n:]]

        text_block = "\n".join(item.text for item in oldest)
        source_chars = len(text_block)

        # 分别统计已有摘要和原始对话的字数，设定差异化压缩目标
        summary_prefix = "[Story Summary]"
        prev_summary_chars = sum(
            len(item.text) for item in oldest
            if item.text.startswith(("[故事摘要]", "[Story Summary]"))
        )
        raw_chars = source_chars - prev_summary_chars
        # 摘要保留 65%，原始对话保留 50%，综合得出目标区间
        overall_target = int(prev_summary_chars * 0.65 + raw_chars * 0.50)
        min_chars = max(200, int(overall_target * 0.85))
        max_chars = max(min_chars + 50, int(overall_target * 1.15))

        summary_system = SUMMARY_SYSTEM_EN.format(
            source_chars=source_chars,
            prev_summary_chars=prev_summary_chars,
            raw_chars=raw_chars,
            min_chars=min_chars,
            max_chars=max_chars,
        )
        summary = self._call_llm_trpg([
            {"role": "system", "content": summary_system},
            {"role": "user", "content": text_block},
        ])

        # ── 超出范围时重试一次 ──────────────────────────────────────────
        actual = len(summary) if summary else 0
        if summary and not (min_chars <= actual <= max_chars):
            if actual > max_chars:
                retry_tmpl = RETRY_HIGH_EN
            else:
                retry_tmpl = RETRY_LOW_EN
            retry_prompt = retry_tmpl.format(
                actual_chars=actual, min_chars=min_chars, max_chars=max_chars
            )
            retry_system = (
                f"HARD CHARACTER LIMIT — Your output MUST be between {min_chars} and {max_chars} characters. "
                f"Do NOT output fewer than {min_chars} characters or more than {max_chars} characters."
            )
            if actual > max_chars:
                # HIGH：基于摘要继续精简，不需要原文
                retry_messages = [
                    {"role": "system", "content": retry_system},
                    {"role": "user", "content": retry_prompt + "\n\n" + summary},
                ]
            else:
                # LOW：附上原文，让模型有材料可补充
                retry_messages = [
                    {"role": "system", "content": retry_system},
                    {"role": "user", "content": (
                        retry_prompt
                        + "\n\n[Original Text]\n" + text_block
                        + "\n\n[Over-compressed summary (please expand on this)]\n" + summary
                    )},
                ]
            retry_summary = self._call_llm_trpg(retry_messages)
            if retry_summary:
                summary = retry_summary
                actual = len(summary)

        # HIGH 仍超出 → 在 max_chars 前最近的句子边界截断
        if actual > max_chars:
            cut = summary[:max_chars]
            # 找最后一个句末标点
            for punct in ("。", "！", "？", ".", "!", "?"):
                idx = cut.rfind(punct)
                if idx > min_chars:
                    cut = cut[:idx + 1]
                    break
            summary = cut
        # LOW 仍不足 → 接受（信息完整优先，不强制截断）

        self.store.reset()
        if summary:
            summary_text = f"{summary_prefix}\n{summary}"
        else:
            summary_text = f"[Story Summary (generation failed — {compress_n} turns omitted)]"
        self.store.add_message(summary_text, role="system")
        for text, role in remaining:
            self.store.add_message(text, role=role)

        self._trpg_compressions += 1

        # 写入压缩日志
        if self.log_dir:
            log_entry = {
                "compression_index": self._trpg_compressions,
                "timestamp": datetime.now().isoformat(),
                "compressed_turns": compress_n,
                "source_chars": source_chars,
                "prev_summary_chars": prev_summary_chars,
                "raw_chars": raw_chars,
                "min_chars": min_chars,
                "max_chars": max_chars,
                "actual_chars": len(summary) if summary else 0,
                "in_range": min_chars <= (len(summary) if summary else 0) <= max_chars,
                "summary": summary_text,
            }
            with open(self.log_dir / "compressions.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

        # 每次压缩后保存 checkpoint
        self._save_trpg_checkpoint("reading", name=f"reading_c{self._trpg_compressions:03d}")
        if verbose:
            print(f"    [Summary] Compressed {compress_n} items → 1 summary, current window={len(self.store.items)}")

    # ── Phase 2: QA ───────────────────────────────────────────────────────────

    def _qa_phase(self, start_idx: int = 0) -> None:
        """QA 阶段：逐题回答并评分"""
        verbose = getattr(self.config, 'verbose', True)
        if not self.env.qa_list:
            print("─── Phase 2: No QA data, skipping ───")
            return

        print("─── Phase 2: QA Evaluation ───")
        n = len(self.env.qa_list)
        # 从已有结果中恢复计数（checkpoint 恢复场景）
        correct = sum(1 for r in self.env.results if r.get("_judge_result") == "CONSISTENT")
        undet = sum(1 for r in self.env.results if r.get("_judge_result") == "UNDETERMINABLE")

        is_en = getattr(self.env.config, "test_language", "en") == "en"

        # QA 阶段记忆不再增加，提前获取一次并缓存
        _all_store = self.store.to_chat_messages()
        story_messages = _all_store[-100:] if len(_all_store) > 100 else _all_store

        for i, qa in enumerate(self.env.qa_list):
            if i < start_idx:
                continue  # 已在 checkpoint 中处理，跳过

            question = qa.get("question", "")
            gold = qa.get("answer", "")
            evidence = qa.get("evidence", [])

            # 如果有 mem_agent，先检索与问题相关的记忆
            retrieved_mem_context = ""
            if self.mem_agent is not None:
                retrieved = self.mem_agent.search_memories(question, top_k=5)
                retrieved = retrieved[:5]  # 确保最多5条，统一 mem0 和 amem 行为
                if retrieved:
                    mem_lines = [f"{idx}. {m.get('text', '')}" for idx, m in enumerate(retrieved, 1)]
                    retrieved_mem_context = "\n".join(mem_lines)
                    if verbose:
                        print(f"    [MemAgent] Retrieved {len(retrieved)} relevant memories")

            qa_user_msg = QA_SYSTEM_EN
            if retrieved_mem_context:
                qa_user_msg += f"\n\n[Retrieved relevant memories]\n{retrieved_mem_context}"
            qa_user_msg += f"\n\nQuestion: {question}\n\nResponse:"
            messages = story_messages + [{"role": "user", "content": qa_user_msg}]
            raw_output = self._call_llm_trpg(messages).strip()

            # 解析结构化输出
            reasoning, predicted = parse_qa_output(raw_output, is_en)

            # C. Acc
            judge_result, judge_reason = judge_answer(
                question, gold, predicted,
                call_llm=self._call_llm_trpg,
                env_config=self.env.config
            )
            if judge_result == "CONSISTENT":
                correct += 1
            elif judge_result == "UNDETERMINABLE":
                undet += 1

            # Inst
            inst_info = judge_inst(reasoning, predicted, is_en, call_llm=self._call_llm_trpg)

            # E. Cit
            cit_score, cit_reason = judge_citation(
                question, evidence, predicted, reasoning, is_en, call_llm=self._call_llm_trpg
            )

            # Read（LLM judge，评估对 gold evidence 的实际阅读理解）
            ev_lookup = getattr(self, "_evidence_lookup", {})
            read_score, read_reason = judge_read(
                question, reasoning, predicted, evidence, ev_lookup, is_en, call_llm=self._call_llm_trpg
            )

            mark = "✓" if judge_result == "CONSISTENT" else ("?" if judge_result == "UNDETERMINABLE" else "✗")
            acc = f"{correct}/{i+1}"

            self.env.results.append({
                **qa,
                "_predicted_reasoning": reasoning,
                "_predicted_answer": predicted,
                "_judge_result": judge_result,
                "_judge_reason": judge_reason,
                "_cit_score": cit_score,
                "_cit_reason": cit_reason,
                "_read_score": read_score,
                "_read_reason": read_reason,
                **inst_info,
            })

            # 每10题保存一次 checkpoint
            if (i + 1) % 10 == 0:
                self._save_trpg_checkpoint("qa", name=f"qa_{i+1:03d}", qa_idx=i, results=self.env.results)

            if verbose:
                print(f"  [{i+1:3d}/{n}] {mark} {judge_result:<16} "
                      f"({acc})  cat={qa.get('category','?')}  "
                      f"cit={cit_score}  read={read_score}  inst={'✓' if inst_info['_inst_pass'] else '✗'}  "
                      f"pred={predicted[:40]!r}")

        pct = correct / n * 100 if n else 0
        print(f"\n  QA complete: {n} questions  correct {correct}  accuracy {pct:.1f}%"
              + (f"  (incl. {undet} UNDETERMINABLE)" if undet else "") + "\n")
        self._print_category_stats()

    def _print_category_stats(self) -> None:
        """打印分类统计信息"""
        results = self.env.results

        # 按 category 统计
        by_cat: Dict[Any, List[str]] = {}
        for r in results:
            cat = r.get("category", "?")
            by_cat.setdefault(cat, []).append(r["_judge_result"])
        if len(by_cat) > 1:
            parts = []
            for cat, rs in sorted(by_cat.items(), key=lambda x: str(x[0])):
                c = sum(1 for r in rs if r == "CONSISTENT")
                parts.append(f"cat{cat}={c}/{len(rs)}")
            print(f"  By category: {'  '.join(parts)}")

        # 按 depth 统计
        _DEPTH_LABELS = {1: "Surface", 2: "Character", 3: "Cross-sect"}
        by_depth: Dict[Any, List[str]] = {}
        for r in results:
            d = r.get("depth", "?")
            by_depth.setdefault(d, []).append(r["_judge_result"])
        if by_depth:
            parts = []
            for d, rs in sorted(by_depth.items(), key=lambda x: str(x[0])):
                c = sum(1 for r in rs if r == "CONSISTENT")
                label = _DEPTH_LABELS.get(d, str(d))
                parts.append(f"{label}={c}/{len(rs)}")
            print(f"  By depth: {'  '.join(parts)}")

        # Inst 通过率
        inst_results = [r for r in results if "_inst_pass" in r]
        if inst_results:
            inst_pass = sum(1 for r in inst_results if r["_inst_pass"])
            print(f"  Inst pass rate: {inst_pass}/{len(inst_results)} ({inst_pass/len(inst_results)*100:.1f}%)")

        # E. Cit 分布
        cit_results = [r for r in results if "_cit_score" in r]
        if cit_results:
            cit_dist = {}
            for r in cit_results:
                s = r["_cit_score"]
                cit_dist[s] = cit_dist.get(s, 0) + 1
            parts = [f"{k}={v}" for k, v in sorted(cit_dist.items())]
            print(f"  E.Cit distribution: {'  '.join(parts)}")

        print()

    # ── LLM 调用 ──────────────────────────────────────────────────────────────

    def _call_llm_trpg(
        self,
        messages: List[Dict[str, str]],
        max_retries: int = 3,
        temperature: Optional[float] = None,
        use_judge_client: bool = False,
    ) -> str:
        """调用 LLM 生成回答（带重试）"""
        temp = temperature if temperature is not None else self._trpg_temperature
        client = self._judge_client if use_judge_client else self._trpg_client
        model = self._judge_model if use_judge_client else self._trpg_model

        for attempt in range(max_retries):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temp,
                )
                return resp.choices[0].message.content or ""
            except Exception as exc:
                print(f"  [LLM] Attempt {attempt+1}/{max_retries} failed: {exc}")
                if attempt < max_retries - 1:
                    time.sleep(5)
        return ""

    # ── Checkpoint ────────────────────────────────────────────────────────────

    def _save_trpg_checkpoint(self, phase: str, name: str, **extra) -> None:
        """保存 TRPG checkpoint 到 checkpoints/{session_id}/trpg_ckpt_{name}.json"""
        if not self.ckpt_dir:
            return
        store_items = [
            {
                "text": item.text,
                "role": item.meta.get("role", "user"),
                "step": item.meta.get("step", ""),
            }
            for item in self.store.items
        ]
        # 当前 section 的实际标题（方便人工核查）
        sections = getattr(self.env, "sections", [])
        if sections and 0 <= self._trpg_sec_idx < len(sections):
            sec_name = sections[self._trpg_sec_idx].get("section", f"Section {self._trpg_sec_idx + 1}")
        else:
            sec_name = ""

        data = {
            "_description": {
                "session_id": "Unique identifier for this run, corresponds to logs/trpg/{session_id}/ directory",
                "story_name": "Name of the story/script being read",
                "model": "LLM model name used",
                "phase": "Current phase: reading (in progress) / reading_complete / qa (answering questions)",
                "sec_idx": "Index of the section at the reading checkpoint (0-based); sections before this are skipped on resume",
                "sec_name": "Title of the section at the checkpoint (for reference only; sec_idx is used for resuming)",
                "turn_idx": "Index of the turn at the checkpoint (-2=not started, -1=header added); turns before this are skipped on resume",
                "total_turns": "Total number of conversation turns read so far",
                "compressions": "Number of compressions performed so far",
                "store_items": "All current memory store items (compressed summaries + raw dialogue lines); restored directly on resume",
                "qa_idx": "(QA phase) Index of the last answered question (0-based); resume continues from qa_idx+1",
                "results": "(QA phase) List of all completed QA results",
            },
            "session_id": self.session_id,
            "story_name": getattr(self.env.config, "story_name", ""),
            "model": self._trpg_model,
            "phase": phase,
            "timestamp": datetime.now().isoformat(),
            "total_turns": self._trpg_total_turns,
            "compressions": self._trpg_compressions,
            "sec_idx": self._trpg_sec_idx,
            "sec_name": sec_name,
            "turn_idx": self._trpg_turn_idx,
            "store_items": store_items,
            "amem_state": self.mem_agent.get_state() if self.mem_agent is not None else None,
            **extra,
        }
        path = self.ckpt_dir / f"trpg_ckpt_{name}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    # ── 结果保存 ──────────────────────────────────────────────────────────────

    def _save_trpg_results(self) -> None:
        """保存评测结果到 results.json"""
        results = self.env.results
        n = len(results)
        correct = sum(1 for r in results if r.get("_judge_result") == "CONSISTENT")
        undet = sum(1 for r in results if r.get("_judge_result") == "UNDETERMINABLE")
        pct = correct / n * 100 if n else 0.0

        # ── C. Acc by category ──────────────────────────────────────────────────
        by_cat: Dict[Any, Dict[str, int]] = {}
        for r in results:
            cat = r.get("category", "?")
            s = by_cat.setdefault(cat, {"total": 0, "correct": 0, "undeterminable": 0})
            s["total"] += 1
            if r.get("_judge_result") == "CONSISTENT":
                s["correct"] += 1
            elif r.get("_judge_result") == "UNDETERMINABLE":
                s["undeterminable"] += 1

        # ── Depth breakdown ─────────────────────────────────────────────────────
        _DEPTH_LABEL = {1: "Surface", 2: "Character", 3: "Cross-section"}
        by_depth: Dict[Any, Dict[str, int]] = {}
        for r in results:
            d = r.get("depth", "?")
            s = by_depth.setdefault(d, {"total": 0, "correct": 0})
            s["total"] += 1
            if r.get("_judge_result") == "CONSISTENT":
                s["correct"] += 1

        # ── Read: group by evidence count ────────────────────────────────────────
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

        # ── Comp: category 2+5 ───────────────────────────────────────────────────
        comp_r = [r for r in results if r.get("category") in (2, 5)]
        comp_cor = sum(1 for r in comp_r if r.get("_judge_result") == "CONSISTENT")

        # ── Inst ────────────────────────────────────────────────────────────────
        inst_r = [r for r in results if "_inst_pass" in r]
        inst_pass = sum(1 for r in inst_r if r.get("_inst_pass"))

        # ── E. Cit ──────────────────────────────────────────────────────────────
        cit_r = [r for r in results if "_cit_score" in r]
        cit_dist: Dict[str, int] = {}
        for r in cit_r:
            s = r["_cit_score"]
            cit_dist[s] = cit_dist.get(s, 0) + 1

        # ── Read Coverage ────────────────────────────────────────────────────────
        read_r = [r for r in results if "_read_score" in r]
        read_dist: Dict[str, int] = {}
        for r in read_r:
            s = r["_read_score"]
            read_dist[s] = read_dist.get(s, 0) + 1
        if read_r:
            read_avg_score = (
                read_dist.get("HIGH", 0) * 1.0 + read_dist.get("MEDIUM", 0) * 0.5
            ) / len(read_r)
        else:
            read_avg_score = 0.0

        output = {
            "story": self.env.config.story_name,
            "language": getattr(self.env.config, "test_language", "en"),
            "model": self._trpg_model,
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "compression_threshold": self.config.compression_threshold,
            "compression_count": self.config.compression_count,
            "total_turns_read": self._trpg_total_turns,
            "compressions_applied": self._trpg_compressions,
            "final_window_size": len(self.store.items),
            # ── C. Acc ──────────────────────────────────────────────────────────
            "accuracy": {
                "total": n,
                "correct": correct,
                "incorrect": n - correct - undet,
                "undeterminable": undet,
                "pct": round(pct, 2),
            },
            "accuracy_by_category": {
                str(cat): {
                    **s,
                    "pct": round(s["correct"] / s["total"] * 100, 2) if s["total"] else 0.0,
                }
                for cat, s in sorted(by_cat.items(), key=lambda x: str(x[0]))
            },
            # ── Depth ───────────────────────────────────────────────────────────
            "accuracy_by_depth": {
                f"{d}_{_DEPTH_LABEL.get(d, str(d))}": {
                    **s,
                    "pct": round(s["correct"] / s["total"] * 100, 2) if s["total"] else 0.0,
                }
                for d, s in sorted(by_depth.items(), key=lambda x: str(x[0]))
            },
            # ── Read ────────────────────────────────────────────────────────────
            "accuracy_by_read_complexity": {
                bucket: {
                    **s,
                    "pct": round(s["correct"] / s["total"] * 100, 2) if s["total"] else 0.0,
                }
                for bucket, s in by_read.items()
            },
            # ── Comp ────────────────────────────────────────────────────────────
            "comp_accuracy": {
                "total": len(comp_r),
                "correct": comp_cor,
                "pct": round(comp_cor / len(comp_r) * 100, 2) if comp_r else 0.0,
                "note": "category 2 (contradiction) + category 5 (skill comparison)",
            },
            # ── Inst ────────────────────────────────────────────────────────────
            "inst_stats": {
                "total": len(inst_r),
                "pass": inst_pass,
                "fail": len(inst_r) - inst_pass,
                "pass_rate": round(inst_pass / len(inst_r) * 100, 2) if inst_r else 0.0,
            },
            # ── E. Cit ──────────────────────────────────────────────────────────
            "cit_score_distribution": {
                **{k: cit_dist.get(k, 0) for k in ("HIGH", "MEDIUM", "LOW")},
                "total": len(cit_r),
                "high_rate": round(cit_dist.get("HIGH", 0) / len(cit_r) * 100, 2) if cit_r else 0.0,
            },
            # ── Read Coverage ────────────────────────────────────────────────────
            "read_coverage": {
                **{k: read_dist.get(k, 0) for k in ("HIGH", "MEDIUM", "LOW")},
                "total": len(read_r),
                "avg_score": round(read_avg_score, 4),
                "pct": round(read_avg_score * 100, 2),
                "high_rate": round(read_dist.get("HIGH", 0) / len(read_r) * 100, 2) if read_r else 0.0,
            },
            "qa_results": results,
        }

        if self.log_dir:
            out_file = self.log_dir / "results.json"
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(output, f, ensure_ascii=False, indent=2)

            print(f"  Results saved to: {out_file}")
            print(f"  Total: {n} questions  correct {correct}  accuracy {pct:.1f}%")
            if inst_r:
                print(f"  Inst pass rate: {inst_pass}/{len(inst_r)} ({output['inst_stats']['pass_rate']}%)")
            if cit_r:
                print(f"  E.Cit HIGH rate: {cit_dist.get('HIGH',0)}/{len(cit_r)} ({output['cit_score_distribution']['high_rate']}%)")
            if read_r:
                print(f"  Read HIGH rate: {read_dist.get('HIGH',0)}/{len(read_r)} ({output['read_coverage']['high_rate']}%)")

            # ── 自动生成 summary.txt ─────────────────────────────────────────
            if _SUMMARIZE_AVAILABLE:
                self._write_summary(output)

    def _write_summary(self, output: dict) -> None:
        """调用 summarize.compute_metrics 生成六维汇总表并保存到 summary.txt"""
        if not self.log_dir:
            return

        row = compute_metrics(output)
        rows = [row]

        buf = io.StringIO()

        def out(*args, **kwargs):
            print(*args, **kwargs)
            print(*args, **kwargs, file=buf)

        cols = ["model", "story", "Overall", "Acc", "Comp.", "Depth",
                "D1(S)", "D2(C)", "D3(X)", "Inst.", "Cit", "Read.", "n_qa"]
        widths = {c: max(len(c), max(len(str(r[c])) for r in rows)) for c in cols}
        header = "  ".join(c.ljust(widths[c]) for c in cols)
        sep    = "  ".join("-" * widths[c] for c in cols)

        out("\n=== Six-Dimensional Evaluation Summary ===\n")
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

        dims = row["_dims"]
        out(f"\n  {'─'*55}")
        out(f"  Score breakdown  model={row['model']}  story={row['story']}")
        out(f"  {'─'*55}")
        out(f"  {'Dimension':<10}  {'Wt':>4}  {'Raw (%)':>10}  {'Weighted':>8}  {'Grade':>5}")
        out(f"  {'-'*10}  {'-'*4}  {'-'*10}  {'-'*8}  {'-'*5}")
        total_weighted = 0.0
        for dim in _WEIGHTS:
            w      = _WEIGHTS[dim]
            raw    = dims.get(dim, 0.0) or 0.0
            wscore = w * raw / 100
            total_weighted += wscore
            out(f"  {dim:<10}  {w:>4.1f}  {raw:>10.2f}  {wscore:>8.4f}  {_grade(raw):>5}")
        out(f"  {'─'*55}")
        pct_overall = total_weighted / _TOTAL_WEIGHT * 100
        out(f"  {'Overall':<10}  {_TOTAL_WEIGHT:>4.1f}  "
            f"{'':>10}  {total_weighted:>8.4f}  "
            f"→ {pct_overall:.2f} / 100  {_grade(pct_overall)}")
        out()
        out("Grade: A+(≥90) A(≥80) B+(≥70) B(≥60) C+(≥50) C(≥40) D(≥30) F(<30)")

        summary_path = self.log_dir / "summary.txt"
        summary_path.write_text(buf.getvalue(), encoding="utf-8")
        print(f"  [Summary] Saved to: {summary_path}")


def run_trpg(
    env: Any,
    store: MemoryStore,
    config: AgentConfig,
    llm_config: LLMConfig,
    judge_llm_config: Optional[LLMConfig] = None,
    log_dir: Optional[Path] = None,
    session_id: str = "",
    ckpt_resume_from: Optional[str] = None,
    mem_agent: Optional[Any] = None,
) -> None:
    """运行 TRPG 评测（入口函数）

    Args:
        env: TRPG 游戏环境
        store: 记忆存储
        config: Agent 配置
        llm_config: 答题模型配置
        judge_llm_config: 判题模型配置
        log_dir: 日志目录
        session_id: 会话ID
        ckpt_resume_from: checkpoint 恢复路径
        mem_agent: A-mem 记忆代理（可选）
    """
    runner = TRPGRunner(
        env=env,
        store=store,
        config=config,
        llm_config=llm_config,
        judge_llm_config=judge_llm_config,
        log_dir=log_dir,
        session_id=session_id,
        ckpt_resume_from=ckpt_resume_from,
        mem_agent=mem_agent,
    )
    runner.run()
