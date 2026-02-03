# env/type_help/utils/game_utils.py
"""Type Help游戏的工具类"""
from __future__ import annotations

from typing import Any, Dict, Optional

from galagent.common.schemas import Observation
from galagent.env.base_game_utils import BaseGameUtils
from env.type_help.utils.file_retriever import TypeHelpFileRetriever


class TypeHelpGameUtils(BaseGameUtils):
    """Type Help游戏的工具类

    处理Type Help游戏特定的：
    - 文件追踪信息获取
    - 日志数据格式化
    - 已读文件列表管理（仅保留文件名和ID）
    """

    def __init__(self):
        super().__init__()
        self.memory_store = None  # 记忆存储引用

    def set_memory_store(self, store):
        """设置记忆存储引用"""
        self.memory_store = store

    def set_env(self, env):
        """设置环境引用"""
        self.env = env

    def get_read_files_text(self) -> str:
        """获取已读文件列表的文本表示（用于添加到prompt）

        Returns:
            格式化后的已读文件列表字符串
        """
        if hasattr(self, 'env') and hasattr(self.env, 'file_tracker'):
            return self.env.file_tracker.get_read_files_text()
        return "尚未阅读任何文件"

    def get_read_files_token_estimate(self, chars_per_token: float = 2.5) -> int:
        """估算已读文件列表的token数量

        Args:
            chars_per_token: 每个token的平均字符数

        Returns:
            估算的token数
        """
        text = self.get_read_files_text()
        return int(len(text) / chars_per_token)

    def retrieve_information(self, retrieval_decision: Dict[str, Any], config: Any) -> tuple[Optional[str], Optional[Dict[str, Any]]]:
        """检索信息（Type Help 游戏：根据LLM决策打开文件）

        Args:
            retrieval_decision: policy.decide_retrieval()返回的检索决策
                包含: need_retrieval, filenames, reason
            config: Agent配置对象

        Returns:
            (格式化的检索结果文本, 检索决策信息字典)
            检索决策信息包含: need_retrieval, filenames, reason
        """
        # 提取检索决策信息
        need_retrieval = retrieval_decision.get("need_retrieval", False)
        filenames = retrieval_decision.get("filenames", [])

        if not need_retrieval or not filenames:
            return None

        # 确保 filenames 是列表
        if not isinstance(filenames, list):
            return None

        # 使用文件检索器获取文件内容
        if not hasattr(self, 'env'):
            return None
        
        retriever = TypeHelpFileRetriever(self.env)
        file_results = retriever.retrieve_files(filenames)
        formatted_result = retriever.format_retrieved_files(file_results)

        return formatted_result


    def post_action_hook(self, decision: Any, action_success: bool = True, step: int = 0) -> None:
        """动作执行后的钩子

        Args:
            decision: 执行的决策
            action_success: 动作是否执行成功
            step: 当前步数
        """
        # 所有逻辑已移到 type_help_env.py 的 choose_by_filename 中
        # 保留此方法以保持接口一致性
        pass

    def _compress_memory(self, count: int, llm_client, llm_config, prompt_builder) -> Optional[str]:
        """压缩最早的n轮对话

        Args:
            count: 要压缩的对话轮次数
            llm_client: LLM客户端
            llm_config: LLM配置
            prompt_builder: Prompt构建器

        Returns:
            压缩后的文本，如果失败则返回None
        """
        if not self.memory_store or len(self.memory_store._items) < count:
            return None

        # 获取最早的n条记忆
        items_to_compress = self.memory_store._items[:count]

        # 构建对话文本列表
        conversations = []
        for item in items_to_compress:
            role = item.meta.get("role", "user")
            conversations.append(f"[{role}] {item.text}")

        # 使用prompt_builder构建压缩prompt
        compression_prompt = prompt_builder.build_compression_prompt(conversations)

        try:
            # 调用LLM进行压缩
            response = llm_client.chat.completions.create(
                model=llm_config.model,
                messages=[{"role": "user", "content": compression_prompt}],
                temperature=0.3  # 使用较低的temperature以保持准确性
            )

            compressed_text = response.choices[0].message.content or ""

            # 删除最早的n条记忆
            for _ in range(count):
                if len(self.memory_store._items) > 0:
                    self.memory_store._items.pop(0)

            # 将压缩后的内容添加到记忆（使用add_message）
            self.memory_store.add_message(
                content=f"[压缩记忆] {compressed_text}",
                role="system",
                compressed=True
            )

            # 将刚添加的记忆移到开头
            if len(self.memory_store._items) > 0:
                compressed_item = self.memory_store._items.pop()
                self.memory_store._items.insert(0, compressed_item)

            return compressed_text
        except Exception as e:
            print(f"[记忆压缩] 压缩失败: {e}")
            return None

    def manage_memory(self, config: Any, full_prompt: str = "", llm_client=None, llm_config=None, prompt_builder=None) -> None:
        """管理记忆，确保完整prompt不超过token限制

        Args:
            config: Agent配置对象
            full_prompt: 完整的prompt文本（可选，包括system_prompt + user_prompt等所有内容）
            llm_client: LLM客户端（用于记忆压缩）
            llm_config: LLM配置（用于记忆压缩）
            prompt_builder: Prompt构建器（用于构建压缩prompt）
        """
        if not self.memory_store:
            return

        # 检查是否启用压缩且对话轮次超过阈值
        if (hasattr(config, 'enable_compression') and config.enable_compression and
            hasattr(config, 'compression_threshold') and hasattr(config, 'compression_count')):

            current_turns = len(self.memory_store._items)

            if current_turns > config.compression_threshold:
                if config.verbose:
                    print(f"[记忆压缩] 对话轮次 {current_turns} 超过阈值 {config.compression_threshold}，开始压缩...")

                # 压缩最早的n轮对话
                if llm_client and llm_config and prompt_builder:
                    compressed_text = self._compress_memory(config.compression_count, llm_client, llm_config, prompt_builder)

                    if compressed_text and config.verbose:
                        print(f"[记忆压缩] 成功压缩 {config.compression_count} 轮对话")
                        print(f"[记忆压缩] 压缩后内容长度: {len(compressed_text)} 字符")
                else:
                    if config.verbose:
                        print(f"[记忆压缩] 警告: 未提供LLM客户端或Prompt构建器，跳过压缩")

        # 如果提供了完整prompt，直接计算token数
        if full_prompt:
            total_tokens = len(full_prompt) / 2.5  # 中文约2.5字符/token
        else:
            # 否则使用估算方式（向后兼容）
            read_files_tokens = self.get_read_files_token_estimate()
            conversation_history_tokens = self.memory_store.get_total_tokens_estimate()
            total_tokens = read_files_tokens + conversation_history_tokens

        # 如果超过token限制，删除记忆
        max_context_tokens = config.max_context_tokens
        if total_tokens > max_context_tokens:
            excess_tokens = total_tokens - max_context_tokens
            delete_count = max(1, excess_tokens // 50)
            deleted = self.memory_store.delete_by_priority(delete_count)

            if deleted > 0 and config.verbose:
                print(f"[上下文管理] 删除了 {deleted} 条记忆（优先assistant）")
                print(f"  Prompt token估算: {int(total_tokens)} (限制: {max_context_tokens})")

    def get_game_context(self, env: Any) -> Dict[str, Any]:
        """获取Type Help游戏的上下文信息

        Args:
            env: Type Help游戏环境实例

        Returns:
            包含记忆、文件追踪信息和已读文件列表的上下文字典
        """
        game_context = {}

        # 获取当前记忆
        if self.memory_store:
            game_context.update(self.memory_store.get_memory_context())

        # 获取文件追踪信息
        if hasattr(env, 'get_file_tracker_info'):
            game_context['file_tracker_info'] = env.get_file_tracker_info()

        # 添加已读文件列表的文本表示
        game_context['read_files_text'] = self.get_read_files_text()

        return game_context

    def format_log_data(
        self,
        step: int,
        obs: Observation,
        decision: Any,
        game_context: Dict[str, Any],
        retrieval_decision: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """格式化Type Help游戏的日志数据

        Args:
            step: 当前步数
            obs: 当前观察
            decision: 决策对象
            game_context: 游戏上下文
            retrieval_decision: 检索决策信息（包含need_retrieval, filenames, reason）

        Returns:
            完整的日志数据字典
        """
        # 格式化choices为新格式：{"text": 文件名, "decision_rationale": 原因}
        choices = {
            "text": decision.choice_text if hasattr(decision, 'choice_text') else "",
            "decision_rationale": decision.rationale,
            "recall": decision.recall if hasattr(decision, 'recall') else []
        }

        # 获取文件追踪信息
        file_tracker_info = game_context.get('file_tracker_info')
        unlocked_files = None
        attempted_files = None
        success_files = None
        failed_files = None
        if file_tracker_info:
            unlocked_files = file_tracker_info.get("unlocked_files", [])
            attempted_files = file_tracker_info.get("attempted_files", [])
            success_files = file_tracker_info.get("success_files", [])
            failed_files = file_tracker_info.get("failed_files", [])

        # 构建日志数据
        log_data = {
            "step": step,
            "node_id": obs.node_id,
            "node_name": obs.name,
            "scene_text": obs.text,
            "choices": choices,
            "file_retrieval": None,
            "unlocked_files": unlocked_files,
            "attempted_files": attempted_files,
            "success_files": success_files,
            "failed_files": failed_files
        }

        # 添加文件检索决策信息
        if retrieval_decision:
            log_data["file_retrieval"] = {
                "need_retrieval": retrieval_decision.get("need_retrieval", False),
                "opened_files": retrieval_decision.get("filenames", []),
                "reason": retrieval_decision.get("reason", "")
            }

        return log_data

    def execute_action(self, env: Any, decision: Any) -> bool:
        """执行Type Help游戏的动作

        Args:
            env: Type Help游戏环境实例
            decision: 决策对象

        Returns:
            bool: 动作是否执行成功
        """
        # Type Help游戏使用文件名选择
        if decision.choice_text:
            if hasattr(env, 'choose_by_filename'):
                return env.choose_by_filename(decision.choice_text)
            else:
                raise RuntimeError("Environment does not support choose_by_filename")
        else:
            raise ValueError("Type Help game requires choice_text in decision")

    def get_console_log_info(
        self,
        obs: Observation,
        search_results: str,
        decision: Any
    ) -> Optional[str]:
        """获取Type Help游戏的控制台日志信息

        Args:
            obs: 当前观察
            search_results: 检索结果
            decision: 决策对象

        Returns:
            格式化的控制台日志字符串
        """
        import json

        lines = []
        lines.append(f"Node: {obs.node_id} ({obs.name})")
        lines.append(f"Text: {obs.text[:100]}...")

        # 显示LLM的决策
        lines.append(f"Decision: {decision.choice_text if hasattr(decision, 'choice_text') else 'N/A'}")
        lines.append(f"Rationale: {decision.rationale}")

        # 显示检索结果
        if search_results:
            data = json.loads(search_results)
            results = data.get("results", [])
            if results:
                lines.append(f"Retrieved Memory ({len(results)} items):")
                for r in results[:3]:  # 只显示前3条
                    lines.append(f"  - {r['text'][:80]}...")

        return "\n".join(lines)

    def get_state(self) -> Dict[str, Any]:
        """获取游戏工具状态用于checkpoint

        Returns:
            包含已读文件列表的状态字典
        """
        read_files = []
        if hasattr(self, 'env') and hasattr(self.env, 'file_tracker'):
            read_files = self.env.file_tracker.get_read_files()
        return {
            "read_files": read_files
        }

    def restore_state(self, state: Dict[str, Any]) -> None:
        """从checkpoint恢复游戏工具状态

        Args:
            state: 游戏工具状态字典
        """
        if hasattr(self, 'env') and hasattr(self.env, 'file_tracker'):
            self.env.file_tracker.read_files = state["read_files"].copy()
            print(f"[GameUtils] 已恢复状态: {len(state['read_files'])}个已读文件")
