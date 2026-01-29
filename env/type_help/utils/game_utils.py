# env/type_help/utils/game_utils.py
"""Type Help游戏的工具类"""
from __future__ import annotations

from typing import Any, Dict, Optional, List

from galagent.common.schemas import Observation
from galagent.env.base_game_utils import BaseGameUtils


class TypeHelpGameUtils(BaseGameUtils):
    """Type Help游戏的工具类

    处理Type Help游戏特定的：
    - 文件追踪信息获取
    - 日志数据格式化
    - 已读文件列表管理（仅保留文件名和ID）
    """

    def __init__(self):
        super().__init__()
        self.read_files: List[Dict[str, str]] = []  # 已读文件列表 [{"id": node_id, "name": name}, ...]
        self.memory_store = None  # 记忆存储引用

    def set_memory_store(self, store):
        """设置记忆存储引用"""
        self.memory_store = store

    def add_read_file(self, node_id: str, name: str) -> None:
        """添加已读文件到列表

        Args:
            node_id: 节点ID
            name: 文件名
        """
        # 检查是否已存在
        for file in self.read_files:
            if file["id"] == node_id and file["name"] == name:
                return

        self.read_files.append({"id": node_id, "name": name})

    def get_read_files_text(self) -> str:
        """获取已读文件列表的文本表示（用于添加到prompt）

        Returns:
            格式化后的已读文件列表字符串
        """
        if not self.read_files:
            return "尚未阅读任何文件"

        lines = ["已阅读的文件:"]
        for file in self.read_files:
            lines.append(f"- {file['name']} (ID: {file['id']})")

        return "\n".join(lines)

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
        from env.type_help.utils.file_retriever import TypeHelpFileRetriever

        # 提取检索决策信息
        need_retrieval = retrieval_decision.get("need_retrieval", False)
        filenames = retrieval_decision.get("filenames", [])

        if not need_retrieval or not filenames:
            return None, retrieval_decision

        # 确保 filenames 是列表
        if not isinstance(filenames, list):
            return None, retrieval_decision

        # 使用文件检索器获取文件内容
        if not hasattr(self, 'env'):
            return None, retrieval_decision

        retriever = TypeHelpFileRetriever(self.env)
        file_results = retriever.retrieve_files(filenames)
        formatted_result = retriever.format_retrieved_files(file_results)

        if config.verbose:
            print(f"Retrieved {len(file_results)} files: {filenames}")

        return formatted_result, retrieval_decision

    def set_env(self, env):
        """设置环境引用"""
        self.env = env

    def observation(self, obs: Observation) -> str:
        """简化观察信息，只保留关键内容

        Args:
            obs: 原始观察对象

        Returns:
            简化后的观察文本，只包含name, key_info, location, character.name, character.number
        """
        lines = [f"文件: {obs.name}"]

        # 从环境中获取节点的memory信息
        if hasattr(self, 'env') and hasattr(self.env, 'nodes'):
            node = self.env.nodes.get(obs.name)
            if node and 'memory' in node:
                memory = node['memory']

                # 地点
                if memory.get('location'):
                    lines.append(f"地点: {memory['location']}")

                # 关键信息
                if memory.get('key_info'):
                    lines.append("关键信息:")
                    for info in memory['key_info']:
                        lines.append(f"  - {info}")

                # 人物（只保留name和number）
                if memory.get('characters'):
                    lines.append("人物:")
                    for char in memory['characters']:
                        char_str = f"  - {char.get('name', '')}"
                        if char.get('number'):
                            char_str += f" (编号: {char['number']})"
                        lines.append(char_str)

        return "\n".join(lines)

    def post_action_hook(self, obs: Any, decision: Any, action_success: bool = True, step: int = 0) -> None:
        """动作执行后的钩子：记录已读文件并解锁新文件

        Args:
            obs: 当前观察
            decision: 执行的决策
            action_success: 动作是否执行成功
            step: 当前步数
        """
        # 如果动作失败（文件不存在），添加失败反馈到记忆
        if not action_success:
            if self.memory_store and hasattr(decision, 'choice_text'):
                failure_msg = f"尝试打开文件 '{decision.choice_text}' 失败：文件不存在。请根据已有信息推断正确的文件名或调整文件名的顺序（数字从小到大）。"
                self.memory_store.add_message(failure_msg, role="system", step=step)
            return  # 失败时不执行后续操作

        # 成功时的处理：添加已读文件到列表
        self.add_read_file(obs.node_id, obs.name)

        # 检查当前节点的memory字段是否有files字段，如果有则解锁这些文件
        if hasattr(self, 'env') and hasattr(self.env, 'nodes'):
            node = self.env.nodes.get(obs.node_id)
            if node and 'memory' in node:
                memory = node['memory']
                files = memory.get('files', [])

                # 解锁files列表中的文件（只解锁完整的文件名，不包含?的）
                for file in files:
                    if "?" not in file:
                        self.env.file_tracker.unlock_file(file)

    def manage_memory(self, max_context_tokens: int, config: Any) -> None:
        """管理记忆，确保已读文件列表始终保留

        Args:
            max_context_tokens: 最大上下文token数
            config: Agent配置对象
        """
        if not self.memory_store:
            return

        # 计算已读文件列表的token数
        read_files_tokens = self.get_read_files_token_estimate()

        # 计算其他记忆的token数
        other_memory_tokens = self.memory_store.get_total_tokens_estimate()

        # 总token数
        total_tokens = read_files_tokens + other_memory_tokens

        # 如果超过限制，删除最早的记忆（但保留已读文件列表）
        if total_tokens > max_context_tokens:
            # 计算需要删除的token数
            excess_tokens = total_tokens - max_context_tokens

            # 计算需要删除的记忆条数（假设每条记忆约50 tokens）
            delete_count = max(1, excess_tokens // 50)

            # 删除最早的记忆
            deleted = self.memory_store.delete_oldest(delete_count)

            if deleted > 0 and config.verbose:
                print(f"[已读文件列表保护] 删除了 {deleted} 条最早的记忆以保留已读文件列表")
                print(f"  已读文件列表: {read_files_tokens} tokens, 其他记忆: {self.memory_store.get_total_tokens_estimate()} tokens")

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
            "decision_rationale": decision.rationale
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
        return {
            "read_files": self.read_files.copy()
        }

    def restore_state(self, state: Dict[str, Any]) -> None:
        """从checkpoint恢复游戏工具状态

        Args:
            state: 游戏工具状态字典
        """
        self.read_files = state["read_files"].copy()
        print(f"[GameUtils] 已恢复状态: {len(self.read_files)}个已读文件")
