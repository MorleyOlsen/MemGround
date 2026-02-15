# env/dust/utils/game_utils.py
"""Dust 游戏的工具类"""
from __future__ import annotations

from typing import Any, Dict, Optional

from galagent.common.schemas import Observation
from galagent.env.base_game_utils import BaseGameUtils


class DustGameUtils(BaseGameUtils):
    """Dust 推理游戏的工具类

    处理 Dust 游戏特定的：
    - 游戏状态上下文获取
    - 日志数据格式化
    - 动作执行（关键词解锁、事件阅读、排序提交、锁解锁）
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

    def get_game_context(self, env: Any) -> Dict[str, Any]:
        """获取 Dust 游戏的上下文信息

        Args:
            env: Dust 游戏环境实例

        Returns:
            包含游戏状态的上下文字典
        """
        game_context = {}

        # 获取当前记忆
        if self.memory_store:
            game_context.update(self.memory_store.get_memory_context())

        # 获取游戏状态信息
        if hasattr(env, 'get_state'):
            state = env.get_state()
            game_context['dust_state'] = {
                'current_node_id': state.get('current_node_id', ''),
                'keyword_pool': state.get('keyword_pool', []),
                'known_events': state.get('known_events', []),
                'event_pool': state.get('event_pool', []),
                'read_events': state.get('read_events', []),
                'locked_events': state.get('locked_events', {}),
                'score': state.get('score_points', 0),
                'keys': state.get('keys', 0),
                'character_orders': state.get('character_orders', {}),
                'order_judgements': state.get('order_judgements', []),
                'awarded_pairs': [list(pair) for pair in state.get('awarded_pairs', [])],
            }

            # 添加 order_gt 用于在 prompt_builder 中生成 trace
            if hasattr(env, 'order_gt'):
                game_context['order_gt'] = env.order_gt

            # 添加 lock_info 用于在 prompt_builder 中显示锁的问题
            if hasattr(env, 'lock_info'):
                game_context['lock_info'] = env.lock_info

        return game_context

    def format_log_data(
        self,
        step: int,
        obs: Observation,
        decision: Any,
        game_context: Dict[str, Any],
        retrieval_decision: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """格式化 Dust 游戏的日志数据

        Args:
            step: 当前步数
            obs: 当前观察
            decision: 决策对象
            game_context: 游戏上下文
            retrieval_decision: 检索决策信息（可选）

        Returns:
            完整的日志数据字典
        """
        # 提取决策信息
        action_type = getattr(decision, 'choice_index', 0)  # choice_index 存储 action_type (0-4)

        # 从 choice_text 解析 action_params
        import json
        action_params = {}
        try:
            choice_data = json.loads(decision.choice_text)
            action_params = choice_data.get("action_params", {})
        except (json.JSONDecodeError, AttributeError):
            pass

        rationale = getattr(decision, 'rationale', '')

        # 获取游戏状态
        dust_state = game_context.get('dust_state', {})

        # 构建 choices 字典（用于日志记录）
        choices = {
            "text": f"action_type={action_type}, params={action_params}",
            "decision_rationale": rationale
        }

        # 构建日志数据（必须包含 log_action 的必需字段）
        log_data = {
            # 标准必需字段
            "step": step,
            "node_id": obs.meta.get('current_node_id', dust_state.get('current_node_id', '')),
            "node_name": obs.meta.get('current_node_id', dust_state.get('current_node_id', '')),
            "scene_text": obs.text if isinstance(obs.text, str) else str(obs.text),
            "choices": choices,
            "file_retrieval": retrieval_decision,

            # Dust 游戏专用字段
            "action_type": action_type,
            "action_params": action_params,
            "current_node_id": dust_state.get('current_node_id', ''),
            "keyword_pool": dust_state.get('keyword_pool', []),
            "known_events": dust_state.get('known_events', []),
            "event_pool": dust_state.get('event_pool', []),
            "read_events": dust_state.get('read_events', []),
            "locked_events": dust_state.get('locked_events', {}),
            "score": dust_state.get('score', 0),
            "keys": dust_state.get('keys', 0),
            "character_orders": dust_state.get('character_orders', {}),
            "order_judgements": dust_state.get('order_judgements', []),
            "awarded_pairs": dust_state.get('awarded_pairs', []),
        }

        return log_data
   
    def execute_action(self, env: Any, decision: Any) -> bool:
        """执行 Dust 游戏的动作

        Args:
            env: Dust 游戏环境实例
            decision: 决策对象，包含 choice_index (动作类型 0-4) 和 choice_text (action_params JSON)

        Returns:
            bool: 动作是否执行成功
        """
        import json

        action_type = decision.choice_index

        # 从 choice_text 解析 action_params
        try:
            action_data = json.loads(decision.choice_text)
            action_params = action_data.get("action_params", {})
        except (json.JSONDecodeError, AttributeError):
            msg = f"无法解析动作参数: {decision.choice_text}"
            print(msg)
            if self.memory_store:
                self.memory_store.add_message(msg, role="system")
            return False

        try:
            action_success = False

            if action_type == 0:
                # unlock_keyword - 通过关键词解锁事件
                keyword = action_params.get('keyword', '')
                if not keyword:
                    return False
                newly_unlocked = env.apply_keyword_unlock(keyword)
                msg = f"使用关键词 '{keyword}' 解锁了 {len(newly_unlocked)} 个事件: {newly_unlocked}"
                print(msg)
                if self.memory_store:
                    self.memory_store.add_message(msg, role="system")
                action_success = True

            elif action_type == 1:
                # read_event - 阅读事件
                event_name = action_params.get('event_name', '')
                if not event_name:
                    return False
                env.read_event(event_name)
                msg = f"阅读事件 '{event_name}'"
                print(msg)
                if self.memory_store:
                    self.memory_store.add_message(msg, role="system")
                action_success = True

            elif action_type == 2:
                # submit_orders - 提交人物事件排序
                orders = action_params.get('orders', {})
                if not orders:
                    return False
                result = env.apply_orders(orders)
                msg = f"提交排序: 新增 {result['new_points']} 分, 获得 {result['keys_earned']} 把钥匙"
                print(msg)
                if self.memory_store:
                    self.memory_store.add_message(msg, role="system")
                action_success = True

            elif action_type == 3:
                # unlock_with_key - 用钥匙解锁黄色锁
                event_name = action_params.get('event_name', '')
                if not event_name:
                    return False
                success = env.unlock_by_key(event_name)
                if success:
                    msg = f"成功用钥匙解锁事件 '{event_name}'"
                else:
                    msg = f"解锁失败: '{event_name}' (钥匙不足或不是黄色锁)"
                print(msg)
                if self.memory_store:
                    self.memory_store.add_message(msg, role="system")
                action_success = success

            elif action_type == 4:
                # answer_lock - 回答问题解锁粉色/紫色锁
                event_name = action_params.get('event_name', '')
                answer = action_params.get('answer', '')
                if not event_name or not answer:
                    return False
                success = env.answer_lock(event_name, answer)
                if success:
                    msg = f"回答正确，解锁事件 '{event_name}'"
                else:
                    msg = f"回答错误，未能解锁事件 '{event_name}'"
                print(msg)
                if self.memory_store:
                    self.memory_store.add_message(msg, role="system")
                action_success = success

            else:
                msg = f"未知的动作类型: {action_type}"
                print(msg)
                if self.memory_store:
                    self.memory_store.add_message(msg, role="system")
                raise ValueError(msg)

            # 检查当前节点是否有 auto_link，如果有则自动跳转
            if action_success:
                self._handle_auto_link(env)

            return action_success

        except Exception as e:
            msg = f"执行动作失败: {e}"
            print(msg)
            if self.memory_store:
                self.memory_store.add_message(msg, role="system")
            return False

    def _handle_auto_link(self, env: Any) -> None:
        """处理节点的自动跳转逻辑

        Args:
            env: Dust 游戏环境实例
        """
        # 获取当前节点
        current_node = env._get_node_by_name(env.current_node_id)
        if not current_node:
            return

        # 检查是否有 auto_link
        auto_link = current_node.get("auto_link", "")
        if not auto_link:
            return

        # 获取目标节点
        target_node = env._get_node_by_name(auto_link)
        if not target_node:
            print(f"[警告] 未找到自动跳转目标节点: {auto_link}")
            return

        # 获取目标节点的 sub_name
        target_sub_name = target_node.get("sub_name", auto_link)

        # 添加到已知事件
        if target_sub_name not in env.known_events:
            env.known_events.add(target_sub_name)

        # 提取并添加关键词
        emphasize = target_node.get("emphasize", [])
        if emphasize:
            for keyword in emphasize:
                # 检查关键词是否已被使用过
                if keyword not in env.used_keywords:
                    env.keyword_pool.add(keyword)

        # 判断锁类型
        lock_info = env._get_lock_info(auto_link)
        lock_type = lock_info.get("type", "none")

        # 根据锁类型决定如何处理
        if lock_type == "none" or not lock_type:
            # 无锁，添加到可阅读事件池
            if target_sub_name not in env.event_pool:
                env.event_pool.append(target_sub_name)
             
        else:
            # 有锁，添加到对应的锁池
            lock_color = None
            if "yellow" in lock_type.lower():
                lock_color = "yellow"
            elif "pink" in lock_type.lower():
                lock_color = "pink"
            elif "purple" in lock_type.lower():
                lock_color = "purple"

            if lock_color and target_sub_name not in env.locked_events[lock_color]:
                env.locked_events[lock_color].add(target_sub_name)
         
                

    def retrieve_information(self, retrieval_decision: Dict[str, Any], config: Any = None) -> Optional[str]:
        """检索信息（Dust 游戏：根据LLM决策检索事件的key_info）

        Args:
            retrieval_decision: policy.decide_retrieval()返回的检索决策
                包含: need_retrieval, filenames, reason
            config: Agent配置对象（保留用于接口一致性）

        Returns:
            格式化的检索结果文本
        """
        # 提取检索决策信息
        need_retrieval = retrieval_decision.get("need_retrieval", False)
        filenames = retrieval_decision.get("filenames", [])

        if not need_retrieval or not filenames:
            return None

        # 确保 filenames 是列表
        if not isinstance(filenames, list):
            return None

        # 获取每个事件的 key_info
        results = []
        for event_name in filenames:
            # 使用环境的 _get_node_by_name 方法获取节点
            node = self.env._get_node_by_name(event_name)
            if node:
                key_info = node.get("key_info", [])
                if key_info:
                    # 格式化输出 - 减少换行符
                    if isinstance(key_info, list):
                        # 将列表元素用空格连接，而不是换行
                        key_info_text = " ".join(key_info)
                    else:
                        key_info_text = str(key_info)
                    results.append(f"事件: {event_name} === {key_info_text}")
            else:
                results.append(f"事件: {event_name} === （未找到该事件）")

        # 拼接所有结果 - 使用分号和空格分隔，而不是换行
        if results:
            return "; ".join(results)
        else:
            return None

    def post_action_hook(self, decision: Any, action_success: bool = True, step: int = 0) -> None:
        """动作执行后的钩子

        Args:
            decision: 执行的决策
            action_success: 动作是否执行成功
            step: 当前步数
        """
        # Dust 游戏目前不需要额外的后处理逻辑
        # 所有状态更新都在 DustEnv 的方法中完成
        pass

    def get_console_log_info(
        self,
        obs: Observation,
        search_results: str,
        decision: Any
    ) -> Optional[str]:
        """获取 Dust 游戏的控制台日志信息

        Args:
            obs: 当前观察
            search_results: 检索结果
            decision: 决策对象

        Returns:
            格式化的控制台日志字符串
        """
        lines = []
        lines.append(f"=== Dust 推理游戏 - Step {obs.meta.get('step', 0)} ===")

        # 显示游戏状态
        meta = obs.meta
        lines.append(f"得分: {meta.get('score', 0)} | 钥匙: {meta.get('keys', 0)}")
        lines.append(f"已知事件: {len(meta.get('known_events', []))} | "
                    f"可阅读: {len(meta.get('event_pool', []))} | "
                    f"已读: {len(meta.get('read_events', []))}")

        # 显示决策
        action_type = getattr(decision, 'action_type', 'unknown')
        action_params = getattr(decision, 'action_params', {})
        lines.append(f"动作: {action_type}")
        lines.append(f"参数: {action_params}")
        lines.append(f"理由: {getattr(decision, 'rationale', 'N/A')}")

        return "\n".join(lines)

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
            total_tokens = self.memory_store.get_total_tokens_estimate()

        # 如果超过token限制，删除记忆
        max_context_tokens = config.max_context_tokens
        if total_tokens > max_context_tokens:
            excess_tokens = total_tokens - max_context_tokens
            delete_count = max(1, excess_tokens // 50)
            deleted = self.memory_store.delete_by_priority(delete_count)

            if deleted > 0 and config.verbose:
                print(f"[上下文管理] 删除了 {deleted} 条记忆（优先assistant）")
                print(f"  Prompt token估算: {int(total_tokens)} (限制: {max_context_tokens})")

    def get_state(self) -> Dict[str, Any]:
        """获取游戏工具状态用于 checkpoint

        Returns:
            状态字典（Dust 游戏的 GameUtils 目前无额外状态）
        """
        return {}

    def restore_state(self, state: Dict[str, Any]) -> None:
        """从 checkpoint 恢复游戏工具状态

        Args:
            state: 游戏工具状态字典
        """
        # Dust 游戏的 GameUtils 目前无额外状态需要恢复
        pass
