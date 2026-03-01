# galagent/env/dust_env.py
"""Dust 推理游戏环境"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Set, Tuple
from pathlib import Path
import base64

from galagent.common.schemas import Observation, Choice, Memory, Character
from galagent.env.base_env import BaseGameEnv, GameConfig
from env.dust.utils.node_index import build_node_indices, get_events_by_tag, get_node_characters
from env.dust.utils.scoring import judge_character_orders, calculate_keys_earned, can_unlock_with_key


@dataclass
class DustConfig(GameConfig):
    """Dust 游戏配置"""
    data_path: Path = Path("dataset/The_dust_settels")
    game_type: str = "dust"
    start_node_id: str = "start"
    key_threshold: int = 5  # 获得一把钥匙所需的分数
    ocr_appcode: str = "244a6973290f4311b061fc1b4969aa4c"  # 阿里云 OCR APPCODE TODO：之后给他改到config.yaml里
    test_language: str = "ch"  # ch or en (Chinese or English prompts)


class DustEnv(BaseGameEnv):
    """Dust 推理游戏环境

    核心机制：
    1. 关键词发现：从文本中识别关键词，形成关键词池
    2. 事件解锁：通过关键词解锁事件名（不包含完整内容）
    3. 事件阅读：选择已解锁事件进行阅读，获得完整内容
    4. 人物事件排序：判断事件在各角色视角下的发生顺序
    5. 计分与钥匙：排序正确获得点数，累积点数获得钥匙
    6. 锁机制：pink/purple 通过回答问题解锁，yellow 通过消耗钥匙解锁
    """

    def __init__(self, config: DustConfig):
        super().__init__(config)

        # 原始数据
        self.raw_nodes: List[Dict] = []  # 原始完整节点数据

        # 数据索引（来自 nodes.json）
        self.nodes: List[Dict] = []  # 节点列表: [{"name": "step_1","sub_name":"xxxxxx", "emphasize": [...], "type": "...", "auto_link": "...", "key_info": [...]}]
        self.tag_index: Dict[str, List[str]] = {}  # keyword/tag -> event_name 列表
        self.lock_info: List[Dict] = []  # 锁信息列表: [{"name": "A", "type": "...", "question": "...", "answer": "..."}]

        # 角色顺序 ground truth（来自 order_gt.json）
        self.order_gt: List[Dict] = []  # 角色列表: [{"name": "角色名", "tag": "角色标签", "dialogue": [{"name": "事件名", "number": 顺序}]}]
        self.character_names: List[str] = []  # 所有角色名列表
        self.character_tags: Dict[str, str] = {}  # 角色名 -> 角色标签的映射

        # 游戏状态（需要持久化）
        self.keyword_pool: Set[str] = set()  # 已发现的关键词
        self.used_keywords: Set[str] = set()  # 已使用过的关键词（用于去重）
        self.known_events: Set[str] = set()  # 已知事件名（包含已解锁、已读、锁池）
        self.event_pool: List[str] = []  # 可阅读事件名池
        self.read_events: Set[str] = set()  # 已阅读事件名
        self.locked_events: Dict[str, Set[str]] = {
            "pink": set(),
            "purple": set(),
            "yellow": set()
        }  # 锁事件名池子（分桶）

        self.character_orders: Dict[str, List[str]] = {}  # 每个人物的事件顺序列表
        self.order_judgements: List[Dict] = []  # 排序判定结果列表
        self.score_points: int = 0  # 当前得分
        self.keys: int = 0  # 当前钥匙数
        self.awarded_pairs: Set[Tuple[str, str, str]] = set()  # 已计分的 (character, earlier, later) 对
        self.submitted_pairs: Set[Tuple[str, str, str]] = set()  # 所有已提交过的 (character, earlier, later) 对（防重复记录）

        # dialogue.json 文本查找表（事件名 -> 完整文本）
        self.dialogue_lookup: Dict[str, str] = {}
        # dialogue.json 时间查找表（事件名 -> 发生时间）
        self.dialogue_time_lookup: Dict[str, str] = {}
        # 通过排序得分已揭露时间的事件集合
        self.time_revealed_events: Set[str] = set()
        # 多事件锁的提交进度（锁事件名 -> 已正确提交的事件名集合）
        self.multi_lock_progress: Dict[str, Set[str]] = {}

        # 加载游戏数据
        self.load_game_data()

    def load_game_data(self) -> None:
        """加载游戏数据并构建索引"""
        data_file = self.config.data_path / "nodes.json"
        if not data_file.exists():
            raise FileNotFoundError(f"Game data not found: {data_file}")

        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 保存原始节点数据
        self.raw_nodes = data.get("game", {}).get("nodes", [])

        # 构建索引
        self.nodes, self.tag_index, self.lock_info = build_node_indices(self.raw_nodes)

        # 加载角色顺序 ground truth（英文模式优先加载 order_gt-en.json）
        if self.config.test_language == "en":
            order_gt_file = self.config.data_path / "order_gt-en.json"
        else:
            order_gt_file = self.config.data_path / "order_gt.json"
        if order_gt_file.exists():
            with open(order_gt_file, 'r', encoding='utf-8') as f:
                order_gt_data = json.load(f)
            self.order_gt = order_gt_data.get("text", [])

            # 提取角色名称和标签
            for character in self.order_gt:
                char_name = character.get("name", "")
                char_tag = character.get("tag", "")
                if char_name:
                    self.character_names.append(char_name)
                    if isinstance(char_tag, str):
                        self.character_tags[char_name] = char_tag
                    elif isinstance(char_tag, list):
                        # 如果 tag 是列表，取第一个或连接所有
                        self.character_tags[char_name] = ", ".join(char_tag)

            # 初始化所有角色的空排序列表
            for char_name in self.character_names:
                self.character_orders[char_name] = []
            print(f"[Dust] 已加载 {order_gt_file.name}，共 {len(self.character_names)} 个角色")
        else:
            print(f"[警告] 未找到角色顺序文件: {order_gt_file}")

        # 加载 dialogue.json 文本查找表（英文模式优先加载 dialogue-en.json）
        if self.config.test_language == "en":
            dialogue_file = self.config.data_path / "dialogue-en.json"
        else:
            dialogue_file = self.config.data_path / "dialogue.json"
        if dialogue_file.exists():
            with open(dialogue_file, 'r', encoding='utf-8') as f:
                dialogue_data = json.load(f)
            # 用 .strip() 规范化 name 键，防止尾部空格造成查找失败
            self.dialogue_lookup = {
                entry["name"].strip(): entry["text"]
                for entry in dialogue_data.get("text", [])
                if "name" in entry and "text" in entry
            }
            self.dialogue_time_lookup = {
                entry["name"].strip(): entry["time"]
                for entry in dialogue_data.get("text", [])
                if "name" in entry and "time" in entry
            }
            print(f"[Dust] 已加载 {dialogue_file.name}，共 {len(self.dialogue_lookup)} 条文本")
        else:
            print(f"[警告] 未找到对话文件: {dialogue_file}")

        # 初始化起始事件
        self._initialize_start_events()

    def _initialize_start_events(self) -> None:
        """初始化起始可读事件"""
        # 获取起始节点
        start_node_name = self.config.start_node_id
        start_node = self._get_node_by_name(start_node_name)

        if not start_node:
            print(f"[警告] 未找到起始节点: {start_node_name}")
            return

        # 检查起始节点是否有 auto_link
        auto_link = start_node.get("auto_link", "")

        if auto_link:
            # 如果有 auto_link，初始化跳转目标节点
            print(f"[初始化] 起始节点 '{start_node_name}' 自动跳转到 '{auto_link}'")

            # 获取目标节点
            target_node = self._get_node_by_name(auto_link)
            if not target_node:
                print(f"[警告] 未找到自动跳转目标节点: {auto_link}")
                return

            # 添加目标节点到可阅读事件池
            target_sub_name = target_node.get("sub_name", auto_link)
            self.event_pool.append(target_sub_name)
            self.known_events.add(target_sub_name)

            # 提取目标节点的关键词（emphasize）并添加到关键词池
            emphasize = target_node.get("emphasize", [])
            for keyword in emphasize:
                # 检查关键词是否已被使用过
                if keyword not in self.used_keywords:
                    self.keyword_pool.add(keyword)

            print(f"[初始化] 添加事件 '{target_sub_name}' 到可阅读池")
            print(f"[初始化] 添加关键词: {emphasize}")
        else:
            # 如果没有 auto_link，使用起始节点本身（如果需要的话）
            # 但根据注释，开始节点只是提供故事背景，不参与对话事件的排序
            # 所以这里不添加起始节点到事件池
            print(f"[初始化] 起始节点 '{start_node_name}' 没有自动跳转")

    def _get_node_by_name(self, node_name: str) -> Dict[str, Any]:
        """根据节点名获取节点数据

        Args:
            node_name: 节点名称（可以是 name 或 sub_name）

        Returns:
            节点数据字典,如果未找到则返回空字典
        """
        for node in self.nodes:
            # 先匹配 name 字段
            if node.get("name") == node_name:
                return node
            # 再匹配 sub_name 字段（如果存在）
            if node.get("sub_name") == node_name:
                return node
        return {}

    def _is_dialogue_node(self, node_name: str) -> bool:
        """判断节点是否为对话条目（对话-数字 或 talk-数字 格式）"""
        return bool(re.match(r'^\s*对话[-－—\uff0d]\d+\s*$', node_name) or
                    re.match(r'^\s*talk-\d+\s*$', node_name, re.IGNORECASE))

    def _get_lock_info(self, event_name: str) -> Dict[str, Any]:
        """根据事件名获取锁信息

        Args:
            event_name: 事件名称

        Returns:
            锁信息字典,如果未找到则返回 {"name": event_name, "type": "none", "question": "", "answer": ""}
        """
        for lock in self.lock_info:
            if lock["sub_name"] == event_name:
                return lock
        return {"name": event_name, "type": "none", "question": "", "answer": ""}

    
    def observe(self) -> Observation:
        """获取当前环境观察

        Returns:
            包含当前游戏状态的 Observation 对象，对于dust游戏而言，只用上了obs.text字段
        """

        # 获取当前节点信息
        current_node = self._get_node_by_name(self.current_node_id)
        node_name = current_node.get("sub_name", self.current_node_id)

        # 如果节点已经被读取过，直接使用 key_info，不需要重新读取
        image_urls = []
        if node_name in self.read_events:
            # 已读节点，直接获取缓存的 key_info
            node_key_info = current_node.get("key_info", []) if current_node else []
        else:
            # 未读节点：根据节点类型决定读取方式
            if self._is_dialogue_node(node_name):
                # 对话条目：直接使用 key_info，禁止 OCR
                node_key_info = current_node.get("key_info", []) if current_node else []
            else:
                # 事件节点：优先从 dialogue.json 读取，找不到才回退到 key_info
                if node_name in self.dialogue_lookup:
                    dialogue_text = self.dialogue_lookup[node_name]
                    node_key_info = [dialogue_text]
                    # 缓存到节点对象，避免重复查找
                    if current_node:
                        current_node["key_info"] = node_key_info
                    print(f"[Dust] 事件 '{node_name}' 从 dialogue.json 读取文本")
                else:
                    # 回退：使用 key_info 中已有的文本，禁止触发 OCR
                    node_key_info = current_node.get("key_info", []) if current_node else []
                    print(f"[Dust] 事件 '{node_name}' 未在 dialogue.json 中找到，使用 key_info 回退")

            # 将节点标记为已读
            if node_name and node_name not in self.read_events:
                self.read_events.add(node_name)
                print(f"[Dust] 节点 '{node_name}' 已添加到已读事件列表")

        # 构建选择项（Dust 游戏的动作空间）
        if self.config.test_language == "en":
            choices = [
                Choice(index=0, text="Use keyword to unlock event"),
                Choice(index=1, text="Read event"),
                Choice(index=2, text="Submit character event ordering"),
                Choice(index=3, text="Use key to unlock yellow-lock event"),
                Choice(index=4, text="Answer question to unlock pink/purple lock event"),
            ]
        else:
            choices = [
                Choice(index=0, text="选择关键词解锁事件"),
                Choice(index=1, text="阅读事件"),
                Choice(index=2, text="提交人物事件排序"),
                Choice(index=3, text="用钥匙解锁黄色锁事件"),
                Choice(index=4, text="回答问题解锁粉色/紫色锁事件"),
            ]

        memory = Memory(
            description="Dust 推理游戏状态",
            key_info=node_key_info
        )

        # 检查是否游戏结束: 得分达到44分（全部排序正确是44分，给他一些宽裕
        ENDING_SCORE_THRESHOLD = 40
        is_ending = self.score_points >= ENDING_SCORE_THRESHOLD

        # 将 node_key_info 列表转换为格式化文本
        if isinstance(node_key_info, list):
            text = "\n".join(node_key_info).lstrip('\n')
        else:
            text = str(node_key_info).lstrip('\n')

        # 如果该事件的时间已通过排序得分被揭露，附加到文本开头
        if node_name in self.time_revealed_events:
            event_time = self.dialogue_time_lookup.get(node_name, "")
            if event_time and text:
                if self.config.test_language == "en":
                    text = f"[Event Occurrence Time: {event_time}]\n{text}"
                else:
                    text = f"[事件发生时间: {event_time}]\n{text}"

        # 计算未读事件（event_pool 中排除已读事件）
        unread_events = [e for e in self.event_pool if e not in self.read_events]

        return Observation(
            node_id=self.current_node_id,
            name=node_name,
            text=text,
            choices=choices,
            memory=memory,
            is_ending=is_ending,
            meta={
                "keyword_pool": list(self.keyword_pool),
                "known_events": list(self.known_events),
                "event_pool": unread_events,  # 修改为未读事件
                "read_events": list(self.read_events),
                "locked_events": {k: list(v) for k, v in self.locked_events.items()},
                "score": self.score_points,
                "keys": self.keys,
                "image_urls": image_urls  # 添加图片URL列表
            }
        )

    def choose(self, choice_index: int) -> None:
        """执行选择（保留用于兼容性，Dust 游戏使用专用方法）"""
        raise NotImplementedError("Dust game uses specialized methods for actions")

    def apply_keyword_unlock(self, keyword: str) -> List[str]:
        """根据关键词解锁事件

        Args:
            keyword: 关键词

        Returns:
            新解锁的事件名列表
        """
        # 检查关键词是否在关键词池中
        if keyword not in self.keyword_pool:
            print(f"[Dust] 关键词 '{keyword}' 不在关键词池中")
            return []

        # 从关键词池中删除这个关键词(一个关键词只能用一次)
        self.keyword_pool.remove(keyword)

        # 将关键词添加到已使用列表
        self.used_keywords.add(keyword)

        # 根据 tag_index 查找关联的事件
        related_events = get_events_by_tag(keyword, self.tag_index)

        newly_unlocked = []
        for event_name in related_events:
            if event_name not in self.known_events:
                # 检查锁类型
                lock_data = self._get_lock_info(event_name)
                lock_type = lock_data.get("type", "none")

                # 标准化锁类型
                if "yellow" in lock_type.lower():
                    # 添加到黄色锁池
                    self.locked_events["yellow"].add(event_name)
                elif "pink" in lock_type.lower():
                    # 添加到粉色锁池
                    self.locked_events["pink"].add(event_name)
                elif "purple" in lock_type.lower():
                    # 添加到紫色锁池
                    self.locked_events["purple"].add(event_name)
                else:
                    # 无锁，直接添加到可阅读池
                    self.event_pool.append(event_name)

                # 标记为已知事件
                self.known_events.add(event_name)
                newly_unlocked.append(event_name)

        return newly_unlocked

    def read_event(self, event_name: str) -> Dict[str, Any]:
        """阅读事件，获取完整内容

        Args:
            event_name: 事件名称

        Returns:
            事件完整数据（包含 description, key_info, characters, tags 等）
        """
        if event_name not in self.event_pool:
            # 允许重新阅读已读事件
            if event_name not in self.read_events:
                raise ValueError(f"Event '{event_name}' is not available for reading")

        # 不在这里添加到已读事件，而是在 observe() 中 OCR 执行后添加
        # self.read_events.add(event_name) # 移到 observe() 中

        # 获取事件节点数据
        node = self._get_node_by_name(event_name)

        # 读取后从可阅读事件列表中删除（无论是否为对话节点）
        if event_name in self.event_pool:
            self.event_pool.remove(event_name)
            print(f"[Dust] 事件 '{event_name}' 已从可阅读事件池中移除")

        # 更新当前节点ID，这样下次 observe 时可以获取该节点的信息
        self.current_node_id = event_name

        # 从事件中提取新的关键词（自动添加到关键词池）
        tags = node.get("emphasize", [])
        for tag in tags:
            # 检查关键词是否已被使用过
            if tag not in self.used_keywords:
                self.keyword_pool.add(tag)

        return node

    def apply_orders(self, orders_dict: Dict[str, List[str]]) -> Dict[str, Any]:
        """应用角色事件排序并计分

        Args:
            orders_dict: 角色名 -> 事件序列的映射

        Returns:
            包含判定结果、新增分数、新增钥匙的字典
        """
        # 调用 scoring.py 进行批量判定
        judgements, new_points, updated_awarded_pairs = judge_character_orders(
            orders_dict,
            self.order_gt,
            self.awarded_pairs
        )

        # 更新状态
        self.character_orders.update(orders_dict)

        # 过滤重复：ignored 直接跳过（awarded_pairs 已追踪），其他对只记录首次提交
        for j in judgements:
            if j["result"] == "ignored":
                continue
            pair_key = (j["character"], j["earlier"], j["later"])
            if pair_key not in self.submitted_pairs:
                self.order_judgements.append(j)
                self.submitted_pairs.add(pair_key)

        self.awarded_pairs = updated_awarded_pairs

        # 更新得分
        old_score = self.score_points
        self.score_points += new_points

        # 计算新增钥匙
        old_keys = calculate_keys_earned(old_score, self.config.key_threshold)
        new_keys = calculate_keys_earned(self.score_points, self.config.key_threshold)
        keys_earned = new_keys - old_keys
        self.keys += keys_earned

        return {
            "judgements": judgements,
            "new_points": new_points,
            "keys_earned": keys_earned,
            "total_score": self.score_points,
            "total_keys": self.keys
        }

    def unlock_by_key(self, event_name: str) -> bool:
        """用钥匙解锁黄色锁事件

        Args:
            event_name: 事件名称

        Returns:
            True 表示解锁成功，False 表示解锁失败
        """
        # 检查是否是黄色锁
        if event_name not in self.locked_events["yellow"]:
            return False

        # 检查钥匙数量
        if self.keys <= 0:
            return False

        # 消耗钥匙
        self.keys -= 1

        # 从锁池移除，添加到可阅读池
        self.locked_events["yellow"].remove(event_name)
        self.event_pool.append(event_name)

        return True

    def answer_lock(self, event_name: str, answer: str) -> Dict[str, Any]:
        """通过回答问题解锁粉色/紫色锁事件

        对于普通锁（answer 为字符串），与原逻辑相同，返回包含 unlocked 等信息的字典。
        对于多事件锁（answer 为列表），每次提交一个事件名，追踪进度，全部正确后解锁。

        Args:
            event_name: 事件名称
            answer: 玩家提供的答案（普通锁为答案字符串，多事件锁为单个事件名）

        Returns:
            包含以下字段的字典：
            - unlocked (bool): 是否完成解锁
            - is_multi (bool): 是否为多事件锁
            - answer_correct (bool): 本次提交的答案是否正确
            - already_submitted (bool): 本次答案是否已经正确提交过（多事件锁专用）
            - progress (int): 已正确提交的数量（多事件锁专用）
            - total (int): 总共需要的数量（多事件锁专用）
        """
        # 检查是否在锁池中
        lock_type = None
        for lt in ["pink", "purple"]:
            if event_name in self.locked_events[lt]:
                lock_type = lt
                break

        if lock_type is None:
            return {"unlocked": False, "is_multi": False, "answer_correct": False,
                    "already_submitted": False, "progress": 0, "total": 0}

        # 从 lock_info 获取正确答案
        lock_data = self._get_lock_info(event_name)
        correct_answer = lock_data.get("answer", "")

        if isinstance(correct_answer, list):
            # ——— 多事件锁：每次提交一个事件名 ———
            if event_name not in self.multi_lock_progress:
                self.multi_lock_progress[event_name] = set()
            progress_set = self.multi_lock_progress[event_name]
            total = len(correct_answer)
            normalized_correct = [a.strip() for a in correct_answer]
            normalized_input = answer.strip()

            # 检查是否已经正确提交过
            if normalized_input in progress_set:
                return {"unlocked": False, "is_multi": True, "answer_correct": True,
                        "already_submitted": True, "progress": len(progress_set), "total": total}

            if normalized_input in normalized_correct:
                progress_set.add(normalized_input)
                if len(progress_set) == total:
                    # 全部正确，正式解锁
                    self.locked_events[lock_type].remove(event_name)
                    self.event_pool.append(event_name)
                    return {"unlocked": True, "is_multi": True, "answer_correct": True,
                            "already_submitted": False, "progress": total, "total": total}
                else:
                    return {"unlocked": False, "is_multi": True, "answer_correct": True,
                            "already_submitted": False, "progress": len(progress_set), "total": total}
            else:
                return {"unlocked": False, "is_multi": True, "answer_correct": False,
                        "already_submitted": False, "progress": len(progress_set), "total": total}
        else:
            # ——— 普通单答案锁 ———
            is_correct = answer.strip().lower() == correct_answer.strip().lower()
            if is_correct:
                self.locked_events[lock_type].remove(event_name)
                self.event_pool.append(event_name)
            return {"unlocked": is_correct, "is_multi": False, "answer_correct": is_correct,
                    "already_submitted": False, "progress": 0, "total": 0}

    def reset(self) -> None:
        """重置环境"""
        self.current_node_id = self.config.start_node_id

        # 重置游戏状态
        self.keyword_pool.clear()
        self.used_keywords.clear()
        self.known_events.clear()
        self.event_pool.clear()
        self.read_events.clear()
        self.locked_events = {
            "pink": set(),
            "purple": set(),
            "yellow": set()
        }

        self.character_orders.clear()
        self.order_judgements.clear()
        self.score_points = 0
        self.keys = 0
        self.awarded_pairs.clear()
        self.submitted_pairs.clear()
        self.time_revealed_events.clear()
        self.multi_lock_progress.clear()

        # 重新初始化起始事件
        self._initialize_start_events()

  
    def get_state(self) -> Dict[str, Any]:
        """获取环境状态用于 checkpoint

        Returns:
            包含环境完整状态的字典
        """
        # 保存已 OCR 的节点的 key_info（避免重新 OCR）
        ocr_cache = {}
        for event_name in self.read_events:
            node = self._get_node_by_name(event_name)
            if node and "key_info" in node:
                ocr_cache[event_name] = node["key_info"]

        return {
            "checkpoint_version": 2,  # 添加版本号，版本 2 包含 OCR 缓存
            "current_node_id": self.current_node_id,
            "keyword_pool": list(self.keyword_pool),
            "used_keywords": list(self.used_keywords),
            "known_events": list(self.known_events),
            "event_pool": self.event_pool.copy(),
            "read_events": list(self.read_events),
            "locked_events": {k: list(v) for k, v in self.locked_events.items()},
            "character_orders": self.character_orders.copy(),
            "order_judgements": self.order_judgements.copy(),
            "score_points": self.score_points,
            "keys": self.keys,
            "awarded_pairs": [list(pair) for pair in self.awarded_pairs],
            "submitted_pairs": [list(pair) for pair in self.submitted_pairs],
            "time_revealed_events": list(self.time_revealed_events),
            "multi_lock_progress": {k: list(v) for k, v in self.multi_lock_progress.items()},
            "ocr_cache": ocr_cache  # 保存 OCR 缓存
        }

    def restore_state(self, state: Dict[str, Any]) -> None:
        """从 checkpoint 恢复环境状态

        Args:
            state: 环境状态字典

        Raises:
            ValueError: 如果 checkpoint 版本不兼容
        """
        # 检查 checkpoint 版本
        checkpoint_version = state.get("checkpoint_version", 1)
        if checkpoint_version < 2:
            raise ValueError(
                f"不兼容的 checkpoint 版本 {checkpoint_version}。"
                f"此版本需要 checkpoint 版本 >= 2（包含 OCR 缓存）。"
                f"请重新运行游戏生成新的 checkpoint。"
            )

        self.current_node_id = state.get("current_node_id", self.config.start_node_id)
        self.keyword_pool = set(state.get("keyword_pool", []))
        self.used_keywords = set(state.get("used_keywords", []))
        self.known_events = set(state.get("known_events", []))
        self.event_pool = state.get("event_pool", []).copy() if isinstance(state.get("event_pool"), list) else []
        self.read_events = set(state.get("read_events", []))

        # 恢复锁定事件
        locked_events_data = state.get("locked_events", {})
        self.locked_events = {
            "pink": set(locked_events_data.get("pink", [])),
            "purple": set(locked_events_data.get("purple", [])),
            "yellow": set(locked_events_data.get("yellow", []))
        }

        self.character_orders = state.get("character_orders", {}).copy()
        self.order_judgements = state.get("order_judgements", []).copy()
        self.score_points = state.get("score_points", 0)
        self.keys = state.get("keys", 0)

        # 恢复 OCR 缓存到 nodes 对象中
        ocr_cache = state.get("ocr_cache", {})
        for event_name, key_info in ocr_cache.items():
            node = self._get_node_by_name(event_name)
            if node:
                node["key_info"] = key_info
                print(f"[Checkpoint] 恢复事件 '{event_name}' 的文本缓存")

        # 恢复已计分事件对
        awarded_pairs_data = state.get("awarded_pairs", [])
        self.awarded_pairs = set(tuple(pair) for pair in awarded_pairs_data)

        # 恢复已提交事件对
        submitted_pairs_data = state.get("submitted_pairs", [])
        self.submitted_pairs = set(tuple(pair) for pair in submitted_pairs_data)
        # 兼容旧 checkpoint：从 order_judgements 重建 submitted_pairs
        if not self.submitted_pairs and self.order_judgements:
            for j in self.order_judgements:
                if j.get("result") != "ignored":
                    self.submitted_pairs.add((j["character"], j["earlier"], j["later"]))

        # 恢复已揭露时间的事件集合
        self.time_revealed_events = set(state.get("time_revealed_events", []))

        # 恢复多事件锁进度
        self.multi_lock_progress = {
            k: set(v) for k, v in state.get("multi_lock_progress", {}).items()
        }

        print(f"[DustEnv] 已恢复状态: 得分={self.score_points}, 钥匙={self.keys}, "
              f"已知事件={len(self.known_events)}个")

    def file_to_data_url(path: Path) -> str:
        ext = path.suffix.lower()
        mime = "image/png" if ext == ".png" else "image/jpeg"
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        return f"data:{mime};base64,{b64}"