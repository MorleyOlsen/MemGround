# galagent/env/dust_env.py
"""Dust 推理游戏环境"""
from __future__ import annotations

import json
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

        # 加载角色顺序 ground truth
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
        else:
            print(f"[警告] 未找到 order_gt.json 文件: {order_gt_file}")

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

        # 如果节点已经被读取过，直接使用 key_info，不需要 OCR
        image_urls = []
        if node_name in self.read_events:
            # 已读节点，直接获取 key_info
            node_key_info = current_node.get("key_info", []) if current_node else []
        else:
            # 未读节点，需要查找图片并进行 OCR
            dialogue_dir = self.config.data_path / "dialogue"
            if dialogue_dir.exists():
                # 图片文件名基于 sub_name（即 current_node_id）
                import glob
                pattern = str(dialogue_dir / f"{self.current_node_id}*.png")
                image_files = glob.glob(pattern)
                # 也查找 jpg 格式
                pattern_jpg = str(dialogue_dir / f"{self.current_node_id}*.jpg")
                image_files.extend(glob.glob(pattern_jpg))

                image_urls = [Path(f) for f in sorted(image_files)]

            # 如果有图片，使用 OCR 解析图片内容替换 key_info
            node_key_info = []
            if image_urls:
                try:
                    from env.dust.utils.ocr_utils import request_many

                    # 从 config 获取 OCR APPCODE
                    appcode = getattr(self.config, 'ocr_appcode', '')
                    if appcode:
                        # 将 Path 对象转换为字符串路径
                        image_paths = [str(url) for url in image_urls]
                        # 调用 OCR 解析所有图片
                        ocr_text = request_many(appcode, image_paths)
                        # 将 OCR 结果作为 key_info
                        node_key_info = [ocr_text]
                        
                       
                        # 持久化保存 OCR 结果到节点对象中
                        if current_node:
                            current_node["key_info"] = node_key_info
                    else:
                        print("[Dust] 警告: config 中未设置 ocr_appcode，使用原始 key_info")
                        node_key_info = current_node.get("key_info", []) if current_node else []
                except Exception as e:
                    print(f"[Dust] OCR 解析失败: {e}，使用原始 key_info")
                    node_key_info = current_node.get("key_info", []) if current_node else []
            else:
                # 没有图片，使用原始 key_info
                node_key_info = current_node.get("key_info", []) if current_node else []

            # OCR 执行完成后，将节点添加到已读事件列表
            if node_name and node_name not in self.read_events:
                self.read_events.add(node_name)
                print(f"[Dust] 节点 '{node_name}' 已添加到已读事件列表")

        # 构建选择项（Dust 游戏的动作空间）
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
                "event_pool": self.event_pool.copy(),
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
            raise ValueError(f"Event '{event_name}' is not available for reading")

        # 不在这里添加到已读事件，而是在 observe() 中 OCR 执行后添加
        # self.read_events.add(event_name) # 移到 observe() 中

        # 获取事件节点数据
        node = self._get_node_by_name(event_name)

        # 特殊处理：如果节点的 name 前缀是"对话"，读取后从可阅读事件列表中删除
        node_name = node.get("name", "")
        if node_name.startswith("对话"):
            if event_name in self.event_pool:
                self.event_pool.remove(event_name)
                print(f"[Dust] 对话节点 '{event_name}' 已从可阅读事件池中移除")
            

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
        self.order_judgements.extend(judgements)
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

    def answer_lock(self, event_name: str, answer: str) -> bool:
        """通过回答问题解锁粉色/紫色锁事件

        Args:
            event_name: 事件名称
            answer: 玩家提供的答案

        Returns:
            True 表示解锁成功，False 表示解锁失败
        """
        # 检查是否在锁池中
        lock_type = None
        for lt in ["pink", "purple"]:
            if event_name in self.locked_events[lt]:
                lock_type = lt
                break

        if lock_type is None:
            return False

        # 从 lock_info 获取正确答案
        lock_data = self._get_lock_info(event_name)
        correct_answer = lock_data.get("answer", "")

        # 判断答案是否正确（简单字符串匹配，可根据需求扩展）
        is_correct = answer.strip().lower() == correct_answer.strip().lower()

        if is_correct:
            # 从锁池移除，添加到可阅读池
            self.locked_events[lock_type].remove(event_name)
            self.event_pool.append(event_name)
            return True
        else:
            return False

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

        # 重新初始化起始事件
        self._initialize_start_events()

  
    def get_state(self) -> Dict[str, Any]:
        """获取环境状态用于 checkpoint

        Returns:
            包含环境完整状态的字典
        """
        return {
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
            "awarded_pairs": [list(pair) for pair in self.awarded_pairs]
        }

    def restore_state(self, state: Dict[str, Any]) -> None:
        """从 checkpoint 恢复环境状态

        Args:
            state: 环境状态字典
        """
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

        # 恢复已计分事件对
        awarded_pairs_data = state.get("awarded_pairs", [])
        self.awarded_pairs = set(tuple(pair) for pair in awarded_pairs_data)

        print(f"[DustEnv] 已恢复状态: 得分={self.score_points}, 钥匙={self.keys}, "
              f"已知事件={len(self.known_events)}个")

    def file_to_data_url(path: Path) -> str:
        ext = path.suffix.lower()
        mime = "image/png" if ext == ".png" else "image/jpeg"
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        return f"data:{mime};base64,{b64}"