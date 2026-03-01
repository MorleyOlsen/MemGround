# env/dust/utils/scoring.py
"""Dust 游戏的排序判定和计分系统"""
from typing import Dict, List, Set, Tuple


def judge_character_orders(
    character_orders: Dict[str, List[str]],  # 模型输出的排序：character -> [event_name, ...]
    order_gt: List[Dict],  # 角色顺序 ground truth
    awarded_pairs: Set[Tuple[str, str, str]],  # 已计分的 (character, earlier, later) 对
) -> Tuple[List[Dict], int, Set[Tuple[str, str, str]]]:
    """批量判定角色事件排序的正确性并计分

    Args:
        character_orders: 模型输出的每个角色的事件排序
        order_gt: 角色顺序 ground truth 列表
        awarded_pairs: 已经计分的事件对集合（用于防重复计分）
                      对于单事件角色，使用 (character, event, event) 格式标记

    Returns:
        (order_judgements, new_points, updated_awarded_pairs)
        - order_judgements: 判定结果列表，每个元素包含 {character, earlier, later, result}
        - new_points: 新增得分
        - updated_awarded_pairs: 更新后的已计分事件对集合
    """
    order_judgements = []
    new_points = 0
    current_awarded_pairs = awarded_pairs.copy()

    # 使用 build_trace_structure 构建 trace 数据结构
    trace = build_trace_structure(character_orders, order_gt)

    for character, data in trace.items():
        events = data["events"]
        gt_numbers = data["gt_numbers"]

        # 特判：只有"伊甸幼儿园"角色且只提交了一个事件时才计分
        if character == "伊甸幼儿园" and len(events) == 1:
            event = events[0]
            gt_num = gt_numbers[0]
            single_key = (character, event, event)

            # 检查是否已经计分过
            if single_key in current_awarded_pairs:
                order_judgements.append({
                    "character": character,
                    "earlier": event,
                    "later": event,
                    "result": "ignored"
                })
                continue

            # 检查该事件是否属于该角色（gt_num != -1）
            if gt_num != -1:
                # 事件属于该角色，给予得分
                order_judgements.append({
                    "character": character,
                    "earlier": event,
                    "later": event,
                    "result": "correct"
                })
                new_points += 1
                current_awarded_pairs.add(single_key)
            else:
                # 事件不属于该角色
                order_judgements.append({
                    "character": character,
                    "earlier": event,
                    "later": event,
                    "result": "incorrect"
                })
            continue

        # 其他角色只有一个事件时，不计分（跳过）
        if len(events) == 1:
            continue

        # 对每个角色的事件序列，判定所有相邻事件对
        for i in range(len(events) - 1):
            earlier_event = events[i]
            later_event = events[i + 1]
            earlier_num = gt_numbers[i]
            later_num = gt_numbers[i + 1]

            # 检查是否已经计分过
            pair_key = (character, earlier_event, later_event)
            if pair_key in current_awarded_pairs:
                # 已经计分过，标记为 ignored
                order_judgements.append({
                    "character": character,
                    "earlier": earlier_event,
                    "later": later_event,
                    "result": "ignored"
                })
                continue

            # 判定排序正确性
            if earlier_num == -1 or later_num == -1:
                result = "unknown"  # 存在事件不属于该角色，无法判断
            elif later_num - earlier_num == 1:
                result = "correct"  # 顺序正确且连续
            elif later_num > earlier_num:
                result = "correct_not_consecutive"  # 顺序正确但不连续
            else:
                result = "incorrect"  # 顺序错误

            order_judgements.append({
                "character": character,
                "earlier": earlier_event,
                "later": later_event,
                "result": result
            })

            # 如果正确且连续，计分并记录
            if result == "correct":
                new_points += 1
                current_awarded_pairs.add(pair_key)

    return order_judgements, new_points, current_awarded_pairs


def calculate_keys_earned(score: int, key_threshold: int) -> int:
    """根据得分计算可获得的钥匙数量

    Args:
        score: 当前得分
        key_threshold: 获得一把钥匙所需的分数

    Returns:
        应获得的钥匙总数
    """
    if key_threshold <= 0:
        return 0
    return score // key_threshold


def can_unlock_with_key(event_name: str, keys: int, lock_type_map: Dict[str, str]) -> bool:
    """检查是否可以用钥匙解锁指定事件

    Args:
        event_name: 事件名称
        keys: 当前拥有的钥匙数量
        lock_type_map: 事件名到锁类型的映射

    Returns:
        True 表示可以解锁（黄色锁且有钥匙），False 表示不能解锁
    """
    lock_type = lock_type_map.get(event_name, "none")
    return lock_type == "yellow" and keys > 0


def build_trace_structure(
    character_orders: Dict[str, List[str]],
    order_gt: List[Dict]
) -> Dict[str, Dict[str, List]]:
    """构建事件追踪数据结构，记录每个事件是否属于角色及其在GT中的位置

    Args:
        character_orders: 模型输出的每个角色的事件排序 {角色名: [事件1, 事件2, ...]}
        order_gt: 角色顺序 ground truth 列表

    Returns:
        追踪数据结构，格式为：
        {
            "角色名": {
                "events": ["事件1", "事件2", ...],
                "gt_numbers": [number1, number2, ...]  # -1表示不属于该角色
            }
        }
    """
    # 构建角色名到角色数据的映射
    character_gt_map = {}
    for char_data in order_gt:
        char_name = char_data.get("name", "")
        if char_name:
            character_gt_map[char_name] = char_data

    trace = {}

    for character, event_sequence in character_orders.items():
        # 获取该角色的 ground truth 数据
        char_data = character_gt_map.get(character)

        # 构建事件名到顺序编号的映射
        event_order_map = {}
        if char_data:
            dialogues = char_data.get("dialogue", [])
            for dialogue in dialogues:
                event_name = dialogue.get("name", "")
                event_number = dialogue.get("number")
                if event_name and event_number is not None:
                    event_order_map[event_name] = event_number

        # 为每个事件获取其在该角色GT中的编号（不属于则为-1）
        gt_numbers = []
        for event_name in event_sequence:
            gt_number = event_order_map.get(event_name, -1)
            gt_numbers.append(gt_number)

        trace[character] = {
            "events": event_sequence,
            "gt_numbers": gt_numbers
        }

    return trace


def describe_trace(trace: Dict[str, Dict[str, List]]) -> str:
    """将追踪数据结构翻译成可读文本

    Args:
        trace: build_trace_structure 返回的追踪数据结构

    Returns:
        格式化的文本描述
    """
    lines = []

    for character, data in trace.items():
        events = data["events"]
        nums = data["gt_numbers"]

        lines.append(f"{character}:")

        # 列出每个事件及其编号
        for ev, n in zip(events, nums):
            if n == -1:
                lines.append(f"  {ev}: 不属于该角色")
            else:
                lines.append(f"  {ev}: 该事件属于该角色") # 如果把n输出出来就可以告诉模型这是第几个事件

        # 判断相邻关系
        for i in range(len(nums) - 1):
            a, b = nums[i], nums[i+1]

            if a ==-1 or b == -1:
                status = "无法判断，存在事件不属于该角色"
            elif b - a == 1:
                status = "顺序正确且连续"
            elif b > a:
                status = "顺序正确但不连续"
            else:
                status = "顺序错误"

            lines.append(f"  {events[i]} -> {events[i+1]} : {status}")

        lines.append("")  # 空行分隔不同角色

    return "\n".join(lines)
