# env/dust/utils/node_index.py
"""Dust 游戏的节点索引构建工具"""
from typing import Dict, List, Set


def build_node_indices(nodes: List[Dict]) -> tuple[
    List[Dict],  # nodes: 节点列表 [{"name": "A", "sub_name":"A", emphasize": ["x","y"], "type": "none", "auto_link": "对话-4", "key_info": [...]}]
    Dict[str, List[str]],  # tag_index: tag -> event_names (记录当前tag能够打开哪些对话)
    List[Dict],  # lock_info: 锁信息列表 [{"name": "A", "type": "none", "question": "...", "answer": "..."}]
]:
    """从 nodes.json 构建索引结构

    Args:
        nodes: 原始节点列表

    Returns:
        (nodes, tag_index, lock_info)
    """
    nodes_list = []
    tag_index: Dict[str, List[str]] = {}
    lock_info = []

    for node in nodes:
        node_name = node.get("name")
        sub_name = node.get("sub_name", "none")
        if not node_name:
            continue

        # 提取 memory 下的 key_info
        memory = node.get("memory", {})
        key_info = memory.get("key_info", [])

        # 构建节点结构 (包含 key_info)
        node_entry = {
            "name": node_name,
            "sub_name":sub_name,
            "emphasize": node.get("emphasize", []),
            "type": node.get("type", "none"),
            "key_info": key_info
        }

        # 提取 auto_link (只有当 links 的 condition 为 "auto" 时才设置)
        links = node.get("links", [])
        node_entry["auto_link"] = ""
        for link in links:
            if link.get("condition", "") == "auto":
                node_entry["auto_link"] = link.get("target", "")
                break

        nodes_list.append(node_entry)

        # 构建 tag 索引，存入tag对应事件的sub_name
        tags = node.get("tag", [])
        for tag in tags:
            if tag not in tag_index:
                tag_index[tag] = []
            if sub_name not in tag_index[tag]:
                tag_index[tag].append(sub_name)

        # 构建锁信息
        lock_type = node.get("type", "none")
        if lock_type != "none":
            lock_entry = {
                "name": node_name,
                "sub_name":sub_name,
                "type": lock_type,
                "question": node.get("question", ""),
                "answer": node.get("answer", "")
            }
            lock_info.append(lock_entry)

    # 调试输出：显示部分数据
    # print("\n=== build_node_indices 调试输出 ===")
    # print(f"\n总节点数: {len(nodes_list)}")
    # print(f"总标签数: {len(tag_index)}")
    # print(f"总锁信息数: {len(lock_info)}")

    # # 输出前3个节点的详细信息
    # print("\n--- 前3个节点 (nodes_list) ---")
    # for i, node in enumerate(nodes_list[:3]):
    #     print(f"\n节点 {i+1}:")
    #     print(f"  name: {node.get('name')}")
    #     print(f"  sub_name: {node.get('sub_name')}")
    #     print(f"  emphasize: {node.get('emphasize')}")
    #     print(f"  type: {node.get('type')}")
    #     print(f"  auto_link: {node.get('auto_link')}")
    #     print(f"  key_info: {node.get('key_info')[:100] if node.get('key_info') else '[]'}...")  # 只显示前100字符

    # # 输出前5个标签及其关联的事件
    # print("\n--- 前5个标签 (tag_index) ---")
    # for i, (tag, events) in enumerate(list(tag_index.items())[:5]):
    #     print(f"\n标签 {i+1}: '{tag}'")
    #     print(f"  关联事件: {events}")

    # # 输出前3个锁信息
    # print("\n--- 前3个锁信息 (lock_info) ---")
    # for i, lock in enumerate(lock_info[:3]):
    #     print(f"\n锁 {i+1}:")
    #     print(f"  name: {lock.get('name')}")
    #     print(f"  sub_name: {lock.get('sub_name')}")
    #     print(f"  type: {lock.get('type')}")
    #     print(f"  question: {lock.get('question')[:50] if lock.get('question') else ''}...")  # 只显示前50字符
    #     print(f"  answer: {lock.get('answer')}")

    # print("\n=== 调试输出结束 ===\n")

    return nodes_list, tag_index, lock_info


def get_events_by_tag(tag: str, tag_index: Dict[str, List[str]]) -> List[str]:
    """根据 tag 获取关联的事件名列表

    Args:
        tag: 关键词
        tag_index: tag -> event_names 的映射

    Returns:
        事件名列表
    """
    return tag_index.get(tag, [])


def get_node_tags(node_name: str, nodes_by_name: Dict[str, Dict]) -> List[str]:
    """获取节点的所有 tags

    Args:
        node_name: 节点名称
        nodes_by_name: 节点名称到节点的映射

    Returns:
        tags 列表
    """
    node = nodes_by_name.get(node_name, {})
    return node.get("tag", [])  # 字段名改为 "tag"


def get_node_characters(node_name: str, nodes_by_name: Dict[str, Dict]) -> List[str]:
    """获取节点关联的角色列表

    Args:
        node_name: 节点名称
        nodes_by_name: 节点名称到节点的映射

    Returns:
        角色名列表
    """
    node = nodes_by_name.get(node_name, {})
    memory = node.get("memory", {})
    characters = memory.get("characters", [])
    # 从 characters 中提取 tag 字段作为角色标识
    return [char.get("tag", "") for char in characters if char.get("tag")]
