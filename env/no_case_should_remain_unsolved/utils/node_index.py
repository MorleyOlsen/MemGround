# env/no_case_should_remain_unsolved/utils/node_index.py
"""Node index building utilities for the No Case Should Remain Unsolved game"""
from typing import Dict, List, Set


def build_node_indices(nodes: List[Dict]) -> tuple[
    List[Dict],  # nodes: node list [{"name": "A", "sub_name":"A", "emphasize": ["x","y"], "type": "none", "auto_link": "talk-4", "key_info": [...]}]
    Dict[str, List[str]],  # tag_index: tag -> event_names (records which dialogues a tag can open)
    List[Dict],  # lock_info: lock info list [{"name": "A", "type": "none", "question": "...", "answer": "..."}]
]:
    """Build index structures from nodes.json

    Args:
        nodes: Raw node list

    Returns:
        (nodes, tag_index, lock_info)
    """
    nodes_list = []
    tag_index: Dict[str, List[str]] = {}
    lock_info = []

    for node in nodes:
        node_name = node.get("name")
        sub_name = node.get("sub_name") or None
        if not node_name:
            continue

        # Extract key_info from memory
        memory = node.get("memory", {})
        key_info = memory.get("key_info", [])

        # Build node structure (including key_info)
        node_entry = {
            "name": node_name,
            "sub_name": sub_name,
            "emphasize": node.get("emphasize", []),
            "type": node.get("type", "none"),
            "key_info": key_info
        }

        # Extract auto_link (only set when a link's condition is "auto")
        links = node.get("links", [])
        node_entry["auto_link"] = ""
        for link in links:
            if link.get("condition", "") == "auto":
                node_entry["auto_link"] = link.get("target", "")
                break

        nodes_list.append(node_entry)

        # Build tag index; store sub_name of the event associated with each tag (skip invalid sub_names)
        tags = node.get("tag", [])
        for tag in tags:
            if tag not in tag_index:
                tag_index[tag] = []
            if sub_name and sub_name not in tag_index[tag]:
                tag_index[tag].append(sub_name)

        # Build lock info
        lock_type = node.get("type", "none")
        if lock_type != "none":
            lock_entry = {
                "name": node_name,
                "sub_name": sub_name,
                "type": lock_type,
                "question": node.get("question", ""),
                "answer": node.get("answer", "")
            }
            lock_info.append(lock_entry)

    # Debug output: display partial data
    # print("\n=== build_node_indices debug output ===")
    # print(f"\nTotal nodes: {len(nodes_list)}")
    # print(f"Total tags: {len(tag_index)}")
    # print(f"Total lock entries: {len(lock_info)}")

    # # Output details for first 3 nodes
    # print("\n--- First 3 nodes (nodes_list) ---")
    # for i, node in enumerate(nodes_list[:3]):
    #     print(f"\nNode {i+1}:")
    #     print(f"  name: {node.get('name')}")
    #     print(f"  sub_name: {node.get('sub_name')}")
    #     print(f"  emphasize: {node.get('emphasize')}")
    #     print(f"  type: {node.get('type')}")
    #     print(f"  auto_link: {node.get('auto_link')}")
    #     print(f"  key_info: {node.get('key_info')[:100] if node.get('key_info') else '[]'}...")  # Show first 100 chars only

    # # Output first 5 tags and their associated events
    # print("\n--- First 5 tags (tag_index) ---")
    # for i, (tag, events) in enumerate(list(tag_index.items())[:5]):
    #     print(f"\nTag {i+1}: '{tag}'")
    #     print(f"  Associated events: {events}")

    # # Output first 3 lock entries
    # print("\n--- First 3 lock entries (lock_info) ---")
    # for i, lock in enumerate(lock_info[:3]):
    #     print(f"\nLock {i+1}:")
    #     print(f"  name: {lock.get('name')}")
    #     print(f"  sub_name: {lock.get('sub_name')}")
    #     print(f"  type: {lock.get('type')}")
    #     print(f"  question: {lock.get('question')[:50] if lock.get('question') else ''}...")  # Show first 50 chars only
    #     print(f"  answer: {lock.get('answer')}")

    # print("\n=== Debug output end ===\n")

    return nodes_list, tag_index, lock_info


def get_events_by_tag(tag: str, tag_index: Dict[str, List[str]]) -> List[str]:
    """Get the list of event names associated with a tag

    Args:
        tag: Keyword/tag
        tag_index: Mapping of tag -> event_names

    Returns:
        List of event names
    """
    return tag_index.get(tag, [])


def get_node_tags(node_name: str, nodes_by_name: Dict[str, Dict]) -> List[str]:
    """Get all tags for a node

    Args:
        node_name: Node name
        nodes_by_name: Mapping of node name to node data

    Returns:
        List of tags
    """
    node = nodes_by_name.get(node_name, {})
    return node.get("tag", [])  # Field name is "tag"


def get_node_characters(node_name: str, nodes_by_name: Dict[str, Dict]) -> List[str]:
    """Get the list of characters associated with a node

    Args:
        node_name: Node name
        nodes_by_name: Mapping of node name to node data

    Returns:
        List of character names
    """
    node = nodes_by_name.get(node_name, {})
    memory = node.get("memory", {})
    characters = memory.get("characters", [])
    # Extract tag field from characters as character identifiers
    return [char.get("tag", "") for char in characters if char.get("tag")]
