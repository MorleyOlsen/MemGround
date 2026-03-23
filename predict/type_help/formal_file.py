import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def _safe_get_opened_files(action: Dict[str, Any]) -> List[Any]:
    fr = action.get("file_retrieval") or {}
    opened = fr.get("opened_files")
    if opened is None:
        return []
    if isinstance(opened, list):
        return opened
    return [opened]


def build_path_edges_from_log(log_obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    构建路径边，并记录从当前节点到下一个节点之间的无用尝试次数 wasted_attempts。
    - 成功跳转：node_name 发生变化
    - wasted_attempts：两次成功跳转之间 node_name 未变化的 action 数量
    """
    actions = log_obj.get("actions")
    if not isinstance(actions, list) or not actions:
        raise ValueError("log json 中未找到非空的 actions 数组")

    edges: List[Dict[str, Any]] = []

    # 第一条：Start -> first node
    first = actions[0]
    first_node = first.get("node_name")
    if not first_node:
        raise ValueError("actions[0] 缺少 node_name")

    edges.append(
        {
            "from": "Start",
            "target": first_node,
            "recall": _safe_get_opened_files(first),
            "wasted_attempts": 0,  # Start 到第一步不计无用尝试（可按需改）
        }
    )

    prev_node = first_node
    last_success_idx = 0  # 上一次“成功跳转”发生的 action 下标（初始为 0）

    # 扫描后续 actions：遇到 node_name 变化就形成一条边
    for i in range(1, len(actions)):
        cur = actions[i]
        cur_node = cur.get("node_name")
        if not cur_node:
            continue

        if cur_node != prev_node:
            wasted = i - last_success_idx - 1
            edges.append(
                {
                    "from": prev_node,
                    "target": cur_node,
                    "recall": _safe_get_opened_files(cur),
                    "wasted_attempts": max(0, wasted),
                }
            )
            prev_node = cur_node
            last_success_idx = i
    
        # ---- 处理最后未成功跳转的 wasted attempts ----
    tail_wasted = len(actions) - last_success_idx - 1

    if tail_wasted > 0:
        edges.append(
            {
                "from": prev_node,
                "target": "END",
                "recall": [],
                "wasted_attempts": tail_wasted,
            }
        )


    return edges


def main(input_path: str, output_path: Optional[str] = None) -> None:
    in_path = Path(input_path)
    if not in_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {in_path}")

    log_obj = json.loads(in_path.read_text(encoding="utf-8"))
    edges = build_path_edges_from_log(log_obj)

    out_text = "\n".join(json.dumps(e, ensure_ascii=False) for e in edges) + "\n"

    if output_path:
        out_path = Path(output_path)
        out_path.write_text(out_text, encoding="utf-8")
        print(f"[OK] wrote {len(edges)} edges -> {out_path}")
    else:
        print(out_text, end="")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    main(args.input, args.output)
