# evaluate.py
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Set, Optional


# ----------------------------
# IO
# ----------------------------
def load_edges(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")

    text = p.read_text(encoding="utf-8").strip()
    if not text:
        return []

    # Heuristic: jsonl if multiple lines and each line starts with '{'
    lines = text.splitlines()
    if len(lines) > 1 and all(l.strip().startswith("{") for l in lines if l.strip()):
        edges = []
        for ln in lines:
            ln = ln.strip()
            if not ln:
                continue
            edges.append(json.loads(ln))
        return edges

    # Otherwise: json (list or dict)
    obj = json.loads(text)
    if isinstance(obj, list):
        return obj
    # If wrapped dict, try common keys
    for key in ("edges", "links", "data", "items"):
        if key in obj and isinstance(obj[key], list):
            return obj[key]
    raise ValueError("Unsupported GT/Pred JSON format: expect list or jsonl.")


# ----------------------------
# Normalization helpers
# ----------------------------
def _recall_item_to_key(x: Any) -> Optional[str]:
    """
    Convert a recall item to a comparable string key.
    Supports:
      - str
      - dict with 'name' / 'id' / 'filename' / 'file' / 'choice_text'
    """
    if x is None:
        return None
    if isinstance(x, str):
        s = x.strip()
        return s if s else None
    if isinstance(x, dict):
        for k in ("name", "id", "filename", "file", "choice_text", "target"):
            v = x.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
    # fallback: stringified (avoid losing info, but may be noisy)
    try:
        s = str(x).strip()
        return s if s else None
    except Exception:
        return None


def normalize_recall_list(recall: Any) -> Set[str]:
    if recall is None:
        return set()
    if not isinstance(recall, list):
        recall = [recall]
    out: Set[str] = set()
    for item in recall:
        key = _recall_item_to_key(item)
        if key is not None:
            out.add(key)
    return out


def edges_to_node_sequence(edges: List[Dict[str, Any]]) -> List[str]:
    """
    Node sequence: Start -> target1 -> target2 -> ...
    If first edge isn't Start->X, we still start with 'Start' for evaluation.
    """
    seq = ["Start"]
    for e in edges:
        tgt = e.get("target")
        if isinstance(tgt, str) and tgt.strip():
            seq.append(tgt.strip())
    return seq


def edges_to_edge_pairs(edges: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
    pairs = []
    for e in edges:
        f = e.get("from")
        t = e.get("target")
        if isinstance(f, str) and isinstance(t, str) and f.strip() and t.strip():
            pairs.append((f.strip(), t.strip()))
    return pairs


def sum_wasted_attempts(edges: List[Dict[str, Any]]) -> int:
    total = 0
    for e in edges:
        w = e.get("wasted_attempts", 0)
        try:
            total += int(w)
        except Exception:
            pass
    return total


# ----------------------------
# Metrics
# ----------------------------
def matched_prefix_len(seq_pred: List[str], seq_gt: List[str]) -> int:
    """
    通关进度=
    Returns number of matched steps in terms of edges = matched_nodes - 1
    """
    m = 0
    for a, b in zip(seq_pred, seq_gt):
        if a != b:
            break
        m += 1
    # m nodes matched -> m-1 edges matched
    return max(0, m - 1)


def levenshtein(a: List[str], b: List[str]) -> int:
    # classic DP edit distance
    n, m = len(a), len(b)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            tmp = dp[j]
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[j] = min(
                dp[j] + 1,        # delete
                dp[j - 1] + 1,    # insert
                prev + cost       # replace
            )
            prev = tmp
    return dp[m]


def jaccard(set_a: Set[Any], set_b: Set[Any]) -> float:
    if not set_a and not set_b:
        return 1.0
    inter = len(set_a & set_b)
    uni = len(set_a | set_b)
    return inter / uni if uni else 0.0


def eval_recall_on_prefix(
    edges_pred: List[Dict[str, Any]],
    edges_gt: List[Dict[str, Any]],
    prefix_edges: int
) -> Dict[str, Any]:
    """
    Evaluate recall quality for the first prefix_edges transitions.
    Alignment: edge-by-edge (position-based) on matched prefix only.
    """
    per_step = []
    sum_tp = sum_fp = sum_fn = 0
    sum_prec = sum_rec = sum_f1 = 0.0
    count = 0

    for i in range(prefix_edges):
        ep = edges_pred[i]
        eg = edges_gt[i]

        P = normalize_recall_list(ep.get("recall"))
        G = normalize_recall_list(eg.get("recall"))

        tp = len(P & G)
        fp = len(P - G)  # redundant
        fn = len(G - P)  # missing

        prec = tp / (tp + fp) if (tp + fp) else 1.0
        rec = tp / (tp + fn) if (tp + fn) else 1.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 1.0

        per_step.append({
            "step": i,
            "from": ep.get("from"),
            "target": ep.get("target"),
            "tp": tp,
            "fp_redundant": fp,
            "fn_missing": fn,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "redundant_items": sorted(list(P - G)),
            "missing_items": sorted(list(G - P)),
        })

        sum_tp += tp
        sum_fp += fp
        sum_fn += fn
        sum_prec += prec
        sum_rec += rec
        sum_f1 += f1
        count += 1

    macro = {
        "avg_precision": (sum_prec / count) if count else 1.0,
        "avg_recall": (sum_rec / count) if count else 1.0,
        "avg_f1": (sum_f1 / count) if count else 1.0,
    }

    # micro (set-based aggregation)
    micro_prec = sum_tp / (sum_tp + sum_fp) if (sum_tp + sum_fp) else 1.0
    micro_rec = sum_tp / (sum_tp + sum_fn) if (sum_tp + sum_fn) else 1.0
    micro_f1 = (2 * micro_prec * micro_rec / (micro_prec + micro_rec)) if (micro_prec + micro_rec) else 1.0

    totals = {
        "tp_total": sum_tp,
        "fp_redundant_total": sum_fp,
        "fn_missing_total": sum_fn,
        "micro_precision": micro_prec,
        "micro_recall": micro_rec,
        "micro_f1": micro_f1,
        "redundancy_rate": (sum_fp / (sum_tp + sum_fp)) if (sum_tp + sum_fp) else 0.0,
        "missing_rate": (sum_fn / (sum_tp + sum_fn)) if (sum_tp + sum_fn) else 0.0,
    }

    return {
        "macro": macro,
        "micro": totals,
        "per_step": per_step,
    }


def evaluate(pred_path: str, gt_path: str) -> Dict[str, Any]:
    edges_pred = load_edges(pred_path)
    edges_gt = load_edges(gt_path)

    seq_pred = edges_to_node_sequence(edges_pred)
    seq_gt = edges_to_node_sequence(edges_gt)
    # 先记录模型的通关情况，然后根据通关进度生成多文档QA（基于前序节点和当前节点的综合问题/同一批次下的综合问题/综合性强的节点作为milestone） 
    # 1. Progress 通关进度
    # 1.1 路径匹配程度 通过前缀是否匹配来判断llm的路径和人类攻略的路径重合程度 <存在问题:我通过log输出的得到的大模型运行结果是
    # 线性的，但是gt是个图的类型，无法完全匹配>
    # 烧烤了一下，way1：人为手动把log输出情况转换成类gt的图形式，也就是结合每个新节点的in_degree字段，如果这个节点解锁时，in_degree字段都已解锁，就统一成gt里的路径
    # way2：对于gt里面同一个time_id的作为一个批次，然后看这个批次的集合是否被全部解锁，如果全解锁就认为llm这个批次的路径重合√
    gt_steps = max(1, len(seq_gt) - 1)
    prefix_edges = matched_prefix_len(seq_pred, seq_gt)
    progress_prefix = prefix_edges / gt_steps
    # 1.2 节点覆盖率 不要求顺序，只是看解锁的节点占总节点数的比例√
    V_pred = set(seq_pred)
    V_gt = set(seq_gt)
    progress_coverage = (len(V_pred & V_gt) / len(V_gt)) if V_gt else 1.0

    # 2.Errors 错误/无用尝试次数 无用尝试次数占总步数的比例
    errors_pred = sum_wasted_attempts(edges_pred)
    success_steps_pred = len(seq_pred) - 1
    error_ratio = errors_pred / max(1, errors_pred + success_steps_pred)

    # 3.Recall 回忆准确度 对于和gt匹配上的路径，比较recall部分的重合比例 <依旧存在上述问题，并且我现在为了限制长度只让最多回忆3个，gt里最多回忆了16个吓晕了>
    # 让模型输出一下跟哪些节点有关系 有效的/总的
    recall_report = eval_recall_on_prefix(edges_pred, edges_gt, prefix_edges)

    # 4. Path difference 
    edit = levenshtein(seq_pred, seq_gt)
    path_edit_norm = edit / max(1, len(seq_gt))

    E_pred = set(edges_to_edge_pairs(edges_pred))
    E_gt = set(edges_to_edge_pairs(edges_gt))
    edges_jacc = jaccard(E_pred, E_gt)

    return {
        "paths": {
            "pred_nodes": seq_pred,
            "gt_nodes": seq_gt,
            "pred_steps": len(seq_pred) - 1,
            "gt_steps": len(seq_gt) - 1,
            "matched_prefix_edges": prefix_edges,
        },
        "metric_1_progress": {
            "progress_prefix": progress_prefix,
            "progress_coverage": progress_coverage,
        },
        "metric_2_errors": {
            "errors_pred": errors_pred,
            "error_ratio": error_ratio,
        },
        "metric_3_recall_quality": recall_report,
        "metric_4_path_difference": {
            "levenshtein_distance": edit,
            "path_edit_norm": path_edit_norm,
            "edges_jaccard": edges_jacc,
        },
    }


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True, help="pred edges (jsonl or json)")
    parser.add_argument("--gt", required=True, help="ground truth edges (json or jsonl)")
    parser.add_argument("--out", default=None, help="output report json path")
    args = parser.parse_args()

    report = evaluate(args.pred, args.gt)

    if args.out:
        Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[OK] report saved to {args.out}")
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
