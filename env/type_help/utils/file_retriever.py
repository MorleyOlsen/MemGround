# env/type_help/utils/file_retriever.py
"""Type Help 游戏的文件检索器

让 LLM 决定打开哪些文件，然后返回这些文件的内容
"""
from __future__ import annotations

from typing import List, Dict, Any, Optional


class TypeHelpFileRetriever:
    """Type Help 游戏的文件检索器

    不使用向量/关键词检索，而是让 LLM 指定要打开的文件名，
    然后返回这些文件的完整信息
    """

    def __init__(self, env):
        """初始化文件检索器

        Args:
            env: Type Help 游戏环境实例
        """
        self.env = env

    def retrieve_files(self, filenames: List[str]) -> List[Dict[str, Any]]:
        """根据文件名列表检索文件内容

        Args:
            filenames: 要检索的文件名列表

        Returns:
            文件信息列表，每个文件包含 name, content, key_info 等
        """
        results = []

        for filename in filenames:
            # 检查文件是否存在
            if filename not in self.env.nodes:
                results.append({
                    "name": filename,
                    "exists": False,
                    "error": "No file found"
                })
                continue

            # 获取文件节点信息
            node = self.env.nodes[filename]
            memory_data = node.get("memory", {})

            # 提取文件信息
            file_info = {
                "name": filename,
                "exists": True,
                "key_info": memory_data.get("key_info", []),
                "location": memory_data.get("location", ""),
                "scenes": memory_data.get("scenes", []),
                "characters": []
            }

            # 提取角色信息
            for char_data in memory_data.get("characters", []):
                file_info["characters"].append({
                    "name": char_data.get("name", ""),
                    "number": char_data.get("number", 0),
                    "description": char_data.get("description", ""),
                    "key_info": char_data.get("key_info", []),
                })

            results.append(file_info)

        return results

    def format_single_file(self, file_info: Dict[str, Any]) -> str:
        """格式化单个文件信息为文本（通用函数）

        Args:
            file_info: 文件信息字典，包含 name, location, key_info, characters 等

        Returns:
            格式化后的文本字符串
        """
        if not file_info.get("exists", False):
            return f"\n[{file_info['name']}] - No file found"

        lines = [f"\n[{file_info['name']}]"]

        # 地点
        if file_info.get("location"):
            lines.append(f"location: {file_info['location']}")

        # 场景描述
        if file_info.get("scenes"):
            lines.append("scenes:")
            for scene in file_info["scenes"]:
                lines.append(f"  - {scene.get('name', '')}: {scene.get('description', '')}")
                for elem in scene.get("key_elements", []):
                    lines.append(f"    * {elem}")

        # 关键信息
        if file_info.get("key_info"):
            lines.append("key info:")
            for info in file_info["key_info"]:
                lines.append(f"  - {info}")

        # 角色（name、number、description、key_info）
        if file_info.get("characters"):
            lines.append("characters:")
            for char in file_info["characters"]:
                char_str = f"  - {char.get('name', '')}"
                if char.get("number"):
                    char_str += f" (number: {char['number']})"
                if char.get("description"):
                    char_str += f": {char['description']}"
                lines.append(char_str)
                for ci in char.get("key_info", []):
                    lines.append(f"    * {ci}")

        return "\n".join(lines)

    def format_retrieved_files(self, file_results: List[Dict[str, Any]]) -> str:
        """格式化检索到的文件信息为文本
        Args:
            file_results: retrieve_files 返回的结果列表

        Returns:
            格式化后的文本字符串，只包含name, key_info, location, character.name, character.number
        """
        if not file_results:
            return "No files found"

        return "\n".join([self.format_single_file(file_info) for file_info in file_results])
