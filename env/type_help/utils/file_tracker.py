# env/type_help/utils/file_tracker.py
"""Type Help 游戏的文件追踪器"""
from __future__ import annotations

from typing import List, Set


class FileTracker:
    """文件解锁追踪器"""

    def __init__(self):
        self.unlocked_files: Set[str] = set()  # 已解锁的文件名
        self.attempted_files: List[str] = []  # 尝试打开的文件名历史（所有尝试）
        self.success_files: List[str] = []  # 成功打开的文件
        self.failed_files: List[str] = []  # 失败的尝试（文件不存在）
        self.file_naming_patterns: List[str] = []  # 发现的命名规则
        self.read_files: List[str] = []  # 已读文件列表（只记录文件名）

    def unlock_file(self, filename: str) -> bool:
        """解锁文件"""
        if filename not in self.unlocked_files:
            self.unlocked_files.add(filename)
            return True
        return False

    def remove_unlocked_file(self, filename: str) -> bool:
        """从已解锁文件列表中删除文件

        Args:
            filename: 要删除的文件名

        Returns:
            是否成功删除（True=文件存在并已删除，False=文件不存在）
        """
        if filename in self.unlocked_files:
            self.unlocked_files.remove(filename)
            return True
        return False

    def attempt_file(self, filename: str, success: bool = True) -> None:
        """记录尝试打开文件

        Args:
            filename: 文件名
            success: 是否成功打开（True=成功，False=失败）
        """
        self.attempted_files.append(filename)
        if success:
            self.success_files.append(filename)
            # 成功打开时，添加到已读文件列表（去重）
            if filename not in self.read_files:
                self.read_files.append(filename)
        else:
            self.failed_files.append(filename)

    # def add_pattern(self, pattern: str) -> None:
    #     """添加发现的命名规则"""
    #     if pattern not in self.file_naming_patterns:
    #         self.file_naming_patterns.append(pattern)

    def get_unlocked_files(self) -> List[str]:
        """获取已解锁文件列表"""
        return sorted(list(self.unlocked_files))

    def get_attempted_files(self) -> List[str]:
        """获取尝试历史（所有尝试）"""
        return self.attempted_files.copy()

    def get_success_files(self) -> List[str]:
        """获取成功打开的文件"""
        return self.success_files.copy()

    def get_failed_files(self) -> List[str]:
        """获取失败的尝试（文件不存在）"""
        return self.failed_files.copy()

    def is_unlocked(self, filename: str) -> bool:
        """检查文件是否已解锁"""
        return filename in self.unlocked_files

    def get_read_files(self) -> List[str]:
        """获取已读文件列表"""
        return self.read_files.copy()

    def get_read_files_text(self) -> str:
        """获取已读文件列表的文本表示（用于添加到prompt）

        Returns:
            格式化后的已读文件列表字符串
        """
        if not self.read_files:
            return "尚未阅读任何文件"

        file_names = [f'"{file}"' for file in self.read_files]
        return f"已阅读的文件: {', '.join(file_names)}"
