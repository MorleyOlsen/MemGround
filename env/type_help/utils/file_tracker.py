# env/type_help/utils/file_tracker.py
"""File tracker for the Type Help game"""
from __future__ import annotations

from typing import List, Set


class FileTracker:
    """File unlock tracker"""

    def __init__(self):
        self.unlocked_files: Set[str] = set()  # Names of unlocked files
        self.attempted_files: List[str] = []  # History of attempted filenames (all attempts)
        self.success_files: List[str] = []  # Successfully opened files
        self.failed_files: List[str] = []  # Failed attempts (file not found)
        self.file_naming_patterns: List[str] = []  # Discovered naming patterns
        self.read_files: List[str] = []  # List of read files (filenames only)

    def unlock_file(self, filename: str) -> bool:
        """Unlock a file"""
        if filename not in self.unlocked_files:
            self.unlocked_files.add(filename)
            return True
        return False

    def remove_unlocked_file(self, filename: str) -> bool:
        """Remove a file from the unlocked files list

        Args:
            filename: Filename to remove

        Returns:
            True if the file was found and removed, False if it was not present
        """
        if filename in self.unlocked_files:
            self.unlocked_files.remove(filename)
            return True
        return False

    def attempt_file(self, filename: str, success: bool = True) -> None:
        """Record an attempt to open a file

        Args:
            filename: Filename
            success: Whether the attempt succeeded (True = success, False = failure)
        """
        self.attempted_files.append(filename)
        if success:
            self.success_files.append(filename)
            # On success, add to the read files list (deduplicated)
            if filename not in self.read_files:
                self.read_files.append(filename)
        else:
            self.failed_files.append(filename)

    # def add_pattern(self, pattern: str) -> None:
    #     """Add a discovered naming pattern"""
    #     if pattern not in self.file_naming_patterns:
    #         self.file_naming_patterns.append(pattern)

    def get_unlocked_files(self) -> List[str]:
        """Get the list of unlocked files"""
        return sorted(list(self.unlocked_files))

    def get_attempted_files(self) -> List[str]:
        """Get the attempt history (all attempts)"""
        return self.attempted_files.copy()

    def get_success_files(self) -> List[str]:
        """Get the list of successfully opened files"""
        return self.success_files.copy()

    def get_failed_files(self) -> List[str]:
        """Get the list of failed attempts (file not found)"""
        return self.failed_files.copy()

    def is_unlocked(self, filename: str) -> bool:
        """Check whether a file is unlocked"""
        return filename in self.unlocked_files

    def get_read_files(self) -> List[str]:
        """Get the list of read files"""
        return self.read_files.copy()

    def get_read_files_text(self) -> str:
        """Get a text representation of the read files list (for inclusion in prompts)

        Returns:
            Formatted read files list string
        """
        if not self.read_files:
            return "No files have been read yet"

        file_names = [f'"{file}"' for file in self.read_files]
        return f"Read files: {', '.join(file_names)}"
