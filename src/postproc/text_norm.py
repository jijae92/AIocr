"""
Text normalization and cleanup utilities.
"""

import re
from typing import Dict, List

from util.logging import get_logger

logger = get_logger(__name__)


class TextNormalizer:
    """Text normalization and cleanup."""

    # Common OCR errors
    COMMON_ERRORS = {
        'l': '1',  # lowercase L to 1
        'O': '0',  # uppercase O to 0
        '|': 'I',  # pipe to I
    }

    @staticmethod
    def normalize_whitespace(text: str) -> str:
        """
        Normalize whitespace.

        Args:
            text: Input text

        Returns:
            Normalized text
        """
        # Replace multiple spaces with single space
        text = re.sub(r' +', ' ', text)

        # Replace multiple newlines with double newline
        text = re.sub(r'\n\n+', '\n\n', text)

        # Remove trailing/leading whitespace from each line
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(lines)

        return text.strip()

    @staticmethod
    def fix_common_errors(text: str, aggressive: bool = False) -> str:
        """
        Fix common OCR errors.

        Args:
            text: Input text
            aggressive: Apply aggressive fixes

        Returns:
            Fixed text
        """
        # Fix common substitutions in numeric contexts
        text = re.sub(r'(\d)l(\d)', r'\g<1>1\g<2>', text)  # 1l1 -> 111
        text = re.sub(r'(\d)O(\d)', r'\g<1>0\g<2>', text)  # 1O1 -> 101

        if aggressive:
            # More aggressive fixes
            text = re.sub(r'\bl\b', '1', text)  # standalone l -> 1
            text = re.sub(r'\bO\b', '0', text)  # standalone O -> 0

        return text

    @staticmethod
    def remove_formatting_artifacts(text: str) -> str:
        """
        Remove PDF formatting artifacts.

        Args:
            text: Input text

        Returns:
            Cleaned text
        """
        # Remove form feed characters
        text = text.replace('\f', '')

        # Remove zero-width spaces
        text = text.replace('\u200b', '')

        # Remove soft hyphens
        text = text.replace('\u00ad', '')

        return text

    @staticmethod
    def fix_line_breaks(text: str) -> str:
        """
        Fix improper line breaks (e.g., hyphenation).

        Args:
            text: Input text

        Returns:
            Fixed text
        """
        # Fix hyphenation at line breaks
        text = re.sub(r'-\n(\w)', r'\g<1>', text)

        # Join lines that don't end with sentence terminators
        lines = text.split('\n')
        fixed_lines = []
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if not line:
                fixed_lines.append('')
                i += 1
                continue

            # Check if line ends with sentence terminator
            if line[-1] not in '.!?:;':
                # Check if next line exists and starts with lowercase
                if i + 1 < len(lines) and lines[i + 1] and lines[i + 1][0].islower():
                    # Join lines
                    line = line + ' ' + lines[i + 1].strip()
                    i += 2
                else:
                    fixed_lines.append(line)
                    i += 1
            else:
                fixed_lines.append(line)
                i += 1

        return '\n'.join(fixed_lines)

    @staticmethod
    def normalize(
        text: str,
        normalize_whitespace: bool = True,
        fix_common_errors: bool = True,
        remove_artifacts: bool = True,
        fix_line_breaks: bool = False,
    ) -> str:
        """
        Apply full normalization pipeline.

        Args:
            text: Input text
            normalize_whitespace: Normalize whitespace
            fix_common_errors: Fix common OCR errors
            remove_artifacts: Remove formatting artifacts
            fix_line_breaks: Fix line breaks

        Returns:
            Normalized text
        """
        if remove_artifacts:
            text = TextNormalizer.remove_formatting_artifacts(text)

        if fix_line_breaks:
            text = TextNormalizer.fix_line_breaks(text)

        if fix_common_errors:
            text = TextNormalizer.fix_common_errors(text)

        if normalize_whitespace:
            text = TextNormalizer.normalize_whitespace(text)

        return text
