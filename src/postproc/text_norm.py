"""
Text normalization and cleanup utilities.
"""

import re
from typing import Dict, List, Optional
from datetime import datetime

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

    @staticmethod
    def normalize_korean_spacing(text: str) -> str:
        """
        Normalize Korean spacing patterns.

        Common Korean spacing rules:
        - Add space after punctuation (. , ! ? etc.)
        - Remove space before punctuation
        - Add space between Korean and numbers/English
        - Fix spacing around particles (은/는, 이/가, etc.)

        Args:
            text: Input text with Korean

        Returns:
            Text with normalized Korean spacing
        """
        # Remove spaces before Korean punctuation
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)

        # Add space after Korean punctuation if followed by character
        text = re.sub(r'([.,!?;:])([^\s\n])', r'\1 \2', text)

        # Add space between Korean and English/numbers
        # Korean followed by English
        text = re.sub(r'([\uac00-\ud7a3])([a-zA-Z0-9])', r'\1 \2', text)
        # English/numbers followed by Korean
        text = re.sub(r'([a-zA-Z0-9])([\uac00-\ud7a3])', r'\1 \2', text)

        # Fix spacing around common particles
        # Remove space before particles
        particles = ['은', '는', '이', '가', '을', '를', '의', '에', '에서', '로', '으로', '와', '과', '도']
        for particle in particles:
            text = re.sub(rf'\s+({particle})\b', rf'\1', text)

        # Fix spacing for Korean units (원, 개, 명, etc.)
        units = ['원', '개', '명', '번', '차', '층', '살', '년', '월', '일', '시', '분', '초']
        for unit in units:
            text = re.sub(rf'(\d+)\s+({unit})', rf'\1{unit}', text)

        return text

    @staticmethod
    def normalize_dates(text: str, format_output: str = 'iso') -> str:
        """
        Normalize date formats.

        Detects various date patterns and normalizes to consistent format.

        Args:
            text: Input text
            format_output: Output format ('iso', 'kr', 'us')

        Returns:
            Text with normalized dates
        """
        # Korean date formats: 2024년 3월 15일
        kr_date_pattern = r'(\d{4})\s*년\s*(\d{1,2})\s*월\s*(\d{1,2})\s*일'

        def replace_kr_date(match):
            year, month, day = match.groups()
            month = month.zfill(2)
            day = day.zfill(2)

            if format_output == 'iso':
                return f'{year}-{month}-{day}'
            elif format_output == 'kr':
                return f'{year}년 {int(month)}월 {int(day)}일'
            elif format_output == 'us':
                return f'{month}/{day}/{year}'
            return match.group(0)

        text = re.sub(kr_date_pattern, replace_kr_date, text)

        # ISO date format: 2024-03-15 or 2024.03.15
        iso_date_pattern = r'(\d{4})[-./](\d{1,2})[-./](\d{1,2})'

        def replace_iso_date(match):
            year, month, day = match.groups()
            month = month.zfill(2)
            day = day.zfill(2)

            if format_output == 'iso':
                return f'{year}-{month}-{day}'
            elif format_output == 'kr':
                return f'{year}년 {int(month)}월 {int(day)}일'
            elif format_output == 'us':
                return f'{month}/{day}/{year}'
            return match.group(0)

        text = re.sub(iso_date_pattern, replace_iso_date, text)

        # US date format: 03/15/2024 or 3/15/2024
        us_date_pattern = r'(\d{1,2})/(\d{1,2})/(\d{4})'

        def replace_us_date(match):
            month, day, year = match.groups()
            month = month.zfill(2)
            day = day.zfill(2)

            if format_output == 'iso':
                return f'{year}-{month}-{day}'
            elif format_output == 'kr':
                return f'{year}년 {int(month)}월 {int(day)}일'
            elif format_output == 'us':
                return f'{month}/{day}/{year}'
            return match.group(0)

        text = re.sub(us_date_pattern, replace_us_date, text)

        return text

    @staticmethod
    def normalize_currency(text: str, standardize_symbol: bool = True) -> str:
        """
        Normalize currency amounts.

        Handles various formats:
        - Korean: 1,000원, 1000원, 1천원, 1만원
        - USD: $1,000, $ 1000, USD 1,000
        - Others: ¥1,000, €1,000

        Args:
            text: Input text
            standardize_symbol: If True, standardize currency symbols

        Returns:
            Text with normalized currency
        """
        # Korean currency
        # Convert 천 (thousand) and 만 (ten thousand) to numbers
        def expand_korean_currency(match):
            number_str = match.group(1)
            unit = match.group(2)

            # Parse number
            number = float(number_str.replace(',', ''))

            # Apply unit
            if unit == '천':
                number *= 1_000
            elif unit == '만':
                number *= 10_000
            elif unit == '억':
                number *= 100_000_000

            # Format with commas
            if number.is_integer():
                return f'{int(number):,}원'
            else:
                return f'{number:,.0f}원'

        text = re.sub(r'([\d,]+\.?\d*)\s*(천|만|억)원', expand_korean_currency, text)

        # Standardize Korean currency format: ensure comma separators
        def format_korean_currency(match):
            number_str = match.group(1).replace(',', '')
            try:
                number = int(number_str)
                return f'{number:,}원'
            except ValueError:
                return match.group(0)

        text = re.sub(r'(\d+)원', format_korean_currency, text)

        # Remove space between number and 원
        text = re.sub(r'(\d+)\s+원', r'\1원', text)

        # USD currency
        # Standardize format: $1,000 (no space)
        text = re.sub(r'\$\s+', '$', text)  # Remove space after $
        text = re.sub(r'USD\s*', '$', text) if standardize_symbol else text

        # Add commas to large numbers
        def format_usd(match):
            symbol = match.group(1)
            number_str = match.group(2).replace(',', '')
            try:
                number = int(number_str)
                return f'{symbol}{number:,}'
            except ValueError:
                return match.group(0)

        text = re.sub(r'([\$])(\d+)', format_usd, text)

        # Normalize other currencies (¥, €)
        for symbol in ['¥', '€', '£']:
            text = re.sub(rf'{symbol}\s+', symbol, text)

        return text

    @staticmethod
    def normalize_numbers(text: str, add_comma_separator: bool = True) -> str:
        """
        Normalize numbers (add comma separators, fix decimals).

        Args:
            text: Input text
            add_comma_separator: Add comma separators to large numbers

        Returns:
            Text with normalized numbers
        """
        if not add_comma_separator:
            return text

        # Find standalone numbers (not part of dates, IDs, etc.)
        # Must be at least 4 digits to warrant commas
        def format_number(match):
            number_str = match.group(0).replace(',', '')
            try:
                # Check if it's an integer
                if '.' not in number_str:
                    number = int(number_str)
                    if number >= 1000:  # Only format numbers >= 1000
                        return f'{number:,}'
                return number_str
            except ValueError:
                return number_str

        # Match numbers not preceded/followed by date-like patterns
        text = re.sub(
            r'(?<![/\-\.])\b(\d{4,})(?![/\-\.])',
            format_number,
            text
        )

        return text

    @staticmethod
    def normalize_codes(text: str) -> str:
        """
        Normalize code-like patterns (IDs, product codes, etc.).

        Ensures consistent formatting for:
        - Product codes: ABC-123
        - Phone numbers: 010-1234-5678
        - Postal codes: 12345

        Args:
            text: Input text

        Returns:
            Text with normalized codes
        """
        # Korean phone numbers: normalize to XXX-XXXX-XXXX format
        # Handle various input formats
        phone_pattern = r'0?1[0-9][\s\-]?\d{3,4}[\s\-]?\d{4}'

        def format_phone(match):
            # Remove all non-digits
            digits = re.sub(r'\D', '', match.group(0))

            if len(digits) == 10:  # 010-1234-5678 format (missing leading 0)
                return f'0{digits[:2]}-{digits[2:6]}-{digits[6:]}'
            elif len(digits) == 11:  # 010-1234-5678 format
                return f'{digits[:3]}-{digits[3:7]}-{digits[7:]}'
            return match.group(0)

        text = re.sub(phone_pattern, format_phone, text)

        # Korean postal code: XXXXX or XXX-XXX
        postal_pattern = r'\b(\d{3})[\s\-]?(\d{3})\b'
        text = re.sub(postal_pattern, r'\1-\2', text)

        # Product codes: ensure consistent hyphenation
        # Pattern: ALPHA-NUMERIC
        code_pattern = r'\b([A-Z]{2,})[\s]?(\d{3,})\b'
        text = re.sub(code_pattern, r'\1-\2', text)

        return text
