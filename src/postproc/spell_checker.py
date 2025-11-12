"""
Spell correction module for OCR results.

Supports English, Korean, and auto-detection.
"""

import re
from typing import Optional

from langdetect import detect, LangDetectException
from spellchecker import SpellChecker

from util.logging import get_logger

logger = get_logger(__name__)


class SpellCorrector:
    """
    Spell correction for OCR results.

    Supports:
    - English spell correction (pyspellchecker)
    - Korean basic corrections (rule-based)
    - Auto language detection
    """

    def __init__(self):
        """Initialize spell correctors."""
        self.english_checker = None
        self.initialized_en = False

    def _init_english(self):
        """Lazy initialization of English spell checker."""
        if not self.initialized_en:
            try:
                self.english_checker = SpellChecker(language='en')
                self.initialized_en = True
                logger.info("English spell checker initialized")
            except Exception as e:
                logger.error(f"Failed to initialize English spell checker: {e}")
                self.initialized_en = True  # Don't retry

    def detect_language(self, text: str) -> Optional[str]:
        """
        Detect language of text.

        Args:
            text: Input text

        Returns:
            Language code ('en', 'ko', etc.) or None
        """
        if not text or len(text.strip()) < 10:
            return None

        try:
            lang = detect(text)
            return lang
        except LangDetectException:
            return None

    def correct_english(self, text: str) -> str:
        """
        Correct English spelling.

        Args:
            text: Input text

        Returns:
            Corrected text
        """
        self._init_english()

        if not self.english_checker:
            return text

        try:
            # Split into words
            words = text.split()
            corrected_words = []

            for word in words:
                # Skip if not alphabetic or too short
                clean_word = re.sub(r'[^a-zA-Z]', '', word)
                if len(clean_word) < 2:
                    corrected_words.append(word)
                    continue

                # Check if misspelled
                if clean_word.lower() in self.english_checker:
                    corrected_words.append(word)
                else:
                    # Get correction
                    correction = self.english_checker.correction(clean_word.lower())
                    if correction and correction != clean_word.lower():
                        # Preserve original capitalization pattern
                        if clean_word.isupper():
                            corrected = correction.upper()
                        elif clean_word[0].isupper():
                            corrected = correction.capitalize()
                        else:
                            corrected = correction

                        # Replace in original word (preserve punctuation)
                        corrected_words.append(word.replace(clean_word, corrected))
                    else:
                        corrected_words.append(word)

            return ' '.join(corrected_words)
        except Exception as e:
            logger.warning(f"English spell correction failed: {e}")
            return text

    def correct_korean(self, text: str) -> str:
        """
        Correct Korean text (rule-based).

        Common OCR errors in Korean:
        - Spacing issues
        - Common character confusions

        Args:
            text: Input text

        Returns:
            Corrected text
        """
        if not text:
            return text

        try:
            corrected = text

            # Fix full-width numbers to half-width
            full_width_corrections = {
                '８': '8',
                '９': '9',
                '０': '0',
                '１': '1',
                '２': '2',
                '３': '3',
                '４': '4',
                '５': '5',
                '６': '6',
                '７': '7',
            }

            for old, new in full_width_corrections.items():
                corrected = corrected.replace(old, new)

            # Fix multiple spaces
            corrected = re.sub(r' {2,}', ' ', corrected)

            # Fix spacing around Korean and English
            # Korean character followed by English: add space
            corrected = re.sub(r'([가-힣])([A-Za-z])', r'\1 \2', corrected)
            # English followed by Korean: add space
            corrected = re.sub(r'([A-Za-z])([가-힣])', r'\1 \2', corrected)

            # Fix spacing around numbers
            corrected = re.sub(r'([가-힣])(\d)', r'\1 \2', corrected)
            corrected = re.sub(r'(\d)([가-힣])', r'\1 \2', corrected)

            return corrected
        except Exception as e:
            logger.warning(f"Korean text correction failed: {e}")
            return text

    def correct(self, text: str, language: str = 'auto') -> str:
        """
        Correct spelling based on language.

        Args:
            text: Input text
            language: 'en', 'ko', 'auto', or 'disabled'

        Returns:
            Corrected text
        """
        if not text or language.lower() == 'disabled':
            return text

        # Auto-detect language
        if language.lower() == 'auto-detect':
            detected = self.detect_language(text)
            if detected:
                language = detected
                logger.debug(f"Detected language: {language}")
            else:
                return text

        # Apply correction based on language
        language = language.lower()

        if language == 'english' or language == 'en':
            return self.correct_english(text)
        elif language == 'korean' or language == 'ko':
            return self.correct_korean(text)
        else:
            # Unknown language, return as-is
            return text


# Global instance
_spell_corrector: Optional[SpellCorrector] = None


def get_spell_corrector() -> SpellCorrector:
    """Get global spell corrector instance."""
    global _spell_corrector

    if _spell_corrector is None:
        _spell_corrector = SpellCorrector()

    return _spell_corrector
