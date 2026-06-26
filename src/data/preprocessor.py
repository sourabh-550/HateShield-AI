# src/data/preprocessor.py
"""
Text Preprocessing Pipeline for Hinglish/Code-Mixed Text.

Design Decision: We build this as a class so it can be:
1. Imported cleanly into training scripts
2. Reused in the FastAPI inference endpoint
3. Tested independently with unit tests

Why not just use a function? A class lets us configure it
(e.g. toggle emoji removal on/off) without changing the interface.
"""

import re
import string
import emoji
from loguru import logger
from typing import Optional


class HinglishPreprocessor:
    """
    Preprocessing pipeline specifically designed for
    code-mixed Hinglish social media text.
    
    Usage:
        preprocessor = HinglishPreprocessor()
        clean_text = preprocessor.clean("Modi ji is best!!!! 😂")
    """

    def __init__(
        self,
        remove_urls: bool = True,
        remove_mentions: bool = True,
        remove_hashtags: bool = False,   # Keep hashtags — they carry meaning
        remove_emojis: bool = False,     # Convert to text — they carry meaning
        remove_numbers: bool = False,    # Numbers carry context
        lowercase: bool = True,
        normalize_whitespace: bool = True,
        min_length: int = 3,
    ):
        """
        Args:
            remove_urls: Remove http/https links
            remove_mentions: Remove @username mentions
            remove_hashtags: Remove #hashtag (default False - hashtags carry meaning)
            remove_emojis: Remove emojis (default False - convert to text instead)
            remove_numbers: Remove numeric tokens
            lowercase: Convert to lowercase
            normalize_whitespace: Collapse multiple spaces
            min_length: Minimum text length after cleaning
        """
        self.remove_urls = remove_urls
        self.remove_mentions = remove_mentions
        self.remove_hashtags = remove_hashtags
        self.remove_emojis = remove_emojis
        self.remove_numbers = remove_numbers
        self.lowercase = lowercase
        self.normalize_whitespace = normalize_whitespace
        self.min_length = min_length

        # Pre-compile regex patterns for speed
        # Why pre-compile? Regex compilation is expensive.
        # Doing it once in __init__ instead of every clean() call
        # gives ~10x speedup on large datasets.
        self._url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )
        self._mention_pattern = re.compile(r'@\w+')
        self._hashtag_pattern = re.compile(r'#(\w+)')  # Keep the word, remove #
        self._number_pattern = re.compile(r'\b\d+\b')
        self._repeated_char_pattern = re.compile(r'(.)\1{2,}')  # aaaa → aa
        self._repeated_punct_pattern = re.compile(r'([!?.]){2,}')  # !!! → !
        self._whitespace_pattern = re.compile(r'\s+')

    def _remove_urls(self, text: str) -> str:
        return self._url_pattern.sub(' ', text)

    def _remove_mentions(self, text: str) -> str:
        return self._mention_pattern.sub(' ', text)

    def _handle_hashtags(self, text: str) -> str:
        if self.remove_hashtags:
            # Remove hashtag completely
            return self._hashtag_pattern.sub(' ', text)
        else:
            # Keep the word, just remove the # symbol
            # #BJP → BJP (the word carries meaning)
            return self._hashtag_pattern.sub(r'\1', text)

    def _handle_emojis(self, text: str) -> str:
        if self.remove_emojis:
            return emoji.replace_emoji(text, replace='')
        else:
            # Convert emoji to text description
            # 😂 → ":face_with_tears_of_joy:"
            # Why? BERT tokenizer will at least see meaningful tokens
            return emoji.demojize(text)  # gives :face_with_tears_of_joy:
        
    def _normalize_repeated_chars(self, text: str) -> str:
        # "sooooo good" → "soo good" (keep 2 for emphasis signal)
        return self._repeated_char_pattern.sub(r'\1\1', text)

    def _normalize_repeated_punct(self, text: str) -> str:
        # "!!!" → "!" 
        return self._repeated_punct_pattern.sub(r'\1', text)

    def _remove_numbers(self, text: str) -> str:
        return self._number_pattern.sub(' ', text)

    def clean(self, text: str) -> Optional[str]:
        """
        Main cleaning pipeline. Applies all steps in order.
        Order matters! e.g. lowercase before regex matching.
        
        Args:
            text: Raw input text
            
        Returns:
            Cleaned text, or None if text is too short after cleaning
        """
        if not isinstance(text, str):
            return None

        # Step 1: URLs (before lowercasing - URLs are case-insensitive anyway)
        if self.remove_urls:
            text = self._remove_urls(text)

        # Step 2: Mentions
        if self.remove_mentions:
            text = self._remove_mentions(text)

        # Step 3: Hashtags
        text = self._handle_hashtags(text)

        # Step 4: Emojis (before lowercase - emojis aren't affected by case)
        text = self._handle_emojis(text)

        # Step 5: Lowercase
        if self.lowercase:
            text = text.lower()

        # Step 6: Normalize repeated characters (sooo → soo)
        text = self._normalize_repeated_chars(text)

        # Step 7: Normalize repeated punctuation (!! → !)
        text = self._normalize_repeated_punct(text)

        # Step 8: Remove numbers
        if self.remove_numbers:
            text = self._remove_numbers(text)

        # Step 9: Normalize whitespace (always last)
        if self.normalize_whitespace:
            text = self._whitespace_pattern.sub(' ', text).strip()

        # Step 10: Length check
        if len(text) < self.min_length:
            return None

        return text

    def clean_batch(self, texts: list, verbose: bool = True) -> list:
        """
        Clean a list of texts. Returns list of (cleaned_text, original_index).
        Skips None results but logs how many were dropped.
        
        Args:
            texts: List of raw texts
            verbose: Whether to log progress
            
        Returns:
            List of cleaned texts (None for failed ones)
        """
        cleaned = []
        failed = 0

        for text in texts:
            result = self.clean(text)
            cleaned.append(result)
            if result is None:
                failed += 1

        if verbose:
            logger.info(f"Cleaned {len(texts)} texts | Failed/too short: {failed}")

        return cleaned


# ── Quick test if run directly ────────────────────────────────────────
if __name__ == "__main__":
    preprocessor = HinglishPreprocessor()

    test_cases = [
        "Modi ji is best!!!! 😂😂😂",
        "Check this out http://t.co/abc123 @narendramodi #BJP",
        "yaaaar tu bohot zyaadaaa bura hai!!!",
        "I hate you so much!!!! You are the worst person ever",
        "normal hinglish text without any special characters",
        "",   # Edge case: empty
        None, # Edge case: None
    ]

    print("Testing HinglishPreprocessor:")
    print("="*60)
    for text in test_cases:
        result = preprocessor.clean(text)
        print(f"\nInput:  {text}")
        print(f"Output: {result}")