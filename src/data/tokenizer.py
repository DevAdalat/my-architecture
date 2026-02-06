"""Simple character-level tokenizer for DPSN-R."""

from enum import IntEnum


class SpecialTokens(IntEnum):
    """Special tokens for the tokenizer."""

    PAD = 0
    BOS = 1
    EOS = 2
    SEP = 3


class CharTokenizer:
    """A simple character-level tokenizer mapping characters to their ordinal values."""

    def __init__(self, vocab_size: int = 256) -> None:
        """Initializes the tokenizer.

        Args:
            vocab_size: The total vocabulary size (including special tokens).
        """
        self.special_token_count = len(SpecialTokens)
        self.vocab_size = vocab_size
        # Max character ordinal we can support
        self.max_char_ord = self.vocab_size - self.special_token_count - 1

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> list[int]:
        """Encodes text into a list of token IDs.

        Args:
            text: The input string to encode.
            add_bos: Whether to prepend the BOS token.
            add_eos: Whether to append the EOS token.

        Returns:
            A list of integer token IDs.
        """
        ids = [min(ord(c) + self.special_token_count, self.vocab_size - 1) for c in text]

        if add_bos:
            ids = [int(SpecialTokens.BOS)] + ids
        if add_eos:
            ids = ids + [int(SpecialTokens.EOS)]

        return ids

    def decode(self, ids: list[int]) -> str:
        """Decodes a list of token IDs back into text.

        Args:
            ids: A list of integer token IDs.

        Returns:
            The decoded string.
        """
        chars = []
        for i in ids:
            if i >= self.special_token_count:
                # Basic mapping back to char
                chars.append(chr(i - self.special_token_count))
            elif i == int(SpecialTokens.SEP):
                chars.append("[SEP]")
            elif i == int(SpecialTokens.BOS):
                chars.append("[BOS]")
            elif i == int(SpecialTokens.EOS):
                chars.append("[EOS]")
            elif i == int(SpecialTokens.PAD):
                chars.append("[PAD]")
        return "".join(chars)
