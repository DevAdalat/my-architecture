"""Synthetic data generation for DPSN-R training.

Generates algorithmic pattern data that tests the model's ability to:
1. Learn deterministic patterns (copy, reverse, arithmetic)
2. Generalize within-distribution
3. Handle multi-step reasoning

All generators produce token sequences using a small vocabulary (256 tokens).
"""

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import IntEnum

import torch
from torch.utils.data import Dataset


class SpecialTokens(IntEnum):
    PAD = 0
    BOS = 1
    EOS = 2
    SEP = 3


@dataclass
class SyntheticSample:
    input_ids: list[int]
    labels: list[int]
    pattern_type: str


class PatternGenerator(ABC):
    """Base class for algorithmic pattern generators."""

    def __init__(self, vocab_size: int = 256, max_len: int = 64) -> None:
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.data_start = int(SpecialTokens.SEP) + 1

    @abstractmethod
    def generate(self) -> SyntheticSample:
        pass

    def _random_tokens(self, length: int) -> list[int]:
        return [random.randint(self.data_start, self.vocab_size - 1) for _ in range(length)]


class CopyPattern(PatternGenerator):
    """Copy the input sequence. Tests basic memorization."""

    def generate(self) -> SyntheticSample:
        length = random.randint(4, self.max_len // 3)
        tokens = self._random_tokens(length)

        input_ids = [int(SpecialTokens.BOS)] + tokens + [int(SpecialTokens.SEP)]
        labels = [-100] * len(input_ids) + tokens + [int(SpecialTokens.EOS)]
        input_ids = input_ids + tokens + [int(SpecialTokens.EOS)]

        return SyntheticSample(input_ids, labels, "copy")


class ReversePattern(PatternGenerator):
    """Reverse the input sequence. Tests sequential reasoning."""

    def generate(self) -> SyntheticSample:
        length = random.randint(4, self.max_len // 3)
        tokens = self._random_tokens(length)

        input_ids = [int(SpecialTokens.BOS)] + tokens + [int(SpecialTokens.SEP)]
        reversed_tokens = tokens[::-1]
        labels = [-100] * len(input_ids) + reversed_tokens + [int(SpecialTokens.EOS)]
        input_ids = input_ids + reversed_tokens + [int(SpecialTokens.EOS)]

        return SyntheticSample(input_ids, labels, "reverse")


class RepeatPattern(PatternGenerator):
    """Repeat each token N times. Tests local pattern learning."""

    def __init__(self, vocab_size: int = 256, max_len: int = 64, repeats: int = 2) -> None:
        super().__init__(vocab_size, max_len)
        self.repeats = repeats

    def generate(self) -> SyntheticSample:
        length = random.randint(3, self.max_len // (self.repeats + 2))
        tokens = self._random_tokens(length)

        input_ids = [int(SpecialTokens.BOS)] + tokens + [int(SpecialTokens.SEP)]
        repeated = []
        for t in tokens:
            repeated.extend([t] * self.repeats)

        labels = [-100] * len(input_ids) + repeated + [int(SpecialTokens.EOS)]
        input_ids = input_ids + repeated + [int(SpecialTokens.EOS)]

        return SyntheticSample(input_ids, labels, f"repeat_{self.repeats}")


class SortPattern(PatternGenerator):
    """Sort the input sequence. Tests global ordering."""

    def generate(self) -> SyntheticSample:
        length = random.randint(4, self.max_len // 3)
        tokens = self._random_tokens(length)

        input_ids = [int(SpecialTokens.BOS)] + tokens + [int(SpecialTokens.SEP)]
        sorted_tokens = sorted(tokens)
        labels = [-100] * len(input_ids) + sorted_tokens + [int(SpecialTokens.EOS)]
        input_ids = input_ids + sorted_tokens + [int(SpecialTokens.EOS)]

        return SyntheticSample(input_ids, labels, "sort")


class ArithmeticPattern(PatternGenerator):
    """Add 1 to each token (mod vocab_size). Tests simple arithmetic."""

    def generate(self) -> SyntheticSample:
        length = random.randint(4, self.max_len // 3)
        tokens = self._random_tokens(length)

        input_ids = [int(SpecialTokens.BOS)] + tokens + [int(SpecialTokens.SEP)]
        incremented = [
            (t - self.data_start + 1) % (self.vocab_size - self.data_start) + self.data_start
            for t in tokens
        ]
        labels = [-100] * len(input_ids) + incremented + [int(SpecialTokens.EOS)]
        input_ids = input_ids + incremented + [int(SpecialTokens.EOS)]

        return SyntheticSample(input_ids, labels, "increment")


class AlternatePattern(PatternGenerator):
    """Output tokens at even indices, then odd indices. Tests index-based reasoning."""

    def generate(self) -> SyntheticSample:
        length = random.randint(6, self.max_len // 3)
        tokens = self._random_tokens(length)

        input_ids = [int(SpecialTokens.BOS)] + tokens + [int(SpecialTokens.SEP)]
        alternated = tokens[::2] + tokens[1::2]
        labels = [-100] * len(input_ids) + alternated + [int(SpecialTokens.EOS)]
        input_ids = input_ids + alternated + [int(SpecialTokens.EOS)]

        return SyntheticSample(input_ids, labels, "alternate")


class AssociativeRecallPattern(PatternGenerator):
    """
    Associative Recall Task (Key-Value Retrieval).
    Format: k1 v1 k2 v2 ... ? query_key -> target_value
    Tests the model's ability to store mappings in memory (Pool) and retrieve them.
    """

    def __init__(self, vocab_size: int, seq_len: int):
        super().__init__(vocab_size, seq_len)
        self.seq_len = seq_len  # Explicitly store for usage
        # -2 for query/separator, //2 for pairs. Ensure integer math.
        self.num_pairs = int((seq_len - 2) // 2)

    def generate(self) -> SyntheticSample:
        # Generate random keys and values
        # Keys must be unique to avoid ambiguity
        # We reserve 0 for padding, 1 for separator '?'
        available_tokens = torch.arange(2, self.vocab_size)

        # Select 2*num_pairs tokens (keys and values)
        # To ensure keys are unique, we pick num_pairs keys
        # Use int cast for slice index
        perm = torch.randperm(len(available_tokens))
        keys = available_tokens[perm[: int(self.num_pairs)]]
        values = torch.randint(2, self.vocab_size, (int(self.num_pairs),))

        # Create input sequence: k1 v1 k2 v2 ...
        seq = torch.empty(self.seq_len, dtype=torch.long).fill_(0)

        # Interleave keys and values
        # Use explicit int slicing
        limit = int(self.num_pairs * 2)
        seq[0:limit:2] = keys
        seq[1:limit:2] = values

        # Choose a query key from the keys we used
        query_idx = torch.randint(0, int(self.num_pairs), (1,)).item()
        query_key = keys[query_idx]
        target_val = values[query_idx]

        # Add query at the end: ... ? query_key
        # Let's put the separator at the second to last position
        seq[-2] = 1  # Separator
        seq[-1] = query_key

        # Targets
        target_val_tensor = target_val.unsqueeze(0) if target_val.ndim == 0 else target_val

        full_seq = torch.cat([seq, target_val_tensor])
        input_ids = full_seq[:-1].tolist()
        labels = full_seq[1:].tolist()

        return SyntheticSample(input_ids, labels, "associative_recall")


class SyntheticDataset(Dataset):
    """Dataset of mixed algorithmic patterns."""

    def __init__(
        self,
        num_samples: int,
        vocab_size: int = 256,
        max_seq_len: int = 128,
        seed: int = 42,
    ) -> None:
        self.num_samples = num_samples
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

        self.generators = [
            CopyPattern(vocab_size, max_seq_len),
            ReversePattern(vocab_size, max_seq_len),
            RepeatPattern(vocab_size, max_seq_len),
            SortPattern(vocab_size, max_seq_len),
            ArithmeticPattern(vocab_size, max_seq_len),
            AlternatePattern(vocab_size, max_seq_len),
            AssociativeRecallPattern(vocab_size, max_seq_len),
        ]

        random.seed(seed)
        self.samples = [self._generate_sample() for _ in range(num_samples)]

    def _generate_sample(self) -> SyntheticSample:
        generator = random.choice(self.generators)
        return generator.generate()

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.samples[idx]
        return {
            "input_ids": torch.tensor(sample.input_ids, dtype=torch.long),
            "labels": torch.tensor(sample.labels, dtype=torch.long),
        }


def collate_fn(
    batch: list[dict[str, torch.Tensor]],
    max_len: int = 128,
    pad_id: int = int(SpecialTokens.PAD),
) -> dict[str, torch.Tensor]:
    """Collate batch with padding to fixed length (XLA compliance)."""
    input_ids_list = []
    labels_list = []

    for item in batch:
        input_ids = item["input_ids"]
        labels = item["labels"]

        if len(input_ids) > max_len:
            input_ids = input_ids[:max_len]
            labels = labels[:max_len]

        pad_len = max_len - len(input_ids)
        input_ids = torch.cat([input_ids, torch.full((pad_len,), pad_id, dtype=torch.long)])
        labels = torch.cat([labels, torch.full((pad_len,), -100, dtype=torch.long)])

        input_ids_list.append(input_ids)
        labels_list.append(labels)

    return {
        "input_ids": torch.stack(input_ids_list),
        "labels": torch.stack(labels_list),
    }


def create_data_splits(
    total_samples: int = 10000,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    vocab_size: int = 256,
    max_seq_len: int = 128,
    seed: int = 42,
) -> tuple[SyntheticDataset, SyntheticDataset, SyntheticDataset]:
    """Create train/val/test splits with different seeds."""
    train_size = int(total_samples * train_ratio)
    val_size = int(total_samples * val_ratio)
    test_size = total_samples - train_size - val_size

    train_ds = SyntheticDataset(train_size, vocab_size, max_seq_len, seed=seed)
    val_ds = SyntheticDataset(val_size, vocab_size, max_seq_len, seed=seed + 1)
    test_ds = SyntheticDataset(test_size, vocab_size, max_seq_len, seed=seed + 2)

    return train_ds, val_ds, test_ds
