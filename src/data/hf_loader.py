"""HuggingFace dataset loader for DPSN-R."""

from functools import partial
from typing import Any

import torch
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from .tokenizer import CharTokenizer
from ..utils.config_loader import DatasetConfig, TrainingConfig


def hf_collate_fn(batch: list[dict], max_len: int) -> dict[str, torch.Tensor]:
    """Collates a batch of examples into tensors."""
    input_ids_list = []
    labels_list = []

    for item in batch:
        input_ids = torch.tensor(item["input_ids"], dtype=torch.long)
        labels = torch.tensor(item["labels"], dtype=torch.long)

        if input_ids.size(0) > max_len:
            input_ids = input_ids[:max_len]
            labels = labels[:max_len]

        input_ids_list.append(input_ids)
        labels_list.append(labels)

    input_ids_padded = pad_sequence(input_ids_list, batch_first=True, padding_value=0)
    labels_padded = pad_sequence(labels_list, batch_first=True, padding_value=-100)

    return {"input_ids": input_ids_padded, "labels": labels_padded}


def get_hf_dataloader(config: DatasetConfig, training_config: TrainingConfig) -> DataLoader:
    """Creates a DataLoader for a HuggingFace dataset.

    Args:
        config: Dataset configuration.
        training_config: Training configuration.

    Returns:
        A PyTorch DataLoader.
    """
    print(f"Loading dataset {config.name} (streaming={config.streaming})...")
    dataset_raw = load_dataset(config.name, name=config.config_name, streaming=config.streaming)
    print("Dataset loaded.")

    # Use 'train' split by default if it's a DatasetDict or IterableDatasetDict
    dataset: Any
    if hasattr(dataset_raw, "keys"):
        if "train" in dataset_raw:
            dataset = dataset_raw["train"]
        else:
            # Fallback to the first available split
            dataset = dataset_raw[next(iter(dataset_raw.keys()))]
    else:
        dataset = dataset_raw

    if config.tokenizer_name:
        print(f"Loading tokenizer {config.tokenizer_name}...")
        tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
        print("Tokenizer loaded.")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer = CharTokenizer()

    def tokenize_function(example: dict) -> dict:
        """Tokenizes a single example and creates input/label pairs."""
        text = example[config.column_name]
        full_ids = tokenizer.encode(text)
        # Standard causal LM task: shift inputs and labels
        input_ids = full_ids[:-1]
        labels = full_ids[1:]
        return {"input_ids": input_ids, "labels": labels}

    if config.streaming:
        # Lazy mapping for streaming datasets
        dataset = dataset.map(tokenize_function)
    else:
        # Pre-tokenize map-style datasets
        column_names = getattr(dataset, "column_names", [])
        dataset = dataset.map(
            tokenize_function,
            batched=False,
            remove_columns=column_names,
        )
        if hasattr(dataset, "set_format"):
            dataset.set_format(type="torch", columns=["input_ids", "labels"])

    # Determine num_workers and prefetch_factor
    num_workers = 0 if config.streaming else 4
    prefetch_factor = config.prefetch_factor if num_workers > 0 else None

    return DataLoader(
        dataset,
        batch_size=training_config.batch_size,
        collate_fn=partial(hf_collate_fn, max_len=training_config.seq_len),
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )
