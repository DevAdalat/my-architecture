"""Configuration loader for DPSN-R model."""

from dataclasses import dataclass

import yaml


@dataclass
class ModelConfig:
    """Configuration for the model architecture."""

    vocab_size: int
    hidden_dim: int
    num_layers: int
    num_heads: int
    head_dim: int
    intermediate_dim: int
    dropout: float
    pool_size: int
    pool_dim: int
    top_k: int
    retrieval_dim: int
    max_reasoning_steps: int
    max_seq_len: int
    act_threshold: float
    ponder_lambda: float
    knowledge_ratio: float
    reasoning_ratio: float
    grammar_ratio: float


@dataclass
class TrainingConfig:
    """Configuration for the training process."""

    batch_size: int
    lr: float
    epochs: int
    seq_len: int
    recurrent_steps: int
    weight_decay: float = 0.01
    generate_steps: int = 100
    save_steps: int = 1000


@dataclass
class DatasetConfig:
    """Configuration for the dataset."""

    name: str
    streaming: bool = False
    column_name: str = "text"
    prefetch_factor: int = 2
    config_name: str | None = None
    tokenizer_name: str | None = None


@dataclass
class FullConfig:
    """Container for all configurations."""

    model: ModelConfig
    training: TrainingConfig
    dataset: DatasetConfig


def load_config(yaml_path: str) -> FullConfig:
    """Loads a YAML configuration file and returns a FullConfig object.

    Args:
        yaml_path: Path to the YAML configuration file.

    Returns:
        A FullConfig object containing the loaded settings.
    """
    with open(yaml_path) as f:
        config_dict = yaml.safe_load(f)

    return FullConfig(
        model=ModelConfig(**config_dict["model"]),
        training=TrainingConfig(**config_dict["training"]),
        dataset=DatasetConfig(**config_dict["dataset"]),
    )
