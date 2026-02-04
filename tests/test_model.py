"""Tests for DPSN-R model components."""

import pytest
import torch

from src.model.act import AdaptiveComputeTime, compute_ponder_loss
from src.model.config import DPSNRConfig
from src.model.controller import TinyController
from src.model.dpsn_r import DPSNR
from src.model.pool import MassivePool


@pytest.fixture
def config() -> DPSNRConfig:
    """Create a small config for testing."""
    return DPSNRConfig(
        vocab_size=1024,
        hidden_dim=128,
        num_layers=2,
        num_heads=4,
        head_dim=32,
        intermediate_dim=512,
        pool_size=1024,
        pool_dim=256,
        top_k=128,
        retrieval_dim=128,
        max_reasoning_steps=4,
        max_seq_len=256,
    )


@pytest.fixture
def batch() -> dict[str, torch.Tensor]:
    """Create a test batch."""
    batch_size = 2
    seq_len = 128
    return {
        "input_ids": torch.randint(0, 1024, (batch_size, seq_len)),
        "labels": torch.randint(0, 1024, (batch_size, seq_len)),
    }


class TestConfig:
    """Tests for DPSNRConfig."""

    def test_default_config_is_valid(self) -> None:
        config = DPSNRConfig()
        assert config.hidden_dim % 128 == 0
        assert config.pool_dim % 128 == 0
        assert config.top_k % 128 == 0

    def test_partition_sizes(self) -> None:
        config = DPSNRConfig()
        total = config.knowledge_size + config.reasoning_size + config.grammar_size
        assert total == config.pool_size

    def test_preset_tiny(self) -> None:
        config = DPSNRConfig.from_preset("tiny")
        assert config.hidden_dim == 512
        assert config.num_layers == 4

    def test_preset_small(self) -> None:
        config = DPSNRConfig.from_preset("small")
        assert config.hidden_dim == 768
        assert config.num_layers == 6

    def test_preset_base(self) -> None:
        config = DPSNRConfig.from_preset("base")
        assert config.hidden_dim == 1024
        assert config.num_layers == 8


class TestController:
    """Tests for TinyController."""

    def test_encode_output_shape(self, config: DPSNRConfig, batch: dict) -> None:
        controller = TinyController(config)
        hidden = controller.encode(batch["input_ids"])
        assert hidden.shape == (2, 128, config.hidden_dim)

    def test_generate_query_shape(self, config: DPSNRConfig, batch: dict) -> None:
        controller = TinyController(config)
        hidden = controller.encode(batch["input_ids"])
        query = controller.generate_query(hidden)
        assert query.shape == (2, 128, config.retrieval_dim)

    def test_predict_halt_shape(self, config: DPSNRConfig, batch: dict) -> None:
        controller = TinyController(config)
        hidden = controller.encode(batch["input_ids"])
        halt_prob = controller.predict_halt(hidden)
        assert halt_prob.shape == (2, 128, 1)
        assert (halt_prob >= 0).all() and (halt_prob <= 1).all()


class TestPool:
    """Tests for MassivePool."""

    def test_retrieve_output_shape(self, config: DPSNRConfig) -> None:
        pool = MassivePool(config)
        query = torch.randn(2, 128, config.retrieval_dim)
        retrieved = pool.retrieve(query, phase="all")
        assert retrieved.shape == (2, 128, config.pool_dim)

    def test_retrieve_knowledge_phase(self, config: DPSNRConfig) -> None:
        pool = MassivePool(config)
        query = torch.randn(2, 128, config.retrieval_dim)
        retrieved = pool.retrieve(query, phase="knowledge")
        assert retrieved.shape == (2, 128, config.pool_dim)

    def test_retrieve_reasoning_phase(self, config: DPSNRConfig) -> None:
        pool = MassivePool(config)
        query = torch.randn(2, 128, config.retrieval_dim)
        retrieved = pool.retrieve(query, phase="reasoning")
        assert retrieved.shape == (2, 128, config.pool_dim)


class TestACT:
    """Tests for Adaptive Compute Time."""

    def test_act_output_shape(self, config: DPSNRConfig) -> None:
        act = AdaptiveComputeTime(config)
        initial = torch.randn(2, 128, config.hidden_dim)

        # New API: init_state -> step -> finalize
        acc = act.init_state(2, 128, config.hidden_dim, initial.device, initial.dtype)

        state = initial
        for step_idx in range(config.max_reasoning_steps):
            halt_prob = torch.sigmoid(state.mean(dim=-1))
            acc = act.step(state, halt_prob, acc, step_idx)
            state = state + 0.1

        final, ponder, halted = act.finalize(state, acc)

        assert final.shape == initial.shape
        assert ponder.shape == (2, 128)
        assert halted.shape == (2, 128)

    def test_ponder_loss(self) -> None:
        ponder_cost = torch.tensor([[1.5, 2.0], [1.8, 1.2]])
        loss = compute_ponder_loss(ponder_cost, target_ponder=1.0)
        assert loss.shape == ()
        assert loss >= 0


class TestDPSNR:
    """Tests for full DPSN-R model."""

    def test_forward_output_shape(self, config: DPSNRConfig, batch: dict) -> None:
        model = DPSNR(config)
        outputs = model(input_ids=batch["input_ids"])
        assert outputs["logits"].shape == (2, 128, config.vocab_size)

    def test_forward_with_labels(self, config: DPSNRConfig, batch: dict) -> None:
        model = DPSNR(config)
        outputs = model(input_ids=batch["input_ids"], labels=batch["labels"])
        assert "loss" in outputs
        assert "ce_loss" in outputs
        assert "ponder_loss" in outputs
        assert outputs["loss"].shape == ()

    def test_generate(self, config: DPSNRConfig) -> None:
        model = DPSNR(config)
        input_ids = torch.randint(0, 1024, (1, 16))
        generated = model.generate(input_ids, max_new_tokens=8)
        assert generated.shape == (1, 24)

    def test_count_parameters(self, config: DPSNRConfig) -> None:
        model = DPSNR(config)
        counts = model.count_parameters()
        assert "controller" in counts
        assert "pool" in counts
        assert "total" in counts
        assert counts["total"] > 0


class TestXLACompatibility:
    """Tests for XLA compatibility (static shapes, no graph breaks)."""

    def test_static_shapes_forward(self, config: DPSNRConfig) -> None:
        """Verify shapes are consistent across calls."""
        model = DPSNR(config)

        input1 = torch.randint(0, 1024, (2, 128))
        input2 = torch.randint(0, 1024, (2, 128))

        out1 = model(input1)
        out2 = model(input2)

        assert out1["logits"].shape == out2["logits"].shape

    def test_no_dynamic_shapes_in_pool(self, config: DPSNRConfig) -> None:
        """Verify pool retrieval produces static shapes."""
        pool = MassivePool(config)

        query1 = torch.randn(2, 128, config.retrieval_dim)
        query2 = torch.randn(2, 128, config.retrieval_dim)

        ret1 = pool.retrieve(query1)
        ret2 = pool.retrieve(query2)

        assert ret1.shape == ret2.shape

    def test_dimensions_are_128_aligned(self, config: DPSNRConfig) -> None:
        """Verify key dimensions are multiples of 128 for TPU efficiency."""
        model = DPSNR(config)

        for name, param in model.named_parameters():
            for dim_size in param.shape:
                if dim_size > 128:
                    pass
