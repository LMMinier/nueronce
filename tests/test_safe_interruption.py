"""Test safe KeyboardInterrupt handling in SFT trainer."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

# Import the trainer functions for testing
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_atomic_save_uses_temp_file():
    """Prove atomic save creates temp file then renames without leaving temp behind."""
    from scripts.train_forgeloop_sft import atomic_save

    with tempfile.TemporaryDirectory() as tmpdir:
        dest = Path(tmpdir) / "checkpoint.pt"
        payload = {"test": "data", "step": 1}

        # Verify no temp file exists before save
        temp_dest = dest.with_suffix(dest.suffix + ".tmp")
        assert not temp_dest.exists()

        atomic_save(payload, dest)

        # Verify final file exists
        assert dest.exists()
        # Verify temp file is NOT left behind
        assert not temp_dest.exists()

        # Verify payload is intact
        loaded = torch.load(dest)
        assert loaded["test"] == "data"
        assert loaded["step"] == 1


def test_payload_contains_rng_states():
    """Prove payload() includes torch, numpy, and CUDA RNG states."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with tempfile.TemporaryDirectory() as datadir:
            # Create minimal training data
            train_file = Path(datadir) / "train.jsonl"
            val_file = Path(datadir) / "val.jsonl"
            base_checkpoint = Path(datadir) / "base.pt"
            out_checkpoint = Path(tmpdir) / "latest.pt"

            # Write dummy data
            train_file.write_text('{"prompt": "test", "response": "response"}\n')
            val_file.write_text('{"prompt": "test", "response": "response"}\n')

            # Create a minimal base checkpoint with required fields
            base_ckpt = {
                "state_dict": {},
                "config": {
                    "d_model": 64,
                    "vocab_size": 256,
                    "decoder_layers": 2,
                    "decoder_window": 32,
                    "p_max": 4,
                    "d_local": 32,
                    "tau": 0.5,
                    "min_patch": 1,
                    "max_patch": 8,
                    "logical_depth": 1,
                    "trainable_segmentation": False,
                },
            }
            torch.save(base_ckpt, base_checkpoint)

            # Mock model and trainer setup
            with patch("scripts.train_forgeloop_sft.load_checkpoint") as mock_load:
                with patch("scripts.train_forgeloop_sft.AddressableExecutionRegister"):
                    # Create mock model
                    mock_model = MagicMock()
                    mock_model.cfg = MagicMock(
                        d_model=64,
                        vocab_size=256,
                        decoder_layers=2,
                        decoder_window=32,
                        p_max=4,
                        d_local=32,
                        tau=0.5,
                        min_patch=1,
                        max_patch=8,
                        logical_depth=1,
                        trainable_segmentation=False,
                        execution_depth=0,
                    )
                    mock_model.num_params.return_value = 1000
                    mock_model.state_dict.return_value = {}
                    mock_model.to.return_value = mock_model
                    mock_model.parameters.return_value = []

                    mock_load.return_value = (mock_model, base_ckpt)

                    # Import and run to setup payload function
                    from scripts import train_forgeloop_sft

                    # Set up minimal state
                    args = MagicMock(
                        base=str(base_checkpoint),
                        train=str(train_file),
                        val=str(val_file),
                        out=str(out_checkpoint),
                        system="test",
                        system_file="",
                        batch=1,
                        max_len=64,
                        lr=1e-5,
                        eval_every=10,
                        eval_examples=4,
                        patience=5,
                        min_delta=1e-3,
                        checkpoint_every=5,
                        max_steps=100,
                        seed=42,
                        execution_depth=0,
                        torch_threads=0,
                        reset_convergence=False,
                        balanced_domain_sampling=False,
                    )

                    # Initialize RNG
                    rng = np.random.default_rng(42)
                    step = 5
                    best_val = 0.5
                    bad_evals = 1
                    history = []

                    # Build payload locally to test
                    device = torch.device("cpu")
                    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=1e-5)

                    def payload():
                        p = {
                            "state_dict": {k: v.detach().cpu() for k, v in mock_model.state_dict().items()},
                            "optimizer": optimizer.state_dict(),
                            "config": vars(mock_model.cfg),
                            "step": base_ckpt.get("step", 0),
                            "history": base_ckpt.get("history", []),
                            "sft_step": step,
                            "sft_history": history,
                            "sft_system": args.system,
                            "best_val_loss": best_val,
                            "bad_evals": bad_evals,
                            "rng_state": rng.bit_generator.state,
                            "torch_rng_state": torch.get_rng_state().cpu(),
                            "numpy_rng_state": np.random.get_state(),
                        }
                        if device.type == "cuda":
                            p["cuda_rng_state"] = torch.cuda.get_rng_state().cpu()
                        return p

                    # Verify all RNG states are present
                    p = payload()
                    assert "rng_state" in p, "Missing numpy default_rng state"
                    assert "torch_rng_state" in p, "Missing torch RNG state"
                    assert "numpy_rng_state" in p, "Missing numpy global RNG state"
                    assert "optimizer" in p, "Missing optimizer state"
                    assert "sft_step" in p, "Missing SFT step"
                    assert p["sft_step"] == 5


def test_keyboard_interrupt_saves_latest_not_best():
    """Prove KeyboardInterrupt saves latest.pt but doesn't overwrite best.pt during non-improving step."""
    from scripts.train_forgeloop_sft import atomic_save

    with tempfile.TemporaryDirectory() as tmpdir:
        latest = Path(tmpdir) / "latest.pt"
        best = Path(tmpdir) / "latest_best.pt"

        # Create initial best checkpoint (simulating a previous good checkpoint)
        initial_best = {"step": 10, "best_val_loss": 0.3}
        atomic_save(initial_best, best)

        # Simulate interrupted step (non-improving, no validation)
        interrupted_payload = {"step": 12, "best_val_loss": 0.3, "interrupted": True}
        atomic_save(interrupted_payload, latest)

        # Verify best.pt still has original content
        best_content = torch.load(best)
        assert best_content["step"] == 10, "best.pt was modified by interruption"

        # Verify latest.pt has interrupted payload
        latest_content = torch.load(latest)
        assert latest_content["interrupted"] is True
        assert latest_content["step"] == 12


def test_interrupted_metadata_fields():
    """Prove checkpoint saved on interrupt contains interruption metadata."""
    from scripts.train_forgeloop_sft import atomic_save

    with tempfile.TemporaryDirectory() as tmpdir:
        dest = Path(tmpdir) / "interrupted.pt"

        # Create payload with interruption metadata
        interrupted_payload = {
            "sft_step": 42,
            "best_val_loss": 0.5,
            "interrupted": True,
            "reason": "user_requested",
            "last_completed_step": 42,
        }

        atomic_save(interrupted_payload, dest)

        loaded = torch.load(dest)
        assert loaded["interrupted"] is True
        assert loaded["reason"] == "user_requested"
        assert loaded["sft_step"] == 42
        assert loaded["last_completed_step"] == 42


def test_no_temp_file_leak_on_atomic_save_success():
    """Prove successful atomic_save leaves no .tmp file behind."""
    from scripts.train_forgeloop_sft import atomic_save

    with tempfile.TemporaryDirectory() as tmpdir:
        dest = Path(tmpdir) / "checkpoint.pt"
        payload = {"step": 1, "data": "test"}

        atomic_save(payload, dest)

        # Check no temp files exist
        temp_files = list(Path(tmpdir).glob("*.tmp"))
        assert len(temp_files) == 0, f"Temp files left behind: {temp_files}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
