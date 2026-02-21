"""
Tests for the ML training engine internals.

Tests core trainer functions: device detection, model loading,
LoRA target module detection, trainer creation, and GPU cleanup.
These run with mocked torch so they work on CI without a GPU.
"""

import pytest
from unittest.mock import patch, MagicMock, PropertyMock


# ── Device Detection ─────────────────────────────────────────────

class TestGetBestDevice:
    """Test get_best_device() helper for all platforms."""

    @patch("torch.cuda.is_available", return_value=True)
    def test_returns_cuda_when_available(self, _):
        from app.ml.trainer import get_best_device
        assert get_best_device() == "cuda"

    @patch("torch.cuda.is_available", return_value=False)
    def test_returns_mps_when_available(self, _):
        import torch
        # Mock MPS availability
        mock_mps = MagicMock()
        mock_mps.is_available.return_value = True
        with patch.object(torch.backends, "mps", mock_mps, create=True):
            from app.ml.trainer import get_best_device
            assert get_best_device() == "mps"

    @patch("torch.cuda.is_available", return_value=False)
    def test_returns_cpu_when_no_gpu(self, _):
        import torch
        # Mock MPS not available
        mock_mps = MagicMock()
        mock_mps.is_available.return_value = False
        with patch.object(torch.backends, "mps", mock_mps, create=True):
            from app.ml.trainer import get_best_device
            assert get_best_device() == "cpu"


# ── Model Dtype Selection ────────────────────────────────────────

class TestGetModelDtype:
    """Test dtype selection for different devices."""

    def test_cuda_returns_float16(self):
        import torch
        from app.ml.trainer import _get_model_dtype
        assert _get_model_dtype("cuda") == torch.float16

    def test_mps_returns_float32(self):
        import torch
        from app.ml.trainer import _get_model_dtype
        assert _get_model_dtype("mps") == torch.float32

    def test_cpu_returns_float32(self):
        import torch
        from app.ml.trainer import _get_model_dtype
        assert _get_model_dtype("cpu") == torch.float32


# ── LoRA Target Modules ──────────────────────────────────────────

class TestFindTargetModules:
    """Test auto-detection of LoRA target modules."""

    def test_finds_common_modules(self):
        from app.ml.trainer import _find_target_modules

        # Create a mock model with LLaMA-style module names
        mock_model = MagicMock()
        mock_model.named_modules.return_value = [
            ("model.layers.0.self_attn.q_proj", MagicMock()),
            ("model.layers.0.self_attn.k_proj", MagicMock()),
            ("model.layers.0.self_attn.v_proj", MagicMock()),
            ("model.layers.0.self_attn.o_proj", MagicMock()),
            ("model.layers.0.mlp.gate_proj", MagicMock()),
            ("model.layers.0.mlp.up_proj", MagicMock()),
            ("model.layers.0.mlp.down_proj", MagicMock()),
        ]

        targets = _find_target_modules(mock_model)
        assert "q_proj" in targets
        assert "v_proj" in targets
        assert len(targets) > 0

    def test_fallback_to_all_linear(self):
        from app.ml.trainer import _find_target_modules

        # Model with no common module names
        mock_model = MagicMock()
        mock_model.named_modules.return_value = [
            ("model.layers.0.custom_layer", MagicMock()),
        ]

        targets = _find_target_modules(mock_model)
        assert "all-linear" in targets


# ── Training Metrics ─────────────────────────────────────────────

class TestTrainingMetrics:
    """Test the shared TrainingMetrics state object."""

    def test_local_mode_defaults(self):
        from app.ml.trainer import TrainingMetrics
        metrics = TrainingMetrics()
        assert metrics.status == "initializing"
        assert metrics.current_step == 0
        assert metrics.current_loss is None

    def test_set_and_get_fields(self):
        from app.ml.trainer import TrainingMetrics
        metrics = TrainingMetrics()
        metrics.status = "running"
        metrics.current_loss = 0.5
        assert metrics.status == "running"
        assert metrics.current_loss == 0.5

    def test_pause_stop_flags(self):
        from app.ml.trainer import TrainingMetrics
        metrics = TrainingMetrics()
        assert metrics.pause_requested is False
        assert metrics.stop_requested is False
        metrics.pause_requested = True
        assert metrics.pause_requested is True
        metrics.stop_requested = True
        assert metrics.stop_requested is True

    def test_log_history_append(self):
        from app.ml.trainer import TrainingMetrics
        metrics = TrainingMetrics()
        metrics.append_log({"step": 1, "loss": 0.5})
        metrics.append_log({"step": 2, "loss": 0.4})
        history = metrics.get_log_history()
        assert len(history) == 2
        assert history[0]["step"] == 1

    def test_log_history_pruning(self):
        from app.ml.trainer import TrainingMetrics
        metrics = TrainingMetrics()
        # Add more than the pruning threshold
        for i in range(10001):
            metrics.append_log({"step": i})
        history = metrics.get_log_history()
        assert len(history) <= 5000  # Pruned to last 5000


# ── GPU Cleanup ──────────────────────────────────────────────────

class TestCleanupGpu:
    """Test GPU memory cleanup for different devices."""

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.empty_cache")
    def test_cuda_cleanup(self, mock_empty, _):
        from app.ml.trainer import _cleanup_gpu
        _cleanup_gpu()
        mock_empty.assert_called_once()

    @patch("torch.cuda.is_available", return_value=False)
    def test_cpu_cleanup_no_error(self, _):
        from app.ml.trainer import _cleanup_gpu
        # Should not raise
        _cleanup_gpu()


# ── Password Validation ─────────────────────────────────────────

class TestPasswordStrength:
    """Test that password strength validation works correctly."""

    def test_strong_password_accepted(self):
        from app.schemas import RegisterRequest
        req = RegisterRequest(email="t@t.com", password="TestPass1!", display_name="User")
        assert req.password == "TestPass1!"

    def test_no_uppercase_rejected(self):
        from app.schemas import RegisterRequest
        with pytest.raises(Exception):
            RegisterRequest(email="t@t.com", password="testpass1!")

    def test_no_digit_rejected(self):
        from app.schemas import RegisterRequest
        with pytest.raises(Exception):
            RegisterRequest(email="t@t.com", password="TestPass!!")

    def test_no_special_char_rejected(self):
        from app.schemas import RegisterRequest
        with pytest.raises(Exception):
            RegisterRequest(email="t@t.com", password="TestPass11")

    def test_too_short_rejected(self):
        from app.schemas import RegisterRequest
        with pytest.raises(Exception):
            RegisterRequest(email="t@t.com", password="Te1!")
