# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
Unit tests for PyTorch to ONNX conversion functionality.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Check if torch is available
try:
    import torch
    import torch.nn as nn

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


class SimpleTestModel(nn.Module):
    """Simple model for testing conversion."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, 10)

    def forward(self, x):
        x = self.relu(self.conv(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="PyTorch not installed")
class TestPyTorchConversion:
    """Tests for PyTorch to ONNX conversion."""

    def test_torchscript_model_conversion(self, tmp_path):
        """TorchScript models should convert successfully."""
        # Create and save a TorchScript model
        model = SimpleTestModel()
        model.eval()
        dummy_input = torch.randn(1, 3, 32, 32)
        traced = torch.jit.trace(model, dummy_input)

        pt_path = tmp_path / "model.pt"
        torch.jit.save(traced, str(pt_path))

        # Import the conversion function
        sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
        from util.model_inspect import _convert_pytorch_to_onnx
        import logging

        logger = logging.getLogger("test")

        # Convert
        onnx_path, temp_file = _convert_pytorch_to_onnx(
            pt_path,
            input_shape_str="1,3,32,32",
            output_path=tmp_path / "output.onnx",
            opset_version=17,
            logger=logger,
        )

        assert onnx_path is not None
        assert onnx_path.exists()
        assert onnx_path.suffix == ".onnx"

    def test_conversion_requires_input_shape(self, tmp_path):
        """Conversion should fail without input shape."""
        model = SimpleTestModel()
        model.eval()
        dummy_input = torch.randn(1, 3, 32, 32)
        traced = torch.jit.trace(model, dummy_input)

        pt_path = tmp_path / "model.pt"
        torch.jit.save(traced, str(pt_path))

        sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
        from util.model_inspect import _convert_pytorch_to_onnx
        import logging

        logger = logging.getLogger("test")

        # Convert without input shape
        onnx_path, _ = _convert_pytorch_to_onnx(
            pt_path,
            input_shape_str=None,  # No input shape
            output_path=None,
            opset_version=17,
            logger=logger,
        )

        assert onnx_path is None

    def test_conversion_invalid_input_shape(self, tmp_path):
        """Conversion should fail with invalid input shape format."""
        model = SimpleTestModel()
        model.eval()
        dummy_input = torch.randn(1, 3, 32, 32)
        traced = torch.jit.trace(model, dummy_input)

        pt_path = tmp_path / "model.pt"
        torch.jit.save(traced, str(pt_path))

        sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
        from util.model_inspect import _convert_pytorch_to_onnx
        import logging

        logger = logging.getLogger("test")

        # Convert with invalid input shape
        onnx_path, _ = _convert_pytorch_to_onnx(
            pt_path,
            input_shape_str="invalid,shape",
            output_path=None,
            opset_version=17,
            logger=logger,
        )

        assert onnx_path is None

    def test_conversion_nonexistent_file(self, tmp_path):
        """Conversion should fail gracefully for nonexistent file."""
        sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
        from util.model_inspect import _convert_pytorch_to_onnx
        import logging

        logger = logging.getLogger("test")

        onnx_path, _ = _convert_pytorch_to_onnx(
            tmp_path / "nonexistent.pt",
            input_shape_str="1,3,32,32",
            output_path=None,
            opset_version=17,
            logger=logger,
        )

        assert onnx_path is None

    def test_conversion_temp_file_cleanup(self, tmp_path):
        """Temp file should be created when no output path specified."""
        model = SimpleTestModel()
        model.eval()
        dummy_input = torch.randn(1, 3, 32, 32)
        traced = torch.jit.trace(model, dummy_input)

        pt_path = tmp_path / "model.pt"
        torch.jit.save(traced, str(pt_path))

        sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
        from util.model_inspect import _convert_pytorch_to_onnx
        import logging

        logger = logging.getLogger("test")

        # Convert without output path (should create temp file)
        onnx_path, temp_file = _convert_pytorch_to_onnx(
            pt_path,
            input_shape_str="1,3,32,32",
            output_path=None,  # No output path
            opset_version=17,
            logger=logger,
        )

        assert onnx_path is not None
        assert temp_file is not None
        assert onnx_path.exists()

        # Clean up
        onnx_path.unlink()

    def test_state_dict_not_supported(self, tmp_path):
        """State dict models should fail with helpful error."""
        model = SimpleTestModel()

        # Save as state_dict (not full model)
        pt_path = tmp_path / "weights.pth"
        torch.save(model.state_dict(), pt_path)

        sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
        from util.model_inspect import _convert_pytorch_to_onnx
        import logging

        logger = logging.getLogger("test")

        onnx_path, _ = _convert_pytorch_to_onnx(
            pt_path,
            input_shape_str="1,3,32,32",
            output_path=None,
            opset_version=17,
            logger=logger,
        )

        assert onnx_path is None


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="PyTorch not installed")
class TestUltralyticsMetadataExtraction:
    """Tests for Ultralytics metadata extraction."""

    def test_extraction_without_ultralytics(self, tmp_path):
        """Should return None gracefully when ultralytics not available."""
        sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
        from util.model_inspect import _extract_ultralytics_metadata
        import logging

        logger = logging.getLogger("test")

        # Mock ultralytics not being available
        with patch.dict("sys.modules", {"ultralytics": None}):
            result = _extract_ultralytics_metadata(tmp_path / "fake.pt", logger)
            # Should return None, not crash
            assert result is None or isinstance(result, dict)


class TestDatasetInfo:
    """Tests for DatasetInfo dataclass."""

    def test_dataset_info_creation(self):
        """DatasetInfo should be created with expected fields."""
        from ..report import DatasetInfo

        info = DatasetInfo(
            task="detect",
            num_classes=5,
            class_names=["cat", "dog", "bird", "fish", "car"],
            source="ultralytics",
        )

        assert info.task == "detect"
        assert info.num_classes == 5
        assert len(info.class_names) == 5
        assert info.source == "ultralytics"

    def test_dataset_info_defaults(self):
        """DatasetInfo should have sensible defaults."""
        from ..report import DatasetInfo

        info = DatasetInfo()

        assert info.task is None
        assert info.num_classes is None
        assert info.class_names == []
        assert info.source is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
