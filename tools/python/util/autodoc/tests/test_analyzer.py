# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
Unit tests for the analyzer module (parameter counting, FLOP estimation, memory estimates).

These tests use programmatically-created tiny ONNX models to ensure deterministic,
reproducible test results without external dependencies.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

# Import the modules under test
import sys

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from autodoc.analyzer import MetricsEngine, ONNXGraphLoader


def create_simple_conv_model() -> onnx.ModelProto:
    """Create a minimal Conv model for testing."""
    # Input: [1, 3, 8, 8]
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 8, 8])

    # Weight: [16, 3, 3, 3] = 16 * 3 * 3 * 3 = 432 params
    W = helper.make_tensor(
        "W",
        TensorProto.FLOAT,
        [16, 3, 3, 3],
        np.random.randn(16, 3, 3, 3).astype(np.float32).flatten().tolist(),
    )

    # Bias: [16] = 16 params
    B = helper.make_tensor(
        "B",
        TensorProto.FLOAT,
        [16],
        np.zeros(16, dtype=np.float32).tolist(),
    )

    # Output: [1, 16, 6, 6]
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 16, 6, 6])

    conv_node = helper.make_node(
        "Conv",
        inputs=["X", "W", "B"],
        outputs=["Y"],
        kernel_shape=[3, 3],
        pads=[0, 0, 0, 0],
    )

    graph = helper.make_graph(
        [conv_node],
        "conv_test",
        [X],
        [Y],
        [W, B],
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    return model


def create_matmul_model() -> onnx.ModelProto:
    """Create a minimal MatMul model for testing."""
    # A: [2, 4, 8]
    A = helper.make_tensor_value_info("A", TensorProto.FLOAT, [2, 4, 8])

    # B weight: [8, 16] = 128 params
    B = helper.make_tensor(
        "B",
        TensorProto.FLOAT,
        [8, 16],
        np.random.randn(8, 16).astype(np.float32).flatten().tolist(),
    )

    # Output: [2, 4, 16]
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 4, 16])

    matmul_node = helper.make_node(
        "MatMul",
        inputs=["A", "B"],
        outputs=["Y"],
    )

    graph = helper.make_graph(
        [matmul_node],
        "matmul_test",
        [A],
        [Y],
        [B],
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    return model


def create_gemm_model() -> onnx.ModelProto:
    """Create a minimal Gemm model for testing."""
    # A: [4, 8]
    A = helper.make_tensor_value_info("A", TensorProto.FLOAT, [4, 8])

    # B weight: [8, 16] = 128 params
    B = helper.make_tensor(
        "B",
        TensorProto.FLOAT,
        [8, 16],
        np.random.randn(8, 16).astype(np.float32).flatten().tolist(),
    )

    # C bias: [16] = 16 params
    C = helper.make_tensor(
        "C",
        TensorProto.FLOAT,
        [16],
        np.zeros(16, dtype=np.float32).tolist(),
    )

    # Output: [4, 16]
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [4, 16])

    gemm_node = helper.make_node(
        "Gemm",
        inputs=["A", "B", "C"],
        outputs=["Y"],
    )

    graph = helper.make_graph(
        [gemm_node],
        "gemm_test",
        [A],
        [Y],
        [B, C],
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    return model


def create_relu_model() -> onnx.ModelProto:
    """Create a minimal ReLU model (no parameters) for testing."""
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 8, 8])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3, 8, 8])

    relu_node = helper.make_node("Relu", inputs=["X"], outputs=["Y"])

    graph = helper.make_graph([relu_node], "relu_test", [X], [Y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    return model


def create_conv_bn_relu_model() -> onnx.ModelProto:
    """Create a Conv-BatchNorm-ReLU sequence for pattern testing."""
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 8, 8])

    # Conv weights: [16, 3, 3, 3] = 432 params
    W = helper.make_tensor(
        "W", TensorProto.FLOAT, [16, 3, 3, 3],
        np.random.randn(16, 3, 3, 3).astype(np.float32).flatten().tolist(),
    )

    # BN params: scale, bias, mean, var each [16] = 64 params total
    scale = helper.make_tensor("scale", TensorProto.FLOAT, [16], np.ones(16, dtype=np.float32).tolist())
    bias = helper.make_tensor("bias", TensorProto.FLOAT, [16], np.zeros(16, dtype=np.float32).tolist())
    mean = helper.make_tensor("mean", TensorProto.FLOAT, [16], np.zeros(16, dtype=np.float32).tolist())
    var = helper.make_tensor("var", TensorProto.FLOAT, [16], np.ones(16, dtype=np.float32).tolist())

    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 16, 6, 6])

    conv_out = "conv_out"
    bn_out = "bn_out"

    conv_node = helper.make_node("Conv", ["X", "W"], [conv_out], kernel_shape=[3, 3])
    bn_node = helper.make_node(
        "BatchNormalization",
        [conv_out, "scale", "bias", "mean", "var"],
        [bn_out],
    )
    relu_node = helper.make_node("Relu", [bn_out], ["Y"])

    graph = helper.make_graph(
        [conv_node, bn_node, relu_node],
        "conv_bn_relu_test",
        [X],
        [Y],
        [W, scale, bias, mean, var],
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    return model


class TestONNXGraphLoader:
    """Tests for ONNXGraphLoader class."""

    def test_load_conv_model(self):
        """Test loading a simple Conv model."""
        model = create_simple_conv_model()

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx.save(model, f.name)
            model_path = Path(f.name)

        try:
            loader = ONNXGraphLoader()
            loaded_model, graph_info = loader.load(model_path)

            assert graph_info.num_nodes == 1
            assert len(graph_info.inputs) == 1
            assert len(graph_info.outputs) == 1
            assert len(graph_info.initializers) == 2  # W and B
            assert "Conv" in graph_info.op_type_counts
        finally:
            model_path.unlink()

    def test_load_extracts_shapes(self):
        """Test that shape information is extracted correctly."""
        model = create_simple_conv_model()

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx.save(model, f.name)
            model_path = Path(f.name)

        try:
            loader = ONNXGraphLoader()
            _, graph_info = loader.load(model_path)

            assert "X" in graph_info.input_shapes
            assert graph_info.input_shapes["X"] == [1, 3, 8, 8]
            assert "Y" in graph_info.output_shapes
            assert graph_info.output_shapes["Y"] == [1, 16, 6, 6]
        finally:
            model_path.unlink()


class TestMetricsEngine:
    """Tests for MetricsEngine class."""

    def test_count_parameters_conv(self):
        """Test parameter counting for Conv model."""
        model = create_simple_conv_model()

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx.save(model, f.name)
            model_path = Path(f.name)

        try:
            loader = ONNXGraphLoader()
            _, graph_info = loader.load(model_path)

            engine = MetricsEngine()
            params = engine.count_parameters(graph_info)

            # W: 16*3*3*3 = 432, B: 16 = 448 total
            assert params.total == 448
            assert "Conv" in params.by_op_type
        finally:
            model_path.unlink()

    def test_count_parameters_matmul(self):
        """Test parameter counting for MatMul model."""
        model = create_matmul_model()

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx.save(model, f.name)
            model_path = Path(f.name)

        try:
            loader = ONNXGraphLoader()
            _, graph_info = loader.load(model_path)

            engine = MetricsEngine()
            params = engine.count_parameters(graph_info)

            # B: 8*16 = 128
            assert params.total == 128
        finally:
            model_path.unlink()

    def test_count_parameters_no_weights(self):
        """Test parameter counting for model without weights."""
        model = create_relu_model()

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx.save(model, f.name)
            model_path = Path(f.name)

        try:
            loader = ONNXGraphLoader()
            _, graph_info = loader.load(model_path)

            engine = MetricsEngine()
            params = engine.count_parameters(graph_info)

            assert params.total == 0
        finally:
            model_path.unlink()

    def test_estimate_flops_conv(self):
        """Test FLOP estimation for Conv model."""
        model = create_simple_conv_model()

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx.save(model, f.name)
            model_path = Path(f.name)

        try:
            loader = ONNXGraphLoader()
            _, graph_info = loader.load(model_path)

            engine = MetricsEngine()
            flops = engine.estimate_flops(graph_info)

            # Conv FLOPs: 2 * K_h * K_w * C_in * C_out * H_out * W_out
            # = 2 * 3 * 3 * 3 * 16 * 6 * 6 + bias = 31,104 + 576 = 31,680
            expected_flops = 2 * 3 * 3 * 3 * 16 * 6 * 6 + 16 * 6 * 6
            assert flops.total == expected_flops
        finally:
            model_path.unlink()

    def test_estimate_flops_matmul(self):
        """Test FLOP estimation for MatMul model."""
        model = create_matmul_model()

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx.save(model, f.name)
            model_path = Path(f.name)

        try:
            loader = ONNXGraphLoader()
            _, graph_info = loader.load(model_path)

            engine = MetricsEngine()
            flops = engine.estimate_flops(graph_info)

            # MatMul FLOPs: 2 * batch * M * N * K
            # = 2 * 2 * 4 * 16 * 8 = 2048
            expected_flops = 2 * 2 * 4 * 16 * 8
            assert flops.total == expected_flops
        finally:
            model_path.unlink()

    def test_estimate_memory(self):
        """Test memory estimation."""
        model = create_simple_conv_model()

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx.save(model, f.name)
            model_path = Path(f.name)

        try:
            loader = ONNXGraphLoader()
            _, graph_info = loader.load(model_path)

            engine = MetricsEngine()
            memory = engine.estimate_memory(graph_info)

            # Model size: 448 params * 4 bytes = 1792 bytes
            assert memory.model_size_bytes == 448 * 4
            assert memory.peak_activation_bytes >= 0
        finally:
            model_path.unlink()


class TestMetricsEngineEdgeCases:
    """Edge case tests for MetricsEngine."""

    def test_gemm_with_bias(self):
        """Test Gemm with bias adds extra FLOPs."""
        model = create_gemm_model()

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx.save(model, f.name)
            model_path = Path(f.name)

        try:
            loader = ONNXGraphLoader()
            _, graph_info = loader.load(model_path)

            engine = MetricsEngine()
            params = engine.count_parameters(graph_info)
            flops = engine.estimate_flops(graph_info)

            # B: 8*16=128, C: 16 = 144 total params
            assert params.total == 144

            # Gemm FLOPs: 2*M*N*K + M*N (bias) = 2*4*16*8 + 4*16 = 1024 + 64 = 1088
            expected_flops = 2 * 4 * 16 * 8 + 4 * 16
            assert flops.total == expected_flops
        finally:
            model_path.unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
