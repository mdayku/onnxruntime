# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
Unit tests for the risks module (risk signal detection).
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from autodoc.analyzer import MetricsEngine, ONNXGraphLoader
from autodoc.patterns import PatternAnalyzer
from autodoc.risks import RiskAnalyzer


def create_deep_no_skip_model(num_layers: int = 60) -> onnx.ModelProto:
    """Create a deep model without skip connections."""
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 64])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 64])

    nodes = []
    initializers = []
    prev_output = "X"

    for i in range(num_layers):
        W = helper.make_tensor(
            f"W_{i}",
            TensorProto.FLOAT,
            [64, 64],
            np.random.randn(64, 64).astype(np.float32).flatten().tolist(),
        )
        initializers.append(W)

        out_name = f"layer_{i}" if i < num_layers - 1 else "Y"
        matmul = helper.make_node(
            "MatMul", [prev_output, f"W_{i}"], [out_name], name=f"matmul_{i}"
        )
        nodes.append(matmul)
        prev_output = out_name

    graph = helper.make_graph(nodes, "deep_no_skip", [X], [Y], initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    return model


def create_dynamic_input_model() -> onnx.ModelProto:
    """Create a model with dynamic input shapes."""
    # Dynamic batch and sequence length
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, ["batch", "seq", 64])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, ["batch", "seq", 64])

    relu = helper.make_node("Relu", ["X"], ["Y"])

    graph = helper.make_graph([relu], "dynamic_test", [X], [Y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    return model


def create_small_model() -> onnx.ModelProto:
    """Create a tiny model that shouldn't trigger any risk signals."""
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 8])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 8])

    W = helper.make_tensor(
        "W",
        TensorProto.FLOAT,
        [8, 8],
        np.random.randn(8, 8).astype(np.float32).flatten().tolist(),
    )

    matmul = helper.make_node("MatMul", ["X", "W"], ["out1"])
    relu = helper.make_node("Relu", ["out1"], ["Y"])

    graph = helper.make_graph([matmul, relu], "small_test", [X], [Y], [W])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    return model


def create_no_activation_model() -> onnx.ModelProto:
    """Create a model with multiple linear layers but no activations.

    Creates 25 MatMul nodes to exceed the MIN_NODES_FOR_DEPTH_CHECK threshold (20).
    """
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 64])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 64])

    nodes = []
    initializers = []
    prev_output = "X"

    # Create 25 MatMul layers with no activations (exceeds 20-node threshold)
    for i in range(25):
        W = helper.make_tensor(
            f"W_{i}",
            TensorProto.FLOAT,
            [64, 64],
            np.random.randn(64, 64).astype(np.float32).flatten().tolist(),
        )
        initializers.append(W)

        out_name = f"layer_{i}" if i < 24 else "Y"
        matmul = helper.make_node(
            "MatMul", [prev_output, f"W_{i}"], [out_name], name=f"matmul_{i}"
        )
        nodes.append(matmul)
        prev_output = out_name

    graph = helper.make_graph(nodes, "no_activation", [X], [Y], initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    return model


class TestRiskAnalyzer:
    """Tests for RiskAnalyzer class."""

    def test_no_risks_for_small_model(self):
        """Small models shouldn't trigger risk signals."""
        model = create_small_model()

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx.save(model, f.name)
            model_path = Path(f.name)

        try:
            loader = ONNXGraphLoader()
            _, graph_info = loader.load(model_path)

            # Compute FLOPs for risk analysis
            engine = MetricsEngine()
            engine.estimate_flops(graph_info)

            pattern_analyzer = PatternAnalyzer()
            blocks = pattern_analyzer.group_into_blocks(graph_info)

            risk_analyzer = RiskAnalyzer()
            signals = risk_analyzer.analyze(graph_info, blocks)

            # Small model shouldn't trigger depth or bottleneck signals
            signal_ids = [s.id for s in signals]
            assert "no_skip_connections" not in signal_ids
            assert "compute_bottleneck" not in signal_ids
        finally:
            model_path.unlink()

    def test_detect_deep_without_skips(self):
        """Detect deep networks without skip connections."""
        model = create_deep_no_skip_model(num_layers=60)

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx.save(model, f.name)
            model_path = Path(f.name)

        try:
            loader = ONNXGraphLoader()
            _, graph_info = loader.load(model_path)

            pattern_analyzer = PatternAnalyzer()
            blocks = pattern_analyzer.group_into_blocks(graph_info)

            risk_analyzer = RiskAnalyzer()
            signals = risk_analyzer.analyze(graph_info, blocks)

            signal_ids = [s.id for s in signals]
            assert "no_skip_connections" in signal_ids
        finally:
            model_path.unlink()

    def test_detect_dynamic_shapes(self):
        """Detect dynamic input shapes."""
        model = create_dynamic_input_model()

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx.save(model, f.name)
            model_path = Path(f.name)

        try:
            loader = ONNXGraphLoader()
            _, graph_info = loader.load(model_path)

            pattern_analyzer = PatternAnalyzer()
            blocks = pattern_analyzer.group_into_blocks(graph_info)

            risk_analyzer = RiskAnalyzer()
            signals = risk_analyzer.analyze(graph_info, blocks)

            signal_ids = [s.id for s in signals]
            assert "dynamic_input_shapes" in signal_ids
        finally:
            model_path.unlink()

    def test_detect_no_activations(self):
        """Detect models without activation functions."""
        model = create_no_activation_model()

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx.save(model, f.name)
            model_path = Path(f.name)

        try:
            loader = ONNXGraphLoader()
            _, graph_info = loader.load(model_path)

            pattern_analyzer = PatternAnalyzer()
            blocks = pattern_analyzer.group_into_blocks(graph_info)

            risk_analyzer = RiskAnalyzer()
            signals = risk_analyzer.analyze(graph_info, blocks)

            signal_ids = [s.id for s in signals]
            # Should detect either no_activations or no_skip_connections
            assert "no_activations" in signal_ids or "no_skip_connections" in signal_ids
        finally:
            model_path.unlink()


class TestRiskSignalSeverity:
    """Tests for risk signal severity levels."""

    def test_severity_levels(self):
        """Verify severity levels are set correctly."""
        model = create_deep_no_skip_model(num_layers=60)

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx.save(model, f.name)
            model_path = Path(f.name)

        try:
            loader = ONNXGraphLoader()
            _, graph_info = loader.load(model_path)

            pattern_analyzer = PatternAnalyzer()
            blocks = pattern_analyzer.group_into_blocks(graph_info)

            risk_analyzer = RiskAnalyzer()
            signals = risk_analyzer.analyze(graph_info, blocks)

            for signal in signals:
                assert signal.severity in ("info", "warning", "high")
                assert signal.id  # Should have an ID
                assert signal.description  # Should have a description
        finally:
            model_path.unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
