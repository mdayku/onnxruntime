# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
Core analysis engine for ONNX Autodoc.

This module provides:
- ONNXGraphLoader: Load ONNX models and extract graph structure
- MetricsEngine: Compute parameters, FLOPs, and memory estimates
- GraphInfo: Internal representation of the parsed graph
"""
from __future__ import annotations

import logging
import pathlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import onnx


# Standalone implementations that work without onnxruntime
def get_opsets_imported(model: onnx.ModelProto) -> dict:
    """Get the opsets imported by the model."""
    opsets = {}
    for entry in model.opset_import:
        domain = entry.domain or "ai.onnx"
        opsets[domain] = entry.version
    return opsets


def iterate_graph_per_node_func(graph, per_node_func, **func_args):
    """Iterate the graph including subgraphs calling the per_node_func for each node."""
    for node in graph.node:
        per_node_func(node, **func_args)
        for attr in node.attribute:
            if attr.HasField("g"):
                iterate_graph_per_node_func(attr.g, per_node_func, **func_args)


# Try to import ORT utilities for enhanced loading (with better shape inference)
try:
    from ..onnx_model_utils import ModelProtoWithShapeInfo

    _HAS_ORT_UTILS = True
except ImportError:
    _HAS_ORT_UTILS = False
    ModelProtoWithShapeInfo = None  # type: ignore


@dataclass
class NodeInfo:
    """Information about a single ONNX node."""

    name: str
    op_type: str
    domain: str
    inputs: list[str]
    outputs: list[str]
    attributes: dict[str, Any]
    # Computed during analysis
    param_count: int = 0
    flops: int = 0


@dataclass
class GraphInfo:
    """Parsed graph structure with extracted metadata."""

    name: str
    nodes: list[NodeInfo]
    inputs: list[str]
    outputs: list[str]
    initializers: dict[str, np.ndarray]  # name -> tensor
    value_shapes: dict[str, list[int | str]]  # name -> shape (may have symbolic dims)

    # Computed summaries
    num_nodes: int = 0
    input_shapes: dict[str, list[int | str]] = field(default_factory=dict)
    output_shapes: dict[str, list[int | str]] = field(default_factory=dict)
    op_type_counts: dict[str, int] = field(default_factory=dict)

    # Node lookup
    node_by_name: dict[str, NodeInfo] = field(default_factory=dict)
    node_by_output: dict[str, NodeInfo] = field(default_factory=dict)


@dataclass
class ParamCounts:
    """Parameter count breakdown."""

    total: int = 0
    trainable: int = 0  # Assumed: all initializers are trainable unless marked
    non_trainable: int = 0
    by_node: dict[str, int] = field(default_factory=dict)
    by_op_type: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "trainable": self.trainable,
            "non_trainable": self.non_trainable,
            "by_op_type": self.by_op_type,
        }


@dataclass
class FlopCounts:
    """FLOP estimate breakdown."""

    total: int = 0
    by_node: dict[str, int] = field(default_factory=dict)
    by_op_type: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "by_op_type": self.by_op_type,
        }


@dataclass
class MemoryEstimates:
    """Memory usage estimates."""

    model_size_bytes: int = 0  # Size of parameters/initializers
    peak_activation_bytes: int = 0  # Estimated peak activation memory (batch=1)
    per_layer_activation_bytes: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "model_size_bytes": self.model_size_bytes,
            "peak_activation_bytes": self.peak_activation_bytes,
        }


class ONNXGraphLoader:
    """
    Load ONNX models and extract graph structure.

    Handles shape inference and creates a GraphInfo representation
    suitable for analysis.
    """

    def __init__(self, logger: logging.Logger | None = None):
        self.logger = logger or logging.getLogger("autodoc.loader")

    def load(self, model_path: str | pathlib.Path) -> tuple[onnx.ModelProto, GraphInfo]:
        """
        Load an ONNX model and extract graph information.

        Args:
            model_path: Path to the ONNX model file.

        Returns:
            Tuple of (ModelProto, GraphInfo)
        """
        model_path = pathlib.Path(model_path)
        self.logger.debug(f"Loading model from {model_path}")

        # Use ORT's helper if available, otherwise fall back to onnx
        if _HAS_ORT_UTILS and ModelProtoWithShapeInfo is not None:
            wrapper = ModelProtoWithShapeInfo(model_path)
            model = wrapper.model_with_shape_info
        else:
            # Fallback: load with onnx and run shape inference
            model = onnx.load(str(model_path))
            try:
                model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
            except Exception as e:
                self.logger.warning(
                    f"Shape inference failed: {e}. Proceeding without shape info."
                )

        graph_info = self._extract_graph_info(model.graph, model)

        self.logger.debug(f"Loaded graph with {graph_info.num_nodes} nodes")
        return model, graph_info

    def _extract_graph_info(
        self, graph: onnx.GraphProto, model: onnx.ModelProto
    ) -> GraphInfo:
        """Extract GraphInfo from an ONNX GraphProto."""

        # Extract initializers (weights/biases)
        initializers = {}
        for init in graph.initializer:
            try:
                initializers[init.name] = onnx.numpy_helper.to_array(init)
            except Exception as e:
                self.logger.warning(f"Could not convert initializer {init.name}: {e}")
                # Store shape info at minimum
                initializers[init.name] = np.zeros(init.dims, dtype=np.float32)

        # Build value shape map from value_info, inputs, and outputs
        value_shapes = {}

        def _extract_shape(value_info: onnx.ValueInfoProto) -> list[int | str]:
            shape = []
            if value_info.type.HasField("tensor_type"):
                tensor_type = value_info.type.tensor_type
                if tensor_type.HasField("shape"):
                    for dim in tensor_type.shape.dim:
                        if dim.HasField("dim_value"):
                            shape.append(dim.dim_value)
                        elif dim.HasField("dim_param"):
                            shape.append(dim.dim_param)
                        else:
                            shape.append("?")
            return shape

        for vi in graph.input:
            value_shapes[vi.name] = _extract_shape(vi)
        for vi in graph.output:
            value_shapes[vi.name] = _extract_shape(vi)
        for vi in graph.value_info:
            value_shapes[vi.name] = _extract_shape(vi)

        # For initializers without explicit value_info, use their tensor shapes
        for name, arr in initializers.items():
            if name not in value_shapes:
                value_shapes[name] = list(arr.shape)

        # Extract nodes
        nodes = []
        op_type_counts: dict[str, int] = {}
        node_by_name: dict[str, NodeInfo] = {}
        node_by_output: dict[str, NodeInfo] = {}

        for node in graph.node:
            # Extract attributes
            attributes = {}
            for attr in node.attribute:
                if attr.HasField("i"):
                    attributes[attr.name] = attr.i
                elif attr.HasField("f"):
                    attributes[attr.name] = attr.f
                elif attr.HasField("s"):
                    attributes[attr.name] = (
                        attr.s.decode("utf-8") if isinstance(attr.s, bytes) else attr.s
                    )
                elif attr.ints:
                    attributes[attr.name] = list(attr.ints)
                elif attr.floats:
                    attributes[attr.name] = list(attr.floats)
                # Skip subgraphs and other complex types for now

            node_info = NodeInfo(
                name=node.name or f"unnamed_{len(nodes)}",
                op_type=node.op_type,
                domain=node.domain or "ai.onnx",
                inputs=list(node.input),
                outputs=list(node.output),
                attributes=attributes,
            )
            nodes.append(node_info)
            node_by_name[node_info.name] = node_info
            for output in node_info.outputs:
                node_by_output[output] = node_info

            # Count op types
            op_type_counts[node.op_type] = op_type_counts.get(node.op_type, 0) + 1

        # Build input/output shape maps (excluding initializers from inputs)
        input_names = [i.name for i in graph.input if i.name not in initializers]
        output_names = [o.name for o in graph.output]

        input_shapes = {name: value_shapes.get(name, []) for name in input_names}
        output_shapes = {name: value_shapes.get(name, []) for name in output_names}

        return GraphInfo(
            name=graph.name or "main",
            nodes=nodes,
            inputs=input_names,
            outputs=output_names,
            initializers=initializers,
            value_shapes=value_shapes,
            num_nodes=len(nodes),
            input_shapes=input_shapes,
            output_shapes=output_shapes,
            op_type_counts=op_type_counts,
            node_by_name=node_by_name,
            node_by_output=node_by_output,
        )


class MetricsEngine:
    """
    Compute model complexity metrics.

    Provides parameter counts, FLOP estimates, and memory estimates
    for ONNX graphs.
    """

    # FLOP multipliers per operation type
    # These are rough estimates; actual FLOPs depend on implementation
    FLOP_FORMULAS = {
        # Conv: 2 * K_h * K_w * C_in * C_out * H_out * W_out
        "Conv": "conv",
        # MatMul: 2 * M * N * K
        "MatMul": "matmul",
        "Gemm": "gemm",
        # Element-wise ops: N elements
        "Add": "elementwise",
        "Sub": "elementwise",
        "Mul": "elementwise",
        "Div": "elementwise",
        "Relu": "elementwise",
        "Sigmoid": "elementwise",
        "Tanh": "elementwise",
        "Sqrt": "elementwise",
        "Exp": "elementwise",
        "Log": "elementwise",
        # Softmax: ~5N (exp, sum, div)
        "Softmax": "softmax",
        # Reduction ops: N elements
        "ReduceMean": "elementwise",
        "ReduceSum": "elementwise",
        "ReduceMax": "elementwise",
        # Attention-related (rough estimates)
        "LayerNormalization": "layernorm",
        "BatchNormalization": "batchnorm",
    }

    def __init__(self, logger: logging.Logger | None = None):
        self.logger = logger or logging.getLogger("autodoc.metrics")

    def count_parameters(self, graph_info: GraphInfo) -> ParamCounts:
        """
        Count parameters in the model.

        Parameters are counted from initializers. All initializers are
        assumed trainable unless specifically marked otherwise.

        Args:
            graph_info: Parsed graph information.

        Returns:
            ParamCounts with total and per-node breakdowns.
        """
        counts = ParamCounts()

        # Count from initializers
        for name, tensor in graph_info.initializers.items():
            param_count = int(np.prod(tensor.shape)) if tensor.shape else 1
            counts.total += param_count
            counts.by_node[name] = param_count

            # Find which node uses this initializer
            for node in graph_info.nodes:
                if name in node.inputs:
                    counts.by_op_type[node.op_type] = (
                        counts.by_op_type.get(node.op_type, 0) + param_count
                    )
                    node.param_count += param_count
                    break

        # For now, assume all are trainable
        # Could be refined with graph analysis (e.g., constants, frozen layers)
        counts.trainable = counts.total
        counts.non_trainable = 0

        return counts

    def estimate_flops(self, graph_info: GraphInfo) -> FlopCounts:
        """
        Estimate FLOPs for each operation in the graph.

        Uses shape information to compute FLOPs. Falls back to
        rough estimates when shapes are unavailable.

        Args:
            graph_info: Parsed graph information.

        Returns:
            FlopCounts with total and per-node breakdowns.
        """
        counts = FlopCounts()

        for node in graph_info.nodes:
            flops = self._estimate_node_flops(node, graph_info)
            node.flops = flops
            counts.total += flops
            counts.by_node[node.name] = flops
            counts.by_op_type[node.op_type] = (
                counts.by_op_type.get(node.op_type, 0) + flops
            )

        return counts

    def _estimate_node_flops(self, node: NodeInfo, graph_info: GraphInfo) -> int:
        """Estimate FLOPs for a single node."""
        formula_type = self.FLOP_FORMULAS.get(node.op_type, "unknown")

        if formula_type == "conv":
            return self._estimate_conv_flops(node, graph_info)
        elif formula_type == "matmul":
            return self._estimate_matmul_flops(node, graph_info)
        elif formula_type == "gemm":
            return self._estimate_gemm_flops(node, graph_info)
        elif formula_type == "elementwise":
            return self._estimate_elementwise_flops(node, graph_info)
        elif formula_type == "softmax":
            return self._estimate_elementwise_flops(node, graph_info) * 5
        elif formula_type == "layernorm":
            return self._estimate_elementwise_flops(node, graph_info) * 5
        elif formula_type == "batchnorm":
            return self._estimate_elementwise_flops(node, graph_info) * 2
        else:
            # Unknown op - estimate based on output size
            return self._estimate_elementwise_flops(node, graph_info)

    def _estimate_conv_flops(self, node: NodeInfo, graph_info: GraphInfo) -> int:
        """Estimate FLOPs for Conv operation: 2 * K_h * K_w * C_in * C_out * H_out * W_out"""
        if len(node.inputs) < 2:
            return 0

        # Get weight shape
        weight_name = node.inputs[1]
        if weight_name in graph_info.initializers:
            weight_shape = list(graph_info.initializers[weight_name].shape)
        elif weight_name in graph_info.value_shapes:
            weight_shape = graph_info.value_shapes[weight_name]
        else:
            return 0

        # Weight shape: [C_out, C_in/groups, K_h, K_w] for 2D conv
        if len(weight_shape) < 4 or not all(isinstance(d, int) for d in weight_shape):
            return 0

        c_out, c_in_per_group, k_h, k_w = weight_shape[:4]

        # Get output shape
        if node.outputs and node.outputs[0] in graph_info.value_shapes:
            output_shape = graph_info.value_shapes[node.outputs[0]]
            if len(output_shape) >= 4 and all(
                isinstance(d, int) for d in output_shape[-2:]
            ):
                h_out, w_out = output_shape[-2], output_shape[-1]
            else:
                h_out, w_out = 1, 1
        else:
            h_out, w_out = 1, 1

        node.attributes.get("group", 1)
        flops = 2 * k_h * k_w * c_in_per_group * c_out * h_out * w_out

        # Add bias if present
        if len(node.inputs) > 2:
            flops += c_out * h_out * w_out

        return int(flops)

    def _estimate_matmul_flops(self, node: NodeInfo, graph_info: GraphInfo) -> int:
        """Estimate FLOPs for MatMul: 2 * M * N * K"""
        if len(node.inputs) < 2:
            return 0

        # Get shapes of both inputs
        shapes = []
        for inp in node.inputs[:2]:
            if inp in graph_info.initializers:
                shapes.append(list(graph_info.initializers[inp].shape))
            elif inp in graph_info.value_shapes:
                shapes.append(graph_info.value_shapes[inp])
            else:
                return 0

        if len(shapes) < 2:
            return 0

        # MatMul: A[..., M, K] @ B[..., K, N] = C[..., M, N]
        shape_a, shape_b = shapes[0], shapes[1]

        # Handle broadcasting and get M, K, N
        if len(shape_a) < 2 or len(shape_b) < 2:
            return 0

        if not all(isinstance(d, int) for d in shape_a[-2:]) or not all(
            isinstance(d, int) for d in shape_b[-2:]
        ):
            return 0

        m, k = shape_a[-2], shape_a[-1]
        k2, n = shape_b[-2], shape_b[-1]

        if k != k2:
            self.logger.warning(
                f"MatMul shape mismatch in node {node.name}: K={k} vs K={k2}"
            )
            return 0

        # Handle batch dimensions
        batch = 1
        for dim in shape_a[:-2]:
            if isinstance(dim, int):
                batch *= dim

        return int(2 * batch * m * n * k)

    def _estimate_gemm_flops(self, node: NodeInfo, graph_info: GraphInfo) -> int:
        """Estimate FLOPs for Gemm: 2 * M * N * K + M * N (bias)"""
        flops = self._estimate_matmul_flops(node, graph_info)

        # Add bias computation if present
        if (
            len(node.inputs) > 2
            and node.outputs
            and node.outputs[0] in graph_info.value_shapes
        ):
            output_shape = graph_info.value_shapes[node.outputs[0]]
            if output_shape and all(isinstance(d, int) for d in output_shape):
                bias_flops = int(np.prod(output_shape))
                flops += bias_flops

        return flops

    def _estimate_elementwise_flops(self, node: NodeInfo, graph_info: GraphInfo) -> int:
        """Estimate FLOPs for element-wise operations: N elements"""
        # Use output shape to determine element count
        if node.outputs and node.outputs[0] in graph_info.value_shapes:
            shape = graph_info.value_shapes[node.outputs[0]]
            if shape and all(isinstance(d, int) for d in shape):
                return int(np.prod(shape))

        # Fallback: use first input shape
        if node.inputs and node.inputs[0] in graph_info.value_shapes:
            shape = graph_info.value_shapes[node.inputs[0]]
            if shape and all(isinstance(d, int) for d in shape):
                return int(np.prod(shape))

        return 0

    def estimate_memory(self, graph_info: GraphInfo) -> MemoryEstimates:
        """
        Estimate memory usage for the model.

        Computes model size (parameters) and peak activation memory.

        Args:
            graph_info: Parsed graph information.

        Returns:
            MemoryEstimates with size and activation memory.
        """
        estimates = MemoryEstimates()

        # Model size: sum of initializer sizes
        for name, tensor in graph_info.initializers.items():
            # Assume float32 (4 bytes) for now
            # Could be refined by checking tensor dtype
            bytes_per_elem = 4
            if hasattr(tensor, "dtype"):
                if tensor.dtype == np.float16:
                    bytes_per_elem = 2
                elif tensor.dtype == np.float64:
                    bytes_per_elem = 8
                elif tensor.dtype in (np.int8, np.uint8):
                    bytes_per_elem = 1
                elif tensor.dtype in (np.int16, np.uint16):
                    bytes_per_elem = 2

            tensor_bytes = (
                int(np.prod(tensor.shape)) * bytes_per_elem
                if tensor.shape
                else bytes_per_elem
            )
            estimates.model_size_bytes += tensor_bytes

        # Peak activation memory: rough estimate based on intermediate tensor sizes
        # This is a simplified model - actual memory depends on execution order
        activation_sizes = []
        for name, shape in graph_info.value_shapes.items():
            # Skip initializers (they're counted in model size)
            if name in graph_info.initializers:
                continue

            if shape and all(isinstance(d, int) for d in shape):
                # Assume float32
                tensor_bytes = int(np.prod(shape)) * 4
                activation_sizes.append((name, tensor_bytes))
                estimates.per_layer_activation_bytes[name] = tensor_bytes

        # Peak is approximate: sum of largest activations that might coexist
        # A more accurate estimate would require analyzing the execution graph
        sorted_activations = sorted(activation_sizes, key=lambda x: -x[1])
        # Rough heuristic: top 3 largest activations might coexist
        top_n = min(3, len(sorted_activations))
        estimates.peak_activation_bytes = sum(
            size for _, size in sorted_activations[:top_n]
        )

        return estimates
