# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
Pattern detection for ONNX Autodoc.

Detects common architectural patterns in ONNX graphs:
- Conv-BatchNorm-ReLU blocks
- Residual/skip connections
- Transformer blocks (attention + MLP)
- Embedding layers
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .analyzer import GraphInfo, NodeInfo


@dataclass
class Block:
    """
    A detected architectural block (group of related nodes).

    Blocks represent higher-level patterns like "ResidualBlock" or
    "TransformerLayer" that consist of multiple ONNX nodes.
    """

    block_type: str  # e.g., "ConvBNRelu", "ResidualBlock", "TransformerBlock"
    name: str
    nodes: list[str]  # Node names in this block
    start_node: str
    end_node: str
    attributes: dict = field(default_factory=dict)  # Block-specific metadata

    def to_dict(self) -> dict:
        return {
            "block_type": self.block_type,
            "name": self.name,
            "nodes": self.nodes,
            "start_node": self.start_node,
            "end_node": self.end_node,
            "attributes": self.attributes,
        }


class PatternAnalyzer:
    """
    Detect architectural patterns in ONNX graphs.

    Identifies common patterns like Conv-BN-ReLU sequences, residual
    blocks, and transformer attention blocks.
    """

    # Operators that commonly appear together
    CONV_ACTIVATIONS = {
        "Relu",
        "LeakyRelu",
        "Sigmoid",
        "Tanh",
        "Clip",
        "HardSwish",
        "Silu",
    }
    NORM_OPS = {
        "BatchNormalization",
        "InstanceNormalization",
        "LayerNormalization",
        "GroupNormalization",
    }
    ATTENTION_OPS = {"MatMul", "Softmax", "Transpose"}
    EMBEDDING_OPS = {"Gather", "Embedding"}

    def __init__(self, logger: logging.Logger | None = None):
        self.logger = logger or logging.getLogger("autodoc.patterns")

    def group_into_blocks(self, graph_info: GraphInfo) -> list[Block]:
        """
        Detect all architectural blocks in the graph.

        Args:
            graph_info: Parsed graph information.

        Returns:
            List of detected Block instances.
        """
        blocks = []

        # Detect various patterns
        blocks.extend(self.detect_conv_bn_relu(graph_info))
        blocks.extend(self.detect_residual_blocks(graph_info))
        blocks.extend(self.detect_transformer_blocks(graph_info))
        blocks.extend(self.detect_embedding_layers(graph_info))

        self.logger.debug(f"Detected {len(blocks)} blocks")
        return blocks

    def detect_conv_bn_relu(self, graph_info: GraphInfo) -> list[Block]:
        """
        Find Conv-BatchNorm-ReLU sequences.

        Matches patterns like:
        - Conv -> BatchNorm -> ReLU
        - Conv -> ReLU
        - Conv -> BatchNorm
        """
        blocks = []
        visited = set()

        for node in graph_info.nodes:
            if node.name in visited:
                continue

            if node.op_type == "Conv":
                block_nodes = [node.name]
                current_output = node.outputs[0] if node.outputs else None
                block_type_parts = ["Conv"]

                # Look for BatchNorm
                if current_output:
                    next_node = self._find_consumer(current_output, graph_info)
                    if next_node and next_node.op_type in self.NORM_OPS:
                        block_nodes.append(next_node.name)
                        block_type_parts.append("BN")
                        current_output = (
                            next_node.outputs[0] if next_node.outputs else None
                        )

                        # Look for activation after BN
                        if current_output:
                            act_node = self._find_consumer(current_output, graph_info)
                            if act_node and act_node.op_type in self.CONV_ACTIVATIONS:
                                block_nodes.append(act_node.name)
                                block_type_parts.append(act_node.op_type)
                    elif next_node and next_node.op_type in self.CONV_ACTIVATIONS:
                        # Conv -> ReLU without BN
                        block_nodes.append(next_node.name)
                        block_type_parts.append(next_node.op_type)

                if len(block_nodes) > 1:
                    visited.update(block_nodes)
                    block = Block(
                        block_type="".join(block_type_parts),
                        name=f"conv_block_{len(blocks)}",
                        nodes=block_nodes,
                        start_node=block_nodes[0],
                        end_node=block_nodes[-1],
                    )
                    blocks.append(block)

        return blocks

    def detect_residual_blocks(self, graph_info: GraphInfo) -> list[Block]:
        """
        Find residual/skip connection patterns.

        Looks for Add nodes where one input comes from earlier in the graph
        (skip connection).
        """
        blocks = []

        for node in graph_info.nodes:
            if node.op_type == "Add" and len(node.inputs) >= 2:
                # Check if this could be a residual connection
                # by looking for inputs that come from different depths
                input_nodes = []
                for inp in node.inputs:
                    if inp in graph_info.node_by_output:
                        input_nodes.append(graph_info.node_by_output[inp])

                if len(input_nodes) >= 2:
                    # Heuristic: if one path is longer (more hops), it's likely the residual path
                    # For now, just detect the pattern exists
                    blocks.append(
                        Block(
                            block_type="ResidualAdd",
                            name=f"residual_{len(blocks)}",
                            nodes=[node.name],
                            start_node=node.name,
                            end_node=node.name,
                            attributes={"inputs": node.inputs},
                        )
                    )

        return blocks

    def detect_transformer_blocks(self, graph_info: GraphInfo) -> list[Block]:
        """
        Find transformer attention patterns.

        Looks for the characteristic Softmax in attention computation
        and MatMul patterns for Q, K, V projections.
        """
        blocks = []
        softmax_nodes = [n for n in graph_info.nodes if n.op_type == "Softmax"]

        for softmax in softmax_nodes:
            # Look for attention pattern: MatMul -> Softmax -> MatMul
            before_nodes = []
            after_nodes = []

            # Find MatMul before softmax
            if softmax.inputs:
                inp = softmax.inputs[0]
                if inp in graph_info.node_by_output:
                    prev = graph_info.node_by_output[inp]
                    if prev.op_type in (
                        "MatMul",
                        "Gemm",
                        "Div",
                        "Mul",
                    ):  # Div for scaling
                        before_nodes.append(prev.name)

            # Find MatMul after softmax
            if softmax.outputs:
                consumer = self._find_consumer(softmax.outputs[0], graph_info)
                if consumer and consumer.op_type in ("MatMul", "Gemm"):
                    after_nodes.append(consumer.name)

            if before_nodes and after_nodes:
                all_nodes = [*before_nodes, softmax.name, *after_nodes]
                blocks.append(
                    Block(
                        block_type="Attention",
                        name=f"attention_{len(blocks)}",
                        nodes=all_nodes,
                        start_node=before_nodes[0],
                        end_node=after_nodes[-1],
                    )
                )

        # Also look for LayerNorm which often brackets transformer blocks
        layernorm_count = sum(
            1 for n in graph_info.nodes if n.op_type == "LayerNormalization"
        )
        if layernorm_count >= 2 and blocks:
            # Likely a transformer architecture
            self.logger.debug(
                f"Found {len(blocks)} attention blocks with {layernorm_count} LayerNorms"
            )

        return blocks

    def detect_embedding_layers(self, graph_info: GraphInfo) -> list[Block]:
        """
        Find embedding lookup patterns.

        Looks for Gather operations on large weight tensors.
        """
        blocks = []

        for node in graph_info.nodes:
            if node.op_type == "Gather":
                # Check if first input is a large initializer (embedding table)
                if node.inputs and node.inputs[0] in graph_info.initializers:
                    embed_table = graph_info.initializers[node.inputs[0]]
                    if len(embed_table.shape) == 2:
                        vocab_size, embed_dim = embed_table.shape
                        blocks.append(
                            Block(
                                block_type="Embedding",
                                name=f"embedding_{len(blocks)}",
                                nodes=[node.name],
                                start_node=node.name,
                                end_node=node.name,
                                attributes={
                                    "vocab_size": int(vocab_size),
                                    "embed_dim": int(embed_dim),
                                },
                            )
                        )

        return blocks

    def classify_architecture(self, graph_info: GraphInfo, blocks: list[Block]) -> str:
        """
        Classify the overall architecture type.

        Args:
            graph_info: Parsed graph information.
            blocks: Detected blocks from group_into_blocks().

        Returns:
            Architecture type: "transformer", "cnn", "mlp", "hybrid", "unknown"
        """
        op_counts = graph_info.op_type_counts
        block_types = [b.block_type for b in blocks]

        # Count key indicators
        has_attention = any("Attention" in bt for bt in block_types)
        has_layernorm = op_counts.get("LayerNormalization", 0) > 0
        has_embedding = any("Embedding" in bt for bt in block_types)

        conv_count = op_counts.get("Conv", 0)
        matmul_count = op_counts.get("MatMul", 0) + op_counts.get("Gemm", 0)
        softmax_count = op_counts.get("Softmax", 0)

        # Classification heuristics
        if has_attention or (has_layernorm and softmax_count >= 2 and has_embedding):
            return "transformer"
        elif conv_count > matmul_count and conv_count >= 5:
            return "cnn"
        elif conv_count > 0 and (has_attention or has_layernorm):
            return "hybrid"
        elif matmul_count > 0:
            return "mlp"
        else:
            return "unknown"

    def _find_consumer(
        self, output_name: str, graph_info: GraphInfo
    ) -> NodeInfo | None:
        """Find the first node that consumes a given output."""
        for node in graph_info.nodes:
            if output_name in node.inputs:
                return node
        return None
