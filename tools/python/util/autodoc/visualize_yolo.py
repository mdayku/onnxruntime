#!/usr/bin/env python3
"""Visualize a YOLO model."""
import sys
sys.path.insert(0, str(__file__).replace("\\", "/").rsplit("/", 4)[0])

from util.autodoc.analyzer import ONNXGraphLoader
from util.autodoc.patterns import PatternAnalyzer
from util.autodoc.edge_analysis import EdgeAnalyzer
from util.autodoc.hierarchical_graph import HierarchicalGraphBuilder
from util.autodoc.html_export import HTMLExporter
from pathlib import Path

model_path = Path(r"C:\Users\marcu\Roomer\room_detection_training\local_training_output\yolo-v8l-200epoch\weights\best.onnx")

print(f"Loading {model_path.name}...")
loader = ONNXGraphLoader()
_, graph_info = loader.load(model_path)

print(f"Nodes: {len(graph_info.nodes)}")
print(f"Initializers: {len(graph_info.initializers)}")

patterns = PatternAnalyzer()
blocks = patterns.group_into_blocks(graph_info)
print(f"Detected {len(blocks)} blocks")

# Get architecture summary
summary = patterns.get_architecture_summary(graph_info, blocks)
arch_type = summary.get("architecture_type", "unknown")
print(f"Architecture: {arch_type}")

edge_analyzer = EdgeAnalyzer()
edge_result = edge_analyzer.analyze(graph_info)
print(f"Edges: {len(edge_result.edges)}")
peak_mb = edge_result.peak_activation_bytes / 1024 / 1024
print(f"Peak memory: {peak_mb:.1f} MB")

builder = HierarchicalGraphBuilder()
hier_graph = builder.build(graph_info, blocks, "YOLOv8-L Room Detection")
print(f"Hierarchy: {hier_graph.total_nodes} nodes, depth {hier_graph.depth}")

exporter = HTMLExporter()
output_path = Path("C:/Users/marcu/onnxruntime/yolo_graph.html")
exporter.export(hier_graph, edge_result, output_path, "YOLOv8-L Room Detection")
print()
print(f"Exported to: {output_path}")
