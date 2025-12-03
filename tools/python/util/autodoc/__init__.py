# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
ONNX Autodoc - Model architecture inspection and documentation tool.

This module provides utilities for analyzing ONNX models to extract:
- Parameter counts and memory estimates
- FLOP estimates per operation
- Architectural pattern detection (transformers, CNNs, residual blocks)
- Risk signals for deployment

Example usage:
    from onnxruntime.tools.util.autodoc import ModelInspector

    inspector = ModelInspector()
    report = inspector.inspect("model.onnx")
    print(report.to_json())
"""

from .analyzer import MetricsEngine, ONNXGraphLoader
from .edge_analysis import EdgeAnalysisResult, EdgeAnalyzer
from .hardware import (
    HARDWARE_PROFILES,
    HardwareDetector,
    HardwareEstimates,
    HardwareEstimator,
    HardwareProfile,
    detect_local_hardware,
    get_profile,
    list_available_profiles,
)
from .hierarchical_graph import (
    HierarchicalGraph,
    HierarchicalGraphBuilder,
    HierarchicalNode,
)
from .hierarchical_graph import generate_summary as generate_graph_summary
from .html_export import HTMLExporter
from .html_export import generate_html as generate_graph_html
from .layer_summary import (
    LayerMetrics,
    LayerSummary,
    LayerSummaryBuilder,
    generate_html_table,
    generate_markdown_table,
)
from .llm_summarizer import LLMSummarizer, LLMSummary, summarize_report
from .llm_summarizer import has_api_key as has_llm_api_key
from .llm_summarizer import is_available as is_llm_available
from .patterns import PatternAnalyzer
from .pdf_generator import PDFGenerator, generate_pdf
from .pdf_generator import is_available as is_pdf_available
from .report import (
    DatasetInfo,
    InspectionReport,
    ModelInspector,
    infer_num_classes_from_output,
)
from .risks import RiskAnalyzer, RiskSignal, RiskThresholds
from .schema import (
    INSPECTION_REPORT_SCHEMA,
    ValidationError,
    get_schema,
    validate_report,
    validate_report_strict,
)
from .visualizations import (
    THEME,
    ChartTheme,
    VisualizationGenerator,
    generate_visualizations,
)
from .visualizations import is_available as is_visualization_available

__all__ = [
    "HARDWARE_PROFILES",
    "INSPECTION_REPORT_SCHEMA",
    "THEME",
    "ChartTheme",
    "DatasetInfo",
    "EdgeAnalysisResult",
    "EdgeAnalyzer",
    "HTMLExporter",
    "HardwareDetector",
    "HardwareEstimates",
    "HardwareEstimator",
    "HardwareProfile",
    "HierarchicalGraph",
    "HierarchicalGraphBuilder",
    "HierarchicalNode",
    "InspectionReport",
    "LLMSummarizer",
    "LLMSummary",
    "LayerMetrics",
    "LayerSummary",
    "LayerSummaryBuilder",
    "MetricsEngine",
    "ModelInspector",
    "ONNXGraphLoader",
    "PDFGenerator",
    "PatternAnalyzer",
    "RiskAnalyzer",
    "RiskSignal",
    "RiskThresholds",
    "ValidationError",
    "VisualizationGenerator",
    "detect_local_hardware",
    "generate_graph_html",
    "generate_graph_summary",
    "generate_html_table",
    "generate_markdown_table",
    "generate_pdf",
    "generate_visualizations",
    "get_profile",
    "get_schema",
    "has_llm_api_key",
    "infer_num_classes_from_output",
    "is_llm_available",
    "is_pdf_available",
    "is_visualization_available",
    "list_available_profiles",
    "summarize_report",
    "validate_report",
    "validate_report_strict",
]
