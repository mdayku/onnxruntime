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
from .patterns import PatternAnalyzer
from .report import InspectionReport, ModelInspector
from .risks import RiskAnalyzer, RiskSignal
from .visualizations import (
    THEME,
    ChartTheme,
    VisualizationGenerator,
    generate_visualizations,
    is_available as is_visualization_available,
)
from .llm_summarizer import (
    LLMSummarizer,
    LLMSummary,
    summarize_report,
    is_available as is_llm_available,
    has_api_key as has_llm_api_key,
)

__all__ = [
    "ModelInspector",
    "InspectionReport",
    "MetricsEngine",
    "ONNXGraphLoader",
    "PatternAnalyzer",
    "RiskAnalyzer",
    "RiskSignal",
    # Hardware
    "HardwareProfile",
    "HardwareDetector",
    "HardwareEstimator",
    "HardwareEstimates",
    "HARDWARE_PROFILES",
    "detect_local_hardware",
    "get_profile",
    "list_available_profiles",
    # Visualization
    "VisualizationGenerator",
    "ChartTheme",
    "THEME",
    "generate_visualizations",
    "is_visualization_available",
    # LLM Summarization
    "LLMSummarizer",
    "LLMSummary",
    "summarize_report",
    "is_llm_available",
    "has_llm_api_key",
]
