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
from .llm_summarizer import (
    LLMSummarizer,
    LLMSummary,
    summarize_report,
)
from .llm_summarizer import (
    has_api_key as has_llm_api_key,
)
from .llm_summarizer import (
    is_available as is_llm_available,
)
from .patterns import PatternAnalyzer
from .report import InspectionReport, ModelInspector
from .risks import RiskAnalyzer, RiskSignal
from .visualizations import (
    THEME,
    ChartTheme,
    VisualizationGenerator,
    generate_visualizations,
)
from .visualizations import (
    is_available as is_visualization_available,
)

__all__ = [
    "HARDWARE_PROFILES",
    "THEME",
    "ChartTheme",
    "HardwareDetector",
    "HardwareEstimates",
    "HardwareEstimator",
    # Hardware
    "HardwareProfile",
    "InspectionReport",
    # LLM Summarization
    "LLMSummarizer",
    "LLMSummary",
    "MetricsEngine",
    "ModelInspector",
    "ONNXGraphLoader",
    "PatternAnalyzer",
    "RiskAnalyzer",
    "RiskSignal",
    # Visualization
    "VisualizationGenerator",
    "detect_local_hardware",
    "generate_visualizations",
    "get_profile",
    "has_llm_api_key",
    "is_llm_available",
    "is_visualization_available",
    "list_available_profiles",
    "summarize_report",
]
