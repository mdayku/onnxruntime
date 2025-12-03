# ONNX Autodoc - Product Requirements Document

## Document Info

| Field | Value |
|-------|-------|
| Project | ONNX Autodoc / Model Architecture Inspector |
| Author | Marcus |
| Version | 1.0 |
| Last Updated | December 2, 2025 |
| Status | In Development |

**Related Documents:**
- [BACKLOG.md](BACKLOG.md) - Epic/Story/Task tracking
- [BRAINLIFT.md](BRAINLIFT.md) - Daily learning logs
- [Architecture.md](Architecture.md) - System design details
- [README.md](README.md) - Quick start guide

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Goals and Non-Goals](#2-goals-and-non-goals)
3. [Personas and User Stories](#3-personas-and-user-stories)
4. [Technical Architecture](#4-technical-architecture)
5. [CLI Design and JSON Schemas](#5-cli-design-and-json-schemas)
6. [Visualization Extension](#6-visualization-extension)
7. [Hardware Profiles and GPU Estimates](#7-hardware-profiles-and-gpu-estimates)
8. [Quantization Impact Panel](#8-quantization-impact-panel)
9. [External Pipeline Integration](#9-external-pipeline-integration)
10. [Testing Strategy](#10-testing-strategy)
11. [Dependencies](#11-dependencies)
12. [Error Handling](#12-error-handling)
13. [Implementation Timeline](#13-implementation-timeline)

---

## 1. Project Overview

### 1.1 One-Liner

**ONNX Autodoc** is a graph-level analysis and documentation tool for ONNX Runtime that inspects a model's architecture, computes static complexity metrics (FLOPs, params, activation memory), surfaces structural risk signals, and auto-generates human-readable reports and model cards that explain *how* the model works, not just how it performs on a test set.

### 1.2 Motivation

Modern ML teams increasingly inherit ONNX models they did not design:

- Vendor-supplied models shipped as `.onnx` artifacts
- Compressed / quantized models whose original training code is unknown
- Legacy models where only checkpoints remain

These models might perform well on benchmarks, but:

- Their internal architecture is opaque
- Potential failure modes are unclear
- Hardware suitability / deployment risk is uncertain

Existing ONNX Runtime tooling focuses on **performance** and **operator support** (e.g., perf tests, model usability checker for NNAPI/CoreML). This project adds a **model architecture inspector** that answers:

> *"What is this model structurally, how expensive is it, what are its likely failure modes, and how should we position it in our system?"*

### 1.3 Core Idea

Given a model `model.onnx`, ONNX Autodoc will:

1. Parse the graph (nodes, tensors, shapes, ops, attributes).
2. Compute structural metrics:
   - Parameter counts per node / block / whole model
   - FLOPs estimates and hot spots
   - Activation tensor sizes and peak memory
   - Attention-specific stats (num layers, heads, context, KV cache footprint) where applicable
3. Identify **risk signals** via heuristics, e.g.:
   - Extremely deep chains without skip connections
   - Non-standard residual patterns
   - Pathologically wide fully-connected layers
   - Mismatched input/output resolutions or shapes
   - Suspect dynamic shape usage for deployment
4. Emit a **structured JSON report** and optional **Markdown/HTML summary**.
5. (Optional) Use an external LLM to summarize findings into a model card.

This becomes a first-class **model inspection** tool in the ONNX Runtime ecosystem.

---

## 2. Goals and Non-Goals

### 2.1 Goals

1. **Graph-Level Architecture Analysis**
   - Robust parsing of ONNX graphs (nodes, edges, initializers, shapes).
   - Layer/block aggregation (e.g., grouping Conv->BN->Relu into a single logical block).

2. **Static Complexity Metrics**
   - Parameter count per node, per block, and globally.
   - FLOP estimates for common ops (Conv, Gemm, MatMul, Self-Attention, etc.).
   - Approximate peak activation memory.

3. **Risk/Smell Detection**
   - Rule-based heuristics to flag potentially problematic architectures.
   - Configurable thresholds for "warning" vs "info" (e.g., model too big for mobile, too deep for training stability, etc.).

4. **Human-Readable Autodoc Output**
   - JSON: full structured report.
   - Markdown: model card + architecture summary. (Designed to integrate nicely with your existing batch-eval outputs.)

5. **Integration with ONNX Runtime Tooling**
   - Exposed as a CLI tool similar to `check_onnx_model_mobile_usability`.
   - Can be invoked via `python -m onnxruntime.tools.model_inspect` or an installed CLI entrypoint `onnxrt-inspect`.

6. **Visualization Support**
   - Matplotlib-based visualizations for architecture analysis
   - Operator distribution histograms, layer depth profiles, parameter distribution charts
   - Embeddable in Markdown/HTML reports

### 2.2 Non-Goals (for MVP)

- Training / fine-tuning models.
- Dynamic runtime profiling or latency benchmarking (already covered by `onnxruntime_perf_test`).
- Auto-fixing models (MVP is read-only analysis; suggestions may be textual only).
- Exhaustive support for every exotic ONNX op (MVP focuses on common image/vision and transformer ops; others fall back to generic estimates).

---

## 3. Personas and User Stories

### 3.1 Personas

| Persona | Role | Primary Need |
|---------|------|--------------|
| **Nina** | ML Infra Engineer | Evaluate deployability of third-party ONNX models on specific hardware |
| **Dev** | Applied Scientist | Quick feedback on complexity and structural differences between architecture variants |
| **Sam** | MLOps/Platform Engineer | Automatic generation of architecture summaries and model cards for internal registry |

### 3.2 Key User Stories

| ID | As a... | I want to... | So that... |
|----|---------|--------------|------------|
| US-1 | ML infra engineer | Run a single command on an ONNX file and see parameter count, FLOPs, and peak memory | I can tell if it fits on our edge devices |
| US-2 | Applied scientist | Diff between two models' architecture reports | I can understand what structurally changed between v1 and v2 |
| US-3 | MLOps engineer | Get a machine-readable JSON report | I can ingest into my model registry for search and governance |
| US-4 | Tech lead | Get a human-readable model card | I can share it with cross-functional stakeholders |
| US-5 | VP Engineering | See visual summaries of model architecture | I can assess maintainability and deployment complexity |

---

## 4. Technical Architecture

### 4.1 Components

```
+-------------------+     +----------------------+     +------------------+
|  CLI / Python     | --> |  Model Inspector     | --> |  JSON Report     |
|  Frontend         |     |  Engine              |     +------------------+
+-------------------+     +----------------------+             |
                                   |                           v
                                   v                  +------------------+
                          +------------------+        |  Markdown/HTML   |
                          |  ONNX Graph      |        |  Model Card      |
                          |  Loader          |        +------------------+
                          +------------------+                 |
                                   |                           v
                                   v                  +------------------+
                          +------------------+        |  LLM Summarizer  |
                          |  Metrics         |        |  (Optional)      |
                          |  Calculator      |        +------------------+
                          +------------------+
                                   |
                                   v
                          +------------------+
                          |  Pattern         |
                          |  Analyzer        |
                          +------------------+
                                   |
                                   v
                          +------------------+
                          |  Risk            |
                          |  Heuristics      |
                          +------------------+
                                   |
                                   v
                          +------------------+
                          |  Visualization   |
                          |  Module          |
                          +------------------+
```

### 4.2 Component Descriptions

- **Graph Analysis Core (Engine)**
  - Implemented as a library module that:
    - Loads an ONNX model.
    - Traverses the graph (nodes, tensors, initializers).
    - Computes structural metrics and risk signals.

- **CLI / Python Frontend**
  - Thin wrapper calling the core engine.
  - Handles argument parsing, output formatting (JSON/Markdown), and optional LLM calls.

- **LLM Summarization (Optional)**
  - Separate module that:
    - Takes the JSON report as input.
    - Produces human-readable model descriptions and model card text.

- **Visualization Module**
  - Matplotlib-based chart generation
  - Produces PNG assets for embedding in reports

### 4.3 Placement in ONNX Runtime Repo

Target repo: `microsoft/onnxruntime` (forked).

**File Locations:**

| Component | Path |
|-----------|------|
| Python CLI & core logic | `onnxruntime/python/tools/model_inspect.py` |
| Compare CLI | `onnxruntime/python/tools/model_inspect_compare.py` |
| Analysis module | `onnxruntime/tools/autodoc/analysis.py` |
| Visualization module | `onnxruntime/tools/autodoc/visualizations.py` |
| Markdown renderer | `onnxruntime/tools/autodoc/render_markdown.py` |
| HTML renderer (optional) | `onnxruntime/tools/autodoc/render_html.py` |
| C++ core (stretch) | `onnxruntime/core/graph/model_inspector.h`, `.cc` |
| Python unit tests | `onnxruntime/test/python/tools/test_model_inspect.py` |
| C++ unit tests (stretch) | `onnxruntime/test/graph/model_inspector_test.cc` |
| Sample scripts | `samples/tools/model_inspect_*.sh` |

### 4.4 Mermaid Diagram

```mermaid
graph TD
    A[CLI / Python Frontend<br/>onnxruntime.tools.model_inspect] --> B[Model Inspector Engine]
    B --> C[ONNX Graph Loader]
    C --> D[Graph Representation<br/>nodes, tensors, initializers]
    D --> E[Metrics Calculator<br/>params, FLOPs, memory]
    D --> F[Pattern Analyzer<br/>blocks, attention, CNN]
    E --> G[Risk Heuristics]
    F --> G
    G --> H[JSON Report]
    H --> I[Markdown Model Card]
    H --> J[LLM Summarizer<br/>optional]
    J --> I
    H --> K[Visualization Module]
    K --> L[PNG Assets]
    L --> I
```

---

## 5. CLI Design and JSON Schemas

### 5.1 CLI Interface (MVP)

**Command:**

```bash
python -m onnxruntime.tools.model_inspect model.onnx \
  --out-json report.json \
  --out-md report.md \
  --llm-summary  # optional
```

**Required arguments:**
- `model_path`: path to `.onnx` model file.

**Optional flags:**

| Flag | Description |
|------|-------------|
| `--out-json PATH` | Write full JSON report to `PATH` (otherwise print to stdout) |
| `--out-md PATH` | Write Markdown summary/model card |
| `--max-layers N` | Truncate per-layer details to top N by FLOPs/params |
| `--llm-summary` | Call external LLM to generate natural-language summaries (reads API key from env) |
| `--log-level {debug,info}` | Logging verbosity |
| `--with-plots` | Generate visualization assets |
| `--assets-dir PATH` | Directory for plot PNGs |
| `--format {md,html}` | Output format for report |

### 5.2 JSON Report Schema (Draft)

```json
{
  "model": {
    "path": "string",
    "onnx_ir_version": "string",
    "opset_imports": [
      { "domain": "", "version": 17 }
    ],
    "producer_name": "string",
    "producer_version": "string",
    "domain": "string",
    "doc_string": "string"
  },
  "graph": {
    "name": "string",
    "num_nodes": 0,
    "num_edges": 0,
    "inputs": [
      {
        "name": "input",
        "elem_type": "float32",
        "shape": [1, 3, 224, 224]
      }
    ],
    "outputs": [
      {
        "name": "output",
        "elem_type": "float32",
        "shape": [1, 1000]
      }
    ]
  },
  "layers": [
    {
      "name": "conv1",
      "type": "Conv",
      "domain": "ai.onnx",
      "inputs": ["input"],
      "outputs": ["conv1_out"],
      "input_shape": [1, 3, 224, 224],
      "output_shape": [1, 64, 112, 112],
      "params": 9408,
      "flops": 118013952,
      "activation_bytes": 3211264,
      "group": "stem_block",
      "attributes": {
        "kernel_shape": [7, 7],
        "strides": [2, 2],
        "pads": [3, 3, 3, 3]
      }
    }
  ],
  "blocks": [
    {
      "name": "layer1_block0",
      "type": "ResidualBlock",
      "nodes": ["conv2", "bn2", "relu2"],
      "params": 102400,
      "flops": 450000000,
      "activation_bytes": 8388608
    }
  ],
  "metrics": {
    "total_params": 25557032,
    "total_flops": 4100000000,
    "peak_activation_bytes": 134217728,
    "attention": {
      "num_layers": 12,
      "num_heads": 12,
      "hidden_size": 768,
      "max_sequence_length": 2048,
      "kv_cache_bytes_per_token": 786432
    }
  },
  "risk_signals": [
    {
      "id": "large_dense_stack",
      "severity": "medium",
      "description": "Two consecutive fully-connected layers with width 4096 may be over-parameterized.",
      "nodes": ["fc1", "fc2"]
    },
    {
      "id": "no_skip_connections",
      "severity": "high",
      "description": "Very deep network (120 layers) with no residual or skip connections detected.",
      "nodes": []
    }
  ],
  "llm_summary": {
    "short": "string",
    "detailed": "string"
  }
}
```

This schema is intentionally verbose so you can:
- Ingest it into your **batch eval** pipeline.
- Join against performance metrics (F1, latency, etc.).
- Build dashboards that correlate *architecture* with *performance* and *deployment risk*.

---

## 6. Visualization Extension

### 6.1 Autodoc Flow With Visualizations

```
ONNX Model -> Analysis (ModelStats) -> Visualization Assets -> Markdown/HTML Report
```

**Pipeline Steps:**

1. **Load Model** - Load ONNX with `onnx.load`
2. **Analyze Model** - Operator distribution, parameter counts, layer depth estimates, shape snapshots
3. **Generate Visualizations** - Using matplotlib (`Agg` backend):
   - Operator type histogram
   - Layer depth profile
   - Parameter distribution
   - Shape evolution
   - Optional graph connectivity
4. **Render Report** - Markdown or HTML embeds visuals for leadership + technical details for ML engineers

### 6.2 The `ModelStats` Object

```python
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

@dataclass
class OpTypeStats:
    counts: Dict[str, int]
    params_per_op: Dict[str, int]


@dataclass
class LayerShapeSnapshot:
    layer_index: int
    node_names: List[str]
    shapes: List[Tuple[int, ...]]


@dataclass
class ModelStats:
    model_path: Path
    model_name: str
    num_parameters: int
    num_initializers: int
    num_nodes: int
    num_inputs: int
    num_outputs: int

    op_stats: OpTypeStats
    layer_shapes: List[LayerShapeSnapshot]
```

This object is the interface between analysis, plots, and rendering.

### 6.3 Visualization Module API

```python
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAVE_MPL = True
except Exception:
    _HAVE_MPL = False

def generate_all_plots(stats: ModelStats, assets_dir: Path) -> Dict[str, str]:
    """Generate all visualization assets.

    Returns:
        Dict mapping plot name to file path, e.g.:
        {"op_type_hist": "assets/op_type_hist.png", ...}
    """
    pass
```

### 6.4 Visualization Types

| Chart | X-Axis | Y-Axis | Purpose |
|-------|--------|--------|---------|
| Operator Histogram | Op types (Conv, Relu, MatMul...) | Node counts | Identify architecture "style" |
| Layer Depth Profile | Layer index | Cumulative params/FLOPs | Show compute distribution |
| Parameter Distribution | Layer name | Param count | Find heavy layers |
| Shape Evolution | Layer index | Tensor dimensions | Track spatial resolution changes |

### 6.5 Markdown Embedding

```markdown
## Operator Type Distribution
![Operator Histogram](assets/op_type_hist.png)

### Executive Summary
- Model size: 123M parameters
- Dominant layers: Attention (72%)
- Architecture: Transformer encoder
```

### 6.6 Audience Framing

| Audience | Focus |
|----------|-------|
| **ML Engineers** | Identify compute/memory hotspots, validate graph assumptions, compare architectures |
| **VP Engineering / CTO** | Assess maintainability, estimate deployment complexity, inform optimization decisions |
| **C-suite / Board** | Understand cost drivers (inference, infra), see high-level architecture without jargon, support roadmap tradeoffs |

### 6.7 Interactive Graph Visualization (Netron-style)

Inspired by Netron and TensorRT EngineXplorer, provide a cleaner, horizontally-oriented graph visualization.

**Key Features:**

| Feature | Description |
|---------|-------------|
| **Horizontal Layout** | Left-to-right flow (inputs -> processing -> outputs) instead of vertical stacking |
| **Collapsible Blocks** | Group Conv-BN-ReLU, attention, and residual blocks into expandable nodes |
| **Node Inspection** | Click any node to see: op type, input/output shapes, params, FLOPs, attributes |
| **Pan/Zoom** | Navigate large graphs with smooth pan and zoom controls |
| **Search** | Find nodes by name, op type, or attribute values |
| **Heatmap Overlay** | Color nodes by latency estimate, memory usage, or parameter count |

**Implementation Options:**

| Library | Pros | Cons |
|---------|------|------|
| **D3.js** | Maximum flexibility, great for custom layouts | Steeper learning curve |
| **Cytoscape.js** | Built for graph viz, good perf with large graphs | Less custom styling |
| **React Flow** | React-native, drag-and-drop, good DX | May need custom layout algo |
| **vis.js** | Simple API, hierarchical layouts built-in | Less modern look |

**Export Formats:**

- Embedded in HTML report (self-contained, interactive)
- Standalone HTML file (shareable)
- SVG export (for documentation/presentations)
- PNG export (static image)

### 6.8 Per-Layer Summary Table

Provide detailed per-layer metrics in a sortable, filterable table.

**Table Columns:**

| Column | Description |
|--------|-------------|
| Layer Name | Node name from ONNX graph |
| Op Type | Conv, MatMul, Attention, etc. |
| Input Shape | Tensor dimensions entering the layer |
| Output Shape | Tensor dimensions exiting the layer |
| Parameters | Weight count for this layer |
| FLOPs | Compute operations for this layer |
| Latency Est. | Estimated execution time (ms) |
| Memory | Activation memory footprint |
| % of Total | Percentage of model's total compute/params |

**Interactive Features:**

- Sort by any column (ascending/descending)
- Filter by op type, parameter threshold, or FLOPs threshold
- Search by layer name
- Click row to highlight in graph visualization
- Export to CSV/JSON

**Example Output:**

```
| Layer           | Op Type  | Input Shape      | Output Shape     | Params   | FLOPs      | Latency | Memory   |
|-----------------|----------|------------------|------------------|----------|------------|---------|----------|
| conv1           | Conv     | [1,3,224,224]    | [1,64,112,112]   | 9.4K     | 118M       | 0.12ms  | 3.1MB    |
| layer1.0.conv1  | Conv     | [1,64,56,56]     | [1,64,56,56]     | 36.9K    | 231M       | 0.24ms  | 0.8MB    |
| layer4.2.conv2  | Conv     | [1,512,7,7]      | [1,512,7,7]      | 2.4M     | 462M       | 0.48ms  | 0.1MB    |
| fc              | Gemm     | [1,2048]         | [1,1000]         | 2.0M     | 4.1M       | 0.01ms  | 8KB      |
```

---

## 7. Hardware Profiles and GPU Estimates

### 7.1 Hardware Profiles

A hardware profile is a small JSON file describing a target device:

```json
{
  "name": "NVIDIA_RTX_4090",
  "vendor": "nvidia",
  "type": "gpu",
  "vram_bytes": 25769803776,
  "peak_fp16_flops": 330000000000000,
  "peak_fp32_flops": 82500000000000,
  "memory_bandwidth_bytes_per_s": 1000000000000
}
```

### 7.2 CLI with Hardware Profile

```bash
python -m onnxruntime.tools.model_inspect model.onnx \
  --hardware-profile profiles/rtx_4090.json \
  --batch-size 8 \
  --precision fp16
```

The inspector uses:
- Model FLOPs and activation bytes (from earlier analysis)
- Hardware peak FLOPs and memory bandwidth
- Requested `batch_size` and `precision`

To estimate:
- **VRAM required** (parameters + activations + simple workspace heuristic)
- **Theoretical latency bounds** (compute- vs bandwidth-limited)
- **Compute utilization estimate** at a given batch size
- Whether the model likely **fits in VRAM** for the given device

### 7.3 JSON Schema Extension - Hardware Estimates

```json
"hardware_estimates": {
  "device": "NVIDIA_RTX_4090",
  "precision": "fp16",
  "batch_size": 8,
  "vram_required_bytes": 1431655765,
  "fits_in_vram": true,
  "theoretical_latency_ms": 13.2,
  "compute_utilization_estimate": 0.72,
  "bottleneck": "memory_bandwidth"
}
```

This provides a static, explainable approximation of `nvidia-smi`-style metrics for a given model + hardware profile, without needing a live GPU.

### 7.4 Supported Hardware Profiles

| Profile | Focus |
|---------|-------|
| NVIDIA GPUs | Initial focus (RTX 4090, A10, T4, etc.) |
| AMD GPUs | Additional JSON profiles |
| Apple M-series | Additional JSON profiles |
| NPUs | Additional JSON profiles |

### 7.5 Hardware Requirements Recommendations (Steam-style)

Generate minimum and recommended hardware specifications based on model analysis, similar to how Steam displays game system requirements.

**Deployment Target Categories:**

| Target | Description | Typical Hardware |
|--------|-------------|------------------|
| **Edge/Jetson** | Embedded devices, real-time inference | Jetson Orin Nano, Jetson Xavier NX |
| **Local Server** | On-premise GPU server, batch processing | RTX 3080/4090, A10, T4 |
| **Cloud Server** | Elastic scaling, high throughput | A100, H100, L4, Inferentia |

**CLI Usage:**

```bash
python -m onnxruntime.tools.model_inspect model.onnx \
  --deployment-target edge \
  --target-latency 50ms \
  --target-throughput 30fps
```

**Output Format (Steam-style):**

```markdown
## System Requirements

### Minimum (Basic Inference)
- **GPU**: NVIDIA Jetson Orin Nano 4GB
- **VRAM**: 2.1 GB
- **Precision**: FP16
- **Expected Latency**: 85ms @ batch=1
- **Power**: 15W TDP

### Recommended (Production Throughput)
- **GPU**: NVIDIA Jetson AGX Orin 32GB
- **VRAM**: 4.2 GB
- **Precision**: FP16
- **Expected Latency**: 12ms @ batch=1
- **Throughput**: 80+ fps
- **Power**: 40W TDP

### Optimal (Maximum Performance)
- **GPU**: NVIDIA A10 (24GB)
- **VRAM**: 6.8 GB
- **Precision**: FP16 with TensorRT
- **Expected Latency**: 3ms @ batch=8
- **Throughput**: 300+ fps
```

**Factors Considered:**

| Factor | Impact |
|--------|--------|
| Model VRAM footprint | Determines minimum GPU memory |
| FLOPs / Compute density | Determines GPU compute tier |
| Memory bandwidth needs | Important for memory-bound models |
| Target latency | Filters out GPUs that can't meet SLA |
| Target throughput | Suggests batch size and GPU count |
| Power constraints | Important for edge deployment |

**JSON Schema Extension:**

```json
"hardware_recommendations": {
  "deployment_target": "edge",
  "user_constraints": {
    "max_latency_ms": 50,
    "min_throughput_fps": 30,
    "max_power_watts": 30
  },
  "minimum": {
    "device": "Jetson_Orin_Nano_4GB",
    "vram_required_gb": 2.1,
    "precision": "fp16",
    "estimated_latency_ms": 85,
    "meets_latency_target": false
  },
  "recommended": {
    "device": "Jetson_AGX_Orin_32GB",
    "vram_required_gb": 4.2,
    "precision": "fp16",
    "estimated_latency_ms": 12,
    "estimated_throughput_fps": 83,
    "meets_all_targets": true
  }
}
```

---

## 8. Quantization Impact Panel

### 8.1 Overview

Compare mode for multiple precision variants of the same model (e.g., fp32 / fp16 / int8).

### 8.2 Inputs

- Multiple ONNX model files for the same architecture:
  - `resnet_fp32.onnx`, `resnet_fp16.onnx`, `resnet_int8.onnx`
- Corresponding **eval/perf metrics JSON** files produced by a batch-eval or perf script

### 8.3 CLI - Compare Mode

```bash
python -m onnxruntime.tools.model_inspect_compare \
  --models resnet_fp32.onnx resnet_fp16.onnx resnet_int8.onnx \
  --eval-metrics eval_fp32.json eval_fp16.json eval_int8.json \
  --hardware-profile profiles/a10.json \
  --baseline-precision fp32 \
  --out-json quant_impact.json \
  --out-md quant_impact.md
```

The tool:
1. Runs (or loads) the core `model_inspect` report for each variant.
2. Loads eval/perf JSON for each variant.
3. Verifies models share the same architecture.
4. Produces a **single compare report** summarizing quantization trade-offs.

### 8.4 JSON Schema - Quantization Impact

```json
{
  "model_family_id": "resnet50_imagenet",
  "baseline_precision": "fp32",
  "variants": [
    {
      "precision": "fp32",
      "quantization_scheme": "none",
      "model_path": "resnet_fp32.onnx",
      "size_bytes": 102400000,
      "metrics": {
        "f1_macro": 0.931,
        "latency_ms_p50": 14.5,
        "throughput_qps": 680,
        "gpu_utilization_pct": 65,
        "vram_used_bytes": 4294967296
      },
      "hardware_estimates": {
        "device": "NVIDIA_A10",
        "batch_size": 8,
        "theoretical_latency_ms": 13.1,
        "compute_utilization_estimate": 0.62
      },
      "deltas_vs_baseline": null
    },
    {
      "precision": "fp16",
      "quantization_scheme": "fp16",
      "model_path": "resnet_fp16.onnx",
      "size_bytes": 51200000,
      "metrics": {
        "f1_macro": 0.929,
        "latency_ms_p50": 9.1,
        "throughput_qps": 1080,
        "gpu_utilization_pct": 79,
        "vram_used_bytes": 2818572288
      },
      "hardware_estimates": {
        "device": "NVIDIA_A10",
        "batch_size": 8,
        "theoretical_latency_ms": 8.7,
        "compute_utilization_estimate": 0.81
      },
      "deltas_vs_baseline": {
        "f1_macro": -0.002,
        "latency_ms_p50": -5.4,
        "throughput_qps": 400,
        "gpu_utilization_pct": 14,
        "vram_used_bytes": -1476395008
      }
    }
  ]
}
```

### 8.5 Resolution and Batch Size Impact Analysis

For computer vision models, analyze how different input resolutions and batch sizes affect performance.

**CLI Usage:**

```bash
python -m onnxruntime.tools.model_inspect model.onnx \
  --input-resolution 224x224,384x384,512x512,640x640 \
  --batch-sizes 1,2,4,8,16,32 \
  --hardware rtx_4090
```

**Resolution Scaling Analysis:**

| Resolution | FLOPs | Memory | Latency Est. | Throughput Est. |
|------------|-------|--------|--------------|-----------------|
| 224x224    | 4.1G  | 1.2GB  | 2.1ms        | 476 fps         |
| 384x384    | 12.1G | 2.8GB  | 5.8ms        | 172 fps         |
| 512x512    | 21.5G | 4.9GB  | 10.2ms       | 98 fps          |
| 640x640    | 33.6G | 7.6GB  | 15.9ms       | 63 fps          |

**Batch Size Scaling Analysis:**

| Batch Size | VRAM Used | Latency | Throughput | GPU Util |
|------------|-----------|---------|------------|----------|
| 1          | 1.2GB     | 2.1ms   | 476 fps    | 23%      |
| 2          | 1.8GB     | 2.4ms   | 833 fps    | 41%      |
| 4          | 3.1GB     | 3.1ms   | 1290 fps   | 62%      |
| 8          | 5.6GB     | 4.8ms   | 1667 fps   | 78%      |
| 16         | 10.7GB    | 8.2ms   | 1951 fps   | 89%      |
| 32         | 20.9GB    | 15.1ms  | 2119 fps   | 94%      |

**Visualization:**

Generate trade-off charts showing:
- Resolution vs Latency curve
- Resolution vs Memory curve
- Batch size vs Throughput curve
- Optimal batch size recommendation for given VRAM constraint

### 8.6 Layer-wise Quantization Analysis (TRT EngineXplorer-inspired)

Inspired by TensorRT EngineXplorer, provide detailed layer-by-layer precision breakdown.

**Layer Precision Breakdown:**

| Layer | Original Precision | Quantized | Speedup | Memory Saved | Accuracy Impact |
|-------|-------------------|-----------|---------|--------------|-----------------|
| conv1 | FP32 | INT8 | 2.8x | 75% | -0.001 |
| layer1.0.conv1 | FP32 | INT8 | 3.1x | 75% | -0.002 |
| layer4.2.conv2 | FP32 | FP16 | 1.9x | 50% | -0.000 |
| fc | FP32 | FP32 | 1.0x | 0% | N/A |

**Sensitive Layer Detection:**

Identify layers where quantization causes significant accuracy degradation:

```json
"sensitive_layers": [
  {
    "layer": "layer3.1.conv2",
    "quantization": "int8",
    "accuracy_drop": 0.015,
    "recommendation": "Keep at FP16 for better accuracy/speed tradeoff"
  }
]
```

**Engine Summary Panel (TRT-style):**

```
+------------------------------------------+
|           Engine Summary                 |
+------------------------------------------+
| Total Layers: 53                         |
| FP32 Layers: 4 (7.5%)                    |
| FP16 Layers: 12 (22.6%)                  |
| INT8 Layers: 37 (69.9%)                  |
+------------------------------------------+
| Model Size: 25.4 MB (vs 97.8 MB FP32)    |
| Speedup: 3.2x                            |
| Accuracy: 92.7% (vs 93.1% FP32)          |
+------------------------------------------+
```

**Calibration Recommendations:**

```markdown
## Quantization Recommendations

1. **Overall Strategy**: Mixed-precision INT8/FP16 recommended
2. **Sensitive Layers**: Keep layers 47-53 (classifier head) at FP16
3. **Calibration Dataset**: Use 1000+ representative samples
4. **Expected Trade-off**: 3.2x speedup for 0.4% accuracy loss
```

---

## 9. External Pipeline Integration

### 9.1 Design Philosophy

ONNX Autodoc is **evaluation-source agnostic**. It does not run benchmarks itself; instead, it consumes eval/perf metrics from external pipelines in a small, generic JSON format.

### 9.2 Generic Eval/Perf JSON Schema

External eval scripts (YOLO, ResNet, BERT, etc.) are expected to emit JSON of the form:

```json
{
  "model_id": "resnet50_fp16",
  "precision": "fp16",
  "task_type": "image_classification",
  "eval": {
    "primary_metric_name": "f1_macro",
    "primary_metric_value": 0.923,
    "metrics": {
      "f1_macro": 0.923,
      "precision_macro": 0.931,
      "recall_macro": 0.917,
      "mAP_50_95": 0.611
    }
  },
  "perf": {
    "latency_ms_p50": 8.2,
    "latency_ms_p95": 9.5,
    "throughput_qps": 1200,
    "gpu_utilization_pct": 78,
    "vram_used_bytes": 2852126720
  }
}
```

**Notes:**
- `task_type` is free-form (e.g., `object_detection`, `nlp`, `speech_recognition`)
- `eval.metrics` can contain any task-specific metrics
- `perf` fields are optional; if missing, Autodoc will still work with architecture + hardware estimates alone

### 9.3 Integration Points

This schema is intentionally minimal so it can be produced by:
- YOLO batch eval pipelines
- ONNX Runtime perf test wrappers
- Custom eval scripts in any language

---

## 10. Testing Strategy

### 10.1 Test Categories

| Category | Scope | Tools |
|----------|-------|-------|
| Unit Tests | Individual functions (param counting, FLOP estimation, risk detection) | pytest |
| Integration Tests | End-to-end CLI runs with real ONNX models | pytest + subprocess |
| Regression Tests | Output stability across code changes | snapshot testing |
| Visual Tests | Chart generation verification | matplotlib + image comparison |

### 10.2 Test Models

Use well-known ONNX models from the ONNX Model Zoo:

| Model | Type | Purpose |
|-------|------|---------|
| ResNet-50 | CNN | Basic conv/bn/relu patterns, residual blocks |
| BERT-base | Transformer | Attention patterns, large param counts |
| YOLOv8-n | Detection | Multi-scale features, complex outputs |
| MobileNetV2 | Mobile CNN | Depthwise separable convs, small model |

### 10.3 Test File Structure

```
onnxruntime/test/python/tools/
    test_model_inspect.py           # Unit tests for core analysis
    test_model_inspect_compare.py   # Unit tests for compare mode
    test_visualizations.py          # Chart generation tests
    test_cli.py                     # CLI integration tests
    fixtures/
        resnet50.onnx              # Test models (or download scripts)
        bert_tiny.onnx
```

### 10.4 Test Coverage Targets

| Component | Target Coverage |
|-----------|----------------|
| Param counting | 95% |
| FLOP estimation | 90% |
| Risk heuristics | 85% |
| CLI argument parsing | 100% |
| JSON schema compliance | 100% |

---

## 11. Dependencies

### 11.1 Required Dependencies

```txt
# Core dependencies (already in ONNX Runtime)
onnx>=1.14.0
numpy>=1.21.0

# Additional for model_inspect
protobuf>=3.20.0
```

### 11.2 Optional Dependencies

```txt
# For visualization
matplotlib>=3.5.0

# For LLM summarization
openai>=1.0.0  # or anthropic, etc.

# For HTML reports
jinja2>=3.0.0
```

### 11.3 Development Dependencies

```txt
pytest>=7.0.0
pytest-cov>=4.0.0
black>=23.0.0
ruff>=0.1.0
mypy>=1.0.0
```

### 11.4 Installation

```bash
# Minimal install (JSON/Markdown output only)
pip install onnxruntime

# With visualization support
pip install onnxruntime[autodoc-viz]

# With LLM summarization
pip install onnxruntime[autodoc-llm]

# Full install
pip install onnxruntime[autodoc-full]
```

---

## 12. Error Handling

### 12.1 Error Categories

| Category | Example | Handling |
|----------|---------|----------|
| **Input Errors** | Invalid ONNX file, missing file | Early exit with clear error message |
| **Analysis Errors** | Unsupported op, shape inference failure | Log warning, continue with fallback estimates |
| **Output Errors** | Can't write to path, permission denied | Raise with actionable message |
| **External Errors** | LLM API failure, network timeout | Graceful degradation (skip LLM summary) |

### 12.2 Error Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Invalid input (file not found, invalid ONNX) |
| 2 | Analysis error (partial results available) |
| 3 | Output error (couldn't write files) |
| 4 | External service error (LLM, network) |

### 12.3 Logging Levels

| Level | Use Case |
|-------|----------|
| DEBUG | Detailed per-node analysis, shape inference steps |
| INFO | Progress updates, summary stats |
| WARNING | Unsupported ops, fallback estimates used |
| ERROR | Failures that prevent completion |

### 12.4 Graceful Degradation

The tool should always produce *some* output, even if partial:

- If shape inference fails for some nodes -> estimate params/FLOPs as 0, flag in warnings
- If matplotlib unavailable -> skip visualizations, produce JSON/Markdown only
- If LLM API fails -> skip LLM summary, produce rule-based summary only

---

## 13. Implementation Timeline

### Day 1 - Repo Exploration and Design

- Fork `microsoft/onnxruntime`
- Build ONNX Runtime locally; run core tests and existing tools (e.g., `check_onnx_model_mobile_usability`)
- Identify where graph structures are represented (C++ graph classes, Python ONNX loader)
- Finalize scope and choose MVP path:
  - Phase 1: Python-only using `onnx` library
  - Phase 2 (stretch): C++ engine with Python wrapper
- Draft detailed design doc in the repo

### Day 2 - Graph Parsing and Basic Metrics

- Implement core graph loader (MVP in Python):
  - Load ONNX model using `onnx`
  - Extract graph, nodes, initializers, shapes
- Implement param counting per node and globally
- Implement FLOPs estimation for key ops:
  - Conv, Gemm, MatMul, Add, Mul
- Write initial JSON emission code

### Day 3 - Advanced Metrics and Block Grouping

- Add block detection:
  - Group Conv+BN+Relu patterns into logical blocks
  - Identify transformer blocks (SelfAttention + MLP pattern) where possible
- Estimate activation sizes and peak memory by walking a topological order
- Add attention metrics for known patterns (BERT/ViT-like models)
- Extend JSON schema to include `blocks` and `metrics.attention`

### Day 4 - Risk Signals and CLI Integration

- Implement heuristic checks:
  - Excessive depth without skips
  - Oversized dense layers
  - Potentially problematic dynamic shapes for deployment
- Implement CLI entrypoint:
  - `python -m onnxruntime.tools.model_inspect`
- Validate CLI on a set of known models (ResNet, BERT, YOLO, etc.)

### Day 5 - Model Card Generation and LLM Summaries

- Implement Markdown model card generator:
  - High-level description
  - Key metrics table (params, FLOPs, memory, attention stats)
  - Risk signals table
- Implement optional LLM integration:
  - Reads API key from environment
  - Summarizes JSON report into short + long narrative
- Add toggles so users without LLM access can still use the tool fully

### Day 6 - Testing, Docs, and Examples

- Add unit tests for:
  - Param counting
  - FLOPs estimation
  - Activation size computation
  - Risk heuristics
- Add example scripts under `samples/`
- Write README for the tool

### Day 7 - Polish and Gauntlet Story

- Clean up code, add comments, and ensure style compliance
- Capture before/after samples for a couple of well-known models
- Prepare Gauntlet deliverables:
  - High-level writeup of how you navigated ONNX Runtime
  - Screenshots/snippets of JSON + Markdown outputs
  - Reflection on design choices and tradeoffs

---

## 14. SaaS Architecture and Distribution

### 14.1 Overview

Transform ONNX Autodoc from a CLI tool into a full SaaS web application, similar to Weights & Biases, enabling team collaboration, model history, and easy access without local installation.

### 14.2 System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Frontend (Vercel)                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────────┐ │
│  │   Upload    │  │  Dashboard  │  │   Report    │  │  Compare   │ │
│  │   Component │  │   View      │  │   Viewer    │  │   View     │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
                              │ REST API / WebSocket
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Backend API (Railway/Render)                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────────┐ │
│  │   FastAPI   │  │   Auth      │  │   Job       │  │   Storage  │ │
│  │   Router    │  │   Middleware│  │   Queue     │  │   Service  │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └────────────┘ │
│                              │                                      │
│                    ┌─────────┴─────────┐                           │
│                    │  Analysis Engine  │  (existing Python code)   │
│                    └───────────────────┘                           │
└─────────────────────────────────────────────────────────────────────┘
         │                    │                      │
         ▼                    ▼                      ▼
┌─────────────┐      ┌─────────────┐        ┌─────────────┐
│  PostgreSQL │      │  S3/R2      │        │  Redis      │
│  (Supabase) │      │  (Models)   │        │  (Queue)    │
└─────────────┘      └─────────────┘        └─────────────┘
```

### 14.3 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/models` | GET | List user's models |
| `/api/models` | POST | Upload new model (multipart) |
| `/api/models/{id}` | GET | Get model details |
| `/api/models/{id}/report` | GET | Get analysis report (JSON/HTML) |
| `/api/models/{id}/compare` | POST | Compare with other models |
| `/api/jobs/{id}` | GET | Check analysis job status |
| `/api/jobs/{id}/ws` | WS | Real-time analysis progress |

**Upload Flow:**

```
Client                    API                     Worker
  │                        │                        │
  │──POST /models─────────►│                        │
  │  (multipart file)      │                        │
  │                        │──Store to S3──────────►│
  │                        │                        │
  │                        │──Queue job────────────►│
  │◄─202 Accepted──────────│                        │
  │  {job_id, status_url}  │                        │
  │                        │                        │
  │──WS /jobs/{id}/ws─────►│                        │
  │◄─progress: loading─────│◄─────────────────────►│
  │◄─progress: analyzing───│                        │
  │◄─progress: complete────│                        │
  │                        │                        │
  │──GET /models/{id}─────►│                        │
  │◄─{report_data}─────────│                        │
```

### 14.4 Frontend Components

| Component | Description |
|-----------|-------------|
| **UploadZone** | Drag-and-drop model upload with progress bar |
| **ModelList** | Paginated list of analyzed models with search/filter |
| **ReportViewer** | Renders HTML report in iframe or React components |
| **GraphViewer** | Interactive Netron-style graph visualization |
| **ComparePanel** | Side-by-side model comparison |
| **TeamSettings** | Workspace management, member invites |

### 14.5 Authentication and Authorization

| Provider | Pros | Implementation |
|----------|------|----------------|
| **Clerk** | Best DX, great React components | `@clerk/nextjs` |
| **Supabase Auth** | Integrated with DB, generous free tier | `@supabase/auth-helpers-nextjs` |
| **Auth0** | Enterprise features, SAML/SSO | `@auth0/nextjs-auth0` |

**Permission Model:**

| Role | Capabilities |
|------|--------------|
| **Owner** | Full access, billing, delete workspace |
| **Admin** | Manage members, all model operations |
| **Editor** | Upload, analyze, compare models |
| **Viewer** | View reports only |

### 14.6 Infrastructure and Costs

| Service | Provider | Estimated Cost (MVP) |
|---------|----------|---------------------|
| Frontend | Vercel (Hobby) | $0/mo |
| Backend | Railway (Starter) | $5/mo |
| Database | Supabase (Free) | $0/mo |
| Storage | Cloudflare R2 | $0.015/GB/mo |
| Queue | Upstash Redis | $0/mo (free tier) |
| **Total MVP** | | **~$5-10/mo** |

### 14.7 Security Considerations

| Concern | Mitigation |
|---------|------------|
| Model privacy | Models stored encrypted, deleted after 30 days (configurable) |
| API abuse | Rate limiting, file size limits (500MB default) |
| Data isolation | Tenant isolation at DB level, S3 prefix per workspace |
| CORS | Strict origin whitelist |

### 14.8 Standalone Package Distribution (PRIMARY PATH)

**Decision: Greenfield over ORT Fork**

The original plan embedded ONNX Autodoc within the ONNX Runtime repository. After analysis, we're pivoting to a standalone package because:

| Factor | ORT Fork | Standalone |
|--------|----------|------------|
| User installation | Clone fork, navigate to tools | `pip install onnx-autodoc` |
| Dependencies | Confusing (ORT vs onnx) | Clean (just `onnx` + `numpy`) |
| Iteration speed | Slow (huge repo) | Fast |
| Distribution | Blocked | Immediate |
| Functionality lost | N/A | **Zero** |

**Key Insight:** We use `onnx.load()` not ORT sessions. All our dependencies are already on PyPI.

**Installation:**

```bash
# From PyPI
pip install onnx-autodoc

# With visualization support
pip install onnx-autodoc[viz]

# With LLM summaries
pip install onnx-autodoc[llm]

# With PDF generation
pip install onnx-autodoc[pdf]

# Everything
pip install onnx-autodoc[full]

# CLI usage
onnx-autodoc analyze model.onnx --html report.html
onnx-autodoc analyze model.onnx --hardware auto --llm-summary
```

**Package Structure:**

```
onnx-autodoc/
├── pyproject.toml
├── README.md
├── LICENSE
├── src/
│   └── onnx_autodoc/
│       ├── __init__.py
│       ├── cli.py              # Click-based CLI
│       ├── analyzer.py         # Core analysis engine
│       ├── patterns.py         # Block/architecture detection
│       ├── risks.py            # Risk heuristics
│       ├── report.py           # JSON/MD/HTML generation
│       ├── visualizations.py   # Matplotlib charts
│       ├── hardware.py         # GPU profiles and estimates
│       ├── llm_summarizer.py   # OpenAI integration
│       ├── pdf_generator.py    # Playwright PDF
│       └── converters/
│           ├── pytorch.py      # PyTorch → ONNX
│           └── tensorflow.py   # TensorFlow → ONNX (future)
├── tests/
├── streamlit_app.py            # Web UI
└── Dockerfile
```

### 14.9 Streamlit Web UI

For maximum accessibility, provide a web interface that requires no installation.

**Features:**

| Feature | Description |
|---------|-------------|
| File Upload | Drag-and-drop ONNX, .pt, .pb files |
| Hardware Selection | Dropdown of 50+ GPU profiles |
| Report Viewer | Embedded HTML report with charts |
| Download Options | JSON, Markdown, PDF export |
| Model Comparison | Side-by-side analysis of 2 models |
| LLM Summary | Optional (user provides API key) |

**Deployment Options:**

| Platform | Cost | GPU Support | URL |
|----------|------|-------------|-----|
| Hugging Face Spaces | Free | Yes (ZeroGPU) | `hf.co/spaces/username/onnx-autodoc` |
| Streamlit Cloud | Free | No | `share.streamlit.io/...` |
| Self-hosted | Varies | Yes | Your infrastructure |

**Streamlit App Structure:**

```python
import streamlit as st
from onnx_autodoc import analyze_model, generate_html_report

st.title("ONNX Autodoc")

uploaded_file = st.file_uploader("Upload ONNX model", type=["onnx"])
hardware = st.selectbox("Target Hardware", list_hardware_profiles())

if uploaded_file and st.button("Analyze"):
    with st.spinner("Analyzing model..."):
        report = analyze_model(uploaded_file, hardware=hardware)
        st.components.v1.html(generate_html_report(report), height=800)
        st.download_button("Download PDF", generate_pdf(report))
```

---

## 15. Inference Platform (Wide Hole Architecture)

### 15.1 Design Philosophy

Build the **platform layer** that enables inference-based analysis, without going deep on task-specific metrics until there's demand.

```
"Dig a wide hole. When someone pays to go deeper, dig deeper there."
```

**Platform = Wide:** Infrastructure that works for any model type
**Plugins = Deep:** Task-specific metrics added on demand (detection mAP, NLP BLEU, etc.)

### 15.2 Core Components

| Component | Purpose | Scope |
|-----------|---------|-------|
| `InferenceRunner` | Wrap ORT session, measure latency | Platform |
| `DataLoader` interface | Load test data (images, arrays, files) | Platform |
| `MetricsCalculator` interface | Extension point for task-specific metrics | Platform |
| `PrecisionComparator` | Run fp32/fp16/int8, compare results | Platform |
| Plugin system | Register custom metrics calculators | Platform |

### 15.3 InferenceRunner API

```python
from onnx_autodoc.inference import InferenceRunner

runner = InferenceRunner("model.onnx")

# Basic inference with timing
results = runner.run(test_data, warmup=10, iterations=100)

print(results.latency_p50_ms)      # 12.3
print(results.latency_p99_ms)      # 15.1
print(results.throughput_per_sec)  # 83.2
print(results.outputs)             # Raw model outputs
```

### 15.4 DataLoader Interface

```python
from abc import ABC, abstractmethod

class DataLoader(ABC):
    @abstractmethod
    def __iter__(self):
        """Yield (input_dict, optional_labels) tuples."""
        pass
    
    @abstractmethod
    def __len__(self):
        """Return number of samples."""
        pass

# Built-in loaders (platform layer)
class ImageFolderLoader(DataLoader): ...
class NumpyArrayLoader(DataLoader): ...

# User can implement custom loaders
class MyCustomLoader(DataLoader): ...
```

### 15.5 MetricsCalculator Interface (Extension Point)

```python
from abc import ABC, abstractmethod

class MetricsCalculator(ABC):
    @abstractmethod
    def calculate(self, outputs, labels) -> dict:
        """Calculate task-specific metrics."""
        pass

# Built-in: just latency, no accuracy
class RawOutputMetrics(MetricsCalculator):
    def calculate(self, outputs, labels):
        return {}  # No accuracy metrics, just timing

# Example plugin (added when someone needs it)
class DetectionMetrics(MetricsCalculator):
    def calculate(self, outputs, labels):
        return {"mAP_50": ..., "mAP_50_95": ...}

# Registration
runner.register_metrics(DetectionMetrics())
```

### 15.6 Multi-Precision Comparison

```python
from onnx_autodoc.inference import compare_precisions

results = compare_precisions(
    models={
        "fp32": "model_fp32.onnx",
        "fp16": "model_fp16.onnx",
        "int8": "model_int8.onnx",
    },
    test_data=loader,
    metrics_calculator=None,  # Optional
)

# Results include:
# - Latency comparison (which is fastest)
# - Output comparison (numerical diff, cosine similarity)
# - Memory usage per precision
```

### 15.7 What's NOT Built Yet (Dig Deeper Later)

| Feature | Status | Build When |
|---------|--------|------------|
| Detection mAP (COCO format) | Not built | Customer needs it |
| Classification accuracy/F1 | Not built | Customer needs it |
| NLP metrics (BLEU, ROUGE) | Not built | Customer needs it |
| Segmentation IoU | Not built | Customer needs it |
| Custom dataset formats | Not built | Customer needs it |

These are **vertical extensions** that plug into the horizontal platform.

### 15.8 Dependency

```
onnx-autodoc[inference]  # Adds onnxruntime-gpu
```

This is a pip package dependency, **not** the ORT fork.

---

## Appendix: Delta Log

*Use this section to track changes to the PRD over time.*

| Date | Section | Change | Reason |
|------|---------|--------|--------|
| Dec 2025 | Initial | Created unified PRD from starter pack + visualization extension | Consolidation |
| Dec 2025 | Structure | Split backlog into BACKLOG.md, brainlift into BRAINLIFT.md | Context window optimization |
| Dec 2, 2025 | 4.3 | Scaffolding complete: `tools/python/util/autodoc/` with analyzer, patterns, risks, report modules | Following ORT patterns from mobile_helpers |
| Dec 2, 2025 | Risk Signals | Added minimum thresholds for risk signals (1M+ params, 1B+ FLOPs) to avoid flagging trivial models | Common sense - don't recommend optimization for 13-param models |
| Dec 2, 2025 | README | Updated README.md to match actual implementation: correct paths, CLI flags, hardware profiles | Documentation accuracy |
| Dec 2, 2025 | Testing | Created comprehensive unit test suite: test_analyzer.py, test_patterns.py, test_risks.py, test_hardware.py, test_report.py | Code quality and regression prevention |
| Dec 2, 2025 | CI/CD | Added `.github/workflows/autodoc-ci.yml` with lint, type check, unit tests, and integration tests | Automated quality gates |
| Dec 2, 2025 | Visualization | Added `visualizations.py` with matplotlib Agg backend, dark theme, 4 chart types, CLI `--with-plots`/`--assets-dir`, Markdown embedding, 17 tests | Epic 5 complete |
| Dec 2, 2025 | Build | C++ ONNX Runtime build complete with CUDA provider (357MB); Python wheel build in progress | Environment setup milestone |
| Dec 2, 2025 | LLM | Added `llm_summarizer.py` with OpenAI integration, prompt templates, CLI `--llm-summary`/`--llm-model`, Executive Summary in Markdown | Epic 7 complete |
| Dec 2, 2025 | Hardware | Adding GPU Saturation metric (model_flops/gpu_capacity) alongside compute utilization | Better hardware insight |
| Dec 2, 2025 | PyTorch | Added PyTorch-to-ONNX conversion (--from-pytorch, --input-shape, --keep-onnx), Ultralytics metadata extraction, DatasetInfo in reports | Epic 4B implementation |
| Dec 2, 2025 | Attention FLOPs | Added _estimate_attention_flops() with full formula: 3*seq*d² + 2*heads*seq²*d_head + 5*heads*seq² + seq*d² | Task 2.3.3 complete |
| Dec 2, 2025 | KV Cache | Added KV cache estimation for transformers (kv_cache_bytes_per_token, kv_cache_bytes_full_context, kv_cache_config) | Task 2.4.3 complete |
| Dec 2, 2025 | Memory Breakdown | Added MemoryBreakdown dataclass with weights_by_op_type, activations_by_op_type, largest_weights/activations | Task 2.4.4 complete |
| Dec 2, 2025 | Progress | Added --progress CLI flag with step-by-step display for large model analysis | Task 4.1.3 complete |
| Dec 2, 2025 | Risk Thresholds | Added RiskThresholds dataclass for configurable severity thresholds | Task 3.2.5 complete |
| Dec 2, 2025 | HTML Parity | Identified gap: HTML missing Operator Distribution, KV Cache, Memory Breakdown, Architecture sections | Added Story 4.4 to backlog |
| Dec 2, 2025 | Non-standard Residuals | Added detect_nonstandard_residual_blocks() for Concat/Gated/Sub skip patterns, check_nonstandard_residuals() risk signal, Architecture sections in MD/HTML reports | Task 3.2.4 complete, Epic 3 complete |
| Dec 2, 2025 | HTML Parity | Added Operator Distribution table, KV Cache section, Memory Breakdown table to HTML report; CSS styling for .kv-cache, .memory-breakdown sections | Story 4.4 complete (4/4 tasks) |
| Dec 2, 2025 | JSON Schema | Added schema.py with Draft 7 JSON schema for InspectionReport, validate_report()/validate_report_strict() functions, report.validate() method, 12 new tests | Task 4.2.2 complete, Epic 4 complete |
| Dec 2, 2025 | Shared Weights | Added fractional weight attribution (Option C) for shared weights - by_op_type sums correctly without over/under counting. Added shared_weights dict with details, num_shared_weights count | Task 2.2.4 edge case 1 |
| Dec 2, 2025 | Quantized Params | Added quantization detection: QUANTIZED_OPS set (15 ops), QUANTIZED_DTYPES set, precision_breakdown dict, is_quantized bool, quantized_ops list. Updated JSON schema, MD/HTML reports | Task 2.2.4 edge case 2 |
| Dec 2, 2025 | Tests | Added 8 new tests: TestSharedWeights (3 tests), TestQuantizedParams (5 tests) covering fractional attribution, precision breakdown, INT8/FP16 detection | Task 2.2.4 complete |
| Dec 2, 2025 | GPU Variants | Added 50+ GPU profiles: H100 (SXM/PCIe/NVL), A100 (40/80GB x SXM/PCIe), V100 variants, RTX 40-series (4090-4060), RTX 30-series (3090Ti-3050), laptop variants | Story 6.5 complete |
| Dec 2, 2025 | Multi-GPU | Added MultiGPUProfile dataclass, create_multi_gpu_profile(), NVLink bandwidth modeling, tensor/pipeline parallelism overhead estimation, DGX H100/A100 profiles | Story 6.6 complete |
| Dec 2, 2025 | Cloud | Added CloudInstanceProfile dataclass, 17 cloud instances (AWS p5/p4d/g5/inf2, Azure NC/ND, GCP a3/a2/g2), hourly cost estimates, --cloud and --list-cloud CLI flags | Story 6.7 complete |
| Dec 2, 2025 | CLI | Added --gpu-count N for multi-GPU scaling, --cloud for cloud instances, --list-cloud to list instances, --out-pdf for PDF reports | CLI enhancements |
| Dec 2, 2025 | PDF | Added pdf_generator.py with Playwright-based PDF generation, PDFGenerator class, --out-pdf CLI flag, header/footer templates | Task 5.3.4 complete |
| Dec 2, 2025 | ML Feedback | Added Section 6.7-6.8 (Graph Viz, Per-Layer Summary), Section 7.5 (HW Recommendations), Section 8.5-8.6 (Resolution/Batch, Layer Quantization), Section 14 (SaaS Architecture) | ML Engineer/MLOps feedback integration |
| Dec 2, 2025 | Backlog | Added Epic 4C (TF Conversion), Epic 10 (SaaS), Epic 10B (PyPI), Stories 5.4-5.5, 6.4 enhancements, Stories 6.8-6.9 | Feature roadmap expansion |
| Dec 3, 2025 | Distribution | **PIVOT**: Greenfield standalone package over ORT fork. Zero functionality lost, immediate PyPI distribution | Distribution was blocked by fork dependency |
| Dec 3, 2025 | Priority | Reordered epics: P0 = Standalone Package + Streamlit UI. Ship usable software first | "Science fair project" → "Real software" pivot |
| Dec 3, 2025 | Streamlit | Added Section 14.9 with Streamlit Web UI spec, HF Spaces deployment plan | Maximize accessibility without installation |
| Dec 3, 2025 | Inference | Added Section 15: Inference Platform with "wide hole" architecture. InferenceRunner, DataLoader interface, MetricsCalculator extension point | Platform-first approach: build horizontal, go vertical when paid |
| Dec 3, 2025 | Backlog | Added Epic 12: Inference Platform (24 tasks across 5 stories). Platform layer for inference, not task-specific metrics | Extensible architecture for future customer needs |
