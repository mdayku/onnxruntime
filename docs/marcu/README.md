# ONNX Autodoc

**Model Architecture Inspector for ONNX Runtime**

A graph-level analysis and documentation tool that inspects ONNX model architectures, computes static complexity metrics, surfaces structural risk signals, and auto-generates human-readable reports.

---

## What is ONNX Autodoc?

ONNX Autodoc answers the question:

> *"What is this model structurally, how expensive is it, what are its likely failure modes, and how should we position it in our system?"*

It provides:

- **Parameter counts** - Per node, per block, and globally
- **FLOP estimates** - Identify compute hotspots
- **Memory analysis** - Peak activation memory and VRAM requirements
- **Risk signals** - Detect problematic architecture patterns
- **Hardware estimates** - GPU utilization predictions for 30+ NVIDIA profiles (including Jetson)
- **Visualizations** - Operator histograms, parameter/FLOPs distribution charts
- **LLM Summaries** - AI-generated executive summaries (optional, requires OpenAI API key)
- **Shareable Reports** - HTML (single file), Markdown, and JSON output formats

---

## Quick Start

### Basic Usage

```bash
# From the tools/python directory:
cd tools/python

# Inspect a single model (auto-detects local hardware)
python model_inspect.py model.onnx

# Use a specific hardware profile
python model_inspect.py model.onnx --hardware a100

# Generate JSON report
python model_inspect.py model.onnx --out-json report.json

# Generate Markdown model card
python model_inspect.py model.onnx --out-md model_card.md

# Generate shareable HTML report (single file with embedded images)
python model_inspect.py model.onnx --out-html report.html --with-plots

# Add AI-generated executive summary (requires OPENAI_API_KEY)
python model_inspect.py model.onnx --llm-summary --out-html report.html

# Full report with everything
python model_inspect.py model.onnx --hardware auto --with-plots --llm-summary --out-html full_report.html

# Specify precision and batch size for hardware estimates
python model_inspect.py model.onnx --hardware rtx4090 --precision fp16 --batch-size 8
```

### List Available Hardware Profiles

```bash
python model_inspect.py --list-hardware
```

Example output:
```
======================================================================
Available Hardware Profiles
======================================================================

Data Center GPUs (Current Gen):
  h100                 NVIDIA H100 SXM                  80 GB  1979.0 TF16
  a100-80gb            NVIDIA A100 80GB                 80 GB   312.0 TF16
  a100-40gb            NVIDIA A100 40GB                 40 GB   312.0 TF16
  ...

Jetson Edge/Embedded (Orin Series):
  jetson-agx-orin-64gb NVIDIA Jetson AGX Orin 64GB      64 GB    10.6 TF16
  jetson-orin-nano-4gb NVIDIA Jetson Orin Nano 4GB       4 GB     1.2 TF16
  ...

Consumer GPUs:
  rtx4090              NVIDIA RTX 4090                  24 GB   165.0 TF16
  ...
======================================================================
```

---

## Example Output

### Console Summary

```
============================================================
Model: resnet50.onnx
============================================================

Nodes: 122
Inputs: 1
Outputs: 1
Initializers: 161

Parameters: 25.56M
FLOPs: 4.11B
Model Size: 97.69 MB

Architecture: cnn
Detected Blocks: 53

--- Hardware Estimates (NVIDIA A100 80GB) ---
Precision: fp16, Batch Size: 1
VRAM Required: 61.31 MB
Fits in VRAM: Yes
Theoretical Latency: 0.01 ms
Bottleneck: memory_bandwidth

Risk Signals: 1
  [INFO] compute_bottleneck
============================================================
```

### JSON Report (excerpt)

```json
{
  "metadata": {
    "path": "resnet50.onnx",
    "ir_version": 8,
    "producer_name": "pytorch",
    "opsets": {"ai.onnx": 17}
  },
  "graph_summary": {
    "num_nodes": 122,
    "num_inputs": 1,
    "num_outputs": 1,
    "op_type_counts": {"Conv": 53, "BatchNormalization": 53, "Relu": 49, ...}
  },
  "param_counts": {
    "total": 25557032,
    "trainable": 25557032
  },
  "flop_counts": {
    "total": 4110000000
  },
  "hardware_estimates": {
    "device": "NVIDIA A100 80GB",
    "precision": "fp16",
    "vram_required_gb": 0.06,
    "fits_in_vram": true,
    "theoretical_latency_ms": 0.01,
    "bottleneck": "memory_bandwidth"
  },
  "risk_signals": [
    {
      "id": "compute_bottleneck",
      "severity": "info",
      "description": "The following operations dominate compute..."
    }
  ]
}
```

---

## Prerequisites

### Required

- Python 3.10+
- `onnx` library (for model loading and shape inference)
- `numpy` (for array operations)

### Optional

- `psutil` - For CPU memory detection (auto-detect mode)
- `matplotlib` - For visualization charts (`--with-plots`)
- `openai` - For LLM-powered summaries (`--llm-summary`)

### Installation

```bash
# Clone the fork
git clone https://github.com/YOUR_USERNAME/onnxruntime.git
cd onnxruntime

# Install dependencies
pip install onnx numpy protobuf

# Optional: for full feature set
pip install psutil matplotlib openai
```

### LLM Summary Setup (Optional)

To enable AI-generated executive summaries:

```bash
# Set your OpenAI API key
# Linux/macOS:
export OPENAI_API_KEY="sk-your-key-here"

# Windows PowerShell:
$env:OPENAI_API_KEY = "sk-your-key-here"

# Windows (Permanent):
[System.Environment]::SetEnvironmentVariable("OPENAI_API_KEY", "sk-your-key", "User")
```

---

## Project Structure

```
onnxruntime/
  tools/
    python/
      model_inspect.py              # CLI entry point
      util/
        model_inspect.py            # CLI implementation
        autodoc/
          __init__.py               # Public API exports
          analyzer.py               # Graph loading, param/FLOP counting
          hardware.py               # 30+ hardware profiles, auto-detection
          patterns.py               # Block detection (Conv-BN-Relu, Attention, etc.)
          risks.py                  # Risk signal heuristics
          report.py                 # ModelInspector orchestrator, JSON/Markdown/HTML output
          visualizations.py         # Chart generation (matplotlib)
          llm_summarizer.py         # LLM-powered summaries (OpenAI)
          tests/                    # Unit and integration tests

docs/marcu/
  README.md                         # This file
  PRD.md                            # Product Requirements Document
  BACKLOG.md                        # Epic/Story/Task tracking
  Architecture.md                   # System architecture
```

---

## Documentation

| Document | Description |
|----------|-------------|
| [PRD.md](PRD.md) | Product requirements, specs, and JSON schemas |
| [BACKLOG.md](BACKLOG.md) | Epic/Story/Task tracking, Jira-style |
| [Architecture.md](Architecture.md) | System design and component diagrams |

---

## CLI Reference

### `model_inspect`

```
usage: model_inspect.py [-h] [--out-json PATH] [--out-md PATH]
                        [--hardware PROFILE] [--list-hardware]
                        [--precision {fp32,fp16,bf16,int8}]
                        [--batch-size N] [--log-level {debug,info,warning,error}]
                        [--quiet]
                        [model_path]

Analyze an ONNX model and generate architecture documentation.

positional arguments:
  model_path            Path to the ONNX model file to analyze.

optional arguments:
  -h, --help            show this help message and exit
  --out-json PATH       Output path for JSON report.
  --out-md PATH         Output path for Markdown model card.
  --quiet               Suppress console output.
  --log-level {debug,info,warning,error}
                        Logging verbosity level (default: info).

Hardware Options:
  --hardware PROFILE    Hardware profile for performance estimates.
                        Use 'auto' to detect local hardware, or specify
                        a profile name (e.g., 'a100', 'rtx4090', 't4').
  --list-hardware       List all available hardware profiles and exit.
  --precision {fp32,fp16,bf16,int8}
                        Precision for hardware estimates (default: fp32).
  --batch-size N        Batch size for hardware estimates (default: 1).
```

### Examples

```bash
# Basic inspection with console output
python model_inspect.py model.onnx

# Auto-detect local GPU and estimate performance
python model_inspect.py model.onnx --hardware auto

# Target a specific device
python model_inspect.py model.onnx --hardware jetson-orin-nano-4gb

# Generate full reports for CI/CD pipeline
python model_inspect.py model.onnx \
  --hardware a100 \
  --precision fp16 \
  --batch-size 32 \
  --out-json artifacts/report.json \
  --out-md artifacts/model_card.md
```

---

## Supported Hardware Profiles

### Data Center GPUs
- **Current Gen**: H100, A100 (40GB/80GB), A10, L4, L40, L40S, T4
- **Previous Gen**: V100 (16GB/32GB), P100, P40

### NVIDIA Jetson Edge/Embedded
- **Orin Series**: AGX Orin (32GB/64GB), Orin NX (8GB/16GB), Orin Nano (4GB/8GB)
- **Xavier Series**: AGX Xavier, Xavier NX
- **Legacy**: TX2, Nano (2GB/4GB)

### Consumer GPUs
- **RTX 40 Series**: 4090, 4080
- **RTX 30 Series**: 3090, 3080

### Other
- `auto` - Auto-detect local GPU via `nvidia-smi`
- `cpu` - Generic CPU profile

---

## Risk Signals

ONNX Autodoc detects common architectural concerns:

| Signal ID | Severity | Description |
|-----------|----------|-------------|
| `no_skip_connections` | warning | Deep network without residual connections |
| `oversized_dense` | info | Dense layers with >100M parameters |
| `dynamic_input_shapes` | info | Inputs with symbolic/dynamic dimensions |
| `missing_normalization` | info | No BatchNorm/LayerNorm in deep network |
| `compute_bottleneck` | info | Single operation using >50% of FLOPs |
| `large_embedding` | info | Embedding tables with >500M parameters |
| `unusual_activations` | info | Non-standard activation functions |
| `no_activations` | warning | Linear layers without activations |

---

## Development

### Running Tests

```bash
# From repo root
pytest tools/python/util/autodoc/tests/ -v
```

### Code Style

```bash
# Format code
black tools/python/util/autodoc/

# Lint
ruff check tools/python/util/autodoc/

# Type check
mypy tools/python/util/autodoc/
```

---

## Roadmap

### Implemented
- [x] Core analysis engine (param counting, FLOP estimation)
- [x] Pattern detection (Conv-BN-Relu, Attention, Residual)
- [x] Risk signal heuristics
- [x] 30+ hardware profiles (data center, Jetson, consumer)
- [x] Auto-detection of local NVIDIA GPUs
- [x] JSON and Markdown output
- [x] CLI with hardware options

### Coming Soon
- [ ] Visualization module (matplotlib charts)
- [ ] Compare mode for quantized variants
- [ ] LLM-powered summaries
- [ ] HTML report output

---

## License

This project is part of the ONNX Runtime fork and follows the same license terms as the parent project.

---

## Acknowledgments

- Microsoft ONNX Runtime team for the base codebase
- ONNX Model Zoo for test models
