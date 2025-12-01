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
- **Hardware estimates** - Static GPU utilization predictions
- **Visual reports** - Matplotlib charts for stakeholder communication
- **Model cards** - Human-readable Markdown/HTML documentation

---

## Quick Start

### Basic Usage

```bash
# Inspect a single model
python -m onnxruntime.tools.model_inspect model.onnx

# Output JSON report
python -m onnxruntime.tools.model_inspect model.onnx --out-json report.json

# Generate Markdown model card
python -m onnxruntime.tools.model_inspect model.onnx --out-md model_card.md

# With visualizations
python -m onnxruntime.tools.model_inspect model.onnx \
  --out-md report.md \
  --with-plots \
  --assets-dir ./assets
```

### Hardware-Aware Analysis

```bash
# Analyze with a specific hardware target
python -m onnxruntime.tools.model_inspect model.onnx \
  --hardware-profile profiles/rtx_4090.json \
  --batch-size 8 \
  --precision fp16
```

### Compare Quantized Variants

```bash
# Compare fp32 vs fp16 vs int8 variants
python -m onnxruntime.tools.model_inspect_compare \
  --models model_fp32.onnx model_fp16.onnx model_int8.onnx \
  --eval-metrics eval_fp32.json eval_fp16.json eval_int8.json \
  --baseline-precision fp32 \
  --out-md quant_impact.md
```

---

## Example Output

### JSON Report (excerpt)

```json
{
  "metrics": {
    "total_params": 25557032,
    "total_flops": 4100000000,
    "peak_activation_bytes": 134217728
  },
  "risk_signals": [
    {
      "id": "no_skip_connections",
      "severity": "high",
      "description": "Very deep network (120 layers) with no residual or skip connections detected."
    }
  ]
}
```

### Markdown Model Card (excerpt)

```markdown
## Model Summary

| Metric | Value |
|--------|-------|
| Total Parameters | 25.6M |
| Total FLOPs | 4.1 GFLOPs |
| Peak Memory | 128 MB |

## Risk Signals

| Severity | Issue |
|----------|-------|
| HIGH | No skip connections in 120-layer network |
```

---

## Prerequisites

### Required

- Python 3.10+
- ONNX Runtime (built from this fork)
- `onnx` library

### Optional

- `matplotlib` - For visualization support
- `openai` / `anthropic` - For LLM-powered summaries
- `jinja2` - For HTML report generation

### Installation

```bash
# Clone the fork
git clone https://github.com/YOUR_USERNAME/onnxruntime.git
cd onnxruntime

# Install dependencies
pip install onnx numpy protobuf

# Optional: visualization support
pip install matplotlib

# Optional: LLM summarization
pip install openai
```

---

## Project Structure

```
onnxruntime/
  tools/
    autodoc/
      __init__.py
      analysis.py          # Core analysis engine
      visualizations.py    # Matplotlib chart generation
      render_markdown.py   # Markdown report generator
      render_html.py       # HTML report generator (optional)
  python/
    tools/
      model_inspect.py           # Main CLI
      model_inspect_compare.py   # Compare mode CLI

docs/marcu/
  README.md                      # This file
  PRD.md                         # Product Requirements Document
  Architecture.md                # System architecture
  THE UNCHARTED TERRITORY CHALLENGE.md  # Challenge requirements
  model_inspect_scaffold.md      # Code scaffolds
```

---

## Documentation

| Document | Description |
|----------|-------------|
| [PRD.md](PRD.md) | Product requirements, specs, and JSON schemas (~600 lines) |
| [BACKLOG.md](BACKLOG.md) | Epic/Story/Task tracking, Jira-style (~350 lines) |
| [BRAINLIFT.md](BRAINLIFT.md) | Daily learning logs for the challenge |
| [Architecture.md](Architecture.md) | System design and component diagrams |
| [THE UNCHARTED TERRITORY CHALLENGE.md](THE%20UNCHARTED%20TERRITORY%20CHALLENGE.md) | Challenge requirements and evaluation criteria |
| [model_inspect_scaffold.md](model_inspect_scaffold.md) | Code scaffolds for implementation |

---

## CLI Reference

### `model_inspect`

```
usage: model_inspect [-h] [--out-json PATH] [--out-md PATH] [--max-layers N]
                     [--llm-summary] [--log-level {debug,info}]
                     [--with-plots] [--assets-dir PATH] [--format {md,html}]
                     [--hardware-profile PATH] [--batch-size N] [--precision {fp32,fp16,int8}]
                     model_path

Inspect an ONNX model and generate architecture reports.

positional arguments:
  model_path            Path to .onnx model file

optional arguments:
  -h, --help            Show this help message
  --out-json PATH       Write full JSON report to PATH
  --out-md PATH         Write Markdown summary/model card
  --max-layers N        Truncate per-layer details to top N
  --llm-summary         Generate LLM-powered narrative summary
  --log-level {debug,info}
                        Logging verbosity
  --with-plots          Generate visualization assets
  --assets-dir PATH     Directory for plot PNGs
  --format {md,html}    Output format for report
  --hardware-profile PATH
                        Hardware profile JSON for estimates
  --batch-size N        Batch size for hardware estimates
  --precision {fp32,fp16,int8}
                        Precision for hardware estimates
```

### `model_inspect_compare`

```
usage: model_inspect_compare [-h] --models MODEL [MODEL ...]
                             [--eval-metrics JSON [JSON ...]]
                             [--hardware-profile PATH]
                             [--baseline-precision {fp32,fp16,int8}]
                             [--out-json PATH] [--out-md PATH]

Compare multiple ONNX model variants (e.g., quantization levels).

optional arguments:
  -h, --help            Show this help message
  --models MODEL [MODEL ...]
                        Paths to ONNX models to compare
  --eval-metrics JSON [JSON ...]
                        Corresponding eval/perf metrics JSONs
  --hardware-profile PATH
                        Hardware profile for estimates
  --baseline-precision {fp32,fp16,int8}
                        Baseline precision for delta calculations
  --out-json PATH       Write comparison report as JSON
  --out-md PATH         Write comparison report as Markdown
```

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | API key for OpenAI LLM summarization |
| `ANTHROPIC_API_KEY` | API key for Anthropic LLM summarization |
| `ONNX_AUTODOC_LOG_LEVEL` | Default log level (debug/info) |

---

## Contributing

This project is part of the Uncharted Territory Challenge. See [THE UNCHARTED TERRITORY CHALLENGE.md](THE%20UNCHARTED%20TERRITORY%20CHALLENGE.md) for context.

### Development Setup

```bash
# Install dev dependencies
pip install pytest pytest-cov black ruff mypy

# Run tests
pytest onnxruntime/test/python/tools/test_model_inspect.py

# Format code
black onnxruntime/tools/autodoc/
ruff check onnxruntime/tools/autodoc/

# Type check
mypy onnxruntime/tools/autodoc/
```

---

## License

This project is part of the ONNX Runtime fork and follows the same license terms as the parent project.

---

## Acknowledgments

- Microsoft ONNX Runtime team for the base codebase
- ONNX Model Zoo for test models
- The Uncharted Territory Challenge for the project framework
