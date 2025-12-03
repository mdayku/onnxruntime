# ONNX Autodoc JSON Schemas

This document describes the JSON output formats produced by ONNX Autodoc.

## Inspection Report Schema

The main output of `model_inspect.py --out-json` follows this schema:

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "$id": "inspection-report.schema.json",
  "title": "ONNX Autodoc Inspection Report"
}
```

### Top-Level Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `metadata` | object | Yes | Model metadata from ONNX proto |
| `generated_at` | string | Yes | ISO 8601 timestamp |
| `autodoc_version` | string | Yes | Version of ONNX Autodoc |
| `graph_summary` | object | No | Graph structure statistics |
| `param_counts` | object | No | Parameter counts by operator |
| `flop_counts` | object | No | FLOPs by operator |
| `memory_estimates` | object | No | Memory usage estimates |
| `hardware_estimates` | object | No | Hardware performance estimates |
| `risks` | array | No | Risk signals and warnings |
| `llm_summary` | object | No | LLM-generated summary |

### metadata

```json
{
  "path": "model.onnx",
  "ir_version": 8,
  "producer_name": "pytorch",
  "producer_version": "2.0.0",
  "opsets": {
    "": 17,
    "com.microsoft": 1
  }
}
```

### graph_summary

```json
{
  "num_nodes": 150,
  "num_inputs": 1,
  "num_outputs": 1,
  "num_initializers": 100,
  "input_shapes": {
    "input": [1, 3, 224, 224]
  },
  "output_shapes": {
    "output": [1, 1000]
  },
  "op_type_counts": {
    "Conv": 20,
    "Relu": 20,
    "Add": 16
  }
}
```

### param_counts

```json
{
  "total": 11689512,
  "by_op_type": {
    "Conv": 11154432,
    "Gemm": 513000
  },
  "trainable": 11689512,
  "non_trainable": 0
}
```

### flop_counts

```json
{
  "total": 1814073344,
  "by_op_type": {
    "Conv": 1800000000,
    "Gemm": 14073344
  }
}
```

### memory_estimates

```json
{
  "model_size_bytes": 46758048,
  "model_size_mb": 44.6,
  "peak_activation_bytes": 12582912,
  "peak_activation_mb": 12.0,
  "total_activation_bytes": 50331648
}
```

### hardware_estimates

When `--hardware` is specified:

```json
{
  "hardware_profile": "rtx4090",
  "precision": "fp16",
  "batch_size": 1,
  "vram_required_gb": 0.5,
  "fits_in_vram": true,
  "latency_ms": 1.2,
  "throughput_fps": 833,
  "compute_utilization": 0.45,
  "memory_bandwidth_utilization": 0.12,
  "bottleneck": "compute"
}
```

### risks

```json
{
  "risks": [
    {
      "severity": "warning",
      "category": "memory",
      "message": "Peak activation memory (12.0 MB) may cause issues on edge devices",
      "details": "Consider reducing batch size or input resolution"
    }
  ]
}
```

### llm_summary

When `--llm-summary` is specified:

```json
{
  "llm_summary": {
    "model": "gpt-4o-mini",
    "short_summary": "ResNet-18 is a CNN for image classification with 11.7M parameters.",
    "detailed_summary": "This ONNX model implements ResNet-18...",
    "generated_at": "2025-12-03T15:00:00Z"
  }
}
```

---

## Comparison Report Schema

The output of `model_inspect_compare.py --out-json` follows this schema:

### Top-Level Fields

| Field | Type | Description |
|-------|------|-------------|
| `baseline` | string | Precision of baseline model |
| `variants` | array | Array of variant comparison data |
| `architecture_compatible` | boolean | Whether models are architecturally comparable |
| `compatibility_warnings` | array | Warnings about architecture differences |

### Variant Object

```json
{
  "precision": "fp16",
  "model_path": "model_fp16.onnx",
  "file_size_bytes": 23379024,
  "file_size_delta_pct": -50.0,
  "params": 11689512,
  "params_delta_pct": 0.0,
  "flops": 1814073344,
  "flops_delta_pct": 0.0,
  "memory_mb": 22.3,
  "memory_delta_pct": -50.0,
  "eval_metrics": {
    "accuracy": 0.754,
    "accuracy_delta": -0.002,
    "latency_ms": 1.2,
    "latency_speedup": 1.8
  }
}
```

---

## Validation

Use the built-in schema validation:

```python
from util.autodoc import validate_report, ValidationError

try:
    validate_report(report_dict)
    print("Report is valid!")
except ValidationError as e:
    print(f"Validation failed: {e}")
    for error in e.errors:
        print(f"  - {error}")
```

---

## CLI Examples

```bash
# Generate JSON report
python -m util.model_inspect model.onnx --out-json report.json

# Generate JSON with hardware estimates
python -m util.model_inspect model.onnx \
  --hardware rtx4090 \
  --precision fp16 \
  --out-json report.json

# Compare models and output JSON
python -m util.model_inspect_compare \
  --models fp32.onnx fp16.onnx int8.onnx \
  --precisions fp32 fp16 int8 \
  --baseline-precision fp32 \
  --out-json comparison.json
```

