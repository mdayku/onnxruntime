# ONNX Autodoc - Project Backlog

Jira-style Epic/Story/Task tracking for the ONNX Autodoc project.

**Related Documents:**
- [PRD.md](PRD.md) - Product requirements and specifications
- [BRAINLIFT.md](BRAINLIFT.md) - Daily learning logs
- [Architecture.md](Architecture.md) - System design details

---

## Progress Summary

| Epic | Status | Stories | Tasks Complete |
|------|--------|---------|----------------|
| Epic 1: Environment Setup | In Progress | 3 | 7/12 |
| Epic 2: Core Analysis Engine | **Complete** | 4 | 17/17 |
| Epic 3: Pattern Analysis | **Complete** | 2 | 9/9 |
| Epic 4: CLI and Output | **Complete** | 4 | 18/18 |
| Epic 4B: PyTorch Integration | In Progress | 2 | 12/14 |
| Epic 5: Visualization | **Complete** | 3 | 12/12 |
| Epic 6: Hardware/Compare | In Progress | 7 | 9/28 |
| Epic 7: LLM Integration | In Progress | 2 | 5/9 |
| Epic 8: Testing & CI/CD | In Progress | 4 | 12/18 |
| Epic 9: Demo/Deliverables | Not Started | 3 | 0/13 |

---

## Epic 1: Environment Setup and Repo Exploration

### Story 1.1: Fork and Build ONNX Runtime
- [x] **Task 1.1.1**: Fork microsoft/onnxruntime repository
- [x] **Task 1.1.2**: Set up local development environment (Python 3.10+, CMake, etc.)
- [x] **Task 1.1.3**: Successfully build ONNX Runtime from source (C++ complete, CUDA provider built)
- [ ] **Task 1.1.4**: Build Python wheel (requires cuDNN install, deferred - Autodoc works without it)
- [ ] **Task 1.1.5**: Run existing test suite to verify build

### Story 1.2: Codebase Familiarization
- [x] **Task 1.2.1**: Map key directories and their purposes
- [x] **Task 1.2.2**: Study existing Python tools (`check_onnx_model_mobile_usability`, etc.)
- [ ] **Task 1.2.3**: Understand graph representation classes (C++ and Python)
- [ ] **Task 1.2.4**: Document extension points and integration patterns

### Story 1.3: Project Scaffolding
- [x] **Task 1.3.1**: Create `tools/python/util/autodoc/` directory structure
- [x] **Task 1.3.2**: Set up `__init__.py` and module structure
- [x] **Task 1.3.3**: Create stub files for all planned modules (analyzer, patterns, risks, report)
- [ ] **Task 1.3.4**: Add project to ONNX Runtime build system

---

## Epic 2: Core Analysis Engine

### Story 2.1: ONNX Graph Loader
- [x] **Task 2.1.1**: Implement model loading with `onnx.load()`
- [x] **Task 2.1.2**: Extract graph metadata (opset, producer, IR version)
- [x] **Task 2.1.3**: Parse nodes, inputs, outputs, initializers
- [x] **Task 2.1.4**: Implement shape inference wrapper (with ORT fallback)

### Story 2.2: Parameter Counting
- [x] **Task 2.2.1**: Count parameters from initializers
- [x] **Task 2.2.2**: Associate parameters with nodes
- [x] **Task 2.2.3**: Aggregate by node type and block
- [ ] **Task 2.2.4**: Handle edge cases (shared weights, quantized params)

### Story 2.3: FLOP Estimation
- [x] **Task 2.3.1**: Implement Conv2D FLOP calculation
- [x] **Task 2.3.2**: Implement MatMul/Gemm FLOP calculation
- [x] **Task 2.3.3**: Implement attention pattern FLOP calculation
- [x] **Task 2.3.4**: Add fallback estimation for unknown ops (elementwise)
- [x] **Task 2.3.5**: Identify and flag FLOP hotspots

### Story 2.4: Memory Estimation
- [x] **Task 2.4.1**: Calculate activation tensor sizes
- [x] **Task 2.4.2**: Implement peak memory estimation (heuristic, top-3 activations)
- [x] **Task 2.4.3**: Estimate KV cache size for attention models (formula: 2*layers*hidden*bytes per token)
- [x] **Task 2.4.4**: Add memory breakdown by component (MemoryBreakdown dataclass with weights/activations by op type)

---

## Epic 3: Pattern Analysis and Risk Detection

### Story 3.1: Block Detection
- [x] **Task 3.1.1**: Implement Conv-BN-Relu pattern detection
- [x] **Task 3.1.2**: Implement residual block detection (Add node heuristic)
- [x] **Task 3.1.3**: Implement transformer block detection (Softmax-based attention)
- [x] **Task 3.1.4**: Group nodes into logical blocks in output

### Story 3.2: Risk Heuristics
- [x] **Task 3.2.1**: Detect deep networks without skip connections
- [x] **Task 3.2.2**: Flag oversized dense layers
- [x] **Task 3.2.3**: Identify problematic dynamic shapes
- [x] **Task 3.2.4**: Detect non-standard residual patterns (Concat, Gated, Sub variants)
- [x] **Task 3.2.5**: Add configurable severity thresholds (RiskThresholds dataclass)

---

## Epic 4: CLI and Output Formats

### Story 4.1: CLI Implementation
- [x] **Task 4.1.1**: Implement argument parser with all flags
- [x] **Task 4.1.2**: Wire CLI to analysis engine
- [x] **Task 4.1.3**: Add progress indicators for large models (--progress flag with step-by-step display)
- [x] **Task 4.1.4**: Implement error handling and exit codes

### Story 4.2: JSON Output
- [x] **Task 4.2.1**: Implement full JSON schema serialization
- [x] **Task 4.2.2**: Add schema validation (jsonschema, Draft 7, validate()/validate_strict() methods)
- [x] **Task 4.2.3**: Support stdout and file output modes
- [x] **Task 4.2.4**: Add pretty-print option (indent=2)

### Story 4.3: Markdown Output
- [x] **Task 4.3.1**: Create Markdown template structure
- [x] **Task 4.3.2**: Implement model card header section
- [x] **Task 4.3.3**: Generate metrics tables
- [x] **Task 4.3.4**: Generate risk signals section
- [x] **Task 4.3.5**: Add hardware estimates section
- [x] **Task 4.3.6**: Add executive summary section (matches HTML, shows LLM summary)

### Story 4.4: HTML Report Parity
- [x] **Task 4.4.1**: Add Operator Distribution table to HTML (op counts)
- [x] **Task 4.4.2**: Add KV Cache section to HTML (per-token, full-context, layers, hidden_dim)
- [x] **Task 4.4.3**: Add Memory Breakdown by Op Type table to HTML
- [x] **Task 4.4.4**: Add Architecture section to HTML (detected type + detected blocks)

---

## Epic 4B: PyTorch Integration

### Story 4B.1: PyTorch to ONNX Conversion
- [x] **Task 4B.1.1**: Add `--from-pytorch` CLI flag with model path argument
- [x] **Task 4B.1.2**: Implement PyTorch model loading (TorchScript + torch.load)
- [x] **Task 4B.1.3**: Require `--input-shape` for conversion
- [x] **Task 4B.1.4**: Implement torch.onnx.export wrapper with sensible defaults
- [x] **Task 4B.1.5**: Generate temp ONNX file for analysis (or --keep-onnx to save)
- [x] **Task 4B.1.6**: Support TorchScript models (.pt from torch.jit.save)
- [x] **Task 4B.1.7**: Add tests for PyTorch conversion flow (9 tests)
- [x] **Task 4B.1.8**: Handle conversion errors gracefully (missing torch, export failures)

### Story 4B.2: Dataset/Class Metadata Extraction
- [x] **Task 4B.2.1**: Detect Ultralytics models and extract class names
- [ ] **Task 4B.2.2**: Parse output shapes to infer number of classes (future)
- [x] **Task 4B.2.3**: Add DatasetInfo dataclass (task, num_classes, class_names)
- [x] **Task 4B.2.4**: Add Dataset Info section to Markdown/HTML reports
- [x] **Task 4B.2.5**: Add --pytorch-weights flag to provide original .pt for metadata

---

## Epic 5: Visualization Module

### Story 5.1: Chart Infrastructure
- [x] **Task 5.1.1**: Set up matplotlib with Agg backend
- [x] **Task 5.1.2**: Create consistent chart styling/theme (ChartTheme dataclass, dark theme)
- [x] **Task 5.1.3**: Implement asset directory management
- [x] **Task 5.1.4**: Add graceful fallback when matplotlib unavailable

### Story 5.2: Individual Charts
- [x] **Task 5.2.1**: Implement operator type histogram
- [x] **Task 5.2.2**: Implement layer depth profile (cumulative params/FLOPs)
- [x] **Task 5.2.3**: Implement parameter distribution chart (pie chart)
- [x] **Task 5.2.4**: Implement FLOPs distribution chart
- [x] **Task 5.2.5**: Implement complexity summary dashboard (3-panel)

### Story 5.3: Report Integration
- [x] **Task 5.3.1**: Embed charts in Markdown output
- [x] **Task 5.3.2**: Add chart captions and descriptions
- [x] **Task 5.3.3**: Support HTML output with embedded images (base64, single shareable file)
- [ ] **Task 5.3.4**: Support PDF output (optional weasyprint dependency, fallback to HTML)

---

## Epic 6: Hardware Profiles and Compare Mode

### Story 6.1: Hardware Profile System
- [x] **Task 6.1.1**: Define hardware profile dataclass (HardwareProfile)
- [x] **Task 6.1.2**: Create comprehensive profile library (30+ profiles):
  - Data Center: H100, A100, A10, L4, L40, L40S, T4, V100, P100, P40
  - Jetson Orin: AGX Orin 64/32GB, Orin NX 16/8GB, Orin Nano 8/4GB
  - Jetson Xavier: AGX Xavier 32/16GB, Xavier NX 16/8GB
  - Jetson Legacy: TX2, TX2 NX, Nano 4GB, Nano 2GB
  - Consumer: RTX 4090/4080/3090/3080
- [x] **Task 6.1.3**: Implement profile loading and auto-detection via nvidia-smi (incl. Jetson)
- [x] **Task 6.1.4**: Add CLI flags (--hardware, --list-hardware, --precision, --batch-size)

### Story 6.2: Hardware Estimates
- [x] **Task 6.2.1**: Implement VRAM requirement estimation
- [x] **Task 6.2.2**: Implement theoretical latency bounds
- [x] **Task 6.2.3**: Estimate compute utilization (roofline-based: compute_time/memory_time)
- [x] **Task 6.2.4**: Identify bottleneck (compute vs memory vs vram)
- [x] **Task 6.2.5**: Add GPU Saturation metric (model_flops / gpu_peak_capacity) - shows % of GPU compute used per inference

### Story 6.3: Compare Mode CLI
- [ ] **Task 6.3.1**: Implement multi-model argument parsing
- [ ] **Task 6.3.2**: Load and validate eval metrics JSONs
- [ ] **Task 6.3.3**: Verify architecture compatibility
- [ ] **Task 6.3.4**: Compute deltas vs baseline

### Story 6.4: Quantization Impact Report
- [ ] **Task 6.4.1**: Generate comparison JSON schema
- [ ] **Task 6.4.2**: Create comparison Markdown table
- [ ] **Task 6.4.3**: Add trade-off analysis section

### Story 6.5: Expanded VRAM Variants
- [ ] **Task 6.5.1**: Add all A100 variants (40GB PCIe, 80GB PCIe, 80GB SXM)
- [ ] **Task 6.5.2**: Add H100 variants (80GB PCIe, 80GB SXM, 94GB NVL)
- [ ] **Task 6.5.3**: Add V100 variants (16GB, 32GB, PCIe vs SXM)
- [ ] **Task 6.5.4**: Add RTX 3080 variants (10GB, 12GB)
- [ ] **Task 6.5.5**: Add RTX 3090 Ti (24GB)
- [ ] **Task 6.5.6**: Add full RTX 40-series lineup (4090, 4080 Super, 4080, 4070 Ti Super, 4070 Ti, 4070, 4060 Ti 16GB/8GB, 4060)
- [ ] **Task 6.5.7**: Add full RTX 30-series lineup (3090 Ti, 3080 Ti, 3070 Ti, 3070, 3060 Ti, 3060, 3050)
- [ ] **Task 6.5.8**: Add laptop GPU variants (Mobile suffix, lower TDP/clocks)

### Story 6.6: Multi-GPU / Cluster Support
- [ ] **Task 6.6.1**: Add multi-GPU profile multiplier (2x, 4x, 8x configurations)
- [ ] **Task 6.6.2**: Model NVLink bandwidth for multi-GPU setups
- [ ] **Task 6.6.3**: Estimate tensor parallelism overhead
- [ ] **Task 6.6.4**: Add DGX system profiles (DGX A100, DGX H100)
- [ ] **Task 6.6.5**: Add CLI flag `--gpu-count N` for multi-GPU estimates
- [ ] **Task 6.6.6**: Estimate pipeline parallelism for models that don't fit on single GPU

### Story 6.7: Cloud Instance Profiles
- [ ] **Task 6.7.1**: Add AWS instance profiles (p4d, p5, g5, inf2)
- [ ] **Task 6.7.2**: Add Azure instance profiles (NC, ND series)
- [ ] **Task 6.7.3**: Add GCP instance profiles (a2, a3, g2)
- [ ] **Task 6.7.4**: Include cost estimates per hour for cloud instances

---

## Epic 7: LLM Integration (Optional)

### Story 7.1: LLM Summarizer - **COMPLETE**
- [x] **Task 7.1.1**: Implement API client abstraction (OpenAI client in llm_summarizer.py)
- [x] **Task 7.1.2**: Create prompt templates for model summarization
- [x] **Task 7.1.3**: Generate short summary (1-2 sentences)
- [x] **Task 7.1.4**: Generate detailed summary (paragraph)
- [x] **Task 7.1.5**: Handle API failures gracefully

### Story 7.2: Secure Config File
- [ ] **Task 7.2.1**: Read API key from `~/.config/onnx-autodoc/config.json` or `~/.onnx-autodoc.json`
- [ ] **Task 7.2.2**: Add `--api-key` CLI flag for explicit key passing
- [ ] **Task 7.2.3**: Priority order: CLI flag > env var > config file
- [ ] **Task 7.2.4**: Add config file location to --help and README

---

## Epic 8: Testing, CI/CD, and Quality

### Story 8.1: Unit Tests
- [x] **Task 8.1.1**: Test param counting with known models
- [x] **Task 8.1.2**: Test FLOP estimation accuracy
- [x] **Task 8.1.3**: Test risk heuristic detection
- [x] **Task 8.1.4**: Test JSON schema compliance

### Story 8.2: Integration Tests
- [ ] **Task 8.2.1**: Test CLI end-to-end with ResNet
- [ ] **Task 8.2.2**: Test CLI end-to-end with BERT
- [ ] **Task 8.2.3**: Test compare mode with quantized variants
- [x] **Task 8.2.4**: Test visualization generation (17 tests, all passing)

### Story 8.3: Documentation
- [x] **Task 8.3.1**: Write tool README
- [ ] **Task 8.3.2**: Add inline code documentation
- [ ] **Task 8.3.3**: Create example scripts
- [ ] **Task 8.3.4**: Document JSON schemas

### Story 8.4: CI/CD Pipeline (GitHub Actions)
- [x] **Task 8.4.1**: Create `.github/workflows/autodoc-ci.yml` workflow file
- [x] **Task 8.4.2**: Add Python linting step (ruff/flake8)
- [x] **Task 8.4.3**: Add type checking step (mypy)
- [x] **Task 8.4.4**: Add unit test step (pytest) with coverage reporting
- [x] **Task 8.4.5**: Add integration test step with sample ONNX models
- [x] **Task 8.4.6**: Cache pip dependencies for faster CI runs

---

## Epic 9: Demo and Deliverables

### Story 9.1: Demo Preparation
- [ ] **Task 9.1.1**: Select demo models (ResNet, BERT, YOLO)
- [ ] **Task 9.1.2**: Generate sample outputs for each model
- [ ] **Task 9.1.3**: Create comparison demo (fp32 vs fp16 vs int8)
- [ ] **Task 9.1.4**: Prepare screenshots and artifacts

### Story 9.2: Video Recording
- [ ] **Task 9.2.1**: Script 5-minute demo video
- [ ] **Task 9.2.2**: Record project introduction
- [ ] **Task 9.2.3**: Record feature walkthrough
- [ ] **Task 9.2.4**: Record technical architecture explanation
- [ ] **Task 9.2.5**: Record reflection on learning

### Story 9.3: Brainlift Documentation
- [ ] **Task 9.3.1**: Complete daily logs
- [ ] **Task 9.3.2**: Document AI prompts used
- [ ] **Task 9.3.3**: Capture learning breakthroughs
- [ ] **Task 9.3.4**: Record technical decisions and rationale
