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
| Epic 1: Environment Setup | In Progress | 3 | 2/12 |
| Epic 2: Core Analysis Engine | Not Started | 4 | 0/17 |
| Epic 3: Pattern Analysis | Not Started | 2 | 0/9 |
| Epic 4: CLI and Output | Not Started | 3 | 0/13 |
| Epic 5: Visualization | Not Started | 3 | 0/11 |
| Epic 6: Hardware/Compare | Not Started | 4 | 0/15 |
| Epic 7: LLM Integration | Not Started | 1 | 0/5 |
| Epic 8: Testing | Not Started | 3 | 0/12 |
| Epic 9: Demo/Deliverables | Not Started | 3 | 0/13 |

---

## Epic 1: Environment Setup and Repo Exploration

### Story 1.1: Fork and Build ONNX Runtime
- [x] **Task 1.1.1**: Fork microsoft/onnxruntime repository
- [x] **Task 1.1.2**: Set up local development environment (Python 3.10+, CMake, etc.)
- [ ] **Task 1.1.3**: Successfully build ONNX Runtime from source *(in progress - CUDA kernels)*
- [ ] **Task 1.1.4**: Run existing test suite to verify build

### Story 1.2: Codebase Familiarization
- [ ] **Task 1.2.1**: Map key directories and their purposes
- [ ] **Task 1.2.2**: Study existing Python tools (`check_onnx_model_mobile_usability`, etc.)
- [ ] **Task 1.2.3**: Understand graph representation classes (C++ and Python)
- [ ] **Task 1.2.4**: Document extension points and integration patterns

### Story 1.3: Project Scaffolding
- [ ] **Task 1.3.1**: Create `onnxruntime/tools/autodoc/` directory structure
- [ ] **Task 1.3.2**: Set up `__init__.py` and module structure
- [ ] **Task 1.3.3**: Create stub files for all planned modules
- [ ] **Task 1.3.4**: Add project to ONNX Runtime build system

---

## Epic 2: Core Analysis Engine

### Story 2.1: ONNX Graph Loader
- [ ] **Task 2.1.1**: Implement model loading with `onnx.load()`
- [ ] **Task 2.1.2**: Extract graph metadata (opset, producer, IR version)
- [ ] **Task 2.1.3**: Parse nodes, inputs, outputs, initializers
- [ ] **Task 2.1.4**: Implement shape inference wrapper

### Story 2.2: Parameter Counting
- [ ] **Task 2.2.1**: Count parameters from initializers
- [ ] **Task 2.2.2**: Associate parameters with nodes
- [ ] **Task 2.2.3**: Aggregate by node type and block
- [ ] **Task 2.2.4**: Handle edge cases (shared weights, quantized params)

### Story 2.3: FLOP Estimation
- [ ] **Task 2.3.1**: Implement Conv2D FLOP calculation
- [ ] **Task 2.3.2**: Implement MatMul/Gemm FLOP calculation
- [ ] **Task 2.3.3**: Implement attention pattern FLOP calculation
- [ ] **Task 2.3.4**: Add fallback estimation for unknown ops
- [ ] **Task 2.3.5**: Identify and flag FLOP hotspots

### Story 2.4: Memory Estimation
- [ ] **Task 2.4.1**: Calculate activation tensor sizes
- [ ] **Task 2.4.2**: Implement peak memory estimation via topological walk
- [ ] **Task 2.4.3**: Estimate KV cache size for attention models
- [ ] **Task 2.4.4**: Add memory breakdown by component

---

## Epic 3: Pattern Analysis and Risk Detection

### Story 3.1: Block Detection
- [ ] **Task 3.1.1**: Implement Conv-BN-Relu pattern detection
- [ ] **Task 3.1.2**: Implement residual block detection
- [ ] **Task 3.1.3**: Implement transformer block detection
- [ ] **Task 3.1.4**: Group nodes into logical blocks in output

### Story 3.2: Risk Heuristics
- [ ] **Task 3.2.1**: Detect deep networks without skip connections
- [ ] **Task 3.2.2**: Flag oversized dense layers
- [ ] **Task 3.2.3**: Identify problematic dynamic shapes
- [ ] **Task 3.2.4**: Detect non-standard residual patterns
- [ ] **Task 3.2.5**: Add configurable severity thresholds

---

## Epic 4: CLI and Output Formats

### Story 4.1: CLI Implementation
- [ ] **Task 4.1.1**: Implement argument parser with all flags
- [ ] **Task 4.1.2**: Wire CLI to analysis engine
- [ ] **Task 4.1.3**: Add progress indicators for large models
- [ ] **Task 4.1.4**: Implement error handling and exit codes

### Story 4.2: JSON Output
- [ ] **Task 4.2.1**: Implement full JSON schema serialization
- [ ] **Task 4.2.2**: Add schema validation
- [ ] **Task 4.2.3**: Support stdout and file output modes
- [ ] **Task 4.2.4**: Add pretty-print option

### Story 4.3: Markdown Output
- [ ] **Task 4.3.1**: Create Markdown template structure
- [ ] **Task 4.3.2**: Implement model card header section
- [ ] **Task 4.3.3**: Generate metrics tables
- [ ] **Task 4.3.4**: Generate risk signals section
- [ ] **Task 4.3.5**: Add executive summary section

---

## Epic 5: Visualization Module

### Story 5.1: Chart Infrastructure
- [ ] **Task 5.1.1**: Set up matplotlib with Agg backend
- [ ] **Task 5.1.2**: Create consistent chart styling/theme
- [ ] **Task 5.1.3**: Implement asset directory management
- [ ] **Task 5.1.4**: Add graceful fallback when matplotlib unavailable

### Story 5.2: Individual Charts
- [ ] **Task 5.2.1**: Implement operator type histogram
- [ ] **Task 5.2.2**: Implement layer depth profile
- [ ] **Task 5.2.3**: Implement parameter distribution chart
- [ ] **Task 5.2.4**: Implement shape evolution chart

### Story 5.3: Report Integration
- [ ] **Task 5.3.1**: Embed charts in Markdown output
- [ ] **Task 5.3.2**: Add chart captions and descriptions
- [ ] **Task 5.3.3**: Support HTML output with embedded images

---

## Epic 6: Hardware Profiles and Compare Mode

### Story 6.1: Hardware Profile System
- [ ] **Task 6.1.1**: Define hardware profile JSON schema
- [ ] **Task 6.1.2**: Create initial profile library (RTX 4090, A10, T4)
- [ ] **Task 6.1.3**: Implement profile loading and validation
- [ ] **Task 6.1.4**: Add CLI flag for hardware profile

### Story 6.2: Hardware Estimates
- [ ] **Task 6.2.1**: Implement VRAM requirement estimation
- [ ] **Task 6.2.2**: Implement theoretical latency bounds
- [ ] **Task 6.2.3**: Estimate compute utilization
- [ ] **Task 6.2.4**: Identify bottleneck (compute vs memory)

### Story 6.3: Compare Mode CLI
- [ ] **Task 6.3.1**: Implement multi-model argument parsing
- [ ] **Task 6.3.2**: Load and validate eval metrics JSONs
- [ ] **Task 6.3.3**: Verify architecture compatibility
- [ ] **Task 6.3.4**: Compute deltas vs baseline

### Story 6.4: Quantization Impact Report
- [ ] **Task 6.4.1**: Generate comparison JSON schema
- [ ] **Task 6.4.2**: Create comparison Markdown table
- [ ] **Task 6.4.3**: Add trade-off analysis section

---

## Epic 7: LLM Integration (Optional)

### Story 7.1: LLM Summarizer
- [ ] **Task 7.1.1**: Implement API client abstraction
- [ ] **Task 7.1.2**: Create prompt templates for model summarization
- [ ] **Task 7.1.3**: Generate short summary (1-2 sentences)
- [ ] **Task 7.1.4**: Generate detailed summary (paragraph)
- [ ] **Task 7.1.5**: Handle API failures gracefully

---

## Epic 8: Testing and Quality

### Story 8.1: Unit Tests
- [ ] **Task 8.1.1**: Test param counting with known models
- [ ] **Task 8.1.2**: Test FLOP estimation accuracy
- [ ] **Task 8.1.3**: Test risk heuristic detection
- [ ] **Task 8.1.4**: Test JSON schema compliance

### Story 8.2: Integration Tests
- [ ] **Task 8.2.1**: Test CLI end-to-end with ResNet
- [ ] **Task 8.2.2**: Test CLI end-to-end with BERT
- [ ] **Task 8.2.3**: Test compare mode with quantized variants
- [ ] **Task 8.2.4**: Test visualization generation

### Story 8.3: Documentation
- [ ] **Task 8.3.1**: Write tool README
- [ ] **Task 8.3.2**: Add inline code documentation
- [ ] **Task 8.3.3**: Create example scripts
- [ ] **Task 8.3.4**: Document JSON schemas

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
