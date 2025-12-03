# Model Analysis Platform - Project Backlog

*Formerly "ONNX Autodoc" - now a universal model analysis and conversion platform.*

**Related Documents:**
- [PRD.md](PRD.md) - Product requirements and specifications
- [BRAINLIFT.md](BRAINLIFT.md) - Daily learning logs
- [Architecture.md](Architecture.md) - System design details

---

## Progress Summary

| Epic | Status | Stories | Tasks Complete | Priority |
|------|--------|---------|----------------|----------|
| Epic 1: Environment Setup | **Complete** | 3 | 11/11 | Done |
| Epic 2: Core Analysis Engine | **Complete** | 4 | 17/17 | Done |
| Epic 3: Pattern Analysis | **Complete** | 2 | 9/9 | Done |
| Epic 4: CLI and Output | **Complete** | 4 | 18/18 | Done |
| Epic 4B: PyTorch Integration | **Complete** | 2 | 14/14 | Done |
| Epic 4C: TensorFlow/Keras/JAX | **Complete** | 3 | 15/15 | Done |
| Epic 5: Visualization | In Progress | 8 | 13/47 | P3 |
| Epic 6: Hardware/Compare | In Progress | 9 | 27/47 | P3 |
| Epic 7: LLM Integration | In Progress | 2 | 5/9 | P3 |
| Epic 8: Testing & CI/CD | In Progress | 4 | 12/18 | P3 |
| Epic 10: SaaS Web App | Not Started | 5 | 0/27 | P4 |
| Epic 10B: Standalone Package | Not Started | 3 | 0/17 | P0 |
| Epic 11: Streamlit Web UI | Not Started | 3 | 0/14 | P0 |
| Epic 12: Inference Platform | Not Started | 5 | 0/24 | P1 |
| Epic 13-17: MLOps Platform | Future | 5 | 0/? | P5 |
| Epic 18: Universal IR | Not Started | 3 | 0/12 | P1 |
| Epic 19: SafeTensors | Not Started | 2 | 0/8 | P2 |
| Epic 20: CoreML | Not Started | 2 | 0/10 | P2 |
| Epic 21: TFLite | Not Started | 2 | 0/10 | P2 |
| Epic 22: TensorRT | Not Started | 2 | 0/8 | P3 |
| Epic 23: OpenVINO | Not Started | 2 | 0/8 | P3 |
| Epic 24: GGUF | Not Started | 2 | 0/6 | P3 |
| Epic 25: Privacy/Trust | Not Started | 3 | 0/9 | P1 |
| **LLM-SCALE ANALYSIS** ||||
| Epic 26: Advanced Quantization | Not Started | 3 | 0/16 | P3 |
| Epic 27: Attention Variants | Not Started | 4 | 0/19 | P3 |
| Epic 28: Memory Patterns | Not Started | 4 | 0/18 | P3 |
| Epic 29: Sparse/Efficient | Not Started | 4 | 0/16 | P3 |
| Epic 30: LLM Deployment | Not Started | 4 | 0/19 | P3 |

---

## Epic 1: Environment Setup (COMPLETE - 11/11)

- [x] Fork and build ONNX Runtime
- [x] Build Python wheel (`onnxruntime_gpu-1.24.0`)
- [x] Codebase familiarization
- [x] Project scaffolding

*Note: Task "Add to ORT build system" removed - this is our IP, not donating to Microsoft.*

---

## Epic 2: Core Analysis Engine (COMPLETE - 17/17)

- [x] ONNX Graph Loader
- [x] Parameter Counting (with shared weights, quantized params)
- [x] FLOP Estimation (Conv, MatMul, Attention)
- [x] Memory Estimation (activations, KV cache)

---

## Epic 3: Pattern Analysis (COMPLETE - 9/9)

- [x] Block Detection (Conv-BN-ReLU, Residual, Transformer)
- [x] Risk Heuristics (deep networks, dynamic shapes, oversized layers)

---

## Epic 4: CLI and Output (COMPLETE - 18/18)

- [x] CLI Implementation (argparse, progress, error handling)
- [x] JSON Output (schema validation)
- [x] Markdown Output (model cards)
- [x] HTML Report (full parity)

---

## Epic 4B: PyTorch Integration (COMPLETE - 14/14)

- [x] PyTorch to ONNX Conversion
- [x] Dataset/Class Metadata Extraction (Ultralytics, output shape inference)

---

## Epic 4C: TensorFlow and Keras Conversion (COMPLETE - 15/15)

*Big user base, enterprise adoption.*

### Story 4C.1: TensorFlow to ONNX Conversion - **COMPLETE**
- [x] **Task 4C.1.1**: Add `--from-tensorflow` CLI flag with SavedModel path argument
- [x] **Task 4C.1.2**: Implement TensorFlow SavedModel loading
- [x] **Task 4C.1.3**: Integrate tf2onnx conversion with sensible defaults
- [x] **Task 4C.1.4**: Support frozen graph (.pb) files (--from-frozen-graph, --tf-inputs, --tf-outputs)
- [x] **Task 4C.1.5**: Handle conversion errors gracefully (missing tf, export failures)
- [x] **Task 4C.1.6**: Add tests for TensorFlow conversion flow (12 tests)

### Story 4C.2: Keras to ONNX Conversion - **COMPLETE**
- [x] **Task 4C.2.1**: Add `--from-keras` CLI flag with .h5/.keras path argument
- [x] **Task 4C.2.2**: Implement Keras model loading (Sequential, Functional, Subclassed)
- [x] **Task 4C.2.3**: Convert via tf2onnx CLI for robustness
- [x] **Task 4C.2.4**: Support both .h5 and .keras formats
- [x] **Task 4C.2.5**: Add tests for Keras conversion flow

### Story 4C.3: JAX/Flax to ONNX Conversion - **COMPLETE**
- [x] **Task 4C.3.1**: Add `--from-jax` CLI flag
- [x] **Task 4C.3.2**: Implement JAX -> TF SavedModel -> ONNX pipeline via jax2tf
- [x] **Task 4C.3.3**: Support .msgpack, .pkl, .npy params formats
- [x] **Task 4C.3.4**: Support Flax modules via --jax-apply-fn module:function pattern

---

## Epic 5: Visualization Module (P3 - 13/47 tasks)

*Expanded for LLM-scale models (70B+ params, 80+ layers, 20k+ ops)*

### Story 5.1: Chart Infrastructure - **COMPLETE**
- [x] **Task 5.1.1**: Set up matplotlib with Agg backend
- [x] **Task 5.1.2**: Create consistent chart styling/theme (ChartTheme dataclass, dark theme)
- [x] **Task 5.1.3**: Implement asset directory management
- [x] **Task 5.1.4**: Add graceful fallback when matplotlib unavailable

### Story 5.2: Individual Charts - **COMPLETE**
- [x] **Task 5.2.1**: Implement operator type histogram
- [x] **Task 5.2.2**: Implement layer depth profile (cumulative params/FLOPs)
- [x] **Task 5.2.3**: Implement parameter distribution chart (pie chart)
- [x] **Task 5.2.4**: Implement FLOPs distribution chart
- [x] **Task 5.2.5**: Implement complexity summary dashboard (3-panel)

### Story 5.3: Report Integration - **COMPLETE**
- [x] **Task 5.3.1**: Embed charts in Markdown output
- [x] **Task 5.3.2**: Add chart captions and descriptions
- [x] **Task 5.3.3**: Support HTML output with embedded images (base64, single shareable file)
- [x] **Task 5.3.4**: Support PDF output (Playwright-based, --out-pdf flag)

### Story 5.4: LLM-Scale Pattern Detection (BLOCKER for 5.7)
*Must handle 70B+ param models with 80+ transformer layers*
- [ ] **Task 5.4.1**: Detect attention patterns (Q/K/V projections, Softmax, Output proj)
- [ ] **Task 5.4.2**: Detect MLP/FFN patterns (up-proj, activation, down-proj)
- [ ] **Task 5.4.3**: Detect embedding patterns (token embed, position embed, RoPE/ALiBi)
- [ ] **Task 5.4.4**: Detect normalization placement (pre-norm vs post-norm)
- [ ] **Task 5.4.5**: Detect repetition - "N identical blocks" → collapse with ×N count
- [ ] **Task 5.4.6**: Add `TransformerBlock`, `AttentionHead`, `MLPBlock` to Block types
- [ ] **Task 5.4.7**: Handle MoE (Mixture of Experts) routing patterns
- [ ] **Task 5.4.8**: Tests with BERT, GPT-2, LLaMA (if ONNX available)

### Story 5.5: Op Type Icon System and Visual Vocabulary
*180+ ONNX ops → ~20 visual categories*
- [ ] **Task 5.5.1**: Define icon/shape for each op category (see table below)
- [ ] **Task 5.5.2**: Map all 180 ONNX ops to visual categories
- [ ] **Task 5.5.3**: Define size scaling function (FLOPs → node size)
- [ ] **Task 5.5.4**: Define color mapping (compute intensity, precision, memory)
- [ ] **Task 5.5.5**: Create SVG icon set for embedding in HTML
- [ ] **Task 5.5.6**: Add legend/key to visualization output

*Op Category Table:*
| Category | Ops | Shape |
|----------|-----|-------|
| Conv | Conv, ConvTranspose | ◼️ Square |
| Linear | MatMul, Gemm | ◆ Diamond |
| Attention | (pattern) | ◉ Target |
| Norm | BatchNorm, LayerNorm, etc. | ▬ Bar |
| Activation | Relu, GELU, Softmax, etc. | ⚡ Bolt |
| Pool | MaxPool, AvgPool, Global* | ▼ Down |
| Residual | Add (skip connection) | ⊕ Plus |
| Reshape | Reshape, Transpose, etc. | 🔄 Transform |
| Slice | Gather, Slice, Split | ✂️ Cut |
| Concat | Concat | ⊕ Join |
| Reduce | ReduceSum, ReduceMean | Σ Sigma |
| Elementwise | Add, Mul, Div, etc. | ± Math |
| Embed | Gather on weights | 📖 Lookup |
| KV Cache | Concat on seq dim | 💾 Cache |

### Story 5.6: Edge-Centric Visualization (BLOCKER for 5.7)
*Edges show tensor flow - THE key insight for bottleneck detection*
- [ ] **Task 5.6.1**: Calculate tensor size at every edge (shape × dtype bytes)
- [ ] **Task 5.6.2**: Map edge thickness to tensor size (log scale for LLMs)
- [ ] **Task 5.6.3**: Color edges by precision (fp32=blue, fp16=green, int8=yellow, bf16=purple)
- [ ] **Task 5.6.4**: Highlight memory bottleneck edges (red for largest tensors)
- [ ] **Task 5.6.5**: Show tensor shape on hover: "[batch, seq, hidden]"
- [ ] **Task 5.6.6**: Detect and highlight skip connections (dashed lines)
- [ ] **Task 5.6.7**: Calculate peak memory point in graph (where activations are largest)
- [ ] **Task 5.6.8**: For attention: show O(seq²) edges prominently (this is why FlashAttention matters)

### Story 5.7: Interactive Hierarchical Graph Visualization (BLOCKED)
*Depends on: 5.4 (patterns), 5.5 (icons), 5.6 (edges)*
- [ ] **Task 5.7.1**: Build hierarchical graph data structure (Model → Layers → Blocks → Ops)
- [ ] **Task 5.7.2**: Implement D3.js or Cytoscape.js renderer
- [ ] **Task 5.7.3**: Default view: collapsed (Input → [Block×N] → Output)
- [ ] **Task 5.7.4**: Click-to-expand: show internal ops of any block
- [ ] **Task 5.7.5**: Pan/zoom for large graphs
- [ ] **Task 5.7.6**: Search by op type, layer name, or tensor name
- [ ] **Task 5.7.7**: Export as standalone HTML (self-contained, shareable)
- [ ] **Task 5.7.8**: Integrate with existing HTML report (--out-html includes graph)
- [ ] **Task 5.7.9**: Performance: handle 20k+ nodes via virtualization/culling

### Story 5.8: Per-Layer Summary Table
- [ ] **Task 5.8.1**: Create per-layer summary table (params, FLOPs, latency estimate, memory)
- [ ] **Task 5.8.2**: Add sortable/filterable table to HTML report
- [ ] **Task 5.8.3**: Click row to highlight in graph visualization
- [ ] **Task 5.8.4**: Export table as CSV

---

## Epic 6: Hardware Profiles and Compare Mode (P3 - 27/47 done)

### Story 6.1: Hardware Profile System - **COMPLETE**
- [x] **Task 6.1.1**: Define hardware profile dataclass (HardwareProfile)
- [x] **Task 6.1.2**: Create comprehensive profile library (30+ profiles)
- [x] **Task 6.1.3**: Implement profile loading and auto-detection via nvidia-smi
- [x] **Task 6.1.4**: Add CLI flags (--hardware, --list-hardware, --precision, --batch-size)

### Story 6.2: Hardware Estimates - **COMPLETE**
- [x] **Task 6.2.1**: Implement VRAM requirement estimation
- [x] **Task 6.2.2**: Implement theoretical latency bounds
- [x] **Task 6.2.3**: Estimate compute utilization (roofline-based)
- [x] **Task 6.2.4**: Identify bottleneck (compute vs memory vs vram)
- [x] **Task 6.2.5**: Add GPU Saturation metric

### Story 6.3: Compare Mode CLI
- [ ] **Task 6.3.1**: Implement multi-model argument parsing
- [ ] **Task 6.3.2**: Load and validate eval metrics JSONs
- [ ] **Task 6.3.3**: Verify architecture compatibility
- [ ] **Task 6.3.4**: Compute deltas vs baseline

### Story 6.4: Quantization Impact Report (TRT EngineXplorer-inspired)
- [ ] **Task 6.4.1**: Generate comparison JSON schema
- [ ] **Task 6.4.2**: Create comparison Markdown table
- [ ] **Task 6.4.3**: Add trade-off analysis section
- [ ] **Task 6.4.4**: Add layer-wise precision breakdown visualization
- [ ] **Task 6.4.5**: Show accuracy vs speedup tradeoff chart
- [ ] **Task 6.4.6**: Display memory savings per layer analysis
- [ ] **Task 6.4.7**: Add engine summary panel
- [ ] **Task 6.4.8**: Show quantization calibration recommendations

### Story 6.5: Expanded VRAM Variants - **COMPLETE**
- [x] **Task 6.5.1-6.5.8**: All GPU variants added (40+ profiles)

### Story 6.6: Multi-GPU / Cluster Support - **COMPLETE**
- [x] **Task 6.6.1-6.6.6**: Multi-GPU support complete

### Story 6.7: Cloud Instance Profiles - **COMPLETE**
- [x] **Task 6.7.1-6.7.4**: AWS/Azure/GCP profiles complete

### Story 6.8: Resolution and Batch Size Impact Analysis
- [ ] **Task 6.8.1**: Add `--input-resolution` CLI flag
- [ ] **Task 6.8.2**: Implement resolution scaling impact estimator
- [ ] **Task 6.8.3**: Add batch size sweep analysis
- [ ] **Task 6.8.4**: Generate resolution/batch vs latency/memory charts
- [ ] **Task 6.8.5**: Add resolution recommendations for target hardware

### Story 6.9: Hardware Requirements Recommendations (Steam-style)
- [ ] **Task 6.9.1**: Define deployment target categories
- [ ] **Task 6.9.2**: Calculate minimum HW spec
- [ ] **Task 6.9.3**: Calculate recommended HW spec
- [ ] **Task 6.9.4**: Add `--deployment-target` CLI flag
- [ ] **Task 6.9.5**: Generate "System Requirements" section
- [ ] **Task 6.9.6**: Factor in latency/throughput requirements

---

## Epic 7: LLM Integration (P3 - 5/9 done)

### Story 7.1: LLM Summarizer - **COMPLETE**
- [x] **Task 7.1.1**: Implement API client abstraction
- [x] **Task 7.1.2**: Create prompt templates
- [x] **Task 7.1.3**: Generate short summary
- [x] **Task 7.1.4**: Generate detailed summary
- [x] **Task 7.1.5**: Handle API failures gracefully

### Story 7.2: Secure Config File
- [ ] **Task 7.2.1**: Read API key from config file
- [ ] **Task 7.2.2**: Add `--api-key` CLI flag
- [ ] **Task 7.2.3**: Priority order: CLI flag > env var > config file
- [ ] **Task 7.2.4**: Document config file location

---

## Epic 8: Testing, CI/CD, and Quality (P3 - 12/18 done)

### Story 8.1: Unit Tests - **COMPLETE**
- [x] **Task 8.1.1-8.1.4**: All unit tests complete

### Story 8.2: Integration Tests
- [ ] **Task 8.2.1**: Test CLI end-to-end with ResNet
- [ ] **Task 8.2.2**: Test CLI end-to-end with BERT
- [ ] **Task 8.2.3**: Test compare mode with quantized variants
- [x] **Task 8.2.4**: Test visualization generation (17 tests)

### Story 8.3: Documentation
- [x] **Task 8.3.1**: Write tool README
- [ ] **Task 8.3.2**: Add inline code documentation
- [ ] **Task 8.3.3**: Create example scripts
- [ ] **Task 8.3.4**: Document JSON schemas

### Story 8.4: CI/CD Pipeline - **COMPLETE**
- [x] **Task 8.4.1-8.4.6**: GitHub Actions workflow complete

---

## Epic 10: SaaS Web Application (P4)

*W&B-style hosted service. Requires Epic 25 (Privacy/Trust) first.*

### Story 10.1: Web Backend API
- [ ] **Task 10.1.1-10.1.6**: FastAPI backend (6 tasks)

### Story 10.2: Frontend MVP
- [ ] **Task 10.2.1-10.2.6**: React/Next.js frontend (6 tasks)

### Story 10.3: Authentication and Users
- [ ] **Task 10.3.1-10.3.5**: Auth integration (5 tasks)

### Story 10.4: Cloud Infrastructure
- [ ] **Task 10.4.1-10.4.6**: Deployment setup (6 tasks)

### Story 10.5: Model History and Comparison
- [ ] **Task 10.5.1-10.5.5**: Versioning features (5 tasks)

---

## Epic 10B: Standalone Package and Distribution (P0)

*Extract from ORT fork, ship as independent product. This is our IP.*

### Story 10B.0: Greenfield Extraction
- [ ] **Task 10B.0.1**: Create new GitHub repo (standalone, not ORT fork)
- [ ] **Task 10B.0.2**: Copy autodoc modules from `tools/python/util/autodoc/`
- [ ] **Task 10B.0.3**: Copy `model_inspect.py` as CLI entrypoint
- [ ] **Task 10B.0.4**: Update all imports to standalone package structure
- [ ] **Task 10B.0.5**: Verify all tests pass in new repo
- [ ] **Task 10B.0.6**: Update README for standalone usage

### Story 10B.1: Python Wheel Packaging
- [ ] **Task 10B.1.1**: Create pyproject.toml with proper metadata
- [ ] **Task 10B.1.2**: Configure build system (hatch or poetry)
- [ ] **Task 10B.1.3**: Define optional dependencies ([viz], [llm], [pdf], [full])
- [ ] **Task 10B.1.4**: Add CLI entrypoint (`model-analyzer` command)
- [ ] **Task 10B.1.5**: Test wheel installation in clean virtual environment
- [ ] **Task 10B.1.6**: Publish to TestPyPI first, then PyPI

### Story 10B.2: Pre-built Docker Image
- [ ] **Task 10B.2.1**: Create Dockerfile with all dependencies
- [ ] **Task 10B.2.2**: Optimize image size (multi-stage build)
- [ ] **Task 10B.2.3**: Add GPU support variant (CUDA base image)
- [ ] **Task 10B.2.4**: Publish to Docker Hub / GitHub Container Registry
- [ ] **Task 10B.2.5**: Create docker-compose.yml for easy local setup

---

## Epic 11: Streamlit Web UI (P0)

*Demo-able web interface for non-CLI users.*

### Story 11.1: Basic Streamlit App
- [ ] **Task 11.1.1**: Create `streamlit_app.py` with file upload widget
- [ ] **Task 11.1.2**: Wire upload to analysis engine
- [ ] **Task 11.1.3**: Display HTML report in Streamlit iframe/component
- [ ] **Task 11.1.4**: Add hardware profile dropdown selector
- [ ] **Task 11.1.5**: Add download buttons (JSON, Markdown, PDF)

### Story 11.2: Enhanced UI Features
- [ ] **Task 11.2.1**: Add interactive charts (Plotly instead of matplotlib)
- [ ] **Task 11.2.2**: Add model comparison tab (upload 2 models)
- [ ] **Task 11.2.3**: Add LLM summary toggle with API key input
- [ ] **Task 11.2.4**: Add session history (analyze multiple models)
- [ ] **Task 11.2.5**: Responsive layout for mobile

### Story 11.3: Deployment
- [ ] **Task 11.3.1**: Deploy to Hugging Face Spaces (free, GPU available)
- [ ] **Task 11.3.2**: Add Streamlit Cloud deployment option
- [ ] **Task 11.3.3**: Create deployment documentation
- [ ] **Task 11.3.4**: Set up CI/CD for auto-deploy on push

---

## Epic 12: Inference Platform (P1)

*Architecture analysis + Inference metrics = Complete picture.*
*User feedback: 2/2 want familiar metrics (accuracy, mAP, F1) not just architecture stats.*

### Story 12.1: Inference Runner Core
- [ ] **Task 12.1.1**: Create `InferenceRunner` class wrapping ORT InferenceSession
- [ ] **Task 12.1.2**: Add warmup runs and timing measurement
- [ ] **Task 12.1.3**: Support batched inference with configurable batch sizes
- [ ] **Task 12.1.4**: Add latency statistics (p50, p95, p99, mean, std)
- [ ] **Task 12.1.5**: Add throughput calculation (items/sec, batches/sec)

### Story 12.2: Data Loader Interface
- [ ] **Task 12.2.1**: Define abstract `DataLoader` interface
- [ ] **Task 12.2.2**: Implement `ImageFolderLoader` (directory of images)
- [ ] **Task 12.2.3**: Implement `NumpyArrayLoader` (precomputed tensors)
- [ ] **Task 12.2.4**: Add preprocessing hooks (resize, normalize, tokenize)
- [ ] **Task 12.2.5**: Support streaming for large datasets

### Story 12.3: Multi-Precision Comparison
- [ ] **Task 12.3.1**: Load multiple precision variants (fp32, fp16, int8)
- [ ] **Task 12.3.2**: Run same test data through all variants
- [ ] **Task 12.3.3**: Compare latency across precisions
- [ ] **Task 12.3.4**: Compare outputs (numerical diff, cosine similarity)
- [ ] **Task 12.3.5**: Generate precision comparison report

### Story 12.4: Metrics Extension Framework
- [ ] **Task 12.4.1**: Define abstract `MetricsCalculator` interface
- [ ] **Task 12.4.2**: Implement `RawOutputMetrics` (just latency, no accuracy)
- [ ] **Task 12.4.3**: Add plugin/registration system for custom metrics
- [ ] **Task 12.4.4**: Create example plugin for classification accuracy
- [ ] **Task 12.4.5**: Document how to add new task-specific metrics

### Story 12.5: Results Schema and Export
- [ ] **Task 12.5.1**: Define standardized inference results schema
- [ ] **Task 12.5.2**: Integrate inference results with existing report generator
- [ ] **Task 12.5.3**: Add inference section to HTML/PDF reports
- [ ] **Task 12.5.4**: Export raw results as CSV/JSON for custom analysis

---

## Epics 13-17: MLOps Platform Vision (P5)

*High-level placeholder. Do not implement until there's demand.*

- **Epic 13**: Cloud Provider Integration (AWS/GCP/Azure)
- **Epic 14**: GPU Orchestration (on-demand spin-up/down)
- **Epic 15**: Training Estimation (dataset → training time)
- **Epic 16**: Model Inventory Management (versions, lineage)
- **Epic 17**: Billing and Usage (metered tracking)

---

## Epic 18: Universal Internal Representation (P1)

*Foundation for format-agnostic analysis. Must be built before other format adapters.*

### Story 18.1: Universal Graph IR
- [ ] **Task 18.1.1**: Design `UniversalGraph` dataclass (nodes, edges, weights, metadata)
- [ ] **Task 18.1.2**: Design `UniversalNode` with op-type abstraction (not tied to ONNX ops)
- [ ] **Task 18.1.3**: Design `UniversalTensor` for weights/activations
- [ ] **Task 18.1.4**: Add source_format tracking and round-trip metadata
- [ ] **Task 18.1.5**: Document IR design decisions in Architecture.md

### Story 18.2: Format Adapter Interface
- [ ] **Task 18.2.1**: Define `FormatAdapter` protocol (can_read, read, can_write, write)
- [ ] **Task 18.2.2**: Implement adapter registry and auto-detection by file extension
- [ ] **Task 18.2.3**: Refactor ONNX loader to use FormatAdapter interface
- [ ] **Task 18.2.4**: Refactor PyTorch loader to use FormatAdapter interface

### Story 18.3: Conversion Matrix
- [ ] **Task 18.3.1**: Define conversion capability enum (FULL, LOSSY, PARTIAL, NONE)
- [ ] **Task 18.3.2**: Implement conversion matrix lookup
- [ ] **Task 18.3.3**: Add CLI flag `--convert-to <format>` for format conversion

---

## Epic 19: SafeTensors Format (P2)

*HuggingFace ecosystem, widely used for LLM weights. Easy win.*

### Story 19.1: SafeTensors Reader
- [ ] **Task 19.1.1**: Add safetensors dependency (optional)
- [ ] **Task 19.1.2**: Implement SafeTensorsAdapter.read() - load tensor dict
- [ ] **Task 19.1.3**: Extract metadata (tensor names, shapes, dtypes)
- [ ] **Task 19.1.4**: Integrate with analysis pipeline (param counts, memory)

### Story 19.2: SafeTensors Writer
- [ ] **Task 19.2.1**: Implement SafeTensorsAdapter.write() - export weights
- [ ] **Task 19.2.2**: Support conversion from ONNX initializers to SafeTensors
- [ ] **Task 19.2.3**: Support conversion from PyTorch state_dict to SafeTensors
- [ ] **Task 19.2.4**: Add `--export-weights safetensors` CLI flag

---

## Epic 20: CoreML Format (P2)

*Apple's ML framework for iOS/macOS deployment.*

### Story 20.1: CoreML Reader
- [ ] **Task 20.1.1**: Add coremltools dependency (optional)
- [ ] **Task 20.1.2**: Implement CoreMLAdapter.read() - load .mlmodel/.mlpackage
- [ ] **Task 20.1.3**: Map CoreML ops to UniversalNode ops
- [ ] **Task 20.1.4**: Extract CoreML-specific metadata (compute units, iOS version)
- [ ] **Task 20.1.5**: Integrate with analysis pipeline

### Story 20.2: CoreML Writer
- [ ] **Task 20.2.1**: Implement CoreMLAdapter.write() via coremltools conversion
- [ ] **Task 20.2.2**: Support ONNX → CoreML conversion path
- [ ] **Task 20.2.3**: Support PyTorch → CoreML conversion path
- [ ] **Task 20.2.4**: Add iOS/macOS deployment target options
- [ ] **Task 20.2.5**: Add `--convert-to coreml` CLI flag

---

## Epic 21: TFLite Format (P2)

*TensorFlow Lite for mobile and edge deployment.*

### Story 21.1: TFLite Reader
- [ ] **Task 21.1.1**: Add tflite-runtime dependency (optional)
- [ ] **Task 21.1.2**: Implement TFLiteAdapter.read() - load .tflite
- [ ] **Task 21.1.3**: Parse FlatBuffer schema for ops and tensors
- [ ] **Task 21.1.4**: Map TFLite ops to UniversalNode ops
- [ ] **Task 21.1.5**: Extract quantization info (int8, float16)

### Story 21.2: TFLite Writer
- [ ] **Task 21.2.1**: Implement TFLiteAdapter.write() via tf.lite.TFLiteConverter
- [ ] **Task 21.2.2**: Support ONNX → TFLite conversion (via TF intermediary)
- [ ] **Task 21.2.3**: Add quantization options (dynamic, full int8)
- [ ] **Task 21.2.4**: Add representative dataset hook for calibration
- [ ] **Task 21.2.5**: Add `--convert-to tflite` CLI flag

---

## Epic 22: TensorRT Format (P3 - Read-Only)

*NVIDIA's optimized inference engine. Read-only (compiled binaries).*

### Story 22.1: TensorRT Engine Reader
- [ ] **Task 22.1.1**: Add tensorrt dependency (optional)
- [ ] **Task 22.1.2**: Implement TensorRTAdapter.read()
- [ ] **Task 22.1.3**: Extract layer info (names, types, shapes, precision)
- [ ] **Task 22.1.4**: Extract timing/profiling data
- [ ] **Task 22.1.5**: Identify precision per layer

### Story 22.2: TensorRT Analysis Features
- [ ] **Task 22.2.1**: Add engine comparison (original vs optimized)
- [ ] **Task 22.2.2**: Show layer fusion visualization
- [ ] **Task 22.2.3**: Estimate memory footprint

---

## Epic 23: OpenVINO Format (P3)

*Intel's inference toolkit for CPU/GPU/VPU deployment.*

### Story 23.1: OpenVINO Reader
- [ ] **Task 23.1.1**: Add openvino dependency (optional)
- [ ] **Task 23.1.2**: Implement OpenVINOAdapter.read()
- [ ] **Task 23.1.3**: Map OpenVINO ops to UniversalNode
- [ ] **Task 23.1.4**: Extract Intel-specific optimizations

### Story 23.2: OpenVINO Writer
- [ ] **Task 23.2.1**: Implement OpenVINOAdapter.write()
- [ ] **Task 23.2.2**: Support ONNX → OpenVINO conversion
- [ ] **Task 23.2.3**: Add precision options
- [ ] **Task 23.2.4**: Add `--convert-to openvino` CLI flag

---

## Epic 24: GGUF Format (P3 - Read-Only)

*llama.cpp format for running LLMs locally.*

### Story 24.1: GGUF Reader
- [ ] **Task 24.1.1**: Implement GGUF header parser (pure Python)
- [ ] **Task 24.1.2**: Extract model metadata
- [ ] **Task 24.1.3**: Extract quantization type per tensor
- [ ] **Task 24.1.4**: Estimate memory footprint

### Story 24.2: GGUF Analysis Features
- [ ] **Task 24.2.1**: Show quantization breakdown
- [ ] **Task 24.2.2**: Estimate VRAM for different context lengths

---

## Epic 25: Privacy and Trust Architecture (P1)

*Enterprise customers need proof we won't steal their model IP.*

### Story 25.1: Local-First Architecture
- [ ] **Task 25.1.1**: Document "model never leaves your machine" guarantee
- [ ] **Task 25.1.2**: Audit code for any network calls or telemetry
- [ ] **Task 25.1.3**: Add `--offline` CLI flag (fail if network detected)
- [ ] **Task 25.1.4**: Create architecture diagram showing data flow

### Story 25.2: Output Controls
- [ ] **Task 25.2.1**: Add `--redact-names` flag (anonymize layer/tensor names)
- [ ] **Task 25.2.2**: Add `--summary-only` flag (stats only, no graph structure)
- [ ] **Task 25.2.3**: Document what information each output format reveals

### Story 25.3: Enterprise Trust Documentation
- [ ] **Task 25.3.1**: Write Privacy Policy / Data Handling document
- [ ] **Task 25.3.2**: Document open-source audit path ("read our code")

*Future: Confidential Computing (TEE) for cloud analysis - see Epic 10 SaaS.*

---

# LLM-SCALE ANALYSIS (P3+)

*Epics 26-30: Handle models like Opus 4.5, GPT-4, LLaMA-70B, Mixtral*

---

## Epic 26: Advanced Quantization Analysis (P3)

*Modern LLMs use complex quantization beyond simple int8/fp16.*

### Story 26.1: Mixed Precision Detection
- [ ] **Task 26.1.1**: Detect per-layer precision (weights vs activations vs accumulation)
- [ ] **Task 26.1.2**: Identify INT4 weights with FP16 activations pattern
- [ ] **Task 26.1.3**: Detect FP32 accumulation in quantized MatMuls
- [ ] **Task 26.1.4**: Report precision breakdown by layer type (attention vs FFN vs embed)
- [ ] **Task 26.1.5**: Visualize precision transitions in graph (where fp16→int8 happens)

### Story 26.2: Quantization Scheme Detection
- [ ] **Task 26.2.1**: Detect GPTQ quantization patterns (group-wise, act_order)
- [ ] **Task 26.2.2**: Detect AWQ quantization patterns (activation-aware)
- [ ] **Task 26.2.3**: Detect GGML/GGUF quantization types (Q4_0, Q4_K_M, Q5_K_S, etc.)
- [ ] **Task 26.2.4**: Detect bitsandbytes NF4/FP4 quantization
- [ ] **Task 26.2.5**: Report expected accuracy degradation per scheme
- [ ] **Task 26.2.6**: Compare memory vs accuracy tradeoffs between schemes

### Story 26.3: Calibration Analysis
- [ ] **Task 26.3.1**: Detect if model has calibration metadata
- [ ] **Task 26.3.2**: Estimate quantization error per layer
- [ ] **Task 26.3.3**: Identify sensitive layers (high quantization error)
- [ ] **Task 26.3.4**: Recommend layers to keep at higher precision

---

## Epic 27: Attention Variant Detection (P3)

*Modern LLMs use many attention optimizations beyond vanilla self-attention.*

### Story 27.1: Attention Architecture Detection
- [ ] **Task 27.1.1**: Detect Multi-Head Attention (MHA) - standard pattern
- [ ] **Task 27.1.2**: Detect Multi-Query Attention (MQA) - single KV head
- [ ] **Task 27.1.3**: Detect Grouped-Query Attention (GQA) - fewer KV heads than Q
- [ ] **Task 27.1.4**: Report num_q_heads, num_kv_heads, head_dim
- [ ] **Task 27.1.5**: Calculate KV cache savings for GQA/MQA vs MHA

### Story 27.2: Attention Pattern Detection
- [ ] **Task 27.2.1**: Detect sliding window attention (Mistral-style)
- [ ] **Task 27.2.2**: Detect local + global attention (Longformer-style)
- [ ] **Task 27.2.3**: Detect sparse attention patterns (BigBird, etc.)
- [ ] **Task 27.2.4**: Detect cross-attention (encoder-decoder models)
- [ ] **Task 27.2.5**: Report effective context length and attention complexity

### Story 27.3: Position Encoding Detection
- [ ] **Task 27.3.1**: Detect RoPE (Rotary Position Embedding)
- [ ] **Task 27.3.2**: Detect ALiBi (Attention with Linear Biases)
- [ ] **Task 27.3.3**: Detect learned positional embeddings
- [ ] **Task 27.3.4**: Detect sinusoidal positional encoding
- [ ] **Task 27.3.5**: Report max context length and extrapolation capability

### Story 27.4: Fused Attention Patterns
- [ ] **Task 27.4.1**: Detect FlashAttention-style fused patterns
- [ ] **Task 27.4.2**: Detect xFormers memory-efficient attention
- [ ] **Task 27.4.3**: Detect cuDNN fused multi-head attention
- [ ] **Task 27.4.4**: Report theoretical vs actual memory usage

---

## Epic 28: Memory Pattern Analysis (P3)

*LLM deployment is memory-bound. Understand where memory goes.*

### Story 28.1: Activation Checkpointing Detection
- [ ] **Task 28.1.1**: Detect activation checkpointing patterns (recompute on backward)
- [ ] **Task 28.1.2**: Identify checkpoint boundaries
- [ ] **Task 28.1.3**: Calculate memory savings vs compute overhead
- [ ] **Task 28.1.4**: Recommend optimal checkpoint granularity

### Story 28.2: KV Cache Analysis
- [ ] **Task 28.2.1**: Calculate KV cache size per layer per token
- [ ] **Task 28.2.2**: Project KV cache for variable context lengths (1k, 4k, 8k, 32k, 128k)
- [ ] **Task 28.2.3**: Detect KV cache quantization (INT8 KV cache)
- [ ] **Task 28.2.4**: Calculate max context length for given VRAM
- [ ] **Task 28.2.5**: Detect PagedAttention patterns (vLLM-style)
- [ ] **Task 28.2.6**: Report KV cache as % of total memory

### Story 28.3: Parallelism Strategy Detection
- [ ] **Task 28.3.1**: Detect tensor parallelism patterns (column/row split)
- [ ] **Task 28.3.2**: Detect pipeline parallelism patterns (layer sharding)
- [ ] **Task 28.3.3**: Detect data parallelism patterns
- [ ] **Task 28.3.4**: Identify all-reduce / all-gather communication ops
- [ ] **Task 28.3.5**: Report memory per GPU for N-way parallelism
- [ ] **Task 28.3.6**: Recommend parallelism strategy for target hardware

### Story 28.4: Memory Waterfall Analysis
- [ ] **Task 28.4.1**: Calculate peak memory at each point in forward pass
- [ ] **Task 28.4.2**: Generate memory waterfall chart (memory over time)
- [ ] **Task 28.4.3**: Identify memory spike locations
- [ ] **Task 28.4.4**: Recommend batch size for given VRAM constraint

---

## Epic 29: Sparse and Efficient Architecture Analysis (P3)

*Mixture of Experts, speculative decoding, and sparsity patterns.*

### Story 29.1: Mixture of Experts (MoE) Analysis
- [ ] **Task 29.1.1**: Detect MoE routing patterns (top-k gating)
- [ ] **Task 29.1.2**: Count total experts and active experts per token
- [ ] **Task 29.1.3**: Calculate effective vs total parameters
- [ ] **Task 29.1.4**: Analyze expert utilization/load balancing
- [ ] **Task 29.1.5**: Report memory for all experts vs active subset
- [ ] **Task 29.1.6**: Detect expert parallelism patterns

### Story 29.2: Speculative Decoding Detection
- [ ] **Task 29.2.1**: Detect draft model + verify model pattern
- [ ] **Task 29.2.2**: Identify draft model architecture
- [ ] **Task 29.2.3**: Calculate speculative decoding speedup potential
- [ ] **Task 29.2.4**: Report token acceptance rate requirements

### Story 29.3: Weight Sparsity Analysis
- [ ] **Task 29.3.1**: Detect structured sparsity (N:M sparsity)
- [ ] **Task 29.3.2**: Detect unstructured sparsity (pruned weights)
- [ ] **Task 29.3.3**: Calculate actual vs theoretical FLOPs with sparsity
- [ ] **Task 29.3.4**: Identify sparse-compatible hardware requirements
- [ ] **Task 29.3.5**: Report sparsity % per layer

### Story 29.4: Efficient Architecture Patterns
- [ ] **Task 29.4.1**: Detect depth-wise separable convolutions
- [ ] **Task 29.4.2**: Detect inverted residual blocks (MobileNet-style)
- [ ] **Task 29.4.3**: Detect squeeze-and-excitation patterns
- [ ] **Task 29.4.4**: Detect neural architecture search (NAS) patterns
- [ ] **Task 29.4.5**: Compare efficiency vs baseline architectures

---

## Epic 30: LLM Deployment Analysis (P3)

*Inference patterns differ from training. Understand production characteristics.*

### Story 30.1: Prefill vs Decode Analysis
- [ ] **Task 30.1.1**: Identify prefill phase (process prompt, compute-bound)
- [ ] **Task 30.1.2**: Identify decode phase (generate tokens, memory-bound)
- [ ] **Task 30.1.3**: Calculate time-to-first-token (TTFT) estimate
- [ ] **Task 30.1.4**: Calculate tokens-per-second decode rate
- [ ] **Task 30.1.5**: Report optimal batch size for each phase

### Story 30.2: Batching Strategy Analysis
- [ ] **Task 30.2.1**: Analyze static batching characteristics
- [ ] **Task 30.2.2**: Detect continuous batching compatibility
- [ ] **Task 30.2.3**: Calculate throughput vs latency tradeoffs
- [ ] **Task 30.2.4**: Report max concurrent requests for given VRAM
- [ ] **Task 30.2.5**: Model request queuing and scheduling impact

### Story 30.3: Context Length Scaling
- [ ] **Task 30.3.1**: Calculate O(n²) attention scaling impact
- [ ] **Task 30.3.2**: Calculate O(n) KV cache scaling impact
- [ ] **Task 30.3.3**: Generate context length vs memory/latency curves
- [ ] **Task 30.3.4**: Identify context length breakpoints (where OOM occurs)
- [ ] **Task 30.3.5**: Recommend context length for target hardware

### Story 30.4: Serving Framework Compatibility
- [ ] **Task 30.4.1**: Check vLLM compatibility (PagedAttention, continuous batching)
- [ ] **Task 30.4.2**: Check TensorRT-LLM compatibility
- [ ] **Task 30.4.3**: Check llama.cpp compatibility
- [ ] **Task 30.4.4**: Check Triton Inference Server compatibility
- [ ] **Task 30.4.5**: Report recommended serving framework for model characteristics
