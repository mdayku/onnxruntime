# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
Entry point for model_inspect CLI tool.

This module provides the CLI interface for ONNX Autodoc model inspection.
"""
from __future__ import annotations

import argparse
import logging
import os
import pathlib
import sys
from typing import Any

from .autodoc import ModelInspector
from .autodoc.hardware import (
    HardwareEstimator,
    detect_local_hardware,
    get_profile,
)
from .autodoc.llm_summarizer import (
    LLMSummarizer,
)
from .autodoc.llm_summarizer import (
    has_api_key as has_llm_api_key,
)
from .autodoc.llm_summarizer import (
    is_available as is_llm_available,
)
from .autodoc.visualizations import (
    VisualizationGenerator,
)
from .autodoc.visualizations import (
    is_available as is_viz_available,
)


class ProgressIndicator:
    """Simple progress indicator for CLI operations."""

    def __init__(self, enabled: bool = True, quiet: bool = False):
        self.enabled = enabled and not quiet
        self._current_step = 0
        self._total_steps = 0

    def start(self, total_steps: int, description: str = "Processing"):
        """Start progress tracking."""
        self._total_steps = total_steps
        self._current_step = 0
        if self.enabled:
            print(f"\n{description}...")

    def step(self, message: str):
        """Mark completion of a step."""
        self._current_step += 1
        if self.enabled:
            pct = (
                (self._current_step / self._total_steps * 100)
                if self._total_steps
                else 0
            )
            print(
                f"  [{self._current_step}/{self._total_steps}] {message} ({pct:.0f}%)"
            )

    def finish(self, message: str = "Done"):
        """Mark completion of all steps."""
        if self.enabled:
            print(f"  {message}\n")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        os.path.basename(__file__),
        description="Analyze an ONNX model and generate architecture documentation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic inspection with console output (auto-detects local hardware)
  python -m onnxruntime.tools.model_inspect model.onnx

  # Use specific NVIDIA GPU profile for estimates
  python -m onnxruntime.tools.model_inspect model.onnx --hardware a100

  # List available hardware profiles
  python -m onnxruntime.tools.model_inspect --list-hardware

  # Generate JSON report with hardware estimates
  python -m onnxruntime.tools.model_inspect model.onnx --hardware rtx4090 --out-json report.json

  # Specify precision and batch size for hardware estimates
  python -m onnxruntime.tools.model_inspect model.onnx --hardware t4 --precision fp16 --batch-size 8
""",
    )

    parser.add_argument(
        "model_path",
        type=pathlib.Path,
        nargs="?",  # Optional now since --list-hardware doesn't need it
        help="Path to the ONNX model file to analyze.",
    )

    parser.add_argument(
        "--out-json",
        type=pathlib.Path,
        default=None,
        help="Output path for JSON report. If not specified, no JSON is written.",
    )

    parser.add_argument(
        "--out-md",
        type=pathlib.Path,
        default=None,
        help="Output path for Markdown model card. If not specified, no Markdown is written.",
    )

    parser.add_argument(
        "--out-html",
        type=pathlib.Path,
        default=None,
        help="Output path for HTML report with embedded images. Single shareable file.",
    )

    # PyTorch conversion options
    pytorch_group = parser.add_argument_group("PyTorch Conversion Options")
    pytorch_group.add_argument(
        "--from-pytorch",
        type=pathlib.Path,
        default=None,
        metavar="MODEL_PATH",
        help="Convert a PyTorch model (.pth, .pt) to ONNX before analysis. Requires torch.",
    )
    pytorch_group.add_argument(
        "--input-shape",
        type=str,
        default=None,
        metavar="SHAPE",
        help="Input shape for PyTorch conversion, e.g., '1,3,224,224'. Required with --from-pytorch.",
    )
    pytorch_group.add_argument(
        "--keep-onnx",
        type=pathlib.Path,
        default=None,
        metavar="PATH",
        help="Save the converted ONNX model to this path (otherwise uses temp file).",
    )
    pytorch_group.add_argument(
        "--opset-version",
        type=int,
        default=17,
        help="ONNX opset version for PyTorch export (default: 17).",
    )
    pytorch_group.add_argument(
        "--pytorch-weights",
        type=pathlib.Path,
        default=None,
        metavar="PATH",
        help="Path to original PyTorch weights (.pt) to extract class names/metadata. "
        "Useful when analyzing a pre-converted ONNX file.",
    )

    # Hardware options
    hardware_group = parser.add_argument_group("Hardware Options")
    hardware_group.add_argument(
        "--hardware",
        type=str,
        default=None,
        metavar="PROFILE",
        help="Hardware profile for performance estimates. Use 'auto' to detect local hardware, "
        "or specify a profile name (e.g., 'a100', 'rtx4090', 't4'). Use --list-hardware to see all options.",
    )

    hardware_group.add_argument(
        "--list-hardware",
        action="store_true",
        help="List all available hardware profiles and exit.",
    )

    hardware_group.add_argument(
        "--precision",
        choices=["fp32", "fp16", "bf16", "int8"],
        default="fp32",
        help="Precision for hardware estimates (default: fp32).",
    )

    hardware_group.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for hardware estimates (default: 1).",
    )

    # Visualization options
    viz_group = parser.add_argument_group("Visualization Options")
    viz_group.add_argument(
        "--with-plots",
        action="store_true",
        help="Generate visualization assets (requires matplotlib).",
    )

    viz_group.add_argument(
        "--assets-dir",
        type=pathlib.Path,
        default=None,
        metavar="PATH",
        help="Directory for plot PNG files (default: same directory as output files, or 'assets/').",
    )

    # LLM options
    llm_group = parser.add_argument_group("LLM Summarization Options")
    llm_group.add_argument(
        "--llm-summary",
        action="store_true",
        help="Generate LLM-powered summaries (requires openai package and OPENAI_API_KEY env var).",
    )

    llm_group.add_argument(
        "--llm-model",
        type=str,
        default="gpt-4o-mini",
        metavar="MODEL",
        help="OpenAI model to use for summaries (default: gpt-4o-mini).",
    )

    parser.add_argument(
        "--log-level",
        choices=["debug", "info", "warning", "error"],
        default="info",
        help="Logging verbosity level (default: info).",
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress console output. Only write to files if --out-json or --out-md specified.",
    )

    parser.add_argument(
        "--progress",
        action="store_true",
        help="Show progress indicators during analysis (useful for large models).",
    )

    return parser.parse_args()


def setup_logging(log_level: str) -> logging.Logger:
    """Configure logging for the CLI."""
    level_map = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
    }

    logging.basicConfig(
        level=level_map.get(log_level, logging.INFO),
        format="%(levelname)s - %(message)s",
    )

    return logging.getLogger("autodoc")


def _generate_markdown_with_extras(
    report, viz_paths: dict, report_dir: pathlib.Path, llm_summary=None
) -> str:
    """Generate markdown with embedded visualizations and LLM summaries."""
    lines = []
    base_md = report.to_markdown()

    # If we have an LLM summary, insert it after the header
    if llm_summary and llm_summary.success:
        # Insert executive summary after the metadata section
        header_end = base_md.find("## Graph Summary")
        if header_end != -1:
            lines.append(base_md[:header_end])
            lines.append("## Executive Summary\n")
            if llm_summary.short_summary:
                lines.append(f"**TL;DR:** {llm_summary.short_summary}\n")
            if llm_summary.detailed_summary:
                lines.append(f"\n{llm_summary.detailed_summary}\n")
            lines.append(f"\n*Generated by {llm_summary.model_used}*\n\n")
            base_md = base_md[header_end:]

    # Split the markdown at the Complexity Metrics section to insert plots
    sections = base_md.split("## Complexity Metrics")

    if len(sections) < 2:
        # No complexity section found, just append plots at end
        lines.append(base_md)
    else:
        lines.append(sections[0])

        # Insert visualizations section before Complexity Metrics
        if viz_paths:
            lines.append("## Visualizations\n")

            if "complexity_summary" in viz_paths:
                rel_path = (
                    viz_paths["complexity_summary"].relative_to(report_dir)
                    if viz_paths["complexity_summary"].is_relative_to(report_dir)
                    else viz_paths["complexity_summary"]
                )
                lines.append(f"### Complexity Overview\n")
                lines.append(f"![Complexity Summary]({rel_path})\n")

            if "op_histogram" in viz_paths:
                rel_path = (
                    viz_paths["op_histogram"].relative_to(report_dir)
                    if viz_paths["op_histogram"].is_relative_to(report_dir)
                    else viz_paths["op_histogram"]
                )
                lines.append(f"### Operator Distribution\n")
                lines.append(f"![Operator Histogram]({rel_path})\n")

            if "param_distribution" in viz_paths:
                rel_path = (
                    viz_paths["param_distribution"].relative_to(report_dir)
                    if viz_paths["param_distribution"].is_relative_to(report_dir)
                    else viz_paths["param_distribution"]
                )
                lines.append(f"### Parameter Distribution\n")
                lines.append(f"![Parameter Distribution]({rel_path})\n")

            if "flops_distribution" in viz_paths:
                rel_path = (
                    viz_paths["flops_distribution"].relative_to(report_dir)
                    if viz_paths["flops_distribution"].is_relative_to(report_dir)
                    else viz_paths["flops_distribution"]
                )
                lines.append(f"### FLOPs Distribution\n")
                lines.append(f"![FLOPs Distribution]({rel_path})\n")

            lines.append("")

        lines.append("## Complexity Metrics" + sections[1])

    return "\n".join(lines)


def _extract_ultralytics_metadata(
    weights_path: pathlib.Path,
    logger: logging.Logger,
) -> dict[str, Any] | None:
    """
    Extract metadata from an Ultralytics model (.pt file).

    Returns dict with task, num_classes, class_names or None if not Ultralytics.
    """
    try:
        from ultralytics import YOLO

        model = YOLO(str(weights_path))

        return {
            "task": model.task,
            "num_classes": len(model.names),
            "class_names": list(model.names.values()),
            "source": "ultralytics",
        }
    except ImportError:
        logger.debug("ultralytics not installed, skipping metadata extraction")
        return None
    except Exception as e:
        logger.debug(f"Could not extract Ultralytics metadata: {e}")
        return None


def _convert_pytorch_to_onnx(
    pytorch_path: pathlib.Path,
    input_shape_str: str | None,
    output_path: pathlib.Path | None,
    opset_version: int,
    logger: logging.Logger,
) -> tuple[pathlib.Path | None, Any]:
    """
    Convert a PyTorch model to ONNX format.

    Args:
        pytorch_path: Path to PyTorch model (.pth, .pt)
        input_shape_str: Input shape as comma-separated string, e.g., "1,3,224,224"
        output_path: Where to save ONNX file (None = temp file)
        opset_version: ONNX opset version
        logger: Logger instance

    Returns:
        Tuple of (onnx_path, temp_file_handle_or_None)
    """
    # Check if torch is available
    try:
        import torch
    except ImportError:
        logger.error("PyTorch not installed. Install with: pip install torch")
        return None, None

    pytorch_path = pytorch_path.resolve()
    if not pytorch_path.exists():
        logger.error(f"PyTorch model not found: {pytorch_path}")
        return None, None

    # Parse input shape
    if not input_shape_str:
        logger.error(
            "--input-shape is required for PyTorch conversion. "
            "Example: --input-shape 1,3,224,224"
        )
        return None, None

    try:
        input_shape = tuple(int(x.strip()) for x in input_shape_str.split(","))
        logger.info(f"Input shape: {input_shape}")
    except ValueError:
        logger.error(
            f"Invalid --input-shape format: '{input_shape_str}'. "
            "Use comma-separated integers, e.g., '1,3,224,224'"
        )
        return None, None

    # Load PyTorch model
    logger.info(f"Loading PyTorch model from: {pytorch_path}")
    model = None

    # Try 1: TorchScript model (.pt files from torch.jit.save)
    try:
        model = torch.jit.load(str(pytorch_path), map_location="cpu")
        logger.info(f"Loaded TorchScript model: {type(model).__name__}")
    except Exception:
        pass

    # Try 2: Regular torch.load (full model or state_dict)
    if model is None:
        try:
            loaded = torch.load(pytorch_path, map_location="cpu", weights_only=False)

            if isinstance(loaded, dict):
                # It's a state_dict - we can't use it directly
                logger.error(
                    "Model file appears to be a state_dict (weights only). "
                    "To convert, you need either:\n"
                    "  1. A TorchScript model: torch.jit.save(torch.jit.script(model), 'model.pt')\n"
                    "  2. A full model: torch.save(model, 'model.pth')  # run from same codebase\n"
                    "  3. Export to ONNX directly in your training code using torch.onnx.export()"
                )
                return None, None

            model = loaded
            logger.info(f"Loaded PyTorch model: {type(model).__name__}")

        except Exception as e:
            error_msg = str(e)
            if "Can't get attribute" in error_msg:
                logger.error(
                    f"Failed to load model - class definition not found.\n"
                    f"The model was saved with torch.save(model, ...) which requires "
                    f"the original class to be importable.\n\n"
                    f"Solutions:\n"
                    f"  1. Save as TorchScript: torch.jit.save(torch.jit.script(model), 'model.pt')\n"
                    f"  2. Export to ONNX in your code: torch.onnx.export(model, dummy_input, 'model.onnx')\n"
                    f"  3. Run this tool from the directory containing your model definition"
                )
            else:
                logger.error(f"Failed to load PyTorch model: {e}")
            return None, None

    if model is None:
        logger.error("Could not load the PyTorch model.")
        return None, None

    model.eval()

    # Create dummy input
    try:
        dummy_input = torch.randn(*input_shape)
        logger.info(f"Created dummy input with shape: {dummy_input.shape}")
    except Exception as e:
        logger.error(f"Failed to create input tensor: {e}")
        return None, None

    # Determine output path
    temp_file = None
    if output_path:
        onnx_path = output_path.resolve()
        onnx_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        import tempfile

        temp_file = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)
        onnx_path = pathlib.Path(temp_file.name)
        temp_file.close()

    # Export to ONNX
    logger.info(f"Exporting to ONNX (opset {opset_version})...")
    try:
        torch.onnx.export(
            model,
            dummy_input,
            str(onnx_path),
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "output": {0: "batch_size"},
            },
            opset_version=opset_version,
            do_constant_folding=True,
        )
        logger.info(f"ONNX model saved to: {onnx_path}")

        # Verify the export
        import onnx

        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)
        logger.info("ONNX model validated successfully")

    except Exception as e:
        logger.error(f"ONNX export failed: {e}")
        if temp_file:
            try:
                onnx_path.unlink()
            except Exception:
                pass
        return None, None

    return onnx_path, temp_file


def run_inspect():
    """Main entry point for the model_inspect CLI."""
    args = parse_args()
    logger = setup_logging(args.log_level)

    # Handle --list-hardware
    if args.list_hardware:
        print("\n" + "=" * 70)
        print("Available Hardware Profiles")
        print("=" * 70)

        print("\nData Center GPUs (Current Gen):")
        for name in [
            "h100",
            "a100-80gb",
            "a100-40gb",
            "a10",
            "l4",
            "l40",
            "l40s",
            "t4",
        ]:
            profile = get_profile(name)
            if profile:
                print(
                    f"  {name:20} {profile.name:30} {profile.vram_bytes // (1024**3):3} GB  {profile.peak_fp16_tflops:6.1f} TF16"
                )

        print("\nData Center GPUs (Previous Gen):")
        for name in ["v100-32gb", "v100-16gb", "p100", "p40"]:
            profile = get_profile(name)
            if profile:
                print(
                    f"  {name:20} {profile.name:30} {profile.vram_bytes // (1024**3):3} GB  {profile.peak_fp16_tflops:6.1f} TF16"
                )

        print("\nJetson Edge/Embedded (Orin Series - Recommended for new projects):")
        for name in [
            "jetson-agx-orin-64gb",
            "jetson-agx-orin-32gb",
            "jetson-orin-nx-16gb",
            "jetson-orin-nx-8gb",
            "jetson-orin-nano-8gb",
            "jetson-orin-nano-4gb",
        ]:
            profile = get_profile(name)
            if profile:
                print(
                    f"  {name:20} {profile.name:30} {profile.vram_bytes // (1024**3):3} GB  {profile.peak_fp16_tflops:6.1f} TF16"
                )

        print("\nJetson Edge/Embedded (Xavier Series):")
        for name in ["jetson-agx-xavier", "jetson-xavier-nx-8gb"]:
            profile = get_profile(name)
            if profile:
                print(
                    f"  {name:20} {profile.name:30} {profile.vram_bytes // (1024**3):3} GB  {profile.peak_fp16_tflops:6.1f} TF16"
                )

        print("\nJetson Edge/Embedded (Legacy - Very Constrained!):")
        for name in ["jetson-tx2", "jetson-nano", "jetson-nano-2gb"]:
            profile = get_profile(name)
            if profile:
                vram_gb = profile.vram_bytes / (1024**3)
                print(
                    f"  {name:20} {profile.name:30} {vram_gb:3.0f} GB  {profile.peak_fp16_tflops:6.3f} TF16"
                )

        print("\nConsumer GPUs:")
        for name in ["rtx4090", "rtx4080", "rtx3090", "rtx3080"]:
            profile = get_profile(name)
            if profile:
                print(
                    f"  {name:20} {profile.name:30} {profile.vram_bytes // (1024**3):3} GB  {profile.peak_fp16_tflops:6.1f} TF16"
                )

        print("\nOther:")
        print("  auto                 Auto-detect local GPU/CPU")
        print("  cpu                  Generic CPU profile")

        print("\n" + "-" * 70)
        print("TF16 = Peak FP16 TFLOPS (higher = faster)")
        print("Jetson uses unified memory (shared between CPU and GPU)")
        print("=" * 70 + "\n")
        sys.exit(0)

    # Handle PyTorch conversion if requested
    temp_onnx_file = None
    if args.from_pytorch:
        model_path, temp_onnx_file = _convert_pytorch_to_onnx(
            args.from_pytorch,
            args.input_shape,
            args.keep_onnx,
            args.opset_version,
            logger,
        )
        if model_path is None:
            sys.exit(1)
    else:
        # Validate model path
        if args.model_path is None:
            logger.error(
                "Model path is required. Use --list-hardware to see available profiles, "
                "or --from-pytorch to convert a PyTorch model."
            )
            sys.exit(1)

        model_path = args.model_path.resolve()
        if not model_path.exists():
            logger.error(f"Model file not found: {model_path}")
            sys.exit(1)

        if model_path.suffix.lower() not in (".onnx", ".pb", ".ort"):
            logger.warning(
                f"Unexpected file extension: {model_path.suffix}. Proceeding anyway."
            )

    # Determine hardware profile
    hardware_profile = None
    if args.hardware:
        if args.hardware.lower() == "auto":
            logger.info("Auto-detecting local hardware...")
            hardware_profile = detect_local_hardware()
            logger.info(f"Detected: {hardware_profile.name}")
        else:
            hardware_profile = get_profile(args.hardware)
            if hardware_profile is None:
                logger.error(f"Unknown hardware profile: {args.hardware}")
                logger.error("Use --list-hardware to see available profiles.")
                sys.exit(1)
            logger.info(f"Using hardware profile: {hardware_profile.name}")

    # Setup progress indicator
    progress = ProgressIndicator(enabled=args.progress, quiet=args.quiet)

    # Calculate total steps based on what will be done
    total_steps = 2  # Load + Analyze always
    if hardware_profile:
        total_steps += 1
    if args.with_plots and is_viz_available():
        total_steps += 1
    if args.llm_summary and is_llm_available() and has_llm_api_key():
        total_steps += 1
    if args.out_json or args.out_md or args.out_html:
        total_steps += 1

    progress.start(total_steps, f"Analyzing {model_path.name}")

    # Run inspection
    try:
        progress.step("Loading model and extracting graph structure")
        inspector = ModelInspector(logger=logger)
        report = inspector.inspect(model_path)
        progress.step("Computing metrics (params, FLOPs, memory)")

        # Add hardware estimates if profile specified
        if (
            hardware_profile
            and report.param_counts
            and report.flop_counts
            and report.memory_estimates
        ):
            progress.step(f"Estimating performance on {hardware_profile.name}")
            estimator = HardwareEstimator(logger=logger)
            hw_estimates = estimator.estimate(
                model_params=report.param_counts.total,
                model_flops=report.flop_counts.total,
                peak_activation_bytes=report.memory_estimates.peak_activation_bytes,
                hardware=hardware_profile,
                batch_size=args.batch_size,
                precision=args.precision,
            )
            report.hardware_estimates = hw_estimates
            report.hardware_profile = hardware_profile

    except Exception as e:
        logger.error(f"Failed to inspect model: {e}")
        if args.log_level == "debug":
            import traceback

            traceback.print_exc()
        sys.exit(1)

    # Extract dataset metadata if PyTorch weights provided
    if args.pytorch_weights or args.from_pytorch:
        weights_path = args.pytorch_weights or args.from_pytorch
        if weights_path.exists():
            logger.info(f"Extracting metadata from: {weights_path}")
            metadata = _extract_ultralytics_metadata(weights_path, logger)
            if metadata:
                from .autodoc.report import DatasetInfo

                report.dataset_info = DatasetInfo(
                    task=metadata.get("task"),
                    num_classes=metadata.get("num_classes"),
                    class_names=metadata.get("class_names", []),
                    source=metadata.get("source"),
                )
                logger.info(
                    f"Extracted {report.dataset_info.num_classes} class(es): "
                    f"{', '.join(report.dataset_info.class_names[:5])}"
                    f"{'...' if len(report.dataset_info.class_names) > 5 else ''}"
                )

    # Generate LLM summaries if requested
    llm_summary = None
    if args.llm_summary:
        if not is_llm_available():
            logger.warning(
                "openai package not installed. Skipping LLM summaries. Install with: pip install openai"
            )
        elif not has_llm_api_key():
            logger.warning("OPENAI_API_KEY not set. Skipping LLM summaries.")
        else:
            try:
                progress.step(f"Generating LLM summary with {args.llm_model}")
                logger.info(f"Generating LLM summaries with {args.llm_model}...")
                summarizer = LLMSummarizer(model=args.llm_model, logger=logger)
                llm_summary = summarizer.summarize(report)
                if llm_summary.success:
                    logger.info(
                        f"LLM summaries generated ({llm_summary.tokens_used} tokens used)"
                    )
                else:
                    logger.warning(
                        f"LLM summarization failed: {llm_summary.error_message}"
                    )
            except Exception as e:
                logger.warning(f"Failed to generate LLM summaries: {e}")

    # Store LLM summary in report for output
    if llm_summary and llm_summary.success:
        # Add to report dict for JSON output
        report._llm_summary = llm_summary  # type: ignore

    # Output results
    has_output = args.out_json or args.out_md or args.out_html
    if has_output:
        progress.step("Writing output files")

    if args.out_json:
        try:
            args.out_json.parent.mkdir(parents=True, exist_ok=True)
            args.out_json.write_text(report.to_json(), encoding="utf-8")
            logger.info(f"JSON report written to: {args.out_json}")
        except Exception as e:
            logger.error(f"Failed to write JSON report: {e}")
            sys.exit(1)

    # Generate visualizations if requested
    viz_paths = {}
    if args.with_plots:
        if not is_viz_available():
            logger.warning(
                "matplotlib not installed. Skipping visualizations. Install with: pip install matplotlib"
            )
        else:
            progress.step("Generating visualizations")
            # Determine assets directory
            if args.assets_dir:
                assets_dir = args.assets_dir
            elif args.out_html:
                # HTML embeds images, but we still generate them for the file
                assets_dir = args.out_html.parent / "assets"
            elif args.out_md:
                assets_dir = args.out_md.parent / "assets"
            elif args.out_json:
                assets_dir = args.out_json.parent / "assets"
            else:
                assets_dir = pathlib.Path("assets")

            try:
                viz_gen = VisualizationGenerator(logger=logger)
                viz_paths = viz_gen.generate_all(report, assets_dir)
                logger.info(
                    f"Generated {len(viz_paths)} visualization assets in {assets_dir}"
                )
            except Exception as e:
                logger.warning(f"Failed to generate some visualizations: {e}")
                if args.log_level == "debug":
                    import traceback

                    traceback.print_exc()

    if args.out_md:
        try:
            args.out_md.parent.mkdir(parents=True, exist_ok=True)
            # Generate markdown with visualizations and/or LLM summaries
            if viz_paths or llm_summary:
                md_content = _generate_markdown_with_extras(
                    report, viz_paths, args.out_md.parent, llm_summary
                )
            else:
                md_content = report.to_markdown()
            args.out_md.write_text(md_content, encoding="utf-8")
            logger.info(f"Markdown model card written to: {args.out_md}")
        except Exception as e:
            logger.error(f"Failed to write Markdown report: {e}")
            sys.exit(1)

    if args.out_html:
        try:
            args.out_html.parent.mkdir(parents=True, exist_ok=True)
            # Add LLM summary to report if available
            if llm_summary and llm_summary.success:
                report.llm_summary = {
                    "success": True,
                    "short_summary": llm_summary.short_summary,
                    "detailed_summary": llm_summary.detailed_summary,
                    "model": args.llm_model,
                }
            # Generate HTML with embedded images
            html_content = report.to_html(image_paths=viz_paths)
            args.out_html.write_text(html_content, encoding="utf-8")
            logger.info(f"HTML report written to: {args.out_html}")
        except Exception as e:
            logger.error(f"Failed to write HTML report: {e}")
            sys.exit(1)

    # Console output
    if not args.quiet and not args.out_json and not args.out_md and not args.out_html:
        # No output files specified - print summary to console
        print("\n" + "=" * 60)
        print(f"Model: {model_path.name}")
        print("=" * 60)

        if report.graph_summary:
            print(f"\nNodes: {report.graph_summary.num_nodes}")
            print(f"Inputs: {report.graph_summary.num_inputs}")
            print(f"Outputs: {report.graph_summary.num_outputs}")
            print(f"Initializers: {report.graph_summary.num_initializers}")

        if report.param_counts:
            print(f"\nParameters: {report._format_number(report.param_counts.total)}")

        if report.flop_counts:
            print(f"FLOPs: {report._format_number(report.flop_counts.total)}")

        if report.memory_estimates:
            print(
                f"Model Size: {report._format_bytes(report.memory_estimates.model_size_bytes)}"
            )

        print(f"\nArchitecture: {report.architecture_type}")
        print(f"Detected Blocks: {len(report.detected_blocks)}")

        # Hardware estimates
        if hasattr(report, "hardware_estimates") and report.hardware_estimates:
            hw = report.hardware_estimates
            print(f"\n--- Hardware Estimates ({hw.device}) ---")
            print(f"Precision: {hw.precision}, Batch Size: {hw.batch_size}")
            print(f"VRAM Required: {report._format_bytes(hw.vram_required_bytes)}")
            print(f"Fits in VRAM: {'Yes' if hw.fits_in_vram else 'NO'}")
            if hw.fits_in_vram:
                print(f"Theoretical Latency: {hw.theoretical_latency_ms:.2f} ms")
                print(f"Bottleneck: {hw.bottleneck}")

        if report.risk_signals:
            print(f"\nRisk Signals: {len(report.risk_signals)}")
            for risk in report.risk_signals:
                severity_icon = {
                    "info": "[INFO]",
                    "warning": "[WARN]",
                    "high": "[HIGH]",
                }
                print(f"  {severity_icon.get(risk.severity, '')} {risk.id}")

        # LLM Summary
        if llm_summary and llm_summary.success:
            print(f"\n--- LLM Summary ({llm_summary.model_used}) ---")
            if llm_summary.short_summary:
                print(f"{llm_summary.short_summary}")

        print("\n" + "=" * 60)
        print("Use --out-json or --out-md for detailed reports.")
        if not args.hardware:
            print("Use --hardware auto or --hardware <profile> for hardware estimates.")
        print("=" * 60 + "\n")

    elif not args.quiet:
        # Files written - just confirm
        print(f"\nInspection complete for: {model_path.name}")
        if args.out_json:
            print(f"  JSON report: {args.out_json}")
        if args.out_md:
            print(f"  Markdown card: {args.out_md}")
        if args.out_html:
            print(f"  HTML report: {args.out_html}")

    # Finish progress indicator
    progress.finish("Analysis complete!")

    # Cleanup temp ONNX file if we created one
    if temp_onnx_file is not None:
        try:
            pathlib.Path(temp_onnx_file.name).unlink()
            logger.debug(f"Cleaned up temp ONNX file: {temp_onnx_file.name}")
        except Exception:
            pass


if __name__ == "__main__":
    run_inspect()
