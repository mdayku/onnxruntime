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

from .autodoc import ModelInspector
from .autodoc.hardware import (
    HARDWARE_PROFILES,
    HardwareEstimator,
    detect_local_hardware,
    get_profile,
    list_available_profiles,
)
from .autodoc.visualizations import (
    VisualizationGenerator,
    is_available as is_viz_available,
)


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


def _generate_markdown_with_plots(report, viz_paths: dict, report_dir: pathlib.Path) -> str:
    """Generate markdown with embedded visualization images."""
    lines = []
    base_md = report.to_markdown()

    # Split the markdown at the Graph Summary section to insert plots
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
                rel_path = viz_paths["complexity_summary"].relative_to(report_dir) if viz_paths["complexity_summary"].is_relative_to(report_dir) else viz_paths["complexity_summary"]
                lines.append(f"### Complexity Overview\n")
                lines.append(f"![Complexity Summary]({rel_path})\n")

            if "op_histogram" in viz_paths:
                rel_path = viz_paths["op_histogram"].relative_to(report_dir) if viz_paths["op_histogram"].is_relative_to(report_dir) else viz_paths["op_histogram"]
                lines.append(f"### Operator Distribution\n")
                lines.append(f"![Operator Histogram]({rel_path})\n")

            if "param_distribution" in viz_paths:
                rel_path = viz_paths["param_distribution"].relative_to(report_dir) if viz_paths["param_distribution"].is_relative_to(report_dir) else viz_paths["param_distribution"]
                lines.append(f"### Parameter Distribution\n")
                lines.append(f"![Parameter Distribution]({rel_path})\n")

            if "flops_distribution" in viz_paths:
                rel_path = viz_paths["flops_distribution"].relative_to(report_dir) if viz_paths["flops_distribution"].is_relative_to(report_dir) else viz_paths["flops_distribution"]
                lines.append(f"### FLOPs Distribution\n")
                lines.append(f"![FLOPs Distribution]({rel_path})\n")

            lines.append("")

        lines.append("## Complexity Metrics" + sections[1])

    return "\n".join(lines)


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
        for name in ["h100", "a100-80gb", "a100-40gb", "a10", "l4", "l40", "l40s", "t4"]:
            profile = get_profile(name)
            if profile:
                print(f"  {name:20} {profile.name:30} {profile.vram_bytes // (1024**3):3} GB  {profile.peak_fp16_tflops:6.1f} TF16")

        print("\nData Center GPUs (Previous Gen):")
        for name in ["v100-32gb", "v100-16gb", "p100", "p40"]:
            profile = get_profile(name)
            if profile:
                print(f"  {name:20} {profile.name:30} {profile.vram_bytes // (1024**3):3} GB  {profile.peak_fp16_tflops:6.1f} TF16")

        print("\nJetson Edge/Embedded (Orin Series - Recommended for new projects):")
        for name in ["jetson-agx-orin-64gb", "jetson-agx-orin-32gb", "jetson-orin-nx-16gb",
                     "jetson-orin-nx-8gb", "jetson-orin-nano-8gb", "jetson-orin-nano-4gb"]:
            profile = get_profile(name)
            if profile:
                print(f"  {name:20} {profile.name:30} {profile.vram_bytes // (1024**3):3} GB  {profile.peak_fp16_tflops:6.1f} TF16")

        print("\nJetson Edge/Embedded (Xavier Series):")
        for name in ["jetson-agx-xavier", "jetson-xavier-nx-8gb"]:
            profile = get_profile(name)
            if profile:
                print(f"  {name:20} {profile.name:30} {profile.vram_bytes // (1024**3):3} GB  {profile.peak_fp16_tflops:6.1f} TF16")

        print("\nJetson Edge/Embedded (Legacy - Very Constrained!):")
        for name in ["jetson-tx2", "jetson-nano", "jetson-nano-2gb"]:
            profile = get_profile(name)
            if profile:
                vram_gb = profile.vram_bytes / (1024**3)
                print(f"  {name:20} {profile.name:30} {vram_gb:3.0f} GB  {profile.peak_fp16_tflops:6.3f} TF16")

        print("\nConsumer GPUs:")
        for name in ["rtx4090", "rtx4080", "rtx3090", "rtx3080"]:
            profile = get_profile(name)
            if profile:
                print(f"  {name:20} {profile.name:30} {profile.vram_bytes // (1024**3):3} GB  {profile.peak_fp16_tflops:6.1f} TF16")

        print("\nOther:")
        print("  auto                 Auto-detect local GPU/CPU")
        print("  cpu                  Generic CPU profile")

        print("\n" + "-" * 70)
        print("TF16 = Peak FP16 TFLOPS (higher = faster)")
        print("Jetson uses unified memory (shared between CPU and GPU)")
        print("=" * 70 + "\n")
        sys.exit(0)

    # Validate model path
    if args.model_path is None:
        logger.error("Model path is required. Use --list-hardware to see available profiles.")
        sys.exit(1)

    model_path = args.model_path.resolve()
    if not model_path.exists():
        logger.error(f"Model file not found: {model_path}")
        sys.exit(1)

    if not model_path.suffix.lower() in (".onnx", ".pb", ".ort"):
        logger.warning(f"Unexpected file extension: {model_path.suffix}. Proceeding anyway.")

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

    # Run inspection
    try:
        inspector = ModelInspector(logger=logger)
        report = inspector.inspect(model_path)

        # Add hardware estimates if profile specified
        if hardware_profile and report.param_counts and report.flop_counts and report.memory_estimates:
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

    # Output results
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
            logger.warning("matplotlib not installed. Skipping visualizations. Install with: pip install matplotlib")
        else:
            # Determine assets directory
            if args.assets_dir:
                assets_dir = args.assets_dir
            elif args.out_md:
                assets_dir = args.out_md.parent / "assets"
            elif args.out_json:
                assets_dir = args.out_json.parent / "assets"
            else:
                assets_dir = pathlib.Path("assets")

            try:
                viz_gen = VisualizationGenerator(logger=logger)
                viz_paths = viz_gen.generate_all(report, assets_dir)
                logger.info(f"Generated {len(viz_paths)} visualization assets in {assets_dir}")
            except Exception as e:
                logger.warning(f"Failed to generate some visualizations: {e}")
                if args.log_level == "debug":
                    import traceback
                    traceback.print_exc()

    if args.out_md:
        try:
            args.out_md.parent.mkdir(parents=True, exist_ok=True)
            # Generate markdown with or without embedded visualizations
            if viz_paths:
                md_content = _generate_markdown_with_plots(report, viz_paths, args.out_md.parent)
            else:
                md_content = report.to_markdown()
            args.out_md.write_text(md_content, encoding="utf-8")
            logger.info(f"Markdown model card written to: {args.out_md}")
        except Exception as e:
            logger.error(f"Failed to write Markdown report: {e}")
            sys.exit(1)

    # Console output
    if not args.quiet and not args.out_json and not args.out_md:
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
            print(f"Model Size: {report._format_bytes(report.memory_estimates.model_size_bytes)}")

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
                severity_icon = {"info": "[INFO]", "warning": "[WARN]", "high": "[HIGH]"}
                print(f"  {severity_icon.get(risk.severity, '')} {risk.id}")

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


if __name__ == "__main__":
    run_inspect()
