# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
Visualization module for ONNX Autodoc.

Generates matplotlib-based charts for model architecture analysis:
- Operator type histogram
- Layer depth profile (cumulative params/FLOPs)
- Parameter distribution by layer type
- Shape evolution through the network

All charts use a consistent dark theme suitable for technical documentation.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .analyzer import FlopCounts, GraphInfo, ParamCounts
    from .report import InspectionReport

# Attempt to import matplotlib with Agg backend (non-interactive)
_MATPLOTLIB_AVAILABLE = False
try:
    import matplotlib

    matplotlib.use("Agg")  # Must be before importing pyplot
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    _MATPLOTLIB_AVAILABLE = True
except ImportError:
    plt = None  # type: ignore
    Figure = None  # type: ignore


def is_available() -> bool:
    """Check if visualization module is available (matplotlib installed)."""
    return _MATPLOTLIB_AVAILABLE


@dataclass
class ChartTheme:
    """Consistent theme for all charts."""

    # Colors - dark theme with vibrant accents
    background: str = "#1a1a2e"
    text: str = "#eaeaea"
    grid: str = "#2d2d44"
    accent_primary: str = "#00d9ff"  # Cyan
    accent_secondary: str = "#ff6b6b"  # Coral
    accent_tertiary: str = "#4ecdc4"  # Teal
    accent_quaternary: str = "#ffe66d"  # Yellow

    # Color palette for multi-series charts
    palette: tuple[str, ...] = (
        "#00d9ff",  # Cyan
        "#ff6b6b",  # Coral
        "#4ecdc4",  # Teal
        "#ffe66d",  # Yellow
        "#a29bfe",  # Lavender
        "#fd79a8",  # Pink
        "#55efc4",  # Mint
        "#ffeaa7",  # Pale yellow
        "#74b9ff",  # Light blue
        "#ff7675",  # Salmon
    )

    # Typography
    font_family: str = "sans-serif"
    title_size: int = 14
    label_size: int = 11
    tick_size: int = 9

    # Figure
    figure_dpi: int = 150
    figure_width: float = 10.0
    figure_height: float = 6.0


# Default theme instance
THEME = ChartTheme()


def _apply_theme(fig: Figure, ax, title: str) -> None:
    """Apply consistent theme to a matplotlib figure and axes."""
    if not _MATPLOTLIB_AVAILABLE:
        return

    # Figure background
    fig.patch.set_facecolor(THEME.background)

    # Axes styling
    ax.set_facecolor(THEME.background)
    ax.set_title(
        title, color=THEME.text, fontsize=THEME.title_size, fontweight="bold", pad=15
    )

    # Spines
    for spine in ax.spines.values():
        spine.set_color(THEME.grid)
        spine.set_linewidth(0.5)

    # Ticks and labels
    ax.tick_params(colors=THEME.text, labelsize=THEME.tick_size)
    ax.xaxis.label.set_color(THEME.text)
    ax.yaxis.label.set_color(THEME.text)
    ax.xaxis.label.set_fontsize(THEME.label_size)
    ax.yaxis.label.set_fontsize(THEME.label_size)

    # Grid
    ax.grid(True, linestyle="--", alpha=0.3, color=THEME.grid)
    ax.set_axisbelow(True)


def _format_count(n: int) -> str:
    """Format large numbers with K/M/B suffixes."""
    if n >= 1e9:
        return f"{n / 1e9:.1f}B"
    if n >= 1e6:
        return f"{n / 1e6:.1f}M"
    if n >= 1e3:
        return f"{n / 1e3:.1f}K"
    return str(n)


class VisualizationGenerator:
    """
    Generate visualization assets for ONNX model reports.

    Usage:
        viz = VisualizationGenerator()
        paths = viz.generate_all(report, output_dir=Path("assets"))
    """

    def __init__(self, logger: logging.Logger | None = None):
        self.logger = logger or logging.getLogger("autodoc.viz")

    def generate_all(
        self,
        report: InspectionReport,
        output_dir: Path,
    ) -> dict[str, Path]:
        """
        Generate all visualization assets for a report.

        Args:
            report: The inspection report containing analysis results.
            output_dir: Directory to save PNG files.

        Returns:
            Dict mapping chart name to file path, e.g.:
            {"op_histogram": Path("assets/op_histogram.png"), ...}
        """
        if not _MATPLOTLIB_AVAILABLE:
            self.logger.warning("matplotlib not available, skipping visualizations")
            return {}

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        paths: dict[str, Path] = {}

        # Generate each chart type
        try:
            if report.graph_summary and report.graph_summary.op_type_counts:
                path = self.operator_histogram(
                    report.graph_summary.op_type_counts,
                    output_dir / "op_histogram.png",
                )
                if path:
                    paths["op_histogram"] = path
        except Exception as e:
            self.logger.warning(f"Failed to generate operator histogram: {e}")

        try:
            if report.param_counts and report.param_counts.by_op_type:
                path = self.param_distribution(
                    report.param_counts.by_op_type,
                    output_dir / "param_distribution.png",
                )
                if path:
                    paths["param_distribution"] = path
        except Exception as e:
            self.logger.warning(f"Failed to generate param distribution: {e}")

        try:
            if report.flop_counts and report.flop_counts.by_op_type:
                path = self.flops_distribution(
                    report.flop_counts.by_op_type,
                    output_dir / "flops_distribution.png",
                )
                if path:
                    paths["flops_distribution"] = path
        except Exception as e:
            self.logger.warning(f"Failed to generate FLOPs distribution: {e}")

        try:
            if report.param_counts and report.flop_counts:
                path = self.complexity_summary(
                    report,
                    output_dir / "complexity_summary.png",
                )
                if path:
                    paths["complexity_summary"] = path
        except Exception as e:
            self.logger.warning(f"Failed to generate complexity summary: {e}")

        self.logger.info(f"Generated {len(paths)} visualization assets in {output_dir}")
        return paths

    def operator_histogram(
        self,
        op_counts: dict[str, int],
        output_path: Path,
        max_ops: int = 15,
    ) -> Path | None:
        """
        Generate operator type histogram.

        Shows distribution of operator types in the model, sorted by frequency.
        """
        if not _MATPLOTLIB_AVAILABLE or not op_counts:
            return None

        # Sort by count and take top N
        sorted_ops = sorted(op_counts.items(), key=lambda x: -x[1])
        if len(sorted_ops) > max_ops:
            top_ops = sorted_ops[:max_ops]
            other_count = sum(count for _, count in sorted_ops[max_ops:])
            if other_count > 0:
                top_ops.append(("Other", other_count))
        else:
            top_ops = sorted_ops

        labels = [op for op, _ in top_ops]
        counts = [count for _, count in top_ops]

        # Create figure
        fig, ax = plt.subplots(
            figsize=(THEME.figure_width, THEME.figure_height), dpi=THEME.figure_dpi
        )

        # Horizontal bar chart
        y_pos = range(len(labels))
        bars = ax.barh(
            y_pos,
            counts,
            color=THEME.accent_primary,
            edgecolor=THEME.background,
            height=0.7,
        )

        # Labels
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Node Count")
        ax.invert_yaxis()  # Largest at top

        # Add value labels on bars
        for bar, count in zip(bars, counts, strict=False):
            ax.text(
                bar.get_width() + max(counts) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                str(count),
                va="center",
                color=THEME.text,
                fontsize=THEME.tick_size,
            )

        _apply_theme(fig, ax, "Operator Type Distribution")

        # Adjust layout
        plt.tight_layout()
        fig.savefig(
            output_path,
            facecolor=THEME.background,
            edgecolor="none",
            bbox_inches="tight",
        )
        plt.close(fig)

        return output_path

    def param_distribution(
        self,
        params_by_op: dict[str, int],
        output_path: Path,
        max_ops: int = 10,
    ) -> Path | None:
        """
        Generate parameter distribution pie chart.

        Shows how parameters are distributed across operator types.
        """
        if not _MATPLOTLIB_AVAILABLE or not params_by_op:
            return None

        # Filter out zero-param ops and sort
        nonzero_ops = {k: v for k, v in params_by_op.items() if v > 0}
        if not nonzero_ops:
            return None

        sorted_ops = sorted(nonzero_ops.items(), key=lambda x: -x[1])
        if len(sorted_ops) > max_ops:
            top_ops = sorted_ops[:max_ops]
            other_count = sum(count for _, count in sorted_ops[max_ops:])
            if other_count > 0:
                top_ops.append(("Other", other_count))
        else:
            top_ops = sorted_ops

        labels = [f"{op}\n({_format_count(count)})" for op, count in top_ops]
        sizes = [count for _, count in top_ops]
        colors = THEME.palette[: len(sizes)]

        # Create figure
        fig, ax = plt.subplots(
            figsize=(THEME.figure_height, THEME.figure_height), dpi=THEME.figure_dpi
        )

        _wedges, _texts, autotexts = ax.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct=lambda pct: f"{pct:.1f}%" if pct > 5 else "",
            startangle=90,
            textprops={"color": THEME.text, "fontsize": THEME.tick_size},
        )

        # Style the percentage text
        for autotext in autotexts:
            autotext.set_color(THEME.background)
            autotext.set_fontweight("bold")

        ax.set_title(
            "Parameter Distribution by Operator Type",
            color=THEME.text,
            fontsize=THEME.title_size,
            fontweight="bold",
            pad=20,
        )

        fig.patch.set_facecolor(THEME.background)

        plt.tight_layout()
        fig.savefig(
            output_path,
            facecolor=THEME.background,
            edgecolor="none",
            bbox_inches="tight",
        )
        plt.close(fig)

        return output_path

    def flops_distribution(
        self,
        flops_by_op: dict[str, int],
        output_path: Path,
        max_ops: int = 10,
    ) -> Path | None:
        """
        Generate FLOPs distribution bar chart.

        Shows computational cost distribution across operator types.
        """
        if not _MATPLOTLIB_AVAILABLE or not flops_by_op:
            return None

        # Filter out zero-FLOP ops and sort
        nonzero_ops = {k: v for k, v in flops_by_op.items() if v > 0}
        if not nonzero_ops:
            return None

        sorted_ops = sorted(nonzero_ops.items(), key=lambda x: -x[1])[:max_ops]

        labels = [op for op, _ in sorted_ops]
        values = [flops for _, flops in sorted_ops]

        # Create figure
        fig, ax = plt.subplots(
            figsize=(THEME.figure_width, THEME.figure_height), dpi=THEME.figure_dpi
        )

        x_pos = range(len(labels))
        bars = ax.bar(
            x_pos,
            values,
            color=THEME.accent_secondary,
            edgecolor=THEME.background,
            width=0.7,
        )

        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel("FLOPs")

        # Add value labels on bars
        for bar, value in zip(bars, values, strict=False):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(values) * 0.02,
                _format_count(value),
                ha="center",
                va="bottom",
                color=THEME.text,
                fontsize=THEME.tick_size,
            )

        _apply_theme(fig, ax, "FLOPs Distribution by Operator Type")

        plt.tight_layout()
        fig.savefig(
            output_path,
            facecolor=THEME.background,
            edgecolor="none",
            bbox_inches="tight",
        )
        plt.close(fig)

        return output_path

    def complexity_summary(
        self,
        report: InspectionReport,
        output_path: Path,
    ) -> Path | None:
        """
        Generate complexity summary dashboard.

        Multi-panel figure showing key metrics at a glance.
        """
        if not _MATPLOTLIB_AVAILABLE:
            return None
        if (
            not report.param_counts
            or not report.flop_counts
            or not report.memory_estimates
        ):
            return None

        fig, axes = plt.subplots(
            1, 3, figsize=(THEME.figure_width, 4), dpi=THEME.figure_dpi
        )
        fig.patch.set_facecolor(THEME.background)

        metrics = [
            ("Parameters", report.param_counts.total, THEME.accent_primary),
            ("FLOPs", report.flop_counts.total, THEME.accent_secondary),
            (
                "Memory (bytes)",
                report.memory_estimates.model_size_bytes,
                THEME.accent_tertiary,
            ),
        ]

        for ax, (label, value, color) in zip(axes, metrics, strict=False):
            ax.set_facecolor(THEME.background)

            # Large centered value
            ax.text(
                0.5,
                0.6,
                _format_count(value),
                ha="center",
                va="center",
                fontsize=28,
                fontweight="bold",
                color=color,
                transform=ax.transAxes,
            )

            # Label below
            ax.text(
                0.5,
                0.25,
                label,
                ha="center",
                va="center",
                fontsize=THEME.label_size,
                color=THEME.text,
                transform=ax.transAxes,
            )

            # Remove all axes elements
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

        # Main title
        fig.suptitle(
            "Model Complexity Summary",
            color=THEME.text,
            fontsize=THEME.title_size,
            fontweight="bold",
            y=0.95,
        )

        plt.tight_layout(rect=[0, 0, 1, 0.9])
        fig.savefig(
            output_path,
            facecolor=THEME.background,
            edgecolor="none",
            bbox_inches="tight",
        )
        plt.close(fig)

        return output_path

    def layer_depth_profile(
        self,
        graph_info: GraphInfo,
        param_counts: ParamCounts,
        flop_counts: FlopCounts,
        output_path: Path,
    ) -> Path | None:
        """
        Generate layer depth profile showing cumulative params/FLOPs.

        Shows how complexity accumulates through the network depth.
        """
        if not _MATPLOTLIB_AVAILABLE:
            return None

        # Get ordered nodes with their metrics
        nodes_with_metrics = []
        for node in graph_info.nodes:
            params = param_counts.by_node.get(node.name, 0)
            flops = flop_counts.by_node.get(node.name, 0)
            if params > 0 or flops > 0:
                nodes_with_metrics.append((node.name, params, flops))

        if not nodes_with_metrics:
            return None

        # Compute cumulative values
        cum_params = []
        cum_flops = []
        running_params = 0
        running_flops = 0
        for _, params, flops in nodes_with_metrics:
            running_params += params
            running_flops += flops
            cum_params.append(running_params)
            cum_flops.append(running_flops)

        x = range(len(nodes_with_metrics))

        # Create figure with two y-axes
        fig, ax1 = plt.subplots(
            figsize=(THEME.figure_width, THEME.figure_height), dpi=THEME.figure_dpi
        )
        ax2 = ax1.twinx()

        # Plot cumulative params
        ax1.fill_between(x, cum_params, alpha=0.3, color=THEME.accent_primary)
        ax1.plot(
            x, cum_params, color=THEME.accent_primary, linewidth=2, label="Parameters"
        )
        ax1.set_ylabel("Cumulative Parameters", color=THEME.accent_primary)
        ax1.tick_params(axis="y", labelcolor=THEME.accent_primary)

        # Plot cumulative FLOPs
        ax2.fill_between(x, cum_flops, alpha=0.3, color=THEME.accent_secondary)
        ax2.plot(x, cum_flops, color=THEME.accent_secondary, linewidth=2, label="FLOPs")
        ax2.set_ylabel("Cumulative FLOPs", color=THEME.accent_secondary)
        ax2.tick_params(axis="y", labelcolor=THEME.accent_secondary)

        ax1.set_xlabel("Layer Index")

        # Apply theme to primary axis
        fig.patch.set_facecolor(THEME.background)
        ax1.set_facecolor(THEME.background)
        ax1.set_title(
            "Layer Depth Profile",
            color=THEME.text,
            fontsize=THEME.title_size,
            fontweight="bold",
            pad=15,
        )

        for spine in ax1.spines.values():
            spine.set_color(THEME.grid)
        for spine in ax2.spines.values():
            spine.set_color(THEME.grid)

        ax1.tick_params(colors=THEME.text, labelsize=THEME.tick_size)
        ax1.xaxis.label.set_color(THEME.text)
        ax1.xaxis.label.set_fontsize(THEME.label_size)

        ax1.grid(True, linestyle="--", alpha=0.3, color=THEME.grid)

        # Legend
        lines = [
            plt.Line2D([0], [0], color=THEME.accent_primary, linewidth=2),
            plt.Line2D([0], [0], color=THEME.accent_secondary, linewidth=2),
        ]
        ax1.legend(
            lines,
            ["Parameters", "FLOPs"],
            loc="upper left",
            facecolor=THEME.background,
            labelcolor=THEME.text,
        )

        plt.tight_layout()
        fig.savefig(
            output_path,
            facecolor=THEME.background,
            edgecolor="none",
            bbox_inches="tight",
        )
        plt.close(fig)

        return output_path


def generate_visualizations(
    report: InspectionReport,
    output_dir: Path | str,
    logger: logging.Logger | None = None,
) -> dict[str, Path]:
    """
    Convenience function to generate all visualizations for a report.

    Args:
        report: The inspection report.
        output_dir: Directory to save PNG files.
        logger: Optional logger.

    Returns:
        Dict mapping chart name to file path.
    """
    generator = VisualizationGenerator(logger=logger)
    return generator.generate_all(report, Path(output_dir))
