# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
Operational profiling and system requirements analysis.

This module implements:
- Batch size scalability analysis (sweeps)
- System requirements generation (Steam-style min/rec/optimal)
- Resolution impact analysis (future)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from .hardware import (
    HARDWARE_PROFILES,
    HardwareEstimates,
    HardwareEstimator,
    HardwareProfile,
)


@dataclass
class BatchSweepPoint:
    """Metrics for a single batch size point."""

    batch_size: int
    vram_required_bytes: int
    estimated_latency_ms: float
    throughput_fps: float
    compute_utilization: float
    bottleneck: str
    fits_in_vram: bool


@dataclass
class BatchSizeSweep:
    """Results of a batch size sweep analysis."""

    batch_sizes: list[int]
    latencies: list[float]
    throughputs: list[float]
    vram_usage_gb: list[float]
    optimal_batch_size: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_sizes": self.batch_sizes,
            "latencies": self.latencies,
            "throughputs": self.throughputs,
            "vram_usage_gb": self.vram_usage_gb,
            "optimal_batch_size": self.optimal_batch_size,
        }


@dataclass
class SystemRequirements:
    """Recommended hardware tiers for deployment.

    This is a lightweight, report-friendly wrapper around :class:`HardwareEstimates`.
    It deliberately mirrors the older `SystemRequirements` helper in `hardware.py`,
    exposing `minimum_gpu`, `recommended_gpu`, and `optimal_gpu` style attributes so
    existing report/HTML code (and mental model) continue to work.
    """

    # Core estimates for each tier
    minimum: HardwareEstimates | None  # The lowest spec that runs it
    recommended: HardwareEstimates | None  # Good balance of cost/perf
    optimal: HardwareEstimates | None  # Maximum performance

    def to_dict(self) -> dict[str, Any]:
        return {
            "minimum": self.minimum.to_dict() if self.minimum else None,
            "recommended": self.recommended.to_dict() if self.recommended else None,
            "optimal": self.optimal.to_dict() if self.optimal else None,
        }

    # Backwards/HTML-friendly convenience properties ---------------------
    #
    # These keep the `reqs.minimum_gpu.name` / `reqs.minimum_vram_gb` style
    # access patterns working in `report.py` and HTML templates without
    # duplicating all the shape logic here.

    @property
    def minimum_gpu(self) -> HardwareEstimates | None:
        return self.minimum

    @property
    def recommended_gpu(self) -> HardwareEstimates | None:
        return self.recommended

    @property
    def optimal_gpu(self) -> HardwareEstimates | None:
        return self.optimal

    @staticmethod
    def _vram_gb(est: HardwareEstimates | None) -> float | None:
        if not est:
            return None
        return round(est.vram_required_bytes / (1024**3), 2)

    @property
    def minimum_vram_gb(self) -> float | None:
        return self._vram_gb(self.minimum)

    @property
    def recommended_vram_gb(self) -> float | None:
        return self._vram_gb(self.recommended)


class OperationalProfiler:
    """
    Analyzes model operational characteristics.
    """

    def __init__(self, logger: logging.Logger | None = None):
        self.logger = logger or logging.getLogger("autodoc.profiler")
        self.hw_estimator = HardwareEstimator(logger=self.logger)

    def run_batch_sweep(
        self,
        model_params: int,
        model_flops: int,
        peak_activation_bytes: int,
        hardware: HardwareProfile,
        batch_sizes: list[int] | None = None,
        precision: str = "fp16",
    ) -> BatchSizeSweep:
        """
        Analyze performance scaling across batch sizes.

        Args:
            model_params: Total parameters
            model_flops: FLOPs per inference (batch=1)
            peak_activation_bytes: Peak activation memory (batch=1)
            hardware: Target hardware profile
            batch_sizes: List of batch sizes to test (default: powers of 2)
            precision: Precision to simulate ("fp32", "fp16", "int8")

        Returns:
            BatchSizeSweep results
        """
        if batch_sizes is None:
            batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]

        latencies = []
        throughputs = []
        vram_usage = []
        optimal_bs = 1
        max_throughput = 0.0

        for bs in batch_sizes:
            est = self.hw_estimator.estimate(
                model_params=model_params,
                model_flops=model_flops,
                peak_activation_bytes=peak_activation_bytes,
                hardware=hardware,
                batch_size=bs,
                precision=precision,
            )

            # Calculate throughput (inferences per second)
            # If latency is infinite (OOM), throughput is 0
            throughput = 0.0
            latency = float("inf")
            vram_gb = est.vram_required_bytes / (1024**3)

            if est.theoretical_latency_ms > 0 and est.fits_in_vram:
                latency = est.theoretical_latency_ms
                throughput = (1000.0 / latency) * bs

                if throughput > max_throughput:
                    max_throughput = throughput
                    optimal_bs = bs

            latencies.append(latency)
            throughputs.append(throughput)
            vram_usage.append(vram_gb)

        return BatchSizeSweep(
            batch_sizes=batch_sizes,
            latencies=latencies,
            throughputs=throughputs,
            vram_usage_gb=vram_usage,
            optimal_batch_size=optimal_bs,
        )

    def determine_system_requirements(
        self,
        model_params: int,
        model_flops: int,
        peak_activation_bytes: int,
        precision: str = "fp16",
        target_fps: float = 30.0,  # For "Recommended" tier
    ) -> SystemRequirements:
        """
        Find suitable hardware tiers ("Steam-style" requirements).

        Strategy:
        - Minimum: Cheapest hardware that fits the model in VRAM (Batch=1)
        - Recommended: Cheapest hardware that hits target_fps (Batch=1) OR fits with good utilization
        - Optimal: Hardware providing highest throughput/lowest latency
        """
        candidates = []

        # Evaluate against all known profiles
        # Filter out mobile/multi-gpu for cleaner list, or keep them?
        # Let's keep single-GPU desktops/servers for simplicity of recommendation
        for name, profile in HARDWARE_PROFILES.items():
            # Skip generic CPU for this analysis unless it's the only option
            if profile.device_type == "cpu":
                continue

            # Skip mobile variants to keep list clean (optional)
            if "mobile" in name:
                continue

            est = self.hw_estimator.estimate(
                model_params=model_params,
                model_flops=model_flops,
                peak_activation_bytes=peak_activation_bytes,
                hardware=profile,
                batch_size=1,
                precision=precision,
            )
            candidates.append((profile, est))

        if not candidates:
            return SystemRequirements(None, None, None)

        # --- Find Minimum ---
        # Sort by VRAM (ascending), then FLOPs (ascending)
        candidates.sort(key=lambda x: (x[0].vram_bytes, x[0].peak_fp16_tflops))

        minimum = None
        for _, est in candidates:
            if est.fits_in_vram:
                minimum = est
                break

        # --- Find Optimal ---
        # Sort by Latency (ascending)
        candidates.sort(key=lambda x: x[1].theoretical_latency_ms)

        optimal = None
        # Filter for ones that fit
        valid_candidates = [x for x in candidates if x[1].fits_in_vram]
        if valid_candidates:
            optimal = valid_candidates[0][1]  # Fastest

        # --- Find Recommended ---
        # Heuristic: Fits VRAM AND (Latency <= 1000/target_fps OR Utilization > 0.5)
        # We want something reasonable, not necessarily the fastest (which is often H100)
        # Let's look for the "cheapest" card that meets a performance bar.

        recommended = None

        # Re-sort by cost proxy (we don't have prices in HardwareProfile, but TFLOPS is a rough proxy)
        valid_candidates.sort(key=lambda x: x[0].peak_fp16_tflops)

        target_latency_ms = 1000.0 / target_fps

        for _, est in valid_candidates:
            if est.theoretical_latency_ms <= target_latency_ms:
                recommended = est
                break

        # If nothing meets strict FPS target, pick the one with decent utilization
        if recommended is None and valid_candidates:
            # Pick median performer? Or just fallback to Minimum if nothing is fast enough?
            # Let's pick the one that is ~4x faster than minimum if possible, or just minimum
            minimum_latency = (
                minimum.theoretical_latency_ms if minimum else float("inf")
            )
            for _, est in valid_candidates:
                if est.theoretical_latency_ms <= minimum_latency / 4.0:
                    recommended = est
                    break

        if recommended is None:
            recommended = minimum  # Fallback

        return SystemRequirements(
            minimum=minimum, recommended=recommended, optimal=optimal
        )
