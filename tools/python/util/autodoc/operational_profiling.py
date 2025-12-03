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
class ResolutionPoint:
    """Metrics for a single resolution point."""

    resolution: tuple[int, int]
    resolution_str: str  # e.g., "224x224"
    flops: int
    memory_bytes: int
    vram_required_bytes: int
    estimated_latency_ms: float
    throughput_fps: float
    fits_in_vram: bool


@dataclass
class ResolutionSweep:
    """Results of a resolution sweep analysis."""

    resolutions: list[str]  # ["224x224", "384x384", ...]
    flops: list[int]
    memory_gb: list[float]
    latencies: list[float]
    throughputs: list[float]
    vram_usage_gb: list[float]
    optimal_resolution: str
    max_resolution: str  # Largest resolution that fits in VRAM

    def to_dict(self) -> dict[str, Any]:
        return {
            "resolutions": self.resolutions,
            "flops": self.flops,
            "memory_gb": self.memory_gb,
            "latencies": self.latencies,
            "throughputs": self.throughputs,
            "vram_usage_gb": self.vram_usage_gb,
            "optimal_resolution": self.optimal_resolution,
            "max_resolution": self.max_resolution,
        }


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

    def run_batch_sweep_benchmark(
        self,
        model_path: str,
        batch_sizes: list[int] | None = None,
        num_warmup: int = 5,
        num_runs: int = 20,
    ) -> BatchSizeSweep | None:
        """
        Benchmark actual inference performance across batch sizes.

        Uses ONNX Runtime to measure real latency and throughput.
        Requires onnxruntime to be installed.

        Args:
            model_path: Path to ONNX model file
            batch_sizes: List of batch sizes to test (default: powers of 2)
            num_warmup: Number of warmup runs before timing
            num_runs: Number of timed runs per batch size

        Returns:
            BatchSizeSweep with measured (not estimated) metrics
        """
        try:
            import numpy as np

            import onnxruntime as ort
        except ImportError:
            self.logger.warning("onnxruntime not available, falling back to estimates")
            return None

        if batch_sizes is None:
            batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]

        # Create session
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        try:
            sess = ort.InferenceSession(model_path, providers=providers)
        except Exception as e:
            self.logger.error(f"Failed to load model for benchmarking: {e}")
            return None

        active_provider = sess.get_providers()[0]
        self.logger.info(f"Benchmarking with {active_provider}")

        # Get input info
        input_info = sess.get_inputs()[0]
        input_name = input_info.name
        input_shape = list(input_info.shape)

        # Replace dynamic dimensions with defaults
        for i, dim in enumerate(input_shape):
            if not isinstance(dim, int) or dim <= 0:
                if i == 0:
                    input_shape[i] = 1  # Batch dim, will be replaced
                elif i == 1:
                    input_shape[i] = 3  # Channels
                else:
                    input_shape[i] = 224  # Spatial dims

        latencies = []
        throughputs = []
        vram_usage = []
        optimal_bs = 1
        max_throughput = 0.0

        for bs in batch_sizes:
            # Create input with current batch size
            input_shape[0] = bs
            try:
                dummy_input = np.random.randn(*input_shape).astype(np.float32)
            except Exception as e:
                self.logger.warning(f"Failed to create input for batch {bs}: {e}")
                latencies.append(float("inf"))
                throughputs.append(0.0)
                vram_usage.append(0.0)
                continue

            # Warmup
            try:
                for _ in range(num_warmup):
                    sess.run(None, {input_name: dummy_input})
            except Exception as e:
                self.logger.warning(f"Batch {bs} failed (OOM?): {e}")
                latencies.append(float("inf"))
                throughputs.append(0.0)
                vram_usage.append(0.0)
                continue

            # Benchmark
            import time

            run_latencies = []
            for _ in range(num_runs):
                start = time.perf_counter()
                sess.run(None, {input_name: dummy_input})
                end = time.perf_counter()
                run_latencies.append((end - start) * 1000)  # ms

            # Use median latency (more stable than mean)
            run_latencies.sort()
            p50_latency = run_latencies[len(run_latencies) // 2]
            throughput = (bs * 1000.0) / p50_latency

            latencies.append(round(p50_latency, 2))
            throughputs.append(round(throughput, 1))
            # VRAM: estimate from input size (actual measurement requires pynvml)
            vram_gb = (dummy_input.nbytes * 2) / (1024**3)  # Input + output approx
            vram_usage.append(round(vram_gb, 3))

            if throughput > max_throughput:
                max_throughput = throughput
                optimal_bs = bs

            self.logger.info(
                f"  Batch {bs}: latency={p50_latency:.2f}ms, "
                f"throughput={throughput:.1f} inf/s"
            )

        return BatchSizeSweep(
            batch_sizes=batch_sizes,
            latencies=latencies,
            throughputs=throughputs,
            vram_usage_gb=vram_usage,
            optimal_batch_size=optimal_bs,
        )

    def run_resolution_sweep(
        self,
        base_flops: int,
        base_activation_bytes: int,
        base_resolution: tuple[int, int],
        model_params: int,
        hardware: HardwareProfile,
        resolutions: list[tuple[int, int]] | None = None,
        batch_size: int = 1,
        precision: str = "fp16",
    ) -> ResolutionSweep:
        """
        Analyze performance scaling across input resolutions.

        For vision models, FLOPs and memory scale approximately quadratically
        with resolution (for most architectures like ResNet, ViT, YOLO).

        Args:
            base_flops: FLOPs at base_resolution
            base_activation_bytes: Activation memory at base_resolution
            base_resolution: The resolution used for base measurements (H, W)
            model_params: Total parameters (doesn't change with resolution)
            hardware: Target hardware profile
            resolutions: List of (H, W) resolutions to test
            batch_size: Batch size for estimates
            precision: Precision ("fp32", "fp16", "int8")

        Returns:
            ResolutionSweep results
        """
        base_h, base_w = base_resolution
        base_pixels = base_h * base_w
        base_aspect = base_w / base_h if base_h > 0 else 1.0

        if resolutions is None:
            # Generate resolutions that:
            # 1. Match the aspect ratio of training data
            # 2. Only go UP TO (not above) the training resolution
            # Running above training resolution typically produces poor results
            resolutions = []

            # Common scale factors (smaller than or equal to 1.0)
            if base_aspect == 1.0:
                # Square aspect ratio
                candidates = [
                    128,
                    160,
                    192,
                    224,
                    256,
                    320,
                    384,
                    416,
                    448,
                    512,
                    640,
                    768,
                    1024,
                ]
                for size in candidates:
                    if size <= base_h:
                        resolutions.append((size, size))
            else:
                # Non-square: generate resolutions matching aspect ratio
                scale_factors = [0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]
                for scale in scale_factors:
                    h = int(base_h * scale)
                    w = int(base_w * scale)
                    # Round to nearest 32 for GPU efficiency
                    h = max(32, (h // 32) * 32)
                    w = max(32, (w // 32) * 32)
                    if h <= base_h and w <= base_w and (h, w) not in resolutions:
                        resolutions.append((h, w))

            # Always include the base resolution
            if base_resolution not in resolutions:
                resolutions.append(base_resolution)

            # Sort by pixel count
            resolutions.sort(key=lambda r: r[0] * r[1])

        resolution_strs = []
        flops_list = []
        memory_gb_list = []
        latencies = []
        throughputs = []
        vram_usage = []
        optimal_res = f"{base_h}x{base_w}"
        max_res = f"{base_h}x{base_w}"
        max_throughput = 0.0
        max_fitting_pixels = 0

        for h, w in resolutions:
            res_str = f"{h}x{w}"
            resolution_strs.append(res_str)

            # Scale FLOPs and memory quadratically with resolution
            pixels = h * w
            scale_factor = pixels / base_pixels

            scaled_flops = int(base_flops * scale_factor)
            scaled_activation = int(base_activation_bytes * scale_factor)

            flops_list.append(scaled_flops)
            memory_gb_list.append(scaled_activation / (1024**3))

            # Get hardware estimates for this resolution
            est = self.hw_estimator.estimate(
                model_params=model_params,
                model_flops=scaled_flops,
                peak_activation_bytes=scaled_activation,
                hardware=hardware,
                batch_size=batch_size,
                precision=precision,
            )

            vram_gb = est.vram_required_bytes / (1024**3)
            vram_usage.append(vram_gb)

            if est.fits_in_vram and est.theoretical_latency_ms > 0:
                latency = est.theoretical_latency_ms
                throughput = (1000.0 / latency) * batch_size

                latencies.append(latency)
                throughputs.append(throughput)

                # Track max resolution that fits
                if pixels > max_fitting_pixels:
                    max_fitting_pixels = pixels
                    max_res = res_str

                # Track optimal (highest throughput)
                if throughput > max_throughput:
                    max_throughput = throughput
                    optimal_res = res_str
            else:
                latencies.append(float("inf"))
                throughputs.append(0.0)

        return ResolutionSweep(
            resolutions=resolution_strs,
            flops=flops_list,
            memory_gb=memory_gb_list,
            latencies=latencies,
            throughputs=throughputs,
            vram_usage_gb=vram_usage,
            optimal_resolution=optimal_res,
            max_resolution=max_res,
        )

    def recommend_resolution(
        self,
        base_flops: int,
        base_activation_bytes: int,
        base_resolution: tuple[int, int],
        model_params: int,
        hardware: HardwareProfile,
        target_fps: float = 30.0,
        batch_size: int = 1,
        precision: str = "fp16",
    ) -> dict[str, Any]:
        """
        Recommend optimal resolution for target hardware and latency requirements.

        Task 6.8.5: Resolution recommendations for target hardware

        Args:
            base_flops: FLOPs at base_resolution
            base_activation_bytes: Activation memory at base_resolution
            base_resolution: The resolution used for base measurements (H, W)
            model_params: Total parameters
            hardware: Target hardware profile
            target_fps: Desired frames per second (default: 30 fps)
            batch_size: Batch size
            precision: Precision for estimates

        Returns:
            Dict with recommended_resolution, max_resolution, and rationale
        """
        target_latency_ms = 1000.0 / target_fps

        # Run sweep with common resolutions
        sweep = self.run_resolution_sweep(
            base_flops=base_flops,
            base_activation_bytes=base_activation_bytes,
            base_resolution=base_resolution,
            model_params=model_params,
            hardware=hardware,
            batch_size=batch_size,
            precision=precision,
        )

        # Find resolution that meets target FPS
        recommended = None
        recommended_idx = -1
        for i, (res, lat) in enumerate(
            zip(sweep.resolutions, sweep.latencies, strict=False)
        ):
            if lat != float("inf") and lat <= target_latency_ms:
                recommended = res
                recommended_idx = i

        # Build recommendation rationale
        rationale_parts = []

        if recommended:
            rationale_parts.append(
                f"Resolution **{recommended}** meets {target_fps} FPS target "
                f"({sweep.latencies[recommended_idx]:.1f}ms latency)."
            )
        else:
            # Find closest resolution that fits
            for i, (res, lat) in enumerate(
                zip(sweep.resolutions, sweep.latencies, strict=False)
            ):
                if lat != float("inf"):
                    recommended = res
                    recommended_idx = i
                    break

            if recommended:
                actual_fps = 1000.0 / sweep.latencies[recommended_idx]
                rationale_parts.append(
                    f"Cannot meet {target_fps} FPS. Best achievable: "
                    f"**{recommended}** at {actual_fps:.1f} FPS."
                )
            else:
                rationale_parts.append("No resolution fits in available VRAM.")

        if sweep.max_resolution and sweep.max_resolution != recommended:
            rationale_parts.append(
                f"Maximum resolution that fits in VRAM: **{sweep.max_resolution}**."
            )

        return {
            "recommended_resolution": recommended,
            "max_resolution": sweep.max_resolution,
            "optimal_resolution": sweep.optimal_resolution,
            "target_fps": target_fps,
            "achievable_fps": (
                1000.0 / sweep.latencies[recommended_idx]
                if recommended and recommended_idx >= 0
                else 0.0
            ),
            "rationale": " ".join(rationale_parts),
            "sweep_results": sweep.to_dict(),
        }

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
