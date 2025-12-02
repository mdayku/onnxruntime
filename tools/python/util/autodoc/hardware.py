# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
Hardware detection and profile management for ONNX Autodoc.

This module provides:
- Automatic detection of local GPU/CPU hardware
- Predefined profiles for common NVIDIA GPUs
- Hardware-aware performance estimates
"""
from __future__ import annotations

import logging
import os
import platform
import subprocess
from dataclasses import dataclass
from typing import Any

# Try to import psutil for CPU info, but don't require it
try:
    import psutil

    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


@dataclass
class HardwareProfile:
    """
    Hardware specification for performance estimates.

    All values are theoretical peaks - actual performance will vary
    based on memory access patterns, kernel efficiency, etc.
    """

    name: str
    vendor: str  # "nvidia", "amd", "apple", "intel", "generic"
    device_type: str  # "gpu", "cpu", "npu"

    # Memory
    vram_bytes: int  # GPU VRAM or system RAM for CPU
    memory_bandwidth_bytes_per_s: int

    # Compute (theoretical peaks)
    peak_fp32_tflops: float
    peak_fp16_tflops: float
    peak_int8_tops: float  # Tera-ops for INT8

    # Optional metadata
    compute_capability: str = ""  # e.g., "8.9" for Ada Lovelace
    tdp_watts: int = 0
    is_detected: bool = False  # True if auto-detected from local hardware

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "vendor": self.vendor,
            "device_type": self.device_type,
            "vram_gb": round(self.vram_bytes / (1024**3), 1),
            "memory_bandwidth_gb_s": round(
                self.memory_bandwidth_bytes_per_s / (1024**3), 1
            ),
            "peak_fp32_tflops": self.peak_fp32_tflops,
            "peak_fp16_tflops": self.peak_fp16_tflops,
            "peak_int8_tops": self.peak_int8_tops,
            "compute_capability": self.compute_capability,
            "tdp_watts": self.tdp_watts,
            "is_detected": self.is_detected,
        }


# ============================================================================
# Predefined Hardware Profiles
# ============================================================================

# NVIDIA Data Center GPUs
NVIDIA_H100_SXM = HardwareProfile(
    name="NVIDIA H100 SXM",
    vendor="nvidia",
    device_type="gpu",
    vram_bytes=80 * (1024**3),  # 80 GB HBM3
    memory_bandwidth_bytes_per_s=3350 * (1024**3),  # 3.35 TB/s
    peak_fp32_tflops=67.0,
    peak_fp16_tflops=1979.0,  # With sparsity: 3958
    peak_int8_tops=3958.0,
    compute_capability="9.0",
    tdp_watts=700,
)

NVIDIA_A100_80GB = HardwareProfile(
    name="NVIDIA A100 80GB",
    vendor="nvidia",
    device_type="gpu",
    vram_bytes=80 * (1024**3),  # 80 GB HBM2e
    memory_bandwidth_bytes_per_s=2039 * (1024**3),  # 2.0 TB/s
    peak_fp32_tflops=19.5,
    peak_fp16_tflops=312.0,  # Tensor Core
    peak_int8_tops=624.0,
    compute_capability="8.0",
    tdp_watts=400,
)

NVIDIA_A100_40GB = HardwareProfile(
    name="NVIDIA A100 40GB",
    vendor="nvidia",
    device_type="gpu",
    vram_bytes=40 * (1024**3),
    memory_bandwidth_bytes_per_s=1555 * (1024**3),  # 1.6 TB/s
    peak_fp32_tflops=19.5,
    peak_fp16_tflops=312.0,
    peak_int8_tops=624.0,
    compute_capability="8.0",
    tdp_watts=400,
)

NVIDIA_A10 = HardwareProfile(
    name="NVIDIA A10",
    vendor="nvidia",
    device_type="gpu",
    vram_bytes=24 * (1024**3),  # 24 GB GDDR6
    memory_bandwidth_bytes_per_s=600 * (1024**3),  # 600 GB/s
    peak_fp32_tflops=31.2,
    peak_fp16_tflops=125.0,  # Tensor Core
    peak_int8_tops=250.0,
    compute_capability="8.6",
    tdp_watts=150,
)

NVIDIA_T4 = HardwareProfile(
    name="NVIDIA T4",
    vendor="nvidia",
    device_type="gpu",
    vram_bytes=16 * (1024**3),  # 16 GB GDDR6
    memory_bandwidth_bytes_per_s=320 * (1024**3),  # 320 GB/s
    peak_fp32_tflops=8.1,
    peak_fp16_tflops=65.0,  # Tensor Core
    peak_int8_tops=130.0,
    compute_capability="7.5",
    tdp_watts=70,
)

NVIDIA_L4 = HardwareProfile(
    name="NVIDIA L4",
    vendor="nvidia",
    device_type="gpu",
    vram_bytes=24 * (1024**3),  # 24 GB GDDR6
    memory_bandwidth_bytes_per_s=300 * (1024**3),  # 300 GB/s
    peak_fp32_tflops=30.3,
    peak_fp16_tflops=121.0,  # Tensor Core
    peak_int8_tops=242.0,
    compute_capability="8.9",
    tdp_watts=72,
)

NVIDIA_L40 = HardwareProfile(
    name="NVIDIA L40",
    vendor="nvidia",
    device_type="gpu",
    vram_bytes=48 * (1024**3),  # 48 GB GDDR6
    memory_bandwidth_bytes_per_s=864 * (1024**3),  # 864 GB/s
    peak_fp32_tflops=90.5,
    peak_fp16_tflops=181.0,
    peak_int8_tops=362.0,
    compute_capability="8.9",
    tdp_watts=300,
)

NVIDIA_L40S = HardwareProfile(
    name="NVIDIA L40S",
    vendor="nvidia",
    device_type="gpu",
    vram_bytes=48 * (1024**3),  # 48 GB GDDR6
    memory_bandwidth_bytes_per_s=864 * (1024**3),  # 864 GB/s
    peak_fp32_tflops=91.6,
    peak_fp16_tflops=183.0,
    peak_int8_tops=733.0,  # Enhanced INT8
    compute_capability="8.9",
    tdp_watts=350,
)

# Older but still common datacenter GPUs
NVIDIA_V100_32GB = HardwareProfile(
    name="NVIDIA V100 32GB",
    vendor="nvidia",
    device_type="gpu",
    vram_bytes=32 * (1024**3),  # 32 GB HBM2
    memory_bandwidth_bytes_per_s=900 * (1024**3),  # 900 GB/s
    peak_fp32_tflops=14.0,
    peak_fp16_tflops=112.0,  # Tensor Core
    peak_int8_tops=0.0,  # No INT8 tensor cores
    compute_capability="7.0",
    tdp_watts=300,
)

NVIDIA_V100_16GB = HardwareProfile(
    name="NVIDIA V100 16GB",
    vendor="nvidia",
    device_type="gpu",
    vram_bytes=16 * (1024**3),  # 16 GB HBM2
    memory_bandwidth_bytes_per_s=900 * (1024**3),  # 900 GB/s
    peak_fp32_tflops=14.0,
    peak_fp16_tflops=112.0,
    peak_int8_tops=0.0,
    compute_capability="7.0",
    tdp_watts=300,
)

NVIDIA_P100 = HardwareProfile(
    name="NVIDIA P100",
    vendor="nvidia",
    device_type="gpu",
    vram_bytes=16 * (1024**3),  # 16 GB HBM2
    memory_bandwidth_bytes_per_s=732 * (1024**3),  # 732 GB/s
    peak_fp32_tflops=9.3,
    peak_fp16_tflops=18.7,
    peak_int8_tops=0.0,
    compute_capability="6.0",
    tdp_watts=250,
)

NVIDIA_P40 = HardwareProfile(
    name="NVIDIA P40",
    vendor="nvidia",
    device_type="gpu",
    vram_bytes=24 * (1024**3),  # 24 GB GDDR5X
    memory_bandwidth_bytes_per_s=346 * (1024**3),  # 346 GB/s
    peak_fp32_tflops=12.0,
    peak_fp16_tflops=0.0,  # No FP16 tensor cores
    peak_int8_tops=47.0,
    compute_capability="6.1",
    tdp_watts=250,
)

# ============================================================================
# NVIDIA Jetson Series (Edge/Embedded)
# ============================================================================

# Jetson Orin Series (2022+)
NVIDIA_JETSON_AGX_ORIN_64GB = HardwareProfile(
    name="NVIDIA Jetson AGX Orin 64GB",
    vendor="nvidia",
    device_type="gpu",
    vram_bytes=64 * (1024**3),  # 64 GB unified memory
    memory_bandwidth_bytes_per_s=204 * (1024**3),  # 204 GB/s
    peak_fp32_tflops=5.3,
    peak_fp16_tflops=10.6,  # Sparse: 21.2
    peak_int8_tops=275.0,  # Sparse
    compute_capability="8.7",
    tdp_watts=60,  # 15W-60W configurable
)

NVIDIA_JETSON_AGX_ORIN_32GB = HardwareProfile(
    name="NVIDIA Jetson AGX Orin 32GB",
    vendor="nvidia",
    device_type="gpu",
    vram_bytes=32 * (1024**3),  # 32 GB unified memory
    memory_bandwidth_bytes_per_s=204 * (1024**3),  # 204 GB/s
    peak_fp32_tflops=5.3,
    peak_fp16_tflops=10.6,
    peak_int8_tops=275.0,
    compute_capability="8.7",
    tdp_watts=60,
)

NVIDIA_JETSON_ORIN_NX_16GB = HardwareProfile(
    name="NVIDIA Jetson Orin NX 16GB",
    vendor="nvidia",
    device_type="gpu",
    vram_bytes=16 * (1024**3),  # 16 GB unified memory
    memory_bandwidth_bytes_per_s=102 * (1024**3),  # 102 GB/s
    peak_fp32_tflops=2.5,
    peak_fp16_tflops=5.0,
    peak_int8_tops=100.0,
    compute_capability="8.7",
    tdp_watts=25,  # 10W-25W configurable
)

NVIDIA_JETSON_ORIN_NX_8GB = HardwareProfile(
    name="NVIDIA Jetson Orin NX 8GB",
    vendor="nvidia",
    device_type="gpu",
    vram_bytes=8 * (1024**3),  # 8 GB unified memory
    memory_bandwidth_bytes_per_s=102 * (1024**3),  # 102 GB/s
    peak_fp32_tflops=2.0,
    peak_fp16_tflops=4.0,
    peak_int8_tops=70.0,
    compute_capability="8.7",
    tdp_watts=25,
)

NVIDIA_JETSON_ORIN_NANO_8GB = HardwareProfile(
    name="NVIDIA Jetson Orin Nano 8GB",
    vendor="nvidia",
    device_type="gpu",
    vram_bytes=8 * (1024**3),  # 8 GB unified memory
    memory_bandwidth_bytes_per_s=68 * (1024**3),  # 68 GB/s
    peak_fp32_tflops=1.0,
    peak_fp16_tflops=2.0,
    peak_int8_tops=40.0,
    compute_capability="8.7",
    tdp_watts=15,  # 7W-15W configurable
)

NVIDIA_JETSON_ORIN_NANO_4GB = HardwareProfile(
    name="NVIDIA Jetson Orin Nano 4GB",
    vendor="nvidia",
    device_type="gpu",
    vram_bytes=4 * (1024**3),  # 4 GB unified memory
    memory_bandwidth_bytes_per_s=68 * (1024**3),  # 68 GB/s
    peak_fp32_tflops=0.625,
    peak_fp16_tflops=1.25,
    peak_int8_tops=20.0,
    compute_capability="8.7",
    tdp_watts=10,
)

# Jetson Xavier Series (2018-2020)
NVIDIA_JETSON_AGX_XAVIER_32GB = HardwareProfile(
    name="NVIDIA Jetson AGX Xavier 32GB",
    vendor="nvidia",
    device_type="gpu",
    vram_bytes=32 * (1024**3),  # 32 GB unified memory
    memory_bandwidth_bytes_per_s=136 * (1024**3),  # 136 GB/s
    peak_fp32_tflops=1.4,
    peak_fp16_tflops=2.8,
    peak_int8_tops=22.0,
    compute_capability="7.2",
    tdp_watts=30,  # 10W-30W configurable
)

NVIDIA_JETSON_AGX_XAVIER_16GB = HardwareProfile(
    name="NVIDIA Jetson AGX Xavier 16GB",
    vendor="nvidia",
    device_type="gpu",
    vram_bytes=16 * (1024**3),  # 16 GB unified memory
    memory_bandwidth_bytes_per_s=136 * (1024**3),  # 136 GB/s
    peak_fp32_tflops=1.4,
    peak_fp16_tflops=2.8,
    peak_int8_tops=22.0,
    compute_capability="7.2",
    tdp_watts=30,
)

NVIDIA_JETSON_XAVIER_NX_16GB = HardwareProfile(
    name="NVIDIA Jetson Xavier NX 16GB",
    vendor="nvidia",
    device_type="gpu",
    vram_bytes=16 * (1024**3),  # 16 GB unified memory
    memory_bandwidth_bytes_per_s=59 * (1024**3),  # 59.7 GB/s
    peak_fp32_tflops=0.5,
    peak_fp16_tflops=1.0,
    peak_int8_tops=21.0,
    compute_capability="7.2",
    tdp_watts=20,  # 10W-20W configurable
)

NVIDIA_JETSON_XAVIER_NX_8GB = HardwareProfile(
    name="NVIDIA Jetson Xavier NX 8GB",
    vendor="nvidia",
    device_type="gpu",
    vram_bytes=8 * (1024**3),  # 8 GB unified memory
    memory_bandwidth_bytes_per_s=59 * (1024**3),  # 59.7 GB/s
    peak_fp32_tflops=0.5,
    peak_fp16_tflops=1.0,
    peak_int8_tops=21.0,
    compute_capability="7.2",
    tdp_watts=20,
)

# Jetson TX2 Series (2017)
NVIDIA_JETSON_TX2 = HardwareProfile(
    name="NVIDIA Jetson TX2",
    vendor="nvidia",
    device_type="gpu",
    vram_bytes=8 * (1024**3),  # 8 GB unified memory
    memory_bandwidth_bytes_per_s=59 * (1024**3),  # 59.7 GB/s
    peak_fp32_tflops=0.67,
    peak_fp16_tflops=1.33,
    peak_int8_tops=0.0,  # No INT8 tensor cores
    compute_capability="6.2",
    tdp_watts=15,  # 7.5W-15W configurable
)

NVIDIA_JETSON_TX2_NX = HardwareProfile(
    name="NVIDIA Jetson TX2 NX",
    vendor="nvidia",
    device_type="gpu",
    vram_bytes=4 * (1024**3),  # 4 GB unified memory
    memory_bandwidth_bytes_per_s=51 * (1024**3),  # 51.2 GB/s
    peak_fp32_tflops=0.5,
    peak_fp16_tflops=1.0,
    peak_int8_tops=0.0,
    compute_capability="6.2",
    tdp_watts=15,
)

# Jetson Nano (2019) - The most constrained!
NVIDIA_JETSON_NANO = HardwareProfile(
    name="NVIDIA Jetson Nano",
    vendor="nvidia",
    device_type="gpu",
    vram_bytes=4 * (1024**3),  # 4 GB unified memory
    memory_bandwidth_bytes_per_s=25 * (1024**3),  # 25.6 GB/s
    peak_fp32_tflops=0.236,
    peak_fp16_tflops=0.472,
    peak_int8_tops=0.0,  # No INT8 tensor cores
    compute_capability="5.3",
    tdp_watts=10,  # 5W-10W configurable
)

NVIDIA_JETSON_NANO_2GB = HardwareProfile(
    name="NVIDIA Jetson Nano 2GB",
    vendor="nvidia",
    device_type="gpu",
    vram_bytes=2 * (1024**3),  # 2 GB unified memory - extremely constrained!
    memory_bandwidth_bytes_per_s=25 * (1024**3),  # 25.6 GB/s
    peak_fp32_tflops=0.236,
    peak_fp16_tflops=0.472,
    peak_int8_tops=0.0,
    compute_capability="5.3",
    tdp_watts=5,
)

# ============================================================================
# NVIDIA Consumer GPUs
# ============================================================================

NVIDIA_RTX_4090 = HardwareProfile(
    name="NVIDIA RTX 4090",
    vendor="nvidia",
    device_type="gpu",
    vram_bytes=24 * (1024**3),  # 24 GB GDDR6X
    memory_bandwidth_bytes_per_s=1008 * (1024**3),  # 1 TB/s
    peak_fp32_tflops=82.6,
    peak_fp16_tflops=165.0,  # Tensor Core ~330 with sparsity
    peak_int8_tops=660.0,
    compute_capability="8.9",
    tdp_watts=450,
)

NVIDIA_RTX_4080 = HardwareProfile(
    name="NVIDIA RTX 4080",
    vendor="nvidia",
    device_type="gpu",
    vram_bytes=16 * (1024**3),  # 16 GB GDDR6X
    memory_bandwidth_bytes_per_s=717 * (1024**3),  # 717 GB/s
    peak_fp32_tflops=48.7,
    peak_fp16_tflops=97.0,
    peak_int8_tops=390.0,
    compute_capability="8.9",
    tdp_watts=320,
)

NVIDIA_RTX_3090 = HardwareProfile(
    name="NVIDIA RTX 3090",
    vendor="nvidia",
    device_type="gpu",
    vram_bytes=24 * (1024**3),  # 24 GB GDDR6X
    memory_bandwidth_bytes_per_s=936 * (1024**3),  # 936 GB/s
    peak_fp32_tflops=35.6,
    peak_fp16_tflops=71.0,
    peak_int8_tops=284.0,
    compute_capability="8.6",
    tdp_watts=350,
)

NVIDIA_RTX_3080 = HardwareProfile(
    name="NVIDIA RTX 3080",
    vendor="nvidia",
    device_type="gpu",
    vram_bytes=10 * (1024**3),  # 10 GB GDDR6X
    memory_bandwidth_bytes_per_s=760 * (1024**3),  # 760 GB/s
    peak_fp32_tflops=29.8,
    peak_fp16_tflops=59.0,
    peak_int8_tops=238.0,
    compute_capability="8.6",
    tdp_watts=320,
)

# Generic CPU profile (will be overridden by detection)
GENERIC_CPU = HardwareProfile(
    name="Generic CPU",
    vendor="generic",
    device_type="cpu",
    vram_bytes=16 * (1024**3),  # Assume 16 GB RAM
    memory_bandwidth_bytes_per_s=50 * (1024**3),  # ~50 GB/s DDR4
    peak_fp32_tflops=0.5,  # Very rough estimate
    peak_fp16_tflops=0.25,  # CPUs typically slower at FP16
    peak_int8_tops=2.0,  # VNNI/AVX-512
    compute_capability="",
    tdp_watts=65,
)

# Registry of all predefined profiles
HARDWARE_PROFILES: dict[str, HardwareProfile] = {
    # -------------------------------------------------------------------------
    # Data Center GPUs (Current Gen)
    # -------------------------------------------------------------------------
    "h100": NVIDIA_H100_SXM,
    "h100-sxm": NVIDIA_H100_SXM,
    "a100": NVIDIA_A100_80GB,  # Default A100 is 80GB
    "a100-80gb": NVIDIA_A100_80GB,
    "a100-40gb": NVIDIA_A100_40GB,
    "a10": NVIDIA_A10,
    "l4": NVIDIA_L4,
    "l40": NVIDIA_L40,
    "l40s": NVIDIA_L40S,
    "t4": NVIDIA_T4,
    # Data Center GPUs (Previous Gen)
    "v100": NVIDIA_V100_32GB,
    "v100-32gb": NVIDIA_V100_32GB,
    "v100-16gb": NVIDIA_V100_16GB,
    "p100": NVIDIA_P100,
    "p40": NVIDIA_P40,
    # -------------------------------------------------------------------------
    # Jetson Edge/Embedded (Orin Series - Current)
    # -------------------------------------------------------------------------
    "jetson-agx-orin-64gb": NVIDIA_JETSON_AGX_ORIN_64GB,
    "jetson-agx-orin-32gb": NVIDIA_JETSON_AGX_ORIN_32GB,
    "jetson-agx-orin": NVIDIA_JETSON_AGX_ORIN_64GB,  # Default to 64GB
    "orin-agx": NVIDIA_JETSON_AGX_ORIN_64GB,
    "jetson-orin-nx-16gb": NVIDIA_JETSON_ORIN_NX_16GB,
    "jetson-orin-nx-8gb": NVIDIA_JETSON_ORIN_NX_8GB,
    "jetson-orin-nx": NVIDIA_JETSON_ORIN_NX_16GB,
    "orin-nx": NVIDIA_JETSON_ORIN_NX_16GB,
    "jetson-orin-nano-8gb": NVIDIA_JETSON_ORIN_NANO_8GB,
    "jetson-orin-nano-4gb": NVIDIA_JETSON_ORIN_NANO_4GB,
    "jetson-orin-nano": NVIDIA_JETSON_ORIN_NANO_8GB,
    "orin-nano": NVIDIA_JETSON_ORIN_NANO_8GB,
    # Jetson Edge/Embedded (Xavier Series)
    "jetson-agx-xavier-32gb": NVIDIA_JETSON_AGX_XAVIER_32GB,
    "jetson-agx-xavier-16gb": NVIDIA_JETSON_AGX_XAVIER_16GB,
    "jetson-agx-xavier": NVIDIA_JETSON_AGX_XAVIER_32GB,
    "xavier-agx": NVIDIA_JETSON_AGX_XAVIER_32GB,
    "jetson-xavier-nx-16gb": NVIDIA_JETSON_XAVIER_NX_16GB,
    "jetson-xavier-nx-8gb": NVIDIA_JETSON_XAVIER_NX_8GB,
    "jetson-xavier-nx": NVIDIA_JETSON_XAVIER_NX_8GB,
    "xavier-nx": NVIDIA_JETSON_XAVIER_NX_8GB,
    # Jetson Edge/Embedded (TX2 Series)
    "jetson-tx2": NVIDIA_JETSON_TX2,
    "tx2": NVIDIA_JETSON_TX2,
    "jetson-tx2-nx": NVIDIA_JETSON_TX2_NX,
    "tx2-nx": NVIDIA_JETSON_TX2_NX,
    # Jetson Edge/Embedded (Nano - Most Constrained!)
    "jetson-nano": NVIDIA_JETSON_NANO,
    "nano": NVIDIA_JETSON_NANO,
    "jetson-nano-2gb": NVIDIA_JETSON_NANO_2GB,
    "nano-2gb": NVIDIA_JETSON_NANO_2GB,
    # -------------------------------------------------------------------------
    # Consumer GPUs
    # -------------------------------------------------------------------------
    "rtx4090": NVIDIA_RTX_4090,
    "4090": NVIDIA_RTX_4090,
    "rtx4080": NVIDIA_RTX_4080,
    "4080": NVIDIA_RTX_4080,
    "rtx3090": NVIDIA_RTX_3090,
    "3090": NVIDIA_RTX_3090,
    "rtx3080": NVIDIA_RTX_3080,
    "3080": NVIDIA_RTX_3080,
    # -------------------------------------------------------------------------
    # Generic / Fallback
    # -------------------------------------------------------------------------
    "cpu": GENERIC_CPU,
}


# ============================================================================
# Hardware Detection
# ============================================================================


class HardwareDetector:
    """
    Detect local hardware configuration.

    Attempts to detect NVIDIA GPUs via nvidia-smi, falls back to CPU info.
    """

    def __init__(self, logger: logging.Logger | None = None):
        self.logger = logger or logging.getLogger("autodoc.hardware")

    def detect(self) -> HardwareProfile:
        """
        Auto-detect local hardware.

        Returns:
            HardwareProfile for the detected hardware.
        """
        # Try NVIDIA GPU first
        gpu_profile = self._detect_nvidia_gpu()
        if gpu_profile:
            return gpu_profile

        # Fall back to CPU
        return self._detect_cpu()

    def _detect_nvidia_gpu(self) -> HardwareProfile | None:
        """Detect NVIDIA GPU using nvidia-smi."""
        try:
            # Query GPU name and memory
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,compute_cap",
                    "--format=csv,noheader,nounits",
                ],
                check=False, capture_output=True,
                text=True,
                timeout=5,
            )

            if result.returncode != 0:
                self.logger.debug("nvidia-smi failed or not found")
                return None

            # Parse first GPU (could extend to multi-GPU)
            line = result.stdout.strip().split("\n")[0]
            parts = [p.strip() for p in line.split(",")]

            if len(parts) < 2:
                return None

            gpu_name = parts[0]
            vram_mb = int(parts[1]) if parts[1].isdigit() else 0
            compute_cap = parts[2] if len(parts) > 2 else ""

            self.logger.info(f"Detected GPU: {gpu_name} ({vram_mb} MB VRAM)")

            # Try to match to a known profile
            profile = self._match_gpu_profile(gpu_name, vram_mb)
            if profile:
                # Create a copy with detected flag and actual VRAM
                return HardwareProfile(
                    name=f"{gpu_name} (detected)",
                    vendor="nvidia",
                    device_type="gpu",
                    vram_bytes=vram_mb * (1024**2),
                    memory_bandwidth_bytes_per_s=profile.memory_bandwidth_bytes_per_s,
                    peak_fp32_tflops=profile.peak_fp32_tflops,
                    peak_fp16_tflops=profile.peak_fp16_tflops,
                    peak_int8_tops=profile.peak_int8_tops,
                    compute_capability=compute_cap or profile.compute_capability,
                    tdp_watts=profile.tdp_watts,
                    is_detected=True,
                )

            # Unknown GPU - create generic profile with detected VRAM
            return HardwareProfile(
                name=f"{gpu_name} (detected)",
                vendor="nvidia",
                device_type="gpu",
                vram_bytes=vram_mb * (1024**2),
                memory_bandwidth_bytes_per_s=500 * (1024**3),  # Conservative estimate
                peak_fp32_tflops=10.0,  # Conservative
                peak_fp16_tflops=20.0,
                peak_int8_tops=40.0,
                compute_capability=compute_cap,
                is_detected=True,
            )

        except FileNotFoundError:
            self.logger.debug("nvidia-smi not found")
            return None
        except subprocess.TimeoutExpired:
            self.logger.warning("nvidia-smi timed out")
            return None
        except Exception as e:
            self.logger.debug(f"GPU detection failed: {e}")
            return None

    def _match_gpu_profile(self, gpu_name: str, vram_mb: int) -> HardwareProfile | None:
        """Match detected GPU name to a known profile."""
        gpu_name_lower = gpu_name.lower()

        # Jetson detection (check first as they're embedded)
        if "jetson" in gpu_name_lower or "tegra" in gpu_name_lower:
            return self._match_jetson_profile(gpu_name_lower, vram_mb)

        # Data center GPU patterns (check more specific patterns first)
        datacenter_matches = [
            ("h100", NVIDIA_H100_SXM),
            ("a100", NVIDIA_A100_80GB if vram_mb > 50000 else NVIDIA_A100_40GB),
            ("a10", NVIDIA_A10),
            ("l40s", NVIDIA_L40S),
            ("l40", NVIDIA_L40),
            ("l4", NVIDIA_L4),
            ("t4", NVIDIA_T4),
            ("v100", NVIDIA_V100_32GB if vram_mb > 20000 else NVIDIA_V100_16GB),
            ("p100", NVIDIA_P100),
            ("p40", NVIDIA_P40),
        ]

        for pattern, profile in datacenter_matches:
            if pattern in gpu_name_lower:
                return profile

        # Consumer GPU patterns
        consumer_matches = [
            ("4090", NVIDIA_RTX_4090),
            ("4080", NVIDIA_RTX_4080),
            ("4070", NVIDIA_RTX_4080),  # Approximate with 4080
            ("4060", NVIDIA_RTX_4080),  # Approximate
            ("3090", NVIDIA_RTX_3090),
            ("3080", NVIDIA_RTX_3080),
            ("3070", NVIDIA_RTX_3080),  # Approximate
            ("3060", NVIDIA_RTX_3080),  # Approximate
        ]

        for pattern, profile in consumer_matches:
            if pattern in gpu_name_lower:
                return profile

        return None

    def _match_jetson_profile(
        self, gpu_name_lower: str, vram_mb: int
    ) -> HardwareProfile | None:
        """Match Jetson device to appropriate profile."""
        # Orin series
        if "orin" in gpu_name_lower:
            if "agx" in gpu_name_lower:
                return (
                    NVIDIA_JETSON_AGX_ORIN_64GB
                    if vram_mb > 40000
                    else NVIDIA_JETSON_AGX_ORIN_32GB
                )
            elif "nx" in gpu_name_lower:
                return (
                    NVIDIA_JETSON_ORIN_NX_16GB
                    if vram_mb > 10000
                    else NVIDIA_JETSON_ORIN_NX_8GB
                )
            elif "nano" in gpu_name_lower:
                return (
                    NVIDIA_JETSON_ORIN_NANO_8GB
                    if vram_mb > 5000
                    else NVIDIA_JETSON_ORIN_NANO_4GB
                )
            # Default Orin
            return NVIDIA_JETSON_ORIN_NX_8GB

        # Xavier series
        if "xavier" in gpu_name_lower:
            if "agx" in gpu_name_lower:
                return (
                    NVIDIA_JETSON_AGX_XAVIER_32GB
                    if vram_mb > 20000
                    else NVIDIA_JETSON_AGX_XAVIER_16GB
                )
            elif "nx" in gpu_name_lower:
                return (
                    NVIDIA_JETSON_XAVIER_NX_16GB
                    if vram_mb > 10000
                    else NVIDIA_JETSON_XAVIER_NX_8GB
                )
            return NVIDIA_JETSON_XAVIER_NX_8GB

        # TX2 series
        if "tx2" in gpu_name_lower:
            if "nx" in gpu_name_lower:
                return NVIDIA_JETSON_TX2_NX
            return NVIDIA_JETSON_TX2

        # Nano (most common constrained device)
        if "nano" in gpu_name_lower:
            return NVIDIA_JETSON_NANO_2GB if vram_mb < 3000 else NVIDIA_JETSON_NANO

        # Generic Jetson fallback based on memory
        if vram_mb <= 2000:
            return NVIDIA_JETSON_NANO_2GB
        elif vram_mb <= 4000:
            return NVIDIA_JETSON_NANO
        elif vram_mb <= 8000:
            return NVIDIA_JETSON_ORIN_NANO_8GB
        else:
            return NVIDIA_JETSON_ORIN_NX_16GB

    def _detect_cpu(self) -> HardwareProfile:
        """Detect CPU and system memory."""
        cpu_name = platform.processor() or "Unknown CPU"

        # Get system memory
        if _HAS_PSUTIL:
            ram_bytes = psutil.virtual_memory().total
        else:
            # Fallback: assume 16 GB
            ram_bytes = 16 * (1024**3)

        # Estimate CPU performance (very rough)
        # Modern CPUs can do ~0.5-2 TFLOPS FP32 depending on cores/frequency
        cpu_count = os.cpu_count() or 4
        estimated_fp32_tflops = 0.1 * cpu_count  # ~0.1 TFLOPS per core

        self.logger.info(
            f"Detected CPU: {cpu_name} ({cpu_count} cores, {ram_bytes / (1024**3):.1f} GB RAM)"
        )

        return HardwareProfile(
            name=f"{cpu_name} (detected)",
            vendor="generic",
            device_type="cpu",
            vram_bytes=ram_bytes,
            memory_bandwidth_bytes_per_s=50 * (1024**3),  # Typical DDR4/DDR5
            peak_fp32_tflops=estimated_fp32_tflops,
            peak_fp16_tflops=estimated_fp32_tflops * 0.5,  # CPUs slower at FP16
            peak_int8_tops=estimated_fp32_tflops * 4,  # VNNI acceleration
            is_detected=True,
        )


# ============================================================================
# Hardware Estimator
# ============================================================================


@dataclass
class HardwareEstimates:
    """Estimated performance characteristics for a model on specific hardware."""

    device: str
    precision: str
    batch_size: int

    # Memory
    vram_required_bytes: int
    fits_in_vram: bool

    # Performance
    theoretical_latency_ms: float
    compute_utilization_estimate: float  # 0.0 - 1.0
    bottleneck: str  # "compute", "memory_bandwidth", "vram"

    # Context
    model_flops: int
    hardware_peak_tflops: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "device": self.device,
            "precision": self.precision,
            "batch_size": self.batch_size,
            "vram_required_gb": round(self.vram_required_bytes / (1024**3), 2),
            "fits_in_vram": self.fits_in_vram,
            "theoretical_latency_ms": round(self.theoretical_latency_ms, 2),
            "compute_utilization_estimate": round(self.compute_utilization_estimate, 2),
            "bottleneck": self.bottleneck,
        }


class HardwareEstimator:
    """
    Estimate hardware requirements and performance.

    Provides theoretical bounds based on model complexity and hardware specs.
    Actual performance will vary based on implementation efficiency.
    """

    def __init__(self, logger: logging.Logger | None = None):
        self.logger = logger or logging.getLogger("autodoc.hardware")

    def estimate(
        self,
        model_params: int,
        model_flops: int,
        peak_activation_bytes: int,
        hardware: HardwareProfile,
        batch_size: int = 1,
        precision: str = "fp32",
    ) -> HardwareEstimates:
        """
        Estimate hardware requirements for a model.

        Args:
            model_params: Total parameter count
            model_flops: FLOPs per inference (batch=1)
            peak_activation_bytes: Peak activation memory (batch=1)
            hardware: Target hardware profile
            batch_size: Batch size for inference
            precision: "fp32", "fp16", or "int8"

        Returns:
            HardwareEstimates with performance predictions
        """
        # Bytes per parameter based on precision
        bytes_per_param = {"fp32": 4, "fp16": 2, "int8": 1, "bf16": 2}.get(precision, 4)

        # Model weights memory
        weights_bytes = model_params * bytes_per_param

        # Activation memory scales with batch size
        activation_bytes = peak_activation_bytes * batch_size

        # Total VRAM required (weights + activations + workspace overhead)
        workspace_overhead = 1.2  # 20% overhead for cuDNN workspace, etc.
        vram_required = int((weights_bytes + activation_bytes) * workspace_overhead)

        fits_in_vram = vram_required <= hardware.vram_bytes

        # Select peak TFLOPS based on precision
        if precision == "int8":
            peak_tflops = hardware.peak_int8_tops  # Note: TOPS, not TFLOPS
        elif precision in ("fp16", "bf16"):
            peak_tflops = hardware.peak_fp16_tflops
        else:
            peak_tflops = hardware.peak_fp32_tflops

        # Theoretical compute time
        total_flops = model_flops * batch_size
        compute_time_ms = (
            (total_flops / (peak_tflops * 1e12)) * 1000
            if peak_tflops > 0
            else float("inf")
        )

        # Memory bandwidth time (moving activations)
        total_memory_access = (
            weights_bytes + activation_bytes * 2
        ) * batch_size  # Read + write activations
        memory_time_ms = (
            total_memory_access / hardware.memory_bandwidth_bytes_per_s
        ) * 1000

        # Bottleneck analysis
        if not fits_in_vram:
            bottleneck = "vram"
            theoretical_latency = float("inf")
            utilization = 0.0
        elif memory_time_ms > compute_time_ms:
            bottleneck = "memory_bandwidth"
            theoretical_latency = memory_time_ms
            utilization = compute_time_ms / memory_time_ms if memory_time_ms > 0 else 0
        else:
            bottleneck = "compute"
            theoretical_latency = compute_time_ms
            utilization = 0.7  # Assume 70% compute utilization in compute-bound case

        return HardwareEstimates(
            device=hardware.name,
            precision=precision,
            batch_size=batch_size,
            vram_required_bytes=vram_required,
            fits_in_vram=fits_in_vram,
            theoretical_latency_ms=theoretical_latency,
            compute_utilization_estimate=min(utilization, 1.0),
            bottleneck=bottleneck,
            model_flops=model_flops,
            hardware_peak_tflops=peak_tflops,
        )


# ============================================================================
# Convenience Functions
# ============================================================================


def list_available_profiles() -> list[str]:
    """List all available hardware profile names."""
    # Deduplicate (some are aliases)
    unique = set()
    for name, profile in HARDWARE_PROFILES.items():
        unique.add(f"{name}: {profile.name}")
    return sorted(unique)


def get_profile(name: str) -> HardwareProfile | None:
    """Get a hardware profile by name."""
    return HARDWARE_PROFILES.get(name.lower())


def detect_local_hardware() -> HardwareProfile:
    """Convenience function to detect local hardware."""
    detector = HardwareDetector()
    return detector.detect()
