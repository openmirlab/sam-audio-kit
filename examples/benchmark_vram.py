#!/usr/bin/env python3
"""
VRAM Benchmark for SAM-Audio-Infer

Measures VRAM usage across different configurations using pynvml.
"""

import gc
import sys
import time
from dataclasses import dataclass
from typing import Optional

import torch
import pynvml


@dataclass
class BenchmarkResult:
    model_size: str
    lite_config: str
    dtype: str
    vram_used_gb: float
    load_time_s: float
    error: Optional[str] = None


def init_nvml():
    """Initialize NVML."""
    pynvml.nvmlInit()


def shutdown_nvml():
    """Shutdown NVML."""
    pynvml.nvmlShutdown()


def get_vram_used_gb(device_index: int = 0) -> float:
    """Get VRAM used in GB using pynvml."""
    handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return info.used / (1024**3)


def get_gpu_info(device_index: int = 0) -> tuple[str, float]:
    """Get GPU name and total memory."""
    handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
    name = pynvml.nvmlDeviceGetName(handle)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    total_gb = info.total / (1024**3)
    return name, total_gb


def cleanup_gpu():
    """Clean up GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    time.sleep(0.5)


def run_benchmark(
    model_size: str,
    lite_config_name: str,
    dtype: str,
) -> BenchmarkResult:
    """Run a single benchmark."""
    from sam_audio_infer import SamAudioInfer, LiteModelConfig

    # Clean and get baseline
    cleanup_gpu()
    baseline_vram = get_vram_used_gb()

    # Build lite config
    lite_configs = {
        "aggressive": LiteModelConfig.aggressive(),
        "with_text_ranker": LiteModelConfig.with_text_ranker(),
        "with_span_predictor": LiteModelConfig.with_span_predictor(),
        "with_all_features": LiteModelConfig.with_all_features(),
        "no_lite": None,
    }
    lite_config = lite_configs.get(lite_config_name)

    try:
        start_time = time.time()

        if lite_config_name == "no_lite":
            model = SamAudioInfer.from_pretrained(
                model_size,
                lite_mode=False,
                device="cuda",
                dtype=dtype,
                verbose=False,
            )
        else:
            model = SamAudioInfer.from_pretrained(
                model_size,
                lite_mode=True,
                lite_config=lite_config,
                device="cuda",
                dtype=dtype,
                verbose=False,
            )

        load_time = time.time() - start_time

        # Sync and measure
        torch.cuda.synchronize()
        vram_after = get_vram_used_gb()
        vram_used = vram_after - baseline_vram

        # Cleanup
        model.unload()
        del model
        cleanup_gpu()

        return BenchmarkResult(
            model_size=model_size,
            lite_config=lite_config_name,
            dtype=dtype,
            vram_used_gb=vram_used,
            load_time_s=load_time,
        )

    except Exception as e:
        cleanup_gpu()
        return BenchmarkResult(
            model_size=model_size,
            lite_config=lite_config_name,
            dtype=dtype,
            vram_used_gb=0,
            load_time_s=0,
            error=str(e)[:100],
        )


def main():
    print("SAM-Audio-Infer VRAM Benchmark")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        sys.exit(1)

    init_nvml()
    gpu_name, gpu_total = get_gpu_info()
    print(f"GPU: {gpu_name}")
    print(f"Total VRAM: {gpu_total:.1f} GB")
    print()

    results = []

    # Test configurations
    model_sizes = ["large"]
    lite_configs = ["aggressive", "with_text_ranker", "with_span_predictor", "with_all_features", "no_lite"]
    dtypes = ["bfloat16"]  # Just bfloat16 for speed

    total = len(model_sizes) * len(lite_configs) * len(dtypes)
    current = 0

    print(f"Running {total} benchmarks...\n")

    for model_size in model_sizes:
        for lite_config in lite_configs:
            for dtype in dtypes:
                current += 1
                print(f"[{current}/{total}] {model_size}/{lite_config}/{dtype}...", end=" ", flush=True)

                result = run_benchmark(model_size, lite_config, dtype)
                results.append(result)

                if result.error:
                    print(f"ERROR: {result.error}")
                else:
                    print(f"{result.vram_used_gb:.2f} GB")

    shutdown_nvml()

    # Print summary tables
    print("\n" + "=" * 80)
    print("RESULTS: VRAM Usage (GB) - bfloat16")
    print("=" * 80)
    print(f"{'Config':<25} {'Small':<10} {'Base':<10} {'Large':<10}")
    print("-" * 55)

    for config in lite_configs:
        row = f"{config:<25}"
        for size in model_sizes:
            r = next((x for x in results if x.model_size == size and x.lite_config == config and x.dtype == "bfloat16" and not x.error), None)
            row += f"{r.vram_used_gb:<10.2f}" if r else f"{'ERR':<10}"
        print(row)

    print("\n" + "=" * 80)
    print("RESULTS: VRAM Usage (GB) - float16")
    print("=" * 80)
    print(f"{'Config':<25} {'Small':<10} {'Base':<10} {'Large':<10}")
    print("-" * 55)

    for config in lite_configs:
        row = f"{config:<25}"
        for size in model_sizes:
            r = next((x for x in results if x.model_size == size and x.lite_config == config and x.dtype == "float16" and not x.error), None)
            row += f"{r.vram_used_gb:<10.2f}" if r else f"{'ERR':<10}"
        print(row)

    print("\n" + "=" * 80)
    print("RESULTS: VRAM Usage (GB) - float32")
    print("=" * 80)
    print(f"{'Config':<25} {'Small':<10} {'Base':<10} {'Large':<10}")
    print("-" * 55)

    for config in lite_configs:
        row = f"{config:<25}"
        for size in model_sizes:
            r = next((x for x in results if x.model_size == size and x.lite_config == config and x.dtype == "float32" and not x.error), None)
            row += f"{r.vram_used_gb:<10.2f}" if r else f"{'ERR':<10}"
        print(row)

    # Save CSV
    with open("benchmark_results.csv", "w") as f:
        f.write("model,config,dtype,vram_gb,load_s,error\n")
        for r in results:
            f.write(f"{r.model_size},{r.lite_config},{r.dtype},{r.vram_used_gb:.2f},{r.load_time_s:.1f},{r.error or ''}\n")

    print("\nResults saved to benchmark_results.csv")


if __name__ == "__main__":
    main()
