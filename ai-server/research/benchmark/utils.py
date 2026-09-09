import csv
import time
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch


@dataclass
class BenchmarkResult:
    model_name: str
    params_m:   float   # 파라미터 수 (백만 단위)
    gflops:     float   # FLOPs (10^9)
    mean_ms:    float
    min_ms:     float
    max_ms:     float
    std_ms:     float
    device:     str
    batch_size: int
    output_dim: int     # 임베딩 벡터 차원


def time_inference(
    fn:      Callable,
    warmup:  int = 10,
    runs:    int = 100,
    device:  str = "cpu",
) -> dict:
    use_cuda = device.startswith("cuda") and torch.cuda.is_available()

    for _ in range(warmup):
        fn()
    if use_cuda:
        torch.cuda.synchronize()

    times = []
    for _ in range(runs):
        if use_cuda:
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            fn()
            e.record()
            torch.cuda.synchronize()
            times.append(s.elapsed_time(e))
        else:
            t0 = time.perf_counter()
            fn()
            times.append((time.perf_counter() - t0) * 1000.0)

    return {
        "mean_ms": float(np.mean(times)),
        "min_ms":  float(np.min(times)),
        "max_ms":  float(np.max(times)),
        "std_ms":  float(np.std(times)),
    }


def measure_flops(model: torch.nn.Module, dummy_input: torch.Tensor) -> tuple[float, float]:
    """thop으로 GFLOPs와 Params(M) 반환. 실패 시 -1 반환."""
    try:
        import thop
        m = deepcopy(model).eval().cpu()
        with torch.no_grad():
            flops, params = thop.profile(m, inputs=(dummy_input.cpu(),), verbose=False)
        del m
        return round(flops / 1e9, 4), round(params / 1e6, 4)
    except Exception as e:
        print(f"    [FLOPs 측정 실패] {e}")
        return -1.0, -1.0


def print_table(results: list[BenchmarkResult]) -> None:
    header = (
        f"{'Model':<22} {'Params(M)':>10} {'GFLOPs':>8} {'Out-Dim':>8} "
        f"{'Mean(ms)':>10} {'Min(ms)':>9} {'Max(ms)':>9} {'Std(ms)':>8} {'Device':<6}"
    )
    sep = "─" * len(header)
    print(f"\n{sep}\n{header}\n{sep}")
    for r in results:
        print(
            f"{r.model_name:<22} {r.params_m:>10.2f} {r.gflops:>8.3f} {r.output_dim:>8} "
            f"{r.mean_ms:>10.2f} {r.min_ms:>9.2f} {r.max_ms:>9.2f} {r.std_ms:>8.2f} {r.device:<6}"
        )
    print(sep)


def save_csv(results: list[BenchmarkResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=asdict(results[0]).keys())
        w.writeheader()
        w.writerows(asdict(r) for r in results)
    print(f"[CSV] {path}")


def save_markdown(results: list[BenchmarkResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    hdr  = "| Model | Params(M) | GFLOPs | Out-Dim | Mean(ms) | Min(ms) | Max(ms) | Std(ms) | Device |"
    sep  = "|:------|----------:|-------:|--------:|---------:|--------:|--------:|--------:|:-------|"
    rows = [
        f"| {r.model_name} | {r.params_m:.2f} | {r.gflops:.3f} | {r.output_dim} | "
        f"{r.mean_ms:.2f} | {r.min_ms:.2f} | {r.max_ms:.2f} | {r.std_ms:.2f} | {r.device} |"
        for r in results
    ]
    with open(path, "w") as f:
        f.write("# ReID Model Benchmark\n\n")
        f.write(f"- input shape : Bx3x256x128 (표준 ReID 입력)\n")
        f.write(f"- batch size  : {results[0].batch_size}\n\n")
        f.write(hdr + "\n" + sep + "\n" + "\n".join(rows) + "\n")
    print(f"[MD]  {path}")
