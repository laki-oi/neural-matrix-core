"""Matrix computation benchmark runner."""

import time
from .core import MatrixEngine


def run_benchmark():
    """Execute matrix kernel speed probes."""
    print("=" * 50)
    print("Initializing Neural Matrix Kernel Benchmark...")
    print("=" * 50)
    engine = MatrixEngine()
    t0 = time.time()
    res = engine.benchmark_gemm(1024, 1024, 1024)
    elapsed = time.time() - t0
    print(f"Matrix benchmark passed: output shape {res.shape} in {elapsed:.4f}s")
    print("Matrix runtime operational.")


if __name__ == "__main__":
    run_benchmark()
