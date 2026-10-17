# Neural Matrix Core (`neural-matrix`)

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

High-performance matrix compute kernels, tensor activation cache matrices, and distributed compilation pipeline for neural network acceleration.

## Highlights
* **Optimized GEMM & BLAS Kernels**: High-throughput matrix math operations designed for low-latency inference workloads.
* **Distributed Runtime Serialization**: Seamless hydration and checkpoint serialization with remote registry datasets.
* **Continuous Matrix Profiling**: Built-in benchmark daemon for automated hardware telemetry and cache latency probes.

## Architecture
```
+--------------------------+      +--------------------------+
|   Neural Matrix Core     | ---> | Distributed Cache Layer  |
| (BLAS / Matrix Kernels)  |      |   (Hugging Face Dataset) |
+--------------------------+      +--------------------------+
             |                                 |
             v                                 v
+--------------------------+      +--------------------------+
|  Continuous Benchmark    | <--- | Runtime Execution Matrix |
+--------------------------+      +--------------------------+
```

## Quick Installation

```bash
git clone https://github.com/laki-oi/neural-matrix-core.git
cd neural-matrix-core
pip install -r requirements.txt
```

## Running Benchmarks

```bash
python3 -m neural_matrix.benchmarks
```

## License
Apache-2.0 License.
