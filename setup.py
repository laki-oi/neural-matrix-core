from setuptools import setup, find_packages

setup(
    name="neural-matrix-core",
    version="0.1.0",
    packages=find_packages(),
    description="High-performance matrix compute kernels and runtime cache pipeline",
    author="Neural Matrix Contributors",
    license="Apache-2.0",
    python_requires=">=3.8",
)
