# 2D Heat Transfer using Jacobi Method (Optimized with CUDA)

This project implements the **Jacobi iterative method** for solving 2D steady-state heat distribution on a grid using:
- **Serial (CPU)** implementation
- **Basic CUDA (GPU)** implementation
- **Optimized CUDA** implementation with shared memory tiling

---

## 📋 Table of Contents
- [Introduction](#introduction)
- [Jacobi Method for Heat Transfer](#jacobi-method-for-heat-transfer)
- [1. Serial Implementation](#1-serial-implementation)
- [2. Basic CUDA Implementation](#2-basic-cuda-implementation)
- [3. Optimized CUDA Implementation](#3-optimized-cuda-implementation)
- [Performance Comparison](#performance-comparison)
- [Nsight Compute Analysis](#nsight-compute-analysis)
- [How to Build and Run](#how-to-build-and-run)

---

## Introduction

This project simulates heat diffusion on a 1000×1000 grid where the top boundary is maintained at 100°C and all other boundaries are at 0°C. The simulation runs until the maximum change between iterations falls below `0.001`.

---

## Jacobi Method for Heat Transfer

The **Jacobi method** is an iterative technique used to solve systems of linear equations. In the context of 2D heat transfer, each interior point in the grid is updated as the average of its four neighboring points:

$$
T_{new}(i,j) = \frac{T(i-1,j) + T(i+1,j) + T(i,j-1) + T(i,j+1)}{4}
$$

This process is repeated until the solution converges (maximum difference between successive iterations < error threshold).

**Grid Setup:**
- Grid size: `1000 × 1000`
- Top row (`i=0`): 100°C (hot boundary)
- All other boundaries: 0°C
- Interior points start at 0°C

*(Insert grid visualization / initial condition image here)*

---

## 1. Serial Implementation

### What is Jacobi Heat Transfer (Serial)?
The serial version performs the Jacobi iteration entirely on the CPU using nested loops. For every iteration:
- A new grid is computed from the old grid.
- The maximum difference is tracked to check convergence.
- Grids are swapped after each iteration.

### Time Taken
**Time taken by serial code:** `178.484957` seconds

*(Insert screenshot of serial code execution here)*

**Conclusion:**  
The pure CPU implementation is straightforward but extremely slow for large grids (1000×1000) due to the high number of iterations required.

---

## 2. Basic CUDA Implementation

### How CUDA Functions Were Added
- Two kernels were implemented:
  1. `heat_transfer` kernel — computes new temperature values and finds maximum difference using `atomicMax`.
  2. `swap_grids` kernel — swaps old and new grids (commented out in optimized version for pointer swapping).
- Memory is allocated on GPU using `cudaMalloc`.
- Data is transferred between host and device using `cudaMemcpy`.
- Thread block size: `16×16`
- Grid dimension: `((M+15)/16) × ((N+15)/16)`

### Time Taken
**GPU Time:** `1.730222` seconds  
**Total Time:** `1.738451` seconds

*(Insert screenshot of basic CUDA code execution here)*

### Speedup
**Speedup vs Serial:** ≈ **103×** faster

### Nsight Compute Analysis

**Key Metrics:**
- Compute (SM) Throughput: **88.43%**
- Memory Throughput: **88.43%**
- Achieved Occupancy: **92.25%**
- High overall throughput (>80%)

**Conclusion from Nsight Compute:**
The basic CUDA version achieves very high compute and memory utilization. The kernel is well-balanced, but there is still room for optimization by reducing global memory accesses using **shared memory tiling**.

*(Insert Nsight Compute screenshots for basic CUDA version here)*

---

## 3. Optimized CUDA Implementation

### Changes Made in the Code
The optimized version introduces **shared memory tiling** to significantly reduce global memory traffic:

**Key Optimizations:**
- Used `__shared__ float tile[18][18];` to load a 16×16 tile + halo (border) cells.
- Each thread loads its own cell + halo cells (left, right, top, bottom) cooperatively.
- All neighboring reads happen from fast **shared memory** after `__syncthreads()`.
- Block-level maximum difference reduction using shared memory + parallel reduction.
- Pointer swapping instead of using `swap_grids` kernel (minor performance gain).

### Time Taken
*(Insert screenshot of optimized CUDA execution here)*

### Speedup
**Speedup vs Serial:** Significantly higher than basic CUDA  
**Speedup vs Basic CUDA:** Improved due to reduced global memory accesses

### Nsight Compute Analysis

**Key Metrics:**
- Compute (SM) Throughput: **47.00%**
- Memory Throughput: **66.81%**
- Duration: **35.74 μs** per kernel launch (much lower than basic version)
- Lower compute throughput but **much better memory efficiency** and faster kernel execution.

**Conclusion from Nsight Compute:**
- Memory is more heavily utilized than compute (as expected in stencil computations).
- Shared memory tiling successfully reduced global memory pressure.
- Kernel runtime per iteration is dramatically reduced.
- Further improvements possible through kernel fusion or better occupancy tuning.

*(Insert Nsight Compute screenshots for optimized CUDA version here - both Summary and Details tabs)*

---

## Performance Comparison

| Implementation       | Execution Time (s) | Speedup vs Serial |
|----------------------|--------------------|-------------------|
| Serial (CPU)         | 178.48            | 1×               |
| Basic CUDA           | 1.738             | ~103×            |
| Optimized CUDA       | ~0.917            | ~195×            |

*(Note: Optimized time taken from your first screenshot ≈ 0.917s)*

---

## Nsight Compute Insights Summary

- **Basic CUDA**: High throughput but higher latency per kernel due to many global memory accesses.
- **Optimized CUDA**: Lower per-kernel duration thanks to shared memory. Better overall performance despite slightly lower SM throughput percentage (because the kernel finishes much faster).
- Memory-bound nature of stencil computations is clearly visible.

---

## How to Build and Run

### Prerequisites
- NVIDIA GPU with CUDA support
- CUDA Toolkit installed
- `nvcc` compiler

### Compile & Run

```bash
# Serial version
gcc serial.c -o serial -lm
./serial

# Basic CUDA version
nvcc cuda_code.cu -o cuda_code
./cuda_code

# Optimized CUDA version
nvcc initial_code.cu -o initial_code
./initial_code