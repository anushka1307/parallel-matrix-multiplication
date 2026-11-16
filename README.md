# Parallel Matrix Multiplication with OpenMP

Dense matrix multiplication in C, optimized step by step with OpenMP on a shared-memory multicore system. The program has four modes: a sequential baseline, a naive OpenMP version, a cache-blocked parallel version, and a cache-blocked version with guided scheduling.

## Results

Measured on NYU CIMS `crunchy3` (AMD Opteron 6272, 4 sockets, 64 cores, 8 NUMA nodes), 8 threads, double precision, averaged over three runs.

| Matrix | Mode 0 (s) | Mode 1 (s) | Mode 2 (s) | Mode 3 (s) |
|---|---|---|---|---|
| 512 | 1.98 | 0.245 | 0.018 | 0.021 |
| 1,024 | 21.56 | 4.31 | 0.24 | 0.29 |
| 2,048 | 386.49 | 78.38 | 0.998 | 1.00 |
| 4,096 | 4865.03 | 895.3 | 9.87 | 9.45 |

- Cache blocking (Mode 2) is 493× faster than the sequential baseline at 4096×4096, and 91× faster than naive parallelization on the same 8 threads.
- The blocked version reaches 7.42× speedup on 8 threads (92% parallel efficiency).
- Guided scheduling (Mode 3) matches static scheduling up to 4096, but is 2.6× slower at 16,384 because of scheduling overhead and NUMA effects.

## Compilation

```bash
gcc -O3 -march=native -funroll-loops -ffast-math -fopenmp \
    matrix-multiplication.c -o matmul
```

## Usage

```bash
./matmul N T M
```

- **N:** size of the square matrix (N × N). For example, 1024 creates three 1024×1024 matrices (A, B, C). Recommended values: 512, 1024, 2048, 4096, 8192.
- **T:** number of OpenMP threads. Recommended: the number of physical CPU cores.
- **M:** execution mode (0–3), described below.

### Example commands

```bash
./matmul 1024 1 0    # sequential baseline on 1024×1024
./matmul 1024 8 1    # naive parallel, 8 threads
./matmul 4096 8 2    # cache-blocked, 8 threads
./matmul 4096 16 3   # guided scheduling, 16 threads
```

## Modes

**Mode 0: Sequential baseline**
- Standard triple-loop matrix multiplication (i-j-k ordering), no parallelization
- Used as the performance baseline

**Mode 1: Naive OpenMP**
- Parallelizes the outer loops using `collapse(2)`, no cache optimization
- Scales poorly because all threads saturate memory bandwidth

**Mode 2: Cache-blocked parallel**
- Loop tiling with 64×64 blocks, sized for L2 cache reuse
- Static work distribution across threads
- Best performance

**Mode 3: Guided scheduling**
- Same blocking as Mode 2, with block size chosen by matrix size (32, 64, 128 or 256)
- Dynamic work distribution using OpenMP `schedule(guided)`
- May improve load balancing but adds scheduling overhead

## Performance tips

Set the thread count to the number of physical cores (not hyperthreads):

```bash
lscpu | grep "Core(s) per socket"
export OMP_NUM_THREADS=8
./matmul 4096 8 2
```

Other useful OpenMP environment variables:

```bash
export OMP_DISPLAY_ENV=TRUE   # display OpenMP configuration
export OMP_DYNAMIC=FALSE      # disable dynamic thread adjustment
```
