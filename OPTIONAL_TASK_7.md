# Optional Task 7: Threshold-Based Vector Pair Search

This document describes the implementation and experiments for Optional Task 7 of Topic 3, Vector Data Management.

## 1. Task Definition

Given a vector set:

\[
V=\{v_1,v_2,\ldots,v_n\}
\]

and a distance threshold \(\epsilon>0\), the goal is to enumerate all unordered vector pairs:

\[
(v_i,v_j),\quad i<j
\]

such that:

\[
dist(v_i,v_j)<\epsilon.
\]

This task is different from the KNN experiments in the main DiskANN evaluation. KNN search returns the top-k nearest neighbors for each query vector, while threshold-based pair search returns all pairs whose distance is below a fixed threshold. Therefore, the output size is not fixed. It depends heavily on the dataset distribution and the value of \(\epsilon\). When \(\epsilon\) is large or the dataset contains dense clusters, the number of output pairs may become very large.

## 2. Relationship to the Existing Project

The implementation follows the same data and execution style as the existing DiskANN and HNSW experiments:

- It reads vectors from the converted `.fvecs` format.
- It uses the same dataset names: `glove`, `coco`, and `fashion`.
- It uses the same metric names: `angular` and `euclidean`.
- It is compiled from the `src/` directory together with the existing C++ programs.
- Its command-line interface starts with the same three arguments as `diskann_test` and `hnsw_test`:

```bash
./epsilon_pairs <data_dir> <dataset_name> <metric> [--epsilon value] [--sample-size N] [--block-size N] [--output pairs.txt]
```

The source code is implemented in:

```text
src/epsilon_pairs.cpp
```

The build target is added to:

```text
src/Makefile
```

## 3. Build and Run

From `src/`, build the executable:

```bash
make epsilon_pairs
```

On Windows without `make`, compile directly:

```bash
g++ -std=c++17 -O3 -march=native -Wall -o epsilon_pairs.exe epsilon_pairs.cpp
```

Run examples:

```bash
./epsilon_pairs /path/to/data glove angular --epsilon 0.28 --sample-size 10000 --block-size 2048
./epsilon_pairs /path/to/data coco angular --epsilon 0.36 --sample-size 10000 --block-size 2048
./epsilon_pairs /path/to/data fashion euclidean --epsilon 1600 --sample-size 10000 --block-size 2048
```

To reproduce the multi-threshold experiment tables, omit `--epsilon`:

```bash
./epsilon_pairs /path/to/data glove angular --sample-size 10000 --block-size 2048
./epsilon_pairs /path/to/data coco angular --sample-size 10000 --block-size 2048
./epsilon_pairs /path/to/data fashion euclidean --sample-size 10000 --block-size 2048
```

Arguments:

| Argument | Meaning |
|---|---|
| `data_dir` | Directory containing converted `.fvecs` files |
| `dataset_name` | `glove`, `coco`, or `fashion` |
| `metric` | `angular` or `euclidean` |
| `--epsilon` | Optional single threshold. If omitted, the program runs a dataset-specific epsilon list for experiments |
| `--sample-size` | Optional number of training vectors used for the pair-scan experiment. If omitted, the full loaded training set is used |
| `--block-size` | Number of vectors processed in each outer block. Default: 2048 |
| `--output` | Optional file for writing matching pairs as `epsilon i j` |

If `--epsilon` is provided, the program solves the literal task for the given threshold. If it is omitted, the program uses a dataset-specific list of \(\epsilon\) values and reports, for each threshold:

- number of matching pairs
- number of pair checks
- elapsed time
- pair-check throughput

By default, the program reports pair counts and performance counters. If an output path is provided, it also writes every matching pair in the format:

```text
epsilon i j
```

The reported experiments omit the output file because the result set can be very large and writing all pairs would make the throughput measurement I/O-bound. This separation is intentional: the algorithm is able to enumerate the actual pairs, while the performance tables focus on the CPU cost of discovering them. In full-dataset mode, the current implementation is an in-memory exact scan; the external-memory approach in Section 5 is a design for scaling beyond memory.

## 4. Algorithm

### 4.1 Exact Blocked All-Pairs Scan

The implemented algorithm is an exact blocked all-pairs scan over the selected training-vector prefix. It enumerates all unordered pairs \((i,j)\) with \(i<j\). This avoids both self-pairs \((i,i)\) and duplicate symmetric pairs \((i,j)\), \((j,i)\).

At a high level, the algorithm is:

```text
Input: vectors V, threshold epsilon, block size B
Output: number of pairs satisfying dist(vi, vj) < epsilon

count = 0
for block_i in blocks(V):
    for block_j in blocks(V), block_j >= block_i:
        for i in block_i:
            for j in block_j:
                if i < j and dist(vi, vj) < epsilon:
                    count += 1
return count
```

The blocking strategy does not change the \(O(n^2d)\) worst-case time complexity, where \(d\) is the vector dimension. However, it makes the access pattern more cache-friendly and gives a direct path to an external-memory implementation.

### 4.2 Euclidean Distance Optimization

For Euclidean distance, the implementation avoids computing a square root for every pair. Instead of checking:

\[
\sqrt{\sum_k (v_i[k]-v_j[k])^2}<\epsilon,
\]

it checks the equivalent condition:

\[
\sum_k (v_i[k]-v_j[k])^2<\epsilon^2.
\]

The implementation also uses early termination. While accumulating the squared distance, if the partial sum already exceeds \(\epsilon^2\), the pair cannot satisfy the threshold and the remaining dimensions are skipped. This optimization is especially effective when \(\epsilon\) is small.

### 4.3 Angular Distance Optimization

For angular distance, directly evaluating:

\[
\arccos\left(\frac{v_i\cdot v_j}{\|v_i\|\|v_j\|}\right)
\]

for every pair is expensive. The implementation uses the monotonicity of `acos` and transforms:

\[
angular(v_i,v_j)<\epsilon
\]

into:

\[
\frac{v_i\cdot v_j}{\|v_i\|\|v_j\|}>\cos(\epsilon).
\]

Thus, each comparison only needs a dot product and precomputed vector norms. This avoids calling `acos` for every candidate pair.

## 5. Large-Scale and External-Memory Design

The exact all-pairs search is quadratic in the number of vectors. For a dataset with \(n\) vectors, the number of unordered pairs is:

\[
\frac{n(n-1)}{2}.
\]

For the full GloVe training set with 1,183,514 vectors, this becomes approximately:

\[
7.0\times 10^{11}
\]

pair checks. Running this exact scan fully is not practical for a course-scale experiment. Therefore, our reported experiment uses a controlled prefix sample. The submitted implementation is an in-memory exact scanner for either the full loaded training set or an explicit sample. The rest of this section describes how to extend the same exact enumeration semantics to datasets that exceed memory.

The key system challenge is that both input and output can exceed memory. The input may be too large to load at once, and the output may contain many more pairs than vectors. A high-performance exact implementation should therefore treat both vectors and result pairs as streams.

### 5.1 Data Layout

The first step is to store vectors in a compact binary layout. The current project already uses `.fvecs`, where each vector is stored as:

```text
[int32 dimension][float32 values...]
```

For external-memory execution, a production implementation can keep a small metadata table:

```text
vector_id -> byte_offset
```

This makes each block directly readable from SSD. If all vectors have the same dimension, we can also store a contiguous matrix layout:

```text
float data[n][d]
```

which allows simpler sequential reads and better prefetching. For angular distance, a preprocessing step can normalize vectors and store unit vectors, so each comparison becomes a dot-product threshold test.

### 5.2 Blocked Exact Enumeration

Let \(B\) be the number of vectors in one block. The vector file is divided into blocks:

\[
X_0, X_1, \ldots, X_{m-1}.
\]

The scheduler enumerates all block pairs:

\[
(X_a, X_b),\quad a\le b.
\]

For diagonal block pairs \((X_a,X_a)\), it checks only pairs \(i<j\). For off-diagonal block pairs \((X_a,X_b)\), all cross-block pairs are checked. This covers every unordered vector pair exactly once.

The memory cost is:

\[
O(Bd)
\]

for a diagonal block and:

\[
O(2Bd)
\]

for an off-diagonal block pair. Therefore, with memory budget \(M\), a practical block size can be chosen as:

\[
B \le \frac{M}{2d\cdot sizeof(float)+metadata}.
\]

This makes the algorithm independent of the full dataset size as long as the vector blocks and output buffers fit in memory.

### 5.3 Output Management

The result set should not be kept in memory. Matching pairs should be written immediately to append-only partition files. A practical format is:

```text
epsilon vector_id_1 vector_id_2
```

or a binary tuple format:

```text
[float epsilon][uint32/uint64 id1][uint32/uint64 id2]
```

Output can be partitioned by block pair, by vector-id range, or by epsilon value. Partitioning avoids write contention in parallel execution and makes downstream processing easier. For example:

```text
results/
  eps_0.28/
    block_0000_0000.bin
    block_0000_0001.bin
    ...
```

If the result set is extremely large, compression can be applied because vector IDs within one block pair are often monotonic and delta-encodable.

### 5.4 Parallel Execution

Block pairs are independent, so the computation is naturally parallel. A high-performance implementation can use a thread pool:

1. A scheduler generates block-pair tasks.
2. Worker threads load the required block or reuse cached blocks.
3. Each worker computes distances and writes matching pairs to its own output buffer.
4. Buffers are flushed to partition files asynchronously.

To reduce I/O, the scheduler should process block pairs in an order that reuses one block for multiple comparisons. For example:

```text
load X_a once, then compare it with X_a, X_{a+1}, ..., X_{m-1}
```

This row-wise schedule reduces repeated reads of the left block. A cache for recently used right blocks can further reduce SSD traffic.

### 5.5 Exact Filters

Safe filters can reduce CPU work while preserving exactness.

For Euclidean distance, vector norms provide a lower bound:

\[
|\|v_i\|-\|v_j\|| \le \|v_i-v_j\|.
\]

If:

\[
|\|v_i\|-\|v_j\|| \ge \epsilon,
\]

then the pair can be skipped without reading all dimensions. During full distance computation, the implementation can also stop as soon as the partial squared distance reaches \(\epsilon^2\).

For angular distance, vectors can be normalized offline. Then:

\[
angular(v_i,v_j)<\epsilon
\iff
v_i\cdot v_j>\cos(\epsilon).
\]

This removes per-pair norm computation. Additional exact upper bounds on the dot product can be built from vector partitions, but any filter must be conservative: it may keep extra candidates, but it must not remove a true matching pair.

Approximate candidate-generation methods such as LSH, clustering, or product quantization can improve speed, but they change the task unless they are followed by exact verification and have a no-false-negative guarantee. Since the task asks to find all pairs, the main design should remain exact.

### 5.6 Fault Tolerance and Resumability

For long-running external-memory jobs, the block-pair scheduler should record completed tasks. Each block-pair output can be written to a temporary file and atomically renamed after completion. If the job is interrupted, the system can skip completed block pairs and resume from the remaining tasks. This is important because a full exact run may take hours or days on very large datasets.

### 5.7 Summary of External-Memory Design

The proposed external-memory implementation is an exact streaming system:

- stream vector blocks from SSD
- enumerate each block pair exactly once
- use safe distance-specific filters
- write result pairs to partitioned output files
- parallelize independent block-pair tasks
- keep memory bounded by block size rather than dataset size

This design satisfies the requirement of finding all vector pairs while still being practical when the vector set exceeds memory capacity.

## 6. Experiment Setup

We evaluated the algorithm on the same three datasets used in the DiskANN and HNSW experiments:

| Dataset | Training Size | Dimension | Metric |
|---|---:|---:|---|
| GloVe-25 | 1,183,514 | 25 | Angular |
| COCO-I2I-512 | 113,287 | 512 | Angular |
| Fashion-MNIST-784 | 60,000 | 784 | Euclidean |

For each dataset, we explicitly used `--sample-size 10000` and `--block-size 2048`. Each run therefore checks:

\[
\frac{10000\times9999}{2}=49,995,000
\]

unordered vector pairs.

The selected \(\epsilon\) values are dataset-specific because the distance scales differ across metrics and datasets. These values are used for experimental sensitivity analysis. The executable also supports a single user-provided threshold through `--epsilon`.

| Dataset | Epsilon Values |
|---|---|
| GloVe-25-Angular | 0.12, 0.16, 0.20, 0.24, 0.28 |
| COCO-512-Angular | 0.12, 0.18, 0.24, 0.30, 0.36 |
| Fashion-MNIST-784-Euclidean | 800, 1000, 1200, 1400, 1600 |

The reported throughput is:

\[
\text{throughput}=\frac{\text{number of pair checks}}{\text{elapsed time}}.
\]

## 7. Experiment Results

### 7.1 GloVe-25-Angular

| Epsilon | Pairs Found | Pair Checks | Time (s) | Throughput (checks/s) |
|---:|---:|---:|---:|---:|
| 0.12 | 0 | 49,995,000 | 0.8129 | 61,505,374 |
| 0.16 | 1 | 49,995,000 | 0.8019 | 62,347,032 |
| 0.20 | 4 | 49,995,000 | 0.8117 | 61,590,563 |
| 0.24 | 12 | 49,995,000 | 0.8411 | 59,442,570 |
| 0.28 | 40 | 49,995,000 | 0.8187 | 61,063,207 |

GloVe has the highest throughput because its dimension is only 25. The number of matching pairs is small under the tested thresholds, indicating that the first 10,000 vectors are sparse under the selected angular thresholds.

### 7.2 COCO-512-Angular

| Epsilon | Pairs Found | Pair Checks | Time (s) | Throughput (checks/s) |
|---:|---:|---:|---:|---:|
| 0.12 | 1 | 49,995,000 | 32.7772 | 1,525,299 |
| 0.18 | 1 | 49,995,000 | 32.7762 | 1,525,344 |
| 0.24 | 4 | 49,995,000 | 32.6418 | 1,531,627 |
| 0.30 | 57 | 49,995,000 | 32.7955 | 1,524,448 |
| 0.36 | 660 | 49,995,000 | 32.8276 | 1,522,958 |

COCO is significantly slower than GloVe because each angular comparison requires a 512-dimensional dot product. The throughput remains stable across different \(\epsilon\) values because angular distance computation does not use early termination in the current implementation.

### 7.3 Fashion-MNIST-784-Euclidean

| Epsilon | Pairs Found | Pair Checks | Time (s) | Throughput (checks/s) |
|---:|---:|---:|---:|---:|
| 800 | 8,231 | 49,995,000 | 12.8601 | 3,887,607 |
| 1000 | 46,703 | 49,995,000 | 17.0643 | 2,929,803 |
| 1200 | 188,673 | 49,995,000 | 21.8213 | 2,291,108 |
| 1400 | 588,207 | 49,995,000 | 27.0874 | 1,845,695 |
| 1600 | 1,456,552 | 49,995,000 | 33.8922 | 1,475,119 |

Fashion-MNIST shows the clearest impact of \(\epsilon\). As \(\epsilon\) increases from 800 to 1600, the number of matching pairs grows from 8,231 to 1,456,552. Meanwhile, throughput decreases from 3.89M checks/s to 1.48M checks/s. This is because early termination is more effective for small thresholds. When \(\epsilon\) is larger, more dimensions must be examined before a pair can be rejected.

## 8. Analysis

The results show three important patterns.

First, the output size is highly sensitive to the threshold. In the 10,000-vector Fashion-MNIST sample, doubling the threshold from 800 to 1600 increases the number of matching pairs by more than two orders of magnitude. This is important because threshold-based pair search can be output-bound as well as compute-bound.

Second, vector dimension strongly affects throughput. GloVe-25 achieves around 60M pair checks per second, while COCO-512 achieves around 1.52M pair checks per second. The difference mainly comes from the number of arithmetic operations required for each distance computation.

Third, distance-specific optimizations matter. For Euclidean distance, early termination reduces work when \(\epsilon\) is small. This explains why Fashion-MNIST is faster at smaller thresholds despite having a higher dimension than COCO. For angular distance, the current implementation avoids `acos`, but still computes the full dot product for each pair, so throughput is almost independent of \(\epsilon\).

Finally, the measured results separate discovery cost from output cost. In the default mode, the program counts matching pairs and measures computation throughput. When the optional output file is enabled, every matching pair is written to disk. This is closer to the literal enumeration output, but the running time then depends on storage bandwidth and output size. For fair algorithmic comparison, the tables report the compute-only mode; for correctness and actual enumeration, the output mode can be used.

## 9. Limitations and Future Work

The current implementation is exact and simple, but it has limitations:

- It uses a single-threaded scan.
- It is an in-memory implementation; the external-memory architecture is provided as a design rather than fully implemented code.
- It evaluates a prefix sample rather than the full dataset in the reported experiments, although the program can run on the full loaded training set when `--sample-size` is omitted.
- It does not persist pair IDs by default, because the output can be very large, but an optional output path can be supplied when enumeration output is needed.
- It does not yet use advanced candidate generation methods.

Possible future improvements include:

- Multi-threading block-pair evaluation.
- Memory-mapped vector blocks for large datasets.
- Output partitioning for very large result sets.
- Safe lower-bound filters for Euclidean distance.
- Approximate candidate generation followed by exact verification.

## 10. Report-Ready Summary

We implemented an exact threshold-based vector pair search algorithm. Given a selected vector set and a threshold \(\epsilon\), the algorithm finds all unordered pairs \((v_i,v_j)\), \(i<j\), satisfying \(dist(v_i,v_j)<\epsilon\). Unlike KNN search, this task has a data-dependent output size and a quadratic candidate space.

Our implementation uses a blocked all-pairs scan over vectors stored in the same `.fvecs` format as the DiskANN and HNSW experiments. For Euclidean distance, we compare squared distances with \(\epsilon^2\) and use early termination during distance accumulation. For angular distance, we precompute vector norms and transform the condition \(angular(v_i,v_j)<\epsilon\) into a cosine-threshold comparison, avoiding expensive `acos` calls. The program reports pair counts by default and can optionally write every discovered pair to disk in the format `epsilon i j`.

We evaluated the implementation on the first 10,000 training vectors of GloVe, COCO, and Fashion-MNIST. Each run checked 49,995,000 unordered vector pairs. GloVe-25-Angular achieved around 60M pair checks per second due to its low dimensionality. COCO-512-Angular achieved around 1.52M pair checks per second because each comparison requires a high-dimensional dot product. Fashion-MNIST-784-Euclidean achieved 3.89M checks/s at \(\epsilon=800\), but throughput dropped to 1.48M checks/s at \(\epsilon=1600\) because early termination became less effective. The number of matching pairs within the sampled set also increased sharply as \(\epsilon\) grew, especially on Fashion-MNIST.

For datasets larger than memory, the same algorithm can be extended using external-memory blocking. The system reads one or two vector blocks from SSD, computes intra-block or inter-block pairs, applies only safe exact filters, and writes matching pairs to append-only partition files. Block-pair tasks can be parallelized, and completed tasks can be recorded for resumability. This keeps memory usage bounded by the block size while preserving exactness and satisfying the requirement to find all qualifying vector pairs.
