<!-- disableFinding(LINE_OVER_80) -->
<!-- disableFinding(LIST_NO_LINE) -->
<!-- disableFinding(WHITESPACE_TRAILING) -->

# ReinsPIRe: High-Throughput Private Information Retrieval and Verifiable PIR

This repository provides the C++ implementation of **ReinsPIRe** (also referred to as InsPIRe 2.0), a high-throughput protocol for Private Information Retrieval (PIR) and Verifiable Private Information Retrieval (vPIR).
ReinsPIRe improves on the protocol the previous protocol, InsPIRe.

---

## 1. Research & System Overview

This repository contains the implementation of two protocols:

### A. ReinsPIRe: Private Information Retrieval (PIR)
PIR enables a client to privately fetch a record from an untrusted server hosting a database of $n$ records without revealing the query index. Some properties of ReinsPIRe include:

- **Server-side Preprocessing:** Supports an offline preprocessing phase where the server computes preprocessed material that can be used to answer online queries quickly.
- **Faster LWE-to-RLWE Packing:** Represents online phase of packing as simple matrix-vector multiplications over a power-of-two moduli, avoiding costly polynomial multiplications and enabling the use of native machine data types.
- **Smaller Preprocessing Material:** Reduces server's memory/disk footprint with smaller preprocessing material compared to prior work.
- **Homomorphic Polynomial Evaluation:** Uses polynomial evaluation as a second dimension of PIR, further reducing response sizes.

### B. vReinsPIRe: Verifiable Private Information Retrieval (vPIR)
vPIR extends PIR by providing cryptographic integrity against malicious or malfunctioning servers:
- **Tamper-Evident Responses:** In addition to the query response, the server generates cryptographic proofs, ensuring that the returned record was unmodified and matches the committed database.
- **Reusable Preprocessed Proof Material:** Preprocessed material can be reused for multiple queries, amortizing the preprocessing cost across many queries.
- **Client Proof Verification:** Client verifies the server's proof before decrypting the queried payload.

---

## 2. Prerequisites & Building

### Prerequisites
- **Operating System:** Linux (x86_64) or macOS
- **Compiler:** C++17 compatible compiler (GCC 10+ or Clang 12+)
- **Build System:** [Bazel](https://bazel.build/) 6.0+ / 7.0+ / 8.0+ / 9.0+ (supports standard Bazel and Bzlmod)
- **External Dependencies:** Fully self-contained. All dependencies (Abseil C++, Google Shell Encryption, Google Highway SIMD, Protobuf, GoogleTest, Google Benchmark) are automatically fetched and built by Bazel. No system libraries or manual installations are required.

### Building All Targets
To compile all libraries, binaries, test suites, and benchmarking tools:

```bash
bazel build -c opt //...
```

For peak performance on hardware supporting AVX-512, compile with the AVX-512 flag:

```bash
bazel build -c opt --copt=-march=skylake-avx512 //...
```

### Running the Test Suite
To run all 15 unit and integration test suites:

```bash
bazel test -c opt //...
```

*Note:* Compiling with `-c opt` enables compiler optimizations essential for cryptographic polynomial arithmetic and NTT/FFT transformations. Using the AVX-512 flag (e.g. `--copt=-march=skylake-avx512` or `--copt=-mavx512f`) enables Google Highway SIMD hardware acceleration for peak performance.

---

## 3. Running Artifact Metrics & Benchmarking Tools

The repository contains standalone benchmarking binaries tailored for empirical evaluation and reproducibility.

### 3.1. PIR Metrics Tool (`pir_metrics_tool`)
Measures preprocessing time, client query encryption, server online response latency, throughput, client decryption, and communication overhead.

```bash
bazel run -c opt //crypto/pir/benchmarking:pir_metrics_tool -- \
  --num_entries=4096 \
  --entry_size_multiple=1 \
  --degree=2048 \
  --interpolation_degree=2 \
  --iterations=5 \
  --use_disk_storage=false
```

**Common Flags:**
- `--num_entries` ($n$): Number of records in the database (e.g. `4096`, `65536`, `262144`, `1048576`). Note: `num_entries / interpolation_degree` must be divisible by `degree`.
- `--degree` ($d$): RLWE polynomial ring degree (e.g. `2048`, `4096`, `8192`).
- `--entry_size_multiple` ($\ell$): Size of each record in multiples of ring dimension.
- `--interpolation_degree` ($t$): Degree of polynomial interpolation (e.g. `2`, `4`, `8`).
- `--iterations`: Number of online timing iterations to average.
- `--use_disk_storage`: `true` to store preprocessed material on disk and mmap it, or `false` to keep in RAM.
- `--disk_storage_file`: Custom path for disk storage file (defaults to temporary directory).
- `--output_file`: Path to append formatted metric results (optional).

### 3.2. vPIR Metrics Tool (`vpir_metrics_tool`)
Evaluates verifiable PIR, measuring offline hint generation, offline proof generation/verification, online server proof evaluation, and client-side proof verification and decryption.

```bash
bazel run -c opt //crypto/vpir/benchmarking:vpir_metrics_tool -- \
  --num_entries=2048 \
  --entry_size_multiple=1 \
  --degree=2048 \
  --kappa=40 \
  --iterations=5
```

**Common Flags:**
- `--num_entries` ($n$): Number of records in the database (e.g. `2048`, `4096`, `16384`).
- `--degree` ($d$): RLWE polynomial ring degree (e.g. `2048`, `4096`).
- `--entry_size_multiple` ($\ell$): Size of database entries.
- `--kappa` ($\kappa$): Number of verifiable proof queries (e.g. `40` for standard statistical security).
- `--local_finalize`: If `true`, activates the local finalize optimization for proof computation.
- `--iterations`: Number of online timing iterations.

### 3.3. Packing and Gadget Decomposition Tool (`pir_packing_metrics_tool`)
Benchmarks the transformation cost of packing multiple LWE ciphertexts into RLWE ciphertexts and gadget decomposition:

```bash
bazel run -c opt //crypto/pir/benchmarking:pir_packing_metrics_tool -- \
  --degree=2048 \
  --pack_log_digit=19 \
  --pack_num_digits=2
```

**Common Flags:**
- `--degree` ($d$): RLWE polynomial ring degree (e.g. `2048`, `4096`).
- `--pack_log_digit`: Gadget decomposition base in $\log_2$ for packing (e.g. `19`).
- `--pack_num_digits`: Number of gadget decomposition digits for packing (e.g. `2`).
- `--iterations`: Number of online timing iterations.

---

## 4. Microbenchmarks (Google Benchmark)

For fine-grained microarchitectural analysis of individual cryptographic primitives:

```bash
# SIMD matrix-vector multiplication benchmarks
bazel run -c opt //crypto:matrix_benchmark

# Polynomial NTT/FFT and polynomial arithmetic benchmarks
bazel run -c opt //crypto:polynomial_benchmark

# Server online processing pipeline microbenchmarks
bazel run -c opt //crypto/pir/server:pir_server_benchmark

# Verifiable server proof processing microbenchmarks
bazel run -c opt //crypto/vpir/server:vpir_server_benchmark
```

---

## 5. Interpreting Metrics Output

When running `pir_metrics_tool` or `vpir_metrics_tool`, the tools print key performance metrics:

| Metric Key | Description | Unit |
| :--- | :--- | :--- |
| `db_size_mb` | Total raw database size | Megabytes (MB) |
| `prep_server_time_ms` | Server offline preprocessing runtime | Milliseconds (ms) |
| `client_query_gen_time_ms` | Client online query generation time | Milliseconds (ms) |
| `server_proc_resp_time_ms` | Server online response processing latency | Milliseconds (ms) |
| `client_proc_resp_time_ms` | Client response decryption & verification time | Milliseconds (ms) |
| `c2s_on_msg_bytes_formula` | Client-to-server online upload size | Bytes |
| `s2c_on_msg_bytes_formula` | Server-to-client online download size | Bytes |
| `verification_status` | Verification status of decrypted output | `SUCCESS` / `1` |

---

## 6. Directory Structure

```
├── MODULE.bazel                         # Bazel module configuration (Bzlmod)
├── WORKSPACE                            # Bazel workspace definition
├── BUILD.bazel                          # Root build package
├── crypto/                              # Core cryptographic library
│   ├── BUILD.bazel
│   ├── encryption.h / .cc               # RLWE & LWE symmetric encryption routines
│   ├── fft.h                            # Pure C++ radix-2 Cooley-Tukey FFT implementation
│   ├── polynomial.h / .cc               # Ring polynomial arithmetic & operations
│   ├── polynomial_fft.h / .cc           # Polynomial FFT / NTT evaluation & interpolation
│   ├── matrix.h / .cc                   # Google Highway SIMD-accelerated matrix operations
│   ├── lwes_to_rlwe.h / .cc             # LWE-to-RLWE ciphertext packing & dimension switching
│   ├── db_row_interpolation.h / .cc     # Fast Lagrange polynomial database interpolation
│   ├── file_util.h / .cc                # Portable POSIX file I/O & disk persistence abstraction
│   ├── status_macros.h                  # Abseil status and testing error propagation macros
│   ├── proto/                           # Protocol buffer schemas
│   │   ├── rlwe.proto, pir.proto, pir_server.proto
│   ├── pir/                             # Private Information Retrieval protocol
│   │   ├── pir_params.h / .cc           # Cryptographic parameters & validation
│   │   ├── client/                      # PIR Client implementation
│   │   ├── server/                      # PIR Server implementation & preprocessing
│   │   └── benchmarking/                # PIR metrics tools & benchmark binaries
│   └── vpir/                            # Verifiable Private Information Retrieval
│       ├── vpir_params.h / .cc          # vPIR parameters & multi-moduli setup
│       ├── client/                      # vPIR Client & proof verification
│       ├── server/                      # vPIR Server & proof generation
│       └── benchmarking/                # vPIR metrics tools & benchmark binaries
└── common/
    └── modular_int/                     # Modular integer arithmetic & Montgomery structures
```

---

## 7. License

Licensed under the [Apache License, Version 2.0](LICENSE).


## Disclaimer

This is not an officially supported Google product. The code is provided as-is,
with no guarantees of correctness or security.