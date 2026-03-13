# GGML FPGA Backend

This directory contains the implementation of the GGML backend for the FPGA accelerator (`/dev/fpga0`).

## Architecture Overview

The FPGA backend is integrated into GGML to offload high-performance matrix-vector multiplications (GEMV). It currently targets the **Q4_K** (4-bit) weight format with **Q8_K** (8-bit) activations.

### Offload Flow
1.  **Interception**: The CPU backend in `ggml/src/ggml-cpu/repack.cpp` intercepts calls to specific GEMV templates.
2.  **Dispatch**: If the FPGA backend is initialized, the call is redirected to `ggml_fpga_gemv_q4_K_8x8_q8_K`.
3.  **Hardware Execution**:
    -   `ioctl` configures the FPGA parameters and internal Look-Up Tables (LUT).
    -   `pwrite` uploads matrix and vector data to physical offsets.
    -   `pread` triggers the hardware computation and retrieves the result vector.

---

## Usage

To enable FPGA support, you must configure the build with `GGML_FPGA=ON`.

### Runtime Switch
You can enable FPGA offloading at runtime using the `GGML_FPGA_OFFLOAD` environment variable:
- **Disabled (Default)**: Unset or `export GGML_FPGA_OFFLOAD=0`
- **Enabled**: `export GGML_FPGA_OFFLOAD=1`

When disabled, the system will automatically use the standard CPU implementation.

---

## How to Build

```bash
mkdir build
cd build
cmake .. -DGGML_FPGA=ON -DGGML_BUILD_EXAMPLES=ON
cmake --build . --config Release -j
```

---

## Implementation Details

### Memory Map (Physical Offsets)
The driver communicates via standard Linux file I/O at specific offsets:
- **0 MB**: Configuration / LUT Generation (via `ioctl`).
- **64 MB**: Output Buffer (`S`). A `pread` at this offset triggers the FPGA hardware and retrieves results.
- **128 MB**: Matrix A Input (`VX`). Weights are uploaded here via `pwrite`.
- **144 MB**: Vector B Input (`VY`). Activations are uploaded here via `pwrite`.

### Buffer Allocation
Currently uses `malloc` to allocate host memory. Data is transferred to the FPGA during the computation phase using `pwrite`.

---

## How to Offload Functions

### 1. Offloading `ggml_gemv_q4_K_8x8_q8_K` (Implemented)
The implementation is found in `ggml-fpga.cpp`. It handles the file descriptor for `/dev/fpga0` and performs the necessary I/O operations.

The interception point is in `ggml/src/ggml-cpu/repack.cpp`:
```cpp
template <> void gemv<block_q4_K, 8, 8, GGML_TYPE_Q8_K>(...) {
    if (!ggml_fpga_gemv_q4_K_8x8_q8_K(...)) {
        ggml_gemv_q4_K_8x8_q8_K(...); // Fallback to CPU
    }
}
```

### 2. Offloading Other Functions
To offload a new operation:
1.  **Define the IOCTL**: If required, add new IOCTL definitions to `ggml-fpga.cpp`.
2.  **Implement the Offload Logic**: Create a new function in `ggml-fpga.cpp` that performs the `pwrite`/`pread` sequence.
3.  **Intercept in GGML**: Modify the corresponding function in `repack.cpp` or `ggml-cpu.cpp` to call your new FPGA function.

---

## Benchmarking & Demos

### Specialized FPGA Benchmark
We provide a standalone benchmark tool to measure raw hardware performance.

```bash
# Must be run with sudo to access /dev/fpga0
sudo ./build/bin/llama-fpga-benchmark
```

### Full Model Benchmark (`llama-bench`)
Verify the integration using `llama-bench` with a Q4_K model. Use batch size 1 to trigger the GEMV path.

```bash
sudo ./build/bin/llama-bench -m models/your-model-q4_k_m.gguf -p 0 -n 128 -b 1
```
