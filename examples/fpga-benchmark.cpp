#include "ggml.h"
#include "ggml-fpga.h"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <cstdint>

// --- FPGA Hardware Definitions ---
#define QK_K 256
typedef struct {
    float   d;
    int8_t  qs[QK_K];
    int16_t bsums[QK_K/16];
} block_q8_K_kernel;

// Declare the generic CPU version for comparison
extern "C" {
    void ggml_gemv_q4_K_8x8_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
}

int main(int argc, char ** argv) {
    const char * path = (argc > 1) ? argv[1] : "/dev/fpga0";
    const int n = 4096;
    const int nc = 4096;
    const int iterations = 10;

    printf("GGML FPGA vs CPU GEMV Benchmark\n");
    printf("Function: ggml_gemv_q4_K_8x8_q8_K\n");
    printf("Device: %s\n", path);
    printf("Matrix: %d x %d, Vector: %d, Iterations: %d\n\n", nc, n, n, iterations);

    size_t vx_size = (size_t)(nc / 8) * (n / 256) * 1152;
    size_t vy_size = (size_t)(n / 256) * sizeof(block_q8_K_kernel);

    std::vector<uint8_t> vx(vx_size, 1);
    std::vector<uint8_t> vy(vy_size, 2);
    std::vector<float> s_fpga(nc, 0.0f);
    std::vector<float> s_cpu(nc, 0.0f);

    double ms_fpga = 0;
    double gflops_fpga = 0;

    // Initialize FPGA backend to set the device path
    ggml_backend_t backend = ggml_backend_fpga_init(path);
    if (!backend) {
	    printf("Failed to initialize FPGA backend for %s\n", path);
    }

    // --- FPGA Benchmark ---
    {
        // Force enable for benchmark
        setenv("GGML_FPGA_OFFLOAD", "1", 1);

        printf("Running FPGA Benchmark (Hardware Offload)...\n");
        auto start_fpga = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < iterations; i++) {
            if (!ggml_fpga_gemv_q4_K_8x8_q8_K(n, s_fpga.data(), 0, vx.data(), vy.data(), 0, nc)) {
                printf("FPGA execution failed at iteration %d\n", i);
                break;
            }
        }

        auto end_fpga = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff_fpga = end_fpga - start_fpga;
        ms_fpga = (diff_fpga.count() * 1000.0) / iterations;
        gflops_fpga = (2.0 * n * nc * iterations) / (diff_fpga.count() * 1e9);

        if (ms_fpga > 0) {
            printf("FPGA: %.4f ms/iter (%.4f GFLOPS)\n\n", ms_fpga, gflops_fpga);
        }
    }

    // --- CPU Benchmark ---
    printf("Running CPU Benchmark (Generic Implementation)...\n");
    auto start_cpu = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iterations; i++) {
        ggml_gemv_q4_K_8x8_q8_K_generic(n, s_cpu.data(), 0, vx.data(), vy.data(), 0, nc);
    }

    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff_cpu = end_cpu - start_cpu;
    double ms_cpu = (diff_cpu.count() * 1000.0) / iterations;
    double gflops_cpu = (2.0 * n * nc * iterations) / (diff_cpu.count() * 1e9);

    printf("CPU:  %.4f ms/iter (%.4f GFLOPS)\n\n", ms_cpu, gflops_cpu);

    // --- Comparison ---
    if (ms_fpga > 0) {
        printf("Comparison: FPGA is %.2fx %s than CPU\n",
               (ms_cpu > ms_fpga) ? (ms_cpu / ms_fpga) : (ms_fpga / ms_cpu),
               (ms_cpu > ms_fpga) ? "FASTER" : "SLOWER");
    }

    return 0;
}
