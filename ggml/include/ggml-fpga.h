#pragma once

#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_fpga_init(const char * device_path);
GGML_BACKEND_API bool ggml_backend_is_fpga(ggml_backend_t backend);

GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_fpga_buffer_type(const char * device_path);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_fpga_reg(void);

// GEMV offload
GGML_BACKEND_API bool ggml_fpga_gemv_q4_K_8x8_q8_K(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc);

#ifdef  __cplusplus
}
#endif
