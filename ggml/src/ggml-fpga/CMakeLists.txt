message(STATUS "Using FPGA backend")

ggml_add_backend_library(ggml-fpga
                         ggml-fpga.cpp
                        )
