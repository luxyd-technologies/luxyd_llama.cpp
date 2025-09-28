# Compiling in Windows machine

To compile for tracing the code in CPU purposes, use command below ( Disabling the CUDA)

```sh
cmake -B build -DGGML_CUDA=OFF
cmake --build build --config Release
```

If you want to run in a profiler, use this instead

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Debug -DGGML_CUDA=OFF
cmake --build build --config Debug
```

# Running the code

Run with `llama-run.exe`, if curl is not installed, download the weight at [Here](https://huggingface.co/DevsDoCode/LLama-3-8b-Uncensored-Q4_K_M-GGUF/tree/main)
and run like below

```sh
llama-run.exe llama-3-8b-uncensored.Q4_K_M.gguf
```

or

```sh
llama-run.exe hf://DevsDoCode/LLama-3-8b-Uncensored-Q4_K_M-GGUF
```

# About the code code

I deleted the AVX512 region of the gemv so that it use the generic gemm for us to trace the code.