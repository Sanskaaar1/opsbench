# GPU operator benchmark

Install the benchmark dependencies into the CUDA-capable Python environment:

```bash
python -m pip install -r requirements.txt
```

Run the full harness with one command:

```bash
./run_benchmark.sh --output-dir results
```

Outside Slurm, the launcher requests one `nvidia_h200_3g.71gb` MIG slice, eight CPUs, and 32 GiB host RAM. Inside an existing Slurm allocation, it runs directly there. Pass normal benchmark options through the launcher, for example `./run_benchmark.sh --operators matmul`.

The matmul benchmark sweeps one workload dimension at a time around `B=8, S=512, H=2048`: batch `1/8/32`, sequence `128/512/4096`, and hidden size `768/2048/8192`. It uses `A[B,S,H] @ B[B,H,H]`, alongside last-dimension softmax, linear+GELU with a shared `[H,H]` weight, and LayerNorm over `H`, in eager and compiled fp32/bf16 modes. Each configuration has five independent runs of 100 CUDA-event measurements after 25 warmups; plots are written to `results/plots/` with run-to-run error bars.

Each JSON case records the B/S/H workload, environment snapshot, modeled FLOPs, input/output payload bytes, arithmetic intensity (FLOPs/byte), raw per-run samples, aggregate latency, TFLOPS, and GB/s. A 128 MiB cache flush runs before each measurement but is excluded from the timed interval. The bytes model intentionally covers operator inputs and outputs; it is not a profiler-derived DRAM counter.
