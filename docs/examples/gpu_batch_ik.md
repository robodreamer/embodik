# GPU Batch IK Example Overview

Overview for `examples/04_gpu_batch_ik.py`.

## What It Demonstrates

- Batched velocity IK benchmark
- CPU sequential solve vs GPU parallel solve using CusADi
- Throughput scaling with batch size

## Run

```bash
python3 examples/04_gpu_batch_ik.py --batch_sizes 10 100 1000
```

Use `--use_gpu` after setting up CusADi codegen artifacts.
