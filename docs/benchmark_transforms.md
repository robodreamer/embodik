# Transform Performance Benchmarks

EmbodiK provides native transform operations (`r2q`, `q2r`, `Rotation`) that replace SciPy in hot paths. This document summarizes performance validation.

## Running Benchmarks

```bash
pixi run benchmark-transforms
```

Options:
- `--latency-iters N` — iterations for latency benchmarks (default: 50000)
- `--throughput N1 N2 ...` — batch sizes for throughput (default: 10000 100000)
- `--json FILE` — save results to JSON

## Baseline vs Native Results

Typical results (Linux, Python 3.12, 50k latency iters):

### Latency (control-loop style, median microseconds)

| Operation        | Native (embodiK) | SciPy   | Speedup |
|-----------------|------------------|---------|---------|
| r2q (matrix→quat) | ~1.6 µs        | ~39 µs  | ~24×    |
| q2r (quat→matrix) | ~3.1 µs        | ~6.8 µs | ~2.2×   |
| Rotation compose | ~1.4 µs          | —       | —       |
| Rotation inv     | ~0.6 µs          | —       | —       |
| Rotation apply   | ~1.0 µs          | —       | —       |

### Throughput (ops/sec, 100k conversions)

| Operation | Native (embodiK) | SciPy   | Speedup |
|----------|------------------|---------|---------|
| r2q      | ~690k            | ~27k    | ~25×    |
| q2r      | ~327k            | ~143k   | ~2.3×   |

## Acceptance Criteria (from plan)

- [x] No SciPy dependency in embodiK core transform utilities
- [x] Equal or better numerical behavior for supported APIs
- [x] Demonstrated speedup in at least one hot-path latency metric (r2q: ~24×)
- [x] Demonstrated speedup in at least one throughput metric (r2q: ~25×)
