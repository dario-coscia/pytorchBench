# pytorchBench
A simple speed benchmark of useful torch functions for different hardware.

## Simple start

**To run the benchmark on CPU** for all the provided torch functions call
```bash
python benchmark.py --device='cpu' --test='all'
```

**For help on available features** run
```bash
python benchmark.py -h
```

## Benchmark Specifics 
Here we provide the specifics for the different possible benchmarks. The output is log-on terminal. 
### Available Precisions
Currently, the available precisions for benchmarking are integer 8-bit `int8`,  integer 16-bit `int16`, integer 32-bit `int32`, integer 64-bit `int64`, float 16-bit`fp16`, bfloat 16-bit `bfloat16`, float 32-bit `fp32`, float 64-bit `fp64`.
### Available Functions
Currently, the available functions for benchmarking are `torch.mm`, `torch.linalg.svd`, `torch.linalg.qr`, `torch.linalg.inv`, `torch.linalg.det`, `torch.linalg.qr`,
        `torch.linalg.pinv`, `torch.linalg.norm`, `torch.linalg.cond`.
### Available Accelerators
Currently, the available accelerators for benchmarking are `cpu`, `cuda`, `mps`.    
