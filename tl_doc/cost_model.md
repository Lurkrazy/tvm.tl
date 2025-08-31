# TVM.TL Cost Model Documentation

## Overview

The TVM.TL (Tensor Language) cost model is a sophisticated performance analysis system designed to estimate the execution cost of TL programs on GPU hardware. It uses a roofline model approach combined with detailed memory usage analysis to provide accurate performance predictions for tensor operations.

## Architecture

The cost model consists of two primary components:

### 1. Python Interface (`python/tvm/tl/cost_model.py`)

```python
def tile_costmodel(f: tvm.tir.PrimFunc):
    return _ffi_api.inst_cnt(f)
```

The Python interface provides a simple entry point for cost analysis. Currently, there is an implementation gap where the Python code calls `_ffi_api.inst_cnt(f)` but only `tl.highlevel_costmodel` is registered in the C++ backend.

### 2. C++ Backend (`src/tl/cost_model/tile_costmodel.cc`)

The C++ implementation contains two main analysis engines:

#### OpCostVisitor
Traverses the TIR (Tensor Intermediate Representation) and analyzes operations to compute costs across multiple categories.

#### MemUseCollector  
Tracks memory resource usage throughout program execution and computes GPU occupancy metrics.

## Cost Categories

The cost model tracks five distinct operation categories defined in `OpCost::OpCostField`:

### kGmemAccess - Global Memory Access
- **Purpose**: Tracks accesses to GPU global memory (device memory)
- **Impact**: Global memory has high latency (~400-600 cycles) but high bandwidth
- **Calculation**: Number of bytes transferred to/from global memory

### kSmemAccess - Shared Memory Access  
- **Purpose**: Tracks accesses to on-chip shared memory within thread blocks
- **Impact**: Low latency (~20-30 cycles) but limited capacity (48-164KB per SM)
- **Calculation**: Number of bytes transferred to/from shared memory

### kFp16TensorCore - FP16 Tensor Core Operations
- **Purpose**: Tracks mixed-precision matrix operations using Tensor Cores
- **Impact**: Highest throughput operations (up to 170+ TFLOPS on modern GPUs)
- **Calculation**: Number of matrix multiply-accumulate operations (M×N×K×2)

### kFloatSIMT - Floating Point SIMT Operations
- **Purpose**: Tracks general floating-point arithmetic on CUDA cores
- **Impact**: Lower throughput than Tensor Cores but more flexible
- **Calculation**: Number of floating-point operations

### kIntSIMT - Integer SIMT Operations  
- **Purpose**: Tracks integer arithmetic operations on CUDA cores
- **Impact**: Similar performance to floating-point for most operations
- **Calculation**: Number of integer operations

## Memory Usage Analysis

### Memory Tracking (`MemUseCollector`)

The memory collector analyzes buffer allocation and usage patterns to compute:

1. **Shared Memory Usage**: Tracks buffers in `shared` and `shared.dyn` scopes
2. **Register Usage**: Tracks buffers in `local` and `local.fragment` scopes  
3. **Pipelined Buffers**: Special handling for buffers used in pipelined loops

### Memory Capacity Calculation

```cpp
class MemCap {
public:
  int64_t smem_, regs_, num_threads_;
};
```

- **Shared Memory**: Maximum concurrent usage across program execution
- **Registers**: Per-thread register usage (accounting for fragment distribution)
- **Thread Count**: Number of threads per thread block

### GPU Occupancy Model

The cost model computes active parallelism using hardware constraints:

```cpp
// GPU hardware limits (example for modern GPUs)
int reg_occupy = (64 << 10) / (mem.num_threads_ * (mem.regs_ / 4));    // 64KB registers per SM
int smem_occupy = (96 << 10) / mem.smem_;                              // 96KB shared memory per SM
int active_blocks = std::min(reg_occupy, smem_occupy);                 // Bottleneck resource
int active_warps = (mem.num_threads_ / 32) * active_blocks;            // 32 threads per warp
```

## Roofline Performance Model

### Theoretical Peak Performance

The cost model uses hardware-specific performance assumptions:

```cpp
double tc_tflops = 170.0 * 1e9;  // 170 TFLOPS tensor core peak throughput
```

### Performance Estimation

For compute-bound workloads (primarily Tensor Core operations):

```cpp
double tc_times_ms = cost_global.num_block * cost.get(OpCost::kFp16TensorCore) / tc_tflops;
```

This calculates execution time based on:
- **num_block**: Total number of thread blocks launched
- **Tensor Core operations**: Total FP16 operations across all blocks  
- **Peak throughput**: Hardware-specific peak performance

### Memory Bandwidth Considerations

While not explicitly implemented in the current version, a complete roofline model would also consider:
- Global memory bandwidth limits (~1TB/s on modern GPUs)
- Memory-bound vs compute-bound operation classification
- Arithmetic intensity analysis

## Operation-Specific Cost Models

### GEMM Operations (`src/tl/op/gemm.cc`)

```cpp
OpCost Gemm::GetOpCost(const Target& target, size_t block_size, arith::Analyzer* analyzer) const {
  OpCost cost;
  auto [warp_m, warp_n] = ComputeWarpPartition(block_size / 32, target);
  
  // Shared memory access costs
  if (A.scope() == "shared" || A.scope() == "shared.dyn") {
    cost.update(OpCost::kSmemAccess, M * K * warp_n * A->dtype.bytes());
  }
  if (B.scope() == "shared" || B.scope() == "shared.dyn") {
    cost.update(OpCost::kSmemAccess, N * K * warp_n * B->dtype.bytes());
  }
  
  // Tensor core computation cost
  cost.update(OpCost::kFp16TensorCore, M * N * K * 2);
  return cost;
}
```

### Element-wise Operations (`src/tl/op/elem.cc`)

Copy and Fill operations delegate to `ParallelOp` for cost calculation, representing them as SIMT operations.

### Parallel Operations (`src/tl/op/parallel.cc`)

The `ParallelOp` cost model uses a visitor pattern to analyze nested expressions and compute costs for general parallel computations.

## Usage and Integration

### Compiler Integration

The cost model is integrated into the TL compilation pipeline:

```python
# In tvm/tl/engine.py
def compile_tl_func(func):
    # ... setup ...
    tl.cost_model.tile_costmodel(mod["main"])  # Cost analysis
    # ... continue compilation ...
```

### Output Format

The cost model produces detailed analysis output:

```
Num blocks: 64
Start OpCost Fields.
  Cost gmem access: 1048576
  Cost smem access: 524288  
  Cost tensorcore : 134217728 roofline 2.34 ms
  Cost flaot inst: 0
End OpCost Fields.
Use regs: 128
Use smem: 49152
Active warps: 16
Active blocks: 8
```

## Implementation Details

### Loop Analysis

The `OpCostVisitor` handles different loop types:

```cpp
Stmt VisitStmt_(const ForNode* node) final {
  if (node->kind == ForKind::kParallel) {
    return GetRef<Stmt>(node);  // Skip parallel loops (handled by thread mapping)
  } else {
    // Calculate loop extent and multiply operation costs
    auto loop_bound = analyzer_->const_int_bound(node->extent);
    int64_t loop_mean_value = (loop_bound->max_value + loop_bound->min_value) / 2;
    cur_loop_extent_ *= loop_mean_value;
    // ... visit body ...
    cur_loop_extent_ /= loop_mean_value;
  }
}
```

### Thread Block Analysis

The visitor tracks GPU thread hierarchy:

```cpp
Stmt VisitStmt_(const AttrStmtNode* node) final {
  if (node->attr_key == tir::attr::thread_extent) {
    IterVar iv = Downcast<IterVar>(node->node);
    if (iv->thread_tag == "blockIdx.x" || iv->thread_tag == "blockIdx.y" || iv->thread_tag == "blockIdx.z") {
      // Track grid dimensions
      grid_extent_ *= *extent_ptr;
    } else if (iv->thread_tag == "threadIdx.x") {
      // Track block size
      block_extent_ = *extent_ptr;
    }
  }
}
```

### Pipelined Buffer Support

The memory collector handles software pipelining:

```cpp
int64_t GetBufferbytes(Buffer buffer) {
  int bytes = buffer->dtype.bytes();
  PrimExpr num_elems = 1;
  for (auto dim : buffer->shape) num_elems *= dim;
  auto int_p = as_const_int(num_elems * bytes);
  
  if (pipelined_buffers_.count(buffer))
    return *int_p * num_stages_;  // Account for pipeline stages
  else
    return *int_p;
}
```

## Extending the Cost Model

### Adding New Operation Types

To add cost analysis for a new operation:

1. **Inherit from Operator** in `src/tl/op/`:
```cpp
class MyOp : public Operator {
public:
  OpCost GetOpCost(const Target& target, size_t block_size, arith::Analyzer* analyzer) const override;
};
```

2. **Implement GetOpCost**:
```cpp
OpCost MyOp::GetOpCost(const Target& target, size_t block_size, arith::Analyzer* analyzer) const {
  OpCost cost;
  // Analyze operation and update cost categories
  cost.update(OpCost::kFloatSIMT, num_float_ops);
  cost.update(OpCost::kGmemAccess, num_bytes_transferred);
  return cost;
}
```

3. **Register the Operation**:
```cpp
TIR_REGISTER_TL_OP(MyOp, my_op)
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));
```

### Customizing Hardware Parameters

Hardware-specific parameters can be adjusted:

```cpp
// In tile_costmodel.cc, modify these values for different GPU architectures:
double tc_tflops = 170.0 * 1e9;      // Tensor core peak throughput
int reg_capacity = (64 << 10);       // Register file capacity per SM  
int smem_capacity = (96 << 10);      // Shared memory capacity per SM
```

## Known Issues and Future Work

### Implementation Gap

**Current Issue**: The Python interface calls `_ffi_api.inst_cnt(f)` but only `tl.highlevel_costmodel` is registered.

**Resolution**: Either:
1. Register `tl.inst_cnt` to map to the existing cost model, or
2. Update Python code to call `_ffi_api.highlevel_costmodel(f)`

### Memory Bandwidth Modeling

**Missing Feature**: The current implementation focuses on compute costs but doesn't model memory bandwidth limitations.

**Future Enhancement**: Add memory bandwidth analysis:
```cpp
// Proposed addition
double gmem_bandwidth = 1000.0 * 1e9;  // 1TB/s global memory bandwidth
double gmem_time_ms = cost.get(OpCost::kGmemAccess) / gmem_bandwidth;
double execution_time = std::max(tc_times_ms, gmem_time_ms);  // Bottleneck analysis
```

### Target-Specific Parameters

**Current Limitation**: Hardware parameters are hardcoded.

**Future Enhancement**: Make parameters target-dependent:
```cpp
// Proposed enhancement
struct GPUSpecs {
  double tensor_core_tflops;
  int64_t smem_capacity;
  int64_t reg_capacity;
  double gmem_bandwidth;
};

GPUSpecs GetGPUSpecs(const Target& target) {
  if (target.arch == "sm_90") return {170.0e9, 96*1024, 64*1024, 1000.0e9};
  if (target.arch == "sm_80") return {156.0e9, 96*1024, 64*1024, 900.0e9};
  // ... other architectures
}
```

## References and Theory

### Roofline Model
The roofline model, introduced by Williams et al., provides a visual and analytical framework for understanding application performance relative to hardware limitations. It plots performance (FLOPS) against arithmetic intensity (operations per byte) to identify whether applications are compute-bound or memory-bound.

### Key Concepts:
- **Arithmetic Intensity**: Ratio of floating-point operations to memory traffic
- **Compute Roof**: Peak computational throughput of the hardware
- **Memory Roof**: Peak memory bandwidth × arithmetic intensity
- **Performance Bound**: min(compute_roof, memory_roof)

### GPU Memory Hierarchy
Modern GPUs have a complex memory hierarchy:
- **Registers**: Fastest, private per thread (~1 cycle latency)
- **Shared Memory**: Fast, shared within thread block (~20-30 cycles)  
- **L1/L2 Cache**: Transparent caching layers
- **Global Memory**: Largest capacity, highest latency (~400-600 cycles)

### Tensor Cores
Specialized matrix computation units introduced in Volta architecture:
- **Mixed Precision**: FP16 inputs, FP32 accumulation
- **High Throughput**: 170+ TFLOPS on modern architectures
- **Structured Computation**: Optimized for 16×16 matrix tiles

This cost model leverages these hardware characteristics to provide accurate performance estimates for TL programs, enabling informed optimization decisions during compilation.