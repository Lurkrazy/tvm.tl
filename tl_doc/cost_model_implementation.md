# TVM.TL Cost Model: Implementation Guide

## Developer's Guide to Extending the Cost Model

This guide provides practical instructions for developers who want to extend, modify, or debug the TVM.TL cost model implementation.

## Quick Start

### Running the Cost Model

The cost model can be invoked during TL program compilation:

```python
import tvm
from tvm import tl

# Define your TL program
@tl.prim_func
def my_kernel(...):
    # Your TL code here
    pass

# Compile and analyze
mod = tvm.IRModule({"main": my_kernel})
target = tvm.target.Target("cuda")
mod = tir.transform.BindTarget(target)(mod)

# Run cost analysis
tl.cost_model.tile_costmodel(mod["main"])
```

### Expected Output

The cost model produces detailed analysis:
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

## Implementation Architecture

### File Organization

```
src/tl/cost_model/
├── tile_costmodel.cc          # Main cost model implementation
python/tvm/tl/
├── cost_model.py              # Python interface
├── _ffi_api.py               # FFI function registry
src/tl/op/
├── op.h                      # OpCost structure definition
├── op.cc                     # OpCost basic operations
├── gemm.cc                   # GEMM cost implementation
├── elem.cc                   # Element-wise operation costs
├── parallel.cc               # Parallel operation costs
└── ...                       # Other operation-specific costs
```

### Class Hierarchy

```cpp
// Base operation class
class Operator {
public:
  virtual OpCost GetOpCost(const Target& target, size_t block_size, 
                          arith::Analyzer* analyzer) const;
};

// Specific operation implementations
class Gemm : public Operator { ... };
class Copy : public Operator { ... };
class Fill : public Operator { ... };
class ParallelOp : public Operator { ... };
```

## Adding Cost Models for New Operations

### Step 1: Define the Operation Class

Create a new operation class in the appropriate file (e.g., `src/tl/op/my_new_op.cc`):

```cpp
#include "op.h"

namespace tvm {
namespace tl {

class MyNewOp : public Operator {
public:
  MyNewOp(Array<PrimExpr> args, BufferMap vmap);
  
  OpCost GetOpCost(const Target& target, size_t block_size, 
                   arith::Analyzer* analyzer) const override;
  
  // Other required methods...
  Stmt Lower(const LowerArgs& T, arith::Analyzer* analyzer) const override;

private:
  // Operation-specific data members
  Buffer input_buffer_;
  Buffer output_buffer_;
  PrimExpr some_parameter_;
};

} // namespace tl
} // namespace tvm
```

### Step 2: Implement Cost Analysis

```cpp
OpCost MyNewOp::GetOpCost(const Target& target, size_t block_size, 
                          arith::Analyzer* analyzer) const {
  OpCost cost;
  
  // Example: Analyze buffer sizes
  int64_t input_bytes = GetBufferBytes(input_buffer_);
  int64_t output_bytes = GetBufferBytes(output_buffer_);
  
  // Example: Add global memory access cost
  if (input_buffer_.scope() == "global") {
    cost.update(OpCost::kGmemAccess, input_bytes);
  }
  if (output_buffer_.scope() == "global") {
    cost.update(OpCost::kGmemAccess, output_bytes);
  }
  
  // Example: Add computational cost
  int64_t num_elements = GetNumElements(output_buffer_);
  if (IsFloatingPoint(output_buffer_->dtype)) {
    cost.update(OpCost::kFloatSIMT, num_elements);
  } else {
    cost.update(OpCost::kIntSIMT, num_elements);
  }
  
  return cost;
}
```

### Step 3: Register the Operation

```cpp
// At the end of your .cc file
TIR_REGISTER_TL_OP(MyNewOp, my_new_op)
    .set_num_inputs(2)  // Adjust based on your operation
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));
```

### Step 4: Helper Functions

Common helper functions for cost analysis:

```cpp
// Calculate buffer size in bytes
int64_t GetBufferBytes(const Buffer& buffer) {
  int bytes = buffer->dtype.bytes();
  PrimExpr num_elems = 1;
  for (auto dim : buffer->shape) {
    num_elems *= dim;
  }
  auto int_p = as_const_int(num_elems * bytes);
  ICHECK(int_p != nullptr) << "Expected static shape";
  return *int_p;
}

// Calculate number of elements
int64_t GetNumElements(const Buffer& buffer) {
  PrimExpr num_elems = 1;
  for (auto dim : buffer->shape) {
    num_elems *= dim;
  }
  auto int_p = as_const_int(num_elems);
  ICHECK(int_p != nullptr) << "Expected static shape";
  return *int_p;
}

// Check if operation involves floating point
bool IsFloatingPoint(const DataType& dtype) {
  return dtype.is_float();
}
```

## Modifying Existing Cost Models

### Updating Hardware Parameters

To update hardware specifications for different GPU architectures:

```cpp
// In tile_costmodel.cc, create architecture-specific parameters
struct GPUArchSpecs {
  double tensor_core_tflops;
  int64_t smem_capacity_bytes;
  int64_t reg_capacity_bytes;
  double gmem_bandwidth_bytes_per_sec;
};

GPUArchSpecs GetArchSpecs(const Target& target) {
  std::string arch = target->GetAttr<String>("arch").value();
  
  if (arch == "sm_90") {  // H100
    return {170.0e9, 96*1024, 64*1024, 1000.0e9};
  } else if (arch == "sm_80") {  // A100
    return {156.0e9, 96*1024, 64*1024, 900.0e9};
  } else if (arch == "sm_75") {  // V100
    return {125.0e9, 96*1024, 64*1024, 900.0e9};
  } else {
    // Default/fallback values
    return {100.0e9, 48*1024, 32*1024, 500.0e9};
  }
}
```

### Enhancing the Roofline Model

Add memory bandwidth analysis:

```cpp
// In the main cost analysis function
OpCostGlobal cost_global = OpCostVisitor::Collect(func);
OpCost cost = cost_global.cost;
auto arch_specs = GetArchSpecs(target);

// Compute time calculations
double tc_time_ms = cost_global.num_block * cost.get(OpCost::kFp16TensorCore) / arch_specs.tensor_core_tflops;
double gmem_time_ms = cost.get(OpCost::kGmemAccess) / arch_specs.gmem_bandwidth_bytes_per_sec;

// Bottleneck analysis
double bottleneck_time_ms = std::max(tc_time_ms, gmem_time_ms);
std::string bottleneck_type = (tc_time_ms > gmem_time_ms) ? "compute" : "memory";

std::cout << "Estimated execution time: " << bottleneck_time_ms << " ms (" 
          << bottleneck_type << " bound)" << std::endl;
```

## Debugging Cost Model Issues

### Common Problems and Solutions

#### Problem 1: Python Interface Mismatch

**Symptoms**: `AttributeError: 'NoneType' object has no attribute ...`

**Cause**: The Python code calls `_ffi_api.inst_cnt(f)` but only `tl.highlevel_costmodel` is registered.

**Solution**: Fix the FFI registration:

```cpp
// Option 1: Register the expected function name
TVM_REGISTER_GLOBAL("tl.inst_cnt").set_body_typed([](PrimFunc func) {
  // Delegate to existing implementation
  return highlevel_costmodel_impl(func);
});

// Option 2: Update Python code
// In python/tvm/tl/cost_model.py:
def tile_costmodel(f: tvm.tir.PrimFunc):
    return _ffi_api.highlevel_costmodel(f)  # Use correct function name
```

#### Problem 2: Cost Analysis Returns Zero

**Symptoms**: All cost fields show zero values.

**Debugging Steps**:

1. **Check operation parsing**:
```cpp
// Add debug output in OpCostVisitor::VisitStmt_(const EvaluateNode* node)
std::cout << "Parsing statement: " << stmt << std::endl;
auto op = ParseOperator(stmt, buffer_data_to_buffer_);
if (op == nullptr) {
  std::cout << "Failed to parse operation" << std::endl;
  return stmt;
}
std::cout << "Successfully parsed operation" << std::endl;
```

2. **Verify operation registration**:
```cpp
// Check if your operation is properly registered
auto op_map = Op::GetAttrMap<OpBuilderFunc>("TLOpBuilder");
std::cout << "Registered operations: " << op_map.size() << std::endl;
```

3. **Check buffer mapping**:
```cpp
// Ensure buffers are properly mapped
std::cout << "Buffer map size: " << buffer_data_to_buffer_.size() << std::endl;
for (const auto& [var, buffer] : buffer_data_to_buffer_) {
  std::cout << "Buffer: " << var << " -> " << buffer << std::endl;
}
```

#### Problem 3: Incorrect Memory Usage Analysis

**Symptoms**: Memory usage calculations seem wrong.

**Debugging Steps**:

1. **Check scope detection**:
```cpp
// In MemUseCollector::ComputeMemCap()
for (auto seq : linear_seq_) {
  for (auto var : event_map[seq.stmt].gen) {
    auto buffer = buffer_data_to_buffer_[GetRef<Var>(var)];
    std::cout << "Buffer scope: " << buffer.scope() << ", size: " << GetBufferbytes(buffer) << std::endl;
  }
}
```

2. **Verify pipelined buffer detection**:
```cpp
// Check if pipelined buffers are correctly identified
std::cout << "Pipelined buffers: " << pipelined_buffers_.size() << std::endl;
for (const auto& buf : pipelined_buffers_) {
  std::cout << "Pipelined: " << buf << std::endl;
}
```

### Debug Output Helpers

Add these utility functions for debugging:

```cpp
// Debug helper for OpCost
void PrintOpCost(const OpCost& cost, const std::string& prefix = "") {
  std::cout << prefix << "OpCost breakdown:" << std::endl;
  std::cout << prefix << "  GMEM: " << cost.get(OpCost::kGmemAccess) << std::endl;
  std::cout << prefix << "  SMEM: " << cost.get(OpCost::kSmemAccess) << std::endl;
  std::cout << prefix << "  TC:   " << cost.get(OpCost::kFp16TensorCore) << std::endl;
  std::cout << prefix << "  FSIMT:" << cost.get(OpCost::kFloatSIMT) << std::endl;
  std::cout << prefix << "  ISIMT:" << cost.get(OpCost::kIntSIMT) << std::endl;
}

// Debug helper for memory analysis
void PrintMemCap(const MemCap& mem, const std::string& prefix = "") {
  std::cout << prefix << "Memory usage:" << std::endl;
  std::cout << prefix << "  SMEM:    " << mem.smem_ << " bytes" << std::endl;
  std::cout << prefix << "  Regs:    " << mem.regs_ << " bytes" << std::endl;
  std::cout << prefix << "  Threads: " << mem.num_threads_ << std::endl;
}
```

## Testing Your Implementation

### Unit Testing

Create tests to verify your cost model implementation:

```cpp
// test_my_new_op_cost.cc
#include <gtest/gtest.h>
#include "src/tl/op/my_new_op.h"

TEST(MyNewOpCostTest, BasicCostCalculation) {
  // Setup test parameters
  Target target("cuda");
  size_t block_size = 256;
  arith::Analyzer analyzer;
  
  // Create test operation
  Array<PrimExpr> args = {/* test arguments */};
  BufferMap vmap = {/* test buffer map */};
  MyNewOp op(args, vmap);
  
  // Calculate cost
  OpCost cost = op.GetOpCost(target, block_size, &analyzer);
  
  // Verify results
  EXPECT_GT(cost.get(OpCost::kFloatSIMT), 0);
  EXPECT_GT(cost.get(OpCost::kGmemAccess), 0);
}
```

### Integration Testing

Test your cost model within the compilation pipeline:

```python
# test_cost_model_integration.py
import tvm
from tvm import tl

def test_my_new_op_cost():
    @tl.prim_func
    def test_kernel(...):
        # Use your new operation
        tl.my_new_op(...)
    
    # Compile and check cost analysis
    mod = tvm.IRModule({"main": test_kernel})
    target = tvm.target.Target("cuda")
    mod = tir.transform.BindTarget(target)(mod)
    
    # This should not crash and should produce reasonable output
    result = tl.cost_model.tile_costmodel(mod["main"])
    assert result is not None
```

## Performance Considerations

### Avoiding Common Pitfalls

1. **Expensive Runtime Analysis**: Avoid complex computations in cost analysis that might slow down compilation.

2. **Static Shape Requirements**: The cost model assumes static shapes. Handle dynamic shapes gracefully:
```cpp
auto extent_ptr = as_const_int(expr);
if (extent_ptr == nullptr) {
  // Handle dynamic case - maybe use heuristic or skip
  return cost;  // Return partial cost
}
```

3. **Target-Specific Code**: Make hardware assumptions explicit and configurable.

### Optimization Tips

1. **Cache Expensive Calculations**: Store frequently computed values.
2. **Early Exit**: Skip cost analysis for trivial operations.
3. **Batch Analysis**: Analyze multiple operations together when possible.

## Contributing Guidelines

### Code Style

Follow TVM's coding conventions:
- Use `snake_case` for variables and functions
- Use `PascalCase` for class names
- Include proper documentation comments
- Add appropriate error checking

### Documentation

When adding new cost models:
1. Document the mathematical model used
2. Explain hardware assumptions
3. Provide usage examples
4. Include performance characteristics

### Testing Requirements

All cost model changes should include:
- Unit tests for individual operations
- Integration tests with the compilation pipeline
- Performance regression tests
- Documentation updates

This implementation guide provides the foundation for extending and maintaining the TVM.TL cost model. The modular design allows for incremental improvements while maintaining compatibility with existing functionality.