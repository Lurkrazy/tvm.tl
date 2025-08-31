# TVM.TL Documentation

This directory contains comprehensive documentation for TVM.TL (Tensor Language), including language reference, cost model analysis, and implementation guides.

## Quick Start

For users new to TVM.TL:
1. Start with [Language Reference](language_ref.md) to understand the language constructs
2. Explore the cost model with [Cost Model Documentation](cost_model.md)

For developers working on the cost model:
1. Read [Cost Model Theory](cost_model_theory.md) for mathematical foundations
2. Follow [Cost Model Implementation Guide](cost_model_implementation.md) to extend functionality

## Document Overview

### [Language Reference](language_ref.md)
- Core TL language constructs (`T.Kernel`, `T.alloc_shared`, etc.)
- Usage patterns and examples
- API reference for all TL operations

### [Cost Model Documentation](cost_model.md)  
- Architecture overview of the cost model system
- Detailed explanation of cost categories and memory analysis
- Roofline performance model implementation
- Usage examples and integration patterns
- Known issues and future enhancements

### [Cost Model Theory](cost_model_theory.md)
- Mathematical framework underlying the cost model
- Roofline model formulation and equations
- GPU occupancy and memory hierarchy analysis
- Performance bottleneck analysis methodologies
- Advanced optimization considerations

### [Cost Model Implementation Guide](cost_model_implementation.md)
- Step-by-step guide for extending the cost model
- Adding cost models for new operations
- Debugging common issues
- Testing and validation procedures
- Performance optimization tips

## Key Features Documented

### Cost Model Capabilities
- **Multi-category Cost Analysis**: Tracks global memory, shared memory, tensor core, and SIMT operations
- **Memory Occupancy Modeling**: Analyzes register and shared memory usage for GPU occupancy calculation
- **Roofline Performance Estimation**: Provides execution time estimates based on hardware characteristics
- **Pipelined Buffer Support**: Handles software pipelining with multi-stage buffers

### Implementation Highlights
- **Modular Design**: Extensible architecture for adding new operation cost models
- **Hardware Abstraction**: Target-specific parameters for different GPU architectures
- **Integration with TL Compiler**: Seamlessly integrated into the compilation pipeline
- **Detailed Analysis Output**: Comprehensive breakdown of performance bottlenecks

## Examples and Usage

### Basic Cost Analysis
```python
import tvm
from tvm import tl

# Define TL program
@tl.prim_func
def my_kernel(...):
    # TL operations
    pass

# Analyze costs
mod = tvm.IRModule({"main": my_kernel})
target = tvm.target.Target("cuda")
mod = tir.transform.BindTarget(target)(mod)
tl.cost_model.tile_costmodel(mod["main"])
```

### Extending Cost Models
```cpp
// Add cost model for new operation
class MyOp : public Operator {
public:
  OpCost GetOpCost(const Target& target, size_t block_size, 
                   arith::Analyzer* analyzer) const override {
    OpCost cost;
    // Analyze operation characteristics
    cost.update(OpCost::kFloatSIMT, num_operations);
    return cost;
  }
};
```

## Contributing

When contributing to TL documentation:
1. Follow the established structure and style
2. Include practical examples alongside theoretical explanations
3. Update all relevant documents when making changes
4. Test code examples to ensure they work correctly

## Related Resources

- [TVM Documentation](https://tvm.apache.org/docs/)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [Roofline Model Paper](https://people.eecs.berkeley.edu/~kubitron/cs252/handouts/papers/RooflineVyNoYellow.pdf)

This documentation suite provides everything needed to understand, use, and extend the TVM.TL cost model system.