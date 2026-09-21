# AI Architecture and Systems Tutorial

This tutorial connects six layers of modern AI systems: **Microarchitecture → Kernel → Compiler → Architecture → System → Algorithms**. The outline below uses consolidated sections for navigation; the complete curriculum and published material live in [CONTENTS.md](CONTENTS.md).

**Maintainers:** [Yufeng Gu](https://yufeng98.github.io/) and [Alireza Khadem](https://www.linkedin.com/in/arkhadem)

| Module | Primary question | Scope |
|---|---|---|
| [Microarchitecture](#microarchitecture) | How does GEMM execute inside an accelerator? | PE, systolic array, memory hierarchy, dataflow, roofline, and operand delivery. |
| [Kernel](#kernel) | How are operators implemented efficiently? | Layout, tiling, pipelines, matrix instructions, libraries, and device-specific kernel programming. |
| [Compiler](#compiler) | How does a model become an executable device program? | Graph/IR lowering, scheduling, memory planning, placement/routing, and executable artifacts. |
| [Architecture](#architecture) | How are hardware resources organized? | Accelerator taxonomy, programming models, fabrics, generations, power/cooling, and design challenges. |
| [System](#system) | How are workloads executed and operated? | Distributed training, serving, post-training, runtimes, heterogeneous fleets, and production infrastructure. |
| [Algorithms](#algorithms) | What computation does the model require? | Model structures, training objectives, generation, compression, and model-family evolution. |

## Microarchitecture

- [Computation Foundations and AI Accelerator Execution](CONTENTS.md#computation-foundations)
- [Numerical Representation, Arithmetic, and Precision](CONTENTS.md#numerical-representation)
- [Execution Organization and Compute Units](CONTENTS.md#execution-and-compute-units)
- [Dataflow, Reuse, and the TPU v1 Case Study](CONTENTS.md#dataflow-and-tpu)
- [Memory Hierarchy](CONTENTS.md#memory-hierarchy)
- [Data Movement, On-Chip Interconnect, and Synchronization](CONTENTS.md#data-movement-and-interconnect)
- [Performance Bounds and Hierarchical GEMM Mapping](CONTENTS.md#performance-and-gemm-mapping)
- [Matrix-Engine Evolution and Non-GEMM Support](CONTENTS.md#matrix-engine-evolution)
- [Hardware Modeling, Implementation, and Verification](CONTENTS.md#modeling-and-verification)

## Kernel

- [Programming Fundamentals and Tensor Layout](CONTENTS.md#programming-and-layout)
- [GEMM Optimization from Naive to Advanced](CONTENTS.md#gemm-optimization)
- [Elementwise Operations, Reductions, and Fusion](CONTENTS.md#elementwise-reduction-and-fusion)
- [Attention and KV-Cache Kernels](CONTENTS.md#attention-and-kv-cache)
- [Sparse, MoE, and Quantized Kernels](CONTENTS.md#sparse-moe-and-quantization)
- [Communication and Non-LLM Kernels](CONTENTS.md#communication-and-non-llm)
- [Kernel Languages and Hardware-Specific Programming](CONTENTS.md#kernel-programming-stacks)
- [Profiling, Benchmarking, and Correctness](CONTENTS.md#profiling-and-correctness)
- [Operator-to-Model Integration](CONTENTS.md#operator-to-model-integration)

## Compiler

- [Compilation Stack, Frontends, and Graph Capture](CONTENTS.md#stack-and-frontends)
- [IR, Autodiff, and Graph Optimization](CONTENTS.md#ir-autodiff-and-graph-optimization)
- [Loop, Layout, and Memory Optimization](CONTENTS.md#loop-layout-and-memory)
- [Code Generation and Runtime Specialization](CONTENTS.md#codegen-and-runtime-specialization)
- [Cost Models, Autotuning, and Distributed Compilation](CONTENTS.md#cost-models-and-distributed-compilation)
- [PyTorch, JAX, and XLA](CONTENTS.md#pytorch-jax-and-xla)
- [MLIR, TVM, and Deployment Stacks](CONTENTS.md#mlir-tvm-and-deployment)
- [Accelerator-Specific and Memory-Centric Compilation](CONTENTS.md#accelerator-specific-compilation)
- [Artifacts, Validation, Debugging, and Case Studies](CONTENTS.md#artifacts-and-validation)

## Architecture

- [Context, Scope, and Evidence Standards](CONTENTS.md#context-and-evidence)
- [Accelerator Taxonomy and Architecture Classes](CONTENTS.md#accelerator-taxonomy)
- [Cross-Category Compute, Memory, and Programming Models](CONTENTS.md#compute-memory-and-programming)
- [Packaging, Host Memory, and Memory-Centric Extensions](CONTENTS.md#packaging-and-memory-extensions)
- [Deployment Hierarchy and Interconnect Domains](CONTENTS.md#deployment-and-interconnect)
- [Collective Communication](CONTENTS.md#collective-communication)
- [Generational Evolution and Cross-Generation Comparison](CONTENTS.md#generational-evolution)
- [Power Delivery and Cooling](CONTENTS.md#power-and-cooling)
- [Future Directions and Architectural Evaluation](CONTENTS.md#future-directions-and-evaluation)

## System

- [Runtime Foundations, Execution Chain, and Capacity Planning](CONTENTS.md#runtime-and-capacity)
- [Distributed Runtime and Parallelism Strategies](CONTENTS.md#distributed-parallelism)
- [Training Memory, Data, Runtime, and Numerical Stability](CONTENTS.md#training-memory-data-and-runtime)
- [Resilient Distributed Training and Frameworks](CONTENTS.md#resilient-training)
- [Inference Lifecycle, Scheduling, and Caching](CONTENTS.md#inference-lifecycle-and-caching)
- [Disaggregated and Distributed Serving](CONTENTS.md#distributed-serving)
- [Optimized Model Serving](CONTENTS.md#optimized-model-serving)
- [Multi-Model Serving and Backend Ecosystems](CONTENTS.md#multi-model-serving)
- [RL Post-Training Infrastructure and Correctness](CONTENTS.md#rl-infrastructure)
- [Multimodal, Generative, and Compound AI Systems](CONTENTS.md#multimodal-and-compound-systems)
- [Production Orchestration, Efficiency, Security, and Observability](CONTENTS.md#production-operations)
- [End-to-End Systems Practice](CONTENTS.md#end-to-end-systems-practice)

## Algorithms

- [Foundations, Lifecycle, and Input Representation](CONTENTS.md#foundations-and-lifecycle)
- [Transformers, Attention, and Sequence Models](CONTENTS.md#transformers-and-attention)
- [Mixture of Experts](CONTENTS.md#mixture-of-experts)
- [Training, Fine-Tuning, Alignment, and Reasoning](CONTENTS.md#training-alignment-and-reasoning)
- [Inference and Generation Strategies](CONTENTS.md#inference-and-generation)
- [Quantization, Sparsity, and Compression](CONTENTS.md#quantization-and-compression)
- [Multimodal, Generative, Retrieval, Agent, and Non-LLM Workloads](CONTENTS.md#multimodal-agents-and-non-llm)
- [Distributed Learning and Model Evolution](CONTENTS.md#distributed-learning-and-evolution)
- [Model-Family Case Studies](CONTENTS.md#model-family-case-studies)
- [Model Evaluation and Cross-Layer Trade-offs](CONTENTS.md#evaluation-and-trade-offs)

The generated tutorial is available in [`docs/`](docs/) and at [yufeng98.github.io/public/blogs/ai-architecture-system/docs/](https://yufeng98.github.io/public/blogs/ai-architecture-system/docs/).
