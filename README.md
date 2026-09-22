# AI Architecture and Systems Tutorial

This tutorial connects six layers of modern AI systems: **Microarchitecture → Kernel → Compiler → Architecture → System → Algorithms**. The outline below uses consolidated sections for navigation; the complete curriculum and published material live in [content/tutorial.md](content/tutorial.md).

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

- [Computation Foundations and AI Accelerator Execution](content/tutorial.md#computation-foundations)
- [Numerical Representation, Arithmetic, and Precision](content/tutorial.md#numerical-representation)
- [Execution Organization and Compute Units](content/tutorial.md#execution-and-compute-units)
- [Dataflow, Reuse, and the TPU v1 Case Study](content/tutorial.md#dataflow-and-tpu)
- [Memory Hierarchy](content/tutorial.md#memory-hierarchy)
- [Data Movement, On-Chip Interconnect, and Synchronization](content/tutorial.md#data-movement-and-interconnect)
- [Performance Bounds and Hierarchical GEMM Mapping](content/tutorial.md#performance-and-gemm-mapping)
- [Matrix-Engine Evolution and Non-GEMM Support](content/tutorial.md#matrix-engine-evolution)
- [Hardware Modeling, Implementation, and Verification](content/tutorial.md#modeling-and-verification)

## Kernel

- [Programming Fundamentals and Tensor Layout](content/tutorial.md#programming-and-layout)
- [GEMM Optimization from Naive to Advanced](content/tutorial.md#gemm-optimization)
- [Elementwise Operations, Reductions, and Fusion](content/tutorial.md#elementwise-reduction-and-fusion)
- [Attention and KV-Cache Kernels](content/tutorial.md#attention-and-kv-cache)
- [Sparse, MoE, and Quantized Kernels](content/tutorial.md#sparse-moe-and-quantization)
- [Communication and Non-LLM Kernels](content/tutorial.md#communication-and-non-llm)
- [Kernel Languages and Hardware-Specific Programming](content/tutorial.md#kernel-programming-stacks)
- [Profiling, Benchmarking, and Correctness](content/tutorial.md#profiling-and-correctness)
- [Operator-to-Model Integration](content/tutorial.md#operator-to-model-integration)

## Compiler

- [Compilation Stack, Frontends, and Graph Capture](content/tutorial.md#stack-and-frontends)
- [IR, Autodiff, and Graph Optimization](content/tutorial.md#ir-autodiff-and-graph-optimization)
- [Loop, Layout, and Memory Optimization](content/tutorial.md#loop-layout-and-memory)
- [Code Generation and Runtime Specialization](content/tutorial.md#codegen-and-runtime-specialization)
- [Cost Models, Autotuning, and Distributed Compilation](content/tutorial.md#cost-models-and-distributed-compilation)
- [PyTorch, JAX, and XLA](content/tutorial.md#pytorch-jax-and-xla)
- [MLIR, TVM, and Deployment Stacks](content/tutorial.md#mlir-tvm-and-deployment)
- [Accelerator-Specific and Memory-Centric Compilation](content/tutorial.md#accelerator-specific-compilation)
- [Artifacts, Validation, Debugging, and Case Studies](content/tutorial.md#artifacts-and-validation)

## Architecture

- [Context, Scope, and Evidence Standards](content/tutorial.md#context-and-evidence)
- [Accelerator Taxonomy and Architecture Classes](content/tutorial.md#accelerator-taxonomy)
- [Cross-Category Compute, Memory, and Programming Models](content/tutorial.md#compute-memory-and-programming)
- [Packaging, Host Memory, and Memory-Centric Extensions](content/tutorial.md#packaging-and-memory-extensions)
- [Deployment Hierarchy and Interconnect Domains](content/tutorial.md#deployment-and-interconnect)
- [Collective Communication](content/tutorial.md#collective-communication)
- [Generational Evolution and Cross-Generation Comparison](content/tutorial.md#generational-evolution)
- [Power Delivery and Cooling](content/tutorial.md#power-and-cooling)
- [Future Directions and Architectural Evaluation](content/tutorial.md#future-directions-and-evaluation)

## System

- [Runtime Foundations, Execution Chain, and Capacity Planning](content/tutorial.md#runtime-and-capacity)
- [Distributed Runtime and Parallelism Strategies](content/tutorial.md#distributed-parallelism)
- [Training Memory, Data, Runtime, and Numerical Stability](content/tutorial.md#training-memory-data-and-runtime)
- [Resilient Distributed Training and Frameworks](content/tutorial.md#resilient-training)
- [Inference Lifecycle, Scheduling, and Caching](content/tutorial.md#inference-lifecycle-and-caching)
- [Disaggregated and Distributed Serving](content/tutorial.md#distributed-serving)
- [Optimized Model Serving](content/tutorial.md#optimized-model-serving)
- [Multi-Model Serving and Backend Ecosystems](content/tutorial.md#multi-model-serving)
- [RL Post-Training Infrastructure and Correctness](content/tutorial.md#rl-infrastructure)
- [Multimodal, Generative, and Compound AI Systems](content/tutorial.md#multimodal-and-compound-systems)
- [Production Orchestration, Efficiency, Security, and Observability](content/tutorial.md#production-operations)
- [End-to-End Systems Practice](content/tutorial.md#end-to-end-systems-practice)

## Algorithms

- [Foundations, Lifecycle, and Input Representation](content/tutorial.md#foundations-and-lifecycle)
- [Transformers, Attention, and Sequence Models](content/tutorial.md#transformers-and-attention)
- [Mixture of Experts](content/tutorial.md#mixture-of-experts)
- [Training, Fine-Tuning, Alignment, and Reasoning](content/tutorial.md#training-alignment-and-reasoning)
- [Inference and Generation Strategies](content/tutorial.md#inference-and-generation)
- [Quantization, Sparsity, and Compression](content/tutorial.md#quantization-and-compression)
- [Multimodal, Generative, Retrieval, Agent, and Non-LLM Workloads](content/tutorial.md#multimodal-agents-and-non-llm)
- [Distributed Learning and Model Evolution](content/tutorial.md#distributed-learning-and-evolution)
- [Model-Family Case Studies](content/tutorial.md#model-family-case-studies)
- [Model Evaluation and Cross-Layer Trade-offs](content/tutorial.md#evaluation-and-trade-offs)

The generated tutorial is available in [`docs/`](docs/) and at [yufeng98.github.io/public/blogs/ai-architecture-system/docs/](https://yufeng98.github.io/public/blogs/ai-architecture-system/docs/).
