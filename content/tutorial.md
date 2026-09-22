# AI Architecture and Systems Tutorial

<a id="microarchitecture"></a>

## 1. Microarchitecture

<a id="computation-foundations"></a>

### 1.1 Computation Foundations and AI Accelerator Execution

#### 1.1.1 From GEMM to AI Accelerators

##### 1.1.1.1 Scalar Multiply-Add, Dot Product, Matrix Multiplication, and Tensor Contraction

##### 1.1.1.2 GEMM, GEMV, Batched GEMM, and Operator Shapes

##### 1.1.1.3 Computation Graphs for Forward Pass, Backward Pass, and Parameter Update

##### 1.1.1.4 Compute Volume, Data Volume, Data Reuse, and Working Set

##### 1.1.1.5 Latency, Throughput, Bandwidth, Capacity, and Energy

##### 1.1.1.6 Execution Hierarchy from Processing Element to Chip

<a id="numerical-representation"></a>

### 1.2 Numerical Representation, Arithmetic, and Precision

#### 1.2.1 Numerical Representation, Arithmetic Circuits, and Precision Contracts

##### 1.2.1.1 Standard Floating-Point Formats: FP64, FP32, and FP16

##### 1.2.1.2 AI-Oriented Formats: BF16, TF32, and FP8

##### 1.2.1.3 Integer Arithmetic and Quantized Data Types: INT8/INT4

##### 1.2.1.4 Block Scaling: MXFP8/MXFP6/MXFP4 and NVFP4

##### 1.2.1.5 Multipliers, Adders, FMA, and Accumulators

##### 1.2.1.6 Separate Precision Conventions for Input, Multiply, Accumulate, and Output

##### 1.2.1.7 Rounding, Overflow, Underflow, Scale Overhead, and Numerical Error

##### 1.2.1.8 Low-Precision Arithmetic, Structured Sparsity, and Effective Data Bandwidth

<a id="execution-and-compute-units"></a>

### 1.3 Execution Organization and Compute Units

#### 1.3.1 Execution Organization: SIMT, Heterogeneous Engines, and Spatial Execution

##### 1.3.1.1 Scalar, SIMD, SIMT, MIMD, and VLIW

##### 1.3.1.2 SM, CU, Warp, Wavefront, and Thread Scheduling

##### 1.3.1.3 Instruction Pipelines, Dependency Checking, and Scoreboarding

##### 1.3.1.4 Latency Hiding, ILP, TLP, and Branch Divergence

##### 1.3.1.5 Division of Labor Among Matrix, Vector, Scalar, and Special-Function Engines

##### 1.3.1.6 PE-Local Execution, BSP, and Data-Triggered Execution

##### 1.3.1.7 Dynamic Hardware Scheduling, Static Software Scheduling, and Their Combination

##### 1.3.1.8 Local Systolic Dataflow versus Chip-Wide Spatial Dataflow

#### 1.3.2 Processing Elements and Matrix Compute Units

##### 1.3.2.1 Processing Element: MAC, Registers, and Local Control

##### 1.3.2.2 Dot-Product Arrays, Outer-Product Arrays, and Reduction Trees

##### 1.3.2.3 Accumulator Width and Partial-Sum Storage

##### 1.3.2.4 Broadcast, Reduction, and Operand Distribution

##### 1.3.2.5 PE-Level Pipelining and Significant-Bit Propagation

##### 1.3.2.6 Multi-Precision Reuse, Zero Skipping, and Sparsity Metadata

#### 1.3.3 Systolic Array Design

##### 1.3.3.1 One-Dimensional and Two-Dimensional Systolic Arrays

##### 1.3.3.2 Input Wavefronts, Data Skew, and Space-Time Scheduling

##### 1.3.3.3 Array Fill, Steady-State Execution, and Drain

##### 1.3.3.4 Weight-Stationary and Output-Stationary Arrays

##### 1.3.3.5 Matrix Tiling, Boundary Handling, and Padding

##### 1.3.3.6 Array Dimensions, Compute Utilization, and Operand-Supply Bandwidth

##### 1.3.3.7 Multi-Array Organization and Reconfigurable Arrays

<a id="dataflow-and-tpu"></a>

### 1.4 Dataflow, Reuse, and the TPU v1 Case Study

#### 1.4.1 Dataflow and Data Reuse

##### 1.4.1.1 Temporal Reuse and Spatial Reuse

##### 1.4.1.2 Weight-Stationary, Output-Stationary, and Input-Stationary

##### 1.4.1.3 Row-Stationary and Hybrid Dataflow

##### 1.4.1.4 Loop Nest, Loop Order, and Reuse Distance

##### 1.4.1.5 Local Reuse, Cross-PE Reuse, and Cross-Tile Reuse

##### 1.4.1.6 Compute, Storage, and Communication Trade-offs in Dataflow

#### 1.4.2 TPU v1 Case Study: From Systolic Array to GEMM Execution

##### 1.4.2.1 Matrix Multiply Unit and the Weight-Stationary Systolic Array

##### 1.4.2.2 Unified Buffer, Weight FIFO, and Accumulator

##### 1.4.2.3 Host Interface, External Weight Storage, and Instruction Control

##### 1.4.2.4 Datapaths for Activation/Weight/Partial Sum

##### 1.4.2.5 Coupling Matrix Multiply, Activation, and Output Write-Back

##### 1.4.2.6 The Complete Execution Path of a Single GEMM Instruction

##### 1.4.2.7 Boundary Between the TPU v1 Teaching Model and Later TPU Generations

<a id="memory-hierarchy"></a>

### 1.5 Memory Hierarchy

#### 1.5.1 Registers and On-Chip Local State

##### 1.5.1.1 Register File Capacity, Ports, and Banking

##### 1.5.1.2 Operand Read Bandwidth and Register Conflicts

##### 1.5.1.3 Thread Registers, Vector Registers, and Accumulator Registers

##### 1.5.1.4 Register Pressure, Occupancy, and Spilling

##### 1.5.1.5 General-Purpose Register Files versus Dedicated Matrix Intermediate-State Storage

##### 1.5.1.6 Register Allocation, Local-State Residency, and Datapath Constraints

#### 1.5.2 On-Chip SRAM Management: Cache, Scratchpad, and Dedicated Buffers

##### 1.5.2.1 SRAM Array, Bank, Ports, and Access Latency

##### 1.5.2.2 Hardware-Managed Cache and Software-Managed Scratchpad

##### 1.5.2.3 The GPU L1/Shared Memory/L2 Hybrid Hierarchy

##### 1.5.2.4 NPU Scratchpad and Spatial-Dataflow PE-Local SRAM

##### 1.5.2.5 Bank Conflict, Address Mapping, Swizzle, and Multicast

##### 1.5.2.6 Double Buffering, Prefetch, Data Residency, and Capacity Allocation

##### 1.5.2.7 Dedicated Accumulator Buffers and Tensor Memory

##### 1.5.2.8 On-Chip Memory Capacity, Bandwidth, Area, and Management Complexity

#### 1.5.3 DRAM and External Memory Interfaces

##### 1.5.3.1 DRAM Cell, Row, Bank, Bank Group, and Channel

##### 1.5.3.2 Row Buffer, Burst Transfer, and Access Timing

##### 1.5.3.3 DDR, LPDDR, GDDR, and HBM

##### 1.5.3.4 HBM Stack, Channel, and Pseudo-Channel

##### 1.5.3.5 Memory Controller and Request Scheduling

##### 1.5.3.6 Bandwidth Utilization, Access Granularity, and Read/Write Turnaround

##### 1.5.3.7 ECC, Refresh, and Capacity/Bandwidth Trade-offs

<a id="data-movement-and-interconnect"></a>

### 1.6 Data Movement, On-Chip Interconnect, and Synchronization

#### 1.6.1 Data Movement and Asynchronous Execution

##### 1.6.1.1 Load/Store Unit and Address Generators

##### 1.6.1.2 DMA, Descriptors, and Asynchronous Copy

##### 1.6.1.3 Gather/Scatter and Non-Contiguous Data Movement

##### 1.6.1.4 Memory-Level Parallelism and Outstanding Requests

##### 1.6.1.5 Producer–Consumer Pipelines and Double Buffering

##### 1.6.1.6 Barrier, Fence, Semaphore, and Data Visibility

##### 1.6.1.7 Overlap Among Compute, Data Movement, and the Memory Hierarchy

#### 1.6.2 On-Chip Interconnect and Synchronization

##### 1.6.2.1 Bus, Crossbar, Ring, and Mesh NoC

##### 1.6.2.2 Unicast, Multicast, and Reduction Networks

##### 1.6.2.3 Routing, Arbitration, Flow Control, and Backpressure

##### 1.6.2.4 On-Chip Bandwidth, Hop Count, and Congestion

##### 1.6.2.5 Multi-Core Shared Memory and Coherence Boundaries

##### 1.6.2.6 Topology Between Compute Arrays and Memory Controllers

<a id="performance-and-gemm-mapping"></a>

### 1.7 Performance Bounds and Hierarchical GEMM Mapping

#### 1.7.1 Roofline and Performance Upper Bounds

##### 1.7.1.1 Arithmetic Intensity and Data Traffic at Each Memory Boundary

##### 1.7.1.2 Compute Roof, Memory Roof, and Ridge Point

##### 1.7.1.3 Peak Roofline and Empirical Roofline

##### 1.7.1.4 Hierarchical Register/SRAM/DRAM Roofline

##### 1.7.1.5 Batch Size, Matrix Shape, and Reuse in GEMM/GEMV

##### 1.7.1.6 Beyond the Bandwidth Bound: Launch Latency, Synchronization, Dependencies, and Utilization Limits

##### 1.7.1.7 Data-Movement Energy and the Energy Roofline

#### 1.7.2 Hierarchical GEMM Mapping and Tiling

##### 1.7.2.1 M/N/K Dimensions and Loop Nest Mapping

##### 1.7.2.2 DRAM Tile, SRAM Tile, and Register Tile

##### 1.7.2.3 Spatial Unrolling, Temporal Reuse, and Array Mapping

##### 1.7.2.4 Residency Strategies for Weights, Activations, and Partial Sums

##### 1.7.2.5 Tile Size, Memory Capacity, and Port Bandwidth Constraints

##### 1.7.2.6 Mapping Differences Among Large, Tall-Skinny, and Small Matrices

##### 1.7.2.7 The Complete Dataflow from DRAM to PE and Back

<a id="matrix-engine-evolution"></a>

### 1.8 Matrix-Engine Evolution and Non-GEMM Support

#### 1.8.1 Evolution of Matrix-Engine Datapaths and Control Granularity

##### 1.8.1.1 Volta: Warp-Cooperative Programming Interface and Sub-Warp Machine Execution

##### 1.8.1.2 Turing: Warp-Level MMA, ldmatrix, and Operand Layout

##### 1.8.1.3 Ampere: cp.async and Global-to-Shared Data Movement

##### 1.8.1.4 Hopper: WGMMA, TMA, and Distributed Shared Memory

##### 1.8.1.5 Blackwell: Tensor Memory and Paired-SM Tensor Execution

##### 1.8.1.6 Independent Evolution of Operand Movement, Compute Issue, and Result Residency

##### 1.8.1.7 Boundaries Among Hardware Capability, ISA Exposure, and Kernel Programming Abstractions

#### 1.8.2 Non-GEMM Operators and Dedicated Support

##### 1.8.2.1 Softmax, LayerNorm, and RMSNorm

##### 1.8.2.2 Activation Functions, Exponential, Reciprocal, and Square Root

##### 1.8.2.3 Reduction, Scan, Sort, and Top-k

##### 1.8.2.4 Embedding, Gather/Scatter, and Sparse Memory Access

##### 1.8.2.5 Hardware Requirements of Attention and MoE

##### 1.8.2.6 Compute-Unit Balance and Amdahl's Law

<a id="modeling-and-verification"></a>

### 1.9 Hardware Modeling, Implementation, and Verification

#### 1.9.1 Hardware Modeling, Implementation, and Verification

##### 1.9.1.1 Analytical Model, Cycle Model, and RTL Model

##### 1.9.1.2 RTL and Functional Verification of Systolic Arrays

##### 1.9.1.3 Memory Models, Latency Models, and Bandwidth Models

##### 1.9.1.4 Area, Timing, Power, and Energy-Efficiency Evaluation

##### 1.9.1.5 Design Space Exploration and Constrained Optimization

##### 1.9.1.6 Timeloop/Accelergy, SCALE-Sim, and gem5

##### 1.9.1.7 FPGA Prototyping, Hardware Counters, and Model Calibration

<a id="kernel"></a>

## 2. Kernel | Operator Implementation and Performance Optimization

<a id="programming-and-layout"></a>

### 2.1 Programming Fundamentals and Tensor Layout

#### 2.1.1 Kernel Programming Models and Execution Fundamentals

##### 2.1.1.1 CPU Thread, GPU Thread, Block, and Grid

##### 2.1.1.2 Warp/Wavefront Cooperation and Cooperative Groups

##### 2.1.1.3 CUDA/HIP Memory Spaces and Synchronization Primitives

##### 2.1.1.4 Kernel Launch, Stream, Event, and Asynchronous Execution

##### 2.1.1.5 Device Memory, Pinned Memory, and Unified Memory

##### 2.1.1.6 Race Conditions, Deadlock, Out-of-Bounds Access, and Memory Consistency

#### 2.1.2 Tensor Layout and Memory Access

##### 2.1.2.1 Shape, Stride, View, and Contiguous Tensor

##### 2.1.2.2 Row-Major, Column-Major, and Blocked Layout

##### 2.1.2.3 Coalescing, Vectorized Load/Store, and Alignment

##### 2.1.2.4 Shared Memory Bank Conflict and Swizzle

##### 2.1.2.5 Transpose, Packing, and Layout Conversion

##### 2.1.2.6 Ragged Tensor, Padding, and Variable-Length Batches

<a id="gemm-optimization"></a>

### 2.2 GEMM Optimization from Naive to Advanced

#### 2.2.1 From Naive GEMM to Tiled GEMM

##### 2.2.1.1 Naive GEMM and Loop Reordering

##### 2.2.1.2 CPU Cache Blocking and SIMD Microkernel

##### 2.2.1.3 GPU Global-Memory GEMM

##### 2.2.1.4 Shared-Memory Tiling and Register Blocking

##### 2.2.1.5 Compute Reuse, Memory Coalescing, and Write-Back Optimization

##### 2.2.1.6 Correctness Checking and Step-by-Step Performance Analysis

#### 2.2.2 Tensor Core GEMM: Matrix Instructions, Data Layout, and Cooperation Scope

##### 2.2.2.1 WMMA/MMA, Matrix Fragments, and Operand Layout

##### 2.2.2.2 Instruction Tile, Warp Tile, CTA Tile, and Cluster Tile

##### 2.2.2.3 ldmatrix, cp.async, and Operand Load Paths

##### 2.2.2.4 Kernel Use and Synchronization of Hopper WGMMA/TMA

##### 2.2.2.5 Blackwell tcgen05/TMEM and Paired-SM Kernels

##### 2.2.2.6 Accumulator Layout, Numerical Precision, and Epilogue

##### 2.2.2.7 Hardware Generation Selection, Fallback, and Operator Correctness

#### 2.2.3 Advanced GEMM Pipelining and Scheduling

##### 2.2.3.1 Double/Multi-Buffering and Software Pipelining

##### 2.2.3.2 Asynchronous Copy and Producer–Consumer Synchronization

##### 2.2.3.3 Warp Specialization and Role Assignment

##### 2.2.3.4 Persistent Kernel and Persistent GEMM

##### 2.2.3.5 Split-K, Stream-K, and Work Distribution

##### 2.2.3.6 CTA Swizzle, Cluster, and Locality

##### 2.2.3.7 Occupancy, Register Pressure, and Wave Quantization

#### 2.2.4 Matrix Computation Across Shapes and Scenarios

##### 2.2.4.1 GEMV and Small-Batch Decode

##### 2.2.4.2 Small/Tall-Skinny GEMM

##### 2.2.4.3 Batched GEMM and Grouped GEMM

##### 2.2.4.4 Variable-Length Matrices, Dynamic Shapes, and Boundary Tiles

##### 2.2.4.5 Low-Rank Matrices, LoRA, and Multi-Adapter GEMM

##### 2.2.4.6 Weight Residency, Weight Packing, and Shape Specialization

<a id="elementwise-reduction-and-fusion"></a>

### 2.3 Elementwise Operations, Reductions, and Fusion

#### 2.3.1 Elementwise, Reduction, and Scan

##### 2.3.1.1 Vector Add and Elementwise Fusion

##### 2.3.1.2 Softmax and Online Softmax

##### 2.3.1.3 LayerNorm, RMSNorm, and Reduction Layout

##### 2.3.1.4 Prefix Sum, Scan, and Histogram

##### 2.3.1.5 Top-k, Argmax, Sort, and Sampling

##### 2.3.1.6 Welford's Algorithm and Stable Reduction

#### 2.3.2 Fusion and Memory-Traffic Elimination

##### 2.3.2.1 Bias/Activation/Residual Epilogue Fusion

##### 2.3.2.2 QKV Projection and RoPE Fusion

##### 2.3.2.3 Norm, Residual, and Quantization Fusion

##### 2.3.2.4 Fused MLP, SwiGLU, and GeGLU

##### 2.3.2.5 Fused Loss and Fused Optimizer

##### 2.3.2.6 Launch Overhead, Memory-Traffic Savings, and Fusion Boundaries

<a id="attention-and-kv-cache"></a>

### 2.4 Attention and KV-Cache Kernels

#### 2.4.1 Exact Attention Kernels

##### 2.4.1.1 Operator Decomposition and Memory Traffic of Standard Attention

##### 2.4.1.2 FlashAttention: IO-Aware Tiling and Recomputation

##### 2.4.1.3 FlashAttention-2/3 and Parallelism and Pipelining Optimizations

##### 2.4.1.4 Causal Mask, Sliding Window, and Variable-Length Attention

##### 2.4.1.5 Forward/Backward Attention Kernels

##### 2.4.1.6 FlexAttention and Programmable Attention

#### 2.4.2 Decode Attention and KV Cache Kernels

##### 2.4.2.1 Paged KV Layout and Block Table

##### 2.4.2.2 Reading and Reduction in PagedAttention

##### 2.4.2.3 Split-KV and Long-Sequence Decode

##### 2.4.2.4 Kernel Mapping of MHA/MQA/GQA

##### 2.4.2.5 MLA Projection Absorption and Low-Rank Cache Access

##### 2.4.2.6 KV Cache Append, Gather, Copy, and Compression

<a id="sparse-moe-and-quantization"></a>

### 2.5 Sparse, MoE, and Quantized Kernels

#### 2.5.1 Sparse and Irregular Kernels

##### 2.5.1.1 COO, CSR, CSC, BSR, and Compressed Layouts

##### 2.5.1.2 SpMV, SpMM, and SDDMM

##### 2.5.1.3 Structured Sparsity, N:M Sparsity, and Sparse Tensor Core

##### 2.5.1.4 Block-Sparse Attention and Dynamic Indexing

##### 2.5.1.5 Sparse Selection, Gather/Scatter, and Load Imbalance

##### 2.5.1.6 Sparsity Metadata Overhead and the Limits of Speedup

#### 2.5.2 MoE Kernels

##### 2.5.2.1 Router, Top-k Gating, and Token Assignment

##### 2.5.2.2 Token Permutation and Unpermutation

##### 2.5.2.3 Grouped GEMM and Expert Padding

##### 2.5.2.4 Dropless MoE and Block-Sparse GEMM

##### 2.5.2.5 Dispatch/Combine and Communication Fusion

##### 2.5.2.6 Shared Experts, Routed Experts, and Compute Overlap

#### 2.5.3 Low-Precision and Quantized Kernels

##### 2.5.3.1 Quantize/Dequantize and Scale Computation

##### 2.5.3.2 Weight-Only GEMM and Weight–Activation GEMM

##### 2.5.3.3 INT8/INT4 and FP8/FP4 GEMM

##### 2.5.3.4 Per-Tensor, Per-Channel, and Block Scaling

##### 2.5.3.5 Microscaling Formats and Scale Layout

##### 2.5.3.6 Online Quantization, Dequantization Fusion, and Numerical Validation

<a id="communication-and-non-llm"></a>

### 2.6 Communication and Non-LLM Kernels

#### 2.6.1 Communication Kernels: Device-Initiated Transfer, Data Movement, and Compute Fusion

##### 2.6.1.1 GPU P2P Copy, Remote Memory Access, and Memory Registration

##### 2.6.1.2 Reduction, AllReduce, and ReduceScatter Kernels

##### 2.6.1.3 NVSHMEM Symmetric Address Space and Explicit Synchronization

##### 2.6.1.4 Device-Initiated Communication and Host-Proxy Paths

##### 2.6.1.5 SM-Driven versus DMA/Copy-Engine Data Movement

##### 2.6.1.6 Communication Tiles, GEMM–Collective Fusion, and Pipelining

##### 2.6.1.7 SM Occupancy, HBM Contention, and Overlap Limits of Communication Kernels

#### 2.6.2 Key Kernels Beyond Language Models

##### 2.6.2.1 Conv2D, Depthwise Conv, and Implicit GEMM

##### 2.6.2.2 Embedding Lookup and Embedding Bag

##### 2.6.2.3 Graph Aggregation and Sparse Adjacency Computation

##### 2.6.2.4 FFT, Convolution, and State-Space Scan

##### 2.6.2.5 Image/Video Preprocessing and Resize

##### 2.6.2.6 Audio Codec, Vocoder, and Streaming Audio Computation

<a id="kernel-programming-stacks"></a>

### 2.7 Kernel Languages and Hardware-Specific Programming

#### 2.7.1 GPU Kernel Languages, DSLs, and Operator/Kernel Libraries

##### 2.7.1.1 CUDA/HIP: Threads, Warp/Wavefront, and Device-Specific Interfaces

##### 2.7.1.2 Triton: Tile Programming, Autotuning, and Backend Differences

##### 2.7.1.3 CuTe/CUTLASS and CuTe DSL: Explicit Layout and Cooperative Scheduling

##### 2.7.1.4 CUDA Tile IR/cuTile: The Boundary Between Tile-Level Abstraction and Low-Level Control

##### 2.7.1.5 cuBLAS/cuBLASLt/cuDNN and CUB/Thrust/libcu++

##### 2.7.1.6 rocBLAS/hipBLASLt/MIOpen and CK/CK-Tile/AITER/Tensile

##### 2.7.1.7 TileLang, FlashInfer, Transformer Engine, and DeepGEMM

##### 2.7.1.8 CPU Counterpart: SIMD/AMX Microkernels and oneDNN

#### 2.7.2 NPU Kernel Programming: Tiles, Scratchpad, and Heterogeneous Engines

##### 2.7.2.1 TPU Pallas: BlockSpec, Memory Space, and MXU/Vector Cooperation

##### 2.7.2.2 Interfacing Pallas/Mosaic with Graph Compilation

##### 2.7.2.3 AWS NKI: Language API, ISA Intrinsics, and SBUF/PSUM

##### 2.7.2.4 Operator Division of Labor Across Neuron Tensor/Vector/Scalar/GPSIMD

##### 2.7.2.5 Ascend C: Data Movement, Cube/Vector, and Pipeline Synchronization

##### 2.7.2.6 Ascend TBE/TIK, CATLASS, and Device-Specific DSLs

##### 2.7.2.7 Cambricon BANG C and Explicit Local-Memory Programming

##### 2.7.2.8 Layout and Capacity Constraints of GEMM/Attention on Different NPUs

#### 2.7.3 Spatial Dataflow Kernels: PEs, Codelets, and Data-Triggered Tasks

##### 2.7.3.1 Tenstorrent: Boundaries Among TT-NN, TT-Metalium, and TT-LLK

##### 2.7.3.2 Reader/Compute/Writer and Unpack/Math/Pack

##### 2.7.3.3 PE-Local Buffers, NoC Data Movement, and Producer–Consumer Synchronization

##### 2.7.3.4 Graphcore: Vertex/Codelet and BSP Compute–Exchange–Sync

##### 2.7.3.5 Cerebras CSL: Wavelet, Color, Task, and Data Structure Descriptor

##### 2.7.3.6 Cooperation Between Local Kernels and Whole-Graph Placement/Routing

##### 2.7.3.7 Groq and SN40L-Era SambaFlow: Compiler-Owned Compute Paths

##### 2.7.3.8 Distinguishing Public Kernel Interfaces, Compiler Internals, and Non-Visible Software Layers

<a id="profiling-and-correctness"></a>

### 2.8 Profiling, Benchmarking, and Correctness

#### 2.8.1 Profiling, Benchmarking, and Correctness

##### 2.8.1.1 Microbenchmarks, Warm-up, Asynchronous Timing, and Timing Boundaries

##### 2.8.1.2 Nsight Compute/Systems and ROCm Profiling

##### 2.8.1.3 Device-Specific Analysis Tools: Neuron Explorer, PopVision, and Compiler Reports

##### 2.8.1.4 Roofline, Stall Reasons, Memory Efficiency, and Communication Occupancy

##### 2.8.1.5 Dense Peak, Sparse Peak, Achieved FLOPS, and Effective Model Work

##### 2.8.1.6 Numerical Tolerance, Gradient Checking, Low-Precision Error, and Differential Testing

##### 2.8.1.7 Race Detection, Regression Testing, and Cross-Device Performance Portability

<a id="operator-to-model-integration"></a>

### 2.9 Operator-to-Model Integration

#### 2.9.1 From Operator to Model: Registration, Porting, and Backend Integration

##### 2.9.1.1 PyTorch Custom Operator, Autograd, and Dispatcher

##### 2.9.1.2 Choosing Among Operator Library Calls, Compiler Fusion, and Custom Kernels

##### 2.9.1.3 Matching Shape, Layout, Precision, Sparse Format, and Device Capability

##### 2.9.1.4 JIT Cache, Precompiled Kernels, and Architecture-Specific Artifacts

##### 2.9.1.5 Graph Capture, Asynchronous Execution, and Custom Operator Compatibility

##### 2.9.1.6 Distinguishing API Compatibility, Numerical Equivalence, and Performance Portability

##### 2.9.1.7 End-to-End Integration from GEMM to Attention to Transformer Block

<a id="compiler"></a>

## 3. Compiler | Compilation and Program Mapping

<a id="stack-and-frontends"></a>

### 3.1 Compilation Stack, Frontends, and Graph Capture

#### 3.1.1 The AI Compilation Stack: Graphs, Operator Libraries, Kernels, and Device Execution

##### 3.1.1.1 Framework → Graph IR → Tensor/Loop IR → Device Program

##### 3.1.1.2 Graph Compiler, Kernel Compiler, Op Library, and Runtime

##### 3.1.1.3 Eager, JIT, AOT, and Static Execution Plans

##### 3.1.1.4 Responsibility Boundaries Across Compile Time, Load Time, Run Time, and Firmware

##### 3.1.1.5 Standalone Software Layers, Compiler-Merged Layers, and Undisclosed Layers

##### 3.1.1.6 Compatibility, Programmability, Portability, and Specialization

#### 3.1.2 Frontend, Tracing, and Graph Capture

##### 3.1.2.1 Python Program, Tensor Graph, and Control Flow

##### 3.1.2.2 Symbolic Tracing, Bytecode Capture, and Export

##### 3.1.2.3 Graph Break, Guard, and Recompilation

##### 3.1.2.4 Dynamic Shapes, Dynamic Branching, and Graph Specialization

##### 3.1.2.5 Custom Operator, Side Effects, and External Calls

<a id="ir-autodiff-and-graph-optimization"></a>

### 3.2 IR, Autodiff, and Graph Optimization

#### 3.2.1 Intermediate Representation

##### 3.2.1.1 Dataflow Graph, SSA, and Control-Flow Graph

##### 3.2.1.2 Tensor IR, Loop IR, and Buffer IR

##### 3.2.1.3 Type, Shape, Layout, and Memory Space

##### 3.2.1.4 MLIR Dialect, Operation, Region, and Pass

##### 3.2.1.5 StableHLO, Linalg, Affine, SCF, and GPU Dialect

#### 3.2.2 Automatic Differentiation and Training-Graph Compilation

##### 3.2.2.1 Reverse-Mode and Forward-Mode AD

##### 3.2.2.2 Backward Graph, VJP, and JVP

##### 3.2.2.3 AOT Autograd and Joint Graph Optimization

##### 3.2.2.4 Activation Liveness and Rematerialization

##### 3.2.2.5 Gradient Accumulation, Gradient Communication, and the Optimizer Graph

#### 3.2.3 Graph-Level Optimization

##### 3.2.3.1 Constant Folding, CSE, and Dead Code Elimination

##### 3.2.3.2 Operator Fusion, Operator Decomposition, and Pattern Rewriting

##### 3.2.3.3 Algebraic Simplification and Operator Reordering

##### 3.2.3.4 Attention, MLP, and Transformer-Specific Fusion

##### 3.2.3.5 Numerical Equivalence and Optimization Legality

<a id="loop-layout-and-memory"></a>

### 3.3 Loop, Layout, and Memory Optimization

#### 3.3.1 Loop Transformation and Tensorization

##### 3.3.1.1 Tiling, Interchange, Unrolling, and Vectorization

##### 3.3.1.2 Loop Fusion, Fission, and Software Pipelining

##### 3.3.1.3 Affine Analysis and Polyhedral Optimization

##### 3.3.1.4 Parallelization and Reduction Transformation

##### 3.3.1.5 Tensorization and Matrix-Instruction Matching

##### 3.3.1.6 Mapping Loop Schedules onto Systolic/SIMT Hardware

#### 3.3.2 Layout and Data-Movement Optimization

##### 3.3.2.1 Layout Inference and Layout Propagation

##### 3.3.2.2 Blocked Layout, Swizzle, and Thread Mapping

##### 3.3.2.3 Insertion and Elimination of Layout Conversions

##### 3.3.2.4 Cache/Scratchpad Placement and Data Reuse

##### 3.3.2.5 Asynchronous DMA, Prefetch, and Transfer Scheduling

#### 3.3.3 Memory Planning and Buffer Management

##### 3.3.3.1 Bufferization, Aliasing, and In-Place Execution

##### 3.3.3.2 Liveness Analysis and Buffer Reuse

##### 3.3.3.3 Static Memory Plans and Dynamic Memory Allocation

##### 3.3.3.4 Register Allocation, Spilling, and Scratchpad Allocation

##### 3.3.3.5 Recomputation, Offloading, and Memory Capacity Constraints

<a id="codegen-and-runtime-specialization"></a>

### 3.4 Code Generation and Runtime Specialization

#### 3.4.1 Kernel Code Generation and ISA Boundaries

##### 3.4.1.1 Instruction Selection, Instruction Scheduling, and Register Allocation

##### 3.4.1.2 LLVM/NVVM/AMDGPU Backends

##### 3.4.1.3 PTX, SASS, and AMD Device Instruction Sets

##### 3.4.1.4 Thread-Level, Tile-Level, and Tensor-Intrinsic Lowering Paths

##### 3.4.1.5 Triton IR → GPU IR → Device Code

##### 3.4.1.6 Cooperation Among Vector, Matrix, and Scalar Paths

##### 3.4.1.7 Distinguishing Public ISAs, Public Intrinsics, and Undisclosed Machine Instructions

##### 3.4.1.8 Cross-Architecture Code Generation, Feature Detection, and Fallback

#### 3.4.2 Dynamic Shape and Runtime Specialization

##### 3.4.2.1 Symbolic Shape, Shape Constraint, and Shape Polymorphism

##### 3.4.2.2 Shape Bucketing, Padding, and Multi-Versioning

##### 3.4.2.3 Autotuning Cache and Compilation Cache

##### 3.4.2.4 Dynamic Sparsity, Dynamic Routing, and Ragged Workloads

##### 3.4.2.5 Compilation Latency, Cold Start, and Steady-State Performance

<a id="cost-models-and-distributed-compilation"></a>

### 3.5 Cost Models, Autotuning, and Distributed Compilation

#### 3.5.1 Cost Models, Autotuning, and Auto-Scheduling

##### 3.5.1.1 Search Space, Legality Constraints, and Schedule Representation

##### 3.5.1.2 Analytical/Learned Cost Models

##### 3.5.1.3 Searching Over Tile, Layout, Fusion, and Parallelism

##### 3.5.1.4 Measurement Feedback, Search Budget, and Generalization

##### 3.5.1.5 Hardware-Aware Optimization and Performance Portability

#### 3.5.2 Distributed Compilation

##### 3.5.2.1 SPMD, Sharding Annotation, and Device Mesh

##### 3.5.2.2 Automatic Parallelization Strategies and Graph Partitioning

##### 3.5.2.3 Collective Insertion and Resharding

##### 3.5.2.4 Compile-Time Scheduling of Communication–Computation Overlap

##### 3.5.2.5 Pipeline Partitioning, Cross-Device Layout, and Memory Constraints

##### 3.5.2.6 OpenXLA, GSPMD, and Shardy

<a id="pytorch-jax-and-xla"></a>

### 3.6 PyTorch, JAX, and XLA

#### 3.6.1 The PyTorch Compilation Stack

##### 3.6.1.1 TorchDynamo and FX Graph

##### 3.6.1.2 AOTAutograd and Training-Graph Capture

##### 3.6.1.3 TorchInductor, Triton, and CPU Codegen

##### 3.6.1.4 Export, Dynamic Shape, and Custom Backends

##### 3.6.1.5 Compilation Modes, Graph Breaks, and Performance Diagnosis

#### 3.6.2 JAX, XLA, and TPU Compilation

##### 3.6.2.1 JAX Tracing, jaxpr, and XLA

##### 3.6.2.2 StableHLO, HLO, and Multi-Stage Optimization

##### 3.6.2.3 Sharding, GSPMD/Shardy, and Collective Insertion

##### 3.6.2.4 Layout Assignment, Buffer Assignment, and Memory Scheduling

##### 3.6.2.5 Cooperation Between Pallas/Mosaic and High-Level Graph Compilation

##### 3.6.2.6 MXU/Vector/DMA Cooperation and Static Scheduling

##### 3.6.2.7 The libtpu Delivery Boundary and the Scope of Public TPU ISA Information

<a id="mlir-tvm-and-deployment"></a>

### 3.7 MLIR, TVM, and Deployment Stacks

#### 3.7.1 MLIR, TVM, and Deployment Compilation Stacks

##### 3.7.1.1 Multi-Level IR and Extensible Dialects in MLIR

##### 3.7.1.2 TVM TensorIR, Relax, and Autotuning

##### 3.7.1.3 IREE and Heterogeneous Runtimes

##### 3.7.1.4 ONNX, ONNX Runtime, and Graph Interchange

##### 3.7.1.5 TensorRT and Deployment Graph Optimization

##### 3.7.1.6 Quantized Graph Conversion, Operator Coverage, and Backend Compatibility

<a id="accelerator-specific-compilation"></a>

### 3.8 Accelerator-Specific and Memory-Centric Compilation

#### 3.8.1 NPU Compilation: Scratchpad, Engine Division of Labor, and Device Executables

##### 3.8.1.1 AWS neuronx-cc: Graph Compilation and Device-Specific Code Generation

##### 3.8.1.2 The Distinct Entry Points of the NKI Compiler and the Graph Compiler

##### 3.8.1.3 SBUF/PSUM Allocation, Prefetch, and Cross-Engine Scheduling

##### 3.8.1.4 NEFF Artifacts, Runtime Loading, and Device Execution

##### 3.8.1.5 Ascend: MindIR/ATC/Graph Engine and CANN

##### 3.8.1.6 Operator Coverage, Static Shapes, Backend Constraints, and Model Porting

##### 3.8.1.7 Separating Public Interfaces, Internal IR, and Inferred Information

#### 3.8.2 Spatial Dataflow Compilation: Graph Mapping, Memory Placement, and Communication Scheduling

##### 3.8.2.1 Tenstorrent: TT-Forge/TT-XLA → TT-MLIR → TT-NN/TT-Metalium

##### 3.8.2.2 Graphcore Poplar: Tile Assignment, Codelets, and BSP Exchange

##### 3.8.2.3 Cerebras: Graph Compilation, CSL Tasks, and Layer-Pipelined/Weight-Streaming Execution

##### 3.8.2.4 SambaFlow (SN40L-Era): PCU/PMU Placement, Routing, and Meta-Pipelines

##### 3.8.2.5 Groq: Functional-Slice Placement, Cycle Scheduling, and Link Scheduling

##### 3.8.2.6 Temporal Reuse, Spatial Unrolling, SRAM Allocation, and Routing Resource Constraints

##### 3.8.2.7 Trade-offs Between Compile-Time Conflict Elimination and Runtime Scheduling

#### 3.8.3 CIM/PNM Compilation and Memory-Centric Offload

##### 3.8.3.1 Identifying and Partitioning Supported Operators, and Host/Accelerator Cooperation

##### 3.8.3.2 Weight Residency, Bank/Array Layout, and Data Reorganization

##### 3.8.3.3 Operator Mapping, Tiling, and Cross-Layer Transfer Under Capacity Limits

##### 3.8.3.4 Compute Coverage, Precision Constraints, and Fallback Paths

##### 3.8.3.5 The Corsair Aviator Case and the Boundary of Undisclosed Raptor Compilation Details

##### 3.8.3.6 Research Extension: CXL-Attached PNM and Heterogeneous Offload Compilation

<a id="artifacts-and-validation"></a>

### 3.9 Artifacts, Validation, Debugging, and Case Studies

#### 3.9.1 Executable Artifacts, Loading Contracts, and Software-Stack Visibility

##### 3.9.1.1 GPU: PTX/SASS, Fat Binary, and Device Code Objects

##### 3.9.1.2 Neuron NEFF, Groq IOP, and SambaFlow PEF

##### 3.9.1.3 Kernel Binaries, Weights, Routing, and Static Scheduling Metadata

##### 3.9.1.4 Runtime Loading, Driver Submission, and Firmware Execution

##### 3.9.1.5 ABI, Version Matching, Compilation Cache, and Reproducible Builds

##### 3.9.1.6 Distinguishing Not Applicable, Not Public, Inferred, and Confirmed

#### 3.9.2 Compiler Validation, Debugging, and Case Studies

##### 3.9.2.1 IR Dump, Pass Tracing, and Minimal Reproductions

##### 3.9.2.2 Differential Testing and Numerical Consistency

##### 3.9.2.3 Ablating Fusion, Layout, and Memory Plan

##### 3.9.2.4 Compilation Performance Regressions and Cross-Version Reproducibility

##### 3.9.2.5 The Complete Compilation Path from Python GEMM to Matrix Instructions

##### 3.9.2.6 From Transformer Graph to Multi-Device Execution Plan

<a id="architecture"></a>

## 4. Architecture | Accelerator Architecture and AI Datacenters

<a id="context-and-evidence"></a>

### 4.1 Context, Scope, and Evidence Standards

#### 4.1.1 Scope, Terminology, and Evidence Standards for Architectural Comparison

##### 4.1.1.1 Generality, Specialization, Programmability, and Deployment Flexibility

##### 4.1.1.2 Dominant Execution/Data-Movement Models and Overlapping Hardware Mechanisms

##### 4.1.1.3 Chip, Die, Package, Card, Server, Rack, and Pod

##### 4.1.1.4 Vendor-Reported Peak, Measured Performance, and Model-Level Results

##### 4.1.1.5 Dense/Sparse Throughput, FMA Counting, and Numerical Formats

##### 4.1.1.6 Unidirectional/Bidirectional Bandwidth, Per-Accelerator/Per-System, and Reference Platforms

##### 4.1.1.7 Confirmed, Inferred, Contested, Not Public, and Not Reported

##### 4.1.1.8 Presenting Disclosed Designs, Planned/Announced Designs, and Historical Platforms Separately

#### 4.1.2 AI Datacenter Context and Four Categories of Resource Pressure

##### 4.1.2.1 From Single-Chip AI Compute to Industrial-Scale Datacenters

##### 4.1.2.2 Model Parameters, Context, Concurrency, and Workload Variation

##### 4.1.2.3 Compute Throughput, Memory Capacity, and Memory Bandwidth

##### 4.1.2.4 Interconnect Bandwidth, Latency, and Synchronization Overhead

##### 4.1.2.5 Power, Cooling, and Practically Operable Compute Capacity

##### 4.1.2.6 Specialized Matrix Arithmetic, Local Reuse, and Multi-Device Aggregation

##### 4.1.2.7 Cross-Generation Comparison: Uneven Evolution of Model Scale and Hardware Resources

<a id="accelerator-taxonomy"></a>

### 4.2 Accelerator Taxonomy and Architecture Classes

#### 4.2.1 Core Taxonomy: Four Classes of AI Accelerator Architecture

##### 4.2.1.1 GPU: SIMT Architecture and Dedicated Tensor/Matrix Units

##### 4.2.1.2 NPU: Matrix/Vector/Scalar Engines Around a Shared Scratchpad

##### 4.2.1.3 Spatial Dataflow: Computation Graphs Mapped onto Distributed Compute, Memory, and Communication Resources

##### 4.2.1.4 Compute-in-Memory: Arithmetic Units Inside or Adjacent to Memory Arrays

##### 4.2.1.5 Three Subclasses of Spatial Dataflow: PE Array, Reconfigurable, Functional-Slice

##### 4.2.1.6 Systolic Execution, Memory Technology, and Their Non-One-to-One Relation to Architecture Class

##### 4.2.1.7 Primary Categories, Generational Differences, and Hybrid Implementations

#### 4.2.2 GPU: General-Purpose Parallel Execution and Specialized Matrix Computation

##### 4.2.2.1 NVIDIA GPU: SM, Tensor Core, and the CUDA Programming Model

##### 4.2.2.2 AMD GPU: CU, Matrix Core, and the ROCm/HIP Programming Model

##### 4.2.2.3 Resource Balance Among General Control Flow, Non-Matrix Operators, and Matrix Throughput

##### 4.2.2.4 SIMT, Cache/Shared Memory, and the Boundary of Programming Responsibility

##### 4.2.2.5 Biren, Hygon DCU, Muxi, and Moore Threads

##### 4.2.2.6 Tianshu Zhixin and Xiwang

#### 4.2.3 NPU: Heterogeneous Compute Engines and Shared Local Memory

##### 4.2.3.1 Google TPU: MXU, Vector/Scalar, and Generation-Specific Units

##### 4.2.3.2 AWS Trainium/Inferentia: NeuronCore and Explicit Scratchpad

##### 4.2.3.3 Huawei Ascend: Da Vinci, Cube/Vector/Scalar, and CANN

##### 4.2.3.4 Intel Gaudi: Matrix Engine, TPC, and Network Integration

##### 4.2.3.5 Microsoft Maia: Matrix/Vector and Cloud Deployment

##### 4.2.3.6 Qualcomm Cloud AI: Matrix, Vector, and Scalar Engines

##### 4.2.3.7 Cambricon MLU: Multi-Core Neural Processor and BANG C

##### 4.2.3.8 Alibaba T-Head, Kunlunxin, and Furiosa

##### 4.2.3.9 Sophgo, Vastai, Tecorigin, and Stream Computing

#### 4.2.4 Spatial Dataflow I: PE Arrays and Distributed Local Memory

##### 4.2.4.1 Tenstorrent: Tensix, RISC-V Control, and NoC Data Movement

##### 4.2.4.2 Meta MTIA: PE Grid and Model–Chip Co-Design

##### 4.2.4.3 Graphcore IPU: MIMD Tiles, Local SRAM, and BSP

##### 4.2.4.4 Tesla Dojo: Tile Processor and Hierarchical Communication

##### 4.2.4.5 Cerebras: Wafer-Scale PE Mesh and Data-Triggered Execution

##### 4.2.4.6 Boundaries Among On-Chip PE Mesh, In-Package Interconnect, and Inter-Accelerator Networks

##### 4.2.4.7 IBM Spyre and Enflame

##### 4.2.4.8 MN-Core, Esperanto, and PEZY

#### 4.2.5 Spatial Dataflow II: Reconfigurable Architectures

##### 4.2.5.1 SambaNova SN40L: Configurable Compute/Memory Tiles and Three-Tier Memory

##### 4.2.5.2 Graph Placement, Memory Placement, and Static Routing

##### 4.2.5.3 Sections, Meta-Pipelines, and Reconfiguration Under Limited Resources

##### 4.2.5.4 Supported Operator Coverage, Graph-Mapping Capability, and Model Evolution

##### 4.2.5.5 Rebellions and Tsingmicro

##### 4.2.5.6 NextSilicon Maverick and Runtime Reconfiguration

#### 4.2.6 Spatial Dataflow III: Functional-Slice Streaming

##### 4.2.6.1 Groq TSP/LPU: Matrix, Vector, SRAM, and Switch Slices

##### 4.2.6.2 Compiler-Determined Operation Placement, SRAM Addresses, and Time Scheduling

##### 4.2.6.3 Cross-Chip Datapaths and Software-Scheduled Networking

##### 4.2.6.4 Predictability of Static Execution, Capacity Limits, and Model Fit

##### 4.2.6.5 Generational/Topological Boundary Between Early Groq Systems and Later Platforms

##### 4.2.6.6 Model-Structure Specialization in Etched Sohu and the Scope of Public Evidence

#### 4.2.7 Compute-in-Memory: SRAM, DRAM, and the Location of Computation

##### 4.2.7.1 d-Matrix Corsair: Digital In-SRAM MAC and Capacity Memory

##### 4.2.7.2 SK hynix AiM: GDDR6 Bank-Adjacent MAC

##### 4.2.7.3 Samsung PIM: HBM and Memory-Side SIMD Compute

##### 4.2.7.4 The Differing Limits of SRAM Capacity, DRAM Internal Bandwidth, and Compute Throughput

##### 4.2.7.5 Mapping Differences Between GEMV/Small-Batch and High-Reuse GEMM

##### 4.2.7.6 Analog Flash Compute in Mythic

##### 4.2.7.7 Historical/Contrast Cases: Untether AI and Rain AI

##### 4.2.7.8 Terminological Boundaries Among CIM, PNM, and 3D Logic–Memory Integration

<a id="compute-memory-and-programming"></a>

### 4.3 Cross-Category Compute, Memory, and Programming Models

#### 4.3.1 Compute Engines: Cross-Category Organization and Trade-offs

##### 4.3.1.1 The Balance Among Scalar/Vector/SIMD/SIMT/Matrix Engines

##### 4.3.1.2 Systolic Array, Tensor Core, Vector MAC, and Configurable Compute Units

##### 4.3.1.3 Matrix Throughput, Non-GEMM Operators, and Control-Flow Coverage

##### 4.3.1.4 Embedding, Gather/Scatter, and Dedicated Acceleration Units

##### 4.3.1.5 Static Scheduling, Dynamic Scheduling, and Data-Triggered Execution

##### 4.3.1.6 Arithmetic Intensity, Compute Utilization, and Area/Energy Efficiency

##### 4.3.1.7 Programming Flexibility, Compiler Burden, and Degree of Specialization

#### 4.3.2 Memory Hierarchy: From Hardware Cache to Software Scratchpad

##### 4.3.2.1 Fully Hardware-Managed, Hybrid, and Fully Software-Managed Memory

##### 4.3.2.2 The CPU Cache Counterpart and the Gaudi Cache-Management Case

##### 4.3.2.3 GPU: L1/Shared Memory/L2 and Dedicated Tensor Memory

##### 4.3.2.4 NPU: Shared Scratchpad, Accumulator Buffer, and Explicit Prefetch

##### 4.3.2.5 PE Array: Distributed Local SRAM and Memory Placement

##### 4.3.2.6 HBM, GDDR, LPDDR, DDR, and SRAM-Centric Organizations

##### 4.3.2.7 Determinism, QoS, Compilation Complexity, and Dynamic Access Patterns

#### 4.3.3 Programming Models: Understanding Hardware Constraints Through Software Layers

##### 4.3.3.1 Framework Integration, Compiler/IR, and Operator Library

##### 4.3.3.2 Kernel Library, Runtime, Driver/Firmware, and Assembler/ISA

##### 4.3.3.3 Communication Libraries and Compiler-Built-In Communication

##### 4.3.3.4 Thread/Warp, Tensor Tile, PE/Vertex, and Static Streaming Programming

##### 4.3.3.5 Kernel-Launched, Graph-Executed, and Data-Triggered Models

##### 4.3.3.6 Registers, Scratchpad, Distributed SRAM, and Explicit Data Movement

##### 4.3.3.7 Standalone Libraries, Compiler-Merged Layers, Undisclosed Interfaces, and Historical Interfaces

##### 4.3.3.8 The Causal Chain from Hardware Resources to Software Responsibility to Programming Model

<a id="packaging-and-memory-extensions"></a>

### 4.4 Packaging, Host Memory, and Memory-Centric Extensions

#### 4.4.1 Packaging, Chiplets, and Logic–Memory Integration

##### 4.4.1.1 Monolithic Die, Multi-Chip Module, and Compute/I/O Die

##### 4.4.1.2 HBM Stack, Interposer, and 2.5D Packaging

##### 4.4.1.3 3D Stacking, Logic–DRAM Bonding, and Local Datapaths

##### 4.4.1.4 Memory Density, Stack Count, Interface Width, and Signaling Rate

##### 4.4.1.5 Chiplet Interconnect, Boards, Baseboards, and Compute Trays

##### 4.4.1.6 SXM/OAM, Packaging Density, and Power and Thermal Constraints

##### 4.4.1.7 The Raptor Early-Silicon Case and Thermal/Reliability Constraints

##### 4.4.1.8 Further Reading: UCIe and General Die-to-Die Interfaces

#### 4.4.2 Host, Remote Memory, and Memory-Centric Extensions

##### 4.4.2.1 Host–Accelerator PCIe, DMA, and CPU/Accelerator NUMA

##### 4.4.2.2 Distinguishing Shared Address Space, Memory Coherence, and Explicit Communication

##### 4.4.2.3 NVSHMEM: Symmetric Memory, Remote Access, and Synchronization

##### 4.4.2.4 Groq: A Compiler-Managed Distributed SRAM Address Space

##### 4.4.2.5 Extension: CXL.io/CXL.cache/CXL.mem and Memory Pooling

##### 4.4.2.6 Extension: CPU/GPU/NPU/PIM Cooperation and CXL-Attached PNM

##### 4.4.2.7 Extension: Storage Offload, Flash/HBF, and the Capacity Hierarchy

<a id="deployment-and-interconnect"></a>

### 4.5 Deployment Hierarchy and Interconnect Domains

#### 4.5.1 Physical Deployment Hierarchy and Scale-Up/Scale-Out Domains

##### 4.5.1.1 Accelerator → Server → Rack → Pod → Datacenter

##### 4.5.1.2 Organization of Compute Tray, Switch Tray, Host, and NIC

##### 4.5.1.3 The Reach of a Scale-Up Fabric Across Server/Rack/Pod

##### 4.5.1.4 Connecting the Scale-Out Network to the Scale-Up Domain

##### 4.5.1.5 Scale-Up Interfaces: NVLink/NVSwitch, Infinity Fabric, and UALink/UALoE

##### 4.5.1.6 NPU Fabrics: TPU ICI, NeuronLink/NeuronSwitch, and Huawei UnifiedBus

##### 4.5.1.7 Distinguishing Physical Rack Boundaries, Network Domain Boundaries, and Failure Domains

##### 4.5.1.8 Capacity, Distance, and Infrastructure Cost of Extending a Scale-Up Domain

#### 4.5.2 Node-Scale Scale-Up Interconnect

##### 4.5.2.1 PCIe-Switched Attachment and Host/Root Complex

##### 4.5.2.2 Direct Full Mesh and the Physical Complete Graph

##### 4.5.2.3 Dedicated Switched Any-to-Any Fabric

##### 4.5.2.4 Gaudi, AMD Baseboard, and HGX/NVSwitch Cases

##### 4.5.2.5 Single-Hop Paths, Cable Count, Switch Capacity, and Endpoint Injection

##### 4.5.2.6 Topology Annotations Tied to System Configuration and Generation

#### 4.5.3 Rack-Scale Scale-Up Interconnect

##### 4.5.3.1 NVL72 and Rack-Scale Switched Fabric

##### 4.5.3.2 Cross-Server Organization of the Trainium UltraServer

##### 4.5.3.3 2D/3D Torus, Nearest-Neighbor, and Wraparound

##### 4.5.3.4 Dragonfly Local Groups and Global Links

##### 4.5.3.5 Switch Chips, Bounded Degree, Diameter, and Cabling Cost

##### 4.5.3.6 Compute Tray/Switch Tray and Interconnect Distance

#### 4.5.4 Pod-Scale Scale-Up Interconnect

##### 4.5.4.1 TPU Pod, 3D Torus, and Optical Circuit Switching

##### 4.5.4.2 Boardfly: Four-Chip Building Block, Group, and Inter-Group Connection

##### 4.5.4.3 UB-Mesh: Hierarchically Localized nD Full Mesh

##### 4.5.4.4 Maia: Fully Connected Quad and Ethernet-Switched Hierarchy

##### 4.5.4.5 Local/Global Connectivity, Oversubscription, and Scaling Limits

##### 4.5.4.6 Distinguishing Deployed, Announced, and Proposed Topologies

#### 4.5.5 Scale-Out, Network Datapaths, and Optical Interconnect

##### 4.5.5.1 Ethernet, InfiniBand, RoCE, and EFA

##### 4.5.5.2 Datapaths Among NIC, RDMA, and Accelerator Memory

##### 4.5.5.3 Leaf–Spine/Clos and Multi-Tier Networks

##### 4.5.5.4 Multi-Rail, Topology-Aware Placement, and Traffic Isolation

##### 4.5.5.5 Flow Control, Congestion Control, Link Failure, and Rerouting

##### 4.5.5.6 Electrical Link, Optical Link, OCS, and Co-Packaged Optics

##### 4.5.5.7 In-Network Reduction, SmartNIC/DPU, and Host Offload

#### 4.5.6 Topology Performance: Connectivity Is Not Effective Communication Performance

##### 4.5.6.1 Physical Link, Logical Reachability, and Communication Path

##### 4.5.6.2 Degree/Radix, Diameter, Hop Count, and Path Diversity

##### 4.5.6.3 Endpoint Injection, Switch Capacity, and Link Direction

##### 4.5.6.4 Bisection Bandwidth, Oversubscription, and Hotspots

##### 4.5.6.5 Topology Partitioning, Message Size, Concurrent Flows, and Link Asymmetry

##### 4.5.6.6 Copper Reach, Optical Propagation, and System Synchronization Latency

##### 4.5.6.7 Aligning Platforms, Bandwidth Units, and Bidirectional Aggregation Conventions

<a id="collective-communication"></a>

### 4.6 Collective Communication

#### 4.6.1 Collective Semantics: Endpoint Data Transformation

##### 4.6.1.1 AllReduce: Aggregation and Full-Result Replication

##### 4.6.1.2 AllGather: Shard Collection and Concatenation

##### 4.6.1.3 ReduceScatter: Aggregation with Retained Result Shards

##### 4.6.1.4 All-to-All: Destination-Rank-Oriented Data Exchange

##### 4.6.1.5 Implementing AllReduce as ReduceScatter + AllGather

##### 4.6.1.6 Communication Requirements of DDP, FSDP/ZeRO, TP/SP, and EP

##### 4.6.1.7 Distinguishing Collective Operation, Algorithm, and Physical Topology

#### 4.6.2 Collective Algorithms: Logical Communication Scheduling

##### 4.6.2.1 Ring ReduceScatter/AllGather and Pipelining

##### 4.6.2.2 Tree and Double Binary Tree

##### 4.6.2.3 Recursive Halving/Doubling and Rabenseifner-Style AllReduce

##### 4.6.2.4 Bruck-Style and Pairwise Exchange

##### 4.6.2.5 Parallel Aggregated Trees (PAT)

##### 4.6.2.6 Hierarchical Local–Global–Local Scheduling

##### 4.6.2.7 TACCL and Topology-Specific Schedule Synthesis

##### 4.6.2.8 Message Size, Rank Count, Startup Rounds, and Data Traffic

#### 4.6.3 Collective–Topology Mapping and Communication Offload

##### 4.6.3.1 Concurrent Ring, Tree, and Pairwise Exchange on Direct/Switched Fabrics

##### 4.6.3.2 Dimensional Decomposition and Multi-Ring Embedding on Mesh/Torus

##### 4.6.3.3 Intra-Group–Inter-Group–Intra-Group Scheduling on Dragonfly/Boardfly

##### 4.6.3.4 Hierarchical Locality and Link Sharing in UB-Mesh/Maia

##### 4.6.3.5 NVLS/CollNet-Style Reduction Offload

##### 4.6.3.6 Data Movement in AllGather/All-to-All and the Limits of Reduction Offload

##### 4.6.3.7 Communication Placement, Link Contention, and Compute–Communication Overlap

<a id="generational-evolution"></a>

### 4.7 Generational Evolution and Cross-Generation Comparison

#### 4.7.1 GPU Generational Evolution: Compute, Data Supply, and Cooperation Scope

##### 4.7.1.1 NVIDIA: Pascal → Volta → Turing → Ampere → Hopper → Blackwell

##### 4.7.1.2 NVIDIA: Cross-Generation Comparison of Tensor Core Datapaths and Control Granularity

##### 4.7.1.3 NVIDIA: Rubin Disclosures and Preliminary Specification Boundaries

##### 4.7.1.4 AMD: MI50 → CDNA/MI100 → MI200 → MI300 → MI350/MI400

##### 4.7.1.5 AMD: Cross-Generation Comparison of Matrix Compute, On-Chip Memory, and Interconnect

##### 4.7.1.6 The Divergent Evolution of Scalar/Vector Throughput and Low-Precision Matrix Throughput

##### 4.7.1.7 Co-Evolution of Arithmetic Capability, Memory Supply, Execution Coordination, and Deployment Form

#### 4.7.2 Specialized Accelerator Generations: Workload Positioning and Hardware–Software Co-Design

##### 4.7.2.1 TPU v1 → v2/v3: Bridging Inference and Training Designs

##### 4.7.2.2 TPU v4/v5e/v5p/v6e/v7: Compute, Memory, and Pod Evolution

##### 4.7.2.3 TPU 8t/8i: Workload Positioning and Disclosure Boundaries

##### 4.7.2.4 AWS Inferentia/Trainium: NeuronCore and Multi-Engine Evolution

##### 4.7.2.5 Trn1/Trn2: Intra-Instance Torus and Inter-UltraServer Connection

##### 4.7.2.6 Trn3: Switched Fabric Evolution

##### 4.7.2.7 MTIA/Maia: Shifts Across Recommendation, Generative AI, and Inference Workloads

#### 4.7.3 Cross-Generation Comparison: Numerical Formats, Structured Sparsity, and Peak Conventions

##### 4.7.3.1 IEEE, AI-Optimized, Block-Scaled, and Integer Formats

##### 4.7.3.2 Separating Input Precision, Accumulation Precision, and Output Precision

##### 4.7.3.3 Shared Scales and Representation Overhead in Microscaling

##### 4.7.3.4 2:4 and Variable M:N Structured Sparsity

##### 4.7.3.5 Compressed Weights, Non-Zero Metadata, and Regular Datapaths

##### 4.7.3.6 Supported Formats, Supported Operations, and Operator Mapping Conditions

##### 4.7.3.7 Dense Peak, Sparse Peak, and End-to-End Effective Performance

#### 4.7.4 Cross-Generation Comparison: Memory Hierarchy and Scale-Up Fabric

##### 4.7.4.1 HBM Capacity: Die Density, Stack Height, and Stack Count

##### 4.7.4.2 HBM Bandwidth: Signaling Rate, Interface Width, and Stack Count

##### 4.7.4.3 Shared Cache, Scratchpad, and On-Chip Reuse Capacity

##### 4.7.4.4 Chiplet Packaging and Memory PHY/Controller Integration

##### 4.7.4.5 GPU Fabric: Direct Connection, Switched Node, and Rack Domain

##### 4.7.4.6 Neuron: Torus, Inter-Instance Ring, and Switched Fabric

##### 4.7.4.7 TPU: Design Trade-offs Among Torus, OCS, and Boardfly

##### 4.7.4.8 Joint Comparison of Device Memory, Platform Topology, and Cross-Generation Data Conventions

<a id="power-and-cooling"></a>

### 4.8 Power Delivery and Cooling

#### 4.8.1 Power Delivery, Cooling, and Practically Operable Capacity

##### 4.8.1.1 PCIe Air Cooling, SXM/OAM, and Rack-Scale Direct Liquid Cooling

##### 4.8.1.2 Power Budgets at Chip, Board, Rack, and Facility

##### 4.8.1.3 Power Distribution, Busbar, Voltage Conversion, and Conductor Constraints

##### 4.8.1.4 54 VDC and the Proposed Higher-Voltage Delivery Direction

##### 4.8.1.5 Coolant Temperature, Thermal Resistance, and Operating Temperature Margin

##### 4.8.1.6 Integration Density, Copper Interconnect Distance, and Cooling Form Factor

##### 4.8.1.7 Average Power, Synchronized Power Swings, and Short-Term Energy Storage

##### 4.8.1.8 Installed Capacity, Power-Capped Capacity, and Cooling-Limited Capacity

##### 4.8.1.9 Reliability Extension: ECC, Link RAS, Hardware Fault Isolation, and Maintenance

<a id="future-directions-and-evaluation"></a>

### 4.9 Future Directions and Architectural Evaluation

#### 4.9.1 Future Design Challenges: Generality versus Specialization

##### 4.9.1.1 Workload Specialization and Fleet Flexibility

##### 4.9.1.2 Data Placement and Movement

##### 4.9.1.3 Communication Across Scale-Up and Scale-Out Networks

##### 4.9.1.4 Power Delivery and Cooling from Rack to Facility

##### 4.9.1.5 Architectural Specialization and Model Evolution

##### 4.9.1.6 The Boundary Between a Reconfigurable Model Graph and a Hard-Wired Base Model

##### 4.9.1.7 Taalas HC1: A Case of Adapter Tunability with a Frozen Base Model

##### 4.9.1.8 Software Change, Hardware Lifetime, and Reuse of Deployed Resources

#### 4.9.2 Extensions and Contrast Cases: Photonics, Neuromorphic, and Limited-Disclosure Designs

##### 4.9.2.1 Distinguishing Photonic Compute from Optical Interconnect

##### 4.9.2.2 Lightmatter: Treating Compute and Interconnect Paths Separately

##### 4.9.2.3 Q.ANT: A Public Photonic NPU Case

##### 4.9.2.4 Luminous: A Historical Photonic-Interconnect/Electronic-Compute Case

##### 4.9.2.5 SpiNNcloud SpiNNaker2: Event/Spike-Driven Manycore

##### 4.9.2.6 Tesla FSD: An Automotive NPU Contrasted with Datacenter Constraints

##### 4.9.2.7 OpenAI–Broadcom/Jalapeño: Public Disclosure versus Unconfirmed Internal Structure

##### 4.9.2.8 Research Extension: FPGAs and Other Reconfigurable Computing

#### 4.9.3 Architectural Evaluation and Full-Stack Case Organization

##### 4.9.3.1 A Unified Case Template: Compute → Data Path → Memory → Host → Fabric

##### 4.9.3.2 A Unified Software Template: Framework/Compiler/Libraries/Runtime/Driver/Firmware/Communication/ISA

##### 4.9.3.3 Programming-Model Rationale and Tracing Public Evidence

##### 4.9.3.4 Comparative Mapping of GEMM/GEMV/Attention/MoE/Embedding

##### 4.9.3.5 Separating Hardware Capability, Kernel Utilization, and Full-Model Performance

##### 4.9.3.6 The Pareto Frontier of Performance, Energy, Area, Cost, and Programmability

##### 4.9.3.7 Annotating Source Dates, Reference Platforms, Software Versions, and Incomparable Data

##### 4.9.3.8 Model Evolution, Resource Reuse, and Cross-Layer Design Trade-offs

<a id="system"></a>

## 5. System | Training, Inference, and AI Infrastructure

<a id="runtime-and-capacity"></a>

### 5.1 Runtime Foundations, Execution Chain, and Capacity Planning

#### 5.1.1 AI Runtime and Operating System Fundamentals

##### 5.1.1.1 Processes, Threads, Coroutines, and the Python GIL

##### 5.1.1.2 CPU Affinity, NUMA, and Host Memory

##### 5.1.1.3 GPU Context, Stream, Event, and CUDA Graph

##### 5.1.1.4 IPC, Shared Memory, RPC, and Serialization

##### 5.1.1.5 Memory Allocators, Memory Pools, and Fragmentation

##### 5.1.1.6 Driver, Runtime, Library, and Container Dependencies

#### 5.1.2 The Execution Chain of Device Runtime, Driver, and Firmware

##### 5.1.2.1 Model Loading, Executable Artifacts, and Device Resource Initialization

##### 5.1.2.2 Context, Command Queue, Doorbell, Event, and Interrupt

##### 5.1.2.3 Device Memory Allocation, Virtual Memory, and Host–Device Transfer

##### 5.1.2.4 The Boundary Between the CUDA Runtime/Driver API and the GPU Kernel Module

##### 5.1.2.5 The Boundary Among HIP/ROCr-HSA, AQL Queue, and amdgpu/KFD

##### 5.1.2.6 Neuron libnrt/NEFF, AscendCL, and Device-Specific Execution

##### 5.1.2.7 TT-Metalium Host/Device Programs and User-Space/Kernel-Space Drivers

##### 5.1.2.8 Distinguishing a Thin Static Executor from a Dynamic Kernel-Launch Runtime

#### 5.1.3 System Metrics, Queueing, and Capacity Planning

##### 5.1.3.1 TTFT, TPOT/ITL, and End-to-End Latency

##### 5.1.3.2 Request Throughput, Token Throughput, and Goodput

##### 5.1.3.3 Latency Percentiles, SLOs, and Tail Latency

##### 5.1.3.4 Arrival Process, Queueing, and Little's Law

##### 5.1.3.5 MFU, HFU, Device Utilization, and Effective Work

##### 5.1.3.6 Tokens/Dollar, Tokens/Joule, and Total Cost of Ownership

##### 5.1.3.7 Capacity, Concurrency, Batch Size, and Load Inflection Points

#### 5.1.4 Model Workloads and Resource Profiles

##### 5.1.4.1 Training, Prefill, Decode, and Verification

##### 5.1.4.2 Parameters, Gradients, Optimizer States, and Activations

##### 5.1.4.3 KV Cache, Recurrent State, and Model State

##### 5.1.4.4 Execution Profiles of Dense, MoE, Multimodal, and Diffusion Models

##### 5.1.4.5 Compute/Memory/Communication/CPU Bottlenecks

##### 5.1.4.6 Batch Size, Context Length, and Output Length Distribution

<a id="distributed-parallelism"></a>

### 5.2 Distributed Runtime and Parallelism Strategies

#### 5.2.1 Distributed Runtime and Collective Communication

##### 5.2.1.1 Rank, World Size, Process Group, and Device Mesh

##### 5.2.1.2 Collective Call Semantics, Completion Semantics, and Consistent Call Ordering

##### 5.2.1.3 Broadcast, Reduce, AllReduce, AllGather, ReduceScatter, and All-to-All

##### 5.2.1.4 NCCL/RCCL, Gloo/MPI/UCX, and Backend Selection

##### 5.2.1.5 The Differing Boundaries of NCCom, ICI Runtime, and Compiler-Built-In Collectives

##### 5.2.1.6 Topology Discovery, Algorithm Selection, and Communication Performance Tuning

##### 5.2.1.7 Asynchronous Communication, Stream/Event, and Compute Overlap

##### 5.2.1.8 Communication Deadlock, Timeout, Failure, and Diagnosis

#### 5.2.2 Data Parallelism and Parameter Sharding

##### 5.2.2.1 Synchronous SGD and Distributed Data Parallel

##### 5.2.2.2 Gradient Bucketing and Communication Overlap

##### 5.2.2.3 ZeRO-1/2/3 and State Sharding

##### 5.2.2.4 FSDP/FSDP2 and Parameter All-Gather

##### 5.2.2.5 Gradient Accumulation and Effective Batch Size

##### 5.2.2.6 Replicated, Sharded, and Hybrid Sharded Training

#### 5.2.3 Tensor Parallelism

##### 5.2.3.1 Column-Parallel and Row-Parallel Linear Layers

##### 5.2.3.2 Partitioning Attention Heads and MLPs

##### 5.2.3.3 Vocabulary, Embedding, and Loss Parallelism

##### 5.2.3.4 All-Reduce/All-Gather/Reduce-Scatter in TP

##### 5.2.3.5 KV Head Count, Replication, and Partitioning Constraints

##### 5.2.3.6 Single-Node and Cross-Node Tensor Parallelism

#### 5.2.4 Pipeline Parallelism

##### 5.2.4.1 Stage Partitioning and Microbatches

##### 5.2.4.2 GPipe, 1F1B, and Interleaved Pipelines

##### 5.2.4.3 Pipeline Bubbles and Load Balance

##### 5.2.4.4 Zero-Bubble Scheduling and Execution Dependencies

##### 5.2.4.5 Activation Communication and Cross-Stage Overlap

##### 5.2.4.6 Virtual Pipeline Stages and Dynamic Repartitioning

#### 5.2.5 Sequence and Context Parallelism

##### 5.2.5.1 The Boundary Between Sequence Parallelism and Context Parallelism

##### 5.2.5.2 Sequence Partitioning for Long-Context Attention

##### 5.2.5.3 Ring Attention and KV Block Passing

##### 5.2.5.4 Ulysses and All-to-All Sequence Redistribution

##### 5.2.5.5 Hierarchical, Hybrid, and Two-Dimensional Context Parallelism

##### 5.2.5.6 Causal Masking, Load Balance, and Communication Hiding

#### 5.2.6 Expert Parallelism

##### 5.2.6.1 Expert Placement, Replication, and Sharding

##### 5.2.6.2 Token Dispatch, All-to-All, and Combine

##### 5.2.6.3 Combining Expert Parallelism with Tensor Parallelism

##### 5.2.6.4 Expert Load Balance and Hot Experts

##### 5.2.6.5 Dropless Routing and Capacity Management

##### 5.2.6.6 Cross-Node MoE Communication and Topology-Aware Deployment

#### 5.2.7 Hybrid Parallelism and Automatic Planning

##### 5.2.7.1 Combining DP/TP/PP/SP/CP/EP

##### 5.2.7.2 Device Mesh, Parallel Groups, and Topological Layering

##### 5.2.7.3 Model Partitioning, Resharding, and Layout Conversion

##### 5.2.7.4 Memory Constraints, Message Size, Communication Latency, and Parallelism Choice

##### 5.2.7.5 Communication Placement Across Scale-Up/Scale-Out Domains

##### 5.2.7.6 Heterogeneous Devices, Stragglers, and Uneven Sharding

##### 5.2.7.7 Automatic Parallelization, Search, and Performance Models

<a id="training-memory-data-and-runtime"></a>

### 5.3 Training Memory, Data, Runtime, and Numerical Stability

#### 5.3.1 Training Memory Management

##### 5.3.1.1 Activation Checkpointing and Selective Recomputation

##### 5.3.1.2 CPU/NVMe Offload and State Tiering

##### 5.3.1.3 Activation Offload, Prefetch, and Asynchronous Transfer

##### 5.3.1.4 Optimizer State Sharding and Low-Precision States

##### 5.3.1.5 Memory Budget, Peak Memory, and Fragmentation

##### 5.3.1.6 Training Graphs, Communication Buffers, and CUDA Graph Memory

#### 5.3.2 Training Data and Checkpoint Pipelines

##### 5.3.2.1 Tokenization, Packing, and Dynamic Batching

##### 5.3.2.2 Dataloader, Prefetch, and the CPU–GPU Pipeline

##### 5.3.2.3 Distributed Data Sharding, Shuffling, and Recovery

##### 5.3.2.4 Checkpoint Save/Load and Parallel I/O

##### 5.3.2.5 Asynchronous Checkpointing and Incremental Saving

##### 5.3.2.6 Consistency Among Data, Parameters, Optimizer, and Random State

#### 5.3.3 Training Runtime and Numerical Stability

##### 5.3.3.1 Training Step, Autograd, and Optimizer Step

##### 5.3.3.2 Mixed-Precision Training and the Scope of Hardware/Kernel Support

##### 5.3.3.3 Loss Scaling, Gradient Clipping, and Anomaly Detection

##### 5.3.3.4 Communication Precision, Accumulation Order, and Reproducibility

##### 5.3.3.5 Diagnosing NaN/Inf, Loss Spikes, and Training Divergence

##### 5.3.3.6 Co-Optimizing Operators, Compilers, and Distributed Training

<a id="resilient-training"></a>

### 5.4 Resilient Distributed Training and Frameworks

#### 5.4.1 Elasticity and Fault Tolerance in Large-Scale Training

##### 5.4.1.1 Failure Domains, Heartbeats, and Health Checks

##### 5.4.1.2 Checkpoint/Restart and Fast Recovery

##### 5.4.1.3 Elastic Training and Dynamic World Size

##### 5.4.1.4 Straggler Detection and Slow-Node Replacement

##### 5.4.1.5 Network Failures, Storage Failures, and Silent Data Corruption

##### 5.4.1.6 Training Efficiency, Failure Cost, and Available Capacity

#### 5.4.2 Distributed Training Frameworks and Accelerator Backends

##### 5.4.2.1 PyTorch Distributed: DDP, FSDP, and DTensor

##### 5.4.2.2 Megatron-LM/Megatron-Core and DeepSpeed

##### 5.4.2.3 TorchTitan and Native PyTorch Training

##### 5.4.2.4 JAX/MaxText and TPU Training

##### 5.4.2.5 NeMo and Training Workflow Integration

##### 5.4.2.6 AWS NxD Training, Ascend Adaptation, and Device-Specific Training Paths

##### 5.4.2.7 Distinguishing Unified Training Concepts from Per-Backend Support Coverage

<a id="inference-lifecycle-and-caching"></a>

### 5.5 Inference Lifecycle, Scheduling, and Caching

#### 5.5.1 The Full Lifecycle of an Inference Request

##### 5.5.1.1 API Server, Tokenizer, and Request Queue

##### 5.5.1.2 Scheduler, Worker, Model Runner, and Executor

##### 5.5.1.3 Model Configuration, Weight Loading, and Parameter Initialization

##### 5.5.1.4 Prefill, Decode, Sampling, and Detokenization

##### 5.5.1.5 Streaming Output, Cancellation, and Resource Reclamation

##### 5.5.1.6 Embedding, Reranking, Reward, and Generation Requests

#### 5.5.2 Batching and Online Scheduling

##### 5.5.2.1 Static Batching, Dynamic Batching, and Continuous Batching

##### 5.5.2.2 Iteration-Level Scheduling and Token Budgets

##### 5.5.2.3 Chunked Prefill and Decode Priority

##### 5.5.2.4 Preemption, Recompute, and Swap

##### 5.5.2.5 CPU–GPU Overlap and Low-Overhead Schedulers

##### 5.5.2.6 Fairness, Priority, and Mixing Short and Long Requests

##### 5.5.2.7 Load-Aware Batch Size and Adaptive Scheduling

#### 5.5.3 KV Cache Allocation, Addressing, and Lifecycle

##### 5.5.3.1 Contiguous, Paged, and Blocked KV Cache

##### 5.5.3.2 Block Table, Memory Pool, and Dynamic Growth

##### 5.5.3.3 Cache Organization for MHA/GQA/MLA

##### 5.5.3.4 Prefix Sharing, Reference Counting, and Copy-on-Write

##### 5.5.3.5 KV Cache Eviction, Compaction, and Fragmentation

##### 5.5.3.6 Diagnosing Cache Hits, OOM, and Memory Budget

#### 5.5.4 Prefix Cache and Multi-Tier Caching

##### 5.5.4.1 Prefix Caching with Radix-Tree/Hash Indexing

##### 5.5.4.2 Cache-Aware Scheduling and Prefix Locality

##### 5.5.4.3 HiCache: HBM, Host DRAM, and External Storage

##### 5.5.4.4 LMCache, Mooncake, and Shared KV Storage

##### 5.5.4.5 Cache Prefetch, Migration, and Asynchronous Write-Back

##### 5.5.4.6 Cache Coherence Across Processes, Devices, and Nodes

##### 5.5.4.7 Multi-Tenant Cache Isolation and Invalidation Management

<a id="distributed-serving"></a>

### 5.6 Disaggregated and Distributed Serving

#### 5.6.1 Disaggregated Serving

##### 5.6.1.1 Prefill–Decode Disaggregation and Per-Phase Resource Needs

##### 5.6.1.2 KV Transfer, Connectors, and the Data Plane

##### 5.6.1.3 Independent Scaling of Prefill and Decode

##### 5.6.1.4 Heterogeneous Hardware Selection, Data Formats, and Migration Cost

##### 5.6.1.5 Attention–FFN Disaggregation and Fine-Grained Splitting

##### 5.6.1.6 Resource-Pool Imbalance, Queue Growth, and Idle Capacity

##### 5.6.1.7 Splitwise, DistServe, and Mooncake Case Studies

#### 5.6.2 Routing and Distributed Serving Scheduling

##### 5.6.2.1 Load Balancing, Join-Shortest-Queue, and Request Routing

##### 5.6.2.2 KV-Aware, Prefix-Aware, and Cache-Aware Routing

##### 5.6.2.3 Replica Placement and Session Affinity

##### 5.6.2.4 Admission Control, Backpressure, and Rate Limiting

##### 5.6.2.5 Autoscaling, Cold Start, and Warm Pools

##### 5.6.2.6 Cross-Rack, Cross-Region, and Tiered Serving Architectures

#### 5.6.3 System Design for MoE Serving

##### 5.6.3.1 Expert Parallel Serving and Communication Paths

##### 5.6.3.2 Combining Attention-DP with Expert Parallelism

##### 5.6.3.3 Expert Parallel Load Balancing

##### 5.6.3.4 Hot Expert Replication and Dynamic Placement

##### 5.6.3.5 Expert Offload, Prefetch, and CPU–GPU Cooperation

##### 5.6.3.6 Shared Experts, Routed Experts, and Cross-Device Execution

<a id="optimized-model-serving"></a>

### 5.7 Optimized Model Serving

#### 5.7.1 Deploying and Tuning Quantized Models

##### 5.7.1.1 Precision Configuration for Weights, Activations, and KV Cache

##### 5.7.1.2 Checkpoint Format, Packing, and Quantization Metadata

##### 5.7.1.3 Matching Quantization Schemes, Kernels, and Hardware Support

##### 5.7.1.4 Offline Quantization, Online Quantization, and Load-Time Conversion

##### 5.7.1.5 Mixed-Precision Deployment for Dense/MoE/Multimodal Models

##### 5.7.1.6 Joint Evaluation of Precision, Latency, Throughput, Capacity, and Cost

#### 5.7.2 System Support for Sparse Models and Long Context

##### 5.7.2.1 Weight Sparsity and Sparse Weight Storage

##### 5.7.2.2 Indexing, Routing, and Cache Access in Sparse Attention

##### 5.7.2.3 Dynamic Token Selection and Batch Organization

##### 5.7.2.4 KV Pruning/Compression and Memory Management

##### 5.7.2.5 Load Balancing and Communication Under Dynamic Sparsity

##### 5.7.2.6 Sparse Compute Gains Versus Preprocessing and Metadata Overhead

#### 5.7.3 System Implementation of Speculative Decoding

##### 5.7.3.1 Draft Model, Target Model, and Verification Worker

##### 5.7.3.2 Same-Device, Separate-Device, and Heterogeneous Draft/Target Deployment

##### 5.7.3.3 Acceptance-Aware and Load-Aware Draft Length

##### 5.7.3.4 Tree/Block Verification and Batch Capacity

##### 5.7.3.5 Draft/Target KV Cache Management and Rollback

##### 5.7.3.6 Trade-offs Among Online Requests, Throughput, and Single-Request Latency

##### 5.7.3.7 Draft Model Training, Updating, and Service Integration

<a id="multi-model-serving"></a>

### 5.8 Multi-Model Serving and Backend Ecosystems

#### 5.8.1 Multi-Model, Multi-Tenant, and Adapter Serving

##### 5.8.1.1 Multi-Model Serving and Model Routing

##### 5.8.1.2 Multi-LoRA Batching and Adapter Cache

##### 5.8.1.3 Weight Sharing, Hot Loading, and Dynamic Model Switching

##### 5.8.1.4 GPU Sharing, MIG, MPS, and Resource Isolation

##### 5.8.1.5 QoS-Constrained Co-Location and Interference Management

##### 5.8.1.6 Tenant Fairness, Quotas, and SLO Isolation

#### 5.8.2 Serving Frameworks, Component Ecosystem, and Device Backends

##### 5.8.2.1 vLLM: Engine, Scheduler, Worker, and PagedAttention

##### 5.8.2.2 SGLang: Scheduler, RadixAttention, and Model Runner

##### 5.8.2.3 TensorRT-LLM and NVIDIA Triton Inference Server

##### 5.8.2.4 NVIDIA Dynamo, llm-d, and Distributed Inference Orchestration

##### 5.8.2.5 Ray Serve and Multi-Stage Serving Pipelines

##### 5.8.2.6 llama.cpp, MLX, and CPU/Local Inference

##### 5.8.2.7 FlashInfer, NIXL, and Reusable Inference Components

##### 5.8.2.8 The Backend Compatibility Matrix: Model, Precision, Kernel, Device, and SDK Version

##### 5.8.2.9 The Boundary Between Vendor SDK/Serving Entry Points and General Serving Frameworks

<a id="rl-infrastructure"></a>

### 5.9 RL Post-Training Infrastructure and Correctness

#### 5.9.1 RL Post-Training Infrastructure

##### 5.9.1.1 Actor, Rollout, Reward, Critic, and Reference Model

##### 5.9.1.2 Combining the Training Engine and the Inference Engine

##### 5.9.1.3 Colocated, Disaggregated, and Hybrid Resource Layouts

##### 5.9.1.4 Synchronous/Asynchronous RL Pipelines

##### 5.9.1.5 Partial Rollout, Dynamic Sampling, and Stragglers

##### 5.9.1.6 Multi-Turn Rollout, Environment Interaction, and Tool Execution

##### 5.9.1.7 Online Weight Update and Weight Broadcast

#### 5.9.2 Training–Inference Consistency and RL Correctness

##### 5.9.2.1 Aligning Tokenizer, Chat Template, and Special Tokens

##### 5.9.2.2 Log Probability, Masking, and Sequence Boundaries

##### 5.9.2.3 Consistency Between the Sampling Distribution and the Training Objective

##### 5.9.2.4 Numerical Differences Across Kernels, Precision, and Reduction Paths

##### 5.9.2.5 Policy Version, Weight Staleness, and Off-Policy Drift

##### 5.9.2.6 Importance Sampling and Mismatch Correction

##### 5.9.2.7 Correctness Validation of Gradients, Samples, and Performance Optimizations

#### 5.9.3 RL Frameworks and Source-Code Studies

##### 5.9.3.1 verl: Initialization, Rollout, and Training Workflow

##### 5.9.3.2 slime: Rollout-First Design and Backend Integration

##### 5.9.3.3 OpenRLHF and Ray-Based Actor Management

##### 5.9.3.4 AReaL and Asynchronous RL Training

##### 5.9.3.5 TRL and Lightweight Post-Training Workflows

##### 5.9.3.6 Weight Update, Memory Sleep/Wake, and Resource Reuse

##### 5.9.3.7 FP8/INT4, Speculative Decoding, and Multi-Turn Tasks in RL

<a id="multimodal-and-compound-systems"></a>

### 5.10 Multimodal, Generative, and Compound AI Systems

#### 5.10.1 Multimodal and Real-Time Speech Serving

##### 5.10.1.1 Image/Video Preprocessing and Vision Encoders

##### 5.10.1.2 Multi-Stage Encoder–Decoder Scheduling

##### 5.10.1.3 Vision Tokens, Cross-Modal Cache, and Batching

##### 5.10.1.4 Audio Codec, Dual-AR, and the Thinker–Talker Pipeline

##### 5.10.1.5 Streaming ASR, TTS, Vocoder, and Full-Duplex Interaction

##### 5.10.1.6 CPU Resources, Audio/Video I/O, and Real-Time Latency Budgets

#### 5.10.2 Diffusion and Non-Autoregressive Model Systems

##### 5.10.2.1 Denoising Steps, Schedulers, and Multi-Stage Execution

##### 5.10.2.2 CFG Parallelism and Model Parallelism

##### 5.10.2.3 Sequence/Patch/Temporal Parallelism

##### 5.10.2.4 Cross-Step Feature Cache and Compute Reuse

##### 5.10.2.5 Block Scheduling and Caching for Diffusion LLMs

##### 5.10.2.6 Memory, Throughput, and Service Orchestration for Image/Video Generation

#### 5.10.3 RAG, Agents, and Compound AI Systems

##### 5.10.3.1 Retrieval, Reranking, Generation, and the Index Pipeline

##### 5.10.3.2 Vector Databases, Embeddings, and Retrieval Caching

##### 5.10.3.3 Tool Calling, Sandboxing, and Environment Resource Management

##### 5.10.3.4 Multi-Turn Sessions, Long-Term State, and Context Management

##### 5.10.3.5 Agent Workflows, Parallel Tools, and Long-Tail Task Scheduling

##### 5.10.3.6 Model Routing, Cascades, and System-Level Caching

##### 5.10.3.7 End-to-End Quality, Latency, and Cost of Compound Tasks

<a id="production-operations"></a>

### 5.11 Production Orchestration, Efficiency, Security, and Observability

#### 5.11.1 Cluster Orchestration and Production Deployment

##### 5.11.1.1 Slurm, Kubernetes, and GPU Resource Scheduling

##### 5.11.1.2 Gang Scheduling, Quotas, and Cluster Fairness

##### 5.11.1.3 Docker, Images, Dependencies, and Reproducible Environments

##### 5.11.1.4 Model Registry, Artifact Storage, and Release Pipelines

##### 5.11.1.5 Canary, Rolling Upgrade, and Rollback

##### 5.11.1.6 Multi-Cluster Deployment, Disaster Recovery, and Security Boundaries

#### 5.11.2 Scheduling Heterogeneous Fleets and Specialized Resource Pools

##### 5.11.2.1 Phase Profiles of Training, Prefill, Decode, Draft/Verify, and Multimodal Stages

##### 5.11.2.2 Unified Resource Pools versus Workload-Partitioned Specialized Pools

##### 5.11.2.3 Scheduling Constraints from Model Compatibility, Precision Support, and Available Kernels

##### 5.11.2.4 Capacity Fragmentation, Idle Devices, Queue Imbalance, and Resource Rebalancing

##### 5.11.2.5 Data-Movement Cost of Migrating Weights, KV Cache, and Activations

##### 5.11.2.6 Rack/Pod Placement and Cross-Network-Domain Communication

##### 5.11.2.7 Throughput, Tail Latency, Cost, and Reuse of Deployed Hardware

#### 5.11.3 Power/Thermal-Aware System Operation

##### 5.11.3.1 Connecting Device, Rack, and Facility Power Budgets

##### 5.11.3.2 Power Capping, Frequency Scaling, and Job Progress

##### 5.11.3.3 Synchronized Training Phases, Power Swings, and Short-Term Energy Buffering

##### 5.11.3.4 Cooling Capacity, Thermal Margin, and Concurrently Operable Device Count

##### 5.11.3.5 Jointly Considering Placement, Concurrency, Batch Size, and Power Limits

##### 5.11.3.6 Average Power, Peak Power, Tokens/Joule, and SLOs

##### 5.11.3.7 The Boundary Between Hardware Design Constraints and Runtime Scheduling Policy

#### 5.11.4 Security, Privacy, and Trustworthy AI Infrastructure

##### 5.11.4.1 Authentication, Authorization, Quotas, and Multi-Tenant Data Isolation

##### 5.11.4.2 Model Weights, Serialization Formats, and Software Supply-Chain Security

##### 5.11.4.3 Confidential Computing, TEEs, and Confidential Accelerator Compute

##### 5.11.4.4 KV Cache Isolation, Sensitive Data, and Side-Channel Defense

##### 5.11.4.5 Tool Sandboxing, Least Privilege, and Prompt Injection Defense

##### 5.11.4.6 Auditing, Data Provenance, and Privacy-Preserving Workflows

#### 5.11.5 Observability, Benchmarking, and Troubleshooting

##### 5.11.5.1 Metrics, Logs, Tracing, and Cross-Component Timelines

##### 5.11.5.2 PyTorch Profiler, Nsight, ROCm Tools, and Device-Specific Profilers

##### 5.11.5.3 Localizing Issues Across Framework/Compiler/Kernel/Runtime/Driver

##### 5.11.5.4 Joint Monitoring of GPU/NPU/CPU/Network/Storage

##### 5.11.5.5 Offline/Online Benchmarking and Real Traffic Replay

##### 5.11.5.6 NCCL Hangs, OOM, Memory Leaks, and CPU Bottlenecks

##### 5.11.5.7 Performance Regressions, Version Pinning, Numerical Consistency, and Experiment Reproducibility

##### 5.11.5.8 MLPerf, Serving Benchmarks, and Reporting Conventions

<a id="end-to-end-systems-practice"></a>

### 5.12 End-to-End Systems Practice

#### 5.12.1 Reading System Source Code and Cross-Layer End-to-End Practice

##### 5.12.1.1 Implementing a Mini Training Runtime from Scratch

##### 5.12.1.2 Implementing a Continuous-Batching LLM Server from Scratch

##### 5.12.1.3 The Complete Path of One Request Through SGLang/vLLM

##### 5.12.1.4 The Complete Path of One RL Step Through Rollout/Reward/Training

##### 5.12.1.5 From HF Checkpoint to Quantization, Multi-Device, and Disaggregated Deployment

##### 5.12.1.6 From Model Frontend to Device Execution: Cross-Platform Paths Within Public Interfaces

##### 5.12.1.7 Cross-Layer Diagnosis from Kernel to Compiler to Runtime to Driver to Hardware

##### 5.12.1.8 Deployment Differences for the Same Model Across Hardware/Programming Models

<a id="algorithms"></a>

## 6. Algorithms | Models, Training Methods, and Generation Algorithms

<a id="foundations-and-lifecycle"></a>

### 6.1 Foundations, Lifecycle, and Input Representation

#### 6.1.1 Deep Learning and Model Computation Fundamentals

##### 6.1.1.1 Tensor, Linear Layer, MLP, and Activation

##### 6.1.1.2 Loss, Gradient, Backpropagation, and Optimizer

##### 6.1.1.3 Batch, Sequence, Hidden Dimension, and Parameter Count

##### 6.1.1.4 Computation Graphs, Automatic Differentiation, and Training/Inference Differences

##### 6.1.1.5 Computational Complexity, Space Complexity, and Data Movement

##### 6.1.1.6 Dense, Sparse, and Conditional Computation

#### 6.1.2 The Full Lifecycle of a Foundation Model

##### 6.1.2.1 Pretraining, Continued Pretraining, and Mid-Training

##### 6.1.2.2 Supervised Fine-Tuning and Instruction Tuning

##### 6.1.2.3 Preference Optimization and Reinforcement Learning

##### 6.1.2.4 Distillation, Compression, and Deployment

##### 6.1.2.5 Inference, Test-Time Compute, and Continuous Evaluation

##### 6.1.2.6 Scaling Laws and Compute-Optimal Training

#### 6.1.3 Tokenization and Input Representation

##### 6.1.3.1 BPE, WordPiece, Unigram, and Byte-Level Tokenization

##### 6.1.3.2 Vocabulary, Embedding, and Output Projection

##### 6.1.3.3 Special Tokens, Chat Templates, and Conversation Formats

##### 6.1.3.4 Padding, Packing, Masking, and Sequence Boundaries

##### 6.1.3.5 Text, Image, Audio, and Video Tokens

##### 6.1.3.6 The Effect of Tokenization on Context Length and Compute Cost

<a id="transformers-and-attention"></a>

### 6.2 Transformers, Attention, and Sequence Models

#### 6.2.1 Transformer Structure and Its Evolution

##### 6.2.1.1 Encoder-Only, Decoder-Only, and Encoder–Decoder

##### 6.2.1.2 Self-Attention, Cross-Attention, and FFN

##### 6.2.1.3 Residual Connection, LayerNorm, and RMSNorm

##### 6.2.1.4 Pre-Norm, Post-Norm, and Deep-Network Stability

##### 6.2.1.5 GELU, GLU, GeGLU, and SwiGLU

##### 6.2.1.6 Weight Tying, Embedding, and the LM Head

#### 6.2.2 The Basic Mechanism of Attention

##### 6.2.2.1 Scaled Dot-Product Attention

##### 6.2.2.2 Q, K, V, and Attention Scores

##### 6.2.2.3 Causal, Bidirectional, and Cross-Attention Masks

##### 6.2.2.4 Softmax, Normalization, and Stability

##### 6.2.2.5 Compute, Memory, and Long-Sequence Complexity of Attention

##### 6.2.2.6 The Boundary Between Exact Attention and Sparse/Approximate Attention

#### 6.2.3 MHA, MQA, GQA, and MLA

##### 6.2.3.1 Multi-Head Attention (MHA)

##### 6.2.3.2 Multi-Query Attention (MQA)

##### 6.2.3.3 Grouped-Query Attention (GQA)

##### 6.2.3.4 Multi-Head Latent Attention (MLA)

##### 6.2.3.5 Head Sharing, Latent Compression, and Cache Capacity

##### 6.2.3.6 KV Cache and Incremental Autoregressive Inference

##### 6.2.3.7 Quality, Bandwidth, and Parallelism Trade-offs Among Attention Variants

#### 6.2.4 Position Encoding and Context Extension

##### 6.2.4.1 Absolute and Relative Position Embeddings

##### 6.2.4.2 RoPE, ALiBi, and Position Representation

##### 6.2.4.3 RoPE Scaling, Position Interpolation, and YaRN

##### 6.2.4.4 Sliding Window, Attention Sink, and Streaming Context

##### 6.2.4.5 Long-Context Training, Extrapolation, and Retrieval Ability

##### 6.2.4.6 Context Compression, Memory, and Cross-Segment State

#### 6.2.5 Sparse and Compressed Attention

##### 6.2.5.1 Token, Block, Head, and Layer-Level Sparsity

##### 6.2.5.2 Local, Global, Dilated, and Sliding-Window Attention

##### 6.2.5.3 Longformer, BigBird, and Sparse Connectivity Patterns

##### 6.2.5.4 Native Sparse Attention (NSA)

##### 6.2.5.5 Mixture of Block Attention (MoBA)

##### 6.2.5.6 DeepSeek Sparse Attention (DSA)

##### 6.2.5.7 Compressed Sparse Attention (CSA) and Heavily Compressed Attention (HCA)

##### 6.2.5.8 Dynamic Token Selection, Top-k, and Trainable Sparsity

#### 6.2.6 Linear Attention, SSMs, and Hybrid Models

##### 6.2.6.1 Kernelized Linear Attention and Linear Complexity

##### 6.2.6.2 RetNet, RWKV, and Recurrent State

##### 6.2.6.3 S4, Mamba, and Mamba-2

##### 6.2.6.4 DeltaNet and Gated DeltaNet

##### 6.2.6.5 Attention–SSM/Linear-Attention Hybrid Architectures

##### 6.2.6.6 Prefill Scan, Recurrent Decode, and State Capacity

<a id="mixture-of-experts"></a>

### 6.3 Mixture of Experts

#### 6.3.1 Mixture of Experts

##### 6.3.1.1 Dense FFN → Sparse MoE

##### 6.3.1.2 Router, Top-k Routing, and Expert Selection

##### 6.3.1.3 Token Choice, Expert Choice, and Capacity Constraints

##### 6.3.1.4 Shared Experts, Fine-Grained Experts, and Expert Partitioning

##### 6.3.1.5 Auxiliary Loss and Auxiliary-Loss-Free Balancing

##### 6.3.1.6 Expert Specialization, Routing Collapse, and Training Stability

##### 6.3.1.7 Activated Parameters, Total Parameter Count, and Effective Compute

<a id="training-alignment-and-reasoning"></a>

### 6.4 Training, Fine-Tuning, Alignment, and Reasoning

#### 6.4.1 Pretraining Objectives and Optimization Algorithms

##### 6.4.1.1 Causal LM, Masked LM, and Denoising Objectives

##### 6.4.1.2 Next-Token Prediction and Multi-Token Prediction

##### 6.4.1.3 SGD, AdamW, Adafactor, and Muon

##### 6.4.1.4 Learning Rate Schedule, Warm-up, and Weight Decay

##### 6.4.1.5 Gradient Clipping, Batch Scaling, and Optimization Stability

##### 6.4.1.6 Low-Precision Training, Error Accumulation, and Precision Sensitivity

#### 6.4.2 Data, Training Recipes, and Model Scaling

##### 6.4.2.1 Data Collection, Filtering, Deduplication, and Quality Assessment

##### 6.4.2.2 Data Mixture, Curriculum, and Multilingual Training

##### 6.4.2.3 Synthetic Data, Self-Training, and Data Distillation

##### 6.4.2.4 Model Width, Depth, Vocabulary, and Sequence Length

##### 6.4.2.5 Continual Learning, Domain Adaptation, and Catastrophic Forgetting

##### 6.4.2.6 Data Quality, Token Budget, and Scaling Trade-offs

#### 6.4.3 SFT and Parameter-Efficient Fine-Tuning

##### 6.4.3.1 Instruction Tuning and Conversational Supervision

##### 6.4.3.2 Full Fine-Tuning and Partial Parameter Updates

##### 6.4.3.3 Adapter, Prefix Tuning, and Prompt Tuning

##### 6.4.3.4 LoRA, QLoRA, DoRA, and Low-Rank Adaptation

##### 6.4.3.5 Multi-Task/Multi-Domain Fine-Tuning

##### 6.4.3.6 Adapter Merging, Model Merging, and Transfer

#### 6.4.4 Alignment and Preference Learning

##### 6.4.4.1 Reward Model, Preference Data, and Pairwise Ranking

##### 6.4.4.2 RLHF and RLAIF

##### 6.4.4.3 PPO, KL Regularization, and Reference Policy

##### 6.4.4.4 DPO, IPO, KTO, and Direct Preference Optimization

##### 6.4.4.5 Rejection Sampling and Best-of-N Data Selection

##### 6.4.4.6 Safety Alignment, Refusal, and Preference Generalization

#### 6.4.5 Reasoning and RL Post-Training

##### 6.4.5.1 Verifiable Rewards and RLVR

##### 6.4.5.2 GRPO, REINFORCE-Style Methods, and Group-Relative Advantage Estimation

##### 6.4.5.3 DAPO, Reward Shaping, and Sample Filtering

##### 6.4.5.4 Outcome Reward and Process Reward

##### 6.4.5.5 On-Policy, Off-Policy, and Importance Sampling

##### 6.4.5.6 Long Chain-of-Thought, Multi-Turn RL, and Agentic RL

##### 6.4.5.7 Reward Hacking, Length Bias, and Training Stability

<a id="inference-and-generation"></a>

### 6.5 Inference and Generation Strategies

#### 6.5.1 Autoregressive Inference and Sampling

##### 6.5.1.1 Prefill, Decode, and Token-by-Token Generation

##### 6.5.1.2 Greedy, Temperature, Top-k, and Top-p

##### 6.5.1.3 Beam Search, Length Penalty, and Repetition Control

##### 6.5.1.4 Logits Processor, Stopping Criteria, and Control Tokens

##### 6.5.1.5 Constrained Decoding, Grammars, and Structured Output

##### 6.5.1.6 Generation Quality, Diversity, Latency, and Reproducibility

#### 6.5.2 Test-Time Compute and Inference Strategies

##### 6.5.2.1 Chain-of-Thought and Explicit Reasoning

##### 6.5.2.2 Self-Consistency, Best-of-N, and Multi-Sample Selection

##### 6.5.2.3 Verifiers, Reward-Guided Search, and Reranking

##### 6.5.2.4 Tree Search, Planning, and Multi-Step Problem Solving

##### 6.5.2.5 Adaptive Reasoning Budget and Early Termination

##### 6.5.2.6 Inference-Time Scaling and Compute Allocation

#### 6.5.3 Speculative Decoding

##### 6.5.3.1 The Draft–Verify Framework and Speculative Sampling

##### 6.5.3.2 Acceptance/Rejection and Target-Distribution Preservation

##### 6.5.3.3 Independent Draft Models and Self-Speculation

##### 6.5.3.4 N-Gram, Prompt Lookup, and Retrieval-Based Drafting

##### 6.5.3.5 The Relationship Between Multi-Token Prediction and Speculative Decoding

##### 6.5.3.6 Medusa, Hydra, ReDrafter, and Multi-Head/Tree Drafting

##### 6.5.3.7 EAGLE, EAGLE-2, and EAGLE-3

##### 6.5.3.8 DFlash: Block-Diffusion Drafting

##### 6.5.3.9 DSpark: Semi-Autoregressive Drafting and Confidence-Scheduled Verification

##### 6.5.3.10 Acceptance Length, Draft Cost, Verification Cost, and the Limits of Speedup

#### 6.5.4 Parallel, Blockwise, and Diffusion Language Generation

##### 6.5.4.1 Autoregressive, Semi-Autoregressive, and Non-Autoregressive Generation

##### 6.5.4.2 Blockwise Parallel Decoding and Iterative Refinement

##### 6.5.4.3 Set Block Decoding (SBD)

##### 6.5.4.4 Masked Diffusion Language Modeling

##### 6.5.4.5 LLaDA, LLaDA 2.0, Dream, and Block Diffusion

##### 6.5.4.6 Token Update Order, Remasking, and Sampling Steps

##### 6.5.4.7 Trade-offs Between KV Cache Compatibility and Generation Quality

<a id="quantization-and-compression"></a>

### 6.6 Quantization, Sparsity, and Compression

#### 6.6.1 Quantization: Algorithms and Numerical Methods

##### 6.6.1.1 PTQ, QAT, and Quantization-Aware Distillation

##### 6.6.1.2 Symmetric/Asymmetric and Uniform/Non-Uniform Quantization

##### 6.6.1.3 Per-Tensor, Per-Channel, Per-Group, and Per-Token Scaling

##### 6.6.1.4 Weight, Activation, and KV Cache Quantization

##### 6.6.1.5 GPTQ, AWQ, SmoothQuant, and Error Compensation

##### 6.6.1.6 Rotation-Based Quantization: QuaRot and SpinQuant

##### 6.6.1.7 Outliers, Clipping, Calibration, and Sensitivity Analysis

##### 6.6.1.8 Mixed Precision, Block Scaling, and Precision Budgets

#### 6.6.2 Sparsification, Pruning, and Model Compression

##### 6.6.2.1 Unstructured, Structured, and N:M Pruning

##### 6.6.2.2 Magnitude, Gradient, and Second-Order Pruning

##### 6.6.2.3 SparseGPT, Wanda, and Post-Training Pruning

##### 6.6.2.4 Activation Sparsity, Token Pruning, and Early Exit

##### 6.6.2.5 Head, Layer, and Expert Pruning

##### 6.6.2.6 Low-Rank Factorization and Structural Compression

##### 6.6.2.7 Knowledge Distillation, Self-Distillation, and Teacher–Student Learning

<a id="multimodal-agents-and-non-llm"></a>

### 6.7 Multimodal, Generative, Retrieval, Agent, and Non-LLM Workloads

#### 6.7.1 Multimodal Foundation Models

##### 6.7.1.1 Vision Encoder, Projector, and Language Backbone

##### 6.7.1.2 CLIP/SigLIP, ViT, and Contrastive Learning

##### 6.7.1.3 LLaVA-Style Alignment and Visual Instruction Tuning

##### 6.7.1.4 Early Fusion, Late Fusion, and Native Multimodality

##### 6.7.1.5 Cross-Attention, Image Tokens, and Dynamic Resolution

##### 6.7.1.6 Audio/Video Tokenization and Cross-Modal Temporal Alignment

##### 6.7.1.7 ASR, TTS, Codec LMs, and Omni Models

#### 6.7.2 Diffusion, Flow Matching, and Visual Generation

##### 6.7.2.1 DDPM, DDIM, and Denoising Diffusion

##### 6.7.2.2 Score-Based Modeling and the Sampling Process

##### 6.7.2.3 Latent Diffusion and VAEs

##### 6.7.2.4 U-Net, Diffusion Transformer (DiT), and MMDiT

##### 6.7.2.5 Flow Matching and Rectified Flow

##### 6.7.2.6 Classifier-Free Guidance and Conditional Generation

##### 6.7.2.7 Distillation, Consistency Models, and Few-Step Generation

##### 6.7.2.8 Video Diffusion, Spatiotemporal Attention, and Long-Video Generation

#### 6.7.3 Retrieval, Tools, and Agent Algorithms

##### 6.7.3.1 Sparse Retrieval, Dense Retrieval, and Hybrid Retrieval

##### 6.7.3.2 Embedding Models, Rerankers, and RAG

##### 6.7.3.3 Retrieval-Augmented Pretraining and Knowledge Updating

##### 6.7.3.4 Tool Use, Function Calling, and the Action Space

##### 6.7.3.5 ReAct, Planning, Memory, and Context Compression

##### 6.7.3.6 Multi-Agent Coordination and Collaborative Problem Solving

##### 6.7.3.7 Tool Feedback, Environment Rewards, and Agent Training

#### 6.7.4 AI Workloads Beyond LLMs

##### 6.7.4.1 CNNs, ResNet, and Visual Recognition

##### 6.7.4.2 Recommendation, DLRM, and Embedding-Heavy Models

##### 6.7.4.3 Graph Neural Networks and Sparse Message Passing

##### 6.7.4.4 Speech, Time-Series, and Sequence Models

##### 6.7.4.5 Vision-Language-Action Models and Robot Policies

##### 6.7.4.6 Scientific ML, Neural Operators, and Structure Prediction

##### 6.7.4.7 Compute, Memory, and Communication Characteristics Across Workloads

<a id="distributed-learning-and-evolution"></a>

### 6.8 Distributed Learning and Model Evolution

#### 6.8.1 Communication-Efficient Learning and Distributed Optimization

##### 6.8.1.1 Local SGD and Periodic Parameter Averaging

##### 6.8.1.2 Gradient Quantization, Sparsification, and Error Feedback

##### 6.8.1.3 Low-Rank Gradient Compression and PowerSGD

##### 6.8.1.4 Asynchronous Training, Staleness, and Convergence

##### 6.8.1.5 Low-Bandwidth Distributed Pretraining and DiLoCo

##### 6.8.1.6 Federated Learning, Secure Aggregation, and Differential Privacy

##### 6.8.1.7 Communication Budget, Statistical Efficiency, and Wall-Clock Training Time

#### 6.8.2 Model Evolution: From Sequence Models to Foundation Models

##### 6.8.2.1 RNN, LSTM, and Seq2Seq

##### 6.8.2.2 Attention and the Original Transformer

##### 6.8.2.3 BERT, T5, and the Pretraining Paradigm

##### 6.8.2.4 GPT, GPT-2, and GPT-3

##### 6.8.2.5 InstructGPT and Instruction/Preference Alignment

##### 6.8.2.6 Dense LM → MoE → Reasoning/Multimodal Models

<a id="model-family-case-studies"></a>

### 6.9 Model-Family Case Studies

#### 6.9.1 Model Family: Llama

##### 6.9.1.1 LLaMA: An Open-Weight Foundation Model

##### 6.9.1.2 Llama 2: Base and Chat

##### 6.9.1.3 Llama 3/3.1: Training Scale and Long Context

##### 6.9.1.4 Llama 3.2/3.3: Model Branches and Capability Evolution

##### 6.9.1.5 Llama 4: Public Model Report and Architectural Evolution

##### 6.9.1.6 Comparing Attention, Data Recipes, and Deployment Across the Llama Family

#### 6.9.2 Model Family: Mistral and Mixtral

##### 6.9.2.1 Mistral 7B and Sliding-Window Attention

##### 6.9.2.2 Mixtral 8×7B and Sparse MoE

##### 6.9.2.3 Mixtral 8×22B and Model Scaling

##### 6.9.2.4 Mistral's Instruction, Code, and Multimodal Branches

##### 6.9.2.5 Routing, Caching, and Deployment Characteristics of Mistral/Mixtral

#### 6.9.3 Model Family: DeepSeek

##### 6.9.3.1 DeepSeek LLM and DeepSeekMoE

##### 6.9.3.2 DeepSeek-V2: MLA and Fine-Grained MoE

##### 6.9.3.3 DeepSeek-V3: Low-Precision Training, Load Balancing, and MTP

##### 6.9.3.4 DeepSeek-R1: Reasoning RL and Distillation

##### 6.9.3.5 DeepSeek-V3.1/V3.2: Reasoning Modes and Sparse Attention

##### 6.9.3.6 DeepSeek-V4: Public Model Report and Deployment Requirements

##### 6.9.3.7 Model–Kernel–System Co-Design Across the DeepSeek Family

#### 6.9.4 Model Family: Qwen

##### 6.9.4.1 Qwen2 and Qwen2.5

##### 6.9.4.2 The Qwen2.5-Coder, Math, VL, and Omni Branches

##### 6.9.4.3 Qwen3: Dense/MoE and Thinking/Non-Thinking

##### 6.9.4.4 Qwen3-Next and Hybrid Sequence Modeling

##### 6.9.4.5 Qwen3.5: Public Model Report and Architectural Evolution

##### 6.9.4.6 Comparing Architecture, Post-Training, and Deployment Across the Qwen Family

#### 6.9.5 Model Family: Kimi

##### 6.9.5.1 Kimi Long-Context Models and Technical Direction

##### 6.9.5.2 Kimi K1.5 and Multimodal Reasoning RL

##### 6.9.5.3 Kimi K2 and Large-Scale MoE

##### 6.9.5.4 Kimi K2 Thinking and Agentic Reasoning

##### 6.9.5.5 Kimi K2.5: Public Model Report and Multimodal/Agent Workloads

##### 6.9.5.6 Attention, Optimizers, and System Requirements Across the Kimi Family

#### 6.9.6 Model Family: GLM

##### 6.9.6.1 GLM and Autoregressive Blank Infilling Pretraining

##### 6.9.6.2 ChatGLM and GLM-4

##### 6.9.6.3 GLM-4.5/4.7 and the Evolution of Agent Capability

##### 6.9.6.4 GLM-5: Public Model Report and Complex-Task Workloads

##### 6.9.6.5 The Text, Vision, and Tool-Calling Branches of the GLM Family

#### 6.9.7 Other Model Families and Reading Public Reports

##### 6.9.7.1 Gemma, Phi, and Small Models/Data Efficiency

##### 6.9.7.2 OLMo and Open Training Recipes

##### 6.9.7.3 DBRX, MiniMax, and Other MoE Directions

##### 6.9.7.4 Stable Diffusion, FLUX, Wan, and Visual Generation

##### 6.9.7.5 Public Model/System Reports for GPT, Claude, and Gemini

##### 6.9.7.6 Evidence Boundaries Among Open-Weight, Open-Source, and Closed Models

<a id="evaluation-and-trade-offs"></a>

### 6.10 Model Evaluation and Cross-Layer Trade-offs

#### 6.10.1 Model Evaluation and Cross-Layer Trade-offs

##### 6.10.1.1 Perplexity, Accuracy, and Task Quality

##### 6.10.1.2 MMLU, GSM8K/MATH, HumanEval, and SWE-bench

##### 6.10.1.3 Long-Context, Multimodal, and Agent Benchmarks

##### 6.10.1.4 Pass@k, Success Rate, and Inference Compute Budget

##### 6.10.1.5 Data Contamination, Judge Bias, and Evaluation Reproducibility

##### 6.10.1.6 The Quality–Latency–Throughput–Memory–Energy–Cost Pareto Frontier

##### 6.10.1.7 Distinguishing Algorithmic Improvement, Kernel Speedup, and End-to-End Gain

##### 6.10.1.8 Model Architecture Change and the Adaptability Range of Deployed Hardware
