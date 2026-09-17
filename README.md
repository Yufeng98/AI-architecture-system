# AI Architecture and Systems Tutorial

*A topic map from GEMM execution to datacenter-scale AI, centered on generality, specialization, and data movement.*

**Revision:** 2026-09-16 · **Curriculum status:** planned topics

This outline connects six layers: **Microarchitecture → Kernel → Compiler → Architecture → System → Algorithms**. Hardware chapters follow the supplied survey, *Balancing Generality and Specialization: A Survey on AI Datacenter Hardware Architecture*, and the [AI Datacenter Accelerator Research Corpus][corpus]. Kernel, compiler, and runtime comparisons use the corpus’s per-chip software mappings. The broader systems and algorithms topics are carried forward from the original outline, with [Awesome-ML-SYS-Tutorial][awesome] as a systems reading index.

| Module | Primary question | Scope |
|---|---|---|
| [1. Microarchitecture](#microarchitecture) | How does GEMM execute inside an accelerator? | PE, systolic array, memory hierarchy, dataflow, roofline, and operand delivery. |
| [2. Kernel](#kernel) | How are operators implemented efficiently? | Layout, tiling, pipelines, matrix instructions, libraries, and device-specific kernel programming. |
| [3. Compiler](#compiler) | How does a model become an executable device program? | Graph/IR lowering, scheduling, memory planning, placement/routing, and executable artifacts. |
| [4. Architecture](#architecture) | How are hardware resources organized? | Accelerator taxonomy, programming models, fabrics, generations, power/cooling, and design challenges. |
| [5. System](#system) | How are workloads executed and operated? | Distributed training, serving, post-training, runtimes, heterogeneous fleets, and production infrastructure. |
| [6. Algorithms](#algorithms) | What computation does the model require? | Model structures, training objectives, generation, compression, and model-family evolution. |

<details>
<summary><strong>Source map and scope</strong></summary>

**Survey.** Yufeng Gu, Jiazhen Wang, and Reetuparna Das, September 2026, supplied 34-page draft. “Survey §…” and figure references refer to that attachment. [Companion project page][survey-project].

**Corpus.** Platform links are pinned to [commit `c49a55c6cdbb`][corpus-commit]. Per-chip records distinguish confirmed, inferred, contested, nonpublic, historical, and announced information. Those qualifications remain relevant when developing the corresponding chapters.

**Core and extensions.** The survey’s core taxonomy is GPU, NPU, Spatial Dataflow, and Compute-in-Memory. Additional vendor cases, photonic/neuromorphic comparisons, and CXL/HBF topics are labeled extensions. Untagged foundations and the detailed systems/algorithms curriculum are retained topics, rather than findings attributed to the hardware survey.

**Cross-layer boundaries.** Numerical hardware support, quantized kernels, compiler transformations, deployment policy, and quantization algorithms are separate topics. Likewise, collective semantics, communication algorithms, and physical topology are separated. A physical server/rack/pod need not coincide with a scale-up domain.

</details>

<a id="microarchitecture"></a>

## 1. Microarchitecture｜微体系结构

> Survey alignment: §3.1–§3.2 and §5.1–§5.2. The GEMM, roofline, circuit, and RTL foundations remain the introductory curriculum.

### 1.1 从 GEMM 到 AI 加速器

#### 1.1.1 标量乘加、点积、矩阵乘法与张量收缩

#### 1.1.2 GEMM、GEMV、Batched GEMM 与算子形状

#### 1.1.3 前向传播、反向传播与参数更新的计算图

#### 1.1.4 计算量、数据量、数据复用与工作集

#### 1.1.5 延迟、吞吐量、带宽、容量与能耗

#### 1.1.6 从处理单元到芯片的执行层次

### 1.2 数值表示、运算电路与精度契约

<sub>Sources: Survey §5.1, Fig. 10; foundational circuit topics retained from the original outline.</sub>

#### 1.2.1 标准浮点格式：FP64、FP32 与 FP16

#### 1.2.2 AI-Oriented Formats：BF16、TF32 与 FP8

#### 1.2.3 整数运算与量化数据类型：INT8／INT4

#### 1.2.4 Block Scaling：MXFP8／MXFP6／MXFP4 与 NVFP4

#### 1.2.5 乘法器、加法器、FMA 与累加器

#### 1.2.6 输入、乘法、累加与输出精度的分别约定

#### 1.2.7 舍入、溢出、下溢、Scale 开销与数值误差

#### 1.2.8 低精度运算、结构化稀疏与有效数据带宽

### 1.3 执行组织：SIMT、异构引擎与空间执行

<sub>Sources: Survey §3 and §3.1; [Graphcore layer mapping][layers-graphcore] and [Cerebras layer mapping][layers-cerebras].</sub>

#### 1.3.1 Scalar、SIMD、SIMT、MIMD 与 VLIW

#### 1.3.2 SM、CU、Warp、Wavefront 与线程调度

#### 1.3.3 指令流水线、依赖检查与 Scoreboarding

#### 1.3.4 Latency Hiding、ILP、TLP 与分支分歧

#### 1.3.5 Matrix、Vector、Scalar 与 Special-Function Engine 的分工

#### 1.3.6 PE-Local Execution、BSP 与数据触发执行

#### 1.3.7 动态硬件调度、静态软件调度与两者结合

#### 1.3.8 局部 Systolic Dataflow 与整芯片 Spatial Dataflow 的区别

### 1.4 PE 与矩阵计算单元

#### 1.4.1 Processing Element：MAC、寄存器与局部控制

#### 1.4.2 点积阵列、外积阵列与归约树

#### 1.4.3 累加器位宽与 Partial Sum 存储

#### 1.4.4 广播、归约与操作数分发

#### 1.4.5 PE 级流水线与有效位传播

#### 1.4.6 多精度复用、零值跳过与稀疏元数据

### 1.5 Systolic Array 的设计

#### 1.5.1 一维与二维 Systolic Array

#### 1.5.2 输入波前、数据偏移与时空调度

#### 1.5.3 阵列填充、稳态执行与排空

#### 1.5.4 Weight-Stationary 与 Output-Stationary 阵列

#### 1.5.5 矩阵分块、边界处理与 Padding

#### 1.5.6 阵列尺寸、计算利用率与供数带宽

#### 1.5.7 多阵列组织与可重构阵列

### 1.6 Dataflow 与数据复用

#### 1.6.1 Temporal Reuse 与 Spatial Reuse

#### 1.6.2 Weight-Stationary、Output-Stationary 与 Input-Stationary

#### 1.6.3 Row-Stationary 与混合 Dataflow

#### 1.6.4 Loop Nest、Loop Order 与复用距离

#### 1.6.5 局部复用、跨 PE 复用与跨 Tile 复用

#### 1.6.6 Dataflow 的计算、存储与通信权衡

### 1.7 [TPU v1 案例：从 Systolic Array 到 GEMM 执行][tpu]

<sub>Sources: [TPU v1 paper][tpu], §2 and Fig. 1; later TPU generations are compared separately in Architecture.</sub>

#### 1.7.1 Matrix Multiply Unit 与 Weight-Stationary Systolic Array

#### 1.7.2 Unified Buffer、Weight FIFO 与 Accumulator

#### 1.7.3 Host Interface、外部权重存储与指令控制

#### 1.7.4 Activation／Weight／Partial Sum 的数据路径

#### 1.7.5 矩阵乘法、激活与输出写回的衔接

#### 1.7.6 一条 GEMM 指令的完整执行路径

#### 1.7.7 TPU v1 教学模型与后续 TPU 代际的边界

### 1.8 Register 与片上局部状态

#### 1.8.1 Register File 的容量、端口与 Banking

#### 1.8.2 操作数读取带宽与寄存器冲突

#### 1.8.3 线程寄存器、向量寄存器与累加器寄存器

#### 1.8.4 寄存器压力、Occupancy 与 Spilling

#### 1.8.5 通用 Register File 与专用矩阵中间状态存储

#### 1.8.6 寄存器分配、局部状态驻留与数据路径约束

### 1.9 片上 SRAM 的管理：Cache、Scratchpad 与专用 Buffer

<sub>Sources: Survey §3.2 and Fig. 5; [NVIDIA][layers-nvidia-gpu] and [AWS Neuron][layers-aws-neuron] layer mappings.</sub>

#### 1.9.1 SRAM Array、Bank、端口与访问延迟

#### 1.9.2 Hardware-Managed Cache 与 Software-Managed Scratchpad

#### 1.9.3 GPU 的 L1／Shared Memory／L2 混合层次

#### 1.9.4 NPU Scratchpad 与 Spatial-Dataflow PE-Local SRAM

#### 1.9.5 Bank Conflict、地址映射、Swizzle 与多播

#### 1.9.6 双缓冲、Prefetch、数据驻留与容量分配

#### 1.9.7 专用 Accumulator Buffer 与 Tensor Memory

#### 1.9.8 片上存储的容量、带宽、面积与管理复杂度

### 1.10 DRAM 与外部存储接口

#### 1.10.1 DRAM Cell、Row、Bank、Bank Group 与 Channel

#### 1.10.2 Row Buffer、突发传输与访问时序

#### 1.10.3 DDR、LPDDR、GDDR 与 HBM

#### 1.10.4 HBM Stack、Channel 与 Pseudo-Channel

#### 1.10.5 Memory Controller 与请求调度

#### 1.10.6 带宽利用率、访问粒度与读写切换

#### 1.10.7 ECC、Refresh 与容量／带宽权衡

### 1.11 数据搬运与异步执行

#### 1.11.1 Load／Store Unit 与地址生成器

#### 1.11.2 DMA、Descriptor 与异步拷贝

#### 1.11.3 Gather／Scatter 与非连续数据搬运

#### 1.11.4 Memory-Level Parallelism 与未完成请求

#### 1.11.5 Producer–Consumer Pipeline 与双缓冲

#### 1.11.6 Barrier、Fence、Semaphore 与数据可见性

#### 1.11.7 计算、搬运与存储层次之间的重叠

### 1.12 片上互连与同步

#### 1.12.1 Bus、Crossbar、Ring 与 Mesh NoC

#### 1.12.2 Unicast、Multicast 与 Reduction Network

#### 1.12.3 路由、仲裁、流控与背压

#### 1.12.4 片上带宽、跳数与拥塞

#### 1.12.5 多核共享存储与一致性边界

#### 1.12.6 计算阵列与内存控制器之间的拓扑

### 1.13 [Roofline 与性能上界][roofline]

#### 1.13.1 Arithmetic Intensity 与每个存储边界的数据流量

#### 1.13.2 Compute Roof、Memory Roof 与 Ridge Point

#### 1.13.3 Peak Roofline 与 Empirical Roofline

#### 1.13.4 Register／SRAM／DRAM 分层 Roofline

#### 1.13.5 GEMM／GEMV 的 Batch Size、矩阵形状与复用

#### 1.13.6 带宽上界之外的启动延迟、同步、依赖与利用率限制

#### 1.13.7 数据搬运能耗与 Energy Roofline

### 1.14 GEMM 的分层映射与 Tiling

#### 1.14.1 M／N／K 维度与 Loop Nest 映射

#### 1.14.2 DRAM Tile、SRAM Tile 与 Register Tile

#### 1.14.3 空间展开、时间复用与阵列映射

#### 1.14.4 权重、激活与 Partial Sum 的驻留策略

#### 1.14.5 Tile Size、存储容量与端口带宽约束

#### 1.14.6 大矩阵、瘦长矩阵与小矩阵的映射差异

#### 1.14.7 从 DRAM 到 PE 再写回的完整数据流

### 1.15 矩阵引擎的数据路径与控制粒度演进

<sub>Sources: Survey §5.2, Figs. 13–14; [NVIDIA layer mapping][layers-nvidia-gpu].</sub>

#### 1.15.1 Volta：Warp-Cooperative 编程接口与 Sub-Warp 机器执行

#### 1.15.2 Turing：Warp-Level MMA、ldmatrix 与操作数布局

#### 1.15.3 Ampere：cp.async 与 Global-to-Shared 数据搬运

#### 1.15.4 Hopper：WGMMA、TMA 与 Distributed Shared Memory

#### 1.15.5 Blackwell：Tensor Memory 与 Paired-SM Tensor Execution

#### 1.15.6 操作数搬运、计算发射与结果驻留的独立演进

#### 1.15.7 硬件能力、ISA 暴露与 Kernel 编程抽象的边界

### 1.16 非 GEMM 算子与专用支持

#### 1.16.1 Softmax、LayerNorm 与 RMSNorm

#### 1.16.2 激活函数、指数、倒数与平方根

#### 1.16.3 Reduction、Scan、Sort 与 Top-k

#### 1.16.4 Embedding、Gather／Scatter 与稀疏访存

#### 1.16.5 Attention 与 MoE 的硬件需求

#### 1.16.6 计算单元均衡与 Amdahl's Law

### 1.17 硬件建模、实现与验证

#### 1.17.1 Analytical Model、Cycle Model 与 RTL Model

#### 1.17.2 Systolic Array 的 RTL 与功能验证

#### 1.17.3 存储模型、延迟模型与带宽模型

#### 1.17.4 面积、时序、功耗与能效评估

#### 1.17.5 Design Space Exploration 与约束优化

#### 1.17.6 Timeloop／Accelergy、SCALE-Sim 与 gem5

#### 1.17.7 FPGA 原型、硬件计数器与模型校准

---

<a id="kernel"></a>

## 2. Kernel｜算子实现与性能优化

> Corpus alignment: distinguish operator libraries, kernel libraries, kernel languages, and compiler-owned execution paths. Kernel exercises do not imply that all vendor interfaces are public.

### 2.1 [Kernel 编程模型与执行基础][cuda]

#### 2.1.1 CPU Thread、GPU Thread、Block 与 Grid

#### 2.1.2 Warp／Wavefront 协作与 Cooperative Groups

#### 2.1.3 CUDA／HIP 的内存空间与同步原语

#### 2.1.4 Kernel Launch、Stream、Event 与异步执行

#### 2.1.5 Device Memory、Pinned Memory 与 Unified Memory

#### 2.1.6 竞态、死锁、越界与内存一致性

### 2.2 Tensor Layout 与内存访问

#### 2.2.1 Shape、Stride、View 与 Contiguous Tensor

#### 2.2.2 Row-Major、Column-Major 与 Blocked Layout

#### 2.2.3 Coalescing、Vectorized Load／Store 与对齐

#### 2.2.4 Shared Memory Bank Conflict 与 Swizzle

#### 2.2.5 Transpose、Packing 与 Layout Conversion

#### 2.2.6 Ragged Tensor、Padding 与变长批次

### 2.3 从朴素 GEMM 到分块 GEMM

#### 2.3.1 Naive GEMM 与循环重排

#### 2.3.2 CPU Cache Blocking 与 SIMD Microkernel

#### 2.3.3 GPU Global-Memory GEMM

#### 2.3.4 Shared-Memory Tiling 与 Register Blocking

#### 2.3.5 计算复用、访存合并与写回优化

#### 2.3.6 正确性校验与逐步性能分析

### 2.4 [Tensor Core GEMM：矩阵指令、数据布局与协作范围][cutlass]

<sub>Sources: Survey §5.2 for hardware mechanisms; [NVIDIA layer mapping][layers-nvidia-gpu] for exposed interfaces.</sub>

#### 2.4.1 WMMA／MMA、矩阵 Fragment 与 Operand Layout

#### 2.4.2 Instruction Tile、Warp Tile、CTA Tile 与 Cluster Tile

#### 2.4.3 ldmatrix、cp.async 与操作数加载路径

#### 2.4.4 Hopper WGMMA／TMA 的 Kernel 使用与同步

#### 2.4.5 Blackwell tcgen05／TMEM 与 Paired-SM Kernel

#### 2.4.6 Accumulator Layout、数值精度与 Epilogue

#### 2.4.7 硬件代际选择、Fallback 与算子正确性

### 2.5 高级 GEMM 流水线与调度

#### 2.5.1 Double／Multi-Buffering 与软件流水线

#### 2.5.2 Asynchronous Copy 与 Producer–Consumer 同步

#### 2.5.3 Warp Specialization 与角色分工

#### 2.5.4 Persistent Kernel 与 Persistent GEMM

#### 2.5.5 Split-K、Stream-K 与工作量分配

#### 2.5.6 CTA Swizzle、Cluster 与局部性

#### 2.5.7 Occupancy、寄存器压力与 Wave Quantization

### 2.6 不同形状与场景的矩阵计算

#### 2.6.1 GEMV 与小 Batch Decode

#### 2.6.2 Small／Tall-Skinny GEMM

#### 2.6.3 Batched GEMM 与 Grouped GEMM

#### 2.6.4 变长矩阵、动态形状与边界 Tile

#### 2.6.5 低秩矩阵、LoRA 与多适配器 GEMM

#### 2.6.6 权重常驻、权重 Packing 与形状专用化

### 2.7 Elementwise、Reduction 与 Scan

#### 2.7.1 Vector Add 与逐元素融合

#### 2.7.2 Softmax 与 Online Softmax

#### 2.7.3 LayerNorm、RMSNorm 与归约布局

#### 2.7.4 Prefix Sum、Scan 与 Histogram

#### 2.7.5 Top-k、Argmax、Sort 与 Sampling

#### 2.7.6 Welford 算法与稳定归约

### 2.8 Fusion 与访存消除

#### 2.8.1 Bias／Activation／Residual Epilogue Fusion

#### 2.8.2 QKV Projection 与 RoPE Fusion

#### 2.8.3 Norm、Residual 与量化融合

#### 2.8.4 Fused MLP、SwiGLU 与 GeGLU

#### 2.8.5 Fused Loss 与 Fused Optimizer

#### 2.8.6 Launch Overhead、访存节省与 Fusion 边界

### 2.9 Exact Attention Kernel

#### 2.9.1 标准 Attention 的算子分解与访存

#### 2.9.2 [FlashAttention：IO-Aware Tiling 与重计算][flashattention]

#### 2.9.3 FlashAttention-2／3 与并行、流水线优化

#### 2.9.4 Causal Mask、Sliding Window 与变长 Attention

#### 2.9.5 Forward／Backward Attention Kernel

#### 2.9.6 FlexAttention 与可编程 Attention

### 2.10 Decode Attention 与 KV Cache Kernel

#### 2.10.1 Paged KV Layout 与 Block Table

#### 2.10.2 PagedAttention 的读取与归约

#### 2.10.3 Split-KV 与长序列 Decode

#### 2.10.4 MHA／MQA／GQA 的 Kernel 映射

#### 2.10.5 MLA 的投影吸收与低秩缓存访问

#### 2.10.6 KV Cache Append、Gather、Copy 与压缩

### 2.11 Sparse 与 Irregular Kernel

#### 2.11.1 COO、CSR、CSC、BSR 与压缩布局

#### 2.11.2 SpMV、SpMM 与 SDDMM

#### 2.11.3 结构化稀疏、N:M 稀疏与 Sparse Tensor Core

#### 2.11.4 Block-Sparse Attention 与动态索引

#### 2.11.5 稀疏选择、Gather／Scatter 与负载不均

#### 2.11.6 稀疏元数据开销与加速收益边界

### 2.12 MoE Kernel

#### 2.12.1 Router、Top-k Gating 与 Token Assignment

#### 2.12.2 Token Permutation 与 Unpermutation

#### 2.12.3 Grouped GEMM 与 Expert Padding

#### 2.12.4 Dropless MoE 与 Block-Sparse GEMM

#### 2.12.5 Dispatch／Combine 与通信融合

#### 2.12.6 共享专家、路由专家与计算重叠

### 2.13 低精度与量化 Kernel

#### 2.13.1 Quantize／Dequantize 与 Scale 计算

#### 2.13.2 Weight-Only GEMM 与 Weight–Activation GEMM

#### 2.13.3 INT8／INT4 与 FP8／FP4 GEMM

#### 2.13.4 Per-Tensor、Per-Channel 与 Block Scaling

#### 2.13.5 Microscaling 格式与 Scale Layout

#### 2.13.6 在线量化、反量化融合与数值校验

### 2.14 Communication Kernel：设备发起、数据搬运与计算融合

<sub>Sources: Survey §3.2 and §4.4; [NVIDIA][layers-nvidia-gpu] and [AMD][layers-amd-gpu] layer mappings.</sub>

#### 2.14.1 GPU P2P Copy、远端访存与内存注册

#### 2.14.2 Reduction、AllReduce 与 ReduceScatter Kernel

#### 2.14.3 NVSHMEM 的对称地址空间与显式同步

#### 2.14.4 Device-Initiated Communication 与 Host-Proxy 路径

#### 2.14.5 SM-Driven 与 DMA／Copy-Engine 数据搬运

#### 2.14.6 通信 Tile、GEMM–Collective Fusion 与流水化

#### 2.14.7 通信 Kernel 的 SM 占用、HBM 竞争与重叠边界

### 2.15 非语言模型的关键 Kernel

#### 2.15.1 Conv2D、Depthwise Conv 与 Implicit GEMM

#### 2.15.2 Embedding Lookup 与 Embedding Bag

#### 2.15.3 Graph Aggregation 与稀疏邻接计算

#### 2.15.4 FFT、Convolution 与 State-Space Scan

#### 2.15.5 图像／视频预处理与 Resize

#### 2.15.6 Audio Codec、Vocoder 与流式音频计算

### 2.16 GPU Kernel Languages、DSL 与算子／Kernel 库

<sub>Sources: [NVIDIA layer mapping][layers-nvidia-gpu] and [AMD layer mapping][layers-amd-gpu]; other original library topics retained.</sub>

#### 2.16.1 CUDA／HIP：线程、Warp／Wavefront 与设备专用接口

#### 2.16.2 [Triton：Tile 编程、自动调优与后端差异][triton]

#### 2.16.3 [CuTe／CUTLASS 与 CuTe DSL：显式布局和协作调度][cutlass]

#### 2.16.4 CUDA Tile IR／cuTile：Tile 级抽象与底层控制的边界

#### 2.16.5 cuBLAS／cuBLASLt／cuDNN 与 CUB／Thrust／libcu++

#### 2.16.6 rocBLAS／hipBLASLt／MIOpen 与 CK／CK-Tile／AITER／Tensile

#### 2.16.7 TileLang、FlashInfer、Transformer Engine 与 DeepGEMM

#### 2.16.8 CPU 对照：SIMD／AMX Microkernel 与 oneDNN

### 2.17 NPU Kernel Programming：Tile、Scratchpad 与异构引擎

<sub>Sources: [TPU][layers-google-tpu], [Neuron][layers-aws-neuron], [Ascend][layers-huawei-ascend] and [Cambricon][chip-cambricon] corpus records.</sub>

#### 2.17.1 TPU Pallas：BlockSpec、Memory Space 与 MXU／Vector 协作

#### 2.17.2 Pallas／Mosaic 与图编译的衔接

#### 2.17.3 AWS NKI：Language API、ISA Intrinsic 与 SBUF／PSUM

#### 2.17.4 Neuron Tensor／Vector／Scalar／GPSIMD 的算子分工

#### 2.17.5 Ascend C：数据搬运、Cube／Vector 与流水线同步

#### 2.17.6 Ascend TBE／TIK、CATLASS 与设备专用 DSL

#### 2.17.7 Cambricon BANG C 与显式局部存储编程

#### 2.17.8 GEMM／Attention 在不同 NPU 上的布局和容量约束

### 2.18 Spatial Dataflow Kernel：PE、Codelet 与数据触发任务

<sub>Sources: [Tenstorrent][layers-tenstorrent], [Graphcore][layers-graphcore], [Cerebras][layers-cerebras], [Groq][layers-groq] and [SambaNova][layers-sambanova] layer mappings. Historical interfaces retain their generation scope.</sub>

#### 2.18.1 Tenstorrent：TT-NN、TT-Metalium 与 TT-LLK 的边界

#### 2.18.2 Reader／Compute／Writer 与 Unpack／Math／Pack

#### 2.18.3 PE-Local Buffer、NoC 数据搬运与 Producer–Consumer 同步

#### 2.18.4 Graphcore：Vertex／Codelet 与 BSP Compute–Exchange–Sync

#### 2.18.5 Cerebras CSL：Wavelet、Color、Task 与 Data Structure Descriptor

#### 2.18.6 局部 Kernel 与全图 Placement／Routing 的协作

#### 2.18.7 Groq 与 SN40L-Era SambaFlow：编译器拥有的计算路径

#### 2.18.8 公开 Kernel 接口、编译器内部实现与不可见软件层的区别

### 2.19 Profiling、Benchmark 与正确性

<sub>Sources: Profiling tools in the corpus layer mappings; benchmark fundamentals retained from the original outline.</sub>

#### 2.19.1 Microbenchmark、Warm-up、异步计时与计时边界

#### 2.19.2 Nsight Compute／Systems 与 ROCm Profiling

#### 2.19.3 设备专用分析工具：Neuron Explorer、PopVision 与编译报告

#### 2.19.4 Roofline、Stall Reason、访存效率与通信占用

#### 2.19.5 Dense Peak、Sparse Peak、Achieved FLOPS 与有效模型工作量

#### 2.19.6 数值容差、梯度检查、低精度误差与差分测试

#### 2.19.7 Race Detection、回归测试与跨设备性能可移植性

### 2.20 从算子到模型：注册、移植与后端集成

<sub>Sources: Corpus software-layer mapping; integration exercises extend the source descriptions.</sub>

#### 2.20.1 PyTorch Custom Operator、Autograd 与 Dispatcher

#### 2.20.2 算子库调用、编译器融合与自定义 Kernel 的选择

#### 2.20.3 形状、布局、精度、稀疏格式与设备能力匹配

#### 2.20.4 JIT Cache、预编译 Kernel 与架构专用产物

#### 2.20.5 Graph Capture、异步执行与自定义算子兼容性

#### 2.20.6 API 兼容性、数值等价性与性能可移植性的区别

#### 2.20.7 GEMM → Attention → Transformer Block 的端到端集成

---

<a id="compiler"></a>

## 3. Compiler｜编译器与程序映射

> Corpus alignment: use platform-specific compilation and loading contracts instead of assuming every accelerator exposes a CUDA-like stack.

### 3.1 AI 编译栈：图、算子库、Kernel 与设备执行

<sub>Sources: [Corpus layer schema][corpus]; [TPU][layers-google-tpu], [Neuron][layers-aws-neuron] and [Groq][layers-groq] provide contrasting stack boundaries.</sub>

#### 3.1.1 Framework → Graph IR → Tensor／Loop IR → Device Program

#### 3.1.2 Graph Compiler、Kernel Compiler、Op Library 与 Runtime

#### 3.1.3 Eager、JIT、AOT 与静态执行计划

#### 3.1.4 编译期、加载期、运行期与 Firmware 的责任边界

#### 3.1.5 独立软件层、编译器内合并的软件层与未公开的软件层

#### 3.1.6 兼容性、可编程性、可移植性与专用化

### 3.2 Frontend、Tracing 与图捕获

#### 3.2.1 Python Program、Tensor Graph 与控制流

#### 3.2.2 Symbolic Tracing、Bytecode Capture 与 Export

#### 3.2.3 Graph Break、Guard 与 Recompilation

#### 3.2.4 动态形状、动态分支与图专用化

#### 3.2.5 Custom Operator、Side Effect 与外部调用

### 3.3 Intermediate Representation

#### 3.3.1 Dataflow Graph、SSA 与 Control-Flow Graph

#### 3.3.2 Tensor IR、Loop IR 与 Buffer IR

#### 3.3.3 类型、Shape、Layout 与内存空间

#### 3.3.4 [MLIR Dialect、Operation、Region 与 Pass][mlir]

#### 3.3.5 StableHLO、Linalg、Affine、SCF 与 GPU Dialect

### 3.4 自动微分与训练图编译

#### 3.4.1 Reverse-Mode 与 Forward-Mode AD

#### 3.4.2 Backward Graph、VJP 与 JVP

#### 3.4.3 AOT Autograd 与联合图优化

#### 3.4.4 Activation Liveness 与 Rematerialization

#### 3.4.5 梯度累加、梯度通信与优化器图

### 3.5 Graph-Level Optimization

#### 3.5.1 Constant Folding、CSE 与 Dead Code Elimination

#### 3.5.2 算子融合、算子分解与 Pattern Rewriting

#### 3.5.3 Algebraic Simplification 与算子重排

#### 3.5.4 Attention、MLP 与 Transformer 专用融合

#### 3.5.5 数值等价性与优化合法性

### 3.6 Loop Transformation 与 Tensorization

#### 3.6.1 Tiling、Interchange、Unrolling 与 Vectorization

#### 3.6.2 Loop Fusion、Fission 与软件流水线

#### 3.6.3 Affine Analysis 与 Polyhedral Optimization

#### 3.6.4 Parallelization 与 Reduction Transformation

#### 3.6.5 Tensorization 与矩阵指令匹配

#### 3.6.6 循环调度到 Systolic／SIMT 硬件的映射

### 3.7 Layout 与数据移动优化

#### 3.7.1 Layout Inference 与 Layout Propagation

#### 3.7.2 Blocked Layout、Swizzle 与线程映射

#### 3.7.3 Layout Conversion 插入与消除

#### 3.7.4 Cache／Scratchpad Placement 与数据复用

#### 3.7.5 异步 DMA、Prefetch 与搬运调度

### 3.8 Memory Planning 与 Buffer Management

#### 3.8.1 Bufferization、Aliasing 与 In-Place Execution

#### 3.8.2 Liveness Analysis 与 Buffer Reuse

#### 3.8.3 Static Memory Plan 与动态内存分配

#### 3.8.4 Register Allocation、Spilling 与 Scratchpad Allocation

#### 3.8.5 重计算、卸载与存储容量约束

### 3.9 Kernel Code Generation 与 ISA 边界

<sub>Sources: [NVIDIA][layers-nvidia-gpu], [AMD][layers-amd-gpu], [TPU][layers-google-tpu] and [Neuron][layers-aws-neuron] layer mappings.</sub>

#### 3.9.1 指令选择、指令调度与寄存器分配

#### 3.9.2 LLVM／NVVM／AMDGPU 后端

#### 3.9.3 PTX、SASS 与 AMD 设备指令集

#### 3.9.4 Thread-Level、Tile-Level 与 Tensor Intrinsic 降低路径

#### 3.9.5 Triton IR → GPU IR → Device Code

#### 3.9.6 Vector、Matrix 与 Scalar 路径协作

#### 3.9.7 公开 ISA、公开 Intrinsic 与未公开机器指令的区别

#### 3.9.8 跨架构 Code Generation、Feature Detection 与 Fallback

### 3.10 Dynamic Shape 与运行时专用化

#### 3.10.1 Symbolic Shape、Shape Constraint 与 Shape Polymorphism

#### 3.10.2 Shape Bucketing、Padding 与 Multi-Versioning

#### 3.10.3 Autotuning Cache 与编译缓存

#### 3.10.4 动态稀疏、动态路由与 Ragged Workload

#### 3.10.5 编译延迟、冷启动与稳态性能

### 3.11 Cost Model、Autotuning 与自动调度

#### 3.11.1 搜索空间、合法性约束与调度表示

#### 3.11.2 Analytical／Learned Cost Model

#### 3.11.3 Tile、Layout、Fusion 与并行度搜索

#### 3.11.4 测量反馈、搜索预算与泛化

#### 3.11.5 硬件感知优化与性能可移植性

### 3.12 Distributed Compilation

#### 3.12.1 SPMD、Sharding Annotation 与 Device Mesh

#### 3.12.2 自动并行策略与图分区

#### 3.12.3 Collective Insertion 与 Resharding

#### 3.12.4 通信与计算重叠的编译期调度

#### 3.12.5 流水线切分、跨设备布局与内存约束

#### 3.12.6 [OpenXLA、GSPMD 与 Shardy][openxla]

### 3.13 PyTorch 编译栈

#### 3.13.1 TorchDynamo 与 FX Graph

#### 3.13.2 AOTAutograd 与训练图捕获

#### 3.13.3 TorchInductor、Triton 与 CPU Codegen

#### 3.13.4 Export、Dynamic Shape 与 Custom Backend

#### 3.13.5 编译模式、Graph Break 与性能诊断

### 3.14 JAX、XLA 与 TPU 编译

<sub>Sources: [Google TPU layer mapping][layers-google-tpu].</sub>

#### 3.14.1 [JAX Tracing、jaxpr 与 XLA][jax]

#### 3.14.2 StableHLO、HLO 与多阶段优化

#### 3.14.3 Sharding、GSPMD／Shardy 与 Collective Insertion

#### 3.14.4 Layout Assignment、Buffer Assignment 与内存调度

#### 3.14.5 Pallas／Mosaic 与高层图编译的协作

#### 3.14.6 MXU／Vector／DMA 协作与静态调度

#### 3.14.7 libtpu 交付边界与 TPU ISA 的公开信息范围

### 3.15 MLIR、TVM 与部署编译栈

#### 3.15.1 [MLIR 的多层 IR 与可扩展 Dialect][mlir]

#### 3.15.2 [TVM TensorIR、Relax 与自动调优][tvm]

#### 3.15.3 IREE 与异构 Runtime

#### 3.15.4 ONNX、ONNX Runtime 与图交换

#### 3.15.5 TensorRT 与部署图优化

#### 3.15.6 量化图转换、算子覆盖与后端兼容

### 3.16 NPU 编译：Scratchpad、引擎分工与设备可执行文件

<sub>Sources: [AWS Neuron][layers-aws-neuron] and [Huawei Ascend][layers-huawei-ascend] layer mappings.</sub>

#### 3.16.1 AWS neuronx-cc：图编译与设备专用 Code Generation

#### 3.16.2 NKI Compiler 与图编译器的不同入口

#### 3.16.3 SBUF／PSUM 分配、Prefetch 与跨引擎调度

#### 3.16.4 NEFF 产物、Runtime 加载与设备执行

#### 3.16.5 Ascend：MindIR／ATC／Graph Engine 与 CANN

#### 3.16.6 算子覆盖、静态 Shape、后端约束与模型移植

#### 3.16.7 公开接口、内部 IR 与推断信息的分离

### 3.17 Spatial Dataflow 编译：图映射、存储放置与通信调度

<sub>Sources: [Tenstorrent][layers-tenstorrent], [Graphcore][layers-graphcore], [Cerebras][layers-cerebras], [SambaNova][layers-sambanova] and [Groq][layers-groq].</sub>

#### 3.17.1 Tenstorrent：TT-Forge／TT-XLA → TT-MLIR → TT-NN／TT-Metalium

#### 3.17.2 Graphcore Poplar：Tile Assignment、Codelet 与 BSP Exchange

#### 3.17.3 Cerebras：图编译、CSL 任务与 Layer-Pipelined／Weight-Streaming 执行

#### 3.17.4 SambaFlow（SN40L-Era）：PCU／PMU Placement、Routing 与 Meta-Pipeline

#### 3.17.5 Groq：Functional-Slice Placement、Cycle Scheduling 与链路调度

#### 3.17.6 时间复用、空间展开、SRAM 分配与路由资源约束

#### 3.17.7 编译期冲突消除与运行期调度的取舍

### 3.18 CIM／PNM 编译与 Memory-Centric Offload

<sub>Sources: Survey §3.2 and §6.2; [d-Matrix layer mapping][layers-d-matrix]. CXL offload remains a curriculum extension.</sub>

#### 3.18.1 支持算子的识别、切分与 Host／Accelerator 协作

#### 3.18.2 权重驻留、Bank／Array Layout 与数据重排

#### 3.18.3 容量限制下的算子映射、分块与跨层搬运

#### 3.18.4 计算支持范围、精度约束与回退路径

#### 3.18.5 Corsair Aviator 案例与 Raptor 未公开编译细节的边界

#### 3.18.6 研究扩展：CXL-Attached PNM 与异构 Offload 编译

### 3.19 Executable Artifact、加载契约与软件栈可见性

<sub>Sources: Corpus layer mappings for NVIDIA, AMD, Neuron, Groq and SambaNova; compatibility topics retained from the original outline.</sub>

#### 3.19.1 GPU：PTX／SASS、Fat Binary 与设备 Code Object

#### 3.19.2 Neuron NEFF、Groq IOP 与 SambaFlow PEF

#### 3.19.3 Kernel Binary、权重、路由与静态调度元数据

#### 3.19.4 Runtime 装载、Driver 提交与 Firmware 执行

#### 3.19.5 ABI、版本匹配、编译缓存与可复现构建

#### 3.19.6 Not Applicable、Not Public、Inferred 与 Confirmed 的区别

### 3.20 编译器验证、调试与案例

#### 3.20.1 IR Dump、Pass Tracing 与最小复现

#### 3.20.2 Differential Testing 与数值一致性

#### 3.20.3 Fusion、Layout 与 Memory Plan 的消融

#### 3.20.4 编译性能回归与跨版本可复现性

#### 3.20.5 从 Python GEMM 到矩阵指令的完整编译路径

#### 3.20.6 从 Transformer Graph 到多设备执行计划

---

<a id="architecture"></a>

## 4. Architecture｜加速器架构与 AI 数据中心

> Primary structure: Survey §2–§6. Core platform classifications follow the survey; additional cases are labeled as corpus extensions or contrasts.

### 4.1 范围、术语与架构比较的证据口径

<sub>Sources: Survey §3 and Tables 2–4; [corpus conventions and per-chip evidence][corpus].</sub>

#### 4.1.1 Generality、Specialization、Programmability 与 Deployment Flexibility

#### 4.1.2 主导执行／数据移动模型与重叠硬件机制

#### 4.1.3 Chip、Die、Package、Card、Server、Rack 与 Pod

#### 4.1.4 Vendor-Reported Peak、实测性能与模型级结果

#### 4.1.5 Dense／Sparse Throughput、FMA 计数与数值格式

#### 4.1.6 单向／双向带宽、Per-Accelerator／Per-System 与参考平台

#### 4.1.7 Confirmed、Inferred、Contested、Not Public 与 Not Reported

#### 4.1.8 已披露设计、计划／预告设计与历史平台的分别呈现

### 4.2 AI 数据中心的背景与四类资源压力

<sub>Sources: Survey §2 and Figs. 1–3.</sub>

#### 4.2.1 从单芯片 AI 计算到工业规模数据中心

#### 4.2.2 模型参数、上下文、并发度与工作负载变化

#### 4.2.3 Compute Throughput、Memory Capacity 与 Memory Bandwidth

#### 4.2.4 Interconnect Bandwidth、Latency 与同步开销

#### 4.2.5 Power、Cooling 与可实际运行的计算容量

#### 4.2.6 专用矩阵运算、局部复用与多设备聚合

#### 4.2.7 跨代比较：模型规模与硬件资源的不均衡演进

### 4.3 核心分类：四类 AI 加速器架构

<sub>Sources: Survey §3, Table 1 and Fig. 4.</sub>

#### 4.3.1 GPU：SIMT 架构与专用 Tensor／Matrix Unit

#### 4.3.2 NPU：围绕共享 Scratchpad 的 Matrix／Vector／Scalar Engine

#### 4.3.3 Spatial Dataflow：映射到分布式计算、存储与通信资源的计算图

#### 4.3.4 Compute-in-Memory：存储阵列内部或近旁的算术单元

#### 4.3.5 Spatial Dataflow 的三种子类：PE Array、Reconfigurable、Functional-Slice

#### 4.3.6 Systolic Execution、Memory Technology 与架构类别的非一一对应

#### 4.3.7 主类别、代际差异与混合实现

### 4.4 GPU：通用并行执行与专用矩阵计算

<sub>Sources: Survey §3 and §3.1; additional GPU records are corpus extensions, not additional survey case studies.</sub>

#### 4.4.1 [NVIDIA GPU：SM、Tensor Core 与 CUDA Programming Model][chip-nvidia-gpu]

#### 4.4.2 [AMD GPU：CU、Matrix Core 与 ROCm／HIP Programming Model][chip-amd-gpu]

#### 4.4.3 通用控制流、非矩阵算子与矩阵吞吐之间的资源配比

#### 4.4.4 SIMT、Cache／Shared Memory 与编程责任边界

#### 4.4.5 Corpus 扩展：[Biren][chip-biren]、[Hygon DCU][chip-hygon-dcu]、[Muxi][chip-muxi] 与 [Moore Threads][chip-mthreads]

#### 4.4.6 Corpus 扩展：[Tianshu Zhixin][chip-tianshu-zhixin] 与 [Xiwang][chip-xiwang]

### 4.5 NPU：异构计算引擎与共享局部存储

<sub>Sources: Survey §3 and Table 2; the final two entries broaden the case pool using the corpus.</sub>

#### 4.5.1 [Google TPU：MXU、Vector／Scalar 与代际专用单元][chip-google-tpu]

#### 4.5.2 [AWS Trainium／Inferentia：NeuronCore 与显式 Scratchpad][chip-aws-neuron]

#### 4.5.3 [Huawei Ascend：Da Vinci、Cube／Vector／Scalar 与 CANN][chip-huawei-ascend]

#### 4.5.4 [Intel Gaudi：Matrix Engine、TPC 与网络集成][chip-intel-gaudi]

#### 4.5.5 [Microsoft Maia：Matrix／Vector 与云端部署][chip-microsoft-maia]

#### 4.5.6 [Qualcomm Cloud AI：矩阵、向量与标量引擎][chip-qualcomm]

#### 4.5.7 [Cambricon MLU：多核 Neural Processor 与 BANG C][chip-cambricon]

#### 4.5.8 Corpus 扩展：[Alibaba T-Head][chip-alibaba-t-head]、[Kunlunxin][chip-kunlunxin] 与 [Furiosa][chip-furiosa]

#### 4.5.9 Corpus 扩展：[Sophgo][chip-sophgo]、[Vastai][chip-vastaitech]、[Tecorigin][chip-tecorigin] 与 [Stream Computing][chip-stream-computing]

### 4.6 Spatial Dataflow I：PE Array 与分布式局部存储

<sub>Sources: Survey §3, Fig. 4(c) and Table 2; additional PE-array and manycore cases follow the corpus labels.</sub>

#### 4.6.1 [Tenstorrent：Tensix、RISC-V 控制与 NoC 数据搬运][chip-tenstorrent]

#### 4.6.2 [Meta MTIA：PE Grid 与模型—芯片协同][chip-meta-mtia]

#### 4.6.3 [Graphcore IPU：MIMD Tile、Local SRAM 与 BSP][chip-graphcore]

#### 4.6.4 [Tesla Dojo：Tile Processor 与分层通信][chip-tesla-dojo]

#### 4.6.5 [Cerebras：Wafer-Scale PE Mesh 与数据触发执行][chip-cerebras]

#### 4.6.6 片上 PE Mesh、封装互连与跨加速器网络的边界

#### 4.6.7 Corpus 扩展：[IBM Spyre][chip-ibm-spyre] 与 [Enflame][chip-enflame]

#### 4.6.8 Corpus 扩展：[MN-Core][chip-preferred-networks-mn-core]、[Esperanto][chip-esperanto] 与 [PEZY][chip-pezy]

### 4.7 Spatial Dataflow II：Reconfigurable Architecture

<sub>Sources: Survey §3 and §6.5; [SambaNova layer mapping][layers-sambanova]. SN40L-era software is a historical case, not a current SDK setup guide.</sub>

#### 4.7.1 [SambaNova SN40L：可配置 Compute／Memory Tile 与三层存储][chip-sambanova]

#### 4.7.2 Graph Placement、Memory Placement 与静态 Routing

#### 4.7.3 Section、Meta-Pipeline 与有限资源下的重配置

#### 4.7.4 支持算子范围、图映射能力与模型演进

#### 4.7.5 Corpus 扩展：[Rebellions][chip-rebellions-atom] 与 [Tsingmicro][chip-tsingmicro]

#### 4.7.6 Corpus 扩展：[NextSilicon Maverick][chip-nextsilicon-maverick] 与运行时重配置

### 4.8 Spatial Dataflow III：Functional-Slice Streaming

<sub>Sources: Survey §3 and §3.2; [Groq layer mapping][layers-groq] and the corpus Etched dossier. Topology remains generation-specific.</sub>

#### 4.8.1 [Groq TSP／LPU：Matrix、Vector、SRAM 与 Switch Slice][chip-groq]

#### 4.8.2 编译器确定的操作放置、SRAM 地址与时间调度

#### 4.8.3 跨芯片数据路径与 Software-Scheduled Networking

#### 4.8.4 静态执行的预测性、容量限制与模型适配

#### 4.8.5 早期 Groq 系统与后续平台的代际／拓扑边界

#### 4.8.6 Corpus 扩展：[Etched Sohu][chip-etched-sohu] 的模型结构专用化与公开证据范围

### 4.9 Compute-in-Memory：SRAM、DRAM 与计算位置

<sub>Sources: Survey §3, §3.2 and §6.2; corpus extends the case pool beyond the three surveyed CIM families.</sub>

#### 4.9.1 [d-Matrix Corsair：Digital In-SRAM MAC 与容量存储][chip-d-matrix]

#### 4.9.2 [SK hynix AiM：GDDR6 Bank-Adjacent MAC][chip-sk-hynix-aim]

#### 4.9.3 [Samsung PIM：HBM 与存储侧 SIMD 计算][chip-samsung-aquabolt-pim]

#### 4.9.4 SRAM 容量、DRAM 内部带宽与计算吞吐的不同限制

#### 4.9.5 GEMV／小 Batch 与高复用 GEMM 的映射差异

#### 4.9.6 Corpus 扩展：[Mythic][chip-mythic] 的 Analog Flash Compute

#### 4.9.7 历史／对照案例：[Untether AI][chip-untether-ai] 与 [Rain AI][chip-rain-ai]

#### 4.9.8 CIM、PNM 与 3D Logic–Memory Integration 的术语边界

### 4.10 Compute Engine：跨类别的组织与取舍

<sub>Sources: Survey §3.1; platform records provide implementation-specific examples.</sub>

#### 4.10.1 Scalar／Vector／SIMD／SIMT／Matrix Engine 的配比

#### 4.10.2 Systolic Array、Tensor Core、Vector MAC 与可配置计算单元

#### 4.10.3 矩阵吞吐、非 GEMM 算子与控制流覆盖

#### 4.10.4 Embedding、Gather／Scatter 与专用加速单元

#### 4.10.5 静态调度、动态调度与数据触发执行

#### 4.10.6 Arithmetic Intensity、计算利用率与面积／能耗效率

#### 4.10.7 编程灵活性、编译负担与专用化程度

### 4.11 Memory Hierarchy：从硬件 Cache 到软件 Scratchpad

<sub>Sources: Survey §3.2 and Fig. 5; memory organization is a comparison axis, not a replacement for the four-category taxonomy.</sub>

#### 4.11.1 Fully Hardware-Managed、Hybrid 与 Fully Software-Managed Memory

#### 4.11.2 CPU Cache 对照与 Gaudi 的缓存管理案例

#### 4.11.3 GPU：L1／Shared Memory／L2 与专用 Tensor Memory

#### 4.11.4 NPU：共享 Scratchpad、Accumulator Buffer 与显式 Prefetch

#### 4.11.5 PE Array：分布式 Local SRAM 与存储放置

#### 4.11.6 HBM、GDDR、LPDDR、DDR 与 SRAM-Centric 组织

#### 4.11.7 确定性、QoS、编译复杂度与动态访问模式

### 4.12 Programming Model：从软件层次理解硬件约束

<sub>Sources: [Corpus software-layer schema][corpus] and the per-chip layer mappings; survey §3 supplies the architectural context.</sub>

#### 4.12.1 Framework Integration、Compiler／IR 与 Operator Library

#### 4.12.2 Kernel Library、Runtime、Driver／Firmware 与 Assembler／ISA

#### 4.12.3 Communication Library 与编译器内置通信

#### 4.12.4 Thread／Warp、Tensor Tile、PE／Vertex 与静态流式编程

#### 4.12.5 Kernel-Launched、Graph-Executed 与 Data-Triggered 模型

#### 4.12.6 寄存器、Scratchpad、分布式 SRAM 与显式数据移动

#### 4.12.7 独立库、编译器合并层、未公开接口与历史接口

#### 4.12.8 硬件资源 → 软件责任 → 编程模型的因果对应

### 4.13 Packaging、Chiplet 与 Logic–Memory Integration

<sub>Sources: Survey §5.3, §5.4 and §6.2; [d-Matrix/Raptor corpus record][chip-d-matrix]. UCIe is retained as an extension.</sub>

#### 4.13.1 Monolithic Die、Multi-Chip Module 与 Compute／I/O Die

#### 4.13.2 HBM Stack、Interposer 与 2.5D Packaging

#### 4.13.3 3D Stacking、Logic–DRAM Bonding 与局部数据路径

#### 4.13.4 Memory Density、Stack Count、Interface Width 与 Signaling Rate

#### 4.13.5 Chiplet 互连、板卡、Baseboard 与计算托盘

#### 4.13.6 SXM／OAM、封装密度与供电散热约束

#### 4.13.7 Raptor Early-Silicon 案例与热／可靠性约束

#### 4.13.8 扩展阅读：UCIe 与通用 Die-to-Die 接口

### 4.14 Host、Remote Memory 与 Memory-Centric 扩展

<sub>Sources: Survey §3.2 and §6.2 cover fabric-level memory and PIM/PNM; CXL, storage offload and HBF are retained curriculum extensions, not survey coverage.</sub>

#### 4.14.1 Host–Accelerator PCIe、DMA 与 CPU／Accelerator NUMA

#### 4.14.2 共享地址空间、内存一致性与显式通信的区别

#### 4.14.3 NVSHMEM：对称内存、远端访问与同步

#### 4.14.4 Groq：编译器管理的分布式 SRAM 地址空间

#### 4.14.5 扩展：CXL.io／CXL.cache／CXL.mem 与 Memory Pooling

#### 4.14.6 扩展：CPU／GPU／NPU／PIM 协作与 CXL-Attached PNM

#### 4.14.7 扩展：Storage Offload、Flash／HBF 与容量层次

### 4.15 物理部署层次与 Scale-Up／Scale-Out 域

<sub>Sources: Survey Introduction, Fig. 1, §4 and §6.3; [NVIDIA][layers-nvidia-gpu], [AMD][layers-amd-gpu], [TPU][layers-google-tpu] and [Neuron][layers-aws-neuron] fabric mappings. Physical and communication-domain boundaries are distinct.</sub>

#### 4.15.1 Accelerator → Server → Rack → Pod → Datacenter

#### 4.15.2 Compute Tray、Switch Tray、Host 与 NIC 的组织

#### 4.15.3 Scale-Up Fabric 跨 Server／Rack／Pod 的范围

#### 4.15.4 Scale-Out Network 与 Scale-Up Domain 的连接

#### 4.15.5 Scale-Up Interfaces：NVLink／NVSwitch、Infinity Fabric 与 UALink／UALoE

#### 4.15.6 NPU Fabrics：TPU ICI、NeuronLink／NeuronSwitch 与 Huawei UnifiedBus

#### 4.15.7 物理机架边界、网络域边界与故障域的区别

#### 4.15.8 扩展 Scale-Up Domain 的容量、距离与基础设施代价

### 4.16 Node-Scale Scale-Up Interconnect

<sub>Sources: Survey §4.1 and Fig. 6; Table 2 and Table 4 provide platform-specific qualifications.</sub>

#### 4.16.1 PCIe-Switched Attachment 与 Host／Root Complex

#### 4.16.2 Direct Full Mesh 与物理完整图

#### 4.16.3 Dedicated Switched Any-to-Any Fabric

#### 4.16.4 Gaudi、AMD Baseboard 与 HGX／NVSwitch 案例

#### 4.16.5 单跳路径、线缆数量、交换容量与端点注入

#### 4.16.6 系统配置与代际相关的拓扑标注

### 4.17 Rack-Scale Scale-Up Interconnect

<sub>Sources: Survey §4.2 and Fig. 7; case names retain the source platform and generation.</sub>

#### 4.17.1 NVL72 与 Rack-Scale Switched Fabric

#### 4.17.2 Trainium UltraServer 的跨服务器组织

#### 4.17.3 2D／3D Torus、Nearest-Neighbor 与 Wraparound

#### 4.17.4 Dragonfly 的 Local Group 与 Global Link

#### 4.17.5 交换芯片、Bounded Degree、Diameter 与布线成本

#### 4.17.6 Compute Tray／Switch Tray 与互连距离

### 4.18 Pod-Scale Scale-Up Interconnect

<sub>Sources: Survey §4.3 and Fig. 8; Boardfly and UB-Mesh retain the announcement/proposal qualifications of the supplied draft.</sub>

#### 4.18.1 TPU Pod、3D Torus 与 Optical Circuit Switching

#### 4.18.2 Boardfly：Four-Chip Building Block、Group 与组间连接

#### 4.18.3 UB-Mesh：Hierarchically Localized nD Full Mesh

#### 4.18.4 Maia：Fully Connected Quad 与 Ethernet-Switched Hierarchy

#### 4.18.5 Local／Global Connectivity、Oversubscription 与扩展边界

#### 4.18.6 已部署、Announced 与 Proposed Topology 的区分

### 4.19 Scale-Out、网络数据路径与光互连

<sub>Sources: Survey §4 and §6.3 motivate these topics; detailed scale-out protocols and operations are extensions from the original outline and corpus.</sub>

#### 4.19.1 Ethernet、InfiniBand、RoCE 与 EFA

#### 4.19.2 NIC、RDMA 与 Accelerator Memory 的数据路径

#### 4.19.3 Leaf–Spine／Clos 与多层网络

#### 4.19.4 Multi-Rail、Topology-Aware Placement 与流量隔离

#### 4.19.5 流控、拥塞控制、链路故障与重路由

#### 4.19.6 Electrical Link、Optical Link、OCS 与 Co-Packaged Optics

#### 4.19.7 In-Network Reduction、SmartNIC／DPU 与主机卸载

### 4.20 拓扑性能：连接能力不等于有效通信性能

<sub>Sources: Survey §4 and §6.3; network-analysis foundations retained from the original outline.</sub>

#### 4.20.1 Physical Link、Logical Reachability 与通信路径

#### 4.20.2 Degree／Radix、Diameter、Hop Count 与路径多样性

#### 4.20.3 Endpoint Injection、Switch Capacity 与链路方向

#### 4.20.4 Bisection Bandwidth、Oversubscription 与热点

#### 4.20.5 拓扑划分、消息大小、并发流与链路不对称

#### 4.20.6 Copper Reach、光学传播与系统同步延迟

#### 4.20.7 平台、带宽单位与双向聚合口径的对齐

### 4.21 Collective Semantics：端点数据变换

<sub>Sources: Survey §4.4 and Fig. 9: collective endpoint semantics.</sub>

#### 4.21.1 AllReduce：聚合与完整结果复制

#### 4.21.2 AllGather：分片收集与拼接

#### 4.21.3 ReduceScatter：聚合后保留结果分片

#### 4.21.4 All-to-All：面向目的 Rank 的数据交换

#### 4.21.5 ReduceScatter ＋ AllGather 的 AllReduce 实现

#### 4.21.6 DDP、FSDP／ZeRO、TP／SP 与 EP 的通信需求

#### 4.21.7 Collective Operation、Algorithm 与 Physical Topology 的区分

### 4.22 Collective Algorithms：逻辑通信调度

<sub>Sources: Survey §4.4: collective algorithms and their communication costs.</sub>

#### 4.22.1 Ring ReduceScatter／AllGather 与流水化

#### 4.22.2 Tree 与 Double Binary Tree

#### 4.22.3 Recursive Halving／Doubling 与 Rabenseifner-Style AllReduce

#### 4.22.4 Bruck-Style 与 Pairwise Exchange

#### 4.22.5 Parallel Aggregated Trees（PAT）

#### 4.22.6 Hierarchical Local–Global–Local Scheduling

#### 4.22.7 TACCL 与 Topology-Specific Schedule Synthesis

#### 4.22.8 消息大小、Rank Count、启动轮次与数据流量

### 4.23 Collective–Topology Mapping 与通信卸载

<sub>Sources: Survey §4.4, especially its topology-mapping and in-network-reduction discussion.</sub>

#### 4.23.1 Direct／Switched Fabric 上的并发 Ring、Tree 与 Pairwise Exchange

#### 4.23.2 Mesh／Torus 的维度分解与多 Ring 嵌入

#### 4.23.3 Dragonfly／Boardfly 的组内—组间—组内调度

#### 4.23.4 UB-Mesh／Maia 的分层局部性与链路共享

#### 4.23.5 NVLS／CollNet-Style Reduction Offload

#### 4.23.6 AllGather／All-to-All 的数据移动与 Reduction Offload 边界

#### 4.23.7 通信放置、链路争用与计算—通信重叠

### 4.24 GPU 代际演进：计算、数据供给与协作范围

<sub>Sources: Survey §5, Tables 3–4 and Figs. 11–15; [NVIDIA][chip-nvidia-gpu] and [AMD][chip-amd-gpu] corpus records. Preliminary specifications retain that status.</sub>

#### 4.24.1 NVIDIA：Pascal → Volta → Turing → Ampere → Hopper → Blackwell

#### 4.24.2 NVIDIA：Tensor Core 数据路径与控制粒度的跨代比较

#### 4.24.3 NVIDIA：Survey 中的 Rubin 披露与初步规格边界

#### 4.24.4 AMD：MI50 → CDNA／MI100 → MI200 → MI300 → MI350／MI400

#### 4.24.5 AMD：矩阵计算、片上存储与互连的跨代比较

#### 4.24.6 标量／向量吞吐与低精度矩阵吞吐的不同演进

#### 4.24.7 算术能力、内存供给、执行协调与部署形态的协同

### 4.25 专用加速器代际演进：工作负载定位与软硬件协同

<sub>Sources: Survey §2.3 and §5; [TPU][chip-google-tpu], [Neuron][chip-aws-neuron], [MTIA][chip-meta-mtia] and [Maia][chip-microsoft-maia] corpus records.</sub>

#### 4.25.1 TPU v1 → v2／v3：推理与训练设计的衔接

#### 4.25.2 TPU v4／v5e／v5p／v6e／v7：计算、存储与 Pod 演进

#### 4.25.3 TPU 8t／8i：Survey 中的工作负载定位与披露边界

#### 4.25.4 AWS Inferentia／Trainium：NeuronCore 与多引擎演进

#### 4.25.5 Trn1／Trn2：Instance 内 Torus 与 UltraServer 间连接

#### 4.25.6 Trn3：Survey 中的 Switched Fabric 演进

#### 4.25.7 MTIA／Maia：推荐、生成式 AI 与推理工作负载的变化

### 4.26 跨代比较：数值格式、结构化稀疏与峰值口径

<sub>Sources: Survey §5.1, Fig. 10, Fig. 12 and Table 3. This chapter compares hardware support; quantization methods remain in Algorithms.</sub>

#### 4.26.1 IEEE、AI-Optimized、Block-Scaled 与 Integer Formats

#### 4.26.2 输入精度、累加精度与输出精度的分离

#### 4.26.3 Microscaling 的共享 Scale 与表示开销

#### 4.26.4 2:4 与可变 M:N Structured Sparsity

#### 4.26.5 压缩权重、非零值 Metadata 与规则数据路径

#### 4.26.6 支持格式、Supported Operation 与算子映射条件

#### 4.26.7 Dense Peak、Sparse Peak 与端到端有效性能

### 4.27 跨代比较：Memory Hierarchy 与 Scale-Up Fabric

<sub>Sources: Survey §5.3, Table 4 and Fig. 15; corpus numerical discrepancies remain separately documented in the linked records.</sub>

#### 4.27.1 HBM Capacity：Die Density、Stack Height 与 Stack Count

#### 4.27.2 HBM Bandwidth：Signaling Rate、Interface Width 与 Stack Count

#### 4.27.3 Shared Cache、Scratchpad 与片上复用容量

#### 4.27.4 Chiplet Packaging 与 Memory PHY／Controller 集成

#### 4.27.5 GPU Fabric：Direct Connection、Switched Node 与 Rack Domain

#### 4.27.6 Neuron：Torus、Inter-Instance Ring 与 Switched Fabric

#### 4.27.7 TPU：Torus、OCS 与 Boardfly 的设计取舍

#### 4.27.8 设备内存、平台拓扑与代际数据口径的联合比较

### 4.28 供电、散热与可实际运行容量

<sub>Sources: Survey §5.4 and §6.4, including Fig. 16; broader RAS and fault-isolation topics are retained extensions.</sub>

#### 4.28.1 PCIe Air Cooling、SXM／OAM 与 Rack-Scale Direct Liquid Cooling

#### 4.28.2 Chip、Board、Rack 与 Facility 的功率预算

#### 4.28.3 Power Distribution、Busbar、Voltage Conversion 与导体约束

#### 4.28.4 54 VDC 与 Survey 中提出的更高电压供电方向

#### 4.28.5 Coolant Temperature、Thermal Resistance 与运行温度余量

#### 4.28.6 集成密度、铜互连距离与冷却形态

#### 4.28.7 平均功率、同步功率波动与短期储能

#### 4.28.8 Installed Capacity、Power-Capped Capacity 与 Cooling-Limited Capacity

#### 4.28.9 可靠性扩展：ECC、Link RAS、硬件故障隔离与维护

### 4.29 未来设计挑战：Generality 与 Specialization

<sub>Sources: Survey §6.1–§6.5.</sub>

#### 4.29.1 Workload Specialization and Fleet Flexibility

#### 4.29.2 Data Placement and Movement

#### 4.29.3 Communication Across Scale-Up and Scale-Out Networks

#### 4.29.4 Power Delivery and Cooling from Rack to Facility

#### 4.29.5 Architectural Specialization and Model Evolution

#### 4.29.6 Reconfigurable Model Graph 与 Hard-Wired Base Model 的边界

#### 4.29.7 Taalas HC1：Adapter 可调整性与基础模型固化的案例

#### 4.29.8 软件变化、硬件寿命与已部署资源的再利用

### 4.30 扩展／对照案例：光子、神经形态与有限披露设计

<sub>Sources: [Corpus extended catalogue][corpus]; these entries do not add categories to the survey taxonomy. Reported, historical and limited-disclosure designs remain qualified.</sub>

#### 4.30.1 Photonic Compute 与 Optical Interconnect 的区别

#### 4.30.2 [Lightmatter][chip-lightmatter]：计算与互连路径的分别讨论

#### 4.30.3 [Q.ANT][chip-q-ant]：Photonic NPU 的公开案例

#### 4.30.4 [Luminous][chip-luminous]：Photonic Interconnect／Electronic Compute 历史案例

#### 4.30.5 [SpiNNcloud SpiNNaker2][chip-spinncloud-spinnaker2]：Event／Spike-Driven Manycore

#### 4.30.6 [Tesla FSD][chip-tesla-fsd]：车载 NPU 与数据中心约束的对照

#### 4.30.7 [OpenAI–Broadcom／Jalapeño][chip-openai-broadcom]：公开披露与未确认内部结构

#### 4.30.8 研究扩展：FPGA 与其他可重构计算

### 4.31 架构评估与贯穿全栈的案例组织

<sub>Sources: Survey §3–§6 and the [corpus dossier structure][corpus]; evaluation exercises are curriculum design, not measured survey findings.</sub>

#### 4.31.1 统一案例模板：Compute → Data Path → Memory → Host → Fabric

#### 4.31.2 统一软件模板：Framework／Compiler／Libraries／Runtime／Driver／Firmware／Communication／ISA

#### 4.31.3 Programming-Model Rationale 与公开证据追踪

#### 4.31.4 GEMM／GEMV／Attention／MoE／Embedding 的对照映射

#### 4.31.5 硬件能力、Kernel 利用率与完整模型性能的分离

#### 4.31.6 Performance、Energy、Area、Cost 与可编程性的 Pareto Frontier

#### 4.31.7 来源日期、参考平台、软件版本与不可比数据的标注

#### 4.31.8 模型演进、资源重用与跨层设计取舍

---

<a id="system"></a>

## 5. System｜训练、推理与 AI 基础设施

> The original training/serving/RL scope is retained. Corpus-derived runtime boundaries and survey-derived fleet, communication, and power constraints are added as explicit cross-layer topics.

### 5.1 AI Runtime 与操作系统基础

#### 5.1.1 进程、线程、Coroutine 与 Python GIL

#### 5.1.2 CPU Affinity、NUMA 与 Host Memory

#### 5.1.3 GPU Context、Stream、Event 与 CUDA Graph

#### 5.1.4 IPC、Shared Memory、RPC 与序列化

#### 5.1.5 内存分配器、Memory Pool 与碎片化

#### 5.1.6 Driver、Runtime、Library 与容器依赖

### 5.2 Device Runtime、Driver 与 Firmware 的执行链

<sub>Sources: [NVIDIA][layers-nvidia-gpu], [AMD][layers-amd-gpu], [Neuron][layers-aws-neuron], [Ascend][layers-huawei-ascend], [Tenstorrent][layers-tenstorrent] and [Groq][layers-groq] layer mappings.</sub>

#### 5.2.1 模型装载、Executable Artifact 与设备资源初始化

#### 5.2.2 Context、Command Queue、Doorbell、Event 与 Interrupt

#### 5.2.3 Device Memory Allocation、Virtual Memory 与 Host–Device Transfer

#### 5.2.4 CUDA Runtime／Driver API 与 GPU Kernel Module 边界

#### 5.2.5 HIP／ROCr-HSA、AQL Queue 与 amdgpu／KFD 边界

#### 5.2.6 Neuron libnrt／NEFF、AscendCL 与专用设备执行

#### 5.2.7 TT-Metalium 的 Host／Device Program 与用户态／内核态驱动

#### 5.2.8 Thin Static Executor 与动态 Kernel-Launch Runtime 的区别

### 5.3 系统指标、排队与容量规划

#### 5.3.1 TTFT、TPOT／ITL 与 End-to-End Latency

#### 5.3.2 Request Throughput、Token Throughput 与 Goodput

#### 5.3.3 Latency Percentile、SLO 与尾延迟

#### 5.3.4 Arrival Process、Queueing 与 Little's Law

#### 5.3.5 MFU、HFU、设备利用率与有效工作

#### 5.3.6 Tokens／Dollar、Tokens／Joule 与总拥有成本

#### 5.3.7 容量、并发度、Batch Size 与负载拐点

### 5.4 模型工作负载与资源画像

#### 5.4.1 Training、Prefill、Decode 与 Verification

#### 5.4.2 参数、梯度、优化器状态与激活

#### 5.4.3 KV Cache、Recurrent State 与模型状态

#### 5.4.4 Dense、MoE、Multimodal 与 Diffusion 的执行画像

#### 5.4.5 Compute／Memory／Communication／CPU 瓶颈

#### 5.4.6 Batch Size、Context Length 与输出长度分布

### 5.5 Distributed Runtime 与 Collective Communication

<sub>Sources: Survey §4.4 for semantics and algorithms; corpus layer mappings for vendor runtime boundaries. Detailed API exercises remain extensions.</sub>

#### 5.5.1 Rank、World Size、Process Group 与 Device Mesh

#### 5.5.2 Collective 调用语义、完成语义与一致的调用次序

#### 5.5.3 Broadcast、Reduce、AllReduce、AllGather、ReduceScatter 与 All-to-All

#### 5.5.4 NCCL／RCCL、Gloo／MPI／UCX 与后端选择

#### 5.5.5 NCCom、ICI Runtime 与编译器内置 Collective 的不同边界

#### 5.5.6 拓扑探测、算法选择与通信性能调优

#### 5.5.7 异步通信、Stream／Event 与计算重叠

#### 5.5.8 通信死锁、Timeout、故障与诊断

### 5.6 Data Parallelism 与参数分片

#### 5.6.1 Synchronous SGD 与 Distributed Data Parallel

#### 5.6.2 Gradient Bucketing 与通信重叠

#### 5.6.3 ZeRO-1／2／3 与状态分片

#### 5.6.4 FSDP／FSDP2 与参数 All-Gather

#### 5.6.5 Gradient Accumulation 与有效 Batch Size

#### 5.6.6 Replicated、Sharded 与 Hybrid Sharded Training

### 5.7 Tensor Parallelism

#### 5.7.1 Column-Parallel 与 Row-Parallel Linear

#### 5.7.2 Attention Head 与 MLP 的切分

#### 5.7.3 Vocabulary、Embedding 与 Loss Parallelism

#### 5.7.4 TP 中的 All-Reduce／All-Gather／Reduce-Scatter

#### 5.7.5 KV Head 数量、复制与切分约束

#### 5.7.6 单节点与跨节点 Tensor Parallelism

### 5.8 Pipeline Parallelism

#### 5.8.1 Stage Partitioning 与 Microbatch

#### 5.8.2 GPipe、1F1B 与交错流水线

#### 5.8.3 Pipeline Bubble 与负载均衡

#### 5.8.4 Zero-Bubble Scheduling 与执行依赖

#### 5.8.5 Activation Communication 与跨 Stage 重叠

#### 5.8.6 Virtual Pipeline Stage 与动态重分区

### 5.9 Sequence 与 Context Parallelism

#### 5.9.1 Sequence Parallelism 与 Context Parallelism 的边界

#### 5.9.2 Long-Context Attention 的序列切分

#### 5.9.3 Ring Attention 与 KV Block 传递

#### 5.9.4 Ulysses 与 All-to-All 序列重排

#### 5.9.5 分层、混合与二维 Context Parallelism

#### 5.9.6 因果掩码、负载均衡与通信隐藏

### 5.10 Expert Parallelism

#### 5.10.1 Expert Placement、Replication 与 Sharding

#### 5.10.2 Token Dispatch、All-to-All 与 Combine

#### 5.10.3 Expert Parallelism 与 Tensor Parallelism 的组合

#### 5.10.4 Expert Load Balance 与 Hot Expert

#### 5.10.5 Dropless Routing 与 Capacity Management

#### 5.10.6 跨节点 MoE 通信与拓扑感知部署

### 5.11 混合并行与自动规划

<sub>Sources: Survey §4.4 and §6.3 motivate communication placement; distributed-training implementation topics are retained.</sub>

#### 5.11.1 DP／TP／PP／SP／CP／EP 的组合

#### 5.11.2 Device Mesh、Parallel Group 与拓扑分层

#### 5.11.3 模型切分、Resharding 与布局转换

#### 5.11.4 内存约束、消息大小、通信延迟与并行度选择

#### 5.11.5 跨 Scale-Up／Scale-Out 域的通信放置

#### 5.11.6 异构设备、Straggler 与不均匀分片

#### 5.11.7 自动并行、搜索与性能模型

### 5.12 训练内存管理

#### 5.12.1 Activation Checkpointing 与 Selective Recomputation

#### 5.12.2 CPU／NVMe Offload 与状态分层

#### 5.12.3 Activation Offload、Prefetch 与异步搬运

#### 5.12.4 Optimizer State Sharding 与低精度状态

#### 5.12.5 Memory Budget、Peak Memory 与 Fragmentation

#### 5.12.6 训练图、通信 Buffer 与 CUDA Graph Memory

### 5.13 训练数据与 Checkpoint Pipeline

#### 5.13.1 Tokenization、Packing 与动态批次

#### 5.13.2 Dataloader、Prefetch 与 CPU–GPU Pipeline

#### 5.13.3 分布式数据 Sharding、Shuffle 与恢复

#### 5.13.4 Checkpoint Save／Load 与并行 I/O

#### 5.13.5 Async Checkpoint 与增量保存

#### 5.13.6 数据、参数、优化器与随机状态的一致性

### 5.14 训练 Runtime 与数值稳定性

#### 5.14.1 Training Step、Autograd 与 Optimizer Step

#### 5.14.2 混合精度训练与硬件／Kernel 支持范围

#### 5.14.3 Loss Scaling、Gradient Clipping 与异常检测

#### 5.14.4 通信精度、累加顺序与可复现性

#### 5.14.5 NaN／Inf、Loss Spike 与训练发散排查

#### 5.14.6 算子、编译器与分布式训练的协同优化

### 5.15 大规模训练的弹性与容错

#### 5.15.1 故障域、Heartbeat 与健康检查

#### 5.15.2 Checkpoint／Restart 与快速恢复

#### 5.15.3 Elastic Training 与动态 World Size

#### 5.15.4 Straggler Detection 与慢节点替换

#### 5.15.5 网络故障、存储故障与静默数据损坏

#### 5.15.6 训练效率、故障成本与可用容量

### 5.16 Distributed Training Frameworks 与加速器后端

<sub>Sources: Original training outline, extended with [TPU][layers-google-tpu], [Neuron][layers-aws-neuron] and [Ascend][layers-huawei-ascend] framework mappings.</sub>

#### 5.16.1 PyTorch Distributed：DDP、FSDP 与 DTensor

#### 5.16.2 Megatron-LM／Megatron-Core 与 DeepSpeed

#### 5.16.3 TorchTitan 与原生 PyTorch 训练

#### 5.16.4 JAX／MaxText 与 TPU 训练

#### 5.16.5 NeMo 与训练工作流集成

#### 5.16.6 AWS NxD Training、Ascend 适配与设备特定训练路径

#### 5.16.7 统一训练概念与不同后端支持范围的区别

### 5.17 一次推理请求的完整生命周期

#### 5.17.1 API Server、Tokenizer 与 Request Queue

#### 5.17.2 Scheduler、Worker、Model Runner 与 Executor

#### 5.17.3 模型配置、权重加载与参数初始化

#### 5.17.4 Prefill、Decode、Sampling 与 Detokenization

#### 5.17.5 Streaming Output、Cancellation 与资源回收

#### 5.17.6 Embedding、Reranking、Reward 与生成请求

### 5.18 Batching 与在线调度

#### 5.18.1 Static Batching、Dynamic Batching 与 Continuous Batching

#### 5.18.2 Iteration-Level Scheduling 与 Token Budget

#### 5.18.3 Chunked Prefill 与 Decode 优先级

#### 5.18.4 Preemption、Recompute 与 Swap

#### 5.18.5 CPU–GPU Overlap 与低开销 Scheduler

#### 5.18.6 公平性、优先级与长短请求混合

#### 5.18.7 负载感知 Batch Size 与自适应调度

### 5.19 KV Cache 的分配、寻址与生命周期

#### 5.19.1 Contiguous、Paged 与分块 KV Cache

#### 5.19.2 Block Table、Memory Pool 与动态扩容

#### 5.19.3 MHA／GQA／MLA 的缓存组织

#### 5.19.4 Prefix Sharing、Reference Counting 与 Copy-on-Write

#### 5.19.5 KV Cache Eviction、Compaction 与碎片

#### 5.19.6 Cache Hit、OOM 与内存预算诊断

### 5.20 Prefix Cache 与多级缓存

#### 5.20.1 Prefix Caching 与 Radix-Tree／Hash 索引

#### 5.20.2 Cache-Aware Scheduling 与前缀局部性

#### 5.20.3 [HiCache：HBM、Host DRAM 与外部存储][hicache]

#### 5.20.4 LMCache、Mooncake 与共享 KV 存储

#### 5.20.5 Cache Prefetch、Migration 与异步写回

#### 5.20.6 跨进程、跨设备与跨节点缓存一致性

#### 5.20.7 多租户缓存隔离与失效管理

### 5.21 Disaggregated Serving

<sub>Sources: Survey §6.1 frames specialization and fleet flexibility; named serving systems are retained from the original outline.</sub>

#### 5.21.1 Prefill–Decode Disaggregation 与阶段资源需求

#### 5.21.2 KV Transfer、Connector 与数据面

#### 5.21.3 Prefill／Decode 的独立扩缩容

#### 5.21.4 异构硬件选择、数据格式与迁移成本

#### 5.21.5 Attention–FFN Disaggregation 与细粒度拆分

#### 5.21.6 资源池不均衡、队列增长与闲置容量

#### 5.21.7 Splitwise、DistServe 与 Mooncake 案例

### 5.22 Routing 与分布式服务调度

#### 5.22.1 负载均衡、Join-Shortest-Queue 与请求路由

#### 5.22.2 KV-Aware、Prefix-Aware 与 Cache-Aware Routing

#### 5.22.3 Replica Placement 与会话亲和性

#### 5.22.4 Admission Control、Backpressure 与 Rate Limiting

#### 5.22.5 Autoscaling、Cold Start 与 Warm Pool

#### 5.22.6 跨机架、跨区域与分层服务架构

### 5.23 MoE Serving 的系统设计

#### 5.23.1 Expert Parallel Serving 与通信路径

#### 5.23.2 Attention-DP 与 Expert-Parallel 组合

#### 5.23.3 Expert Parallel Load Balancing

#### 5.23.4 Hot Expert Replication 与动态放置

#### 5.23.5 Expert Offload、Prefetch 与 CPU–GPU 协作

#### 5.23.6 共享专家、路由专家与跨设备执行

### 5.24 量化模型的部署与调优

#### 5.24.1 权重、激活与 KV Cache 的精度配置

#### 5.24.2 Checkpoint Format、Packing 与量化元数据

#### 5.24.3 量化方案、Kernel 与硬件支持匹配

#### 5.24.4 离线量化、在线量化与加载时转换

#### 5.24.5 Dense／MoE／Multimodal 的混合精度部署

#### 5.24.6 精度、延迟、吞吐、容量与成本联合评估

### 5.25 稀疏模型与长上下文的系统支持

#### 5.25.1 Weight Sparsity 与稀疏权重存储

#### 5.25.2 Sparse Attention 的索引、路由与缓存访问

#### 5.25.3 动态 Token Selection 与批次组织

#### 5.25.4 KV Pruning／Compression 与内存管理

#### 5.25.5 动态稀疏的负载均衡与通信

#### 5.25.6 稀疏计算收益与预处理、元数据开销

### 5.26 Speculative Decoding 的系统实现

#### 5.26.1 Draft Model、Target Model 与 Verification Worker

#### 5.26.2 Draft／Target 的同卡、分卡与异构部署

#### 5.26.3 Acceptance-Aware 与 Load-Aware Draft Length

#### 5.26.4 Tree／Block Verification 与批次容量

#### 5.26.5 Draft／Target KV Cache 管理与回滚

#### 5.26.6 在线请求、吞吐与单请求延迟的权衡

#### 5.26.7 Draft Model 训练、更新与服务集成

### 5.27 多模型、多租户与适配器服务

#### 5.27.1 Multi-Model Serving 与模型路由

#### 5.27.2 Multi-LoRA Batching 与 Adapter Cache

#### 5.27.3 权重共享、热加载与动态模型切换

#### 5.27.4 GPU Sharing、MIG、MPS 与资源隔离

#### 5.27.5 QoS-Constrained Co-Location 与干扰管理

#### 5.27.6 租户公平性、配额与 SLO 隔离

### 5.28 Serving Frameworks、组件生态与设备后端

<sub>Sources: Original serving outline; corpus mappings motivate backend-specific coverage checks rather than assuming every framework runs on every accelerator.</sub>

#### 5.28.1 [vLLM：Engine、Scheduler、Worker 与 PagedAttention][vllm]

#### 5.28.2 [SGLang：Scheduler、RadixAttention 与 Model Runner][sglang]

#### 5.28.3 TensorRT-LLM 与 NVIDIA Triton Inference Server

#### 5.28.4 NVIDIA Dynamo、llm-d 与分布式推理编排

#### 5.28.5 Ray Serve 与多阶段服务 Pipeline

#### 5.28.6 llama.cpp、MLX 与 CPU／本地推理

#### 5.28.7 FlashInfer、NIXL 与可复用推理组件

#### 5.28.8 后端兼容矩阵：模型、精度、Kernel、设备与 SDK 版本

#### 5.28.9 专用硬件的 SDK／服务入口与通用 Serving Framework 的边界

### 5.29 RL Post-Training Infrastructure

#### 5.29.1 Actor、Rollout、Reward、Critic 与 Reference Model

#### 5.29.2 Training Engine 与 Inference Engine 的组合

#### 5.29.3 Colocated、Disaggregated 与混合资源布局

#### 5.29.4 Synchronous／Asynchronous RL Pipeline

#### 5.29.5 Partial Rollout、Dynamic Sampling 与 Straggler

#### 5.29.6 Multi-Turn Rollout、环境交互与工具执行

#### 5.29.7 Online Weight Update 与权重广播

### 5.30 训练—推理一致性与 RL 正确性

#### 5.30.1 Tokenizer、Chat Template 与 Special Token 对齐

#### 5.30.2 Log Probability、Mask 与序列边界

#### 5.30.3 Sampling Distribution 与训练目标的一致性

#### 5.30.4 Kernel、精度与 Reduction 路径的数值差异

#### 5.30.5 Policy Version、Weight Staleness 与 Off-Policy Drift

#### 5.30.6 Importance Sampling 与失配修正

#### 5.30.7 梯度、样本与性能优化的正确性验证

### 5.31 RL Frameworks 与源码专题

#### 5.31.1 verl：初始化、Rollout 与训练工作流

#### 5.31.2 slime：Rollout-First 设计与后端集成

#### 5.31.3 OpenRLHF 与 Ray-Based Actor 管理

#### 5.31.4 AReaL 与异步 RL 训练

#### 5.31.5 TRL 与轻量后训练工作流

#### 5.31.6 权重更新、Memory Sleep／Wake 与资源复用

#### 5.31.7 RL 中的 FP8／INT4、Speculative Decoding 与多轮任务

### 5.32 Multimodal 与实时语音服务

#### 5.32.1 图像／视频预处理与视觉 Encoder

#### 5.32.2 Encoder–Decoder 多阶段调度

#### 5.32.3 视觉 Token、跨模态 Cache 与 Batching

#### 5.32.4 Audio Codec、Dual-AR 与 Thinker–Talker Pipeline

#### 5.32.5 Streaming ASR、TTS、Vocoder 与全双工交互

#### 5.32.6 CPU 资源、音视频 I/O 与实时延迟预算

### 5.33 Diffusion 与非自回归模型系统

#### 5.33.1 Denoising Step、Scheduler 与多阶段执行

#### 5.33.2 CFG Parallelism 与模型并行

#### 5.33.3 Sequence／Patch／Temporal Parallelism

#### 5.33.4 跨 Step 的 Feature Cache 与计算复用

#### 5.33.5 Diffusion LLM 的 Block Scheduling 与缓存

#### 5.33.6 图像／视频生成的显存、吞吐与服务编排

### 5.34 RAG、Agent 与复合 AI 系统

#### 5.34.1 Retrieval、Reranking、Generation 与 Index Pipeline

#### 5.34.2 Vector Database、Embedding 与检索缓存

#### 5.34.3 Tool Calling、Sandbox 与环境资源管理

#### 5.34.4 Multi-Turn Session、Long-Term State 与上下文管理

#### 5.34.5 Agent Workflow、并行工具与长尾任务调度

#### 5.34.6 Model Routing、Cascade 与系统级缓存

#### 5.34.7 复合任务的端到端质量、延迟与成本

### 5.35 集群编排与生产部署

#### 5.35.1 Slurm、Kubernetes 与 GPU Resource Scheduling

#### 5.35.2 Gang Scheduling、Quota 与集群公平性

#### 5.35.3 Docker、镜像、依赖与可复现环境

#### 5.35.4 Model Registry、Artifact Storage 与发布流水线

#### 5.35.5 Canary、Rolling Upgrade 与回滚

#### 5.35.6 多集群部署、灾备与安全边界

### 5.36 Heterogeneous Fleet 与专用资源池的调度

<sub>Sources: Survey §6.1–§6.3; scheduling mechanisms extend the survey’s design-challenge framing.</sub>

#### 5.36.1 Training、Prefill、Decode、Draft／Verify 与多模态阶段画像

#### 5.36.2 统一资源池与按工作负载划分的专用资源池

#### 5.36.3 模型兼容性、精度支持与可用 Kernel 的调度约束

#### 5.36.4 容量碎片、闲置设备、队列不均衡与资源再平衡

#### 5.36.5 权重、KV Cache 与激活迁移的数据移动成本

#### 5.36.6 机架／Pod 放置与跨网络域通信

#### 5.36.7 吞吐、尾延迟、成本与已部署硬件复用

### 5.37 Power／Thermal-Aware 系统运行

<sub>Sources: Survey §5.4 and §6.4; runtime policies are tutorial extensions, not evaluated algorithms in the survey.</sub>

#### 5.37.1 设备、机架与 Facility Power Budget 的衔接

#### 5.37.2 Power Capping、频率调整与任务进度

#### 5.37.3 同步训练阶段、功率波动与短期能量缓冲

#### 5.37.4 冷却能力、热余量与可同时运行的设备数

#### 5.37.5 放置、并发度、Batch Size 与功率限制的联合考虑

#### 5.37.6 平均功率、峰值功率、Tokens／Joule 与 SLO

#### 5.37.7 硬件设计约束与运行期调度策略的边界

### 5.38 安全、隐私与可信 AI 基础设施

#### 5.38.1 认证、授权、配额与多租户数据隔离

#### 5.38.2 模型权重、序列化格式与软件供应链安全

#### 5.38.3 Confidential Computing、TEE 与机密加速器计算

#### 5.38.4 KV Cache 隔离、敏感数据与侧信道防护

#### 5.38.5 Tool Sandbox、最小权限与 Prompt Injection 防护

#### 5.38.6 审计、数据溯源与隐私保护工作流

### 5.39 Observability、Benchmark 与故障排查

#### 5.39.1 Metrics、Logs、Tracing 与跨组件时间线

#### 5.39.2 PyTorch Profiler、Nsight、ROCm 工具与设备专用分析器

#### 5.39.3 Framework／Compiler／Kernel／Runtime／Driver 的分层定位

#### 5.39.4 GPU／NPU／CPU／网络／存储的联合监控

#### 5.39.5 Offline／Online Benchmark 与真实流量回放

#### 5.39.6 NCCL Hang、OOM、Memory Leak 与 CPU Bottleneck

#### 5.39.7 性能回归、版本固定、数值一致性与实验可复现性

#### 5.39.8 MLPerf、Serving Benchmark 与报告口径

### 5.40 [系统源码阅读与跨层端到端实践][awesome]

<sub>Sources: Original systems reading index and corpus software-layer organization; planned walkthrough exercises.</sub>

#### 5.40.1 从零实现 Mini Training Runtime

#### 5.40.2 从零实现 Continuous-Batching LLM Server

#### 5.40.3 一个请求穿过 SGLang／vLLM 的完整路径

#### 5.40.4 一个 RL Step 穿过 Rollout／Reward／Training 的完整路径

#### 5.40.5 从 HF Checkpoint 到量化、多卡与分离式部署

#### 5.40.6 从模型前端到设备执行：公开接口范围内的跨平台路径

#### 5.40.7 Kernel → Compiler → Runtime → Driver → Hardware 的跨层诊断

#### 5.40.8 同一模型在不同硬件／编程模型上的部署差异

---

<a id="algorithms"></a>

## 6. Algorithms｜模型、训练方法与生成算法

> Retained model and algorithm curriculum. The hardware survey motivates workload diversity and adaptability, but is not the source for the detailed algorithms or model-family histories below.

### 6.1 深度学习与模型计算基础

#### 6.1.1 Tensor、Linear Layer、MLP 与 Activation

#### 6.1.2 Loss、Gradient、Backpropagation 与 Optimizer

#### 6.1.3 Batch、Sequence、Hidden Dimension 与 Parameter Count

#### 6.1.4 计算图、自动微分与训练／推理差异

#### 6.1.5 计算复杂度、空间复杂度与数据移动

#### 6.1.6 Dense、Sparse 与 Conditional Computation

### 6.2 Foundation Model 的完整生命周期

#### 6.2.1 Pretraining、Continued Pretraining 与 Mid-Training

#### 6.2.2 Supervised Fine-Tuning 与 Instruction Tuning

#### 6.2.3 Preference Optimization 与 Reinforcement Learning

#### 6.2.4 Distillation、Compression 与 Deployment

#### 6.2.5 Inference、Test-Time Compute 与持续评测

#### 6.2.6 Scaling Laws 与 Compute-Optimal Training

### 6.3 Tokenization 与输入表示

#### 6.3.1 BPE、WordPiece、Unigram 与 Byte-Level Tokenization

#### 6.3.2 Vocabulary、Embedding 与输出投影

#### 6.3.3 Special Token、Chat Template 与对话格式

#### 6.3.4 Padding、Packing、Mask 与序列边界

#### 6.3.5 文本、图像、音频与视频 Token

#### 6.3.6 Tokenization 对上下文长度与计算成本的影响

### 6.4 Transformer 的结构与演变

#### 6.4.1 Encoder-Only、Decoder-Only 与 Encoder–Decoder

#### 6.4.2 Self-Attention、Cross-Attention 与 FFN

#### 6.4.3 Residual Connection、LayerNorm 与 RMSNorm

#### 6.4.4 Pre-Norm、Post-Norm 与深层网络稳定性

#### 6.4.5 GELU、GLU、GeGLU 与 SwiGLU

#### 6.4.6 Weight Tying、Embedding 与 LM Head

### 6.5 Attention 的基本机制

#### 6.5.1 Scaled Dot-Product Attention

#### 6.5.2 Q、K、V 与 Attention Score

#### 6.5.3 Causal、Bidirectional 与 Cross-Attention Mask

#### 6.5.4 Softmax、归一化与稳定性

#### 6.5.5 Attention 的计算、存储与长序列复杂度

#### 6.5.6 Exact Attention 与稀疏／近似 Attention 的边界

### 6.6 MHA、MQA、GQA 与 MLA

#### 6.6.1 Multi-Head Attention（MHA）

#### 6.6.2 Multi-Query Attention（MQA）

#### 6.6.3 Grouped-Query Attention（GQA）

#### 6.6.4 Multi-Head Latent Attention（MLA）

#### 6.6.5 Head Sharing、Latent Compression 与缓存容量

#### 6.6.6 KV Cache 与增量自回归推理

#### 6.6.7 Attention 变体的质量、带宽与并行性权衡

### 6.7 Position Encoding 与上下文扩展

#### 6.7.1 Absolute 与 Relative Position Embedding

#### 6.7.2 RoPE、ALiBi 与位置表示

#### 6.7.3 RoPE Scaling、Position Interpolation 与 YaRN

#### 6.7.4 Sliding Window、Attention Sink 与流式上下文

#### 6.7.5 长上下文训练、外推与检索能力

#### 6.7.6 Context Compression、Memory 与跨段状态

### 6.8 Sparse 与 Compressed Attention

#### 6.8.1 Token、Block、Head 与层级稀疏

#### 6.8.2 Local、Global、Dilated 与 Sliding-Window Attention

#### 6.8.3 Longformer、BigBird 与稀疏连接模式

#### 6.8.4 [Native Sparse Attention（NSA）][nsa]

#### 6.8.5 Mixture of Block Attention（MoBA）

#### 6.8.6 DeepSeek Sparse Attention（DSA）

#### 6.8.7 [Compressed Sparse Attention（CSA）与 Heavily Compressed Attention（HCA）][deepseek4]

#### 6.8.8 动态 Token Selection、Top-k 与可训练稀疏

### 6.9 Linear Attention、SSM 与 Hybrid Model

#### 6.9.1 Kernelized Linear Attention 与线性复杂度

#### 6.9.2 RetNet、RWKV 与递归状态

#### 6.9.3 S4、Mamba 与 Mamba-2

#### 6.9.4 DeltaNet 与 Gated DeltaNet

#### 6.9.5 Attention–SSM／Linear-Attention 混合结构

#### 6.9.6 Prefill Scan、Recurrent Decode 与状态容量

### 6.10 Mixture of Experts

#### 6.10.1 Dense FFN → Sparse MoE

#### 6.10.2 Router、Top-k Routing 与 Expert Selection

#### 6.10.3 Token Choice、Expert Choice 与容量约束

#### 6.10.4 Shared Expert、Fine-Grained Expert 与专家划分

#### 6.10.5 Auxiliary Loss 与 Auxiliary-Loss-Free Balancing

#### 6.10.6 Expert Specialization、Routing Collapse 与训练稳定性

#### 6.10.7 Activated Parameters、总参数量与有效计算

### 6.11 预训练目标与优化算法

#### 6.11.1 Causal LM、Masked LM 与 Denoising Objective

#### 6.11.2 Next-Token Prediction 与 Multi-Token Prediction

#### 6.11.3 SGD、AdamW、Adafactor 与 Muon

#### 6.11.4 Learning Rate Schedule、Warm-up 与 Weight Decay

#### 6.11.5 Gradient Clipping、Batch Scaling 与优化稳定性

#### 6.11.6 低精度训练、误差积累与精度敏感性

### 6.12 数据、训练配方与模型扩展

#### 6.12.1 数据收集、过滤、去重与质量评估

#### 6.12.2 数据配比、Curriculum 与多语言训练

#### 6.12.3 Synthetic Data、Self-Training 与数据蒸馏

#### 6.12.4 模型宽度、深度、词表与序列长度

#### 6.12.5 Continual Learning、Domain Adaptation 与灾难性遗忘

#### 6.12.6 数据质量、Token Budget 与 Scaling Trade-off

### 6.13 SFT 与 Parameter-Efficient Fine-Tuning

#### 6.13.1 Instruction Tuning 与对话监督

#### 6.13.2 Full Fine-Tuning 与部分参数更新

#### 6.13.3 Adapter、Prefix Tuning 与 Prompt Tuning

#### 6.13.4 LoRA、QLoRA、DoRA 与低秩适配

#### 6.13.5 Multi-Task／Multi-Domain Fine-Tuning

#### 6.13.6 适配器合并、模型合并与迁移

### 6.14 Alignment 与偏好学习

#### 6.14.1 Reward Model、Preference Data 与 Pairwise Ranking

#### 6.14.2 RLHF 与 RLAIF

#### 6.14.3 PPO、KL Regularization 与 Reference Policy

#### 6.14.4 DPO、IPO、KTO 与直接偏好优化

#### 6.14.5 Rejection Sampling 与 Best-of-N 数据筛选

#### 6.14.6 Safety Alignment、Refusal 与偏好泛化

### 6.15 Reasoning 与 RL Post-Training

#### 6.15.1 可验证奖励与 RLVR

#### 6.15.2 GRPO、REINFORCE-Style 与组内优势估计

#### 6.15.3 DAPO、Reward Shaping 与样本过滤

#### 6.15.4 Outcome Reward 与 Process Reward

#### 6.15.5 On-Policy、Off-Policy 与 Importance Sampling

#### 6.15.6 长链推理、Multi-Turn RL 与 Agentic RL

#### 6.15.7 Reward Hacking、长度偏置与训练稳定性

### 6.16 自回归推理与采样

#### 6.16.1 Prefill、Decode 与逐 Token 生成

#### 6.16.2 Greedy、Temperature、Top-k 与 Top-p

#### 6.16.3 Beam Search、长度惩罚与重复控制

#### 6.16.4 Logits Processor、Stopping Criteria 与控制 Token

#### 6.16.5 Constrained Decoding、Grammar 与 Structured Output

#### 6.16.6 生成质量、多样性、延迟与可复现性

### 6.17 Test-Time Compute 与推理策略

#### 6.17.1 Chain-of-Thought 与显式推理

#### 6.17.2 Self-Consistency、Best-of-N 与多样本选择

#### 6.17.3 Verifier、Reward-Guided Search 与重排序

#### 6.17.4 Tree Search、规划与多步求解

#### 6.17.5 Adaptive Reasoning Budget 与提前终止

#### 6.17.6 Inference-Time Scaling 与计算分配

### 6.18 Speculative Decoding：推测解码

#### 6.18.1 Draft–Verify 框架与 Speculative Sampling

#### 6.18.2 Acceptance／Rejection 与目标分布保持

#### 6.18.3 Independent Draft Model 与 Self-Speculation

#### 6.18.4 N-Gram、Prompt Lookup 与 Retrieval-Based Drafting

#### 6.18.5 Multi-Token Prediction 与推测解码的关系

#### 6.18.6 Medusa、Hydra、ReDrafter 与多头／树式草稿

#### 6.18.7 [EAGLE、EAGLE-2 与 EAGLE-3][eagle3]

#### 6.18.8 [DFlash：Block-Diffusion Drafting][dflash]

#### 6.18.9 [DSpark：Semi-Autoregressive Drafting 与 Confidence-Scheduled Verification][dspark]

#### 6.18.10 接受长度、Draft Cost、Verification Cost 与加速边界

### 6.19 并行、块式与 Diffusion Language Generation

#### 6.19.1 Autoregressive、Semi-Autoregressive 与 Non-Autoregressive Generation

#### 6.19.2 Blockwise Parallel Decoding 与迭代修正

#### 6.19.3 [Set Block Decoding（SBD）][sbd]

#### 6.19.4 Masked Diffusion Language Modeling

#### 6.19.5 LLaDA、LLaDA 2.0、Dream 与 Block Diffusion

#### 6.19.6 Token Update Order、Remasking 与采样步数

#### 6.19.7 KV Cache 兼容性与生成质量的权衡

### 6.20 Quantization：算法与数值方法

#### 6.20.1 PTQ、QAT 与 Quantization-Aware Distillation

#### 6.20.2 Symmetric／Asymmetric 与 Uniform／Non-Uniform Quantization

#### 6.20.3 Per-Tensor、Per-Channel、Per-Group 与 Per-Token Scaling

#### 6.20.4 Weight、Activation 与 KV Cache Quantization

#### 6.20.5 GPTQ、AWQ、SmoothQuant 与误差补偿

#### 6.20.6 Rotation-Based Quantization：QuaRot 与 SpinQuant

#### 6.20.7 Outlier、Clipping、Calibration 与敏感性分析

#### 6.20.8 Mixed Precision、Block Scaling 与精度预算

### 6.21 Sparsification、Pruning 与模型压缩

#### 6.21.1 Unstructured、Structured 与 N:M Pruning

#### 6.21.2 Magnitude、Gradient 与 Second-Order Pruning

#### 6.21.3 SparseGPT、Wanda 与训练后剪枝

#### 6.21.4 Activation Sparsity、Token Pruning 与 Early Exit

#### 6.21.5 Head、Layer 与 Expert Pruning

#### 6.21.6 Low-Rank Factorization 与结构压缩

#### 6.21.7 Knowledge Distillation、Self-Distillation 与 Teacher–Student 学习

### 6.22 Multimodal Foundation Models

#### 6.22.1 Vision Encoder、Projector 与 Language Backbone

#### 6.22.2 CLIP／SigLIP、ViT 与对比学习

#### 6.22.3 LLaVA-Style 对齐与视觉指令微调

#### 6.22.4 Early Fusion、Late Fusion 与 Native Multimodality

#### 6.22.5 Cross-Attention、Image Token 与动态分辨率

#### 6.22.6 Audio／Video Tokenization 与跨模态时序

#### 6.22.7 ASR、TTS、Codec LM 与 Omni Model

### 6.23 Diffusion、Flow Matching 与视觉生成

#### 6.23.1 DDPM、DDIM 与去噪扩散

#### 6.23.2 Score-Based Modeling 与采样过程

#### 6.23.3 Latent Diffusion 与 VAE

#### 6.23.4 U-Net、Diffusion Transformer（DiT）与 MMDiT

#### 6.23.5 Flow Matching 与 Rectified Flow

#### 6.23.6 Classifier-Free Guidance 与条件生成

#### 6.23.7 Distillation、Consistency Model 与少步生成

#### 6.23.8 Video Diffusion、时空 Attention 与长视频生成

### 6.24 Retrieval、Tools 与 Agent Algorithms

#### 6.24.1 Sparse Retrieval、Dense Retrieval 与 Hybrid Retrieval

#### 6.24.2 Embedding Model、Reranker 与 RAG

#### 6.24.3 Retrieval-Augmented Pretraining 与知识更新

#### 6.24.4 Tool Use、Function Calling 与行动空间

#### 6.24.5 ReAct、规划、记忆与上下文压缩

#### 6.24.6 Multi-Agent Coordination 与协同求解

#### 6.24.7 工具反馈、环境奖励与 Agent 训练

### 6.25 LLM 之外的 AI 工作负载

#### 6.25.1 CNN、ResNet 与视觉识别

#### 6.25.2 Recommendation、DLRM 与 Embedding-Heavy Model

#### 6.25.3 Graph Neural Network 与稀疏消息传递

#### 6.25.4 Speech、Time-Series 与 Sequence Model

#### 6.25.5 Vision-Language-Action 与机器人策略

#### 6.25.6 Scientific ML、Neural Operator 与结构预测

#### 6.25.7 不同工作负载的计算、存储与通信特征

### 6.26 Communication-Efficient Learning 与分布式优化

#### 6.26.1 Local SGD 与周期性参数平均

#### 6.26.2 Gradient Quantization、Sparsification 与 Error Feedback

#### 6.26.3 Low-Rank Gradient Compression 与 PowerSGD

#### 6.26.4 异步训练、Staleness 与收敛

#### 6.26.5 低带宽分布式预训练与 DiLoCo

#### 6.26.6 Federated Learning、Secure Aggregation 与 Differential Privacy

#### 6.26.7 通信预算、统计效率与 Wall-Clock Training Time

### 6.27 模型演变：从序列模型到基础模型

#### 6.27.1 RNN、LSTM 与 Seq2Seq

#### 6.27.2 Attention 与原始 Transformer

#### 6.27.3 BERT、T5 与预训练范式

#### 6.27.4 GPT、GPT-2 与 GPT-3

#### 6.27.5 InstructGPT 与指令／偏好对齐

#### 6.27.6 Dense LM → MoE → Reasoning／Multimodal Model

### 6.28 模型家族：Llama

#### 6.28.1 LLaMA：开放权重基础模型

#### 6.28.2 Llama 2：Base 与 Chat

#### 6.28.3 Llama 3／3.1：训练规模与长上下文

#### 6.28.4 Llama 3.2／3.3：模型分支与能力演进

#### 6.28.5 [Llama 4：公开模型报告与架构演进][llama4]

#### 6.28.6 Llama 家族的 Attention、数据配方与部署比较

### 6.29 模型家族：Mistral 与 Mixtral

#### 6.29.1 Mistral 7B 与 Sliding-Window Attention

#### 6.29.2 Mixtral 8×7B 与 Sparse MoE

#### 6.29.3 Mixtral 8×22B 与模型扩展

#### 6.29.4 Mistral 的指令、代码与多模态分支

#### 6.29.5 Mistral／Mixtral 的路由、缓存与部署特征

### 6.30 模型家族：DeepSeek

#### 6.30.1 DeepSeek LLM 与 DeepSeekMoE

#### 6.30.2 DeepSeek-V2：MLA 与细粒度 MoE

#### 6.30.3 DeepSeek-V3：低精度训练、负载均衡与 MTP

#### 6.30.4 DeepSeek-R1：Reasoning RL 与蒸馏

#### 6.30.5 DeepSeek-V3.1／V3.2：推理模式与稀疏 Attention

#### 6.30.6 [DeepSeek-V4：公开模型报告与部署需求][deepseek4]

#### 6.30.7 DeepSeek 家族的模型—Kernel—系统协同

### 6.31 模型家族：Qwen

#### 6.31.1 Qwen2 与 Qwen2.5

#### 6.31.2 Qwen2.5-Coder、Math、VL 与 Omni 分支

#### 6.31.3 Qwen3：Dense／MoE 与 Thinking／Non-Thinking

#### 6.31.4 Qwen3-Next 与混合序列建模

#### 6.31.5 [Qwen3.5：公开模型报告与结构演进][qwen35]

#### 6.31.6 Qwen 家族的结构、后训练与部署比较

### 6.32 模型家族：Kimi

#### 6.32.1 Kimi 长上下文模型与技术路线

#### 6.32.2 Kimi K1.5 与多模态 Reasoning RL

#### 6.32.3 Kimi K2 与大规模 MoE

#### 6.32.4 Kimi K2 Thinking 与 Agentic Reasoning

#### 6.32.5 [Kimi K2.5：公开模型报告与多模态／Agent 工作负载][kimi25]

#### 6.32.6 Kimi 家族的 Attention、优化器与系统需求

### 6.33 模型家族：GLM

#### 6.33.1 GLM 与自回归空白填充预训练

#### 6.33.2 ChatGLM 与 GLM-4

#### 6.33.3 GLM-4.5／4.7 与 Agent 能力演进

#### 6.33.4 [GLM-5：公开模型报告与复杂任务工作负载][glm5]

#### 6.33.5 GLM 家族的文本、视觉与工具调用分支

### 6.34 其他模型家族与公开报告阅读

#### 6.34.1 Gemma、Phi 与小模型／数据效率

#### 6.34.2 OLMo 与开放训练配方

#### 6.34.3 DBRX、MiniMax 与其他 MoE 路线

#### 6.34.4 Stable Diffusion、FLUX、Wan 与视觉生成

#### 6.34.5 GPT、Claude、Gemini 的公开模型／系统报告

#### 6.34.6 Open-Weight、Open-Source 与闭源模型的证据边界

### 6.35 模型评测与系统协同取舍

#### 6.35.1 Perplexity、Accuracy 与任务质量

#### 6.35.2 MMLU、GSM8K／MATH、HumanEval 与 SWE-bench

#### 6.35.3 Long-Context、Multimodal 与 Agent Benchmark

#### 6.35.4 Pass@k、成功率与推理计算预算

#### 6.35.5 数据污染、Judge Bias 与评测可复现性

#### 6.35.6 质量—延迟—吞吐—内存—能耗—成本的 Pareto Frontier

#### 6.35.7 算法改进、Kernel 加速与端到端收益的区分

#### 6.35.8 模型结构变化与已部署硬件的可适配范围

<!-- Reading references: original references are retained; corpus references are pinned to the revision snapshot. -->

[awesome]: https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial "Awesome-ML-SYS-Tutorial"
[tpu]: https://arxiv.org/abs/1704.04760 "In-Datacenter Performance Analysis of a Tensor Processing Unit"
[roofline]: https://doi.org/10.1145/1498765.1498785 "Roofline: An Insightful Visual Performance Model for Multicore Architectures"
[cuda]: https://docs.nvidia.com/cuda/cuda-programming-guide/index.html "NVIDIA CUDA Programming Guide"
[cutlass]: https://docs.nvidia.com/cutlass/latest/overview.html "NVIDIA CUTLASS Documentation"
[triton]: https://triton-lang.org/main/getting-started/tutorials/index.html "Triton Tutorials"
[flashattention]: https://arxiv.org/abs/2205.14135 "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"
[mlir]: https://mlir.llvm.org/ "MLIR"
[openxla]: https://openxla.org/xla "OpenXLA"
[jax]: https://docs.jax.dev/en/latest/ "JAX Documentation"
[tvm]: https://tvm.apache.org/docs/ "Apache TVM Documentation"
[vllm]: https://docs.vllm.ai/en/latest/ "vLLM Documentation"
[sglang]: https://docs.sglang.io/ "SGLang Documentation"
[hicache]: https://docs.sglang.io/docs/advanced_features/hicache_design "HiCache System Design and Optimization"
[nsa]: https://arxiv.org/abs/2502.11089 "Native Sparse Attention"
[eagle3]: https://arxiv.org/abs/2503.01840 "EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test"
[dflash]: https://arxiv.org/abs/2602.06036 "DFlash: Block Diffusion for Flash Speculative Decoding"
[dspark]: https://arxiv.org/abs/2607.05147 "DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation"
[sbd]: https://arxiv.org/abs/2509.04185 "Set Block Decoding is a Language Model Inference Accelerator"
[deepseek4]: https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro "DeepSeek-V4-Pro Official Model Card"
[qwen35]: https://huggingface.co/Qwen/Qwen3.5-397B-A17B "Qwen3.5 Official Model Card"
[kimi25]: https://huggingface.co/moonshotai/Kimi-K2.5 "Kimi K2.5 Official Model Card"
[glm5]: https://huggingface.co/zai-org/GLM-5 "GLM-5 Official Model Card"
[llama4]: https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct "Llama 4 Official Model Card"
[corpus]: https://github.com/Yufeng98/AI-datacenter/tree/c49a55c6cdbbf387ca8d42bce1fac224f60d5588 "AI Datacenter Accelerator Research Corpus — source snapshot"
[survey-project]: https://yufeng98.github.io/ai-datacenter-survey/ "Survey companion project page"
[corpus-commit]: https://github.com/Yufeng98/AI-datacenter/commit/c49a55c6cdbbf387ca8d42bce1fac224f60d5588 "Corpus snapshot used for this revision"
[chip-nvidia-gpu]: https://github.com/Yufeng98/AI-datacenter/blob/c49a55c6cdbbf387ca8d42bce1fac224f60d5588/chips/nvidia-gpu/summary.md
[chip-amd-gpu]: https://github.com/Yufeng98/AI-datacenter/blob/c49a55c6cdbbf387ca8d42bce1fac224f60d5588/chips/amd-gpu/summary.md
[chip-biren]: https://github.com/Yufeng98/AI-datacenter/blob/c49a55c6cdbbf387ca8d42bce1fac224f60d5588/chips/biren/summary.md
[chip-hygon-dcu]: https://github.com/Yufeng98/AI-datacenter/blob/c49a55c6cdbbf387ca8d42bce1fac224f60d5588/chips/hygon-dcu/summary.md
[chip-muxi]: https://github.com/Yufeng98/AI-datacenter/blob/c49a55c6cdbbf387ca8d42bce1fac224f60d5588/chips/muxi/summary.md
[chip-mthreads]: https://github.com/Yufeng98/AI-datacenter/blob/c49a55c6cdbbf387ca8d42bce1fac224f60d5588/chips/mthreads/summary.md
[chip-tianshu-zhixin]: https://github.com/Yufeng98/AI-datacenter/blob/c49a55c6cdbbf387ca8d42bce1fac224f60d5588/chips/tianshu-zhixin/summary.md
[chip-xiwang]: https://github.com/Yufeng98/AI-datacenter/blob/c49a55c6cdbbf387ca8d42bce1fac224f60d5588/chips/xiwang/summary.md
[chip-google-tpu]: https://github.com/Yufeng98/AI-datacenter/blob/c49a55c6cdbbf387ca8d42bce1fac224f60d5588/chips/google-tpu/summary.md
[chip-aws-neuron]: https://github.com/Yufeng98/AI-datacenter/blob/c49a55c6cdbbf387ca8d42bce1fac224f60d5588/chips/aws-neuron/summary.md
[chip-huawei-ascend]: https://github.com/Yufeng98/AI-datacenter/blob/c49a55c6cdbbf387ca8d42bce1fac224f60d5588/chips/huawei-ascend/summary.md
[chip-intel-gaudi]: https://github.com/Yufeng98/AI-datacenter/blob/c49a55c6cdbbf387ca8d42bce1fac224f60d5588/chips/intel-gaudi/summary.md
[chip-microsoft-maia]: https://github.com/Yufeng98/AI-datacenter/blob/c49a55c6cdbbf387ca8d42bce1fac224f60d5588/chips/microsoft-maia/summary.md
[chip-qualcomm]: https://github.com/Yufeng98/AI-datacenter/blob/c49a55c6cdbbf387ca8d42bce1fac224f60d5588/chips/qualcomm/summary.md
[chip-cambricon]: https://github.com/Yufeng98/AI-datacenter/blob/c49a55c6cdbbf387ca8d42bce1fac224f60d5588/chips/cambricon/summary.md
[chip-alibaba-t-head]: https://github.com/Yufeng98/AI-datacenter/blob/c49a55c6cdbbf387ca8d42bce1fac224f60d5588/chips/alibaba-t-head/summary.md
[chip-kunlunxin]: https://github.com/Yufeng98/AI-datacenter/blob/c49a55c6cdbbf387ca8d42bce1fac224f60d5588/chips/kunlunxin/summary.md
[chip-furiosa]: https://github.com/Yufeng98/AI-datacenter/blob/c49a55c6cdbbf387ca8d42bce1fac224f60d5588/chips/furiosa/summary.md
[chip-sophgo]: https://github.com/Yufeng98/AI-datacenter/blob/c49a55c6cdbbf387ca8d42bce1fac224f60d5588/chips/sophgo/summary.md
[chip-vastaitech]: https://github.com/Yufeng98/AI-datacenter/blob/c49a55c6cdbbf387ca8d42bce1fac224f60d5588/chips/vastaitech/summary.md
[chip-tecorigin]: https://github.com/Yufeng98/AI-datacenter/blob/c49a55c6cdbbf387ca8d42bce1fac224f60d5588/chips/tecorigin/summary.md
[chip-stream-computing]: https://github.com/Yufeng98/AI-datacenter/blob/c49a55c6cdbbf387ca8d42bce1fac224f60d5588/chips/stream-computing/summary.md
[chip-tesla-fsd]: https://github.com/Yufeng98/AI-datacenter/blob/c49a55c6cdbbf387ca8d42bce1fac224f60d5588/chips/tesla-fsd/summary.md
[chip-openai-broadcom]: https://github.com/Yufeng98/AI-datacenter/blob/c49a55c6cdbbf387ca8d42bce1fac224f60d5588/chips/openai-broadcom/summary.md
[chip-tenstorrent]: https://github.com/Yufeng98/AI-datacenter/blob/c49a55c6cdbbf387ca8d42bce1fac224f60d5588/chips/tenstorrent/summary.md
[chip-meta-mtia]: https://github.com/Yufeng98/AI-datacenter/blob/c49a55c6cdbbf387ca8d42bce1fac224f60d5588/chips/meta-mtia/summary.md
[chip-graphcore]: https://github.com/Yufeng98/AI-datacenter/blob/c49a55c6cdbbf387ca8d42bce1fac224f60d5588/chips/graphcore/summary.md
[chip-tesla-dojo]: https://github.com/Yufeng98/AI-datacenter/blob/c49a55c6cdbbf387ca8d42bce1fac224f60d5588/chips/tesla-dojo/summary.md
[chip-cerebras]: https://github.com/Yufeng98/AI-datacenter/blob/c49a55c6cdbbf387ca8d42bce1fac224f60d5588/chips/cerebras/summary.md
[chip-ibm-spyre]: https://github.com/Yufeng98/AI-datacenter/blob/c49a55c6cdbbf387ca8d42bce1fac224f60d5588/chips/ibm-spyre/summary.md
[chip-enflame]: https://github.com/Yufeng98/AI-datacenter/blob/c49a55c6cdbbf387ca8d42bce1fac224f60d5588/chips/enflame/summary.md
[chip-preferred-networks-mn-core]: https://github.com/Yufeng98/AI-datacenter/blob/c49a55c6cdbbf387ca8d42bce1fac224f60d5588/chips/preferred-networks-mn-core/summary.md
[chip-esperanto]: https://github.com/Yufeng98/AI-datacenter/blob/c49a55c6cdbbf387ca8d42bce1fac224f60d5588/chips/esperanto/summary.md
[chip-pezy]: https://github.com/Yufeng98/AI-datacenter/blob/c49a55c6cdbbf387ca8d42bce1fac224f60d5588/chips/pezy/summary.md
[chip-sambanova]: https://github.com/Yufeng98/AI-datacenter/blob/c49a55c6cdbbf387ca8d42bce1fac224f60d5588/chips/sambanova/summary.md
[chip-rebellions-atom]: https://github.com/Yufeng98/AI-datacenter/blob/c49a55c6cdbbf387ca8d42bce1fac224f60d5588/chips/rebellions-atom/summary.md
[chip-tsingmicro]: https://github.com/Yufeng98/AI-datacenter/blob/c49a55c6cdbbf387ca8d42bce1fac224f60d5588/chips/tsingmicro/summary.md
[chip-nextsilicon-maverick]: https://github.com/Yufeng98/AI-datacenter/blob/c49a55c6cdbbf387ca8d42bce1fac224f60d5588/chips/nextsilicon-maverick/summary.md
[chip-groq]: https://github.com/Yufeng98/AI-datacenter/blob/c49a55c6cdbbf387ca8d42bce1fac224f60d5588/chips/groq/summary.md
[chip-etched-sohu]: https://github.com/Yufeng98/AI-datacenter/blob/c49a55c6cdbbf387ca8d42bce1fac224f60d5588/chips/etched-sohu/summary.md
[chip-d-matrix]: https://github.com/Yufeng98/AI-datacenter/blob/c49a55c6cdbbf387ca8d42bce1fac224f60d5588/chips/d-matrix/summary.md
[chip-sk-hynix-aim]: https://github.com/Yufeng98/AI-datacenter/blob/c49a55c6cdbbf387ca8d42bce1fac224f60d5588/chips/sk-hynix-aim/summary.md
[chip-samsung-aquabolt-pim]: https://github.com/Yufeng98/AI-datacenter/blob/c49a55c6cdbbf387ca8d42bce1fac224f60d5588/chips/samsung-aquabolt-pim/summary.md
[chip-mythic]: https://github.com/Yufeng98/AI-datacenter/blob/c49a55c6cdbbf387ca8d42bce1fac224f60d5588/chips/mythic/summary.md
[chip-untether-ai]: https://github.com/Yufeng98/AI-datacenter/blob/c49a55c6cdbbf387ca8d42bce1fac224f60d5588/chips/untether-ai/summary.md
[chip-rain-ai]: https://github.com/Yufeng98/AI-datacenter/blob/c49a55c6cdbbf387ca8d42bce1fac224f60d5588/chips/rain-ai/summary.md
[chip-lightmatter]: https://github.com/Yufeng98/AI-datacenter/blob/c49a55c6cdbbf387ca8d42bce1fac224f60d5588/chips/lightmatter/summary.md
[chip-q-ant]: https://github.com/Yufeng98/AI-datacenter/blob/c49a55c6cdbbf387ca8d42bce1fac224f60d5588/chips/q-ant/summary.md
[chip-luminous]: https://github.com/Yufeng98/AI-datacenter/blob/c49a55c6cdbbf387ca8d42bce1fac224f60d5588/chips/luminous/summary.md
[chip-spinncloud-spinnaker2]: https://github.com/Yufeng98/AI-datacenter/blob/c49a55c6cdbbf387ca8d42bce1fac224f60d5588/chips/spinncloud-spinnaker2/summary.md
[layers-nvidia-gpu]: https://github.com/Yufeng98/AI-datacenter/blob/c49a55c6cdbbf387ca8d42bce1fac224f60d5588/chips/nvidia-gpu/layer-table.md
[layers-amd-gpu]: https://github.com/Yufeng98/AI-datacenter/blob/c49a55c6cdbbf387ca8d42bce1fac224f60d5588/chips/amd-gpu/layer-table.md
[layers-google-tpu]: https://github.com/Yufeng98/AI-datacenter/blob/c49a55c6cdbbf387ca8d42bce1fac224f60d5588/chips/google-tpu/layer-table.md
[layers-aws-neuron]: https://github.com/Yufeng98/AI-datacenter/blob/c49a55c6cdbbf387ca8d42bce1fac224f60d5588/chips/aws-neuron/layer-table.md
[layers-huawei-ascend]: https://github.com/Yufeng98/AI-datacenter/blob/c49a55c6cdbbf387ca8d42bce1fac224f60d5588/chips/huawei-ascend/layer-table.md
[layers-tenstorrent]: https://github.com/Yufeng98/AI-datacenter/blob/c49a55c6cdbbf387ca8d42bce1fac224f60d5588/chips/tenstorrent/layer-table.md
[layers-graphcore]: https://github.com/Yufeng98/AI-datacenter/blob/c49a55c6cdbbf387ca8d42bce1fac224f60d5588/chips/graphcore/layer-table.md
[layers-cerebras]: https://github.com/Yufeng98/AI-datacenter/blob/c49a55c6cdbbf387ca8d42bce1fac224f60d5588/chips/cerebras/layer-table.md
[layers-groq]: https://github.com/Yufeng98/AI-datacenter/blob/c49a55c6cdbbf387ca8d42bce1fac224f60d5588/chips/groq/layer-table.md
[layers-sambanova]: https://github.com/Yufeng98/AI-datacenter/blob/c49a55c6cdbbf387ca8d42bce1fac224f60d5588/chips/sambanova/layer-table.md
[layers-d-matrix]: https://github.com/Yufeng98/AI-datacenter/blob/c49a55c6cdbbf387ca8d42bce1fac224f60d5588/chips/d-matrix/layer-table.md
