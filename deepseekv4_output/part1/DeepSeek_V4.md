# DeepSeek-V4:

# Towards Highly Efficient Million-Token Context Intelligence

DeepSeek-AI research@deepseek.com

## Abstract

We present a preview version of DeepSeek-V4 series, including two strong Mixture-of-Experts (MoE) language models — DeepSeek-V4-Pro with 1.6T parameters (49B activated) and DeepSeek-V4-Flash with 284B parameters (13B activated) — both supporting a context length of one million tokens. DeepSeek-V4 series incorporate several key upgrades in architecture and optimization: (1) a hybrid attention architecture that combines Compressed Sparse Attention (CSA) and Heavily Compressed Attention (HCA) to improve long-context efficiency; (2) Manifold-Constrained Hyper-Connections (mHC) that enhance conventional residual connections; (3) and the Muon optimizer for faster convergence and greater training stability. We pre-train both models on more than 32T diverse and high-quality tokens, followed by a comprehensive post-training pipeline that unlocks and further enhances their capabilities. DeepSeek-V4-Pro-Max, the maximum reasoning effort mode of DeepSeek-V4-Pro, redefines the state-of-the-art for open models, outperforming its predecessors in core tasks. Meanwhile, DeepSeek-V4 series are highly efficient in long-context scenarios. In the one-million-token context setting, DeepSeek-V4-Pro requires only 27% of single-token inference FLOPs and 10% of KV cache compared with DeepSeek-V3.2. This enables us to routinely support one-million-token contexts, thereby making long-horizon tasks and further test-time scaling more feasible. The model checkpoints are available at https://huggingface.co/collections/deepseek-ai/deepseek-v4.

<!-- image-->

<!-- image-->

<!-- image-->  
Figure 1 | Left: benchmark performance of DeepSeek-V4-Pro-Max and its counterparts. Right: inference FLOPs and KV cache size of DeepSeek-V4 series and DeepSeek-V3.2.

## Contents

1 Introduction 4   
2 Architecture 6   
2.1 Designs Inherited from DeepSeek-V3 . 7   
2.2 Manifold-Constrained Hyper-Connections 7   
2.3 Hybrid Attention with CSA and HCA 9   
2.3.1 Compressed Sparse Attention . 9   
2.3.2 Heavily Compressed Attention . 11   
2.3.3 Other Details 12   
2.3.4 Efficiency Discussion . 13   
2.4 Muon Optimizer 14   
3 General Infrastructures 15   
3.1 Fine-Grained Communication-Computation Overlap in Expert Parallelism . . . 15   
3.2 Flexible and Efficient Kernel Development with TileLang . 16   
3.3 High-Performance Batch-Invariant and Deterministic Kernel Libraries 18   
3.4 Training Framework 19   
3.4.1 Efficient Implementation of Muon 19   
3.4.2 Cost-Effective and Memory-Efficient Implementation of mHC 20   
3.4.3 Contextual Parallelism for Long-Context Attention 20   
3.4.4 Extended Automatic Differentiation for Flexible Activation Checkpointing 21   
3.5 Inference Framework . 21   
3.5.1 KV Cache Structure and Management 21   
3.5.2 On-Disk KV Cache Storage 23   
4 Pre-Training 24   
4.1 Data Construction . . 24   
4.2 Pre-Training Setups . 24   
4.2.1 Model Setups 24   
4.2.2 Training Setups 25   
4.2.3 Mitigating Training Instability 26   
4.3 Evaluations 27   
4.3.1 Evaluation Benchmarks 27   
4.3.2 Evaluation Results 27   
5 Post-Training 28   
5.1 Post-Training Pipeline 28   
5.1.1 Specialist Training 28   
5.1.2 On-Policy Distillation 32   
5.2 Post-Training Infrastructures 33   
5.2.1 FP4 Quantization-Aware Training 33   
5.2.2 Efficient Teacher Scheduling for Full-Vocabulary OPD 34   
5.2.3 Preemptible and Fault-Tolerant Rollout Service 34   
5.2.4 Scaling RL Framework for Million-Token Context 35   
5.2.5 Sandbox Infrastructure for Agentic AI . 35   
5.3 Standard Benchmark Evaluation 36   
5.3.1 Evaluation Setup . 36   
5.3.2 Evaluation Results 37   
5.4 Performance on Real-World Tasks 41   
5.4.1 Chinese Writing . 41   
5.4.2 Search 42   
5.4.3 White-Collar Task . 42   
5.4.4 Code Agent . 44   
6 Conclusion, Limitations, and Future Directions 44   
A Author List and Acknowledgment 54   
A.1 Author List . 54   
A.2 Acknowledgment . . 55   
B Evaluation Details 5 5

## 1. Introduction

The emergence of reasoning models (DeepSeek-AI, 2025; OpenAI, 2024c) has established a new paradigm of test-time scaling, driving substantial performance gains for Large Language Models (LLMs). However, this scaling paradigm is fundamentally constrained by the quadratic computational complexity of the vanilla attention mechanism (Vaswani et al., 2017), which creates a prohibitive bottleneck for ultra-long contexts and reasoning processes. Concurrently, the emergence of long-horizon scenarios and tasks — from complex agentic workflows to massive cross-document analysis — has also made efficient support for ultra-long contexts critical for future progress. While recent open-source efforts (Bai et al., 2025a; DeepSeek-AI, 2024; MiniMax, 2025; Qwen, 2025) have advanced general capabilities, this core architectural inefficiency in handling ultra-long sequences remains a key impediment, limiting further gains from test-time scaling and hindering further exploration into long-horizon scenarios and tasks.

In order to break the efficiency barrier in ultra-long contexts, we develop the DeepSeek-V4 series, including the preview versions of DeepSeek-V4-Pro with 1.6T parameters (49B activated) and DeepSeek-V4-Flash with 284B parameters (13B activated). Through architectural innovations, DeepSeek-V4 series achieve a dramatic leap in computational efficiency for processing ultra-long sequences. This breakthrough enables efficient support for a context length of one million tokens, ushering in a new era of million-length contexts for next-generation LLMs. We believe our capability to efficiently handle ultra-long sequences unlocks the next frontier of test-time scaling, paves the way for deeper research into long-horizon tasks, and establishes a necessary foundation for exploring future paradigms like online learning.

Compared with the DeepSeek-V3 architecture (DeepSeek-AI, 2024), DeepSeek-V4 series retain the DeepSeekMoE framework (Dai et al., 2024) and Multi-Token Prediction (MTP) strategy, while introducing several key innovations in architecture and optimization. To enhance longcontext efficiency, we design a hybrid attention mechanism combining Compressed Sparse Attention (CSA) and Heavily Compressed Attention (HCA). CSA compresses the KV caches along the sequence dimension and then performs DeepSeek Sparse Attention (DSA) (DeepSeek-AI, 2025), whereas HCA applies more aggressive compression to the KV caches but keeps dense attention. To strengthen modeling capability, we incorporate Manifold-Constrained Hyper-Connections (mHC) (Xie et al., 2026) that upgrade conventional residual connections. Additionally, we introduce the Muon (Jordan et al., 2024; Liu et al., 2025) optimizer to the training of DeepSeek-V4 series, leading to faster convergence and improved training stability.

To enable efficient training and inference for DeepSeek-V4 series as well as productive development, we introduce several infrastructure optimizations. First, we design and implement a single fused kernel for MoE modules that fully overlaps computation, communication, and memory access. Second, we employ TileLang (Wang et al., 2026), a Domain-Specific Language (DSL) to balance development productivity and runtime efficiency. Third, we provide efficient batch-invariant and deterministic kernel libraries to ensure bitwise reproducibility across training and inference. Fourth, for the training framework, we extend the autograd framework with tensor-level checkpointing for fine-grained recomputation control; and we enhance training efficiency with a hybrid ZeRO strategy for the Muon optimizer, cost-effective mHC implementations via recomputation and fused kernels, and two-stage contextual parallelism to manage compressed attention. Fifth, for the inference framework, we design a heterogeneous KV cache structure with on-disk storage strategies to enable efficient shared-prefix reuse. In addition, during the post-training stage, we incorporate FP4 quantization-aware training for MoE expert weights and the indexer QK path to reduce memory and computation.

By employing hybrid CSA and HCA, along with precision optimizations on computation and storage, DeepSeek-V4 series achieve significantly lower inference FLOPs and a substantially reduced KV cache size compared with DeepSeek-V3.2, especially in long-context settings. The right part of Figure 1 demonstrates the estimated single-token inference FLOPs and accumulated KV cache size of DeepSeek-V3.2 and DeepSeek-V4 series. In the scenario of 1M-token context, even DeepSeek-V4-Pro, which has a larger number of activated parameters, attains only 27% of the single-token FLOPs (measured in equivalent FP8 FLOPs) and 10% of the KV cache size relative to DeepSeek-V3.2. Furthermore, DeepSeek-V4-Flash, with its smaller number of activated parameters, pushes efficiency even further: in the 1M-token context setting, it achieves only 10% of the single-token FLOPs and 7% of the KV cache size compared with DeepSeek-V3.2. Additionally, for DeepSeek-V4 series, the routed expert parameters utilize FP4 precision. While the peak FLOPs for FP4 × FP8 operations are currently the same as FP8 × FP8 on existing hardware, they can theoretically be implemented to be 1/3 more efficient on future hardware, which will further enhance the efficiency of DeepSeek-V4 series.

During pre-training, we train DeepSeek-V4-Flash on 32T tokens and DeepSeek-V4-Pro on 33T tokens, respectively. After pre-training, these two models can natively and efficiently support 1M-length contexts. In our internal evaluations, DeepSeek-V4-Flash-Base already surpasses DeepSeek-V3.2-Base across a majority of benchmarks with its more parameter-efficient design. DeepSeek-V4-Pro-Base further extends this advantage to set a new performance standard among DeepSeek foundation models, achieving comprehensive superiority across reasoning, coding, long-context, and world knowledge tasks.

The post-training pipeline of DeepSeek-V4 series features a two-stage paradigm: the independent cultivation of domain-specific experts, followed by unified model consolidation via on-policy distillation (Gu et al., 2024; Lu and Lab, 2025). Initially, for each target domain — such as mathematics, coding, agent, and instruction following — a separate expert model is trained independently. The base model first undergoes Supervised Fine-Tuning (SFT) on high-quality, domain-specific data to establish foundational capabilities. Subsequently, Reinforcement Learning (RL) is applied using Group Relative Policy Optimization (GRPO) (DeepSeek-AI, 2025), which further optimizes the model for domain-aligned behaviors guided by reward models tailored to specific success criteria. This phase yields a diverse set of specialized experts, each excelling in its respective field. Finally, to integrate these distinct proficiencies, a single unified model is trained through on-policy distillation, wherein the unified model acts as the student learning to optimize the reverse KL loss with teacher models.

## Summary of Core Evaluation Results

• Knowledge: In assessments of broad world knowledge, DeepSeek-V4-Pro-Max, the maximum reasoning effort mode of DeepSeek-V4-Pro, significantly outperforms leading opensource models on the SimpleQA (OpenAI, 2024d) and Chinese-SimpleQA (He et al., 2024) benchmarks. Regarding educational knowledge — evaluated via MMLU-Pro (Wang et al., 2024b), HLE (Phan et al., 2025), and GPQA (Rein et al., 2023) — DeepSeek-V4-Pro-Max shows a marginal lead over its open-source counterparts. DeepSeek-V4-Pro-Max has significantly closed the gap with the leading proprietary model, Gemini-3.1-Pro, despite still trailing it in these knowledge-based evaluations.

• Reasoning: Through the expansion of reasoning tokens, DeepSeek-V4-Pro-Max demonstrates superior performance relative to GPT-5.2 and Gemini-3.0-Pro on standard reasoning benchmarks. Nevertheless, its performance falls marginally short of GPT-5.4 and Gemini-3.1-Pro, suggesting a developmental trajectory that trails state-of-the-art frontier models by approximately 3 to 6 months. Furthermore, DeepSeek-V4-Flash-Max achieves comparable performance to GPT-5.2 and Gemini-3.0-Pro, establishing itself as a highly cost-effective architecture for complex reasoning tasks.

<!-- image-->  
Figure 2 | Overall architecture of DeepSeek-V4 series. We use hybrid CSA (Compressed Sparse Attention) and HCA (Heavily Compressed Attention) for attention layers, DeepSeekMoE for feed-forward layers, and strengthen conventional residual connections with mHC.

• Agent: On public benchmarks, DeepSeek-V4-Pro-Max is on par with leading open-source models, such as Kimi-K2.6 and GLM-5.1, but slightly worse than frontier closed models. In our internal evaluation, DeepSeek-V4-Pro-Max outperforms Claude Sonnet 4.5 and approaches the level of Opus 4.5.

• Long-Context: DeepSeek-V4-Pro-Max delivers strong results on synthetic and real use cases with a 1-million-token context window, surpassing even Gemini-3.1-Pro on academic benchmarks.

• DeepSeek-V4-Pro v.s. DeepSeek-V4-Flash: DeepSeek-V4-Flash-Max exhibits lower performance in knowledge evaluations due to its smaller parameter scale. However, it achieves comparable results on reasoning tasks when allocated a larger thinking budget. In agent evaluations, while DeepSeek-V4-Flash-Max matches the performance of DeepSeek-V4-Pro-Max on several benchmarks, it still trails its larger counterpart on more complex, high-difficulty tasks.

## 2. Architecture

Overall, DeepSeek-V4 series retain the Transformer (Vaswani et al., 2017) architecture and Multi-Token Prediction (MTP) modules (DeepSeek-AI, 2024; Gloeckle et al., 2024), while introducing several key upgrades over DeepSeek-V3: (1) firstly, we introduce the Manifold-Constrained Hyper-Connections (mHC) (Xie et al., 2026) to strengthen conventional residual connections;

(2) secondly, we design a hybrid attention architecture, which greatly improves long-context efficiency through Compressed Sparse Attention and Heavily Compressed Attention. (3) thirdly, we employ Muon (Jordan et al., 2024; Liu et al., 2025) as the optimizer. For the Mixture-of-Experts (MoE) components, we still adopt the DeepSeekMoE (Dai et al., 2024) architecture, with only minor adjustments from DeepSeek-V3. The Multi-Token Prediction (MTP) (DeepSeek-AI, 2024; Gloeckle et al., 2024; Li et al., 2024; Qi et al., 2020) configuration remains identical to that of DeepSeek-V3. All other unspecified details follow the settings established in DeepSeek-V3 (DeepSeek-AI, 2024). Figure 2 illustrates the overall architecture of DeepSeek-V4, and the details are described below.

## 2.1. Designs Inherited from DeepSeek-V3

Mixture-of-Experts. As previous DeepSeek-series models (DeepSeek-AI, 2024; DeepSeek-AI, 2024), DeepSeek-V4 series also adopt the DeepSeekMoE paradigm (Dai et al., 2024) for Feed-Forward Networks (FFNs), which sets fine-grained routed experts and shared experts. Different from DeepSeek-V3, we change the activation function that computes the affinity scores from Sigmoid(·) into Sqrt(Softplus(·)). For load balancing, we also employ the auxiliary-loss-free strategy (DeepSeek-AI, 2024; Wang et al., 2024a), augmented by a slight sequence-wise balance loss that prevents extreme imbalance within individual sequences. For DeepSeek-V4, we remove the constraint on the number of routing target nodes, and carefully redesign the parallelism strategy to maintain training efficiency. Furthermore, compared with DeepSeek-V3, we replace the dense FFN layers in the initial several Transformer blocks with MoE layers that employ Hash routing (Roller et al., 2021). The Hash routing strategy determines the target experts of each token according to a predefined hash function with regard to the input token ID.

Multi-Token Prediction. As DeepSeek-V3, DeepSeek-V4 series also set MTP modules and objectives. Given that the MTP strategy has been validated in DeepSeek-V3, we adopt the same strategy for DeepSeek-V4 series without modification.

## 2.2. Manifold-Constrained Hyper-Connections

As shown in Figure 2, DeepSeek-V4 series incorporate Manifold-Constrained Hyper-Connections (mHC) (Xie et al., 2026) to strengthen the conventional residual connections between adjacent Transformer blocks. Compared with naive Hyper-Connections (HC) (Zhu et al., 2025), the core idea of mHC is to constrain the residual mapping onto a specific manifold, and thus enhance the stability of signal propagation across layers while preserving model expressivity. This subsection briefly introduces the standard HC and describes how we design mHC for stable training.

Standard Hyper-Connections. The standard HC expands the width of the residual stream by a factor of $n _ { \mathrm { h c } } .$ . Specifically, the shape of the residual stream is expanded from $\mathbb { R } ^ { d }$ to $\mathbb { R } ^ { n _ { \mathrm { h c } } \times d }$ where ?? is the hidden size of the actual layer input. Let $X _ { l } = [ \mathbf { x } _ { l , 1 } ; \ldots ; \mathbf { x } _ { l , n _ { \mathrm { h c } } } ] ^ { T } \in \mathbb { R } ^ { n _ { \mathrm { h c } } \times d }$ be the residual state before the ??-th layer. HC introduces three linear mappings: an input mapping $A _ { l } \in \mathbb { R } ^ { 1 \times n _ { \mathrm { h c } } }$ , a residual transformation $B _ { l } \in \mathbb { R } ^ { n _ { \mathrm { h c } } \times n _ { \mathrm { h c } } }$ , and an output mapping $\bar { C _ { l } } \bar { \in } \mathbb { R } ^ { n _ { \mathrm { h c } } \times \hat { 1 } }$ . The update of the residual state is then formulated as:

$$
X _ { l + 1 } = B _ { l } X _ { l } + C _ { l } { \mathcal { F } } _ { l } ( A _ { l } X _ { l } ) ,\tag{1}
$$

where $\mathcal { F } _ { l }$ denotes the ??-th layer $( \mathrm { e . g . } ,$ an MoE layer), whose input and output shapes are both $\mathbb { R } ^ { d }$ . Note that the actual layer input $A _ { l } X _ { l } \in \mathbb { R } ^ { d }$ is also ??-dimensional, so the expanded residual width does not influence the design of the inner layers. HC decouples the residual width from the actual hidden size, offering a complementary scaling axis with minimal computational overhead, as $n _ { \mathrm { h c } }$ is typically much smaller than the hidden size ??. However, even though HC has demonstrated potential in improving model performance, we find that the training will frequently exhibit numerical instability when stacking multiple layers, which hinders the scaling of HC.

Manifold-Constrained Residual Mapping. The core innovation of mHC is to constrain the residual mapping matrix $B _ { l }$ to the manifold of doubly stochastic matrices (the Birkhoff polytope) M, and thus enhance the stability of signal propagation across layers:

$$
B _ { l } \in { \mathcal { M } } : = \{ M \in \mathbb { R } ^ { n \times n } ~ | ~ M { \bf 1 } _ { n } = { \bf 1 } _ { n } , ~ { \bf 1 } _ { n } ^ { T } M = { \bf 1 } _ { n } ^ { T } , ~ M \geqslant 0 \} .\tag{2}
$$

This constraint ensures that the spectral norm of the mapping matrix $\| B _ { l } \| _ { 2 }$ is bounded by 1, so the residual transformation is non-expansive, which increases the numerical stability during both the forward pass and backpropagation. Besides, the set M is closed under multiplication, which guarantees stability in the scenarios of deep stacks of mHC. In addition, the input transformation $A _ { l }$ and output transformation $C _ { l }$ are also constrained to be non-negative and bounded via a Sigmoid function to avoid the risk of signal cancellation.

Dynamic Parameterization. The parameters of three linear mappings are dynamically generated, which are decomposed into a dynamic (input-dependent) component and a static (input-independent) component. Given the input $\bar { X _ { l } } \in \mathbb { R } ^ { n _ { \mathrm { h c } } \times d }$ , it is first flattened and normalized: $\hat { X } _ { l } = \mathrm { R M S N o r m } ( \mathrm { v e c } ( \bar { X } _ { l } ) ) \in \mathbb { R } ^ { 1 \times n _ { \mathrm { h c } } d }$ . Then, we follow the conventional HC to generate the unconstrained raw parameters $\tilde { A } _ { l } \in \mathbb { R } ^ { 1 \times n _ { \mathrm { h c } } } , \tilde { B } _ { l } \in \mathbb { R } ^ { n _ { \mathrm { h c } } \times n _ { \mathrm { h c } } }$ , and $\tilde { C } _ { l } \in \mathbb { R } ^ { n _ { \mathrm { h c } } \times 1 }$ :

$$
\tilde { A } _ { l } = \alpha _ { l } ^ { \mathrm { p r e } } \cdot ( \hat { X } _ { l } W _ { l } ^ { \mathrm { p r e } } ) + S _ { l } ^ { \mathrm { p r e } } ,\tag{3}
$$

$$
\tilde { B } _ { l } = \alpha _ { l } ^ { \mathrm { r e s } } \cdot \mathrm { M a t } ( \hat { X } _ { l } W _ { l } ^ { \mathrm { r e s } } ) + S _ { l } ^ { \mathrm { r e s } } ,\tag{4}
$$

$$
\tilde { C } _ { l } = \alpha _ { l } ^ { \mathrm { p o s t } } \cdot ( \hat { X } _ { l } W _ { l } ^ { \mathrm { p o s t } } ) ^ { T } + S _ { l } ^ { \mathrm { p o s t } } ,\tag{5}
$$

where $W _ { \jmath } ^ { \mathrm { p r e } } , W _ { \jmath } ^ { \mathrm { p o s t } } \in \mathbb { R } ^ { n _ { \mathrm { h c } } d \times n _ { \mathrm { h c } } }$ and $W _ { l } ^ { \mathrm { r e s } } \in \mathbb { R } ^ { n _ { \mathrm { h c } } d \times n _ { \mathrm { h c } } ^ { 2 } }$ are learnable parameters for generating the dynamic components; Mat(·) reshapes a vector of size $1 \times n _ { \mathrm { h c } } ^ { 2 }$ into a matrix of size $n _ { \mathrm { h c } } \times n _ { \mathrm { h c } } ;$ $S _ { l } ^ { \mathrm { p r e } } \in \mathbb { R } ^ { 1 \times n _ { \mathrm { h c } } } , S _ { l } ^ { \mathrm { p o s t } } \in \mathbb { R } ^ { n _ { \mathrm { h c } } \times 1 }$ , and $S _ { \boldsymbol { I } } ^ { \mathrm { r e s } } \in \mathbb { R } ^ { n _ { \mathrm { h c } } \times n _ { \mathrm { h c } } }$ are learnable static biases; and $\alpha _ { l } ^ { \mathrm { p r e } } , \alpha _ { l } ^ { \mathrm { r e s } } , \alpha _ { l } ^ { \mathrm { p o s t } } \in \mathbb { R }$ are learnable gating factors initialized to small values.

Applying Parameter Constraints. After obtaining the unconstrained raw parameters $\tilde { A } _ { l } , \tilde { B } _ { l } , \tilde { C } _ { l } ,$ we then apply constraints described earlier to them to enhance the numerical stability. To be specific, for the input and output mappings, we employ a Sigmoid function ??(·) to ensure their non-negativity and boundedness:

$$
A _ { l } = \sigma ( \tilde { A } _ { l } ) ,\tag{6}
$$

$$
C _ { l } = 2 \sigma ( \tilde { C } _ { l } ) .\tag{7}
$$

As for the residual mapping $\tilde { B } _ { l } ,$ , we project it onto the manifold of doubly stochastic matrices M. This is achieved by the Sinkhorn-Knopp algorithm, which first applies an exponential function to $\tilde { B } _ { l }$ to ensure positivity, getting $M ^ { ( 0 ) } = \exp ( \tilde { B } _ { l } )$ , and then iteratively performs column and row normalization:

$$
\boldsymbol { M } ^ { ( t ) } = \mathcal { T } ( \mathcal { T } _ { c } ( \boldsymbol { M } ^ { ( t - 1 ) } ) ) ,\tag{8}
$$

where $\mathcal { T } _ { r }$ and $\mathcal { T } _ { c }$ denote row and column normalization, respectively. This iteration converges to a constrained doubly stochastic matrix $B _ { l } = M ^ { ( t _ { \operatorname* { m a x } } ) }$ . We choose $t _ { \mathrm { m a x } } = 2 0$ as a practical value.

<!-- image-->  
Figure 3 | Core architectures of CSA. It compresses the number of KV entries to $\textstyle { \frac { 1 } { m } }$ times, and then applies DeepSeek Sparse Attention for further acceleration. Additionally, a small set of sliding window KV entries is combined with the selected compressed KV entries to enhance local fine-grained dependencies.

## 2.3. Hybrid Attention with CSA and HCA

As the context length reaches extreme scales, the attention mechanism emerges as the dominant computational bottleneck in a model. For DeepSeek-V4, we design two efficient attention architectures — Compressed Sparse Attention (CSA) and Heavily Compressed Attention (HCA) — and employ their interleaved hybrid configuration, which substantially reduces the computational cost of attention in long-text scenarios. CSA integrates both compression and sparse attention strategies: it first compresses the Key-Value (KV) cache of every ?? tokens into one entry, and then applies DeepSeek Sparse Attention (DSA) (DeepSeek-AI, 2025) where each query token attends to only ?? compressed KV entries. HCA aims for extreme compression by consolidating the KV cache of every ??′ (≫ ??) tokens into a single entry. The hybrid architecture of CSA and HCA remarkably improves the long-context efficiency of DeepSeek-V4 series, making one-million-token context feasible in practice. This subsection describes the core techniques of our hybrid attention architecture, and we also provide an open-source implementation1 to specify more details unambiguously.

## 2.3.1. Compressed Sparse Attention

The core architecture of CSA is illustrated in Figure 3, which first compresses the KV cache of each ?? tokens into one entry, and then applies DeepSeek Sparse Attention for further acceleration.

Compressed Key-Value Entries. Let $H \in \mathbb { R } ^ { n \times d }$ be a sequence of input hidden states, where ?? is the sequence length and ?? is the hidden size. CSA first computes two series of KV entries $C ^ { a } , C ^ { b } \in \mathbb { R } ^ { \bar { n } \times c }$ and their corresponding compression weights $Z ^ { a } , { \bar { Z } } ^ { b } \in \mathbb { R } ^ { n \times c }$ , where ?? is the head

dimension:

$$
C ^ { a } = H \cdot W ^ { a K V } , \quad C ^ { b } = H \cdot W ^ { b K V } ,\tag{9}
$$

$$
Z ^ { a } = H \cdot W ^ { a Z } , ~ Z ^ { b } = H \cdot W ^ { b Z } ,\tag{10}
$$

where $W ^ { a K V } , W ^ { b K V } , W ^ { a Z } , W ^ { b Z } \in \mathbb { R } ^ { d \times c }$ are trainable parameters. Next, each ?? KV entries in $C ^ { a }$ and $C ^ { b }$ will be compressed into one entry according to their compression weights and learnable positional biases $B ^ { a } , B ^ { b } \in \mathbb { R } ^ { m \times c }$ , producing $C ^ { \mathsf { C o m p } } \in \mathbb { R } ^ { \frac { n } { m } \times c }$ . Each compressed entry $C _ { i } ^ { \mathrm { C o m p } } \in \mathbb { R } ^ { c }$ is computed by

$$
\begin{array} { r } { [ S _ { m i : m ( i + 1 ) - 1 } ^ { a } ; S _ { m ( i - 1 ) : m i - 1 } ^ { b } ] = \mathrm { { S o f t m a x } _ { r o w } } ( [ Z _ { m i : m ( i + 1 ) - 1 } ^ { a } + B ^ { a } ; Z _ { m ( i - 1 ) : m i - 1 } ^ { b } + B ^ { b } ] ) , } \end{array}\tag{11}
$$

$$
C _ { i } ^ { \mathrm { C o m p } } = \sum _ { j = m i } ^ { m ( i + 1 ) - 1 } S _ { j } ^ { a } \odot C _ { j } ^ { a } + \sum _ { j = m ( i - 1 ) } ^ { m i - 1 } S _ { j } ^ { b } \odot C _ { j } ^ { b } ,\tag{12}
$$

where $\odot$ denotes the Hadamard product; $\mathrm { S o f t m a x } _ { \mathrm { r o w } } ( \cdot )$ denotes the softmax operation along the row dimension, which performs normalization across the total of 2?? elements from both $Z ^ { a }$ and $Z ^ { b }$ . When $i = 0 , Z _ { m ( i - 1 ) : m i - 1 } ^ { b }$ is padded with negative infinity and $C _ { m ( i - 1 ) : m i - 1 } ^ { b }$ is padded with zeros. Note that each $C _ { i } ^ { \mathrm { C o m p } }$ is derived from 2?? KV entries, but the indexes of $C ^ { b }$ used for $C _ { i } ^ { \mathrm { C o m p } }$ and the indexes of $C ^ { a }$ used for $C _ { i - 1 } ^ { \mathsf { C o m p } }$ are overlapped. Therefore, CSA in fact compresses the sequence length to $\frac { 1 } { m }$ times.

Lightning Indexer for Sparse Selection. After obtaining the compressed KV entries $C ^ { \mathrm { C o m p } }$ , CSA applies the DSA strategy to select top-k compressed KV entries for core attention. First, CSA performs the same compression operation used for $C ^ { \mathrm { C o m p } }$ to get compressed indexer keys $K ^ { \mathrm { I C o m } { \bar { \mathrm { p } } } } \in \mathbb { R } ^ { { \frac { n } { m } } \times c ^ { I } }$ , where $c ^ { I }$ is the indexer head dimension. Then, for a query token ??, we produce the indexer queries $\{ \mathbf { q } _ { t , 1 } ^ { I } ; \mathbf { q } _ { t , 2 } ^ { I } ; . . . ; \mathbf { q } _ { t , n _ { h } ^ { I } } ^ { I } \}$ in a low-rank manner:

$$
\begin{array} { r } { \mathbf { c } _ { t } ^ { Q } = \mathbf { h } _ { t } \cdot W ^ { D Q } , } \end{array}\tag{13}
$$

$$
\begin{array} { r } { [ \ P _ { t , 1 } ^ { I } ; \ P _ { t , 2 } ^ { I } ; . . . ; \ P _ { t , n _ { h } ^ { I } } ^ { I } ] = \ P _ { t } ^ { I } = \mathbf { c } _ { t } ^ { Q } \cdot W ^ { I U Q } , } \end{array}\tag{14}
$$

where $\mathbf { h } _ { t } \ \in \ \mathbb { R } ^ { d }$ is the input hidden state of the query token ??; $\mathbf { c } _ { t } ^ { Q } \in \mathbb { R } ^ { d _ { c } }$ is the compressed latent vector for queries; $d _ { c }$ denotes the query compression dimension; $n _ { h } ^ { I }$ denotes the number of indexer query heads; $W ^ { D Q } \in \mathbb { R } ^ { d \times d _ { c } }$ and $W ^ { I U Q } \in \mathbb { R } ^ { d _ { c } \times c ^ { I } n _ { h } ^ { I } }$ are the down-projection and upprojection matrices for indexer queries, respectively. Next, the index score $I _ { t , s } \in \mathbb { R }$ between the query token ?? and a preceding compressed block $\hat { \cdot } ( s < \mathrm { F l o o r } ( \frac { t } { m } ) )$ is computed by

$$
\begin{array} { r } { [ { w _ { t , 1 } ^ { I } } ; { w _ { t , 2 } ^ { I } } ; . . . ; { w _ { t , { n _ { h } ^ { I } } } ^ { I } } ] = \mathbf { w } _ { t } ^ { I } = \mathbf { h } _ { t } \cdot W ^ { w } , } \end{array}\tag{15}
$$

$$
I _ { t , s } = \sum _ { h = 1 } ^ { n _ { h } ^ { I } } \boldsymbol { w } _ { t , h } ^ { I } \cdot \mathrm { R e L U } \left( \mathbf { q } _ { t , h } ^ { I } \cdot K _ { s } ^ { \mathrm { I C o m p } } \right) ,\tag{16}
$$

where $W ^ { w } \in \mathbb { R } ^ { d \times n _ { h } ^ { I } }$ is a learnable matrix; $w _ { t , h } ^ { I } \in \mathbb { R }$ is the weight of the ℎ-th indexer head. For a query token ??, given its index scores $I _ { t , : } ,$ we employ a top-k selector to selectively retain a subset of compressed KV entries $C _ { t } ^ { \mathsf { S p r s C o m p } }$ for subsequent core attention:

$$
C _ { t } ^ { S \mathrm { p r s C o m p } } = \left\{ C _ { s } ^ { \mathrm { C o m p } } \ : \middle | \ : I _ { t , s } \in \mathrm { T o p - k } ( I _ { t , : } ) \right\} .\tag{17}
$$

<!-- image-->  
Figure 4 | Core architectures of HCA. It performs heavier compression, where the KV entries of $m ^ { \prime } \left( \gg m \right)$ tokens will be consolidated into one. Also, we additionally introduce a small set of sliding window KV entries to enhance local fine-grained dependencies.

Shared Key-Value MQA. After selecting the sparse KV entries, CSA then performs core attention in a Multi-Query Attention (MQA) (Shazeer, 2019) manner, where each compressed KV entry in $C _ { t } ^ { \mathsf { S p r s C o m p } }$ serves as both attention key and value. To be specific, for a query token $t ,$ we first produce attention queries $\{ \mathbf { q } _ { t , 1 } ; \mathbf { q } _ { t , 2 } ; . . . ; \mathbf { q } _ { t , n _ { h } } \}$ from the compressed latent vector $\mathbf { c } _ { t } ^ { Q } ;$

$$
[ \mathbf { q } _ { t , 1 } ; \mathbf { q } _ { t , 2 } ; . . . ; \mathbf { q } _ { t , n _ { h } } ] = \mathbf { q } _ { t } = \mathbf { c } _ { t } ^ { Q } \cdot W ^ { U Q } ,\tag{18}
$$

where $n _ { h }$ denotes the number of query heads; $W ^ { U Q } \in \mathbb { R } ^ { d _ { c } \times c n _ { h } }$ is the up-projection matrices for queries. Note that the latent query vector $\mathbf { c } _ { t } ^ { Q }$ is shared with that used for the indexer queries. Next, we perform MQA on $\{ \pmb q _ { t , i } \}$ and $C _ { t } ^ { \mathsf { S p r s C o m p } }$ :

$$
\mathbf { o } _ { t , i } = \mathrm { C o r e A t t i n } \left( \mathtt { q u e r y } = \mathbf { q } _ { t , i } , \mathtt { k e y } = C _ { t } ^ { \mathrm { S p r s C o m p } } , \mathtt { v a l u e } = C _ { t } ^ { \mathrm { S p r s C o m p } } \right) ,\tag{19}
$$

where $\mathbf { o } _ { t , i } \in \mathbb { R } ^ { c }$ is the core attention output of the ??-th head at the ??-th token; CoreAttn(·) denotes the core attention operation.

Grouped Output Projection. In the configuration of DeepSeek-V4, $c n _ { h }$ is quite large. Therefore, directly projecting the outputs of the core attention operation $\left[ \mathbf { o } _ { t , 1 } ; \mathbf { o } _ { t , 2 } ; . . . ; \mathbf { o } _ { t , n _ { h } } \right] = \mathbf { o } _ { t } \in \mathbb { R } ^ { c n _ { h } }$ to a ??-dimensional hidden state will impose a substantial computational burden. To mitigate this cost, we design a grouped output projection strategy. To be specific, we first split $n _ { h }$ outputs into ?? groups, and then for each group of output ${ \bf o } _ { t , i } ^ { G } \in \mathbb { R } ^ { c \frac { n _ { h } } { g } }$ , we project it to a $d _ { g } .$ -dimensional intermediate output ${ \bf o } _ { t , i } ^ { G ^ { \prime } } \in \mathbb { R } ^ { d _ { g } }$ , where $d _ { g } \ < \ c \frac { n _ { h } } { g }$ . Finally, we project the intermediate output $[ \mathbf { o } _ { t , 1 } ^ { G ^ { \prime } } ; \mathbf { o } _ { t , 2 } ^ { G ^ { \prime } } ; . . . ; \mathbf { o } _ { t , g } ^ { G ^ { \prime } } ] \in \mathbb { R } ^ { d _ { g } g }$ to the final attention output $\hat { \mathbf { o } } _ { t } \in \mathbb { R } ^ { d }$

## 2.3.2. Heavily Compressed Attention

The core architecture of HCA is illustrated in Figure 4, which compresses the KV cache in a heavier manner, but does not employ sparse attention.

Compressed Key-Value Entries. By and large, the compression strategy of HCA is similar to that of CSA, but employs a larger compression rate $m ^ { \prime } \left( \gg m \right)$ and does not perform overlapped

compression. Let $H \in \mathbb { R } ^ { n \times d }$ be a sequence of input hidden states, HCA first computes the original KV entries $C \in \mathbb { R } ^ { n \times c }$ and their corresponding compression weights $Z \in \mathbb { R } ^ { n \times c }$ :

$$
C = H \cdot W ^ { K V } ,\tag{20}
$$

$$
Z = H \cdot W ^ { Z } ,\tag{21}
$$

where $W ^ { K V } , W ^ { Z } \in \mathbb { R } ^ { d \times c }$ are trainable parameters. Next, each ??′ KV entries in ?? will be compressed into one according to the compression weights and learnable positional biases $B \in \mathbb { R } ^ { m ^ { \prime } \times c }$ 1 producing $C ^ { \mathsf { C o m p } } \in \mathbb { R } ^ { \frac { n } { m ^ { \prime } } \times c }$ . Each compressed entry $C _ { i } ^ { \mathrm { C o m p } } \in \mathbb { R } ^ { c }$ is computed by

$$
\begin{array} { r } { S _ { m ^ { \prime } i : m ^ { \prime } ( i + 1 ) - 1 } = \mathrm { S o f t m a x } _ { \mathrm { r o w } } ( Z _ { m ^ { \prime } i : m ^ { \prime } ( i + 1 ) - 1 } + B ) , } \end{array}\tag{22}
$$

$$
C _ { i } ^ { \mathrm { C o m p } } = \sum _ { j = m ^ { \prime } i } ^ { m ^ { \prime } ( i + 1 ) - 1 } S _ { j } \odot C _ { j } .\tag{23}
$$

Through this compression operation, HCA compresses the sequence length to $\scriptstyle { \frac { 1 } { m ^ { \prime } } }$ times.

Shared Key-Value MQA and Grouped Output Projection. HCA also employs the shared KV MQA and grouped output projection strategies as CSA does. After the KV compression, for a query token ??, HCA first produces attention queries $\{ \mathbf { q } _ { t , 1 } ; \mathbf { q } _ { t , 2 } ; . . . ; \mathbf { q } _ { t , n _ { h } } \}$ in a low-rank manner:

$$
\begin{array} { r } { \mathbf { c } _ { t } ^ { Q } = \mathbf { h } _ { t } \cdot W ^ { D Q } , } \end{array}\tag{24}
$$

$$
[ \mathbf { q } _ { t , 1 } ; \mathbf { q } _ { t , 2 } ; . . . ; \mathbf { q } _ { t , n _ { h } } ] = \mathbf { q } _ { t } = \mathbf { c } _ { t } ^ { Q } \cdot W ^ { U Q } ,\tag{25}
$$

where $\mathbf h _ { t } \in \mathbb R ^ { d }$ is the input hidden state of the query token $t ; n _ { h }$ denotes the number of query heads; $W ^ { D Q } \in \mathbb { R } ^ { d \times d _ { c } }$ and $W ^ { U Q } \in \mathbb { R } ^ { d _ { c } \times c n _ { h } }$ are the down-projection and up-projection matrices for queries, respectively. Next, we perform MQA on $\{ \mathbf { q } _ { t , i } \}$ and $C ^ { \mathrm { C o m p } }$ :

$$
{ \bf o } _ { t , i } = \mathrm { C o r e A t t n } \left( { \tt q u e r y = { q } } _ { t , i } , \tt k e y = \it C ^ { \mathrm { C o m p } } , \tt v a l u e = \it C ^ { \mathrm { C o m p } } \right) ,\tag{26}
$$

where $\mathbf { o } _ { t , i } \in \mathbb { R } ^ { c }$ is the core attention output of the ??-th head at the ??-th token. Next, as CSA does, HCA splits $n _ { h }$ outputs into ?? groups, and for each group of output ${ \bf o } _ { t , i } ^ { G } \in \mathbb { R } ^ { c \frac { n _ { h } } { g } }$ , HCA projects it to a $d _ { g }$ -dimensional intermediate output ${ \bf o } _ { t , i } ^ { G ^ { \prime } } \in \mathbb { R } ^ { d _ { g } }$ , where $d _ { g } < c \frac { n _ { h } } { g }$ . Finally, HCA projects the intermediate output $[ \mathbf { o } _ { t , 1 } ^ { G ^ { \prime } } ; \mathbf { o } _ { t , 2 } ^ { G ^ { \prime } } ; . . . ; \mathbf { o } _ { t , g } ^ { G ^ { \prime } } ] \in \mathbb { R } ^ { d _ { g } g }$ to the final attention output $ { \hat { \mathbf { o } } } _ { t } \in \mathbb { R } ^ { d }$

## 2.3.3. Other Details

In addition to the core architectures of CSA and HCA described above, our hybrid attention incorporates several other techniques. For writing clarity, we omit these additional techniques from the above introduction and will briefly describe them in this subsection. Also, this subsection focuses only on the core ideas of them and may omit some tiny details for simplicity. We encourage the readers to refer to our open-source implementation for unambiguous details.

Query and Key-Value Entry Normalization. For both CSA and HCA, we perform an additional RMSNorm operation on each head of the queries and the only head of the compressed KV entries, just before the core attention operation. This normalization avoids exploding attention logits and may improve training stability.

Partial Rotary Positional Embedding. For both CSA and HCA, we partially employ the Rotary Positional Embedding (RoPE) (Su et al., 2024) to the attention queries, KV entries, and the core attention outputs. To be specific, for each query vector and KV entry vector used in CSA and HCA, we apply RoPE to its last 64 dimensions. Since the KV entries serve as both attention keys and values, the naive core attention outputs $\left\{ \mathbf { o } _ { t , i } \right\}$ will carry absolute position embeddings, derived from the weighted sum of KV entries. As a countermeasure, we also apply RoPE with position −?? on the last 64 dimensions of each $\mathbf { o } _ { t , i }$ . In this way, the output of the core attention will also carry relative position embeddings — the contribution of each KV entry to the core attention outputs will also be related to the distance between the query and the KV entry.

Additional Branch of Sliding Window Attention. In order to strictly preserve causality in CSA and HCA, each query attends to only preceding compressed KV blocks. Consequently, a query cannot access information from other tokens within its own compressed block. Meanwhile, recent tokens usually possess greater relevance to the query token in language modeling. For these reasons, we introduce a supplementary attention branch to both CSA and HCA in a sliding window manner, for better modeling of local dependencies. To be specific, for each query token, we additionally produce $n _ { \mathrm { w i n } }$ uncompressed KV entries corresponding to the recent $n _ { \mathrm { w i n } }$ tokens. In the core attention of CSA and HCA, these KV entries in the sliding window will be used along with the compressed KV entries.

Attention Sink. In the core attention of CSA and HCA, we employ the trick of attention sink (OpenAI, 2025; Xiao et al., 2024). To be specific, we set a series of learnable sink logits $\{ z _ { 1 } ^ { \prime } , z _ { 2 } ^ { \prime } , . . . , z _ { n _ { h } } ^ { \prime } \}$ . For the ℎ-th attention head, Ex $\mathsf { p } ( z _ { h } ^ { \prime } )$ will be added to the denominator of the attention score:

$$
s _ { h , i , j } = \frac { \mathrm { E x p } ( z _ { h , i , j } ) } { \sum _ { k } \mathrm { E x p } ( z _ { h , i , k } ) + \mathrm { E x p } ( z _ { h } ^ { \prime } ) } ,\tag{27}
$$

where $s _ { h , i , j } , z _ { h , i , j } \in \mathbb { R }$ denote the attention score and attention logit of the ℎ-th attention head between the ??-th query token and the ??-th preceding token or compressed block. This technique allows each query head to adjust its total attention scores to be not equal to 1, and even to be near 0.

## 2.3.4. Efficiency Discussion

Due to the employment of hybrid CSA and HCA, together with low-precision computation and storage, the attention module of DeepSeek-V4 series achieves remarkable efficiency in both attention FLOPs and KV cache size, especially in long-context scenarios. First, we adopt a mixed storage format for KV entries: BF16 precision is used for the rotary positional embedding (RoPE) dimensions, while FP8 precision is applied to the remaining dimensions. This hybrid representation reduces the KV cache size by nearly half compared with pure BF16 storage. Second, attention computation within the lightning indexer is performed in FP4 precision, which accelerates the attention operation under extremely long contexts. Third, relative to DeepSeek-V3.2, a smaller attention top-k is chosen in DeepSeek-V4 series, thereby improving model efficiency on short- and medium-length texts. Finally, and most importantly, compressed attention and hybrid attention techniques substantially reduce both the KV cache size and the computational FLOPs.

Taking BF16 GQA8 (Ainslie et al., 2023) with a head dimension of 128 as the baseline — one of the common configurations of LLM attention — the KV cache size of DeepSeek-V4 series can be dramatically reduced to approximately 2% times of that baseline in the 1M-context setting.

Algorithm 1 Muon Optimizer for DeepSeek-V4   
Require: Learning rate ??, momentum ??, weight decay ??, update rescaling factor ??   
1: for each training step ?? do   
2: for each logically independent weight $W \in \mathbb { R } ^ { n \times m }$ do   
3: $G _ { t } = \nabla _ { W } \mathcal { L } _ { t } ( W _ { t - 1 } )$ ⊲ Compute gradients   
4: $M _ { t } = \mu M _ { t - 1 } + G _ { t }$ ⊲ Accumulate momentum buffer   
5: ??′?? = HybridNewtonSchulz $\left( \mu M _ { t } + G _ { t } \right)$ ⊲ Nesterov trick and hybrid Newton-Schulz   
6: $O _ { t } = O _ { t } ^ { \prime } \cdot \sqrt { \operatorname* { m a x } ( n , m ) } \cdot \gamma$ ⊲ Rescale the update RMS   
7: $W _ { t } = W _ { t - 1 } \cdot ( 1 - \eta \lambda ) - \eta O _ { t }$ ⊲ Perform weight decay and update   
8: end for   
9: end for

Moreover, even when compared with DeepSeek-V3.2 (DeepSeek-AI, 2025) — already an efficient baseline — DeepSeek-V4 series still exhibits substantial advantages in efficiency. A comparison of their inference FLOPs and KV cache size is provided in the right part of Figure 1.

## 2.4. Muon Optimizer

We employ the Muon (Jordan et al., 2024; Liu et al., 2025) optimizer for the majority of modules in DeepSeek-V4 series due to its faster convergence and improved training stability. The full algorithm of our Muon optimization is summarized in Algorithm 1.

Basic Configurations. We maintain the AdamW (Loshchilov and Hutter, 2017) optimizer for the embedding module, the prediction head module, the static biases and gating factors of mHC modules, and the weights of all RMSNorm modules. All other modules are updated with Muon. Following Liu et al. (2025), we also apply weight decay to Muon parameters, use the Nesterov (Jordan et al., 2024; Nesterov, 1983) trick, and rescale the Root Mean Square (RMS) of the update matrix for reutilization of our AdamW hyper-parameters. Different from them, we use hybrid Newton-Schulz iterations for orthogonalization.

Hybrid Newton-Schulz Iterations. For a given matrix ??, let its Singular Value Decomposition (SVD) be $M = U \Sigma V ^ { T }$ . The Newton-Schulz iterations aim to approximately orthogonalize ?? to be ?????? . Usually, ?? will be first normalized as $M _ { 0 } = M / | | \boldsymbol { M } | | _ { F }$ to ensure its maximum singular value does not exceed 1. Then, each Newton-Schulz iteration performs the following operation:

$$
M _ { k } = a M _ { k - 1 } + b ( M _ { k - 1 } M _ { k - 1 } ^ { T } ) M _ { k - 1 } + c ( M _ { k - 1 } M _ { k - 1 } ^ { T } ) ^ { 2 } M _ { k - 1 } .\tag{28}
$$

Our hybrid Newton-Schulz performs 10 iterations over two distinct stages. During the first 8 steps, we use coefficients $( a , b , c ) = ( 3 . 4 4 4 5 , - 4 . 7 7 5 0 , 2 . 0 3 1 5 )$ to drive rapid convergence, bringing the singular values close to 1. In the final 2 steps, we switch to coefficients $( a , b , c ) = ( 2 , - 1 . 5 , 0 . 5 )$ / which stabilize the singular values precisely at 1.

Avoiding Exploding Attention Logits. The attention architecture of DeepSeek-V4 series allows us to directly apply RMSNorm on the attention queries and KV entries, which effectively prevents attention logits from exploding. Consequently, we do not employ the QK-Clip technique (Liu et al., 2025) in our Muon optimizer.

## 3. General Infrastructures

## 3.1. Fine-Grained Communication-Computation Overlap in Expert Parallelism

Mixture-of-Experts (MoE) can be accelerated via Expert Parallelism (EP). However, EP requires complex inter-node communication and imposes substantial demands on interconnect bandwidth and latency. To alleviate the communication bottleneck in EP and achieve higher end-to-end performance under lower interconnection bandwidth requirements, we propose a fine-grained EP scheme that fuses communication and computation into a single pipelined kernel for communication-computation overlapping.

Communication Latency Can Be Hidden. The key insight of our EP scheme is that the communication latency can be effectively hidden beneath computation in MoE layers. As shown in Figure 5, in DeepSeek-V4 series, each MoE layer can be decomposed mainly into four stages: two communication-bound stages, Dispatch and Combine, and two computation-bound stages, Linear-1 and Linear-2. Our profiling reveals that within a single MoE layer, the total time of communication is less than that of the computation. Therefore, after fusing communication and computation into a unified pipeline, computation remains the dominant bottleneck, implying that the system can tolerate lower interconnect bandwidth without degrading end-to-end performance.

<!-- image-->  
Figure 5 | Illustration of our EP scheme with related works. Comet (Zhang et al., 2025b) overlaps Dispatch with Linear-1, and Linear-2 with Combine, separately. Our EP scheme achieves a finergrained overlapping by splitting and scheduling experts into waves. The theoretical speedup is evaluated in the configuration of the DeepSeek-V4-Flash architecture.

Fine-Grained EP Scheme. To further lower the interconnect bandwidth requirement and amplify the benefits of overlapping, we introduce a finer-grained expert partitioning scheme. Inspired by many related works (Aimuyo et al., 2025; Zhang et al., 2025b), we split and schedule the experts into waves. Each wave consists of a small portion of experts. As soon as all experts within the wave have completed their communication, computation can commence immediately without waiting for other experts. In steady state, computation of current wave, token transfer for the next wave, and result sending of completed experts all proceed concurrently, as demonstrated in Figure 5. This forms a fine-grained pipeline among experts, keeping both computation and communication continuous throughout the wave. The wave-based scheduling speeds up the performance on extreme cases such as Reinforcement Learning (RL) rollout, which usually encounters long-tail small batches.

Performance and Open-Sourced Mega-Kernel. We validated the fine-grained EP scheme on both NVIDIA GPUs and HUAWEI Ascend NPUs platforms. Compared against strong non-fused baselines, it achieves 1.50 ∼ 1.73× speedup for general inference workloads, and up to 1.96× for latency-sensitive scenarios such as RL rollouts and high-speed agent serving. We have open-sourced the CUDA-based mega-kernel implementation named MegaMoE2 as a component of DeepGEMM.

Observations and Proposals. We share observations and lessons from kernel development and offer some proposals to hardware vendors, in the hope of aiding efficient hardware design and achieving better software-hardware co-design:

• Computation-Communication Ratio. Full communication-computation overlap hinges on the computation-communication ratio, rather than the bandwidth solely. Denoting peak compute throughput as ?? and interconnect bandwidth as ??, communication can be fully hidden when $C / B \leqslant V _ { \mathrm { c o m p } } / V _ { \mathrm { c o m m } } ,$ where $V _ { \mathrm { c o m p } }$ denotes the computation volume and ??comm refers to the communication volume. For DeepSeek-V4-Pro, where each token-expert pair requires 6ℎ?? FLOPs (SwiGLU gate, up, and down projections) but only 3ℎ bytes of communication (FP8 Dispatch + BF16 Combine), this simplifies to:

$$
{ \frac { C } { B } } \leqslant 2 d = 6 1 4 4 { \mathrm { ~ F L O P s / B y t e } } .
$$

That is, each GBps of interconnect bandwidth suffices to hide the communication for 6.1 TFLOP/s of compute. Once bandwidth meets this threshold, it ceases to be the bottleneck, and devoting additional silicon area to further bandwidth brings diminishing returns. We encourage future hardware designs to target such balance points rather than scale bandwidth unconditionally.

• Power Budget. Extreme kernel fusion drives compute, memory, and network to high load simultaneously, making power throttling a key performance limiter. We suggest that future hardware designs provide sufficient power headroom for such fully concurrent workloads.

• Communication Primitives. In the dispatch stage, we adopt a pull-based approach where each GPU actively reads activations from remote GPUs, avoiding the high notification latency that fine-grained push entails. Future hardware with lower-latency cross-GPU signaling would make push viable and enable more natural communication patterns.

• Activation Function. We propose replacing SwiGLU with a low-cost element-wise activation that involves no exponential or division operations. This directly reduces the overhead of post-GEMM processing, preventing the GEMM pipeline from being stalled by activation function computation, thereby enhancing overall computational throughput and resource utilization.

## 3.2. Flexible and Efficient Kernel Development with TileLang

In practice, our elaborate model architecture would have resulted in hundreds of fine-grained Torch ATen operators. We adopt TileLang (Wang et al., 2026) to develop a set of fused kernels to replace the vast majority of them, delivering optimal performance with minimal effort. It also allows us to quickly prototype operators like attention variants during validation. These kernels play critical roles in model architecture development, large-scale training, and ultimately production deployment of inference services. As a Domain-Specific Language (DSL), TileLang balances development productivity with runtime efficiency, enabling rapid development while supporting deep, iterative optimizations within the same codebase. Additionally, we collaborate closely with the TileLang community to foster a more agile, efficient, and stable kernel development workflow.

Reducing Invocation Overhead with Host Codegen. As accelerators continue to grow in performance, CPU-side orchestration overhead becomes increasingly prominent. For small, highly optimized kernels, such fixed host overhead can easily cap utilization and throughput. A common source of this overhead is that host-side logic, such as runtime contract checks, is typically written in Python for flexibility and thus incurs a fixed per-invocation cost.

We mitigate this overhead with Host Codegen, which moves most host-side logic into generated host code. Specifically, we first co-generate the device kernel and a lightweight host launcher at the IR (Intermediate Representation) level, embedding the necessary metadata—such as data types, rank/shape constraints, and stride/layout assumptions—parsed from the language frontend. The launcher is then lowered to the host source code built on top of the TVM-FFI (Chen et al., 2018) framework, whose compact calling convention and zero-copy tensor interop together minimize host-side overhead. At runtime, this generated host code performs validation and argument marshaling, shifting all per-invocation checks out of the Python execution path. Our measurements show that CPU-side validation overhead drops from tens or hundreds of microseconds to less than one microsecond per invocation.

SMT-Solver-Assisted Formal Integer Analysis. TileLang kernels involve complex tensor index arithmetic that requires strong formal integer analysis. During compilation passes such as layout inference, memory hazard detection, and bound analysis, the compiler must verify whether integer expressions satisfy specific properties to enable the corresponding optimizations. Therefore, stronger formal analysis capabilities can unlock more advanced and complex optimization opportunities.

To this end, we integrate the Z3 SMT solver (De Moura and Bjørner, 2008) into TileLang’s algebraic system, providing formal analysis capability for most integer expressions in tensor programs. We strike a balance between computational overhead and formal expressiveness by translating TileLang’s integer expressions into Z3’s quantifier-free non-linear integer arithmetic (QF_NIA). Based on Integer Linear Programming (ILP) solvers, QF_NIA seamlessly resolves standard linear integer expressions common in kernels. Furthermore, its inherent non-linear reasoning capacity effectively addresses advanced challenges like vectorization over variable tensor shapes. Under reasonable resource limits, Z3 elevates overall optimization performance while restricting compilation time overhead to just a few seconds. The impact is substantial across multiple passes, including vectorization, barrier insertion, and code simplification.

Numerical Precision and Bitwise Reproducibility. In production settings, numerical correctness and reproducibility are as critical as raw throughput. We therefore prioritize accuracy by default: fast-math optimizations are disabled at the compiler level, and precision-affecting approximations are provided only as explicit, opt-in frontend operators (e.g., T.__exp, T.__log, and T.__sin). Conversely, when strict IEEE-754 semantics are required, TileLang provides

IEEE-compliant intrinsics with explicit rounding modes (e.g., T.ieee_fsqrt, T.ieee_fdiv, and T.ieee_add), enabling developers to precisely specify numerical behavior.

We also target bitwise reproducibility for validating kernels against hand-written CUDA baselines. We align TileLang’s algebraic simplification and lowering rules with mainstream CUDA toolchains (e.g., NVCC) to avoid transformations that introduce unintended bit-level differences. Layout annotations (e.g., T.annotate_layout) further allow users to pin down layout-dependent lowering decisions, keeping evaluation and accumulation order consistent with the reference CUDA implementation and thus enabling bit-identical outputs when desired.

Our evaluation shows that these accuracy- and reproducibility-oriented design choices do not sacrifice performance: under conservative defaults, TileLang kernels remain competitive, while exposing knobs to selectively relax numerical constraints for higher speed.

## 3.3. High-Performance Batch-Invariant and Deterministic Kernel Libraries

To enable efficient training and inference, we develop a comprehensive set of high-performance computational kernels. Beyond basic functionalities and maximizing hardware utilization, another pivotal design goal is to ensure training reproducibility and bitwise alignment among pre-training, post-training, and inference pipelines. Therefore, we implement end-to-end, bitwise batch-invariant, and deterministic kernels with minimal performance overhead. These kernels are helpful for debugging, stability analysis, and consistent post-training behavior.

Batch Invariance. Batch invariance ensures that the output of any given token remains bitwise identical, regardless of its position within a batch. To implement batch invariance, the primary challenges are listed as follows:

• Attention. To achieve batch invariance, we cannot use the split-KV method (Dao et al., 2023), which distributes the attention computation for a single sequence across multiple Stream Multiprocessors (SMs) to balance the load of SMs. However, abandoning this technique will lead to severe wave-quantization problems3, which can adversely affect GPU utilization. To address this, we develop a dual-kernel strategy for batch-invariant decoding. The first kernel computes the attention output for an entire sequence within a single SM, ensuring high throughput for fully occupied waves. The second kernel, to minimize the latency of the final partially-filled wave and thus alleviate wave-quantization, uses multiple SMs for a single sequence. For the bitwise identity of these two kernels, we carefully design the calculation path of the second kernel to ensure its accumulation order is the same as that of the first kernel. Additionally, the second kernel utilizes distributed shared memory4 within thread-block clusters, enabling high-speed data exchange across SMs. This dual-kernel method effectively confines the overhead of batch-invariant decoding to be negligible.

• Matrix Multiplication. Traditional cuBLAS library (NVIDIA Corporation, 2024) cannot achieve batch invariance. Therefore, we replace it end-to-end with DeepGEMM (Zhao et al., 2025). Furthermore, for very small batch sizes, conventional implementation usually employs split-k (Osama et al., 2023) techniques to improve performance. Unfortunately, split-k techniques cannot guarantee batch invariance, a pivotal feature in DeepSeek-V4.

Therefore, we abandon split-k in most scenarios, which, however, may cause performance degradation. To address this, we introduce a set of optimizations that enable our implementation of matrix multiplication to match or even surpass the performance of standard split-k in most major scenarios.

Determinism. Deterministic training is highly beneficial for debugging hardware or software issues. Moreover, when training exhibits anomalies such as loss spikes, determinism enables researchers to more easily pinpoint numerical causes and further refine the model design. Nondeterminism in training typically stems from non-deterministic accumulation order, often due to the use of atomic addition instructions. This issue primarily occurs during the backward pass, notably at the following parts:

• Attention Backward. In conventional implementations of backward propagation for sparse attention, we use atomicAdd to accumulate gradients for the KV tokens. This introduces non-determinism due to the non-associativity of floating-point addition. To address this problem, we allocate separate accumulation buffers for each SM, followed by a global deterministic summation across all buffers.

• MoE Backward. When multiple SMs from different ranks concurrently write data to the same buffer on a receiving rank, negotiating writing positions also introduces nondeterminism. To resolve this, we design a token order pre-processing mechanism within each single rank, combined with buffer isolation across multiple ranks. This strategy ensures determinism of both the send results of expert parallelism and the accumulation order in the MoE backward pass.

• Matrix Multiplication in mHC. mHC involves a matrix multiplication with an output dimension of only 24. For very small batch sizes, we are compelled to use the split-k (Osama et al., 2023) algorithm, whose naive implementation will cause non-determinism. To overcome this, we output each split part separately and perform a deterministic reduction in a subsequent kernel, thereby preserving both performance and determinism.

## 3.4. Training Framework

Our training framework is built upon the scalable and efficient infrastructure developed for DeepSeek-V3 (DeepSeek-AI, 2024). In training DeepSeek-V4, we inherit this robust foundation while introducing several key innovations to accommodate its novel architectural components — specifically the Muon optimizer, mHC, and the hybrid attention mechanism — while maintaining high training efficiency and stability.

## 3.4.1. Efficient Implementation of Muon

The Muon optimizer requires the full gradient matrix to compute parameter updates, which presents a challenge when combined with the Zero Redundancy Optimizer (ZeRO) (Rajbhandari et al., 2020). Traditional ZeRO is designed for element-wise optimizers like AdamW, where a single parameter matrix can be partitioned and updated across multiple ranks. To address this conflict, we design a hybrid strategy of ZeRO bucket assignment for Muon.

For dense parameters, we limit the maximum size of ZeRO parallelism and employ a knapsack algorithm to assign parameter matrices to these ranks, ensuring each rank manages a roughly balanced load. The bucket on each rank is padded to match the size of the largest bucket across ranks, facilitating efficient reduce-scatter operations. This padding typically incurs less than 10% memory overhead in our setup, where each rank manages no more than five parameter matrices. When the overall size of data parallelism exceeds the limit for ZeRO, we compute the Muon update redundantly across the extra data-parallel groups, trading computation for reduced total bucket memory.

For MoE parameters, we optimize each expert independently. We first flatten all down projection matrices in SwiGLU (Shazeer, 2020) of all experts across all layers, followed by flattened up projection matrices and gate matrices. Then, we pad the flattened vector to ensure we can evenly distribute this vector across all ranks without splitting any logically independent matrix. Given the large number of experts, we do not impose a limit of ZeRO parallelism for MoE parameters, and the padding overhead is also negligible.

Additionally, on each rank, consecutive parameters of identical shape will be automatically merged, enabling batched execution of the Newton-Schulz iterations for better hardware utilization. Furthermore, we observe that the Newton-Schulz iterations in Muon remain stable when computed with BF16 matrix multiplications. Leveraging this, we further quantize, in a stochastic rounding manner, the MoE gradients to be synchronized across data-parallel ranks to the BF16 precision, halving the communication volume. To avoid accumulation errors introduced by low-precision adders, we replace conventional tree- or ring-based reduce-scatter collectives with a two-phase approach. First, an all-to-all operation exchanges local gradients across ranks, and then each rank performs a local sum in FP32. This design maintains numerical robustness.

## 3.4.2. Cost-Effective and Memory-Efficient Implementation of mHC

The introduction of mHC increases both activation memory consumption and communication volume between pipeline stages, compared with conventional residual connections. To mitigate these costs, we implement several optimization strategies.

Firstly, we carefully design and implement fused kernels of mHC for both training and inference. Secondly, we introduce a recomputation strategy that selectively checkpoints intermediate tensors. Specifically, we recompute most hidden states between layers and all normalized layer inputs, while avoiding recomputation of compute-intensive operations. This achieves a balance between memory saving and computational overhead. Thirdly, we adjust the DualPipe 1F1B overlapping scheme to accommodate the increased pipeline communication and enable concurrent execution of some operations in mHC.

Collectively, these optimizations constrain the wall-time overhead of mHC to only 6.7% of the overlapped 1F1B pipeline stage. More details of the engineering optimization can be found in the dedicated mHC paper (Xie et al., 2026).

## 3.4.3. Contextual Parallelism for Long-Context Attention

Conventional Context Parallelism (CP) partitions the sequence dimension, with each rank maintaining contiguous ?? tokens. This introduces two challenges to our compressed attention mechanisms (i.e., CSA and HCA). On the one hand, training samples are packed from multiple sequences, and each sequence is compressed independently by a factor of ?? (or ??′), with any trailing tokens fewer than ?? being discarded. Consequently, the compressed KV lengths are typically less than $\frac { s } { m }$ and vary across ranks. On the other hand, the compression requires ?? consecutive KV entries, which may straddle the boundary between two neighboring CP ranks.

To address these challenges, we design a two-stage communication approach. In the first stage, each rank ?? sends its last ?? uncompressed KV entries to rank ?? + 1. Then, rank ?? + 1 compresses some of these received entries together with its local ?? uncompressed KV entries,