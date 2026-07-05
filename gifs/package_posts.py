# -*- coding: utf-8 -*-
"""
Source of truth for the packaged deliverable: every GIF paired with a LinkedIn
caption written in a loose, first-person engineer voice (not polished AI copy).
package_build.py turns this into per-day .txt files, an ALL_POSTS.md, and a zip.

Each entry: (day, filename, concept, post)
"""

POSTS = [
(1, "day_01_gpu_memory_coalescing.gif",
 "Split screen: 32 threads hit scattered memory (red, 56 transactions, latency climbing) vs. one contiguous sweep (green, 1 transaction, throughput maxed). Punchline: 'Coalesce your loads. Your DRAM will thank you.'",
 """Spent an afternoon last week convinced a kernel was compute-bound. It wasn't. It was starving on memory the whole time.

The culprit was the access pattern. 32 threads in the warp were reading scattered addresses, so the GPU needed a pile of separate memory transactions to feed them. Reordered the indexing so neighboring threads read neighboring addresses and it collapsed to basically one transaction per load. Big speedup, zero change to the math.

What makes this sneaky is the slow version still gives correct results. Nothing complains. You only catch it when you look at the memory counters.

Coalescing is probably the cheapest CUDA win there is. What's the first thing you check on a memory-bound kernel?

#CUDA #GPU #HPC #PerformanceOptimization"""),

(2, "day_02_warp_divergence.gif",
 "32 lanes in lockstep; an `if (threadIdx.x & 1)` drops and half the lanes freeze while the other half run, then swap. '2 passes for 1 warp.' Punchline: 'Both branches run. Nobody wins.'",
 """Added one if-statement to handle an edge case. Kernel got 2x slower. The FLOP count said nothing had changed.

That branch split the warp. Threads in a warp share a program counter, so the hardware can't run both sides at once — it runs the if, parks the other lanes, then runs the else. Half your threads sit idle on each pass.

What actually helps: group your data so a warp takes one path, or go branchless with predication for the small stuff.

Warp divergence almost never shows up as a bug. It shows up as "why is this slower than the math says it should be."

How do you usually catch it?

#CUDA #GPUProgramming #WarpDivergence #HPC"""),

(3, "day_03_cuda_out_of_memory.gif",
 "Memory climbs as batch size rises, hits an 'OOM' wall short of the 24 GiB line while trying to alloc 2 GiB. Punchline: 'It's never the last 1 GiB. It's the fragmentation.'",
 """64 → out of memory. 63 → out of memory. 32 → fine. We've all been here.

The thing nobody tells you early on: "CUDA out of memory" often isn't about total capacity. nvidia-smi shows free space, but the allocator can't find one contiguous block. It's fragmentation.

Stuff that's saved me more than buying a bigger GPU: gradient checkpointing, bf16, expandable_segments, gradient accumulation with smaller micro-batches. And once, embarrassingly, just finding the tensor I forgot to stop holding a reference to.

What's your first move when OOM hits right before a deadline?

#DeepLearning #CUDA #PyTorch #GPU"""),

(4, "day_04_tensor_cores_unleashed.gif",
 "A CUDA core fills a result matrix cell-by-cell (FLOPs tick by 1s) while a Tensor Core fills whole 4x4 tiles at once (FLOPs leap). Punchline: 'Your Tensor Cores are napping.'",
 """If your matmuls are running in FP32, your Tensor Cores are basically switched off. And they're most of the GPU you paid for.

They do a whole little matrix multiply-accumulate in one shot, but only in mixed precision (bf16/fp16, fp8 on newer cards) and only when your dimensions line up with the tile sizes.

The trap I keep seeing: someone turns on AMP, leaves a dimension unaligned, and the library quietly drops to a slower path. Nobody notices because the loss still goes down. Check Nsight — if Tensor Core utilization is zero, they're asleep.

Do you write your own Tensor Core kernels, or leave it to cuBLAS/CUTLASS?

#TensorCores #CUDA #GEMM #DeepLearning #GPU"""),

(5, "day_05_kv_cache_growth.gif",
 "A memory curve accelerates as tokens are generated, labeled 'KV cache', dwarfing the model weights. Punchline: 'Your 7B model is small. Your KV cache is not.'",
 """Everyone sizes their GPU around the model weights. Then they hit long context and the KV cache ends up eating more memory than the model.

Every token you generate parks a key and value for every layer and every head. Push to 32k context with a few concurrent users and your "small" 7B suddenly won't serve.

PagedAttention in vLLM was the big unlock for us on the fragmentation side. Grouped-query attention shrinks the footprint, and quantizing the KV cache buys even more room.

If your serving throughput is mysteriously bad, look at the cache before you blame the weights.

How are you managing KV cache at high concurrency?

#LLM #Inference #GPU #KVCache #vLLM"""),

(6, "day_06_nsight_profiler_reveal.gif",
 "Two bars: 'What you assumed: COMPUTE-BOUND' (15%) vs 'What Nsight showed: MEMORY-STALLED' (90%). Punchline: 'You're not compute-bound. You never were.'",
 """Two days optimizing the math on a kernel. Opened Nsight Compute. It had been memory-bound the entire time. Cool. Cool cool cool.

This happens to everyone. You're sure you're compute-bound, the profiler shows 80%+ of the time waiting on memory or warp stalls, and all your clever arithmetic changes did nothing.

I profile first now, in order: roofline, memory throughput vs peak, warp stall reasons. Then I touch the code. Change one thing, measure again.

Optimizing without a profiler is just slower guessing.

What's the most surprising thing a profiler ever told you about your code?

#CUDA #Nsight #Profiling #GPU"""),

(7, "day_07_cuda_synchronize_bug.gif",
 "Device output flickers as garbage, then stabilizes to correct once cudaDeviceSynchronize() is added. Punchline: 'It's not flaky. You forgot to sync.'",
 """You add cudaDeviceSynchronize(), the bug disappears, you move on. Please don't move on.

That's not a fix — it means you were reading a result before the GPU had actually finished. Kernel launches are async. The CPU runs ahead, and if you read, copy, or time without syncing properly, you get bugs that vanish the moment you attach a debugger.

Use CUDA events for timing, learn stream ordering before you go multi-stream, and check errors after a sync, not before. And don't just sprinkle syncs everywhere in prod — you'll kill all your overlap.

"A sync makes it work" is a clue, not a solution.

What's the nastiest async bug you've chased?

#CUDA #GPU #Debugging #ParallelComputing"""),

(8, "day_08_llm_inference_latency.gif",
 "Two bars: 'PREFILL / TTFT (compute-bound, parallel)' fills fast; 'DECODE / ITL (memory-bound, one token at a time)' crawls. Punchline: 'Prefill is fast. It's the decode that hurts.'",
 """The demo felt instant to me and slow to the user. Both true, because LLM latency is really two different numbers.

Time to first token is prefill — one big parallel matmul over the prompt, compute-bound. Inter-token latency is decode — one token at a time, memory-bound, mostly shuffling weights and KV cache around.

Optimize the wrong one and you burn a week. Long prompts? Fix prefill with batching and FlashAttention. Long outputs? Fix decode with quantization, speculative decoding, a bigger batch.

Figure out which half you're bound on before you optimize anything.

Are your workloads more prefill or decode heavy?

#LLM #Inference #GPU #Optimization"""),

(9, "day_09_gemm_tiling.gif",
 "A GEMM loads 4x4 tiles into shared memory one at a time and reuses each before moving on; completed tiles stay green. Punchline: 'It's about not re-fetching.'",
 """A fast GEMM doesn't really do more math than a slow one. It just stops reading the same numbers out of memory over and over.

Naive matmul re-fetches operands from global memory constantly. The fast version tiles the matrices, pulls each tile into shared memory once, and reuses it across a bunch of multiply-accumulates before moving on.

The ladder goes: block tiling in shared memory, register tiling per thread, double buffering, Tensor Core MMA in the inner loop... and then honestly you hand the parts you'd get wrong to CUTLASS.

The whole game is FLOPs per byte moved.

Ever written a GEMM by hand, or is it cuBLAS/CUTLASS all the way?

#GEMM #CUDA #TensorCores #HPC #DeepLearning"""),

(10, "day_10_nemo_pipeline.gif",
 "Loose parts (data, tokenize, model, parallelism, checkpoint) get assembled into one running pipeline; a 'batch' packet flows through. Punchline: 'Gluing your own training loop together is a personality, not a strategy.'",
 """Nobody warns you that most of "training a model" is plumbing.

Data loading, tokenization, the distributed strategy, mixed precision, checkpointing, eval — and keeping all of it from falling over across a few hundred GPUs. That's the actual work. The architecture is the easy part.

That's the whole pitch for something like NVIDIA NeMo: the tensor/pipeline/data parallelism is already worked out, and there are recipes for pretraining and fine-tuning (LoRA, SFT, alignment) so you're not rebuilding distributed training from scratch every time.

Scale is an infrastructure problem wearing a modeling costume.

Do you build pipelines from scratch, or start from a framework?

#NeMo #NVIDIA #LLM #DistributedTraining #DeepLearning"""),

(11, "day_11_flashattention_memory.gif",
 "Two bars: 'Standard attention — O(N^2) memory' (huge) vs 'FlashAttention — O(N) memory, streamed in SRAM' (linear). Punchline: 'O(N^2) memory was a choice. FlashAttention un-chose it.'",
 """FlashAttention is one of those rare things that's faster AND uses less memory, which is why everyone switched to it basically overnight.

Regular attention builds the full N×N score matrix in HBM. Flash just doesn't. It tiles the work, keeps everything in on-chip SRAM, and computes the softmax in a streaming pass. Quadratic memory becomes linear.

The reason it's faster even though it recomputes more: it's built around the memory hierarchy, not the FLOP count. Fewer trips to slow memory wins.

Longer context on the same card, more or less for free.

When did you make the switch, and what did it unlock for you?

#FlashAttention #LLM #GPU #Attention #DeepLearning"""),

(12, "day_12_infiniband_vs_ethernet.gif",
 "Two lanes GPU 0 → GPU 1: 'Ethernet / TCP' stalls through CPU and kernel hops (red); 'InfiniBand / RDMA' shoots straight across (green). Punchline: 'At 1000 GPUs, the network IS the computer.'",
 """Watched a training run crawl once because the gradient sync was going over plain TCP. Fastest GPUs money can buy, bottlenecked by the networking stack.

Past a few hundred nodes the interconnect is the story. InfiniBand with RDMA writes straight into another node's memory, skipping the CPU and the kernel entirely. With GPUDirect it's GPU-to-GPU across the fabric, low latency, high bandwidth.

Why it matters: all-reduce is communication-bound at scale. If the network can't keep up, thousands of GPUs just sit there waiting to agree on gradients.

You can be limited by the network long before you're limited by compute.

How much of your step time is comms?

#InfiniBand #RDMA #DistributedTraining #GPU #HPC #NCCL"""),

(13, "day_13_quantization_int8.gif",
 "An FP32 memory bar (24 GB) shrinks to an INT8 bar (6 GB) with a tiny accuracy note. Punchline: 'Do you really need all 32 bits? (You don't.)'",
 """32 bits to store a number the model basically rounds off anyway. Quantization is close to a free lunch when you're careful.

INT8 (or fp16/fp8) shrinks the model, cuts memory bandwidth, and runs faster on hardware with low-precision paths. For a lot of models the accuracy hit is tiny.

Where it gets real: per-channel scaling beats per-tensor, quantization-aware training recovers more than post-training, and you have to deal with activation outliers — that's what SmoothQuant/AWQ/GPTQ are all fighting. Quantizing the KV cache gets you longer context too.

It's not about fewer bits for their own sake, it's throughput per dollar.

What's your quantization stack in production?

#Quantization #LLM #Inference #INT8 #GPU"""),

(14, "day_14_shared_vs_global_memory.gif",
 "Two lanes thread → data: 'Global memory (HBM) ~400 cycles' crawls; 'Shared memory (on-chip) ~20 cycles' is instant. Punchline: 'Shared memory: the closet you keep forgetting exists.'",
 """Almost every "why is my kernel slow" story ends the same way: the data it kept driving to global memory for was sitting in shared memory the whole time.

Global (HBM) is huge but hundreds of cycles away. Shared memory is tiny, on-chip, and you manage it by hand. Staging reuse into shared memory is usually the optimization that actually matters.

Just watch for bank conflicts (a one-column pad usually fixes it) and don't blow your whole occupancy budget on it.

The memory hierarchy really does reward people who respect it.

Favorite use of shared memory you've written?

#CUDA #GPU #SharedMemory #HPC #ParallelComputing"""),

(15, "day_15_nemotron_reasoning.gif",
 "A hard query flows through 'read → reason → self-check → answer'; a deliberate model lands the correct answer. Punchline: 'Fast wrong vs. deliberate right. Pick your model accordingly.'",
 """A confidently wrong answer in half a second is worse than a right one that took three. We finally have models tuned to know the difference.

NVIDIA's Nemotron line leans into reasoning, tool use, and agentic work, with open weights and enough training detail to actually fine-tune and self-host. The interesting shift is that "spend more compute at inference to think harder" is a real feature now, not a prompt trick.

In practice: route the hard queries to a reasoning model, keep a fast one for the easy stuff, and distill the behavior down when you can.

Do you route by difficulty, or run one model for everything?

#Nemotron #NVIDIA #LLM #Reasoning #AI"""),

(16, "day_16_kernel_launch_overhead.gif",
 "A CPU 'launch' row and a GPU 'work' row: many tiny kernels each preceded by a launch stamp bigger than the work itself. Punchline: '1000 tiny kernels = 1000 tiny regrets.'",
 """Split the work into a thousand tiny kernels to be "efficient." Ended up spending more time launching kernels than running them.

Every launch has fixed overhead. One is nothing. A thousand small ones and the overhead is your bottleneck — the profiler timeline turns into little slivers of work with gaps in between.

Fixes: fuse the element-wise ops (torch.compile does this), or capture the sequence as a CUDA Graph and replay it. Basically, make each launch earn its keep.

Lots of tiny gaps between kernels? That's launch overhead eating you.

CUDA Graphs, torch.compile, or hand-fused — what's your go-to?

#CUDA #GPU #CUDAGraphs #torchcompile #Optimization"""),

(17, "day_17_gpu_utilization_lie.gif",
 "Two bars: 'nvidia-smi utilization' (100%) vs 'Model FLOPs Utilization / useful work' (38%). Punchline: 'nvidia-smi says 100%. Your FLOPs say otherwise.'",
 """nvidia-smi says 100% GPU utilization. Feels great. Means almost nothing.

That number just tells you a kernel was running when it sampled. You can be "100% utilized" while memory-bound, running at a fraction of peak FLOPs, or spinning on a busy-wait.

The number I actually trust is MFU — model FLOPs utilization, useful math over peak. 40-50% on a big training run is genuinely good. 100% util tells you basically nothing about efficiency.

Don't celebrate the util graph. Check MFU.

What do you actually use to measure GPU efficiency?

#GPU #MFU #Nsight #DeepLearning #PerformanceEngineering"""),

(18, "day_18_mixed_precision_training.gif",
 "Two bars: 'FP32 training throughput' (1x) vs 'BF16 on Tensor Cores' (~2x). Punchline: 'BF16 for peace of mind. FP16 for the loss-scaling drama.'",
 """Training in full FP32 is like commuting in a tank. Safe, comfortable, and leaving half the GPU unused.

Mixed precision is standard now but the details still bite people. Heavy matmuls in low precision to hit the Tensor Cores, master weights and the sensitive reductions kept in FP32.

Quick version: BF16 has FP32's range so it usually just works, no loss scaling. FP16 has a tiny range, so it needs loss scaling or your gradients underflow to zero. FP8 is showing up on the newest cards with its own scaling dance.

Roughly 2x throughput and half the activation memory for a bit of numerical care.

BF16 or FP16 for you, and why?

#MixedPrecision #BF16 #TensorCores #DeepLearning #GPU"""),

(19, "day_19_debugging_cuda_kernel.gif",
 "A terminal floods with interleaved printf lines from thousands of threads. Punchline: '32,768 threads said hi. None said where the bug is.'",
 """Dropped a printf into a CUDA kernel to see what was going on. 32,768 threads all answered at once. The terminal has not fully recovered.

Debugging GPU code isn't like the CPU — you can't just step through one thread. What actually works: compute-sanitizer (memcheck / racecheck / synccheck) catches most real bugs in seconds, cuda-gdb when you need to step a specific thread, and printf only if you guard it to thread 0.

Honestly, 99% of my "kernel returns garbage" bugs turn out to be an out-of-bounds write or a race — and compute-sanitizer finds both.

What's in your CUDA debugging kit?

#CUDA #Debugging #GPU #HPC"""),

(20, "day_20_speculative_decoding.gif",
 "A small model drafts 4 tokens; the big model verifies all in one parallel pass, accepts 3, corrects 1. Punchline: 'Let the small model type. The big model just proofreads.'",
 """Your big model spends most of decode waiting on memory, not thinking. So stop making it type one token at a time.

Speculative decoding: a small draft model guesses the next few tokens, the big model checks all of them in one parallel pass, keeps the good ones, fixes the first wrong one. The output is provably identical to normal decoding — it's exact, not an approximation.

2-3x faster generation is common, and with Medusa-style heads you don't even need a separate draft model.

Same words the model would've said, just sooner.

Running it in prod? What acceptance rate are you seeing?

#LLM #Inference #SpeculativeDecoding #GPU #Optimization"""),

(21, "day_21_bank_conflicts.gif",
 "Threads all hammer one shared-memory bank (red, serialized), then skew to a diagonal (green, parallel). Punchline: 'Padding by one column: the dumbest fix that always works.'",
 """Your shared-memory kernel is correct and running at one-eighth of its speed. Welcome to bank conflicts.

Shared memory is split into 32 banks. If threads in a warp hit the same bank at different addresses, those accesses serialize — your parallel code quietly queues up single file, and the output looks perfectly fine.

Nsight Compute reports them directly. The classic fix is almost dumb: pad a 2D shared array by one column ([32][33]) to skew the stride. Suddenly it's parallel again.

Ever chased a mystery slowdown that turned out to be this?

#CUDA #GPU #SharedMemory #BankConflicts #HPC"""),

(22, "day_22_torch_compile_first_run.gif",
 "Iteration time is huge on the first run ('compiling...') then drops flat and fast for every run after. Punchline: 'Slow once, fast forever (until you change a shape).'",
 """The first iteration after torch.compile takes forever and you're convinced it hung. It didn't. That's the compile tax.

It traces your model, fuses ops, and generates kernels (usually via Triton) on that first call. After that you're running the optimized version and it flies.

What actually trips people up is recompilation — dynamic shapes trigger it, so pad to fixed shapes or pass dynamic=True. Graph breaks from data-dependent control flow eat into the win too. mode="max-autotune" searches harder: slower compile, faster runtime.

For long training runs and inference servers it's almost always worth it.

Win for you, or graph-break whack-a-mole?

#PyTorch #torchcompile #GPU #Triton #Optimization"""),

(23, "day_23_occupancy_myth.gif",
 "A performance curve rises with occupancy, peaks, then falls as register spills appear. Punchline: 'Occupancy is a means, not a trophy.'",
 """Cranked occupancy to 100% expecting a speedup. Got a slowdown. Occupancy was never the goal.

You need enough of it to hide memory latency, sure. But past that point chasing 100% forces register spills or starves each thread of the work it needs to be efficient. Some of the fastest kernels I've seen run at pretty modest occupancy with heavy register reuse.

Use the occupancy calculator to get into the right range, then measure actual performance. Optimize the kernel, not the number.

Lowest occupancy you've shipped a fast kernel at?

#CUDA #GPU #Occupancy #PerformanceEngineering #HPC"""),

(24, "day_24_all_reduce_gradients.gif",
 "6 GPUs in a ring pass gradient chunks around (reduce-scatter + all-gather) until all hold the same averaged gradient. Punchline: 'Ring all-reduce: the group project that actually works.'",
 """Every training step, thousands of GPUs stop and agree on one averaged gradient. That agreement is where a surprising amount of your speed goes.

That's the all-reduce, and at scale it can dominate step time. Ring all-reduce (what NCCL does) is the neat trick: chop the gradients into chunks, pass them around a ring, and the bandwidth each GPU needs stays constant no matter how many you add.

What makes it fast or slow: NVLink inside the node, InfiniBand across nodes, and overlapping the comms with the backward pass so it's not just dead time.

At thousands of GPUs, training is half a networking problem.

How much of your step is comms vs compute?

#DistributedTraining #NCCL #GPU #InfiniBand #NVLink"""),

(25, "day_25_cuda_version_hell.gif",
 "Mismatched pieces (driver 12.1, toolkit 11.8, wheel cu124) each fail red until a green 'NGC container' fixes it. Punchline: 'Solved by never leaving the container.'",
 """Driver 12.1, toolkit 11.8, a wheel built for 12.4, and a cuDNN that likes none of them. I didn't write a bug, I assembled an incompatibility.

Everyone loses an afternoon to this. The driver, toolkit, framework build, and low-level libs all have compatibility constraints, and a mismatch throws errors that have nothing to do with your code.

What finally fixed it for me: just live in the NGC containers. Known-good stacks, already pinned. And remember nvidia-smi (driver) and nvcc --version (toolkit) are not the same number.

A reproducible env beats "worked yesterday" every single time.

How do you tame the CUDA stack?

#CUDA #Docker #NGC #MLOps #GPU"""),

(26, "day_26_batching_throughput.gif",
 "Bars for batch=1 (5% GPU), batch=8, batch=32 (throughput maxed via continuous batching). Punchline: 'Batch size 1 is a luxury nobody's paying for.'",
 """Batch size 1 uses maybe 5% of the GPU. You're renting a supercomputer to run a calculator.

GPUs are throughput machines — batching spreads the cost of loading weights across many requests and transforms your cost per token. The tension is latency: bigger batch, longer any single request might wait.

Continuous (in-flight) batching is what makes this actually work now. It adds and drops requests mid-generation instead of waiting for a whole batch to finish, and chunked prefill keeps one long prompt from stalling everyone else.

The whole game in serving is filling the GPU without blowing your latency budget.

Where do you draw that line?

#LLM #Inference #Batching #GPU #vLLM"""),

(27, "day_27_atomic_contention.gif",
 "Thousands of threads funnel through a single 'atomicAdd' toll booth, serialized, then privatize per-block. Punchline: 'One atomic to rule them all... and serialize them.'",
 """Made the kernel massively parallel, then routed every thread through one atomicAdd. Built myself a traffic jam.

Atomics are fine until everyone hammers the same address — then they serialize, and your parallel kernel suddenly has a sequential heart. Thousands of threads bumping one global counter, one at a time.

The fix is hierarchical: reduce within the warp with shuffle intrinsics (no memory at all), then within the block via shared memory, and let just one atomic per block touch global memory. Contention drops from thousands to a handful.

Same answer, wildly less contention. "Privatize then combine" is everywhere once you see it.

How do you handle high-contention reductions?

#CUDA #GPU #Atomics #Reduction #HPC"""),

(28, "day_28_nemo_guardrails.gif",
 "A user prompt passes through a 'guardrail' gate before reaching the LLM; a sketchy prompt gets redirected. Punchline: 'Your LLM is brilliant. It still needs bumpers.'",
 """Your model is brilliant right up until someone types "ignore your previous instructions." Brilliant is a demo. Bounded is a product.

Shipping an LLM isn't just the model, it's what happens when a user does something you didn't plan for. Something like NeMo Guardrails sits between your app and the model as a policy layer: topic control, safety filtering, dialog flow, output checks.

It's the difference between something you show off internally and something you actually put in front of customers.

Where do you handle safety — model, app, or both?

#LLM #NeMo #Guardrails #AISafety #NVIDIA"""),

(29, "day_29_pcie_vs_nvlink.gif",
 "Two lanes GPU 0 → GPU 1: PCIe (narrow, slow) vs NVLink (wide, fast). Punchline: 'NVLink where it counts. PCIe where it hurts.'",
 """Split a model across GPUs, forgot to check how they were wired, and every layer's activations went crawling across PCIe while NVLink sat idle.

When you split a model the link between GPUs is suddenly on your critical path. NVLink has way more GPU-to-GPU bandwidth than PCIe. Push tensor-parallel activations over a slow PCIe hop every layer and you cancel out the whole point of parallelism.

nvidia-smi topo -m shows how your GPUs actually connect. Keep tensor-parallel groups inside an NVLink domain, use pipeline/data parallel across the slower links.

The best plan on paper can be the worst one on your actual box.

Do you match your parallelism to your topology?

#GPU #NVLink #PCIe #DistributedTraining #ModelParallelism"""),

(30, "day_30_gradient_checkpointing.gif",
 "'Store every activation' memory bar (OOM risk) vs 'gradient checkpointing' (fits) plus a small '+25% recompute' bar. Punchline: 'Recompute is cheaper than OOM. Always.'",
 """You probably don't need a bigger GPU. You need to stop keeping every activation you'll look at exactly once during backward.

Gradient checkpointing keeps a sparse set of activations and recomputes the rest during the backward pass. You trade maybe 20-30% extra compute for a big drop in activation memory — often enough to fit a longer sequence or a bigger batch on the same card.

It's a one-line change in most frameworks and it stacks nicely with mixed precision and FlashAttention.

Recompute is almost always cheaper than an OOM crash at hour six.

Checkpoint everything, selectively, or not at all?

#DeepLearning #GPU #GradientCheckpointing #Training #PyTorch"""),

(31, "day_31_fp8_training.gif",
 "A precision dial drops to FP8; two formats (E4M3 for weights/activations, E5M2 for gradients) with a scaling factor; throughput jumps again. Punchline: 'FP8: maximum speed, minimum margin for error.'",
 """8-bit floats sound like a joke until you watch them double your throughput again over BF16 — as long as you respect the scaling.

FP8 is where the newest GPUs are pushing. With only 8 bits the range is tight, so it leans hard on scaling factors. There are even two formats: E4M3 (more mantissa, for forward-pass tensors) and E5M2 (more range, for gradients).

Not every layer wants to be FP8, dynamic scaling matters for stability, and Transformer Engine handles most of the bookkeeping so you're not doing it by hand.

Nice example of hardware and numerics evolving together.

Running FP8 yet, or waiting on the tooling to settle?

#FP8 #MixedPrecision #GPU #TransformerEngine #DeepLearning"""),

(32, "day_32_prompt_injection.gif",
 "A retrieved web doc hides 'ignore previous instructions'; a sanitizer flags it before the LLM. Punchline: 'Your context window is an attack surface.'",
 """The moment your model reads a web page, that web page can talk back to it.

That's prompt injection, and it's one of the ugliest open problems in applied AI. The model can't reliably tell "instructions from the developer" apart from "instructions hidden in the document it just retrieved." A malicious page can try to hijack the chat, leak your context, or misuse your tools.

There's no single fix, so you layer: treat retrieved and tool content as data, not instructions, lock down tool permissions, gate sensitive actions behind confirmation, and log anything that looks off.

Building agents means thinking like a security engineer, not just an ML one.

How are you hardening against injection?

#LLM #AISecurity #PromptInjection #RAG #AISafety"""),

(33, "day_33_streaming_multiprocessor.gif",
 "A GPU die zooms into a grid of SM tiles, each with its own warps and shared memory, lighting up as blocks are scheduled. Punchline: 'Thousands of small workers beat one big one — if you keep them all busy.'",
 """A GPU isn't one giant brain. It's a city of small workers, and most slow kernels are just half the city standing around.

Once the Streaming Multiprocessor clicked for me, a lot of CUDA made sense. The GPU is an array of SMs, each with its own schedulers, registers, and shared memory. Your blocks get spread across them, and within each SM the warps get scheduled to hide latency.

So your grid/block sizing should give every SM enough blocks to stay busy, and latency hiding comes from having lots of ready warps — not from any one warp being fast.

What made the SM finally click for you?

#CUDA #GPU #GPUArchitecture #HPC #ParallelComputing"""),

(34, "day_34_lora_finetuning.gif",
 "A frozen 7B model (locked) with tiny trainable low-rank adapters; bars show 7.0B vs ~4M trainable params. Punchline: 'Full fine-tuning walked so LoRA could run (on one GPU).'",
 """You don't need to retrain 7 billion parameters to teach a model your domain. You need to train about four million of them.

LoRA freezes the pretrained weights and slots in small low-rank adapters. You train a sliver of the parameters, which slashes memory and compute — often enough to fine-tune a big model on a single GPU.

The part I love: adapters are swappable. One base model, a folder of task-specific LoRAs. QLoRA quantizes the base to 4-bit for even less memory, and you can merge the adapter back in for zero inference overhead.

It genuinely democratized fine-tuning.

LoRA, QLoRA, full fine-tune, or prompt-tuning as your default?

#LoRA #QLoRA #FineTuning #LLM #PEFT"""),

(35, "day_35_race_condition_shared_mem.gif",
 "Two threads write one shared slot; it flickers between values until a __syncthreads() barrier makes reads clean. Punchline: '__syncthreads(): the barrier between you and 3 hours of confusion.'",
 """It works on the small input, breaks on the big one, and only sometimes. That's not flaky — that's a missing __syncthreads() and an evening of your life.

Shared-memory races are the worst kind of CUDA bug: nondeterministic, invisible in small tests, dependent on scheduling. If threads in a block write and read shared memory without syncing at the right points, you get torn reads and results that change run to run.

Rule of thumb: barrier after you write shared memory, before you read what other threads wrote. Never put __syncthreads() inside a divergent branch. And run racecheck — it catches these automatically.

How do you hunt GPU races?

#CUDA #GPU #RaceCondition #Debugging #ParallelComputing"""),

(36, "day_36_moe_routing.gif",
 "A router sends each token to just 2 of 8 experts; the rest stay dark. Punchline: 'MoE: pay for the whole model, run a slice of it.'",
 """You're paying to store a giant model and only running a sliver of it per token. That's not a bug, that's the whole idea behind mixture-of-experts.

A router sends each token to just a couple of expert subnetworks. Total params are huge, active params per token stay small — capacity of a big model, compute of a much smaller one.

The catch is all in the engineering. Routing has to load-balance or some experts overload while others idle, and spreading experts across GPUs means all-to-all comms — which is where your InfiniBand earns its salary.

MoE trades a FLOP problem for a routing and communication problem.

Serving MoE in prod? How are you doing expert parallelism?

#MoE #LLM #GPU #ExpertParallelism #AIInfrastructure"""),

(37, "day_37_nsight_systems_timeline.gif",
 "An Nsight Systems timeline: the CPU row is busy while the GPU row has big idle gaps, starving. Punchline: 'Your $30k GPU is bottlenecked by a Python for-loop.'",
 """The most expensive thing in the room was idle. A $30k GPU, sitting there waiting on a Python for-loop in the data loader.

That's the usual plot twist in Nsight Systems. You open the timeline expecting a compute problem and instead you see the GPU starving — big gaps while the CPU does data loading and preprocessing.

Fixes: more dataloader workers, prefetch, pinned memory to overlap the H2D copy with compute, and moving preprocessing onto the GPU (DALI) when the CPU can't keep up.

A fast GPU fed by a slow pipeline is a very expensive space heater.

What finally fixed your input pipeline?

#Nsight #GPU #Profiling #DataPipeline #Training"""),

(38, "day_38_tokenizer_surprise.gif",
 "'strawberry' splits into tokens st / raw / berry; '10 characters → 3 tokens'. Punchline: 'The model can't count the R's because it never saw them.'",
 """The reason a model can't count the R's in "strawberry" is that it never saw the letters. It saw "st", "raw", "berry".

Tokenization quietly shapes everything above it — cost, context limits, which tasks feel weirdly hard. Text gets split into subword tokens before the model sees a thing. That's why letter puzzles are hard, why numbers and code fragment in strange ways, and why your token bill never matches your sense of length.

Token count isn't word count isn't character count — measure it. And context limits are in tokens, so tokenizer efficiency is effective context.

So many "why did it do that" moments trace right back here.

Strangest tokenization behavior you've run into?

#LLM #Tokenization #NLP #AI #PromptEngineering"""),

(39, "day_39_pinned_memory_transfer.gif",
 "Two lanes CPU → GPU: pageable memory routes through a 'staging' copy; pinned memory goes direct via DMA and overlaps. Punchline: 'Pinned memory: one flag, free bandwidth.'",
 """Small thing that quietly speeds up a lot of training loops: pinned memory.

Pageable host memory can get moved around by the OS, so CUDA stages your CPU→GPU copy through a temporary buffer — an extra hop you didn't ask for. Allocate pinned (page-locked) memory and the DMA engine moves it directly, and it can overlap the copy with kernel execution on a stream.

In practice that's pin_memory=True in your DataLoader, plus async copies on a non-default stream and double-buffering so the next batch loads while this one trains. Just don't over-pin — it's a limited, non-swappable resource.

One flag, real bandwidth back.

Using pinned memory + streams to hide transfers?

#CUDA #GPU #PinnedMemory #PyTorch #Optimization"""),

(40, "day_40_context_window_overflow.gif",
 "Message chips scroll through a fixed context window; the oldest slide out and vanish. Punchline: 'It didn't forget. It never had it anymore.'",
 """"The model forgot what I told it earlier." It didn't forget — it literally can't see it anymore.

An LLM only attends to what's inside its context window. Go past that budget and something gets dropped: truncation, summarization, or retrieval. Whatever falls out is just gone from the model's view. It's arithmetic, not amnesia.

So you manage it: RAG to pull the relevant history back in, rolling summaries to compress old turns, memory stores outside the model. Long-context models help but cost more compute and KV cache per token — bigger windows aren't free.

Deciding what stays in the window is half the job of building these apps.

How do you handle conversations that outgrow the limit?

#LLM #ContextWindow #RAG #AI #AIInfrastructure"""),

(41, "day_41_cuda_graphs_capture.gif",
 "The same 200 kernels launch every step with CPU gaps, then get captured once and replayed as a single graph. Punchline: 'Record once, replay forever.'",
 """If you're launching the same 200 kernels every training step, you're re-explaining yourself to the GPU 200 times a step.

CUDA Graphs let you capture that sequence once and replay it as a single unit. For small-kernel, high-iteration workloads the CPU-side launch overhead and sync gaps mostly disappear.

Best fit: static shapes, a fixed op sequence (training steps, decode loops), lots of little kernels, and cases where the CPU is the thing feeding the GPU too slowly. torch.compile and framework cuda_graphs modes expose it.

Catch: graphs assume the structure is fixed, so dynamic control flow needs care.

Have CUDA Graphs been worth the integration for you?

#CUDA #CUDAGraphs #GPU #Optimization #Performance"""),

(42, "day_42_nemo_curator_data.gif",
 "Raw web text flows through dedup → quality → PII scrub → clean tokens. Punchline: 'Nobody loves the cleaning that decides if it works.'",
 """Nobody puts "data cleaning" on a slide. But the model you're proud of is only as good as the garbage you managed to filter out.

Boring truth of this field: data quality often beats architecture, and cleaning at scale is a real engineering problem. NVIDIA's NeMo Curator is aimed right at it — GPU-accelerated dedup, quality filtering, PII removal, language ID, built for web-scale corpora.

Dedup cuts memorization and wasted compute. Quality filtering lifts downstream results more than most model tweaks. And doing it on GPUs is what makes web-scale curation actually feasible.

Model quality is downstream of data quality.

How much of your effort goes to curation vs modeling?

#NeMo #DataCuration #NVIDIA #LLM #DataQuality"""),

(43, "day_43_gpu_thermal_throttle.gif",
 "Clock speed runs high, temperature crosses the thermal limit, and the clock steps down while the benchmark sags. Punchline: 'Your benchmark was fast. Your cooling wasn't.'",
 """Benchmark was screaming for the first 30 seconds. Then the card got hot, clocked down, and started telling the truth.

A number you can't sustain isn't a benchmark, it's a first impression. GPUs boost clocks when they have thermal and power headroom and back off when they don't. So a kernel can look incredible briefly and then settle into a much lower steady state under real load.

Measure sustained throughput, not the first few iterations. Watch clocks, temps, and power (nvidia-smi dmon) across a long run. Cooling and power delivery are part of your throughput whether you think about them or not.

Do you measure sustained, or does your benchmark stop before throttling kicks in?

#GPU #Performance #Benchmarking #HPC #DataCenter"""),

(44, "day_44_rag_retrieval.gif",
 "A query embeds, hits a vector search, pulls 3 chunks into context, and the LLM answers with citations. Punchline: 'RAG: giving your model an open-book exam.'",
 """Stop trying to cram all of human knowledge into the weights. Just give the model an open-book exam.

RAG is still one of the most useful patterns going: pull relevant context at query time and let the model reason over it. Easy to describe, fiddly to get right — embed your docs, index them, retrieve the top matches, drop them into the prompt.

Where these systems actually live or die: chunking (too big is noise, too small loses context), embedding quality, and retrieval quality. Honestly, reranking often matters more than which LLM you picked.

The model is usually the easy part. Retrieval is where the work is.

Biggest win you've gotten from tuning a RAG pipeline?

#RAG #LLM #VectorSearch #Embeddings #AI"""),

(45, "day_45_warp_shuffle.gif",
 "A warp sums values by passing them register-to-register via __shfl_down_sync, halving active lanes each step — no memory touched. Punchline: 'The reduction that never touches memory.'",
 """Watched someone bounce values through shared memory to sum 32 numbers. The threads could've just handed the values to each other.

Warp-level primitives are a bit of a CUDA superpower people skip. Shuffle intrinsics (__shfl_down_sync and friends) let threads in a warp swap register values directly — no shared memory, no __syncthreads(), because the lanes are already in lockstep.

For warp-level reductions, scans, and broadcasts it's faster and simpler than the shared-memory version, and it's the natural innermost step of a hierarchical reduction. Cooperative groups give you a cleaner API over the same idea.

Do you drop to warp intrinsics, or stay at the shared-memory level?

#CUDA #GPU #WarpShuffle #HPC #Reduction"""),

(46, "day_46_hallucination_confidence.gif",
 "A model answers about a fake API with 100% confidence and ~0% factual accuracy, beautifully formatted. Punchline: 'It's not lying. It's autocompleting with confidence.'",
 """It invented a function, gave it parameters, documented the return type, and formatted it beautifully. None of it exists.

The scary LLM failure mode isn't being wrong, it's being fluently, confidently wrong. These models are trained to produce plausible continuations, not to flag uncertainty — so a hallucinated API shows up in the exact same polished tone as a correct one. No "I'm guessing" signal anywhere.

What helps in practice: ground answers with retrieval and require citations, constrain outputs to a schema you can check, add a verification pass or tool call, sample a few times and look for agreement, and keep a human in the loop when it matters.

Fluent isn't the same as correct. Design for verification.

Most effective anti-hallucination guardrail you've shipped?

#LLM #Hallucination #AISafety #RAG #AI"""),

(47, "day_47_batch_size_sweet_spot.gif",
 "Throughput rises with batch size, plateaus at a 'sweet spot' knee, then an OOM wall. Punchline: 'Bigger batch until it stops helping, not until it stops fitting.'",
 """Cranked batch size until it OOM'd and called it optimized. The best throughput was actually back at the knee, before the cliff.

Batch size tuning is a roofline problem in disguise. Small batches leave you memory-bound and underusing compute, so throughput climbs fast. At some point you saturate the compute units and it flattens — now you're just adding latency. Past that, OOM.

So sweep it and plot tokens/sec, and look for the knee. The sweet spot is usually before the memory limit, not at it. (For training, remember effective batch size and learning rate move together.)

"Max it until it crashes" leaves throughput on the table and adds latency you didn't need.

Sweep, formula, or gut feel for finding it?

#GPU #DeepLearning #BatchSize #Roofline #Optimization"""),

(48, "day_48_infiniband_topology.gif",
 "A fat-tree InfiniBand fabric moves an all-reduce smoothly until one oversubscribed link jams everything. Punchline: 'Your cluster is only as fast as its worst hop.'",
 """Wired up a cluster, oversubscribed one layer of the network, and every all-reduce started running at the speed of that one bad link.

At cluster scale, topology is a real performance concern, not an infra detail. InfiniBand fabrics are usually fat-trees (or newer rail-optimized designs) so any group of GPUs can talk to any other without a bottleneck. Get it wrong or oversubscribe a layer and collectives slow to the weakest hop.

NCCL is topology-aware and will use a good layout, but it can't fix bandwidth you didn't build. Congestion control and adaptive routing matter when everyone talks at once.

Fast GPUs and fast NICs aren't enough — how you wire them decides whether they cooperate.

How much does your team weigh topology when planning runs?

#InfiniBand #HPC #DistributedTraining #NetworkTopology #GPU #NCCL"""),

(49, "day_49_model_parallelism_types.gif",
 "Layers stack: data parallel, tensor parallel, pipeline parallel, then 3D parallelism + ZeRO/FSDP. Punchline: 'Scaling laws are easy. Scaling infrastructure is the job.'",
 """The scaling law fits in a tweet. Making 3D parallelism map onto your actual interconnect is what eats the quarter.

Too big for one GPU means choosing how to split, and at scale you don't pick one — you stack them. Data parallel replicates and all-reduces. Tensor parallel splits each layer's matrices (heavy comms, keep it inside NVLink). Pipeline parallel puts layers on different GPUs and flows micro-batches through, minding the bubble.

Real runs use all three at once, plus sharded optimizer state (ZeRO/FSDP) to fit memory.

Understanding each axis is easy. Mapping them onto real hardware so comms don't dominate is the hard part.

Which parallelism strategy has been the biggest pain to tune?

#DistributedTraining #ModelParallelism #GPU #FSDP #HPC"""),

(50, "day_50_triton_kernel.gif",
 "A wall of dense CUDA C++ gets replaced by a compact Triton kernel in Python that runs nearly as fast; bars compare lines of code and runtime. Punchline: 'Fast kernels AND your weekend.'",
 """You can write a fast kernel in CUDA C++ and lose a weekend to pointer math, or write it in Triton and keep the weekend.

Custom kernels used to mean committing to CUDA C++ and hand-managing every index. Triton lets you write them in Python at the block level and hands a lot of the low-level stuff — coalescing, shared memory, scheduling — to the compiler.

Way less boilerplate for a lot of kernels, performance that's competitive for common patterns, and it's literally what torch.compile generates under the hood. Great for fused ops and custom attention variants.

CUDA C++ still wins for the last few percent and full control. But Triton dropped the barrier a lot.

Triton or CUDA C++ for your custom kernels?

#Triton #CUDA #GPU #torchcompile #KernelProgramming"""),

(51, "day_51_gemm_arithmetic_intensity.gif",
 "A roofline: a small GEMM sits under the memory-bound slope, then slides up to the compute ceiling as size grows. Punchline: 'Small GEMMs don't fail at math. They fail at feeding.'",
 """A small matmul isn't slow because the GPU can't do the math. It's slow because it's sitting there starving for bytes.

The roofline model explains in one picture why some GEMMs fly and others crawl on the exact same GPU. It comes down to arithmetic intensity — FLOPs per byte moved. Low-intensity ops (small or skinny matmuls, element-wise) are memory-bound; you hit the bandwidth ceiling long before the compute one. Big square GEMMs are compute-bound and can actually approach peak.

The useful part: it tells you whether an op can even reach peak before you spend time on it. Small GEMMs in LLM decode are memory-bound, which is exactly why batching helps.

Optimize the math on a memory-bound kernel and nothing moves.

Do you roofline before optimizing, or dive straight in?

#GEMM #Roofline #GPU #CUDA #Performance"""),

(52, "day_52_checkpoint_save_load.gif",
 "Training progress climbs, a node crashes to zero, then resumes from the last checkpoint. Punchline: 'The only run that never crashes is the one that already finished.'",
 """Three days into a run, a node died. The only difference between a shrug and a disaster was whether we were checkpointing.

At scale, hardware failure isn't an edge case — it's a when. A long run will hit a dead node, a network blip, or a preemption. Checkpointing turns that from catastrophe into a coffee break.

And it's more than "save the weights" — save optimizer state, LR scheduler, RNG state, and step count, or you can't truly resume. Async/distributed checkpointing so saving doesn't stall training, sharded checkpoints for big models, and please, test the restore path before you need it.

Teams that scale smoothly treat fault tolerance as a design requirement.

What's your checkpointing setup for long multi-node runs?

#DistributedTraining #Checkpointing #FaultTolerance #GPU #MLOps"""),

(53, "day_53_nemotron_distillation.gif",
 "A large teacher model generates data; a small student learns it and ends up nearly as accurate but faster and cheaper. Punchline: 'The student graduated smaller AND faster than the teacher.'",
 """The best small model you've used probably wasn't trained on scraped data. It was taught by a much bigger one.

Distillation is how frontier-ish quality trickles down into something you can actually afford to serve. A big teacher generates outputs (or soft targets), a smaller student learns to imitate them, and you end up with most of the quality at a fraction of the serving cost. NVIDIA's Nemotron work leans on exactly this — big models generating high-quality data to train deployable small ones.

Serving cost scales with size, distilled students can pick up reasoning behavior (not just surface patterns), and synthetic data from a strong teacher often beats scraped data for your target task.

The future of practical AI isn't just bigger teachers, it's better students.

Distilling for prod, or serving the big models directly?

#Nemotron #Distillation #NVIDIA #LLM #ModelOptimization"""),

(54, "day_54_deadlock_multi_gpu.gif",
 "GPU 0 waits at all-reduce while GPU 1 waits at a different collective; both freeze as a timeout ticks. Punchline: 'NCCL deadlock: where your whole cluster holds its breath.'",
 """No crash. No error. Just the whole cluster silently holding its breath until a timeout finally fires.

Collective deadlocks are a special flavor of distributed pain, because every rank has to show up to the same collective in the same order. One rank hits an all-reduce while another — after taking a different branch, an early return, a shape mismatch — is waiting somewhere else. Now they wait forever.

To debug: make sure control flow is identical across ranks for collective calls, watch for shape/dtype mismatches that make a rank skip one, set NCCL timeouts, and turn on NCCL debug logging to see who's stuck where.

"It just hangs" is the distributed version of a heisenbug.

Worst training hang you've had to debug?

#DistributedTraining #NCCL #GPU #Debugging #HPC"""),

(55, "day_55_inference_server_scaling.gif",
 "Traffic surges; replicas autoscale and a load balancer fans requests out while p99 latency stays flat. Punchline: 'Scaling inference: the model was never the hard part.'",
 """Getting the model to run was the weekend project. Keeping it up under real traffic is the actual job.

Production inference is a pile of concerns that have nothing to do with the model: autoscaling replicas for spiky demand, load balancing across GPUs, continuous batching to stay full without wrecking latency, cold-start when you scale up, and observability — p50/p95/p99, queue depth, tokens/sec, cost per request.

Triton Inference Server, TensorRT-LLM, vLLM all exist because these problems are hard and everyone hits them.

The model is the ingredient. The serving stack is the restaurant.

Hardest part of running inference at scale for you?

#Inference #LLM #GPU #MLOps #vLLM #TensorRT"""),

(56, "day_56_debugging_nan_loss.gif",
 "A loss curve descends beautifully, then spikes to NaN at step 4,000 until gradient clipping is added. Punchline: 'NaN loss: we need to talk about your learning rate.'",
 """Loss curve was gorgeous. Step 4,000 hit and it went straight to NaN. Somewhere a gradient just went to infinity.

NaN loss is almost always numerical instability, and the suspects are short: learning rate too high (clip the gradients), FP16 overflow/underflow (loss scaling or BF16), a log(0) or divide-by-zero in a custom op, one corrupted sample with wild values, or an unstable softmax/norm.

The workflow that works: turn on anomaly detection to find the exact op, log gradient norms (the spike right before the NaN is your smoking gun), and bisect — fixed batch? fixed step? clipping off?

Feels random, almost always has a concrete cause.

First move when the loss goes NaN?

#DeepLearning #Training #Debugging #MixedPrecision #PyTorch"""),

(57, "day_57_tensorrt_optimization.gif",
 "A PyTorch model enters TensorRT: layers fuse, precision drops to INT8/FP8, kernels auto-tune, and a faster engine comes out. Punchline: 'Training optimizes the weights. TensorRT optimizes everything else.'",
 """There's usually a big gap between "my model runs" and "my model runs well on this exact GPU." Compilers close it.

TensorRT (and TensorRT-LLM) fuse layers to cut launches and memory traffic, pick the fastest kernels for your specific card, apply reduced precision with calibration, and for LLMs add in-flight batching and paged KV cache. What comes out is a hardware-specific engine that's often several times faster than the eager model — same weights, same GPU.

The tradeoff is build time and some rigidity (fixed shapes, a compile step), which is easily worth it for high-volume serving.

Training tuned the weights. This tunes everything about how they run.

Do you compile for inference, or serve eager for the flexibility?

#TensorRT #Inference #GPU #Optimization #LLM #NVIDIA"""),

(58, "day_58_precision_debugging.gif",
 "A row of layers goes FP16-green except one sensitive layer that stays FP32; accuracy recovers. Punchline: 'Mixed precision: emphasis on MIXED.'",
 """Cast the whole model to FP16, accuracy drifted, and I spent a while blaming everything except the one layer that actually mattered.

"Mixed precision" has "mixed" in the name for a reason. Most ops tolerate low precision fine; a few really don't — big reductions, softmax denominators, layer-norm stats, long accumulations. That's where FP16's tiny range or limited mantissa bites.

So keep the sensitive reductions and norms in FP32 (frameworks often do by default), accumulate in higher precision even when inputs are low, and bisect to find the one layer that drifts instead of reverting everything. Prefer BF16 when it's a range problem.

Blindly casting everything to FP16 gets you a fast model that's subtly wrong.

Ever traced an accuracy bug to a single precision-sensitive layer?

#MixedPrecision #GPU #DeepLearning #FP16 #BF16"""),

(59, "day_59_agentic_tool_loop.gif",
 "An agent loops: think → act (tool call) → observe, retrying when a tool fails. Punchline: 'An agent is a while-loop with good judgment (and a big API bill).'",
 """An AI agent is basically a while-loop with good judgment and a shockingly large API bill.

The concept is simple: reason, call a tool, look at the result, repeat until done. The reliability is where all the actual engineering lives — solid tool calling and structured outputs, retries when a tool returns garbage, context management across many steps (the loop eats tokens fast), permissions on what the agent can actually do, and enough observability to debug the trace.

The intelligence part is honestly the easy bit now. The scaffolding around it is what makes an agent trustworthy.

Biggest challenge you've hit making agents reliable?

#AgenticAI #LLM #AI #ToolUse #Agents"""),

(60, "day_60_gpu_full_stack_journey.gif",
 "A montage climbs the stack: CUDA thread → warp/SM → GEMM+Tensor Cores → FlashAttention+KV cache → quantize+compile → NVLink+InfiniBand → NeMo+serving → a live AI product. Punchline: 'Every AI product is a tower of optimizations — all the way down to a single warp.'",
 """Day 60. Zooming all the way out.

Every AI product you use is a tower of optimizations stacked on optimizations, and it goes surprisingly far down:

a CUDA thread does one tiny piece of work, a warp runs 32 of them in lockstep, an SM schedules warps to hide latency, GEMM turns that into the matmuls behind every layer, Tensor Cores accelerate them, FlashAttention and KV caching make Transformers efficient, quantization and compilation cut inference cost, NVLink and InfiniBand let thousands of GPUs cooperate, and frameworks like NeMo make the whole thing usable.

The "magic" of modern AI is really thousands of concrete optimizations, each fixing a real bottleneck, stacked until it feels effortless.

Thanks for following along these 60 days — the comments were the best part.

Which layer of the stack do you most enjoy working on?

#GPU #CUDA #AI #LLM #DeepLearning #HPC #TensorCores #InfiniBand"""),
]
