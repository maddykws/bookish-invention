"""
days.py — data-driven configs for the 60-day GIF series.

Days 1, 2, 4 have bespoke scripts (day_01/02/04_*.py). Everything else is
defined here as a config consumed by build.py. Each entry:

  day, file, title, subtitle, caption, arch, params, punch, punch_sub

`params` shape depends on `arch` (see archetypes.py). Callables (grid patterns,
gauge curves, timeline layouts) are plain Python functions — configs are code.
"""
import giflib as g

# ---------------------------------------------------------------------------
# small helpers for archetype callables
# ---------------------------------------------------------------------------
def flicker(t, r, col, rows, cols):
    """garbage that stabilizes to green once 'sync' is added (t>0.55)."""
    if t > 0.55:
        return "good"
    return "bad" if (r * 7 + col * 3 + int(t * 60)) % 3 == 0 else "warn"


def tile_sweep(t, r, col, rows, cols):
    """4x4 tiles load one at a time; completed tiles stay 'good'."""
    tsz = 4
    tiles_x = cols // tsz
    tr, tc = r // tsz, col // tsz
    tindex = tr * tiles_x + tc
    active = int(t * (tiles_x * (rows // tsz)))
    if tindex < active:
        return "good"
    if tindex == active:
        return "on"
    return "off"


def column_conflict(t, r, col, rows, cols):
    """all threads hammer one bank (col 3) -> serialized; then skew to diagonal."""
    if t < 0.55:
        return "bad" if col == 3 else "off"
    return "good" if col == (r % cols) else "off"


def sm_fill(t, r, col, rows, cols):
    idx = r * cols + col
    return "nv" if idx < int(t * rows * cols) else "off"


def moe_route(t, r, col, rows, cols):
    idx = r * cols + col
    chosen = {int(t * 7) % 8, (int(t * 7) + 3) % 8}
    return "alt" if idx in chosen else "off"


def shuffle_tree(t, r, col, rows, cols):
    step = int(t * 4)                     # 0..3 reduction steps
    active = max(1, cols >> step)
    return "good" if col < active else "off"


def one_sensitive(t, r, col, rows, cols):
    bad_col = 6
    if col == bad_col:
        return "bad" if t < 0.55 else "warn"   # kept in FP32
    return "good" if t > 0.3 else "on"


# gauge curves --------------------------------------------------------------
def oom_curve(x):
    return min(0.97, x * 1.15)

def kv_curve(x):
    return 0.15 + 0.85 * (x ** 1.8)

def compile_curve(x):
    return 0.95 if x < 0.13 else 0.22

def occupancy_curve(x):
    # perf rises, peaks ~0.6 occupancy, then register spills drag it down
    return max(0.0, 1.0 - 3.2 * (x - 0.58) ** 2)

def throttle_curve(x):
    return 0.95 if x < 0.45 else max(0.6, 0.95 - (x - 0.45) * 0.9)

def knee_curve(x):
    return min(0.92, 1.15 * (1 - 2.718 ** (-3.4 * x)))

def roofline_curve(x):
    return min(0.9, x * 1.7)              # memory-bound slope -> compute ceiling

def nan_curve(x):
    if x < 0.68:
        return 0.8 - x * 0.85            # loss descends
    return 0.24 + (x - 0.68) * 6.0       # spikes to NaN

def resume_curve(x):
    if x < 0.55:
        return x * 1.5                    # progress climbs
    if x < 0.62:
        return 0.05                       # crash
    return 0.83 * (x - 0.62) / 0.38 + 0.05  # resume from checkpoint


# timeline layouts ----------------------------------------------------------
def launch_overhead_layout(t):
    # GPU: many tiny kernels with big gaps; CPU: launch 'stamps' before each
    gpu, cpu = [], []
    k = 10
    reveal = int(t * k) + 1
    for i in range(min(reveal, k)):
        base = i / k
        cpu.append((base, base + 0.02, g.AMBER))          # launch overhead
        gpu.append((base + 0.03, base + 0.06, g.BLUE))     # tiny kernel
    return cpu, gpu

def graphs_layout(t):
    # first half gappy launches; second half one solid replayed graph
    if t < 0.5:
        gpu, cpu = [], []
        k = 10
        for i in range(k):
            base = i / k
            cpu.append((base, base + 0.02, g.AMBER))
            gpu.append((base + 0.03, base + 0.06, g.BLUE))
        return cpu, gpu
    else:
        return [(0.02, 0.06, g.AMBER)], [(0.06, 0.97, g.GREEN)]

def nsight_layout(t):
    # CPU busy (dense) while GPU starves (sparse blocks + gaps)
    cpu = [(i/8, i/8 + 0.10, g.AMBER) for i in range(8)]
    gpu = []
    k = 4
    reveal = min(int(t * k) + 1, k)
    for i in range(reveal):
        base = i / k + 0.12
        gpu.append((base, base + 0.05, g.BLUE))
    return cpu, gpu


DAYS = [
    dict(day=3, file="day_03_cuda_out_of_memory.gif",
         title="pytorch ~ cuda oom", subtitle="batch_size += 1",
         caption="It's never the total capacity. It's the fragmentation.",
         arch="gauge_spike",
         params=dict(fn=oom_curve, y_label="GPU MEMORY", x_label="batch size",
                     color=g.AMBER, head_color=g.RED, wall=0.85,
                     wall_label="OOM", threshold=0.92,
                     threshold_label="24 GiB", annot="alloc 2 GiB"),
         punch="CUDA out of memory.",
         punch_sub="It's never the last 1 GiB. It's the fragmentation."),

    dict(day=5, file="day_05_kv_cache_growth.gif",
         title="llm ~ kv cache", subtitle="context grows",
         caption="Every token you generate, the cache remembers.",
         arch="gauge_spike",
         params=dict(fn=kv_curve, y_label="GPU MEMORY", x_label="tokens generated",
                     color=g.PURPLE, head_color=g.PURPLE, annot="KV cache"),
         punch="Your 7B model is small.",
         punch_sub="Your KV cache is not.  #LLM #KVCache"),

    dict(day=6, file="day_06_nsight_profiler_reveal.gif",
         title="nsight ~ compute", subtitle="roofline says...",
         caption="You were sure it was compute-bound.",
         arch="bars_race",
         params=dict(bars=[
             dict(label="What you assumed:  COMPUTE-BOUND", target=0.15,
                  color=g.BLUE, value="15% FLOPs"),
             dict(label="What Nsight showed:  MEMORY-STALLED", target=0.9,
                  color=g.RED, value="90% stalls", speed=0.9),
         ], note="Measure before you optimize."),
         punch="You're not compute-bound.",
         punch_sub="You never were.  #CUDA #Nsight"),

    dict(day=7, file="day_07_cuda_synchronize_bug.gif",
         title="cuda ~ async", subtitle="host raced ahead",
         caption="Read the result before the GPU finished? Garbage.",
         arch="grid_states",
         params=dict(rows=4, cols=8, pattern=flicker, cell=42, gap=8,
                     labels=("device output", "add cudaDeviceSynchronize() -> correct"),
                     caption_color=g.GREEN),
         punch="It's not flaky.",
         punch_sub="You forgot to sync.  #CUDA #GPU"),

    dict(day=8, file="day_08_llm_inference_latency.gif",
         title="llm ~ latency", subtitle="two bottlenecks",
         caption="LLM latency is really two numbers with opposite limits.",
         arch="bars_race",
         params=dict(bars=[
             dict(label="PREFILL / TTFT  (compute-bound, parallel)", target=0.92,
                  color=g.GREEN, value="fast", speed=1.5),
             dict(label="DECODE / ITL  (memory-bound, one token at a time)",
                  target=0.55, color=g.AMBER, value="slow", speed=0.6),
         ], note="Are your workloads prefill-heavy or decode-heavy?"),
         punch="Prefill is fast.",
         punch_sub="It's the decode that hurts.  #LLM #Inference"),

    dict(day=9, file="day_09_gemm_tiling.gif",
         title="cuda ~ gemm tiling", subtitle="C = A x B",
         caption="A fast GEMM is about memory reuse, not multiplying.",
         arch="grid_states",
         params=dict(rows=8, cols=8, pattern=tile_sweep, cell=34, gap=6,
                     labels=("load one tile into shared memory, reuse it",
                             "tile by tile -> maximize FLOPs per byte")),
         punch="GEMM isn't about multiplying.",
         punch_sub="It's about not re-fetching.  #GEMM #CUDA"),

    dict(day=10, file="day_10_nemo_pipeline.gif",
         title="nvidia ~ nemo", subtitle="one config",
         caption="Six fragile scripts, or one training framework?",
         arch="flow_pipeline",
         params=dict(stages=[
             dict(label="data", color=g.BLUE), dict(label="tokenize", color=g.BLUE),
             dict(label="model", color=g.NVIDIA), dict(label="parallelism", color=g.NVIDIA),
             dict(label="checkpoint", color=g.GREEN),
         ], packet_label="batch", packet_color=g.AMBER),
         punch="Gluing your own training loop together",
         punch_sub="is a personality, not a strategy.  #NeMo #NVIDIA"),

    dict(day=11, file="day_11_flashattention_memory.gif",
         title="attention ~ flash", subtitle="never materialize",
         caption="Standard attention builds the whole N x N matrix. Flash doesn't.",
         arch="bars_race",
         params=dict(bars=[
             dict(label="Standard attention  -  O(N^2) memory", target=1.0,
                  color=g.RED, value="huge"),
             dict(label="FlashAttention  -  O(N) memory (streamed in SRAM)",
                  target=0.26, color=g.GREEN, value="linear", speed=1.3),
         ], note="IO-aware: fewer trips to slow HBM."),
         punch="O(N^2) memory was a choice.",
         punch_sub="FlashAttention un-chose it.  #FlashAttention #LLM"),

    dict(day=12, file="day_12_infiniband_vs_ethernet.gif",
         title="net ~ rdma", subtitle="GPU to GPU",
         caption="At scale, the interconnect becomes the bottleneck.",
         arch="compare_paths",
         params=dict(top_label="Ethernet / TCP", bottom_label="InfiniBand / RDMA",
                     src="GPU 0", dst="GPU 1", top_speed=0.5, bottom_speed=1.0,
                     top_hops=[(0.4, "CPU"), (0.66, "kernel")],
                     note="RDMA writes straight into remote memory, bypassing the CPU."),
         punch="At 1000 GPUs,",
         punch_sub="the network IS the computer.  #InfiniBand #RDMA"),

    dict(day=13, file="day_13_quantization_int8.gif",
         title="deploy ~ quantization", subtitle="fp32 -> int8",
         caption="Do you really need all 32 bits?",
         arch="bars_race",
         params=dict(bars=[
             dict(label="FP32 model  -  memory footprint", target=1.0,
                  color=g.RED, value="24 GB"),
             dict(label="INT8 model  -  memory footprint", target=0.25,
                  color=g.GREEN, value="6 GB", speed=1.2),
         ], note="~4x smaller, ~0.3% accuracy drop with good calibration."),
         punch="Do you really need all 32 bits?",
         punch_sub="(You don't.)  #Quantization #INT8"),

    dict(day=14, file="day_14_shared_vs_global_memory.gif",
         title="cuda ~ memory hierarchy", subtitle="cycles matter",
         caption="Global memory is huge. Shared memory is next door.",
         arch="compare_paths",
         params=dict(top_label="Global memory (HBM)  ~400 cycles",
                     bottom_label="Shared memory (on-chip)  ~20 cycles",
                     src="thread", dst="data", top_speed=0.42, bottom_speed=1.0,
                     note="Stage reuse into shared memory; watch for bank conflicts."),
         punch="Shared memory:",
         punch_sub="the closet you keep forgetting exists.  #CUDA #GPU"),

    dict(day=15, file="day_15_nemotron_reasoning.gif",
         title="nvidia ~ nemotron", subtitle="thinking is a feature",
         caption="Fast wrong, or deliberate right?",
         arch="flow_pipeline",
         params=dict(stages=[
             dict(label="read", color=g.BLUE), dict(label="reason", color=g.NVIDIA),
             dict(label="self-check", color=g.NVIDIA), dict(label="answer", color=g.GREEN),
         ], packet_label="hard query", packet_color=g.AMBER,
             note="Trade latency for accuracy on demand."),
         punch="Fast wrong vs. deliberate right.",
         punch_sub="Pick your model accordingly.  #Nemotron #Reasoning"),

    dict(day=16, file="day_16_kernel_launch_overhead.gif",
         title="cuda ~ launch overhead", subtitle="1000 tiny kernels",
         caption="The launch cost more than the kernel.",
         arch="timeline_track",
         params=dict(rows=[("CPU launch", 230), ("GPU work", 330)],
                     layout=launch_overhead_layout,
                     note="Fuse kernels or capture a CUDA Graph."),
         punch="1000 tiny kernels",
         punch_sub="= 1000 tiny regrets.  #CUDA #Optimization"),

    dict(day=17, file="day_17_gpu_utilization_lie.gif",
         title="gpu ~ utilization", subtitle="the 100% lie",
         caption="A kernel was running. That's all 'utilization' means.",
         arch="bars_race",
         params=dict(bars=[
             dict(label="nvidia-smi  'GPU utilization'", target=1.0,
                  color=g.AMBER, value="100%"),
             dict(label="Model FLOPs Utilization (useful work)", target=0.38,
                  color=g.RED, value="38% MFU", speed=0.9),
         ], note="MFU is the number serious training teams track."),
         punch="nvidia-smi says 100%.",
         punch_sub="Your FLOPs say otherwise.  #GPU #MFU"),

    dict(day=18, file="day_18_mixed_precision_training.gif",
         title="training ~ mixed precision", subtitle="bf16 vs fp16",
         caption="Half the memory, roughly double the throughput.",
         arch="bars_race",
         params=dict(bars=[
             dict(label="FP32 training throughput", target=0.5,
                  color=g.BLUE, value="1x"),
             dict(label="BF16 on Tensor Cores", target=1.0,
                  color=g.GREEN, value="~2x", speed=1.3),
         ], note="BF16: FP32's range, no loss scaling needed."),
         punch="BF16 for peace of mind.",
         punch_sub="FP16 for the loss-scaling drama.  #MixedPrecision"),

    dict(day=19, file="day_19_debugging_cuda_kernel.gif",
         title="cuda ~ debugging", subtitle="printf in a kernel",
         caption="You added printf. All 32,768 threads answered.",
         arch="type_reveal",
         params=dict(mode="printf_flood"),
         punch="Reach for compute-sanitizer.",
         punch_sub="printf in a kernel is a cry for help.  #CUDA #Debugging"),

    dict(day=20, file="day_20_speculative_decoding.gif",
         title="llm ~ speculative decoding", subtitle="guess, then verify",
         caption="Decode is memory-bound, so verification is nearly free.",
         arch="flow_pipeline",
         params=dict(stages=[
             dict(label="draft x4", sub="small model", color=g.BLUE),
             dict(label="verify", sub="1 parallel pass", color=g.NVIDIA),
             dict(label="accept 3", color=g.GREEN),
             dict(label="correct 1", color=g.AMBER),
         ], packet_label="tokens", packet_color=g.PURPLE,
             note="Output distribution is provably identical."),
         punch="Let the small model type.",
         punch_sub="The big model just proofreads.  #LLM #Inference"),

    dict(day=21, file="day_21_bank_conflicts.gif",
         title="cuda ~ bank conflicts", subtitle="32 banks, one line",
         caption="Shared memory is fast, until everyone hits one bank.",
         arch="grid_states",
         params=dict(rows=4, cols=8, pattern=column_conflict, cell=42, gap=8,
                     labels=("threads -> shared memory banks",
                             "pad by one column to skew the stride"),
                     caption_color=g.GREEN),
         punch="Padding by one column:",
         punch_sub="the dumbest fix that always works.  #CUDA #GPU"),

    dict(day=22, file="day_22_torch_compile_first_run.gif",
         title="pytorch ~ torch.compile", subtitle="the first-run tax",
         caption="Iteration one traces, fuses, and codegens. Then it flies.",
         arch="gauge_spike",
         params=dict(fn=compile_curve, y_label="ITER TIME", x_label="iteration",
                     color=g.GREEN, head_color=g.GREEN, annot="compiling..."),
         punch="Slow once, fast forever",
         punch_sub="(until you change a shape).  #PyTorch #torchcompile"),

    dict(day=23, file="day_23_occupancy_myth.gif",
         title="cuda ~ occupancy", subtitle="not a trophy",
         caption="More threads isn't more speed past the point of latency-hiding.",
         arch="gauge_spike",
         params=dict(fn=occupancy_curve, y_label="PERFORMANCE",
                     x_label="occupancy ->", color=g.BLUE, head_color=g.RED,
                     annot="register spills"),
         punch="Occupancy is a means,",
         punch_sub="not a trophy.  #CUDA #Performance"),

    dict(day=24, file="day_24_all_reduce_gradients.gif",
         title="nccl ~ all-reduce", subtitle="everyone agrees",
         caption="After every step, all GPUs must agree on the gradient.",
         arch="ring_net",
         params=dict(mode="allreduce", n=6, dot_color=g.BLUE,
                     center_label=("ring all-reduce", "reduce-scatter + all-gather")),
         punch="Ring all-reduce:",
         punch_sub="the group project that actually works.  #NCCL #GPU"),

    dict(day=25, file="day_25_cuda_version_hell.gif",
         title="mlops ~ cuda stack", subtitle="version matrix",
         caption="Driver, toolkit, wheel, cuDNN... none of them agree.",
         arch="flow_pipeline",
         params=dict(stages=[
             dict(label="driver", sub="12.1", color=g.RED),
             dict(label="toolkit", sub="11.8", color=g.RED),
             dict(label="wheel", sub="cu124", color=g.RED),
             dict(label="container", sub="NGC image", color=g.GREEN),
         ], packet_label="import torch", packet_color=g.AMBER,
             note="Pin a known-good stack and never leave the container."),
         punch="CUDA dependency hell:",
         punch_sub="solved by never leaving the container.  #Docker #MLOps"),

    dict(day=26, file="day_26_batching_throughput.gif",
         title="serving ~ batching", subtitle="fill the GPU",
         caption="Feed a throughput machine one request at a time?",
         arch="bars_race",
         params=dict(bars=[
             dict(label="batch = 1", target=0.12, color=g.RED, value="5% GPU"),
             dict(label="batch = 8", target=0.5, color=g.AMBER, value="", speed=1.1),
             dict(label="batch = 32  (continuous batching)", target=0.95,
                  color=g.GREEN, value="throughput ↑", speed=1.25),
         ], note="Continuous batching fills the GPU without wrecking latency."),
         punch="Batch size 1 is a luxury",
         punch_sub="nobody's paying for.  #LLM #Inference"),

    dict(day=27, file="day_27_atomic_contention.gif",
         title="cuda ~ atomics", subtitle="one toll booth",
         caption="Thousands of threads, one atomicAdd, one queue.",
         arch="flow_pipeline",
         params=dict(stages=[
             dict(label="32 threads", color=g.BLUE),
             dict(label="1 atomic", sub="serialized", color=g.RED),
             dict(label="result", color=g.GREEN),
         ], packet_label="atomicAdd", packet_color=g.AMBER,
             note="Privatize per-block, then combine: contention drops to a handful."),
         punch="One atomic to rule them all...",
         punch_sub="and in the darkness serialize them.  #CUDA #GPU"),

    dict(day=28, file="day_28_nemo_guardrails.gif",
         title="nvidia ~ nemo guardrails", subtitle="rails in, rails out",
         caption="A capable model without guardrails is a demo.",
         arch="flow_pipeline",
         params=dict(stages=[
             dict(label="user", color=g.BLUE),
             dict(label="guardrail", sub="topic + safety", color=g.NVIDIA),
             dict(label="LLM", color=g.PURPLE),
             dict(label="response", color=g.GREEN),
         ], packet_label="prompt", packet_color=g.AMBER,
             note="Programmable rails: topics, safety, dialog, output checks."),
         punch="Your LLM is brilliant.",
         punch_sub="It still needs bumpers.  #NeMo #AISafety"),

    dict(day=29, file="day_29_pcie_vs_nvlink.gif",
         title="gpu ~ interconnect", subtitle="know your topology",
         caption="Split a model across a slow link and parallelism evaporates.",
         arch="compare_paths",
         params=dict(top_label="PCIe  (narrow)", bottom_label="NVLink  (wide)",
                     src="GPU 0", dst="GPU 1", top_speed=0.45, bottom_speed=1.0,
                     note="Keep tensor-parallel groups inside an NVLink domain."),
         punch="NVLink where it counts.",
         punch_sub="PCIe where it hurts.  #GPU #NVLink"),

    dict(day=30, file="day_30_gradient_checkpointing.gif",
         title="training ~ checkpointing", subtitle="trade FLOPs for GB",
         caption="Keep a few activations, recompute the rest in backward.",
         arch="bars_race",
         params=dict(bars=[
             dict(label="store every activation  -  memory", target=1.0,
                  color=g.RED, value="OOM risk"),
             dict(label="gradient checkpointing  -  memory", target=0.35,
                  color=g.GREEN, value="fits", speed=1.2),
             dict(label="extra recompute cost", target=0.27,
                  color=g.AMBER, value="+25%", speed=1.1),
         ], note="Compounds with mixed precision and FlashAttention."),
         punch="Recompute is cheaper than OOM.",
         punch_sub="Always.  #DeepLearning #GPU"),

    dict(day=31, file="day_31_fp8_training.gif",
         title="training ~ fp8", subtitle="8 bits, handle with care",
         caption="Another throughput bump over BF16 - if you respect scaling.",
         arch="bars_race",
         params=dict(bars=[
             dict(label="BF16 throughput", target=0.62, color=g.BLUE, value="1x"),
             dict(label="FP8  (E4M3 fwd / E5M2 grad + scaling)", target=1.0,
                  color=g.NVIDIA, value="faster", speed=1.3),
         ], note="Transformer Engine manages per-tensor scaling factors."),
         punch="FP8: maximum speed,",
         punch_sub="minimum margin for error.  #FP8 #NVIDIA"),

    dict(day=32, file="day_32_prompt_injection.gif",
         title="ai security ~ injection", subtitle="retrieved != trusted",
         caption="Hidden in the retrieved doc: 'ignore previous instructions'.",
         arch="flow_pipeline",
         params=dict(stages=[
             dict(label="retrieve", sub="web doc", color=g.BLUE),
             dict(label="sanitize", sub="flag injection", color=g.RED),
             dict(label="LLM", color=g.PURPLE),
             dict(label="safe answer", color=g.GREEN),
         ], packet_label="untrusted data", packet_color=g.AMBER,
             note="Treat all retrieved/tool content as data, not instructions."),
         punch="Your context window",
         punch_sub="is an attack surface.  #AISecurity #PromptInjection"),

    dict(day=33, file="day_33_streaming_multiprocessor.gif",
         title="gpu ~ SM array", subtitle="a city, not a genius",
         caption="A GPU isn't one brain. It's an array of SMs.",
         arch="grid_states",
         params=dict(rows=4, cols=6, pattern=sm_fill, cell=54, gap=12,
                     labels=("thread blocks scheduled across SMs",
                             "keep every SM busy to hide latency")),
         punch="Thousands of small workers beat one big one",
         punch_sub="- if you keep them all busy.  #CUDA #GPU"),

    dict(day=34, file="day_34_lora_finetuning.gif",
         title="peft ~ lora", subtitle="freeze the giant",
         caption="Freeze the pretrained weights, train tiny low-rank adapters.",
         arch="bars_race",
         params=dict(bars=[
             dict(label="Full fine-tune  -  trainable params", target=1.0,
                  color=g.RED, value="7.0B"),
             dict(label="LoRA  -  trainable params", target=0.02,
                  color=g.GREEN, value="~4M", speed=1.4),
         ], note="One base model, many swappable adapters. QLoRA goes lower."),
         punch="Full fine-tuning walked",
         punch_sub="so LoRA could run (on one GPU).  #LoRA #PEFT"),

    dict(day=35, file="day_35_race_condition_shared_mem.gif",
         title="cuda ~ race condition", subtitle="order isn't free",
         caption="Two threads, one shared slot, no barrier. Torn reads.",
         arch="grid_states",
         params=dict(rows=4, cols=8, pattern=flicker, cell=42, gap=8,
                     labels=("shared memory writes without a barrier",
                             "__syncthreads() -> clean, deterministic reads"),
                     caption_color=g.GREEN),
         punch="__syncthreads():",
         punch_sub="the barrier between you and 3 hours of confusion.  #CUDA"),

    dict(day=36, file="day_36_moe_routing.gif",
         title="llm ~ mixture of experts", subtitle="sparse spend",
         caption="Each token flows through just 2 of 8 experts.",
         arch="grid_states",
         params=dict(rows=2, cols=4, pattern=moe_route, cell=70, gap=18,
                     labels=("router picks top-2 experts per token",
                             "huge total params, small active params")),
         punch="MoE: pay for the whole model,",
         punch_sub="run a slice of it.  #MoE #LLM"),

    dict(day=37, file="day_37_nsight_systems_timeline.gif",
         title="nsight systems ~ timeline", subtitle="GPU starving",
         caption="The GPU sat idle, waiting for the CPU to feed it.",
         arch="timeline_track",
         params=dict(rows=[("CPU", 230), ("GPU", 330)], layout=nsight_layout,
                     note="More dataloader workers, prefetch, pinned memory."),
         punch="Your $30k GPU is bottlenecked",
         punch_sub="by a Python for-loop.  #Nsight #Profiling"),

    dict(day=38, file="day_38_tokenizer_surprise.gif",
         title="llm ~ tokenizer", subtitle="characters -> tokens",
         caption="You wrote a word. It bills you in tokens.",
         arch="type_reveal",
         params=dict(mode="tokenize", word="strawberry",
                     tokens=["st", "raw", "berry"]),
         punch="The model can't count the R's",
         punch_sub="because it never saw them.  #LLM #Tokenization"),

    dict(day=39, file="day_39_pinned_memory_transfer.gif",
         title="cuda ~ pinned memory", subtitle="one flag, free bandwidth",
         caption="Pageable memory copies through a staging buffer. Pinned doesn't.",
         arch="compare_paths",
         params=dict(top_label="pageable memory  (extra copy)",
                     bottom_label="pinned memory  (direct DMA + overlap)",
                     src="CPU", dst="GPU", top_speed=0.5, bottom_speed=1.0,
                     top_hops=[(0.5, "staging")],
                     note="pin_memory=True + cudaMemcpyAsync on a stream."),
         punch="Pinned memory:",
         punch_sub="one flag, free bandwidth.  #CUDA #GPU"),

    dict(day=40, file="day_40_context_window_overflow.gif",
         title="llm ~ context window", subtitle="the window moved on",
         caption="It only attends to what's inside the window.",
         arch="type_reveal",
         params=dict(mode="context_window"),
         punch="It didn't forget.",
         punch_sub="It never had it anymore.  #LLM #ContextWindow"),

    dict(day=41, file="day_41_cuda_graphs_capture.gif",
         title="cuda ~ graphs", subtitle="record once, replay",
         caption="Same 200 kernels every step? Capture them once.",
         arch="timeline_track",
         params=dict(rows=[("CPU", 230), ("GPU", 330)], layout=graphs_layout,
                     note="CUDA Graphs replay a whole sequence as one unit."),
         punch="Record once, replay forever.",
         punch_sub="Stop re-explaining the same 200 kernels.  #CUDA"),

    dict(day=42, file="day_42_nemo_curator_data.gif",
         title="nvidia ~ nemo curator", subtitle="garbage in, garbage model",
         caption="Model quality is downstream of data quality.",
         arch="flow_pipeline",
         params=dict(stages=[
             dict(label="raw web", color=g.RED),
             dict(label="dedup", color=g.AMBER),
             dict(label="quality", color=g.BLUE),
             dict(label="PII scrub", color=g.NVIDIA),
             dict(label="clean tokens", color=g.GREEN),
         ], packet_label="corpus", packet_color=g.AMBER,
             note="GPU-accelerated curation at web scale."),
         punch="Everyone loves training.",
         punch_sub="Nobody loves the cleaning that decides if it works.  #NeMo"),

    dict(day=43, file="day_43_gpu_thermal_throttle.gif",
         title="gpu ~ thermals", subtitle="sustained != peak",
         caption="A benchmark you can't sustain is a first impression.",
         arch="gauge_spike",
         params=dict(fn=throttle_curve, y_label="CLOCK", x_label="time under load",
                     color=g.GREEN, head_color=g.RED, threshold=0.9,
                     threshold_label="thermal limit", annot="throttle"),
         punch="Your benchmark was fast.",
         punch_sub="Your cooling wasn't.  #GPU #Performance"),

    dict(day=44, file="day_44_rag_retrieval.gif",
         title="ai ~ rag", subtitle="open-book exam",
         caption="Don't memorize. Look it up at query time.",
         arch="flow_pipeline",
         params=dict(stages=[
             dict(label="query", color=g.BLUE),
             dict(label="embed", color=g.PURPLE),
             dict(label="vector search", color=g.PURPLE),
             dict(label="context", color=g.AMBER),
             dict(label="grounded answer", color=g.GREEN),
         ], packet_label="question", packet_color=g.AMBER,
             note="Retrieval quality matters more than the LLM choice."),
         punch="RAG: giving your model",
         punch_sub="an open-book exam.  #RAG #LLM"),

    dict(day=45, file="day_45_warp_shuffle.gif",
         title="cuda ~ warp shuffle", subtitle="registers talking",
         caption="Reduce across a warp without touching memory.",
         arch="grid_states",
         params=dict(rows=1, cols=8, pattern=shuffle_tree, cell=64, gap=14,
                     index=True,
                     labels=("__shfl_down_sync: register-to-register",
                             "no shared memory, no __syncthreads()")),
         punch="Warp shuffle:",
         punch_sub="the reduction that never touches memory.  #CUDA #GPU"),

    dict(day=46, file="day_46_hallucination_confidence.gif",
         title="llm ~ hallucination", subtitle="fluent != correct",
         caption="It described a fake API with total confidence.",
         arch="type_reveal",
         params=dict(mode="confidence",
                     prompt='"Explain the foobar() function"  (foobar does not exist)'),
         punch="It's not lying.",
         punch_sub="It's autocompleting with confidence.  #LLM #AISafety"),

    dict(day=47, file="day_47_batch_size_sweet_spot.gif",
         title="gpu ~ batch size", subtitle="the knee, not the cliff",
         caption="Throughput climbs, then plateaus. Then OOM.",
         arch="gauge_spike",
         params=dict(fn=knee_curve, y_label="THROUGHPUT", x_label="batch size ->",
                     color=g.GREEN, head_color=g.GREEN, wall=0.9,
                     wall_label="OOM", annot="sweet spot"),
         punch="Bigger batch until it stops helping,",
         punch_sub="not until it stops fitting.  #GPU #Optimization"),

    dict(day=48, file="day_48_infiniband_topology.gif",
         title="hpc ~ ib fabric", subtitle="worst hop wins",
         caption="Balanced fat-tree, or a bottlenecked layer?",
         arch="ring_net",
         params=dict(mode="topology", n=8, dot_color=g.BLUE,
                     center_label=("fat-tree fabric", "high bisection bandwidth")),
         punch="Your cluster is only as fast",
         punch_sub="as its worst hop.  #InfiniBand #HPC"),

    dict(day=49, file="day_49_model_parallelism_types.gif",
         title="training ~ parallelism", subtitle="one axis isn't enough",
         caption="Too big for one GPU? Split it - on several axes at once.",
         arch="stack_build",
         params=dict(layers=[
             dict(label="Data parallel", sub="replicate + all-reduce", color=g.BLUE),
             dict(label="Tensor parallel", sub="split each layer", color=g.PURPLE),
             dict(label="Pipeline parallel", sub="layers across GPUs", color=g.AMBER),
             dict(label="3D parallelism + ZeRO/FSDP", sub="all at once", color=g.GREEN),
         ]),
         punch="Scaling laws are easy.",
         punch_sub="Scaling infrastructure is the job.  #DistributedTraining"),

    dict(day=50, file="day_50_triton_kernel.gif",
         title="gpu ~ triton", subtitle="kernels in python",
         caption="Kernel performance, Python ergonomics.",
         arch="bars_race",
         params=dict(bars=[
             dict(label="CUDA C++  -  lines of boilerplate", target=1.0,
                  color=g.BLUE, value="lots"),
             dict(label="Triton  -  lines of code", target=0.32,
                  color=g.GREEN, value="few", speed=1.3),
             dict(label="runtime performance", target=0.94,
                  color=g.NVIDIA, value="~on par", speed=1.1),
         ], note="Triton is what torch.compile generates under the hood."),
         punch="Triton: for when you want fast kernels",
         punch_sub="AND your weekend.  #Triton #CUDA"),

    dict(day=51, file="day_51_gemm_arithmetic_intensity.gif",
         title="gpu ~ roofline", subtitle="FLOPs per byte",
         caption="Small GEMMs are memory-bound before they reach peak FLOPs.",
         arch="gauge_spike",
         params=dict(fn=roofline_curve, y_label="ACHIEVED FLOPs",
                     x_label="arithmetic intensity ->", color=g.BLUE,
                     head_color=g.GREEN, threshold=0.9,
                     threshold_label="compute ceiling", annot="compute-bound"),
         punch="Small GEMMs don't fail at math.",
         punch_sub="They fail at feeding.  #GEMM #Roofline"),

    dict(day=52, file="day_52_checkpoint_save_load.gif",
         title="training ~ checkpointing", subtitle="save early, save often",
         caption="Node crashed at 80%. The checkpoint saved the run.",
         arch="gauge_spike",
         params=dict(fn=resume_curve, y_label="PROGRESS", x_label="time",
                     color=g.GREEN, head_color=g.GREEN, annot="resume"),
         punch="The only run that never crashes",
         punch_sub="is the one that already finished.  #MLOps #Training"),

    dict(day=53, file="day_53_nemotron_distillation.gif",
         title="nvidia ~ distillation", subtitle="inherit the smarts",
         caption="A big teacher trains a small, deployable student.",
         arch="flow_pipeline",
         params=dict(stages=[
             dict(label="teacher", sub="large model", color=g.PURPLE),
             dict(label="synth data", color=g.NVIDIA),
             dict(label="student", sub="small model", color=g.BLUE),
             dict(label="deploy", sub="fast + cheap", color=g.GREEN),
         ], packet_label="knowledge", packet_color=g.AMBER,
             note="Frontier-ish quality at a footprint you can serve widely."),
         punch="The student graduated smaller",
         punch_sub="AND faster than the teacher.  #Nemotron #Distillation"),

    dict(day=54, file="day_54_deadlock_multi_gpu.gif",
         title="nccl ~ deadlock", subtitle="everyone holds their breath",
         caption="Ranks reached different collectives. Now they wait forever.",
         arch="ring_net",
         params=dict(mode="deadlock"),
         punch="NCCL deadlock:",
         punch_sub="where your whole cluster holds its breath.  #NCCL #GPU"),

    dict(day=55, file="day_55_inference_server_scaling.gif",
         title="serving ~ scaling", subtitle="elastic, not heroic",
         caption="Traffic spiked. Replicas scaled. Latency stayed flat.",
         arch="bars_race",
         params=dict(bars=[
             dict(label="incoming requests", target=0.95, color=g.AMBER,
                  value="surge", speed=1.4),
             dict(label="model replicas (autoscaled)", target=0.8,
                  color=g.BLUE, value="scale out", speed=1.0),
             dict(label="p99 latency", target=0.3, color=g.GREEN,
                  value="held", speed=0.7),
         ], note="Continuous batching, load balancing, backpressure."),
         punch="Scaling inference:",
         punch_sub="the model was never the hard part.  #Inference #MLOps"),

    dict(day=56, file="day_56_debugging_nan_loss.gif",
         title="training ~ nan loss", subtitle="a gradient went to infinity",
         caption="The loss descended beautifully... then snapped to NaN.",
         arch="gauge_spike",
         params=dict(fn=nan_curve, y_label="LOSS", x_label="step",
                     color=g.BLUE, head_color=g.RED, annot="NaN"),
         punch="NaN loss:",
         punch_sub="'we need to talk about your learning rate.'  #DeepLearning"),

    dict(day=57, file="day_57_tensorrt_optimization.gif",
         title="nvidia ~ tensorrt", subtitle="same GPU, more speed",
         caption="Fuse layers, drop precision, autotune for this exact GPU.",
         arch="flow_pipeline",
         params=dict(stages=[
             dict(label="model", sub="eager graph", color=g.BLUE),
             dict(label="fuse layers", color=g.NVIDIA),
             dict(label="quantize", sub="INT8/FP8", color=g.NVIDIA),
             dict(label="autotune", color=g.AMBER),
             dict(label="engine", sub="faster", color=g.GREEN),
         ], packet_label="weights", packet_color=g.AMBER,
             note="In-flight batching + paged KV cache for LLMs."),
         punch="Training optimizes the weights.",
         punch_sub="TensorRT optimizes everything else.  #TensorRT #NVIDIA"),

    dict(day=58, file="day_58_precision_debugging.gif",
         title="training ~ precision", subtitle="emphasis on MIXED",
         caption="One numerically sensitive layer drifts. Keep it FP32.",
         arch="grid_states",
         params=dict(rows=1, cols=10, pattern=one_sensitive, cell=58, gap=10,
                     labels=("cast layers to FP16...",
                             "...but keep the sensitive reduction in FP32")),
         punch="Mixed precision:",
         punch_sub="emphasis on MIXED.  #MixedPrecision #GPU"),

    dict(day=59, file="day_59_agentic_tool_loop.gif",
         title="ai ~ agentic loop", subtitle="think, act, observe",
         caption="Reason, call a tool, observe, repeat.",
         arch="flow_pipeline",
         params=dict(stages=[
             dict(label="think", color=g.BLUE),
             dict(label="act", sub="tool call", color=g.PURPLE),
             dict(label="observe", color=g.AMBER),
         ], loop=True, packet_label="task", packet_color=g.GREEN,
             note="The intelligence impresses; the scaffolding makes it reliable."),
         punch="An agent is a while-loop",
         punch_sub="with good judgment (and a big API bill).  #AgenticAI"),

    dict(day=60, file="day_60_gpu_full_stack_journey.gif",
         title="the whole stack", subtitle="one thread -> a product",
         caption="Every AI product is a tower of optimizations.",
         arch="stack_build",
         params=dict(layers=[
             dict(label="CUDA thread", sub="one FMA", color=g.BLUE),
             dict(label="warp / SM", sub="32 lanes, latency hidden", color=g.BLUE),
             dict(label="GEMM + Tensor Cores", sub="the matmuls", color=g.PURPLE),
             dict(label="FlashAttention + KV cache", sub="efficient transformers", color=g.PURPLE),
             dict(label="quantize + compile", sub="cheap inference", color=g.AMBER),
             dict(label="NVLink + InfiniBand", sub="1000s of GPUs cooperate", color=g.NVIDIA),
             dict(label="NeMo + serving stack", sub="a thinking machine", color=g.GREEN),
         ]),
         punch="Every AI product is a tower of optimizations",
         punch_sub="- all the way down to a single warp.  #GPU #AI"),
]
