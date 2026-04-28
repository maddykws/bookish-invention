"""
Gradio Streaming Demo UI
Run: python agent/ui.py   →   localhost:7860

Features:
  - Streaming token output (feels alive, not frozen)
  - Answer + reasoning + confidence + tools used
  - Cost & latency metrics per run
  - Architecture diagram in sidebar
  - Phoenix link for live traces
  - Mode indicator (multi-agent vs single)
"""

import sys
import asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()


def _load_diagram_md() -> str:
    docs = Path(__file__).parent.parent / "docs" / "architecture.md"
    if docs.exists():
        return docs.read_text()
    return "_Run `make diagram` to generate the architecture diagram._"


def _run_agent_streaming(task: str):
    """
    Generator that yields streamed tokens then final metadata.
    Gradio calls this as a generator for real-time streaming.
    """
    if not task.strip():
        yield "", "", "", "", ""
        return

    import anthropic
    import yaml
    import os
    from agent.config import (
        MODEL, ACTIVE_PROMPT_VERSION, PROMPTS_DIR,
        VALIDATOR_ENABLED, MULTI_AGENT_MODE, HUMAN_IN_THE_LOOP,
    )
    from agent.agent import setup_observability

    setup_observability()

    # Load prompt
    prompt_file = PROMPTS_DIR / f"system_v{ACTIVE_PROMPT_VERSION}.yaml"
    if prompt_file.exists():
        system_prompt = yaml.safe_load(prompt_file.read_text())["prompt"]
    else:
        system_prompt = "You are a helpful AI agent."

    # Stream the answer token-by-token using Anthropic streaming
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    streamed_answer = ""
    mode_label = "multi-agent" if MULTI_AGENT_MODE else "single"
    flags = f"validator={'on' if VALIDATOR_ENABLED else 'off'} | HITL={'on' if HUMAN_IN_THE_LOOP else 'off'}"

    yield f"[{mode_label} | {flags}] Thinking...", "", "", "", ""

    import time
    t0 = time.monotonic()

    try:
        with client.messages.stream(
            model=MODEL,
            max_tokens=1500,
            system=system_prompt,
            messages=[{"role": "user", "content": task}],
        ) as stream:
            for text in stream.text_stream:
                streamed_answer += text
                yield streamed_answer, "", "", "", ""

        latency_ms = int((time.monotonic() - t0) * 1000)

        # After streaming, run full pipeline for structured output + metrics
        import asyncio
        from agent.agent import main as run_agent
        result, run_meta = asyncio.run(run_agent(task, return_meta=True))

        reasoning     = result.reasoning
        confidence    = f"{result.confidence:.0%}"
        tools         = ", ".join(result.tools_used) if result.tools_used else "none"
        metrics       = (
            f"Mode: {run_meta.get('mode', mode_label)} | "
            f"Latency: {latency_ms}ms | "
            f"Reflections: {run_meta.get('reflections', '—')} | "
            f"Score: {run_meta.get('final_score', 0):.2f} | "
            f"Est. cost: ${run_meta.get('cost_usd', 0):.4f}"
        )

        yield result.answer, reasoning, confidence, tools, metrics

    except Exception as e:
        yield f"Error: {e}", "", "", "", ""


def build_ui():
    try:
        import gradio as gr
    except ImportError:
        print("Gradio not installed. Run: pip install gradio")
        sys.exit(1)

    diagram_md = _load_diagram_md()

    with gr.Blocks(
        title="HackerRank Orchestrate — Agent Demo",
        theme=gr.themes.Soft(),
    ) as demo:

        gr.Markdown(
            "# Agent Demo — HackerRank Orchestrate\n"
            "Multi-agent system with Reflexion, Meta-Validator, and ChromaDB memory.  \n"
            "Traces: [localhost:6006](http://localhost:6006) (Arize Phoenix)"
        )

        with gr.Row():
            # ── Left: input + output ─────────────────────────────────────────
            with gr.Column(scale=3):
                task_input = gr.Textbox(
                    label="Task / Question",
                    placeholder="Enter the task for the agent...",
                    lines=3,
                )
                run_btn = gr.Button("Run Agent", variant="primary")

                answer_out = gr.Textbox(
                    label="Answer (streaming)",
                    lines=8,
                    interactive=False,
                )

                with gr.Accordion("Reasoning", open=False):
                    reasoning_out = gr.Textbox(
                        label="Agent Reasoning",
                        lines=4,
                        interactive=False,
                    )

                with gr.Row():
                    confidence_out = gr.Textbox(label="Confidence", interactive=False, scale=1)
                    tools_out      = gr.Textbox(label="Tools Used",  interactive=False, scale=3)

                metrics_out = gr.Textbox(label="Performance Metrics", interactive=False)

            # ── Right: architecture diagram ──────────────────────────────────
            with gr.Column(scale=2):
                gr.Markdown("### Architecture")
                gr.Markdown(diagram_md)

        # Wire up streaming
        run_btn.click(
            fn=_run_agent_streaming,
            inputs=[task_input],
            outputs=[answer_out, reasoning_out, confidence_out, tools_out, metrics_out],
        )
        task_input.submit(
            fn=_run_agent_streaming,
            inputs=[task_input],
            outputs=[answer_out, reasoning_out, confidence_out, tools_out, metrics_out],
        )

    return demo


if __name__ == "__main__":
    ui = build_ui()
    ui.launch(server_name="0.0.0.0", server_port=7860, share=False)
