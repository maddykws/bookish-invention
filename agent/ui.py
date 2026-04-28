"""
Gradio Demo UI
Run: python agent/ui.py   →   opens at http://localhost:7860

Shows: input box, answer, reasoning, confidence, tools used, cost, latency.
Phoenix traces link is displayed in the sidebar.
Use this during the AI judge session — live demo beats CLI output every time.
"""

import sys
import asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()


def _run_agent(task: str):
    """Synchronous wrapper for the async agent — called by Gradio."""
    if not task.strip():
        return "", "", "", "", ""

    from agent.agent import main as run_agent, setup_observability
    setup_observability()

    try:
        result, run_meta = asyncio.run(run_agent(task, return_meta=True))
    except TypeError:
        # fallback if return_meta not supported yet
        result = asyncio.run(run_agent(task))
        run_meta = {}

    answer        = result.answer
    reasoning     = result.reasoning
    confidence    = f"{result.confidence:.0%}"
    tools         = ", ".join(result.tools_used) if result.tools_used else "none"
    cost_latency  = (
        f"Tokens in: {run_meta.get('tokens_in', '—')} | "
        f"Tokens out: {run_meta.get('tokens_out', '—')} | "
        f"Est. cost: ${run_meta.get('cost_usd', 0):.4f} | "
        f"Latency: {run_meta.get('latency_ms', '—')}ms | "
        f"Reflections: {run_meta.get('reflections', '—')}"
    )

    return answer, reasoning, confidence, tools, cost_latency


def build_ui():
    try:
        import gradio as gr
    except ImportError:
        print("Gradio not installed. Run: pip install gradio")
        sys.exit(1)

    with gr.Blocks(
        title="HackerRank Orchestrate — Agent Demo",
        theme=gr.themes.Soft(),
        css=".footer { display: none !important; }",
    ) as demo:

        gr.Markdown(
            "# Agent Demo\n"
            "**HackerRank Orchestrate Hackathon**  \n"
            "Traces visible at [localhost:6006](http://localhost:6006) (Arize Phoenix)"
        )

        with gr.Row():
            with gr.Column(scale=2):
                task_input = gr.Textbox(
                    label="Task / Question",
                    placeholder="Enter the task for the agent...",
                    lines=4,
                )
                run_btn = gr.Button("Run Agent", variant="primary")

            with gr.Column(scale=1):
                gr.Markdown(
                    "### Stack\n"
                    "- **Framework**: PydanticAI\n"
                    "- **Observability**: Arize Phoenix\n"
                    "- **Prompts**: Langfuse + local YAML\n"
                    "- **Self-correction**: Reflexion (3 loops)\n"
                    "- **Eval**: DeepEval\n"
                    "- **Memory**: ChromaDB\n"
                )

        with gr.Row():
            answer_out = gr.Textbox(label="Answer", lines=6, interactive=False)

        with gr.Accordion("Reasoning", open=False):
            reasoning_out = gr.Textbox(label="Agent Reasoning", lines=4, interactive=False)

        with gr.Row():
            confidence_out = gr.Textbox(label="Confidence", interactive=False, scale=1)
            tools_out      = gr.Textbox(label="Tools Used", interactive=False, scale=2)

        metrics_out = gr.Textbox(label="Cost & Latency", interactive=False)

        run_btn.click(
            fn=_run_agent,
            inputs=[task_input],
            outputs=[answer_out, reasoning_out, confidence_out, tools_out, metrics_out],
        )

        task_input.submit(
            fn=_run_agent,
            inputs=[task_input],
            outputs=[answer_out, reasoning_out, confidence_out, tools_out, metrics_out],
        )

    return demo


if __name__ == "__main__":
    ui = build_ui()
    ui.launch(server_name="0.0.0.0", server_port=7860, share=False)
