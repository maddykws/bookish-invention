.PHONY: install pipeline agent ui eval ab test diagram self-eval checklist judge-tools

# ── Setup ─────────────────────────────────────────────────────────────────────
install:
	pip install -r requirements.txt

# ── On question drop: run this first ─────────────────────────────────────────
pipeline:
	python runner.py

# ── Build loop ────────────────────────────────────────────────────────────────

# Run agent on a task.  Usage: make agent TASK="your task here"
agent:
	python agent/agent.py --input "$(TASK)"

# Gradio streaming demo → localhost:7860
ui:
	python agent/ui.py

# Pytest — run after filling in agent/tools.py TODO blocks
# Target: all tests green before submitting
test:
	pytest tests/ -v --tb=short

# DeepEval scores for active prompt version
eval:
	python agent/eval.py

# A/B compare two prompt versions.  Usage: make ab VA=1.0 VB=2.0
ab:
	python agent/eval.py --ab $(VA) $(VB)

# Regenerate Mermaid architecture diagram → docs/architecture.md
diagram:
	python modules/diagram.py

# ── Pre-submission ────────────────────────────────────────────────────────────

# AI judge simulation.  Usage: make self-eval DESC="describe what you built"
self-eval:
	python runner.py --self-eval --agent-description "$(DESC)"

# Pre-submission checklist
checklist:
	python runner.py --checklist

# ── Utilities ─────────────────────────────────────────────────────────────────

# Show pre-researched judge tools
judge-tools:
	python runner.py --judge-tools-only
