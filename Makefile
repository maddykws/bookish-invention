.PHONY: install pipeline agent ui eval ab self-eval checklist judge-tools

# First-time setup
install:
	pip install -r requirements.txt

# On question drop: research + scaffold (run this first)
pipeline:
	python runner.py

# Run agent on a task
# Usage: make agent TASK="your task here"
agent:
	python agent/agent.py --input "$(TASK)"

# Gradio demo UI → localhost:7860
ui:
	python agent/ui.py

# DeepEval scores for active prompt version
eval:
	python agent/eval.py

# A/B compare two prompt versions
# Usage: make ab VA=1.0 VB=2.0
ab:
	python agent/eval.py --ab $(VA) $(VB)

# AI judge simulation (run before submitting)
# Usage: make self-eval DESC="describe what you built"
self-eval:
	python runner.py --self-eval --agent-description "$(DESC)"

# Pre-submission checklist
checklist:
	python runner.py --checklist

# Show pre-researched judge tools
judge-tools:
	python runner.py --judge-tools-only
