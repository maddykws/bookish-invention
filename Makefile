.PHONY: install pipeline agent eval self-eval checklist judge-tools

# First-time setup
install:
	pip install -r requirements.txt

# On question drop: run this first (research + scaffold)
pipeline:
	python runner.py

# Run agent on a task (set TASK="your task here")
agent:
	python agent/agent.py --input "$(TASK)"

# Get evaluation scores (run after implementing tools)
eval:
	python agent/eval.py

# AI judge simulation (run before submitting)
self-eval:
	python runner.py --self-eval --agent-description "$(DESC)"

# Pre-submission checklist
checklist:
	python runner.py --checklist

# Show pre-researched judge tools
judge-tools:
	python runner.py --judge-tools-only
