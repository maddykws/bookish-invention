# Multi-Modal Evidence Review System

Automated damage-claim verification. For each claim (a chat transcript + one or
more images + user history + minimum evidence requirements), the system decides
whether the images **support**, **contradict**, or give **not enough
information** for the claim — and produces a 14-column `output.csv`.

The images are the primary source of truth; the conversation defines what to
check; user history adds risk context but never overrides clear visual evidence.

## Architecture (Strategy B — multi-model cascade, one API spine)

```
Stage 0  Provider resolution + capability probe (graceful degradation)
Stage 1  Transcript -> structured claim          (Claude Haiku 4.5)
Stage 2  Local image preprocessing (no GPU)      blank/blur/FFT/EXIF/SHA-256, resize
Stage 3  Verdict (images primary)                Claude Opus 4.8 + prompt caching
Stage 3.6 Multi-vendor jury (escalated claims)   Gemini + Llama (free) [+ GPT-4o + Grok on hard cases]
Stage 4  Validate / consistency / repair / safe-defaults
Output   14-column output.csv (enforced contract)
```

Every model is reached through one OpenAI-compatible client. **Primary provider:
OpenRouter (one key).** If OpenRouter is unavailable, the pipeline falls back to
direct vendor APIs by key presence (Anthropic required; OpenAI/xAI/Google/Groq
optional) — the Opus verdict is preserved and consensus degrades safe.

## Setup (under 2 minutes)

```bash
pip install -r requirements.txt          # core — torch-free, runs the full pipeline
# optional, only with a CUDA GPU for local pre-checks:
#   pip install -r requirements-local.txt
# optional, for tracing / richer eval / demo UI:
#   pip install -r requirements-dev.txt
```

Set the API key (one variable):

```bash
export OPENROUTER_API_KEY=your_key_here        # or copy .env.example -> .env
```

## Run

```bash
python code/main.py                 # dataset/claims.csv -> output.csv
python code/evaluation/main.py      # Strategy A vs B + metrics -> code/evaluation/report.json
```

Useful flags:

```bash
python code/main.py --dry-run       # offline plumbing test: no network, no tokens, stub verdicts
python code/main.py --limit 5       # first 5 claims only
python code/main.py --input dataset/claims.csv --output output.csv
```

The dataset goes in `dataset/` (provided by the grader): `claims.csv`,
`sample_claims.csv`, `user_history.csv`, `evidence_requirements.csv`, and
`images/sample/` + `images/test/`.

## Run on Windows (PowerShell)

OpenRouter is reachable from your own machine without any network-allowlist
changes, so running locally is the simplest path:

```powershell
# 1. Python 3.11+
winget install Python.Python.3.11

# 2. clone + branch
git clone https://github.com/maddykws/bookish-invention.git
cd bookish-invention
git checkout claude/hackathon-system-design-jwts6w

# 3. venv
python -m venv .venv
Set-ExecutionPolicy -Scope Process -Bypass      # if activation is blocked
.\.venv\Scripts\Activate.ps1

# 4. deps
pip install -r requirements.txt

# 5. key (gitignored .env)
"OPENROUTER_API_KEY=sk-or-v1-your_key" | Out-File -Encoding ascii .env

# 6. run (place the dataset in dataset\ first)
python code\main.py
python code\evaluation\main.py
```

## Output schema (14 columns, fixed order)

`user_id, image_paths, user_claim, claim_object, evidence_standard_met,
evidence_standard_met_reason, risk_flags, issue_type, object_part, claim_status,
claim_status_justification, supporting_image_ids, valid_image, severity`

Booleans serialize lowercase (`true`/`false`); one row per input row.

## Layout

```
code/
  main.py                 OFFICIAL entry point -> output.csv
  config.py               single Config — every setting lives here
  providers.py            OpenRouter-first provider resolution + fallback
  capabilities.py         Stage-0 GPU/torch probe (graceful degradation)
  pipeline/               stages 1-4, models, loaders, prompts, orchestrator, writer
  utils/                  logger, injection screener, image utils, cache, LLM client
  evaluation/main.py      OFFICIAL eval entry point -> report.json
dataset/                  provided at runtime (gitignored)
requirements*.txt         core / optional-GPU / optional-dev
```

Design rationale and decisions: `SYSTEM_DESIGN.md`. Pre-run verification of the
real dataset's columns/paths: `DATASET_LANDING_CHECKLIST.md`.

## Environment variables

Only `OPENROUTER_API_KEY` is required. Optional direct-vendor fallback keys
(used only if OpenRouter is unavailable): `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`,
`XAI_API_KEY`, `GOOGLE_API_KEY`, `GROQ_API_KEY`. Never commit real keys — put
them in `.env` (gitignored).
