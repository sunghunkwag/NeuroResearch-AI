# NeuroResearch-AI

**Advanced Multi-Agent Research System with Local/Cloud Modes, Meta-Learning, and AutoGen (Beta)**

[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Overview

NeuroResearch-AI is an enhanced research automation system building on Denario, now with:
- **Local/Offline Mode**: Run fully offline with no API keys (heuristic engine)
- **Multi-LLM Support**: OpenAI, Anthropic, Google (keys optional; see Matrix)
- **Meta-Learning**: ε-greedy bandit to optimize prompts/params over iterations
- **AutoGen Integration (Beta)**: Minimal agent-pair conversations to inform methodology
- **Expanded Workflow**: All 8 agent roles participate with role prompts
- **Dynamic Quality Assessment**: Heuristic multi-metric scoring; stops when threshold met

> Note: AutoGen integration is marked Beta. When API keys are not provided, it falls back to local mock conversations.

## Capability Matrix

| Capability | Local Mode (No Keys) | Cloud (OpenAI) | Cloud (Anthropic) | Cloud (Google) |
|---|---|---|---|---|
| Research pipeline | ✅ | ✅ | ✅ | ✅ |
| LLM provider | ❌ (heuristic engine) | ✅ | ✅ | ✅ |
| AutoGen conversations | ✅ (mock) | ✅ | ✅ | ✅ |
| Meta-learning | ✅ | ✅ | ✅ | ✅ |
| Quality assessment | ✅ | ✅ | ✅ | ✅ |

## Installation

```bash
git clone https://github.com/sunghunkwag/NeuroResearch-AI.git
cd NeuroResearch-AI
pip install -e .
```

Optional providers:
```bash
pip install langchain-openai langchain-anthropic langchain-google-genai
pip install autogen  # optional (beta)
```

## Quick Start

### A) Local/Offline (no API keys)
```bash
neuroresearch create-project myproj
neuroresearch conduct-research ./myproj --question "How to optimize attention for long sequences?" --local
```

### B) Cloud with OpenAI (example)
```bash
neuroresearch conduct-research ./myproj --question "..." --provider openai
# prompts for keys will be shown
```

### C) Mixed providers + AutoGen (beta)
```bash
neuroresearch conduct-research ./myproj --question "..." --provider mixed --use-autogen
```

Key options:
- `--quality-threshold` (default 0.75)
- `--max-iterations` (default 5)

## Python Usage

```python
from neuroresearch_ai import NeuroResearchAI, ResearchContext, ResearchDomain

context = ResearchContext(
    project_id="demo",
    domain=ResearchDomain.AI_ML,
    research_question="How can we improve attention for 100k token sequences?",
)

# Local (no keys)
system = NeuroResearchAI(
    project_dir="./demo",
    research_context=context,
    use_local_mode=True,
    quality_threshold=0.75,
    max_iterations=5,
)
results = await system.conduct_research(context.research_question)
print(results["quality_score"])  # ~0.8+

# Cloud (OpenAI)
system = NeuroResearchAI(
    project_dir="./demo",
    research_context=context,
    api_keys={"openai": "YOUR_KEY"},
    use_local_mode=False,
)
```

## What’s Implemented (Claim vs Reality aligned)

- **Multi-LLM selection**: OpenAI/Anthropic/Google branches; keys validated, clear errors if missing
- **AutoGen (Beta)**: Two agent-pair loops (Literature↔Synthesis, Review↔Methodology) with mock fallback
- **Workflow**: DIRECTOR → LITERATURE_SCOUT → CROSS_DOMAIN_SYNTHESIZER → METHODOLOGY_DESIGNER → DATA_ANALYST → PEER_REVIEWER → ETHICS_GUARDIAN → PUBLICATION_EXPERT
- **Prompts**: Role prompts externalized to `templates/prompts/*.md` (auto-created on first run)
- **Quality**: Heuristic metrics (novelty, rigor, reproducibility, ethics, cross-domain, stats)
- **Meta-learning**: ε-greedy bandit optimizing temperature/tokens/prompt style

## Project Structure

```
neuroresearch_ai/
  core.py                 # local/cloud, multi-LLM, AutoGen, meta-learning
  cli.py                  # --local, --use-autogen, --provider, thresholds
  templates/prompts/*.md  # role-specific prompts (auto-generated)
  ...
```

## Troubleshooting
- Running cloud without keys → use `--local` or provide the required provider key
- Missing prompt files → created automatically; fallback exists
- AutoGen not installed → Beta features degrade to mock conversation

## Contributing
- Areas: stronger AutoGen workflows, richer metrics, provider plugins, prompt packs, tests
- PRs welcome. Please run: `black`, `isort`, `flake8`, `pytest`

## License
MIT
