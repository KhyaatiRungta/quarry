# Getting Started

## Prerequisites

- Python 3.11+
- OpenRouter API key (optional for offline mode)

## Installation

`ash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
`

## Quick Start

`ash
python -m quarry ask --offline --data data/sales.csv "Which region has the highest revenue?"
`
"@ | Out-File "docs\getting-started.md" -Encoding utf8
@"
# Configuration

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| OPENROUTER_API_KEY | API key for OpenRouter | (required for live mode) |
| QUARRY_MODEL | Model to use | anthropic/claude-3.5-sonnet |

## .env Setup

Copy .env.example to .env and fill in your API key.
