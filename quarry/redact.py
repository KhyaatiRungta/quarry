"""Redaction - keep secrets out of traces, logs, and run files."""

import os


def redact(text: str) -> str:
    """Replace the OpenRouter API key (and bare sk-or tokens) with ***."""
    if not text:
        return text

    key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if key and len(key) > 8:
        text = text.replace(key, "***REDACTED***")

    # Fallback: any sk-or-v1-... token shape
    parts = text.split("sk-or-")
    if len(parts) > 1:
        rebuilt = parts[0] + "sk-or-***"
        for seg in parts[1:]:
            # keep at most 4 chars of entropy visible, cut at next separator
            cut = min((i for i, ch in enumerate(seg) if ch in " \t\n\"',);:"), default=8)
            rebuilt += seg[:cut].replace(seg[:cut], "***") + seg[cut:]
            rebuilt = rebuilt.replace("***", "***", 1)
        text = rebuilt

    return text


def redact_structure(obj):
    """Recursively redact strings inside dicts / lists (for trace JSON)."""
    if isinstance(obj, str):
        return redact(obj)
    if isinstance(obj, dict):
        return {k: redact_structure(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [redact_structure(v) for v in obj]
    return obj
