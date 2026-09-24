"""FastAPI web interface for Quarry."""

import base64
import io
import tempfile
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles

from quarry.agent import MAX_STEPS_MSG, QuarryAgent
from quarry.config import ConfigError, load_settings
from quarry.llm import LLMError
from quarry.redact import redact, redact_structure

MAX_UPLOAD_BYTES = 5 * 1024 * 1024  # 5 MB

app = FastAPI(title="Quarry API", version="1.0.0")


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/ask")
def ask(question: str = Form(...), file: UploadFile = File(...)):
    """Answer a question about an uploaded CSV.

    Declared sync on purpose: FastAPI runs sync endpoints in a worker
    thread, keeping the event loop free.
    """
    # --- validation ---
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are supported")

    question = question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question must not be empty")

    payload = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large (max 5 MB)")

    try:
        pd.read_csv(io.BytesIO(payload))
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not parse CSV: {exc}") from exc

    try:
        settings = load_settings(require_key=True)
    except ConfigError as exc:
        raise HTTPException(status_code=500, detail=f"Server config error: {exc}") from exc

    # --- run the agent in an isolated temp workspace ---
    with tempfile.TemporaryDirectory() as tmp:
        dataset = Path(tmp) / "data.csv"
        dataset.write_bytes(payload)
        artifacts_dir = Path(tmp) / "artifacts"
        artifacts_dir.mkdir()

        try:
            agent = QuarryAgent(
                dataset_path=str(dataset),
                max_steps=settings.max_steps,
                max_repairs=settings.max_repairs,
                artifacts_dir=str(artifacts_dir),
            )
            answer = agent.ask(question)
        except LLMError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ConfigError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        charts = [
            {
                "name": png.name,
                "data_base64": base64.b64encode(png.read_bytes()).decode(),
            }
            for png in sorted(artifacts_dir.glob("*.png"))
        ]

    grounded = answer != MAX_STEPS_MSG
    return {
        "question": question,
        "answer": redact(answer),
        "grounded": grounded,
        "steps": len(agent.trace),
        "repairs": {
            "recovered": agent.repair_count,
            "attempts": agent.repair_attempts,
        },
        "tokens": {
            "input": agent.client.tokens_in,
            "output": agent.client.tokens_out,
        },
        "trace": redact_structure(agent.trace),
        "charts": charts,
    }


# Static frontend - mounted last so /api/* routes win
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
