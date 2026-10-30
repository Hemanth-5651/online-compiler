
import os
import re
import time
from contextlib import asynccontextmanager

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
JUDGE0_URL = os.getenv(
    "JUDGE0_URL", "https://ce.judge0.com"
).rstrip("/")

MAX_SOURCE_BYTES = 1_000_000
MAX_STDIN_BYTES = 100_000
JOB_TTL_SECONDS = 600

LANGUAGES = {
    "python": 71,
    "c": 50,
    "cpp": 54,
    "java": 62,
    "javascript": 63,
}

# Local-development job registry.
# Replace with Redis before running multiple API workers.
jobs = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.http = httpx.AsyncClient(
        timeout=httpx.Timeout(10.0, connect=3.0),
        limits=httpx.Limits(
            max_connections=100,
            max_keepalive_connections=20,
        ),
    )
    yield
    await app.state.http.aclose()


app = FastAPI(title="CodeLab Compiler", lifespan=lifespan)

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(
    RateLimitExceeded, _rate_limit_exceeded_handler
)

app.mount(
    "/static",
    StaticFiles(directory=os.path.join(BASE_DIR, "static")),
    name="static",
)


class RunRequest(BaseModel):
    language: str
    code: str
    stdin: str = ""

    @field_validator("language")
    @classmethod
    def validate_language(cls, value):
        value = value.lower()
        if value not in LANGUAGES:
            raise ValueError("Unsupported language")
        return value

    @field_validator("code")
    @classmethod
    def validate_code(cls, value):
        if not value.strip():
            raise ValueError("Code cannot be empty")
        if len(value.encode("utf-8")) > MAX_SOURCE_BYTES:
            raise ValueError("Source code exceeds 1 MB")
        return value

    @field_validator("stdin")
    @classmethod
    def validate_stdin(cls, value):
        if len(value.encode("utf-8")) > MAX_STDIN_BYTES:
            raise ValueError("Standard input exceeds 100 KB")
        return value


def cleanup_jobs():
    now = time.time()
    expired = [
        key for key, job in jobs.items()
        if now - job["created_at"] > JOB_TTL_SECONDS
    ]
    for key in expired:
        jobs.pop(key, None)


@app.get("/")
async def home():
    return FileResponse(
        os.path.join(BASE_DIR, "static", "index.html")
    )


@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.post("/api/run")
@limiter.limit("5/minute")
async def run_code(request: Request, body: RunRequest):
    cleanup_jobs()

    payload = {
        "language_id": LANGUAGES[body.language],
        "source_code": body.code,
        "stdin": body.stdin,
        "cpu_time_limit": 2,
        "cpu_extra_time": 1,
        "wall_time_limit": 5,
        "memory_limit": 128000 if body.language == "java" else 65536,
        "stack_limit": 64000 if body.language == "java" else 16384,
        "max_processes_and_or_threads": 60 if body.language == "java" else 20,
        "enable_network": False,
    }

    try:
        response = await request.app.state.http.post(
            f"{JUDGE0_URL}/submissions",
            params={"wait": "false", "base64_encoded": "false"},
            json=payload,
        )
        response.raise_for_status()
        data = response.json()

    except httpx.HTTPError:
        raise HTTPException(
            status_code=502,
            detail="Execution service unavailable",
        )

    token = data.get("token")
    if not token:
        raise HTTPException(
            status_code=502,
            detail="Execution service returned no job token",
        )

    jobs[token] = {
        "created_at": time.time(),
        "language": body.language,
    }

    return {"token": token, "status": "Queued"}


@app.get("/api/result/{token}")
@limiter.limit("60/minute")
async def get_result(request: Request, token: str):
    cleanup_jobs()

    if not re.fullmatch(r"[a-fA-F0-9-]{16,64}", token):
        raise HTTPException(status_code=400, detail="Invalid job token")

    if token not in jobs:
        raise HTTPException(
            status_code=404,
            detail="Job not found or expired",
        )

    try:
        response = await request.app.state.http.get(
            f"{JUDGE0_URL}/submissions/{token}",
            params={
                "base64_encoded": "false",
                "fields": (
                    "token,status,stdout,stderr,compile_output,"
                    "message,time,memory"
                ),
            },
        )
        response.raise_for_status()
        result = response.json()

    except httpx.HTTPError:
        raise HTTPException(
            status_code=502,
            detail="Could not retrieve execution result",
        )

    status_data = result.get("status") or {}
    status_id = status_data.get("id")

    if status_id in (1, 2):
        return {
            "status": status_data.get("description", "Processing"),
            "done": False,
        }

    # Results are returned only after execution has finished.
    # This local registry is not durable across restarts.
    jobs.pop(token, None)

    return {
        "done": True,
        "status": status_data.get("description", "Unknown"),
        "stdout": result.get("stdout") or "",
        "stderr": result.get("stderr") or "",
        "compile_output": result.get("compile_output") or "",
        "message": result.get("message") or "",
        "time": result.get("time"),
        "memory": result.get("memory"),
    }
