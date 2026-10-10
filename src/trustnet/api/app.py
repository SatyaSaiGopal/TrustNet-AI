"""FastAPI web service for TrustNet AI.

Serves the REST API under /api and the static frontend under /.
Includes multi-domain support, model switching, and database seeding.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import httpx
from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ValidationError

from trustnet.agents import ANALYST, DEFAULT_ROLES, RESEARCHER, SKEPTIC, LLMAgent, build_agents
from trustnet.config import get_settings
from trustnet.evaluation import run_benchmark
from trustnet.evaluation.demo import Demo, seed_demo
from trustnet.evaluation.showcase import SHOWCASE_SCENARIOS, ShowcaseScenario
from trustnet.evaluation.synthetic import Case, SimulatedAgent
from trustnet.pipeline import (
    TaskEvaluation,
    build_engine,
    build_evaluator,
    build_policy,
    evaluate_task,
    record_outcome,
)
from trustnet.providers import (
    OpenAIProvider,
    ProviderConfigurationError,
    ProviderError,
    create_provider,
)
from trustnet.providers.ollama import OllamaProvider
from trustnet.schemas import Outcome, Task, TrustSignals
from trustnet.storage import (
    DuplicateOutcomeError,
    Repository,
    connect,
    initialize,
)
from trustnet.storage.seeder import seed_live_database

STATIC_DIR = Path(__file__).resolve().parent.parent.parent.parent / "static"
SOURCES_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data" / "sources"

# Runtime selected active model (defaults to llama3.2:1b)
runtime_active_model: str = "llama3.2:1b"


class DemoState:
    """Persistent in-memory state for synthetic demonstrations."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.connection = connect(":memory:", check_same_thread=False)
        initialize(self.connection)
        self.repository = Repository(self.connection)
        self.settings = get_settings()
        self.engine = build_engine(self.settings)
        self.policy = build_policy(self.settings)
        self.demo = seed_demo(self.repository, self.engine)
        self.cases_by_id: dict[str, Case] = {
            case.task.task_id: case for case in self.demo.cases
        }


demo_state = DemoState()


def get_live_repository() -> Repository:
    settings = get_settings()
    settings.database_path.parent.mkdir(parents=True, exist_ok=True)
    connection = connect(str(settings.database_path), check_same_thread=False)
    initialize(connection)
    repo = Repository(connection)
    # Auto-seed if database is completely empty
    if not repo.tasks():
        seed_live_database(repo, settings)
    return repo


app = FastAPI(
    title="TrustNet AI",
    description="AI Agent Trust Network API",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class EvaluateRequest(BaseModel):
    question: str = Field(min_length=1, max_length=10_000)
    category: str = "general"
    options: list[str] | None = None
    mode: Literal["synthetic", "live"] = "synthetic"
    task_id: str | None = None
    model: str | None = None
    ensemble: bool = False
    provider: Literal["ollama", "openai"] | None = None
    api_key: str | None = None
    base_url: str | None = None


class RecordOutcomeRequest(BaseModel):
    task_id: str = Field(min_length=1)
    correct_answer: str | None = None
    resolution_source: str = Field(min_length=1)
    mode: Literal["synthetic", "live"] = "synthetic"


class SelectModelRequest(BaseModel):
    model: str = Field(min_length=1)


@app.get("/api/status")
def get_status() -> dict[str, Any]:
    """Backend connection status and runtime configuration."""
    settings = get_settings()
    return {
        "status": "connected",
        "version": "0.1.0",
        "environment": settings.environment,
        "ollama_base_url": settings.ollama_base_url,
        "ollama_model": runtime_active_model or settings.ollama_model,
        "openai_configured": bool(settings.openai_api_key),
        "openai_model": settings.openai_model,
        "default_provider": settings.llm_provider,
        "database_path": str(settings.database_path),
        "min_samples": settings.trust_min_samples,
    }


@app.get("/api/providers")
def get_providers() -> dict[str, Any]:
    """Return available LLM provider integrations and presets."""
    settings = get_settings()
    return {
        "configured_provider": settings.llm_provider,
        "has_openai_key": bool(settings.openai_api_key),
        "presets": [
            {
                "id": "openai-gpt4omini",
                "name": "OpenAI — GPT-4o Mini (Recommended)",
                "provider": "openai",
                "model": "gpt-4o-mini",
                "base_url": "https://api.openai.com/v1",
                "note": "Fast, high-precision JSON schema following",
            },
            {
                "id": "openai-gpt4o",
                "name": "OpenAI — GPT-4o (Frontier Model)",
                "provider": "openai",
                "model": "gpt-4o",
                "base_url": "https://api.openai.com/v1",
                "note": "Highest capability frontier reasoning",
            },
            {
                "id": "groq-llama33",
                "name": "Groq — LLaMA 3.3 70B Versatile",
                "provider": "openai",
                "model": "llama-3.3-70b-versatile",
                "base_url": "https://api.groq.com/openai/v1",
                "note": "Ultra-fast inference (hundreds of tok/sec)",
            },
            {
                "id": "openrouter",
                "name": "OpenRouter — Meta LLaMA 3.3 70B",
                "provider": "openai",
                "model": "meta-llama/llama-3.3-70b-instruct",
                "base_url": "https://openrouter.ai/api/v1",
                "note": "Multi-provider unified API router",
            },
            {
                "id": "ollama-local",
                "name": f"Local Ollama — {runtime_active_model or 'llama3.2:1b'}",
                "provider": "ollama",
                "model": runtime_active_model or "llama3.2:1b",
                "base_url": settings.ollama_base_url,
                "note": "100% private, on-device offline inference",
            },
        ],
    }


@app.get("/api/models")
def get_installed_models() -> dict[str, Any]:
    """Query local Ollama instance for installed models."""
    settings = get_settings()
    installed = []
    try:
        resp = httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=5.0)
        if resp.status_code == 200:
            data = resp.json()
            raw_models = [m.get("name") for m in data.get("models", []) if m.get("name")]
            # Filter out embedding models or raw hashes for clean display
            installed = sorted(list(set(
                m for m in raw_models if not m.startswith("nomic-embed") and not m.startswith("llamacpp:")
            )))
    except Exception:
        installed = ["llama3.2:1b", "gemma2:2b"]

    return {
        "installed_models": installed,
        "active_model": runtime_active_model,
        "recommended": ["llama3.2:1b", "gemma2:2b"],
    }


@app.post("/api/models/select")
def select_model(req: SelectModelRequest) -> dict[str, Any]:
    """Switch the active local LLM model for live agents."""
    global runtime_active_model
    runtime_active_model = req.model.strip()
    return {
        "status": "success",
        "active_model": runtime_active_model,
        "message": f"Active model set to '{runtime_active_model}'",
    }


@app.get("/api/showcase/cases")
def get_showcase_cases() -> list[dict[str, Any]]:
    """Return curated enterprise showcase scenarios across 7 mission-critical domains."""
    return [s.to_dict() for s in SHOWCASE_SCENARIOS]


@app.post("/api/database/seed")
def trigger_database_seed() -> dict[str, Any]:
    """Seed or refresh the live database with multi-domain verified outcomes."""
    settings = get_settings()
    repo = get_live_repository()
    return seed_live_database(repo, settings)


@app.get("/api/demo/cases")
def get_demo_cases() -> list[dict[str, Any]]:
    """List of preloaded synthetic benchmark tasks for demonstration."""
    # Combine synthetic cases and showcase scenarios for rich selection
    cases = []
    for s in SHOWCASE_SCENARIOS:
        cases.append({
            "task_id": s.task_id,
            "question": s.question,
            "category": s.domain,
            "options": list(s.options),
            "is_answerable": s.is_answerable,
            "correct_answer": s.correct_answer,
            "domain_title": s.title,
            "reference": s.reference,
            "is_showcase": True,
        })
    for case in demo_state.demo.cases:
        cases.append({
            "task_id": case.task.task_id,
            "question": case.task.question,
            "category": case.task.category,
            "options": list(case.task.options),
            "is_answerable": case.outcome.answerable,
            "correct_answer": case.outcome.correct_answer,
            "domain_title": f"Benchmark: {case.task.category.title()}",
            "reference": "",
            "is_showcase": False,
        })
    return cases


@app.post("/api/task/evaluate")
def evaluate(request: EvaluateRequest) -> dict[str, Any]:
    """Execute agents, verify citations, score trust, and produce a decision."""
    clean_question = request.question.strip()
    if not clean_question:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Question cannot be empty."
        )

    settings = get_settings()
    engine = build_engine(settings)
    policy = build_policy(settings)
    chosen_model = request.model or runtime_active_model or settings.ollama_model or "llama3.2:1b"

    if request.mode == "synthetic":
        # Check showcase scenarios
        showcase_match = next((s for s in SHOWCASE_SCENARIOS if s.task_id == request.task_id), None)
        if showcase_match:
            task = Task(
                task_id=showcase_match.task_id,
                question=clean_question,
                category=showcase_match.domain,
                options=showcase_match.options,
                is_synthetic=True,
            )
            # Register showcase outcome in synthetic demo agents
            outcome = Outcome(
                task_id=task.task_id,
                correct_answer=showcase_match.correct_answer,
                resolution_source=showcase_match.resolution_source,
            )
            for agent in demo_state.demo.agents:
                agent._outcomes[task.task_id] = outcome
            if hasattr(demo_state.demo.evaluator._sources, "_sources") and showcase_match.reference:
                demo_state.demo.evaluator._sources._sources[f"synthetic://{task.task_id}"] = showcase_match.excerpt
        else:
            case = demo_state.cases_by_id.get(request.task_id) if request.task_id else None
            if case:
                task = Task(
                    task_id=case.task.task_id,
                    question=clean_question,
                    category=case.task.category,
                    options=case.task.options,
                    equivalence_group=case.task.equivalence_group,
                    is_synthetic=True,
                )
            else:
                task_id = request.task_id or f"synthetic-adhoc-{uuid4().hex[:8]}"
                category = request.category if request.category in ("ledger", "inventory") else "ledger"
                options = (
                    tuple(request.options)
                    if request.options and len(request.options) >= 2
                    else ("1000", "2000", "3000", "4000")
                )
                task = Task(
                    task_id=task_id,
                    question=clean_question,
                    category=category,
                    options=options,
                    is_synthetic=True,
                )
                outcome = Outcome(
                    task_id=task_id,
                    correct_answer=options[0],
                    resolution_source="synthetic-ground-truth",
                )
                for agent in demo_state.demo.agents:
                    agent._outcomes[task_id] = outcome
                if hasattr(demo_state.demo.evaluator._sources, "_sources"):
                    demo_state.demo.evaluator._sources._sources[f"synthetic://{task_id}"] = (
                        f"Item {task_id} has recorded value {options[0]}."
                    )

        try:
            evaluation = evaluate_task(
                task,
                demo_state.demo.agents,
                demo_state.repository,
                demo_state.engine,
                demo_state.demo.evaluator,
                demo_state.policy,
            )
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Evaluation failed: {exc}",
            )

    else:
        # Live multi-agent execution (Local Ollama or Cloud API)
        repository = get_live_repository()
        task_id = request.task_id or uuid4().hex
        task = Task(
            task_id=task_id,
            question=clean_question,
            category=request.category.strip() or "general",
            options=tuple(request.options) if request.options else (),
            is_synthetic=False,
        )

        provider_type = (
            request.provider
            or ("openai" if (request.api_key or settings.llm_provider == "openai") else "ollama")
        )

        try:
            if provider_type == "openai":
                api_key = request.api_key or settings.openai_api_key
                if not api_key:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="An API key is required for Cloud API execution. Enter your key in the provider settings or configure TRUSTNET_OPENAI_API_KEY.",
                    )
                base_url = request.base_url or settings.openai_base_url or "https://api.openai.com/v1"
                cloud_model = request.model or settings.openai_model or "gpt-4o-mini"
                timeout = settings.openai_timeout_seconds

                if request.ensemble:
                    analyst_provider = OpenAIProvider(api_key=api_key, model=cloud_model, base_url=base_url, timeout_seconds=timeout)
                    skeptic_provider = OpenAIProvider(api_key=api_key, model=cloud_model, base_url=base_url, timeout_seconds=timeout)
                    researcher_provider = OpenAIProvider(api_key=api_key, model=cloud_model, base_url=base_url, timeout_seconds=timeout)
                    agents = [
                        LLMAgent(ANALYST, analyst_provider),
                        LLMAgent(SKEPTIC, skeptic_provider),
                        LLMAgent(RESEARCHER, researcher_provider),
                    ]
                else:
                    provider = OpenAIProvider(api_key=api_key, model=cloud_model, base_url=base_url, timeout_seconds=timeout)
                    agents = build_agents(provider)

            else:
                # Local Ollama execution
                if request.ensemble:
                    # Heterogeneous multi-model ensemble
                    analyst_provider = OllamaProvider(chosen_model, settings.ollama_base_url, settings.ollama_timeout_seconds)
                    skeptic_model = "gemma2:2b" if chosen_model != "gemma2:2b" else "llama3.2:1b"
                    skeptic_provider = OllamaProvider(skeptic_model, settings.ollama_base_url, settings.ollama_timeout_seconds)
                    researcher_provider = OllamaProvider(chosen_model, settings.ollama_base_url, settings.ollama_timeout_seconds)
                    agents = [
                        LLMAgent(ANALYST, analyst_provider),
                        LLMAgent(SKEPTIC, skeptic_provider),
                        LLMAgent(RESEARCHER, researcher_provider),
                    ]
                else:
                    provider = OllamaProvider(chosen_model, settings.ollama_base_url, settings.ollama_timeout_seconds)
                    agents = build_agents(provider)

            evaluator = build_evaluator(settings, None)
            evaluation = evaluate_task(task, agents, repository, engine, evaluator, policy)

        except HTTPException:
            raise
        except ProviderConfigurationError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Provider configuration error: {exc}",
            )
        except ProviderError as exc:
            target_name = (request.model or "gpt-4o-mini") if provider_type == "openai" else chosen_model
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY if provider_type == "openai" else status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"{provider_type.title()} provider error with model '{target_name}': {exc}",
            )
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Live execution failed: {exc}",
            )

    data = evaluation.model_dump(mode="json")
    data["conflicting_answers"] = evaluation.conflicting_answers
    data["decision"] = evaluation.decision.model_dump(mode="json")
    return data


@app.post("/api/outcome/record")
def record_verified_outcome(request: RecordOutcomeRequest) -> dict[str, Any]:
    """Store a verified outcome and recompute reliability metrics."""
    clean_answer = request.correct_answer.strip() if request.correct_answer and request.correct_answer.strip() else None
    clean_source = request.resolution_source.strip()
    if not clean_source:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Resolution source (who/what verified the outcome) is required.",
        )

    outcome = Outcome(
        task_id=request.task_id,
        correct_answer=clean_answer,
        resolution_source=clean_source,
    )
    settings = get_settings()
    engine = build_engine(settings)
    repo = demo_state.repository if request.mode == "synthetic" else get_live_repository()

    try:
        updated_profiles = record_outcome(repo, outcome, engine)
    except DuplicateOutcomeError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Outcome already recorded: {exc}",
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task not found: {exc}",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Recording failed: {exc}",
        )

    serialized = {
        agent_id: {
            (cat or "overall"): signals.model_dump(mode="json")
            for cat, signals in by_cat.items()
        }
        for agent_id, by_cat in updated_profiles.items()
    }
    return {
        "status": "success",
        "message": "Outcome recorded successfully. Reliability metrics recomputed.",
        "task_id": request.task_id,
        "correct_answer": clean_answer,
        "is_answerable": clean_answer is not None,
        "resolution_source": clean_source,
        "profiles": serialized,
    }


@app.get("/api/benchmark")
def get_benchmark(
    seed: int = Query(default=7, ge=0),
    development_size: int = Query(default=300, ge=10, le=1000),
    test_size: int = Query(default=300, ge=10, le=1000),
) -> dict[str, Any]:
    """Execute reproducible held-out synthetic benchmark comparing the 4 strategies."""
    try:
        report = run_benchmark(
            seed=seed,
            development_size=development_size,
            test_size=test_size,
        )
        return report.model_dump(mode="json")
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Benchmark execution failed: {exc}",
        )


@app.get("/api/agents")
def get_agents(mode: Literal["synthetic", "live"] = "synthetic") -> dict[str, Any]:
    """Retrieve full reliability profiles for all agents in the current repository."""
    settings = get_settings()
    engine = build_engine(settings)
    repo = demo_state.repository if mode == "synthetic" else get_live_repository()
    profiles = repo.refresh_reliability(engine)
    return {
        agent_id: {
            (cat or "overall"): signals.model_dump(mode="json")
            for cat, signals in by_cat.items()
        }
        for agent_id, by_cat in profiles.items()
    }


@app.get("/api/sources/{filename}")
def get_source_document(filename: str) -> dict[str, Any]:
    """Retrieve ground-truth text document cited in evidence."""
    file_path = (SOURCES_DIR / filename).resolve()
    if not file_path.is_relative_to(SOURCES_DIR) or not file_path.is_file():
        raise HTTPException(status_code=404, detail="Source document not found.")
    return {
        "filename": filename,
        "content": file_path.read_text(encoding="utf-8", errors="replace"),
    }


@app.api_route("/healthz", methods=["GET", "HEAD"])
@app.api_route("/health", methods=["GET", "HEAD"])
def health_check() -> dict[str, str]:
    """Health check endpoint for Render, Docker, and Kubernetes probes."""
    return {"status": "ok", "app": "TrustNet AI"}


# Static files mount
if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.api_route("/", methods=["GET", "HEAD"])
    def serve_frontend() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

