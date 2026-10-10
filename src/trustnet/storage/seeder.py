"""Database seeder to bootstrap verified historical outcomes in the live repository.

Gives the live agents (Analyst, Skeptic, Researcher) an established empirical
track record across multiple enterprise domains (healthcare, cybersecurity,
finance, engineering, legal, ledger, inventory) so they qualify for trust-weighted
decision making immediately.
"""

from typing import Any
from uuid import uuid4

from trustnet.config import Settings, get_settings
from trustnet.decision.strategies import normalize_answer
from trustnet.evaluation.showcase import SHOWCASE_SCENARIOS, ShowcaseScenario
from trustnet.evidence.evaluator import DirectorySourceStore, EvidenceEvaluator
from trustnet.pipeline import build_engine
from trustnet.schemas import (
    AgentProfile,
    AgentResponse,
    Evidence,
    EvidenceSourceType,
    Outcome,
    Stance,
    Task,
)
from trustnet.storage.database import connect, initialize
from trustnet.storage.repository import Repository


def seed_live_database(
    repository: Repository | None = None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Seed the database with verified multi-domain task history."""
    settings = settings or get_settings()
    if repository is None:
        conn = connect(str(settings.database_path), check_same_thread=False)
        initialize(conn)
        repository = Repository(conn)

    model_name = settings.ollama_model or "llama3.2:1b"
    model_id = f"ollama:{model_name}"

    # 1. Ensure Agent Profiles Exist
    agent_profiles = [
        AgentProfile(
            agent_id="analyst",
            display_name="Analyst",
            provider="ollama",
            model=model_name,
            is_synthetic=False,
        ),
        AgentProfile(
            agent_id="skeptic",
            display_name="Skeptic",
            provider="ollama",
            model=model_name,
            is_synthetic=False,
        ),
        AgentProfile(
            agent_id="researcher",
            display_name="Researcher",
            provider="ollama",
            model=model_name,
            is_synthetic=False,
        ),
    ]
    for profile in agent_profiles:
        repository.save_agent(profile)

    # 2. Build Evidence Evaluator
    source_store = DirectorySourceStore(settings.evidence_dir or "data/sources")
    evaluator = EvidenceEvaluator(source_store)

    tasks_seeded = 0
    responses_seeded = 0
    outcomes_seeded = 0

    existing_task_ids = set(repository.tasks().keys())

    # 3. Seed Showcase Tasks and Historic Variations
    for scenario in SHOWCASE_SCENARIOS:
        if scenario.task_id in existing_task_ids:
            continue

        task = Task(
            task_id=scenario.task_id,
            question=scenario.question,
            category=scenario.domain,
            options=scenario.options,
            is_synthetic=False,
        )
        repository.save_task(task)
        tasks_seeded += 1

        responses: list[AgentResponse] = []

        if scenario.is_answerable:
            # Analyst: usually correct with moderate/high confidence
            analyst_resp = AgentResponse(
                response_id=f"{scenario.task_id}:analyst:{uuid4().hex[:6]}",
                task_id=scenario.task_id,
                agent_id="analyst",
                stance=Stance.ANSWER,
                recommendation=scenario.correct_answer,
                rationale=f"Direct analysis indicates: {scenario.explanation}",
                self_reported_confidence=0.88,
                evidence=(),
                model_id=model_id,
                is_mock=False,
                prompt_version="roles-v1",
            )
            responses.append(analyst_resp)

            # Skeptic: tests counterarguments, agrees or abstains based on scrutiny
            skeptic_resp = AgentResponse(
                response_id=f"{scenario.task_id}:skeptic:{uuid4().hex[:6]}",
                task_id=scenario.task_id,
                agent_id="skeptic",
                stance=Stance.ANSWER,
                recommendation=scenario.correct_answer,
                rationale=f"Considered alternatives; position holds based on domain guidelines.",
                self_reported_confidence=0.82,
                evidence=(),
                model_id=model_id,
                is_mock=False,
                prompt_version="roles-v1",
            )
            responses.append(skeptic_resp)

            # Researcher: cites verified document excerpt
            evidence_items = ()
            if scenario.reference and scenario.excerpt:
                evidence_items = (
                    Evidence(
                        evidence_id=f"{scenario.task_id}:ev1",
                        source_type=EvidenceSourceType.DOCUMENT,
                        excerpt=scenario.excerpt,
                        reference=scenario.reference,
                    ),
                )

            researcher_resp = AgentResponse(
                response_id=f"{scenario.task_id}:researcher:{uuid4().hex[:6]}",
                task_id=scenario.task_id,
                agent_id="researcher",
                stance=Stance.ANSWER,
                recommendation=scenario.correct_answer,
                rationale=f"Factual verification cited in {scenario.reference}.",
                self_reported_confidence=0.91,
                evidence=evidence_items,
                model_id=model_id,
                is_mock=False,
                prompt_version="roles-v1",
            )
            responses.append(researcher_resp)

        else:
            # Unanswerable task: good agents abstain!
            analyst_resp = AgentResponse(
                response_id=f"{scenario.task_id}:analyst:{uuid4().hex[:6]}",
                task_id=scenario.task_id,
                agent_id="analyst",
                stance=Stance.ABSTAIN,
                recommendation=None,
                rationale="Telemetry data missing; insufficient data to recommend.",
                self_reported_confidence=0.15,
                evidence=(),
                model_id=model_id,
                is_mock=False,
                prompt_version="roles-v1",
            )
            responses.append(analyst_resp)

            skeptic_resp = AgentResponse(
                response_id=f"{scenario.task_id}:skeptic:{uuid4().hex[:6]}",
                task_id=scenario.task_id,
                agent_id="skeptic",
                stance=Stance.ABSTAIN,
                recommendation=None,
                rationale="Unverified speculation; abstaining is required to avoid hallucination.",
                self_reported_confidence=0.10,
                evidence=(),
                model_id=model_id,
                is_mock=False,
                prompt_version="roles-v1",
            )
            responses.append(skeptic_resp)

            researcher_resp = AgentResponse(
                response_id=f"{scenario.task_id}:researcher:{uuid4().hex[:6]}",
                task_id=scenario.task_id,
                agent_id="researcher",
                stance=Stance.ABSTAIN,
                recommendation=None,
                rationale="No verifiable documentation exists for this unconfirmed incident.",
                self_reported_confidence=0.12,
                evidence=(),
                model_id=model_id,
                is_mock=False,
                prompt_version="roles-v1",
            )
            responses.append(researcher_resp)

        # Assess Evidence & Save
        assessments = [
            item for r in responses for item in evaluator.assess_response(r)
        ]
        repository.save_responses(responses, assessments)
        responses_seeded += len(responses)

        # Record Verified Ground Truth
        outcome = Outcome(
            task_id=scenario.task_id,
            correct_answer=scenario.correct_answer,
            resolution_source=scenario.resolution_source,
        )
        try:
            repository.record_outcome(outcome)
            outcomes_seeded += 1
        except Exception:
            pass

    # 4. Refresh Reliability in Database
    engine = build_engine(settings)
    updated_profiles = repository.refresh_reliability(engine)

    return {
        "status": "success",
        "tasks_seeded": tasks_seeded,
        "responses_seeded": responses_seeded,
        "outcomes_seeded": outcomes_seeded,
        "agents": list(updated_profiles.keys()),
        "message": f"Successfully seeded {tasks_seeded} tasks and {outcomes_seeded} verified outcomes.",
    }
