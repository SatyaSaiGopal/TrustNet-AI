"""Reads and writes domain objects in SQLite. The only module that issues SQL for them."""

import json
import sqlite3
from collections.abc import Iterable

from trustnet.schemas import (
    AgentProfile,
    AgentResponse,
    Decision,
    Evidence,
    EvidenceAssessment,
    Outcome,
    Task,
    TrustSignals,
)
from trustnet.trust.engine import TrustEngine

ALL_CATEGORIES = ""


class DuplicateOutcomeError(ValueError):
    """A verified outcome is already recorded for this task."""


class Repository:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self._db = connection

    # --- writes -------------------------------------------------------------------------

    def save_agent(self, agent: AgentProfile) -> None:
        self._db.execute(
            "INSERT INTO agents (agent_id, display_name, provider, model, is_synthetic) "
            "VALUES (?, ?, ?, ?, ?) ON CONFLICT (agent_id) DO UPDATE SET "
            "display_name = excluded.display_name, provider = excluded.provider, "
            "model = excluded.model, is_synthetic = excluded.is_synthetic",
            (agent.agent_id, agent.display_name, agent.provider, agent.model, agent.is_synthetic),
        )
        self._db.commit()

    def save_task(self, task: Task) -> None:
        self._db.execute(
            "INSERT OR IGNORE INTO tasks "
            "(task_id, question, is_synthetic, category, equivalence_group, options) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                task.task_id,
                task.question,
                task.is_synthetic,
                task.category,
                task.equivalence_group,
                json.dumps(task.options),
            ),
        )
        self._db.commit()

    def save_responses(
        self,
        responses: Iterable[AgentResponse],
        assessments: Iterable[EvidenceAssessment] = (),
    ) -> None:
        """Store responses with their evidence, and any assessments of that evidence."""
        with self._db:
            for response in responses:
                self._db.execute(
                    "INSERT INTO agent_responses (response_id, task_id, agent_id, stance, answer, "
                    "rationale, self_reported_confidence, uncertainties, model_id, is_mock, "
                    "prompt_version) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        response.response_id,
                        response.task_id,
                        response.agent_id,
                        response.stance,
                        response.recommendation,
                        response.rationale,
                        response.self_reported_confidence,
                        json.dumps(response.uncertainties),
                        response.model_id,
                        response.is_mock,
                        response.prompt_version,
                    ),
                )
                self._db.executemany(
                    "INSERT INTO evidence_items (evidence_id, response_id, source_type, excerpt, "
                    "reference, verification_status) VALUES (?, ?, ?, ?, ?, ?)",
                    [
                        (
                            item.evidence_id,
                            response.response_id,
                            item.source_type,
                            item.excerpt,
                            item.reference,
                            item.verification_status,
                        )
                        for item in response.evidence
                    ],
                )
            self._db.executemany(
                "INSERT OR REPLACE INTO evidence_assessments "
                "(evidence_id, status, source_available, relevance, method, note) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (
                        item.evidence_id,
                        item.status,
                        item.source_available,
                        item.relevance,
                        item.method,
                        item.note,
                    )
                    for item in assessments
                ],
            )

    def record_outcome(self, outcome: Outcome) -> None:
        """Record the verified outcome of a task. A task can have only one."""
        try:
            with self._db:
                self._db.execute(
                    "INSERT INTO outcomes (task_id, correct_answer, resolution_source) "
                    "VALUES (?, ?, ?)",
                    (outcome.task_id, outcome.correct_answer, outcome.resolution_source),
                )
        except sqlite3.IntegrityError as exc:
            if "FOREIGN KEY" in str(exc).upper():
                raise ValueError(f"unknown task '{outcome.task_id}'") from exc
            raise DuplicateOutcomeError(
                f"task '{outcome.task_id}' already has a verified outcome"
            ) from exc

    def save_decisions(self, decisions: Iterable[Decision]) -> None:
        with self._db:
            self._db.executemany(
                "INSERT OR REPLACE INTO decisions (decision_id, task_id, outcome, answer, "
                "rationale, strategy, abstention_reason, support, contributions) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        decision.decision_id,
                        decision.task_id,
                        decision.outcome,
                        decision.answer,
                        decision.rationale,
                        decision.strategy,
                        decision.abstention_reason,
                        decision.support,
                        json.dumps([item.model_dump() for item in decision.contributions]),
                    )
                    for decision in decisions
                ],
            )

    def refresh_reliability(self, engine: TrustEngine) -> dict[str, dict[str | None, TrustSignals]]:
        """Recompute and store every agent's profiles from the recorded history.

        Accuracy and calibration come only from responses whose task has a verified
        outcome; responses without one cannot move them.
        """
        tasks, outcomes, assessments = self.tasks(), self.outcomes(), self.assessments()
        responses = self.responses()
        profiles = {
            agent_id: engine.profiles(agent_id, responses, outcomes, tasks, assessments)
            for agent_id in self.agent_ids()
        }
        with self._db:
            self._db.executemany(
                "INSERT OR REPLACE INTO reliability_metrics "
                "(agent_id, category, trust_score, resolved_sample_size, signals) "
                "VALUES (?, ?, ?, ?, ?)",
                [
                    (
                        agent_id,
                        category or ALL_CATEGORIES,
                        signals.trust_score,
                        signals.resolved_sample_size,
                        signals.model_dump_json(),
                    )
                    for agent_id, by_category in profiles.items()
                    for category, signals in by_category.items()
                ],
            )
        return profiles

    # --- reads --------------------------------------------------------------------------

    def agent_ids(self) -> list[str]:
        rows = self._db.execute("SELECT agent_id FROM agents ORDER BY agent_id")
        return [row["agent_id"] for row in rows]

    def tasks(self) -> dict[str, Task]:
        return {
            row["task_id"]: Task(
                task_id=row["task_id"],
                question=row["question"],
                is_synthetic=bool(row["is_synthetic"]),
                category=row["category"],
                equivalence_group=row["equivalence_group"],
                options=tuple(json.loads(row["options"])),
            )
            for row in self._db.execute("SELECT * FROM tasks")
        }

    def responses(self, task_id: str | None = None) -> list[AgentResponse]:
        evidence: dict[str, list[Evidence]] = {}
        for row in self._db.execute("SELECT * FROM evidence_items ORDER BY rowid"):
            evidence.setdefault(row["response_id"], []).append(
                Evidence(
                    evidence_id=row["evidence_id"],
                    source_type=row["source_type"],
                    excerpt=row["excerpt"],
                    reference=row["reference"],
                    verification_status=row["verification_status"],
                )
            )
        query = "SELECT * FROM agent_responses"
        parameters: tuple[str, ...] = ()
        if task_id is not None:
            query, parameters = f"{query} WHERE task_id = ?", (task_id,)
        return [
            AgentResponse(
                response_id=row["response_id"],
                task_id=row["task_id"],
                agent_id=row["agent_id"],
                stance=row["stance"],
                recommendation=row["answer"],
                rationale=row["rationale"],
                self_reported_confidence=row["self_reported_confidence"],
                evidence=tuple(evidence.get(row["response_id"], ())),
                uncertainties=tuple(json.loads(row["uncertainties"])),
                model_id=row["model_id"],
                is_mock=bool(row["is_mock"]),
                prompt_version=row["prompt_version"],
            )
            for row in self._db.execute(f"{query} ORDER BY created_at, rowid", parameters)
        ]

    def outcomes(self) -> dict[str, Outcome]:
        return {
            row["task_id"]: Outcome(
                task_id=row["task_id"],
                correct_answer=row["correct_answer"],
                resolution_source=row["resolution_source"],
            )
            for row in self._db.execute("SELECT * FROM outcomes")
        }

    def assessments(self) -> dict[str, EvidenceAssessment]:
        return {
            row["evidence_id"]: EvidenceAssessment(
                evidence_id=row["evidence_id"],
                status=row["status"],
                source_available=None
                if row["source_available"] is None
                else bool(row["source_available"]),
                relevance=row["relevance"],
                method=row["method"],
                note=row["note"],
            )
            for row in self._db.execute("SELECT * FROM evidence_assessments")
        }

    def metrics(self, agent_id: str) -> dict[str | None, TrustSignals]:
        rows = self._db.execute(
            "SELECT category, signals FROM reliability_metrics WHERE agent_id = ?", (agent_id,)
        )
        return {
            row["category"] or None: TrustSignals.model_validate_json(row["signals"])
            for row in rows
        }

    def decisions(self, task_id: str) -> list[Decision]:
        rows = self._db.execute(
            "SELECT * FROM decisions WHERE task_id = ? ORDER BY rowid", (task_id,)
        )
        return [
            Decision(
                decision_id=row["decision_id"],
                task_id=row["task_id"],
                outcome=row["outcome"],
                answer=row["answer"],
                rationale=row["rationale"],
                strategy=row["strategy"],
                abstention_reason=row["abstention_reason"],
                support=row["support"],
                contributions=tuple(json.loads(row["contributions"])),
            )
            for row in rows
        ]
