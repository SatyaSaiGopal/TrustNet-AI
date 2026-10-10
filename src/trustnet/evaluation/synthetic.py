"""SYNTHETIC benchmark fixtures: simulated agents, tasks, sources and outcomes.

Nothing here comes from a real model. Each simulated agent answers from a fixed
recipe (how often it is right, how confident it claims to be, how often it abstains
or cites a source), drawn from a seeded random generator so runs are reproducible.
Every response is labelled as mock output.
"""

import random
from collections.abc import Mapping
from dataclasses import dataclass

from trustnet.agents.base import AgentOutputError
from trustnet.schemas import (
    MOCK_PROVIDER_NAME,
    AgentProfile,
    AgentResponse,
    Evidence,
    EvidenceSourceType,
    Outcome,
    Stance,
    Task,
)

SYNTHETIC_NOTICE = (
    "SYNTHETIC DATA: simulated agents, questions, sources and outcomes. "
    "These are not results from real models."
)
SYNTHETIC_LABEL = "[SYNTHETIC]"
CATEGORIES = ("ledger", "inventory")
UNANSWERABLE_SHARE = 0.2
PARAPHRASE_EVERY = 5


@dataclass(frozen=True)
class AgentSpec:
    agent_id: str
    # Agents sharing a model draw identical answers: they are perfectly correlated.
    model: str
    # Chance of a correct answer to an answerable task, per category.
    accuracy: Mapping[str, float]
    # Fixed stated confidence, or None to state roughly its true accuracy.
    confidence: float | None
    abstains_when_unanswerable: float
    cites_source: float


AGENT_SPECS: tuple[AgentSpec, ...] = (
    AgentSpec("steady", "synthetic-steady", {"ledger": 0.85, "inventory": 0.85}, None, 0.85, 0.9),
    AgentSpec("specialist", "synthetic-specialist", {"ledger": 0.95, "inventory": 0.45}, None, 0.6, 0.5),
    AgentSpec("overconfident", "synthetic-loud", {"ledger": 0.5, "inventory": 0.5}, 0.97, 0.0, 0.0),
    AgentSpec("clone-1", "synthetic-clone", {"ledger": 0.55, "inventory": 0.55}, 0.8, 0.1, 0.2),
    AgentSpec("clone-2", "synthetic-clone", {"ledger": 0.55, "inventory": 0.55}, 0.8, 0.1, 0.2),
)


@dataclass(frozen=True)
class Case:
    task: Task
    outcome: Outcome
    # (reference, text) of the source that settles the task; None when unanswerable.
    source: tuple[str, str] | None


def _item(task: Task) -> str:
    return (task.equivalence_group or task.task_id).rsplit("-", 1)[-1]


def make_cases(count: int, seed: int, prefix: str) -> list[Case]:
    """``count`` base tasks plus a reworded twin of every fifth one, for consistency checks."""
    rng = random.Random(f"{seed}|{prefix}|cases")
    cases = []
    for index in range(count):
        base_id = f"synthetic-{prefix}-{index:04d}"
        category = rng.choice(CATEGORIES)
        options = tuple(str(value) for value in rng.sample(range(1000, 10000), 4))
        answerable = rng.random() >= UNANSWERABLE_SHARE
        correct = rng.choice(options) if answerable else None
        wordings = [
            (base_id, f"What value does the {category} record for item {index:04d}?"),
        ]
        if index % PARAPHRASE_EVERY == 0:
            wordings.append(
                (f"{base_id}-b", f"For item {index:04d}, which value is in the {category}?")
            )
        for task_id, wording in wordings:
            note = "" if answerable else " The record for this item is missing."
            reference = f"synthetic://{task_id}"
            cases.append(
                Case(
                    task=Task(
                        task_id=task_id,
                        question=f"{SYNTHETIC_LABEL} {wording}{note}",
                        is_synthetic=True,
                        category=category,
                        equivalence_group=base_id,
                        options=options,
                    ),
                    outcome=Outcome(
                        task_id=task_id,
                        correct_answer=correct,
                        resolution_source="synthetic-ground-truth",
                    ),
                    source=(reference, f"Item {index:04d} has recorded value {correct}.")
                    if answerable
                    else None,
                )
            )
    return cases


class SimulatedAgent:
    """Answers benchmark tasks from its recipe. Satisfies the ``Agent`` interface."""

    def __init__(self, spec: AgentSpec, outcomes: Mapping[str, Outcome], seed: int = 0) -> None:
        self.spec = spec
        self._outcomes = outcomes
        self._seed = seed
        self._calls = 0
        self.profile = AgentProfile(
            agent_id=spec.agent_id,
            display_name=f"{SYNTHETIC_LABEL} {spec.agent_id}",
            provider=MOCK_PROVIDER_NAME,
            model=spec.model,
            is_synthetic=True,
        )

    def respond(self, task: Task) -> AgentResponse:
        outcome = self._outcomes.get(task.task_id)
        if outcome is None or not task.options:
            raise AgentOutputError(
                f"simulated agent '{self.spec.agent_id}' can only answer synthetic benchmark tasks"
            )
        self._calls += 1
        spec = self.spec
        rng = random.Random(f"{self._seed}|{spec.model}|{task.task_id}")
        accuracy = spec.accuracy.get(task.category, sum(spec.accuracy.values()) / len(spec.accuracy))

        if not outcome.answerable:
            answer = None if rng.random() < spec.abstains_when_unanswerable else rng.choice(task.options)
        elif rng.random() < accuracy:
            answer = outcome.correct_answer
        else:
            answer = rng.choice([option for option in task.options if option != outcome.correct_answer])
        stated = spec.confidence
        if stated is None:
            stated = min(1.0, max(0.0, accuracy + rng.uniform(-0.05, 0.05)))
        cites = answer is not None and rng.random() < spec.cites_source

        response_id = f"{task.task_id}:{spec.agent_id}:{self._calls}"
        return AgentResponse(
            response_id=response_id,
            task_id=task.task_id,
            agent_id=spec.agent_id,
            stance=Stance.ABSTAIN if answer is None else Stance.ANSWER,
            recommendation=answer,
            rationale=f"{SYNTHETIC_LABEL} simulated response",
            self_reported_confidence=stated,
            evidence=(
                Evidence(
                    evidence_id=f"{response_id}:e1",
                    source_type=EvidenceSourceType.DOCUMENT,
                    excerpt=f"Item {_item(task)} has recorded value {answer}.",
                    reference=f"synthetic://{task.task_id}",
                ),
            )
            if cites
            else (),
            model_id=f"{MOCK_PROVIDER_NAME}:{spec.model}",
            is_mock=True,
            prompt_version="synthetic",
        )


def build_agents(outcomes: Mapping[str, Outcome], seed: int = 0) -> list[SimulatedAgent]:
    return [SimulatedAgent(spec, outcomes, seed) for spec in AGENT_SPECS]
