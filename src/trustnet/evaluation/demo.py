"""A SYNTHETIC demo: a database preloaded with simulated history, plus unseen tasks to run."""

from dataclasses import dataclass

from trustnet.evaluation.synthetic import Case, SimulatedAgent, build_agents, make_cases
from trustnet.evidence.evaluator import EvidenceEvaluator, InMemorySourceStore
from trustnet.storage.repository import Repository
from trustnet.trust.engine import TrustEngine


@dataclass(frozen=True)
class Demo:
    agents: list[SimulatedAgent]
    # Tasks the agents have not been scored on; their outcomes are not in the database.
    cases: list[Case]
    evaluator: EvidenceEvaluator


def seed_demo(
    repository: Repository,
    engine: TrustEngine | None = None,
    seed: int = 7,
    history_size: int = 120,
    unseen_size: int = 25,
) -> Demo:
    """Store simulated agents, their past responses and the verified outcomes of those."""
    history = make_cases(history_size, seed, "demo-history")
    unseen = make_cases(unseen_size, seed, "demo-unseen")
    cases = history + unseen
    evaluator = EvidenceEvaluator(
        InMemorySourceStore(dict(case.source for case in cases if case.source))
    )
    agents = build_agents({case.task.task_id: case.outcome for case in cases}, seed)

    for agent in agents:
        repository.save_agent(agent.profile)
    responses = []
    for case in history:
        repository.save_task(case.task)
        responses += [agent.respond(case.task) for agent in agents]
    assessments = [item for response in responses for item in evaluator.assess_response(response)]
    repository.save_responses(responses, assessments)
    for case in history:
        repository.record_outcome(case.outcome)
    repository.refresh_reliability(engine or TrustEngine())
    return Demo(agents=agents, cases=unseen, evaluator=evaluator)
