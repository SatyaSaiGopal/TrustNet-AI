"""TrustNet AI dashboard. Run with:  streamlit run src/trustnet/ui/app.py

Display only: every number shown comes from ``trustnet.pipeline`` and the modules
beneath it.
"""

from uuid import uuid4

import streamlit as st
from pydantic import ValidationError

from trustnet.agents import build_agents
from trustnet.config import get_settings
from trustnet.evaluation import SYNTHETIC_NOTICE, run_benchmark
from trustnet.evaluation.demo import Demo, seed_demo
from trustnet.pipeline import (
    TaskEvaluation,
    build_engine,
    build_evaluator,
    build_policy,
    evaluate_task,
    record_outcome,
)
from trustnet.providers import ProviderError, create_provider
from trustnet.schemas import DecisionOutcome, Outcome, Task
from trustnet.storage import DuplicateOutcomeError, Repository, connect, initialize

SYNTHETIC = "Synthetic demo (no model needed)"
LIVE = "Live agents (local Ollama)"
NOT_MEASURED = "not measured"


def shown(value: float | None) -> str:
    return NOT_MEASURED if value is None else f"{value:.2f}"


def open_repository(path: str) -> Repository:
    connection = connect(path, check_same_thread=False)
    initialize(connection)
    return Repository(connection)


@st.cache_resource
def synthetic_demo() -> tuple[Repository, Demo]:
    repository = open_repository(":memory:")
    return repository, seed_demo(repository)


@st.cache_resource
def live_repository(path: str) -> Repository:
    return open_repository(path)


@st.cache_data
def benchmark_rows() -> tuple[str, list[dict[str, object]]]:
    report = run_benchmark()
    summary = (
        f"{report.development_cases} development cases, {report.test_cases} held-out test "
        f"cases ({report.test_unanswerable} unanswerable), seed {report.seed}"
    )
    rows = [
        {
            "strategy": row.strategy,
            "accuracy": f"{row.accuracy:.3f}",
            "95% interval": f"{row.accuracy_interval[0]:.3f} to {row.accuracy_interval[1]:.3f}",
            "abstention rate": f"{row.abstention_rate:.3f}",
            "answered": row.answered,
            "answered-case accuracy": shown(row.answered_accuracy),
            "Brier score": shown(row.brier_score),
        }
        for row in report.strategies
    ]
    return summary, rows


def render(evaluation: TaskEvaluation) -> None:
    result, decision = evaluation.result, evaluation.decision
    st.subheader("Final decision (trust-weighted)")
    if decision.outcome is DecisionOutcome.DECIDED:
        st.success(f"**{decision.answer}** — {decision.rationale}")
    else:
        st.warning(f"**Abstained.** Reason: {decision.abstention_reason}")

    for failure in result.failures:
        st.error(f"Agent '{failure.agent_id}' failed: {failure.message}")
    answers = evaluation.conflicting_answers
    if len(answers) > 1:
        sides = "; ".join(f"'{answer}' from {', '.join(who)}" for answer, who in answers.items())
        st.error(f"Conflict: the agents gave {len(answers)} different answers. {sides}")
    elif answers:
        st.info("No conflict: every agent that answered gave the same answer.")

    counted = {item.agent_id: item for item in decision.contributions}
    st.subheader("Agent recommendations")
    st.dataframe(
        [
            {
                "agent": response.agent_id,
                "recommendation": response.recommendation or "(abstained)",
                "stated confidence": f"{response.self_reported_confidence:.2f}",
                "weight in decision": shown(counted[response.agent_id].weight),
                "how it counted": counted[response.agent_id].note,
                "model": response.model_id,
                "mock output": response.is_mock,
                "rationale": response.rationale,
            }
            for response in result.responses
        ],
        hide_index=True,
    )

    st.subheader("Reliability metrics")
    st.caption(
        "Measured from verified outcomes only. Stated confidence is never used as accuracy. "
        f"'{NOT_MEASURED}' means there is no data, which is different from a score of 0."
    )
    rows = []
    for agent_id in result.expected_agent_ids:
        profile = evaluation.trust.get(agent_id)
        rows.append(
            {
                "agent": agent_id,
                "profile": (profile.category or "all categories") if profile else "none",
                "verified answers": profile.resolved_sample_size if profile else 0,
                "enough history": bool(profile and profile.has_sufficient_history),
                "trust score": shown(profile.trust_score if profile else None),
                "accuracy": shown(profile.historical_accuracy if profile else None),
                "accuracy (95% lower bound)": shown(
                    profile.accuracy_lower_bound if profile else None
                ),
                "calibration": shown(profile.calibration if profile else None),
                "evidence quality": shown(profile.evidence_quality if profile else None),
                "consistency": shown(profile.consistency if profile else None),
                "abstention behaviour": shown(
                    profile.insufficient_information_behavior if profile else None
                ),
            }
        )
    st.dataframe(rows, hide_index=True)

    st.subheader("Evidence status")
    status = {item.evidence_id: item for item in evaluation.assessments}
    evidence = [
        {
            "agent": response.agent_id,
            "claim": item.excerpt,
            "reference": item.reference or "(none)",
            "status": status[item.evidence_id].status,
            "checked by": status[item.evidence_id].method,
            "note": status[item.evidence_id].note,
        }
        for response in result.responses
        for item in response.evidence
    ]
    if evidence:
        st.dataframe(evidence, hide_index=True)
    else:
        st.info("No agent cited any evidence.")

    st.subheader("Strategy comparison")
    st.dataframe(
        [
            {
                "strategy": item.strategy,
                "result": item.answer if item.answer is not None else "ABSTAINED",
                "support": shown(item.support),
                "explanation": item.rationale,
            }
            for item in evaluation.decisions
        ],
        hide_index=True,
    )


def outcome_recorder(repository: Repository, task: Task, known: Outcome | None) -> None:
    st.subheader("Verified outcome")
    st.caption("Reliability only changes when a verified outcome is recorded. One per task.")
    if known is None:
        answer = st.text_input("Correct answer (leave blank if the task was unanswerable)")
        source = st.text_input("Who or what verified it")
        if not st.button("Record outcome", key="record"):
            return
        try:
            known = Outcome(
                task_id=task.task_id,
                correct_answer=answer.strip() or None,
                resolution_source=source.strip(),
            )
        except ValidationError:
            st.error("Say who or what verified the outcome.")
            return
    elif not st.button("Reveal and record the synthetic outcome", key="record"):
        return
    try:
        record_outcome(repository, known, build_engine(get_settings()))
    except DuplicateOutcomeError as exc:
        st.error(f"Not recorded: {exc}.")
    else:
        label = known.correct_answer if known.answerable else "unanswerable (abstaining was right)"
        st.success(f"Recorded: {label}. Reliability metrics were recomputed.")


def main() -> None:
    st.set_page_config(page_title="TrustNet AI", layout="wide")
    st.title("TrustNet AI")
    st.caption("Research prototype. Do not use it for consequential decisions.")
    settings = get_settings()
    mode = st.sidebar.radio("Data source", [SYNTHETIC, LIVE])
    engine, policy = build_engine(settings), build_policy(settings)

    if mode == SYNTHETIC:
        st.warning(SYNTHETIC_NOTICE)
        repository, demo = synthetic_demo()
        cases = {case.task.task_id: case for case in demo.cases}
        task_id = st.selectbox(
            "Unseen synthetic task", list(cases), format_func=lambda key: cases[key].task.question
        )
        task, known = cases[task_id].task, cases[task_id].outcome
        if st.button("Run agents", key="run"):
            st.session_state[mode] = evaluate_task(
                task, demo.agents, repository, engine, demo.evaluator, policy
            )
    else:
        repository = live_repository(str(settings.database_path))
        question = st.text_area("Question")
        category = st.text_input("Task category", value="general")
        task, known = None, None
        if st.button("Run agents", key="run"):
            try:
                task = Task(task_id=uuid4().hex, question=question, category=category or "general")
                provider = create_provider(settings)
            except ValidationError:
                st.error("Enter a question.")
            except ProviderError as exc:
                st.error(str(exc))
            else:
                with st.spinner("Running the agents on the local model. This can take minutes."):
                    st.session_state[mode] = evaluate_task(
                        task,
                        build_agents(provider),
                        repository,
                        engine,
                        build_evaluator(settings, provider),
                        policy,
                    )

    evaluation = st.session_state.get(mode)
    if evaluation is not None and (mode == LIVE or evaluation.result.task.task_id == task.task_id):
        render(evaluation)
        outcome_recorder(repository, evaluation.result.task, known)

    with st.sidebar.expander("Synthetic benchmark"):
        st.caption(SYNTHETIC_NOTICE)
        if st.button("Run benchmark", key="benchmark"):
            summary, rows = benchmark_rows()
            st.caption(summary)
            st.dataframe(rows, hide_index=True)


main()
