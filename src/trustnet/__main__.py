"""Run the agent workflow on one question: ``python -m trustnet "your question"``.

Prints the collected agent outputs as JSON. Exit code 0 means every agent responded,
1 means at least one agent failed, 2 means the input or configuration was invalid.
"""

import argparse
import json
import sys
from collections.abc import Sequence
from uuid import uuid4

from pydantic import ValidationError

from trustnet.agents import build_agents, run_workflow
from trustnet.config import get_settings
from trustnet.providers import ProviderConfigurationError, create_provider
from trustnet.schemas import Task


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m trustnet",
        description="Run the Analyst, Skeptic and Researcher agents on one question.",
    )
    parser.add_argument("question", help="the problem for the agents to work on")
    args = parser.parse_args(argv)

    try:
        task = Task(task_id=uuid4().hex, question=args.question)
    except ValidationError as exc:
        reasons = "; ".join(item["msg"] for item in exc.errors())
        print(f"Invalid question: {reasons}", file=sys.stderr)
        return 2
    try:
        provider = create_provider(get_settings())
    except (ProviderConfigurationError, ValidationError) as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    result = run_workflow(task, build_agents(provider))
    print(json.dumps(result.model_dump(mode="json"), indent=2))
    for failure in result.failures:
        print(f"AGENT FAILED [{failure.agent_id}] {failure.message}", file=sys.stderr)
    return 0 if result.is_complete else 1


if __name__ == "__main__":
    sys.exit(main())
