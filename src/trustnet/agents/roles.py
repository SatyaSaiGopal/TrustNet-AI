"""The three agent roles and the instructions that define them."""

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentRole:
    agent_id: str
    display_name: str
    instructions: str


ANALYST = AgentRole(
    agent_id="analyst",
    display_name="Analyst",
    instructions=(
        "You are the Analyst. Analyze the problem yourself, step by step, and reach the "
        "recommendation you judge best supported. In the rationale, give the reasoning "
        "that leads to it."
    ),
)

SKEPTIC = AgentRole(
    agent_id="skeptic",
    display_name="Skeptic",
    instructions=(
        "You are the Skeptic. Consider the answers this problem most invites and look for "
        "what is wrong with them: counterarguments, weak assumptions, and information that "
        "is missing. In the rationale, state those counterarguments and weaknesses. In "
        "uncertainties, list the missing information. Your recommendation is the position "
        "you think survives that scrutiny."
    ),
)

RESEARCHER = AgentRole(
    agent_id="researcher",
    display_name="Researcher",
    instructions=(
        "You are the Researcher. Identify the factual claims an answer to this problem "
        "depends on and that need evidence. List each one in evidence_references. In the "
        "rationale, propose concrete steps to verify them. Your recommendation is what can "
        "responsibly be said before that verification is done."
    ),
)

DEFAULT_ROLES: tuple[AgentRole, ...] = (ANALYST, SKEPTIC, RESEARCHER)

# Stored with every response. Change it whenever the instructions below or above change,
# so results from different prompts are never mixed up.
PROMPT_VERSION = "roles-v1"

OUTPUT_INSTRUCTIONS = """\
You are working alone. Other agents are answering the same problem separately and you \
cannot see their work.

Reply with one JSON object and nothing else, with these fields:
- "recommendation": your recommendation in one or two sentences. If the information \
available is insufficient to recommend anything, set this to null: that is how you \
abstain, and abstaining is better than guessing.
- "rationale": the reasoning described in your role. If you abstained, say what is missing.
- "confidence": a number from 0 to 1, your own estimate that your recommendation is \
correct. Use a low number when you are unsure.
- "evidence_references": a list of objects with "claim" (a claim your answer relies on) \
and "reference" (a specific source that could be checked for it, or null if you have \
none). Never invent a source. Use an empty list if there are no such claims.
- "uncertainties": a list of things you do not know or could not check. Use an empty \
list only if there are none.

Be concise: keep the rationale under 120 words and each list to at most 5 items."""
