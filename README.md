# TrustNet AI

A research prototype for an **AI Agent Trust Network**: a system that scores multiple
AI agents on how far they can be trusted, uses those scores to combine conflicting
recommendations, and abstains when the evidence is insufficient.

> **Status: functional prototype.** Agents run independently, their evidence is checked,
> their reliability is measured from verified outcomes, and a trust-aware engine
> combines their recommendations or abstains. Results are stored in SQLite and shown
> in a Streamlit dashboard. There is no HTTP API.
>
> This is a student research prototype. Do not use it for consequential decisions.

## Architecture

```
   Task --> agents (LangGraph) --> evidence --> trust --> decision --> Decision
               |                      |           ^          |         or abstention
           providers               sources     storage <-----+
            (Ollama)                           (SQLite)

   pipeline ties the steps together; ui (Streamlit) only displays what pipeline returns.
   config and schemas are shared by every module.
```

### The agent workflow

```
   START --> analyst ----+
         --> skeptic ----+--> collect_outputs --> END
         --> researcher -+
```

| Agent | Role |
| --- | --- |
| Analyst | Analyzes the problem on its own and gives the recommendation it judges best supported. |
| Skeptic | Looks for counterarguments, weak assumptions and missing information. |
| Researcher | Identifies the claims that need evidence and proposes how to verify them. |

All three start in the same step and each is given only the task, so no agent sees
another's recommendation. An agent that fails (model down, timeout, invalid output) is
reported in `failures` and never replaced with invented output.

### How trust is measured

All five signals are in `trustnet/trust/engine.py`. A signal with no data is `None`
("not measured"), never 0.

| Signal | Measured as |
| --- | --- |
| Historical accuracy | Share of the agent's answers that match a **verified outcome**. Stated confidence plays no part. |
| Calibration | 1 minus the Brier score of stated confidence against verified outcomes. |
| Evidence quality | Per answer, the share of checked evidence found `verified`. An answer citing nothing scores 0; evidence that could not be checked is left out. |
| Consistency | Agreement between the agent's answers to the same task, or to tasks in the same `equivalence_group` (reworded prompts). |
| Insufficient-information behaviour | Mean of the abstention rate on tasks verified as unanswerable and the answer rate on answerable ones. |

The **trust score** is the weighted mean of the signals that could be measured
(default weights 0.40 accuracy, 0.20 calibration, 0.15 evidence, 0.15 consistency,
0.10 abstention; configurable in `.env`). It uses the Wilson 95% lower bound of
accuracy, so a short record counts for less than a long one. An agent has **no trust
score** until it has `TRUSTNET_TRUST_MIN_SAMPLES` verified answers (default 5).
Profiles are kept per task category as well as overall; the category profile is used
when it has enough history.

### How evidence is checked

`trustnet/evidence/evaluator.py` gives each cited item one of four statuses:

| Status | Meaning |
| --- | --- |
| `verified` | The cited source was read and states the claim |
| `contradictory` | The source was read and gives different figures or the opposite |
| `unsupported` | No reference was given, or the source does not support the claim |
| `unverified` | A reference was given but could not be read (missing file, remote URL) |

Checks are deterministic text heuristics (term overlap, number and negation
mismatch). `verified` and `contradictory` are impossible unless a source was actually
read. Sources are text files in `TRUSTNET_EVIDENCE_DIR`, cited by file name; URLs are
not fetched. Setting `TRUSTNET_EVIDENCE_LLM_JUDGE=true` additionally asks the local
model, only about sources that were read; if the model fails, the deterministic
result is kept.

### How decisions are made

`trustnet/decision/strategies.py` holds four strategies that run on identical inputs:

| Strategy | Rule |
| --- | --- |
| `trust_weighted` | Each response is weighted by the agent's trust score (details below) |
| `majority_vote` | Decide only if more than half of all agents give the same answer |
| `equal_weight_vote` | Most common answer among agents that answered; a tie means abstain |
| `highest_confidence` | Answer of the agent with the highest self-reported confidence |

The trust-weighted engine:

- does not count an agent that has no trust score, and assumes no default for it;
- treats agents on the same model giving the same response as correlated, sharing
  one weight between them;
- cuts a response's weight to a quarter when its own cited source contradicts it;
- counts abstaining agents' weight against deciding;
- **abstains** when nothing can be weighted, when the leading answer has under 60% of
  the weight, when the weight that abstained matches or exceeds it, or when no
  supporter of the leading answer has a trust score of at least 0.5;
- records each agent's weight, trust score, evidence status and how it counted.

Answers are compared after normalising case, spacing and edge punctuation, so the
engine works best when a task supplies `options` for agents to choose from. Free-text
answers worded differently count as different answers.

### Storage

SQLite, through `trustnet/storage/repository.py`: agents, tasks, responses (with
model id, prompt version and timestamp), evidence items and assessments, verified
outcomes, reliability metrics and decisions with their explanations. A task can have
only one verified outcome; a second is refused. Reliability is recomputed from the
stored history, and only verified outcomes can move accuracy or calibration.

### Modules

| Module | Purpose |
| --- | --- |
| `trustnet.config` | Settings from `TRUSTNET_*` environment variables or `.env` |
| `trustnet.schemas` | Shared Pydantic models and the data rules they enforce |
| `trustnet.providers` | Model access: `OllamaProvider` (ChatOllama); a scripted provider for tests |
| `trustnet.agents` | Agent roles, LLM agents, the LangGraph workflow; mock agents for tests |
| `trustnet.evidence` | Evidence evaluation and source stores |
| `trustnet.trust` | Scoring functions and the trust engine |
| `trustnet.decision` | The four decision strategies |
| `trustnet.storage` | SQLite schema, migrations and repository |
| `trustnet.pipeline` | One call that runs agents, checks evidence, decides and stores |
| `trustnet.evaluation` | Synthetic benchmark and synthetic demo data |
| `trustnet.ui` | Streamlit dashboard |
| `trustnet.api` | Placeholder; not implemented |

## Setup (Windows, PowerShell)

Requires Python 3.11 or newer. [Ollama](https://ollama.com/download) is needed only
for live agents; the tests, the benchmark and the synthetic demo run without it.
Run every command from the project directory.

### 1. Install

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev,ui]"
```

`ui` adds Streamlit for the dashboard; use `".[dev]"` if you only want the tests and
command line. Calling `.\.venv\Scripts\python.exe` directly avoids activating the
environment, which PowerShell's execution policy can block. If you set the project up
earlier, run the install command again; an existing database upgrades automatically.

### 2. Choose a model (live agents only)

No model is assumed. See what you have, and pull one if the list is empty:

```powershell
ollama list
ollama pull llama3.2:1b
```

### 3. Configure (live agents only)

```powershell
Copy-Item .env.example .env
```

Set `TRUSTNET_OLLAMA_MODEL` in `.env` to a name exactly as `ollama list` prints it.

| Variable | Default | Meaning |
| --- | --- | --- |
| `TRUSTNET_OLLAMA_MODEL` | none, required for live agents | The Ollama model the agents use |
| `TRUSTNET_OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server address |
| `TRUSTNET_OLLAMA_TIMEOUT_SECONDS` | `600` | Time each agent may take, including time queued |
| `TRUSTNET_DATABASE_PATH` | `data/trustnet.db` | SQLite file used by the dashboard's live mode |
| `TRUSTNET_TRUST_MIN_SAMPLES` | `5` | Verified answers needed before an agent gets a trust score |
| `TRUSTNET_WEIGHT_*` | see `.env.example` | Weight of each signal in the trust score |
| `TRUSTNET_DECISION_MIN_SUPPORT` | `0.6` | Share of reliability weight the leading answer needs |
| `TRUSTNET_DECISION_MIN_TRUST` | `0.5` | Trust score a supporter of the leading answer must have |
| `TRUSTNET_EVIDENCE_DIR` | unset | Folder of text sources agents may cite |
| `TRUSTNET_EVIDENCE_LLM_JUDGE` | `false` | Also ask the local model to judge evidence |

`.env` is git-ignored. The project uses no API keys or paid services.

## Running the tests

```powershell
.\.venv\Scripts\python.exe -m pytest
```

The suite is deterministic and offline: no Ollama server, model, network, API key or
`.env` file is needed. The dashboard test is skipped if Streamlit is not installed.

## Running the dashboard

```powershell
.\.venv\Scripts\python.exe -m streamlit run src/trustnet/ui/app.py
```

- **Synthetic demo** (default) needs no model. It uses simulated agents with a
  preloaded history and is labelled as synthetic throughout.
- **Live agents** runs the Analyst, Skeptic and Researcher on your local Ollama model
  and stores results in `TRUSTNET_DATABASE_PATH`. New agents have no verified history,
  so the trust-weighted engine abstains until you record verified outcomes for their
  answers in the "Verified outcome" box.

The page shows each agent's recommendation, the reliability metrics, evidence
status, conflicts, the decision from all four strategies, and the reason for any
abstention.

## Running the benchmark

```powershell
.\.venv\Scripts\python.exe -m trustnet.evaluation
```

This compares the four strategies on a seeded **synthetic** benchmark: simulated
agents with fixed recipes (one steady, one specialist, one overconfident, two clones
on a shared model), about 20% unanswerable tasks, reliability learned on a development
split and scored on a held-out test split. It prints sample counts, accuracy with 95%
intervals, abstention rate, answered-case accuracy and Brier score. Add `--seed N` to
vary it or `--json` for machine-readable output.

Read the numbers for what they are: they show how the strategies behave under the
simulation's assumptions (for example, that sources state answers plainly and that
clones err together). They are not evidence about real language models.

## Running the agents from the command line

```powershell
.\.venv\Scripts\python.exe -m trustnet "Should a small team rewrite a messy service or refactor it?"
```

Prints the three agents' collected outputs as JSON, without scoring or storing them.
Exit code `0` means all agents responded, `1` that one failed, `2` that the question
or configuration was invalid. The agents share one local model, so a run can take
several minutes on a laptop.

## Troubleshooting

| Message | Fix |
| --- | --- |
| `TRUSTNET_OLLAMA_MODEL is not set` | Set it in `.env` or the terminal (step 3) |
| `Ollama model '...' is not installed` | `ollama pull <name>`, or use a name from `ollama list` |
| `Cannot reach Ollama at ...` | Start Ollama; check `TRUSTNET_OLLAMA_BASE_URL` |
| `did not finish replying within ... seconds` | Raise the timeout or use a smaller model |
| `returned output that is not a valid response` | The model's reply failed validation; smaller models do this more often |
| Trust-weighted decision always abstains in live mode | Expected until agents have verified outcomes on record |

## Known limitations

- Evidence checks are word-overlap heuristics. They can miss paraphrases and can be
  fooled; remote URLs are never fetched.
- The live agents all use one model, so they are correlated and share one weight.
- Free-text answers are only grouped when they match after normalisation.
- The benchmark is synthetic. No evaluation on real labelled data is included.
- Reliability is recomputed from the full history on every run, which is fine for a
  prototype and slow for a large database.
