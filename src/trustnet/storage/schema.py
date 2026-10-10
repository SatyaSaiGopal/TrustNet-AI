"""SQLite schema, as an ordered list of migrations.

Append new migrations; never edit one that has shipped. The CHECK constraints
mirror the validators in ``trustnet.schemas`` so the rules hold even for rows
written outside the Pydantic models.
"""

_V1 = """
CREATE TABLE agents (
    agent_id      TEXT PRIMARY KEY,
    display_name  TEXT NOT NULL,
    provider      TEXT NOT NULL,
    model         TEXT NOT NULL,
    is_synthetic  INTEGER NOT NULL DEFAULT 0 CHECK (is_synthetic IN (0, 1)),
    created_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE TABLE tasks (
    task_id       TEXT PRIMARY KEY,
    question      TEXT NOT NULL,
    is_synthetic  INTEGER NOT NULL DEFAULT 0 CHECK (is_synthetic IN (0, 1)),
    created_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE TABLE agent_responses (
    response_id               TEXT PRIMARY KEY,
    task_id                   TEXT NOT NULL REFERENCES tasks (task_id),
    agent_id                  TEXT NOT NULL REFERENCES agents (agent_id),
    stance                    TEXT NOT NULL CHECK (stance IN ('answer', 'abstain')),
    answer                    TEXT,
    rationale                 TEXT NOT NULL DEFAULT '',
    self_reported_confidence  REAL CHECK (self_reported_confidence BETWEEN 0.0 AND 1.0),
    created_at                TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    CHECK (
        (stance = 'abstain' AND answer IS NULL)
        OR (stance = 'answer' AND answer IS NOT NULL AND answer <> '')
    )
);

CREATE TABLE evidence_items (
    evidence_id          TEXT PRIMARY KEY,
    response_id          TEXT NOT NULL REFERENCES agent_responses (response_id),
    source_type          TEXT NOT NULL
        CHECK (source_type IN ('document', 'dataset', 'tool_output', 'model_assertion')),
    excerpt              TEXT NOT NULL,
    reference            TEXT,
    verification_status  TEXT NOT NULL DEFAULT 'unverified'
        CHECK (verification_status IN ('unverified', 'verified', 'refuted')),
    CHECK (
        verification_status <> 'verified'
        OR (source_type <> 'model_assertion' AND reference IS NOT NULL AND reference <> '')
    )
);

CREATE TABLE outcomes (
    task_id            TEXT PRIMARY KEY REFERENCES tasks (task_id),
    correct_answer     TEXT NOT NULL,
    resolution_source  TEXT NOT NULL,
    resolved_at        TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE TABLE decisions (
    decision_id  TEXT PRIMARY KEY,
    task_id      TEXT NOT NULL REFERENCES tasks (task_id),
    outcome      TEXT NOT NULL CHECK (outcome IN ('decided', 'abstained')),
    answer       TEXT,
    rationale    TEXT NOT NULL DEFAULT '',
    created_at   TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    CHECK (
        (outcome = 'abstained' AND answer IS NULL)
        OR (outcome = 'decided' AND answer IS NOT NULL AND answer <> '')
    )
);

CREATE INDEX idx_agent_responses_task ON agent_responses (task_id);
CREATE INDEX idx_agent_responses_agent ON agent_responses (agent_id);
CREATE INDEX idx_evidence_items_response ON evidence_items (response_id);
CREATE INDEX idx_decisions_task ON decisions (task_id);
"""

# Level 3: task categories, response provenance, evidence assessments, reliability
# metrics and explained decisions. outcomes is rebuilt so correct_answer can be NULL,
# which records a task verified as unanswerable. Its primary key on task_id is what
# prevents a second outcome for the same task.
_V2 = """
ALTER TABLE tasks ADD COLUMN category TEXT NOT NULL DEFAULT 'general';
ALTER TABLE tasks ADD COLUMN equivalence_group TEXT;
ALTER TABLE tasks ADD COLUMN options TEXT NOT NULL DEFAULT '[]';

ALTER TABLE agent_responses ADD COLUMN uncertainties TEXT NOT NULL DEFAULT '[]';
ALTER TABLE agent_responses ADD COLUMN model_id TEXT NOT NULL DEFAULT 'unknown';
ALTER TABLE agent_responses ADD COLUMN is_mock INTEGER NOT NULL DEFAULT 0;
ALTER TABLE agent_responses ADD COLUMN prompt_version TEXT;

CREATE TABLE outcomes_v2 (
    task_id            TEXT PRIMARY KEY REFERENCES tasks (task_id),
    correct_answer     TEXT CHECK (correct_answer IS NULL OR correct_answer <> ''),
    resolution_source  TEXT NOT NULL CHECK (resolution_source <> ''),
    resolved_at        TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);
INSERT INTO outcomes_v2 SELECT task_id, correct_answer, resolution_source, resolved_at FROM outcomes;
DROP TABLE outcomes;
ALTER TABLE outcomes_v2 RENAME TO outcomes;

CREATE TABLE evidence_assessments (
    evidence_id       TEXT PRIMARY KEY REFERENCES evidence_items (evidence_id),
    status            TEXT NOT NULL
        CHECK (status IN ('verified', 'unsupported', 'contradictory', 'unverified')),
    source_available  INTEGER CHECK (source_available IN (0, 1)),
    relevance         REAL CHECK (relevance BETWEEN 0.0 AND 1.0),
    method            TEXT NOT NULL CHECK (method IN ('deterministic', 'llm')),
    note              TEXT NOT NULL DEFAULT '',
    assessed_at       TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    CHECK (status NOT IN ('verified', 'contradictory') OR source_available = 1)
);

-- category '' is the profile across all categories. signals holds the full TrustSignals.
CREATE TABLE reliability_metrics (
    agent_id              TEXT NOT NULL REFERENCES agents (agent_id),
    category              TEXT NOT NULL DEFAULT '',
    trust_score           REAL CHECK (trust_score BETWEEN 0.0 AND 1.0),
    resolved_sample_size  INTEGER NOT NULL CHECK (resolved_sample_size >= 0),
    signals               TEXT NOT NULL,
    computed_at           TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    PRIMARY KEY (agent_id, category)
);

ALTER TABLE decisions ADD COLUMN strategy TEXT NOT NULL DEFAULT 'unspecified';
ALTER TABLE decisions ADD COLUMN abstention_reason TEXT;
ALTER TABLE decisions ADD COLUMN support REAL;
ALTER TABLE decisions ADD COLUMN contributions TEXT NOT NULL DEFAULT '[]';
"""

MIGRATIONS: tuple[str, ...] = (_V1, _V2)
