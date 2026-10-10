from pathlib import Path

import pytest
from pydantic import ValidationError

from trustnet.evidence import (
    DirectorySourceStore,
    EvidenceEvaluator,
    InMemorySourceStore,
    LLMEvidenceJudge,
    is_url,
)
from trustnet.providers import DeterministicProvider, GenerationRequest, ProviderUnavailableError
from trustnet.schemas import Evidence, EvidenceAssessment, EvidenceSourceType, EvidenceStatus

SOURCE = (
    "MOCK REPORT. The bridge was completed in 1932 and is 503 metres long.\n"
    "The bridge is not open to heavy vehicles. Maintenance is funded by the city."
)


def evidence(claim: str, reference: str | None = "report") -> Evidence:
    return Evidence(
        evidence_id="e1",
        source_type=EvidenceSourceType.DOCUMENT,
        excerpt=claim,
        reference=reference,
    )


def evaluator(**kwargs: object) -> EvidenceEvaluator:
    return EvidenceEvaluator(InMemorySourceStore({"report": SOURCE}), **kwargs)


def test_claim_stated_by_its_source_is_verified() -> None:
    result = evaluator().assess(evidence("The bridge was completed in 1932."))
    assert result.status is EvidenceStatus.VERIFIED
    assert result.source_available is True
    assert result.relevance == 1.0
    assert result.method == "deterministic"


def test_claim_with_different_figures_is_contradictory() -> None:
    result = evaluator().assess(evidence("The bridge was completed in 1951."))
    assert result.status is EvidenceStatus.CONTRADICTORY
    assert "different figures" in result.note


def test_claim_with_opposite_polarity_is_contradictory() -> None:
    result = evaluator().assess(evidence("The bridge is open to heavy vehicles."))
    assert result.status is EvidenceStatus.CONTRADICTORY
    assert "opposite" in result.note


def test_claim_the_source_does_not_address_is_unsupported() -> None:
    result = evaluator().assess(evidence("Ticket prices doubled last winter."))
    assert result.status is EvidenceStatus.UNSUPPORTED
    assert result.source_available is True
    assert result.relevance < 0.5


def test_claim_without_a_reference_is_unsupported() -> None:
    for reference in (None, "", "   "):
        result = evaluator().assess(evidence("The bridge was completed in 1932.", reference))
        assert result.status is EvidenceStatus.UNSUPPORTED
        assert result.source_available is None
        assert "no reference" in result.note


def test_missing_source_is_unverified_not_refuted() -> None:
    result = evaluator().assess(evidence("The bridge was completed in 1932.", "lost-file"))
    assert result.status is EvidenceStatus.UNVERIFIED
    assert result.source_available is False


def test_remote_url_is_not_fetched_and_stays_unverified() -> None:
    result = evaluator().assess(evidence("Anything at all.", "https://example.invalid/page"))
    assert result.status is EvidenceStatus.UNVERIFIED
    assert result.source_available is None
    assert "not fetched" in result.note


def test_nothing_is_verified_without_a_source_store() -> None:
    result = EvidenceEvaluator().assess(evidence("The bridge was completed in 1932."))
    assert result.status is EvidenceStatus.UNVERIFIED


@pytest.mark.parametrize("status", [EvidenceStatus.VERIFIED, EvidenceStatus.CONTRADICTORY])
@pytest.mark.parametrize("available", [None, False])
def test_a_definite_verdict_cannot_exist_without_a_source_that_was_read(
    status: EvidenceStatus, available: bool | None
) -> None:
    with pytest.raises(ValidationError, match="actually read"):
        EvidenceAssessment(evidence_id="e1", status=status, source_available=available)


@pytest.mark.parametrize(
    "reference, expected",
    [("https://example.org/a", True), ("HTTP://example.org", True), ("report.txt", False),
     ("https://", False), ("ftp://example.org", False)],
)
def test_url_references_are_recognised(reference: str, expected: bool) -> None:
    assert is_url(reference) is expected


# --- source directory -------------------------------------------------------------------


def test_directory_store_reads_files_inside_its_root(tmp_path: Path) -> None:
    (tmp_path / "sources").mkdir()
    (tmp_path / "sources" / "report.txt").write_text(SOURCE, encoding="utf-8")
    store = DirectorySourceStore(tmp_path / "sources")

    assert store.fetch("report.txt") == SOURCE
    assert store.fetch("file:report.txt") == SOURCE
    assert store.fetch("absent.txt") is None


def test_directory_store_refuses_paths_outside_its_root(tmp_path: Path) -> None:
    (tmp_path / "sources").mkdir()
    secret = tmp_path / "secret.txt"
    secret.write_text("MOCK private", encoding="utf-8")
    store = DirectorySourceStore(tmp_path / "sources")

    assert store.fetch("../secret.txt") is None
    assert store.fetch(str(secret)) is None
    assert store.fetch("https://example.org/report.txt") is None


# --- optional LLM judge -----------------------------------------------------------------


def judge_replying(text: str) -> tuple[LLMEvidenceJudge, DeterministicProvider]:
    provider = DeterministicProvider(default=text)
    return LLMEvidenceJudge(provider), provider


def test_llm_judge_is_off_by_default_and_never_called() -> None:
    judge, provider = judge_replying('{"verdict": "supported"}')
    evaluator().assess(evidence("The bridge was completed in 1932."))
    assert provider.calls == []
    assert judge is not None


def test_llm_judge_verdict_is_used_and_labelled() -> None:
    judge, provider = judge_replying('{"verdict": "contradicted"}')
    result = evaluator(judge=judge).assess(evidence("The bridge was completed in 1932."))

    assert result.status is EvidenceStatus.CONTRADICTORY
    assert result.method == "llm"
    assert "deterministic said" in result.note
    assert "CLAIM:" in provider.calls[0].prompt and "SOURCE:" in provider.calls[0].prompt


def test_llm_judge_is_never_asked_about_a_source_that_could_not_be_read() -> None:
    judge, provider = judge_replying('{"verdict": "supported"}')
    checker = evaluator(judge=judge)

    for reference in (None, "lost-file", "https://example.invalid/page"):
        result = checker.assess(evidence("The bridge was completed in 1932.", reference))
        assert result.status is not EvidenceStatus.VERIFIED
    assert provider.calls == []


def test_failed_llm_judge_falls_back_to_the_deterministic_result() -> None:
    class DownProvider:
        name = "ollama"
        model = "m"

        def generate(self, request: GenerationRequest) -> None:
            raise ProviderUnavailableError("Cannot reach Ollama")

    result = evaluator(judge=LLMEvidenceJudge(DownProvider())).assess(
        evidence("The bridge was completed in 1932.")
    )
    assert result.status is EvidenceStatus.VERIFIED
    assert result.method == "deterministic"
    assert "LLM judge failed" in result.note and "Cannot reach Ollama" in result.note


def test_invalid_llm_judge_output_falls_back_to_the_deterministic_result() -> None:
    judge, _ = judge_replying("MOCK not json")
    result = evaluator(judge=judge).assess(evidence("The bridge was completed in 1951."))
    assert result.status is EvidenceStatus.CONTRADICTORY
    assert result.method == "deterministic"
