"""Curated enterprise showcase scenarios across diverse mission-critical domains.

Includes verified questions, options, ground-truth outcomes, and evidence citations
pointing to real files in data/sources/.
"""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ShowcaseScenario:
    task_id: str
    domain: str
    title: str
    question: str
    options: tuple[str, ...]
    correct_answer: str | None
    is_answerable: bool
    reference: str
    excerpt: str
    resolution_source: str
    explanation: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "domain": self.domain,
            "title": self.title,
            "question": self.question,
            "options": list(self.options),
            "correct_answer": self.correct_answer,
            "is_answerable": self.is_answerable,
            "reference": self.reference,
            "excerpt": self.excerpt,
            "resolution_source": self.resolution_source,
            "explanation": self.explanation,
        }


SHOWCASE_SCENARIOS: tuple[ShowcaseScenario, ...] = (
    ShowcaseScenario(
        task_id="showcase-health-001",
        domain="healthcare",
        title="Healthcare: High-Risk Anticoagulant Interaction",
        question="A 68-year-old patient on Warfarin therapy is prescribed high-dose Aspirin for joint pain. Should the clinical team approve concurrent administration, or discontinue concurrent NSAIDs and monitor INR?",
        options=(
            "Discontinue concurrent NSAID therapy and monitor INR",
            "Approve concurrent administration with no dosage changes",
            "Double the Warfarin dosage immediately",
            "Administer NSAID with oral vitamin K",
        ),
        correct_answer="Discontinue concurrent NSAID therapy and monitor INR",
        is_answerable=True,
        reference="clinical_warfarin_guideline.txt",
        excerpt="Discontinue concurrent NSAID therapy and monitor INR levels weekly.",
        resolution_source="Clinical Practice Guideline CP-402 Audit",
        explanation="CP-402 states concomitant Warfarin and high-dose NSAIDs increases gastrointestinal bleeding risk 3.8-fold without thrombotic benefit.",
    ),
    ShowcaseScenario(
        task_id="showcase-cyber-002",
        domain="cybersecurity",
        title="Cybersecurity: CVE-2024-8812 Authentication Bypass Remediation",
        question="Production AuthGateway v2.3 is vulnerable to CVE-2024-8812 (unauthenticated JWT algorithm 'none' exploit). What is the mandatory immediate remediation?",
        options=(
            "Upgrade to AuthGateway v2.5 or apply hotfix patch 2024-8812-P1",
            "Disable token validation headers on all API gateways",
            "Switch all client tokens to unencrypted plaintext cookies",
            "Reboot gateway instances nightly to flush tokens",
        ),
        correct_answer="Upgrade to AuthGateway v2.5 or apply hotfix patch 2024-8812-P1",
        is_answerable=True,
        reference="cve_2024_auth_bypass.txt",
        excerpt="Upgrade to AuthGateway v2.5 or apply hotfix patch 2024-8812-P1.",
        resolution_source="CISA / SEC Advisory SEC-2024-8812",
        explanation="Official security advisory prohibits disabling token validation and mandates v2.5 upgrade or 2024-8812-P1 hotfix.",
    ),
    ShowcaseScenario(
        task_id="showcase-fin-003",
        domain="finance",
        title="Finance: AML Cross-Border Wire Compliance Threshold",
        question="A customer initiates an international wire transfer of $45,000 USD from a non-FATF compliant jurisdiction. Does regulatory compliance permit automated straight-through processing or require enhanced due diligence (EDD) with a 48-hour compliance hold?",
        options=(
            "Require enhanced customer due diligence (EDD) and mandatory 48-hour compliance hold",
            "Execute automated straight-through processing immediately",
            "Waive all compliance checks if transfer fee is paid",
            "Split wire transfer into smaller transactions to bypass hold",
        ),
        correct_answer="Require enhanced customer due diligence (EDD) and mandatory 48-hour compliance hold",
        is_answerable=True,
        reference="aml_compliance_rule31.txt",
        excerpt="transfers exceeding $10,000 USD originating from non-FATF compliant jurisdictions require enhanced customer due diligence (EDD) and mandatory 48-hour compliance hold.",
        resolution_source="FinCEN / BSA Advisory AML-31.4",
        explanation="Transfers exceeding $10,000 from non-FATF regions trigger mandatory 48-hour compliance holds under BSA Section 5318.",
    ),
    ShowcaseScenario(
        task_id="showcase-arch-004",
        domain="engineering",
        title="Engineering: Distributed System P99 SLA Architecture",
        question="A tier-1 API service handling 65,000 req/sec experiences P99 latency degradation. Should the team immediately rewrite the entire codebase in Go, or implement Redis caching and connection pooling per SLA-109?",
        options=(
            "Implement Redis caching layer and connection pooling before authorizing rewrite",
            "Authorize immediate total service rewrite in Go without caching",
            "Disable all database replication to save memory",
            "Increase thread count to 10,000 without connection bounds",
        ),
        correct_answer="Implement Redis caching layer and connection pooling before authorizing rewrite",
        is_answerable=True,
        reference="distributed_cache_sla.txt",
        excerpt="Implement Redis caching layer and connection pooling before authorizing rewrite.",
        resolution_source="Infrastructure Architecture Review SLA-109",
        explanation="SLA-109 specifies that Redis caching provides sub-5ms P99 latency and total rewrites are unjustified unless GC pauses exceed 120ms.",
    ),
    ShowcaseScenario(
        task_id="showcase-legal-005",
        domain="legal",
        title="Legal: GDPR Article 17 Erasure vs Statutory Tax Retention",
        question="A former client submits an automated GDPR Article 17 erasure request for accounting records dated 3 years ago. Should the data protection officer permanently delete the financial records immediately or verify against statutory retention periods?",
        options=(
            "Verify against statutory tax and audit retention periods before deletion",
            "Permanently delete all financial transaction records immediately",
            "Publish records to a public repository to satisfy erasure",
            "Charge the client a penalty fee for exercising GDPR rights",
        ),
        correct_answer="Verify against statutory tax and audit retention periods before deletion",
        is_answerable=True,
        reference="gdpr_article17_erasure.txt",
        excerpt="Automated deletion requests must be verified against statutory tax and audit retention periods of 7 years.",
        resolution_source="Data Protection Authority Guidance GDPR-Art17",
        explanation="GDPR Article 17(3)(b) explicitly exempts data necessary for compliance with statutory legal obligations (such as 7-year tax records).",
    ),
    ShowcaseScenario(
        task_id="showcase-ledger-006",
        domain="ledger",
        title="Ledger: Q3 Accounts Payable Reconciliation",
        question="What is the verified reconciled accounts payable balance for Item 4082 following supplier credit note CN-9912?",
        options=(
            "$14,250.00 USD",
            "$89,400.00 USD",
            "$0.00 USD",
            "$5,500.00 USD",
        ),
        correct_answer="$14,250.00 USD",
        is_answerable=True,
        reference="financial_ledger_q3.txt",
        excerpt="Item 4082 accounts payable variance is officially reconciled at $14,250.00 USD after supplier credit note CN-9912.",
        resolution_source="Corporate Internal Audit Q3-2026",
        explanation="Official audit certifies $14,250.00 USD as the reconciled accounts payable amount.",
    ),
    ShowcaseScenario(
        task_id="showcase-inv-007",
        domain="inventory",
        title="Inventory: Critical Safety Stock Reorder Threshold",
        question="What is the verified physical inventory count for SKU-8841 at the Central Distribution Hub in report 2026-W41?",
        options=(
            "420 units",
            "150 units",
            "200 units",
            "50 units",
        ),
        correct_answer="420 units",
        is_answerable=True,
        reference="supply_chain_inventory_2026.txt",
        excerpt="Item SKU-8841 physical inventory count at Central Distribution Hub is verified at 420 units.",
        resolution_source="Warehouse Stock Audit Report 2026-W41",
        explanation="Physical count verified at 420 units, well above safety threshold of 150 units.",
    ),
    ShowcaseScenario(
        task_id="showcase-unanswerable-008",
        domain="cybersecurity",
        title="Unanswerable Test: Undisclosed Zero-Day Breach Scope",
        question="What were the exact IP addresses and exfiltrated record count of the unconfirmed incident reported by an anonymous forum post on May 3rd? No incident telemetry or logs exist.",
        options=(
            "IP 192.168.1.100 with 50,000 records exfiltrated",
            "IP 10.0.0.1 with 1,000,000 records exfiltrated",
            "IP 172.16.0.5 with 0 records exfiltrated",
            "All internal IP addresses were completely compromised",
        ),
        correct_answer=None,  # Intentionally unanswerable!
        is_answerable=False,
        reference="",
        excerpt="",
        resolution_source="Forensic Telemetry Gap Report",
        explanation="Information is completely unverified and unrecorded. A trustworthy decision architecture MUST abstain rather than guess an invented IP.",
    ),
)
