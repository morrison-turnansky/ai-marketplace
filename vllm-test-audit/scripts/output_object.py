"""Structured output objects for the vLLM test oracle auditor."""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class AuditCandidate:
    """Phase 1 evidence for a single test candidate."""

    candidate: str
    file: str
    line: int
    comparison: str
    oracle: str
    helper: str
    batch_invariant_enabled: bool
    code_path_verified: bool
    fixtures: str
    c1_weak_oracle: str
    c2_realistic_breakage: str
    c3_no_strong_contract: str
    classification: str
    coincidentally_correct: bool
    code_snippet: str = ""


@dataclass
class ReviewCandidate:
    """Phase 2 review verdict for a single test candidate."""

    candidate: str
    phase_1_classification: str
    phase_1_coincidentally_correct: bool
    review: str
    file: str
    line: int
    comparison: str
    oracle: str
    helper: str
    batch_invariant_enabled: bool
    code_path_verified: bool
    fixtures: str
    c1_weak_oracle: str
    c2_realistic_breakage: str
    c3_no_strong_contract: str
    classification: str
    coincidentally_correct: bool
    code_snippet: str = ""


@dataclass
class AuditReport:
    """Full Phase 1 audit report."""

    test_files_in_scope: int
    candidates_analyzed: int
    candidates: list[AuditCandidate] = field(default_factory=list)

    def verify_coverage(self, expected_tests: list[str]) -> None:
        """Assert every test from list_tests.py output has a candidate entry.

        Args:
            expected_tests: CSV lines from list_tests.py in DIR,FILE,FUNCTION format.

        Raises:
            AssertionError: If any expected test is missing from the report.
        """
        expected = set()
        for line in expected_tests:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) == 3:
                expected.add(parts[2])
        analyzed = {c.candidate for c in self.candidates}
        missing = expected - analyzed
        if missing:
            raise AssertionError(
                f"Missing {len(missing)} test(s) from report:\n" + "\n".join(sorted(missing))
            )

    def write_split(self, cc_path: str, not_cc_path: str) -> None:
        """Write CC and not-CC candidates to separate JSON files.

        Verifies that cc + not_cc == total candidates before writing.

        Args:
            cc_path: Output path for coincidentally correct candidates.
            not_cc_path: Output path for not coincidentally correct candidates.

        Raises:
            AssertionError: If the split counts don't add up.
        """
        cc = [c for c in self.candidates if c.coincidentally_correct]
        not_cc = [c for c in self.candidates if not c.coincidentally_correct]

        if len(cc) + len(not_cc) != len(self.candidates):
            raise AssertionError(
                f"Split mismatch: {len(cc)} CC + {len(not_cc)} not-CC "
                f"!= {len(self.candidates)} total"
            )

        cc_report = {
            "test_files_in_scope": self.test_files_in_scope,
            "candidates_analyzed": len(cc),
            "candidates": [asdict(c) for c in cc],
        }
        not_cc_report = {
            "test_files_in_scope": self.test_files_in_scope,
            "candidates_analyzed": len(not_cc),
            "candidates": [asdict(c) for c in not_cc],
        }

        Path(cc_path).write_text(json.dumps(cc_report, indent=2) + "\n")
        Path(not_cc_path).write_text(json.dumps(not_cc_report, indent=2) + "\n")

        print(
            f"Total: {len(self.candidates)}, "
            f"CC: {len(cc)} -> {cc_path}, "
            f"Not CC: {len(not_cc)} -> {not_cc_path}"
        )


@dataclass
class ReviewReport:
    """Full Phase 2 review report."""

    test_files_in_scope: int
    candidates_analyzed: int
    phase_1_agreed: int
    phase_1_reclassified: int
    candidates: list[ReviewCandidate] = field(default_factory=list)

    def verify_coverage(self, cc_input_path: str) -> None:
        """Assert every CC candidate from Phase 1 has a review entry.

        Args:
            cc_input_path: Path to the Phase 1 CC JSON file.

        Raises:
            AssertionError: If any CC candidate is missing a review.
        """
        cc_data = json.loads(Path(cc_input_path).read_text())
        expected = {c["candidate"] for c in cc_data["candidates"]}
        reviewed = {c.candidate for c in self.candidates}
        missing = expected - reviewed
        if missing:
            raise AssertionError(
                f"Missing {len(missing)} review(s):\n" + "\n".join(sorted(missing))
            )

    def write_to_file(self, file_name: str) -> None:
        """Write report as JSON to the given file path.

        Args:
            file_name: Output file path.
        """
        Path(file_name).write_text(json.dumps(asdict(self), indent=2) + "\n")
