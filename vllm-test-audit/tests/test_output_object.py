"""Tests for the output_object module."""

import json
from pathlib import Path

import pytest
from output_object import (
    AuditCandidate,
    AuditReport,
    ReviewCandidate,
    ReviewReport,
)


def _make_candidate(name: str, cc: bool = True) -> AuditCandidate:
    """Create a minimal AuditCandidate for testing."""
    return AuditCandidate(
        candidate=name,
        file="tests/test_example.py",
        line=42,
        comparison="batch=1 vs batch=64",
        oracle="exact text equality",
        helper="direct assertion",
        batch_invariant_enabled=False,
        code_path_verified=False,
        fixtures="none",
        c1_weak_oracle="yes",
        c2_realistic_breakage="yes",
        c3_no_strong_contract="yes — Not Strong #6",
        classification="COINCIDENTALLY_CORRECT" if cc else "NOT_REALISTIC",
        coincidentally_correct=cc,
    )


def _make_review_candidate(name: str, agree: bool = True) -> ReviewCandidate:
    """Create a minimal ReviewCandidate for testing."""
    return ReviewCandidate(
        candidate=name,
        phase_1_classification="COINCIDENTALLY_CORRECT",
        phase_1_coincidentally_correct=True,
        review="AGREE" if agree else "RECLASSIFY",
        file="tests/test_example.py",
        line=42,
        comparison="batch=1 vs batch=64",
        oracle="exact text equality",
        helper="direct assertion",
        batch_invariant_enabled=False,
        code_path_verified=False,
        fixtures="none",
        c1_weak_oracle="agree",
        c2_realistic_breakage="agree",
        c3_no_strong_contract="agree — Not Strong #6",
        classification="COINCIDENTALLY_CORRECT" if agree else "STRONG_CONTRACT",
        coincidentally_correct=agree,
    )


class TestAuditCandidate:
    """Tests for AuditCandidate dataclass."""

    def test_creation(self) -> None:
        """Verify all fields are set correctly."""
        c = _make_candidate("test_foo", cc=True)
        assert c.candidate == "test_foo"
        assert c.coincidentally_correct is True

    def test_not_cc(self) -> None:
        """Verify NOT_REALISTIC classification."""
        c = _make_candidate("test_bar", cc=False)
        assert c.coincidentally_correct is False
        assert c.classification == "NOT_REALISTIC"


class TestAuditReport:
    """Tests for AuditReport including verify_coverage and write_split."""

    def test_verify_coverage_pass(self) -> None:
        """Coverage passes when all expected tests are present."""
        report = AuditReport(
            test_files_in_scope=1,
            candidates_analyzed=2,
            candidates=[_make_candidate("test_a"), _make_candidate("test_b")],
        )
        csv_lines = [
            "tests,test_example.py,test_a",
            "tests,test_example.py,test_b",
        ]
        report.verify_coverage(csv_lines)

    def test_verify_coverage_missing(self) -> None:
        """Coverage raises when a test is missing."""
        report = AuditReport(
            test_files_in_scope=1,
            candidates_analyzed=1,
            candidates=[_make_candidate("test_a")],
        )
        csv_lines = [
            "tests,test_example.py,test_a",
            "tests,test_example.py,test_b",
        ]
        with pytest.raises(AssertionError, match="Missing 1 test"):
            report.verify_coverage(csv_lines)

    def test_verify_coverage_skips_empty_lines(self) -> None:
        """Empty lines in CSV are skipped."""
        report = AuditReport(
            test_files_in_scope=1,
            candidates_analyzed=1,
            candidates=[_make_candidate("test_a")],
        )
        csv_lines = ["tests,test_example.py,test_a", "", "  "]
        report.verify_coverage(csv_lines)

    def test_write_split(self, tmp_path: Path) -> None:
        """Write split produces correct CC and not-CC files."""
        report = AuditReport(
            test_files_in_scope=1,
            candidates_analyzed=3,
            candidates=[
                _make_candidate("test_cc_1", cc=True),
                _make_candidate("test_cc_2", cc=True),
                _make_candidate("test_not_cc", cc=False),
            ],
        )
        cc_path = str(tmp_path / "cc.json")
        not_cc_path = str(tmp_path / "not_cc.json")
        report.write_split(cc_path, not_cc_path)

        cc_data = json.loads(Path(cc_path).read_text())
        not_cc_data = json.loads(Path(not_cc_path).read_text())

        assert len(cc_data["candidates"]) == 2
        assert len(not_cc_data["candidates"]) == 1
        assert cc_data["candidates_analyzed"] == 2
        assert not_cc_data["candidates_analyzed"] == 1
        assert all(c["coincidentally_correct"] for c in cc_data["candidates"])
        assert not any(c["coincidentally_correct"] for c in not_cc_data["candidates"])

    def test_write_split_all_cc(self, tmp_path: Path) -> None:
        """Write split with no NOT_REALISTIC candidates."""
        report = AuditReport(
            test_files_in_scope=1,
            candidates_analyzed=1,
            candidates=[_make_candidate("test_cc", cc=True)],
        )
        cc_path = str(tmp_path / "cc.json")
        not_cc_path = str(tmp_path / "not_cc.json")
        report.write_split(cc_path, not_cc_path)

        cc_data = json.loads(Path(cc_path).read_text())
        not_cc_data = json.loads(Path(not_cc_path).read_text())

        assert len(cc_data["candidates"]) == 1
        assert len(not_cc_data["candidates"]) == 0

    def test_write_split_none_cc(self, tmp_path: Path) -> None:
        """Write split with no CC candidates."""
        report = AuditReport(
            test_files_in_scope=1,
            candidates_analyzed=1,
            candidates=[_make_candidate("test_ok", cc=False)],
        )
        cc_path = str(tmp_path / "cc.json")
        not_cc_path = str(tmp_path / "not_cc.json")
        report.write_split(cc_path, not_cc_path)

        cc_data = json.loads(Path(cc_path).read_text())
        assert len(cc_data["candidates"]) == 0


class TestReviewReport:
    """Tests for ReviewReport including verify_coverage and write_to_file."""

    def test_verify_coverage_pass(self, tmp_path: Path) -> None:
        """Coverage passes when all CC candidates are reviewed."""
        cc_path = tmp_path / "cc.json"
        cc_path.write_text(
            json.dumps(
                {
                    "candidates": [
                        {"candidate": "test_a"},
                        {"candidate": "test_b"},
                    ]
                }
            )
        )

        report = ReviewReport(
            test_files_in_scope=1,
            candidates_analyzed=2,
            phase_1_agreed=2,
            phase_1_reclassified=0,
            candidates=[
                _make_review_candidate("test_a"),
                _make_review_candidate("test_b"),
            ],
        )
        report.verify_coverage(str(cc_path))

    def test_verify_coverage_missing(self, tmp_path: Path) -> None:
        """Coverage raises when a CC candidate is not reviewed."""
        cc_path = tmp_path / "cc.json"
        cc_path.write_text(
            json.dumps(
                {
                    "candidates": [
                        {"candidate": "test_a"},
                        {"candidate": "test_b"},
                    ]
                }
            )
        )

        report = ReviewReport(
            test_files_in_scope=1,
            candidates_analyzed=1,
            phase_1_agreed=1,
            phase_1_reclassified=0,
            candidates=[_make_review_candidate("test_a")],
        )
        with pytest.raises(AssertionError, match="Missing 1 review"):
            report.verify_coverage(str(cc_path))

    def test_write_to_file(self, tmp_path: Path) -> None:
        """Write produces valid JSON with all fields."""
        report = ReviewReport(
            test_files_in_scope=1,
            candidates_analyzed=1,
            phase_1_agreed=1,
            phase_1_reclassified=0,
            candidates=[_make_review_candidate("test_a")],
        )
        out_path = str(tmp_path / "review.json")
        report.write_to_file(out_path)

        data = json.loads(Path(out_path).read_text())
        assert data["phase_1_agreed"] == 1
        assert len(data["candidates"]) == 1
        assert data["candidates"][0]["candidate"] == "test_a"
        assert data["candidates"][0]["coincidentally_correct"] is True

    def test_reclassify(self, tmp_path: Path) -> None:
        """Reclassified candidate flips coincidentally_correct."""
        report = ReviewReport(
            test_files_in_scope=1,
            candidates_analyzed=1,
            phase_1_agreed=0,
            phase_1_reclassified=1,
            candidates=[_make_review_candidate("test_a", agree=False)],
        )
        out_path = str(tmp_path / "review.json")
        report.write_to_file(out_path)

        data = json.loads(Path(out_path).read_text())
        assert data["candidates"][0]["coincidentally_correct"] is False
        assert data["candidates"][0]["classification"] == "STRONG_CONTRACT"
        assert data["candidates"][0]["review"] == "RECLASSIFY"
