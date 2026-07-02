"""Tests for the list_tests module."""

import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest
from list_tests import (
    _build_test_func_map,
    is_pr,
    list_from_directory,
    list_from_pr,
    list_functions,
)


class TestIsPr:
    """Tests for PR detection."""

    def test_github_url(self) -> None:
        """Full GitHub PR URL is detected."""
        assert is_pr("https://github.com/vllm-project/vllm/pull/1234")

    def test_plain_number(self) -> None:
        """Bare number is detected as PR."""
        assert is_pr("1234")

    def test_directory(self) -> None:
        """Directory path is not a PR."""
        assert not is_pr("tests/compile/")

    def test_file(self) -> None:
        """File path is not a PR."""
        assert not is_pr("tests/test_foo.py")

    def test_file_function(self) -> None:
        """file::function is not a PR."""
        assert not is_pr("tests/test_foo.py::test_bar")


class TestListFunctions:
    """Tests for listing test functions from a file."""

    def test_finds_test_functions(self, tmp_path: Path) -> None:
        """Extracts all def test_ functions from a file."""
        f = tmp_path / "test_example.py"
        f.write_text(
            textwrap.dedent("""\
            def test_foo():
                pass

            def test_bar():
                pass

            def helper():
                pass

            class TestStuff:
                def test_baz(self):
                    pass
        """)
        )
        results = list_functions(str(f))
        funcs = [r[2] for r in results]
        assert funcs == ["test_foo", "test_bar", "test_baz"]

    def test_async_test_functions(self, tmp_path: Path) -> None:
        """Extracts async def test_ functions."""
        f = tmp_path / "test_async.py"
        f.write_text(
            textwrap.dedent("""\
            async def test_async_thing():
                pass
        """)
        )
        results = list_functions(str(f))
        assert len(results) == 1
        assert results[0][2] == "test_async_thing"

    def test_nonexistent_file(self) -> None:
        """Returns empty for nonexistent file."""
        assert list_functions("/nonexistent/test_foo.py") == []

    def test_no_test_functions(self, tmp_path: Path) -> None:
        """Returns empty for file with no test functions."""
        f = tmp_path / "test_empty.py"
        f.write_text("def helper():\n    pass\n")
        assert list_functions(str(f)) == []

    def test_csv_format(self, tmp_path: Path) -> None:
        """Output tuples have (dir, filename, function) structure."""
        d = tmp_path / "tests" / "unit"
        d.mkdir(parents=True)
        f = d / "test_example.py"
        f.write_text("def test_one():\n    pass\n")
        results = list_functions(str(f))
        assert results[0] == (str(d), "test_example.py", "test_one")


class TestListFromDirectory:
    """Tests for listing test functions from a directory."""

    def test_finds_all_test_files(self, tmp_path: Path) -> None:
        """Finds test functions across multiple files."""
        (tmp_path / "test_a.py").write_text("def test_alpha():\n    pass\n")
        (tmp_path / "test_b.py").write_text("def test_beta():\n    pass\n")
        (tmp_path / "helper.py").write_text("def test_ignored():\n    pass\n")
        results = list_from_directory(str(tmp_path))
        funcs = [r[2] for r in results]
        assert "test_alpha" in funcs
        assert "test_beta" in funcs
        assert "test_ignored" not in funcs

    def test_recursive(self, tmp_path: Path) -> None:
        """Finds test files in subdirectories."""
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "test_deep.py").write_text("def test_nested():\n    pass\n")
        results = list_from_directory(str(tmp_path))
        assert any(r[2] == "test_nested" for r in results)

    def test_empty_directory(self, tmp_path: Path) -> None:
        """Returns empty for directory with no test files."""
        assert list_from_directory(str(tmp_path)) == []


class TestListFromPr:
    """Tests for extracting test functions from PR diffs."""

    def test_new_test_functions_detected(
        self, pr_adds_tests_diff: str, pr_adds_tests_expected: set[str]
    ) -> None:
        """PR that adds new test functions to existing files."""
        with patch("list_tests.get_pr_diff", return_value=pr_adds_tests_diff):
            results = list_from_pr(["47185"])
        found_funcs = {result[2] for result in results}
        assert pr_adds_tests_expected == found_funcs

    def test_modified_test_function_detected(
        self,
        pr_modifies_test_diff: str,
        pr_modifies_test_expected: set[str],
        mock_tracing_file: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """PR that modifies an existing test function body."""
        monkeypatch.chdir(mock_tracing_file)
        with patch("list_tests.get_pr_diff", return_value=pr_modifies_test_diff):
            results = list_from_pr(["47299"])
        found_funcs = {result[2] for result in results}
        assert pr_modifies_test_expected == found_funcs

    def test_empty_diff(self) -> None:
        """Empty diff returns no results."""
        with patch("list_tests.get_pr_diff", return_value=""):
            results = list_from_pr(["99999"])
        assert results == []

    def test_diff_with_no_test_files(self) -> None:
        """Diff that only touches non-test files returns no results."""
        diff = (
            "diff --git a/src/main.py b/src/main.py\n"
            "--- a/src/main.py\n"
            "+++ b/src/main.py\n"
            "@@ -1,3 +1,4 @@\n"
            " import os\n"
            "+import sys\n"
            " def main():\n"
            "     pass\n"
        )
        with patch("list_tests.get_pr_diff", return_value=diff):
            results = list_from_pr(["99999"])
        assert results == []

    def test_csv_format(self, pr_adds_tests_diff: str) -> None:
        """PR results have (directory, filename, function) structure."""
        with patch("list_tests.get_pr_diff", return_value=pr_adds_tests_diff):
            results = list_from_pr(["47185"])
        for _dir_name, base_name, func_name in results:
            assert base_name.startswith("test_")
            assert base_name.endswith(".py")
            assert func_name.startswith("test_")


class TestBuildTestFuncMap:
    """Tests for mapping line numbers to test functions."""

    def test_maps_lines_to_functions(self, tmp_path: Path) -> None:
        """Builds correct line->function mapping."""
        f = tmp_path / "test_example.py"
        f.write_text(
            textwrap.dedent("""\
            import os

            def test_first():
                x = 1
                assert x == 1

            def test_second():
                y = 2
                assert y == 2

            def helper():
                pass
        """)
        )
        mapping = _build_test_func_map(str(f))
        assert len(mapping) == 2
        assert mapping[0] == (3, "test_first")
        assert mapping[1] == (7, "test_second")

    def test_indented_methods(self, tmp_path: Path) -> None:
        """Handles class-level test methods."""
        f = tmp_path / "test_class.py"
        f.write_text(
            textwrap.dedent("""\
            class TestFoo:
                def test_method(self):
                    pass
        """)
        )
        mapping = _build_test_func_map(str(f))
        assert len(mapping) == 1
        assert mapping[0][1] == "test_method"
