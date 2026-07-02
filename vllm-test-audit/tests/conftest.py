"""Shared fixtures for vllm-test-audit tests."""

from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture()
def pr_adds_tests_diff() -> str:
    """Diff from vllm PR #47185 — adds 11 new test functions to existing files."""
    return (FIXTURES_DIR / "pr_47185_adds_tests.diff").read_text()


@pytest.fixture()
def pr_adds_tests_expected() -> set[str]:
    """Expected test function names from PR #47185."""
    return {
        "test_commentary_with_no_recipient_creates_message",
        "test_function_recipient_creates_function_call",
        "test_builtin_recipient_creates_reasoning",
        "test_non_function_non_builtin_recipient_creates_mcp_call",
        "test_browser_search_recipient_respects_incomplete",
        "test_zero_delta_items_should_preserve_streaming_lifecycle",
        "test_reasoning_tokens_counting",
        "test_preamble_tokens_not_counted_as_reasoning",
        "test_commentary_with_recipient_counted_as_reasoning",
        "test_streaming_multi_turn_token_counting",
        "test_streaming_message_synchronization",
    }


@pytest.fixture()
def pr_modifies_test_diff() -> str:
    """Diff from vllm PR #47299 — modifies test_traces body, no new functions."""
    return (FIXTURES_DIR / "pr_47299_modifies_test.diff").read_text()


@pytest.fixture()
def pr_modifies_test_expected() -> set[str]:
    """Expected test function names from PR #47299."""
    return {"test_traces"}


@pytest.fixture()
def mock_tracing_file(tmp_path: Path) -> Path:
    """Create a mock test file matching the structure of test_tracing.py.

    The diff for PR #47299 modifies lines around line 25, inside test_traces
    which starts at line 20. This mock mirrors that structure so
    _build_test_func_map correctly maps changed lines to test_traces.
    """
    test_dir = tmp_path / "tests" / "v1" / "tracing"
    test_dir.mkdir(parents=True)
    test_file = test_dir / "test_tracing.py"
    lines = [
        "import pytest",  # 1
        "from unittest.mock import patch",  # 2
        "",  # 3
        "from vllm import something",  # 4
        "",  # 5
        "OTEL_EXPORTER = 'foo'",  # 6
        "",  # 7
        "",  # 8
        "def helper():",  # 9
        "    pass",  # 10
        "",  # 11
        "",  # 12
        "def another_helper():",  # 13
        "    pass",  # 14
        "",  # 15
        "",  # 16
        "@pytest.mark.parametrize('model', ['m'])",  # 17
        "@pytest.mark.parametrize('backend', ['b'])",  # 18
        "@pytest.mark.asyncio",  # 19
        "def test_traces(",  # 20
        "    monkeypatch,",  # 21
        "    model,",  # 22
        "    backend,",  # 23
        "):",  # 24
        "    with monkeypatch.context() as m:",  # 25
        "        m.setenv('FOO', 'true')",  # 26
        "        m.setenv('BAR', 'spawn')",  # 27
        "        pass",  # 28
        "",  # 29
        "",  # 30
        "",  # 31
        "",  # 32
        "",  # 33
        "",  # 34
        "",  # 35
        "",  # 36
        "",  # 37
        "",  # 38
        "",  # 39
        "def test_other_trace():",  # 40
        "    pass",  # 41
    ]
    test_file.write_text("\n".join(lines) + "\n")
    return tmp_path
