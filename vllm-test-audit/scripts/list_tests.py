"""List test functions from PRs, directories, files, or file::function targets.

Detects input type automatically. Always outputs DIR,FILE,FUNCTION CSV.

For PRs:
  - New test files: extracts added test function definitions from diff.
  - Modified test files: maps changed line numbers to enclosing test function.

Usage:
    python list_tests.py https://github.com/vllm-project/vllm/pull/1234
    python list_tests.py 1234 --repo vllm-project/vllm
    python list_tests.py tests/compile/correctness_e2e/
    python list_tests.py tests/v1/e2e/spec_decode/test_spec_decode.py
    python list_tests.py tests/v1/e2e/spec_decode/test_spec_decode.py::test_mtp_correctness
"""

import os
import re
import subprocess
import sys
from pathlib import Path

TEST_FUNC_RE = re.compile(r"(?:    )?(?:async )?def (test_\w+)")
ADDED_FUNC_RE = re.compile(r"\+\s*(?:async )?def (test_\w+)")


def list_functions(file_path: str) -> list[tuple[str, str, str]]:
    """List all test functions in a single test file.

    Args:
        file_path: Path to a test_*.py file.

    Returns:
        List of (directory, filename, function_name) tuples.
    """
    results = []
    path = Path(file_path)
    if not path.is_file():
        return results
    for line in path.read_text().splitlines():
        match = TEST_FUNC_RE.match(line)
        if match:
            results.append((str(path.parent), path.name, match.group(1)))
    return results


def list_from_directory(dir_path: str) -> list[tuple[str, str, str]]:
    """List all test functions in all test_*.py files in a directory.

    Args:
        dir_path: Path to a directory to search recursively.

    Returns:
        List of (directory, filename, function_name) tuples.
    """
    results = []
    for path in sorted(Path(dir_path).rglob("test_*.py")):
        results.extend(list_functions(str(path)))
    return results


def get_pr_diff(args: list[str]) -> str:
    """Fetch PR diff via gh CLI.

    Args:
        args: Arguments to pass to ``gh pr diff`` (PR URL/number, --repo, etc).

    Returns:
        The unified diff as a string.
    """
    result = subprocess.run(
        ["gh", "pr", "diff", *args],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print("Failed to fetch PR diff", file=sys.stderr)
        sys.exit(1)
    return result.stdout


def _build_test_func_map(file_path: str) -> list[tuple[int, str]]:
    """Build a mapping of line numbers to test function names.

    Args:
        file_path: Path to a Python test file.

    Returns:
        List of (line_number, function_name) tuples, ordered by line number.
    """
    test_starts = []
    with open(file_path) as source_file:
        for line_num, line in enumerate(source_file, 1):
            match = TEST_FUNC_RE.match(line)
            if match:
                test_starts.append((line_num, match.group(1)))
    return test_starts


def list_from_pr(args: list[str]) -> list[tuple[str, str, str]]:
    """Extract changed/added test functions from a PR diff.

    For new files, extracts ``+def test_`` lines from the diff. For modified
    files, maps changed line numbers to enclosing test functions by reading
    the local checkout.

    Args:
        args: Arguments to pass to ``gh pr diff``.

    Returns:
        Sorted list of (directory, filename, function_name) tuples.
    """
    diff = get_pr_diff(args)

    current_file: str | None = None
    current_line = 0
    prev_was_dev_null = False

    new_funcs: dict[str, set[str]] = {}
    file_lines: dict[str, set[int]] = {}
    new_files: set[str] = set()

    for line in diff.splitlines():
        if line.startswith("--- "):
            prev_was_dev_null = line.startswith("--- /dev/null")
        elif line.startswith("+++ b/"):
            file_path = line[6:]
            if "/test_" in file_path and file_path.endswith(".py"):
                current_file = file_path
                if prev_was_dev_null:
                    new_files.add(file_path)
            else:
                current_file = None
        elif line.startswith("@@"):
            hunk_match = re.match(r"@@ -\d+(?:,\d+)? \+(\d+)", line)
            if hunk_match:
                current_line = int(hunk_match.group(1))
        elif not current_file:
            pass
        elif line.startswith("+"):
            func_match = ADDED_FUNC_RE.match(line)
            if func_match:
                new_funcs.setdefault(current_file, set()).add(func_match.group(1))
            file_lines.setdefault(current_file, set()).add(current_line)
            current_line += 1
        elif line.startswith("-"):
            file_lines.setdefault(current_file, set()).add(current_line)
        else:
            current_line += 1

    all_files = sorted(set(list(new_funcs.keys()) + list(file_lines.keys())))
    results: set[tuple[str, str, str]] = set()

    for file_path in all_files:
        dir_name = os.path.dirname(file_path)
        base_name = os.path.basename(file_path)

        if file_path in new_files:
            for func_name in new_funcs.get(file_path, set()):
                results.add((dir_name, base_name, func_name))
        else:
            if os.path.isfile(file_path):
                test_starts = _build_test_func_map(file_path)
                for changed_line in file_lines.get(file_path, set()):
                    for start_line, func_name in reversed(test_starts):
                        if start_line <= changed_line:
                            results.add((dir_name, base_name, func_name))
                            break

            for func_name in new_funcs.get(file_path, set()):
                results.add((dir_name, base_name, func_name))

    return sorted(results)


def is_pr(arg: str) -> bool:
    """Check if the argument looks like a PR URL or number.

    Args:
        arg: Command-line argument to check.

    Returns:
        True if the argument is a GitHub PR URL or a plain number.
    """
    if re.match(r"^https://github\.com/.*/pull/\d+", arg):
        return True
    if re.match(r"^\d+$", arg):
        return True
    return False


def main() -> None:
    """Entry point for CLI usage."""
    if len(sys.argv) < 2:
        print(
            "Usage: list_tests.py <pr-url|pr-number|directory|file|file::function> "
            "[--repo owner/repo]",
            file=sys.stderr,
        )
        sys.exit(1)

    target = sys.argv[1]
    results: list[tuple[str, str, str]] = []

    if is_pr(target):
        results = list_from_pr(sys.argv[1:])
    elif "::" in target:
        file_path, func_name = target.split("::", 1)
        results = [(os.path.dirname(file_path), os.path.basename(file_path), func_name)]
    elif os.path.isfile(target) and "/test_" in target:
        results = list_functions(target)
    elif os.path.isdir(target):
        results = list_from_directory(target)

    for dir_name, base_name, func_name in results:
        print(f"{dir_name},{base_name},{func_name}")


if __name__ == "__main__":
    main()
