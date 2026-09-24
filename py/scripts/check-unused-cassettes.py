#!/usr/bin/env python3
"""Find integration cassette files that no test loads.

``check-stale-cassettes.py`` catches whole version directories that fell out of
the matrix. This script catches individual files inside valid directories that
nothing reads any more: cassettes left behind by deleted or renamed tests, and
cassettes for tests that always skip at that version (e.g. a feature the old
SDK doesn't have).

It works from what tests actually open rather than from name matching. Tests
record every cassette they read when ``BRAINTRUST_CASSETTE_USAGE_DIR`` is set
(see ``src/braintrust/_test_cassette_usage.py``), and this script compares
those reads with the files on disk.

Usage:
    # Run every nox session that reads the given integrations' cassettes
    # (all matrix versions, replay-only) and report unused files.
    python scripts/check-unused-cassettes.py run anthropic openai
    python scripts/check-unused-cassettes.py run --all
    python scripts/check-unused-cassettes.py run anthropic --clean

    # Report from usage logs collected elsewhere (e.g. merged CI artifacts).
    # Only meaningful when the logs cover every session that reads the
    # reported integrations' cassettes.
    python scripts/check-unused-cassettes.py report --usage-dir DIR [INTEGRATION ...]
"""

import argparse
import ast
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict


_PROJECT_DIR = pathlib.Path(__file__).resolve().parent.parent
_PACKAGE_DIR = _PROJECT_DIR / "src" / "braintrust"
_INTEGRATIONS_DIR = _PACKAGE_DIR / "integrations"
_NOXFILE = _PROJECT_DIR / "noxfile.py"

USAGE_DIR_ENV = "BRAINTRUST_CASSETTE_USAGE_DIR"


def integrations_with_cassettes() -> list[str]:
    return sorted(
        d.name
        for d in _INTEGRATIONS_DIR.iterdir()
        if d.is_dir() and d.name != "__pycache__" and (d / "cassettes").is_dir()
    )


def cassette_files(integrations: list[str]) -> set[str]:
    """Cassette files on disk, as POSIX paths relative to the braintrust package."""
    files = set()
    for name in integrations:
        for path in (_INTEGRATIONS_DIR / name / "cassettes").rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts:
                files.add(path.relative_to(_PACKAGE_DIR).as_posix())
    return files


def load_usage(usage_dir: pathlib.Path) -> set[str]:
    used = set()
    for log in usage_dir.glob("usage-*.txt"):
        used.update(line.strip() for line in log.read_text(encoding="utf-8").splitlines() if line.strip())
    return used


def find_unused(integrations: list[str], used: set[str]) -> list[str]:
    return sorted(cassette_files(integrations) - used)


def _render_test_path(node: ast.expr) -> str | None:
    """Render the test-path argument of ``_run_tests`` (a literal or f-string)."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        parts = []
        for value in node.values:
            if isinstance(value, ast.Constant):
                parts.append(str(value.value))
            elif isinstance(value, ast.FormattedValue) and isinstance(value.value, ast.Name):
                if value.value.id != "INTEGRATION_DIR":
                    return None
                parts.append("braintrust/integrations")
            else:
                return None
        return "".join(parts)
    return None


def sessions_by_integration(noxfile: pathlib.Path = _NOXFILE) -> dict[str, set[str]]:
    """Map integration directory -> nox session function names that read its cassettes.

    A session reads ``integrations/<name>/cassettes`` when it runs tests under
    ``integrations/<name>/``, or when it passes ``<name>`` as an env value to
    ``_run_tests`` (the btx sessions select their provider's cassettes that way).
    """
    known = set(integrations_with_cassettes())
    mapping: dict[str, set[str]] = defaultdict(set)
    tree = ast.parse(noxfile.read_text())
    for func in tree.body:
        if not isinstance(func, ast.FunctionDef):
            continue
        for call in ast.walk(func):
            if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == "_run_tests"):
                continue
            if len(call.args) >= 2:
                path = _render_test_path(call.args[1])
                prefix = "braintrust/integrations/"
                if path and path.startswith(prefix):
                    name = path[len(prefix) :].split("/")[0]
                    if name in known:
                        mapping[name].add(func.name)
            for keyword in call.keywords:
                if keyword.arg == "env" and isinstance(keyword.value, ast.Dict):
                    for value in keyword.value.values:
                        if isinstance(value, ast.Constant) and value.value in known:
                            mapping[value.value].add(func.name)
    return dict(mapping)


def _nox_command() -> list[str]:
    nox = shutil.which("nox")
    return [nox] if nox else [sys.executable, "-m", "nox"]


def run_sessions(integrations: list[str], usage_dir: pathlib.Path) -> tuple[list[str], dict[str, list[str]]]:
    """Run every session that reads these integrations' cassettes.

    Returns (integrations that ran completely, {integration: [problem sessions]}).
    """
    mapping = sessions_by_integration()
    listing = json.loads(
        subprocess.run(
            [*_nox_command(), "-l", "--json"], cwd=_PROJECT_DIR, check=True, capture_output=True, text=True
        ).stdout
    )
    concrete: dict[str, list[str]] = defaultdict(list)
    for entry in listing:
        concrete[entry["name"]].append(entry["session"])

    to_run: dict[str, list[str]] = {}
    incomplete: dict[str, list[str]] = {}
    for name in integrations:
        funcs = sorted(mapping.get(name, ()))
        if not funcs:
            incomplete[name] = ["no nox session found that runs these tests"]
            continue
        to_run[name] = [s for func in funcs for s in concrete.get(func, [])]

    sessions = sorted({s for group in to_run.values() for s in group})
    env = {**os.environ, USAGE_DIR_ENV: str(usage_dir), "CI": "1"}
    results: dict[str, str] = {}
    for session in sessions:
        print(f"==> nox -s {session}", flush=True)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as report:
            report_path = report.name
        try:
            proc = subprocess.run(
                [*_nox_command(), "-s", session, "--report", report_path],
                cwd=_PROJECT_DIR,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            try:
                outcome = json.loads(pathlib.Path(report_path).read_text())["sessions"][0]["result"]
            except (OSError, ValueError, KeyError, IndexError):
                outcome = "success" if proc.returncode == 0 else "failed"
        finally:
            pathlib.Path(report_path).unlink(missing_ok=True)
        results[session] = outcome
        print(f"    {outcome}", flush=True)
        if outcome == "failed":
            print("\n".join(f"    | {line}" for line in proc.stdout.splitlines()[-15:]), flush=True)

    complete = []
    for name, group in to_run.items():
        problems = [f"{s} ({results[s]})" for s in group if results.get(s) != "success"]
        if problems:
            incomplete[name] = problems
        else:
            complete.append(name)
    return complete, incomplete


def print_report(unused: list[str], clean: bool) -> None:
    if not unused:
        print("No unused cassette files found.")
        return
    by_dir: dict[str, list[str]] = defaultdict(list)
    for path in unused:
        parent, _, filename = path.rpartition("/")
        by_dir[parent].append(filename)
    action = "Deleted" if clean else "Found"
    print(f"{action} {len(unused)} unused cassette file{'' if len(unused) == 1 else 's'}:")
    for parent in sorted(by_dir):
        print(f"  src/braintrust/{parent}/")
        for filename in by_dir[parent]:
            print(f"    {filename}")
    if clean:
        for path in unused:
            (_PACKAGE_DIR / path).unlink()
    else:
        print("\nRun with --clean to delete them.")


def _resolve_integrations(names: list[str], all_: bool) -> list[str]:
    available = integrations_with_cassettes()
    if all_ or not names:
        return available
    unknown = sorted(set(names) - set(available))
    if unknown:
        sys.exit(f"No cassette directory for: {', '.join(unknown)}. Choose from: {', '.join(available)}")
    return sorted(set(names))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run", help="Run the relevant nox sessions, then report")
    run_parser.add_argument("integrations", nargs="*", help="Integration directory names (e.g. anthropic)")
    run_parser.add_argument("--all", action="store_true", help="Check every integration with cassettes")
    run_parser.add_argument("--usage-dir", type=pathlib.Path, help="Keep usage logs here instead of a temp dir")
    run_parser.add_argument("--clean", action="store_true", help="Delete unused cassette files")

    report_parser = sub.add_parser("report", help="Report from existing usage logs")
    report_parser.add_argument("integrations", nargs="*", help="Limit the report to these integrations")
    report_parser.add_argument("--usage-dir", type=pathlib.Path, required=True)
    report_parser.add_argument("--clean", action="store_true", help="Delete unused cassette files")

    args = parser.parse_args()

    if args.command == "run":
        if not args.integrations and not args.all:
            parser.error("pass integration names or --all")
        integrations = _resolve_integrations(args.integrations, args.all)
        usage_dir = args.usage_dir or pathlib.Path(tempfile.mkdtemp(prefix="cassette-usage-"))
        usage_dir.mkdir(parents=True, exist_ok=True)
        complete, incomplete = run_sessions(integrations, usage_dir)
        print()
        if incomplete:
            print("Skipped these integrations because not every session that reads their cassettes succeeded:")
            for name, problems in sorted(incomplete.items()):
                print(f"  {name}: {', '.join(problems)}")
            print()
        unused = find_unused(complete, load_usage(usage_dir))
        print_report(unused, args.clean)
        sys.exit(1 if unused or incomplete else 0)

    integrations = _resolve_integrations(args.integrations, False)
    used = load_usage(args.usage_dir)
    if not used:
        sys.exit(f"No usage recorded in {args.usage_dir}. Run tests with {USAGE_DIR_ENV}={args.usage_dir} first.")
    unused = find_unused(integrations, used)
    print_report(unused, args.clean)
    sys.exit(1 if unused else 0)


if __name__ == "__main__":
    main()
