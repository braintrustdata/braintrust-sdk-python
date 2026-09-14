#!/usr/bin/env python3
"""Update the generated reference sections in the public REST API README."""

import argparse
import ast
import sys
from pathlib import Path
from typing import Iterable

from openapi_codegen import (
    CONFIG_PATH,
    SPEC_PATH,
    collect_generated_operations,
    generated_resource_name,
    load_config,
    read_and_verify_spec,
)


API_ROOT = Path(__file__).resolve().parents[1] / "src" / "braintrust" / "api"
README_PATH = API_ROOT / "README.md"
RESOURCE_START = "<!-- BEGIN GENERATED RESOURCE REFERENCE -->"
RESOURCE_END = "<!-- END GENERATED RESOURCE REFERENCE -->"
API_EXPORT_START = "<!-- BEGIN GENERATED API EXPORTS -->"
API_EXPORT_END = "<!-- END GENERATED API EXPORTS -->"
TYPE_START = "<!-- BEGIN GENERATED REST TYPES -->"
TYPE_END = "<!-- END GENERATED REST TYPES -->"


def _literal_all(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    assignment = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets)
    )
    value = ast.literal_eval(assignment.value)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{path}.__all__ must be a literal list of strings")
    return value


def _render_resource_reference() -> str:
    config = load_config(CONFIG_PATH)
    spec = read_and_verify_spec(config, SPEC_PATH)
    operations, _ = collect_generated_operations(spec, config)
    operations_by_tag = {tag: [] for tag in config["endpoint_generator"]["generated_tags"]}
    for operation in operations:
        operations_by_tag[operation.tag].append(operation)

    rows = ["| Client property | Methods |", "| --- | --- |"]
    for tag, tag_operations in operations_by_tag.items():
        methods = "<br>".join(
            f"`{operation.constant_name.lower()}` — `{operation.method} {operation.path}`"
            for operation in tag_operations
        )
        rows.append(f"| `client.{generated_resource_name(tag)}` | {methods} |")
    return "\n".join(rows)


def _render_name_table(names: Iterable[str], columns: int = 3) -> str:
    values = [f"`{name}`" for name in names]
    rows = ["| " + " | ".join(["Name"] * columns) + " |", "| " + " | ".join(["---"] * columns) + " |"]
    for index in range(0, len(values), columns):
        row = values[index : index + columns]
        row.extend([""] * (columns - len(row)))
        rows.append("| " + " | ".join(row) + " |")
    return "\n".join(rows)


def _replace_section(content: str, start: str, end: str, replacement: str) -> str:
    if content.count(start) != 1 or content.count(end) != 1:
        raise ValueError(f"{README_PATH} must contain exactly one {start!r} and {end!r} marker")
    prefix, remainder = content.split(start, 1)
    _, suffix = remainder.split(end, 1)
    return f"{prefix}{start}\n{replacement}\n{end}{suffix}"


def render_readme(content: str) -> str:
    content = _replace_section(content, RESOURCE_START, RESOURCE_END, _render_resource_reference())
    content = _replace_section(
        content,
        API_EXPORT_START,
        API_EXPORT_END,
        _render_name_table(_literal_all(API_ROOT / "__init__.py")),
    )
    return _replace_section(
        content,
        TYPE_START,
        TYPE_END,
        _render_name_table(_literal_all(API_ROOT / "types" / "__init__.py")),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Report README drift without changing the file.")
    args = parser.parse_args()

    current = README_PATH.read_text(encoding="utf-8")
    rendered = render_readme(current)
    if current == rendered:
        print(f"Public REST API documentation is current: {README_PATH}")
        return 0
    if args.check:
        print(
            "Public REST API documentation drift detected. Run `cd py && make generate-api-client`.",
            file=sys.stderr,
        )
        return 1
    README_PATH.write_text(rendered, encoding="utf-8")
    print(f"Updated public REST API documentation: {README_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
