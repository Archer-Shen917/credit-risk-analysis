"""Verify the saved deliverable without reading any customer-level source data."""

from __future__ import annotations

import ast
import csv
import json
from pathlib import Path


def verify_repository(root: Path) -> dict:
    """Audit source contracts, table accounting, and key artifact existence.

    Args:
        root: Project root containing src, outputs, reports, and models.

    Returns:
        Summary of explicit checks; exceptions identify invalid deliverables.
    """
    sources = list((root / "src").rglob("*.py"))
    functions = 0
    for path in sources:
        text = path.read_text(encoding="utf-8")
        assert "\t" not in text, f"Tab indentation: {path}"
        assert max(map(len, text.splitlines()), default=0) <= 88, path
        tree = ast.parse(text)
        assert ast.get_docstring(tree), f"Missing module docstring: {path}"
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                functions += 1
                documentation = ast.get_docstring(node) or ""
                assert "Args:" in documentation and "Returns:" in documentation
                assert node.returns is not None, node.name
                for argument in node.args.args:
                    if argument.arg not in {"self", "cls"}:
                        assert argument.annotation is not None, node.name
            if isinstance(node, ast.ExceptHandler):
                assert node.type is not None, f"Bare except in {path}"
    with (root / "outputs/phase3/strategy_tiers_test.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        records = list(csv.DictReader(handle))
    totals = {}
    for record in records:
        name = record["strategy"]
        totals[name] = totals.get(name, 0) + int(record["n"])
    assert all(total == 92254 for total in totals.values())
    assert (root / "reports/final/项目结项总结_5页.pdf").is_file()
    assert (root / "models/model_bundle.joblib").is_file()
    files = [path for path in root.rglob("*") if path.is_file()]
    largest = max(path.stat().st_size for path in files)
    assert largest < 25 * 1024 * 1024
    return {
        "source_modules": len(sources),
        "documented_typed_functions": functions,
        "source_line_width_maximum": 88,
        "strategy_counts_match_test": totals,
        "largest_file_bytes": largest,
        "third_party_lint_status": "not executed in this environment",
    }


def main() -> None:
    """Run delivery checks from the repository root.

    Args:
        None; the repository path is inferred from this script.

    Returns:
        None; an auditable JSON result is written to outputs/qa/.
    """
    root = Path(__file__).resolve().parents[1]
    result = verify_repository(root)
    output = root / "outputs/qa/delivery_checks.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
