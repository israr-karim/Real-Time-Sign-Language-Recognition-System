"""Check notebook hygiene and extracted definitions without loading ML models."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import black
import nbformat

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BLACK_MODE = black.Mode(line_length=88, target_versions={black.TargetVersion.PY311})


def validate_notebooks() -> tuple[int, int]:
    """Validate schemas, syntax, formatting, and cleared runtime state."""
    notebook_count = 0
    code_cell_count = 0
    for path in sorted((REPOSITORY_ROOT / "notebooks").glob("*.ipynb")):
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        notebook_count += 1
        for index, cell in enumerate(notebook.cells, 1):
            if cell.cell_type != "code":
                continue
            location = f"{path.name}, cell {index}"
            if cell.outputs or cell.execution_count is not None:
                raise ValueError(f"Stored runtime state must be cleared: {location}")
            compile(cell.source, location, "exec")
            if (
                black.format_str(cell.source, mode=BLACK_MODE).rstrip()
                != cell.source.rstrip()
            ):
                raise ValueError(f"Code formatting differs from Black: {location}")
            code_cell_count += 1
    if notebook_count != 4:
        raise ValueError(
            "Expected the overview and three recreated experiment notebooks."
        )
    return notebook_count, code_cell_count


def validate_definition_provenance() -> int:
    """Match extracted classes to the recorded original syntax-tree hashes."""
    manifest = json.loads(
        (REPOSITORY_ROOT / "docs" / "source_manifest.json").read_text()
    )
    parsed_modules = {}
    for record in manifest["extracted_definitions"]:
        module_path = record["module"]
        if module_path not in parsed_modules:
            parsed_modules[module_path] = ast.parse(
                (REPOSITORY_ROOT / module_path).read_text()
            )
        definitions = [
            node
            for node in parsed_modules[module_path].body
            if isinstance(node, ast.ClassDef) and node.name == record["class"]
        ]
        if len(definitions) != 1:
            raise ValueError(f"Ambiguous definition: {module_path}: {record['class']}")
        digest = hashlib.sha256(
            ast.dump(definitions[0], include_attributes=False).encode()
        ).hexdigest()
        if digest != record["ast_sha256"]:
            raise ValueError(
                f"Extracted class changed; review and update provenance: "
                f"{module_path}: {record['class']}"
            )
    for record in manifest["notebooks"]:
        if not (REPOSITORY_ROOT / record["recreated_file"]).is_file():
            raise ValueError(f"Missing recreated notebook: {record['recreated_file']}")
    return len(manifest["extracted_definitions"])


def main() -> None:
    notebooks, cells = validate_notebooks()
    definitions = validate_definition_provenance()
    print(f"Validated {notebooks} notebooks and {cells} code cells.")
    print(f"Verified {definitions} extracted-class provenance records.")
    print("No research model was imported, trained, or evaluated.")


if __name__ == "__main__":
    main()
