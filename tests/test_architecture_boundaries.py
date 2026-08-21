"""Dependency contracts between the PyQt frontend and scientific backend."""

from __future__ import annotations

import ast
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).parents[1]
BACKEND_MODULES = (
    "xrd_backend.py",
    "xrd_analyzer.py",
    "xrd_crystallography.py",
    "xrd_io.py",
    "xrd_peaks.py",
    "xrd_preprocessing.py",
    "xrd_project.py",
    "xrd_session.py",
)


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    return imported


def test_frontend_uses_only_the_backend_public_entrypoint() -> None:
    imports = _imported_modules(REPOSITORY_ROOT / "xrd_gui.py")
    direct_xrd_dependencies = {
        module for module in imports if module.startswith("xrd_")
    }

    assert direct_xrd_dependencies == {"xrd_backend"}


def test_backend_dependency_closure_does_not_import_pyqt() -> None:
    violations = {}
    for module_name in BACKEND_MODULES:
        imports = _imported_modules(REPOSITORY_ROOT / module_name)
        pyqt_imports = sorted(
            module for module in imports if module == "PyQt5" or module.startswith("PyQt5.")
        )
        if pyqt_imports:
            violations[module_name] = pyqt_imports

    assert violations == {}


def test_application_has_one_main_entrypoint() -> None:
    main_path = REPOSITORY_ROOT / "main.py"
    main_tree = ast.parse(main_path.read_text(encoding="utf-8"), filename=str(main_path))
    main_functions = {
        node.name for node in main_tree.body if isinstance(node, ast.FunctionDef)
    }

    gui_path = REPOSITORY_ROOT / "xrd_gui.py"
    gui_tree = ast.parse(gui_path.read_text(encoding="utf-8"), filename=str(gui_path))
    gui_functions = {
        node.name for node in gui_tree.body if isinstance(node, ast.FunctionDef)
    }

    project_configuration = (REPOSITORY_ROOT / "pyproject.toml").read_text(
        encoding="utf-8"
    )

    assert "main" in main_functions
    assert "main" not in gui_functions
    assert 'xrd-analyzer = "main:main"' in project_configuration
