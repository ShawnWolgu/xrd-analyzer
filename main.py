"""XRD Analyzer application entry point."""

from __future__ import annotations

import importlib
import sys
from collections.abc import Sequence

from app_metadata import startup_banner_text


REQUIRED_PACKAGES = {
    "PyQt5": "PyQt5",
    "numpy": "numpy",
    "scipy": "scipy",
    "matplotlib": "matplotlib",
    "pandas": "pandas",
    "lmfit": "lmfit",
    "openpyxl": "openpyxl",
    "threadpoolctl": "threadpoolctl",
}


def find_missing_packages() -> list[str]:
    """Return install names for runtime dependencies that cannot be imported."""
    missing = []
    for module_name, package_name in REQUIRED_PACKAGES.items():
        try:
            importlib.import_module(module_name)
        except ImportError:
            missing.append(package_name)
    return missing


def _print_missing_packages(packages: Sequence[str]) -> None:
    print("=" * 60)
    print("缺少必要的Python包！")
    print("=" * 60)
    print("\n请运行以下命令安装：")
    print(f"\npip install {' '.join(packages)}")
    print("\n或者：")
    print(f"\nconda install {' '.join(packages)}")
    print("\n" + "=" * 60)


def main() -> int:
    """Create the desktop application and run its event loop."""
    print(startup_banner_text())
    missing_packages = find_missing_packages()
    if missing_packages:
        _print_missing_packages(missing_packages)
        return 1

    # Keep GUI imports inside the entry function so dependency diagnostics can run first.
    from PyQt5.QtGui import QFont
    from PyQt5.QtWidgets import QApplication

    from xrd_gui import XRDAnalyzerGUI

    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setFont(QFont("Microsoft YaHei", 10))

    window = XRDAnalyzerGUI()
    window.show()
    return int(app.exec_())


if __name__ == "__main__":
    raise SystemExit(main())
