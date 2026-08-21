"""Repository entry point for XRD Analyzer."""

from xrd_analyzer.application import (
    find_missing_packages,
    main as _application_main,
)
from xrd_analyzer.metadata import startup_banner_text


def main() -> int:
    """Start the installed XRD Analyzer application."""
    return _application_main()


if __name__ == "__main__":
    raise SystemExit(main())
