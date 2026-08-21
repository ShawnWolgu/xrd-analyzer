"""Allow ``python -m xrd_analyzer`` to launch the desktop application."""

from .application import main


if __name__ == "__main__":
    raise SystemExit(main())
