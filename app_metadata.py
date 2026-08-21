"""User-facing product identity shared by all application entry points."""

APP_NAME = "XRD Analyzer"
APP_VERSION = "1.0.0"
APP_TAGLINE_ZH = "通用 X 射线衍射分析工具"
WINDOW_TITLE = f"{APP_NAME} v{APP_VERSION} - {APP_TAGLINE_ZH}"


def startup_banner_text() -> str:
    """Return the terminal banner shown for every supported launch command."""
    return "\n".join(
        (
            "=" * 60,
            f"{APP_NAME} v{APP_VERSION}",
            APP_TAGLINE_ZH,
            "支持衍射数据处理、峰形拟合、项目恢复与结果导出",
            "正在启动图形界面...",
            "=" * 60,
        )
    )
