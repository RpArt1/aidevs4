from __future__ import annotations

import logging
import sys
import threading
from copy import deepcopy
from pathlib import Path
from typing import Any

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_FILE = LOG_DIR / "app.log"

_global_error_hooks_installed = False


def _install_global_error_hooks() -> None:
    """Log uncaught exceptions and warnings through the logging tree (once per process).

    Does not run for exceptions caught inside ASGI/FastAPI: Uvicorn logs those on the
    ``uvicorn`` loggers (``propagate: False``), which never reach the root file handler.
    Use :func:`build_uvicorn_log_config` with ``uvicorn.run(..., log_config=...)`` for that.
    """
    global _global_error_hooks_installed
    if _global_error_hooks_installed:
        return
    _global_error_hooks_installed = True

    logging.captureWarnings(True)

    uncaught = logging.getLogger("uncaught")

    def excepthook(
        exc_type: type[BaseException],
        exc_value: BaseException,
        exc_traceback: object,
    ) -> None:
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        uncaught.error(
            "Uncaught exception (main thread)",
            exc_info=(exc_type, exc_value, exc_traceback),
        )
        sys.__excepthook__(exc_type, exc_value, exc_traceback)

    sys.excepthook = excepthook

    if hasattr(threading, "excepthook"):
        prev_thread_hook = threading.excepthook
        tlog = logging.getLogger("uncaught.thread")

        def thread_excepthook(args: threading.ExceptHookArgs) -> None:
            tlog.error(
                "Uncaught exception in thread %r",
                args.thread.name,
                exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
            )
            prev_thread_hook(args)

        threading.excepthook = thread_excepthook


def setup_logging(level: int = logging.INFO) -> None:

    root = logging.getLogger()
    root.setLevel(level)

    for name in root.handlers[:]:
        root.removeHandler(name)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    root.addHandler(console)

    LOG_DIR.mkdir(exist_ok=True)
    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

    _install_global_error_hooks()


def build_uvicorn_log_config(*, use_colors: bool | None = None) -> dict[str, Any]:
    """Return Uvicorn's logging config plus the same ``app.log`` file handler as :func:`setup_logging`.

    Without this, request/traceback lines from Uvicorn stay on stderr/stdout only because
    the ``uvicorn`` / ``uvicorn.access`` loggers do not propagate to the root logger.
    """
    from uvicorn.config import LOGGING_CONFIG

    cfg = deepcopy(LOGGING_CONFIG)
    cfg["formatters"]["project"] = {
        "format": "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        "datefmt": "%H:%M:%S",
    }
    cfg["handlers"]["project_file"] = {
        "formatter": "project",
        "class": "logging.FileHandler",
        "filename": str(LOG_FILE),
        "encoding": "utf-8",
    }
    cfg["loggers"]["uvicorn"]["handlers"] = ["default", "project_file"]
    cfg["loggers"]["uvicorn.access"]["handlers"] = ["access", "project_file"]
    if use_colors is not None:
        cfg["formatters"]["default"]["use_colors"] = use_colors
        cfg["formatters"]["access"]["use_colors"] = use_colors
    return cfg


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
