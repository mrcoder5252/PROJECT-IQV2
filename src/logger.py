"""
Project-IQ: Structured logging.

One console handler and ONE rotating file handler are attached to the root
``project_iq`` logger; every ``project_iq.<module>`` logger propagates to it.
(Previously each module opened its own handle on the same log file, which breaks
log rotation on Windows.)
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Dict, Optional

from src.config import config

ROOT_LOGGER_NAME = "project_iq"
_LOGGERS: Dict[str, logging.Logger] = {}
_FORMAT = "[%(asctime)s] [%(levelname)-7s] [%(name)s]: %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def _attach_handlers(logger: logging.Logger, log_file: Optional[Path], level: int) -> None:
    formatter = logging.Formatter(fmt=_FORMAT, datefmt=_DATEFMT)

    if hasattr(sys.stdout, "reconfigure"):  # keep Windows consoles from crashing on unicode
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(formatter)
    logger.addHandler(console)

    target = log_file or (config.logs_dir / "project_iq.log")
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(target, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except OSError as exc:
        logger.warning("File logging disabled (%s): %s", target, exc)


def setup_logger(name: str = ROOT_LOGGER_NAME, log_file: Optional[Path] = None, level: Optional[str] = None) -> logging.Logger:
    """Return a configured logger; safe to call repeatedly."""
    if name in _LOGGERS:
        return _LOGGERS[name]

    numeric = getattr(logging, (level or config.log_level).upper(), logging.INFO)
    is_child = name.startswith(ROOT_LOGGER_NAME + ".")

    if is_child:
        setup_logger(ROOT_LOGGER_NAME, log_file, level)  # make sure the root is configured once
        logger = logging.getLogger(name)
        logger.setLevel(numeric)
    else:
        logger = logging.getLogger(name)
        logger.setLevel(numeric)
        logger.propagate = False
        if not logger.handlers:
            _attach_handlers(logger, log_file, numeric)

    _LOGGERS[name] = logger
    return logger


def get_logger(name: str = ROOT_LOGGER_NAME) -> logging.Logger:
    """Convenience accessor."""
    return _LOGGERS.get(name) or setup_logger(name)
