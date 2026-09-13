"""
Project-IQ: Production Structured Logging System
Provides unified, thread-safe console and rotating file logging.
"""

from __future__ import annotations
import logging
from logging.handlers import RotatingFileHandler
import os
import sys
from pathlib import Path
from typing import Optional

from src.config import config

_LOGGERS = {}


def setup_logger(
    name: str = "project_iq",
    log_file: Optional[Path] = None,
    level: Optional[str] = None
) -> logging.Logger:
    """
    Configure and return a structured logger with console and file handlers.
    Thread-safe and avoids duplicate handlers if called repeatedly.
    """
    if name in _LOGGERS:
        return _LOGGERS[name]

    logger = logging.getLogger(name)
    log_level_str = level or config.log_level
    numeric_level = getattr(logging, log_level_str.upper(), logging.INFO)
    logger.setLevel(numeric_level)
    logger.propagate = False

    # Prevent duplicate handlers
    if logger.handlers:
        return logger

    # Standard formatter with ISO timestamp and clear module tagging
    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)-7s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Ensure Windows console does not crash on unicode/emojis
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    # Console Handler (UTF-8 safe)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Optional Rotating File Handler
    target_file = log_file or (config.logs_dir / "project_iq.log")
    try:
        target_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            target_file,
            maxBytes=10 * 1024 * 1024,  # 10 MB per file
            backupCount=5,
            encoding="utf-8"
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        logger.warning(f"Failed to initialize file log handler at {target_file}: {e}")

    _LOGGERS[name] = logger
    return logger


def get_logger(name: str = "project_iq") -> logging.Logger:
    """Convenience accessor to get a logger instance."""
    if name in _LOGGERS:
        return _LOGGERS[name]
    return setup_logger(name)
