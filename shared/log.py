"""
Logging configuration for the Citation Agent pipeline.

Usage:
    from shared.log import get_logger
    logger = get_logger("agent3")
    logger.info("Processing %s...", pdf_path)
"""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler

from config import PROJECT_ROOT

# Overridable so a test run, or a second checkout, does not append to the
# project's own log. Set CITATION_LOG_DIR to redirect, or CITATION_LOG_FILE=0
# to turn file logging off entirely and keep console output only.
LOG_DIR  = os.environ.get("CITATION_LOG_DIR", os.path.join(PROJECT_ROOT, "logs"))
LOG_FILE = os.path.join(LOG_DIR, "citation_agent.log")
LOG_TO_FILE = os.environ.get("CITATION_LOG_FILE", "1").lower() not in ("0", "false", "no")


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a named logger that writes to both console and a rotating log file."""
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if called multiple times
    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(name)s] %(levelname)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # Rotating file handler (5 MB, keep 3 backups).
    #
    # delay=True defers opening the file until something is actually logged, so
    # merely importing a module no longer touches the filesystem — importing
    # config or shared.search used to create logs/ and open a file handle as a
    # side effect.
    if LOG_TO_FILE:
        os.makedirs(LOG_DIR, exist_ok=True)
        file_handler = RotatingFileHandler(
            LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3, delay=True
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
