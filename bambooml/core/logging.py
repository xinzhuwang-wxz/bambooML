"""Logging configuration utilities."""
import logging
import logging.config
import logging.handlers
import sys
from pathlib import Path
from typing import Optional

from ..core.config import LOGS_DIR

# Logging configuration dictionary (similar to Made-With-ML)
logging_config = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "minimal": {"format": "%(message)s"},
        "detailed": {
            "format": "%(levelname)s %(asctime)s [%(name)s:%(filename)s:%(funcName)s:%(lineno)d]\n%(message)s\n"
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "stream": sys.stdout,
            "formatter": "minimal",
            "level": logging.DEBUG,
        },
        "info": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": str(Path(LOGS_DIR, "info.log")),
            "maxBytes": 10485760,  # 10 MB
            "backupCount": 10,
            "formatter": "detailed",
            "level": logging.INFO,
        },
        "error": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": str(Path(LOGS_DIR, "error.log")),
            "maxBytes": 10485760,  # 10 MB
            "backupCount": 10,
            "formatter": "detailed",
            "level": logging.ERROR,
        },
    },
    "root": {
        "handlers": ["console", "info", "error"],
        "level": logging.INFO,
        "propagate": True,
    },
}


def get_logger(
    name: str,
    stdout: bool = True,
    filename: Optional[str] = None,
    level: int = logging.INFO,
    use_config: bool = True,
) -> logging.Logger:
    """Create or get a logger with specified handlers.

    Args:
        name: Logger name.
        stdout: Whether to output logs to stdout. Defaults to True.
        filename: Log file path. Defaults to None.
        level: Logging level. Defaults to logging.INFO.
        use_config: Whether to use the global logging config. Defaults to True.

    Returns:
        Configured logger instance.
    """
    # Configure root logger if not already configured
    if use_config and not logging.root.handlers:
        logging.config.dictConfig(logging_config)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # If filename is provided, create a custom logger with file handlers
    if filename:
        logger.handlers = []  # Avoid duplicate handlers

        # Formatters
        minimal_formatter = logging.Formatter("%(message)s")
        detailed_formatter = logging.Formatter(
            "%(levelname)s %(asctime)s [%(name)s:%(filename)s:%(funcName)s:%(lineno)d]\n%(message)s\n"
        )

        # Console handler
        if stdout:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(minimal_formatter)
            console_handler.setLevel(logging.DEBUG)
            logger.addHandler(console_handler)

        # File handlers
        file_path = Path(filename)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Info file handler with rotation
        info_handler = logging.handlers.RotatingFileHandler(
            Path(LOGS_DIR, "info.log"),
            maxBytes=10485760,  # 10 MB
            backupCount=10,
        )
        info_handler.setFormatter(detailed_formatter)
        info_handler.setLevel(logging.INFO)
        logger.addHandler(info_handler)

        # Error file handler with rotation
        error_handler = logging.handlers.RotatingFileHandler(
            Path(LOGS_DIR, "error.log"),
            maxBytes=10485760,  # 10 MB
            backupCount=10,
        )
        error_handler.setFormatter(detailed_formatter)
        error_handler.setLevel(logging.ERROR)
        logger.addHandler(error_handler)

    return logger
