"""Logging handler que emite mensajes a la GUI (INFO/WARNING/ERROR/DEBUG)."""

from __future__ import annotations

import logging
from typing import Callable, Optional


class GuiLogHandler(logging.Handler):
    """Redirige registros de logging a un callback ``(message, level)``."""

    def __init__(self, callback: Callable[[str, str], None], level: int = logging.DEBUG):
        super().__init__(level)
        self.callback = callback
        self.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            self.callback(msg, record.levelname)
        except Exception:
            self.handleError(record)


def attach_gui_logging(
    callback: Callable[[str, str], None],
    logger_names: Optional[list] = None,
    level: int = logging.DEBUG,
) -> GuiLogHandler:
    """Attach handler to train_detector and related loggers. Returns handler for removal."""
    handler = GuiLogHandler(callback, level=level)
    names = logger_names or [
        "train_detector",
        "src.training.detector_trainer",
        "src.data.detector_data_engine",
        "src.data.detection_dataset",
    ]
    for name in names:
        log = logging.getLogger(name)
        log.addHandler(handler)
        if log.level > level:
            log.setLevel(level)
    return handler


def detach_gui_logging(handler: GuiLogHandler, logger_names: Optional[list] = None) -> None:
    names = logger_names or [
        "train_detector",
        "src.training.detector_trainer",
        "src.data.detector_data_engine",
        "src.data.detection_dataset",
    ]
    for name in names:
        logging.getLogger(name).removeHandler(handler)
