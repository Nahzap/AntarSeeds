"""
Structured alarm logging for training runtime diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class AlarmEvent:
    timestamp: str
    level: str
    code: str
    message: str
    context: Dict[str, Any]


class TrainingAlarmManager:
    """
    Emits human-readable alarm lines to logger and a structured JSONL stream.
    """

    def __init__(self, run_dir: Optional[str], logger):
        self.logger = logger
        self.alarms_path: Optional[Path] = None
        if run_dir:
            alarms_dir = Path(run_dir) / "logs"
            alarms_dir.mkdir(parents=True, exist_ok=True)
            self.alarms_path = alarms_dir / "training_alarms.jsonl"

    def _emit(self, level: str, code: str, message: str, **context: Any):
        stamp = datetime.now().isoformat(timespec="seconds")
        prefix = f"[ALARM][{code}] {message}"
        if level == "ERROR":
            self.logger.error(prefix)
        elif level == "WARNING":
            self.logger.warning(prefix)
        else:
            self.logger.info(prefix)

        if self.alarms_path is None:
            return

        event = AlarmEvent(
            timestamp=stamp,
            level=level,
            code=code,
            message=message,
            context=context,
        )
        with open(self.alarms_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event.__dict__, ensure_ascii=False) + "\n")

    def info(self, code: str, message: str, **context: Any):
        self._emit("INFO", code, message, **context)

    def warning(self, code: str, message: str, **context: Any):
        self._emit("WARNING", code, message, **context)

    def error(self, code: str, message: str, **context: Any):
        self._emit("ERROR", code, message, **context)
