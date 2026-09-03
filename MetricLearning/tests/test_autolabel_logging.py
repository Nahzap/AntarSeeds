"""
Trazabilidad del autoetiquetador en la terminal.

El fallo que cubren: `setup_logger` solo configuraba un logger nombrado con
`propagate=False`, así que los módulos del buscador ROI no escribían nada en la
consola y el lote parecía una caja negra.
"""

import logging

import numpy as np
import pytest

from src.grain_detection.seeded_contour import (
    SeededParams,
    passes_validity_filters,
    validity_reason,
)
from src.utils import logging_utils
from src.utils.logging_utils import (
    set_autolabel_verbosity,
    setup_root_logging,
    start_session_logs,
)

AUTOLABEL_LOGGERS = (
    "src.grain_detection.seeded_contour",
    "src.grain_detection.grain_detector",
    "src.grain_detection.annotator",
)


def _rect(x0, y0, x1, y1):
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.int32)


# ---------------------------------------------------------------------------
# Motivo de descarte: la terminal debe decir QUÉ filtro cortó el candidato
# ---------------------------------------------------------------------------

def test_valid_body_has_no_reason():
    params = SeededParams(min_area=1000, max_area=200000, min_circularity=0.1,
                          max_aspect_ratio=8.0, max_bbox_side_frac=0.9)
    assert validity_reason(_rect(50, 50, 200, 200), 600, 600, params) is None


@pytest.mark.parametrize(
    "contour, params_kwargs, expected",
    [
        (_rect(10, 10, 20, 20), {"min_area": 5000}, "min_area"),
        (_rect(10, 10, 500, 500), {"max_area": 1000}, "max_area"),
        (_rect(10, 10, 560, 120), {"max_bbox_side_frac": 0.5}, "max_bbox_side_frac"),
        (_rect(10, 10, 500, 60), {"max_aspect_ratio": 2.0}, "max_aspect_ratio"),
        (_rect(10, 10, 200, 200), {"min_circularity": 0.99}, "min_circularity"),
    ],
)
def test_reason_names_the_filter_that_rejected(contour, params_kwargs, expected):
    base = dict(min_area=100, max_area=500000, max_area_frac=1.0,
                min_circularity=0.05, max_aspect_ratio=20.0, max_bbox_side_frac=1.0)
    base.update(params_kwargs)
    params = SeededParams(**base)

    reason = validity_reason(contour, 600, 600, params)
    assert reason is not None
    assert expected in reason
    assert not passes_validity_filters(contour, 600, 600, params)


def test_empty_contour_reports_reason():
    assert "vac" in validity_reason(None, 100, 100, SeededParams())


# ---------------------------------------------------------------------------
# El logging llega realmente a la consola
# ---------------------------------------------------------------------------

def test_start_session_logs_clears_previous_and_names_with_date(tmp_path):
    old = tmp_path / "viejo.log"
    old.write_text("basura", encoding="utf-8")
    path = start_session_logs(str(tmp_path))
    try:
        assert path.name.startswith("antarseeds_")
        assert path.suffix == ".log"
        assert not old.exists()
        assert path.exists()
        assert "Sesión:" in path.read_text(encoding="utf-8")
    finally:
        logging_utils.SESSION_LOG_PATH = None


def test_root_logging_attaches_console_and_file(tmp_path):
    root = setup_root_logging(log_dir=str(tmp_path))
    streams = [h for h in root.handlers if isinstance(h, logging.StreamHandler)]
    assert streams, "sin handler de consola el lote no escribe nada"
    assert list(tmp_path.glob("antarseeds_*.log"))


def test_root_logging_is_idempotent(tmp_path):
    setup_root_logging(log_dir=str(tmp_path))
    n_before = len(logging.getLogger().handlers)
    setup_root_logging(log_dir=str(tmp_path))
    assert len(logging.getLogger().handlers) == n_before


def test_autolabel_modules_reach_root(tmp_path, caplog):
    setup_root_logging(log_dir=str(tmp_path))
    set_autolabel_verbosity(False)
    with caplog.at_level(logging.INFO, logger="src.grain_detection.seeded_contour"):
        logging.getLogger("src.grain_detection.seeded_contour").info("paso de prueba")
    assert "paso de prueba" in caplog.text


def test_verbosity_switch_moves_between_info_and_debug(tmp_path):
    setup_root_logging(log_dir=str(tmp_path))

    set_autolabel_verbosity(True)
    for name in AUTOLABEL_LOGGERS:
        assert logging.getLogger(name).level == logging.DEBUG

    set_autolabel_verbosity(False)
    for name in AUTOLABEL_LOGGERS:
        assert logging.getLogger(name).level == logging.INFO
