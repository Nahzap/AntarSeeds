"""
Banco de .seg: conteo y limpieza.

El banco es el conjunto de anotaciones que alimenta la fase de entrenamiento,
así que borrar tiene que ser exacto: nunca debe tocar clases no pedidas ni
dejar backups huérfanos que finjan datos.
"""

import os

import pytest

from src.grain_detection.annotator import count_annotations, delete_annotations


@pytest.fixture
def bank(tmp_path):
    """Banco con 3 clases: 2, 3 y 1 .seg (esta última con backup)."""
    root = tmp_path / "annotations"
    for cls, n in (("C_Quitensis", 2), ("D_Antartica", 3), ("P_Annua", 1)):
        d = root / cls
        d.mkdir(parents=True)
        for i in range(n):
            (d / f"img_{i:03d}.seg").write_text("# seg\n", encoding="utf-8")
    (root / "P_Annua" / "img_000.seg.bak").write_text("# viejo\n", encoding="utf-8")
    return root


# ---------------------------------------------------------------------------
# Conteo
# ---------------------------------------------------------------------------

def test_count_reports_every_class(bank):
    assert count_annotations(str(bank)) == {
        "C_Quitensis": 2, "D_Antartica": 3, "P_Annua": 1
    }


def test_count_honours_class_filter(bank):
    assert count_annotations(str(bank), ["D_Antartica"]) == {"D_Antartica": 3}


def test_count_ignores_backups(bank):
    """Un .seg.bak no es dato: no puede inflar el banco."""
    assert count_annotations(str(bank))["P_Annua"] == 1


def test_count_missing_root_is_empty(tmp_path):
    assert count_annotations(str(tmp_path / "no_existe")) == {}


# ---------------------------------------------------------------------------
# Limpieza por clases marcadas
# ---------------------------------------------------------------------------

def test_delete_selected_leaves_other_classes_intact(bank):
    result = delete_annotations(str(bank), ["C_Quitensis"])

    assert result["seg_removed"] == 2
    assert result["per_class"] == {"C_Quitensis": 2}
    assert result["errors"] == 0
    assert count_annotations(str(bank)) == {"D_Antartica": 3, "P_Annua": 1}


def test_delete_selected_removes_empty_class_dir(bank):
    delete_annotations(str(bank), ["C_Quitensis"])
    assert not (bank / "C_Quitensis").exists()
    assert (bank / "D_Antartica").exists()


def test_delete_selected_takes_backups_with_it(bank):
    result = delete_annotations(str(bank), ["P_Annua"])
    assert result["bak_removed"] == 1
    assert list((bank).rglob("*.seg.bak")) == []


def test_delete_can_keep_backups(bank):
    result = delete_annotations(str(bank), ["P_Annua"], remove_backups=False)
    assert result["seg_removed"] == 1
    assert result["bak_removed"] == 0
    assert len(list(bank.rglob("*.seg.bak"))) == 1


def test_unknown_class_in_filter_deletes_nothing(bank):
    antes = count_annotations(str(bank))
    result = delete_annotations(str(bank), ["No_Existe"])
    assert result["seg_removed"] == 0
    assert count_annotations(str(bank)) == antes


# ---------------------------------------------------------------------------
# Limpieza total
# ---------------------------------------------------------------------------

def test_delete_all_empties_the_bank(bank):
    result = delete_annotations(str(bank), None)

    assert result["seg_removed"] == 6
    assert result["bak_removed"] == 1
    assert count_annotations(str(bank)) == {}
    assert list(bank.rglob("*.seg")) == []


def test_delete_all_on_empty_bank_is_safe(tmp_path):
    result = delete_annotations(str(tmp_path / "vacio"), None)
    assert result == {"seg_removed": 0, "bak_removed": 0, "per_class": {}, "errors": 0}


def test_delete_is_idempotent(bank):
    delete_annotations(str(bank), None)
    result = delete_annotations(str(bank), None)
    assert result["seg_removed"] == 0
    assert result["errors"] == 0


# ---------------------------------------------------------------------------
# Los dos botones son una sola implementación con distinto alcance
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def contour_tab():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PyQt5.QtWidgets")
    from lib.contour_analysis_tab import ContourAnalysisTab

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    tab = ContourAnalysisTab()
    yield tab
    tab.deleteLater()
    app.processEvents()


def test_both_bank_buttons_exist(contour_tab):
    assert contour_tab.btn_clear_selected_seg.text() == "Limpiar clases marcadas"
    assert contour_tab.btn_clear_all_seg.text() == "Limpiar TODO"


def test_bank_buttons_share_one_implementation(contour_tab):
    """Distinto alcance, misma función: nada de dos rutas de borrado."""
    llamadas = []
    contour_tab._clear_annotations = lambda cf: llamadas.append(cf)

    contour_tab._on_clear_all_annotations()
    assert llamadas == [None]

    del contour_tab._clear_annotations


def test_bank_buttons_locked_while_segmenting(contour_tab):
    contour_tab._set_segmentation_busy(True)
    assert not contour_tab.btn_clear_all_seg.isEnabled()
    assert not contour_tab.btn_clear_selected_seg.isEnabled()

    contour_tab._set_segmentation_busy(False)
    assert contour_tab.btn_clear_all_seg.isEnabled()
    assert contour_tab.btn_clear_selected_seg.isEnabled()


def test_clear_selected_requires_a_tick(contour_tab, monkeypatch):
    """Sin tick no se borra nada, ni se pide confirmación."""
    from lib import contour_analysis_tab as tab_mod

    monkeypatch.setattr(contour_tab, "_get_checked_classes", lambda: [])
    llamadas = []
    monkeypatch.setattr(contour_tab, "_clear_annotations", lambda cf: llamadas.append(cf))
    monkeypatch.setattr(tab_mod.QMessageBox, "warning", lambda *a, **k: None)

    contour_tab._on_clear_selected_annotations()
    assert llamadas == []


def test_bank_info_reports_empty_state(contour_tab):
    contour_tab._update_bank_info()
    assert "data/annotations" in contour_tab.bank_info_label.text()
