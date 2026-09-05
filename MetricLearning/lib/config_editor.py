"""
Editor de Configuración Visual para config.yaml
Implementa mejores prácticas IEEE 9241-143 (Formularios)

Características:
- Validación en tiempo real
- Feedback visual inmediato
- Agrupación lógica de campos
- Tooltips informativos
- Guardado automático con backup
"""

import yaml
import shutil
from pathlib import Path
from datetime import datetime
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QLineEdit, QSpinBox, QDoubleSpinBox, QCheckBox, QComboBox,
    QPushButton, QScrollArea, QLabel, QMessageBox, QFileDialog,
    QTabWidget, QTextEdit, QDialog, QDialogButtonBox
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont, QColor, QPalette

from lib.styles import Styles

# Import para auto-configuración de backbones
from src.data.transforms_enhanced import get_input_size_for_backbone


def set_nested_value(data: dict, dotted_key: str, value):
    """Escribe un valor en config_data usando clave con puntos (ej. data.loader_policy.x)."""
    keys = dotted_key.split('.')
    cur = data
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value


class ConfigChangesDialog(QDialog):
    """
    Diálogo para mostrar cambios automáticos aplicados al cambiar backbone.
    Permite al usuario ver y copiar los cambios realizados.
    """
    def __init__(self, changes, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Información - Cambios Aplicados")
        self.setModal(False)  # No modal para poder seguir trabajando
        self.setMinimumWidth(500)
        self.setMinimumHeight(300)
        
        layout = QVBoxLayout()
        
        # Header
        header = QLabel("✅ Configuración Actualizada Automáticamente")
        header.setStyleSheet("""
            QLabel {
                font-size: 12pt;
                font-weight: bold;
                color: #0e639c;
                padding: 10px;
                background-color: #e8f4f8;
                border-radius: 5px;
            }
        """)
        layout.addWidget(header)
        
        # Descripción
        desc = QLabel("Los siguientes parámetros se actualizaron según las características del backbone seleccionado:")
        desc.setWordWrap(True)
        desc.setStyleSheet("padding: 10px; color: #666;")
        layout.addWidget(desc)
        
        # Lista de cambios
        self.changes_text = QTextEdit()
        self.changes_text.setReadOnly(True)
        self.changes_text.setStyleSheet("""
            QTextEdit {
                background-color: #f9f9f9;
                border: 1px solid #cccccc;
                border-radius: 3px;
                padding: 8px;
                font-family: 'Courier New', monospace;
                font-size: 10pt;
            }
        """)
        
        # Formatear cambios
        changes_html = "<table cellpadding='5' style='width: 100%;'>"
        changes_html += "<tr style='background-color: #e0e0e0; font-weight: bold;'>"
        changes_html += "<td>Parámetro</td><td>Valor Anterior</td><td>→</td><td>Nuevo Valor</td></tr>"
        
        for param, old_val, new_val in changes:
            changes_html += f"<tr>"
            changes_html += f"<td style='color: #0e639c;'><b>{param}</b></td>"
            changes_html += f"<td style='color: #888;'>{old_val}</td>"
            changes_html += f"<td>→</td>"
            changes_html += f"<td style='color: #4ec9b0; font-weight: bold;'>{new_val}</td>"
            changes_html += "</tr>"
        
        changes_html += "</table>"
        self.changes_text.setHtml(changes_html)
        layout.addWidget(self.changes_text)
        
        # Nota informativa
        note = QLabel("💡 Estos cambios se aplicaron automáticamente pero AÚN NO se han guardado en config.yaml")
        note.setWordWrap(True)
        note.setStyleSheet("""
            QLabel {
                background-color: #fff4e6;
                border: 1px solid #ffa94d;
                border-radius: 3px;
                padding: 8px;
                color: #d9480f;
            }
        """)
        layout.addWidget(note)
        
        # Botones
        button_box = QDialogButtonBox()
        copy_btn = button_box.addButton("📋 Copiar", QDialogButtonBox.ActionRole)
        copy_btn.clicked.connect(self.copy_to_clipboard)
        close_btn = button_box.addButton("Cerrar", QDialogButtonBox.AcceptRole)
        close_btn.clicked.connect(self.accept)
        
        layout.addWidget(button_box)
        self.setLayout(layout)
    
    def copy_to_clipboard(self):
        """Copia los cambios al portapapeles"""
        from PyQt5.QtWidgets import QApplication
        text = self.changes_text.toPlainText()
        QApplication.clipboard().setText(text)
        QMessageBox.information(self, "Copiado", "Cambios copiados al portapapeles")


class ConfigField(QWidget):
    """
    Widget base para campos de configuración con validación.
    Implementa IEEE 9241-143: Diseño de formularios.
    """
    valueChanged = pyqtSignal()
    
    def __init__(self, label, tooltip="", parent=None):
        super().__init__(parent)
        self.label_text = label
        self.tooltip_text = tooltip
        self.is_valid = True
        
    def get_value(self):
        """Obtiene valor del campo"""
        raise NotImplementedError
    
    def set_value(self, value):
        """Establece valor del campo"""
        raise NotImplementedError
    
    def validate(self):
        """Valida el campo"""
        return True
    
    def mark_invalid(self, message=""):
        """Marca campo como inválido"""
        self.is_valid = False
        self.setStyleSheet("border: 2px solid #f48771; border-radius: 3px;")
        if message:
            self.setToolTip(f"❌ {message}")
    
    def mark_valid(self):
        """Marca campo como válido"""
        self.is_valid = True
        self.setStyleSheet("")
        self.setToolTip(self.tooltip_text)


class StringField(ConfigField):
    """Campo de texto simple"""
    def __init__(self, label, default="", tooltip="", placeholder="", parent=None):
        super().__init__(label, tooltip, parent)
        self.widget = QLineEdit()
        self.widget.setText(default)
        self.widget.setPlaceholderText(placeholder)
        self.widget.setToolTip(tooltip)
        self.widget.textChanged.connect(self.valueChanged.emit)
    
    def get_value(self):
        return self.widget.text()
    
    def set_value(self, value):
        self.widget.setText(str(value))


class IntField(ConfigField):
    """Campo numérico entero"""
    def __init__(self, label, default=0, min_val=0, max_val=1000, tooltip="", parent=None):
        super().__init__(label, tooltip, parent)
        self.widget = QSpinBox()
        self.widget.setRange(min_val, max_val)
        self.widget.setValue(default)
        self.widget.setToolTip(tooltip)
        self.widget.valueChanged.connect(self.valueChanged.emit)
    
    def get_value(self):
        return self.widget.value()
    
    def set_value(self, value):
        self.widget.setValue(int(value))


class FloatField(ConfigField):
    """Campo numérico decimal"""
    def __init__(self, label, default=0.0, min_val=0.0, max_val=1.0, 
                 decimals=4, step=0.01, tooltip="", parent=None):
        super().__init__(label, tooltip, parent)
        self.widget = QDoubleSpinBox()
        self.widget.setRange(min_val, max_val)
        self.widget.setDecimals(decimals)
        self.widget.setSingleStep(step)
        self.widget.setValue(default)
        self.widget.setToolTip(tooltip)
        self.widget.valueChanged.connect(self.valueChanged.emit)
    
    def get_value(self):
        return self.widget.value()
    
    def set_value(self, value):
        self.widget.setValue(float(value))


class BoolField(ConfigField):
    """Campo booleano (checkbox)"""
    def __init__(self, label, default=False, tooltip="", parent=None):
        super().__init__(label, tooltip, parent)
        self.widget = QCheckBox()
        self.widget.setChecked(default)
        self.widget.setToolTip(tooltip)
        self.widget.stateChanged.connect(self.valueChanged.emit)
    
    def get_value(self):
        return self.widget.isChecked()
    
    def set_value(self, value):
        self.widget.setChecked(bool(value))


class ChoiceField(ConfigField):
    """Campo de selección (combobox)"""
    def __init__(self, label, choices, default=None, tooltip="", parent=None):
        super().__init__(label, tooltip, parent)
        self.widget = QComboBox()
        self.widget.addItems(choices)
        if default:
            index = self.widget.findText(default)
            if index >= 0:
                self.widget.setCurrentIndex(index)
        self.widget.setToolTip(tooltip)
        self.widget.currentTextChanged.connect(self.valueChanged.emit)
    
    def get_value(self):
        return self.widget.currentText()
    
    def set_value(self, value):
        index = self.widget.findText(str(value))
        if index >= 0:
            self.widget.setCurrentIndex(index)


class PathField(ConfigField):
    """Campo de ruta con botón de búsqueda"""
    def __init__(self, label, default="", tooltip="", is_dir=False, parent=None):
        super().__init__(label, tooltip, parent)
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.widget = QLineEdit()
        self.widget.setText(default)
        self.widget.setToolTip(tooltip)
        self.widget.textChanged.connect(self.valueChanged.emit)
        
        self.browse_btn = QPushButton("📁")
        self.browse_btn.setMaximumWidth(40)
        self.browse_btn.clicked.connect(lambda: self.browse(is_dir))
        
        layout.addWidget(self.widget)
        layout.addWidget(self.browse_btn)
        
        container = QWidget()
        container.setLayout(layout)
        self.widget_container = container
    
    def browse(self, is_dir):
        if is_dir:
            path = QFileDialog.getExistingDirectory(self, "Seleccionar Directorio")
        else:
            path, _ = QFileDialog.getOpenFileName(self, "Seleccionar Archivo")
        
        if path:
            self.widget.setText(path)
    
    def get_value(self):
        return self.widget.text()
    
    def set_value(self, value):
        self.widget.setText(str(value))


class ListField(ConfigField):
    """Campo para listas (valores separados por comas)"""
    def __init__(self, label, default=None, tooltip="", parent=None):
        super().__init__(label, tooltip, parent)
        self.widget = QLineEdit()
        if default:
            self.widget.setText(", ".join(map(str, default)))
        self.widget.setPlaceholderText("Valores separados por comas")
        self.widget.setToolTip(tooltip)
        self.widget.textChanged.connect(self.valueChanged.emit)
    
    def get_value(self):
        text = self.widget.text().strip()
        if not text:
            return []
        # Intenta convertir a números si es posible
        values = [v.strip() for v in text.split(',')]
        try:
            return [float(v) if '.' in v else int(v) for v in values]
        except:
            return values
    
    def set_value(self, value):
        if isinstance(value, list):
            self.widget.setText(", ".join(map(str, value)))
        else:
            self.widget.setText(str(value))


class ConfigEditorTab(QWidget):
    """
    Tab principal del editor de configuración.
    Implementa IEEE 9241-143 para diseño de formularios.
    """
    configChanged = pyqtSignal()
    configSaved = pyqtSignal(str)
    
    @staticmethod
    def resolve_config_path(config_path: str) -> Path:
        """Resolve config.yaml relative to project root (parent of lib/)."""
        p = Path(config_path)
        if p.is_absolute():
            return p.resolve()
        root = Path(__file__).resolve().parent.parent
        for candidate in (root / p, Path.cwd() / p):
            if candidate.exists():
                return candidate.resolve()
        return (root / p).resolve()

    def __init__(self, config_path="config.yaml", parent=None):
        super().__init__(parent)
        self.config_path = self.resolve_config_path(config_path)
        self.fields = {}
        self.config_data = {}
        self.parent_window = parent
        self._dirty = False
        self._suppress_dirty = False
        
        self.init_ui()
        self._connect_dirty_tracking()
        self._wire_ablation_controls()
        self.load_config()
    
    def init_ui(self):
        """Inicializa interfaz de usuario"""
        main_layout = QVBoxLayout()
        
        # Header con info
        header = QLabel("⚙️ Editor de Configuración")
        header.setStyleSheet(Styles.HEADER_SECTION)
        main_layout.addWidget(header)
        
        # Info del archivo
        self.file_info = QLabel(f"📄 Archivo: {self.config_path.absolute()}")
        self.file_info.setStyleSheet("padding: 5px; color: #666;")
        main_layout.addWidget(self.file_info)
        
        # Scroll area para formulario
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        # Widget contenedor del formulario
        form_widget = QWidget()
        form_layout = QVBoxLayout()
        
        # === TABS para secciones ===
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #cccccc;
                border-radius: 5px;
            }
            QTabBar::tab {
                background-color: #e0e0e0;
                padding: 8px 15px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: #0e639c;
                color: white;
            }
        """)
        
        # Crear tabs por sección
        self.create_model_tab()
        self.create_data_tab()
        self.create_grain_tab()
        self.create_augmentation_tab()
        self.create_training_tab()
        self.create_evaluation_tab()
        self.create_inference_tab()
        self.create_paths_tab()
        self.create_hardware_tab()
        
        form_layout.addWidget(self.tabs)
        form_widget.setLayout(form_layout)
        scroll.setWidget(form_widget)
        main_layout.addWidget(scroll)
        
        # Botones de acción
        buttons_layout = QHBoxLayout()
        
        self.load_btn = QPushButton("📂 Recargar")
        self.load_btn.clicked.connect(self.load_config)
        self.load_btn.setToolTip("Recargar configuración desde archivo")
        
        self.validate_btn = QPushButton("✓ Validar")
        self.validate_btn.clicked.connect(self.validate_all)
        self.validate_btn.setToolTip("Validar todos los campos")
        
        self.save_btn = QPushButton("💾 Guardar")
        self.save_btn.setStyleSheet("""
            QPushButton {
                background-color: #0e639c;
                color: white;
                font-weight: bold;
                padding: 10px 20px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #1177bb;
            }
        """)
        self.save_btn.clicked.connect(self.save_config)
        self.save_btn.setToolTip("Guardar cambios en config.yaml")
        
        self.reset_btn = QPushButton("↺ Restaurar")
        self.reset_btn.clicked.connect(self.reset_to_defaults)
        self.reset_btn.setToolTip("Restaurar valores originales")
        
        buttons_layout.addWidget(self.load_btn)
        buttons_layout.addWidget(self.validate_btn)
        buttons_layout.addStretch()
        buttons_layout.addWidget(self.reset_btn)
        buttons_layout.addWidget(self.save_btn)
        
        main_layout.addLayout(buttons_layout)
        
        # Status
        self.status_label = QLabel("Listo")
        self.status_label.setStyleSheet("padding: 5px; color: #666;")
        main_layout.addWidget(self.status_label)
        
        self.setLayout(main_layout)
    
    def _on_sampler_param_edited(self):
        if getattr(self, '_updating_batch_params', False): return
        self._updating_batch_params = True
        try:
            p = self.fields['data.classes_per_batch'].get_value()
            k = self.fields['data.samples_per_class'].get_value()
            new_batch = p * k
            if self.fields['data.batch_size'].get_value() != new_batch:
                self.fields['data.batch_size'].set_value(new_batch)
                self.batch_warning_label.setText(f"✓ Batch Size actualizado a {new_batch} ({p} × {k})")
            else:
                self.batch_warning_label.setText("")
        finally:
            self._updating_batch_params = False

    def _on_batch_size_edited(self):
        if getattr(self, '_updating_batch_params', False): return
        self._updating_batch_params = True
        try:
            b = self.fields['data.batch_size'].get_value()
            k = self.fields['data.samples_per_class'].get_value()
            p = max(2, b // k)
            new_batch = p * k
            if self.fields['data.classes_per_batch'].get_value() != p:
                self.fields['data.classes_per_batch'].set_value(p)
            if b != new_batch:
                self.fields['data.batch_size'].set_value(new_batch)
                self.batch_warning_label.setText(f"⚠ Batch auto-ajustado a {new_batch} (múltiplo de {k})")
            else:
                self.batch_warning_label.setText(f"✓ Batch OK ({p} × {k})")
        finally:
            self._updating_batch_params = False

    def create_model_tab(self):
        """Tab de configuración del modelo"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Experiment group (nuevo: tracking y metadata)
        exp_group = QGroupBox("Experimento")
        exp_layout = QFormLayout()
        
        self.fields['experiment.name'] = StringField(
            "Name",
            default="dinov2_vits14_baseline",
            tooltip="Nombre del experimento para tracking",
            placeholder="nombre_modelo_baseline"
        )
        exp_layout.addRow("Name:", self.fields['experiment.name'].widget)
        
        self.fields['experiment.description'] = StringField(
            "Description",
            default="Baseline experiment",
            tooltip="Descripción breve del experimento",
            placeholder="Descripción del modelo y configuración"
        )
        exp_layout.addRow("Description:", self.fields['experiment.description'].widget)
        
        exp_group.setLayout(exp_layout)
        layout.addWidget(exp_group)
        
        # Model group
        model_group = QGroupBox("Arquitectura del Modelo")
        model_layout = QFormLayout()
        
        self.fields['model.backbone_name'] = ChoiceField(
            "Backbone",
            ['dinov2_vits14', 'dinov2_vitb14', 'resnet50', 'convnext_v2_tiny',
             'deit_small_patch16_224', 'efficientnet_b4', 'swin_tiny_patch4_window7_224',
             'resnet18', 'dinov2_vits14_reg', 'dinov2_vitb14_reg'],
            tooltip="Arquitectura base del modelo (DINOv2 recomendado para fine-grained)"
        )
        # Conectar cambio de backbone a auto-configuración
        self.fields['model.backbone_name'].widget.currentTextChanged.connect(self.on_backbone_changed)
        model_layout.addRow("Backbone:", self.fields['model.backbone_name'].widget)
        
        self.fields['model.embedding_dim'] = IntField(
            "Embedding Dim",
            default=128, min_val=64, max_val=512,
            tooltip="Dimensión del vector de embeddings (recomendado: 128)"
        )
        model_layout.addRow("Embedding Dim:", self.fields['model.embedding_dim'].widget)
        
        self.fields['model.pretrained'] = BoolField(
            "Pretrained",
            tooltip="Usar pesos pre-entrenados de ImageNet"
        )
        model_layout.addRow("Pretrained:", self.fields['model.pretrained'].widget)

        self.fields['model.freeze_backbone'] = BoolField(
            "Freeze Backbone",
            tooltip="Congelar backbone (solo entrena projection head; más rápido, menos calidad)"
        )
        model_layout.addRow("Freeze Backbone:", self.fields['model.freeze_backbone'].widget)

        self.fields['model.use_attention'] = BoolField(
            "Use Attention",
            tooltip="Capa de atención adicional en el head"
        )
        model_layout.addRow("Use Attention:", self.fields['model.use_attention'].widget)
        
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        # Projection Head group
        proj_group = QGroupBox("Projection Head")
        proj_layout = QFormLayout()
        
        self.fields['model.projection_head.hidden_dims'] = ListField(
            "Hidden Dims",
            tooltip="Dimensiones de capas ocultas (ej: 512, 256)"
        )
        proj_layout.addRow("Hidden Dims:", self.fields['model.projection_head.hidden_dims'].widget)
        
        self.fields['model.projection_head.use_batch_norm'] = BoolField(
            "Batch Norm",
            default=False,
            tooltip="BatchNorm (desaconsejado con batch pequeño; preferir LayerNorm)"
        )
        proj_layout.addRow("Batch Norm:", self.fields['model.projection_head.use_batch_norm'].widget)

        self.fields['model.projection_head.use_layer_norm'] = BoolField(
            "Layer Norm",
            default=True,
            tooltip="LayerNorm en capas ocultas del projection head (D-01)"
        )
        proj_layout.addRow("Layer Norm:", self.fields['model.projection_head.use_layer_norm'].widget)
        
        self.fields['model.projection_head.dropout'] = FloatField(
            "Dropout",
            default=0.0, min_val=0.0, max_val=0.9, decimals=2, step=0.1,
            tooltip="Tasa de dropout (0.0 = sin dropout)"
        )
        proj_layout.addRow("Dropout:", self.fields['model.projection_head.dropout'].widget)
        
        self.fields['model.projection_head.normalize_embeddings_ln'] = BoolField(
            "Normalize LN",
            default=False,
            tooltip="Aplicar LayerNorm final al projection head"
        )
        proj_layout.addRow("Normalize LN:", self.fields['model.projection_head.normalize_embeddings_ln'].widget)
        
        self.fields['model.projection_head.use_hardtanh'] = BoolField(
            "Use HardTanh",
            default=False,
            tooltip="Aplicar HardTanh al final de la proyección"
        )
        proj_layout.addRow("Use HardTanh:", self.fields['model.projection_head.use_hardtanh'].widget)
        
        self.fields['model.projection_head.hardtanh_min'] = FloatField(
            "HardTanh Min",
            default=-2.0, min_val=-10.0, max_val=0.0, decimals=1, step=0.5,
            tooltip="Límite inferior HardTanh"
        )
        proj_layout.addRow("HardTanh Min:", self.fields['model.projection_head.hardtanh_min'].widget)
        
        self.fields['model.projection_head.hardtanh_max'] = FloatField(
            "HardTanh Max",
            default=2.0, min_val=0.0, max_val=10.0, decimals=1, step=0.5,
            tooltip="Límite superior HardTanh"
        )
        proj_layout.addRow("HardTanh Max:", self.fields['model.projection_head.hardtanh_max'].widget)
        
        proj_group.setLayout(proj_layout)
        layout.addWidget(proj_group)

        pool_group = QGroupBox("Pooling y Proyección (enlazado a ablaciones F-01, F-03, F-04)")
        pool_layout = QFormLayout()

        self.fields['model.pooling.strategy'] = ChoiceField(
            "Pooling Strategy",
            ['multi_spectral', 'gem', 'mean', 'max', 'attention', 'fsw'],
            default='multi_spectral',
            tooltip="Estrategia de pooling espacial (F-01)"
        )
        pool_layout.addRow("Pooling:", self.fields['model.pooling.strategy'].widget)

        self.fields['model.pooling.layer_fusion'] = ChoiceField(
            "Layer Fusion",
            ['per_layer', 'concat'],
            default='per_layer',
            tooltip="Fusión de capas ViT (F-03 layout)"
        )
        pool_layout.addRow("Layer fusion:", self.fields['model.pooling.layer_fusion'].widget)

        self.fields['model.projection.layout'] = ChoiceField(
            "Projection Layout",
            ['lf_hf_dual', 'slice_native', 'multi_tower'],
            default='multi_tower',
            tooltip="Layout del projection head (F-03)"
        )
        pool_layout.addRow("Layout:", self.fields['model.projection.layout'].widget)

        self.fields['model.pooling.masked.enabled'] = BoolField(
            "Pooling Masked",
            default=True,
            tooltip="Pooling con máscara de grano (F-04; sincronizar con Grano → Masked Pooling)"
        )
        pool_layout.addRow("Masked pooling:", self.fields['model.pooling.masked.enabled'].widget)

        self.fields['model.num_unfrozen_blocks'] = IntField(
            "Unfrozen Blocks",
            default=2, min_val=0, max_val=24,
            tooltip="Bloques ViT descongelados si freeze_backbone=true (F-06)"
        )
        pool_layout.addRow("Bloques unfrozen:", self.fields['model.num_unfrozen_blocks'].widget)

        pool_group.setLayout(pool_layout)
        layout.addWidget(pool_group)
        
        layout.addStretch()
        tab.setLayout(layout)
        self.tabs.addTab(tab, "🤖 Modelo")
    
    def create_data_tab(self):
        """Tab de configuración de datos"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Paths group
        paths_group = QGroupBox("Rutas de Datos")
        paths_layout = QFormLayout()
        
        self.fields['data.train_dir'] = PathField(
            "Train Dir",
            tooltip="Directorio de entrenamiento",
            is_dir=True
        )
        paths_layout.addRow("Train:", self.fields['data.train_dir'].widget_container)
        
        self.fields['data.val_dir'] = PathField(
            "Val Dir",
            tooltip="Directorio de validación",
            is_dir=True
        )
        paths_layout.addRow("Validation:", self.fields['data.val_dir'].widget_container)
        
        self.fields['data.test_dir'] = PathField(
            "Test Dir",
            tooltip="Directorio de test",
            is_dir=True
        )
        paths_layout.addRow("Test:", self.fields['data.test_dir'].widget_container)

        self.fields['data.annotation_root'] = PathField(
            "Annotation Root",
            tooltip="Raíz de anotaciones .seg (modo grain on-the-fly)",
            is_dir=True
        )
        paths_layout.addRow("Anotaciones (.seg):", self.fields['data.annotation_root'].widget_container)
        
        paths_group.setLayout(paths_layout)
        layout.addWidget(paths_group)
        
        # Sampler group
        sampler_group = QGroupBox("Sampler (P×K Strategy)")
        sampler_layout = QFormLayout()
        
        self.fields['data.classes_per_batch'] = IntField(
            "Classes per Batch",
            default=3, min_val=2, max_val=20,
            tooltip="P: Número de clases por batch"
        )
        sampler_layout.addRow("Classes (P):", self.fields['data.classes_per_batch'].widget)
        
        self.fields['data.samples_per_class'] = IntField(
            "Samples per Class",
            default=8, min_val=2, max_val=32,
            tooltip="K: Muestras por clase"
        )
        sampler_layout.addRow("Samples (K):", self.fields['data.samples_per_class'].widget)
        
        self.fields['data.batch_size'] = IntField(
            "Batch Size",
            default=24, min_val=4, max_val=256,
            tooltip="Tamaño del batch (P × K)"
        )
        sampler_layout.addRow("Batch Size:", self.fields['data.batch_size'].widget)

        self.batch_warning_label = QLabel()
        self.batch_warning_label.setStyleSheet("color: #d97706; font-size: 11px; font-weight: bold;")
        sampler_layout.addRow("", self.batch_warning_label)

        # Connect signals for auto-correction
        self.fields['data.classes_per_batch'].widget.valueChanged.connect(self._on_sampler_param_edited)
        self.fields['data.samples_per_class'].widget.valueChanged.connect(self._on_sampler_param_edited)
        self.fields['data.batch_size'].widget.valueChanged.connect(self._on_batch_size_edited)

        self.fields['data.val_batch_size'] = IntField(
            "Val Batch Size",
            default=0, min_val=0, max_val=512,
            tooltip="0 = mismo que train. Puede ser mayor en val (sin backward) para menos batches"
        )
        sampler_layout.addRow("Val Batch Size:", self.fields['data.val_batch_size'].widget)

        self.fields['data.auto_tune_resources'] = BoolField(
            "Auto Tune Resources",
            default=True,
            tooltip="Ajusta workers/prefetch/caché según CPU y RAM disponibles al crear DataLoaders"
        )
        sampler_layout.addRow("Auto-tune CPU/RAM:", self.fields['data.auto_tune_resources'].widget)
        
        sampler_group.setLayout(sampler_layout)
        layout.addWidget(sampler_group)
        
        # DataLoader group
        loader_group = QGroupBox("DataLoader")
        loader_layout = QFormLayout()
        
        self.fields['data.num_workers'] = IntField(
            "Num Workers",
            default=0, min_val=0, max_val=16,
            tooltip="Workers del DataLoader de entrenamiento (0 = hilo principal)"
        )
        loader_layout.addRow("Num Workers:", self.fields['data.num_workers'].widget)

        self.fields['data.prefetch_factor'] = IntField(
            "Prefetch Factor",
            default=2, min_val=1, max_val=8,
            tooltip="Batches prefetched por worker (requiere num_workers > 0)"
        )
        loader_layout.addRow("Prefetch:", self.fields['data.prefetch_factor'].widget)

        self.fields['data.val_num_workers'] = IntField(
            "Val Workers",
            default=-1, min_val=-1, max_val=16,
            tooltip="-1 = mismo que train; 0 = validación en hilo principal"
        )
        loader_layout.addRow("Val Workers:", self.fields['data.val_num_workers'].widget)
        
        self.fields['data.pin_memory'] = BoolField(
            "Pin Memory",
            tooltip="Habilitar pin_memory para GPU"
        )
        loader_layout.addRow("Pin Memory:", self.fields['data.pin_memory'].widget)
        
        self.fields['data.drop_last'] = BoolField(
            "Drop Last",
            tooltip="Descartar último batch incompleto"
        )
        loader_layout.addRow("Drop Last:", self.fields['data.drop_last'].widget)
        
        loader_group.setLayout(loader_layout)
        layout.addWidget(loader_group)

        policy_group = QGroupBox("Política DataLoader")
        policy_layout = QFormLayout()

        self.fields['data.loader_policy.exclusive_train_workers_during_val'] = BoolField(
            "Exclusive Train Workers",
            default=True,
            tooltip="Libera workers de train durante validación (evita duplicar RAM)"
        )
        policy_layout.addRow(
            "Workers exclusivos en val:",
            self.fields['data.loader_policy.exclusive_train_workers_during_val'].widget
        )

        self.fields['data.loader_policy.persistent_workers'] = BoolField(
            "Persistent Workers",
            default=True,
            tooltip="Mantener workers vivos entre épocas (menos overhead de spawn)"
        )
        policy_layout.addRow(
            "Persistent workers:",
            self.fields['data.loader_policy.persistent_workers'].widget
        )

        self.fields['data.worker_ram_budget_mb'] = IntField(
            "Worker RAM Budget",
            default=900, min_val=400, max_val=32000,
            tooltip="Si la RAM estimada de workers supera este valor, se reducen workers/cache automáticamente"
        )
        policy_layout.addRow("RAM budget (MB):", self.fields['data.worker_ram_budget_mb'].widget)

        policy_group.setLayout(policy_layout)
        layout.addWidget(policy_group)
        
        # Image preprocessing
        img_group = QGroupBox("Preprocesamiento de Imágenes")
        img_layout = QFormLayout()
        
        self.fields['data.image_size'] = IntField(
            "Image Size",
            default=224, min_val=64, max_val=512,
            tooltip="Resolución canónica del pipeline (crop HDF5, transforms, modelo). "
                    "Al cambiar, resize y crop de grano se sincronizan automáticamente."
        )
        img_layout.addRow("Image Size:", self.fields['data.image_size'].widget)
        self.fields['data.image_size'].widget.valueChanged.connect(self._on_image_size_changed)
        
        self.fields['data.resize_size'] = IntField(
            "Resize Size",
            default=256, min_val=64, max_val=512,
            tooltip="Espejo de Image Size (sincronizado automáticamente al guardar)"
        )
        self.fields['data.resize_size'].widget.setReadOnly(True)
        self.fields['data.resize_size'].widget.setButtonSymbols(QSpinBox.NoButtons)
        img_layout.addRow("Resize Size:", self.fields['data.resize_size'].widget)
        
        img_group.setLayout(img_layout)
        layout.addWidget(img_group)

        norm_group = QGroupBox("Normalización (data.normalize)")
        norm_layout = QFormLayout()

        self.fields['data.normalize.mode'] = ChoiceField(
            "Norm Mode",
            ['imagenet', 'dataset', 'macenko'],
            default='imagenet',
            tooltip="imagenet | dataset (stats_path) | macenko (stain + ImageNet)"
        )
        norm_layout.addRow("Mode:", self.fields['data.normalize.mode'].widget)

        self.fields['data.normalize.stats_path'] = PathField(
            "Stats Path",
            tooltip="JSON generado por scripts/compute_dataset_norm.py",
            is_dir=False
        )
        norm_layout.addRow("Stats JSON:", self.fields['data.normalize.stats_path'].widget_container)

        self.fields['data.normalize.stain_method'] = ChoiceField(
            "Stain Method",
            ['null', 'reinhard', 'macenko'],
            default='null',
            tooltip="Pre-normalize opcional (E-02)"
        )
        norm_layout.addRow("Stain:", self.fields['data.normalize.stain_method'].widget)

        norm_group.setLayout(norm_layout)
        layout.addWidget(norm_group)
        
        layout.addStretch()
        tab.setLayout(layout)
        self.tabs.addTab(tab, "💾 Datos")

    def create_grain_tab(self):
        """Tab de detección de grano on-the-fly (.seg)"""
        tab = QWidget()
        layout = QVBoxLayout()

        mode_group = QGroupBox("Modo Dataset")
        mode_layout = QFormLayout()

        self.fields['grain_detection.enabled'] = BoolField(
            "Pipeline grain (.seg)",
            default=True,
            tooltip="Siempre on-the-fly: 1 muestra = 1 grano desde anotaciones .seg. "
                    "Desactivar solo si cada carpeta de clase ya tiene imágenes recortadas listas."
        )
        mode_layout.addRow("Habilitado:", self.fields['grain_detection.enabled'].widget)

        mode_note = QLabel(
            "Con el pipeline activo, cada época lee symlinks + .seg y genera crops/máscaras "
            "en memoria (no hay modo «image» separado en entrenamiento)."
        )
        mode_note.setWordWrap(True)
        mode_note.setStyleSheet("color: #666; padding: 4px 0;")
        mode_layout.addRow(mode_note)

        mode_group.setLayout(mode_layout)
        layout.addWidget(mode_group)

        crop_group = QGroupBox("Crop y Máscara")
        crop_layout = QFormLayout()

        self.fields['grain_detection.crop_size'] = IntField(
            "Crop Size",
            default=252, min_val=64, max_val=512,
            tooltip="Espejo de Datos → Image Size (solo lectura; edítalo en Preprocesamiento)"
        )
        self.fields['grain_detection.crop_size'].widget.setReadOnly(True)
        self.fields['grain_detection.crop_size'].widget.setButtonSymbols(QSpinBox.NoButtons)
        crop_layout.addRow("Crop Size (sync):", self.fields['grain_detection.crop_size'].widget)

        self.fields['grain_detection.crop_padding'] = FloatField(
            "Crop Padding",
            default=0.2, min_val=0.0, max_val=1.0, decimals=2, step=0.05,
            tooltip="Padding relativo al bbox del grano"
        )
        crop_layout.addRow("Crop Padding:", self.fields['grain_detection.crop_padding'].widget)

        self.fields['grain_detection.use_mask'] = BoolField(
            "Use Mask",
            default=True,
            tooltip="Aplicar máscara del grano en augmentación"
        )
        crop_layout.addRow("Use Mask:", self.fields['grain_detection.use_mask'].widget)

        self.fields['grain_detection.masked_pooling'] = BoolField(
            "Masked Pooling",
            default=True,
            tooltip="Pooling atento a la máscara en el backbone"
        )
        crop_layout.addRow("Masked Pooling:", self.fields['grain_detection.masked_pooling'].widget)

        self.fields['grain_detection.train_mask_bg_mode'] = ChoiceField(
            "Train BG Mode",
            ['imagenet_neutral', 'gray', 'random', 'black', 'none'],
            default='imagenet_neutral',
            tooltip="imagenet_neutral = zero activation (fondo → ~0 tras Normalize ImageNet)"
        )
        crop_layout.addRow("Train BG:", self.fields['grain_detection.train_mask_bg_mode'].widget)

        self.fields['grain_detection.val_mask_bg_mode'] = ChoiceField(
            "Val BG Mode",
            ['imagenet_neutral', 'gray', 'random', 'black', 'none'],
            default='imagenet_neutral',
            tooltip="Fondo fuera de máscara en validación/inferencia"
        )
        crop_layout.addRow("Val BG:", self.fields['grain_detection.val_mask_bg_mode'].widget)

        crop_group.setLayout(crop_layout)
        layout.addWidget(crop_group)

        cache_group = QGroupBox("Caché de Imágenes (workers)")
        cache_layout = QFormLayout()

        self.fields['grain_detection.image_cache_size'] = IntField(
            "Cache Size",
            default=6, min_val=0, max_val=64,
            tooltip="Entradas LRU por worker (0 = sin caché)"
        )
        cache_layout.addRow("Entradas LRU:", self.fields['grain_detection.image_cache_size'].widget)

        self.fields['grain_detection.image_cache_max_mb'] = IntField(
            "Cache Max MB",
            default=48, min_val=0, max_val=512,
            tooltip="Límite de RAM por worker para caché de imágenes"
        )
        cache_layout.addRow("Máx MB/worker:", self.fields['grain_detection.image_cache_max_mb'].widget)

        self.fields['grain_detection.max_cached_image_mb'] = IntField(
            "Max Frame Cache MB",
            default=12, min_val=4, max_val=64,
            tooltip="No cachear en LRU frames más grandes que esto (microscopía full-res ~19MB)"
        )
        cache_layout.addRow("Máx frame (MB):", self.fields['grain_detection.max_cached_image_mb'].widget)

        self.fields['grain_detection.max_load_side'] = IntField(
            "Max Load Side",
            default=1536, min_val=512, max_val=4096,
            tooltip="Redimensiona al cargar si el lado mayor supera este valor (menos RAM/CPU, mismo crop)"
        )
        cache_layout.addRow("Max load side:", self.fields['grain_detection.max_load_side'].widget)

        cache_group.setLayout(cache_layout)
        layout.addWidget(cache_group)

        det_group = QGroupBox("Detección (U²-Net / saliency)")
        det_layout = QFormLayout()

        self.fields['grain_detection.model_type'] = ChoiceField(
            "Model Type",
            ['u2netp', 'u2net'],
            default='u2netp',
            tooltip="Modelo de segmentación auxiliar"
        )
        det_layout.addRow("Modelo:", self.fields['grain_detection.model_type'].widget)

        self.fields['grain_detection.input_size'] = IntField(
            "Input Size",
            default=320, min_val=128, max_val=512,
            tooltip="Resolución de entrada del detector"
        )
        det_layout.addRow("Input Size:", self.fields['grain_detection.input_size'].widget)

        self.fields['grain_detection.saliency_threshold'] = FloatField(
            "Saliency Threshold",
            default=0.3, min_val=0.0, max_val=1.0, decimals=2, step=0.05,
            tooltip="Umbral de mapa de saliencia"
        )
        det_layout.addRow("Saliency:", self.fields['grain_detection.saliency_threshold'].widget)

        self.fields['grain_detection.adaptive_k'] = FloatField(
            "Adaptive K",
            default=0.3, min_val=0.0, max_val=1.0, decimals=2, step=0.05,
            tooltip="Factor k adaptativo para umbral"
        )
        det_layout.addRow("Adaptive K:", self.fields['grain_detection.adaptive_k'].widget)

        self.fields['grain_detection.min_area'] = IntField(
            "Min Area",
            default=1000, min_val=0, max_val=1000000,
            tooltip="Área mínima del grano (px²)"
        )
        det_layout.addRow("Min Area:", self.fields['grain_detection.min_area'].widget)

        self.fields['grain_detection.max_area'] = IntField(
            "Max Area",
            default=120000, min_val=0, max_val=10000000,
            tooltip="Área máxima del grano (px²)"
        )
        det_layout.addRow("Max Area:", self.fields['grain_detection.max_area'].widget)

        self.fields['grain_detection.min_circularity'] = FloatField(
            "Min Circularity",
            default=0.3, min_val=0.0, max_val=1.0, decimals=2, step=0.05,
            tooltip="Circularidad mínima aceptada"
        )
        det_layout.addRow("Min Circularity:", self.fields['grain_detection.min_circularity'].widget)

        self.fields['grain_detection.morph_kernel_size'] = IntField(
            "Morph Kernel",
            default=3, min_val=1, max_val=21,
            tooltip="Tamaño del kernel morfológico"
        )
        det_layout.addRow("Morph Kernel:", self.fields['grain_detection.morph_kernel_size'].widget)

        self.fields['grain_detection.distance_threshold'] = FloatField(
            "Distance Threshold",
            default=0.7, min_val=0.0, max_val=2.0, decimals=2, step=0.05,
            tooltip="Umbral de distancia entre granos"
        )
        det_layout.addRow("Distance:", self.fields['grain_detection.distance_threshold'].widget)

        det_group.setLayout(det_layout)
        layout.addWidget(det_group)

        layout.addStretch()
        tab.setLayout(layout)
        self.tabs.addTab(tab, "🌾 Grano")
    
    def create_augmentation_tab(self):
        """Tab de augmentación"""
        tab = QWidget()
        layout = QVBoxLayout()

        sota_group = QGroupBox("Modo de Augmentación (SOTA)")
        sota_layout = QFormLayout()

        self.fields['sota.augmentation_mode'] = ChoiceField(
            "Augmentation Mode",
            ['full', 'deit3', 'deit3_simple'],
            default='deit3_simple',
            tooltip="full = microscopía; deit3/deit3_simple = DeiT III 3-Augment (recomendado ViT)"
        )
        sota_layout.addRow("Modo:", self.fields['sota.augmentation_mode'].widget)

        sota_group.setLayout(sota_layout)
        layout.addWidget(sota_group)
        
        # Geometric
        geom_group = QGroupBox("Augmentaciones Geométricas")
        geom_layout = QFormLayout()
        
        self.fields['augmentation.random_horizontal_flip'] = FloatField(
            "H Flip",
            default=0.5, min_val=0.0, max_val=1.0, decimals=2, step=0.1,
            tooltip="Probabilidad de flip horizontal"
        )
        geom_layout.addRow("Horizontal Flip:", self.fields['augmentation.random_horizontal_flip'].widget)
        
        self.fields['augmentation.random_vertical_flip'] = FloatField(
            "V Flip",
            default=0.5, min_val=0.0, max_val=1.0, decimals=2, step=0.1,
            tooltip="Probabilidad de flip vertical"
        )
        geom_layout.addRow("Vertical Flip:", self.fields['augmentation.random_vertical_flip'].widget)
        
        self.fields['augmentation.random_rotation'] = IntField(
            "Rotation",
            default=180, min_val=0, max_val=360,
            tooltip="Grados de rotación máxima"
        )
        geom_layout.addRow("Rotation (deg):", self.fields['augmentation.random_rotation'].widget)
        
        self.fields['augmentation.random_crop'] = BoolField(
            "Random Crop",
            tooltip="Aplicar random crop"
        )
        geom_layout.addRow("Random Crop:", self.fields['augmentation.random_crop'].widget)
        
        geom_group.setLayout(geom_layout)
        layout.addWidget(geom_group)
        
        # Color
        color_group = QGroupBox("Augmentaciones de Color")
        color_layout = QFormLayout()
        
        self.fields['augmentation.color_jitter.brightness'] = FloatField(
            "Brightness",
            default=0.3, min_val=0.0, max_val=1.0, decimals=2, step=0.1,
            tooltip="Factor de brillo"
        )
        color_layout.addRow("Brightness:", self.fields['augmentation.color_jitter.brightness'].widget)
        
        self.fields['augmentation.color_jitter.contrast'] = FloatField(
            "Contrast",
            default=0.3, min_val=0.0, max_val=1.0, decimals=2, step=0.1,
            tooltip="Factor de contraste"
        )
        color_layout.addRow("Contrast:", self.fields['augmentation.color_jitter.contrast'].widget)
        
        self.fields['augmentation.color_jitter.saturation'] = FloatField(
            "Saturation",
            default=0.2, min_val=0.0, max_val=1.0, decimals=2, step=0.1,
            tooltip="Factor de saturación"
        )
        color_layout.addRow("Saturation:", self.fields['augmentation.color_jitter.saturation'].widget)
        
        self.fields['augmentation.color_jitter.hue'] = FloatField(
            "Hue",
            default=0.1, min_val=0.0, max_val=0.5, decimals=2, step=0.05,
            tooltip="Factor de matiz"
        )
        color_layout.addRow("Hue:", self.fields['augmentation.color_jitter.hue'].widget)
        
        color_group.setLayout(color_layout)
        layout.addWidget(color_group)

        # Grayscale (RGB + B&N robustness)
        gray_group = QGroupBox("Escala de grises (RGB + blanco y negro)")
        gray_layout = QFormLayout()

        self.fields['augmentation.grayscale.enabled'] = BoolField(
            "Grayscale",
            default=True,
            tooltip="Convierte aleatoriamente a escala de grises (3 canales). "
                    "Entrena resiliencia RGB y B&N."
        )
        gray_layout.addRow("Habilitar B&N:", self.fields['augmentation.grayscale.enabled'].widget)

        self.fields['augmentation.grayscale.p'] = FloatField(
            "Grayscale P",
            default=0.5, min_val=0.0, max_val=1.0, decimals=2, step=0.1,
            tooltip="Probabilidad de aplicar ToGray en cada sample"
        )
        gray_layout.addRow("Probabilidad:", self.fields['augmentation.grayscale.p'].widget)

        gray_group.setLayout(gray_layout)
        layout.addWidget(gray_group)

        perf_group = QGroupBox("Rendimiento")
        perf_layout = QFormLayout()
        self.fields['augmentation.performance_mode'] = BoolField(
            "Performance Mode",
            default=True,
            tooltip="Desactiva augmentaciones muy costosas en CPU (optical, domain)"
        )
        perf_layout.addRow("Modo rendimiento:", self.fields['augmentation.performance_mode'].widget)
        perf_group.setLayout(perf_layout)
        layout.addWidget(perf_group)

        micro_group = QGroupBox("Microscopía (Albumentations)")
        micro_layout = QFormLayout()

        self.fields['augmentation.microscopy_specific.random_gamma.enabled'] = BoolField(
            "Gamma Enabled", default=True, tooltip="Random gamma correction"
        )
        micro_layout.addRow("Gamma:", self.fields['augmentation.microscopy_specific.random_gamma.enabled'].widget)

        self.fields['augmentation.microscopy_specific.random_gamma.p'] = FloatField(
            "Gamma P", default=0.5, min_val=0.0, max_val=1.0, decimals=2, step=0.1,
            tooltip="Probabilidad random gamma"
        )
        micro_layout.addRow("Gamma P:", self.fields['augmentation.microscopy_specific.random_gamma.p'].widget)

        self.fields['augmentation.microscopy_specific.random_gamma.gamma_limit'] = ListField(
            "Gamma Limit", default=[80, 120],
            tooltip="Límites gamma (min, max) en %"
        )
        micro_layout.addRow("Gamma Limit:", self.fields['augmentation.microscopy_specific.random_gamma.gamma_limit'].widget)

        self.fields['augmentation.microscopy_specific.gaussian_noise.enabled'] = BoolField(
            "Noise Enabled", default=True, tooltip="Ruido gaussiano"
        )
        micro_layout.addRow("Noise:", self.fields['augmentation.microscopy_specific.gaussian_noise.enabled'].widget)

        self.fields['augmentation.microscopy_specific.gaussian_noise.p'] = FloatField(
            "Noise P", default=0.5, min_val=0.0, max_val=1.0, decimals=2, step=0.1,
            tooltip="Probabilidad ruido gaussiano"
        )
        micro_layout.addRow("Noise P:", self.fields['augmentation.microscopy_specific.gaussian_noise.p'].widget)

        self.fields['augmentation.microscopy_specific.gaussian_noise.var_limit'] = ListField(
            "Noise Var Limit", default=[10.0, 50.0],
            tooltip="Varianza mínima y máxima del ruido"
        )
        micro_layout.addRow("Var Limit:", self.fields['augmentation.microscopy_specific.gaussian_noise.var_limit'].widget)

        self.fields['augmentation.microscopy_specific.blur.enabled'] = BoolField(
            "Blur Enabled", default=True, tooltip="Desenfoque gaussiano"
        )
        micro_layout.addRow("Blur:", self.fields['augmentation.microscopy_specific.blur.enabled'].widget)

        self.fields['augmentation.microscopy_specific.blur.p'] = FloatField(
            "Blur P", default=0.3, min_val=0.0, max_val=1.0, decimals=2, step=0.1,
            tooltip="Probabilidad blur"
        )
        micro_layout.addRow("Blur P:", self.fields['augmentation.microscopy_specific.blur.p'].widget)

        self.fields['augmentation.microscopy_specific.blur.blur_limit'] = IntField(
            "Blur Limit", default=3, min_val=1, max_val=15,
            tooltip="Kernel máximo de blur"
        )
        micro_layout.addRow("Blur Limit:", self.fields['augmentation.microscopy_specific.blur.blur_limit'].widget)

        self.fields['augmentation.microscopy_specific.optical_distortion.enabled'] = BoolField(
            "Optical Distortion", default=True,
            tooltip="Distorsión óptica (costosa en CPU)"
        )
        micro_layout.addRow(
            "Optical Dist:",
            self.fields['augmentation.microscopy_specific.optical_distortion.enabled'].widget
        )

        self.fields['augmentation.microscopy_specific.optical_distortion.p'] = FloatField(
            "Optical P", default=0.3, min_val=0.0, max_val=1.0, decimals=2, step=0.1,
            tooltip="Probabilidad distorsión óptica"
        )
        micro_layout.addRow(
            "Optical P:",
            self.fields['augmentation.microscopy_specific.optical_distortion.p'].widget
        )

        self.fields['augmentation.microscopy_specific.optical_distortion.distort_limit'] = FloatField(
            "Distort Limit", default=0.1, min_val=0.0, max_val=1.0, decimals=2, step=0.05,
            tooltip="Magnitud de distorsión"
        )
        micro_layout.addRow(
            "Distort Limit:",
            self.fields['augmentation.microscopy_specific.optical_distortion.distort_limit'].widget
        )

        self.fields['augmentation.microscopy_specific.clahe.enabled'] = BoolField(
            "CLAHE Enabled", default=True, tooltip="CLAHE adaptativo"
        )
        micro_layout.addRow("CLAHE:", self.fields['augmentation.microscopy_specific.clahe.enabled'].widget)

        self.fields['augmentation.microscopy_specific.clahe.p'] = FloatField(
            "CLAHE P", default=0.5, min_val=0.0, max_val=1.0, decimals=2, step=0.1,
            tooltip="Probabilidad CLAHE"
        )
        micro_layout.addRow("CLAHE P:", self.fields['augmentation.microscopy_specific.clahe.p'].widget)

        self.fields['augmentation.microscopy_specific.clahe.clip_limit'] = FloatField(
            "CLAHE Clip", default=2.0, min_val=0.1, max_val=10.0, decimals=1, step=0.1,
            tooltip="Clip limit de CLAHE"
        )
        micro_layout.addRow("Clip Limit:", self.fields['augmentation.microscopy_specific.clahe.clip_limit'].widget)

        self.fields['augmentation.microscopy_specific.clahe.tile_grid_size'] = ListField(
            "CLAHE Grid", default=[8, 8],
            tooltip="Tamaño de rejilla CLAHE (filas, cols)"
        )
        micro_layout.addRow("Tile Grid:", self.fields['augmentation.microscopy_specific.clahe.tile_grid_size'].widget)

        micro_group.setLayout(micro_layout)
        layout.addWidget(micro_group)

        domain_group = QGroupBox("Domain Augmentation")
        domain_layout = QFormLayout()

        self.fields['augmentation.domain_augmentation.enabled'] = BoolField(
            "Domain Aug Enabled", default=True,
            tooltip="Augmentaciones de dominio (costosas; desactivar para más velocidad)"
        )
        domain_layout.addRow("Habilitado:", self.fields['augmentation.domain_augmentation.enabled'].widget)

        self.fields['augmentation.domain_augmentation.brightness_contrast_limit'] = FloatField(
            "BC Limit", default=0.4, min_val=0.0, max_val=1.0, decimals=2, step=0.05,
            tooltip="Límite brillo/contraste"
        )
        domain_layout.addRow("BC Limit:", self.fields['augmentation.domain_augmentation.brightness_contrast_limit'].widget)

        self.fields['augmentation.domain_augmentation.rgb_shift_limit'] = IntField(
            "RGB Shift", default=25, min_val=0, max_val=100,
            tooltip="Desplazamiento RGB máximo"
        )
        domain_layout.addRow("RGB Shift:", self.fields['augmentation.domain_augmentation.rgb_shift_limit'].widget)

        self.fields['augmentation.domain_augmentation.tone_curve_scale'] = FloatField(
            "Tone Curve", default=0.2, min_val=0.0, max_val=1.0, decimals=2, step=0.05,
            tooltip="Escala de curva de tono"
        )
        domain_layout.addRow("Tone Curve:", self.fields['augmentation.domain_augmentation.tone_curve_scale'].widget)

        self.fields['augmentation.domain_augmentation.downscale_min'] = FloatField(
            "Downscale Min", default=0.5, min_val=0.1, max_val=1.0, decimals=2, step=0.05,
            tooltip="Factor mínimo de downscale"
        )
        domain_layout.addRow("Downscale Min:", self.fields['augmentation.domain_augmentation.downscale_min'].widget)

        self.fields['augmentation.domain_augmentation.coarse_dropout_holes'] = IntField(
            "Coarse Dropout", default=4, min_val=0, max_val=32,
            tooltip="Número de huecos coarse dropout"
        )
        domain_layout.addRow("Dropout Holes:", self.fields['augmentation.domain_augmentation.coarse_dropout_holes'].widget)

        domain_group.setLayout(domain_layout)
        layout.addWidget(domain_group)
        
        # Normalization
        norm_group = QGroupBox("Normalización (ImageNet)")
        norm_layout = QFormLayout()
        
        self.fields['augmentation.normalize.mean'] = ListField(
            "Mean",
            tooltip="Media para normalización (R, G, B)"
        )
        norm_layout.addRow("Mean:", self.fields['augmentation.normalize.mean'].widget)
        
        self.fields['augmentation.normalize.std'] = ListField(
            "Std",
            tooltip="Desviación estándar (R, G, B)"
        )
        norm_layout.addRow("Std:", self.fields['augmentation.normalize.std'].widget)
        
        norm_group.setLayout(norm_layout)
        layout.addWidget(norm_group)
        
        layout.addStretch()
        tab.setLayout(layout)
        self.tabs.addTab(tab, "🎨 Augmentación")
    
    def create_training_tab(self):
        """Tab de entrenamiento"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Epochs
        epochs_group = QGroupBox("Épocas")
        epochs_layout = QFormLayout()
        
        self.fields['training.epochs'] = IntField(
            "Epochs",
            default=30, min_val=1, max_val=1000,
            tooltip="Número total de épocas"
        )
        epochs_layout.addRow("Epochs:", self.fields['training.epochs'].widget)
        
        self.fields['training.warmup_epochs'] = IntField(
            "Warmup",
            default=3, min_val=0, max_val=20,
            tooltip="Épocas de warmup"
        )
        epochs_layout.addRow("Warmup:", self.fields['training.warmup_epochs'].widget)
        
        epochs_group.setLayout(epochs_layout)
        layout.addWidget(epochs_group)
        
        # Loss
        loss_group = QGroupBox("Función de Pérdida")
        loss_layout = QFormLayout()
        
        self.fields['training.loss.type'] = ChoiceField(
            "Loss Type",
            ['multi_similarity', 'triplet', 'arcface', 'hybrid_proxy_ms', 'sliced_ms', 'sliced_proxy'],
            default='sliced_ms',
            tooltip="Tipo de función de pérdida. sliced_ms requiere classes_per_batch >= 2."
        )
        loss_layout.addRow("Type:", self.fields['training.loss.type'].widget)
        
        self.fields['training.loss.params.alpha'] = FloatField(
            "Alpha",
            default=2.0, min_val=0.1, max_val=10.0, decimals=2, step=0.1,
            tooltip="Peso de pares positivos (Multi-Similarity)"
        )
        loss_layout.addRow("Alpha:", self.fields['training.loss.params.alpha'].widget)
        
        self.fields['training.loss.params.beta'] = FloatField(
            "Beta",
            default=50.0, min_val=1.0, max_val=100.0, decimals=1, step=1.0,
            tooltip="Peso de pares negativos (Multi-Similarity)"
        )
        loss_layout.addRow("Beta:", self.fields['training.loss.params.beta'].widget)

        self.fields['training.loss.params.base_margin'] = FloatField(
            "Base Margin",
            default=0.5, min_val=0.1, max_val=2.0, decimals=2, step=0.1,
            tooltip="Margen base para Multi-Similarity Loss (y LDM)"
        )
        loss_layout.addRow("Base Margin:", self.fields['training.loss.params.base_margin'].widget)

        self.fields['training.loss.params.margin'] = FloatField(
            "ArcFace Margin",
            default=28.6, min_val=0.0, max_val=180.0, decimals=1, step=0.1,
            tooltip="Margen angular ArcFace (grados). Para forzar uso de dimensiones, valores altos (90-120) son efectivos."
        )
        loss_layout.addRow("ArcFace Margin:", self.fields['training.loss.params.margin'].widget)

        self.fields['training.loss.params.scale'] = FloatField(
            "ArcFace Scale",
            default=64.0, min_val=10.0, max_val=100.0, decimals=1, step=1.0,
            tooltip="Escala para ArcFace logits"
        )
        loss_layout.addRow("ArcFace Scale:", self.fields['training.loss.params.scale'].widget)

        self.fields['training.loss.params.proxy_scale'] = FloatField(
            "Proxy Scale",
            default=30.0, min_val=10.0, max_val=100.0, decimals=1, step=1.0,
            tooltip="Escala para Proxy logits (usado en SlicedProxy)"
        )
        loss_layout.addRow("Proxy Scale:", self.fields['training.loss.params.proxy_scale'].widget)

        self.fields['training.loss.params.proxy_margin'] = FloatField(
            "Proxy Margin",
            default=0.2, min_val=0.0, max_val=1.0, decimals=2, step=0.05,
            tooltip="Margen aditivo para Proxies (usado en SlicedProxy)"
        )
        loss_layout.addRow("Proxy Margin:", self.fields['training.loss.params.proxy_margin'].widget)

        self.fields['training.loss.params.num_slices'] = IntField(
            "Num Slices",
            default=4, min_val=2, max_val=16,
            tooltip="Número de subespacios para Sliced MS Loss (Spectral Embedding Expansion)"
        )
        loss_layout.addRow("Slices (SlicedMS):", self.fields['training.loss.params.num_slices'].widget)

        self.fields['training.gradient_accumulation_steps'] = IntField(
            "Gradient Accumulation",
            default=1, min_val=1, max_val=32,
            tooltip="Pasos de acumulación de gradientes. Batch efectivo = batch_size × este valor."
        )
        loss_layout.addRow("Grad Accum Steps:", self.fields['training.gradient_accumulation_steps'].widget)
        
        loss_group.setLayout(loss_layout)
        layout.addWidget(loss_group)

        metric_group = QGroupBox("Métrica de Validación")
        metric_layout = QFormLayout()

        self.fields['training.validation_metric'] = ChoiceField(
            "Validation Metric",
            ['mAP@R', 'recall_at_1', 'F1_macro', 'val_loss'],
            default='mAP@R',
            tooltip="Métrica MLRC para selección de checkpoint (F-07 ablación)"
        )
        metric_layout.addRow("Métrica:", self.fields['training.validation_metric'].widget)

        metric_group.setLayout(metric_layout)
        layout.addWidget(metric_group)

        mining_group = QGroupBox("Hard Mining (ArcFace OHEM)")
        mining_layout = QFormLayout()

        self.fields['training.hard_mining.enabled'] = BoolField(
            "Hard Mining",
            default=True,
            tooltip="Online hard example mining en ArcFace"
        )
        mining_layout.addRow("Habilitado:", self.fields['training.hard_mining.enabled'].widget)

        self.fields['training.hard_mining.ratio'] = FloatField(
            "Ratio",
            default=0.3, min_val=0.0, max_val=1.0, decimals=2, step=0.05,
            tooltip="Fracción de ejemplos difíciles por batch"
        )
        mining_layout.addRow("Ratio:", self.fields['training.hard_mining.ratio'].widget)

        self.fields['training.hard_mining.min_samples'] = IntField(
            "Min Samples",
            default=32, min_val=1, max_val=512,
            tooltip="Mínimo de muestras para activar mining"
        )
        mining_layout.addRow("Min Samples:", self.fields['training.hard_mining.min_samples'].widget)

        self.fields['training.hard_mining.warmup_epochs'] = IntField(
            "Mining Warmup",
            default=0, min_val=0, max_val=50,
            tooltip="Épocas antes de activar hard mining"
        )
        mining_layout.addRow("Warmup épocas:", self.fields['training.hard_mining.warmup_epochs'].widget)

        self.fields['training.hard_mining.adaptive'] = BoolField(
            "Adaptive",
            default=False,
            tooltip="Ajustar ratio de mining dinámicamente"
        )
        mining_layout.addRow("Adaptativo:", self.fields['training.hard_mining.adaptive'].widget)

        mining_group.setLayout(mining_layout)
        layout.addWidget(mining_group)

        triplet_mining_group = QGroupBox("Triplet Mining (legacy)")
        triplet_mining_layout = QFormLayout()

        self.fields['training.mining.enabled'] = BoolField(
            "Mining Enabled",
            default=False,
            tooltip="Mining para triplet/multi-similarity (no ArcFace)"
        )
        triplet_mining_layout.addRow("Habilitado:", self.fields['training.mining.enabled'].widget)

        self.fields['training.mining.type'] = ChoiceField(
            "Mining Type",
            ['semihard', 'hard', 'easy'],
            default='semihard',
            tooltip="Estrategia de mining triplet"
        )
        triplet_mining_layout.addRow("Tipo:", self.fields['training.mining.type'].widget)

        self.fields['training.mining.margin'] = FloatField(
            "Mining Margin",
            default=0.2, min_val=0.0, max_val=2.0, decimals=2, step=0.05,
            tooltip="Margen para triplet mining"
        )
        triplet_mining_layout.addRow("Margen:", self.fields['training.mining.margin'].widget)

        triplet_mining_group.setLayout(triplet_mining_layout)
        layout.addWidget(triplet_mining_group)

        # Academic Regularizers
        reg_group = QGroupBox("Regularizadores Académicos (128D)")
        reg_layout = QFormLayout()

        self.fields['sota.amplitude_penalty.enabled'] = BoolField(
            "Amplitude Penalty",
            default=False,
            tooltip="Penalizar amplitudes extremas en embeddings (antes de LN)"
        )
        reg_layout.addRow("Amplitude Penalty:", self.fields['sota.amplitude_penalty.enabled'].widget)

        self.fields['sota.amplitude_penalty.lambda_weight'] = FloatField(
            "Amplitude Lambda",
            default=0.1, min_val=0.0, max_val=10.0, decimals=2, step=0.1,
            tooltip="Peso de la pérdida de amplitud"
        )
        reg_layout.addRow("Amplitude Lambda:", self.fields['sota.amplitude_penalty.lambda_weight'].widget)

        self.fields['sota.amplitude_penalty.penalty_type'] = ChoiceField(
            "Amplitude Type",
            ['clamping', 'huber', 'l2'],
            default='clamping',
            tooltip="Tipo de penalización de amplitud"
        )
        reg_layout.addRow("Amplitude Type:", self.fields['sota.amplitude_penalty.penalty_type'].widget)

        self.fields['sota.amplitude_penalty.margin'] = FloatField(
            "Amplitude Margin",
            default=0.5, min_val=0.0, max_val=5.0, decimals=2, step=0.1,
            tooltip="Margen de amplitud segura"
        )
        reg_layout.addRow("Amplitude Margin:", self.fields['sota.amplitude_penalty.margin'].widget)


        self.fields['sota.uniformity.enabled'] = BoolField(
            "Uniformity Enabled",
            default=False,
            tooltip="Activar Uniformidad Hiperesférica (Wang & Isola, 2020) para maximizar ortogonalidad"
        )
        reg_layout.addRow("Uniformidad:", self.fields['sota.uniformity.enabled'].widget)

        self.fields['sota.uniformity.lambda'] = FloatField(
            "Uniformity Lambda",
            default=1.0, min_val=0.0, max_val=10.0, decimals=2, step=0.1,
            tooltip="Peso de la pérdida de uniformidad"
        )
        reg_layout.addRow("Uniformidad (Peso):", self.fields['sota.uniformity.lambda'].widget)

        self.fields['sota.uniformity.t'] = FloatField(
            "Uniformity t (RBF)",
            default=2.0, min_val=0.1, max_val=10.0, decimals=1, step=0.1,
            tooltip="Parámetro t del núcleo Gaussiano RBF"
        )
        reg_layout.addRow("Uniformidad (t):", self.fields['sota.uniformity.t'].widget)

        self.fields['sota.decorrelation.enabled'] = BoolField(
            "Decorrelation Enabled",
            default=False,
            tooltip="Activar Whitening / Decorrelación (VICReg) para forzar uso de las 128D"
        )
        reg_layout.addRow("Decorrelación:", self.fields['sota.decorrelation.enabled'].widget)

        self.fields['sota.decorrelation.lambda'] = FloatField(
            "Decorrelation Lambda",
            default=0.5, min_val=0.0, max_val=10.0, decimals=2, step=0.1,
            tooltip="Peso de la pérdida de decorrelación de covarianza"
        )
        reg_layout.addRow("Decorrelación (Peso):", self.fields['sota.decorrelation.lambda'].widget)

        self.fields['sota.decorrelation.target_variance'] = FloatField(
            "Target Variance",
            default=1.0, min_val=0.1, max_val=10.0, decimals=2, step=0.1,
            tooltip="Varianza objetivo (target) para las dimensiones"
        )
        reg_layout.addRow("Varianza Target:", self.fields['sota.decorrelation.target_variance'].widget)

        self.fields['sota.disentanglement.enabled'] = BoolField(
            "Disentanglement Enabled",
            default=False,
            tooltip="Activar Desentrelazamiento (MIM) entre Forma y Textura"
        )
        reg_layout.addRow("Desentrelazamiento:", self.fields['sota.disentanglement.enabled'].widget)

        self.fields['sota.disentanglement.lambda'] = FloatField(
            "Disentanglement Lambda",
            default=5.0, min_val=0.0, max_val=50.0, decimals=2, step=0.5,
            tooltip="Peso de la penalización de desentrelazamiento"
        )
        reg_layout.addRow("Desentrelazamiento (Peso):", self.fields['sota.disentanglement.lambda'].widget)

        self.fields['sota.self_distillation.enabled'] = BoolField(
            "Self-Distillation",
            default=False,
            tooltip="Desactivar con slicing (compite con expansión espectral)"
        )
        reg_layout.addRow("Self-Distillation:", self.fields['sota.self_distillation.enabled'].widget)

        self.fields['sota.anti_collapse.enabled'] = BoolField(
            "Anti-Collapse",
            default=False,
            tooltip="Regularizador de uniformidad en hipersfera (IEEE Trans. Multimedia 2024)"
        )
        reg_layout.addRow("Anti-Collapse:", self.fields['sota.anti_collapse.enabled'].widget)

        self.fields['sota.anti_collapse.lambda'] = FloatField(
            "Anti-Collapse Lambda",
            default=0.1, min_val=0.0, max_val=5.0, decimals=2, step=0.05,
            tooltip="Peso del término anti-colapso"
        )
        reg_layout.addRow("Anti-Collapse λ:", self.fields['sota.anti_collapse.lambda'].widget)

        self.fields['sota.anti_collapse.t'] = FloatField(
            "Anti-Collapse t",
            default=2.0, min_val=0.1, max_val=10.0, decimals=1, step=0.1,
            tooltip="Temperatura del kernel RBF de uniformidad"
        )
        reg_layout.addRow("Anti-Collapse t:", self.fields['sota.anti_collapse.t'].widget)

        reg_group.setLayout(reg_layout)
        layout.addWidget(reg_group)
        
        # Optimizer
        opt_group = QGroupBox("Optimizador")
        opt_layout = QFormLayout()
        
        self.fields['training.optimizer'] = ChoiceField(
            "Optimizer",
            ['adam', 'adamw', 'sgd'],
            tooltip="Tipo de optimizador"
        )
        opt_layout.addRow("Type:", self.fields['training.optimizer'].widget)
        
        self.fields['training.learning_rate'] = FloatField(
            "Learning Rate",
            default=0.0001, min_val=0.00001, max_val=0.1, decimals=6, step=0.00001,
            tooltip="Tasa de aprendizaje"
        )
        opt_layout.addRow("Learning Rate:", self.fields['training.learning_rate'].widget)
        
        self.fields['training.weight_decay'] = FloatField(
            "Weight Decay",
            default=0.0001, min_val=0.0, max_val=0.01, decimals=6, step=0.00001,
            tooltip="Regularización L2"
        )
        opt_layout.addRow("Weight Decay:", self.fields['training.weight_decay'].widget)
        
        self.fields['training.backbone_lr_factor'] = FloatField(
            "Backbone LR Factor",
            default=0.1, min_val=0.001, max_val=1.0, decimals=3, step=0.01,
            tooltip="Factor multiplicativo para LR del backbone (0.1 = 10x menor que projection head)"
        )
        opt_layout.addRow("Backbone LR Factor:", self.fields['training.backbone_lr_factor'].widget)

        self.fields['training.gradient_clip_norm'] = FloatField(
            "Gradient Clip Norm",
            default=5.0, min_val=0.0, max_val=50.0, decimals=1, step=0.5,
            tooltip="Norma máxima para gradient clipping (0 = desactivado)"
        )
        opt_layout.addRow("Gradient Clip:", self.fields['training.gradient_clip_norm'].widget)

        opt_group.setLayout(opt_layout)
        layout.addWidget(opt_group)
        
        # Scheduler
        sched_group = QGroupBox("Scheduler")
        sched_layout = QFormLayout()
        
        self.fields['training.scheduler'] = ChoiceField(
            "Scheduler",
            ['reduce_on_plateau', 'cosine', 'onecycle'],
            tooltip="Tipo de scheduler"
        )
        sched_layout.addRow("Type:", self.fields['training.scheduler'].widget)

        self.fields['training.scheduler_params.T_max'] = IntField(
            "T_max",
            default=50, min_val=1, max_val=1000,
            tooltip="Período del cosine scheduler (debe coincidir con epochs para evitar restart)"
        )
        sched_layout.addRow("T_max:", self.fields['training.scheduler_params.T_max'].widget)

        self.fields['training.scheduler_params.eta_min'] = FloatField(
            "Eta Min",
            default=0.000001, min_val=0.0, max_val=0.01, decimals=7, step=0.0000001,
            tooltip="LR mínimo del cosine scheduler"
        )
        sched_layout.addRow("Eta Min:", self.fields['training.scheduler_params.eta_min'].widget)

        sched_group.setLayout(sched_layout)
        layout.addWidget(sched_group)

        # Early Stopping
        es_group = QGroupBox("Early Stopping")
        es_layout = QFormLayout()

        self.fields['training.early_stopping.enabled'] = BoolField(
            "Enabled",
            default=True,
            tooltip="Activar early stopping"
        )
        es_layout.addRow("Activado:", self.fields['training.early_stopping.enabled'].widget)

        self.fields['training.early_stopping.patience'] = IntField(
            "Patience",
            default=10, min_val=1, max_val=100,
            tooltip="Épocas sin mejora antes de detener el entrenamiento"
        )
        es_layout.addRow("Patience:", self.fields['training.early_stopping.patience'].widget)

        self.fields['training.early_stopping.min_delta'] = FloatField(
            "Min Delta",
            default=0.001, min_val=0.0, max_val=0.1, decimals=4, step=0.0001,
            tooltip="Mejora mínima para considerar progreso"
        )
        es_layout.addRow("Min Delta:", self.fields['training.early_stopping.min_delta'].widget)

        self.fields['training.early_stopping.restore_best_weights'] = BoolField(
            "Restore Best",
            default=True,
            tooltip="Restaurar pesos del mejor epoch al detenerse"
        )
        es_layout.addRow("Restore Best:", self.fields['training.early_stopping.restore_best_weights'].widget)

        es_group.setLayout(es_layout)
        layout.addWidget(es_group)

        # Checkpoints
        ckpt_group = QGroupBox("Checkpoints")
        ckpt_layout = QFormLayout()

        self.fields['training.save_every_n_epochs'] = IntField(
            "Save Every N",
            default=10, min_val=1, max_val=100,
            tooltip="Guardar checkpoint cada N épocas"
        )
        ckpt_layout.addRow("Guardar cada:", self.fields['training.save_every_n_epochs'].widget)

        self.fields['training.save_best_only'] = BoolField(
            "Save Best Only",
            default=True,
            tooltip="Solo guardar el mejor modelo (además de checkpoints periódicos)"
        )
        ckpt_layout.addRow("Solo mejor:", self.fields['training.save_best_only'].widget)

        ckpt_group.setLayout(ckpt_layout)
        layout.addWidget(ckpt_group)

        layout.addStretch()
        tab.setLayout(layout)
        self.tabs.addTab(tab, "🚀 Entrenamiento")

    def create_evaluation_tab(self):
        """Tab de evaluación durante entrenamiento"""
        tab = QWidget()
        layout = QVBoxLayout()

        eval_group = QGroupBox("Validación y Métricas")
        eval_layout = QFormLayout()

        self.fields['evaluation.progressive_plot_every_epochs'] = IntField(
            "Progressive Plots",
            default=2, min_val=0, max_val=10,
            tooltip="0 = desactivado; actualiza curvas de entrenamiento cada N épocas (no afecta métricas)"
        )
        eval_layout.addRow("Plots progresivos:", self.fields['evaluation.progressive_plot_every_epochs'].widget)

        self.fields['evaluation.compute_precision_at_k'] = ListField(
            "Precision@K",
            default=[1, 5, 10],
            tooltip="Valores k para Precision@k"
        )
        eval_layout.addRow("Precision@K:", self.fields['evaluation.compute_precision_at_k'].widget)

        self.fields['evaluation.visualize_embeddings'] = BoolField(
            "Visualize Embeddings",
            default=True,
            tooltip="Generar visualización de embeddings"
        )
        eval_layout.addRow("Visualizar:", self.fields['evaluation.visualize_embeddings'].widget)

        self.fields['evaluation.embedding_visualization_method'] = ChoiceField(
            "Viz Method",
            ['tsne', 'umap', 'pca'],
            default='tsne',
            tooltip="Método de reducción dimensional para plots"
        )
        eval_layout.addRow("Método viz:", self.fields['evaluation.embedding_visualization_method'].widget)

        eval_group.setLayout(eval_layout)
        layout.addWidget(eval_group)

        abl_group = QGroupBox("Ablaciones (automáticas tras entrenar)")
        abl_layout = QFormLayout()

        self.fields['evaluation.ablation.enabled'] = BoolField(
            "Ablation Enabled",
            default=False,
            tooltip="Tras el entrenamiento principal, lanzar variantes de ablación configuradas "
                    "(resultados en runs/<run>/ablations/)"
        )
        abl_layout.addRow("Habilitar:", self.fields['evaluation.ablation.enabled'].widget)

        self.fields['evaluation.ablation.when'] = ChoiceField(
            "When",
            ['after_training'],
            default='after_training',
            tooltip="Momento de ejecución (solo after_training por ahora)"
        )
        abl_layout.addRow("Cuándo:", self.fields['evaluation.ablation.when'].widget)

        self.fields['evaluation.ablation.skip_baseline_variant'] = BoolField(
            "Skip Baseline",
            default=True,
            tooltip="Omitir variantes idénticas a la config actual (evita re-entrenar el baseline)"
        )
        abl_layout.addRow("Saltar baseline:", self.fields['evaluation.ablation.skip_baseline_variant'].widget)

        self.fields['evaluation.ablation.on_failure'] = ChoiceField(
            "On Failure",
            ['continue', 'abort'],
            default='continue',
            tooltip="continue: el run principal sigue OK aunque falle una variante; "
                    "abort: detener ablaciones al primer fallo"
        )
        abl_layout.addRow("Si falla variante:", self.fields['evaluation.ablation.on_failure'].widget)

        self.fields['evaluation.ablation.epochs'] = IntField(
            "Ablation Epochs",
            default=0, min_val=0, max_val=500,
            tooltip="0 = automático (mismas épocas que training.epochs); ≥1 = override por variante"
        )
        self.fields['evaluation.ablation.epochs'].widget.setSpecialValueText("auto")
        abl_layout.addRow("Épocas ablación:", self.fields['evaluation.ablation.epochs'].widget)

        matrix_labels = {
            'pooling': 'F-01 Pooling',
            'slices': 'F-02 Slices',
            'layout': 'F-03 Layout',
            'masked': 'F-04 Masked',
            'augment': 'F-05 Augment',
            'freeze': 'F-06 Freeze',
            'validation_metric': 'F-07 Val metric',
        }
        for key, label in matrix_labels.items():
            field_key = f'evaluation.ablation.matrices.{key}'
            self.fields[field_key] = BoolField(
                label,
                default=False,
                tooltip=f"Incluir matriz {label} en el estudio automático"
            )
            abl_layout.addRow(f"{label}:", self.fields[field_key].widget)

        self.ablation_hint_label = QLabel(
            "Las variantes usan los mismos campos que Modelo, Entrenamiento, Grano y Augmentación. "
            "Se ejecutan automáticamente al pulsar Entrenar (tras el run principal)."
        )
        self.ablation_hint_label.setWordWrap(True)
        self.ablation_hint_label.setStyleSheet("color: #666; font-size: 11px; padding: 4px;")
        abl_layout.addRow(self.ablation_hint_label)

        abl_group.setLayout(abl_layout)
        layout.addWidget(abl_group)

        repro_group = QGroupBox("Reproducibilidad")
        repro_layout = QFormLayout()

        self.fields['reproducibility.seed'] = IntField(
            "Seed",
            default=42, min_val=0, max_val=999999,
            tooltip="Semilla aleatoria global"
        )
        repro_layout.addRow("Seed:", self.fields['reproducibility.seed'].widget)

        self.fields['reproducibility.deterministic'] = BoolField(
            "Deterministic",
            default=False,
            tooltip="Modo determinista (más lento)"
        )
        repro_layout.addRow("Determinista:", self.fields['reproducibility.deterministic'].widget)

        self.fields['reproducibility.benchmark'] = BoolField(
            "Benchmark",
            default=True,
            tooltip="cudnn.benchmark para convoluciones"
        )
        repro_layout.addRow("cuDNN benchmark:", self.fields['reproducibility.benchmark'].widget)

        repro_group.setLayout(repro_layout)
        layout.addWidget(repro_group)

        layout.addStretch()
        tab.setLayout(layout)
        self.tabs.addTab(tab, "📊 Evaluación")
    
    def create_inference_tab(self):
        """Tab de inferencia"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # FAISS
        faiss_group = QGroupBox("Configuración FAISS")
        faiss_layout = QFormLayout()
        
        self.fields['inference.index_type'] = ChoiceField(
            "Index Type",
            ['flat', 'ivf', 'ivfpq'],
            tooltip="Tipo de índice FAISS"
        )
        faiss_layout.addRow("Index Type:", self.fields['inference.index_type'].widget)
        
        self.fields['inference.metric_type'] = ChoiceField(
            "Metric",
            ['inner_product', 'l2'],
            tooltip="Métrica de distancia"
        )
        faiss_layout.addRow("Metric:", self.fields['inference.metric_type'].widget)
        
        self.fields['inference.k_neighbors'] = IntField(
            "K Neighbors",
            default=5, min_val=1, max_val=100,
            tooltip="Número de vecinos a buscar"
        )
        faiss_layout.addRow("K Neighbors:", self.fields['inference.k_neighbors'].widget)
        
        self.fields['inference.use_gpu'] = BoolField(
            "Use GPU",
            tooltip="Usar GPU para FAISS"
        )
        faiss_layout.addRow("Use GPU:", self.fields['inference.use_gpu'].widget)
        
        faiss_group.setLayout(faiss_layout)
        layout.addWidget(faiss_group)
        
        # Thresholds
        thresh_group = QGroupBox("Umbrales")
        thresh_layout = QFormLayout()
        
        self.fields['inference.similarity_threshold'] = FloatField(
            "Similarity",
            default=0.6, min_val=0.0, max_val=1.0, decimals=2, step=0.05,
            tooltip="Umbral de similitud"
        )
        thresh_layout.addRow("Similarity:", self.fields['inference.similarity_threshold'].widget)
        
        self.fields['inference.confidence_threshold'] = FloatField(
            "Confidence",
            default=0.8, min_val=0.0, max_val=1.0, decimals=2, step=0.05,
            tooltip="Umbral de confianza"
        )
        thresh_layout.addRow("Confidence:", self.fields['inference.confidence_threshold'].widget)
        
        thresh_group.setLayout(thresh_layout)
        layout.addWidget(thresh_group)
        
        # Device
        device_group = QGroupBox("Dispositivo")
        device_layout = QFormLayout()
        
        self.fields['inference.device'] = ChoiceField(
            "Device",
            ['cuda', 'cpu'],
            tooltip="Dispositivo para inferencia"
        )
        device_layout.addRow("Device:", self.fields['inference.device'].widget)
        
        device_group.setLayout(device_layout)
        layout.addWidget(device_group)
        
        layout.addStretch()
        tab.setLayout(layout)
        self.tabs.addTab(tab, "🔍 Inferencia")
    
    def create_paths_tab(self):
        """Tab de rutas"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        paths_group = QGroupBox("Rutas del Sistema")
        paths_layout = QFormLayout()
        
        self.fields['paths.model_save_dir'] = PathField(
            "Models",
            tooltip="Directorio para modelos",
            is_dir=True
        )
        paths_layout.addRow("Models:", self.fields['paths.model_save_dir'].widget_container)
        
        self.fields['paths.checkpoint_dir'] = PathField(
            "Checkpoints",
            tooltip="Directorio para checkpoints",
            is_dir=True
        )
        paths_layout.addRow("Checkpoints:", self.fields['paths.checkpoint_dir'].widget_container)
        
        self.fields['paths.log_dir'] = PathField(
            "Logs",
            tooltip="Directorio para logs",
            is_dir=True
        )
        paths_layout.addRow("Logs:", self.fields['paths.log_dir'].widget_container)
        
        self.fields['paths.experiment_dir'] = PathField(
            "Experiments",
            tooltip="Directorio para experimentos",
            is_dir=True
        )
        paths_layout.addRow("Experiments:", self.fields['paths.experiment_dir'].widget_container)
        
        paths_group.setLayout(paths_layout)
        layout.addWidget(paths_group)
        
        layout.addStretch()
        tab.setLayout(layout)
        self.tabs.addTab(tab, "📁 Rutas")
    
    def create_hardware_tab(self):
        """Tab de hardware"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # GPU
        gpu_group = QGroupBox("GPU")
        gpu_layout = QFormLayout()
        
        self.fields['hardware.use_gpu'] = BoolField(
            "Use GPU",
            tooltip="Habilitar GPU"
        )
        gpu_layout.addRow("Use GPU:", self.fields['hardware.use_gpu'].widget)
        
        self.fields['hardware.gpu_ids'] = ListField(
            "GPU IDs",
            tooltip="IDs de GPUs a usar (ej: 0, 1)"
        )
        gpu_layout.addRow("GPU IDs:", self.fields['hardware.gpu_ids'].widget)
        
        self.fields['hardware.use_amp'] = BoolField(
            "Mixed Precision",
            tooltip="Usar Automatic Mixed Precision"
        )
        gpu_layout.addRow("Mixed Precision:", self.fields['hardware.use_amp'].widget)

        self.fields['hardware.faiss_gpu'] = BoolField(
            "FAISS GPU",
            default=True,
            tooltip="Usar GPU para índices FAISS en evaluación/inferencia"
        )
        gpu_layout.addRow("FAISS GPU:", self.fields['hardware.faiss_gpu'].widget)
        
        gpu_group.setLayout(gpu_layout)
        layout.addWidget(gpu_group)
        
        # Distributed
        dist_group = QGroupBox("Entrenamiento Distribuido")
        dist_layout = QFormLayout()
        
        self.fields['hardware.distributed'] = BoolField(
            "Distributed",
            tooltip="Habilitar entrenamiento distribuido"
        )
        dist_layout.addRow("Distributed:", self.fields['hardware.distributed'].widget)
        
        self.fields['hardware.world_size'] = IntField(
            "World Size",
            default=1, min_val=1, max_val=16,
            tooltip="Número de procesos"
        )
        dist_layout.addRow("World Size:", self.fields['hardware.world_size'].widget)
        
        dist_group.setLayout(dist_layout)
        layout.addWidget(dist_group)
        
        layout.addStretch()
        tab.setLayout(layout)
        self.tabs.addTab(tab, "⚡ Hardware")
    
    def _wire_ablation_controls(self):
        """Habilita/deshabilita controles de ablación según evaluation.ablation.enabled."""
        matrix_field_keys = [
            f"evaluation.ablation.matrices.{k}"
            for k in (
                "pooling", "slices", "layout", "masked",
                "augment", "freeze", "validation_metric",
            )
        ]
        dependent_keys = matrix_field_keys + [
            "evaluation.ablation.when",
            "evaluation.ablation.skip_baseline_variant",
            "evaluation.ablation.on_failure",
            "evaluation.ablation.epochs",
        ]

        def _refresh_ablation_enabled_state():
            enabled = self.fields["evaluation.ablation.enabled"].get_value()
            for key in dependent_keys:
                if key in self.fields:
                    self.fields[key].widget.setEnabled(enabled)

        self._refresh_ablation_enabled_state = _refresh_ablation_enabled_state
        self.fields["evaluation.ablation.enabled"].widget.stateChanged.connect(
            lambda _state: _refresh_ablation_enabled_state()
        )
        self._connect_ablation_preview()

    @staticmethod
    def _normalize_field_value(full_key: str, val):
        if full_key == "evaluation.ablation.epochs" and val == 0:
            return None
        return val

    def get_draft_config(self) -> dict:
        """Config actual según los campos de la UI (puede diferir del YAML en disco)."""
        import copy

        draft = copy.deepcopy(self.config_data)
        for full_key, field in self.fields.items():
            val = self._normalize_field_value(full_key, field.get_value())
            set_nested_value(draft, full_key, val)
        return draft

    def _connect_ablation_preview(self):
        """Actualiza resumen en pestaña Entrenamiento al editar bloque de ablación."""
        preview_keys = [
            "evaluation.ablation.enabled",
            "evaluation.ablation.when",
            "evaluation.ablation.skip_baseline_variant",
            "evaluation.ablation.on_failure",
            "evaluation.ablation.epochs",
        ] + [
            f"evaluation.ablation.matrices.{k}"
            for k in (
                "pooling", "slices", "layout", "masked",
                "augment", "freeze", "validation_metric",
            )
        ]

        def _preview_ablation_summary():
            if not self.parent_window or not hasattr(self.parent_window, "training_tab"):
                return
            from src.utils.ablation_matrix import (
                ensure_ablation_defaults,
                format_ablation_startup_summary,
            )

            draft = self.get_draft_config()
            ensure_ablation_defaults(draft)
            summary = format_ablation_startup_summary(draft)
            if self._dirty:
                summary += " (vista previa — guarde 💾)"
            self.parent_window.training_tab.ablation_summary_label.setText(summary)

        for key in preview_keys:
            if key in self.fields:
                self.fields[key].valueChanged.connect(_preview_ablation_summary)

    def _connect_dirty_tracking(self):
        """Conecta valueChanged de todos los campos para detectar ediciones sin guardar."""
        for field in self.fields.values():
            field.valueChanged.connect(self._mark_dirty)
    
    def _mark_dirty(self):
        if self._suppress_dirty:
            return
        self._dirty = True
        self.status_label.setText("● Cambios sin guardar")
        self.status_label.setStyleSheet("color: #ffa94d; padding: 5px; font-weight: bold;")
    
    def is_dirty(self) -> bool:
        """True si hay cambios en la UI que no se han guardado en disco."""
        return self._dirty
    
    def get_config_path_str(self) -> str:
        return str(self.config_path.resolve())
    
    def load_config(self):
        """Carga configuración desde archivo"""
        try:
            self._suppress_dirty = True
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config_data = yaml.safe_load(f) or {}

            from src.utils.ablation_matrix import ensure_ablation_defaults

            migrated = ensure_ablation_defaults(self.config_data)
            
            # Cargar valores en campos
            self.load_values_from_dict(self.config_data)
            self._sync_resolution_fields()
            if hasattr(self, "_refresh_ablation_enabled_state"):
                self._refresh_ablation_enabled_state()
            self._dirty = False
            
            self.status_label.setText(f"✅ Configuración cargada: {datetime.now().strftime('%H:%M:%S')}")
            self.status_label.setStyleSheet("color: #4ec9b0; padding: 5px;")
            
            if migrated and self.parent_window and hasattr(self.parent_window, 'log_widget'):
                self.parent_window.log_widget.append_log(
                    "Bloque evaluation.ablation aplicado desde defaults — guarde para persistir",
                    "WARNING",
                )
            if self.parent_window and hasattr(self.parent_window, 'log_widget'):
                self.parent_window.log_widget.append_log("Configuración cargada", "SUCCESS")
            if self.parent_window and hasattr(self.parent_window, "training_tab"):
                self.parent_window.training_tab.refresh_ablation_summary(
                    str(self.config_path)
                )
        
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error cargando configuración:\n{str(e)}")
            self.status_label.setText(f"❌ Error: {str(e)}")
            self.status_label.setStyleSheet("color: #f48771; padding: 5px;")
        finally:
            self._suppress_dirty = False
    
    def load_values_from_dict(self, data, prefix=""):
        """Carga valores recursivamente desde diccionario"""
        for key, value in data.items():
            full_key = f"{prefix}.{key}" if prefix else key
            
            if isinstance(value, dict):
                self.load_values_from_dict(value, full_key)
            elif full_key in self.fields:
                if value is None and full_key == "evaluation.ablation.epochs":
                    self.fields[full_key].set_value(0)
                else:
                    self.fields[full_key].set_value(value)
    
    def save_config(self):
        """Guarda configuración en archivo"""
        # Validar primero
        if not self.validate_all():
            QMessageBox.warning(
                self,
                "Validación Fallida",
                "Hay campos con valores inválidos. Por favor corríjalos antes de guardar."
            )
            return
        
        # Confirmar
        reply = QMessageBox.question(
            self,
            "Confirmar Guardado",
            f"¿Guardar cambios en {self.config_path.name}?\n\nSe creará un backup automático.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes
        )
        
        if reply != QMessageBox.Yes:
            return
        
        try:
            # Crear backup
            backup_path = self.config_path.with_suffix('.yaml.backup')
            shutil.copy2(self.config_path, backup_path)
            
            self._sync_resolution_fields()
            # Actualizar diccionario con valores de campos (incluye claves nuevas anidadas)
            for full_key, field in self.fields.items():
                val = self._normalize_field_value(full_key, field.get_value())
                set_nested_value(self.config_data, full_key, val)
            
            # Guardar
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config_data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
            
            self.status_label.setText(f"✅ Guardado exitosamente: {datetime.now().strftime('%H:%M:%S')}")
            self.status_label.setStyleSheet("color: #4ec9b0; padding: 5px;")
            
            if self.parent_window:
                self.parent_window.log_widget.append_log(
                    f"Configuración guardada en {self.config_path.name}",
                    "SUCCESS"
                )
            
            QMessageBox.information(
                self,
                "Éxito",
                f"Configuración guardada exitosamente.\n\nBackup creado: {backup_path.name}"
            )
            
            self._dirty = False
            saved_path = str(self.config_path.resolve())
            self.configChanged.emit()
            self.configSaved.emit(saved_path)
            if self.parent_window and hasattr(self.parent_window, "training_tab"):
                self.parent_window.training_tab.refresh_ablation_summary(saved_path)
        
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error guardando configuración:\n{str(e)}")
            self.status_label.setText(f"❌ Error: {str(e)}")
            self.status_label.setStyleSheet("color: #f48771; padding: 5px;")
    
    def update_dict_from_fields(self, data, prefix=""):
        """Actualiza diccionario recursivamente desde campos"""
        for key in list(data.keys()):
            full_key = f"{prefix}.{key}" if prefix else key
            
            if isinstance(data[key], dict):
                self.update_dict_from_fields(data[key], full_key)
            elif full_key in self.fields:
                data[key] = self.fields[full_key].get_value()
    
    def validate_all(self):
        """Valida todos los campos"""
        all_valid = True
        
        for key, field in self.fields.items():
            if not field.validate():
                all_valid = False
                field.mark_invalid()
            else:
                field.mark_valid()
        
        if all_valid:
            from src.utils.config_validator import validate_config
            draft = self.get_draft_config()
            pipeline = validate_config(draft, str(self.config_path))
            if not pipeline["valid"]:
                all_valid = False
                for err in pipeline["errors"]:
                    self.status_label.setText(f"❌ {err}")
                    self.status_label.setStyleSheet("color: #f48771; padding: 5px;")
                    break
            elif pipeline["warnings"]:
                self.status_label.setText(
                    f"✅ Campos OK — {pipeline['pipeline_summary']} "
                    f"({len(pipeline['warnings'])} aviso(s))"
                )
                self.status_label.setStyleSheet("color: #4ec9b0; padding: 5px;")
            else:
                self.status_label.setText(
                    f"✅ Válido — {pipeline['pipeline_summary']}"
                )
                self.status_label.setStyleSheet("color: #4ec9b0; padding: 5px;")
        else:
            self.status_label.setText("❌ Hay campos inválidos")
            self.status_label.setStyleSheet("color: #f48771; padding: 5px;")
        
        return all_valid
    
    def reset_to_defaults(self):
        """Restaura valores por defecto"""
        reply = QMessageBox.question(
            self,
            "Confirmar Reset",
            "¿Restaurar todos los valores a sus valores por defecto?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.load_config()
    
    def _on_image_size_changed(self, _value=None):
        """Propaga data.image_size a resize y crop (única fuente de verdad)."""
        self._sync_resolution_fields()

    def _sync_resolution_fields(self):
        """Alinea resize_size y grain crop_size con data.image_size."""
        if 'data.image_size' not in self.fields:
            return
        size = int(self.fields['data.image_size'].get_value())
        if 'data.resize_size' in self.fields:
            if self.fields['data.resize_size'].get_value() != size:
                self.fields['data.resize_size'].set_value(size)
        if 'grain_detection.crop_size' in self.fields:
            if self.fields['grain_detection.crop_size'].get_value() != size:
                self.fields['grain_detection.crop_size'].set_value(size)

    def on_backbone_changed(self, backbone_name):
        """
        Auto-configura parámetros al cambiar el backbone.
        Actualiza resolución, batch_size si es necesario, y otros parámetros.
        """
        if not backbone_name:
            return
        
        changes = []  # Lista de (parámetro, valor_anterior, valor_nuevo)
        
        try:
            # 1. Obtener resolución óptima para el backbone
            resolved_size = get_input_size_for_backbone(backbone_name)
            
            # 2. Actualizar data.image_size
            if 'data.image_size' in self.fields:
                old_size = self.fields['data.image_size'].get_value()
                if old_size != resolved_size:
                    self.fields['data.image_size'].set_value(resolved_size)
                    changes.append(('data.image_size', old_size, resolved_size))
            
            # 3. Actualizar augmentation.image_size si existe
            if 'augmentation.image_size' in self.fields:
                old_aug_size = self.fields['augmentation.image_size'].get_value()
                if old_aug_size != resolved_size:
                    self.fields['augmentation.image_size'].set_value(resolved_size)
                    changes.append(('augmentation.image_size', old_aug_size, resolved_size))
            
            # 4. Ajustar samples_per_class según VRAM del modelo
            # Modelos grandes necesitan menos samples por batch
            samples_per_class_map = {
                'efficientnet_b4': 5,          # Modelo grande
                'dinov2_vitb14': 7,            # Modelo mediano-grande
                'dinov2_vitb14_reg': 7,        # Modelo mediano-grande
                'convnext_v2_tiny': 7,         # Modelo mediano
                'swin_tiny_patch4_window7_224': 7,  # Modelo mediano
                'dinov2_vits14': 14,           # Modelo pequeño (default)
                'dinov2_vits14_reg': 14,       # Modelo pequeño
                'resnet50': 14,                # CNN clásico, eficiente
                'resnet18': 14,                # CNN pequeño
                'deit_small_patch16_224': 14,  # ViT pequeño
            }
            
            if 'data.samples_per_class' in self.fields:
                old_samples = self.fields['data.samples_per_class'].get_value()
                recommended_samples = samples_per_class_map.get(backbone_name, 14)
                if old_samples != recommended_samples:
                    self.fields['data.samples_per_class'].set_value(recommended_samples)
                    changes.append(('data.samples_per_class', old_samples, recommended_samples))
            
            # 5. Recalcular batch_size = classes_per_batch × samples_per_class
            if 'data.classes_per_batch' in self.fields and 'data.samples_per_class' in self.fields:
                classes_per_batch = self.fields['data.classes_per_batch'].get_value()
                samples_per_class = self.fields['data.samples_per_class'].get_value()
                calculated_batch = classes_per_batch * samples_per_class
                
                if 'data.batch_size' in self.fields:
                    old_batch = self.fields['data.batch_size'].get_value()
                    if old_batch != calculated_batch:
                        self.fields['data.batch_size'].set_value(calculated_batch)
                        changes.append(('data.batch_size', old_batch, f'{calculated_batch} ({classes_per_batch}×{samples_per_class})'))
            
            # 6. Sincronizar grain_detection.crop_size con image_size
            if 'grain_detection.crop_size' in self.fields:
                old_crop_size = self.fields['grain_detection.crop_size'].get_value()
                if old_crop_size != resolved_size:
                    self.fields['grain_detection.crop_size'].set_value(resolved_size)
                    changes.append(('grain_detection.crop_size', old_crop_size, resolved_size))
            
            # 7. Actualizar experiment.name si existe
            if 'experiment.name' in self.fields:
                old_exp_name = self.fields['experiment.name'].get_value()
                # Generar nombre sugerido: backbone + baseline
                suggested_name = f"{backbone_name}_baseline"
                if old_exp_name != suggested_name:
                    self.fields['experiment.name'].set_value(suggested_name)
                    changes.append(('experiment.name', old_exp_name, suggested_name))
            
            # 8. Actualizar experiment.description si existe
            if 'experiment.description' in self.fields:
                backbone_descriptions = {
                    'dinov2_vits14': 'DINOv2 ViT-S/14 + Sliced MS (metric learning)',
                    'dinov2_vitb14': 'DINOv2 ViT-B/14 + Sliced MS (metric learning)',
                    'resnet50': 'ResNet-50 baseline (CNN clásico)',
                    'convnext_v2_tiny': 'ConvNeXt V2 Tiny baseline (CNN moderno)',
                    'deit_small_patch16_224': 'DeiT-S baseline (ViT supervisado)',
                    'efficientnet_b4': 'EfficientNet-B4 baseline (CNN eficiente)',
                    'swin_tiny_patch4_window7_224': 'Swin Transformer Tiny baseline',
                }
                if backbone_name in backbone_descriptions:
                    old_desc = self.fields['experiment.description'].get_value()
                    new_desc = backbone_descriptions[backbone_name]
                    if old_desc != new_desc:
                        self.fields['experiment.description'].set_value(new_desc)
                        changes.append(('experiment.description', old_desc, new_desc))
            
            # Mostrar diálogo con cambios si hubo alguno
            if changes:
                dialog = ConfigChangesDialog(changes, self)
                dialog.show()
                
                # Log en la aplicación principal
                if self.parent_window and hasattr(self.parent_window, 'log_widget'):
                    self.parent_window.log_widget.append_log(
                        f"Auto-configuración aplicada para '{backbone_name}': {len(changes)} parámetros actualizados",
                        "SUCCESS"
                    )
                
                self.status_label.setText(f"✅ Auto-configurado para {backbone_name} ({len(changes)} cambios)")
                self.status_label.setStyleSheet("color: #4ec9b0; padding: 5px;")
            
        except Exception as e:
            error_msg = f"Error en auto-configuración: {str(e)}"
            QMessageBox.warning(self, "Error", error_msg)
            if self.parent_window and hasattr(self.parent_window, 'log_widget'):
                self.parent_window.log_widget.append_log(error_msg, "ERROR")
