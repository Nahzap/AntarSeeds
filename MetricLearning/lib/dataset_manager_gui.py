"""
Dataset Manager GUI - Interfaz Visual para Gestión de Datasets
Implementa IEEE 9241-16 (Manipulación Directa) para sliders interactivos.

Características:
- Sliders de 2 puntos para balanceo train/val/test
- Visualización en tiempo real de distribución
- Generación de mapas vectoriales
- Validación de integridad
- Estadísticas detalladas
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QSlider, QSpinBox, QTableWidget, QTableWidgetItem,
    QFileDialog, QMessageBox, QProgressBar, QTextEdit, QLineEdit,
    QFormLayout, QHeaderView, QListWidget, QListWidgetItem
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont, QColor
from pathlib import Path
import warnings
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from lib.dataset_manager import DatasetMapManager
from lib.styles import Styles, Colors


class DoubleSlider(QWidget):
    """
    Slider de 2 puntos para definir 3 regiones (train/val/test).
    Implementa IEEE 9241-16 para manipulación directa.
    """
    valuesChanged = pyqtSignal(float, float, float)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Labels de valores - MÁS GRANDES Y CLAROS
        values_layout = QHBoxLayout()
        self.train_label = QLabel("Train: 70%")
        self.train_label.setStyleSheet(f"color: {Colors.SUCCESS}; font-weight: bold; font-size: 14pt;")
        self.val_label = QLabel("Val: 15%")
        self.val_label.setStyleSheet(f"color: {Colors.WARNING}; font-weight: bold; font-size: 14pt;")
        self.test_label = QLabel("Test: 15%")
        self.test_label.setStyleSheet(f"color: {Colors.ERROR}; font-weight: bold; font-size: 14pt;")
        
        values_layout.addWidget(self.train_label)
        values_layout.addStretch()
        values_layout.addWidget(self.val_label)
        values_layout.addStretch()
        values_layout.addWidget(self.test_label)
        layout.addLayout(values_layout)
        
        # Sliders
        sliders_layout = QHBoxLayout()
        
        # Slider 1: Train/Val boundary (0-100)
        self.slider1 = QSlider(Qt.Horizontal)
        self.slider1.setRange(50, 90)  # Train entre 50% y 90%
        self.slider1.setValue(70)
        self.slider1.setTickPosition(QSlider.TicksBelow)
        self.slider1.setTickInterval(10)
        self.slider1.valueChanged.connect(self.update_values)
        
        # Slider 2: Val/Test boundary (0-100)
        self.slider2 = QSlider(Qt.Horizontal)
        self.slider2.setRange(5, 40)  # Val entre 5% y 40%
        self.slider2.setValue(15)
        self.slider2.setTickPosition(QSlider.TicksBelow)
        self.slider2.setTickInterval(5)
        self.slider2.valueChanged.connect(self.update_values)
        
        sliders_layout.addWidget(QLabel("Train:"))
        sliders_layout.addWidget(self.slider1, stretch=3)
        sliders_layout.addWidget(QLabel("Val:"))
        sliders_layout.addWidget(self.slider2, stretch=1)
        
        layout.addLayout(sliders_layout)
        
        # Barra visual de distribución
        self.distribution_bar = QLabel()
        self.distribution_bar.setFixedHeight(30)
        self.update_distribution_bar()
        layout.addWidget(self.distribution_bar)
        
        self.setLayout(layout)
    
    def update_values(self):
        """Actualiza valores y emite señal"""
        train = self.slider1.value() / 100.0
        val = self.slider2.value() / 100.0
        test = 1.0 - train - val
        
        # Validar que test no sea negativo
        if test < 0:
            test = 0
            val = 1.0 - train
            self.slider2.setValue(int(val * 100))
            return
        
        # Actualizar labels
        self.train_label.setText(f"Train: {train*100:.1f}%")
        self.val_label.setText(f"Val: {val*100:.1f}%")
        self.test_label.setText(f"Test: {test*100:.1f}%")
        
        # Actualizar barra visual
        self.update_distribution_bar()
        
        # Emitir señal
        self.valuesChanged.emit(train, val, test)
    
    def update_distribution_bar(self):
        """Actualiza barra visual de distribución"""
        train = self.slider1.value()
        val = self.slider2.value()
        test = 100 - train - val
        
        if test < 0:
            test = 0
        
        # Crear HTML con barras de colores
        html = f"""
        <div style="display: flex; width: 100%; height: 30px; border: 1px solid #ccc; border-radius: 5px; overflow: hidden;">
            <div style="width: {train}%; background-color: #4ec9b0; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold;">
                {train}%
            </div>
            <div style="width: {val}%; background-color: #dcdcaa; display: flex; align-items: center; justify-content: center; color: black; font-weight: bold;">
                {val}%
            </div>
            <div style="width: {test}%; background-color: #f48771; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold;">
                {test}%
            </div>
        </div>
        """
        self.distribution_bar.setText(html)
        self.distribution_bar.setTextFormat(Qt.RichText)
    
    def get_ratios(self):
        """Obtiene ratios actuales"""
        train = self.slider1.value() / 100.0
        val = self.slider2.value() / 100.0
        test = 1.0 - train - val
        return train, val, max(0, test)


class DistributionChart(FigureCanvasQTAgg):
    """Gráfico de distribución de clases"""
    
    def __init__(self, parent=None, width=8, height=5, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        fig.subplots_adjust(left=0.08, right=0.95, top=0.92, bottom=0.30)
        self.axes = fig.add_subplot(111)
        super().__init__(fig)
        self.setParent(parent)
        self.setMinimumHeight(320)
    
    def plot_distribution(self, stats: dict):
        """Dibuja distribución de clases"""
        self.axes.clear()
        
        if not stats or 'class_distribution' not in stats:
            self.axes.text(0.5, 0.5, 'No hay datos', 
                          ha='center', va='center', transform=self.axes.transAxes)
            self.draw()
            return
        
        classes = list(stats['class_distribution'].keys())
        counts = list(stats['class_distribution'].values())
        
        # Gráfico de barras
        bars = self.axes.bar(range(len(classes)), counts, color='#0e639c', alpha=0.7)
        
        # Etiquetas
        self.axes.set_xlabel('Clases')
        self.axes.set_ylabel('Número de Imágenes')
        self.axes.set_title('Distribución de Clases')
        self.axes.set_xticks(range(len(classes)))
        self.axes.set_xticklabels(classes, rotation=45, ha='right')
        
        # Valores sobre barras
        for bar in bars:
            height = bar.get_height()
            self.axes.text(bar.get_x() + bar.get_width()/2., height,
                          f'{int(height)}',
                          ha='center', va='bottom')
        
        self.axes.grid(axis='y', alpha=0.3)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            try:
                self.figure.tight_layout()
            except Exception:
                self.figure.subplots_adjust(left=0.08, right=0.95, top=0.90, bottom=0.25)
        self.draw()


class DatasetManagerTab(QWidget):
    """
    Tab de gestión de datasets con mapas vectoriales.
    Implementa IEEE 9241-16 para manipulación directa.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.manager = None
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Header
        header = QLabel("📊 Gestor de Datos")
        header.setStyleSheet(Styles.HEADER_SECTION)
        layout.addWidget(header)
        
        # === Configuración ===
        config_group = QGroupBox("Configuración")
        config_layout = QVBoxLayout()
        
        # Directorios fuente (múltiples)
        source_label = QLabel("Directorios Fuente (se unificarán):")
        source_label.setStyleSheet(Styles.LABEL_BOLD)
        config_layout.addWidget(source_label)
        
        # Tabla de directorios con clases
        self.source_dirs_table = QTableWidget()
        self.source_dirs_table.setColumnCount(2)
        self.source_dirs_table.setHorizontalHeaderLabels(['Carpeta', 'Nombre de Clase'])
        self.source_dirs_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.source_dirs_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Interactive)
        self.source_dirs_table.setMinimumHeight(100)
        self.source_dirs_table.setMaximumHeight(180)
        self.source_dirs_table.setStyleSheet(Styles.TABLE_WIDGET)
        config_layout.addWidget(self.source_dirs_table)
        
        # Botones para gestionar directorios
        source_buttons_layout = QHBoxLayout()
        add_dir_btn = QPushButton("➕ Añadir Carpeta")
        add_dir_btn.clicked.connect(self.add_source_directory)
        remove_dir_btn = QPushButton("➖ Quitar Seleccionada")
        remove_dir_btn.clicked.connect(self.remove_source_directory)
        clear_dirs_btn = QPushButton("🗑️ Limpiar Todo")
        clear_dirs_btn.clicked.connect(self.clear_source_directories)
        source_buttons_layout.addWidget(add_dir_btn)
        source_buttons_layout.addWidget(remove_dir_btn)
        source_buttons_layout.addWidget(clear_dirs_btn)
        config_layout.addLayout(source_buttons_layout)
        
        # Directorio salida
        output_form = QFormLayout()
        output_layout = QHBoxLayout()
        self.output_dir = QLineEdit("data/maps")
        self.output_dir.setPlaceholderText("Directorio para mapas vectoriales")
        output_browse = QPushButton("📁")
        output_browse.setMaximumWidth(40)
        output_browse.clicked.connect(self.browse_output)
        output_layout.addWidget(self.output_dir)
        output_layout.addWidget(output_browse)
        output_form.addRow("Directorio Salida:", output_layout)
        config_layout.addLayout(output_form)
        
        # Botones de acción principal
        scan_buttons_layout = QHBoxLayout()
        
        self.scan_btn = QPushButton("🔍 Escanear Directorio")
        self.scan_btn.clicked.connect(self.scan_directory)
        self.scan_btn.setStyleSheet(Styles.BUTTON_PRIMARY)
        
        self.load_maps_btn = QPushButton("📂 Cargar Mapas Existentes")
        self.load_maps_btn.clicked.connect(self.load_existing_maps)
        self.load_maps_btn.setToolTip(
            "Carga mapas previamente guardados (train/val/test_map.json + metadata)\n"
            "desde el directorio de salida, restaurando clases, splits y estadísticas."
        )
        
        scan_buttons_layout.addWidget(self.scan_btn)
        scan_buttons_layout.addWidget(self.load_maps_btn)
        config_layout.addLayout(scan_buttons_layout)
        
        config_group.setLayout(config_layout)
        layout.addWidget(config_group)
        
        # === Estadísticas ===
        stats_group = QGroupBox("Estadísticas del Dataset")
        stats_layout = QVBoxLayout()
        
        # Info básica
        info_layout = QHBoxLayout()
        self.total_label = QLabel("Total: 0 imágenes")
        self.classes_label = QLabel("Clases: 0")
        info_layout.addWidget(self.total_label)
        info_layout.addWidget(self.classes_label)
        info_layout.addStretch()
        stats_layout.addLayout(info_layout)
        
        # Gráfico de distribución
        self.chart = DistributionChart(self, width=8, height=5)
        stats_layout.addWidget(self.chart)
        
        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)
        
        # === Balanceo de Splits ===
        split_group = QGroupBox("Balanceo de Datos")
        split_layout = QVBoxLayout()
        
        # Info
        info = QLabel("Ajusta las proporciones de Train, Validation y Test:")
        info.setStyleSheet(Styles.LABEL_INFO)
        split_layout.addWidget(info)
        
        # Double slider
        self.double_slider = DoubleSlider()
        self.double_slider.valuesChanged.connect(self.update_split_preview)
        split_layout.addWidget(self.double_slider)
        
        # Tabla de preview
        self.split_table = QTableWidget()
        self.split_table.setColumnCount(3)
        self.split_table.setHorizontalHeaderLabels(['Split', 'Imágenes', 'Porcentaje'])
        self.split_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.split_table.setMinimumHeight(100)
        self.split_table.setMaximumHeight(140)
        split_layout.addWidget(self.split_table)
        
        split_group.setLayout(split_layout)
        layout.addWidget(split_group)
        
        # === Acciones ===
        actions_layout = QHBoxLayout()
        
        self.generate_btn = QPushButton("⚙️ Generar Splits")
        self.generate_btn.clicked.connect(self.generate_splits)
        self.generate_btn.setEnabled(False)
        
        self.save_btn = QPushButton("💾 Guardar Mapas")
        self.save_btn.clicked.connect(self.save_maps)
        self.save_btn.setEnabled(False)
        
        self.validate_btn = QPushButton("✓ Validar")
        self.validate_btn.clicked.connect(self.validate_maps)
        self.validate_btn.setEnabled(False)
        
        self.prepare_btn = QPushButton("📦 Preparar Datos para Entrenamiento")
        self.prepare_btn.clicked.connect(self.prepare_processed_data)
        self.prepare_btn.setEnabled(False)
        self.prepare_btn.setStyleSheet(Styles.BUTTON_PRIMARY)
        self.prepare_btn.setToolTip(
            "Copia las imágenes a data/processed/{train,val,test}/Clase/ "
            "para que el entrenamiento pueda encontrarlas."
        )
        
        actions_layout.addWidget(self.generate_btn)
        actions_layout.addWidget(self.validate_btn)
        actions_layout.addWidget(self.save_btn)
        actions_layout.addStretch()
        actions_layout.addWidget(self.prepare_btn)
        
        layout.addLayout(actions_layout)
        
        # === Status ===
        self.status_label = QLabel("Listo para escanear directorio")
        self.status_label.setStyleSheet(Styles.LABEL_INFO)
        layout.addWidget(self.status_label)
        
        self.setLayout(layout)
    
    def add_source_directory(self):
        """Añadir directorio fuente con nombre de clase"""
        dir_path = QFileDialog.getExistingDirectory(
            self, "Seleccionar Directorio Fuente"
        )
        if dir_path:
            # Verificar que no esté duplicado
            for row in range(self.source_dirs_table.rowCount()):
                if self.source_dirs_table.item(row, 0).text() == dir_path:
                    QMessageBox.warning(self, "Advertencia", "Este directorio ya está en la lista")
                    return
            
            # Solicitar nombre de clase
            from PyQt5.QtWidgets import QInputDialog
            class_name, ok = QInputDialog.getText(
                self,
                "Nombre de Clase",
                "Ingrese el nombre de la clase para esta carpeta:\n" + dir_path,
                QLineEdit.Normal,
                Path(dir_path).name  # Sugerir nombre de carpeta
            )
            
            if ok and class_name:
                # Validar nombre de clase
                class_name = class_name.strip()
                if not class_name:
                    QMessageBox.warning(self, "Error", "El nombre de clase no puede estar vacío")
                    return
                
                # Añadir a la tabla
                row = self.source_dirs_table.rowCount()
                self.source_dirs_table.insertRow(row)
                
                # Columna 0: Carpeta (no editable)
                folder_item = QTableWidgetItem(dir_path)
                folder_item.setFlags(folder_item.flags() & ~Qt.ItemIsEditable)
                self.source_dirs_table.setItem(row, 0, folder_item)
                
                # Columna 1: Clase (editable)
                class_item = QTableWidgetItem(class_name)
                self.source_dirs_table.setItem(row, 1, class_item)
                
                if self.parent_window:
                    self.parent_window.log_widget.append_log(
                        f"Carpeta añadida: {dir_path} → Clase: {class_name}",
                        "INFO"
                    )
            elif ok:
                QMessageBox.warning(self, "Error", "Debe ingresar un nombre de clase")
    
    def remove_source_directory(self):
        """Quitar directorio seleccionado de la tabla"""
        current_row = self.source_dirs_table.currentRow()
        if current_row >= 0:
            folder = self.source_dirs_table.item(current_row, 0).text()
            self.source_dirs_table.removeRow(current_row)
            if self.parent_window:
                self.parent_window.log_widget.append_log(
                    f"Directorio eliminado: {folder}",
                    "INFO"
                )
    
    def clear_source_directories(self):
        """Limpiar todos los directorios"""
        self.source_dirs_table.setRowCount(0)
        if self.parent_window:
            self.parent_window.log_widget.append_log("Lista de directorios limpiada", "INFO")
    
    def browse_output(self):
        """Buscar directorio salida"""
        dir_path = QFileDialog.getExistingDirectory(
            self, "Seleccionar Directorio de Salida"
        )
        if dir_path:
            self.output_dir.setText(dir_path)
    
    def load_existing_maps(self):
        """Carga mapas existentes desde el directorio de salida y restaura todo el estado."""
        maps_dir = self.output_dir.text()
        
        if not maps_dir or not Path(maps_dir).exists():
            QMessageBox.warning(self, "Error", f"Directorio de mapas no existe: {maps_dir}")
            return
        
        metadata_file = Path(maps_dir) / "dataset_metadata.json"
        if not metadata_file.exists():
            QMessageBox.warning(
                self, "Error",
                f"No se encontró dataset_metadata.json en:\n{maps_dir}\n\n"
                "Primero debe escanear y guardar mapas."
            )
            return
        
        try:
            self.status_label.setText("🔄 Cargando mapas existentes...")
            self.status_label.setStyleSheet(Styles.LABEL_INFO)
            
            if self.parent_window:
                self.parent_window.log_widget.append_log(
                    f"Cargando mapas desde {maps_dir}...", "INFO"
                )
            
            # Cargar manager desde mapas
            self.manager = DatasetMapManager.load_from_maps_directory(maps_dir)
            
            # Restaurar tabla de directorios fuente
            self.source_dirs_table.setRowCount(0)
            folder_class_map = self.manager.metadata.get('folder_class_map')
            if folder_class_map:
                for folder, class_name in folder_class_map.items():
                    row = self.source_dirs_table.rowCount()
                    self.source_dirs_table.insertRow(row)
                    
                    folder_item = QTableWidgetItem(folder)
                    folder_item.setFlags(folder_item.flags() & ~Qt.ItemIsEditable)
                    self.source_dirs_table.setItem(row, 0, folder_item)
                    
                    class_item = QTableWidgetItem(class_name)
                    self.source_dirs_table.setItem(row, 1, class_item)
            
            # Actualizar estadísticas
            total = self.manager.metadata.get('total_images', 0)
            classes = self.manager.metadata.get('classes', [])
            self.total_label.setText(f"Total: {total} imágenes")
            self.classes_label.setText(f"Clases: {len(classes)}")
            
            # Actualizar gráfico de distribución
            self.chart.plot_distribution(self.manager.metadata)
            
            # Restaurar ratios en sliders
            ratios = self.manager.ratios
            train_pct = int(ratios.get('train', 0.7) * 100)
            val_pct = int(ratios.get('val', 0.15) * 100)
            self.double_slider.slider1.setValue(train_pct)
            self.double_slider.slider2.setValue(val_pct)
            
            # Actualizar preview de splits
            self.update_split_preview(
                ratios.get('train', 0.7),
                ratios.get('val', 0.15),
                ratios.get('test', 0.15)
            )
            
            # Habilitar todos los botones
            self.generate_btn.setEnabled(True)
            self.save_btn.setEnabled(True)
            self.validate_btn.setEnabled(True)
            self.prepare_btn.setEnabled(True)
            
            self.status_label.setText(
                f"✅ Mapas cargados: {total} imágenes, {len(classes)} clases"
            )
            self.status_label.setStyleSheet(Styles.LABEL_SUCCESS)
            
            if self.parent_window:
                self.parent_window.log_widget.append_log(
                    f"Mapas cargados exitosamente: {total} imágenes, {len(classes)} clases",
                    "SUCCESS"
                )
                for cls in classes:
                    count = self.manager.metadata['class_distribution'].get(cls, 0)
                    self.parent_window.log_widget.append_log(
                        f"  {cls}: {count} imágenes", "DEBUG"
                    )
                
                # Mostrar info de splits cargados
                for split_name, split_data in self.manager.splits.items():
                    split_total = sum(len(imgs) for imgs in split_data.values())
                    self.parent_window.log_widget.append_log(
                        f"  Split {split_name}: {split_total} imágenes", "INFO"
                    )
        
        except Exception as e:
            self.status_label.setText(f"❌ Error: {str(e)}")
            self.status_label.setStyleSheet(Styles.LABEL_ERROR)
            if self.parent_window:
                self.parent_window.log_widget.append_log(f"Error cargando mapas: {e}", "ERROR")
            QMessageBox.critical(self, "Error", f"Error cargando mapas:\n{str(e)}")
    
    def scan_directory(self):
        """Escanea directorio(s) y construye mapa vectorial unificado con clases asignadas"""
        output = self.output_dir.text()
        
        # Obtener lista de directorios con sus clases
        folder_class_map = {}
        for row in range(self.source_dirs_table.rowCount()):
            folder = self.source_dirs_table.item(row, 0).text()
            class_name = self.source_dirs_table.item(row, 1).text().strip()
            
            if not class_name:
                QMessageBox.warning(
                    self,
                    "Error",
                    f"La carpeta '{folder}' no tiene nombre de clase asignado"
                )
                return
            
            folder_class_map[folder] = class_name
        
        if not folder_class_map:
            QMessageBox.warning(self, "Error", "Añada al menos una carpeta con su clase")
            return
        
        # Verificar que existan
        for folder in folder_class_map.keys():
            if not Path(folder).exists():
                QMessageBox.warning(self, "Error", f"El directorio no existe: {folder}")
                return
        
        try:
            # Crear manager con mapeo carpeta->clase
            self.manager = DatasetMapManager(folder_class_map, output)
            
            # Escanear
            self.status_label.setText("🔄 Escaneando directorio...")
            distribution = self.manager.scan_source_directory()
            
            # Actualizar UI
            total = sum(distribution.values())
            self.total_label.setText(f"Total: {total} imágenes")
            self.classes_label.setText(f"Clases: {len(distribution)}")
            
            # Actualizar gráfico
            self.chart.plot_distribution(self.manager.metadata)
            
            # Habilitar botones
            self.generate_btn.setEnabled(True)
            
            # Log
            num_folders = len(folder_class_map)
            self.status_label.setText(f"✅ Escaneado: {total} imágenes en {len(distribution)} clases desde {num_folders} carpeta(s)")
            self.status_label.setStyleSheet(Styles.LABEL_SUCCESS)
            
            if self.parent_window:
                self.parent_window.log_widget.append_log(
                    f"Dataset escaneado: {total} imágenes, {len(distribution)} clases desde {num_folders} carpeta(s)",
                    "SUCCESS"
                )
            
            # Actualizar preview
            self.update_split_preview(0.7, 0.15, 0.15)
        
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error escaneando directorio:\n{str(e)}")
            self.status_label.setText(f"❌ Error: {str(e)}")
            self.status_label.setStyleSheet(Styles.LABEL_ERROR)
    
    def update_split_preview(self, train, val, test):
        """Actualiza preview de splits"""
        if not self.manager:
            return
        
        total = self.manager.metadata['total_images']
        
        # Actualizar tabla
        self.split_table.setRowCount(3)
        
        splits = [
            ('Train', train, '#4ec9b0'),
            ('Validation', val, '#dcdcaa'),
            ('Test', test, '#f48771')
        ]
        
        for row, (name, ratio, color) in enumerate(splits):
            count = int(total * ratio)
            
            # Nombre
            name_item = QTableWidgetItem(name)
            name_item.setBackground(QColor(color))
            name_item.setForeground(QColor('white' if row != 1 else 'black'))
            self.split_table.setItem(row, 0, name_item)
            
            # Imágenes
            count_item = QTableWidgetItem(str(count))
            count_item.setTextAlignment(Qt.AlignCenter)
            self.split_table.setItem(row, 1, count_item)
            
            # Porcentaje
            pct_item = QTableWidgetItem(f"{ratio*100:.1f}%")
            pct_item.setTextAlignment(Qt.AlignCenter)
            self.split_table.setItem(row, 2, pct_item)
    
    def generate_splits(self):
        """Genera splits con proporciones actuales"""
        if not self.manager:
            return
        
        try:
            # Obtener ratios
            train, val, test = self.double_slider.get_ratios()
            
            # Establecer ratios
            self.manager.set_split_ratios(train, val, test)
            
            # Generar splits
            self.status_label.setText("🔄 Generando splits...")
            self.manager.generate_splits(seed=42, stratified=True)
            
            # Habilitar guardado y preparación
            self.save_btn.setEnabled(True)
            self.validate_btn.setEnabled(True)
            self.prepare_btn.setEnabled(True)
            
            self.status_label.setText("✅ Splits generados correctamente")
            self.status_label.setStyleSheet(Styles.LABEL_SUCCESS)
            
            if self.parent_window:
                self.parent_window.log_widget.append_log(
                    f"Splits generados: {train*100:.1f}% / {val*100:.1f}% / {test*100:.1f}%",
                    "SUCCESS"
                )
            
            QMessageBox.information(
                self,
                "Éxito",
                f"Splits generados:\n\n"
                f"Train: {train*100:.1f}%\n"
                f"Validation: {val*100:.1f}%\n"
                f"Test: {test*100:.1f}%"
            )
        
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error generando splits:\n{str(e)}")
            self.status_label.setText(f"❌ Error: {str(e)}")
            self.status_label.setStyleSheet(Styles.LABEL_ERROR)
    
    def save_maps(self):
        """Guarda mapas vectoriales"""
        if not self.manager:
            return
        
        try:
            self.status_label.setText("🔄 Guardando mapas vectoriales...")
            
            # Guardar en formato texto
            output_files = self.manager.save_split_maps(format='txt')
            
            # También guardar en JSON
            json_files = self.manager.save_split_maps(format='json')
            
            self.status_label.setText("✅ Mapas guardados correctamente")
            self.status_label.setStyleSheet(Styles.LABEL_SUCCESS)
            
            if self.parent_window:
                self.parent_window.log_widget.append_log(
                    f"Mapas guardados en {self.output_dir.text()}",
                    "SUCCESS"
                )
            
            # Mostrar archivos generados
            files_list = "\n".join([f"• {f.name}" for f in output_files.values()])
            QMessageBox.information(
                self,
                "Éxito",
                f"Mapas vectoriales guardados en:\n{self.output_dir.text()}\n\n"
                f"Archivos generados:\n{files_list}"
            )
        
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error guardando mapas:\n{str(e)}")
            self.status_label.setText(f"❌ Error: {str(e)}")
            self.status_label.setStyleSheet(Styles.LABEL_ERROR)
    
    def prepare_processed_data(self):
        """Prepara directorio data/processed/ con imágenes para entrenamiento"""
        if not self.manager:
            QMessageBox.warning(self, "Error", "Primero genere los splits")
            return
        
        # Confirmar operación
        total = sum(
            len(imgs) for split_data in self.manager.splits.values()
            for imgs in split_data.values()
        )
        reply = QMessageBox.question(
            self,
            "Preparar Datos",
            f"Se copiarán {total} imágenes a data/processed/\n"
            f"(estructura ImageFolder para entrenamiento).\n\n"
            f"Esto puede tomar unos minutos.\n"
            f"¿Desea continuar?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes
        )
        
        if reply != QMessageBox.Yes:
            return
        
        try:
            self.prepare_btn.setEnabled(False)
            self.status_label.setText("🔄 Preparando datos (copiando imágenes)...")
            self.status_label.setStyleSheet(Styles.LABEL_INFO)
            
            if self.parent_window:
                self.parent_window.log_widget.append_log(
                    f"Preparando {total} imágenes en data/processed/...", "INFO"
                )
            
            # Forzar repaint
            from PyQt5.QtWidgets import QApplication
            QApplication.processEvents()
            
            def progress_cb(current, total_files, msg):
                self.status_label.setText(f"🔄 Copiando: {current}/{total_files} — {msg}")
                if self.parent_window and current % 500 == 0:
                    self.parent_window.log_widget.append_log(
                        f"Progreso: {current}/{total_files} archivos", "DEBUG"
                    )
                QApplication.processEvents()
            
            # Primero guardar mapas si no se han guardado
            self.manager.save_split_maps(format='txt')
            self.manager.save_split_maps(format='json')
            
            # Preparar directorio procesado
            counts = self.manager.prepare_processed_directory(
                processed_dir="data/processed",
                strategy="symlink",
                progress_callback=progress_cb
            )
            
            # Actualizar config.yaml con el número correcto de clases
            self._update_config_classes()
            
            total_prepared = sum(counts.values())
            self.status_label.setText(f"✅ Datos preparados: {total_prepared} imágenes copiadas")
            self.status_label.setStyleSheet(Styles.LABEL_SUCCESS)
            
            if self.parent_window:
                for split_name, count in counts.items():
                    self.parent_window.log_widget.append_log(
                        f"  {split_name}: {count} imágenes", "SUCCESS"
                    )
                self.parent_window.log_widget.append_log(
                    f"Datos listos en data/processed/ — puede iniciar entrenamiento", "SUCCESS"
                )
            
            QMessageBox.information(
                self,
                "Datos Preparados",
                f"Imágenes preparadas en data/processed/:\n\n"
                + "\n".join(f"  {s}: {c} imágenes" for s, c in counts.items())
                + f"\n\nTotal: {total_prepared}\n\n"
                f"Ahora puede ir a la pestaña Entrenamiento."
            )
            
        except Exception as e:
            self.status_label.setText(f"❌ Error: {str(e)}")
            self.status_label.setStyleSheet(Styles.LABEL_ERROR)
            if self.parent_window:
                self.parent_window.log_widget.append_log(f"Error preparando datos: {e}", "ERROR")
            QMessageBox.critical(self, "Error", f"Error preparando datos:\n{str(e)}")
        finally:
            self.prepare_btn.setEnabled(True)
    
    def _update_config_classes(self):
        """Actualiza config.yaml con el número correcto de clases"""
        try:
            import yaml
            config_path = Path("config.yaml")
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                
                num_classes = len(self.manager.metadata.get('classes', []))
                if num_classes > 0:
                    config['data']['classes_per_batch'] = min(num_classes, config['data'].get('classes_per_batch', 3))
                    config['data']['batch_size'] = config['data']['classes_per_batch'] * config['data']['samples_per_class']
                    
                    with open(config_path, 'w', encoding='utf-8') as f:
                        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
                    
                    if self.parent_window:
                        self.parent_window.log_widget.append_log(
                            f"Config actualizado: classes_per_batch={config['data']['classes_per_batch']}, "
                            f"batch_size={config['data']['batch_size']}", "INFO"
                        )
        except Exception as e:
            if self.parent_window:
                self.parent_window.log_widget.append_log(
                    f"Advertencia: no se pudo actualizar config.yaml: {e}", "WARNING"
                )
    
    def validate_maps(self):
        """Valida integridad de mapas"""
        if not self.manager:
            return
        
        try:
            self.status_label.setText("🔄 Validando mapas...")
            
            validation = self.manager.validate_maps()
            
            if validation['valid']:
                self.status_label.setText("✅ Validación exitosa")
                self.status_label.setStyleSheet(Styles.LABEL_SUCCESS)
                
                stats_text = f"Total: {validation['stats']['total_images']} imágenes\n\n"
                for split, count in validation['stats']['splits'].items():
                    stats_text += f"{split.capitalize()}: {count} imágenes\n"
                
                QMessageBox.information(self, "Validación Exitosa", stats_text)
            else:
                self.status_label.setText("⚠️ Validación con errores")
                self.status_label.setStyleSheet(Styles.LABEL_WARNING)
                
                errors_text = "Errores encontrados:\n\n"
                for error in validation['errors']:
                    errors_text += f"• {error['type']}: {error['count']} archivos\n"
                
                QMessageBox.warning(self, "Validación con Errores", errors_text)
        
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error validando:\n{str(e)}")
