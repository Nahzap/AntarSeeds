"""
Sistema centralizado de estilos para la aplicación MetricLearning.
Define todos los estilos CSS y colores en un solo lugar para mantener consistencia.
"""

# ============================================================================
# PALETA DE COLORES
# ============================================================================

class Colors:
    """Paleta de colores centralizada"""
    # Colores primarios
    PRIMARY = "#0e639c"
    PRIMARY_HOVER = "#1177bb"
    PRIMARY_LIGHT = "#f0f8ff"
    
    # Colores de estado
    SUCCESS = "#4ec9b0"
    WARNING = "#dcdcaa"
    ERROR = "#f48771"
    INFO = "#0e639c"
    
    # Colores de fondo
    BG_MAIN = "#f5f5f5"
    BG_WIDGET = "#ffffff"
    BG_ALTERNATE = "#f9f9f9"
    BG_HEADER = "#f0f0f0"
    BG_DARK = "#1e1e1e"
    
    # Colores de texto
    TEXT_PRIMARY = "#333333"
    TEXT_SECONDARY = "#666666"
    TEXT_LIGHT = "#d4d4d4"
    TEXT_WHITE = "#ffffff"
    
    # Colores de borde
    BORDER_LIGHT = "#cccccc"
    BORDER_MEDIUM = "#3c3c3c"
    BORDER_DARK = "#ddd"
    
    # Colores de gráficos
    CHART_1 = "#0e639c"
    CHART_2 = "#4ec9b0"
    CHART_3 = "#dcdcaa"
    CHART_4 = "#f48771"
    
    # Colores de botones deshabilitados
    DISABLED = "#555555"
    DISABLED_BG = "#cccccc"


# ============================================================================
# ESTILOS DE COMPONENTES
# ============================================================================

class Styles:
    """Estilos CSS centralizados para componentes PyQt5"""
    
    # ------------------------------------------------------------------------
    # BOTONES
    # ------------------------------------------------------------------------
    
    BUTTON_PRIMARY = f"""
        QPushButton {{
            background-color: {Colors.PRIMARY};
            color: {Colors.TEXT_WHITE};
            font-size: 12pt;
            font-weight: bold;
            padding: 10px;
            border-radius: 5px;
            border: none;
        }}
        QPushButton:hover {{
            background-color: {Colors.PRIMARY_HOVER};
        }}
        QPushButton:disabled {{
            background-color: {Colors.DISABLED};
            color: #999999;
        }}
    """
    
    BUTTON_SAVE = """
        QPushButton {
            background-color: #0e639c;
            color: white;
            font-weight: bold;
            padding: 8px 16px;
            border-radius: 5px;
        }
        QPushButton:hover {
            background-color: #1177bb;
        }
    """

    BUTTON_SECONDARY = f"""
        QPushButton {{
            background-color: {Colors.BG_ALTERNATE};
            color: {Colors.TEXT_PRIMARY};
            font-weight: bold;
            padding: 8px;
            border-radius: 5px;
            border: 1px solid {Colors.BORDER_LIGHT};
        }}
        QPushButton:hover {{
            background-color: {Colors.BG_HEADER};
            border-color: {Colors.PRIMARY};
        }}
        QPushButton:disabled {{
            background-color: {Colors.DISABLED_BG};
            color: #999999;
        }}
    """
    
    BUTTON_SMALL = f"""
        QPushButton {{
            background-color: {Colors.PRIMARY};
            color: {Colors.TEXT_WHITE};
            font-weight: bold;
            padding: 5px 10px;
            border-radius: 3px;
            border: none;
        }}
        QPushButton:hover {{
            background-color: {Colors.PRIMARY_HOVER};
        }}
    """
    
    BUTTON_ICON = """
        QPushButton {
            background-color: transparent;
            border: none;
            padding: 5px;
        }
        QPushButton:hover {
            background-color: rgba(0, 0, 0, 0.05);
            border-radius: 3px;
        }
    """
    
    # ------------------------------------------------------------------------
    # LABELS Y HEADERS
    # ------------------------------------------------------------------------
    
    HEADER_MAIN = f"""
        QLabel {{
            font-size: 16pt;
            font-weight: bold;
            color: {Colors.PRIMARY};
            padding: 8px 12px;
            background-color: {Colors.BG_HEADER};
            border-radius: 5px;
        }}
    """
    
    HEADER_SECTION = f"""
        QLabel {{
            font-size: 13pt;
            font-weight: bold;
            color: {Colors.PRIMARY};
            padding: 6px 10px;
            background-color: {Colors.BG_HEADER};
            border-radius: 5px;
        }}
    """
    
    LABEL_BOLD = """
        QLabel {
            font-weight: bold;
            padding: 5px;
        }
    """
    
    LABEL_INFO = f"""
        QLabel {{
            color: {Colors.TEXT_SECONDARY};
            padding: 5px;
            font-size: 11pt;
        }}
    """
    
    LABEL_SUCCESS = f"""
        QLabel {{
            color: {Colors.SUCCESS};
            padding: 5px;
        }}
    """
    
    LABEL_ERROR = f"""
        QLabel {{
            color: {Colors.ERROR};
            padding: 5px;
        }}
    """
    
    LABEL_WARNING = f"""
        QLabel {{
            color: {Colors.WARNING};
            padding: 5px;
        }}
    """
    
    # ------------------------------------------------------------------------
    # INPUTS
    # ------------------------------------------------------------------------
    
    LINE_EDIT = f"""
        QLineEdit {{
            padding: 8px;
            border: 1px solid {Colors.BORDER_LIGHT};
            border-radius: 4px;
            background-color: {Colors.BG_WIDGET};
            font-size: 10pt;
        }}
        QLineEdit:focus {{
            border-color: {Colors.PRIMARY};
        }}
        QLineEdit:disabled {{
            background-color: {Colors.BG_HEADER};
            color: {Colors.TEXT_SECONDARY};
        }}
    """
    
    TEXT_EDIT = f"""
        QTextEdit {{
            padding: 8px;
            border: 1px solid {Colors.BORDER_LIGHT};
            border-radius: 4px;
            background-color: {Colors.BG_WIDGET};
            font-size: 10pt;
        }}
        QTextEdit:focus {{
            border-color: {Colors.PRIMARY};
        }}
    """
    
    SPIN_BOX = f"""
        QSpinBox {{
            padding: 5px;
            border: 1px solid {Colors.BORDER_LIGHT};
            border-radius: 4px;
            background-color: {Colors.BG_WIDGET};
        }}
        QSpinBox:focus {{
            border-color: {Colors.PRIMARY};
        }}
    """
    
    COMBO_BOX = f"""
        QComboBox {{
            padding: 5px;
            border: 1px solid {Colors.BORDER_LIGHT};
            border-radius: 4px;
            background-color: {Colors.BG_WIDGET};
        }}
        QComboBox:focus {{
            border-color: {Colors.PRIMARY};
        }}
        QComboBox::drop-down {{
            border: none;
        }}
        QComboBox::down-arrow {{
            image: none;
            border-left: 5px solid transparent;
            border-right: 5px solid transparent;
            border-top: 5px solid {Colors.TEXT_PRIMARY};
        }}
    """
    
    # ------------------------------------------------------------------------
    # GROUPBOX
    # ------------------------------------------------------------------------
    
    GROUP_BOX = f"""
        QGroupBox {{
            font-weight: bold;
            border: 2px solid {Colors.BORDER_LIGHT};
            border-radius: 5px;
            margin-top: 10px;
            padding-top: 10px;
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 0 5px;
            color: {Colors.PRIMARY};
        }}
    """
    
    # ------------------------------------------------------------------------
    # PROGRESS BAR
    # ------------------------------------------------------------------------
    
    PROGRESS_BAR = f"""
        QProgressBar {{
            border: 2px solid {Colors.PRIMARY};
            border-radius: 5px;
            text-align: center;
            font-weight: bold;
            background-color: {Colors.BG_WIDGET};
        }}
        QProgressBar::chunk {{
            background-color: {Colors.PRIMARY};
            border-radius: 3px;
        }}
    """
    
    # ------------------------------------------------------------------------
    # TABLES
    # ------------------------------------------------------------------------
    
    TABLE_WIDGET = f"""
        QTableWidget {{
            border: 1px solid {Colors.BORDER_LIGHT};
            border-radius: 4px;
            background-color: {Colors.BG_WIDGET};
            gridline-color: {Colors.BORDER_LIGHT};
        }}
        QTableWidget::item {{
            padding: 5px;
        }}
        QTableWidget::item:selected {{
            background-color: {Colors.PRIMARY};
            color: {Colors.TEXT_WHITE};
        }}
        QHeaderView::section {{
            background-color: {Colors.PRIMARY};
            color: {Colors.TEXT_WHITE};
            padding: 8px;
            border: none;
            font-weight: bold;
        }}
    """
    
    # ------------------------------------------------------------------------
    # LIST WIDGET
    # ------------------------------------------------------------------------
    
    LIST_WIDGET = f"""
        QListWidget {{
            border: 1px solid {Colors.BORDER_LIGHT};
            border-radius: 4px;
            background-color: {Colors.BG_WIDGET};
            padding: 5px;
        }}
        QListWidget::item {{
            padding: 5px;
            border-radius: 3px;
        }}
        QListWidget::item:selected {{
            background-color: {Colors.PRIMARY};
            color: {Colors.TEXT_WHITE};
        }}
        QListWidget::item:hover {{
            background-color: {Colors.BG_ALTERNATE};
        }}
    """
    
    # ------------------------------------------------------------------------
    # TABS
    # ------------------------------------------------------------------------
    
    TAB_WIDGET = f"""
        QTabWidget::pane {{
            border: 1px solid {Colors.BORDER_LIGHT};
            border-radius: 5px;
            padding: 10px;
            background-color: {Colors.BG_WIDGET};
        }}
        QTabBar::tab {{
            background-color: {Colors.BG_HEADER};
            padding: 10px 20px;
            margin-right: 2px;
            border-top-left-radius: 5px;
            border-top-right-radius: 5px;
            color: {Colors.TEXT_PRIMARY};
        }}
        QTabBar::tab:selected {{
            background-color: {Colors.PRIMARY};
            color: {Colors.TEXT_WHITE};
        }}
        QTabBar::tab:hover {{
            background-color: {Colors.PRIMARY_HOVER};
            color: {Colors.TEXT_WHITE};
        }}
    """
    
    # ------------------------------------------------------------------------
    # LOG WIDGET (Console-like)
    # ------------------------------------------------------------------------
    
    LOG_WIDGET = f"""
        QTextEdit {{
            background-color: {Colors.BG_DARK};
            color: {Colors.TEXT_LIGHT};
            font-family: 'Consolas', 'Courier New', monospace;
            font-size: 10pt;
            border: 1px solid {Colors.BORDER_MEDIUM};
            border-radius: 4px;
            padding: 5px;
        }}
    """
    
    # ------------------------------------------------------------------------
    # SLIDERS
    # ------------------------------------------------------------------------
    
    SLIDER = f"""
        QSlider::groove:horizontal {{
            border: 1px solid {Colors.BORDER_LIGHT};
            height: 8px;
            background: {Colors.BG_HEADER};
            border-radius: 4px;
        }}
        QSlider::handle:horizontal {{
            background: {Colors.PRIMARY};
            border: 1px solid {Colors.PRIMARY};
            width: 18px;
            margin: -5px 0;
            border-radius: 9px;
        }}
        QSlider::handle:horizontal:hover {{
            background: {Colors.PRIMARY_HOVER};
        }}
    """
    
    # ------------------------------------------------------------------------
    # CHECKBOXES
    # ------------------------------------------------------------------------
    
    CHECKBOX = f"""
        QCheckBox {{
            spacing: 5px;
        }}
        QCheckBox::indicator {{
            width: 18px;
            height: 18px;
            border: 2px solid {Colors.BORDER_LIGHT};
            border-radius: 3px;
            background-color: {Colors.BG_WIDGET};
        }}
        QCheckBox::indicator:checked {{
            background-color: {Colors.PRIMARY};
            border-color: {Colors.PRIMARY};
        }}
        QCheckBox::indicator:hover {{
            border-color: {Colors.PRIMARY};
        }}
    """
    
    # ------------------------------------------------------------------------
    # SCROLLBAR
    # ------------------------------------------------------------------------
    
    SCROLLBAR = f"""
        QScrollBar:vertical {{
            border: none;
            background: {Colors.BG_HEADER};
            width: 10px;
            margin: 0px;
        }}
        QScrollBar::handle:vertical {{
            background: {Colors.BORDER_LIGHT};
            min-height: 20px;
            border-radius: 5px;
        }}
        QScrollBar::handle:vertical:hover {{
            background: {Colors.PRIMARY};
        }}
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
            height: 0px;
        }}
    """


# ============================================================================
# UTILIDADES
# ============================================================================

def apply_global_styles(app):
    """
    Aplica estilos globales a la aplicación.
    
    Args:
        app: QApplication instance
    """
    from PyQt5.QtGui import QPalette, QColor
    from PyQt5.QtCore import Qt
    
    # Configurar paleta global
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(Colors.BG_MAIN))
    palette.setColor(QPalette.WindowText, QColor(Colors.TEXT_PRIMARY))
    palette.setColor(QPalette.Base, QColor(Colors.BG_WIDGET))
    palette.setColor(QPalette.AlternateBase, QColor(Colors.BG_ALTERNATE))
    palette.setColor(QPalette.ToolTipBase, QColor(Colors.BG_WIDGET))
    palette.setColor(QPalette.ToolTipText, QColor(Colors.TEXT_PRIMARY))
    palette.setColor(QPalette.Text, QColor(Colors.TEXT_PRIMARY))
    palette.setColor(QPalette.Button, QColor(Colors.BG_HEADER))
    palette.setColor(QPalette.ButtonText, QColor(Colors.TEXT_PRIMARY))
    palette.setColor(QPalette.Link, QColor(Colors.PRIMARY))
    palette.setColor(QPalette.Highlight, QColor(Colors.PRIMARY))
    palette.setColor(QPalette.HighlightedText, QColor(Colors.TEXT_WHITE))
    
    app.setPalette(palette)
    app.setStyle("Fusion")


def get_status_style(status: str) -> str:
    """
    Obtiene estilo según el estado.
    
    Args:
        status: 'success', 'error', 'warning', 'info'
        
    Returns:
        String con estilo CSS
    """
    status_map = {
        'success': Styles.LABEL_SUCCESS,
        'error': Styles.LABEL_ERROR,
        'warning': Styles.LABEL_WARNING,
        'info': Styles.LABEL_INFO
    }
    return status_map.get(status.lower(), Styles.LABEL_INFO)


def get_color(color_name: str) -> str:
    """
    Obtiene un color de la paleta.
    
    Args:
        color_name: Nombre del color (ej: 'PRIMARY', 'SUCCESS')
        
    Returns:
        Código hexadecimal del color
    """
    return getattr(Colors, color_name.upper(), Colors.PRIMARY)
