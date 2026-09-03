"""
Widget de botón de ayuda con popup explicativo
"""

from PyQt5.QtWidgets import QPushButton, QMessageBox
from PyQt5.QtCore import Qt


class HelpButton(QPushButton):
    """
    Botón de ayuda (?) que muestra popup explicativo
    """
    
    def __init__(self, title, explanation, parent=None):
        super().__init__("?", parent)
        self.title = title
        self.explanation = explanation
        
        # Estilo del botón
        self.setMaximumWidth(25)
        self.setMaximumHeight(25)
        self.setStyleSheet("""
            QPushButton {
                background-color: #0e639c;
                color: white;
                border-radius: 12px;
                font-weight: bold;
                font-size: 12pt;
            }
            QPushButton:hover {
                background-color: #1177bb;
            }
        """)
        self.setToolTip("Click para más información")
        self.clicked.connect(self.show_help)
    
    def show_help(self):
        """Muestra popup con explicación"""
        msg = QMessageBox(self)
        msg.setWindowTitle(f"ℹ️ {self.title}")
        msg.setText(self.explanation)
        msg.setIcon(QMessageBox.Information)
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()


def create_field_with_help(label_text, widget, help_title, help_text):
    """
    Crea un layout con label, widget y botón de ayuda
    
    Returns:
        QHBoxLayout con label, widget y botón ?
    """
    from PyQt5.QtWidgets import QHBoxLayout, QLabel
    
    layout = QHBoxLayout()
    
    # Label
    label = QLabel(label_text)
    layout.addWidget(label)
    
    # Widget
    layout.addWidget(widget, stretch=1)
    
    # Botón de ayuda
    help_btn = HelpButton(help_title, help_text)
    layout.addWidget(help_btn)
    
    return layout
