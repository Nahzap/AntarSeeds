"""
MetricLearning - Aplicación Principal
=====================================================

ÚNICO PUNTO DE ENTRADA DEL SISTEMA

Uso:
    python app.py              # Inicia interfaz gráfica (GUI)
    python app.py --cli        # Inicia línea de comandos (CLI)
    python app.py --help       # Muestra ayuda

Autor: MetricLearning Team
Versión: 2.0
Fecha: 2026-01-10
"""

import sys
import os
import io
import argparse
from datetime import datetime
from pathlib import Path

# Raíz del proyecto: rutas relativas (config.yaml, logs/, models/) no dependen del CWD
PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parent
VENV_DIR = REPO_ROOT / ".venv"
VENV_PYTHON = VENV_DIR / "Scripts" / "python.exe"


def require_project_venv():
    """Obliga el .venv de AntarSeeds (no el Python de uv / sistema)."""
    expected = VENV_DIR.resolve()
    prefix = Path(sys.prefix).resolve()
    exe = Path(sys.executable).resolve()
    in_venv = prefix == expected or exe.parent.parent == expected
    if in_venv:
        return
    print("\nERROR: este programa solo corre con el entorno virtual de AntarSeeds.")
    print(f"   Python usado:  {exe}")
    print(f"   Python correcto: {VENV_PYTHON.resolve()}")
    print("\n   Desde la raíz del repo:")
    print("     .\\.venv\\Scripts\\python.exe MetricLearning\\app.py")
    print("     .\\start.ps1")
    print("   O selecciona el intérprete `.venv` en Cursor (no cpython de uv).\n")
    sys.exit(1)


def ensure_project_root():
    """Usar el directorio del proyecto aunque PowerShell/CMD arranquen desde otro sitio."""
    os.chdir(PROJECT_ROOT)
    root = str(PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


class TeeStream:
    """
    Duplicates writes to both the original stream and a log file.
    Every line gets a timestamp prefix in the log file.
    Handles tqdm, print, logging, and any other stdout/stderr output.
    """
    def __init__(self, original_stream, log_file, stream_name="stdout"):
        self.original = original_stream
        self.log_file = log_file
        self.stream_name = stream_name
        self._line_buffer = ""
    
    def write(self, text):
        if text:
            try:
                self.original.write(text)
            except UnicodeEncodeError:
                self.original.write(
                    text.encode(getattr(self.original, 'encoding', 'utf-8') or 'utf-8', errors='replace')
                    .decode(getattr(self.original, 'encoding', 'utf-8') or 'utf-8', errors='replace')
                )
            # Write to log file with timestamp for each line
            for char in text:
                if char == '\n':
                    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                    self.log_file.write(f"[{ts}] [{self.stream_name}] {self._line_buffer}\n")
                    self._line_buffer = ""
                elif char == '\r':
                    # Carriage return (tqdm progress bars) — overwrite line
                    if self._line_buffer.strip():
                        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                        self.log_file.write(f"[{ts}] [{self.stream_name}] {self._line_buffer}\n")
                    self._line_buffer = ""
                else:
                    self._line_buffer += char
            self.log_file.flush()
    
    def flush(self):
        self.original.flush()
        self.log_file.flush()
    
    def fileno(self):
        return self.original.fileno()
    
    def isatty(self):
        return self.original.isatty()
    
    @property
    def encoding(self):
        return getattr(self.original, 'encoding', 'utf-8')


def setup_persistent_logging():
    """
    Vacía logs/ y captura stdout/stderr en antarseeds_YYYY-MM-DD_HH-MM-SS.log.
    Cada reinicio de la app empieza un log nuevo.
    """
    from src.utils.logging_utils import start_session_logs

    log_path = start_session_logs(str(PROJECT_ROOT / "logs"))
    log_file = open(log_path, "a", encoding="utf-8", buffering=1)
    
    # Redirect stdout and stderr
    sys.stdout = TeeStream(sys.__stdout__, log_file, "stdout")
    sys.stderr = TeeStream(sys.__stderr__, log_file, "stderr")
    
    return log_path


def print_banner():
    """Muestra banner de la aplicación"""
    print("\n" + "="*70)
    print("  MetricLearning v2.0")
    print("  Sistema de Clasificación de Imágenes con Metric Learning")
    print("="*70 + "\n")


def launch_gui():
    """Lanza interfaz gráfica"""
    print("🖥️  Iniciando interfaz gráfica...")
    from lib.gui_app import main as gui_main
    gui_main()


def launch_cli(args):
    """Lanza interfaz de línea de comandos"""
    print("⌨️  Iniciando línea de comandos...")
    from lib.cli_app import main as cli_main
    
    # Pasar argumentos al CLI
    sys.argv = ['cli_app.py'] + args
    cli_main()


def main():
    """Función principal"""
    parser = argparse.ArgumentParser(
        description="MetricLearning - Sistema de Clasificación de Imágenes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modos de uso:

  1. INTERFAZ GRÁFICA (por defecto):
     python app.py
     
     - Interfaz visual intuitiva
     - Editor de configuración
     - Gestor de datasets
     - Entrenamiento, inferencia y evaluación
     
  2. LÍNEA DE COMANDOS:
     python app.py --cli <comando> [opciones]
     
     Comandos disponibles:
       train       - Entrenar modelo
       inference   - Clasificar imágenes
       evaluate    - Evaluar modelo
       batch       - Evaluación en batch
       prepare     - Preparar dataset
     
     Ejemplos:
       python app.py --cli train --config config.yaml
       python app.py --cli inference --checkpoint model.pth --query img.jpg
       python app.py --cli evaluate --checkpoint model.pth

Para más información sobre comandos CLI:
  python app.py --cli --help
        """
    )
    
    parser.add_argument(
        '--cli',
        action='store_true',
        help='Usar interfaz de línea de comandos en lugar de GUI'
    )
    
    # Parsear solo los argumentos conocidos
    args, remaining = parser.parse_known_args()
    
    # Mostrar banner
    print_banner()
    
    # Decidir qué interfaz lanzar
    if args.cli:
        # Modo CLI
        launch_cli(remaining)
    else:
        # Modo GUI (por defecto)
        if remaining:
            print("⚠️  Argumentos ignorados en modo GUI:", remaining)
            print("    Use --cli para modo línea de comandos\n")
        launch_gui()


def install_exception_hooks():
    """Instala hooks globales para capturar excepciones no manejadas."""
    import threading
    import traceback as tb

    def global_excepthook(exc_type, exc_value, exc_tb):
        msg = "".join(tb.format_exception(exc_type, exc_value, exc_tb))
        print(f"\n💥 EXCEPCIÓN NO MANEJADA:\n{msg}", file=sys.__stderr__)
        sys.__excepthook__(exc_type, exc_value, exc_tb)

    sys.excepthook = global_excepthook

    # Hook para excepciones en threads secundarios
    _original_init = threading.Thread.__init__

    def _patched_init(self, *args, **kwargs):
        _original_init(self, *args, **kwargs)
        _original_run = self.run

        def wrapped_run(*a, **kw):
            try:
                _original_run(*a, **kw)
            except Exception:
                msg = tb.format_exc()
                print(f"\n💥 EXCEPCIÓN EN THREAD {self.name}:\n{msg}", file=sys.__stderr__)

        self.run = wrapped_run

    threading.Thread.__init__ = _patched_init


if __name__ == "__main__":
    ensure_project_root()
    require_project_venv()

    # Setup persistent logging BEFORE anything else
    log_path = setup_persistent_logging()
    print(f"📝 Log de sesión: {log_path.absolute()}")

    # Enable faulthandler for segfaults / native crashes
    import faulthandler
    faulthandler.enable(file=sys.__stderr__, all_threads=True)

    install_exception_hooks()
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Operación cancelada por el usuario")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
