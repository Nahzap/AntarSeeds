#!/usr/bin/env python
"""
Setup U²-Net — Descarga de pesos pre-entrenados para detección de granos.

Uso:
    python setup_u2net.py [--model u2netp|u2net] [--full]

Modelos:
    u2netp: Versión pequeña (~4.7 MB), rápida (~30ms GPU)
    u2net:  Versión completa (~176 MB), más precisa
"""

import os
import sys
from pathlib import Path


class Colors:
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BLUE = "\033[94m"
    RESET = "\033[0m"
    BOLD = "\033[1m"


def print_header():
    print(f"\n{Colors.BOLD}{'=' * 60}")
    print("  Setup U²-Net — Pesos para Detección de Granos de Polen")
    print(f"{'=' * 60}{Colors.RESET}\n")


def check_pytorch():
    print(f"{Colors.BLUE}[1/3] Verificando PyTorch...{Colors.RESET}")
    try:
        import torch

        print(f"  {Colors.GREEN}✓ PyTorch {torch.__version__}{Colors.RESET}")
        if torch.cuda.is_available():
            print(f"  {Colors.GREEN}✓ CUDA: {torch.cuda.get_device_name(0)}{Colors.RESET}")
        else:
            print(f"  {Colors.YELLOW}⚠ CUDA no disponible — se usará CPU{Colors.RESET}")
        return True
    except ImportError:
        print(f"  {Colors.RED}✗ PyTorch no instalado{Colors.RESET}")
        print(f"  Ejecuta: pip install torch torchvision")
        return False


def check_gdown():
    print(f"\n{Colors.BLUE}[2/3] Verificando gdown...{Colors.RESET}")
    try:
        import gdown

        print(f"  {Colors.GREEN}✓ gdown instalado{Colors.RESET}")
        return True
    except ImportError:
        print(f"  {Colors.YELLOW}⚠ Instalando gdown...{Colors.RESET}")
        os.system(f"{sys.executable} -m pip install gdown")
        try:
            import gdown

            print(f"  {Colors.GREEN}✓ gdown instalado{Colors.RESET}")
            return True
        except ImportError:
            print(f"  {Colors.RED}✗ No se pudo instalar gdown{Colors.RESET}")
            return False


def download_weights(model_type="u2netp"):
    print(f"\n{Colors.BLUE}[3/3] Descargando pesos {model_type}...{Colors.RESET}")

    import gdown

    base_dir = Path(__file__).parent
    weights_dir = base_dir / "models" / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    urls = {
        "u2netp": {
            "url": "https://drive.google.com/uc?id=1rbSTGKAE-MTxBYHd-51l2hMOQPT_7EPy",
            "file": "u2netp.pth",
            "size": "~4.7 MB",
        },
        "u2net": {
            "url": "https://drive.google.com/uc?id=1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ",
            "file": "u2net.pth",
            "size": "~176 MB",
        },
    }

    if model_type not in urls:
        print(f"  {Colors.RED}✗ Modelo desconocido: {model_type}{Colors.RESET}")
        return False

    info = urls[model_type]
    output_path = weights_dir / info["file"]

    if output_path.exists():
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"  {Colors.GREEN}✓ Ya existe: {info['file']} ({size_mb:.1f} MB){Colors.RESET}")
        return True

    print(f"  Descargando {info['file']} ({info['size']})...")

    try:
        gdown.download(info["url"], str(output_path), quiet=False)
        if output_path.exists():
            size_mb = output_path.stat().st_size / (1024 * 1024)
            print(f"  {Colors.GREEN}✓ Descargado: {info['file']} ({size_mb:.1f} MB){Colors.RESET}")
            return True
        else:
            print(f"  {Colors.RED}✗ Archivo no creado{Colors.RESET}")
            return False
    except Exception as e:
        print(f"  {Colors.RED}✗ Error: {e}{Colors.RESET}")
        print(f"\n  Descarga manual: {info['url']}")
        print(f"  Guardar en: {output_path}")
        return False


def verify():
    print(f"\n{Colors.BOLD}{'=' * 60}")
    print("  Verificación Final")
    print(f"{'=' * 60}{Colors.RESET}\n")

    base_dir = Path(__file__).parent
    weights_path = base_dir / "models" / "weights" / "u2netp.pth"

    if not weights_path.exists():
        print(f"  {Colors.RED}✗ u2netp.pth no encontrado{Colors.RESET}")
        return False

    print(f"  {Colors.GREEN}✓ Pesos u2netp.pth listos{Colors.RESET}")

    try:
        sys.path.insert(0, str(base_dir))
        from src.models.u2net import U2NETP

        print(f"  {Colors.GREEN}✓ Modelo U2NETP importable{Colors.RESET}")
    except Exception as e:
        print(f"  {Colors.RED}✗ Error importando: {e}{Colors.RESET}")
        return False

    print(f"\n{Colors.GREEN}{Colors.BOLD}✓ Setup U²-Net completado!{Colors.RESET}")
    return True


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Setup U²-Net weights")
    parser.add_argument(
        "--model",
        choices=["u2netp", "u2net"],
        default="u2netp",
        help="Modelo a descargar (default: u2netp)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Descargar ambos modelos",
    )
    args = parser.parse_args()

    print_header()

    if not check_pytorch():
        sys.exit(1)

    if not check_gdown():
        sys.exit(1)

    if args.full:
        download_weights("u2netp")
        download_weights("u2net")
    else:
        download_weights(args.model)

    verify()


if __name__ == "__main__":
    main()
