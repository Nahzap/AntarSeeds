# AntarSeeds

Interfaz y pipeline (GUI + CLI) para trabajar con **imagenes microscopicas de semillas** usando un flujo tipo:
**U2-Net (propuestas) -> `.seg` (anotaciones) -> entrenamiento/inferencia de Metric Learning (ViT/DINOv2)**.

El repo está pensado para que **no subas datos sensibles o pesados**:
- Las imágenes/datasets grandes se mantienen fuera de Git (gitignore).
- Los artefactos de entrenamiento (runs/logs/checkpoints) tampoco se suben.

## Requisitos

- Windows (recomendado) o Python soportado por el proyecto.
- **Python 3.11** (el launcher espera un entorno en `.venv`).
- Dependencias listadas en `requirements.txt`.

## Instalación

Desde la raíz del repo:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Ejecutar (GUI)

```powershell
.\start.ps1
```

En modo CLI:

```powershell
.\start.ps1 --cli --help
```

O directamente:

```powershell
.\.venv\Scripts\python.exe .\MetricLearning\app.py --cli
```

## Estructura de datos (lo que se usa / dónde)

El código usa rutas relativas desde la raíz del repo. Estructura esperada:

### Datos de entrada (locales)
- `Data/`
  - Incluye calibración/nomenclatura (archivos pequeños).
  - **Imágenes raw/datasets grandes** deben ir en `Data/images/` (la carpeta está gitignored).

### Datos para entrenamiento / anotaciones (locales)
- `MetricLearning/data/processed/`
  - `train/`, `val/`, `test/` → subcarpetas por clase.
  - (Se crean/enlazan desde el dataset preparado por la app.)
- `MetricLearning/data/annotations/`
  - Archivos `.seg` por clase generados desde la GUI (panel de ROI/Contornos).

> Nota: `MetricLearning/data/*` está gitignored para evitar subir datos grandes o sensibles.

### Artefactos (NO se suben a Git)
- `logs/`, `MetricLearning/runs/`, checkpoints/pesos, etc. están gitignored.

## Qué hace la app (flujo típico)

1. Abres la **GUI** (`start.ps1`).
2. Usas el flujo de **contornos/ROI** para generar `.seg` en `MetricLearning/data/annotations/`.
3. Preparas el dataset en `MetricLearning/data/processed/` (splits).
4. Entrenas/evaluas/inferes desde la misma app (o por CLI).

## Pesos del detector (U2-Net)

El sistema descarga pesos desde Google Drive cuando corresponda (si faltan en disco). Si falla la descarga, revisa que tengas `gdown` instalado y que haya conexión.

## Contribuir

Si vas a subir cambios:
- Asegúrate de mantener el `.gitignore` (no subir `Data/images/` ni `MetricLearning/data/`).
- Mantén `README.md` actualizado con instrucciones de uso.

