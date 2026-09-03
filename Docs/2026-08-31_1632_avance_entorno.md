# Avance entorno

| Campo | Valor |
| --- | --- |
| Fecha | 2026-08-31 16:32 (UTC-04:00) |
| Codigo | no |

## Funciona

- **venv:** `C:\Users\askna\Documents\GitHub\AntarSeeds\.venv` (`python.exe`).
- **CUDA:** disponible (`NVIDIA GeForce RTX 3070 Laptop GPU`).
- **Pesos U2-NetP:** `MetricLearning/models/weights/u2netp.pth`.
- **Junctions raw:** 16 (`Data/images/raw/SAMPLE_001` ... `SAMPLE_016` -> `F:\MICROSCOPIA\ANTARSEEDS\SAMPLE_xxx`).
- **Smoke detect:** `SAMPLE_004` `J_BUFONIUS_0001` f1 via junction ASCII; shape `(1942, 2590, 3)`, `n_grains=4`, `SMOKE_DETECT_OK`.
- **bpof:** 684 file symlinks `SAMPLE_xxx__*_f1.png` en `Data/images/bpof/{SPECIES}/` (prefijo de placa para evitar colisiones de nombre entre replicas). Excluidos 4 FOV de `BAD_IMAGES` (`C_QUITENSIS_0047` X11800, `C_QUITENSIS_0044` X17200 en disco, `D_ANTARTICA_0036`, `T_REPENS_0042`).
- **train:** 8 junctions `MetricLearning/data/processed/train/{SPECIES}` -> `bpof/{SPECIES}`.

## Bloqueado

Entrenamiento (detector ViT-denso / DML) hasta que existan etiquetas `.seg` revisadas.

---

## 2026-08-31 16:40 (UTC−04:00) — intérprete

Cursor lanzó `app.py` con **uv cpython 3.14** (`No module named 'PyQt5'`). PyQt5 está en `.venv` (3.11), no en uv.

- `app.py` ahora **sale** si `sys.prefix` no es `AntarSeeds/.venv`.
- Intérprete del workspace: `.venv\Scripts\python.exe`.
- Arranque: `.\start.ps1` o `.\start.bat` desde la raíz del repo.