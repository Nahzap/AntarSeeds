# Plan de implementación — enlaces, máscaras y MetricLearning

| Campo | Valor |
| --- | --- |
| Fecha | 2026-08-31 16:14 (UTC−04:00) |
| Código | no |
| Clases | `Data/NOMENCLATURA_DATA.csv`, sin renombrar |

`T_officinale` · `J_Bufonius` · `C_Quitensis` · `D_Antartica` · `T_Repens` · `P_Pratensis` · `P_Annua` · `L_Vulgare`

---

## 1. Imágenes: enlaces, no copias

Los PNG (~14 GB) se quedan en `F:\MICROSCOPIA\ANTARSEEDS`. AntarSeeds y MetricLearning solo enlazan.

```
AntarSeeds/Data/images/          # gitignored
  raw/SAMPLE_xxx  → junction a F:\...\SAMPLE_xxx
  bpof/{SPECIES}/ → symlink de un PNG por FOV (plano de trabajo)
  exclude/        → BAD_IMAGES (fuera de bpof)
```

- `raw`: 16 junctions (`mklink /J`). JSON se leen ahí.
- `bpof`: un plano por FOV (`f1` / BPOF). Carpetas = `SPECIES`. Ahí se separan taxa, malas y stacks.
- Fuera de `bpof`: `f0`/`f2`, JSON huérfano SAMPLE_001 `X12600`, FOVs de `BAD_IMAGES.csv` (corregir coord. de `C_QUITENSIS_0044`).
- Split **por placa** (`SAMPLE_xxx`), nunca por archivo.

MetricLearning: `data/processed/{train,val,test}/{SPECIES}/` → symlinks a `bpof`. Los `.seg` sí se escriben (livianos), p. ej. `AntarSeeds/Data/annotations/{SPECIES}/`.

Windows: Modo de desarrollador para file symlinks. Si `os.symlink` falla, MetricLearning copia: **prohibido**. Probar un enlace antes del lote.

---

## 2. MetricLearning — viabilidad

MeliVision ya encadena **U²-Net → `.seg` → crops on-the-fly → DINOv2 / ViT-denso**. Se reusa; no se reimplementa. Datos de semillas por enlace. No se usa el checkpoint de polen.

**Sirve:** U²-Net (propuestas de objeto, no “polen”); `.seg`; GUI de contornos; `dataset_manager` con symlinks; ViT-denso (tras `.seg` buenos); AnalogyNet (tras detector).

**No sirve tal cual:** `best_detector.pth` 2026-06-13 (polen); perfil `balanced` (circularidad 0,32 / aspect 2,8 tira semillas alargadas); split 70/15/15 al azar (fuga por solape ~93 %); `max_load_side: 1536` frente a 2590×1942; alimentar el stack f0–f2 en vez de solo `bpof`.

Hace falta un perfil **semilla** (circularidad baja, aspect alto, áreas en px con ~10 µm/px). Eso es config.

---

## 3. Orden

1. Enlaces `bpof/{SPECIES}` + split por `SAMPLE`.
2. U²-Net → `.seg` propuestos.
3. Revisión GUI (fibras, solapes). Ahí termina el *gold set* mínimo.
4. Entrenar ViT-denso (saliency = unión de contornos).
5. DML AnalogyNet, 8 clases = `SPECIES`.
6. Inferencia: instancias → área px × $s_x s_y$.

Hasta el paso 3 no se entrena.

**Veredicto:** viable como toolchain de máscaras y entrenamiento. No es plug-and-play: retuning morfológico, split por placa, sin `.pth` de polen.
