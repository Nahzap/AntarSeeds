# Registro de ediciones — AntarSeeds

Un solo diario. Entradas nuevas **arriba**. Fecha local UTC−04:00. Sin planes largos.

Taxa (nombres CSV, no binomiales): `T_officinale` · `J_Bufonius` · `C_Quitensis` · `D_Antartica` · `T_Repens` · `P_Pratensis` · `P_Annua` · `L_Vulgare`.

---

## 2026-09-02 (16) — El encuadre no es un objeto; el borde es el umbral

`Incl: 5 029 780` era `2590×1942`: el grano de `T_officinale` tocaba (0,0) y `fill_holes` invertía el fondo. Un CLOSE extra en la máscara verde dilataba el borde.

`fill_holes` ahora inunda desde un marco de 1 px (el origen de la imagen puede ser objeto). El buscador no dilata: el borde es el isocontorno del umbral objeto/fondo. El rectángulo del encuadre se descarta al resolver, al acumular y al aplicar. `grow_mask_with_saliency` era un segundo método; se eliminó.

Cada arranque vacía `logs/` y abre `antarseeds_YYYY-MM-DD_HH-MM-SS.log`.

---

## 2026-09-02 (15) — Trazo izquierdo: U²-Net cierra el área

En `T_OFFICINALE_0001` el recorte sobre la textura (arriba-izquierda) era todo saliente (`sal 1.0 / 1.0`). Eso se trataba como fallo y bloqueaba el etiquetado. Es el interior del grano.

El click izquierdo ya no lanza U²-Net por punto. Se juntan puntos al arrastrar; al soltar, un recorte del trazo y **una** inferencia. Se cierra el componente de mayor puntaje que cubre el trazo. CLAHE sigue siendo solo el preproceso (IoU del mapa). Un recorte local lleno se acepta; el rectángulo del encuadre entero no.

---

## 2026-09-02 (14) — El click no se cuelga en T_officinale

`T_OFFICINALE_0004`: el recorte de 990 px separaba (sal 0.79). Ampliar a 1584/2178/marco devolvía sal 0 en la semilla y **tiraba** el recorte bueno. Luego el fallo relanzaba U²-Net solo. La cola quedaba en «ocupado» y la imagen no se segmentaba.

Como máximo dos inferencias (radio → encuadre si truncó). Si ampliar pierde el objeto, se conserva el primero. El registro ya no restaura `crop_radius` (no es de la especie). Un fallo no dispara otro click automático; sí vacía la cola.

---

## 2026-09-02 (13) — Orquestación: sin módulos sueltos ni dos umbrales

`clahe.py` y `object_background.py` eran satélites con varias funciones y un segundo cálculo del umbral (resolver + extractor). Eso no es orquestación.

CLAHE vive en `SalientObjectDetector` (único preproceso de U²-Net). La comparación objeto/fondo vive en `seeded_contour` (dueño del buscador). `resolve_seeded_object` mide una vez y pasa ese umbral a `u2net_seeded_contour`. Click y lote no pueden divergir. La luminancia solo se registra; no decide. No hay fallback al blob más cercano.

La cola de clicks queda en 2 (no 106). «Detalle en terminal» enciende INFO de objeto/fondo.

---

## 2026-09-02 (12) — Camino único: CLAHE + U²-Net

El click de `T_officinale` devolvía `Incl: 5 025 836` (el encuadre). Tres atajos distintos lo causaban: recorte = marco (radio aprendido 2072), `max_crop_fill=1` + aceptar el rectángulo, y Canny «snap a borde». El ajuste Lab era un segundo método.

Queda un solo camino: CLAHE en L → U²-Net → contorno sembrado. El click no mueve la semilla con Canny. El aprendizaje ya no escribe `crop_radius` (eso no es tamaño de grano). El rectángulo del recorte no es un cuerpo.

---

## 2026-09-02 (11) — Borde objeto/fondo a resolución nativa

U²-Net entra a 320 px. En `T_officinale` el grano llena el FOV: el mapa de saliencia queda borroso y el umbral no coincide con el borde real contra el fondo de microscopio.

Tras el contorno sembrado, `snap_mask_to_color_edge` reasigna la banda del borde en Lab: color del interior vs color del fondo. Si no hay contraste (imagen negra de test), no toca nada. El `approxPolyDP` de cuerpos grandes deja de usar 0.25 % del perímetro (borraba ~20 px de perfil). El crecimiento por saliencia usa la mediana del interior, no la media del encuadre.

---

## 2026-09-02 (10) — Click de muestra: cuerpos que llenan el encuadre

`T_officinale` llega a lado ~1900 px en un marco de ~1942. El recorte chico y el guarda «si el bbox es el 96 % del recorte, es fondo» descartaban el click aunque `for_demonstration` ya hubiera abierto min/max área. Watershed (`split_touching`) partía el cuerpo texturado.

El click ahora usa el encuadre completo, no parte por contacto, no aplica el 96 % ni el fill del recorte, y si el grano toca el borde se acepta. Esos números del panel («lado máx 1696») son solo del lote.

---

## 2026-09-02 (9) — El click de muestra no tiene tope de área

El calibrador del (8) cerró el círculo al revés: tras etiquetar `C_Quitensis`, `max_area` bajaba a ~682 kpx² y el click de U2-Net rechazaba cualquier grano más grande. Sin poder marcarlos, no había forma de enseñar semillas mayores.

**Cambio.** `SeededParams.for_demonstration()` abre área, circularidad, aspecto y lado. El `ManualContourWorker` (click) usa esa envolvente. Threshold y k se conservan (son de saliencia). El lote / «Probar aquí» / «Segmentar» siguen con los topes aprendidos. Al aceptar el cuerpo grande (A), la envolvente de la clase crece.

`max_crop_fill` del recorte local sube a 0.95 en el click para que un grano que llena el recorte no se descarte como “fondo”.

---

## 2026-09-02 (8) — El etiquetado manual calibra el automático

223 tests pasan (1 omitido).

El lote de `J_Bufonius` devolvía `0 semillas → 0 cuerpos` porque los topes venían de `C_Quitensis` (`min_area=80 000`). Un grano de `J_Bufonius` mide ~60 000 px²: el buscador los descartaba a todos. Configurar área/circularidad/aspecto a mano por especie era el trabajo que el usuario ya hace al etiquetar.

**Cambio.** Cada cuerpo que aceptas (ROI aplicado, o el primer `.seg` de una clase) mide área, circularidad, aspecto y lado. Esos valores se acumulan **por clase** y se escriben en los mismos controles que usa `Segmentar` / `Probar aquí`. Un click en `J_Bufonius` baja `min_area` de 80 000 a ~22 000; un click en `C_Quitensis` lo sube a ~127 000. Al volver a una clase se recuperan sus topes.

**Política.** El área se re-escala a la especie. Los límites de forma solo se **relajan**: un círculo perfecto no puede subir la circularidad mínima a 0.75 y dejar fuera semillas menos redondas. El radio de recorte crece si el cuerpo lo pide.

**Persistencia.** `data/annotations/class_envelopes.json`. Sobrevive al reinicio. Checkbox `Aprender de mi etiquetado` (on por defecto) en el panel de ROI.

Módulo: `src/grain_detection/param_learning.py`. Tests: `tests/test_param_learning.py`.

---

## 2026-09-02 (7) — Efectos de borde: fuera los lados rectos

194 tests pasan (1 omitido).

Un contorno con un lado perfectamente recto no es el perfil de una semilla. Había **dos** causas distintas que producían la misma recta:

**1. Watershed cortando un cuerpo real.** Una semilla alargada tiene el máximo de distancia desdoblado, así que se partía por la mitad. Nuevo `merge_shallow_splits()`: un corte solo es legítimo si el istmo es más delgado que los cuerpos. Se compara la silla de la distancia en la frontera con el pico de cada región:

```
silla <= waist_frac * min(pico_a, pico_b)  ->  corte real
```

Dos semillas tangentes dan silla/pico ≈ 0.5 (se corta). Una semilla alargada con doble máximo espurio da ≈ 0.95 (no se corta). `distance_cores` valida con el mismo criterio, así que tampoco emite dos semillas donde hay un cuerpo.

**2. Contorno truncado por el recorte, aceptado como resultado.** El lado recto era el canto del recorte. `touches_interior_crop_border()` distingue un lado del recorte que **no** es borde del encuadre (artificio) de uno que sí lo es (el cuerpo sale de la imagen, corte real). Los radios ahora escalan hasta cubrir el encuadre completo, así que un objeto legítimo nunca se pierde, y un contorno truncado ya **no** se devuelve como `fallback`.

**Controles nuevos:** `Cintura` (0.20–0.99, def. 0.80) y `Descartar cuerpos en el borde` (def. off; con el solape de placa ~93 % la semilla aparece completa en el FOV vecino, así que activarlo mejora el banco sin perder datos).

**Medido con U²-Net real** sobre 4 semillas (2 tangentes): lado recto máximo 7.3 % del perímetro (normal para un polígono), tangentes separadas, y con `crop_radius=90` los cuerpos siguen completos en vez de truncarse.

Tests: `tests/test_edge_artifacts.py`.

---

## 2026-09-02 (6) — Separar semillas en contacto

177 tests pasan.

**Causa raíz.** `peak_seeds` tomaba **un centroide por componente conectado**. Dos semillas que se tocan son un solo componente, así que la semilla automática caía en **la unión** y salía un cuerpo fusionado. Por eso el click manual funcionaba (el humano apunta al centro de cada semilla) y el lote no: no era un problema de configuración, era el sembrador.

**Solución.** El sembrador automático ahora hace lo que hace un humano: apuntar al punto más interior de cada cuerpo.

- `distance_peaks()` — máximos locales de la transformada de distancia. Un umbral global no basta: con 20 px de solape el cuello queda al 51 % del máximo y los núcleos siguen unidos. Los máximos locales sí distinguen los dos centros.
- `watershed_regions()` + `region_containing()` — el contorno se recorta a la región del cuerpo sembrado.
- `grow_mask_with_saliency(..., allowed=)` — la expansión queda confinada a esa región; antes crecía sobre la vecina.

**Controles nuevos** (afectan click y lote por igual): `Separar contacto` y `Núcleo` (prominencia mínima, 0.20–0.80, def. 0.45). También en `config.yaml`, en el preset `seed` y en la cabecera del `.seg`.

**Medido con U²-Net real**, dos semillas ovaladas pegadas: 1 cuerpo de 396×200 px → 2 cuerpos de 201×186 y 196×177 px. Semillas separadas: sin cambio (2 → 2).

Tests: `tests/test_touching_seeds.py` (19 casos, incluido el comportamiento anterior como referencia del fallo).

---

## 2026-09-02 (5) — Banco de datos: limpiar marcadas / limpiar todo

158 tests pasan.

Fila nueva **Banco de datos (.seg)** en la sección de configuración, con el total vivo (`N .seg en M clase(s) — data/annotations/`):

- **Limpiar clases marcadas** — borra los `.seg` de las clases con tick; el resto no se toca.
- **Limpiar TODO** — vacía el banco completo para etiquetar de cero.

Ambos entran a `_clear_annotations(class_filter)`: un solo borrado, dos alcances. Sobre `src/grain_detection/annotator.py`:

- `count_annotations(root, class_filter)` — conteo por clase; ignora `.seg.bak` para que un backup no finja dato.
- `delete_annotations(root, class_filter, remove_backups=True)` — borra `.seg` + `.seg.bak`, retira carpetas de clase vacías, devuelve `seg_removed / bak_removed / per_class / errors`.

Efectos que evitan estado inconsistente: se invalida `contour_registry.json` de los tres splits (su fast path devolvería conteos viejos), se descarta el estado en memoria de la imagen abierta (granos, historial, ROI, prueba) y los botones quedan bloqueados mientras hay una segmentación en curso. Confirmación previa con el desglose exacto por clase.

Tests: `tests/test_annotation_bank.py`.

---

## 2026-09-02 (4) — Terminal paso a paso y layout legible

141 tests pasan.

**El autoetiquetador ya no es una caja negra.** `setup_logger("gui_app")` configuraba **solo** ese logger con `propagate=False`, así que `src.grain_detection.*` no tenía handlers y no escribía nada en consola. Nuevo `setup_root_logging()` (llamado desde `gui_app`) manda todo a stdout **y** a `logs/antarseeds_<fecha>.log`.

Qué se ve ahora, por defecto una línea por imagen:

```
[27/1442] C_Quitensis/C_QUITENSIS_0006.png — segmentando...
buscador ROI: 5 semillas -> 4 cuerpos (1 fusionados por NMS, 0 repetidas) en 76 ms
[27/1442] C_Quitensis/C_QUITENSIS_0006.png -> 4 cuerpos en 669 ms | .seg: ... | acumulado: 112 granos
```

Con **Detalle en terminal** marcado, una línea por semilla con el motivo exacto del descarte:

```
semilla (340,359) r=200: descartada — aspecto 2.82 > 2.0 (max_aspect_ratio)
buscador ROI: 1 semillas -> 0 cuerpos | descartes: aspecto 5.17 > 2.0 (max_aspect_ratio) x1
```

`validity_reason()` devuelve qué filtro cortó el candidato; `passes_validity_filters` ahora es un envoltorio.

**Layout.** El tab vive dentro de un `QScrollArea`, así que el `QSplitter` vertical nunca se acotaba: por eso "el slider no bajaba más" y las estadísticas quedaban fuera. Cambios: splitter vertical eliminado; parámetros **plegables** (`▾ Parámetros`, 837 → 705 px de alto mínimo); barra de estadísticas sin `setMaximumHeight(80)`, en dos líneas propias (totales + granos por clase); panel derecho de grano/ROI con scroll propio para que no imponga su altura.

**`Probar aquí` movido junto a la imagen** (antes estaba en la sección de config, que se te iba de pantalla) con `Quitar prueba` y un diagnóstico que dice qué control ajustar según el resultado.

Tests: `tests/test_autolabel_logging.py` y ampliación de `tests/test_segmentation_scope_and_cancel.py`.

---

## 2026-09-02 (3) — Cancelar lote, tick obligatorio, calibración del panel

123 tests pasan.

- **Cancelar**: `annotate_directory` acepta `cancel_check`; corta **entre imágenes**, nunca a mitad de un `.seg`. Botón rojo `Cancelar` activo solo mientras corre; `stats["cancelled"]` se propaga al resumen y al log.
- **El tick manda en cualquier alcance**: `Train + Val + Test` pasaba `class_filter=None` y segmentaba las 8 clases. Ahora ambos alcances exigen tick y el diálogo de sobrescritura lista las clases afectadas.
- **`Probar aquí`**: corre el buscador ROI en la imagen abierta con los valores actuales y dibuja las propuestas en **blanco** sin escribir `.seg`. Caduca al mover un filtro o cambiar de imagen. Es el lote aplicado a un encuadre: mismas semillas, mismos filtros.
- **`Threshold` no refrescaba el panel** (faltaba en `_filter_widgets`). La lectura efectiva ahora separa **semilla** (threshold + k + radio) de **cuerpo válido** (área, lado, aspecto, circularidad).

Tests: `tests/test_segmentation_scope_and_cancel.py`.

---

## 2026-09-02 (2) — Limpieza: código muerto, redundancias y bugs

`contour_analysis_tab.py` 3407 → 3077 líneas. 114 tests pasan.

**Bugs corregidos**

- **Caché del detector indexado por config completa**: cada movimiento de slider recargaba los pesos de U²-Net en GPU. Ahora la clave es `(model_type, input_size, device)`.
- **Estadísticas buscaban `.seg` junto a la imagen** (`f.with_suffix`) mientras el resto usa `data/annotations/<clase>/`. La barra inferior siempre marcaba 0. Helpers `seg_path_for` / `class_images`.
- **`Limpiar ROI` era un reset parcial**: sobrevivían traza de semillas, cola pendiente y última semilla. Un solo `_reset_manual_roi_session`.
- **UNDO con click derecho era inalcanzable** tras hacer ROI el único modo. Reordenado: sin ROI en curso → eliminar grano cercano, si no hay → deshacer.
- **`detect_exhaustive` cambiaba de algoritmo** (volvía al detector de polen) con `object_finder: roi_seed`. Ahora relaja `adaptive_k` sobre el mismo buscador.

**Unificaciones**

- `resolve_seeded_object` (semilla → cuerpo) es el único camino; lo usan el click y `propose_objects`. El lote ejecuta la red **por recorte** vía `saliency_provider`, igual que el click.
- `AnnotationWorker` + `AllSplitsAnnotationWorker` → un worker con lista de splits; `accumulate` es pura y testeada.
- `_on_save_registry` + `_save_registry_silent` → `_save_registry(interactive=)`.
- Dos textos de ayuda divergentes → `_interaction_hint()`.
- Eliminados: `_on_manual_contour_result/_error` (~100 líneas muertas), `edit_mode_combo`, `_on_edit_mode_changed`, `_edit_mode`, `_is_manual_roi_mode`, `_fill_holes_binary` (usa `seeded_contour.fill_holes`), helpers duplicados del worker.

**Tests nuevos**: `test_contour_tab_paths.py` (rutas .seg, caché, resumen) y ampliación de `test_seeded_contour.py` (click == lote, coords globales, truncamiento).

---

## 2026-09-02 — Buscador ROI unificado (autoetiquetador weak)

**Problema.** Los sliders de segmentación no gobernaban el click de ROI. El lote usaba `extract_grains_from_saliency` (detector de polen). Había un modo “Auto asistido” paralelo. Crop padding/size DML se confundían con el recorte del objeto.

**Cambio.**

- Un algoritmo: `MetricLearning/src/grain_detection/seeded_contour.py` (`u2net_seeded_contour` + `propose_objects`). Click = semilla; lote = picos de saliencia como clicks.
- GUI: solo ROI. Threshold, k, área, circ., aspecto, kernel y **radio semilla** van a `SeededParams`. DML padding/crop etiquetados aparte.
- `PollenGrainDetector.detect()` con `object_finder: roi_seed` (default en `config.yaml`).
- Tests: `tests/test_seeded_contour.py`; GUI en `tests/test_weak_labeler_config.py`.

**Arranque.** `AntarSeeds/start.ps1` o `.\.venv\Scripts\python.exe MetricLearning\app.py`. No uv cpython 3.14.

---

## 2026-08-31 16:40 — Intérprete

Cursor lanzó `app.py` con uv **cpython 3.14** (`No module named 'PyQt5'`). PyQt5 está en `.venv` (3.11).

- `app.py` sale si `sys.prefix` no es `AntarSeeds/.venv`.
- Workspace: `.venv\Scripts\python.exe`.

---

## 2026-08-31 16:32 — Entorno y datos

- venv: `AntarSeeds/.venv`, CUDA RTX 3070, U²-NetP `MetricLearning/models/weights/u2netp.pth`.
- 16 junctions `Data/images/raw/SAMPLE_xxx` → `F:\MICROSCOPIA\ANTARSEEDS\SAMPLE_xxx` (ASCII, no `ANTÁRSEEDS`).
- 684 symlinks BPOF `f1` en `Data/images/bpof/{SPECIES}/`. Excluidos 4 FOV de `BAD_IMAGES`.
- Train: 8 junctions `MetricLearning/data/processed/train/{SPECIES}` → bpof.
- Smoke: Basler `(1942, 2590, 3)`. Calibración cerrada: ~10.04 × 10.30 µm/px.

Entrenamiento DML/ViT bloqueado hasta `.seg` revisados.

---

## 2026-08-31 16:14 — Enlaces, no copias

PNG (~14 GB) permanecen en `F:\MICROSCOPIA\ANTARSEEDS`. Split **por placa**. MetricLearning reusa U²-Net → `.seg` → crops; no se usa el checkpoint de polen.

Detalle: `Docs/2026-08-31_1614_plan_implementacion.md`.

---

## Cómo añadir una entrada

```
## YYYY-MM-DD HH:MM — título corto

Qué cambió y por qué. Archivos clave. Qué quedó bloqueado.
```
