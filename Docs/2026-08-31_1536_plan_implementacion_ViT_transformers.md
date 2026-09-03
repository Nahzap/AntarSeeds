# Plan de implementación — visión artificial con ViT (transformers)

| Campo | Valor |
| --- | --- |
| Proyecto | AntarSeeds |
| Documento | Plan teórico de implementación (aprendizaje con Vision Transformer) |
| Fecha | 2026-08-31 |
| Hora (v1) | 15:36 (UTC−04:00) |
| Revisión | 15:53 — escala 2590×1942 y FOV 2,6×2,0 cm fijados; math en `$$` para preview |
| Alcance | Teoría y plan de trabajo **antes** de entrenar; no incluye código ni hiperparámetros definitivos |
| Estado de las máscaras | **No existen** segmentaciones, cajas ni IDs de instancia |

---

## 1. Propósito

Este documento fija **cómo proceder desde la teoría** para aprender un modelo de visión basado en *transformers* (familia ViT) sobre el acervo actual de AntarSeeds.

La restricción dominante no es la arquitectura: es el **régimen de supervisión**. Hay imágenes, metadatos de captura y una etiqueta de *especie a nivel de placa/muestra*. No hay contornos, máscaras semánticas ni *instance IDs*. Entrenar de inmediato un segmentador supervisado (ViT + cabeza densa, Mask2Former, SAM *fine-tuned*, etc.) **no está justificado**: no hay función de pérdida con ground truth geométrico.

El plan ordena el trabajo en tres capas:

1. **Metrología** — convertir píxeles en micrómetros de forma trazable.
2. **Representación** — aprender un encoder ViT *sin* máscaras (auto-supervisión + clasificación débil).
3. **Geometría** — llegar a instancias y morfometría solo después de un protocolo de etiquetado mínimo, apoyado en propuestas automáticas.

La interfaz de captura ya anticipa el producto final: tabla *Objetos detectados* con *Score* y *Área (px)*. El modelo debe, a término, llenar esa tabla y reportar área/longitud en **unidades físicas**, no solo en píxeles.

---

## 2. Situación general del sistema (auditoría)

### 2.1 Organización del workspace

| Rol | Ruta | Contenido |
| --- | --- | --- |
| Repositorio de trabajo | `C:\Users\askna\Documents\GitHub\AntarSeeds` | `Data/` (calibración, nomenclatura, imágenes malas), `Docs/` (este plan). Aún **no** es un repositorio git inicializado. |
| Acervo de imagen | `F:\MICROSCOPIA\ANTARSEEDS` | 16 carpetas `SAMPLE_001` … `SAMPLE_016`. No debe versionarse en git (~14 GB de PNG). |
| Workspace VS Code/Cursor | `AntarSeeds.code-workspace` | Enlaza ambas raíces. |

No hay código de entrenamiento, ni carpeta de máscaras, ni `dataset.yaml`, ni *splits*. El sistema de captura (UI *Vista de Cámara — Tiempo Real*, modos ROI/contornos) existe como evidencia en `Data/calibracion.jpg`; el *pipeline* de ML está por nacer.

### 2.2 Inventario de captura (31-08-2026)

Conteo sobre disco, no sobre el CSV de nomenclatura.

| Muestra | Taxón (código en archivo) | FOVs con `*_f1.png` | PNG (f0+f1+f2) | `*_focus.json` | `*_position.json` |
| --- | --- | --- | --- | --- | --- |
| SAMPLE_001 | T_OFFICINALE | 37 | 111 | 38 | 38 |
| SAMPLE_002 | T_OFFICINALE | 48 | 144 | 48 | 48 |
| SAMPLE_003 | J_BUFONIUS | 37 | 111 | 37 | 37 |
| SAMPLE_004 | J_BUFONIUS | 35 | 105 | 35 | 35 |
| SAMPLE_005 | C_QUITENSIS | 48 | 144 | 48 | 48 |
| SAMPLE_006 | C_QUITENSIS | 48 | 144 | 48 | 48 |
| SAMPLE_007 | D_ANTARTICA | 47 | 141 | 47 | 47 |
| SAMPLE_008 | D_ANTARTICA | 43 | 129 | 43 | 43 |
| SAMPLE_009 | T_REPENS | 45 | 135 | 45 | 45 |
| SAMPLE_010 | T_REPENS | 48 | 144 | 48 | 48 |
| SAMPLE_011 | P_PRATENSIS | 48 | 144 | 48 | 48 |
| SAMPLE_012 | P_PRATENSIS | 38 | 114 | 38 | 38 |
| SAMPLE_013 | P_ANNUA | 42 | 126 | 42 | 42 |
| SAMPLE_014 | P_ANNUA | 42 | 126 | 42 | 42 |
| SAMPLE_015 | L_VULGARE | 45 | 135 | 45 | 45 |
| SAMPLE_016 | L_VULGARE | 37 | 111 | 37 | 37 |
| **Total** | 8 clases × 2 placas | **688** | **2064** | **689** | **689** |

Resumen:

- **8 especies**, **2 réplicas espaciales** (placas) por especie — diseño factorial correcto para no mezclar *train/test* dentro de la misma placa.
- Cada FOV es un **stack de 3 planos** (`f0`, `f1`, `f2`), RGB 24 bit, **2590 × 1942** px, ~7 MB/archivo, **~13,95 GB** en total.
- Metadatos por FOV: `*_focus.json` (planos Z, métrica de nitidez `S`, archivo BPOF) y `*_position.json` (consigna vs. posición real del host, error en µm).
- **Cero** archivos cuyo nombre sugiera máscara, etiqueta densa, COCO/YOLO o anotación (`mask`, `seg`, `label`, `annot`).

FOVs por clase (suma de las dos placas): T_officinale 85, J_Bufonius 72, C_Quitensis 96, D_Antartica 90, T_Repens 93, P_Pratensis 86, P_Annua 84, L_Vulgare 82. El desbalance entre clases es **leve** (~72–96 FOVs). El desbalance **dentro** del FOV (número de semillas, solapes, aristas, fondos vacíos) no está medido: no hay instancias.

### 2.3 Nomenclatura y taxones

Fuente: `Data/NOMENCLATURA_DATA.csv`. Los nombres de clase son **exactamente** estos códigos. No se sustituyen por binomios.

| Nº | SPECIES | FOLDER_CODES |
| --- | --- | --- |
| 1 | T_officinale | SAMPLE_001, SAMPLE_002 |
| 2 | J_Bufonius | SAMPLE_003, SAMPLE_004 |
| 3 | C_Quitensis | SAMPLE_005, SAMPLE_006 |
| 4 | D_Antartica | SAMPLE_007, SAMPLE_008 |
| 5 | T_Repens | SAMPLE_009, SAMPLE_010 |
| 6 | P_Pratensis | SAMPLE_011, SAMPLE_012 |
| 7 | P_Annua | SAMPLE_013, SAMPLE_014 |
| 8 | L_Vulgare | SAMPLE_015, SAMPLE_016 |

Patrón de archivo:

```text
{TAXON}_{NNNN}_X{x}um_Y{y}um_f{0|1|2}.png
{TAXON}_{NNNN}_X{x}um_Y{y}um_focus.json
{TAXON}_{NNNN}_X{x}um_Y{y}um_position.json
```

La etiqueta de clase está **en el nombre y en la carpeta**, no en polígonos. Es supervisión **a nivel de imagen/placa**, y además es **débil respecto al contenido**: un FOV puede contener 0–N semillas, restos, fibras, solapes y fuera de foco. La clase de la carpeta no afirma que *cada* objeto del marco sea una semilla de esa especie; afirma que la *placa* corresponde a ese taxón.

### 2.4 Rejilla, foco y calidad de captura

- Rejilla típica (p. ej. SAMPLE_002/004/010): X ∈ [10000, 19000] µm (paso mediano **1800 µm**), Y ∈ [10000, 19800] µm (paso mediano **1400 µm**). SAMPLE_001 usa una malla más densa (paso X **1300 µm**, Y **1000 µm**) y un rango Y más corto.
- Protocolo de foco: tres capturas alrededor del plano de trabajo; **`f1` está marcado `is_bpof: true` en el 100 % de los JSON auditados**. No es una elección *a posteriori* entre los tres planos: el stack se define como (bajo, BPOF, alto). La métrica `S` a veces **discrepa** (el máximo `S` cae en `f0` o `f2` mientras `is_bpof` sigue en `f1`).
- Paso Z típico ~10–30 µm cuando el eje está lejos del tope; en el límite inferior del stage (`z ≈ 0,021 µm`) **f0 y f1 pueden compartir la misma Z**. El rango BPOF observado es ~0,02–90 µm.
- `fov_verify_passed` es **false** en todos los JSON revisados (el verificador de FOV no corrió o no se persistió).
- Error de posicionamiento del host: de ~15 µm (mejor caso SAMPLE_001) hasta medias de **~280–360 µm** en otras placas. Eso es una fracción no trivial del paso de malla y debe considerarse al pensar en mosaicos o en *non-maximum suppression* entre FOVs vecinos.

Inconsistencias puntuales (higiene, no bloqueantes):

| Hallazgo | Detalle |
| --- | --- |
| JSON huérfano | `SAMPLE_001/T_OFFICINALE_0002_X12600um_Y10000um_focus.json` **sin** PNG. El índice `0002` se reutiliza en otra coordenada (`X11300um`) que sí tiene imagen. |
| Lista de malas vs. disco | `Data/BAD_IMAGES.csv` cita `C_QUITENSIS_0044_X11800um_Y19800um`, pero en disco `0044` está en **`X17200um_Y19800um`**. `X11800um_Y19800um` corresponde a `0047`. Hay que corregir el CSV o filtrar por `(SAMPLE, índice)`, no por *string* de coordenada. |
| Malas que sí existen | `005/0047`, `007/0036`, `010/0042` (tres planos cada una). Deben excluirse de todo aprendizaje. |
| Nota `calibracion.txt` (2560) | Redondeo de memoria. **Resolución operativa = 2590 × 1942** (PNG del acervo). La nota en `Data/` queda corregida a 2590. |

---

## 3. Calibración geométrica — escala de trabajo

Fuentes: `Data/calibracion.txt`, `Data/calibracion.jpg` (captura real de la regla en el FOV de trabajo), `Data/ruler_calibration.jpg` (*Microscope Micrometer Calibration Ruler*, PET).

### 3.1 Datos de trabajo (cerrados)

| Magnitud | Valor | Origen |
| --- | --- | --- |
| Sensor / frame | **2590 × 1942 px** | PNG del acervo (la nota 2560 era un redondeo) |
| FOV observado | **2,6 cm × 2,0 cm** (X × Y) | Imagen de calibración sobre la regla PET |
| FOV en µm | 26 000 µm × 20 000 µm | conversión cm → µm |
| *Target* | regla micrométrica PET, DIV = 0,1 mm | `ruler_calibration.jpg` |

La escala al píxel es el cociente FOV / resolución. No hace falta otra hipótesis de unidades:

$$
s_x = \frac{26000~\mu\mathrm{m}}{2590~\mathrm{px}} = 10{,}0386~\mu\mathrm{m/px}
$$

$$
s_y = \frac{20000~\mu\mathrm{m}}{1942~\mathrm{px}} = 10{,}2987~\mu\mathrm{m/px}
$$

Valores de trabajo (redondeo usable):

| Eje | Escala | Uso |
| --- | --- | --- |
| X | **10,039 µm/px** | longitudes horizontales |
| Y | **10,299 µm/px** | longitudes verticales |
| Área | **103,38 µm²/px²** | $A = N_{\mathrm{px}} \cdot s_x \cdot s_y$ |

Hay anisotropía de ~2,6 % entre ejes. Un ViT con *patches* cuadrados **no la corrige**: opera en píxeles. La conversión a micrómetros es **post-proceso** de la máscara, eje a eje. No usar un único “10 µm/px” genérico si se van a reportar áreas.

La tilde de la nota (`~[2.6*2.0]`) queda como incertidumbre de la medida del FOV, no como duda sobre el orden de magnitud. Si más adelante se cuentan ticks de 0,1 mm sobre un PNG 2590×1942, se refina el tercer decimal; **no se cambia el factor de escala**.

### 3.2 Consecuencia inmediata: solape de la malla

Con FOV 26 mm × 20 mm y pasos típicos de **1,8 mm (X)** y **1,4 mm (Y)**:

$$
\text{solape}_x = 1 - \frac{1{,}8}{26} \approx 93\%,\qquad
\text{solape}_y = 1 - \frac{1{,}4}{20} \approx 93\%.
$$

SAMPLE_001 es aún más denso (1,3 mm × 1,0 mm). Los 688 FOVs **no** son 688 escenas independientes: la misma semilla entra en muchos marcos. Eso obliga a (a) *split* por placa, nunca por PNG; (b) NMS espacial en µm usando el stage al contar individuos.

### 3.3 Modelo geométrico píxel → muestra

Sea $(u,v)$ el píxel (origen: esquina superior izquierda del PNG, $u$ a la derecha, $v$ hacia abajo) y $(x,y)$ el plano de la muestra en µm, en el sistema del host. Con $u_0 = 0$, $v_0 = 0$ si el origen del stage se amarra a esa esquina:

$$
\begin{aligned}
x &= x_{\mathrm{stage}} + (u - u_0)\, s_x \\
y &= y_{\mathrm{stage}} + (v - v_0)\, s_y
\end{aligned}
$$

Para morfometría **dentro** del FOV (área, eje mayor, circularidad) bastan $s_x$ y $s_y$. Distorsión de lente y homografía entran solo si se mosaica la placa. El mosaico es viable en teoría (solape ~93 %), pero el error de host (hasta ~0,3 mm) hay que meterlo como incertidumbre de asociación entre FOVs, no como si el stage fuera exacto.

### 3.4 Qué queda para la Fase 0

La escala de trabajo **ya está**. Fase 0 no es “descubrir si son cm o mm”. Es persistirla:

1. Escribir `Data/calibration.json` con 2590×1942, FOV 26×20 mm, $s_x$, $s_y$, fecha y fuente (`calibracion.jpg` + PNG del acervo).
2. Opcional: conteo de ticks 0,1 mm sobre un PNG full-res para acotar el `~` del FOV.
3. Contrato de coordenadas: origen top-left, Y de imagen hacia abajo, vs. signo del eje Y del stage (`error_y_um`, `move_dir_y` en los JSON).
4. Manifiesto de FOVs usables (excluir malas + JSON huérfano).

---

## 4. Qué es el problema de aprendizaje (sin máscaras)

### 4.1 Tareas que *parecen* el objetivo, y su supervisión real

| Tarea deseada | Supervisión disponible hoy | ¿Se puede entrenar un ViT “clásico”? |
| --- | --- | --- |
| Clasificar especie de la **placa** | Sí (8 clases, 2 placas) | Sí, con fugas si se parte mal el *split* |
| Clasificar especie de **cada semilla** | No (no hay instancias) | No de forma honesta |
| Segmentación semántica (fondo / semilla) | No | No supervisada; sí *zero-shot* / *bootstrapping* |
| Segmentación de instancia (semilla nº k) | No | No supervisada |
| Detección (cajas + clase) | No | No supervisada |
| Morfometría (área, eje, color) | Escala µm/px **lista**; faltan máscaras | No (hace falta $N_{\mathrm{px}}$ de cada instancia) |

El cuello de botella es **la falta de geometría etiquetada**, no la falta de *transformers*.

### 4.2 Por qué un ViT “de libro” no se alimenta con este disco

Un ViT de clasificación (Dosovitskiy et al., 2020) parte la imagen en *patches*, los proyecta a *tokens* y usa auto-atención. Eso **sí** puede entrenarse con la etiqueta de carpeta: “esta imagen es `C_QUITENSIS`”. Lo que **no** puede hacer, sin más, es:

- separar dos semillas del mismo color que se tocan;
- ignorar una fibra o un resto;
- devolver un contorno usable en *Área (px)*;
- generalizar a una novena especie.

Un ViT de segmentación (SETR, SegFormer, Mask2Former) necesita **mapas densos** o *queries* con máscaras. No existen.

Conclusión operativa: el primer modelo ViT de AntarSeeds no es un segmentador. Es un **encoder de representación** y, en paralelo, un **clasificador débil de FOV**. La segmentación es un segundo sistema, alimentado por propuestas + un *gold set* pequeño.

### 4.3 Fenomenología visual que el modelo tendrá que absorber

Observación cualitativa sobre BPOF (`f1`) de varias clases:

- Fondo relativamente uniforme (verde-gris claro, tipo bandeja/placa), buen contraste de color frente a semillas ocre/ámbar — favorece umbralizado y SAM.
- **Solape** frecuente (T_officinale, C_quitensis, D_antarctica): el problema real es *instance*, no *semantic*.
- Texturas de largo alcance (costillas, aristas, testa lisa). Eso favorece atención global (transformer) frente a un CNN muy local.
- Profundidad de campo estrecha: bordes y aristas se desenfocan. El stack `f0–f2` es información, no ruido, pero **f1 no siempre maximiza `S`**.
- Contaminantes: fibras, polvo, fragmentos. Una máscara “todo lo que no es fondo” **no** es una máscara de semilla.

### 4.4 Unidad atómica de dato

Definiciones que el proyecto debe adoptar y no mezclar:

| Unidad | Qué es | Etiqueta actual |
| --- | --- | --- |
| Placa / `SAMPLE_xxx` | Réplica física | Especie |
| FOV | Un (x,y) de la malla, 3 PNG | Misma especie (heredada) |
| Plano | `f0`/`f1`/`f2` | Ninguna propia |
| Instancia | Una semilla (aún no existe) | — |

**Regla de *split*:** nunca poner FOVs de la **misma placa** en train y en test. El solape espacial y el mismo lote de semillas inflarían cualquier métrica. El diseño natural es *leave-one-sample-out* por especie (entrenar en la placa A, testear en la placa B, y al revés). Con solo 2 placas no hay *validation* independiente de verdad: la validación debe ser *k-fold* entre placas **o** un hold-out de FOVs *dentro* de train, sabiendo que está correlacionado.

---

## 5. Marco teórico: transformers de visión aplicables aquí

### 5.1 Encoder ViT (clasificación / representación)

El ViT recorta la imagen en parches $P \times P$, añade *positional embeddings* y apila bloques de atención. Para 2590 × 1942:

- Un ViT-B/16 nativo exigiría ~ (162 × 121) ≈ **19 600 tokens** — inviable en memoria y absurdo para un objeto que ocupa una región compacta.
- La práctica correcta es **teselar** (tiles 224 / 256 / 518) con solape, o redimensionar **solo** para clasificación débil de FOV (perdiendo el detalle de aristas).
- *Patch size* pequeño (8–16) conserva costillas y pelos; *patch size* grande (32) las borra. Para este material, $P=14$ o $16$ sobre tiles de 512 px es el orden de magnitud razonable.

Preentrenos que importan más que “ViT from scratch” (688 FOVs es **poco** para un ViT aleatorio):

| Familia | Qué aprende | Encaje AntarSeeds |
| --- | --- | --- |
| ImageNet supervisado (DeiT, ViT-B) | Objetos naturales | Arranque aceptable, sesgo de dominio fuerte |
| MAE (He et al.) | Reconstruir parches | Bueno para textura de testa |
| DINO / **DINOv2** | Features densas, *objectness* | El más útil *antes* de tener máscaras: atención y PCA de tokens ya “dibujan” objetos |
| SAM / SAM2 (ViT-H + decodificador) | Máscaras *zero-shot* | Propuestas de instancia, no clasificador de especie |

Recomendación teórica de encoder: **DINOv2 ViT-S/14 o ViT-B/14** como *backbone* congelable. No porque sea moda, sino porque (a) produce mapas densos usables para *discovery*, (b) admite *linear probe* de especie con pocas imágenes, (c) más adelante se enchufa a Mask2Former / un decodificador ligero.

### 5.2 De tokens a geometría (cuando existan etiquetas)

Tres familias, de menor a mayor compromiso con instancias:

1. **SegFormer (MiT)** — encoder jerárquico tipo transformer, decodificador MLP. Bueno para semántica fondo/semilla. Débil en separar instancias que se tocan.
2. **SETR / DPT / ViT-Adapter** — ViT “plano” + decodificador denso. Mismo límite de instancia.
3. **Mask2Former / Mask DINO** — *queries* de objeto + máscaras. Es el destino si el producto es “lista de semillas”. Requiere IDs de instancia.
4. **DETR / DINO-DETR / RT-DETR** — detección en cajas. Útil como paso intermedio (anotar cajas es más barato que polígonos), insuficiente para área precisa.

Ninguna de estas se entrena en Fase 1.

### 5.3 Caminos *sin* máscara humana (teoría de supervisión débil)

Estos métodos **no reemplazan** un *gold set*; reducen su tamaño.

1. **Class Activation / atención ViT.** Tras un clasificador de especie, los mapas de atención o *Chefer relevance* destacan regiones “que apoyan la clase”. Sirven como *heatmap*, no como contorno metrológico. Útiles para *sanity check* (“¿el modelo mira la semilla o el fondo de la placa?”).
2. **Object discovery.** LOST, TokenCut, CutLER: particionan el grafo de similitud entre tokens DINO. En fondos lisos suelen extraer el objeto dominante; fallan con N semillas y con aristas finas.
3. **SAM / SAM2 *everything* o *points*.** En este material (alto contraste, fondo homogéneo) es el generador de **pseudo-máscaras** más plausible. Hay que filtrar por área, solidez, excentricidad y color para tirar fibras y recortes de borde.
4. **Visión clásica como prior.** Otsu / *grabcut* / watershed. Si ya segmenta FOVs de semillas aisladas, el ViT se justifica sobre todo para **instancias en contacto** y para **clase**.
5. **MIL (Multiple Instance Learning).** El FOV es una *bag*; los tiles son instancias. La etiqueta de especie se predice si *alguna* instancia es semilla. Encaja con FOVs medio vacíos, pero no da contornos.

La secuencia teórica honesta es: **clásico + SAM → pseudo-etiquetas ruidosas → corrección humana mínima → entrenamiento denso**. El ViT entra como encoder y, después, como segmentador; no como primer etiquetador mágico.

### 5.4 Qué hacer con los tres planos de foco

No tratar `f0,f1,f2` como tres clases. Opciones, de más simple a más rica:

| Estrategia | Idea | Riesgo |
| --- | --- | --- |
| A. Solo `f1` (BPOF de protocolo) | Un RGB por FOV | A veces no es el más nítido (`S` máximo en otro plano) |
| B. Reelección por $\arg\max S$ | Usar el plano de mayor nitidez | `S` no está calibrada entre sesiones; puede preferir artefactos |
| C. Focus stacking (EDF) | Un RGB fusionado | Alucinación de bordes si el registro Z no es perfecto |
| D. Entrada multiplano | 9 canales, o 3 *tokens* de vista, o *late fusion* | Más parámetros; 688 FOVs no bastan sin preentreno |

**Fase 1–2: estrategia B** (un solo RGB, plano de máximo `S`, *fallback* a `f1` si $\Delta Z = 0$). Reservar D para un experimento controlado cuando el encoder ya esté estable.

No usar `f0` y `f2` como *augmentation* de la misma etiqueta en clasificación de FOV sin cuidado: son el mismo contenido, almost-i.i.d., e inflan el N efectivo.

---

## 6. Plan de implementación por fases

Las fases son secuenciales en **dependencias teóricas**, no necesariamente en calendario de calendario largo. Cada fase tiene un criterio de salida. No se pasa a la siguiente sin ese criterio.

### Fase 0 — Metrología y contrato de datos

**Objetivo.** Persistir $s_x$, $s_y$ (ya calculados) y el manifiesto de FOVs.

**Trabajo**

- Escribir `calibration.json` según §3.4.
- Corregir `BAD_IMAGES.csv` (coordenada de `C_QUITENSIS_0044`).
- Excluir malas + JSON huérfano.
- Escribir un *dataset card*: unidad atómica, *splits* por placa, hash de archivos, resolución, ROI. Nombres de clase = `SPECIES` del CSV.
- Decidir origen de coordenadas de máscaras futuras (píxel top-left, Y hacia abajo, como PNG).

**Criterio de salida.** `calibration.json` + lista canónica de FOVs usables (se espera **~684** FOVs de 688, salvo más descartes visuales).

**Aún no:** entrenar nada.

### Fase 1 — Higiene, teselado y *baselines* no neuronales

**Objetivo.** Entender el dato como señal, no como “dataset de deep learning”.

**Trabajo teórico-práctico**

- Histograma de color fondo vs. objeto; estimación grosera de *objectness* por FOV (¿cuántos marcos están casi vacíos?).
- *Baseline* A: umbral + morfología + componentes conexas → cajas/máscaras crudas.
- *Baseline* B: SAM2 automático + filtros geométricos.
- Teselado documentado: p. ej. ventanas 512 × 512 con solape 25 %, sobre el plano elegido (§5.4). Guardar el mapeo tile → FOV → µm.
- **No** usar tiles de la misma placa a ambos lados del *split*.

**Criterio de salida.** Un informe cuantitativo *sin* red: precisión/recall *aproximados* de propuestas vs. un **muestreo visual de ~50 FOVs** (no es gold estándar, es control de cordura). Si el *baseline* clásico ya es excelente en clases de semillas aisladas, el valor del transformer se acota a solapes y a clasificación.

### Fase 2 — Encoder ViT auto-supervisado (el verdadero “aprendizaje sin máscaras”)

**Objetivo.** Un espacio de *tokens* en el que semillas de la misma especie se agrupen y el fondo se separe, **sin polígonos**.

**Trabajo**

- Inicializar DINOv2 (o MAE) preentrenado; *fine-tune* corto en tiles de AntarSeeds **o** incluso *linear probe* primero (a menudo basta).
- Diagnósticos que sustituyen a mAP mientras no hay cajas:
  - PCA / t-SNE de tokens o de CLS por FOV, coloreado por especie.
  - Mapas de atención / primer componente PCA de parches (deben resaltar semillas, no viñeteado).
  - *k-NN* de especie a nivel FOV (métrica honesta con *split* por placa).
- Opcional: DINO *head* entrenada solo con las 8 etiquetas de placa como *weak labels* (no es auto-supervisión pura, pero es barata).

**Criterio de salida.** El *k-NN* o *linear probe* de especie, **placa held-out**, supera un baseline tonto (color medio del FOV / histograma HSV). Si **no** lo supera, o las dos placas de la misma especie no se parecen, el problema no es el ViT: es deriva de iluminación, foco o protocolo entre sesiones. Hay que parar y mirar captura, no apilar capas.

Esta fase es el corazón teórico del encargo: **sí se puede aprender un ViT ahora**, en el sentido de representación y clasificación débil. **No** se puede aprender un segmentador de instancias.

### Fase 3 — Clasificador débil de FOV (producto intermedio)

**Objetivo.** Un ViT que, dado un BPOF, predice las 8 especies.

**Diseño**

- Entrada: tile central o *multi-crop* (varios tiles, agregación por *max* o MIL). Un único resize 518² del FOV entero perderá aristas; usarlo solo como *ablation*.
- Pérdida: cross-entropy. Regularización fuerte (drop-path, Mixup/CutMix **con cautela**: CutMix de dos especies inventa quimeras morfológicas).
- Evaluación: accuracy y macro-F1 **por placa held-out**, matriz de confusión entre los 8 `SPECIES`.
- Explicabilidad: mapas de atención; si miran el fondo, el modelo está usando el color de la bandeja (fuga de dataset). Mitigación: restar fondo, normalizar color por placa, o entrenar solo sobre tiles con *objectness* alto.

**Criterio de salida.** Macro-F1 held-out documentado. Este modelo **no** alimenta *Área (px)*. Sí sirve como control de que la señal taxonómica está en la imagen.

### Fase 4 — De propuestas a *gold set* mínimo (la fase que desbloquea segmentación)

**Objetivo.** Pocas máscaras **correctas**, no muchas máscaras automáticas.

**Protocolo teórico de anotación** (human-in-the-loop):

1. Muestreo estratificado: las 8 especies × 2 placas × {FOV vacío, 1 semilla, N semillas, solape, borde de frame, fuera de foco} — del orden de **150–300 FOVs** (no 688).
2. Pre-relleno con SAM / *baseline* clásico.
3. Corrección en herramienta (Label Studio, CVAT, or SAM-assisted): clase **semilla** vs. **rechazo** (fibra, fragmento, burbuja).
4. Convención de instancia: cada semilla un ID; apéndices pegados al cuerpo **sí**; polvo **no**.
5. Exportación COCO o *labelme*, con `image_id` ligado al *stem* del PNG y a `s_x,s_y`.

**Criterio de salida.** Acuerdo inter-anotador en un 10 % doble-etiquetado (IoU de instancia > 0,8 en objetos no ambiguos). Sin esto, el “mAP del ViT” no significa nada.

Hasta no tener este *gold set*, **está prohibido** reportar métricas de segmentación como resultado del proyecto.

### Fase 5 — Cabeza densa / instancias (el ViT de producto)

**Objetivo.** Lista de semillas por FOV: máscara, score, especie, geometría en µm.

**Arquitectura de trabajo (cuando exista Fase 4)**

- *Backbone:* encoder de Fase 2 (DINOv2 / ViT), preferentemente congelado al inicio.
- *Cabeza:* Mask2Former (instancia) **o** SAM2 *fine-tune* ligero + clasificador de especie por recorte.
- Alternativa parsimoniosa: detector de cajas (DINO-DETR) + SAM en el *box prompt* + ViT clasificador en el *crop*. Tres módulos, cada uno con supervisión más barata.
- Post-proceso: NMS espacial usando $(x,y)$ de stage para no contar dos veces la misma semilla en FOVs vecinos (el error de host de cientos de µm entra aquí como incertidumbre de asociación).

**Métricas**

- Detección: AP50, AP75 (COCO), **por placa held-out**.
- Semántica auxiliar: IoU fondo/semilla.
- Morfometría: error de área y de eje mayor **en µm**, usando $s_x$ y $s_y$, sobre el *gold set* (no sobre pseudo-máscaras).
- Biología: no optimizar solo mAP; optimizar error de **área equivalente** y tasa de fusión/separación de instancias en contacto.

**Criterio de salida.** Error de área y tasa de *split/merge* aceptables para el uso científico que se defina (ese umbral es una decisión de laboratorio, no de arquitectura).

### Fase 6 — Integración conceptual con el sistema de captura

La UI ya tiene *Contornos*, *ROI listo* y *Objetos detectados* (Nº, Score, Área px). El modelo de Fase 5 es un **servicio de inferencia** sobre el frame (o sobre el PNG BPOF):

- *Score* ← confianza de instancia o de clasificador.
- *Área (px)* ← $\sum$ máscara; en paralelo *Área (µm²)* ← Área px $\times s_x s_y$.
- *Contornos* ← polígono simplificado de la máscara.

Esto es diseño de sistema, no de red. No requiere GPU en el host de captura si la inferencia es *offline* sobre el acervo; sí la requiere si se quiere en el loop *LIVE*.

---

## 7. Lo que **no** hay que hacer (anti-plan)

1. Entrenar SegFormer/Mask2Former **desde cero** sobre 688 FOVs sin máscaras, rellenando con SAM sin revisión, y llamar a eso “ground truth”.
2. Partir train/test al azar por archivo PNG: fuga por solape de malla y por la segunda foto de la misma semilla.
3. Tratar `f0`,`f1`,`f2` como tres muestras i.i.d. de la clase.
4. Reportar áreas con un único “10 µm/px” ignorando la anisotropía $s_x \neq s_y$.
5. Redimensionar 2590 × 1942 → 224 × 224 como único *pipeline* de segmentación: se destruye la metrología y las aristas.
6. Optimizar accuracy de especie en FOV y dar el proyecto por cerrado: esa métrica no produce *Área (px)*.

---

## 8. Decisiones abiertas (requieren criterio de laboratorio, no de modelo)

| Decisión | Por qué importa |
| --- | --- |
| ¿Apéndice ⊂ semilla? | Cambia área y circularidad en clases con aristas |
| ¿Mosaico de placa o análisis por FOV? | Solape ~93 % hace el mosaico posible; el error de stage (~0,3 mm) define si conviene |
| Presupuesto de anotación (¿150 o 300 FOVs?) | Determina si Mask2Former es realista o si basta SAM+clasificador |
| Inferencia *LIVE* vs. *offline* | Define latencia, hardware y si el ViT debe ser ViT-S o puede ser ViT-B |

---

## 9. Orden de ejecución resumido

```text
[0] Persistir calibration.json + manifiesto de FOVs
        ↓
[1] Baselines clásico/SAM + teselado + split por PLACA
        ↓
[2] Encoder ViT (DINOv2/MAE) — representación sin máscaras
        ↓
[3] Clasificador débil de especie (diagnóstico de señal)
        ↓
[4] Gold set mínimo (SAM-assisted, instancias)
        ↓
[5] Cabeza de instancias + morfometría en µm
        ↓
[6] Consumo en “Objetos detectados”
```

**Dónde estamos hoy:** escala µm/px **cerrada** (§3.1); Fase 0 es persistirla y limpiar el manifiesto. El aprendizaje ViT **legítimo** con el disco actual empieza en la Fase 2, no en la 5.

---

## 10. Referencias conceptuales (para el diseño, no para citar en un paper aún)

- Dosovitskiy et al., *An Image is Worth 16×16 Words* (ViT).
- Caron et al., DINO; Oquab et al., DINOv2 (features densas y *objectness* sin cajas).
- He et al., MAE.
- Kirillov et al., SAM; Ravi et al., SAM 2 (propuestas de máscara).
- Cheng et al., Mask2Former; Xie et al., SegFormer.
- Chefer et al., *Transformer Interpretability Beyond Attention Visualization* (mapas de relevancia).
- Wang et al., TokenCut / LOST (descubrimiento de objetos con tokens DINO).

---

## 11. Conclusión

El sistema de captura está poblado: **8 taxones, 16 placas, 688 FOVs × 3 planos, ~14 GB**, con metadatos de stage y de foco. La escala de trabajo está fijada: **2590 × 1942 px** sobre un FOV **2,6 × 2,0 cm**, esto es **10,039 × 10,299 µm/px**. El software de adquisición ya piensa en objetos (score, área). **Falta el eslabón geométrico etiquetado** (máscaras / instancias), no la metrología.

Un Vision Transformer no se “entrena a segmentar” sobre este estado. Se **preentrena o se adapta como encoder**, se usa para ver si la especie es separable entre placas, y se combina con SAM/visión clásica para no anotar las 2064 imágenes a mano. Solo un *gold set* de instancias, convertido a µm con $s_x$ y $s_y$, convierte el ViT en instrumento de morfometría.

El siguiente documento técnico del repositorio debería ser, en este orden: (1) `calibration.json` con los valores de §3.1, (2) el manifiesto de FOVs y *splits* por placa, (3) el protocolo de anotación de instancias. El primer *training script* de ViT corresponde a la Fase 2, no antes.
)
