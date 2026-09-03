"""
Genera un documento Word editable para NotebookLM / presentaciones,
a partir del contenido del manuscrito MeliVision (IEEE TPAMI preliminary).
"""

from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt


def _add_bullets(doc: Document, items: list[str]) -> None:
    for item in items:
        doc.add_paragraph(item, style="List Bullet")


def _add_table(doc: Document, headers: list[str], rows: list[list[str]]) -> None:
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = "Table Grid"
    hdr = table.rows[0].cells
    for i, h in enumerate(headers):
        hdr[i].text = h
    for r_idx, row in enumerate(rows):
        cells = table.rows[r_idx + 1].cells
        for c_idx, val in enumerate(row):
            cells[c_idx].text = val
    doc.add_paragraph()


def build_document() -> Document:
    doc = Document()

    # Título
    title = doc.add_heading(
        "MeliVision: Detección y Clasificación de Polen en Microscopía Óptica",
        level=0,
    )
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    sub = doc.add_paragraph(
        "ViT-denso para localización + AnalogyNet (métrica slice-aware) para identificación taxonómica"
    )
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub.runs[0].italic = True

    meta = doc.add_paragraph("Autor: Rodrigo Jofré Cerda | Proyecto MeliVision | Junio 2026")
    meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_paragraph()

    # Resumen ejecutivo (ideal para NotebookLM)
    doc.add_heading("Resumen ejecutivo (1 minuto)", level=1)
    doc.add_paragraph(
        "MeliVision automatiza la melisopalinología: localiza granos de polen en imágenes "
        "de campo completo de microscopía y los identifica a nivel de especie. "
        "El sistema usa un paradigma detectar-y-luego-clasificar con dos módulos independientes: "
        "(1) un detector ViT-denso que produce mapas de saliencia pixel a pixel, y "
        "(2) un clasificador AnalogyNet que mapea cada grano recortado a un embedding unitario "
        "en una hiperesfera de 128 dimensiones, donde la identidad de especie es un problema "
        "de proximidad angular (coseno), no de softmax sobre clases fijas."
    )
    doc.add_paragraph(
        "Dataset: PollenBB16 (Zenodo, DOI 10.5281/zenodo.19830051) — 16 especies de flora "
        "chilena (región del Biobío), 16.198 imágenes de campo y 36.383 instancias con polígonos expertos."
    )

    # Problema
    doc.add_heading("1. Problema y motivación", level=1)
    doc.add_paragraph(
        "La melisopalinología y la certificación botánica de origen dependen de identificar "
        "miles de granos microscópicos por muestra. Los expertos reconocen especies por ornamentación "
        "del exino, aperturas y tamaño bajo óptica de campo brillante — un proceso lento y "
        "dependiente del operador."
    )
    doc.add_paragraph(
        "La mayoría del trabajo previo en palinología computacional trata el reconocimiento "
        "como clasificación de imágenes ya recortadas, ignorando que en la práctica hay que "
        "encontrar cada grano antes de nombrarlo."
    )
    _add_bullets(
        doc,
        [
            "Localización = saliencia (detección de objetos salientes, SOD), no clasificación.",
            "Identificación = recuperación métrica en S^127, no softmax sobre logits fijos.",
            "Pipeline desacoplado: localización e identificación tienen supervisión, métricas y fallos distintos.",
        ],
    )

    # Contribuciones
    doc.add_heading("2. Contribuciones principales", level=1)
    _add_bullets(
        doc,
        [
            "Arquitectura detect-then-classify en tres fases: anotación, entrenamiento dual offline, acoplamiento E2E.",
            "Localización ViT-denso con formalismo SOD (compartido con U²-Net), búsqueda coarse-to-fine y quality gates.",
            "AnalogyNet: pooling multi-espectral enmascarado, proyección multi-torre [11,11,3,3] → 4×32D, inferencia slice-aware.",
            "Evaluación en tres niveles: detección SOD, recuperación MLRC, auditoría E2E en campo completo.",
        ],
    )

    # Dataset
    doc.add_heading("3. Dataset PollenBB16", level=1)
    doc.add_paragraph(
        "PollenBB16 es un corpus RGB de polen adquirido por microscopía óptica de campo brillante "
        "desde flora chilena identificada botánicamente (región del Biobío). "
        "Disponible en Zenodo: https://zenodo.org/records/19830051"
    )
    _add_bullets(
        doc,
        [
            "16.198 imágenes de campo a resolución nativa 3088×2064 px.",
            "36.383 instancias con polígonos pixel-precisos verificados por palinólogo experto.",
            "16 especies en 13 familias botánicas (endémicas, nativas e introducidas).",
            "Tres planos focales por posición (fp0, fp1, fp2) para variación de profundidad.",
        ],
    )
    _add_table(
        doc,
        ["Split", "Granos", "Rol"],
        [
            ["Train", "20.101", "Entrenamiento métrico"],
            ["Validation", "4.362", "Checkpoint clasificador (mAP@R)"],
            ["Test", "4.353", "Clasificación held-out"],
            ["Total", "28.816", "16 clases, split estratificado"],
        ],
    )

    # Arquitectura
    doc.add_heading("4. Arquitectura MeliVision (tres fases)", level=1)

    doc.add_heading("Fase A — Anotación", level=2)
    doc.add_paragraph(
        "Convierte imágenes de campo en registros .seg (polígono, bbox, clase, saliencia). "
        "En producción se usan polígonos expertos; U²-Net puede bootstrapear campos nuevos."
    )

    doc.add_heading("Fase B — Entrenamiento offline (dos módulos independientes)", level=2)
    doc.add_paragraph("Detector f_det: imagen I → mapa de saliencia Ŝ, pérdida BCE+Dice sobre máscara unión.")
    doc.add_paragraph(
        "Clasificador f_cls: crop (x, M) → embedding e ∈ S^127, pérdida Multi-Similarity sliced (4×32D)."
    )

    doc.add_heading("Fase C — Despliegue E2E", level=2)
    doc.add_paragraph(
        "FullImageClassifier acopla ambos módulos sin fine-tuning conjunto. "
        "Por cada imagen: detectar granos → recortar con misma geometría que entrenamiento → "
        "clasificar con SliceAwareClassifier (prototipos por slice)."
    )
    _add_bullets(
        doc,
        [
            "Paso 1: Detección (opcional downscale para GPU).",
            "Paso 2: PollenViTDetector → lista de granos detectados.",
            "Paso 3–4: Rescale, split multi-pico, NMS, filtros de calidad.",
            "Paso 5–6: prepare_grain_crop_and_mask → resize 252×252, normalizar.",
            "Paso 7–8: AnalogyNet → embedding → predicción por prototipos slice-aware.",
            "Paso 9: Etiquetas por grano + overlay de saliencia opcional.",
        ],
    )

    # Detector
    doc.add_heading("5. Etapa I: Localización (ViT-denso)", level=1)
    doc.add_paragraph(
        "El localizador de producción usa DINOv2 ViT-S/14 con decoder convolucional denso. "
        "Entrada letterbox 504×504; salida Ŝ ∈ [0,1]^(H×W). Instancias extraídas post-hoc: "
        "umbral adaptativo, morfología, contornos, NMS."
    )
    doc.add_paragraph(
        "Búsqueda coarse-to-fine en imágenes grandes: (1) pasada global, (2) tiles deslizantes, "
        "(3) verificación local, (4) fusión NMS. Quality gates rechazan granos truncados, "
        "blobs fusionados y detecciones multi-pico espurias."
    )
    _add_table(
        doc,
        ["Métrica detector (test, 2.375 imágenes)", "Valor"],
        [
            ["Det-F1 (IoU ≥ 0.5)", "87.8%"],
            ["Det-Precision", "90.3%"],
            ["Det-Recall", "90.6%"],
            ["Fm (SOD)", "94.1%"],
            ["Mask IoU", "83.2%"],
            ["Grains MAE", "0.264"],
        ],
    )

    # Clasificador
    doc.add_heading("6. Etapa II: Identificación taxonómica (AnalogyNet)", level=1)
    doc.add_paragraph(
        "Cada grano se representa como vector unitario en S^127. La similitud coseno es la única "
        "señal discriminativa — compatible con protocolos MLRC (R@1, mAP@R) y extensible a especies nuevas."
    )
    _add_bullets(
        doc,
        [
            "Pooling multi-espectral enmascarado: capas ViT 3 y 11, estadísticas μ/max/σ sobre tokens enmascarados.",
            "Multi-torre [11,11,3,3]: 4 torres de 32D alineadas con slices de entrenamiento.",
            "Multi-Similarity Loss por slice (S=4) para evitar colapso neural en especies fine-grained.",
            "Inferencia slice-aware: prototipos μ_c^(s) por especie y slice; score = media de cosenos.",
        ],
    )
    _add_table(
        doc,
        ["Métrica clasificador (test, 4.353 granos)", "Valor"],
        [
            ["R@1 (MLRC, 1-NN LOO)", "97.3%"],
            ["R@5", "98.1%"],
            ["mAP@R", "95.5%"],
            ["NMI", "94.0%"],
            ["Macro-F1", "97.0%"],
            ["Slice-aware accuracy", "97.1%"],
            ["Ratio inter/intra distancia", "2.42"],
        ],
    )

    # E2E
    doc.add_heading("7. Resultados end-to-end (auditoría honesta)", level=1)
    doc.add_paragraph(
        "La brecha entre clasificación aislada (~97% R@1) y operación en pipeline completo (~90% E2E) "
        "está dominada por errores de localización (falsos positivos y granos perdidos), no por colapso "
        "del espacio de embeddings (rango efectivo 29/128, NMI 94%)."
    )
    _add_table(
        doc,
        ["Métrica E2E (galería 46 imágenes)", "Valor", "Conteo"],
        [
            ["Detection recall (IoU ≥ 0.3)", "95.1%", "77/81"],
            ["Clasificación en matched", "90.8%", "69/77"],
            ["Legacy E2E accuracy", "89.5%", "85/95"],
            ["Anotaciones perdidas", "—", "4"],
            ["Falsos positivos extra", "—", "19"],
            ["Matched pero clase incorrecta", "—", "7"],
        ],
    )
    doc.add_paragraph(
        "Presupuesto de error: dominan las detecciones extra (19) y granos perdidos (4) "
        "sobre errores puros de clasificación en granos emparejados (7)."
    )

    # Geometría y XAI
    doc.add_heading("8. Geometría del embedding y explicabilidad", level=1)
    _add_bullets(
        doc,
        [
            "t-SNE y análisis espectral: rango efectivo 29/128; solo 2/120 pares de clases colapsados.",
            "Pares más difíciles: Castanea/Mentha (F1 per-class más bajos: 91.8% y 93.6%).",
            "Grad-CAM sobre DINOv2: atención en ornamentación del exino para discriminación angular.",
        ],
    )

    # Discusión
    doc.add_heading("9. Discusión y limitaciones", level=1)
    _add_bullets(
        doc,
        [
            "SOD evita forzar contornos irregulares a cajas axis-aligned en entrenamiento.",
            "Coarse-to-fine recupera granos pequeños; quality gates evitan filtrar especies grandes legítimas.",
            "Limitación: un solo seed de entrenamiento; intervalos MLRC con n_runs=10 pendientes.",
            "Galería E2E de 46 imágenes; métricas de detección/clasificación usan splits completos.",
        ],
    )

    # Conclusión
    doc.add_heading("10. Conclusión", level=1)
    doc.add_paragraph(
        "MeliVision demuestra que el análisis automatizado de polen se beneficia de una arquitectura "
        "explícita detect-then-classify: saliencia pixel para localización y aprendizaje métrico "
        "en hiperesfera para identificación. Recuperación fuerte a nivel crop (97.3% R@1) combinada "
        "con auditoría E2E honesta (89.5% legacy, 95.1% recall de detección) ofrece una base "
        "reproducible para automatización melisopalinológica."
    )

    # Guía para NotebookLM
    doc.add_page_break()
    doc.add_heading("Guía para NotebookLM — estructura sugerida de presentación", level=1)
    doc.add_paragraph(
        "Use este documento en NotebookLM y pida: "
        "'Genera una presentación de 12–15 diapositivas para audiencia técnica en visión por computador "
        "y melisopalinología, en español, con una diapositiva de título, problema, solución, "
        "arquitectura en 3 fases, dataset, resultados por etapa, E2E, limitaciones y cierre.'"
    )

    slides = [
        ("Diapositiva 1 — Título", "MeliVision: polen en microscopía con ViT-denso + métrica slice-aware. Autor, institución, fecha."),
        ("Diapositiva 2 — El problema", "Melisopalinología manual; miles de granos; necesidad de encontrar Y nombrar."),
        ("Diapositiva 3 — Idea clave", "Dos principios: saliencia (SOD) + geometría angular en S^127."),
        ("Diapositiva 4 — PollenBB16", "16 especies chilenas, Zenodo, polígonos expertos, 3 planos focales."),
        ("Diapositiva 5 — Pipeline 3 fases", "A: anotación .seg | B: entrenar det + cls | C: FullImageClassifier E2E."),
        ("Diapositiva 6 — Detector ViT-denso", "DINOv2 + decoder denso, BCE+Dice, coarse-to-fine, quality gates."),
        ("Diapositiva 7 — Resultados detección", "Det-F1 87.8%, Fm 94.1%, tabla resumida."),
        ("Diapositiva 8 — AnalogyNet", "Pooling enmascarado, 4×32D slices, Multi-Similarity, prototipos."),
        ("Diapositiva 9 — Resultados clasificación", "R@1 97.3%, mAP@R 95.5%, matriz de confusión (mencionar)."),
        ("Diapositiva 10 — E2E audit", "95.1% recall det, 90.8% cls matched, 89.5% legacy; presupuesto de error."),
        ("Diapositiva 11 — Por qué baja E2E vs crop", "Localización domina; no colapso del embedding."),
        ("Diapositiva 12 — XAI y especies difíciles", "Grad-CAM, Castanea/Mentha, espectro."),
        ("Diapositiva 13 — Demo / software", "GUI MeliVision, pestaña Inferencia E2E, carpeta de fotos."),
        ("Diapositiva 14 — Limitaciones y trabajo futuro", "Multi-seed, más especies, ablaciones U²-Net vs ViT-denso."),
        ("Diapositiva 15 — Cierre", "Código abierto, datos Zenodo, contacto."),
    ]
    for title_text, body in slides:
        p = doc.add_paragraph()
        p.add_run(title_text).bold = True
        doc.add_paragraph(body)

    doc.add_heading("Preguntas que NotebookLM puede responder desde este documento", level=2)
    _add_bullets(
        doc,
        [
            "¿Por qué detect-then-classify y no un solo modelo?",
            "¿Qué es S^127 y por qué no softmax?",
            "¿Cuál es la brecha entre crop-level y E2E y por qué?",
            "¿Qué es PollenBB16 y dónde descargarlo?",
            "¿Cómo funciona la inferencia en la GUI MeliVision?",
        ],
    )

    doc.add_heading("Referencias clave", level=2)
    _add_bullets(
        doc,
        [
            "PollenBB16: Zenodo 19830051, DOI 10.5281/zenodo.19830051",
            "U²-Net (Qin et al., 2020) — SOD baseline",
            "DINOv2 (Oquab et al., 2024) — backbone ViT",
            "Multi-Similarity Loss (Wang et al., 2019)",
            "MLRC protocol (Musgrave et al., 2020)",
        ],
    )

    return doc


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    out_dir = root / "paper" / "ieee_tpami_preliminary"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "MeliVision_Presentacion_NotebookLM.docx"

    doc = build_document()
    doc.save(str(out_path))
    print(f"Guardado: {out_path}")


if __name__ == "__main__":
    main()
