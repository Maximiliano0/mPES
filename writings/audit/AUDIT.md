# Auditoría LaTeX — `writings/`

Raíz: `C:\Users\maxvega\Documents\Win_mPES\writings`

## 1. Sintaxis y compilación

**Archivos `.tex` huérfanos en `01_Chapters/` (no incluidos por `Main.tex`):**

- `01_Chapters/Acknowledgement.tex`

- ✅ Compilación exitosa — **59 páginas**.
- 📄 PDF: `out/mPES-Esquemas-para-Toma-de-Decision-Artificial-en-Escenarios-Secuenciales.pdf`

- ✅ Sin Overfull \hbox.

- ✅ Sin Underfull \hbox.

- ✅ Sin warnings LaTeX/paquete.

## 2. Figuras, tablas y numeración

- Figuras con label: **21**.
- Tablas con label: **13**.
- Referencias internas (`\ref`/`\autoref`/`\eqref`): **87**.

- ✅ Sin referencias rotas.

- ✅ Sin etiquetas duplicadas.

- Imágenes referenciadas con `\includegraphics`: **24**.
- ✅ Todas las imágenes referenciadas existen.

**Imágenes huérfanas en `02_Images/` (no referenciadas, 9 archivos):**

- `02_Images/agent_internals/trf_agent_cumulative_performance.png`
- `02_Images/agent_internals/trf_agent_normalised_performance.png`
- `02_Images/agent_internals/trf_agent_remapped_confidences.png`
- `02_Images/heatmaps/ens_02_degradacion_por_escenario.png`
- `02_Images/heatmaps/ens_08_ranking_desempeno.png`
- `02_Images/heatmaps/ind_02_degradacion_por_escenario.png`
- `02_Images/heatmaps/ind_08_ranking_desempeno.png`
- `02_Images/ood_curves/ind_06_curvas_extrapolacion.png`
- `02_Images/per_sequence/ens_05_curvas_por_familia.png`

## 3. Citas y bibliografía (APA)

- ✅ Todas las claves citadas existen en `References.bib`.

## 4. Coherencia y cohesión

**Orden de capítulos en `Main.tex`:**

- `01_Chapters/000NHH-Frontpage.tex`
- `01_Chapters/00Abstract.tex`
- `01_Chapters/00Abstract_en.tex`
- `01_Chapters/01Introduction.tex`
- `01_Chapters/02Background.tex`
- `01_Chapters/03StateOfTheArt.tex`
- `01_Chapters/04Materials.tex`
- `01_Chapters/05Results.tex`
- `01_Chapters/06Discussion.tex`
- `01_Chapters/07Conclusion.tex`
- `01_Chapters/Appendix.tex`

## 5. Idioma único (español)

- ✅ No se detectaron palabras inglesas frecuentes.

## 6. Cobertura de los `doc/` del proyecto

| Paquete | Mencionado | `doc/` presente |
|---------|------------|-----------------|
| `pes_base` | ✅ | 2 archivos |
| `pes_ql` | ✅ | 2 archivos |
| `pes_dql` | ✅ | 2 archivos |
| `pes_dqn` | ✅ | 2 archivos |
| `pes_rdqn` | ✅ | 2 archivos |
| `pes_a2c` | ✅ | 2 archivos |
| `pes_trf` | ✅ | 2 archivos |
| `pes_ens` | ✅ | 2 archivos |

**Conceptos clave en `writings/`:**

- ✅ MDP
- ✅ Q-Learning
- ✅ Double Q-Learning
- ✅ PBRS / potential-based reward shaping
- ✅ Experience Replay
- ✅ Target Network
- ✅ Optuna / TPE
- ✅ Cohen d / Welch / KL
- ✅ Shannon entropy / UQ
- ✅ Atención causal / Transformer

## 7. Marcadores pendientes (TODO / FIXME)

- ✅ Sin marcadores TODO/FIXME pendientes.
