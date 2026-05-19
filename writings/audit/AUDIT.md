# Auditoría LaTeX — `writings/`

Raíz: `C:\Users\maxvega\Documents\Win_mPES\writings`

## 1. Sintaxis y compilación

- No hay archivos `.tex` huérfanos.

**Avisos de bibtex:**

- empty publisher in Hasselt2010

- Compilación OK: `C:/Users/maxvega/Documents/Win_mPES/writings/out/mPES-Inteligencia-Artificial-para-la-Gestion-de-Crisis-Pandemicas.pdf` (55 páginas, 2918 KB).

## 2. Figuras, tablas y numeración

- Figuras con label: **9**.
- Tablas con label: **4**.
- Referencias internas (`\ref`/`\autoref`/`\eqref`): **60**.

- Sin problemas detectados.

## 3. Citas y bibliografía (APA)

- Todas las 61 claves citadas existen en `References.bib`.

**Entradas no citadas (13):** `AIAMA`, `Endo`, `Giuste2021`, `Gymnasium2023`, `Hasselt2016`, `Janner2021`, `MNE`, `MNE-PYTHON`, `Silver2017`, `Wiering2008`, `ramele2019histogram`, `uriguen2015eeg`, `wolpaw2012brain`

## 4. Coherencia y cohesión

**Orden de capítulos en `Main.tex`:**

- `Introducción` → `01Introduction.tex` (1317 palabras)
- `Marco Teórico` → `02Background.tex` (1694 palabras)
- `Estado de la Cuestión` → `03StateOfTheArt.tex` (1621 palabras)
- `Metodología` → `04Materials.tex` (2566 palabras)
- `Resultados y Solución` → `05Results.tex` (2327 palabras)
- `Discusión` → `06Discussion.tex` (1990 palabras)
- `Conclusiones` → `07Conclusion.tex` (1265 palabras)
- `Agradecimientos` → `Acknowledgement.tex` (300 palabras)

## 5. Idioma único (español)

- No se detectaron palabras inglesas frecuentes.

## 6. Cobertura de los `doc/` del proyecto

| Paquete | Mencionado | `doc/` presente |
|---------|------------|-----------------|
| `pes_base` | ✅ | 4 archivos |
| `pes_ql` | ✅ | 2 archivos |
| `pes_dql` | ✅ | 2 archivos |
| `pes_dqn` | ✅ | 2 archivos |
| `pes_a2c` | ✅ | 2 archivos |
| `pes_trf` | ✅ | 2 archivos |
| `pes_rdqn` | ✅ | 2 archivos |
| `pes_ens` | ✅ | 4 archivos |

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

