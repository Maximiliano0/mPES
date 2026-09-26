# Actualización local de la tesis mPES (versión del 2026-09-26)

Este paquete trae la versión revisada de `writings/` (37 páginas). Está pensado
para que Claude local (o el autor) actualice el repositorio y recompile.

El paquete se probó por separado: al descomprimirlo y correr `python audit/audit.py`
desde `writings/`, compila y la auditoría da todo en verde (tabla al final).

## Contenido

```
writings/
  00_Main/        Main.tex, References.bib
  01_Chapters/    los 12 capítulos .tex (Acknowledgement.tex sigue huérfano a propósito)
  02_Images/      las 21 imágenes que usa la tesis, en sus carpetas
  audit/          audit.py (corregido) y AUDIT.md (último informe)
  auxiliar/scripts/  sync_figures.py, ensemble_decisions.py, rebuild_thesis.py
referencia/       PDF compilado, para comparar
```

No se incluyen `apa.bst`, `IEEEtran.cls`, `mPES_citation.bib`,
`Lakshminarayanan2017.bib` ni `.latexmkrc`. No cambiaron y quedan como están en
el repositorio. `Main.tex` usa el estilo estándar `unsrtnat`.

## Pasos

1. **Resguardar el estado actual.** En la raíz del repositorio, revisar
   `git status`. Si hay cambios sin guardar en `writings/`, hacer commit o crear
   una rama antes de sobrescribir, por ejemplo `git switch -c revision-2026-09-26`.
2. **Copiar.** Copiar el contenido de `writings/` de este paquete sobre
   `writings/` del repositorio, reemplazando los archivos existentes.
3. **Quitar las imágenes que ya no se usan.** Si existen, borrar:
   - `writings/02_Images/individual/ind_04_kl_acciones_por_escenario.png`
   - `writings/02_Images/ensemble/ens_04_kl_acciones_por_escenario.png`
   - `writings/02_Images/agent_internals/` (con `trf_agent_confidences.png`)

   El paso 4 las borra solo; si se omite ese paso, hay que borrarlas a mano, porque
   `audit.py` marca como falla las imágenes huérfanas.
4. **Opcional: regenerar las figuras desde los resultados.** Desde la raíz, con el
   entorno virtual activado:
   `python writings\auxiliar\scripts\sync_figures.py`.
   Copia a `02_Images/` las figuras de `h1/general/results/` y de
   `h1/general/work/`, y borra todo lo que la tesis no usa. Deja las mismas 21
   imágenes que trae el paquete.
5. **Compilar y auditar.** Desde `writings/`: `python audit\audit.py`.
   El PDF queda en `writings/out/` y el informe, en `writings/audit/AUDIT.md`.
   Para compilar a mano, desde `writings/00_Main/`:
   `pdflatex -file-line-error Main.tex`, `bibtex Main` y dos veces más
   `pdflatex -file-line-error Main.tex` (o `latexmk` con el `.latexmkrc` del
   repositorio).
6. **Verificar.** El resultado debe coincidir con la tabla de abajo, y el PDF,
   con `referencia/`.

## Resultado esperado de `audit.py`

```
Criterio              Esperado
Compilación           37 páginas, 0 errores
Overfull/Underfull    0 / 0
Warnings              0
Capítulos huérfanos   sólo Acknowledgement.tex (a propósito)
Figuras / tablas      19 / 12 con label
Referencias           48, 0 rotas, 0 etiquetas duplicadas
Imágenes              21 referenciadas, 0 faltantes, 0 huérfanas
Bibliografía          todas las claves citadas existen
Idioma y marcadores   sin palabras inglesas estructurales, sin TODO/FIXME
```

## Qué cambió respecto de la versión anterior del repositorio

- `Main.tex`: el preámbulo no cambió. El cuerpo cambió en dos cosas:
  - la maquetación: índice y listas juntos, sin saltos entre capítulos,
    `\clearpage` antes de la Introducción, de la bibliografía y del Apéndice;
  - la separación entre figuras o tablas y el texto:
    `\addtolength{\intextsep}{0.5\baselineskip}` y lo mismo para
    `\textfloatsep`.
- `References.bib`: se quitaron `Hochreiter1997`, `Kullback1951` y `Packer2018`,
  que ya no se citan. Quedan 34 entradas.
- **Capítulos:** todos se revisaron, sección por sección, según las anotaciones del
  autor. El detalle está en `claude/estado_revision.md`, en el proyecto de claude.ai.
- `audit.py`: detecta los errores de compilación en formato `archivo.tex:línea:`,
  que es como los informa `pdflatex -file-line-error`. Antes podía informar
  «compilación exitosa» aunque hubiera errores.
- `sync_figures.py`: lista actualizada. Ya no copia los mapas de KL ni la confianza
  del Transformer.

## Cuidados al editar

- **Porcentajes:** escribirlos como `$36\%$`, nunca `$36\,\%$`. Con
  babel-spanish, la segunda forma da «Incompatible glue units».
- **Figuras y tablas del Capítulo 5:** usan `[H]` para respetar el orden que pidió
  el autor. Entre figuras seguidas hay un `\vspace` negativo que compensa el
  espacio extra.
- **Final de las Conclusiones:** la sección 7.4 empieza con
  `\enlargethispage{2\baselineskip}` para que las últimas líneas no queden solas
  en una página nueva. Si cambia el texto de los Capítulos 6 o 7, conviene revisar
  si sigue haciendo falta.

## Sumas SHA-256 (para verificar la copia)

```
0f384f681096740a57de9e364386499da5ae22535eb93e73435cc75f52c77128  writings/00_Main/Main.tex
e20c846ec0304b712932be894c95f2958b02aafc5940456d79c83a10059b83ae  writings/00_Main/References.bib
05cf27e352912cce82bf569410b89c3400d999980ad7970384495ba91bb9e4ef  writings/01_Chapters/000NHH-Frontpage.tex
c305e63b8d4bdc3673f7d31bedaafbe2c3abeb6a621476adeabddfe4556ae9e9  writings/01_Chapters/00Abstract.tex
06b02dfb4f8e75e66d30ee3bdd1124537ed51815d9976b36687b3f0e40e2815f  writings/01_Chapters/00Abstract_en.tex
c3f2c4cbe719a24350052b25786b37d18ef3571001427086961ff7c8419aa9d5  writings/01_Chapters/01Introduction.tex
b64d346497cd6682501cf1d393fe4beb27d164573885fa4406395df7fb6fe5d0  writings/01_Chapters/02Background.tex
4dbf6ad3e5a9a4a5c4a6eea28fb69155844d4ac0a9604a4546a8451481164053  writings/01_Chapters/03StateOfTheArt.tex
f6aa92885a208bf8aee1bc4fe3a6fae508920af8e6a6e1e90c27f2c88257525c  writings/01_Chapters/04Materials.tex
1e149d13d7777d1dfabe93b271f4ec410acaced2fd0dffc97297b3e59e3e53fd  writings/01_Chapters/05Results.tex
dc18e361ab65279350cd3423e2cf421aa80b91e85fe5ad5dc7123c9d7b5b0f64  writings/01_Chapters/06Discussion.tex
2e116e9bd088d44377ac4f1d54effa4ffeeb3ea116f1cb3bc278fcc8335a4209  writings/01_Chapters/07Conclusion.tex
7922fc76ce8731fee0aebf865effb5383a7d3fcf0d53849e15e4af23ee12f278  writings/01_Chapters/Acknowledgement.tex
ce6552f091d94144d58f7b21fe6f50b791c95bffc3a9d998a70ec9c7e49a78cc  writings/01_Chapters/Appendix.tex
544e278680d2aa5c2bddcd16f506df65af776b533ddf2d9450ad0a476c02408b  writings/02_Images/baseline/random_player_normalised_performance.png
645221743f6e6cb2c183f4789ab019dfc94a6e7276f787b0fdbd24625fadb194  writings/02_Images/baseline/random_player_sequence_performance.png
5ddd09e2c62cadc615ccd861e495cbbe0efc7f7ccbca828ab5772e4df5eff579  writings/02_Images/ensemble/ens_01_desempeno_por_escenario.png
2fac9ac2cc6089ec6f51f3975a629529fa9fe70dc48b04b8db3c910ac2fe051e  writings/02_Images/ensemble/ens_03_welch_logp_por_escenario.png
cdddb1d43866a7b14729770b5e15821f80663c7122cd938784bc451c6f7f377c  writings/02_Images/ensemble/ens_06_curvas_extrapolacion.png
db1557c96f543edcfe38f013e6647bd83596b91bb33097da02b04d9ebce5cafe  writings/02_Images/ensemble/ens_07_cohen_d_por_escenario.png
47e9b45bcdcf0d2a1062995afe9ee5a4aba3db4a5e05a7fcf30851534753dee1  writings/02_Images/ensemble/ens_12_pares_welch_logp.png
d5a70ca0042a45b20728bbe83cd6096ecfc95401f2ceed2ab401543fac0e3073  writings/02_Images/ensemble/ens_13_pares_cohen_d.png
b66e91e58ba78f13ab198e1ffe62c5e9dfeaaee454d36ae3a78d60e20e9acf4f  writings/02_Images/frontpage/LOGO-ITBA.jpg
57221e7a59c9f9be387af13d3458aa7cff530a0b50c0d5c8e0d67a320267a642  writings/02_Images/individual/ind_01_desempeno_por_escenario.png
f47ade4529ead6e1a4bd0e7aea4e8c963cefc34a03ef4c79597149c15acd169d  writings/02_Images/individual/ind_03_welch_logp_por_escenario.png
7443bcbd0ffb7b6f2426b3097db95c25b78fefbc5b3f3a8f83dd103d11cb92bd  writings/02_Images/individual/ind_05_curvas_por_familia.png
17deaf704e19373c9719e3a24318e2a68a1f2b0864187baff5b4408c76d1b4d2  writings/02_Images/individual/ind_13_pares_cohen_d.png
1a3978f0207760d7eaa489e5d3625777bd1d594ec3ce1eec838f05a3f299c520  writings/02_Images/per_model/PES_A2C_results.png
b9ee9c9ef0e2cd4eabb99f7faf6648811e7636f7156e8c0f7c521d3949d4d763  writings/02_Images/per_model/PES_BASE_results.png
c9bffcbf07568d029ad53a0ae8138f70b54320546b95c702df0a4134f8c96737  writings/02_Images/per_model/PES_DQL_results.png
f899475bee2aaf59b8640e407da12035b8a814fd7d06624920a0c98323a027d0  writings/02_Images/per_model/PES_DQN_results.png
6bcbf0f4222436ed76018e13374f59c497ab44f5ecaa4b300600ae2a26aa6aac  writings/02_Images/per_model/PES_ENS_results.png
5ca3902b75b9642f056d7d2550bffa910f6dd3ef711bdc4bff7aab67d599fbc5  writings/02_Images/per_model/PES_QL_results.png
d60ae9de973fbace03f715282c40fc51af106ea379868b5576a6c5b46f9aa995  writings/02_Images/per_model/PES_RDQN_results.png
f71a4d5e8d934ea98307fa13f839a7f02ea3731d4d18dffe1352df3b8ea3547d  writings/02_Images/per_model/PES_TRF_results.png
f882fd449b12c13af893a38767f6e18d4c7019df40832f093ffc3e89ecfe8b9b  writings/audit/AUDIT.md
4fd390147f6abb29ddf40f31b2c629706aa6f651eed9ca02d893d4a5eb3e01fc  writings/audit/audit.py
f2a11359b0827c159ab6b1f636d2c02df4160c3d77bff4dffb5b1eabb236946d  writings/auxiliar/scripts/ensemble_decisions.py
c1ee747e0eb7e1a17e20c5cc8374c0fb36d7a1fec89b016de2ce2164b698a30b  writings/auxiliar/scripts/rebuild_thesis.py
89b5fb8ee643360a9db2da53438a852d6af9f10f64436d8ff223d6ec9cec3b09  writings/auxiliar/scripts/sync_figures.py
```
