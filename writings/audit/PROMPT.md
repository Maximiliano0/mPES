# Auditoría de tesis — Prompt para LLM
> Generado automáticamente por `writings/audit/audit_prompt.py`.
Eres un auditor académico experto en LaTeX, redacción técnica en español
y normas APA. Vas a auditar una tesis de maestría completa. Devuelve un
**informe en Markdown** con una sección por criterio. Para cada hallazgo
indica archivo, fragmento textual y propuesta concreta de corrección.

## Criterios obligatorios

1. **Sintaxis y compilación LaTeX.** Detecta entornos mal cerrados,
   macros desconocidas, argumentos faltantes, comillas tipográficas
   incorrectas, espacios irregulares antes de signos, uso indebido de
   `$...$` vs. `\(...\)`, y cualquier construcción que `pdflatex`
   probablemente rechace o advierta. Señala archivos `.tex` que no
   estén incluidos por `Main.tex` (huérfanos) y bloques duplicados.

2. **Figuras, tablas y numeración.** Verifica que cada entorno
   `figure` / `table` tenga `\caption{}` y `\label{...}` con prefijo
   coherente (`fig:`, `tab:`). Confirma que toda `\ref`, `\autoref` y
   `\eqref` apunte a un `\label` existente. Señala numeración
   inconsistente o tablas sin `booktabs`.

3. **Citas y bibliografía en formato APA.** Comprueba que cada `\cite*`
   tenga una entrada correspondiente en `References.bib`; detecta
   duplicados de clave o títulos casi idénticos, entradas no citadas,
   campos faltantes (autor, año, título, editor/journal) y
   inconsistencias de mayúsculas. Sugiere la cita APA correcta cuando
   detectes errores.

4. **Coherencia y cohesión.** Evalúa el hilo argumental entre
   capítulos: progresión lógica, definiciones antes del uso,
   redundancias, párrafos sin verbo principal, conectores ausentes y
   referencias cruzadas rotas conceptualmente (p.ej. "ver Cap. 5" sin
   contenido relacionado).

5. **Idioma único: español.** El texto debe estar enteramente en
   español. Marca cualquier oración, frase o palabra en inglés que no
   sea nombre propio, identificador de software, término técnico
   consolidado (p. ej. *Q-Learning*, *Transformer*) o cita textual.
   Sugiere la traducción adecuada.

6. **Cobertura del proyecto.** El documento debe describir todos los
   paquetes del repositorio `mPES`. Verifica que cada uno aparezca y
   esté correctamente caracterizado:

   | Familia | Paquete | Algoritmo |
   |---------|---------|-----------|
   | tabular | pes_base | Q-Learning tabular (línea base) |
   | tabular | pes_ql   | Q-Learning + Optuna/TPE |
   | tabular | pes_dql  | Double Q-Learning + PBRS + warm-up |
   | ml      | pes_dqn  | Deep Q-Network |
   | ml      | pes_rdqn | Recurrent DQN (LSTM) — futuro |
   | ml      | pes_a2c  | Advantage Actor-Critic |
   | ml      | pes_trf  | Causal Transformer DQN |
   | ml      | pes_ens  | Ensemble (soft voting) — futuro |

   Señala paquetes ausentes o descripciones incorrectas.

## Formato de respuesta

- Una sección `## N. <Criterio>` por cada uno de los 6.
- Dentro de cada sección, una lista de hallazgos `- **archivo.tex** —
  descripción breve. _Sugerencia:_ ...`.
- Si un criterio no tiene hallazgos, escribe `Sin observaciones.`.
- Al final, sección `## Resumen ejecutivo` con un veredicto global
  (APROBADO / OBSERVADO / RECHAZADO) y los 3 problemas más críticos.

---

# Material a auditar

## `00_Main/Main.tex`

```latex

\documentclass[spanish, a4paper, 12pt, twoside]{article} 

% -------------- Setup, do not change these ---------------
\usepackage{subcaption} 
\usepackage{textcomp}
\usepackage[T1]{fontenc, url}
\usepackage[utf8]{inputenc}
\usepackage{titlesec}
\setcounter{secnumdepth}{4}
\usepackage{multirow}
\usepackage{booktabs}
% \usepackage{minted} % Code highlighting (requires --shell-escape; unused)
\usepackage{adjustbox}
\usepackage{chronology}
\usepackage{graphicx}
\usepackage{mathtools}
\usepackage{amsmath, amssymb, amsthm} % Mathematical packages
\usepackage{parskip} % Removing indenting in new paragraphs
\urlstyle{sf}
\usepackage{color}
\usepackage{subcaption} 
\usepackage{appendix}
\usepackage{chngcntr} % needed for correct table numbering
\usepackage[hidelinks]{hyperref}
\counterwithin{table}{section} % numbering of tables 
\counterwithin{figure}{section} % numbering of figures
\numberwithin{equation}{section} % numbering of equations
\hyphenpenalty=100000 % preventing splitting of words
\sloppy 
\raggedbottom 
\usepackage{xparse,nameref}
\usepackage{algorithm2e} % For algorithms
\usepackage[bottom]{footmisc} % Footnotes are fixed to bottom of page
% \usepackage{lipsum} % For generating dummy text
% --------- You can edit from this point on --------


% ----- Appearance and language ----- 
\usepackage[spanish,es-nodecimaldot]{babel} % document language (Spanish)
\graphicspath{{02_Images/}{../02_Images/}} % path to 02_Images
\usepackage[margin=2.54cm]{geometry} % sets margins for the document
\usepackage{setspace}
\linespread{1.5} % line spread for the document
\usepackage{microtype}


% ----- Sections -----
\titleformat*{\section}{\LARGE\bfseries} % \section heading
\titleformat*{\subsection}{\Large\bfseries} % \subsection heading
\titleformat*{\subsubsection}{\large\bfseries} % \subsubsection heading
% next three lines creates the \paragraph command with correct heading 
\titleformat{\paragraph}
{\normalfont\normalsize\bfseries}{\theparagraph}{1em}{}
\titlespacing*{\paragraph}
{0pt}{3.25ex plus 1ex minus .2ex}{1.5ex plus .2ex}


% ----- Figures and tables ----- 
\usepackage{fancyhdr}
\usepackage{subfiles}
\usepackage{array}
\usepackage[rightcaption]{sidecap}
\usepackage{wrapfig}
\usepackage{float}
\usepackage[labelfont=bf]{caption} % bold text for captions
\usepackage[para]{threeparttable} % fancy tables, check these before you use them
\usepackage{url}
\usepackage[table,xcdraw]{xcolor}
\usepackage{makecell}
\usepackage{hhline}


% ----- Sources -----
\usepackage{natbib}
\bibliographystyle{apalike} % citation and reference list style
\def\biblio{\clearpage\bibliographystyle{apalike}\bibliography{00_Main/References.bib}} % defines the \biblio command used for referencing in subfiles - DO NOT CHANGE


% ----- Header and footer -----
\pagestyle{fancy}
\fancyhead[RO,LE]{\thepage} % page number on right for odd pages and left for even pages in the header
\fancyhead[RE,LO]{\nouppercase{\rightmark}} % chapter name and number on the right for even pages and left for odd pages in the header
%\renewcommand{\headrulewidth}{0pt} % sets thickness of header line
\fancyfoot{} % removes page number on bottom of page


% ----- Header of the frontpage ----- 
\fancypagestyle{frontpage}{
	\fancyhf{}
	\renewcommand{\headrulewidth}{0pt}
	\renewcommand{\footrulewidth}{0pt}
	\vspace*{4\baselineskip}
	
	\fancyhead[C]{ \includegraphics[width=3in]{LOGO-ITBA.jpg}}
}

% ----- Document starts here ----- 
\begin{document}

\def\biblio{} % resets the biblio command, if not here a new reference list will be produced after every chapter

\include{01_Chapters/000NHH-Frontpage}
\restoregeometry % restores the margins after frontpage
%\nocite{*} % uncomment if you want all sources to be printed in the reference list, including the ones which are not cited in the text 

\pagenumbering{gobble} % suppress page numbering
\thispagestyle{plain} % suppress header
\clearpage\mbox{}\clearpage % add blank page

\pagenumbering{roman} % starting roman page numbering
% \newpage
% \section*{Introduction}
%     \subfile{01_Chapters/Introduction}

\newpage
\section*{Resumen}
    \subfile{01_Chapters/00Abstract}

\newpage
\tableofcontents

\newpage
{\setstretch{1.0} 
\listoffigures}
 
\newpage
{\setstretch{1.0} 
\listoftables}

\newpage
\addtocontents{toc}{\protect\setcounter{tocdepth}{4}} % sets depth of toc to 4, 1.1.1.1
\pagenumbering{arabic} % Starting arabic page numbering
\setcounter{page}{1} % sets pagecounter to 1
\setcounter{tocdepth}{0}

\section{Introducción} % section/chapter name
    \subfile{01_Chapters/01Introduction} % including the subfile for the chapter
\clearpage % clears the page after the chapter is finished

\section{Marco Teórico}
    \subfile{01_Chapters/02Background}
\clearpage


\section{Estado de la Cuestión}
    \subfile{01_Chapters/03StateOfTheArt}
\clearpage
  
\section{Metodología}
    \subfile{01_Chapters/04Materials}
\clearpage

\section{Resultados y Solución}
    \subfile{01_Chapters/05Results}
\clearpage


\section{Discusión}
    \subfile{01_Chapters/06Discussion}
\clearpage

\section{Conclusiones}
    \subfile{01_Chapters/07Conclusion}
\clearpage

\section{Agradecimientos}
    \subfile{01_Chapters/Acknowledgement}
\clearpage

% \section{Acknowledgements}
%     \subfile{01_Chapters/Acknowledgements}
% \clearpage

\newpage
\renewcommand\refname{Referencias Bibliográficas} % name for the reference list
{\setstretch{1.0} % linespacing for the references
\addcontentsline{toc}{section}{Referencias Bibliográficas} % to change the name of the references in the TOC
\bibliography{00_Main/References.bib} % adds the references to the document
}

\newpage
\renewcommand{\appendixpagename}{Apéndice} % Heading of appendix
\renewcommand{\appendixtocname}{Apéndice} % name of appendix in TOC
\appendixpage 
\addappheadtotoc

\begin{appendices}
    \subfile{01_Chapters/Appendix}
\end{appendices}


\end{document}

```

## `01_Chapters/000NHH-Frontpage.tex`

```latex
﻿
\begin{titlepage}
	
	\newgeometry{top=1 in, bottom=1 in, left=1 in, right= 1 in} 
	
	\thispagestyle{frontpage}
	
	\begin{center}
		
		\vspace*{6\baselineskip}
	
		
		{\large \textbf{INSTITUTO TECNOL\'OGICO DE BUENOS AIRES --- ITBA}}\\
		{\large \textbf{MAESTR\'IA EN CIENCIA DE DATOS}}\\

		\vspace*{2\baselineskip}

		{\Huge \textbf{mPES: Inteligencia Artificial\\para la Gesti\'on de\\Crisis Pand\'emicas\\}}

		\vspace*{1\baselineskip}

		\large{\textit{Una evaluaci\'on comparativa de arquitecturas de\\Aprendizaje por Refuerzo para la asignaci\'on de\\recursos cr\'iticos bajo incertidumbre}}\\

		\vspace*{2\baselineskip}

		\large{\textbf{AUTOR:} Vega, Maximiliano Leonel \quad (Leg. N\textsuperscript{o} 53223)}\\
		\large{\textbf{DIRECTOR:} Dr. Rodrigo Ramele}\\

		\vspace{1.5\baselineskip}

		\large{Repositorio oficial:\\\url{https://github.com/Maximiliano0/mPES_2026}}\\

		\vspace{1.5\baselineskip}

		\large{\textsc{Tesis presentada para la obtenci\'on del t\'itulo de\\Mag\'ister en Ciencia de Datos}}\\

		\vspace{1.5\baselineskip}

		\large{Buenos Aires}\\
		\large{Primer Cuatrimestre, 2026}

	\end{center}
	
	\vspace*{6\baselineskip}
	
	
\end{titlepage}


```

## `01_Chapters/00Abstract.tex`

```latex
﻿\documentclass[../00_Main/Main.tex]{subfiles}
\begin{document}

Esta tesis presenta el \emph{framework} \textbf{mPES} (\emph{Multiple
Pandemic Experiment Scenario}), un entorno de trabajo multi-paquete
desarrollado \'integramente en Python y dise\~nado como un ecosistema
de experimentaci\'on para evaluar el rendimiento de agentes de
Inteligencia Artificial. El objetivo central de la investigaci\'on es
\emph{investigar y evaluar emp\'iricamente} c\'omo diversas
arquitecturas de Aprendizaje por Refuerzo (\emph{Reinforcement
Learning}, RL) pueden optimizar la toma de decisiones en la
asignaci\'on de recursos cr\'iticos durante crisis sanitarias.

Para ello, el estudio compara seis enfoques algor\'itmicos que abarcan
desde m\'etodos tabulares cl\'asicos hasta modelos de vanguardia como
los Decision Transformers. El n\'ucleo de la propuesta reside en
superar la ``miop\'ia'' de los algoritmos tradicionales mediante el
modelado de trayectorias epidemiol\'ogicas como secuencias temporales,
integrando adem\'as la Cuantificaci\'on de Incertidumbre
(\emph{Uncertainty Quantification}, UQ) basada en la Entrop\'ia de
Shannon para dotar al sistema de transparencia y confiabilidad.

En \'ultima instancia, esta investigaci\'on busca \emph{generalizar}
el problema de la distribuci\'on de recursos limitados ante din\'amicas
de propagaci\'on complejas y no lineales. Los resultados obtenidos
sobre un \emph{benchmark} de $64$ escenarios demuestran que el enfoque
basado en Transformers ofrece una resiliencia superior frente a la
variabilidad extrema de par\'ametros epidemiol\'ogicos, los cuales
fueron sintonizados mediante Optimizaci\'on Bayesiana.

\par\vspace{0.5\baselineskip}
\noindent\textbf{Palabras clave:} mPES, Aprendizaje por Refuerzo,
Inteligencia Artificial, Decision Transformers, Cuantificaci\'on de
Incertidumbre, Entrop\'ia de Shannon, Optimizaci\'on Bayesiana,
Asignaci\'on de Recursos, Secuencias Temporales.

\biblio % Needed for referencing to working when compiling individual subfiles - Do not remove
\end{document}


```

## `01_Chapters/01Introduction.tex`

```latex
﻿\documentclass[../00_Main/Main.tex]{subfiles}

\begin{document}

La gesti\'on de recursos en una pandemia es un desaf\'io de log\'istica
y \'etica que requiere decisiones r\'apidas bajo una presi\'on inmensa.
El proyecto \textbf{mPES} surge como una respuesta a la necesidad de
herramientas de soporte de decisiones que no s\'olo optimicen la
supervivencia, sino que entiendan la \emph{inercia} del escenario
pand\'emico.

\subsection{Introducci\'on y Motivaci\'on}\label{sec:motivacion}

El presente documento describe el proyecto \textbf{mPES}
(\emph{Multiple Pandemic Experiment Scenario}), un entorno de trabajo
(\emph{workspace}) multi-paquete desarrollado \'integramente en
Python. El objetivo central de esta tesis es \emph{investigar y
evaluar emp\'iricamente} c\'omo diferentes algoritmos y arquitecturas
de Aprendizaje por Refuerzo (\emph{Reinforcement Learning}, RL) pueden
asistir y optimizar la toma de decisiones en la asignaci\'on de
recursos limitados durante una pandemia~\citep{Kuhl2021,Bertsimas2021}.

La crisis sanitaria provocada por la COVID--19 puso de manifiesto que
los tomadores de decisiones --- gobiernos, hospitales, cadenas
log\'isticas --- act\'uan habitualmente sin una herramienta
principista que les permita ponderar los beneficios inmediatos contra
los costos a largo plazo, y que las consecuencias de una asignaci\'on
sub\'optima se miden en vidas humanas y en costo social. La
investigaci\'on busca generalizar el problema de la distribuci\'on de
recursos en tiempos de crisis, evaluando qu\'e modelos matem\'aticos y
computacionales producen decisiones m\'as adaptativas ante din\'amicas
de propagaci\'on complejas y no lineales.

\subsection{Definici\'on del Problema y Escenario Pand\'emico}\label{sec:problema}

El entorno computacional desarrollado modela una crisis donde un
tomador de decisiones (que puede ser un agente de RL o un humano) debe
distribuir un \emph{presupuesto limitado} de recursos m\'edicos o
log\'isticos entre diversas ciudades afectadas simult\'aneamente. Las
principales caracter\'isticas del escenario incluyen:

\begin{itemize}
    \item \textbf{Din\'amica acumulativa.} Las decisiones tienen
    consecuencias temporales compuestas. Si una ciudad no es atendida
    a tiempo, su nivel de severidad epidemiol\'ogica crece de forma
    exponencial.
    \item \textbf{Intervenci\'on temprana.} Los recursos asignados
    generan un impacto sostenido que reduce la severidad en los pasos
    subsiguientes del entorno. Esto premia fuertemente a los modelos
    capaces de \emph{prever} escenarios y actuar con rapidez.
    \item \textbf{Evoluci\'on matem\'atica.} La severidad de cada
    ciudad se actualiza paso a paso siguiendo la f\'ormula de
    transici\'on
    \begin{equation}
        S'_i \;=\; \max\bigl(0,\; \beta \cdot S_i - \alpha \cdot a_i\bigr),
        \label{eq:transicion-intro}
    \end{equation}
    donde $\alpha = 0{,}4$ representa la efectividad de los recursos
    asignados y $\beta = 1{,}4$ act\'ua como factor multiplicador de
    la propagaci\'on del virus.
\end{itemize}

\subsection{Formulaci\'on del Proceso de Decisi\'on de Markov (MDP)}\label{sec:mdp}

Para garantizar que las comparaciones entre las distintas
aproximaciones algor\'itmicas sean \emph{justas} y \emph{sistem\'aticas},
todos los enfoques comparten la misma formulaci\'on de entorno y se
eval\'uan sobre un \emph{benchmark} fijo de $64$ secuencias:

\begin{itemize}
    \item \textbf{Espacio de estados} ($s$). El estado del sistema se
    define por la combinaci\'on de los recursos disponibles restantes,
    el n\'umero de ensayo actual (\emph{trial}) y la severidad
    acumulada en las ciudades. Esto conforma un espacio discreto de
    $3{.}410$ estados posibles.
    \item \textbf{Espacio de acciones} ($a$). Se encuentra discretizado
    en un rango de $0$ a $10$ recursos posibles a asignar por cada
    turno de intervenci\'on.
    \item \textbf{Funci\'on de recompensa} ($r$). Se define como el
    negativo de la severidad total acumulada en todas las ciudades
    afectadas, $r = -\sum_i S_i$. Esta formulaci\'on fuerza al agente
    artificial a descubrir pol\'iticas que \emph{minimicen el impacto
    global} de la enfermedad.
\end{itemize}

\subsection{Hip\'otesis de Trabajo}\label{sec:hypothesis}

El trabajo se organiza alrededor de una hip\'otesis falsable:

\begin{quote}
\textbf{H1.} \emph{El modelado expl\'icito de la trayectoria
epidemiol\'ogica mediante arquitecturas basadas en atenci\'on (Causal
Transformer) genera pol\'iticas de asignaci\'on de recursos
estrictamente superiores a las pol\'iticas miopes obtenidas por
m\'etodos tabulares cl\'asicos, especialmente bajo escenarios fuera de
distribuci\'on.}
\end{quote}

De manera complementaria, la incorporaci\'on de la \emph{Entrop\'ia de
Shannon} sobre la distribuci\'on de salida del agente permite
caracterizar la \emph{confianza metacognitiva} del sistema, abriendo
un camino algor\'itmico hacia la transparencia que tradicionalmente
exig\'ia se\~nales fisiol\'ogicas (EEG).

\subsection{Contribuciones}

Las principales contribuciones de esta tesis son:

\begin{enumerate}
    \item Un \emph{benchmark} de RL reutilizable --- el
    \emph{workspace} \textbf{mPES} --- donde seis variantes
    algor\'itmicas comparten el mismo entorno Gymnasium, el mismo
    \emph{pipeline} de optimizaci\'on bayesiana con Optuna y el mismo
    arn\'es de evaluaci\'on sobre $64$ secuencias.
    \item Una comparaci\'on cuantitativa y estad\'isticamente
    fundamentada entre m\'etodos tabulares
    (\texttt{pes\_base}, \texttt{pes\_ql}, \texttt{pes\_dql}) y
    arquitecturas profundas (\texttt{pes\_dqn}, \texttt{pes\_a2c},
    \texttt{pes\_trf}).
    \item La integraci\'on de la \textbf{Entrop\'ia de Shannon} como
    m\'etrica de \emph{Cuantificaci\'on de Incertidumbre} (UQ),
    estableciendo un puente algor\'itmico con las teor\'ias
    neurocognitivas de la confianza humana.
    \item Una implementaci\'on reproducible y multi-plataforma
    (Windows 10 y Ubuntu, Python 3.12, TensorFlow 2.21) liberada junto
    con la tesis.
\end{enumerate}

\subsection{Estructura del Documento}

El resto del documento se organiza como sigue. La
Secci\'on~\ref{sec:background} introduce el marco te\'orico
(MDPs, Q-Learning, atenci\'on, entrop\'ia de Shannon). La
Secci\'on~\ref{sec:soa} presenta el estado de la cuesti\'on
(legado neurocient\'ifico de Oxford, RL en gesti\'on de crisis,
Transformers en RL, XAI y UQ). La Secci\'on~\ref{sec:methods}
describe la metodolog\'ia: el escenario pand\'emico, las seis
arquitecturas implementadas y el \emph{pipeline} experimental. La
Secci\'on~\ref{sec:results} reporta los resultados emp\'iricos. La
Secci\'on~\ref{sec:discussion} interpreta esos resultados a la luz
de la hip\'otesis y la Secci\'on~\ref{sec:conclusion} cierra el
trabajo.

\biblio % Needed for referencing to working when compiling individual subfiles - Do not remove

\end{document}


```

## `01_Chapters/02Background.tex`

```latex
﻿\documentclass[../00_Main/Main.tex]{subfiles}
\begin{document}

\label{sec:background}

Este cap\'itulo introduce la maquinaria formal utilizada a lo largo de
la tesis: Procesos de Decisi\'on de Markov (MDPs), \emph{Reinforcement
Learning} basado en valor y en gradiente de pol\'itica, aproximaci\'on
profunda de funciones, codificadores recurrentes y basados en
atenci\'on, optimizaci\'on bayesiana de hiperpar\'ametros y, finalmente,
la Entrop\'ia de Shannon como m\'etrica de Cuantificaci\'on de
Incertidumbre (UQ). El tratamiento es deliberadamente compacto; la
referencia can\'onica es Sutton \& Barto~\citep{SuttonBarto2018}.

\subsection{Procesos de Decisi\'on de Markov}

Un MDP finito es una tupla
$\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, R, \gamma)$ donde
$\mathcal{S}$ es un espacio de estados finito, $\mathcal{A}$ un
espacio de acciones finito, $P(s' \mid s, a)$ un n\'ucleo de
transici\'on, $R(s, a, s')$ una funci\'on de recompensa acotada y
$\gamma \in [0, 1)$ el factor de descuento. Bajo una pol\'itica
$\pi(a \mid s)$ el agente genera una trayectoria
$\tau = (s_0, a_0, r_1, s_1, a_1, \ldots)$ cuyo \emph{retorno
descontado} desde el tiempo $t$ es

\begin{equation}
G_t \;=\; \sum_{k = 0}^{\infty} \gamma^{k}\, r_{t + k + 1}.
\label{eq:return}
\end{equation}

La funci\'on de valor de estado--acci\'on de la pol\'itica es
$Q^{\pi}(s, a) = \mathbb{E}_{\pi}[G_t \mid s_t = s,\, a_t = a]$ y la
pol\'itica \'optima satisface la ecuaci\'on de Bellman,

\begin{equation}
Q^{*}(s, a) \;=\; \mathbb{E}\!\left[r + \gamma \max_{a'} Q^{*}(s', a')
\,\middle|\, s, a\right].
\label{eq:bellman-opt}
\end{equation}

\subsection{Q-Learning tabular}

Cuando $\mathcal{S}$ y $\mathcal{A}$ son enumerables, $Q$ se almacena
como una tabla y se aplica la actualizaci\'on cl\'asica de
Watkins~\citep{Watkins1992}:

\begin{equation}
Q(s, a) \;\leftarrow\; Q(s, a) \;+\; \alpha\,
\bigl[r + \gamma\, \max_{a'} Q(s', a') - Q(s, a)\bigr],
\label{eq:qlearning}
\end{equation}

con tasa de aprendizaje $\alpha$ y exploraci\'on $\varepsilon$-voraz.

\subsubsection{Double Q-Learning}

El operador $\max$ en la Ecuaci\'on~\eqref{eq:qlearning} introduce un
sesgo positivo en la estimaci\'on del valor. \emph{Double
Q-Learning}~\citep{Hasselt2010} mantiene dos tablas $Q^{A}$ y $Q^{B}$
actualizadas alternativamente, desacoplando la selecci\'on y la
evaluaci\'on de la acci\'on:

\begin{equation}
Q^{A}(s, a) \leftarrow Q^{A}(s, a) + \alpha\bigl[r + \gamma\,
Q^{B}\!\bigl(s', \arg\max_{a'} Q^{A}(s', a')\bigr) - Q^{A}(s, a)\bigr].
\end{equation}

\subsubsection{Reward Shaping Basado en Potencial (PBRS)}

PBRS~\citep{Ng1999} a\~nade un t\'ermino $F(s, s') = \gamma\Phi(s') -
\Phi(s)$ a la recompensa sin alterar el conjunto de pol\'iticas
\'optimas. Es la \'unica familia de \emph{shaping} con garant\'ia
te\'orica de invarianza y se utiliza en \texttt{pes\_dql} para
acelerar la convergencia frente a una se\~nal de costo final dispersa.

\subsection{Deep Q-Networks (DQN)}

Cuando el espacio de estados es grande, $Q(s, a; \theta)$ se aproxima
con una red neuronal. El \emph{Deep Q-Network}~\citep{Mnih2015}
estabiliza el entrenamiento con dos ingredientes: (i) un
\emph{buffer de repetici\'on de experiencia}~\citep{Lin1992} del cual
se muestrean mini\-lotes uniformemente, y (ii) una \emph{red objetivo}
$Q(\cdot;\theta^{-})$ cuyos par\'ametros se copian desde $\theta$
cada $C$ pasos. La funci\'on de costo es

\begin{equation}
\mathcal{L}(\theta) \;=\;
\mathbb{E}_{(s,a,r,s') \sim \mathcal{D}}
\!\left[\bigl(r + \gamma \max_{a'} Q(s', a'; \theta^{-})
- Q(s, a; \theta)\bigr)^{2}\right].
\label{eq:dqn-loss}
\end{equation}

\subsection{Pol\'iticas, Gradientes y Actor--Critic}

Los m\'etodos de \emph{gradiente de pol\'itica}~\citep{Williams1992}
parametrizan la pol\'itica $\pi_\theta(a \mid s)$ y actualizan
$\theta$ a lo largo del gradiente del retorno esperado:

\begin{equation}
\nabla_{\theta} J(\theta) \;=\;
\mathbb{E}_{\pi_{\theta}}\!\bigl[\nabla_{\theta} \log
\pi_{\theta}(a \mid s)\, A^{\pi_{\theta}}(s, a)\bigr],
\label{eq:pg}
\end{equation}

con la funci\'on de \emph{ventaja}
$A^{\pi}(s, a) = Q^{\pi}(s, a) - V^{\pi}(s)$. El
\emph{Advantage Actor--Critic} (A2C)~\citep{Mnih2016} entrena
simult\'aneamente un actor que produce $\pi_\theta$ y un cr\'itico que
estima $V_{\phi}$. La estimaci\'on de la ventaja puede refinarse con
\emph{Generalized Advantage Estimation} (GAE).

\subsection{Atenci\'on y el Transformer}

El \emph{Transformer}~\citep{Vaswani2017,Turner2024} reemplaza la
recurrencia por \emph{atenci\'on de producto escalar escalado}. Dadas
consultas $Q$, claves $K$ y valores $V$,

\begin{equation}
\mathrm{Atenci\acute{o}n}(Q, K, V) \;=\;
\mathrm{softmax}\!\left(\frac{Q K^{\top}}{\sqrt{d_k}}\right) V.
\label{eq:attention}
\end{equation}

En un entorno autorregresivo, una m\'ascara triangular
$M_{ij} = -\infty$ para $j > i$ proh\'ibe que una posici\'on atienda
al futuro:

\begin{equation}
\mathrm{Atenci\acute{o}n}_{\mathrm{causal}}(Q, K, V) =
\mathrm{softmax}\!\left(\frac{Q K^{\top}}{\sqrt{d_k}} + M\right) V.
\label{eq:causal-attention}
\end{equation}

En RL, trabajos recientes~\citep{Parisotto2020,Chen2021} demuestran
que los codificadores Transformer pueden reemplazar funciones de
valor recurrentes con mejor capacidad de captura de dependencias de
largo plazo, ofreciendo una soluci\'on al problema de la
``miop\'ia temporal'' que afecta a los m\'etodos tabulares en
escenarios de inercia epidemiol\'ogica.

\subsection{Cuantificaci\'on de Incertidumbre y Entrop\'ia de Shannon}

Dada una distribuci\'on de probabilidad discreta $p(a \mid s)$ sobre
$|\mathcal{A}|$ acciones, la \emph{entrop\'ia de Shannon} es

\begin{equation}
H(p) \;=\; -\sum_{a \in \mathcal{A}} p(a \mid s)\, \log p(a \mid s).
\label{eq:shannon}
\end{equation}

Normalizada por $\log |\mathcal{A}|$, $H \in [0, 1]$ se interpreta como
una medida de \emph{incertidumbre intr\'inseca} del agente: $H \to 0$
significa que el modelo concentra toda su masa en una sola acci\'on
(alta confianza); $H \to 1$ indica una distribuci\'on uniforme
(m\'axima duda). En esta tesis se propone usar $1 - H_{\mathrm{norm}}$
como proxy de \emph{confianza metacognitiva} del agente, en analog\'ia
con los marcadores neuronales de confianza identificados por Boldt
\& Yeung~\citep{Boldt2015} y revisados por Fleming~\citep{Fleming2024}.
Trabajos como el de Pei \emph{et al.}~\citep{Pei2021} formalizan la
estimaci\'on de incertidumbre en Transformers mediante \emph{atenci\'on
estoc\'astica jer\'arquica}, y Chen \emph{et al.}~\citep{Chen2025}
ofrecen una panor\'amica accesible del campo UQ.

\subsection{Optimizaci\'on Bayesiana de Hiperpar\'ametros}

Un agente de RL tiene varios hiperpar\'ametros no diferenciables
($\alpha$, $\gamma$, tama\~no del \emph{replay buffer},
\emph{schedule} de $\varepsilon$, anchura de red, etc.) cuyas
interacciones son altamente no lineales. La \emph{Optimizaci\'on
Bayesiana} trata el puntaje de validaci\'on como una funci\'on de caja
negra ruidosa y la explora con un \emph{surrogate}
probabil\'istico~\citep{Snoek2012}. Se utiliza el estimador
estructurado en \'arbol de Parzen (TPE)~\citep{Bergstra2011}
implementado en Optuna~\citep{Akiba2019}, que modela
$p(x \mid y < y^{*})$ y $p(x \mid y \geq y^{*})$ con KDEs separados.

\subsection{Comparaci\'on Estad\'istica de Agentes}

Para comparar dos agentes estoc\'asticos $A$ y $B$ sobre $n$ semillas
independientes reportamos tres cantidades:

\begin{description}
    \item[Tama\~no del efecto.] $d$ de Cohen~\citep{Cohen1988},
    \begin{equation}
    d \;=\; \frac{\bar{r}_A - \bar{r}_B}
    {\sqrt{(\sigma_A^{2} + \sigma_B^{2})/2}},
    \end{equation}
    clasificado como peque\~no ($|d| \geq 0{,}2$), medio
    ($|d| \geq 0{,}5$) y grande ($|d| \geq 0{,}8$).
    \item[Significancia.] $t$ de Welch~\citep{Welch1947} con grados
    de libertad de Welch--Satterthwaite.
    \item[Cambio de pol\'itica.] Divergencia de Kullback--Leibler
    entre los histogramas de acci\'on de ambos agentes,
    $D_{\mathrm{KL}}(p_A \,\|\, p_B)$.
\end{description}

\biblio % Needed for referencing to working when compiling individual subfiles - Do not remove
\end{document}


```

## `01_Chapters/03StateOfTheArt.tex`

```latex
﻿\documentclass[../00_Main/Main.tex]{subfiles}
\begin{document}

\label{sec:soa}

La presente investigaci\'on se sit\'ua en la intersecci\'on de la
epidemiolog\'ia computacional~\citep{Kuhl2021,Csefalvay2023}, la
neurociencia cognitiva de la confianza y el aprendizaje profundo
secuencial. El estado del arte actual revela un creciente inter\'es
en sistemas de soporte de decisiones que no s\'olo optimicen una
m\'etrica de rendimiento, sino que incorporen nociones de
\emph{confianza} y \emph{transparencia}.

\subsection{El Legado de Oxford: de la Metacognici\'on Humana a la IA}

El punto de partida conceptual de este trabajo es el experimento
original de la Universidad de Oxford (\emph{dataset} p\'ublico
\texttt{ds004477}), el cual investig\'o las firmas neuronales de la
confianza humana ante decisiones bajo presi\'on. Estudios
fundamentales como los de Boldt \& Yeung~\citep{Boldt2015} y Fleming
\& Dolan~\citep{Fleming2012} identificaron que el cerebro humano
utiliza biomarcadores espec\'ificos (observables mediante EEG) para
monitorear errores y ajustar la certeza de una acci\'on. Trabajos
te\'oricos posteriores~\citep{Grimaldi2015,Fleming2024,Griffith2021}
formalizaron la relaci\'on entre confianza, an\'alisis secuencial y
teor\'ia de detecci\'on de se\~nales. Investigaciones recientes
sobre conflicto cognitivo~\citep{Kuc2021,Nawaz2020} han demostrado
que estos biomarcadores son cruciales en tareas de alta carga.

La originalidad de mPES radica en \emph{tomar esta premisa
neurofisiol\'ogica y trasladarla al plano algor\'itmico}: sustituir
la confianza extra\'ida del EEG por la Entrop\'ia de Shannon
(Ecuaci\'on~\ref{eq:shannon}) como m\'etrica de incertidumbre
intr\'inseca del agente de IA.

\subsection{Aprendizaje por Refuerzo en la Gesti\'on de Crisis}

La literatura en RL ha evolucionado desde m\'etodos tabulares como
el Q-Learning~\citep{Watkins1992,Peng1994} hacia aproximaciones
capaces de manejar la alta dimensionalidad de los problemas del
mundo real~\citep{Ghasemi2024,SuttonBarto2018}.

\begin{itemize}
    \item \textbf{Deep Q-Networks (DQN).} La introducci\'on del
    \emph{Experience Replay}~\citep{Lin1992} y las
    \emph{Target Networks} por Mnih \emph{et al.}~\citep{Mnih2015}
    permiti\'o estabilizar el aprendizaje en entornos complejos.
    Sin embargo, en el contexto epidemiol\'ogico~\citep{Kuhl2021},
    estos modelos suelen sufrir de un \emph{sesgo de
    sobreestimaci\'on} que el Double Q-Learning~\citep{Hasselt2010}
    intenta mitigar mediante el desacoplamiento de la selecci\'on y
    evaluaci\'on de acciones.
    \item \textbf{Mejoras de optimizaci\'on.} El uso de Adam con
    decaimiento c\'iclico~\citep{Kingma2015,Loshchilov2017} y
    \emph{schedules} adaptativos de exploraci\'on como RBED
    (\emph{Reward Based Epsilon Decay})~\citep{Maroti2019} aceleran
    la convergencia.
    \item \textbf{Actor--Critic (A2C).} La arquitectura Actor--Cr\'itico
    se ha identificado como una soluci\'on robusta para reducir la
    varianza en los gradientes de pol\'itica~\citep{Mnih2016}. Su
    capacidad para modelar una funci\'on de ventaja es fundamental
    para gestionar la volatilidad extrema de los datos de contagio.
\end{itemize}

\subsection{El Cambio de Paradigma: Transformers y Modelado de Secuencias}

El mayor obst\'aculo en la gesti\'on de pandemias es la
\emph{inercia temporal}: una decisi\'on hoy afecta la curva de
mortalidad semanas despu\'es. Los modelos tradicionales basados en
MDPs suelen ser ``miopes'', ya que priorizan el estado inmediato.

La propuesta de Vaswani \emph{et al.}~\citep{Vaswani2017} con la
arquitectura Transformer y su posterior adaptaci\'on al RL mediante
el \emph{Decision Transformer}~\citep{Chen2021}, representa un
cambio de paradigma. Al tratar el RL como un problema de
predicci\'on de secuencias y utilizar mecanismos de
\emph{auto-atenci\'on} (Ecuaci\'on~\ref{eq:causal-attention}), el
agente puede capturar dependencias de largo plazo, entendiendo la
\emph{trayectoria completa} de la pandemia en lugar de observar
fotograf\'ias aisladas del presente. Trabajos como el
\emph{Gated Transformer-XL}~\citep{Parisotto2020} mostraron que,
con compuertas e inicializaci\'on adecuadas, los Transformers
igualan o superan a las redes LSTM en tareas que requieren memoria.

\subsection{IA Explicable (XAI) y Cuantificaci\'on de Incertidumbre}

Como se identific\'o en la Revisi\'on Sistem\'atica de la Literatura
realizada para el Entregable 3 de esta maestr\'ia, la adopci\'on de
IA en salud p\'ublica est\'a limitada por la opacidad de los
modelos. La Cuantificaci\'on de Incertidumbre (UQ) mediante el uso
de entrop\'ia jer\'arquica en Transformers~\citep{Pei2021} permite
que el sistema informe cu\'ando ``no est\'a seguro'' de una
decisi\'on. El tutorial de Chen \emph{et al.}~\citep{Chen2025}
sintetiza los enfoques modernos en UQ.

Esto cierra el c\'irculo iniciado en Oxford: si el agente detecta
una alta entrop\'ia (incertidumbre), emula el proceso de duda
humana, permitiendo una intervenci\'on supervisada m\'as segura en
escenarios de crisis. La originalidad de esta tesis consiste,
precisamente, en \emph{operacionalizar} ese puente neurocognitivo
mediante la Entrop\'ia de Shannon sobre la distribuci\'on de salida
del Transformer Causal.

\subsection{Posici\'on de este trabajo}

Comparado con la literatura precedente, la contribuci\'on de esta
tesis es:

\begin{itemize}
    \item \textbf{Seis arquitecturas, un mismo arn\'es experimental.}
    Una comparaci\'on \emph{lado a lado} que abarca m\'etodos
    tabulares, profundos densos, actor--cr\'itico y Transformer
    causal, todos entrenados sobre el mismo MDP y evaluados con
    $n = 64$ secuencias por modelo.
    \item \textbf{UQ algor\'itmica.} La adopci\'on expl\'icita de la
    Entrop\'ia de Shannon como proxy de confianza, en l\'inea con
    el legado neurocient\'ifico de Oxford.
    \item \textbf{Rigor estad\'istico.} Cada diferencia reportada
    viene acompa\~nada de un tama\~no de efecto ($d$ de Cohen) y
    una significancia ($p$ de Welch).
\end{itemize}

\biblio % Needed for referencing to working when compiling individual subfiles - Do not remove
\end{document}


```

## `01_Chapters/04Materials.tex`

```latex
﻿\documentclass[../00_Main/Main.tex]{subfiles}
\begin{document}

\label{sec:methods}

Este cap\'itulo describe el aparato experimental: el repositorio
\texttt{mPES}, la definici\'on formal del entorno, los seis paquetes
algor\'itmicos comparados, el pipeline de optimizaci\'on bayesiana de
hiperpar\'ametros y el protocolo de evaluaci\'on.

\subsection{Repositorio y reproducibilidad}

Todos los experimentos se ejecutan dentro del \emph{workspace}
\texttt{mPES} (\url{https://github.com/Maximiliano0/mPES_2026}), un
proyecto Python organizado en dos familias: \texttt{tabular/}
(m\'etodos enumerativos) y \texttt{ml/} (aprendizaje profundo). Cada
paquete es autocontenido: provee su propio entorno
\texttt{gymnasium}~\citep{Brockman2016,Towers2024}, su rutina de
entrenamiento, su m\'odulo de optimizaci\'on bayesiana y su carpeta
\texttt{outputs/} versionada por fecha. Las dependencias principales
son TensorFlow 2.21, Keras 3.13, NumPy 2.4, Optuna 4.7,
Gymnasium 1.2 y SciPy 1.17.

\subsection{El entorno: \emph{Pandemic Scenario}}

\subsubsection{Definici\'on formal}

El entorno modela una pandemia simplificada como un MDP finito en el
que el agente decide, en cada \emph{trial}, cu\'antas unidades de un
recurso escaso destinar a contener un brote. El espacio de estados
es

\begin{equation}
\mathcal{S} \;=\;
\{(R, t, S) : R \in \{0, \ldots, 39\},\ t \in \{1, \ldots, T\},\
S \in \{0, \ldots, S_{\max}\}\},
\end{equation}

con $R$ los recursos restantes, $t$ el \'indice de \emph{trial} y $S$
la severidad actual; $|\mathcal{S}| = 3\,410$ estados alcanzables.
El espacio de acciones es $\mathcal{A} = \{0, 1, \ldots, 10\}$
(asignaciones que exceden $R$ son enmascaradas).

\subsubsection{Din\'amica y recompensa}

La din\'amica de la severidad sigue la ley exponencial-corregida

\begin{equation}
S_{t+1} \;=\; \max\!\bigl(0,\ \beta \cdot S_{t} - \alpha \cdot a_{t}\bigr),
\qquad \alpha = 0{,}4,\ \beta = 1{,}4,
\label{eq:transicion}
\end{equation}

con $\beta > 1$ capturando el crecimiento natural del contagio
(inercia) y $\alpha$ la eficacia marginal de cada unidad asignada. La
recompensa instant\'anea es

\begin{equation}
r_t \;=\; -\,S_{t},
\end{equation}

de modo que la pol\'itica \'optima minimiza la severidad
acumulada total, en analog\'ia directa con la mortalidad agregada de
la pandemia simulada.

\subsubsection{Estructura del benchmark}

La evaluaci\'on se realiza sobre $n = 64$ secuencias fijas, divididas
en bloques de longitud variable (3--10 \emph{trials}), garantizando
que los seis modelos enfrenten el \emph{mismo} conjunto
\emph{out-of-distribution}.

\subsection{Los seis paquetes algor\'itmicos}

La Tabla~\ref{tab:packages} resume las seis arquitecturas evaluadas.
\textbf{Nota:} la versi\'on completa del repositorio \texttt{mPES}
incluye dos paquetes adicionales (\texttt{pes\_rdqn} y
\texttt{pes\_ens}) que quedaron fuera del alcance del presente
trabajo por razones de tiempo computacional y se reservan para
trabajo futuro (Sec.~\ref{sec:conclusion}).

\begin{table}[h!]
\centering
\caption{Los seis paquetes evaluados.}
\label{tab:packages}
\small
\begin{tabular}{@{}lll@{}}
\toprule
\textbf{Paquete} & \textbf{Familia} & \textbf{Caracter\'istica distintiva} \\
\midrule
\texttt{pes\_base} & Tabular & Q-Learning de referencia~\citep{Watkins1992}.\\
\texttt{pes\_ql}   & Tabular & Q-Learning + Optuna/TPE.\\
\texttt{pes\_dql}  & Tabular & Double Q-Learning + PBRS + $\varepsilon$ \emph{warm-up}.\\
\texttt{pes\_dqn}  & Profundo & DQN con \emph{replay} y red objetivo~\citep{Mnih2015}.\\
\texttt{pes\_a2c}  & Profundo & Actor--Cr\'itico (A2C)~\citep{Mnih2016}.\\
\texttt{pes\_trf}  & Profundo & Codificador Transformer causal sobre historia.\\
\bottomrule
\end{tabular}
\end{table}

\subsubsection{Familia tabular}

\paragraph{\texttt{pes\_base}.} Q-Learning enumerativo con
hiperpar\'ametros fijos; sirve de l\'inea de base inferior.

\paragraph{\texttt{pes\_ql}.} A\~nade un bucle externo Optuna/TPE que
muestrea $\alpha,\gamma$ y la pol\'itica de $\varepsilon$.

\paragraph{\texttt{pes\_dql}.} Implementa Double
Q-Learning~\citep{Hasselt2010}, una fase de calentamiento de
$\varepsilon$ y \emph{Potential-Based Reward Shaping}
$\Phi(s) = -S$~\citep{Ng1999}, especialmente \'util para acelerar la
convergencia ante una se\~nal de recompensa concentrada al final del
\emph{trial}.

\subsubsection{Familia profunda}

\paragraph{\texttt{pes\_dqn}.} Red densa Multi-Layer Perceptron con
buffer de repetici\'on de experiencia~\citep{Lin1992} y red objetivo
sincronizada cada $C$ pasos (Ecuaci\'on~\ref{eq:dqn-loss}).

\paragraph{\texttt{pes\_a2c}.} Dos cabezales independientes (actor y
cr\'itico) sobre un tronco compartido; entrenamiento s\'incrono
n-\emph{step} con estimaci\'on de ventaja
(Ecuaci\'on~\ref{eq:pg})~\citep{Mnih2016}.

\paragraph{\texttt{pes\_trf}.} Codificador Transformer causal de
$L$ capas, $h$ cabezas y dimensi\'on $d_{\mathrm{model}}$. La entrada
es la \emph{historia} de los \'ultimos $W$ tuplas
$(s_{t-W+1}, a_{t-W+1}, r_{t-W+1}, \ldots, s_t)$; la salida sobre la
\'ultima posici\'on alimenta una cabeza lineal que produce los
$|\mathcal{A}|$ \emph{logits}. La m\'ascara causal
(Ecuaci\'on~\ref{eq:causal-attention}) impide cualquier fuga
temporal. La distribuci\'on \texttt{softmax} sobre los \emph{logits}
es la base del c\'alculo de la \textbf{entrop\'ia de Shannon}
(Ecuaci\'on~\ref{eq:shannon}) usada como m\'etrica de UQ.

\subsection{Optimizaci\'on Bayesiana de Hiperpar\'ametros}

Cada paquete profundo expone un m\'odulo \texttt{optimize\_*.py} que
ejecuta un estudio Optuna con estimador estructurado en \'arbol de
Parzen (TPE)~\citep{Bergstra2011,Snoek2012,Akiba2019}. La funci\'on
objetivo $f(\boldsymbol{\theta})$ es la mediana del retorno
descontado sobre las $n = 64$ secuencias de validaci\'on. Los
estudios incluyen \emph{pruning} mediante \texttt{MedianPruner}
para descartar tempranamente configuraciones inviables y conservan
el almacenamiento SQLite (\texttt{optuna\_storage.db}) en
\texttt{outputs/}, permitiendo reanudaci\'on y visualizaci\'on con
\texttt{optuna-dashboard}.

\subsection{Protocolo de entrenamiento y evaluaci\'on}

\begin{enumerate}
\item \textbf{B\'usqueda Bayesiana.} 100--300 \emph{trials} Optuna por
paquete (las redes profundas requieren m\'as \emph{trials} por su
mayor dimensi\'on de hiperpar\'ametros).
\item \textbf{Reentrenamiento final.} El mejor conjunto
$\boldsymbol{\theta}^{*}$ se reentrena desde cero por
$E_{\max} = 2000$ episodios.
\item \textbf{Evaluaci\'on \emph{out-of-policy}.} Se evaluan las
$n = 64$ secuencias de validaci\'on con la pol\'itica
$\varepsilon$-greedy ($\varepsilon = 0$).
\item \textbf{An\'alisis estad\'istico.} Para cada par de paquetes
$(A, B)$ se reporta la media, desv\'iaci\'on est\'andar, $d$ de
Cohen~\citep{Cohen1988}, $t$/$p$ de Welch~\citep{Welch1947} y
divergencia KL entre los histogramas de acci\'on, agregadas como
matrices y \emph{heatmaps} (Cap.~\ref{sec:results}).
\end{enumerate}

\subsection{Generaci\'on autom\'atica del reporte}

El orquestador \texttt{general/scripts/orchestrate.py} ejecuta todos
los paquetes, agrega resultados con \texttt{aggregate.py}, produce
los \emph{heatmaps} con \texttt{plot\_matrix.py} y emite el reporte
final \texttt{benchmark\_report.md} mediante \texttt{report.py}. Esto
garantiza que cada cifra reportada en el Cap.~\ref{sec:results}
provenga de un pipeline reproducible y versionado.

\biblio % Needed for referencing to working when compiling individual subfiles - Do not remove
\end{document}


```

## `01_Chapters/05Results.tex`

```latex
﻿\documentclass[../00_Main/Main.tex]{subfiles}
\begin{document}

\label{sec:results}

Este cap\'itulo presenta los resultados emp\'iricos del benchmark
mPES, comparando las seis arquitecturas sobre $n = 64$ secuencias
\emph{out-of-distribution} agrupadas en cinco familias de
distribuci\'on de severidad y cinco familias de longitud de
secuencia. La m\'etrica principal es la \emph{recompensa normalizada
media} $\bar{r} \in [0, 1]$, donde $1$ representa la pol\'itica
\'optima.

\subsection{Visi\'on global}

La Tabla~\ref{tab:global-mean} resume el desempe\~no global agregado
sobre las 22 condiciones del benchmark. \texttt{pes\_trf} domina con
$\bar{r} = 0{,}927$, seguido por la familia profunda
(\texttt{pes\_dqn}, \texttt{pes\_a2c} y \texttt{pes\_dql}, todos
$\approx 0{,}89$). Los m\'etodos tabulares puros
(\texttt{pes\_base}/\texttt{pes\_ql}) se quedan en
$\bar{r} \approx 0{,}88$, mostrando que el l\'imite de la
representaci\'on enumerativa se alcanza r\'apido en este MDP.

\begin{table}[h!]
\centering
\caption{Recompensa normalizada media por modelo (agregada sobre
22 escenarios; $n = 64$ secuencias por escenario).}
\label{tab:global-mean}
\begin{tabular}{@{}lc@{}}
\toprule
\textbf{Modelo} & $\bar{r}$ \\
\midrule
\texttt{pes\_ql}  & 0{,}887 \\
\texttt{pes\_dql} & 0{,}896 \\
\texttt{pes\_dqn} & 0{,}894 \\
\texttt{pes\_a2c} & 0{,}887 \\
\textbf{\texttt{pes\_trf}} & \textbf{0{,}927} \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Reducci\'on de la mortalidad}

Tomando \texttt{pes\_ql} como l\'inea de base y midiendo la
mortalidad (severidad acumulada) como $1 - \bar{r}$ promediada sobre
las 22 condiciones, se obtiene:

\begin{equation}
\Delta_{\mathrm{trf}} \;=\;
\frac{(1 - \bar{r}_{\mathrm{ql}}) - (1 - \bar{r}_{\mathrm{trf}})}
{1 - \bar{r}_{\mathrm{ql}}}
\;=\; \frac{0{,}113 - 0{,}073}{0{,}113}
\;\approx\; \mathbf{0{,}35},
\end{equation}

es decir, \texttt{pes\_trf} reduce la severidad acumulada en torno a
un \textbf{22--35\%} respecto de la l\'inea de base tabular, con el
m\'inimo correspondiente al escenario emp\'irico est\'andar y el
m\'aximo a las condiciones extrapoladas. Este resultado confirma la
hip\'otesis $H_1$ formulada en el Cap.~\ref{sec:problema}.

\subsection{Comportamiento por familia algor\'itmica}

\begin{description}
    \item[Tabular (\texttt{pes\_base, pes\_ql, pes\_dql}).] Eficaz en
    escenarios cortos y con dispersi\'on de severidad acotada
    (sev\_empirical, len\_all\_short). Su desempe\~no se degrada en
    escenarios largos (len\_all\_long, joint\_high\_long), donde la
    inercia exponencial del contagio (Ec.~\ref{eq:transicion}) excede
    la capacidad de la tabla para generalizar entre estados similares.
    PBRS aporta una mejora marginal pero no estructural.
    \item[\texttt{pes\_dqn}.] El reemplazo de la tabla por un MLP
    permite una mejor exploraci\'on del espacio de estados denso. Es
    notablemente m\'as robusto en escenarios extrapolados
    (\texttt{sev\_extrapolate\_high}: 0{,}890 vs 0{,}762 de
    \texttt{pes\_ql}), validando la ventaja de la aproximaci\'on
    param\'etrica.
    \item[\texttt{pes\_a2c}.] M\'as estable que DQN en escenarios
    de alta varianza (joint\_extrap\_both: 0{,}949), aunque
    ligeramente inferior en el agregado. La separaci\'on actor/cr\'itico
    suaviza la varianza de las actualizaciones.
    \item[\texttt{pes\_trf}.] Domina en pr\'acticamente todas las
    columnas, con el salto m\'as notable en condiciones extremas:
    \texttt{sev\_extrapolate\_high} alcanza $0{,}996$ y
    \texttt{joint\_extrap\_both} $0{,}997$. La auto-atenci\'on causal
    captura la \emph{trayectoria} del brote, permitiendo decisiones
    anticipatorias frente al crecimiento exponencial.
\end{description}

\subsection{An\'alisis de Cuantificaci\'on de Incertidumbre}

Calculando la entrop\'ia de Shannon normalizada
$H_{\mathrm{norm}}(p)$ (Ec.~\ref{eq:shannon}) sobre la distribuci\'on
\texttt{softmax} del Transformer durante las 64 secuencias, se
observa un patr\'on consistente:

\begin{itemize}
    \item Durante las fases de crecimiento exponencial
    (severidad creciente), $H_{\mathrm{norm}}$ desciende a valores
    $< 0{,}25$, indicando alta \emph{confianza} del agente en su
    asignaci\'on agresiva de recursos.
    \item Durante mesetas o decrecimientos, $H_{\mathrm{norm}}$
    sube por encima de $0{,}5$, reflejando que m\'ultiples
    acciones (incluyendo $a = 0$) son aceptables.
\end{itemize}

Este comportamiento es \emph{exactamente} el espejo algor\'itmico
del marcador neurofisiol\'ogico de confianza descrito por Boldt
\& Yeung~\citep{Boldt2015}: el sistema reduce la duda en los momentos
de mayor demanda decisional, y la incrementa cuando el horizonte se
estabiliza. Se discuten las implicaciones en el Cap.~\ref{sec:discussion}.

\subsection{Salidas autom\'aticas del orquestador}

El orquestador \texttt{general/scripts/orchestrate.py} genera de
forma reproducible:

\begin{itemize}
    \item \texttt{general/results/matrix\_global\_mean.csv} (Tabla
    completa de los seis modelos $\times$ 22 escenarios).
    \item Heatmaps \texttt{matrix\_cohen\_d.csv},
    \texttt{matrix\_welch\_p.csv}, \texttt{matrix\_action\_kl.csv}.
    \item Histogramas por secuencia en
    \texttt{general/results/per\_sequence\_histograms/}.
    \item Reporte final \texttt{general/results/benchmark\_report.md}.
\end{itemize}

% Las figuras se pueden incluir comentadas para futura inserci\'on:
% \begin{figure}[h!]
%   \centering
%   \includegraphics[width=0.9\textwidth]{../02_Images/heatmap_cohen_d.pdf}
%   \caption{Tama\~no del efecto $d$ de Cohen entre cada par de modelos.}
%   \label{fig:heatmap-cohen}
% \end{figure}

\biblio % Needed for referencing to working when compiling individual subfiles - Do not remove
\end{document}


```

## `01_Chapters/06Discussion.tex`

```latex
﻿\documentclass[../00_Main/Main.tex]{subfiles}
\begin{document}

\label{sec:discussion}

Los resultados del Cap.~\ref{sec:results} permiten responder, de
manera afirmativa, la hip\'otesis $H_1$: la arquitectura
\emph{Transformer causal} es la que mejor resuelve el problema de
asignaci\'on pand\'emica modelado en mPES, no s\'olo en m\'etrica
agregada sino sobre todo en escenarios extrapolados. Esta secci\'on
discute las implicaciones m\'as relevantes.

\subsection{Por qu\'e gana el Transformer}

La din\'amica de severidad
$S_{t+1} = \max(0, \beta S_t - \alpha a_t)$ (Ec.~\ref{eq:transicion})
es ex\'oticamente \emph{ag\'rega} en el tiempo: un d\'eficit de
recursos en $t$ no se traduce en costo inmediato, sino en una
amplificaci\'on geom\'etrica sobre los \emph{trials} siguientes. Esta
estructura penaliza fuertemente a los m\'etodos ``mi\'opes'' (tabular
y, en menor medida, DQN feed-forward) que tienden a sub-asignar al
principio de las secuencias largas.

El codificador Transformer, al observar la \emph{ventana hist\'orica}
completa $\{s_{t-W+1}, \ldots, s_t\}$ con atenci\'on causal
(Ec.~\ref{eq:causal-attention}), construye una representaci\'on que
codifica expl\'icitamente la \emph{velocidad y aceleraci\'on} del
crecimiento de la severidad. Esto le permite anticipar el
crecimiento exponencial y movilizar recursos antes de que se vuelva
incontenible.

\subsection{Entrop\'ia de Shannon como confianza algor\'itmica}

El hallazgo m\'as conceptualmente importante de esta tesis no es la
mejora en $\bar{r}$ sino el patr\'on de entrop\'ia: la
$H_{\mathrm{norm}}$ del Transformer \emph{baja} justo cuando la
amenaza es m\'as grave, y \emph{sube} cuando el entorno se
estabiliza. Esto sugiere que la red ha aprendido, sin supervisi\'on
expl\'icita, una forma de \emph{metacognici\'on emergente}:

\begin{quote}
\emph{El agente sabe cu\'ando sabe.}
\end{quote}

Este comportamiento es exactamente el descrito por
Boldt \& Yeung~\citep{Boldt2015} y formalizado por
Fleming~\citep{Fleming2024} para la confianza humana medida con EEG.
La aportaci\'on metodol\'ogica de la tesis es haber traducido ese
biomarcador a una m\'etrica reproducible:
$1 - H_{\mathrm{norm}}(\pi(a \mid s))$, calculable en cualquier
modelo con salida \texttt{softmax}, sin necesidad de hardware de
neuroimagen.

\subsection{Del EEG al \emph{softmax}: validez externa}

La transici\'on del paradigma neurocognitivo (Oxford, ds004477) al
algor\'itmico (mPES) admite dos lecturas complementarias:

\begin{enumerate}
    \item \textbf{Operacional.} Permite construir sistemas de soporte
    de decisi\'on que comunican su confianza al humano supervisor,
    abilitando \emph{human-in-the-loop} significativo. Cuando el
    agente reporta $H_{\mathrm{norm}} > 0{,}7$, el sistema puede
    delegar al operador humano la decisi\'on, alineando IA y
    epidemiolog\'ia~\citep{Kuhl2021}.
    \item \textbf{Te\'orica.} Aporta evidencia computacional a la
    hip\'otesis de que la confianza es un fen\'omeno
    \emph{representacional}, no exclusivamente neuronal: cualquier
    sistema que mantenga una distribuci\'on de creencias sobre sus
    propias acciones genera, naturalmente, una se\~nal de UQ
    interpretable~\citep{Chen2025,Pei2021}.
\end{enumerate}

\subsection{Limitaciones}

\begin{itemize}
    \item \textbf{Cobertura algor\'itmica.} El benchmark se limit\'o
    a seis paquetes; los m\'odulos recurrente (\texttt{pes\_rdqn}) y
    ensemble (\texttt{pes\_ens}) presentes en el repositorio quedaron
    fuera del alcance temporal del Entregable.
    \item \textbf{Costo computacional.} \texttt{pes\_trf} requiere
    aproximadamente $5\times$ m\'as tiempo de entrenamiento que
    \texttt{pes\_dqn}, lo que puede ser un factor decisivo en
    despliegues con presupuesto limitado.
    \item \textbf{Simplicidad del entorno.} El \emph{Pandemic
    Scenario} es un proxy abstracto; la validaci\'on externa contra
    datos reales (p.~ej.~series temporales de hospitalizaciones)
    queda como trabajo futuro.
    \item \textbf{$H$ como UQ.} La entrop\'ia de Shannon es una
    medida \emph{epist\'emica} sobre la pol\'itica, no sobre la
    funci\'on de valor; refinamientos como la entrop\'ia jer\'arquica
    de Pei \emph{et al.}~\citep{Pei2021} merecen una evaluaci\'on
    dedicada.
\end{itemize}

\biblio % Needed for referencing to working when compiling individual subfiles - Do not remove
\end{document}


```

## `01_Chapters/07Conclusion.tex`

```latex
﻿\documentclass[../00_Main/Main.tex]{subfiles}
\begin{document}

\label{sec:conclusion}

Esta tesis present\'o \textbf{mPES}, una plataforma de aprendizaje
por refuerzo para la asignaci\'on \'optima de recursos en una
pandemia simulada, y la utiliz\'o como banco de prueba para una
comparaci\'on rigurosa entre seis arquitecturas: tres tabulares
(\texttt{pes\_base}, \texttt{pes\_ql}, \texttt{pes\_dql}) y tres
profundas (\texttt{pes\_dqn}, \texttt{pes\_a2c},
\texttt{pes\_trf}).

\subsection{Mensajes principales}

\begin{enumerate}
    \item \textbf{La arquitectura importa, no s\'olo los
    hiperpar\'ametros.} El salto entre \texttt{pes\_ql}
    (Q-Learning optimizado con Optuna) y \texttt{pes\_trf}
    (Transformer causal) es estructural: aproximadamente un
    \textbf{22--35\% de reducci\'on en mortalidad acumulada}, no
    alcanzable por una mejor b\'usqueda de hiperpar\'ametros sobre
    la representaci\'on tabular.
    \item \textbf{El Transformer es la mejor opci\'on.} La
    auto-atenci\'on causal permite anticipar el crecimiento
    exponencial del contagio, condici\'on \emph{sine qua non} para
    decisiones de inercia alta.
    \item \textbf{La Entrop\'ia de Shannon es una m\'etrica
    leg\'itima de UQ.} El comportamiento de
    $H_{\mathrm{norm}}(\pi)$ a lo largo de la trayectoria reproduce
    cualitativamente los marcadores neurofisiol\'ogicos de confianza
    identificados en el experimento de Oxford
    (\texttt{ds004477}~\citep{Boldt2015,Fleming2024}), validando la
    transici\'on del paradigma neurocognitivo al algor\'itmico.
    \item \textbf{El benchmark es reproducible.} El pipeline
    \texttt{general/scripts/orchestrate.py} regenera todos los CSVs,
    heatmaps y el reporte \texttt{benchmark\_report.md}, lo que
    facilita la extensi\'on y la auditor\'ia externa.
\end{enumerate}

\subsection{Trabajo futuro}

\begin{itemize}
    \item Integrar las arquitecturas excluidas del Entregable:
    \texttt{pes\_rdqn} (Recurrent DQN basado en LSTM) y
    \texttt{pes\_ens} (ensemble con voto suave), para verificar si
    aportan robustez adicional respecto del Transformer.
    \item Sustituir el escenario simulado por series temporales
    reales (p.~ej.~datos COVID-19 desagregados por regi\'on) y
    evaluar generalizaci\'on.
    \item Explorar variantes m\'as ricas de UQ: \emph{atenci\'on
    estoc\'astica jer\'arquica}~\citep{Pei2021}, redes bayesianas
    y \emph{deep ensembles}~\citep{Chen2025}.
    \item Estudiar el acoplamiento humano--agente: dise\~nar
    interfaces que comuniquen $H_{\mathrm{norm}}$ al decisor humano
    y medir su impacto en la calidad de la decisi\'on conjunta.
\end{itemize}

\subsection{Cierre}

mPES demuestra que, en problemas de decisi\'on bajo crisis con
fuerte componente temporal, la elecci\'on de la arquitectura es
decisiva, y que la combinaci\'on de Transformers y Cuantificaci\'on
de Incertidumbre ofrece a la vez \emph{eficiencia} (mejor pol\'itica)
y \emph{transparencia} (confianza interpretable). Esa doble virtud es
lo que distingue una herramienta utilizable de un experimento
acad\'emico.

\biblio % Needed for referencing to working when compiling individual subfiles - Do not remove
\end{document}


```

## `01_Chapters/Acknowledgement.tex`

```latex
﻿\documentclass[../00_Main/Main.tex]{subfiles}
\begin{document}

A mi director, \textbf{Dr.\ Rodrigo Ramele}, por la gu\'ia paciente
y rigurosa a lo largo de la Maestr\'ia en Ciencia de Datos, y por
plantear desde el primer d\'ia el desaf\'io de tender un puente entre
la neurociencia de la confianza y los modelos de aprendizaje por
refuerzo. Esta tesis es deudora directa de esa visi\'on.

Al cuerpo docente del \textbf{Instituto Tecnol\'ogico de Buenos
Aires (ITBA)} por la formaci\'on s\'olida y la apertura para discutir
ideas fuera del programa.

A mi familia y a Magal\'i, por la paciencia infinita durante los
meses de entrenamiento de redes que se negaban a converger.

A la comunidad \emph{open-source} detr\'as de Python, NumPy,
TensorFlow, Keras, Gymnasium y Optuna, sin la cual este trabajo
ser\'ia simplemente impensable.

\biblio % Needed for referencing to working when compiling individual subfiles - Do not remove
\end{document}


```

## `01_Chapters/Appendix.tex`

```latex
﻿\documentclass[../00_Main/Main.tex]{subfiles}
\begin{document}

Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum

\biblio % Needed for referencing to working when compiling individual subfiles - Do not remove
\end{document}

```

## `00_Main/References.bib`

```bibtex
% References.bib
% BibTeX database for the thesis "A Comparative Evaluation of
% Reinforcement Learning Architectures for Resource Allocation under
% Uncertainty in Pandemic Scenarios" (Vega, ITBA, 2026).


@manual{Microsoft.2024,
title  = "Neurips Conference Analytics",
mendeley-groups = {Balance},
author = "Microsoft Academics",
note   = "\url{https://www.microsoft.com/en-us/research/project/academic/articles/neurips-conference-analytics/}",
year   = "2024 (accessed Feb 27, 2024)"
}


@article{Endo,
abstract = {Home Oxydgen Therapy (H.O.T.) is a medical treatment for severe lung diseases in which the patients are supplied concentrated oxygen. This paper investigates the use of a follower robot as a support device for H.O.T. patients, consisting of a two-wheeled differential drive robot connected to the user by tether. Two different control algorithms were studied using dynamic simulation and motion capture experiments with healthy subjects. In further experiments with H.O.T. patients, including a questionnaire survey, it was confirmed that Follow the Leader control was capable of following the user's trajectory more accurately than Pseudo-Joystick control, and that overall H.O.T. patients showed a preference for Follow the Leader control.},
author = {Endo, Gen and Allan, Ben and Iemura, Yu and Fukushima, Edwardo F. and Iribe, Masatsugu and Takubo, Toshio and Ohira, Mineko},
doi = {10.1186/s40648-014-0026-3},
file = {:Users/rramele/Library/Application Support/Mendeley Desktop/Downloaded/Endo et al. - 2015 - Mobile follower robot as an assistive device for home oxygen therapy – evaluation of tether control algorithms.pdf:pdf},
issn = {21974225},
journal = {ROBOMECH Journal},
keywords = {Home oxygen therapy,Leader following,Mobile robot,Tether},
mendeley-groups = {AlpiBot},
month = {dec},
number = {1},
pages = {6},
publisher = {Springer International Publishing},
title = {{Mobile follower robot as an assistive device for home oxygen therapy – evaluation of tether control algorithms}},
url = {http://www.robomechjournal.com/content/2/1/6},
volume = {2},
year = {2015}
}


@article{ramele2019histogram,
  title={Histogram of gradient orientations of signal plots applied to P300 detection},
  author={Ramele, Rodrigo and Villar, Ana Julia and Santos, Juan Miguel},
  journal={Frontiers in computational neuroscience},
  volume={13},
  year={2019},
  publisher={Frontiers Media SA}
}

@article{uriguen2015eeg,
  title={EEG artifact removal—state-of-the-art and guidelines},
  author={Urig{\"u}en, Jose Antonio and Garcia-Zapirain, Bego{\~n}a},
  journal={Journal of neural engineering},
  volume={12},
  number={3},
  pages={031001},
  year={2015},
  publisher={IOP Publishing}
}

@book{AIAMA,
  title={Artificial Intelligence: A Modern Approach},
  author={Norvig, Peter and Russell, Stuart},
  year={2009},
  publisher={Prentice Hall Press Upper Saddle River}
}

@book{wolpaw2012brain,
  title={Brain-computer interfaces: principles and practice},
  author={Wolpaw, Jonathan and Wolpaw, Elizabeth Winter},
  year={2012},
  publisher={OUP USA}
}

@Manual{MNE,
    title = {MNE software for processing MEG and EEG data, NeuroImage, Volume 86},
    author = {A. Gramfort, M. Luessi, E. Larson, D. Engemann, D. Strohmeier, C. Brodbeck, L. Parkkonen, M. Hämäläinen},
    year = {2014},
    url = {https://mne-tools.github.io/0.13/index.html}
  }
  
@Manual{MNE-PYTHON,
    title = {MEG and EEG data analysis with MNE-Python, Frontiers in Neuroscience, Volume 7},
    author = {A. Gramfort, M. Luessi, E. Larson, D. Engemann, D. Strohmeier, C. Brodbeck, R. Goj, M. Jas, T. Brooks, L. Parkkonen, M. Hämäläinen},
    year = {2013},
    url = {https://mne-tools.github.io/0.13/index.html}
  }


% ==================================================================
% Entries added for the mPES thesis (Vega, ITBA, 2026).
% ==================================================================

% ----- Foundations of Reinforcement Learning -----

@book{SuttonBarto2018,
  title     = {Reinforcement Learning: An Introduction},
  author    = {Sutton, Richard S. and Barto, Andrew G.},
  year      = {2018},
  edition   = {2nd},
  publisher = {The MIT Press},
  address   = {Cambridge, MA}
}

@article{Watkins1992,
  title   = {Q-learning},
  author  = {Watkins, Christopher J. C. H. and Dayan, Peter},
  journal = {Machine Learning},
  volume  = {8},
  number  = {3--4},
  pages   = {279--292},
  year    = {1992},
  publisher = {Springer}
}

@incollection{Hasselt2010,
  title     = {Double {Q}-learning},
  author    = {van Hasselt, Hado},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {23},
  pages     = {2613--2621},
  year      = {2010}
}

@inproceedings{Hasselt2016,
  title     = {Deep Reinforcement Learning with Double {Q}-learning},
  author    = {van Hasselt, Hado and Guez, Arthur and Silver, David},
  booktitle = {Proceedings of the {AAAI} Conference on Artificial Intelligence},
  year      = {2016}
}

@inproceedings{Ng1999,
  title     = {Policy invariance under reward transformations: Theory and application to reward shaping},
  author    = {Ng, Andrew Y. and Harada, Daishi and Russell, Stuart},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {1999}
}

@article{Williams1992,
  title   = {Simple statistical gradient-following algorithms for connectionist reinforcement learning},
  author  = {Williams, Ronald J.},
  journal = {Machine Learning},
  volume  = {8},
  number  = {3--4},
  pages   = {229--256},
  year    = {1992}
}

% ----- Deep RL -----

@article{Mnih2015,
  title   = {Human-level control through deep reinforcement learning},
  author  = {Mnih, Volodymyr and Kavukcuoglu, Koray and Silver, David and Rusu, Andrei A. and Veness, Joel and Bellemare, Marc G. and Graves, Alex and Riedmiller, Martin and Fidjeland, Andreas K. and Ostrovski, Georg and others},
  journal = {Nature},
  volume  = {518},
  number  = {7540},
  pages   = {529--533},
  year    = {2015},
  publisher = {Nature Publishing Group}
}

@article{Silver2017,
  title   = {Mastering the game of {Go} without human knowledge},
  author  = {Silver, David and Schrittwieser, Julian and Simonyan, Karen and Antonoglou, Ioannis and Huang, Aja and Guez, Arthur and Hubert, Thomas and others},
  journal = {Nature},
  volume  = {550},
  number  = {7676},
  pages   = {354--359},
  year    = {2017}
}

@inproceedings{Mnih2016,
  title     = {Asynchronous methods for deep reinforcement learning},
  author    = {Mnih, Volodymyr and Badia, Adri{\`a} Puigdom{\`e}nech and Mirza, Mehdi and Graves, Alex and Lillicrap, Timothy and Harley, Tim and Silver, David and Kavukcuoglu, Koray},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2016}
}

@article{Hochreiter1997,
  title   = {Long short-term memory},
  author  = {Hochreiter, Sepp and Schmidhuber, J{\"u}rgen},
  journal = {Neural Computation},
  volume  = {9},
  number  = {8},
  pages   = {1735--1780},
  year    = {1997},
  publisher = {MIT Press}
}

@inproceedings{Hausknecht2015,
  title     = {Deep recurrent {Q}-learning for partially observable {MDPs}},
  author    = {Hausknecht, Matthew and Stone, Peter},
  booktitle = {AAAI Fall Symposium Series},
  year      = {2015}
}

@inproceedings{Kingma2015,
  title     = {{Adam}: A Method for Stochastic Optimization},
  author    = {Kingma, Diederik P. and Ba, Jimmy},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2015}
}

% ----- Attention / Transformers -----

@inproceedings{Vaswani2017,
  title     = {Attention Is All You Need},
  author    = {Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N. and Kaiser, {\L}ukasz and Polosukhin, Illia},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2017}
}

@inproceedings{Parisotto2020,
  title     = {Stabilizing transformers for reinforcement learning},
  author    = {Parisotto, Emilio and Song, H. Francis and Rae, Jack W. and Pascanu, Razvan and Gulcehre, Caglar and Jayakumar, Siddhant M. and Jaderberg, Max and Kaufman, Raphael Lopez and Clark, Aidan and Noury, Seb and Botvinick, Matthew M. and Heess, Nicolas and Hadsell, Raia},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2020}
}

@inproceedings{Chen2021,
  title     = {Decision Transformer: Reinforcement Learning via Sequence Modeling},
  author    = {Chen, Lili and Lu, Kevin and Rajeswaran, Aravind and Lee, Kimin and Grover, Aditya and Laskin, Michael and Abbeel, Pieter and Srinivas, Aravind and Mordatch, Igor},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2021}
}

@inproceedings{Janner2021,
  title     = {Offline Reinforcement Learning as One Big Sequence Modeling Problem},
  author    = {Janner, Michael and Li, Qiyang and Levine, Sergey},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2021}
}

% ----- Ensembles -----

@incollection{Dietterich2000,
  title     = {Ensemble methods in machine learning},
  author    = {Dietterich, Thomas G.},
  booktitle = {Multiple Classifier Systems},
  pages     = {1--15},
  year      = {2000},
  publisher = {Springer}
}

@inproceedings{Osband2016,
  title     = {Deep exploration via bootstrapped {DQN}},
  author    = {Osband, Ian and Blundell, Charles and Pritzel, Alexander and Van Roy, Benjamin},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2016}
}

@inproceedings{Anschel2017,
  title     = {{Averaged-DQN}: Variance reduction and stabilization for deep reinforcement learning},
  author    = {Anschel, Oron and Baram, Nir and Shimkin, Nahum},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2017}
}

@incollection{Wiering2008,
  title     = {Ensemble algorithms in reinforcement learning},
  author    = {Wiering, Marco A. and van Hasselt, Hado},
  booktitle = {IEEE Transactions on Systems, Man, and Cybernetics, Part B},
  volume    = {38},
  number    = {4},
  pages     = {930--936},
  year      = {2008}
}

% ----- Generalisation and OOD -----

@inproceedings{Cobbe2019,
  title     = {Quantifying generalization in reinforcement learning},
  author    = {Cobbe, Karl and Klimov, Oleg and Hesse, Christopher and Kim, Taehoon and Schulman, John},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2019}
}

@inproceedings{Cobbe2020,
  title     = {Leveraging procedural generation to benchmark reinforcement learning},
  author    = {Cobbe, Karl and Hesse, Christopher and Hilton, Jacob and Schulman, John},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2020}
}

@article{Packer2018,
  title   = {Assessing Generalization in Deep Reinforcement Learning},
  author  = {Packer, Charles and Gao, Katelyn and Kos, Jernej and Kr{\"a}henb{\"u}hl, Philipp and Koltun, Vladlen and Song, Dawn},
  journal = {arXiv:1810.12282},
  year    = {2018}
}

@article{Kirk2023,
  title   = {A survey of zero-shot generalisation in deep reinforcement learning},
  author  = {Kirk, Robert and Zhang, Amy and Grefenstette, Edward and Rockt{\"a}schel, Tim},
  journal = {Journal of Artificial Intelligence Research},
  volume  = {76},
  pages   = {201--264},
  year    = {2023}
}

@inproceedings{Henderson2018,
  title     = {Deep reinforcement learning that matters},
  author    = {Henderson, Peter and Islam, Riashat and Bachman, Philip and Pineau, Joelle and Precup, Doina and Meger, David},
  booktitle = {AAAI Conference on Artificial Intelligence},
  year      = {2018}
}

% ----- Hyper-parameter optimisation -----

@inproceedings{Akiba2019,
  title     = {{Optuna}: A next-generation hyperparameter optimization framework},
  author    = {Akiba, Takuya and Sano, Shotaro and Yanase, Toshihiko and Ohta, Takeru and Koyama, Masanori},
  booktitle = {ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD)},
  year      = {2019}
}

@inproceedings{Bergstra2011,
  title     = {Algorithms for hyper-parameter optimization},
  author    = {Bergstra, James and Bardenet, R{\'e}mi and Bengio, Yoshua and K{\'e}gl, Bal{\'a}zs},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2011}
}

% ----- Statistics -----

@book{Cohen1988,
  title     = {Statistical Power Analysis for the Behavioral Sciences},
  author    = {Cohen, Jacob},
  year      = {1988},
  edition   = {2nd},
  publisher = {Lawrence Erlbaum Associates},
  address   = {Hillsdale, NJ}
}

@article{Welch1947,
  title   = {The generalization of {Student}'s problem when several different population variances are involved},
  author  = {Welch, Bernard L.},
  journal = {Biometrika},
  volume  = {34},
  number  = {1/2},
  pages   = {28--35},
  year    = {1947}
}

% ----- RL for pandemic / public-health resource allocation -----

@article{Bertsimas2021,
  title   = {From predictions to prescriptions: A data-driven response to {COVID-19}},
  author  = {Bertsimas, Dimitris and Boussioux, L{\'e}onard and Cory-Wright, Ryan and Delarue, Arthur and Digalakis, Vassilis and Jacquillat, Alexandre and Kitane, Driss Lahlou and Lukin, Galit and Li, Michael Lingzhi and Mingardi, Luca and others},
  journal = {Health Care Management Science},
  volume  = {24},
  pages   = {253--272},
  year    = {2021}
}

@article{Yanez2020,
  title   = {{COVID-19} epidemic control using short-term lockdowns for collective gain},
  author  = {Y{\'a}{\~n}ez, Andrea and Hayes, Conor and Glavin, Frank},
  journal = {arXiv:2003.13546},
  year    = {2020}
}

@article{Kompella2020,
  title   = {Reinforcement learning for optimization of {COVID-19} mitigation policies},
  author  = {Kompella, Varun and Capobianco, Roberto and Jong, Stacy and Browne, Jonathan and Fox, Spencer and Meyers, Lauren and Wurman, Peter and Stone, Peter},
  journal = {arXiv:2010.10560},
  year    = {2020}
}

@article{Khalilpourazari2021,
  title   = {Designing emergency flood evacuation plans using robust optimization and artificial intelligence},
  author  = {Khalilpourazari, Soheyl and Hashemi Doulabi, Hossein},
  journal = {Journal of Combinatorial Optimization},
  year    = {2021}
}

% ----- Gymnasium -----

@misc{Gymnasium2023,
  title  = {{Gymnasium}: A standard interface for reinforcement learning environments},
  author = {Towers, Mark and Terry, Jordan K. and Kwiatkowski, Ariel and Balis, John U. and {de Cola}, Gianluca and Deleu, Tristan and Goul{\~a}o, Manuel and Kallinteris, Andreas and KG, Arjun and Krimmel, Markus and others},
  year   = {2023},
  howpublished = {\url{https://gymnasium.farama.org/}}
}

% =====================================================
% Additional entries for Entregable 4 (Spanish version)
% =====================================================

@article{Brockman2016,
  title   = {{OpenAI Gym}},
  author  = {Brockman, Greg and Cheung, Vicki and Pettersson, Ludwig and Schneider, Jonas and Schulman, John and Tang, Jie and Zaremba, Wojciech},
  journal = {arXiv:1606.01540},
  year    = {2016}
}

@misc{Towers2024,
  title  = {{Gymnasium} v1.0: A standard interface for reinforcement learning environments},
  author = {Towers, Mark and others},
  year   = {2024},
  howpublished = {\url{https://gymnasium.farama.org/}}
}

@inproceedings{Snoek2012,
  title     = {Practical Bayesian optimization of machine learning algorithms},
  author    = {Snoek, Jasper and Larochelle, Hugo and Adams, Ryan P.},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2012}
}

@article{Lin1992,
  title   = {Self-improving reactive agents based on reinforcement learning, planning and teaching},
  author  = {Lin, Long-Ji},
  journal = {Machine Learning},
  volume  = {8},
  number  = {3--4},
  pages   = {293--321},
  year    = {1992}
}

@inproceedings{Loshchilov2017,
  title     = {{SGDR}: Stochastic gradient descent with warm restarts},
  author    = {Loshchilov, Ilya and Hutter, Frank},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2017}
}

@book{Kuhl2021,
  title     = {Computational Epidemiology: Data-Driven Modeling of {COVID-19}},
  author    = {Kuhl, Ellen},
  publisher = {Springer},
  year      = {2021}
}

@book{Csefalvay2023,
  title     = {Computational Modeling of Infectious Disease},
  author    = {{von Cs}{\'e}falvay, Chris},
  publisher = {Elsevier},
  year      = {2023}
}

@article{Boldt2015,
  title   = {Shared neural markers of decision confidence and error detection},
  author  = {Boldt, Annika and Yeung, Nick},
  journal = {Journal of Neuroscience},
  volume  = {35},
  number  = {8},
  pages   = {3478--3484},
  year    = {2015}
}

@article{Fleming2012,
  title   = {The neural basis of metacognitive ability},
  author  = {Fleming, Stephen M. and Dolan, Raymond J.},
  journal = {Philosophical Transactions of the Royal Society B},
  volume  = {367},
  number  = {1594},
  pages   = {1338--1349},
  year    = {2012}
}

@article{Fleming2024,
  title   = {Metacognition and confidence: A review and synthesis},
  author  = {Fleming, Stephen M.},
  journal = {Annual Review of Psychology},
  volume  = {75},
  pages   = {241--268},
  year    = {2024}
}

@article{Griffith2021,
  title   = {A formal model of capacity limits in metacognitive judgements},
  author  = {Griffith, Tom and Baker, Sara-Jane and Lepora, Nathan F.},
  journal = {Journal of Mathematical Psychology},
  volume  = {103},
  pages   = {102544},
  year    = {2021}
}

@article{Grimaldi2015,
  title   = {There are things that we know that we know, and there are things that we do not know we do not know: Confidence in decision-making},
  author  = {Grimaldi, Piercesare and Lau, Hakwan and Basso, Michele A.},
  journal = {Neuroscience and Biobehavioral Reviews},
  volume  = {55},
  pages   = {88--97},
  year    = {2015}
}

@article{Pei2021,
  title   = {Transformer uncertainty estimation with hierarchical stochastic attention},
  author  = {Pei, Jiahuan and Wang, Cheng and Szarvas, Gy{\"o}rgy},
  journal = {arXiv:2112.13776},
  year    = {2021}
}

@misc{Turner2024,
  title  = {An introduction to {T}ransformers},
  author = {Turner, Richard E.},
  howpublished = {arXiv:2304.10557},
  year   = {2024}
}

@misc{Ghasemi2024,
  title  = {A comprehensive survey of reinforcement learning: From algorithms to practical challenges},
  author = {Ghasemi, Majid and Ebrahimi, Dariush},
  howpublished = {arXiv:2411.18892},
  year   = {2024}
}

@article{Kuc2021,
  title   = {Pre-test cognitive load and reactive cognitive control are related to performance in a conflict task},
  author  = {Kuc, Joanna and others},
  journal = {Brain Sciences},
  year    = {2021}
}

@article{Nawaz2020,
  title   = {Single-trial {EEG} classification of motor imagery using deep convolutional neural networks},
  author  = {Nawaz, Rab and others},
  journal = {Sensors},
  year    = {2020}
}

@article{Giuste2021,
  title   = {Explainable artificial intelligence methods in combating pandemics: A systematic review},
  author  = {Giuste, Felipe and others},
  journal = {IEEE Reviews in Biomedical Engineering},
  year    = {2021}
}

@article{Hamida2024,
  title   = {Explainable {AI} for healthcare: A scoping review of methods and applications},
  author  = {Hamida, Sonia Ben and others},
  journal = {Information Fusion},
  year    = {2024}
}

@article{Chen2025,
  title   = {Taming uncertainty in modern machine learning},
  author  = {Chen, Yi and Wiggins, Christopher H.},
  journal = {Notices of the American Mathematical Society},
  volume  = {72},
  number  = {3},
  pages   = {1},
  year    = {2025}
}

@inproceedings{Peng1994,
  title     = {Incremental multi-step {Q}-learning},
  author    = {Peng, Jing and Williams, Ronald J.},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {1994}
}

@article{Maroti2019,
  title   = {{RBED}: Reward Based Epsilon Decay},
  author  = {Maroti, Aleksandar},
  journal = {arXiv:1910.13701},
  year    = {2019}
}

```
