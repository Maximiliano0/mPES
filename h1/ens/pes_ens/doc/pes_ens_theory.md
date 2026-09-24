# pes_ens — fundamentos del ensamble ponderado

## Voto suave ponderado por confianza

Sea $p_k(a \mid s)$ la distribución del miembro $k$ sobre las acciones
factibles, obtenida con una *softmax* de temperatura $\tau = 15$ sobre sus
valores $Q$. El ensamble combina los miembros con pesos que crecen con su
confianza, medida como uno menos la entropía de Shannon normalizada:

$$
c_k = 1 - \frac{H(p_k)}{\log_2 11}, \qquad
p_{\mathrm{ens}}(a \mid s) \propto \eta(a) \sum_k w_k\,(0{,}1 + c_k)\,p_k(a \mid s)
$$

donde $w_k$ son los pesos base normalizados y $\eta(0) = 0{,}3$ si quedan
recursos ($\eta(a) = 1$ en otro caso). El término $0{,}1 + c_k$ evita que un
miembro con distribución uniforme quede anulado.

Una temperatura alta aplana cada $p_k$. En las 64 secuencias de referencia,
con $\tau = 15$ la acción más probable de cada miembro tiene en promedio una
probabilidad de 0,19 y supera a la segunda por sólo 0,02–0,03, y la
confianza media $c_k$ es de 0,12–0,14. Con diferencias tan pequeñas entre
acciones, el *prior* y el factor $\eta$ pesan mucho en la decisión final: el
*prior* cambia la acción votada en el 41 % de las decisiones con recursos
disponibles, y la acción final coincide con la preferida por el Transformer
sólo en el 39 % de ellas.

## *Prior* de severidad y cota de seguridad

La distribución se mezcla con un *prior* gaussiano centrado en la severidad
actual $S$:

$$
\pi_{\mathrm{prior}}(a \mid s) \propto \exp\!\left[-\frac{(a - S)^2}{2\sigma_S^2}\right],
\qquad
p'(a \mid s) \propto (1 - w_\pi)\,p_{\mathrm{ens}} + w_\pi\,\pi_{\mathrm{prior}}
$$

con $w_\pi = 0{,}17$ y $\sigma_S = 3$. Se elige $\arg\max_a p'(a \mid s)$ y,
si $S \ge 6$, la acción se eleva a $\lfloor S/2 \rfloor$ cuando es menor
que esa cota y la cota es factible.

## Referencias

- Shannon, C. E. (1948). A mathematical theory of communication. *The Bell
  System Technical Journal*, 27(3), 379–423.
- Wiering, M. A., & van Hasselt, H. (2008). Ensemble algorithms in
  reinforcement learning. *IEEE Transactions on Systems, Man, and
  Cybernetics, Part B*, 38(4), 930–936.
- Dietterich, T. G. (2000). Ensemble methods in machine learning. In *Multiple Classifier Systems* (pp. 1–15). Springer.
- Kuncheva, L. I. (2004). *Combining Pattern Classifiers: Methods and Algorithms*. Wiley.
