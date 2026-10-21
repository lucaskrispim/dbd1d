# SPDX-License-Identifier: GPL-3.0-or-later
"""Tensão de entrada do caso de Takashima et al. 2012 (gerador CPT) a partir do
campo aplicado medido a 60 Torr (docs/literatura/dados/takashima2012/
fig7_campo_aplicado_CPT.csv), para `simula.py --forma tabela`.

O campo aplicado do modelo deles é a tensão dividida por d + 2 l/ε (gap de
10 mm, paredes de quartzo de 1,75 mm, ε = 3,8 aqui): V = E · 1,092 cm. A curva
começa em 67 ns (zero antes); termina em 180 ns com +3,4 kV/cm e volta a zero em
rampa linear de 30 ns. Reamostragem de 0,5 ns, t deslocado para começar em zero.
A mesma forma serve para todas as pressões (o artigo só mostra a de 60 Torr).

Uso: python3 validacao/takashima2012/prepara_pulso.py [EPS_R]
  (grava pulso_cpt.txt com ε = 3,8, ou pulso_cpt_epsEPS_R.txt)
"""
import sys
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parents[1]

def main():
    eps = float(sys.argv[1]) if len(sys.argv) > 1 else 3.8
    DISTANCIA_CM = 1.0 + 2 * 0.175 / eps
    nome = "pulso_cpt.txt" if eps == 3.8 else f"pulso_cpt_eps{sys.argv[1]}.txt"
    d = np.loadtxt(RAIZ / "docs" / "literatura" / "dados" / "takashima2012" / "fig7_campo_aplicado_CPT.csv",
                   delimiter=",")
    t, e = d[:, 0], d[:, 1]
    t0 = t[0] - 1.0
    t, e = np.concatenate([[t0], t, [t[-1] + 30.0]]), np.concatenate([[0.0], e, [0.0]])
    grade = np.arange(0.0, t[-1] - t0 + 0.25, 0.5)
    v = np.interp(grade + t0, t, e) * DISTANCIA_CM
    v[0] = v[-1] = 0.0
    np.savetxt(AQUI / nome, np.column_stack([grade * 1e-9, v]), fmt="%.6e",
               header=f"Takashima 2012, gerador CPT, a partir do campo aplicado a 60 Torr (fig. 7), ε = {eps}; "
                      "t (s), V (V)")
    print(f"{nome}: {len(grade)} pontos até {grade[-1]:.1f} ns, de {v.min() / 1e3:.1f} a {v.max() / 1e3:.1f} kV")


if __name__ == "__main__":
    main()
