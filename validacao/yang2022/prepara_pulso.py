# SPDX-License-Identifier: GPL-3.0-or-later
"""Tensão de entrada do caso de Yang et al. 2022 a partir do pulso 50 medido
(docs/literatura/dados/yang2022/fig2_tensao_pulso50.csv), para
`simula.py --forma tabela --tensao-tabela pulso50.txt --periodo 1e-5`.

Tratamento, todo registrado aqui:
- antes de -1 ns a tensão é zero (o patamar de 0,2 a 1,1 kV da digitalização
  antes da subida é a linha do zero cruzando a curva, não sinal);
- a curva vai até 198,4 ns, onde a figura termina com -5,8 kV; dali a tensão
  volta a zero em rampa linear de 30 ns (o que vem depois não foi medido);
- reamostragem uniforme de 0,5 ns (a subida leva ~3 ns), com t deslocado para
  começar em zero;
- a mediana por coluna corta o pico estreito: 25,2 kV aqui contra 26,2 kV pela
  borda da linha (4 %).

Uso: python3 validacao/yang2022/prepara_pulso.py  (grava pulso50.txt ao lado)
"""
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parents[1]


def main():
    d = np.loadtxt(RAIZ / "docs" / "literatura" / "dados" / "yang2022" / "fig2_tensao_pulso50.csv", delimiter=",")
    t_ns, v_kv = d[:, 0], d[:, 1]
    ok = t_ns >= -1.0
    t_ns, v_kv = np.concatenate([[-1.0], t_ns[ok]]), np.concatenate([[0.0], v_kv[ok]])
    fim = t_ns[-1]
    t_ns, v_kv = np.concatenate([t_ns, [fim + 30.0]]), np.concatenate([v_kv, [0.0]])
    grade = np.arange(0.0, fim + 1.0 + 30.0 + 0.25, 0.5)
    v = np.interp(grade - 1.0, t_ns, v_kv, left=0.0, right=0.0) * 1e3
    v[0] = v[-1] = 0.0
    np.savetxt(AQUI / "pulso50.txt", np.column_stack([grade * 1e-9, v]), fmt="%.6e",
               header="Yang 2022, pulso 50 em N2 a 150 Torr (fig. 2(a)); t (s), V (V); ver prepara_pulso.py")
    print(f"pulso50.txt: {len(grade)} pontos, de 0 a {grade[-1]:.1f} ns, pico {v.max() / 1e3:.1f} kV, "
          f"mínimo {v.min() / 1e3:.1f} kV")


if __name__ == "__main__":
    main()
