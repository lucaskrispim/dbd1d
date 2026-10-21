# SPDX-License-Identifier: GPL-3.0-or-later
"""Extração da figura 4 de Loffhagen et al. 2020, PCPP, doi
10.1007/s11090-020-10121-y: tensão de ignição medida e calculada de uma DBD de
argônio com TMS (gap de 1 mm, vidro borofloat de 2 mm com ε_r = 4,6, 86,2 kHz,
1 atm, 300 K). Figura vetorial, página 30 do PDF.

Calibração (pt, y para baixo): eixo x com traços principais em 261,00 (0 ppm)
e 385,45 (200 ppm); eixo y em 548,11 (1,0 kV) e 456,42 (3,5 kV).

Uso: python3 docs/literatura/dados/loffhagen2020.py  (grava em loffhagen2020/)
"""
import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(RAIZ / "ferramentas"))
import digitaliza as dg  # noqa: E402

PDF = RAIZ / "docs" / "artigos" / "loffhagen2020.pdf"
SAIDA = Path(__file__).with_name("loffhagen2020")
MOLDURA = (254.8, 445.4, 391.7, 548.1)
LEGENDA = (300.0, 445.4, 391.7, 510.0)


def main():
    tr = dg.tracos_pdf(PDF, 30)
    tx = dg.tracos_de_escala(tr, MOLDURA, "x", min_comp=2.5)
    ty = dg.tracos_de_escala(tr, MOLDURA, "y", min_comp=2.5)
    assert len(tx) == 5 and len(ty) == 6, (tx, ty)
    ex, ey = dg.Eixo(tx[0], 0.0, tx[-1], 200.0), dg.Eixo(ty[0], 3.5, ty[-1], 1.0)
    # pontos medidos: círculos cheios; o centro vertical e a barra de erro vêm
    # dos segmentos verticais da barra (dois por ponto, encostados no centro)
    pontos = []
    for t in tr:
        p = t["p"]
        if dg._mesma_cor(t["preenchimento"], (0, 0, 0)) and (p.max(0) - p.min(0)).max() < 8 \
                and dg._dentro(p.mean(0)[None, :], MOLDURA)[0] and not dg._dentro(p.mean(0)[None, :], LEGENDA)[0]:
            pontos.append(0.5 * (p.max(0) + p.min(0)))
    pontos = np.array(sorted(pontos, key=lambda c: c[0]))
    barras = []
    for c in pontos:
        segs = [(min(a[1], b[1]), max(a[1], b[1])) for t in tr if dg._mesma_cor(t["traco"], (0, 0, 0))
                for a, b in zip(t["p"][:-1], t["p"][1:])
                if abs(a[0] - b[0]) < 0.05 and abs(a[0] - c[0]) < 0.6 and abs(a[1] - b[1]) < 12
                and min(a[1], b[1]) < MOLDURA[3] - 0.5]
        topo, base = min(s[0] for s in segs), max(s[1] for s in segs)
        barras.append((topo, base))
    barras = np.array(barras)
    x = np.round(ex.valor(pontos[:, 0])) + 0.0
    u = ey.valor(pontos[:, 1])
    u_max, u_min = ey.valor(barras[:, 0]), ey.valor(barras[:, 1])
    SAIDA.mkdir(exist_ok=True)
    np.savetxt(SAIDA / "fig4_ignicao_medida.csv", np.column_stack([x, u, u_min, u_max]), fmt="%.4g", delimiter=",",
               header="Loffhagen 2020 fig. 4: tensão de ignição medida (amplitude da senoide), Ar com x ppm de TMS\n"
                      "x_ppm,Ui_kV,Ui_min_kV,Ui_max_kV", comments="# ")
    for nome, cor in (("k3.0e-16_a0.25", (0, 0, 0)), ("k2.4e-16_a0.30", (1, 0, 0)), ("k2.2e-17_a0.55", (0, 0, 1))):
        c = [t["p"] for t in tr if dg._mesma_cor(t["traco"], cor) and len(t["p"]) == 12
             and dg._dentro(t["p"], MOLDURA).all()]
        assert len(c) == 1, nome
        np.savetxt(SAIDA / f"fig4_ignicao_modelo_{nome}.csv", np.column_stack([ex.valor(c[0][:, 0]), ey.valor(c[0][:, 1])]),
                   fmt="%.4g", delimiter=",", comments="# ",
                   header=f"Loffhagen 2020 fig. 4: tensão de ignição calculada, k_M,Ar* e α_Pl = {nome}\nx_ppm,Ui_kV")
    for linha in np.column_stack([x, u, u_min, u_max]):
        print("x = %3.0f ppm: U_i = %.3f kV (%.3f a %.3f)" % tuple(linha))


if __name__ == "__main__":
    main()
