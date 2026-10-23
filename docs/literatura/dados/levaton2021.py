# SPDX-License-Identifier: GPL-3.0-or-later
"""Extraction of figures 2 and 3 of Levaton, Klein and Amorim 2021, Braz. J. Phys.
51, 75, doi 10.1007/s13538-020-00809-z: positive column of a flowing N2 DC
discharge at 60 mA (tube radius 0.3 cm), at 230 Pa (fig. 2, E/N = 94 Td) and
530 Pa (fig. 3, E/N = 48 Td). Each figure has the N2(B) (solid black), N2(C)
(dashed red) and N2+(B) (dotted green) densities calculated by the authors
against the residence time, and the three relative densities measured at the end
of the column, normalized by the authors so that N2(C, v = 0 to 4) matches the
calculated N2(C).

Vector figures (strokes), pages 3 and 4 of the PDF. Calibration (pt, y downwards)
from the major tick marks, log-log; the x labels are text and the y labels were
read on the rendered page:
  fig. 2: x 1e-9 and 1e-1 s at 340.54 and 536.99; y 1e5 and 1e12 cm-3 at 632.47 and 506.59;
  fig. 3: x 1e-9 and 1e-1 s at 85.73 and 282.18; y 1e5 and 1e12 cm-3 at 197.88 and 72.00.
The markers of fig. 3 are drawn at 0.99 ms, inside the 0.76 +- 0.27 ms of
table 1. Around each measured marker (1.6 to 2.6 pt, the size of its outline) the
curve points are left out, since the outline has the colour of the curve.

Usage: python3 docs/literatura/dados/levaton2021.py  (writes to levaton2021/)
"""
import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(RAIZ / "ferramentas"))
import digitaliza as dg  # noqa: E402

PDF = RAIZ / "docs" / "artigos" / "s13538-020-00809-z.pdf"
SAIDA = Path(__file__).with_name("levaton2021")
PRETO, VERMELHO, VERDE = (0, 0, 0), (1, 0, 0), (0, 1, 0)

# figure: page, frame, x axis, y axis, pressure (Pa), residence time at the end of
# the column and its uncertainty (s, table 1)
FIGURAS = {
    "fig2": (3, (340.54, 494.07, 536.99, 632.47), (340.54, 536.99), (632.47, 506.59), 230, (0.36e-3, 0.13e-3)),
    "fig3": (4, (85.73, 59.48, 282.18, 197.88), (85.73, 282.18), (197.88, 72.00), 530, (0.76e-3, 0.27e-3)),
}
ESTADOS = (("N2_B", PRETO), ("N2_C", VERMELHO), ("N2p_B", VERDE))
# radius (pt) of the outline of each marker: square, circle and triangle
RAIO = {"N2_B": 1.8, "N2_C": 1.6, "N2p_B": 2.6}


def grava(nome, cabecalho, colunas, nota):
    SAIDA.mkdir(exist_ok=True)
    np.savetxt(SAIDA / nome, np.column_stack(colunas), fmt="%.6g", delimiter=",",
               header=f"{nota}\n{cabecalho}", comments="# ")
    print(f"{nome}: {len(colunas[0])} points")


def marcador(tr, cor, moldura):
    """Centre (pt) of the single filled marker of the colour inside the frame."""
    centros = {(round(c[0], 1), round(c[1], 1)) for t in tr if t["preenchimento"]
               and dg._mesma_cor(t["preenchimento"], cor, 0.12)
               for c in [0.5 * (t["p"].min(0) + t["p"].max(0))]
               if 2.0 < (t["p"].max(0) - t["p"].min(0)).max() < 7.0 and dg._dentro(c[None, :], moldura)[0]}
    assert len(centros) == 1, (cor, centros)
    return np.array(centros.pop())


def barra(tr, c, alcance=15.0):
    """Top and bottom (pt) of the error bar: black strokes crossing the column of
    the marker."""
    ys = [c[1]]
    for t in tr:
        p = t["p"]
        if (t["traco"] and dg._mesma_cor(t["traco"], PRETO) and len(p) <= 3
                and p[:, 0].min() - 0.3 <= c[0] <= p[:, 0].max() + 0.3 and np.ptp(p[:, 0]) < 5.0
                and np.abs(p[:, 1] - c[1]).max() < alcance):
            ys.extend(p[:, 1])
    return min(ys), max(ys)


def figura(nome, tr):
    _, moldura, (x0, x1), (y0, y1), pressao, t_fim = FIGURAS[nome]
    ex, ey = dg.Eixo(x0, 1e-9, x1, 1e-1, log=True), dg.Eixo(y0, 1e5, y1, 1e12, log=True)
    interior = (moldura[0] + 0.8, moldura[1] + 0.8, moldura[2] - 0.8, moldura[3] - 0.8)
    medidos = []
    marcas = {}
    for estado, cor in ESTADOS:
        cor_marca = (0.07, 0.08, 0.07) if cor == PRETO else cor
        marcas[estado] = marcador(tr, cor_marca, moldura)
    for estado, cor in ESTADOS:
        # black: 4 pt from the frame (tick marks) and out of the column of the error bars
        janela = interior if cor != PRETO else (moldura[0] + 4, moldura[1] + 4, moldura[2] - 4, moldura[3] - 4)
        c = dg.curva_vetorial(tr, cor, janela, min_pontos=2)
        longe = np.min([np.hypot(*(c - m).T) / RAIO[e] for e, m in marcas.items()], axis=0) > 1.0
        if cor == PRETO:
            longe &= np.abs(c[:, 0] - marcas[estado][0]) > 1.7
        c = c[longe]
        grava(f"{nome}_{estado}_calculado.csv", "t_s,n_cm3", (ex.valor(c[:, 0]), ey.valor(c[:, 1])),
              f"Levaton 2021 {nome}, {estado} calculated by the authors, 60 mA, {pressao} Pa")
        m = marcas[estado]
        topo, base = barra(tr, m)
        medidos.append((estado, ex.valor(m[0]), ey.valor(m[1]), ey.valor(base), ey.valor(topo)))
    t_med = medidos[0][1]
    # the markers are drawn within the uncertainty of table 1 (fig. 3: at 0.99 ms,
    # for 0.76 +- 0.27 ms)
    assert abs(t_med - t_fim[0]) <= t_fim[1], (nome, t_med, t_fim)
    with open(SAIDA / f"{nome}_medidos.csv", "w") as f:
        f.write(f"# Levaton 2021 {nome}: relative densities measured at the end of the column (60 mA, {pressao} Pa),\n"
                "# normalized by the authors so that the sum of N2(C, v = 0 to 4) equals the calculated N2(C)\n"
                "# estado,t_s,n_cm3,n_inf,n_sup\n")
        for linha in medidos:
            f.write(",".join([linha[0]] + [f"{v:.6g}" for v in linha[1:]]) + "\n")
    calc = np.loadtxt(SAIDA / f"{nome}_N2_C_calculado.csv", delimiter=",")
    fim = np.interp(np.log(medidos[1][1]), np.log(calc[:, 0]), calc[:, 1])
    assert abs(medidos[1][2] / fim - 1.0) < 0.15, (nome, medidos[1], fim)
    print(f"{nome}_medidos.csv:", [(m[0], f"{m[2]:.3g}") for m in medidos], f"(N2(C) calculated there: {fim:.3g})")


def main():
    for nome, (pagina, *_) in FIGURAS.items():
        figura(nome, dg.tracos_pdf(PDF, pagina))


if __name__ == "__main__":
    main()
