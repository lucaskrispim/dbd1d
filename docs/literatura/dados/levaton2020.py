# SPDX-License-Identifier: GPL-3.0-or-later
"""Extraction of figures 3, 4 and 10 of Levaton, Klein and Amorim 2020, Phys.
Plasmas 27, 063503, doi 10.1063/1.5144467: positive column of a flowing N2 DC
discharge (tube radius 0.3 cm, 12 cm between the electrodes).

Figures 3 and 4: E/N and n_e measured between the probes at 60 mA, against the gas
flow rate, at 230 Pa (red) and 1070 Pa (blue in fig. 3, green in fig. 4). Figure
10: N2(B), N2(C) and N2+(B) calculated by the authors at 10 mA against the
residence time, and the three measured relative densities at the end of the
column (t = 4.3e-4 s), normalized by the authors so that N2(C) matches the model.

Vector figures, page 12 and 14 of the PDF, drawn only with filled paths (lines,
markers and tick marks are fills). Calibration (pt, y downwards) from the major
tick marks:
  fig. 3: x 0.0 and 1.0 Slm at 94.86 and 278.11; y 40 and 100 Td at 243.07 and 111.75;
  fig. 4: x 0.2 and 1.0 Slm at 123.18 and 286.10; y 2e11 and 6e11 cm-3 at 615.73 and 502.96;
  fig. 10 (log-log): x 1e-9 and 1e-4 s at 88.57 and 268.29 (the measured points at
  4.3e-4 s fall at 291.1, as they should); y 1e6 and 1e13 cm-3 at 252.20 and 119.01.
The curves of fig. 10 are thin filled outlines of the lines: the curve is the mean
y of the outline in bins of 0.5 pt in x.
The error bars are the black fills crossing the column of each marker.

The text gives the means over the flow rates (section III): 90 +- 9 Td and
(1.54 +- 0.10)e11 cm-3 at 230 Pa, 38 +- 2 Td and (5.16 +- 0.17)e11 cm-3 at 1070 Pa;
the script checks the extracted points against them.

Usage: python3 docs/literatura/dados/levaton2020.py  (writes to levaton2020/)
"""
import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(RAIZ / "ferramentas"))
import digitaliza as dg  # noqa: E402

PDF = RAIZ / "docs" / "artigos" / "levaton2020.pdf"
SAIDA = Path(__file__).with_name("levaton2020")
PRETO, VERMELHO, AZUL, VERDE = (0, 0, 0), (1, 0, 0), (0, 0, 1), (0, 1, 0)


def grava(nome, cabecalho, colunas, nota):
    SAIDA.mkdir(exist_ok=True)
    np.savetxt(SAIDA / nome, np.column_stack(colunas), fmt="%.6g", delimiter=",",
               header=f"{nota}\n{cabecalho}", comments="# ")
    print(f"{nome}: {len(colunas[0])} points")


def marcadores(tr, cor, janela):
    """Centres (pt) of the filled markers of the colour, sorted in x: each marker is
    drawn twice (fill and outline), so the centres are rounded and merged."""
    centros = set()
    for t in tr:
        p = t["p"]
        w = p.max(0) - p.min(0)
        if t["preenchimento"] and dg._mesma_cor(t["preenchimento"], cor) and 2.0 < w.max() < 6.0:
            c = 0.5 * (p.min(0) + p.max(0))
            if dg._dentro(c[None, :], janela)[0]:
                centros.add((round(c[0], 1), round(c[1], 1)))
    return np.array(sorted(centros))


def barras(tr, centros, alcance=40.0, fino=0.4):
    """Top and bottom (pt) of the error bar of each marker: the extent in y of the
    thin black fills (bar and caps, 0.2 pt thick, 0.5 pt in fig. 10) that cross
    the column of the marker; the tick marks (0.7 pt) are left out."""
    topo, base = [], []
    for cx, cy in centros:
        ys = [cy]
        for t in tr:
            p = t["p"]
            w = p.max(0) - p.min(0)
            if (t["preenchimento"] and dg._mesma_cor(t["preenchimento"], PRETO) and len(p) > 1 and w.max() < 2 * alcance
                    and p[:, 0].min() - 0.3 <= cx <= p[:, 0].max() + 0.3 and abs(p[:, 1].mean() - cy) < alcance
                    and w[0] < 4.0 and w.min() < fino):
                ys.extend(p[:, 1])
        topo.append(min(ys))
        base.append(max(ys))
    return np.array(topo), np.array(base)


def figura(tr, janela, ex, ey, series, grandeza, unidade, medias):
    for pressao, cor in series:
        c = marcadores(tr, cor, janela)
        assert len(c) == 10, (pressao, len(c))
        t, b = barras(tr, c)
        x, y = ex.valor(c[:, 0]), ey.valor(c[:, 1])
        sup, inf = ey.valor(t), ey.valor(b)
        media, ref = y.mean(), medias[pressao]
        assert abs(media / ref[0] - 1.0) < 0.03, (pressao, media, ref)
        grava(f"fig{grandeza}_{pressao}Pa.csv", f"Q_slm,{unidade},err_inf,err_sup", (x, y, y - inf, sup - y),
              f"Levaton 2020 fig. {'3' if grandeza == '3_EN' else '4'}, 60 mA, {pressao} Pa; mean {media:.4g} "
              f"(text: {ref[0]:g} +- {ref[1]:g})")


def curva_contorno(tr, cor, janela, minimo=50):
    """Line drawn as a filled outline: mean y of the outline points in bins of
    0.5 pt in x."""
    partes = [t["p"] for t in tr if t["preenchimento"] and dg._mesma_cor(t["preenchimento"], cor, 0.05)
              and len(t["p"]) >= minimo and dg._dentro(t["p"].mean(0)[None, :], janela)[0]]
    assert partes, cor
    p = np.concatenate(partes)
    p = p[dg._dentro(p, janela)]
    caixa = np.round(p[:, 0] / 0.5).astype(int)
    xs = np.unique(caixa)
    return np.array([[p[caixa == k, 0].mean(), p[caixa == k, 1].mean()] for k in xs])


def figura10(tr):
    janela = (89.5, 106.5, 295.5, 251.5)
    ex, ey = dg.Eixo(88.57, 1e-9, 268.29, 1e-4, log=True), dg.Eixo(252.20, 1e6, 119.01, 1e13, log=True)
    estados = (("N2_B", PRETO, PRETO), ("N2_C", VERMELHO, VERMELHO), ("N2p_B", VERDE, VERDE))
    medidos = []
    for nome, cor_linha, cor_ponto in estados:
        c = curva_contorno(tr, cor_linha if nome != "N2_B" else PRETO, janela)
        grava(f"fig10_{nome}_calculado.csv", "t_s,n_cm3", (ex.valor(c[:, 0]), ey.valor(c[:, 1])),
              f"Levaton 2020 fig. 10, {nome} calculated by the authors, 10 mA, 230 Pa")
    # measured points at the end of the column: square (black), circle (red), triangle (green)
    for nome, cor in (("N2_B", PRETO), ("N2_C", VERMELHO), ("N2p_B", VERDE)):
        if cor == PRETO:
            c = np.array(sorted({(round(0.5 * (t["p"].min(0) + t["p"].max(0))[0], 1),
                                  round(0.5 * (t["p"].min(0) + t["p"].max(0))[1], 1)) for t in tr
                                 if t["preenchimento"] and dg._mesma_cor(t["preenchimento"], PRETO)
                                 and 2.0 < (t["p"].max(0) - t["p"].min(0)).min() and (t["p"].max(0) - t["p"].min(0)).max() < 6.0
                                 and abs(0.5 * (t["p"].min(0) + t["p"].max(0))[0] - 291.1) < 1.0}))
        else:
            c = marcadores(tr, cor, (285.0, 106.5, 296.0, 251.5))
        assert len(c) == 1, (nome, c)
        t, b = barras(tr, c, alcance=8.0, fino=0.6)
        medidos.append((nome, ex.valor(c[0, 0]), ey.valor(c[0, 1]), ey.valor(b[0]), ey.valor(t[0])))
    with open(SAIDA / "fig10_medidos.csv", "w") as f:
        f.write("# Levaton 2020 fig. 10: relative densities measured at the end of the column (10 mA, 230 Pa),\n"
                "# normalized by the authors so that the sum of N2(C, v = 0 to 4) equals the calculated N2(C)\n"
                "# estado,t_s,n_cm3,n_inf,n_sup\n")
        for linha in medidos:
            f.write(",".join([linha[0]] + [f"{v:.6g}" for v in linha[1:]]) + "\n")
    # the authors normalized the measurements to the calculated N2(C) at the end of the column
    calc_c = np.loadtxt(SAIDA / "fig10_N2_C_calculado.csv", delimiter=",")
    fim = np.interp(np.log(medidos[1][1]), np.log(calc_c[:, 0]), calc_c[:, 1])
    assert abs(medidos[1][2] / fim - 1.0) < 0.1, (medidos[1], fim)
    print("fig10_medidos.csv:", [(m[0], f"{m[2]:.3g}") for m in medidos], f"(N2(C) calculated at the end: {fim:.3g})")


def main():
    tr = dg.tracos_pdf(PDF, 12)
    figura(tr, (90, 95, 300, 256), dg.Eixo(94.86, 0.0, 278.11, 1.0), dg.Eixo(243.07, 40.0, 111.75, 100.0),
           ((230, VERMELHO), (1070, AZUL)), "3_EN", "EN_Td", {230: (90.0, 9.0), 1070: (38.0, 2.0)})
    figura(tr, (90, 490, 300, 645), dg.Eixo(123.18, 0.2, 286.10, 1.0), dg.Eixo(615.73, 2.0e11, 502.96, 6.0e11),
           ((230, VERMELHO), (1070, VERDE)), "4_ne", "ne_cm3", {230: (1.54e11, 0.10e11), 1070: (5.16e11, 0.17e11)})
    figura10(dg.tracos_pdf(PDF, 14))


if __name__ == "__main__":
    main()
