# SPDX-License-Identifier: GPL-3.0-or-later
"""Extraction of the comparison data of figures 7 to 10 of Crispim et al. 2024,
Phys. Scr. 99 065521: electron swarm parameters in argon versus E/N, measured or
computed by other authors (each series with the original source cited in the
legend). Vector gnuplot figures, pages 12 and 13 of the PDF.

The axes are logarithmic. The calibration uses two decade ticks of each axis
(pt), found by the pattern of the minor ticks (the one of 2 lies log10(2) of a
decade after the tick of the previous decade). The frame does not always fall on
a decade: in fig. 8 the 1e1 is 5.4 pt below the top, in fig. 9 the 1e3 is 4.9 pt
before the right edge.

Usage: python3 docs/literatura/dados/crispim2024.py  (writes to crispim2024/)
"""
import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(RAIZ / "ferramentas"))
import digitaliza as dg  # noqa: E402

PDF = RAIZ / "docs" / "artigos" / "Crispim_2024_Phys._Scr._99_065521.pdf"
SAIDA = Path(__file__).with_name("crispim2024")

# figure: page, frame (pt), (c, value) of two decade ticks in x and in y, legend box, quantity
FIGURAS = {
    "fig7": (12, (221.6, 77.6, 481.7, 266.3), ((221.6, 1e-4), (481.7, 1e3)), ((266.3, 1e-4), (77.6, 1e1)),
             (440.0, 195.0, 482.0, 262.0), "electron drift velocity, 1e5 m/s"),
    "fig8": (12, (219.5, 377.4, 481.6, 567.5), ((219.5, 1e-5), (481.6, 1e3)), ((567.5, 1e-4), (382.8, 1e1)),
             (440.0, 500.0, 482.0, 567.0), "D_e/mu_e, eV"),
    "fig9": (13, (222.2, 72.8, 486.5, 264.5), ((222.2, 1e-3), (481.6, 1e3)), ((264.5, 1e23), (99.1, 1e26)),
             (330.0, 75.0, 487.0, 135.0), "mu_e N, (m V s)^-1"),
    "fig10b": (13, (367.1, 365.5, 537.6, 489.2), ((367.1, 1e-1), (537.6, 1e3)), ((489.2, 1e-1), (370.2, 1e1)),
               (440.0, 455.0, 538.0, 490.0), "electron mean energy, eV"),
}
# series: figure, colour, 'traco' (stroke) or 'preenchimento' (fill), number of points of the subpath, source
SERIES = [
    ("fig7_pack_phelps_1961_medida", "fig7", (1.0, 0.08, 0.58), "traco", 2, "J L Pack and A V Phelps 1961, measured"),
    ("fig7_ellis_1976_medida", "fig7", (0.0, 0.0, 0.0), "traco_x", 2, "H W Ellis 1976, measured"),
    ("fig7_kucukarpaci_lucas_1981_medida", "fig7", (1.0, 0.55, 0.0), "preenchimento", 5,
     "H N Kucukarpaci and J Lucas 1981, measured"),
    ("fig7_nielsen_1936_medida", "fig7", (0.42, 0.35, 0.8), "preenchimento", 4, "R A Nielsen 1936, measured"),
    ("fig8_warren_parker_1962_77K_medida", "fig8", (0.0, 0.29, 0.02), "preenchimento", 5,
     "Warren and Parker 1962, 77 K, measured"),
    ("fig8_townsend_bailey_1922_288K_medida", "fig8", (0.0, 1.0, 0.0), "preenchimento", 5,
     "Townsend and Bailey 1922, 288 K, measured"),
    ("fig8_townsend_bailey_1922_77K_medida", "fig8", (0.29, 0.0, 0.51), "preenchimento", 5,
     "Townsend and Bailey 1922, 77 K, measured"),
    ("fig9_kucukarpaci_lucas_1981_medida", "fig9", (0.0, 1.0, 0.87), "preenchimento", 5,
     "H N Kucukarpaci and J Lucas 1981, measured"),
    ("fig9_robertson_1977_medida", "fig9", (0.0, 0.0, 0.0), "preenchimento_losango", 5, "A G Robertson et al. 1977, measured"),
    ("fig9_pack_1992_medida", "fig9", (0.0, 0.04, 1.0), "traco", 2, "Pack et al. 1992, measured"),
    ("fig9_errett_1951_medida", "fig9", (0.65, 0.04, 0.61), "traco", 2, "Errett 1951, measured"),
    ("fig10b_escada_2007_calculada", "fig10b", (0.0, 0.75, 1.0), "preenchimento", 5, "J Escada et al. 2007, computed"),
    ("fig10b_benhenni_2005_calculada", "fig10b", (0.0, 1.0, 0.0), "preenchimento", 5, "M Benhenni et al. 2005, computed"),
]


def centros(tr, moldura, legenda, cor, tipo, n):
    """Centres of the markers of a series: small subpaths of the given colour,
    type and number of points; markers made of several strokes (x and *) count
    once; those of the legend are left out."""
    out = []
    for t in tr:
        p = t["p"]
        tam = p.max(0) - p.min(0)
        if len(p) != n or tam.max() > 6.0 or tam.max() < 1.0:
            continue
        if tipo.startswith("traco"):
            if not dg._mesma_cor(t["traco"], cor, 0.02):
                continue
            # the black x of Ellis: only the diagonals (the ticks are horizontal or vertical)
            if tipo == "traco_x" and (tam.min() < 1.0):
                continue
        else:
            if not dg._mesma_cor(t["preenchimento"], cor, 0.02) or t["traco"] is not None:
                continue
            if tipo == "preenchimento_losango" and not (abs(tam[0] - 2.8) < 0.2 and abs(tam[1] - 2.8) < 0.2):
                continue
        c = 0.5 * (p.max(0) + p.min(0))
        if not dg._dentro(c[None, :], moldura, 0.5)[0] or dg._dentro(c[None, :], legenda)[0]:
            continue
        if not any(np.hypot(*(c - q)) < 0.8 for q in out):
            out.append(c)
    return np.array(out).reshape(-1, 2)


def main():
    SAIDA.mkdir(exist_ok=True)
    tracos = {pg: dg.tracos_pdf(PDF, pg) for pg in (12, 13)}
    for nome, fig, cor, tipo, n, fonte in SERIES:
        pg, m, (x1, x2), (y1, y2), leg, grandeza = FIGURAS[fig]
        ex = dg.Eixo(*x1, *x2, log=True)
        ey = dg.Eixo(*y1, *y2, log=True)
        c = centros(tracos[pg], m, leg, cor, tipo, n)
        c = c[np.argsort(c[:, 0])]
        en, v = ex.valor(c[:, 0]), ey.valor(c[:, 1])
        np.savetxt(SAIDA / f"{nome}.csv", np.column_stack([en, v]), fmt="%.5g", delimiter=",", comments="# ",
                   header=f"Crispim 2024 {fig}: {grandeza} in argon; {fonte} (as reproduced in the figure)\n"
                          "EN_Td,valor")
        print(f"{nome}: {len(c)} points, E/N from {en.min():.3g} to {en.max():.3g} Td")


if __name__ == "__main__":
    main()
