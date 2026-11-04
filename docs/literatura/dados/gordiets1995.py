# SPDX-License-Identifier: GPL-3.0-or-later
"""Pure N2 points (0% O2) of figures 4, 5 and 8 of Gordiets, Ferreira, Guerra,
Loureiro, Nahorny, Pagnon, Touzeau and Vialle 1995, IEEE Trans. Plasma Sci. 23,
750, doi 10.1109/27.467998: positive column of a flowing DC glow discharge,
Pyrex tube of 16 mm inner diameter, 2 Torr, 100 sccm, I = 15, 30 and 80 mA.
Measured gas temperature (fig. 4, from the rotational distribution of the
N2(C-B) 337 nm band), electron density (fig. 5, microwave cavity) and reduced
field (fig. 8, two electrostatic probes). The vibrational temperature of the
lowest levels is given in the text (section III): 3000, 4000 and 5500 K (+- 500).

The PDF is a scan: pages 10, 11 and 13 are rendered at 300 dpi and read per
pixel. Calibration from the rows of the frame (top and bottom of each panel),
found as the darkest rows; the markers at 0% sit on the y axis and are located
as the centroid of the dark pixels in a window around each one, leaving out a
band of +-3 px around the axis column (the axis and the steep start of the model
curves). Windows (pixels at 300 dpi) chosen on the rendered page and recorded
below. Moving each window by 4 px in any direction changes the result by at
most 0.6% (E/N), 1% (Tg) and 4% (ne). Fig. 5 has no 0% point for 80 mA (the
first is at 3%); the text gives 5e9 to 2e10 cm-3 from 15 to 80 mA (Gordiets et
al. 1996, J. Phys. D 29, 1021).

Figure 13 (page 15): vibrational distribution N2(X, v)/N2 calculated by the
authors, v = 1 to 30, for 0.1, 0.5 and 2% O2 at 80 mA (solid) and 30 mA (dashed);
the axis label reads cm-3 but the values (1e-9 to 1) are fractions. Only the 0.1%
curves are extracted, the closest to pure N2; below v ~ 10 the three percentages
coincide. Calibration (pixels at 300 dpi) from the dotted grid lines: v = 5, 10,
15, 20, 25 at columns 590.0, 703.0, 816.0, 929.5, 1041.5 (22.67 px per level, v = 0
at 476.7); decades 1e-1 to 1e-8 at rows 320.0 to 844.5 (75.0 px per decade). The
solid curves form one connected component of dark pixels: the 0.1% curve is its
top run in each column. The dashed curves are dotted: the dots are the small
components (2 to 150 px) away from the horizontal grid lines (3.5 px), from the labels
(the large components, expanded by 6 px), 5 px from the vertical ones; the 0.1% curve is followed from v = 12
to the right, at each level the dot closest to the extrapolation of the two
previous levels within 8 px in x and 20 px in y; below v = 12, where the dashed
curves coincide, the top dot.

Usage: python3 docs/literatura/dados/gordiets1995.py  (writes to gordiets1995/)
"""
import sys
import tempfile
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(RAIZ / "ferramentas"))
import digitaliza as dg  # noqa: E402

PDF = RAIZ / "docs" / "artigos" / "gordiets1995.pdf"
SAIDA = Path(__file__).with_name("gordiets1995")

# quantity: page, axis column, (top row, value), (bottom row, value), {current mA: (row, col) of the window centre}
PAINEIS = {
    "Tg_K": (10, 1575, (233.7, 1100.0), (845.3, 300.0), {15: (688, 1576), 30: (627, 1575), 80: (470, 1573)}),
    "ne_cm3": (11, 473, (241.7, 3.5e10), (917.3, 0.0), {15: (838, 478), 30: (753, 477)}),
    "EN_Td_15mA": (13, 508, (233.5, 85.0), (846.0, 50.0), {15: (551, 508)}),
    "EN_Td_30mA": (13, 508, (1053.5, 85.0), (1665.5, 50.0), {30: (1438, 508)}),
    "EN_Td_80mA": (13, 508, (1865.5, 100.0), (2477.5, 50.0), {80: (2235, 508)}),
}
TV = {15: 3000.0, 30: 4000.0, 80: 5500.0}


def centro(escuro, linha, coluna, eixo, meia=16, banda=3):
    """Centroid row of the dark pixels in the window, without the axis band."""
    y0, y1, x0, x1 = linha - meia, linha + meia + 1, coluna - meia, coluna + meia + 1
    w = escuro[y0:y1, x0:x1].copy()
    w[:, max(0, eixo - banda - x0):eixo + banda + 1 - x0] = False
    ys, _ = np.nonzero(w)
    assert len(ys) > 10, (linha, coluna, len(ys))
    return y0 + ys.mean()


def figura13(pagina):
    from scipy import ndimage
    escuro = dg.le_png(pagina).sum(2) / 3.0 < 0.55
    x0, x1, y0, y1 = 480, 1150, 248, 915
    m = escuro[y0:y1, x0:x1]
    rotulo, n = ndimage.label(m, structure=np.ones((3, 3)))
    tamanhos = ndimage.sum(m, rotulo, range(1, n + 1))
    caixas = ndimage.find_objects(rotulo)
    centros = ndimage.center_of_mass(m, rotulo, range(1, n + 1))
    coluna = lambda v: 476.7 + 22.67 * v  # noqa: E731
    fracao = lambda y: 10.0 ** (-1.0 - (y - 320.0) / 75.0)  # noqa: E731
    grade_y = [245.0, 320.0, 394.0, 468.5, 543.5, 618.8, 693.5, 768.8, 844.5, 919.5]
    grade_x = [coluna(v) for v in (5, 10, 15, 20, 25)]
    continua = int(np.argmax(tamanhos)) + 1
    rotulos = [(c[1].start + x0 - 6, c[1].stop + x0 + 6, c[0].start + y0 - 6, c[0].stop + y0 + 6)
               for k, c in enumerate(caixas) if tamanhos[k] > 150 and k + 1 != continua]
    pontos = []
    for s, (cy, cx) in zip(tamanhos, centros):
        x, y = cx + x0, cy + y0
        if not 2 <= s <= 150 or min(abs(y - g) for g in grade_y) < 3.5 or min(abs(x - g) for g in grade_x) < 5.0:
            continue
        if any(a <= x <= b and c <= y <= d for a, b, c, d in rotulos):
            continue
        pontos.append((x, y))
    pontos = np.array(pontos)
    solida = []
    for v in range(1, 30):
        linhas = np.flatnonzero(rotulo[:, int(round(coluna(v))) - x0] == continua) + y0
        topo = np.split(linhas, np.flatnonzero(np.diff(linhas) > 1) + 1)[0]      # the top run
        solida.append((v, fracao(topo.mean())))
    tracejada = []
    for v in range(1, 30):
        perto = pontos[np.abs(pontos[:, 0] - coluna(v)) < 8.0]
        if not len(perto):
            continue
        if v < 12 or len(tracejada) < 2:
            y = perto[:, 1].min()
        else:
            (va, ya), (vb, yb) = [(u, 320.0 - 75.0 * (np.log10(f) + 1.0)) for u, f in tracejada[-2:]]
            previsto = yb + (yb - ya) * (v - vb) / (vb - va)
            y = perto[np.argmin(np.abs(perto[:, 1] - previsto)), 1]
            if abs(y - previsto) > 20.0:
                continue
        tracejada.append((v, fracao(y)))
    SAIDA.mkdir(exist_ok=True)
    for nome, corrente, curva in (("solid", 80, solida), ("dashed", 30, tracejada)):
        v, f = np.array(curva).T
        np.savetxt(SAIDA / f"fig13_vdf_{corrente}mA_0.1pctO2.csv", np.column_stack((v, f)), fmt="%.6g", delimiter=",",
                   header=f"Gordiets 1995 fig. 13, N2(X, v)/N2 calculated by the authors, {corrente} mA, 0.1% O2 "
                          f"({nome} line)\nv,fracao", comments="# ")
        print(f"fig13_vdf_{corrente}mA_0.1pctO2.csv: {len(v)} points")


def main():
    valores = {}
    with tempfile.TemporaryDirectory() as d:
        paginas = {}
        for nome, (pagina, eixo, (r0, v0), (r1, v1), janelas) in PAINEIS.items():
            if pagina not in paginas:
                paginas[pagina] = dg.le_png(dg.pagina_png(PDF, pagina, Path(d) / f"p{pagina}.png", dpi=300)).mean(2) < 0.5
            e = dg.Eixo(r0, v0, r1, v1)
            for corrente, (linha, coluna) in janelas.items():
                grandeza = nome.split("_")[0]
                valores[(grandeza, corrente)] = float(e.valor(centro(paginas[pagina], linha, coluna, eixo)))
    SAIDA.mkdir(exist_ok=True)
    with open(SAIDA / "n2_puro.csv", "w") as f:
        f.write("# Gordiets et al. 1995, pure N2 (0% O2), 2 Torr, tube radius 0.8 cm, 100 sccm\n"
                "# Tg (fig. 4), ne (fig. 5; none at 0% for 80 mA), E/N (fig. 8, 1e-16 V cm2 = 10 Td), Tv (text)\n"
                "# corrente_mA,Tg_K,ne_cm3,EN_Td,Tv_K\n")
        for i in (15, 30, 80):
            ne = valores.get(("ne", i))
            f.write(f"{i},{valores[('Tg', i)]:.0f},{'' if ne is None else f'{ne:.3g}'},{valores[('EN', i)]:.1f},"
                    f"{TV[i]:.0f}\n")
    for k, v in sorted(valores.items()):
        print(k, f"{v:.4g}")
    with tempfile.TemporaryDirectory() as d:
        figura13(dg.pagina_png(PDF, 15, Path(d) / "p15.png", dpi=300))


if __name__ == "__main__":
    main()
