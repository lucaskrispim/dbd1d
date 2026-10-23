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


if __name__ == "__main__":
    main()
