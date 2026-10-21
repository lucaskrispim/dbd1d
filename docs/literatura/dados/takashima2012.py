# SPDX-License-Identifier: GPL-3.0-or-later
"""Extraction of the coupled energy per pulse versus pressure in Takashima et al.
2012, AIAA 2012-3093 (nanosecond pulse DBD in dry air, plane to plane, quartz
channel 10 mm high with 1.75 mm walls). Figures 8 (right, CPT generator, pulses
of ~50 ns) and 10 (right, FID generator, pulses of 5 to 10 ns, 10 kHz), JPEG
images of 150 ppi on pages 18 and 19.

Calibration (pixels of the embedded image): figure 8, x at 43.5 (0 Torr) and 353
(300 Torr), y at 344 (0 mJ) and 44 (3 mJ); figure 10, x at 27.5 (0 Torr) and 388
(500 Torr), y at 344 (0 mJ) and 43.5 (6 mJ). Checked against the labels.

Usage: python3 docs/literatura/dados/takashima2012.py  (writes to takashima2012/)
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(RAIZ / "ferramentas"))
import digitaliza as dg  # noqa: E402

PDF = RAIZ / "docs" / "artigos" / "takashima2012.pdf"
SAIDA = Path(__file__).with_name("takashima2012")
VERMELHO, AZUL = (0.95, 0.05, 0.05), (0.05, 0.05, 0.95)


def fora(c, caixa):
    x0, y0, x1, y1 = caixa
    return c[~((c[:, 0] > x0) & (c[:, 0] < x1) & (c[:, 1] > y0) & (c[:, 1] < y1))]


def campo_aplicado(d):
    """Measured applied field (solid red curve) of figs. 7 (CPT, 60 Torr) and 9
    (FID, 60 Torr), in V/cm. In fig. 7 the red texts inside the plot are erased
    (boxes that do not cross the curve) and, before breakdown (92 ns), the blue
    curve of the field in the plasma is used: it is drawn over the red one and
    equals it up to there. The dashed Gaussian fit, of the same colour, is left
    out because it breaks into pieces (only the largest connected component
    counts). Calibration: ticks every 20 ns at x = 60.5 to 419 px; 0 V/cm at
    179.5 px (fig. 7) and 210 px (fig. 9), every 5000 V/cm each 33.3 and 27.3 px."""
    for nome, png, t0, y0, y1, v1, textos in (
            ("fig7_campo_aplicado_CPT", d / "img-000.png", 60.0, 179.5, 46.5, 20000.0,
             [(62, 243, 142, 263), (196, 303, 282, 323)]),
            ("fig9_campo_aplicado_FID", d / "img-003.png", 80.0, 210.0, 46.5, 30000.0,
             [])):
        img = dg.le_png(png).copy()
        for x0, ya, x1, yb in textos:
            img[ya:yb, x0:x1] = 1.0
        c = dg.curva_raster(img, VERMELHO, (61, 40, 418, 352), 0.35, maior_componente=True)
        ex, ey = dg.Eixo(60.5, t0, 419.0, t0 + 120.0), dg.Eixo(y0, 0.0, y1, v1)
        if nome.startswith("fig7"):
            # before breakdown (92 ns) the blue curve, field in the plasma, equal to
            # the applied one up to there, is drawn over the red one
            azul = dg.curva_raster(img, AZUL, (61, 40, 418, 352), 0.35, maior_componente=True)
            corte = ex.c1 + (92.0 - t0) * (ex.c2 - ex.c1) / 120.0
            c = np.concatenate([azul[azul[:, 0] < corte], c[c[:, 0] >= corte]])
        np.savetxt(SAIDA / f"{nome}.csv", np.column_stack([ex.valor(c[:, 0]), ey.valor(c[:, 1])]), fmt="%.5g",
                   delimiter=",", comments="# ",
                   header=f"Takashima 2012 {nome.split('_')[0]}: measured applied field in air at 60 Torr (V/cm); the "
                          "electrode voltage is the field times d + 2 l / eps of the barrier, according to their model\nt_ns,E_V_cm")
        print(f"{nome}: {len(c)} points, min {ey.valor(c[:, 1]).min():.0f} and max {ey.valor(c[:, 1]).max():.0f} V/cm")


def main():
    d = Path(tempfile.mkdtemp())
    subprocess.run(["pdfimages", "-f", "18", "-l", "19", "-png", str(PDF), str(d / "img")], check=True)
    SAIDA.mkdir(exist_ok=True)
    campo_aplicado(d)
    casos = [
        ("fig8_energia_CPT", d / "img-001.png", (44, 44, 405, 344), dg.Eixo(43.5, 0.0, 353.0, 300.0),
         dg.Eixo(344.0, 0.0, 44.0, 3.0), (225, 50, 350, 110),
         [("40kHz", VERMELHO), ("20kHz", AZUL)], "CPT generator (~50 ns, 25 kV), coupled energy per pulse"),
        ("fig10_energia_FID", d / "img-005.png", (28, 43, 389, 344), dg.Eixo(27.5, 0.0, 388.0, 500.0),
         dg.Eixo(344.0, 0.0, 43.5, 6.0), (140, 255, 215, 300),
         [("incidente", VERMELHO), ("incidente_e_refletidos", AZUL)], "FID generator (5 to 10 ns, 30 kV), 10 kHz"),
    ]
    for nome, png, jan, ex, ey, legenda, series, desc in casos:
        img = dg.le_png(png)
        for rotulo, cor in series:
            c = fora(dg.marcadores_raster(img, cor, jan, 0.35, min_px=10, abertura=4, separa=True), legenda)
            c = c[np.argsort(c[:, 0])]
            p, e = ex.valor(c[:, 0]), ey.valor(c[:, 1])
            np.savetxt(SAIDA / f"{nome}_{rotulo}.csv", np.column_stack([p, e]), fmt="%.4g", delimiter=",", comments="# ",
                       header=f"Takashima 2012 {nome.split('_')[0]}: dry air, {desc}; {rotulo}; left out are the points "
                              "on the 0 Torr axis, those hidden by markers of another series (several in the 20 kHz series) and "
                              "the 492 Torr point of the incident series, too small\np_torr,E_mJ")
            print(f"{nome}_{rotulo}: {len(c)} points, p from {p.min():.0f} to {p.max():.0f} Torr, E max {e.max():.2f} mJ")


if __name__ == "__main__":
    main()
