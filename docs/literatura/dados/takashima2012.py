# SPDX-License-Identifier: GPL-3.0-or-later
"""Extração da energia acoplada por pulso contra a pressão em Takashima et al.
2012, AIAA 2012-3093 (DBD de pulsos de nanossegundos em ar seco, plano a
plano, canal de quartzo de 10 mm de altura com paredes de 1,75 mm). Figuras
8 (direita, gerador CPT, pulsos de ~50 ns) e 10 (direita, gerador FID, pulsos
de 5 a 10 ns, 10 kHz), imagens JPEG de 150 ppi nas páginas 18 e 19.

Calibração (pixels da imagem embutida): figura 8, x em 43,5 (0 Torr) e 353
(300 Torr), y em 344 (0 mJ) e 44 (3 mJ); figura 10, x em 27,5 (0 Torr) e 388
(500 Torr), y em 344 (0 mJ) e 43,5 (6 mJ). Conferidas pelos rótulos.

Uso: python3 docs/literatura/dados/takashima2012.py  (grava em takashima2012/)
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
    """Campo aplicado medido (curva vermelha contínua) das figs. 7 (CPT, 60 Torr)
    e 9 (FID, 60 Torr), em V/cm. Na fig. 7, os textos vermelhos dentro do
    gráfico são apagados (caixas que não cruzam a curva) e, antes da ruptura
    (92 ns), vale a curva azul do campo no plasma, desenhada por cima da
    vermelha e igual a ela até ali. O ajuste gaussiano tracejado, da mesma cor, fica de fora por se
    partir em pedaços (só o maior componente conexo conta). Calibração: traços de 20 em 20 ns em
    x = 60,5 a 419 px; 0 V/cm em 179,5 px (fig. 7) e 210 px (fig. 9), de
    5000 em 5000 V/cm a cada 33,3 e 27,3 px."""
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
            # antes da ruptura (92 ns) a curva azul, campo no plasma, igual ao
            # aplicado até ali, está desenhada por cima da vermelha
            azul = dg.curva_raster(img, AZUL, (61, 40, 418, 352), 0.35, maior_componente=True)
            corte = ex.c1 + (92.0 - t0) * (ex.c2 - ex.c1) / 120.0
            c = np.concatenate([azul[azul[:, 0] < corte], c[c[:, 0] >= corte]])
        np.savetxt(SAIDA / f"{nome}.csv", np.column_stack([ex.valor(c[:, 0]), ey.valor(c[:, 1])]), fmt="%.5g",
                   delimiter=",", comments="# ",
                   header=f"Takashima 2012 {nome.split('_')[0]}: campo aplicado medido em ar a 60 Torr (V/cm); a "
                          "tensão nos eletrodos é o campo vezes d + 2 l / ε da barreira, segundo o modelo deles\nt_ns,E_V_cm")
        print(f"{nome}: {len(c)} pontos, mín {ey.valor(c[:, 1]).min():.0f} e máx {ey.valor(c[:, 1]).max():.0f} V/cm")


def main():
    d = Path(tempfile.mkdtemp())
    subprocess.run(["pdfimages", "-f", "18", "-l", "19", "-png", str(PDF), str(d / "img")], check=True)
    SAIDA.mkdir(exist_ok=True)
    campo_aplicado(d)
    casos = [
        ("fig8_energia_CPT", d / "img-001.png", (44, 44, 405, 344), dg.Eixo(43.5, 0.0, 353.0, 300.0),
         dg.Eixo(344.0, 0.0, 44.0, 3.0), (225, 50, 350, 110),
         [("40kHz", VERMELHO), ("20kHz", AZUL)], "gerador CPT (~50 ns, 25 kV), energia acoplada por pulso"),
        ("fig10_energia_FID", d / "img-005.png", (28, 43, 389, 344), dg.Eixo(27.5, 0.0, 388.0, 500.0),
         dg.Eixo(344.0, 0.0, 43.5, 6.0), (140, 255, 215, 300),
         [("incidente", VERMELHO), ("incidente_e_refletidos", AZUL)], "gerador FID (5 a 10 ns, 30 kV), 10 kHz"),
    ]
    for nome, png, jan, ex, ey, legenda, series, desc in casos:
        img = dg.le_png(png)
        for rotulo, cor in series:
            c = fora(dg.marcadores_raster(img, cor, jan, 0.35, min_px=10, abertura=4, separa=True), legenda)
            c = c[np.argsort(c[:, 0])]
            p, e = ex.valor(c[:, 0]), ey.valor(c[:, 1])
            np.savetxt(SAIDA / f"{nome}_{rotulo}.csv", np.column_stack([p, e]), fmt="%.4g", delimiter=",", comments="# ",
                       header=f"Takashima 2012 {nome.split('_')[0]}: ar seco, {desc}; {rotulo}; ficam de fora os pontos "
                              "sobre o eixo de 0 Torr, os encobertos por marcadores de outra série (na série de 20 kHz, vários) e "
                              "o ponto de 492 Torr da série incidente, pequeno demais\np_torr,E_mJ")
            print(f"{nome}_{rotulo}: {len(c)} pontos, p de {p.min():.0f} a {p.max():.0f} Torr, E máx {e.max():.2f} mJ")


if __name__ == "__main__":
    main()
