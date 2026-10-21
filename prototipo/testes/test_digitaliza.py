# SPDX-License-Identifier: GPL-3.0-or-later
"""Digitalização de figuras de artigos (ferramentas/digitaliza.py): um gráfico
com dados conhecidos é gerado pelo matplotlib em PDF (vetorial) e em PNG
(raster), e o digitalizador tem de devolver os mesmos dados a partir da
moldura dos eixos e da cor de cada série."""
import shutil
import sys

import numpy as np
import pytest

from conftest import RAIZ

sys.path.insert(0, str(RAIZ / "ferramentas"))
import digitaliza as dg  # noqa: E402

# a leitura vetorial passa o PDF pelo pdftocairo (pacote poppler-utils)
precisa_pdftocairo = pytest.mark.skipif(shutil.which("pdftocairo") is None,
                                        reason="pdftocairo (poppler-utils) não instalado")

X_PONTOS = np.array([100.0, 300.0, 500.0, 650.0])
Y_PONTOS = np.array([0.559, 0.935, 1.535, 2.474])
X_CURVA = np.linspace(0.0, 40.0, 400)
Y_CURVA = 0.012 * np.exp(-((X_CURVA - 18.7) / 0.6) ** 2) + 0.003 * np.sin(X_CURVA / 6.0)


def _figura(caminho, log=False, pretos=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a, b) = plt.subplots(1, 2, figsize=(8, 3.5))
    a.plot(X_PONTOS, Y_PONTOS, "s", color="red", ms=5)
    if pretos:
        # marcador preto dentro do gráfico: lados verticais curtos que não são traços de escala
        a.plot([350.0], [0.1], "s", color="black", mfc="none", ms=5)
    a.set_xlim(0.0, 700.0)
    a.set_ylim(0.0, 3.0)
    a.set_xticks(np.arange(0.0, 701.0, 100.0))
    a.set_yticks(np.arange(0.0, 3.01, 0.5))
    b.plot(X_CURVA, Y_CURVA, "-", color="blue", lw=3)
    b.set_xlim(0.0, 40.0)
    b.set_ylim(-0.02, 0.02)
    if log:
        a.set_yscale("log")
        a.set_ylim(0.1, 10.0)
    fig.savefig(caminho, dpi=200)
    plt.close(fig)


def _janelas(molduras):
    # duas molduras, ordenadas da esquerda para a direita
    assert len(molduras) == 2
    return sorted(molduras, key=lambda m: m[0])


@precisa_pdftocairo
@pytest.mark.parametrize("log", [False, True])
def test_vetorial_marcadores_e_curva(tmp_path, log):
    pdf = tmp_path / "fig.pdf"
    _figura(pdf, log)
    tr = dg.tracos_pdf(pdf, 1)
    esq, dir_ = _janelas(dg.molduras_vetoriais(tr))
    ex = dg.Eixo(esq[0], 0.0, esq[2], 700.0)
    ey = dg.Eixo(esq[3], 0.1, esq[1], 10.0, log=True) if log else dg.Eixo(esq[3], 0.0, esq[1], 3.0)
    p = dg.marcadores_vetoriais(tr, (1.0, 0.0, 0.0), esq)
    x, y = ex.valor(p[:, 0]), ey.valor(p[:, 1])
    ordem = np.argsort(x)
    np.testing.assert_allclose(x[ordem], X_PONTOS, atol=1e-3 * 700)
    np.testing.assert_allclose(y[ordem], Y_PONTOS, rtol=2e-3)
    c = dg.curva_vetorial(tr, (0.0, 0.0, 1.0), dir_)
    ex2 = dg.Eixo(dir_[0], 0.0, dir_[2], 40.0)
    ey2 = dg.Eixo(dir_[3], -0.02, dir_[1], 0.02)
    xc, yc = ex2.valor(c[:, 0]), ey2.valor(c[:, 1])
    np.testing.assert_allclose(yc, np.interp(xc, X_CURVA, Y_CURVA), atol=1e-4)


def test_raster_marcadores_e_curva(tmp_path):
    png = tmp_path / "fig.png"
    _figura(png)
    img = dg.le_png(png)
    esq, dir_ = _janelas(dg.molduras_raster(img))
    ex = dg.Eixo(esq[0], 0.0, esq[2], 700.0)
    ey = dg.Eixo(esq[3], 0.0, esq[1], 3.0)
    p = dg.marcadores_raster(img, (1.0, 0.0, 0.0), esq)
    x, y = ex.valor(p[:, 0]), ey.valor(p[:, 1])
    ordem = np.argsort(x)
    np.testing.assert_allclose(x[ordem], X_PONTOS, atol=4e-3 * 700)
    np.testing.assert_allclose(y[ordem], Y_PONTOS, atol=4e-3 * 3)
    c = dg.curva_raster(img, (0.0, 0.0, 1.0), dir_)
    ex2 = dg.Eixo(dir_[0], 0.0, dir_[2], 40.0)
    ey2 = dg.Eixo(dir_[3], -0.02, dir_[1], 0.02)
    xc, yc = ex2.valor(c[:, 0]), ey2.valor(c[:, 1])
    # no raster o erro é da ordem de um pixel; no pico estreito a coluna
    # de pixels mistura a subida e a descida
    longe = np.abs(xc - 18.7) > 1.5
    np.testing.assert_allclose(yc[longe], np.interp(xc[longe], X_CURVA, Y_CURVA), atol=6e-3 * 0.04)
    xp, yp = dg.extremo_raster(img, (0.0, 0.0, 1.0), dir_)
    assert abs(ey2.valor(yp) - Y_CURVA.max()) < 0.01 * Y_CURVA.max()
    assert abs(ex2.valor(xp) - X_CURVA[np.argmax(Y_CURVA)]) < 0.1


def test_eixo_log_e_linear():
    e = dg.Eixo(100.0, 1.0, 300.0, 100.0, log=True)
    np.testing.assert_allclose(e.valor(np.array([100.0, 200.0, 300.0])), [1.0, 10.0, 100.0])
    e = dg.Eixo(50.0, 0.0, 10.0, 4.0)
    np.testing.assert_allclose(e.valor(np.array([50.0, 30.0])), [0.0, 2.0])


@precisa_pdftocairo
def test_tracos_de_escala_vetoriais(tmp_path):
    # matplotlib põe traços em 0, 100, ..., 700 no eixo x e 0, 0,5, ..., 3 no y
    pdf = tmp_path / "fig.pdf"
    _figura(pdf, pretos=True)
    tr = dg.tracos_pdf(pdf, 1)
    esq, _ = _janelas(dg.molduras_vetoriais(tr))
    # desenhado duas vezes (como em figuras com painéis sobrepostos), conta uma
    tr = tr + tr
    xs = dg.tracos_de_escala(tr, esq, "x", min_comp=2.0)
    ys = dg.tracos_de_escala(tr, esq, "y", min_comp=2.0)
    assert len(xs) == 8 and len(ys) == 7
    np.testing.assert_allclose(np.diff(xs), (esq[2] - esq[0]) / 7, rtol=1e-3)
    assert abs(ys[0] - esq[1]) < 0.5 and abs(ys[-1] - esq[3]) < 0.5


def test_tracos_de_escala_raster(tmp_path):
    png = tmp_path / "fig.png"
    _figura(png, pretos=True)
    img = dg.le_png(png)
    esq, _ = _janelas(dg.molduras_raster(img))
    xs = dg.tracos_de_escala_raster(img, esq[3], "x", (esq[0] - 3, esq[2] + 3))
    ys = dg.tracos_de_escala_raster(img, esq[0], "y", (esq[1] - 3, esq[3] + 3))
    assert len(xs) == 8 and len(ys) == 7
    np.testing.assert_allclose(np.diff(xs), (esq[2] - esq[0]) / 7, atol=1.0)
    assert abs(ys[0] - esq[1]) < 1.5 and abs(ys[-1] - esq[3]) < 1.5


def test_raster_marcadores_com_barra_e_linha(tmp_path):
    # pontos com barra de erro e uma linha de modelo na mesma cor, como em Yang 2022
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = np.array([10.0, 40.0, 70.0, 100.0, 130.0])
    y = np.array([1.0, 2.2, 2.9, 3.4, 3.6])
    err = np.array([0.1, 0.3, 0.15, 0.2, 0.25])
    fig, a = plt.subplots(figsize=(5, 4))
    a.errorbar(x, y, yerr=err, fmt="o", color="blue", ms=6, capsize=3, elinewidth=1)
    xx = np.linspace(0, 150, 300)
    a.plot(xx, 4.0 * (1 - np.exp(-xx / 50.0)), "-", color="blue", lw=1.2)
    a.set_xlim(0, 150)
    a.set_ylim(0, 5)
    png = tmp_path / "f.png"
    fig.savefig(png, dpi=150)
    plt.close(fig)
    img = dg.le_png(png)
    m = dg.molduras_raster(img)[0]
    ex, ey = dg.Eixo(m[0], 0.0, m[2], 150.0), dg.Eixo(m[3], 0.0, m[1], 5.0)
    c = dg.marcadores_raster(img, (0.0, 0.0, 1.0), m, abertura=5)
    c = c[np.argsort(c[:, 0])]
    assert len(c) == 5
    np.testing.assert_allclose(ex.valor(c[:, 0]), x, atol=0.01 * 150)
    np.testing.assert_allclose(ey.valor(c[:, 1]), y, atol=0.01 * 5)
    topo, base = dg.barras_raster(img, (0.0, 0.0, 1.0), c, m, alcance=60)
    np.testing.assert_allclose(ey.valor(topo), y + err, atol=0.015 * 5)
    np.testing.assert_allclose(ey.valor(base), y - err, atol=0.015 * 5)


@precisa_pdftocairo
def test_moldura_desenhada_como_caminho_fechado(tmp_path):
    # gnuplot desenha a moldura como um retângulo fechado (M L L L Z), não
    # como quatro linhas: o lado que o Z fecha também conta
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig = plt.figure(figsize=(4, 3))
    a = fig.add_axes([0, 0, 1, 1])
    a.set_axis_off()
    a.set_xlim(0, 10)
    a.set_ylim(0, 10)
    a.add_patch(Rectangle((2, 2), 6, 5, fill=False, edgecolor="black", lw=1))
    # e desenha a moldura duas vezes, com uma diferença de décimos de ponto
    a.add_patch(Rectangle((2, 2), 6.003, 5, fill=False, edgecolor="black", lw=1))
    pdf = tmp_path / "r.pdf"
    fig.savefig(pdf)
    plt.close(fig)
    m = dg.molduras_vetoriais(dg.tracos_pdf(pdf, 1))
    assert len(m) == 1
    x0, y0, x1, y1 = m[0]
    np.testing.assert_allclose([(x1 - x0) / (y1 - y0)], [6 * 4 / (5 * 3)], rtol=1e-3)


def _figura_pontos(png, x, y, err, alfa_barra=1.0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, a = plt.subplots(figsize=(5, 4))
    a.errorbar(x, y, yerr=err, fmt="o", color="red", ms=7, capsize=3, elinewidth=1,
               ecolor=(1.0, 0.0, 0.0, alfa_barra))
    a.set_xlim(0, 150)
    a.set_ylim(0, 5)
    fig.savefig(png, dpi=150)
    plt.close(fig)
    img = dg.le_png(png)
    m = dg.molduras_raster(img)[0]
    return img, m, dg.Eixo(m[0], 0.0, m[2], 150.0), dg.Eixo(m[3], 0.0, m[1], 5.0)


def test_marcadores_encostados_sao_separados(tmp_path):
    # como os pulsos 2 e 5 da fig. 3 de Yang: dois marcadores que se tocam
    x = np.array([10.0, 13.0, 60.0, 100.0])
    y = np.array([3.0, 3.05, 2.0, 1.0])
    img, m, ex, ey = _figura_pontos(tmp_path / "f.png", x, y, np.zeros(4))
    c = dg.marcadores_raster(img, (1.0, 0.0, 0.0), m, abertura=3, separa=True)
    c = c[np.argsort(c[:, 0])]
    assert len(c) == 4
    np.testing.assert_allclose(ex.valor(c[:, 0]), x, atol=0.012 * 150)
    np.testing.assert_allclose(ey.valor(c[:, 1]), y, atol=0.015 * 5)


def test_barra_de_erro_clara_e_seguida(tmp_path):
    # barras semitransparentes (claras), como nas figs. 3, 9 e 14 de Yang
    x = np.array([20.0, 70.0, 120.0])
    y = np.array([2.0, 3.0, 2.5])
    err = np.array([0.6, 1.0, 0.4])
    img, m, ex, ey = _figura_pontos(tmp_path / "f.png", x, y, err, alfa_barra=0.35)
    c = dg.marcadores_raster(img, (1.0, 0.0, 0.0), m, abertura=3)
    c = c[np.argsort(c[:, 0])]
    topo, base = dg.barras_raster(img, (1.0, 0.0, 0.0), c, m, alcance=150, alfa_min=0.2)
    np.testing.assert_allclose(ey.valor(topo), y + err, atol=0.02 * 5)
    np.testing.assert_allclose(ey.valor(base), y - err, atol=0.02 * 5)


def test_curva_seguida_separa_continua_de_tracejada(tmp_path):
    # como o campo aplicado de Takashima 2012: a medida (contínua) e o ajuste
    # (tracejado) na mesma cor, juntos no começo e separados depois
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = np.linspace(0, 10, 500)
    real = np.where(x < 3, 0.0, np.sin(x - 3) * 2)
    ajuste = np.where(x < 3, 0.0, -(x - 3) * 0.5)
    fig, a = plt.subplots(figsize=(6, 4))
    a.plot(x, real, "-", color="red", lw=1.5)
    a.plot(x, ajuste, "--", color="red", lw=1.5)
    a.set_xlim(0, 10)
    a.set_ylim(-4, 4)
    png = tmp_path / "f.png"
    fig.savefig(png, dpi=150)
    plt.close(fig)
    img = dg.le_png(png)
    m = dg.molduras_raster(img)[0]
    ex, ey = dg.Eixo(m[0], 0.0, m[2], 10.0), dg.Eixo(m[3], -4.0, m[1], 4.0)
    for c in (dg.curva_raster_seguida(img, (1.0, 0.0, 0.0), m),
              dg.curva_raster(img, (1.0, 0.0, 0.0), m, maior_componente=True)):
        xc, yc = ex.valor(c[:, 0]), ey.valor(c[:, 1])
        np.testing.assert_allclose(yc, np.interp(xc, x, real), atol=0.06)
        assert xc.max() > 9.5
