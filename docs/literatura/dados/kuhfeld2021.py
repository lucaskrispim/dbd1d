# SPDX-License-Identifier: GPL-3.0-or-later
"""Extraction of the figures of Kuhfeld, Luggenhölscher and Czarnetzki 2021, J. Phys.
D 54, 305205 (paper II, analysis), on the CARS measurements in a ns-pulsed N2 jet
(1 mm gap between molybdenum electrodes, 200 mbar, 20 sccm, 1 kHz) of Kuhfeld,
Lepikhin, Luggenhölscher and Czarnetzki 2021, J. Phys. D 54, 305204 (paper I).

Paper II, figures 8, A1 and A2: relative populations N_v/N_gas (in %) of v = 1 to
4 over the pulse period, measured (markers with error bars) and calculated by the
authors (black line), for 3 kV 200 ns, 3 kV 250 ns and 4 kV 200 ns.

Vector figures (pdftocairo). The frame of each panel is the white filled
rectangle; the tick marks are thin black filled rectangles inside the frame, the
major ones 2.47 pt long and the minor ones 1.65 pt. The x axis is logarithmic
with a major tick at each decade, the last one at 1e-3 s (labels read on the
rendered page, 1e-7 to 1e-3); the minor ticks are 4 per decade equally spaced in
the linear value (2.8, 4.6, 6.4 and 8.2), as pgfplots draws them on a log axis,
which the script checks. The y axes are linear, with the values of the major
ticks read on the rendered page and checked by their count. The markers are
filled with orange; the error bar of each marker is the black vertical strokes in
its column. On these log axes the pulse starts at 50 ns (blue band, 50 to 250 ns
or 50 to 300 ns, measured in A1 and A2); the panels with linear time in ns put
its start at 0. In A1 the calculated v = 1 reaches the top of the frame near 1e-4
s and is clipped there by the authors.

Paper I (305204): figure 13, the field measured by E-FISH 500 um from the
cathode (kV/cm); figure 15, the fraction of vibrationally hot molecules R_h;
figure 16, their temperature T_vib,h (K); figure 20, the vibrational quanta per
molecule over the period (log time in us, the pulse from 0.05 us). Paper II,
figure 6: the electron density from the current (1e19 m-3), one curve per
condition. Series by colour: blue circles 3 kV 200 ns, orange squares 3 kV 250
ns, green triangles 4 kV 200 ns; the markers of the legend (inside its white box
with a black outline) are left out. Same tick conventions, labels read on the
rendered pages.

Usage: python3 docs/literatura/dados/kuhfeld2021.py  (writes to kuhfeld2021/)
"""
import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(RAIZ / "ferramentas"))
import digitaliza as dg  # noqa: E402

PDF_I = RAIZ / "docs" / "jato_n2" / "artigos" / "kuhfeld2021_I.pdf"
PDF_II = RAIZ / "docs" / "jato_n2" / "artigos" / "kuhfeld2021_II.pdf"
SAIDA = Path(__file__).with_name("kuhfeld2021")
LARANJA = (0.89, 0.435, 0.28)
PRETO, BRANCO, AZUL = (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (0.5, 0.5, 1.0)

# figure: page, condition, top y (pt) of its first panel, and the values of the
# major y ticks of the panels v = 4, 3, 2, 1 (from the top), read on the page
POPULACOES = {
    "fig8": (8, "3kV_200ns", 80.0, [np.arange(0, 0.41, 0.1), np.arange(0, 0.81, 0.2),
                                    np.arange(0, 2.01, 0.5), np.arange(0, 10.01, 2.5)]),
    "figA1": (10, "3kV_250ns", 80.0, [np.arange(0, 0.61, 0.1), np.arange(0, 1.01, 0.25),
                                      np.arange(0, 2.51, 0.5), np.arange(0, 10.01, 2.5)]),
    "figA2": (10, "4kV_200ns", 440.0, [np.arange(0, 0.81, 0.2), np.arange(0, 1.26, 0.25),
                                       np.arange(0, 3.01, 1.0), np.arange(0, 15.01, 5.0)]),
}

SERIES = {"3kV_200ns": (0.0, 0.6, 0.98), "3kV_250ns": LARANJA, "4kV_200ns": (0.24, 0.64, 0.31)}
# figure: pdf, page, frame (pt, approximate), x axis ('lin' with the values of the
# major ticks, in s, or 'log' with the last decade, in s), values of the major y
# ticks, quantity and unit
FIGURAS = {
    "I_fig13": ("I", 11, (82.6, 87.7, 293.5, 249.5), ("lin", np.arange(0, 301e-9, 100e-9)),
                np.arange(0, 20.1, 5.0), "E_kV_cm"),
    "I_fig15": ("I", 11, (351.1, 428.7, 550.9, 582.0), ("lin", np.arange(0, 301e-9, 100e-9)),
                np.arange(0, 0.1001, 0.025), "R_h"),
    "I_fig16": ("I", 12, (92.8, 87.4, 293.1, 241.1), ("lin", np.arange(0, 301e-9, 100e-9)),
                np.arange(0, 8001, 2000.0), "T_vib_h_K"),
    "I_fig20": ("I", 14, (346.0, 322.0, 543.2, 473.3), ("log", 1e-3), np.arange(0, 0.2501, 0.05),
                "quanta_por_molecula"),
}
# paper II, figure 6: electron density (1e19 m-3) against time
FIG6 = ("II", 6, (336.9, 83.3, 550.9, 247.5), ("lin", np.arange(0, 301e-9, 100e-9)), np.arange(0, 3.01, 1.0))


def grava(nome, cabecalho, colunas, nota):
    SAIDA.mkdir(exist_ok=True)
    np.savetxt(SAIDA / nome, np.column_stack(colunas), fmt="%.6g", delimiter=",",
               header=f"{nota}\n{cabecalho}", comments="# ")
    print(f"{nome}: {len(colunas[0])} points")


def molduras(tr, y_min, y_max):
    """Frames (x0, y_top, x1, y_bottom) of the panels between y_min and y_max: the
    white filled rectangles wider than 100 pt, from the top."""
    saida = []
    for t in tr:
        p = t["p"]
        if (t["preenchimento"] and dg._mesma_cor(t["preenchimento"], BRANCO) and len(p) == 5
                and np.ptp(p[:, 0]) > 100 and np.ptp(p[:, 1]) > 30 and y_min <= p[:, 1].min() <= y_max):
            saida.append((p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()))
    return sorted(saida, key=lambda m: m[1])


def marcas(tr, moldura, eixo):
    """(position, length) in pt of the tick marks of the bottom ('x') or left ('y')
    edge: thin black filled rectangles touching the edge from inside."""
    x0, y0, x1, y1 = moldura
    saida = set()
    for t in tr:
        p = t["p"]
        if not (t["preenchimento"] and dg._mesma_cor(t["preenchimento"], PRETO)) or len(p) < 4:
            continue
        w, h = np.ptp(p[:, 0]), np.ptp(p[:, 1])
        if eixo == "x" and w < 0.3 and 1.0 < h < 4.0 and abs(p[:, 1].max() - y1) < 0.6 \
                and x0 - 0.5 <= p[:, 0].mean() <= x1 + 0.5:
            saida.add((round(p[:, 0].mean(), 3), round(h, 2)))
        if eixo == "y" and h < 0.3 and 1.0 < w < 4.0 and abs(p[:, 0].min() - x0) < 0.6 \
                and y0 - 0.5 <= p[:, 1].mean() <= y1 + 0.5:
            saida.add((round(p[:, 1].mean(), 3), round(w, 2)))
    return sorted(saida)


def maiores(lista):
    longo = max(c for _, c in lista)
    return np.array([p for p, c in lista if c > longo - 0.3])


def eixo_x_log(tr, moldura, ultima=1e-3):
    """Log x axis from the major ticks (one per decade, the last at 'ultima'),
    checking the pgfplots minor ticks (2.8, 4.6, 6.4 and 8.2 in each decade)."""
    lista = marcas(tr, moldura, "x")
    principais = maiores(lista)
    d = np.diff(principais)
    assert len(principais) >= 3 and np.ptp(d) < 0.05, principais
    n = len(principais)
    ex = dg.Eixo(principais[0], ultima / 10.0 ** (n - 1), principais[-1], ultima, log=True)
    menores = np.array([p for p, c in lista if p not in principais])
    frac = np.log10(ex.valor(menores)) % 1.0
    esperado = np.log10([2.8, 4.6, 6.4, 8.2])
    assert np.all(np.min(np.abs(frac[:, None] - esperado[None, :]), axis=1) < 0.01), frac
    return ex


def eixo_y_lin(tr, moldura, valores):
    """Linear y axis from the major ticks, whose count must match the labels."""
    principais = maiores(marcas(tr, moldura, "y"))
    assert len(principais) == len(valores), (moldura, principais, valores)
    assert np.ptp(np.diff(principais)) < 0.05, principais
    # pt grows downwards: the bottom tick is the first value
    return dg.Eixo(principais[-1], valores[0], principais[0], valores[-1])


def barra(tr, c, moldura):
    """Bottom and top (pt) of the error bar of the marker at c. The bars are black
    vertical strokes or, under a shaded region, black filled rectangles narrower
    than 1 pt, inside the frame of the panel. Of those in the column of the marker
    and reaching it (within 3 pt), the one whose middle is closest to the marker
    is taken, since the bars are symmetric and those of the other series at the
    same time overlap; when no middle is within 1.5 pt, the bar is drawn in two
    halves and the union is taken. Without a bar, both ends are the marker."""
    candidatos = []
    for t in tr:
        p = t["p"]
        traco = t["traco"] and dg._mesma_cor(t["traco"], PRETO) and len(p) <= 3 and np.ptp(p[:, 0]) < 0.05
        cheio = (t["preenchimento"] and dg._mesma_cor(t["preenchimento"], PRETO) and len(p) == 5
                 and np.ptp(p[:, 0]) < 1.0)
        if not (traco or cheio) or abs(p[:, 0].mean() - c[0]) > 0.6 or np.ptp(p[:, 1]) < 0.3:
            continue
        lo, hi = p[:, 1].min(), p[:, 1].max()
        if lo >= moldura[1] - 0.5 and hi <= moldura[3] + 0.5 and lo <= c[1] + 3.0 and hi >= c[1] - 3.0:
            candidatos.append((lo, hi))
    if not candidatos:
        return c[1], c[1]
    lo, hi = min(candidatos, key=lambda b: abs(0.5 * (b[0] + b[1]) - c[1]))
    if abs(0.5 * (lo + hi) - c[1]) > 1.5:
        lo, hi = min(b[0] for b in candidatos), max(b[1] for b in candidatos)
    return hi, lo


def faixa_do_pulso(tr, moldura):
    """Ends (pt) of the blue band that marks the pulse in the panel, or None when
    it is drawn as a transparent image (figure 8)."""
    for t in tr:
        p = t["p"]
        if (t["preenchimento"] and dg._mesma_cor(t["preenchimento"], AZUL) and np.ptp(p[:, 1]) > 20
                and dg._dentro(p.mean(0)[None, :], moldura)[0]):
            return p[:, 0].min(), p[:, 0].max()
    return None


def populacoes(nome, tr):
    pagina, condicao, y_topo, valores = POPULACOES[nome]
    paineis = molduras(tr, y_topo, y_topo + 300.0)
    assert len(paineis) == 4, (nome, paineis)
    ex = eixo_x_log(tr, paineis[-1])
    faixa = faixa_do_pulso(tr, paineis[-1])
    if faixa is not None:
        print(f"{nome}: pulse band from {ex.valor(faixa[0]) * 1e9:.0f} to {ex.valor(faixa[1]) * 1e9:.0f} ns")
    for v, moldura, vals in zip((4, 3, 2, 1), paineis, valores):
        ey = eixo_y_lin(tr, moldura, vals)
        janela = (moldura[0] + 0.5, moldura[1] + 0.5, moldura[2] - 0.5, moldura[3] - 0.5)
        mk = dg.marcadores_vetoriais(tr, LARANJA, janela)
        mk = mk[np.argsort(mk[:, 0])]
        linhas = []
        for c in mk:
            base, topo = barra(tr, c, moldura)
            # near zero the bars are clipped by the frame (one sided); a bar that
            # does not reach the marker belongs to a neighbour and is dropped
            if not topo - 0.6 <= c[1] <= base + 0.6:
                base = topo = c[1]
            linhas.append((ex.valor(c[0]), ey.valor(c[1]), ey.valor(base), ey.valor(topo)))
        linhas = np.array(linhas)
        nota = (f"Kuhfeld et al 2021 J. Phys. D 54 305205, {nome}, {condicao}: N_v/N_gas (%) for v = {v}, "
                "measured by CARS (paper I); n_inf and n_sup are the ends of the error bar")
        grava(f"{nome}_v{v}_medidos.csv", "t_s,pct,pct_inf,pct_sup", linhas.T, nota)
        c = dg.curva_vetorial(tr, PRETO, (moldura[0] + 3.0, moldura[1] + 0.5, moldura[2] - 0.5, moldura[3] - 0.5),
                              min_pontos=20)
        grava(f"{nome}_v{v}_calculado.csv", "t_s,pct", (ex.valor(c[:, 0]), ey.valor(c[:, 1])),
              f"Kuhfeld et al 2021 J. Phys. D 54 305205, {nome}, {condicao}: N_v/N_gas (%) for v = {v}, "
              "calculated by the authors (V-V, V-T and wall model of the afterglow)")


def moldura_perto(tr, aprox):
    """The white frame closest to the approximate one."""
    todas = [m for m in molduras(tr, 0.0, 1e4)]
    m = min(todas, key=lambda m: np.abs(np.subtract(m, aprox)).max())
    assert np.abs(np.subtract(m, aprox)).max() < 1.0, (m, aprox)
    return m


def legenda(tr, moldura):
    """Box of the legend: white filled rectangle inside the frame, smaller than
    it."""
    for t in tr:
        p = t["p"]
        if (t["preenchimento"] and dg._mesma_cor(t["preenchimento"], BRANCO) and len(p) == 5
                and 15.0 < np.ptp(p[:, 0]) < 0.8 * (moldura[2] - moldura[0])
                and dg._dentro(p.mean(0)[None, :], moldura)[0]):
            return p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()
    return None


def eixo_x(tr, moldura, spec):
    if spec[0] == "log":
        return eixo_x_log(tr, moldura, spec[1])
    principais = maiores(marcas(tr, moldura, "x"))
    assert len(principais) == len(spec[1]) and np.ptp(np.diff(principais)) < 0.05, (principais, spec)
    return dg.Eixo(principais[0], spec[1][0], principais[-1], spec[1][-1])


def triangulos(tr, cor, janela):
    """Centres (pt) of the triangle markers: the centroid of the three vertices,
    which is the data point of a pgfplots triangle (the centre of its bounding box
    lies a sixth of the height higher)."""
    centros = []
    for t in tr:
        p = t["p"]
        if t["preenchimento"] and dg._mesma_cor(t["preenchimento"], cor) and len(p) == 4 \
                and np.ptp(p[:, 0]) < 12.0:
            c = p[:3].mean(0)
            if dg._dentro(c[None, :], janela, 1.0)[0] and not any(np.hypot(*(c - q)) < 1.0 for q in centros):
                centros.append(c)
    return np.array(centros).reshape(-1, 2)


def series(nome, tr):
    _, _, aprox, xspec, yvals, grandeza = FIGURAS[nome]
    moldura = moldura_perto(tr, aprox)
    ex, ey = eixo_x(tr, moldura, xspec), eixo_y_lin(tr, moldura, yvals)
    caixa = legenda(tr, moldura)
    janela = (moldura[0] + 0.5, moldura[1] + 0.5, moldura[2] - 0.5, moldura[3] - 0.5)
    for condicao, cor in SERIES.items():
        mk = triangulos(tr, cor, janela) if condicao == "4kV_200ns" else dg.marcadores_vetoriais(tr, cor, janela)
        if caixa is not None:
            mk = mk[~dg._dentro(mk, caixa, 0.5)]
        mk = mk[np.argsort(mk[:, 0])]
        linhas = []
        for c in mk:
            base, topo = barra(tr, c, moldura)
            # the bars are symmetric: their middle must fall on the marker
            assert base == topo or abs(0.5 * (base + topo) - c[1]) < 0.6, (nome, condicao, c, base, topo)
            linhas.append((ex.valor(c[0]), ey.valor(c[1]), ey.valor(base), ey.valor(topo)))
        linhas = np.array(linhas).reshape(-1, 4)
        grava(f"{nome}_{condicao}.csv", f"t_s,{grandeza},inf,sup", linhas.T,
              f"Kuhfeld et al 2021 J. Phys. D 54 305204, {nome[2:]}, {condicao}: {grandeza} measured; "
              "inf and sup are the ends of the error bar (equal to the value without a bar)")


def densidade_eletronica(tr):
    _, _, aprox, xspec, yvals = FIG6
    moldura = moldura_perto(tr, aprox)
    ex, ey = eixo_x(tr, moldura, xspec), eixo_y_lin(tr, moldura, yvals)
    janela = (moldura[0] + 0.5, moldura[1] + 0.5, moldura[2] - 0.5, moldura[3] - 0.5)
    caixa = legenda(tr, moldura)
    for condicao, cor in SERIES.items():
        c = dg.curva_vetorial(tr, cor, janela, min_pontos=20)
        if caixa is not None:
            c = c[~dg._dentro(c, caixa, 0.5)]
        grava(f"II_fig6_{condicao}.csv", "t_s,ne_cm3", (ex.valor(c[:, 0]), ey.valor(c[:, 1]) * 1e13),
              f"Kuhfeld et al 2021 J. Phys. D 54 305205, figure 6, {condicao}: electron density from the current "
              "and the E-FISH field (eq. 12), from 20 ns")


def main():
    paginas = {}

    def pagina(pdf, n):
        return paginas.setdefault((pdf, n), dg.tracos_pdf(PDF_I if pdf == "I" else PDF_II, n))
    for nome, (n, *_) in POPULACOES.items():
        populacoes(nome, pagina("II", n))
    for nome, (pdf, n, *_) in FIGURAS.items():
        series(nome, pagina(pdf, n))
    densidade_eletronica(pagina(*FIG6[:2]))
    confere_com_o_texto()


def confere_com_o_texto():
    """Values stated in the text of paper I, section 5.1: T_vib,h averaged (here
    weighted by the error bars) for t >
    100 ns (> 50 ns at 4 kV) of 5200 +- 300, 4600 +- 200 and 5300 +- 100 K; R_h
    averaged over the first three points (t <= 0) of 0.015, 0.021 and 0.022; the
    field of about 3.5 kV/cm."""
    def le(nome):
        return np.loadtxt(SAIDA / nome, delimiter=",", ndmin=2)
    for condicao, t_min, esperado, tol in (("3kV_200ns", 100e-9, 5200, 300), ("3kV_250ns", 100e-9, 4600, 200),
                                            ("4kV_200ns", 50e-9, 5300, 100)):
        d = le(f"I_fig16_{condicao}.csv")
        d = d[d[:, 0] > t_min + 1e-9]
        # mean weighted by the error bars (the unweighted one gives 5170, 4580 and 5450 K)
        peso = 1.0 / (0.5 * (d[:, 3] - d[:, 2])) ** 2
        media = np.sum(peso * d[:, 1]) / np.sum(peso)
        assert abs(media - esperado) < tol, (condicao, media)
        print(f"T_vib,h {condicao}: {media:.0f} K (text {esperado})")
    for condicao, esperado in (("3kV_200ns", 0.015), ("3kV_250ns", 0.021), ("4kV_200ns", 0.022)):
        d = le(f"I_fig15_{condicao}.csv")
        media = d[d[:, 0] < 1e-9, 1][:3].mean()
        assert abs(media - esperado) < 0.004, (condicao, media)
        print(f"R_h before the pulse {condicao}: {media:.3f} (text {esperado})")
    campo = np.concatenate([le(f"I_fig13_{c}.csv")[:, 1] for c in SERIES])
    assert abs(campo.mean() - 3.5) < 0.3, campo.mean()
    print(f"E-FISH field: mean {campo.mean():.2f} kV/cm (text about 3.5)")


if __name__ == "__main__":
    main()
