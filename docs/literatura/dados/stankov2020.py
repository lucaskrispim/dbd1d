# SPDX-License-Identifier: GPL-3.0-or-later
"""Extraction of the data of Stankov et al. 2020, PSST 29 125009 (argon DBD,
3 mm gap, 1 mm quartz, 24 kHz). The figures are vector graphics: the coordinates
come exactly from the PDF and the calibration uses the tick marks of each panel,
written below in pt (y downwards) with the value of each.

Usage: python3 docs/literatura/dados/stankov2020.py  (writes to stankov2020/)
"""
import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(RAIZ / "ferramentas"))
import digitaliza as dg  # noqa: E402

PDF = RAIZ / "docs" / "artigos" / "Stankov_2020_Plasma_Sources_Sci._Technol._29_125009.pdf"
SAIDA = Path(__file__).with_name("stankov2020")
PERIODO_US = 1e6 / 24e3
PRESSOES = (100.0, 300.0, 500.0, 650.0)
PRETO, VERMELHO, AZUL, VERDE, LARANJA, MAGENTA = (0, 0, 0), (1, 0, 0), (0, 0, 1), (0, 0.5, 0), (1, 0.5, 0), (1, 0, 1)
AZUL_FIG4 = (0.12, 0.24, 1.0)   # gamma = 0.01 in figure 4
AZUL_CLARO = (0.0, 0.62, 1.0)   # applied voltage in figure 4


def grava(nome, cabecalho, colunas, nota):
    SAIDA.mkdir(exist_ok=True)
    np.savetxt(SAIDA / nome, np.column_stack(colunas), fmt="%.6g", delimiter=",",
               header=f"{nota}\n{cabecalho}", comments="# ")
    print(f"{nome}: {len(colunas[0])} points")


def curva(tr, cor, janela, ex, ey, min_pontos=20):
    c = dg.curva_vetorial(tr, cor, janela, min_pontos)
    return ex.valor(c[:, 0]), ey.valor(c[:, 1])


def principais(tr, moldura, eixo, n, min_comp):
    """Major tick marks; checks the expected number."""
    t = dg.tracos_de_escala(tr, moldura, eixo, min_comp=min_comp)
    assert len(t) == n, (eixo, t)
    return t


def figura3(tr):
    """Current measured at 100 mbar and 1.8 kV in the periodic regime (main panel
    and the two insets)."""
    m = (108.33, 64.55, 260.14, 167.68)
    tx = principais(tr, m, "x", 9, 3.5)            # 0, 5, ..., 40 us
    ty = principais(tr, m, "y", 5, 2.5)            # 0.02 ... -0.02 A
    ex, ey = dg.Eixo(tx[0], 0.0, tx[-1], 40.0), dg.Eixo(ty[0], 0.02, ty[-1], -0.02)
    c = dg.curva_vetorial(tr, PRETO, m, 100)
    # the two insets lie inside the main frame
    for x0, y0, x1, y1 in ((169.4, 118.4, 210.07, 156.5), (214.89, 66.96, 255.53, 105.02)):
        c = c[~((c[:, 0] >= x0 - 1) & (c[:, 0] <= x1 + 1) & (c[:, 1] >= y0 - 1) & (c[:, 1] <= y1 + 1))]
    t, i = ex.valor(c[:, 0]), ey.valor(c[:, 1])
    grava("fig3_corrente_medida_100mbar.csv", "t_us,I_A", [t, i],
          "Stankov 2020 fig. 3: measured current, 100 mbar, U0 = 1.8 kV, 24 kHz, t = 0 at the start of a periodic period")
    # detail of the peak (lower inset): ticks at 17.5 to 20 us every 0.5 and at 0 and 0.010 A
    m2 = (169.4, 118.4, 210.07, 156.5)
    ex2, ey2 = dg.Eixo(172.94, 17.5, 206.71, 20.0), dg.Eixo(156.5, 0.0, 135.34, 0.010)
    t2, i2 = curva(tr, PRETO, m2, ex2, ey2, 100)
    grava("fig3_pico_medido_100mbar.csv", "t_us,I_A", [t2, i2], "Stankov 2020 fig. 3, detail of the main peak")
    # detail of the hump (upper inset): 20 to 28 us; 0.006 to 0.001 A
    m3 = (214.89, 66.96, 255.53, 105.02)
    ex3, ey3 = dg.Eixo(221.22, 20.0, 250.77, 28.0), dg.Eixo(66.96, 0.006, 105.02, 0.001)
    t3, i3 = curva(tr, PRETO, (m3[0] + 1, m3[1], m3[2], m3[3]), ex3, ey3, 100)
    grava("fig3_corcova_medida_100mbar.csv", "t_us,I_A", [t3, i3], "Stankov 2020 fig. 3, detail of the hump after the peak")
    return ex, ey


def figura4(tr):
    """Computed by them at 100 mbar and 1.8 kV for each gamma: gap voltage, applied
    voltage and gap-averaged n_e."""
    # major ticks of the time axis at 369.6 (0 us) and 480.86 (40 us); of the gap
    # voltage at 108.3 (0 V) and 66.09 (800 V); of the applied voltage (right axis,
    # light blue) at 108.3 (0 V) and 70.8 (2000 V); of n_e at 256.03 (1e13 m-3) and
    # 185.11 (1e16 m-3)
    ex = dg.Eixo(369.6, 0.0, 480.86, 40.0)
    ma = (369.59, 66.09, 485.07, 150.39)
    ey_gap = dg.Eixo(108.3, 0.0, 66.09, 800.0)
    ey_ua = dg.Eixo(108.3, 0.0, 70.8, 2000.0)
    t, ua = curva(tr, AZUL_CLARO, ma, ex, ey_ua)
    grava("fig4_tensao_aplicada.csv", "t_us,Ua_V", [t, ua], "Stankov 2020 fig. 4(a): applied voltage (right axis)")
    mb = (369.64, 169.72, 485.11, 256.03)
    ey_ne = dg.Eixo(256.03, 1e13, 185.11, 1e16, log=True)
    for g, cor in (("0.005", VERMELHO), ("0.01", AZUL_FIG4), ("0.02", VERDE), ("0.04", LARANJA), ("0.08", MAGENTA)):
        t, u = curva(tr, cor, ma, ex, ey_gap)
        grava(f"fig4_ugap_modelo_gama{g}.csv", "t_us,Ugap_V", [t, u], f"Stankov 2020 fig. 4(a): computed U_gap, gamma = {g}")
    for g, cor in (("0.005", VERMELHO), ("0.01", AZUL), ("0.02", VERDE), ("0.04", LARANJA), ("0.08", MAGENTA)):
        t, n = curva(tr, cor, mb, ex, ey_ne)
        grava(f"fig4_ne_medio_modelo_gama{g}.csv", "t_us,ne_m3", [t, n],
              f"Stankov 2020 fig. 4(b): computed gap-averaged n_e, gamma = {g}"
              + ("; the green curve ends at 40.0 us in the PDF" if g == "0.02" else ""))


def figura6(tr):
    """Current measured at 650 mbar and 3.4 kV, ten periods (first panel)."""
    m = (373.21, 194.18, 513.43, 256.7)
    ey = dg.Eixo(199.95, 0.10, 256.7, -0.10)
    tx = dg.tracos_de_escala(tr, m, "x", min_comp=3.0)
    # 0, T, ..., 10T: the first tick is the axis at x = 373.21
    assert len(tx) == 11, tx
    ex = dg.Eixo(tx[0], 0.0, tx[-1], 10 * PERIODO_US)
    t, i = curva(tr, PRETO, (m[0] - 0.1, m[1], m[2] + 0.1, m[3]), ex, ey, 100)
    t, i = t[t >= 0.0], i[t >= 0.0]
    grava("fig6_corrente_medida_650mbar.csv", "t_us,I_A", [t, i],
          "Stankov 2020 fig. 6: measured current, 650 mbar, U0 = 3.4 kV, ten periods of 41.67 us")


def figura7(tr):
    """Mean power versus pressure: measured and computed for each gamma."""
    m = (117.16, 65.31, 253.68, 193.18)
    ex = dg.Eixo(127.7, 100.0, 253.68, 700.0)
    ey = dg.Eixo(180.38, 0.5, 78.12, 2.5)
    legenda = (117.0, 65.0, 205.0, 120.0)
    linhas = {}
    for nome, cor in (("medida", PRETO), ("gama0.005", VERMELHO), ("gama0.01", AZUL), ("gama0.02", VERDE),
                      ("gama0.04", LARANJA), ("gama0.08", MAGENTA)):
        p = nas_pressoes(dg.marcadores_vetoriais(tr, cor, m), ex, legenda)
        assert len(p) == 4, (nome, p)
        linhas[nome] = ey.valor(p[:, 1])
        pres = np.array(PRESSOES)
    grava("fig7_potencia.csv", "p_mbar," + ",".join(f"P_{k}_W" for k in linhas), [np.round(pres)] + list(linhas.values()),
          "Stankov 2020 fig. 7: mean power measured (P_medida) and computed (eps_r = 4.2) for each gamma; "
          "centre of the bounding box of the marker (triangles: error up to 0.015 W)")


def figura12(tr):
    """Mean power versus pressure computed for each eps_r (gamma = 0.02)."""
    m = (373.0, 68.1, 509.6, 196.1)
    ty = principais(tr, m, "y", 11, 3.0)            # 3.0 to 0.5 W every 0.25
    tx = principais(tr, m, "x", 7, 3.0)             # 100 to 700 mbar
    ex, ey = dg.Eixo(tx[0], 100.0, tx[-1], 700.0), dg.Eixo(ty[0], 3.0, ty[-1], 0.5)
    legenda = (373.0, 68.0, 460.0, 125.0)
    linhas = {}
    for nome, cor in (("medida", PRETO), ("eps3.3", VERMELHO), ("eps3.75", AZUL), ("eps4.2", VERDE),
                      ("eps4.65", LARANJA), ("eps5.1", MAGENTA)):
        p = nas_pressoes(dg.marcadores_vetoriais(tr, cor, m), ex, legenda)
        assert len(p) == 4, (nome, p)
        linhas[nome] = ey.valor(p[:, 1])
        pres = np.array(PRESSOES)
    grava("fig12_potencia_eps.csv", "p_mbar," + ",".join(f"P_{k}_W" for k in linhas),
          [np.round(pres)] + list(linhas.values()),
          "Stankov 2020 fig. 12: mean power measured (P_medida) and computed (gamma = 0.02) for each eps_r")


def nas_pressoes(p, ex, legenda):
    """Only the markers at the measured pressures and outside the legend box: the
    letters of the legend are paths too and fall inside the frame."""
    x = ex.valor(p[:, 0])
    p = p[np.min(np.abs(x[:, None] - np.array(PRESSOES)[None, :]), axis=1) < 4.0]
    x0, y0, x1, y1 = legenda
    p = p[~((p[:, 0] > x0) & (p[:, 0] < x1) & (p[:, 1] > y0) & (p[:, 1] < y1))]
    return p[np.argsort(p[:, 0])]


def tabelas():
    """Tables 2 and 3 (computed, 100 mbar and 1.8 kV), copied from the text."""
    grava("tabela2_gama.csv", "gama,Ugap_ruptura_V,Imax_mA,ne_max_1e16m3",
          [[0.005, 0.01, 0.02, 0.04, 0.08], [649.2, 646.6, 608.6, 559.3, 497.1], [164, 163, 156, 145, 129],
           [4.02, 3.75, 3.38, 2.88, 2.26]], "Stankov 2020 table 2: eps_r = 4.2. I_max is ten times the peak of the "
          "model in fig. 3 (~16 mA), probably an error of the article; the breakdown is not the maximum of U_gap of "
          "fig. 4 for gamma = 0.005")
    grava("tabela3_eps.csv", "eps_r,Ugap_ruptura_V,Imax_mA,ne_max_1e16m3",
          [[3.3, 3.75, 4.2, 4.65, 5.1], [645.5, 628.6, 608.6, 583.3, 557.4], [160, 159, 156, 149, 138],
           [3.11, 3.26, 3.38, 3.37, 3.26]], "Stankov 2020 table 3: gamma = 0.02. I_max is ten times the peak of the "
          "model in fig. 8 (~16 mA), probably an error of the article")


def main():
    tr7 = dg.tracos_pdf(PDF, 7)
    figura3(tr7)
    figura4(tr7)
    figura6(dg.tracos_pdf(PDF, 8))
    figura7(dg.tracos_pdf(PDF, 9))
    figura12(dg.tracos_pdf(PDF, 12))
    tabelas()


if __name__ == "__main__":
    main()
