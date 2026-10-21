# SPDX-License-Identifier: GPL-3.0-or-later
"""Extração dos dados de Stankov et al. 2020, PSST 29 125009 (DBD de argônio,
gap de 3 mm, quartzo de 1 mm, 24 kHz). As figuras são vetoriais: as
coordenadas saem exatas do PDF e a calibração usa os traços de escala de cada
painel, escritos abaixo em pt (y para baixo) com o valor de cada um.

Uso: python3 docs/literatura/dados/stankov2020.py  (grava em stankov2020/)
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
AZUL_FIG4 = (0.12, 0.24, 1.0)   # γ = 0,01 na figura 4
AZUL_CLARO = (0.0, 0.62, 1.0)   # tensão aplicada na figura 4


def grava(nome, cabecalho, colunas, nota):
    SAIDA.mkdir(exist_ok=True)
    np.savetxt(SAIDA / nome, np.column_stack(colunas), fmt="%.6g", delimiter=",",
               header=f"{nota}\n{cabecalho}", comments="# ")
    print(f"{nome}: {len(colunas[0])} pontos")


def curva(tr, cor, janela, ex, ey, min_pontos=20):
    c = dg.curva_vetorial(tr, cor, janela, min_pontos)
    return ex.valor(c[:, 0]), ey.valor(c[:, 1])


def principais(tr, moldura, eixo, n, min_comp):
    """Traços de escala principais; confere a quantidade esperada."""
    t = dg.tracos_de_escala(tr, moldura, eixo, min_comp=min_comp)
    assert len(t) == n, (eixo, t)
    return t


def figura3(tr):
    """Corrente medida a 100 mbar e 1,8 kV no regime periódico (painel
    principal e os dois detalhes)."""
    m = (108.33, 64.55, 260.14, 167.68)
    tx = principais(tr, m, "x", 9, 3.5)            # 0, 5, ..., 40 µs
    ty = principais(tr, m, "y", 5, 2.5)            # 0,02 ... -0,02 A
    ex, ey = dg.Eixo(tx[0], 0.0, tx[-1], 40.0), dg.Eixo(ty[0], 0.02, ty[-1], -0.02)
    c = dg.curva_vetorial(tr, PRETO, m, 100)
    # os dois detalhes inseridos ficam dentro da moldura principal
    for x0, y0, x1, y1 in ((169.4, 118.4, 210.07, 156.5), (214.89, 66.96, 255.53, 105.02)):
        c = c[~((c[:, 0] >= x0 - 1) & (c[:, 0] <= x1 + 1) & (c[:, 1] >= y0 - 1) & (c[:, 1] <= y1 + 1))]
    t, i = ex.valor(c[:, 0]), ey.valor(c[:, 1])
    grava("fig3_corrente_medida_100mbar.csv", "t_us,I_A", [t, i],
          "Stankov 2020 fig. 3: corrente medida, 100 mbar, U0 = 1,8 kV, 24 kHz, t = 0 no início de um período periódico")
    # detalhe do pico (inserção de baixo): traços em 17,5 a 20 µs de 0,5 em 0,5 e em 0 e 0,010 A
    m2 = (169.4, 118.4, 210.07, 156.5)
    ex2, ey2 = dg.Eixo(172.94, 17.5, 206.71, 20.0), dg.Eixo(156.5, 0.0, 135.34, 0.010)
    t2, i2 = curva(tr, PRETO, m2, ex2, ey2, 100)
    grava("fig3_pico_medido_100mbar.csv", "t_us,I_A", [t2, i2], "Stankov 2020 fig. 3, detalhe do pico principal")
    # detalhe da corcova (inserção de cima): 20 a 28 µs; 0,006 a 0,001 A
    m3 = (214.89, 66.96, 255.53, 105.02)
    ex3, ey3 = dg.Eixo(221.22, 20.0, 250.77, 28.0), dg.Eixo(66.96, 0.006, 105.02, 0.001)
    t3, i3 = curva(tr, PRETO, (m3[0] + 1, m3[1], m3[2], m3[3]), ex3, ey3, 100)
    grava("fig3_corcova_medida_100mbar.csv", "t_us,I_A", [t3, i3], "Stankov 2020 fig. 3, detalhe da corcova depois do pico")
    return ex, ey


def figura4(tr):
    """Calculado por eles a 100 mbar e 1,8 kV para cada γ: tensão no gap,
    tensão aplicada e n_e médio no gap."""
    # traços principais do eixo de tempo em 369,6 (0 µs) e 480,86 (40 µs); da
    # tensão no gap em 108,3 (0 V) e 66,09 (800 V); da tensão aplicada (eixo da
    # direita, em azul claro) em 108,3 (0 V) e 70,8 (2000 V); de n_e em 256,03
    # (1e13 m-3) e 185,11 (1e16 m-3)
    ex = dg.Eixo(369.6, 0.0, 480.86, 40.0)
    ma = (369.59, 66.09, 485.07, 150.39)
    ey_gap = dg.Eixo(108.3, 0.0, 66.09, 800.0)
    ey_ua = dg.Eixo(108.3, 0.0, 70.8, 2000.0)
    t, ua = curva(tr, AZUL_CLARO, ma, ex, ey_ua)
    grava("fig4_tensao_aplicada.csv", "t_us,Ua_V", [t, ua], "Stankov 2020 fig. 4(a): tensão aplicada (eixo da direita)")
    mb = (369.64, 169.72, 485.11, 256.03)
    ey_ne = dg.Eixo(256.03, 1e13, 185.11, 1e16, log=True)
    for g, cor in (("0.005", VERMELHO), ("0.01", AZUL_FIG4), ("0.02", VERDE), ("0.04", LARANJA), ("0.08", MAGENTA)):
        t, u = curva(tr, cor, ma, ex, ey_gap)
        grava(f"fig4_ugap_modelo_gama{g}.csv", "t_us,Ugap_V", [t, u], f"Stankov 2020 fig. 4(a): U_gap calculado, γ = {g}")
    for g, cor in (("0.005", VERMELHO), ("0.01", AZUL), ("0.02", VERDE), ("0.04", LARANJA), ("0.08", MAGENTA)):
        t, n = curva(tr, cor, mb, ex, ey_ne)
        grava(f"fig4_ne_medio_modelo_gama{g}.csv", "t_us,ne_m3", [t, n],
              f"Stankov 2020 fig. 4(b): n_e médio no gap calculado, γ = {g}"
              + ("; a curva verde termina em 40,0 µs no PDF" if g == "0.02" else ""))


def figura6(tr):
    """Corrente medida a 650 mbar e 3,4 kV, dez períodos (primeiro painel)."""
    m = (373.21, 194.18, 513.43, 256.7)
    ey = dg.Eixo(199.95, 0.10, 256.7, -0.10)
    tx = dg.tracos_de_escala(tr, m, "x", min_comp=3.0)
    # 0, T, ..., 10T: o primeiro traço é o eixo em x = 373,21
    assert len(tx) == 11, tx
    ex = dg.Eixo(tx[0], 0.0, tx[-1], 10 * PERIODO_US)
    t, i = curva(tr, PRETO, (m[0] - 0.1, m[1], m[2] + 0.1, m[3]), ex, ey, 100)
    t, i = t[t >= 0.0], i[t >= 0.0]
    grava("fig6_corrente_medida_650mbar.csv", "t_us,I_A", [t, i],
          "Stankov 2020 fig. 6: corrente medida, 650 mbar, U0 = 3,4 kV, dez períodos de 41,67 µs")


def figura7(tr):
    """Potência média contra a pressão: medida e calculada para cada γ."""
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
          "Stankov 2020 fig. 7: potência média medida e calculada (ε_r = 4,2) para cada γ; "
          "centro da caixa do marcador (triângulos: erro de até 0,015 W)")


def figura12(tr):
    """Potência média contra a pressão calculada para cada ε_r (γ = 0,02)."""
    m = (373.0, 68.1, 509.6, 196.1)
    ty = principais(tr, m, "y", 11, 3.0)            # 3,0 a 0,5 W de 0,25 em 0,25
    tx = principais(tr, m, "x", 7, 3.0)             # 100 a 700 mbar
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
          "Stankov 2020 fig. 12: potência média medida e calculada (γ = 0,02) para cada ε_r")


def nas_pressoes(p, ex, legenda):
    """Só os marcadores nas pressões medidas e fora da caixa da legenda: as
    letras da legenda também são caminhos e caem dentro da moldura."""
    x = ex.valor(p[:, 0])
    p = p[np.min(np.abs(x[:, None] - np.array(PRESSOES)[None, :]), axis=1) < 4.0]
    x0, y0, x1, y1 = legenda
    p = p[~((p[:, 0] > x0) & (p[:, 0] < x1) & (p[:, 1] > y0) & (p[:, 1] < y1))]
    return p[np.argsort(p[:, 0])]


def tabelas():
    """Tabelas 2 e 3 (calculadas, 100 mbar e 1,8 kV), copiadas do texto."""
    grava("tabela2_gama.csv", "gama,Ugap_ruptura_V,Imax_mA,ne_max_1e16m3",
          [[0.005, 0.01, 0.02, 0.04, 0.08], [649.2, 646.6, 608.6, 559.3, 497.1], [164, 163, 156, 145, 129],
           [4.02, 3.75, 3.38, 2.88, 2.26]], "Stankov 2020 tabela 2: ε_r = 4,2. I_max é dez vezes o pico do modelo "
          "na fig. 3 (~16 mA), provável erro do artigo; a ruptura não é o máximo de U_gap da fig. 4 para γ = 0,005")
    grava("tabela3_eps.csv", "eps_r,Ugap_ruptura_V,Imax_mA,ne_max_1e16m3",
          [[3.3, 3.75, 4.2, 4.65, 5.1], [645.5, 628.6, 608.6, 583.3, 557.4], [160, 159, 156, 149, 138],
           [3.11, 3.26, 3.38, 3.37, 3.26]], "Stankov 2020 tabela 3: γ = 0,02. I_max é dez vezes o pico do modelo "
          "na fig. 8 (~16 mA), provável erro do artigo")


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
