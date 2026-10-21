# SPDX-License-Identifier: GPL-3.0-or-later
"""Extração dos dados de Yang et al. 2022, PSST 31 015017 (DBD de pulsos de
nanossegundos em N2 e N2/H2, 150 Torr, gap de 8 mm, rajadas de 150 pulsos a
100 kHz). As figuras são imagens JPEG de 200 ppi embutidas no PDF; o script as
extrai com pdfimages e lê por cor.

Calibração de cada painel: posições dos traços de escala principais (pixels
da imagem embutida, y para baixo), achadas por
digitaliza.tracos_de_escala_raster, e o valor de cada traço lido no rótulo
da figura. Cada par foi conferido contra a posição do texto do rótulo (a
coluna ou a linha de pixels escuros ao lado do traço), porque traços
secundários do mesmo comprimento enganam a leitura a olho. A incerteza de leitura é de um a dois pixels, da ordem de 0,5 % do
fundo de escala, além da compressão JPEG.

Uso: python3 docs/literatura/dados/yang2022.py  (grava em yang2022/)
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy import ndimage

RAIZ = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(RAIZ / "ferramentas"))
import digitaliza as dg  # noqa: E402

PDF = RAIZ / "docs" / "artigos" / "Yang_2022_Plasma_Sources_Sci._Technol._31_015017.pdf"
SAIDA = Path(__file__).with_name("yang2022")
VERMELHO, AZUL, PRETO = (0.93, 0.08, 0.08), (0.08, 0.08, 0.93), (0.0, 0.0, 0.0)


def grava(nome, cabecalho, colunas, nota):
    if "_min" in cabecalho:
        nota += ("; mín. e máx.: extremos da barra de erro (barras claras incluídas), ou a borda do marcador quando "
                 "não há barra visível; a barra pode se estender alguns pixels onde a curva do modelo, da mesma cor, "
                 "cruza a coluna do ponto")
    SAIDA.mkdir(exist_ok=True)
    np.savetxt(SAIDA / nome, np.column_stack(colunas), fmt="%.5g", delimiter=",",
               header=f"{nota}\n{cabecalho}", comments="# ")
    print(f"{nome}: {len(colunas[0])} pontos")


def imagens():
    """Imagens embutidas, pelo número que o pdfimages -list mostra."""
    d = Path(tempfile.mkdtemp())
    subprocess.run(["pdfimages", "-png", str(PDF), str(d / "img")], check=True)
    return lambda n: dg.le_png(d / f"img-{n:03d}.png")


def curva(img, cor, janela, ex, ey, tol=0.3, excluir=()):
    """Curva da cor dada; as caixas em excluir (x0, y0, x1, y1 em pixels, como
    a legenda) são pintadas de branco antes, para não entrarem na mediana."""
    if excluir:
        img = img.copy()
        for x0, y0, x1, y1 in excluir:
            img[y0:y1, x0:x1] = 1.0
    c = dg.curva_raster(img, cor, janela, tol)
    return ex.valor(c[:, 0]), ey.valor(c[:, 1])


def pontos(img, cor, janela, ex, ey, tol=0.3, abertura=3, alcance=150, min_px=12, separa=False):
    """Marcadores, com a barra de erro vertical (sem barra, os extremos são as
    bordas do marcador)."""
    # separa só onde marcadores se tocam (fig. 3): onde um marcador encosta numa
    # curva de modelo da mesma cor, a mancha alongada seria partida em dois
    c = dg.marcadores_raster(img, cor, janela, tol, min_px=min_px, abertura=abertura, separa=separa)
    c = c[np.argsort(c[:, 0])]
    # as barras são claras (semitransparentes): a cor misturada ao branco também vale
    topo, base = dg.barras_raster(img, cor, c, janela, alcance, tol, alfa_min=0.15)
    return ex.valor(c[:, 0]), ey.valor(c[:, 1]), ey.valor(base), ey.valor(topo)


def envoltoria(img, janela, ex, ey):
    """Envoltória de cima e de baixo da curva medida de N2(A), que oscila a cada
    pulso: pixels avermelhados (o vermelho e a mistura com o azul do modelo por
    cima) por coluna."""
    x0, y0, x1, y1 = [int(v) for v in janela]
    sub = img[y0 + 3:y1 - 2, x0 + 3:x1 - 2]
    m = (sub[..., 0] - sub[..., 2] > 0.25) & (sub[..., 1] < 0.45)
    xs = np.flatnonzero(m.any(axis=0))
    topo = np.array([np.flatnonzero(m[:, x]).min() for x in xs]) + y0 + 3
    base = np.array([np.flatnonzero(m[:, x]).max() for x in xs]) + y0 + 3
    t = ex.valor(xs + x0 + 3)
    return t, ey.valor(topo), ey.valor(base)


def figura2(img):
    """Pulso 50 em N2 a 150 Torr: tensão, corrente, potência e energia acoplada."""
    ex = dg.Eixo(82.0, 0.0, 596.0, 200.0)
    ev = dg.Eixo(342.5, 0.0, 140.5, 30.0)
    ei = dg.Eixo(340.0, 0.0, 153.0, 60.0)
    jan = (57, 106, 595, 511)
    legenda = [(430, 115, 592, 180)]
    t, v = curva(img, PRETO, jan, ex, ev, 0.35, legenda)
    limpa = img.copy()
    limpa[115:180, 430:592] = 1.0
    xp, yp = dg.extremo_raster(limpa, PRETO, jan, "max", 0.35)
    grava("fig2_tensao_pulso50.csv", "t_ns,V_kV", [t, v],
          "Yang 2022 fig. 2(a): tensão medida no pulso 50 de uma rajada de 150, N2, 150 Torr, 100 kHz; "
          f"mediana por coluna, que corta o pico estreito: pico pela borda da linha {ev.valor(yp):.1f} kV "
          f"em {ex.valor(xp):.1f} ns")
    t, i = curva(img, VERMELHO, (jan[0], jan[1], jan[2] - 4, jan[3]), ex, ei, 0.35, legenda)
    grava("fig2_corrente_pulso50.csv", "t_ns,I_A", [t, i], "Yang 2022 fig. 2(a): corrente medida no pulso 50")
    ex2 = dg.Eixo(836.0, 0.0, 1350.0, 200.0)
    ep = dg.Eixo(387.5, 0.0, 237.0, 1.0)
    ee = dg.Eixo(387.5, 0.0, 117.5, 8.0)
    jan2 = (814, 118, 1346, 522)
    legenda2 = [(1085, 125, 1345, 190)]
    t, p = curva(img, VERMELHO, jan2, ex2, ep, 0.35, legenda2)
    grava("fig2_potencia_pulso50.csv", "t_ns,P_MW", [t, p],
          "Yang 2022 fig. 2(b): potência instantânea no pulso 50; só a forma: a integral desta curva dá 3,7 mJ "
          "em 200 ns, 11 % acima da curva de energia (3,3 mJ, o valor do texto)")
    t, e = curva(img, AZUL, jan2, ex2, ee, 0.35, legenda2)
    grava("fig2_energia_pulso50.csv", "t_ns,E_mJ", [t, e],
          "Yang 2022 fig. 2(b): energia acoplada acumulada no pulso 50 (3,3 mJ no fim, segundo o texto)")


def figura3(img):
    """Energia acoplada por pulso ao longo da rajada."""
    ex = dg.Eixo(88.0, 0.0, 640.5, 150.0)
    ey = dg.Eixo(471.5, 0.0, 33.5, 4.0)
    jan = (90, 30, 659, 470)
    for nome, cor, desc in (("total", VERMELHO, "incluindo os pulsos refletidos, medida em t = 200 ns"),
                            ("incidente", AZUL, "só o pulso incidente, medida em t = 30 ns")):
        x, y, lo, hi = pontos(img, cor, jan, ex, ey, 0.35, separa=True)
        ok = ~((x > ex.valor(420)) & (y < ey.valor(380)))   # legenda
        grava(f"fig3_energia_{nome}.csv", "pulso,E_mJ,E_min_mJ,E_max_mJ", [np.round(x[ok]), y[ok], lo[ok], hi[ok]],
              f"Yang 2022 fig. 3: energia acoplada por pulso, {desc}; N2, 150 Torr; barras: desvio-padrão em 100 rajadas")


def figura7(img):
    """N2(A, v=0) e N2(A, v=1) medidos em N2: envoltória do sinal que oscila a
    cada pulso."""
    for nome, jan, ex, ey in (
            ("v0", (108, 141, 610, 560), dg.Eixo(121.5, 0.0, 596.5, 1750.0), dg.Eixo(541.5, 0.0, 141.5, 5e13)),
            ("v1", (828, 145, 1329, 564), dg.Eixo(841.5, 0.0, 1315.5, 1750.0), dg.Eixo(554.0, 0.0, 145.5, 1e14))):
        t, topo, base = envoltoria(img, jan, ex, ey)
        ok = ~((t > ex.valor(jan[0] + 290)) & (topo > ey.valor(jan[1] + 80)))   # legenda
        grava(f"fig7_n2a_{nome}_N2.csv", "t_us,max_cm3,min_cm3", [t[ok], topo[ok], base[ok]],
              f"Yang 2022 fig. 7: N2(A, {nome}) medido por TDLAS, rajada de 150 pulsos em N2, 150 Torr; "
              "envoltória de cima e de baixo (o sinal oscila a cada pulso de 10 µs); a de cima é aproximada onde "
              "o modelo, desenhado por cima em azul semitransparente, encobre o topo da curva medida")


def figura8(img):
    """Temperatura rotacional e translacional ao longo da rajada em N2."""
    ex = dg.Eixo(865.0, 0.0, 1387.5, 1600.0)
    ey = dg.Eixo(549.5, 300.0, 140.0, 450.0)
    jan = (868, 139, 1388, 574)
    # os pontos medidos se encostam numa faixa contínua: depois de apagar as
    # barras de erro (abertura 3 × 3), a mediana por coluna, de 10 em 10 µs
    m = ndimage.binary_opening(dg.mascara(img, PRETO, 0.35, jan), np.ones((3, 3), dtype=bool))
    m[380:, 1150:] = False                                 # legenda (canto de baixo à direita)
    xs = np.flatnonzero(m.any(axis=0))
    ys = np.array([np.median(np.flatnonzero(m[:, x])) for x in xs])
    t, temp = ex.valor(xs), ey.valor(ys)
    grade = np.arange(np.ceil(t.min() / 10) * 10, t.max(), 10.0)
    grava("fig8_temperatura_N2.csv", "t_us,T_K", [grade, np.interp(grade, t, temp)],
          "Yang 2022 fig. 8(b): temperatura rotacional e translacional medida, N2, 150 Torr; pontos encostados "
          "lidos como curva (mediana por coluna) e amostrados de 10 em 10 µs; incerteza de 10 K segundo o texto")
    t, tm = curva(img, AZUL, jan, ex, ey, 0.35)
    ok = ~((t > ex.valor(1150)) & (tm < ey.valor(470)))
    grava("fig8_temperatura_modelo_N2.csv", "t_us,T_K", [t[ok], tm[ok]], "Yang 2022 fig. 8(b): modelo deles")


def figura9(img):
    """N acumulado e N produzido por pulso em N2."""
    ex = dg.Eixo(125.0, 0.0, 610.5, 150.0)
    e_por = dg.Eixo(411.0, 0.0, 12.0, 7e13)
    e_acum = dg.Eixo(416.0, 0.0, 12.0, 4.5e15)
    jan = (127, 12, 614, 434)
    x, y, lo, hi = pontos(img, AZUL, jan, ex, e_acum, 0.35)
    grava("fig9_N_acumulado_N2.csv", "pulso,N_cm3,N_min_cm3,N_max_cm3", [np.round(x), y, lo, hi],
          "Yang 2022 fig. 9: N(4S) medido por TALIF 100 µs depois de cada pulso, rajada em N2, 150 Torr")
    x, y, lo, hi = pontos(img, VERMELHO, jan, ex, e_por, 0.35)
    grava("fig9_N_por_pulso_N2.csv", "pulso,dN_cm3,dN_min_cm3,dN_max_cm3", [np.round(x), y, lo, hi],
          "Yang 2022 fig. 9: N produzido por pulso, diferença entre medidas a 10 pulsos de distância")


def figura11(img):
    """N2(A, v=0 e 1) medidos em N2 com 1 % de H2."""
    for nome, jan, ex in (("v0", (110, 139, 612, 560), dg.Eixo(122.5, 0.0, 597.5, 1750.0)),
                          ("v1", (828, 140, 1330, 559), dg.Eixo(841.0, 0.0, 1316.5, 1750.0))):
        ey = dg.Eixo(544.5 if nome == "v0" else 545.0, 0.0, 140.0 if nome == "v0" else 140.5, 7e13)
        t, topo, base = envoltoria(img, jan, ex, ey)
        ok = ~((t > ex.valor(jan[0] + 290)) & (topo > ey.valor(jan[1] + 80)))
        grava(f"fig11_n2a_{nome}_1pcH2.csv", "t_us,max_cm3,min_cm3", [t[ok], topo[ok], base[ok]],
              f"Yang 2022 fig. 11: N2(A, {nome}) medido, rajada de 150 pulsos em N2 com 1 % de H2, 150 Torr; envoltória; "
              "a de cima é aproximada onde o modelo encobre a curva medida")


def figura14(img):
    """N acumulado, N por pulso e H acumulado em N2 com 1 % de H2."""
    ex = dg.Eixo(139.5, 0.0, 624.5, 150.0)
    e_por = dg.Eixo(498.5, 0.0, 88.0, 2e13)
    e_acum = dg.Eixo(500.5, 0.0, 87.5, 1.2e15)
    jan = (142, 88, 628, 508)
    x, y, lo, hi = pontos(img, AZUL, jan, ex, e_acum, 0.35)
    ok = ~((x > ex.valor(380)) & (x < ex.valor(620)) & (y > e_acum.valor(330)) & (y < e_acum.valor(260)))
    grava("fig14_N_acumulado_1pcH2.csv", "pulso,N_cm3,N_min_cm3,N_max_cm3", [np.round(x[ok]), y[ok], lo[ok], hi[ok]],
          "Yang 2022 fig. 14(a): N medido 100 µs depois de cada pulso, N2 com 1 % de H2, 150 Torr")
    x, y, lo, hi = pontos(img, VERMELHO, jan, ex, e_por, 0.35, alcance=120)
    ok = ~((x > ex.valor(380)) & (x < ex.valor(620)) & (y > e_por.valor(330)) & (y < e_por.valor(260)))
    grava("fig14_N_por_pulso_1pcH2.csv", "pulso,dN_cm3,dN_min_cm3,dN_max_cm3", [np.round(x[ok]), y[ok], lo[ok], hi[ok]],
          "Yang 2022 fig. 14(a): N produzido por pulso (barras grandes: diferença de medidas próximas); "
          "o ponto do pulso 110, sob a curva do modelo, não é separado")
    ex2 = dg.Eixo(944.5, 0.0, 1312.5, 1500.0)
    ey2 = dg.Eixo(488.5, 0.0, 116.5, 1e16)
    jan2 = (935, 116, 1386, 487)
    x, y, lo, hi = pontos(img, VERMELHO, jan2, ex2, ey2, 0.35)
    ok = ~((x > ex2.valor(1240)) & (y < ey2.valor(380)) & (y > ey2.valor(470)))
    grava("fig14_H_acumulado_1pcH2.csv", "t_us,H_cm3,H_min_cm3,H_max_cm3", [x[ok], y[ok], lo[ok], hi[ok]],
          "Yang 2022 fig. 14(b): H medido por TALIF durante a rajada, N2 com 1 % de H2, 150 Torr")


def figura16(img):
    """N e H 100 µs depois de uma rajada de 50 pulsos contra a fração de H2."""
    ex = dg.Eixo(96.5, 0.0, 513.5, 35.0)
    ey = dg.Eixo(401.5, 1e14, 45.5, 1e16, log=True)
    jan = (90, 45, 514, 400)
    for esp, cor in (("N", VERMELHO), ("H", AZUL)):
        # medidos: círculos cheios; os triângulos vazados do modelo somem na abertura
        x, y, lo, hi = pontos(img, cor, jan, ex, ey, 0.35, abertura=6, alcance=15)
        # frações da figura: 0, 1, 5, 9, 17 e 33 %; o marcador em 0 fica cortado
        # pelo eixo e o centroide sai deslocado (0,14 %)
        x = np.array([min((0.0, 1.0, 5.0, 9.0, 17.0, 33.0), key=lambda f: abs(f - v)) for v in x])
        grava(f"fig16_{esp}_contra_H2.csv", "H2_pct,dens_cm3,dens_min_cm3,dens_max_cm3", [x, y, lo, hi],
              f"Yang 2022 fig. 16: {esp} medido 100 µs depois de rajada de 50 pulsos, 150 Torr")


def main():
    img = imagens()
    figura2(img(14))
    figura3(img(15))
    figura7(img(23))
    figura8(img(24))
    figura9(img(26))
    figura11(img(29))
    figura14(img(34))
    figura16(img(37))


if __name__ == "__main__":
    main()
