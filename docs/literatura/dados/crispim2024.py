# SPDX-License-Identifier: GPL-3.0-or-later
"""Extração dos dados de comparação das figuras 7 a 10 de Crispim et al. 2024,
Phys. Scr. 99 065521: parâmetros de enxame de elétrons em argônio contra E/N,
medidos ou calculados por outros autores (cada série com a fonte original
citada na legenda). Figuras vetoriais do gnuplot, páginas 12 e 13 do PDF.

Os eixos são logarítmicos. A calibração usa dois traços de década de cada eixo
(pt), achados pelo padrão dos traços secundários (o do 2 fica a log10(2) de
década do traço da década anterior). A moldura nem sempre cai numa década: na
fig. 8 o 1e1 está 5,4 pt abaixo do topo, na fig. 9 o 1e3 está 4,9 pt antes da
borda direita.

Uso: python3 docs/literatura/dados/crispim2024.py  (grava em crispim2024/)
"""
import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(RAIZ / "ferramentas"))
import digitaliza as dg  # noqa: E402

PDF = RAIZ / "docs" / "artigos" / "Crispim_2024_Phys._Scr._99_065521.pdf"
SAIDA = Path(__file__).with_name("crispim2024")

# figura: página, moldura (pt), (c, valor) de dois traços de década em x e em y, caixa da legenda, grandeza
FIGURAS = {
    "fig7": (12, (221.6, 77.6, 481.7, 266.3), ((221.6, 1e-4), (481.7, 1e3)), ((266.3, 1e-4), (77.6, 1e1)),
             (440.0, 195.0, 482.0, 262.0), "velocidade de deriva dos elétrons, 1e5 m/s"),
    "fig8": (12, (219.5, 377.4, 481.6, 567.5), ((219.5, 1e-5), (481.6, 1e3)), ((567.5, 1e-4), (382.8, 1e1)),
             (440.0, 500.0, 482.0, 567.0), "D_e/μ_e, eV"),
    "fig9": (13, (222.2, 72.8, 486.5, 264.5), ((222.2, 1e-3), (481.6, 1e3)), ((264.5, 1e23), (99.1, 1e26)),
             (330.0, 75.0, 487.0, 135.0), "μ_e N, (m V s)^-1"),
    "fig10b": (13, (367.1, 365.5, 537.6, 489.2), ((367.1, 1e-1), (537.6, 1e3)), ((489.2, 1e-1), (370.2, 1e1)),
               (440.0, 455.0, 538.0, 490.0), "energia média dos elétrons, eV"),
}
# série: figura, cor, 'traco' ou 'preenchimento', número de pontos do subcaminho, fonte
SERIES = [
    ("fig7_pack_phelps_1961_medida", "fig7", (1.0, 0.08, 0.58), "traco", 2, "J L Pack e A V Phelps 1961, medida"),
    ("fig7_ellis_1976_medida", "fig7", (0.0, 0.0, 0.0), "traco_x", 2, "H W Ellis 1976, medida"),
    ("fig7_kucukarpaci_lucas_1981_medida", "fig7", (1.0, 0.55, 0.0), "preenchimento", 5,
     "H N Kucukarpaci e J Lucas 1981, medida"),
    ("fig7_nielsen_1936_medida", "fig7", (0.42, 0.35, 0.8), "preenchimento", 4, "R A Nielsen 1936, medida"),
    ("fig8_warren_parker_1962_77K_medida", "fig8", (0.0, 0.29, 0.02), "preenchimento", 5,
     "Warren e Parker 1962, 77 K, medida"),
    ("fig8_townsend_bailey_1922_288K_medida", "fig8", (0.0, 1.0, 0.0), "preenchimento", 5,
     "Townsend e Bailey 1922, 288 K, medida"),
    ("fig8_townsend_bailey_1922_77K_medida", "fig8", (0.29, 0.0, 0.51), "preenchimento", 5,
     "Townsend e Bailey 1922, 77 K, medida"),
    ("fig9_kucukarpaci_lucas_1981_medida", "fig9", (0.0, 1.0, 0.87), "preenchimento", 5,
     "H N Kucukarpaci e J Lucas 1981, medida"),
    ("fig9_robertson_1977_medida", "fig9", (0.0, 0.0, 0.0), "preenchimento_losango", 5, "A G Robertson et al. 1977, medida"),
    ("fig9_pack_1992_medida", "fig9", (0.0, 0.04, 1.0), "traco", 2, "Pack et al. 1992, medida"),
    ("fig9_errett_1951_medida", "fig9", (0.65, 0.04, 0.61), "traco", 2, "Errett 1951, medida"),
    ("fig10b_escada_2007_calculada", "fig10b", (0.0, 0.75, 1.0), "preenchimento", 5, "J Escada et al. 2007, calculada"),
    ("fig10b_benhenni_2005_calculada", "fig10b", (0.0, 1.0, 0.0), "preenchimento", 5, "M Benhenni et al. 2005, calculada"),
]


def centros(tr, moldura, legenda, cor, tipo, n):
    """Centros dos marcadores de uma série: subcaminhos pequenos da cor, do
    tipo e com o número de pontos dados; marcadores feitos de vários traços
    (× e *) contam uma vez; os da legenda ficam de fora."""
    out = []
    for t in tr:
        p = t["p"]
        tam = p.max(0) - p.min(0)
        if len(p) != n or tam.max() > 6.0 or tam.max() < 1.0:
            continue
        if tipo.startswith("traco"):
            if not dg._mesma_cor(t["traco"], cor, 0.02):
                continue
            # o × do Ellis em preto: só as diagonais (os traços de escala são horizontais ou verticais)
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
                   header=f"Crispim 2024 {fig}: {grandeza} em argônio; {fonte} (como reproduzida na figura)\n"
                          "EN_Td,valor")
        print(f"{nome}: {len(c)} pontos, E/N de {en.min():.3g} a {en.max():.3g} Td")


if __name__ == "__main__":
    main()
