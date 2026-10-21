# SPDX-License-Identifier: GPL-3.0-or-later
"""V4 de validacao/LEIAME.md: parâmetros de enxame do BOLSIG+ com o bolsigdb.dat
do argônio (tabela_bolsig.dat da raiz) contra os dados reunidos nas figuras 7 a
10 de Crispim et al. 2024 (docs/literatura/dados/crispim2024/).

Da tabela: v_d = μN · (E/N), D/μ = DN/μN (V, igual a eV por elétron) e energia
média 3/2 k T_e. Para cada série, o desvio relativo da tabela interpolada em
log-log no E/N de cada ponto, com a mediana e o máximo por faixa de E/N.

Uso: python3 scripts/compara_enxame.py [SAIDA.png]
"""
import glob
import pathlib
import sys

import numpy as np

RAIZ = pathlib.Path(__file__).resolve().parent.parent
DADOS = RAIZ / "docs" / "literatura" / "dados" / "crispim2024"
KB_EV = 8.617333262e-5
FAIXAS = [(0.0, 0.01), (0.01, 1.0), (1.0, 10.0), (10.0, 100.0), (100.0, 1e4)]


def tabela():
    t = np.loadtxt(RAIZ / "tabela_bolsig.dat", comments="#")
    en, te, mun, dn = t[:, 0], t[:, 1], t[:, 2], t[:, 3]
    return en, {
        "fig7": mun * en * 1e-17 / 1e7,      # v_d = μN (1/(cm V s)) · E/N (1e-17 V cm²) em cm/s; 1e5 m/s = 1e7 cm/s
        "fig8": dn / mun,                     # D/μ em V
        "fig9": mun * 100.0,                  # μN em 1/(m V s)
        "fig10b": 1.5 * KB_EV * te,           # energia média em eV
    }


def compara():
    en, calc = tabela()
    linhas = []
    for arq in sorted(glob.glob(str(DADOS / "*.csv"))):
        nome = pathlib.Path(arq).stem
        fig = nome.split("_")[0]
        d = np.loadtxt(arq, delimiter=",", ndmin=2)
        x, y = d[:, 0], d[:, 1]
        dentro = (x >= en.min()) & (x <= en.max())
        modelo = np.exp(np.interp(np.log(x[dentro]), np.log(en), np.log(calc[fig])))
        desvio = modelo / y[dentro] - 1.0
        linhas.append((nome, fig, x[dentro], desvio))
    return en, calc, linhas


def main():
    en, calc, linhas = compara()
    print(f"{'série':<44s} {'n':>3s} " + " ".join(f"{a:g}-{b:g} Td".rjust(14) for a, b in FAIXAS))
    for nome, fig, x, dev in linhas:
        cel = []
        for a, b in FAIXAS:
            m = (x >= a) & (x < b)
            cel.append(f"{np.median(dev[m]) * 100:+5.0f}/{np.abs(dev[m]).max() * 100:3.0f}%" if m.any() else "")
        print(f"{nome:<44s} {len(x):3d} " + " ".join(c.rjust(14) for c in cel))
    print("cada célula: mediana do desvio (tabela/dado - 1) e o maior desvio absoluto na faixa")
    if len(sys.argv) < 2:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rotulos = {"fig7": "v_d (1e5 m/s)", "fig8": "D/μ (eV)", "fig9": "μN (1/(m V s))", "fig10b": "⟨ε⟩ (eV)"}
    fig, eixos = plt.subplots(2, 2, figsize=(12, 9))
    for eixo, chave in zip(eixos.flat, ("fig7", "fig8", "fig9", "fig10b")):
        eixo.loglog(en, calc[chave], "k-", lw=1.5, label="BOLSIG+ (bolsigdb.dat do projeto), 300 K")
        for arq in sorted(glob.glob(str(DADOS / f"{chave}_*.csv"))):
            d = np.loadtxt(arq, delimiter=",", ndmin=2)
            eixo.loglog(d[:, 0], d[:, 1], "o", ms=3.5, label=pathlib.Path(arq).stem[len(chave) + 1:].replace("_", " "))
        eixo.set_xlabel("E/N (Td)")
        eixo.set_ylabel(rotulos[chave])
        eixo.grid(alpha=0.3, which="both")
        eixo.legend(fontsize=7)
    fig.suptitle("Parâmetros de enxame em argônio: tabela do BOLSIG+ contra medidas e cálculos da literatura", fontsize=11)
    fig.tight_layout()
    fig.savefig(sys.argv[1], dpi=110)


if __name__ == "__main__":
    main()
