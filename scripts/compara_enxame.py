# SPDX-License-Identifier: GPL-3.0-or-later
"""V4 of validacao/README.md: swarm parameters of BOLSIG+ with the argon cross
sections (tabela_bolsig.dat at the repository root) against the data compiled in figures 7
to 10 of Crispim et al. 2024 (docs/literatura/dados/crispim2024/).

From the table: v_d = muN (E/N), D/mu = DN/muN (V, equal to eV per electron) and
mean energy 3/2 k T_e. For each series, the relative deviation of the table
interpolated in log-log at the E/N of each point, with the median and the
maximum per range of E/N.

Usage: python3 scripts/compara_enxame.py [OUTPUT.png]
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
        "fig7": mun * en * 1e-17 / 1e7,      # v_d = muN (1/(cm V s)) E/N (1e-17 V cm2) in cm/s; 1e5 m/s = 1e7 cm/s
        "fig8": dn / mun,                     # D/mu in V
        "fig9": mun * 100.0,                  # muN in 1/(m V s)
        "fig10b": 1.5 * KB_EV * te,           # mean energy in eV
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
    print(f"{'series':<44s} {'n':>3s} " + " ".join(f"{a:g}-{b:g} Td".rjust(14) for a, b in FAIXAS))
    for nome, fig, x, dev in linhas:
        cel = []
        for a, b in FAIXAS:
            m = (x >= a) & (x < b)
            cel.append(f"{np.median(dev[m]) * 100:+5.0f}/{np.abs(dev[m]).max() * 100:3.0f}%" if m.any() else "")
        print(f"{nome:<44s} {len(x):3d} " + " ".join(c.rjust(14) for c in cel))
    print("each cell: median of the deviation (table/data - 1) and the largest absolute deviation in the range")
    if len(sys.argv) < 2:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rotulos = {"fig7": "v_d (1e5 m/s)", "fig8": "D/mu (eV)", "fig9": "mu N (1/(m V s))", "fig10b": "mean energy (eV)"}
    fig, eixos = plt.subplots(2, 2, figsize=(12, 9))
    for eixo, chave in zip(eixos.flat, ("fig7", "fig8", "fig9", "fig10b")):
        eixo.loglog(en, calc[chave], "k-", lw=1.5, label="BOLSIG+ table, 300 K")
        for arq in sorted(glob.glob(str(DADOS / f"{chave}_*.csv"))):
            d = np.loadtxt(arq, delimiter=",", ndmin=2)
            eixo.loglog(d[:, 0], d[:, 1], "o", ms=3.5, label=pathlib.Path(arq).stem[len(chave) + 1:].replace("_", " "))
        eixo.set_xlabel("E/N (Td)")
        eixo.set_ylabel(rotulos[chave])
        eixo.grid(alpha=0.3, which="both")
        eixo.legend(fontsize=7)
    fig.suptitle("Swarm parameters in argon: BOLSIG+ table against measurements and calculations", fontsize=11)
    fig.tight_layout()
    fig.savefig(sys.argv[1], dpi=110)


if __name__ == "__main__":
    main()
