# SPDX-License-Identifier: GPL-3.0-or-later
"""Figure for checking the digitization of Levaton et al. 2021 (levaton2021.py)
against pages 3 and 4 of the PDF: figures 2 and 3.

Usage: python3 docs/literatura/dados/levaton2021_figura.py OUTPUT.png
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

DADOS = Path(__file__).with_name("levaton2021")
ESTILO = {"N2_B": ("k", "-", "s"), "N2_C": ("red", "--", "o"), "N2p_B": ("green", ":", "^")}


def main():
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for a, (nome, titulo) in zip(ax, (("fig2", "fig. 2, 60 mA, 230 Pa"), ("fig3", "fig. 3, 60 mA, 530 Pa"))):
        for estado, (cor, ls, m) in ESTILO.items():
            d = np.loadtxt(DADOS / f"{nome}_{estado}_calculado.csv", delimiter=",")
            a.loglog(d[:, 0], d[:, 1], ls=ls, color=cor, label=f"{estado}, calculado")
        with open(DADOS / f"{nome}_medidos.csv") as f:
            for linha in (l.split(",") for l in f if not l.startswith("#")):
                cor, _, m = ESTILO[linha[0]]
                t, n, inf, sup = map(float, linha[1:])
                a.errorbar([t], [n], yerr=[[n - inf], [sup - n]], fmt=m, color=cor, capsize=3)
        a.set(xlim=(1e-9, 1e-1), ylim=(1e5, 5e12), xlabel="tempo de residência (s)",
              ylabel="densidade (cm$^{-3}$)", title=titulo)
        a.legend(fontsize=8)
    fig.suptitle("Levaton, Klein e Amorim 2021, dados digitalizados")
    fig.tight_layout()
    fig.savefig(sys.argv[1], dpi=100)


if __name__ == "__main__":
    main()
