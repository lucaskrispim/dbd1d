# SPDX-License-Identifier: GPL-3.0-or-later
"""Figure for checking the digitization of Levaton et al. 2020 (levaton2020.py)
against pages 12 and 14 of the PDF: figures 3, 4 and 10.

Usage: python3 docs/literatura/dados/levaton2020_figura.py OUTPUT.png
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

DADOS = Path(__file__).with_name("levaton2020")


def le(nome):
    return np.loadtxt(DADOS / nome, delimiter=",")


def main():
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    for k, (pref, rot) in enumerate((("fig3_EN", "E/N (Td)"), ("fig4_ne", "n$_e$ (cm$^{-3}$)"))):
        for pressao, cor, m in ((230, "red", "o"), (1070, "blue" if k == 0 else "green", "v")):
            d = le(f"{pref}_{pressao}Pa.csv")
            ax[k].errorbar(d[:, 0], d[:, 1], yerr=[d[:, 2], d[:, 3]], fmt=m, color=cor, capsize=3,
                           label=f"{pressao} Pa")
        ax[k].set(xlabel="Q (Slm)", ylabel=rot, title=f"fig. {3 + k}, 60 mA")
        ax[k].legend()
    for nome, cor, m in (("N2_B", "k", "s"), ("N2_C", "red", "o"), ("N2p_B", "green", "^")):
        d = le(f"fig10_{nome}_calculado.csv")
        ax[2].loglog(d[:, 0], d[:, 1], color=cor, label=f"{nome}, calculado")
    with open(DADOS / "fig10_medidos.csv") as f:
        linhas = [l.split(",") for l in f if not l.startswith("#")]
    for (nome, t, n, inf, sup), (cor, m) in zip(linhas, (("k", "s"), ("red", "o"), ("green", "^"))):
        t, n, inf, sup = map(float, (t, n, inf, sup))
        ax[2].errorbar([t], [n], yerr=[[n - inf], [sup - n]], fmt=m, color=cor, capsize=3)
    ax[2].set(xlabel="tempo de residência (s)", ylabel="densidade (cm$^{-3}$)", title="fig. 10, 10 mA",
              xlim=(1e-9, 1e-3), ylim=(1e6, 5e13))
    ax[2].legend(fontsize=8)
    fig.suptitle("Levaton, Klein e Amorim 2020, dados digitalizados")
    fig.tight_layout()
    fig.savefig(sys.argv[1], dpi=100)


if __name__ == "__main__":
    main()
