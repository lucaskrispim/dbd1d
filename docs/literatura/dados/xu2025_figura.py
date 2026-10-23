# SPDX-License-Identifier: GPL-3.0-or-later
"""Figure for checking the digitization of Xu et al. 2025 (xu2025.py) against
pages 26 and 28 of the PDF: Q-U loops of fig. 1(d-f), N and H of fig. 3(a,b)
and the cell voltage.

Usage: python3 docs/literatura/dados/xu2025_figura.py OUTPUT.png
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

DADOS = Path(__file__).with_name("xu2025")


def le(nome):
    return np.loadtxt(DADOS / nome, delimiter=",")


def main():
    fig, ax = plt.subplots(2, 3, figsize=(13, 7))
    for a, (f, gas) in zip(ax[0], [("fig1d_N2", "N2"), ("fig1e_mistura", "25% N2 75% H2"), ("fig1f_H2", "H2")]):
        for barreira, cor in (("dbd", "#7caee0"), ("fbd", "#e89c94")):
            d = le(f"{f}_{barreira}.csv")
            a.plot(d[:, 0], d[:, 1], ".", ms=1, color=cor, label=barreira.upper())
        a.set(title=f"fig. 1, Q-U, {gas}", xlabel="U_cell (kV)", ylabel="Q (µC/cm²)")
        a.legend()
    for a, esp, gas, topo in ((ax[1, 0], "N", "N2", 1.3e16), (ax[1, 1], "H", "H2", 2.6e17)):
        for rotulo, cor in ((f"dbd_{gas}", "#00476f"), ("dbd_mistura", "#7caee0"),
                            (f"fbd_{gas}", "#870303"), ("fbd_mistura", "#e89c94")):
            d = le(f"fig3_n{esp}_{rotulo}.csv")
            a.plot(d[:, 0], d[:, 1], lw=1, color=cor, label=rotulo)
        a.set(title=f"fig. 3, n_{esp}(t)", xlabel="t (µs)", ylabel=f"n_{esp} (cm⁻³)", ylim=(-0.03 * topo, topo))
        a.legend(fontsize=7)
    for rotulo, cor in (("N_dbd_N2", "#00476f"), ("N_dbd_mistura", "#7caee0")):
        d = le(f"fig3_U_{rotulo}.csv")
        ax[1, 2].plot(d[:, 0], d[:, 1], color=cor, label=rotulo)
    ax[1, 2].set(title="fig. 3, U_cell(t)", xlabel="t (µs)", ylabel="U (kV)")
    ax[1, 2].legend()
    fig.suptitle("Xu et al. 2025, dados digitalizados")
    fig.tight_layout()
    fig.savefig(sys.argv[1], dpi=110)


if __name__ == "__main__":
    main()
