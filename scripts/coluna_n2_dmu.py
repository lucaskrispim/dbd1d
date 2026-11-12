# SPDX-License-Identifier: GPL-3.0-or-later
"""Sensitivity of the sustaining E/N of chemistry A to the electron temperature in the ambipolar
loss: (2/3)<eps> of the tables (the model) against kTe/e = D_e/mu_e of the tables, in the eight
conditions (docs/coluna_positiva_n2/falta/dmu.csv).

Usage: python3 scripts/coluna_n2_dmu.py
"""
import csv
import pathlib
import sys

import numpy as np

RAIZ = pathlib.Path(__file__).resolve().parent.parent
sys.path[:0] = [str(RAIZ / "dbd1d"), str(RAIZ / "scripts")]
import coluna  # noqa: E402
import coluna_n2_en as ce  # noqa: E402
sys.path.insert(0, str(RAIZ / "ferramentas"))
import proveniencia  # noqa: E402
from coluna_n2 import condicoes  # noqa: E402

KB, QE = 1.380649e-23, 1.602176634e-19
ORIGINAL = coluna.Coluna0D.frequencias_de_perda


def com_dmu(self, e_n, te):
    v = self.tab(np.array([e_n]), np.array([self.t_gas]))
    return ORIGINAL(self, e_n, v[0, 2] / v[0, 1] * QE / KB)


def main():
    linhas = []
    for cond in condicoes():
        coluna.Coluna0D.frequencias_de_perda = ORIGINAL
        base = ce.en("n2_amonia", cond)
        coluna.Coluna0D.frequencias_de_perda = com_dmu
        dmu = ce.en("n2_amonia", cond)
        coluna.Coluna0D.frequencias_de_perda = ORIGINAL
        linhas.append(dict(id=cond["id"], en_base=base, en_dmu=dmu, diferenca=dmu - base, relativa=dmu / base - 1))
        print(f"{cond['id']:10s} {base:7.2f} {dmu:7.2f} Td  {dmu - base:+.2f} ({100 * (dmu / base - 1):+.1f}%)", flush=True)
    with open(RAIZ / "docs" / "coluna_positiva_n2" / "falta" / "dmu.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(linhas[0]))
        w.writeheader()
        w.writerows(linhas)
    proveniencia.grava_origem(RAIZ / "docs" / "coluna_positiva_n2" / "falta" / "dmu.csv", __file__, ['n2_amonia'])  # code and inputs beside the CSV


if __name__ == "__main__":
    main()
