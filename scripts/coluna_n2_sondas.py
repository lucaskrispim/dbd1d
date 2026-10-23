# SPDX-License-Identifier: GPL-3.0-or-later
"""Sensitivity of the sustaining E/N to where the electron balance is imposed. The
E/N of Levaton et al. is measured between double probes in the middle of the
column, the residence time t of docs/coluna_positiva_n2/condicoes.csv is the one of
the optical measurement near the end; the balance is imposed here at t and at t/2,
for the three chemistries (the conditions of Gordiets et al. are steady and are
left out).

Usage: python3 scripts/coluna_n2_sondas.py SAIDA.csv
"""
import csv
import pathlib
import sys

RAIZ = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "dbd1d"))
sys.path.insert(0, str(RAIZ / "scripts"))
from coluna import Coluna0D  # noqa: E402
from coluna_n2 import GAMAS, QUIMICAS, condicoes  # noqa: E402
from coluna_n2_en import CORRENTE_MA  # noqa: E402


def main():
    linhas = []
    for cond in condicoes():
        if not cond["id"].startswith(("L20", "B21")):
            continue
        for caso in QUIMICAS:
            c = Coluna0D(RAIZ / "casos" / caso, cond["p"], cond["tg"], cond["r"], gamas=GAMAS, estrito=False)
            i_a = CORRENTE_MA[cond["id"]] * 1e-3
            fim, meio = c.en_sustentado(i_a, cond["t"]), c.en_sustentado(i_a, 0.5 * cond["t"])
            linhas.append(dict(id=cond["id"], quimica=caso, t=cond["t"], en_med=cond["en"], en_t=fim, en_t2=meio))
            print(f"{cond['id']:10s} {caso:11s} t {fim:6.1f}  t/2 {meio:6.1f}", flush=True)
    with open(sys.argv[1], "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(linhas[0]))
        w.writeheader()
        w.writerows(linhas)


if __name__ == "__main__":
    main()
