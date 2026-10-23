# SPDX-License-Identifier: GPL-3.0-or-later
"""Numerical convergence of the 0D positive column (dbd1d/coluna.py): Tv, [N]/[N2],
N2(B)/N2(C) and the sustaining E/N of some conditions of docs/coluna_positiva_n2
with the default tolerances of the integrator (rtol 1e-6, atol 1 cm-3) and with
tighter ones (rtol 1e-8, atol 1e-3 cm-3); prints the relative differences.

Usage: python3 scripts/coluna_n2_convergencia.py SAIDA.csv
"""
import csv
import pathlib
import sys

import numpy as np

RAIZ = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "dbd1d"))
sys.path.insert(0, str(RAIZ / "scripts"))
from coluna import Coluna0D  # noqa: E402
from coluna_n2 import GAMAS, condicoes, ne_nosso, treanor  # noqa: E402
from coluna_n2_en import CORRENTE_MA  # noqa: E402

TOLERANCIAS = {"padrao": (1e-6, 1.0), "apertada": (1e-8, 1e-3)}
CASOS = ("L20_10mA", "B21_530Pa", "G95_30mA")


class ColunaTol(Coluna0D):
    def __init__(self, *a, tol, **k):
        super().__init__(*a, **k)
        self._tol = tol

    def integra(self, e_n, ne, t_fim, n_saidas=50, y0=None, rtol=None, atol=None):
        return super().integra(e_n, ne, t_fim, n_saidas, y0, *self._tol)


def grandezas(caso, cond, tol):
    c = ColunaTol(RAIZ / "casos" / caso, cond["p"], cond["tg"], cond["r"], gamas=GAMAS, tol=tol)
    ne = ne_nosso(c, cond["ne"], cond["en"]) if cond["mobilidade_levaton"] else cond["ne"]
    _, y = c.integra(cond["en"], ne, cond["t"], n_saidas=3)
    nomes = [s.upper() for s in c.especies]
    n0 = c.densidades(y[-1])[[s.upper() for s in c.q.especies].index("N2")]
    tv = treanor(np.r_[n0, [y[-1, nomes.index(f"N2(V{k})")] for k in range(1, 5)]], cond["tg"])
    return dict(tv=tv, dis=y[-1, nomes.index("N")] / n0, b_c=y[-1, nomes.index("N2(B3)")] / y[-1, nomes.index("N2(C3)")],
                en=c.en_sustentado(CORRENTE_MA[cond["id"]] * 1e-3, cond["t"]))


def main():
    linhas = []
    for cond in (c for c in condicoes() if c["id"] in CASOS):
        for caso in ("n2_amonia", "n2_tese"):
            r = {nome: grandezas(caso, cond, tol) for nome, tol in TOLERANCIAS.items()}
            for g in ("tv", "dis", "b_c", "en"):
                a, b = r["padrao"][g], r["apertada"][g]
                linhas.append(dict(id=cond["id"], quimica=caso, grandeza=g, padrao=a, apertada=b,
                                   dif_rel=abs(a - b) / abs(b)))
                print(f"{cond['id']:10s} {caso:10s} {g:4s} {a:12.5g} {b:12.5g}  {abs(a - b) / abs(b):.1e}", flush=True)
    with open(sys.argv[1], "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(linhas[0]))
        w.writeheader()
        w.writerows(linhas)
    print(f"largest relative difference: {max(l['dif_rel'] for l in linhas):.1e}")


if __name__ == "__main__":
    main()
