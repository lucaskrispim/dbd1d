# SPDX-License-Identifier: GPL-3.0-or-later
"""Sensitivity of the N2 positive column to the wall probabilities
(docs/coluna_positiva_n2/README.md): Tv, [N]/[N2] and the sustaining
E/N of all conditions, with the chemistries of the ammonia and of the thesis,
for the base probabilities of scripts/coluna_n2.py and with one group changed at
a time: N(4S) x10 and /10, N2(v) x10 and /10, the electronic metastables (N2(A),
N2(a'), N(2D), N(2P)) at 0.1 and 0.01 instead of 1.

Usage: python3 scripts/coluna_n2_parede.py SAIDA_DIR
Writes parede.csv.
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

VIB = [f"N2(V{k})" for k in range(1, 9)]
METAESTAVEIS = ["N2(A3)", "N2(a'1)", "N2(a`1)", "N(2D)", "N(2P)"]
GAMA_VIB = 4.5e-4


def variantes():
    v = {"base": dict(GAMAS)}
    for nome, f in (("N_x10", 10.0), ("N_div10", 0.1)):
        g = dict(GAMAS)
        g["N"] = GAMAS["N"] * f
        v[nome] = g
    for nome, f in (("vib_x10", 10.0), ("vib_div10", 0.1)):
        g = dict(GAMAS)
        g.update({s: GAMA_VIB * f for s in VIB})
        v[nome] = g
    for nome, gm in (("meta_0.1", 0.1), ("meta_0.01", 0.01)):
        g = dict(GAMAS)
        g.update({s: gm for s in METAESTAVEIS})
        v[nome] = g
    return v


def grandezas(caso, cond, gamas):
    c = Coluna0D(RAIZ / "casos" / caso, cond["p"], cond["tg"], cond["r"], gamas=gamas, estrito=False)
    ne = ne_nosso(c, cond["ne"], cond["en"]) if cond["mobilidade_levaton"] else cond["ne"]
    _, y = c.integra(cond["en"], ne, cond["t"], n_saidas=3)
    nomes = [s.upper() for s in c.especies]
    n0 = c.densidades(y[-1])[[s.upper() for s in c.q.especies].index("N2")]
    tv = treanor(np.r_[n0, [y[-1, nomes.index(f"N2(V{k})")] for k in range(1, 5)]], cond["tg"])
    en = c.en_sustentado(CORRENTE_MA[cond["id"]] * 1e-3, cond["t"])
    return dict(tv=tv, dis=y[-1, nomes.index("N")] / n0, en=en)


def main():
    saida = pathlib.Path(sys.argv[1])
    saida.mkdir(parents=True, exist_ok=True)
    linhas = []
    for cond in condicoes():
        for caso in ("n2_amonia", "n2_tese"):
            for nome, gamas in variantes().items():
                g = grandezas(caso, cond, gamas)
                linhas.append(dict(id=cond["id"], quimica=caso, variante=nome, **g))
                print(f"{cond['id']:10s} {caso:10s} {nome:10s} Tv {g['tv']:8.0f}  [N]/[N2] {100 * g['dis']:.4f}%  "
                      f"E/N {g['en']:6.1f}", flush=True)
    with open(saida / "parede.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(linhas[0]))
        w.writeheader()
        w.writerows(linhas)


if __name__ == "__main__":
    main()
