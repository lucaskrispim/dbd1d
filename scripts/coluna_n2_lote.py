# SPDX-License-Identifier: GPL-3.0-or-later
"""One run of the N2 positive column for a batch on a cluster node: a case, a
condition and a variant of the wall probabilities (the variants of
scripts/coluna_n2_parede.py, with the vibrational levels up to v = 45). Writes
one line of CSV: Tv and [N]/[N2] at the measured E/N, and the sustaining E/N.

Usage: python3 scripts/coluna_n2_lote.py CASO ID VARIANTE SAIDA.csv
"""
import pathlib
import re
import sys

import numpy as np

RAIZ = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "dbd1d"))
sys.path.insert(0, str(RAIZ / "scripts"))
from coluna import Coluna0D  # noqa: E402
from coluna_n2 import GAMAS, condicoes, ne_nosso, treanor  # noqa: E402
from coluna_n2_en import CORRENTE_MA  # noqa: E402

VIB = [f"N2(V{k})" for k in range(1, 46)]
METAESTAVEIS = ["N2(A3)", "N2(a'1)", "N2(a`1)", "N(2D)", "N(2P)"]


def gamas(variante):
    g = dict(GAMAS)
    if variante == "base":
        return g
    grupo, _, valor = variante.partition("_")
    if grupo == "N" and valor in ("x10", "div10"):
        g["N"] = GAMAS["N"] * (10.0 if valor == "x10" else 0.1)
    elif grupo == "vib":
        # gordiets: 7e-3, Gordiets et al. 1995, eq. (24), Pyrex tube of the same experiment
        fatores = {"x10": 10.0, "div10": 0.1, "gordiets": 7e-3 / 4.5e-4}
        if valor not in fatores:
            raise ValueError(f"unknown wall variant {variante}: vib_ takes {', '.join(fatores)}")
        fator = fatores[valor]
        g.update({s: 4.5e-4 * fator for s in VIB})
    elif grupo == "meta" and re.fullmatch(r"\d*\.?\d+(e-?\d+)?", valor) and 0.0 < float(valor) <= 1.0:
        g.update({s: float(valor) for s in METAESTAVEIS})
    else:
        raise ValueError(f"unknown wall variant {variante}: base, N_x10, N_div10, vib_x10, vib_div10, "
                         "vib_gordiets or meta_VALUE with VALUE in (0, 1]")
    return g


def main():
    caso, cid, variante, arquivo = sys.argv[1:5]
    cond = next(c for c in condicoes() if c["id"] == cid)
    c = Coluna0D(RAIZ / "casos" / caso, cond["p"], cond["tg"], cond["r"], gamas=gamas(variante), estrito=False)
    ne = ne_nosso(c, cond["ne"], cond["en"]) if cond["mobilidade_levaton"] else cond["ne"]
    _, y = c.integra(cond["en"], ne, cond["t"], n_saidas=3)
    nomes = [s.upper() for s in c.especies]
    n0 = c.densidades(y[-1])[[s.upper() for s in c.q.especies].index("N2")]
    tv = treanor(np.r_[n0, [y[-1, nomes.index(f"N2(V{k})")] for k in range(1, 5)]], cond["tg"])
    en = c.en_sustentado(CORRENTE_MA[cid] * 1e-3, cond["t"])
    with open(arquivo, "w") as f:
        f.write("caso,id,variante,tv,dis,en\n")
        f.write(f"{caso},{cid},{variante},{tv:.6g},{y[-1, nomes.index('N')] / n0:.6g},{en:.6g}\n")


if __name__ == "__main__":
    main()
