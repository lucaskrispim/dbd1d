# SPDX-License-Identifier: GPL-3.0-or-later
"""N2(a''1) separated from N2(C3) in the N2 positive column (docs/coluna_positiva_n2):
chemistries A and B lump the excitation to N2(a'') into N2(C), so that its energy
comes out as N2(C) -> N2(B) emission; casos/n2_amonia_a2 and casos/n2_tese_a2 make
it a species (ferramentas/casos_coluna.py, separa_a2). For each condition, with the
lumped and the separated a'': the sustaining E/N, and at the measured E/N and n_e
Tv, [N]/[N2], N2(B)/N2(C) and [N2(a'')]/[N2(C)] at the end of the column. Same
choices as scripts/coluna_n2_perfil.py, without the radial correction.

Usage: python3 scripts/coluna_n2_a2.py SAIDA_DIR
Writes a2.csv.
"""
import csv
import pathlib
import sys

import numpy as np

RAIZ = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "scripts"))
sys.path.insert(0, str(RAIZ / "ferramentas"))
import proveniencia  # noqa: E402
from coluna_n2 import condicoes  # noqa: E402
from coluna_n2_lote import gamas  # noqa: E402
from coluna_n2_perfil import estado  # noqa: E402

PARES = (("n2_amonia", "n2_amonia_a2"), ("n2_tese", "n2_tese_a2"))


def main():
    saida = pathlib.Path(sys.argv[1])
    saida.mkdir(parents=True, exist_ok=True)
    linhas = []
    for cond in condicoes():
        for junto, separado in PARES:
            _, a = estado(junto, cond, gamas("base"), False)
            c, b = estado(separado, cond, gamas("base"), False)
            _, y = c.integra(cond["en"], _ne(c, cond), cond["t"], n_saidas=3)
            nomes = [s.upper() for s in c.especies]
            a2_c = y[-1, nomes.index("N2(A``1)")] / y[-1, nomes.index("N2(C3)")]
            linhas.append(dict(id=cond["id"], quimica=junto, **{f"{k}_junto": v for k, v in a.items()},
                               **{f"{k}_separado": v for k, v in b.items()}, a2_sobre_c=a2_c))
            print(f"{cond['id']:10s} {junto:10s} E/N {a['en']:6.1f} -> {b['en']:6.1f}  Tv {a['tv']:6.0f} -> {b['tv']:6.0f}  "
                  f"[N]/[N2] x{b['dis'] / a['dis']:.3f}  B/C {a['b_c']:.3g} -> {b['b_c']:.3g}  a''/C {a2_c:.3g}", flush=True)
    with open(saida / "a2.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(linhas[0]))
        w.writeheader()
        w.writerows(linhas)
    proveniencia.grava_origem(saida / "a2.csv", __file__, ['n2_amonia', 'n2_amonia_a2', 'n2_tese', 'n2_tese_a2'])  # code and inputs beside the CSV


def _ne(c, cond):
    from coluna_n2 import ne_nosso
    return ne_nosso(c, cond["ne"], cond["en"]) if cond["mobilidade_levaton"] else cond["ne"]


if __name__ == "__main__":
    main()
