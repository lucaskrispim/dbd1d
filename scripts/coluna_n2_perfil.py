# SPDX-License-Identifier: GPL-3.0-or-later
"""Radial profile in the N2 positive column (docs/coluna_positiva_n2): the 0D model
uses the means over the cross section, so a rate between two species with the
Bessel profile J0(2.405 r/R) is underestimated by the factor <J0^2>/<J0>^2 = 1.446
(dbd1d/coluna.py, perfil_radial). For each condition and chemistry, with and
without the correction: the sustaining E/N (n_e from the current), and at the
measured E/N and n_e the vibrational temperature, [N]/[N2] and N2(B)/N2(C) at the
end of the column; and the reactions whose rate the correction changes by more
than 1%. Same choices as scripts/coluna_n2.py and scripts/coluna_n2_en.py; the
ladder up to v = 45 with the two vibrational wall probabilities.

Usage: python3 scripts/coluna_n2_perfil.py SAIDA_DIR
Writes perfil.csv.
"""
import csv
import pathlib
import sys

import numpy as np

RAIZ = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "dbd1d"))
sys.path.insert(0, str(RAIZ / "scripts"))
from coluna import Coluna0D  # noqa: E402
from coluna_n2 import QUIMICAS, condicoes, n2_total, ne_nosso, treanor  # noqa: E402
from coluna_n2_en import CORRENTE_MA  # noqa: E402
from coluna_n2_lote import gamas  # noqa: E402

B_C = {"n2_amonia": ("N2(B3)", "N2(C3)"), "n2_tese": ("N2(B3)", "N2(C3)"), "n2_exemplo": ("N2(B)", "N2(C)"),
       "n2_amonia_v45": ("N2(B3)", "N2(C3)"), "n2_amonia_a2": ("N2(B3)", "N2(C3)"), "n2_tese_a2": ("N2(B3)", "N2(C3)")}
CASOS = [(q, "base") for q in QUIMICAS] + [("n2_amonia_v45", "base"), ("n2_amonia_v45", "vib_gordiets")]


def estado(caso, cond, g, perfil):
    c = Coluna0D(RAIZ / "casos" / caso, cond["p"], cond["tg"], cond["r"], gamas=g, estrito=False, perfil_radial=perfil)
    ne = ne_nosso(c, cond["ne"], cond["en"]) if cond["mobilidade_levaton"] else cond["ne"]
    _, y = c.integra(cond["en"], ne, cond["t"], n_saidas=3)
    nomes = [s.upper() for s in c.especies]
    n0 = c.densidades(y[-1])[[s.upper() for s in c.q.especies].index("N2")]
    vib = [f"N2(V{k})" for k in range(1, 5)]
    tv = treanor(np.r_[n0, [y[-1, nomes.index(s)] for s in vib]], cond["tg"]) if all(s in nomes for s in vib) else np.nan
    b, cc = (nomes.index(s.upper()) for s in B_C[caso])
    en = c.en_sustentado(CORRENTE_MA[cond["id"]] * 1e-3, cond["t"])
    return c, dict(en=en, tv=tv, dis=y[-1, nomes.index("N")] / n2_total(c, y[-1]), b_c=y[-1, b] / y[-1, cc])


def main():
    saida = pathlib.Path(sys.argv[1])
    saida.mkdir(parents=True, exist_ok=True)
    linhas = []
    for cond in condicoes():
        for caso, variante in CASOS:
            g = gamas(variante)
            _, sem = estado(caso, cond, g, False)
            c, com = estado(caso, cond, g, True)
            mudadas = [(c.q.reacoes[r], f) for r, f in enumerate(c.fator_radial) if abs(f - 1.0) > 0.01]
            linha = dict(id=cond["id"], quimica=caso, variante=variante, en_med=cond["en"],
                         **{f"{k}_0d": v for k, v in sem.items()}, **{f"{k}_perfil": v for k, v in com.items()},
                         n_reacoes=len(mudadas), fator_max=max((f for _, f in mudadas), default=1.0),
                         reacoes="; ".join(f"{r} x{f:.3f}" for r, f in mudadas))
            linhas.append(linha)
            print(f"{cond['id']:10s} {caso:14s} {variante:12s} E/N {sem['en']:6.1f} -> {com['en']:6.1f}  "
                  f"Tv {sem['tv']:6.0f} -> {com['tv']:6.0f}  [N]/[N2] x{com['dis'] / sem['dis']:.3f}  "
                  f"B/C x{com['b_c'] / sem['b_c']:.3f}  {len(mudadas)} reactions, max x{linha['fator_max']:.3f}",
                  flush=True)
    with open(saida / "perfil.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(linhas[0]))
        w.writeheader()
        w.writerows(linhas)


if __name__ == "__main__":
    main()
