# SPDX-License-Identifier: GPL-3.0-or-later
"""Test of a mechanism for the high-excitation disagreement
(docs/coluna_positiva_n2/README.md), third test: the N2
chemistry of the ammonia with the full vibrational ladder of N2(X) up to v = 45
(casos/n2_amonia_v45, made by ferramentas/escada_vibracional.py with the V-T and
V-V rates of Capitelli et al. 2000, chapter 7). For each condition: Tv and
[N]/[N2] at the measured E/N (as scripts/coluna_n2.py), and the sustaining E/N
(as scripts/coluna_n2_en.py), with the base wall probabilities and with the
electronic metastables at 0.01.

Usage: python3 scripts/coluna_n2_v45.py SAIDA_DIR [ID]
Writes v45.csv, or v45_ID.csv with only the condition ID (one process per condition).
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

CASO = RAIZ / "casos" / "n2_amonia_v45"
META = dict(GAMAS, **{s: 0.01 for s in ("N2(A3)", "N2(a'1)", "N(2D)", "N(2P)")})


def main():
    saida = pathlib.Path(sys.argv[1])
    saida.mkdir(parents=True, exist_ok=True)
    linhas = []
    so = sys.argv[2] if len(sys.argv) > 2 else None
    for cond in (c for c in condicoes() if so is None or c["id"] == so):
        c = Coluna0D(CASO, cond["p"], cond["tg"], cond["r"], gamas=GAMAS, estrito=False)
        ne = ne_nosso(c, cond["ne"], cond["en"]) if cond["mobilidade_levaton"] else cond["ne"]
        _, y = c.integra(cond["en"], ne, cond["t"], n_saidas=3)
        nomes = [s.upper() for s in c.especies]
        n0 = c.densidades(y[-1])[[s.upper() for s in c.q.especies].index("N2")]
        tv = treanor(np.r_[n0, [y[-1, nomes.index(f"N2(V{k})")] for k in range(1, 5)]], cond["tg"])
        dis = y[-1, nomes.index("N")] / n0
        i_a = CORRENTE_MA[cond["id"]] * 1e-3
        en = c.en_sustentado(i_a, cond["t"])
        en_meta = Coluna0D(CASO, cond["p"], cond["tg"], cond["r"], gamas=META, estrito=False).en_sustentado(i_a, cond["t"])
        linhas.append(dict(id=cond["id"], tv_med=cond["tv"], tv=tv, dis=dis, en_med=cond["en"], en=en,
                           en_meta001=en_meta))
        print(f"{cond['id']:10s} Tv {tv:6.0f} (measured {cond['tv']:.0f})  [N]/[N2] {100 * dis:.4f}%  "
              f"E/N {en:6.1f}, with gamma_meta 0.01 {en_meta:6.1f} (measured {cond['en']:.1f})", flush=True)
    with open(saida / (f"v45_{so}.csv" if so else "v45.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(linhas[0]))
        w.writeheader()
        w.writerows(linhas)


if __name__ == "__main__":
    main()
