# SPDX-License-Identifier: GPL-3.0-or-later
"""E/N that sustains the N2 positive column (V1 of docs/coluna_positiva_n2): for
each measured condition, the E/N at which the electron production at the end of
the residence time equals the ambipolar loss, with n_e carried by the measured
current (dbd1d/coluna.py, en_sustentado), for the three N2 chemistries; then the
same with the associative ionization reactions (two excited N2 molecules giving
N4+ + e) multiplied by 1/5, 1/2, 2 and 5, and with T_g at the edges of its error
bar. Same wall probabilities and residence times as scripts/coluna_n2.py.

Usage: python3 scripts/coluna_n2_en.py SAIDA_DIR
Writes en.csv and en_sustentado.png.
"""
import csv
import pathlib
import sys

import numpy as np

RAIZ = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "dbd1d"))
sys.path.insert(0, str(RAIZ / "scripts"))
from coluna import Coluna0D  # noqa: E402
from coluna_n2 import GAMAS, QUIMICAS, condicoes  # noqa: E402

CORRENTE_MA = {"L20_10mA": 10, "L20_20mA": 20, "L20_50mA": 50, "B21_230Pa": 60, "B21_530Pa": 60,
               "G95_15mA": 15, "G95_30mA": 30, "G95_80mA": 80}
FATORES = (0.2, 0.5, 2.0, 5.0)


def associativas(q):
    """Indices of the reactions of two excited neutral N2 giving N4+ + e."""
    r = []
    for i, txt in enumerate(q.reacoes):
        if "=>" not in txt or txt.startswith("bolsig"):
            continue
        esq, dir_ = txt.upper().split("=>")
        reag = esq.split("+")
        if sorted(dir_.replace("^+", "").split("+")) == ["E", "N4"] and all(t.startswith("N2(") for t in reag):
            r.append(i)
    return r


def en(caso, cond, tg=None, fator=1.0):
    base = Coluna0D(RAIZ / "casos" / caso, cond["p"], cond["tg"] if tg is None else tg, cond["r"], gamas=GAMAS, estrito=False)
    fatores = {i: fator for i in associativas(base.q)} if fator != 1.0 else None
    c = Coluna0D(RAIZ / "casos" / caso, cond["p"], cond["tg"] if tg is None else tg, cond["r"], gamas=GAMAS, estrito=False,
                 fatores=fatores)
    return c.en_sustentado(CORRENTE_MA[cond["id"]] * 1e-3, cond["t"])


def main():
    saida = pathlib.Path(sys.argv[1])
    saida.mkdir(parents=True, exist_ok=True)
    linhas = []
    for cond in condicoes():
        for caso in QUIMICAS:
            centro = en(caso, cond)
            por_fator = {f: en(caso, cond, fator=f) for f in FATORES}
            por_tg = [en(caso, cond, tg=cond["tg"] + s * cond["tg_err"]) for s in (-1, 1)]
            linha = dict(id=cond["id"], quimica=caso, en_med=cond["en"], en_err=cond["en_err"], en=centro,
                         tg_min=min(por_tg), tg_max=max(por_tg),
                         **{f"fator_{f:g}": v for f, v in por_fator.items()})
            linhas.append(linha)
            print(f"{cond['id']:10s} {caso:11s} measured {cond['en']:6.1f}  predicted {centro:6.1f}  "
                  f"Tg: {min(por_tg):.1f} to {max(por_tg):.1f}  associative x0.2..x5: "
                  + " ".join(f"{por_fator[f]:.1f}" for f in FATORES), flush=True)
    with open(saida / "en.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(linhas[0]))
        w.writeheader()
        w.writerows(linhas)
    figura(linhas, saida / "en_sustentado.png")


def figura(linhas, arquivo):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ids = list(dict.fromkeys(l["id"] for l in linhas))
    x = np.arange(len(ids))
    fig, ax = plt.subplots(figsize=(11, 5))
    med = [next(l for l in linhas if l["id"] == i) for i in ids]
    ax.errorbar(x, [m["en_med"] for m in med], yerr=[m["en_err"] for m in med], fmt="ks", capsize=4, ms=7,
                label="measured")
    for k, (caso, cor) in enumerate(zip(QUIMICAS, ("tab:red", "tab:blue", "tab:green"))):
        sel = [next(l for l in linhas if l["id"] == i and l["quimica"] == caso) for i in ids]
        xs = x + 0.17 * (k - 1)
        c = np.array([s["en"] for s in sel])
        lo = np.array([min(s["fator_0.2"], s["fator_5"], s["tg_min"]) for s in sel])
        hi = np.array([max(s["fator_0.2"], s["fator_5"], s["tg_max"]) for s in sel])
        ax.errorbar(xs, c, yerr=[c - lo, hi - c], fmt="o", color=cor, capsize=3,
                    label=f"{caso} (bar: associative x0.2 to x5 and Tg)")
    ax.set_xticks(x)
    ax.set_xticklabels(ids, rotation=30, ha="right")
    ax.set_ylabel("E/N (Td)")
    ax.set_title("E/N that sustains the positive column (V1)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(arquivo, dpi=110)


if __name__ == "__main__":
    main()
