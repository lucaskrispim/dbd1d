# SPDX-License-Identifier: GPL-3.0-or-later
"""Radiative states of the N2 positive column (V3 of docs/coluna_positiva_n2): the
ratio N2(B)/N2(C) at the end of the column, from the 0D model with the three
chemistries, against the ratio measured by Levaton et al. (2020, fig. 10 at
10 mA; 2021, figs. 2 and 3 at 60 mA, 230 and 530 Pa) and against the ratio of
their own model. The measured densities are relative (normalized by the authors
to their calculated N2(C)), so only the ratio is compared; N2+(B) is left out
because the chemistries do not have it (its main source, N2(X, v >= 12) + N2+,
needs vibrational levels above v = 8).

Same choices as scripts/coluna_n2.py: imposed E/N, T_g and n_e (n_e recalculated
with our mobility), the wall probabilities of Levaton, residence time of the table;
the range comes from the edges of the error bars of E/N, n_e and T_g.

Usage: python3 scripts/coluna_n2_v3.py SAIDA_DIR
Writes v3.csv and razao_b_c.png.
"""
import csv
import pathlib
import sys

import numpy as np

RAIZ = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "dbd1d"))
sys.path.insert(0, str(RAIZ / "scripts"))
from coluna import Coluna0D  # noqa: E402
from coluna_n2 import GAMAS, QUIMICAS, condicoes, ne_nosso  # noqa: E402

DADOS = RAIZ / "docs" / "literatura" / "dados"
# condition: (file of the measured points, file of the N2(B) and N2(C) calculated by the authors)
MEDIDAS = {"L20_10mA": ("levaton2020/fig10_medidos.csv", "levaton2020/fig10_{}_calculado.csv"),
           "B21_230Pa": ("levaton2021/fig2_medidos.csv", "levaton2021/fig2_{}_calculado.csv"),
           "B21_530Pa": ("levaton2021/fig3_medidos.csv", "levaton2021/fig3_{}_calculado.csv")}
NOMES_B_C = {"n2_amonia": ("N2(B3)", "N2(C3)"), "n2_tese": ("N2(B3)", "N2(C3)"), "n2_exemplo": ("N2(B)", "N2(C)")}


def medido(cid):
    pontos, calculado = MEDIDAS[cid]
    with open(DADOS / pontos) as f:
        d = {l.split(",")[0]: [float(x) for x in l.split(",")[1:]] for l in f if not l.startswith("#")}
    (t, b, b_inf, b_sup), (_, c, c_inf, c_sup) = d["N2_B"], d["N2_C"]
    razao, faixa = b / c, (b_inf / c_sup, b_sup / c_inf)
    curvas = {e: np.loadtxt(DADOS / calculado.format(e), delimiter=",") for e in ("N2_B", "N2_C")}
    deles = (np.interp(np.log(t), np.log(curvas["N2_B"][:, 0]), curvas["N2_B"][:, 1])
             / np.interp(np.log(t), np.log(curvas["N2_C"][:, 0]), curvas["N2_C"][:, 1]))
    return razao, faixa, deles


def razao(caso, cond, en, ne, tg):
    c = Coluna0D(RAIZ / "casos" / caso, cond["p"], tg, cond["r"], gamas=GAMAS, estrito=False)
    _, y = c.integra(en, ne_nosso(c, ne, en), cond["t"], n_saidas=3)
    b, cc = NOMES_B_C[caso]
    return y[-1, c.especies.index(b)] / y[-1, c.especies.index(cc)]


def main():
    saida = pathlib.Path(sys.argv[1])
    saida.mkdir(parents=True, exist_ok=True)
    linhas = []
    for cond in (c for c in condicoes() if c["id"] in MEDIDAS):
        r_med, faixa_med, r_deles = medido(cond["id"])
        for caso in QUIMICAS:
            centro = razao(caso, cond, cond["en"], cond["ne"], cond["tg"])
            bordas = [razao(caso, cond, cond["en"] + s * cond["en_err"], cond["ne"], cond["tg"]) for s in (-1, 1)]
            bordas += [razao(caso, cond, cond["en"], cond["ne"] + s * cond["ne_err"], cond["tg"]) for s in (-1, 1)]
            bordas += [razao(caso, cond, cond["en"], cond["ne"], cond["tg"] + s * cond["tg_err"]) for s in (-1, 1)]
            linhas.append(dict(id=cond["id"], quimica=caso, medido=r_med, medido_min=faixa_med[0],
                               medido_max=faixa_med[1], levaton_modelo=r_deles, previsto=centro,
                               previsto_min=min(bordas + [centro]), previsto_max=max(bordas + [centro])))
            print(f"{cond['id']:10s} {caso:11s} B/C measured {r_med:6.1f} ({faixa_med[0]:.1f} to {faixa_med[1]:.1f})  "
                  f"their model {r_deles:6.1f}  ours {centro:6.1f} ({min(bordas):.1f} to {max(bordas):.1f})",
                  flush=True)
    with open(saida / "v3.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(linhas[0]))
        w.writeheader()
        w.writerows(linhas)
    figura(linhas, saida / "razao_b_c.png")


def figura(linhas, arquivo):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ids = list(dict.fromkeys(l["id"] for l in linhas))
    x = np.arange(len(ids))
    fig, ax = plt.subplots(figsize=(8, 5))
    med = [next(l for l in linhas if l["id"] == i) for i in ids]
    r = np.array([m["medido"] for m in med])
    ax.errorbar(x, r, yerr=[r - [m["medido_min"] for m in med], np.array([m["medido_max"] for m in med]) - r],
                fmt="ks", capsize=4, ms=7, label="measured")
    ax.plot(x, [m["levaton_modelo"] for m in med], "kx", ms=9, label="model of Levaton et al.")
    for k, (caso, cor) in enumerate(zip(QUIMICAS, ("tab:red", "tab:blue", "tab:green"))):
        sel = [next(l for l in linhas if l["id"] == i and l["quimica"] == caso) for i in ids]
        p = np.array([s["previsto"] for s in sel])
        ax.errorbar(x + 0.15 * (k - 1), p, yerr=[p - [s["previsto_min"] for s in sel],
                                                 np.array([s["previsto_max"] for s in sel]) - p],
                    fmt="o", color=cor, capsize=3, label=caso)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(ids)
    ax.set_ylabel("N2(B) / N2(C) at the end of the column")
    ax.set_title("radiative states (V3)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(arquivo, dpi=110)


if __name__ == "__main__":
    main()
