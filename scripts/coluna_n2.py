# SPDX-License-Identifier: GPL-3.0-or-later
"""Positive column of N2 (docs/coluna_positiva_n2): the 0D model (dbd1d/coluna.py)
in the measured conditions of Levaton et al. 2020 and 2021 (condicoes.csv) and of
Gordiets et al. 1995 (pure N2 points), with the three N2 chemistries, against the
measured vibrational temperature (V2) and the dissociation (V5).

Choices, as in docs/coluna_positiva_n2/README.md:
- E/N, T_g and n_e imposed; the n_e of Levaton, inferred with mu_e = 4.2e5/p
  (p in Torr), is recalculated with our electron mobility at that E/N and T_g
  (n_e is inversely proportional to mu_e); the n_e of Gordiets comes from a
  microwave cavity and is used as given;
- wall probabilities of Levaton 2020 (Pyrex): N(4S) 3.2e-4 and N2(v) 4.5e-4; the
  electronic metastables N2(A), N2(a'), N(2D) and N(2P), 1;
- residence time: that of the table (Levaton) or 0.1 s, the steady state far
  downstream (Gordiets);
- Tv: Treanor fit to N2(v = 0 to 4), anharmonic levels of N2(X) (we 2358.57,
  wexe 14.324 cm-1), as Levaton fits the first five levels; v = 0 is the
  background, depleted by what the other species hold (coluna.py);
- dissociation: [N]/[N2];
- uncertainty: each condition also at the edges of its error bars of E/N, n_e
  and T_g (one at a time; 10% for Gordiets, which gives no error bars); the
  result is the envelope.

Usage: python3 scripts/coluna_n2.py SAIDA_DIR
Writes resultados.csv and the figure tv_dissociacao.png.
"""
import csv
import pathlib
import sys

import numpy as np

RAIZ = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "dbd1d"))
from coluna import Coluna0D  # noqa: E402

KB = 1.380649e-23
HC_K = 1.438776877           # cm K: h c / k
WE, WEXE = 2358.57, 14.324   # cm-1, N2(X)
QUIMICAS = ("n2_amonia", "n2_tese", "n2_exemplo")
GAMAS = {"N": 3.2e-4, "N2(A3)": 1.0, "N2(A)": 1.0, "N2(a'1)": 1.0, "N2(a`1)": 1.0, "N2(a')": 1.0,
         "N(2D)": 1.0, "N(2P)": 1.0}


def energia_vib(v):
    """Energy (K) of N2(X, v) above v = 0."""
    g = lambda x: WE * (x + 0.5) - WEXE * (x + 0.5) ** 2  # noqa: E731
    return (g(np.asarray(v, dtype=float)) - g(0.0)) * HC_K


def treanor(n, t_gas):
    """Treanor temperature theta (K) from the populations n[v], v = 0..4:
    ln(n_v/n_0) = -v E1/theta + (v E1 - E_v)/T_g, least squares in 1/theta."""
    v = np.arange(len(n))
    e1, ev = energia_vib(1), energia_vib(v)
    y = np.log(n / n[0]) - (v * e1 - ev) / t_gas
    x = -v * e1
    inv = np.sum(x * y) / np.sum(x * x)
    return 1.0 / inv


def condicoes():
    lista = []
    with open(RAIZ / "docs" / "coluna_positiva_n2" / "condicoes.csv") as f:
        for linha in csv.DictReader(l for l in f if not l.startswith("#")):
            lista.append(dict(id=linha["id"], p=float(linha["pressao_Pa"]), r=0.3, tg=float(linha["Tg_K"]),
                              tg_err=float(linha["Tg_err"]), en=float(linha["EN_Td"]), en_err=float(linha["EN_err"]),
                              ne=float(linha["ne_cm3"]), ne_err=float(linha["ne_err"]),
                              t=float(linha["t_res_ms"]) * 1e-3, tv=float(linha["Tv_K"]),
                              tv_err=float(linha["Tv_err"]), mobilidade_levaton=True))
    with open(RAIZ / "docs" / "literatura" / "dados" / "gordiets1995" / "n2_puro.csv") as f:
        for linha in (l.strip().split(",") for l in f if not l.startswith("#")):
            i, tg, ne, en, tv = linha
            ne = float(ne) if ne else 2.0e10       # 80 mA: from the text of Gordiets et al. 1996
            lista.append(dict(id=f"G95_{i}mA", p=266.64, r=0.8, tg=float(tg), tg_err=0.1 * float(tg), en=float(en),
                              en_err=0.1 * float(en), ne=ne, ne_err=0.1 * ne, t=0.1, tv=float(tv), tv_err=500.0,
                              mobilidade_levaton=False))
    return lista


def ne_nosso(c, ne, en):
    """n_e of Levaton recalculated with our mobility: n_e mu_e is the measured current density."""
    v = c.tab(np.array([en]), np.array([c.t_gas]))
    mu_nosso = v[0, 1] / c.n_gas
    mu_levaton = 4.2e5 / (c.pressao_pa / 133.322)
    return ne * mu_levaton / mu_nosso


def roda(caso, cond, en, ne, tg):
    c = Coluna0D(RAIZ / "casos" / caso, cond["p"], tg, cond["r"], gamas=GAMAS, estrito=False)
    if cond["mobilidade_levaton"]:
        ne = ne_nosso(c, ne, en)
    _, y = c.integra(en, ne, cond["t"], n_saidas=3)
    nomes = [s.upper() for s in c.especies]
    fim = y[-1]
    vib = [f"N2(V{k})" for k in range(1, 5)]
    n0 = c.densidades(fim)[[s.upper() for s in c.q.especies].index("N2")]
    tv = treanor(np.r_[n0, [fim[nomes.index(s)] for s in vib]], tg) if all(s in nomes for s in vib) else np.nan
    return dict(tv=tv, dis=fim[nomes.index("N")] / n0, ne=ne)


def main():
    saida = pathlib.Path(sys.argv[1])
    saida.mkdir(parents=True, exist_ok=True)
    linhas = []
    for cond in condicoes():
        for caso in QUIMICAS:
            centro = roda(caso, cond, cond["en"], cond["ne"], cond["tg"])
            bordas = [roda(caso, cond, cond["en"] + s * cond["en_err"], cond["ne"], cond["tg"]) for s in (-1, 1)]
            bordas += [roda(caso, cond, cond["en"], cond["ne"] + s * cond["ne_err"], cond["tg"]) for s in (-1, 1)]
            bordas += [roda(caso, cond, cond["en"], cond["ne"], cond["tg"] + s * cond["tg_err"]) for s in (-1, 1)]
            tvs = [b["tv"] for b in bordas] + [centro["tv"]]
            dis = [b["dis"] for b in bordas] + [centro["dis"]]
            linhas.append(dict(id=cond["id"], quimica=caso, ne_usado=centro["ne"], tv_med=cond["tv"],
                               tv_err=cond["tv_err"], tv=centro["tv"], tv_min=np.nanmin(tvs), tv_max=np.nanmax(tvs),
                               dis=centro["dis"], dis_min=min(dis), dis_max=max(dis)))
            print(f"{cond['id']:10s} {caso:11s} ne {centro['ne']:.3g}  Tv {centro['tv']:7.0f} "
                  f"({np.nanmin(tvs):.0f} to {np.nanmax(tvs):.0f}) measured {cond['tv']:.0f}  "
                  f"[N]/[N2] {100 * centro['dis']:.3f}%", flush=True)
    with open(saida / "resultados.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(linhas[0]))
        w.writeheader()
        w.writerows(linhas)
    figura(linhas, saida / "tv_dissociacao.png")


def figura(linhas, arquivo):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ids = list(dict.fromkeys(l["id"] for l in linhas))
    x = np.arange(len(ids))
    # Tv only at the short residence times (Levaton): with v <= 8, the steady state far
    # downstream (Gordiets, 0.1 s) piles the population at the top of the ladder
    curtos = [k for k, i in enumerate(ids) if not i.startswith("G95")]
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    med = [next(l for l in linhas if l["id"] == i) for i in ids]
    ax[0].errorbar(x[curtos], [med[k]["tv_med"] for k in curtos], yerr=[med[k]["tv_err"] for k in curtos],
                   fmt="ks", capsize=4, label="measured")
    ax[0].plot(x[:3], [1600, 2300, 3800], "kx", ms=9, label="Levaton 2020, calculated")
    for k, (caso, cor) in enumerate(zip(QUIMICAS, ("tab:red", "tab:blue", "tab:green"))):
        sel = [next(l for l in linhas if l["id"] == i and l["quimica"] == caso) for i in ids]
        xs = x + 0.15 * (k - 1)
        tv = np.array([s["tv"] for s in sel])
        if not np.all(np.isnan(tv)):
            c = curtos
            ax[0].errorbar(xs[c], tv[c], yerr=[tv[c] - np.array([sel[j]["tv_min"] for j in c]),
                                               np.array([sel[j]["tv_max"] for j in c]) - tv[c]],
                           fmt="o", color=cor, capsize=3, label=caso)
        dis = np.array([s["dis"] for s in sel]) * 100
        ax[1].errorbar(xs, dis, yerr=[dis - 100 * np.array([s["dis_min"] for s in sel]),
                                      100 * np.array([s["dis_max"] for s in sel]) - dis],
                       fmt="o", color=cor, capsize=3, label=caso)
    ax[1].plot([0, 2], [0.07, 0.2], "k^", ms=9, label="Levaton 2020, calculated (10 and 50 mA)")
    ax[1].set_yscale("log")
    for a, rot in zip(ax, ("Tv (K)", "[N]/[N2] (%)")):
        a.set_xticks(x)
        a.set_xticklabels(ids, rotation=30, ha="right")
        a.set_ylabel(rot)
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    ax[0].set_title("vibrational temperature (V2), short residence only")
    ax[0].set_xlim(-0.5, len(ids) - 0.5)
    ax[1].set_title("dissociation at the end of the column (V5)")
    fig.tight_layout()
    fig.savefig(arquivo, dpi=110)


if __name__ == "__main__":
    main()
