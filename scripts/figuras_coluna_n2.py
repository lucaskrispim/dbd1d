# SPDX-License-Identifier: GPL-3.0-or-later
"""Figures of the N2 positive column article (artigo_coluna_n2/), from the results
recorded in docs/coluna_positiva_n2: etapa4 (Tv, dissociation), etapa5 (sustaining
E/N, three chemistries), etapa6 (N2(B)/N2(C)), falta/ (wall sensitivity,
superelastic, stepwise ionization, vibrational ladder), pendencias/ (coupling at the
self-consistent root, vibrational distributions, Kumar and Ghosh 1991).

Usage: python3 scripts/figuras_coluna_n2.py artigo_coluna_n2/figuras
"""
import csv
import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Patch, Rectangle  # noqa: E402
import numpy as np  # noqa: E402

RAIZ = pathlib.Path(__file__).resolve().parents[1]
D = RAIZ / "docs" / "coluna_positiva_n2"
IDS = ["L20_10mA", "L20_20mA", "L20_50mA", "B21_230Pa", "B21_530Pa", "G95_15mA", "G95_30mA", "G95_80mA"]
ROTULOS = ["L 10 mA\n230 Pa", "L 20 mA\n230 Pa", "L 50 mA\n230 Pa", "L 60 mA\n230 Pa", "L 60 mA\n530 Pa",
           "G 15 mA\n267 Pa", "G 30 mA\n267 Pa", "G 80 mA\n267 Pa"]
EN_MED = {"L20_10mA": (118, 14), "L20_20mA": (108, 13), "L20_50mA": (90, 10), "B21_230Pa": (94, 10),
          "B21_530Pa": (48, 6), "G95_15mA": (66.7, 6.7), "G95_30mA": (63.0, 6.3), "G95_80mA": (69.9, 7.0)}
TV_MED = {"L20_10mA": (1900, 350), "L20_20mA": (2700, 500), "L20_50mA": (5100, 900), "B21_230Pa": (6300, 1100),
          "B21_530Pa": (6900, 1200), "G95_15mA": (3000, 500), "G95_30mA": (4000, 500), "G95_80mA": (5500, 500)}
plt.rcParams.update({"font.size": 9})


def le_csv(caminho):
    with open(caminho) as f:
        return list(csv.DictReader(l for l in f if not l.startswith("#")))


PAREDES = (("base", "Levaton wall"), ("vib_gordiets", "Gordiets wall"))


def acoplado(cid, variante="base", modo="completo_ev"):
    """Self-consistent root of a coupled run (scripts/coluna_n2_raiz.py): the point at
    the converged E/N, with Tv, dissociation and populations; None if the chain has
    not finished or found no root. The default mode is the full coupling with e-V
    from the excited levels."""
    nome = cid + ("" if variante == "base" else f"_{variante}") + ("" if modo == "completo" else f"_{modo}")
    f = D / "pendencias" / "raiz" / f"{nome}_raiz.json"
    if not f.exists():
        return None
    d = json.loads(f.read_text())
    if not d.get("convergiu"):
        return None
    return next(p for p in d["pontos"] if p["en"] == d["en"])


def figura_en(saida):
    en5 = {(l["id"], l["quimica"]): float(l["en"]) for l in le_csv(D / "etapa5" / "en.csv")}
    v45 = {(l["caso"], l["id"], l["variante"]): float(l["en"]) for l in le_csv(D / "falta" / "escada_v45.csv")}
    x = np.arange(len(IDS))
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    med = np.array([EN_MED[i][0] for i in IDS])
    ax.errorbar(x, med, yerr=[EN_MED[i][1] for i in IDS], fmt="ks", ms=6, capsize=3, label="measured", zorder=5)
    estilos = [("A, v $\\leq$ 8", [en5[(i, "n2_amonia")] for i in IDS], "o", "tab:red", -0.27),
               ("B, v $\\leq$ 8", [en5[(i, "n2_tese")] for i in IDS], "o", "tab:blue", -0.18),
               ("C", [en5[(i, "n2_exemplo")] for i in IDS], "o", "tab:green", -0.09),
               ("A, v $\\leq$ 45", [v45[("n2_amonia_v45", i, "base")] for i in IDS], "D", "tab:orange", 0.0)]
    for rot, y, m, cor, dx in estilos:
        ax.plot(x + dx, y, m, color=cor, ms=5, mfc="none", label=rot)
    for (var, rot), m, cor, dx in zip(PAREDES, ("^", "v"), ("tab:purple", "black"), (0.12, 0.24)):
        ys = [acoplado(i, var) for i in IDS]
        ok = [k for k, j in enumerate(ys) if j]
        ax.plot(x[ok] + dx, [ys[k]["en"] for k in ok], m, color=cor, ms=6, label=f"A, coupled, {rot}")
    ax.set_xticks(x)
    ax.set_xticklabels(ROTULOS, fontsize=8)
    ax.set_ylabel("E/N (Td)")
    ax.set_ylim(0, 200)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5, ncol=2, loc="upper left")
    fig.tight_layout()
    fig.savefig(saida / "en.pdf")
    plt.close(fig)


def figura_tv(saida):
    e4 = {(l["id"], l["quimica"]): float(l["tv"]) for l in le_csv(D / "etapa4" / "resultados.csv")}
    v45 = {(l["caso"], l["id"], l["variante"]): float(l["tv"]) for l in le_csv(D / "falta" / "escada_v45.csv")}
    x = np.arange(len(IDS))
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.errorbar(x, [TV_MED[i][0] for i in IDS], yerr=[TV_MED[i][1] for i in IDS], fmt="ks", ms=6, capsize=3,
                label="measured", zorder=5)
    curtos = [k for k, i in enumerate(IDS) if not i.startswith("G95")]
    ax.plot(x[curtos] - 0.25, [e4[(IDS[k], "n2_amonia")] for k in curtos], "o", color="tab:red", mfc="none",
            label="A, v $\\leq$ 8")
    for (var, rot), m, cor, dx in zip(PAREDES, ("D", "d"), ("tab:orange", "tab:brown"), (-0.12, 0.0)):
        ax.plot(x + dx, [v45[("n2_amonia_v45", i, var)] for i in IDS], m, color=cor, mfc="none",
                label=f"A, v $\\leq$ 45, {rot}")
    for (var, rot), m, cor, dx in zip(PAREDES, ("^", "v"), ("tab:purple", "black"), (0.12, 0.24)):
        ys = [acoplado(i, var) for i in IDS]
        ok = [k for k, j in enumerate(ys) if j]
        ax.plot(x[ok] + dx, [ys[k]["tv"] for k in ok], m, color=cor, ms=6, label=f"A, coupled, {rot}")
    ax.plot([0, 1, 2, 3, 4], [1600, 2300, 3800, 5400, 6100], "x", color="gray", ms=7, label="Levaton et al. model")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(ROTULOS, fontsize=8)
    ax.set_ylabel("$T_v$ (K)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7.5, ncol=2, loc="upper left")
    fig.tight_layout()
    fig.savefig(saida / "tv.pdf")
    plt.close(fig)


def figura_mecanismos(saida):
    """Left: E/N shift from chemistry A (v <= 8) for each candidate mechanism.
    Right: the coupling decomposed, for each wall: the ladder up to v = 45, the
    levels N2(v) as electron targets, the shifted thresholds and the superelastic
    collisions, stacked."""
    altos = ["L20_50mA", "B21_230Pa", "B21_530Pa", "G95_15mA", "G95_30mA", "G95_80mA"]
    en5 = {(l["id"], l["quimica"]): float(l["en"]) for l in le_csv(D / "etapa5" / "en.csv")}
    parede = {(l["id"], l["quimica"], l["variante"]): float(l["en"]) for l in le_csv(D / "falta" / "parede.csv")}
    sup = {l["id"]: l for l in le_csv(D / "falta" / "superelastica.csv")}
    etp = {l["id"]: l for l in le_csv(D / "falta" / "etapas.csv")}
    v45 = {(l["caso"], l["id"], l["variante"]): float(l["en"]) for l in le_csv(D / "falta" / "escada_v45.csv")}
    base = np.array([en5[(i, "n2_amonia")] for i in altos])
    med = np.array([EN_MED[i][0] for i in altos])
    x = np.arange(len(altos))
    rotulos = [ROTULOS[IDS.index(i)] for i in altos]
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.6), sharey=True)
    deltas = {
        "metastable wall $\\gamma$ = 0.01": [parede[(i, "n2_amonia", "meta_0.01")] for i in altos],
        "superelastic, $T_v$ = 6000 K": [float(sup[i]["en_tv6000"]) for i in altos],
        "stepwise ionization": [float(etp[i]["en_base"]) for i in altos],
        "ladder v $\\leq$ 45": [v45[("n2_amonia_v45", i, "base")] for i in altos],
    }
    w = 0.2
    for k, (rot, y) in enumerate(deltas.items()):
        ax[0].bar(x + (k - 1.5) * w, np.array(y) - base, w, label=rot)
    cores = {"alvo_ev": "tab:cyan", "deslocado_ev": "tab:blue", "completo_ev": "tab:purple"}
    nomes = {"alvo_ev": "N$_2(v)$ as targets", "deslocado_ev": "+ shifted thresholds", "completo_ev": "+ superelastic"}
    for j, (var, rot) in enumerate(PAREDES):
        xs = x + (j - 0.5) * 0.36
        for k, i in enumerate(altos):
            anterior = v45[("n2_amonia_v45", i, var)]
            ax[1].bar(xs[k], anterior - base[k], 0.34, color="0.8", edgecolor="k", lw=0.3)
            faltou = False
            for m, cor in cores.items():
                final = acoplado(i, var, m)
                if final is None:
                    # step not finished: the next bar holds both steps, hatched
                    faltou = True
                    continue
                ax[1].bar(xs[k], final["en"] - anterior, 0.34, bottom=anterior - base[k], color=cor, edgecolor="k",
                          lw=0.3, hatch="//" if faltou else None)
                anterior, faltou = final["en"], False
            ax[1].text(xs[k], 1.5, "L" if var == "base" else "G", ha="center", fontsize=7)
    for a in ax:
        a.plot(x, med - base, "k_", ms=16, mew=2, label="needed")
        a.axhline(0, color="k", lw=0.5)
        a.set_xticks(x)
        a.set_xticklabels(rotulos, fontsize=7.5)
        a.grid(alpha=0.3, axis="y")
    ax[0].set_ylabel("$\\Delta$E/N from chemistry A, v $\\leq$ 8 (Td)")
    ax[0].legend(fontsize=7, loc="lower left")
    marcas = [Patch(facecolor="0.8", edgecolor="k", lw=0.3, label="ladder v $\\leq$ 45 with the wall")]
    marcas += [Patch(facecolor=cor, edgecolor="k", lw=0.3, label=nomes[m]) for m, cor in cores.items()]
    ax[1].legend(handles=marcas + ax[1].get_legend_handles_labels()[0], fontsize=7, loc="lower left")
    ax[0].set_ylim(top=6)
    ax[0].set_title("(a) candidate mechanisms", fontsize=9)
    ax[1].set_title("(b) coupling, L and G walls", fontsize=9)
    fig.tight_layout()
    fig.savefig(saida / "mecanismos.pdf")
    plt.close(fig)


def figura_radiativos(saida):
    v3 = le_csv(D / "etapa6" / "v3.csv")
    e4 = le_csv(D / "etapa4" / "resultados.csv")
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))
    ids3 = list(dict.fromkeys(l["id"] for l in v3))
    x = np.arange(len(ids3))
    med = [next(l for l in v3 if l["id"] == i) for i in ids3]
    r = np.array([float(m["medido"]) for m in med])
    ax[0].errorbar(x, r, yerr=[r - [float(m["medido_min"]) for m in med],
                               np.array([float(m["medido_max"]) for m in med]) - r],
                   fmt="ks", capsize=3, label="measured")
    ax[0].plot(x, [float(m["levaton_modelo"]) for m in med], "x", color="gray", ms=7, label="Levaton et al. model")
    for k, (q, cor, rot) in enumerate((("n2_amonia", "tab:red", "A"), ("n2_tese", "tab:blue", "B"),
                                        ("n2_exemplo", "tab:green", "C"))):
        ax[0].plot(x + 0.12 * (k - 1), [float(next(l for l in v3 if l["id"] == i and l["quimica"] == q)["previsto"])
                                        for i in ids3], "o", color=cor, mfc="none", label=rot)
    ax[0].set_yscale("log")
    ax[0].set_xticks(x)
    ax[0].set_xticklabels([ROTULOS[IDS.index(i)] for i in ids3], fontsize=7)
    ax[0].set_ylabel("N$_2$(B)/N$_2$(C)")
    ax[0].legend(fontsize=6.5)
    x = np.arange(len(IDS))
    for k, (q, cor, rot) in enumerate((("n2_amonia", "tab:red", "A"), ("n2_tese", "tab:blue", "B"),
                                        ("n2_exemplo", "tab:green", "C"))):
        ax[1].plot(x + 0.12 * (k - 1), [100 * float(next(l for l in e4 if l["id"] == i and l["quimica"] == q)["dis"])
                                        for i in IDS], "o", color=cor, mfc="none", label=rot)
    ax[1].plot([0, 2], [0.07, 0.2], "x", color="gray", ms=7, label="Levaton et al. model")
    ax[1].set_yscale("log")
    ax[1].set_xticks(x)
    ax[1].set_xticklabels([r.replace("\n", " ") for r in ROTULOS], fontsize=7, rotation=90)
    ax[1].set_ylabel("[N]/[N$_2$] (%)")
    ax[1].legend(fontsize=6.5)
    for a in ax:
        a.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(saida / "radiativos_dissociacao.pdf")
    plt.close(fig)


LIT = RAIZ / "docs" / "literatura" / "dados"


def le_lit(caminho):
    with open(caminho) as f:
        return np.array([[float(x) for x in l.split(",")] for l in f if l.strip() and not l.startswith("#")])


def figura_vdf(saida):
    """Vibrational distributions: (a, b) Levaton et al. fig. 11 (densities, three
    times) against chemistry A with the ladder up to v = 45 at their E/N and n_e;
    (c, d) Gordiets et al. fig. 13 (fractions, 0.1% O2, steady state) against the
    ladder with the two walls; coupled distributions at the self-consistent root
    where available."""
    vdf = le_csv(D / "pendencias" / "vdf.csv")
    fig, ax = plt.subplots(2, 2, figsize=(7.2, 5.6))
    cores_t = ("tab:blue", "tab:orange", "tab:red")
    for a, (cid, ma, tfim, tit) in zip(ax[0], (("L20_10mA", 10, "4.3e-4", "(a) L 10 mA"),
                                                ("L20_50mA", 50, "3.5e-4", "(b) L 50 mA"))):
        for t, cor in zip(("1e-6", "1e-5", tfim), cores_t):
            dig = le_lit(LIT / "levaton2020" / f"fig11_vdf_{ma}mA_t{t}.csv")
            a.plot(dig[:, 0], dig[:, 1], "o", color=cor, ms=3.5, mfc="none")
            mod = [l for l in vdf if l["id"] == cid and l["caso"] == "n2_amonia_v45" and float(l["t"]) == float(t)]
            a.semilogy([int(l["v"]) for l in mod], [float(l["n"]) for l in mod], "-", color=cor, lw=1,
                       label=f"t = {float(t):.0e} s".replace("e-0", "e-"))
        final = acoplado(cid)
        if final:
            n0 = next(float(l["n"]) / float(l["fracao"]) for l in vdf if l["id"] == cid and float(l["t"]) == float(tfim))
            a.semilogy(range(len(final["populacoes"])), n0 * np.array(final["populacoes"]), "--", color="k", lw=1,
                       label="coupled, root")
        a.set_ylim(1e10, 1e18)
        a.set_xlim(0, 15)
        a.set_ylabel("N$_2$($v$) (cm$^{-3}$)")
        a.set_title(tit, fontsize=9)
    ax[0, 0].legend(fontsize=7, title="lines: A, v $\\leq$ 45; circles: Levaton et al.", title_fontsize=7)
    for a, (cid, ma, tit) in zip(ax[1], (("G95_30mA", 30, "(c) G 30 mA"), ("G95_80mA", 80, "(d) G 80 mA"))):
        dig = le_lit(LIT / "gordiets1995" / f"fig13_vdf_{ma}mA_0.1pctO2.csv")
        a.semilogy(dig[:, 0], dig[:, 1], "ks", ms=3.5, mfc="none", label="Gordiets et al., 0.1% O$_2$")
        for (par, var, rot), cor in zip((("levaton", "base", "Levaton wall"), ("gordiets", "vib_gordiets", "Gordiets wall")),
                                        ("tab:purple", "tab:brown")):
            mod = [l for l in vdf if l["id"] == cid and l["parede"] == par and l["t"] != "t"]
            a.semilogy([int(l["v"]) for l in mod], [float(l["fracao"]) for l in mod], "-", color=cor, lw=1,
                       label=f"A, v $\\leq$ 45, {rot}")
            final = acoplado(cid, var)
            if final:
                a.semilogy(range(len(final["populacoes"])), final["populacoes"], "--", color=cor, lw=1,
                           label=f"coupled, {rot}")
        a.set_ylim(1e-6, 1)
        a.set_xlim(0, 45)
        a.set_ylabel("N$_2$($v$)/N$_2$")
        a.set_title(tit, fontsize=9)
    ax[1, 0].legend(fontsize=6.5)
    for a in ax.flat:
        a.set_xlabel("$v$")
        a.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(saida / "vdf.pdf")
    plt.close(fig)


def figura_kumar(saida):
    """Dissociation degree in the DC glow of Kumar and Ghosh 1991 (R = 0.6 cm, 1 to
    3 Torr, 30 to 80 mA), measured and calculated at the end of the residence time,
    bars for T_g from 350 to 450 K."""
    k = le_csv(D / "pendencias" / "kumar.csv")
    conds = list(dict.fromkeys((l["torr"], l["corrente_mA"]) for l in k))
    x = np.arange(len(conds))
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    med = np.array([100 * float(next(l for l in k if (l["torr"], l["corrente_mA"]) == c)["medido"]) for c in conds])
    ax.errorbar(x, med, yerr=[0 * med, 1.5 * med], fmt="ks", ms=6, capsize=3, label="measured (bar: 2.5$\\times$, residence)",
                zorder=5)
    casos = (("n2_amonia", "base", "A", "tab:red", "o"), ("n2_tese", "base", "B", "tab:blue", "o"),
             ("n2_exemplo", "base", "C", "tab:green", "o"),
             ("n2_amonia_v45", "base", "A, v $\\leq$ 45, Levaton wall", "tab:orange", "D"),
             ("n2_amonia_v45", "vib_gordiets", "A, v $\\leq$ 45, Gordiets wall", "tab:brown", "d"))
    for j, (q, var, rot, cor, m) in enumerate(casos):
        ls = [next(l for l in k if (l["torr"], l["corrente_mA"]) == c and l["quimica"] == q and l["variante"] == var)
              for c in conds]
        y = np.array([100 * float(l["dis_fim"]) for l in ls])
        lo = y - [100 * float(l["dis_fim_min"]) for l in ls]
        hi = np.array([100 * float(l["dis_fim_max"]) for l in ls]) - y
        ax.errorbar(x + 0.1 * (j - 1.5), y, yerr=[lo, hi], fmt=m, color=cor, ms=4.5, mfc="none", capsize=2, lw=0.8,
                    label=rot)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{float(t):.0f} Torr\n{float(i):.0f} mA" for t, i in conds], fontsize=8)
    ax.set_ylabel("[N]/[N$_2$] (%)")
    ax.set_ylim(1e-4, 10)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7, ncol=3, loc="lower center")
    fig.tight_layout()
    fig.savefig(saida / "kumar.pdf")
    plt.close(fig)


FS = 8.5      # labels of the schematic, pt at the printed size (width 153 mm)


def _tubo(ax, x0, x1, y, raio, corte=None):
    """Tube from x0 to x1; corte: x of a break mark (//) for an omitted length."""
    ax.add_patch(FancyBboxPatch((x0, y - raio), x1 - x0, 2 * raio, boxstyle="round,pad=0,rounding_size=0.15",
                                fc="#e6e6e6", ec="k", lw=1.0))
    if corte is not None:
        ax.add_patch(Rectangle((corte - 0.18, y - raio - 0.2), 0.36, 2 * raio + 0.4, fc="white", ec="none", zorder=3))
        for dx in (-0.18, 0.18):
            ax.plot([corte + dx - 0.12, corte + dx + 0.12], [y - raio - 0.25, y + raio + 0.25], color="k", lw=0.9,
                    zorder=4)


def _eletrodo(ax, x, y, raio, nome):
    ax.add_patch(Rectangle((x - 0.12, y + raio), 0.24, 0.5, fc="0.45", ec="k", lw=0.6))
    ax.text(x, y + raio + 0.6, nome, ha="center", va="bottom", fontsize=FS)


def _fluxo(ax, y):
    ax.annotate("", (-0.7, y), (-2.3, y), arrowprops=dict(arrowstyle="-|>", lw=1.2))
    ax.text(-2.3, y + 0.2, "N$_2$ in", fontsize=FS, va="bottom")
    ax.annotate("", (15.0, y), (13.7, y), arrowprops=dict(arrowstyle="-|>", lw=1.2))
    ax.text(13.75, y + 0.2, "pump", fontsize=FS, va="bottom")


def _sondas(ax, xs, y, raio):
    for x in xs:
        ax.plot([x, x], [y - raio - 0.15, y + raio + 0.15], color="k", lw=2.2)


def _fibra(ax, x, y, raio, texto, lado):
    ax.annotate("", (x, y - raio), (x, y - raio - 0.7), arrowprops=dict(arrowstyle="->", color="k", lw=1.0))
    ax.text(x + (0.2 if lado == "left" else -0.2), y - raio - 0.55, texto, ha=lado, va="center", fontsize=FS)


def figura_esquema(saida):
    """Schematic of the two experiments and of what the zero-dimensional model does.
    Levaton et al. 2020 (fig. 2 and section III): side-armed electrodes 12 cm apart,
    tube radius 0.3 cm, gas entering at the anode (t = 0), two double probes 5 cm
    apart in the middle, optical fibre near the cathode at the end of the column.
    Gordiets et al. 1995 (fig. 2 and section III): Pyrex tube of radius 0.8 cm, gas
    entering at electrode 1, microwave cavity, optical fibre and electrostatic probes
    downstream, far from the entrance where the column is steady, discharge between
    electrodes 1 and 3; distances and polarity not given. Drawn with the flow from left
    to right in both (mirror image of the original for Gordiets et al.); radii not to
    scale."""
    fig, ax = plt.subplots(figsize=(6.02, 5.6))
    ax.set_xlim(-3.2, 15.2)
    ax.set_ylim(-5.5, 9.6)
    ax.axis("off")
    # (a) Levaton et al., to scale along the axis (1 unit = 1 cm)
    y, r = 6.3, 0.35
    ax.text(-3.1, 9.5, "(a) Levaton et al.: $R$ = 0.3 cm, 230 and 530 Pa, 10 to 60 mA", fontsize=FS + 0.5, va="top")
    _tubo(ax, -0.8, 13.6, y, r)
    _eletrodo(ax, 0.0, y, r, "anode")
    _eletrodo(ax, 12.0, y, r, "cathode")
    _fluxo(ax, y)
    ax.annotate("", (0.0, y + r + 1.45), (12.0, y + r + 1.45), arrowprops=dict(arrowstyle="<->", lw=0.8))
    ax.text(6.0, y + r + 1.55, "12 cm", ha="center", fontsize=FS)
    _sondas(ax, (3.5, 8.5), y, r)
    ax.annotate("", (3.5, y + r + 0.4), (8.5, y + r + 0.4), arrowprops=dict(arrowstyle="<->", lw=0.8))
    ax.text(6.0, y + r + 0.5, "double probes, 5 cm: $E/N$ ($n_e$ from $I$)", ha="center", fontsize=FS)
    _fibra(ax, 11.3, y, r, "optical fibre: $T_g$, $T_v$, N$_2$(B), N$_2$(C), N$_2^+$(B)", "right")
    # (b) Gordiets et al., not to scale
    y2, r2 = 2.0, 0.75
    ax.text(-3.1, 4.75, "(b) Gordiets et al.: $R$ = 0.8 cm, 267 Pa, 15 to 80 mA (not to scale)", fontsize=FS + 0.5,
            va="top")
    _tubo(ax, -0.8, 13.6, y2, r2, corte=1.5)
    _eletrodo(ax, 0.0, y2, r2, "1")
    _eletrodo(ax, 12.0, y2, r2, "3")
    _fluxo(ax, y2)
    ax.add_patch(Rectangle((2.8, y2 - r2 - 0.2), 1.3, 2 * r2 + 0.4, fc="none", ec="k", lw=1.2, ls="--"))
    ax.text(3.45, y2 + r2 + 0.3, "microwave cavity: $n_e$", ha="center", va="bottom", fontsize=FS)
    _fibra(ax, 5.0, y2, r2, "fibre: $T_g$, $T_v$", "left")
    _sondas(ax, (7.5, 10.0), y2, r2)
    ax.text(8.75, y2 + r2 + 0.3, "electrostatic probes: $E/N$", ha="center", va="bottom", fontsize=FS)
    ax.annotate("", (2.8, y2 - r2 - 1.15), (10.0, y2 - r2 - 1.15), arrowprops=dict(arrowstyle="-", lw=0.8))
    for x in (2.8, 10.0):
        ax.plot([x, x], [y2 - r2 - 1.15, y2 - r2 - 1.0], color="k", lw=0.8)
    ax.text(6.4, y2 - r2 - 1.3, "steady state, far downstream of the entrance", ha="center", va="top", fontsize=FS)
    # (c) the model
    yc = -2.75
    ax.text(-3.1, -0.85, "(c) Model: a gas parcel in plug flow, radially averaged, along the residence time",
            fontsize=FS + 0.5, va="top")
    ax.add_patch(FancyBboxPatch((-0.6, yc - 0.8), 12.6, 1.6, boxstyle="round,pad=0.05", fc="#e6e6e6", ec="k",
                                lw=0.8))
    linhas = (r"$dn_i/dt = S_i - \nu_i n_i$,   $n_e = I/(e\mu_e E\pi R^2)$,   $T_g$ measured",
              r"$\nu_i$: ambipolar diffusion (ions), wall with $\gamma$ (N, N$_2(v)$, metastables)",
              r"electron balance at $t$:  $S_e = n_e\langle\nu\rangle_{ions}$")
    for k, texto in enumerate(linhas):
        ax.text(5.7, yc + 0.5 - 0.5 * k, texto, ha="center", va="center", fontsize=FS)
    ax.annotate("", (13.6, yc - 1.3), (0.0, yc - 1.3), arrowprops=dict(arrowstyle="-|>", lw=1.0))
    ax.text(13.75, yc - 1.3, "$t$", va="center", fontsize=FS + 1)
    for xt, rot in ((0.0, "0"), (6.0, "$t$/2: probes (a)"), (11.3, "$t$: optics (a)")):
        ax.plot([xt, xt], [yc - 1.42, yc - 1.18], color="k", lw=1)
        ax.text(xt, yc - 1.55, rot, ha="center", va="top", fontsize=FS)
    ax.text(5.7, yc - 2.2, "(b): run to 0.1 s, steady state, balance at the end", ha="center", va="top", fontsize=FS)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    fig.savefig(saida / "esquema.pdf")
    plt.close(fig)


def main():
    saida = pathlib.Path(sys.argv[1])
    saida.mkdir(parents=True, exist_ok=True)
    figura_esquema(saida)
    figura_en(saida)
    figura_tv(saida)
    figura_mecanismos(saida)
    figura_radiativos(saida)
    figura_vdf(saida)
    figura_kumar(saida)
    print(sorted(p.name for p in saida.glob("*.pdf")))


if __name__ == "__main__":
    main()
