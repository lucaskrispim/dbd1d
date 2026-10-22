# SPDX-License-Identifier: GPL-3.0-or-later
"""Figures of the article, from the validation runs in execucoes/validacao and
the data extracted in docs/literatura/dados.

Usage: python3 artigo/figuras.py [RUNS_DIR]   (writes to artigo/figuras/)
"""
import json
import pathlib
import sys

import numpy as np

RAIZ = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "scripts"))
from compara_stankov import potencias_por_periodo, ultimo_periodo  # noqa: E402
from detalha_stankov import ne_medio  # noqa: E402
import compara_yang  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DADOS = RAIZ / "docs" / "literatura" / "dados"
SAIDA = pathlib.Path(__file__).with_name("figuras")
EPS0 = 8.8541878128e-12
# style applied only by main, so as not to change matplotlib for whoever just imports the module
ESTILO = {"font.size": 9, "axes.grid": True, "grid.alpha": 0.3, "legend.fontsize": 7.5,
          "figure.dpi": 150, "savefig.bbox": "tight"}


def legenda(eixo, **kw):
    """Marks the panel to have its legend below, outside the data area (it is
    placed by fecha, once the size of the labels is known)."""
    eixo._legenda_embaixo = kw


def fecha(fig, saida):
    """Legends below the x labels of each panel, so as not to cover curves or
    labels, and the figure written. The distance is measured in points (height of
    the tick labels and of the axis label), which does not change with the layout."""
    from matplotlib.transforms import ScaledTranslation
    fig.tight_layout()
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    for eixo in fig.axes:
        kw = getattr(eixo, "_legenda_embaixo", None)
        if kw is None:
            continue
        abaixo = (eixo.get_window_extent(r).y0 - eixo.get_tightbbox(r).y0) * 72.0 / fig.dpi + 3.0
        desloca = eixo.transAxes + ScaledTranslation(0.0, -abaixo / 72.0, fig.dpi_scale_trans)
        # two columns when they fit within the width of the panel, otherwise one: wider legends
        # ran into the legend of the panel beside it
        for ncol in (2, 1):
            leg = eixo.legend(loc="upper center", bbox_to_anchor=(0.5, 0.0), bbox_transform=desloca,
                              borderaxespad=0.0, ncol=ncol, frameon=False, columnspacing=1.2, handlelength=1.8, **kw)
            if leg.get_window_extent(r).width <= eixo.get_window_extent(r).width:
                break
    fig.tight_layout()
    fig.savefig(saida)


def le(*partes):
    return np.loadtxt(DADOS.joinpath(*partes), delimiter=",", ndmin=2)


def stankov(base):
    st = base / "validacao_stankov"
    fig, e = plt.subplots(2, 2, figsize=(7.2, 6.8))
    med = le("stankov2020", "fig3_corrente_medida_100mbar.csv")
    e[0, 0].plot(med[:, 0], med[:, 1] * 1e3, "k", lw=0.8, label="measured [Stankov 2020]")
    for n, cor in ((100, "C0"), (400, "C3")):
        r = ultimo_periodo(st / f"S100_n{n}")
        e[0, 0].plot(r["t"] * 1e6, r["I"] * 1e3, cor, lw=0.9, label=f"dbd1d, {n} cells")
    if (st / "J100").is_dir():
        r = ultimo_periodo(st / "J100")
        e[0, 0].plot(r["t"] * 1e6, r["I"] * 1e3, "C2", lw=0.9, label="dbd1d, Jovanović coefficients")
    e[0, 0].set(xlabel="t (µs)", ylabel="I (mA)", title="(a) current, 100 mbar, 1.8 kV")
    legenda(e[0, 0])
    pres = [100, 300, 500, 650]
    f7 = le("stankov2020", "fig7_potencia.csv")
    e[0, 1].plot(f7[:, 0], f7[:, 1], "ks-", label="measured")
    e[0, 1].plot(f7[:, 0], f7[:, 4], "^--", color="gray", label=r"model of Stankov, $\gamma$ = 0.02")
    # mean of the last three periods and their range: at 300 mbar the discharge is not periodic
    for prefixo, sufixo, estilo, rot in (("S", "_n100", "C0o-", "dbd1d"),
                                         ("J", "", "C2v-", "dbd1d, Jovanović coefficients"),
                                         ("R", "_r07", "C1D-", r"dbd1d, $r_e$ = 0.7")):
        if all((st / f"{prefixo}{p}{sufixo}").is_dir() for p in pres):
            ps = np.array([potencias_por_periodo(st / f"{prefixo}{p}{sufixo}", 3) for p in pres])
            media = ps.mean(axis=1)
            e[0, 1].errorbar(pres, media, yerr=[media - ps.min(axis=1), ps.max(axis=1) - media], fmt=estilo,
                             ms=4, lw=0.8, capsize=2, label=rot)
    e[0, 1].set(xlabel="p (mbar)", ylabel="P (W)", title="(b) mean power")
    legenda(e[0, 1])
    gamas = [0.005, 0.01, 0.02, 0.04, 0.08]
    pg = [ultimo_periodo(st / ("S100_n100" if g == 0.02 else f"G{g}"))["P"] for g in gamas]
    e[1, 0].semilogx(gamas, pg, "C0o-", label="dbd1d")
    e[1, 0].semilogx(gamas, f7[0, 2:], "^--", color="gray", label="model of Stankov")
    e[1, 0].axhline(f7[0, 1], color="k", lw=0.8, label="measured")
    e[1, 0].set(xlabel=r"$\gamma$", ylabel="P (W)", title=r"(c) power vs $\gamma$, 100 mbar")
    legenda(e[1, 0])
    epss = [3.3, 3.75, 4.2, 4.65, 5.1]
    pe = [ultimo_periodo(st / ("S100_n100" if x == 4.2 else f"E{x}"))["P"] for x in epss]
    f12 = le("stankov2020", "fig12_potencia_eps.csv")
    e[1, 1].plot(epss, pe, "C0o-", label="dbd1d")
    e[1, 1].plot(epss, f12[0, 2:], "^--", color="gray", label="model of Stankov")
    e[1, 1].axhline(f12[0, 1], color="k", lw=0.8, label="measured")
    e[1, 1].set(xlabel=r"$\varepsilon_r$", ylabel="P (W)", title=r"(d) power vs $\varepsilon_r$, 100 mbar")
    legenda(e[1, 1])
    fecha(fig, SAIDA / "stankov.pdf")

    # gap voltage and electron density against their model
    d = st / "S100_n100"
    a = json.loads((d / "parametros.json").read_text())["argumentos"]
    per = a["periodo"]
    r = ultimo_periodo(d)
    tn, ne = ne_medio(d, (r["periodo"] - 1) * per, r["periodo"] * per)
    ug = le("stankov2020", "fig4_ugap_modelo_gama0.02.csv")
    nr = le("stankov2020", "fig4_ne_medio_modelo_gama0.02.csv")
    fig, e = plt.subplots(1, 2, figsize=(7.2, 3.5))
    e[0].plot(r["t"] * 1e6, r["Ugap"], "C0", label="dbd1d")
    if (st / "J100").is_dir():
        rj = ultimo_periodo(st / "J100")
        e[0].plot(rj["t"] * 1e6, rj["Ugap"], "C2", lw=0.9, label="dbd1d, Jovanović coefficients")
    e[0].plot(ug[:, 0], ug[:, 1], "k--", lw=0.9, label="model of Stankov")
    e[0].set(xlabel="t (µs)", ylabel=r"$U_{gap}$ (V)", title="(a) gap voltage")
    e[1].semilogy(tn * 1e6, ne, "C0", label="dbd1d")
    e[1].semilogy(nr[:, 0], nr[:, 1], "k--", lw=0.9, label="model of Stankov")
    e[1].set(xlabel="t (µs)", ylabel=r"$\bar n_e$ (m$^{-3}$)", title="(b) gap-averaged electron density")
    for x in e:
        legenda(x)
    fecha(fig, SAIDA / "stankov_detalhes.pdf")


def loffhagen(base):
    fig, e = plt.subplots(1, 2, figsize=(7.2, 3.5))
    series = (("L[0-9]*", "C0o-", r"dbd1d, $\gamma$ = 0.02"), ("LG0.01_*", "C1s--", r"$\gamma$ = 0.01"),
              ("LG0.08_*", "C3^--", r"$\gamma$ = 0.08"), ("LJ_*", "C2v-", "Jovanović coefficients"))
    for padrao, estilo, rot in series:
        us, rz = [], []
        for d in sorted(x for x in (base / "validacao_loffhagen").glob(padrao) if x.is_dir()):
            a = json.loads((d / "parametros.json").read_text())["argumentos"]
            s = np.loadtxt(d / "superficie.dat", skiprows=1, ndmin=2)
            t, v, phi = s[:, 0], s[:, 1], s[:, 2]
            per = a["periodo"]
            if t[-1] < 10 * per * (1 - 1e-9):
                continue  # run not finished
            i = np.gradient(EPS0 * a["eps_dieletrico"] * (v - phi) / a["espessura"] * 8e-4, t)
            m = t >= 9 * per - 1e-12
            c = EPS0 * 8e-4 / (a["gap"] + 2 * a["espessura"] / a["eps_dieletrico"])
            us.append(a["v_pico"] / 1e3)
            rz.append(np.abs(i[m]).max() / np.abs(np.gradient(c * v, t))[m].max())
            if padrao == "L[0-9]*" and a["v_pico"] in (2800, 2850, 3500):
                e[1].plot((t[m] - 9 * per) * 1e6, i[m] * 1e3, label=f"$U_0$ = {a['v_pico'] / 1e3:.2f} kV")
        if us:
            o = np.argsort(us)
            e[0].plot(np.array(us)[o], np.array(rz)[o], estilo, ms=3.5, lw=0.8, label=rot)
    e[0].axvspan(2.865, 3.174, color="tab:orange", alpha=0.3, label="measured ignition")
    e[0].axvline(3.649, color="gray", ls="--", label="model of Loffhagen")
    e[0].set(xlabel=r"$U_0$ (kV)", ylabel="peak current / capacitive peak", title="(a) ignition, Ar at 1 atm")
    e[1].set(xlabel="t (µs)", ylabel="I (mA)", title="(b) dbd1d current, 10th period")
    for x in e:
        legenda(x)
    fecha(fig, SAIDA / "loffhagen.pdf")


def takashima(base):
    tk = base / "validacao_takashima"
    res = []
    for d in sorted(x for x in tk.glob("T[0-9]*_p5") if x.is_dir()):
        a = json.loads((d / "parametros.json").read_text())["argumentos"]
        s = np.loadtxt(d / "superficie.dat", skiprows=1)
        q = EPS0 * a["eps_dieletrico"] * (s[:, 1] - s[:, 2]) / a["espessura"] * 9.1e-4
        res.append((a["pressao"] / 133.322, np.trapezoid(s[:, 1], q) * 1e3))
    res.sort()
    m40 = le("takashima2012", "fig8_energia_CPT_40kHz.csv")
    m20 = le("takashima2012", "fig8_energia_CPT_20kHz.csv")
    fig, e = plt.subplots(figsize=(3.6, 3.4))
    e.plot(m40[:, 0], m40[:, 1], "o", color="k", ms=3, label="measured, 40 kHz")
    e.plot(m20[:, 0], m20[:, 1], "s", color="gray", ms=3, label="measured, 20 kHz")
    # markers only, with a thin dashed line: there are no values between the computed pressures
    e.plot([x for x, _ in res], [y for _, y in res], "C0^--", lw=0.6, label="dbd1d, 5th pulse")
    e.set(xlabel="p (Torr)", ylabel="coupled energy per pulse (mJ)", title="dry air, ns pulses")
    legenda(e)
    fecha(fig, SAIDA / "takashima.pdf")


PULSOS_DO_ARTIGO = {"YN2_n50": 40, "YH2_n50": 25}


def yang(base):
    """Bursts of ns pulses in N2 and N2 + 1% H2 (Yang et al. 2022): what the runs of
    150 pulses have reached."""
    yg = base / "validacao_yang"
    per = compara_yang.PERIODO

    def series(nome):
        # the article shows the pulses computed when it was submitted, also when the run has more
        t, med, cen, tg = compara_yang.series(yg / nome)
        m = t <= PULSOS_DO_ARTIGO[nome] * per * (1 + 1e-9)
        return t[m], {k: v[m] for k, v in med.items()}, cen, tg[m]
    fim = 1e6 * max(series(x)[0][-1] for x in ("YN2_n50", "YH2_n50"))
    ate = 1.15 * fim  # window of the plots: what the runs reached and a little more
    fig, e = plt.subplots(2, 2, figsize=(7.2, 6.8))
    for nome, rot, cor, arq in (("YN2_n50", "N$_2$", "C0", "fig9_N_acumulado_N2.csv"),
                                ("YH2_n50", "N$_2$ + 1% H$_2$", "C3", "fig14_N_acumulado_1pcH2.csv")):
        t, med, _, tg = series(nome)
        n = np.arange(1, int(np.floor(t[-1] / per + 1e-9)) + 1)
        antes = [int(np.argmin(np.abs(t - k * per + 1e-6))) for k in n]
        m = le("yang2022", arq)
        m = m[m[:, 0] * per * 1e6 <= ate]
        e[0, 0].errorbar(m[:, 0], m[:, 1] / 1e15, yerr=[(m[:, 1] - m[:, 2]) / 1e15, (m[:, 3] - m[:, 1]) / 1e15],
                         fmt="o", ms=3, color=cor, mfc="none", capsize=2, label=f"measured, {rot}")
        e[0, 0].plot(n, med["N"][antes] / 1e15, "-", color=cor, label=f"dbd1d, {rot}")
        if compara_yang.mistura(yg / nome) != "N2":
            h = le("yang2022", "fig14_H_acumulado_1pcH2.csv")
            h = h[h[:, 0] <= ate]
            e[0, 1].plot(h[:, 0], h[:, 1] / 1e15, "o", ms=3, color=cor, mfc="none", label="measured")
            e[0, 1].plot(t * 1e6, med["H"] / 1e15, "-", color=cor, lw=0.8, label="dbd1d, gap average")
        if nome == "YN2_n50":
            v0 = le("yang2022", "fig7_n2a_v0_N2.csv")
            v1 = le("yang2022", "fig7_n2a_v1_N2.csv")
            soma = v0[:, 1] + np.interp(v0[:, 0], v1[:, 0], v1[:, 1])
            e[1, 0].plot(v0[:, 0], soma / 1e14, "k-", lw=0.8, label="measured, v = 0 + v = 1 (upper envelope)")
            e[1, 0].plot(t * 1e6, med["N2(A3)"] / 1e14, "-", color=cor, lw=0.6,
                         label="dbd1d, gap average (1 µs samples)")
            tm = le("yang2022", "fig8_temperatura_N2.csv")
            tm = tm[tm[:, 0] <= ate]
            tmod = le("yang2022", "fig8_temperatura_modelo_N2.csv")
            tmod = tmod[tmod[:, 0] <= ate]
            e[1, 1].plot(tm[:, 0], tm[:, 1], "ko", ms=2.5, mfc="none", label="measured")
            e[1, 1].plot(tmod[:, 0], tmod[:, 1], "--", color="gray", label="model of Yang")
            e[1, 1].plot(t * 1e6, tg, "-", color=cor, label="dbd1d, gap average")
    e[0, 0].set(xlabel="pulse number", ylabel=r"$n_\mathrm{N}$ (10$^{15}$ cm$^{-3}$)", title="(a) N atoms after each pulse")
    e[0, 1].set(xlabel="t (µs)", ylabel=r"$n_\mathrm{H}$ (10$^{15}$ cm$^{-3}$)", title=r"(b) H atoms, N$_2$ + 1% H$_2$")
    e[1, 0].set(xlabel="t (µs)", ylabel=r"$n_{\mathrm{N_2(A)}}$ (10$^{14}$ cm$^{-3}$)", title=r"(c) N$_2$(A), N$_2$")
    e[1, 1].set(xlabel="t (µs)", ylabel=r"$T_g$ (K)", title=r"(d) gas temperature, N$_2$")
    for x in e[0, 1], e[1, 0], e[1, 1]:
        x.set_xlim(0, ate)
    e[0, 0].set_xlim(0, ate / (per * 1e6))
    for x in e.flat:
        legenda(x)
    fecha(fig, SAIDA / "yang.pdf")


def rotulo_enxame(nome):
    """ellis_1976_medida -> Ellis 1976; townsend_bailey_1922_77K_medida ->
    Townsend and Bailey 1922, 77 K; escada_2007_calculada -> Escada 2007 (calc.)."""
    partes = nome.split("_")
    calc = partes[-1] == "calculada"
    partes = partes[:-1]
    i = next(k for k, p in enumerate(partes) if p.isdigit())
    autores = [p.capitalize() for p in partes[:i]]
    rot = " and ".join(autores) + " " + partes[i]
    if len(partes) > i + 1:
        rot += ", " + partes[i + 1][:-1] + " K"
    return rot + (" (calc.)" if calc else "")


def enxame():
    import glob
    t = np.loadtxt(RAIZ / "tabela_bolsig.dat", comments="#")
    en, te, mun, dn = t[:, 0], t[:, 1], t[:, 2], t[:, 3]
    calc = {"fig7": (mun * en * 1e-17 / 1e7, r"$v_d$ (10$^5$ m/s)"), "fig8": (dn / mun, r"$D/\mu$ (eV)"),
            "fig9": (mun * 100.0, r"$\mu N$ (m$^{-1}$V$^{-1}$s$^{-1}$)"),
            "fig10b": (1.5 * 8.617333262e-5 * te, r"$\langle\varepsilon\rangle$ (eV)")}
    fig, e = plt.subplots(2, 2, figsize=(7.2, 6.6))
    titulos = ["(a) drift velocity", "(b) characteristic energy", "(c) reduced mobility", "(d) mean energy"]
    for x, titulo, (chave, (y, rot)) in zip(e.flat, titulos, calc.items()):
        x.set_title(titulo)
        x.loglog(en, y, "k-", lw=1.2, label="BOLSIG+ table, 300 K")
        for arq in sorted(glob.glob(str(DADOS / "crispim2024" / f"{chave}_*.csv"))):
            d = np.loadtxt(arq, delimiter=",", ndmin=2)
            x.loglog(d[:, 0], d[:, 1], "o", ms=2.5, label=rotulo_enxame(pathlib.Path(arq).stem[len(chave) + 1:]))
        x.set(xlabel="E/N (Td)", ylabel=rot)
        legenda(x, fontsize=6.5)
    fecha(fig, SAIDA / "enxame.pdf")


def main():
    # folder with validacao_stankov, validacao_loffhagen, validacao_takashima and validacao_yang
    # (artigo/README.md lists the runs)
    base = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else RAIZ / "execucoes"
    SAIDA.mkdir(exist_ok=True)
    plt.rcParams.update(ESTILO)
    stankov(base)
    loffhagen(base)
    takashima(base)
    yang(base)
    enxame()
    print("figuras em", SAIDA)


if __name__ == "__main__":
    main()
