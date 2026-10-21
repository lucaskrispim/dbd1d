# SPDX-License-Identifier: GPL-3.0-or-later
"""Figuras do artigo (em inglês), a partir das execuções de validação em
execucoes/validacao e dos dados extraídos em docs/literatura/dados.

Uso: python3 artigo/figuras.py [DIR_DAS_EXECUCOES]   (grava em artigo/figuras/)
"""
import json
import pathlib
import sys

import numpy as np

RAIZ = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "scripts"))
from compara_stankov import ultimo_periodo  # noqa: E402
from detalha_stankov import ne_medio  # noqa: E402
import compara_yang  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DADOS = RAIZ / "docs" / "literatura" / "dados"
SAIDA = pathlib.Path(__file__).with_name("figuras")
EPS0 = 8.8541878128e-12
# estilo aplicado só por main, para não mudar o matplotlib de quem só importa o módulo
ESTILO = {"font.size": 9, "axes.grid": True, "grid.alpha": 0.3, "legend.fontsize": 7.5,
          "figure.dpi": 150, "savefig.bbox": "tight"}


def legenda(eixo, **kw):
    """Marca o painel para ter a legenda embaixo, fora da área dos dados (ela
    é posta por fecha, depois que o tamanho dos rótulos é conhecido)."""
    eixo._legenda_embaixo = kw


def fecha(fig, saida):
    """Legendas embaixo dos rótulos do eixo x de cada painel, para não cobrir
    curva nem rótulo, e a figura gravada. A distância é medida em pontos
    (altura dos números e do rótulo do eixo), que não muda com o layout."""
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
        eixo.legend(loc="upper center", bbox_to_anchor=(0.5, 0.0), bbox_transform=desloca, borderaxespad=0.0,
                    ncol=2, frameon=False, columnspacing=1.2, handlelength=1.8, **kw)
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
    e[0, 0].set(xlabel="t (µs)", ylabel="I (mA)", title="(a) current, 100 mbar, 1.8 kV")
    legenda(e[0, 0])
    pres = [100, 300, 500, 650]
    p1d = [ultimo_periodo(st / f"S{p}_n100")["P"] for p in pres]
    f7 = le("stankov2020", "fig7_potencia.csv")
    e[0, 1].plot(f7[:, 0], f7[:, 1], "ks-", label="measured")
    e[0, 1].plot(f7[:, 0], f7[:, 4], "^--", color="gray", label=r"model of Stankov, $\gamma$ = 0.02")
    e[0, 1].plot(pres, p1d, "C0o-", label="dbd1d")
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

    # tensão no gap e densidade de elétrons contra o modelo deles
    d = st / "S100_n100"
    a = json.loads((d / "parametros.json").read_text())["argumentos"]
    per = a["periodo"]
    r = ultimo_periodo(d)
    tn, ne = ne_medio(d, (r["periodo"] - 1) * per, r["periodo"] * per)
    ug = le("stankov2020", "fig4_ugap_modelo_gama0.02.csv")
    nr = le("stankov2020", "fig4_ne_medio_modelo_gama0.02.csv")
    fig, e = plt.subplots(1, 2, figsize=(7.2, 3.5))
    e[0].plot(r["t"] * 1e6, r["Ugap"], "C0", label="dbd1d")
    e[0].plot(ug[:, 0], ug[:, 1], "k--", lw=0.9, label="model of Stankov")
    e[0].set(xlabel="t (µs)", ylabel=r"$U_{gap}$ (V)", title="(a) gap voltage")
    e[1].semilogy(tn * 1e6, ne, "C0", label="dbd1d")
    e[1].semilogy(nr[:, 0], nr[:, 1], "k--", lw=0.9, label="model of Stankov")
    e[1].set(xlabel="t (µs)", ylabel=r"$\bar n_e$ (m$^{-3}$)", title="(b) gap-averaged electron density")
    for x in e:
        legenda(x)
    fecha(fig, SAIDA / "stankov_detalhes.pdf")


def loffhagen(base):
    us, rz = [], []
    fig, e = plt.subplots(1, 2, figsize=(7.2, 3.5))
    for d in sorted(x for x in (base / "validacao_loffhagen").glob("L[0-9]*") if x.is_dir()):
        a = json.loads((d / "parametros.json").read_text())["argumentos"]
        s = np.loadtxt(d / "superficie.dat", skiprows=1)
        t, v, phi = s[:, 0], s[:, 1], s[:, 2]
        i = np.gradient(EPS0 * a["eps_dieletrico"] * (v - phi) / a["espessura"] * 8e-4, t)
        per = a["periodo"]
        m = t >= 9 * per - 1e-12
        c = EPS0 * 8e-4 / (a["gap"] + 2 * a["espessura"] / a["eps_dieletrico"])
        us.append(a["v_pico"] / 1e3)
        rz.append(np.abs(i[m]).max() / np.abs(np.gradient(c * v, t))[m].max())
        if a["v_pico"] in (2800, 2850, 3500):
            e[1].plot((t[m] - 9 * per) * 1e6, i[m] * 1e3, label=f"$U_0$ = {a['v_pico'] / 1e3:.2f} kV")
    o = np.argsort(us)
    e[0].plot(np.array(us)[o], np.array(rz)[o], "C0o-", ms=3.5, label="dbd1d, 10th period")
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
    for d in sorted(x for x in tk.glob("T*_p5") if x.is_dir()):
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
    # só marcadores, com tracejado fino: entre as pressões calculadas não há valores
    e.plot([x for x, _ in res], [y for _, y in res], "C0^--", lw=0.6, label="dbd1d, 5th pulse")
    e.set(xlabel="p (Torr)", ylabel="coupled energy per pulse (mJ)", title="dry air, ns pulses")
    legenda(e)
    fecha(fig, SAIDA / "takashima.pdf")


def yang(base):
    """Rajadas de pulsos de ns em N2 e N2 + 1% H2 (Yang et al. 2022): o que as
    execuções de 150 pulsos já alcançaram."""
    yg = base / "validacao_yang"
    per = compara_yang.PERIODO
    fim = 1e6 * max(compara_yang.series(yg / x)[0][-1] for x in ("YN2_n50", "YH2_n50"))
    ate = 1.15 * fim  # janela dos gráficos: o que as execuções alcançaram e um pouco mais
    fig, e = plt.subplots(2, 2, figsize=(7.2, 6.8))
    for nome, rot, cor, arq in (("YN2_n50", "N$_2$", "C0", "fig9_N_acumulado_N2.csv"),
                                ("YH2_n50", "N$_2$ + 1% H$_2$", "C3", "fig14_N_acumulado_1pcH2.csv")):
        t, med, _, tg = compara_yang.series(yg / nome)
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
    base = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else RAIZ / "execucoes" / "validacao"
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
