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

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DADOS = RAIZ / "docs" / "literatura" / "dados"
SAIDA = pathlib.Path(__file__).with_name("figuras")
EPS0 = 8.8541878128e-12
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3, "legend.fontsize": 7.5,
                     "figure.dpi": 150, "savefig.bbox": "tight"})


def le(*partes):
    return np.loadtxt(DADOS.joinpath(*partes), delimiter=",", ndmin=2)


def stankov(base):
    st = base / "validacao_stankov"
    fig, e = plt.subplots(2, 2, figsize=(7.2, 5.6))
    med = le("stankov2020", "fig3_corrente_medida_100mbar.csv")
    e[0, 0].plot(med[:, 0], med[:, 1] * 1e3, "k", lw=0.8, label="measured [Stankov 2020]")
    for n, cor in ((100, "C0"), (400, "C3")):
        r = ultimo_periodo(st / f"S100_n{n}")
        e[0, 0].plot(r["t"] * 1e6, r["I"] * 1e3, cor, lw=0.9, label=f"1D, {n} cells")
    e[0, 0].set(xlabel="t (µs)", ylabel="I (mA)", title="(a) current, 100 mbar, 1.8 kV")
    e[0, 0].legend()
    pres = [100, 300, 500, 650]
    p1d = [ultimo_periodo(st / f"S{p}_n100")["P"] for p in pres]
    f7 = le("stankov2020", "fig7_potencia.csv")
    e[0, 1].plot(f7[:, 0], f7[:, 1], "ks-", label="measured")
    e[0, 1].plot(f7[:, 0], f7[:, 4], "^--", color="gray", label=r"model of Stankov, $\gamma$ = 0.02")
    e[0, 1].plot(pres, p1d, "C0o-", label="1D")
    e[0, 1].set(xlabel="p (mbar)", ylabel="P (W)", title="(b) mean power")
    e[0, 1].legend()
    gamas = [0.005, 0.01, 0.02, 0.04, 0.08]
    pg = [ultimo_periodo(st / ("S100_n100" if g == 0.02 else f"G{g}"))["P"] for g in gamas]
    e[1, 0].semilogx(gamas, pg, "C0o-", label="1D")
    e[1, 0].semilogx(gamas, f7[0, 2:], "^--", color="gray", label="model of Stankov")
    e[1, 0].axhline(f7[0, 1], color="k", lw=0.8, label="measured")
    e[1, 0].set(xlabel=r"$\gamma$", ylabel="P (W)", title=r"(c) power vs $\gamma$, 100 mbar")
    e[1, 0].legend()
    epss = [3.3, 3.75, 4.2, 4.65, 5.1]
    pe = [ultimo_periodo(st / ("S100_n100" if x == 4.2 else f"E{x}"))["P"] for x in epss]
    f12 = le("stankov2020", "fig12_potencia_eps.csv")
    e[1, 1].plot(epss, pe, "C0o-", label="1D")
    e[1, 1].plot(epss, f12[0, 2:], "^--", color="gray", label="model of Stankov")
    e[1, 1].axhline(f12[0, 1], color="k", lw=0.8, label="measured")
    e[1, 1].set(xlabel=r"$\varepsilon_r$", ylabel="P (W)", title=r"(d) power vs $\varepsilon_r$, 100 mbar")
    e[1, 1].legend()
    fig.tight_layout()
    fig.savefig(SAIDA / "stankov.pdf")

    # tensão no gap e densidade de elétrons contra o modelo deles
    d = st / "S100_n100"
    a = json.loads((d / "parametros.json").read_text())["argumentos"]
    per = a["periodo"]
    r = ultimo_periodo(d)
    tn, ne = ne_medio(d, (r["periodo"] - 1) * per, r["periodo"] * per)
    ug = le("stankov2020", "fig4_ugap_modelo_gama0.02.csv")
    nr = le("stankov2020", "fig4_ne_medio_modelo_gama0.02.csv")
    fig, e = plt.subplots(1, 2, figsize=(7.2, 2.8))
    e[0].plot(r["t"] * 1e6, r["Ugap"], "C0", label="1D")
    e[0].plot(ug[:, 0], ug[:, 1], "k--", lw=0.9, label="model of Stankov")
    e[0].set(xlabel="t (µs)", ylabel=r"$U_{gap}$ (V)", title="(a) gap voltage")
    e[1].semilogy(tn * 1e6, ne, "C0", label="1D")
    e[1].semilogy(nr[:, 0], nr[:, 1], "k--", lw=0.9, label="model of Stankov")
    e[1].set(xlabel="t (µs)", ylabel=r"$\bar n_e$ (m$^{-3}$)", title="(b) gap-averaged electron density")
    for x in e:
        x.legend()
    fig.tight_layout()
    fig.savefig(SAIDA / "stankov_detalhes.pdf")


def loffhagen(base):
    us, rz = [], []
    fig, e = plt.subplots(1, 2, figsize=(7.2, 2.8))
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
    e[0].plot(np.array(us)[o], np.array(rz)[o], "C0o-", ms=3.5, label="1D, 10th period")
    e[0].axvspan(2.865, 3.174, color="tab:orange", alpha=0.3, label="measured ignition")
    e[0].axvline(3.649, color="gray", ls="--", label="model of Loffhagen")
    e[0].set(xlabel=r"$U_0$ (kV)", ylabel="peak current / capacitive peak", title="(a) ignition, Ar at 1 atm")
    e[1].set(xlabel="t (µs)", ylabel="I (mA)", title="(b) 1D current, 10th period")
    for x in e:
        x.legend()
    fig.tight_layout()
    fig.savefig(SAIDA / "loffhagen.pdf")


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
    fig, e = plt.subplots(figsize=(3.6, 2.8))
    e.plot(m40[:, 0], m40[:, 1], "o", color="k", ms=3, label="measured, 40 kHz")
    e.plot(m20[:, 0], m20[:, 1], "s", color="gray", ms=3, label="measured, 20 kHz")
    e.plot([x for x, _ in res], [y for _, y in res], "C0^-", label="1D, 5th pulse")
    e.set(xlabel="p (Torr)", ylabel="coupled energy per pulse (mJ)", title="dry air, ns pulses")
    e.legend()
    fig.tight_layout()
    fig.savefig(SAIDA / "takashima.pdf")


def enxame():
    import glob
    t = np.loadtxt(RAIZ / "tabela_bolsig.dat", comments="#")
    en, te, mun, dn = t[:, 0], t[:, 1], t[:, 2], t[:, 3]
    calc = {"fig7": (mun * en * 1e-17 / 1e7, r"$v_d$ (10$^5$ m/s)"), "fig8": (dn / mun, r"$D/\mu$ (eV)"),
            "fig9": (mun * 100.0, r"$\mu N$ (m$^{-1}$V$^{-1}$s$^{-1}$)"),
            "fig10b": (1.5 * 8.617333262e-5 * te, r"$\langle\varepsilon\rangle$ (eV)")}
    fig, e = plt.subplots(2, 2, figsize=(7.2, 5.4))
    for x, (chave, (y, rot)) in zip(e.flat, calc.items()):
        x.loglog(en, y, "k-", lw=1.2, label="BOLSIG+ table, 300 K")
        for arq in sorted(glob.glob(str(DADOS / "crispim2024" / f"{chave}_*.csv"))):
            d = np.loadtxt(arq, delimiter=",", ndmin=2)
            nome = pathlib.Path(arq).stem[len(chave) + 1:].replace("_medida", "").replace("_calculada", " (calc.)")
            x.loglog(d[:, 0], d[:, 1], "o", ms=2.5, label=nome.replace("_", " "))
        x.set(xlabel="E/N (Td)", ylabel=rot)
        x.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(SAIDA / "enxame.pdf")


def main():
    base = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else RAIZ / "execucoes" / "validacao"
    SAIDA.mkdir(exist_ok=True)
    stankov(base)
    loffhagen(base)
    takashima(base)
    enxame()
    print("figuras em", SAIDA)


if __name__ == "__main__":
    main()
