# SPDX-License-Identifier: GPL-3.0-or-later
"""Scans of gamma and eps_r of V1 (100 mbar, 1.8 kV) against the model of Stankov
et al. 2020 (figs. 7 and 12, tables 2 and 3) and the measured power, and the mesh
convergence of the reference condition.

Usage: python3 scripts/figura_varreduras_stankov.py RUNS_DIR OUTPUT.png
  (RAIZ_DAS_EXECUCOES tem G0.005 ... G0.08, E3.3 ... E5.1, S100_n100, S100_n200, S100_n400)
"""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from compara_stankov import DADOS, ultimo_periodo  # noqa: E402


def main():
    base, saida = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    gamas = [0.005, 0.01, 0.02, 0.04, 0.08]
    epss = [3.3, 3.75, 4.2, 4.65, 5.1]
    rg = [ultimo_periodo(base / ("S100_n100" if g == 0.02 else f"G{g}")) for g in gamas]
    re_ = [ultimo_periodo(base / ("S100_n100" if e == 4.2 else f"E{e}")) for e in epss]
    rm = {n: ultimo_periodo(base / f"S100_n{n}") for n in (100, 200, 400)}
    f7 = np.loadtxt(DADOS / "fig7_potencia.csv", delimiter=",")[0]   # 100 mbar: measured and five gamma values
    f12 = np.loadtxt(DADOS / "fig12_potencia_eps.csv", delimiter=",")[0]
    t2 = np.loadtxt(DADOS / "tabela2_gama.csv", delimiter=",")
    t3 = np.loadtxt(DADOS / "tabela3_eps.csv", delimiter=",")
    ruptura = lambda r: max(r["rupturas"], key=lambda x: x[2])  # noqa: E731
    print(f"{'gamma':>6s} {'P dbd1d':>7s} {'P Stankov':>9s} {'U_bd':>9s} {'U_bd St.':>10s} {'n_e max':>10s} {'St.':>8s}")
    for g, r, ps, row in zip(gamas, rg, f7[2:], t2):
        print(f"{g:6.3f} {r['P']:7.3f} {ps:9.3f} {ruptura(r)[1]:9.0f} {row[1]:10.0f} {r['ne_max']:10.2e} {row[3] * 1e16:8.2e}")
    print(f"{'eps_r':>6s}")
    for e, r, ps, row in zip(epss, re_, f12[2:], t3):
        print(f"{e:6.2f} {r['P']:7.3f} {ps:9.3f} {ruptura(r)[1]:9.0f} {row[1]:10.0f} {r['ne_max']:10.2e} {row[3] * 1e16:8.2e}")
    print(f"measured: {f7[1]:.3f} W")
    print("mesh: " + "; ".join(f"{n} cells P = {r['P']:.4f} W, U_bd = {ruptura(r)[1]:.0f} V, "
                               f"t peak = {ruptura(r)[0] * 1e6:.2f} us" for n, r in rm.items()))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, e = plt.subplots(2, 3, figsize=(16, 9))
    e[0, 0].semilogx(gamas, [r["P"] for r in rg], "C0o-", label="dbd1d")
    e[0, 0].semilogx(gamas, f7[2:], "k^--", label="model of Stankov (fig. 7)")
    e[0, 0].axhline(f7[1], color="tab:red", label="measured")
    e[0, 0].set_xlabel(r"$\gamma$")
    e[0, 0].set_ylabel("P (W)")
    e[0, 0].set_title(r"power versus $\gamma$ ($\varepsilon_r$ = 4.2)")
    e[0, 1].semilogx(gamas, [ruptura(r)[1] for r in rg], "C0o-", label="dbd1d: U_gap at the current peak")
    e[0, 1].semilogx(gamas, t2[:, 1], "k^--", label="Stankov (table 2)")
    e[0, 1].set_xlabel(r"$\gamma$")
    e[0, 1].set_ylabel("breakdown voltage (V)")
    e[0, 2].semilogx(gamas, [r["ne_max"] for r in rg], "C0o-", label="dbd1d")
    e[0, 2].semilogx(gamas, t2[:, 3] * 1e16, "k^--", label="Stankov (table 2)")
    e[0, 2].set_xlabel(r"$\gamma$")
    e[0, 2].set_ylabel("maximum gap-averaged n_e (m-3)")
    e[1, 0].plot(epss, [r["P"] for r in re_], "C0o-", label="dbd1d")
    e[1, 0].plot(epss, f12[2:], "k^--", label="model of Stankov (fig. 12)")
    e[1, 0].axhline(f12[1], color="tab:red", label="measured")
    e[1, 0].set_xlabel(r"$\varepsilon_r$")
    e[1, 0].set_ylabel("P (W)")
    e[1, 0].set_title(r"power versus $\varepsilon_r$ ($\gamma$ = 0.02)")
    e[1, 1].plot(epss, [ruptura(r)[1] for r in re_], "C0o-", label="dbd1d")
    e[1, 1].plot(epss, t3[:, 1], "k^--", label="Stankov (table 3)")
    e[1, 1].set_xlabel(r"$\varepsilon_r$")
    e[1, 1].set_ylabel("breakdown voltage (V)")
    for n, r in rm.items():
        e[1, 2].plot(r["t"] * 1e6, r["I"] * 1e3, lw=1, label=f"{n} cells, P = {r['P']:.3f} W")
    e[1, 2].set_xlim(18.5, 21.5)
    e[1, 2].set_xlabel("t in the tenth period (us)")
    e[1, 2].set_ylabel("I (mA)")
    e[1, 2].set_title("mesh convergence, current peak")
    for x in e.flat:
        x.grid(alpha=0.3)
        x.legend(fontsize=8)
    fig.suptitle(r"Stankov 2020, 100 mbar and 1.8 kV: scans of $\gamma$ and $\varepsilon_r$", fontsize=11)
    fig.tight_layout()
    fig.savefig(saida, dpi=100)


if __name__ == "__main__":
    main()
