# SPDX-License-Identifier: GPL-3.0-or-later
"""Varreduras de γ e de ε_r do V1 (100 mbar, 1,8 kV) contra o modelo de Stankov
et al. 2020 (figs. 7 e 12, tabelas 2 e 3) e a potência medida, e a
convergência em malha da condição de referência.

Uso: python3 scripts/figura_varreduras_stankov.py RAIZ_DAS_EXECUCOES SAIDA.png
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
    f7 = np.loadtxt(DADOS / "fig7_potencia.csv", delimiter=",")[0]   # 100 mbar: medida e cinco γ
    f12 = np.loadtxt(DADOS / "fig12_potencia_eps.csv", delimiter=",")[0]
    t2 = np.loadtxt(DADOS / "tabela2_gama.csv", delimiter=",")
    t3 = np.loadtxt(DADOS / "tabela3_eps.csv", delimiter=",")
    ruptura = lambda r: max(r["rupturas"], key=lambda x: x[2])  # noqa: E731
    print(f"{'γ':>6s} {'P 1D':>7s} {'P Stankov':>9s} {'U_rupt 1D':>9s} {'U_rupt St.':>10s} {'n_e máx 1D':>10s} {'St.':>8s}")
    for g, r, ps, row in zip(gamas, rg, f7[2:], t2):
        print(f"{g:6.3f} {r['P']:7.3f} {ps:9.3f} {ruptura(r)[1]:9.0f} {row[1]:10.0f} {r['ne_max']:10.2e} {row[3] * 1e16:8.2e}")
    print(f"{'ε_r':>6s}")
    for e, r, ps, row in zip(epss, re_, f12[2:], t3):
        print(f"{e:6.2f} {r['P']:7.3f} {ps:9.3f} {ruptura(r)[1]:9.0f} {row[1]:10.0f} {r['ne_max']:10.2e} {row[3] * 1e16:8.2e}")
    print(f"medida: {f7[1]:.3f} W")
    print("malha: " + "; ".join(f"{n} células P = {r['P']:.4f} W, U_rupt = {ruptura(r)[1]:.0f} V, "
                                f"t pico = {ruptura(r)[0] * 1e6:.2f} µs" for n, r in rm.items()))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, e = plt.subplots(2, 3, figsize=(16, 9))
    e[0, 0].semilogx(gamas, [r["P"] for r in rg], "C0o-", label="1D")
    e[0, 0].semilogx(gamas, f7[2:], "k^--", label="modelo de Stankov (fig. 7)")
    e[0, 0].axhline(f7[1], color="tab:red", label="medida")
    e[0, 0].set_xlabel("γ")
    e[0, 0].set_ylabel("P (W)")
    e[0, 0].set_title("potência contra γ (ε_r = 4,2)")
    e[0, 1].semilogx(gamas, [ruptura(r)[1] for r in rg], "C0o-", label="1D: U_gap no pico de corrente")
    e[0, 1].semilogx(gamas, t2[:, 1], "k^--", label="Stankov (tabela 2)")
    e[0, 1].set_xlabel("γ")
    e[0, 1].set_ylabel("tensão de ruptura (V)")
    e[0, 2].semilogx(gamas, [r["ne_max"] for r in rg], "C0o-", label="1D")
    e[0, 2].semilogx(gamas, t2[:, 3] * 1e16, "k^--", label="Stankov (tabela 2)")
    e[0, 2].set_xlabel("γ")
    e[0, 2].set_ylabel("n̄_e máximo (m⁻³)")
    e[1, 0].plot(epss, [r["P"] for r in re_], "C0o-", label="1D")
    e[1, 0].plot(epss, f12[2:], "k^--", label="modelo de Stankov (fig. 12)")
    e[1, 0].axhline(f12[1], color="tab:red", label="medida")
    e[1, 0].set_xlabel("ε_r")
    e[1, 0].set_ylabel("P (W)")
    e[1, 0].set_title("potência contra ε_r (γ = 0,02)")
    e[1, 1].plot(epss, [ruptura(r)[1] for r in re_], "C0o-", label="1D")
    e[1, 1].plot(epss, t3[:, 1], "k^--", label="Stankov (tabela 3)")
    e[1, 1].set_xlabel("ε_r")
    e[1, 1].set_ylabel("tensão de ruptura (V)")
    for n, r in rm.items():
        e[1, 2].plot(r["t"] * 1e6, r["I"] * 1e3, lw=1, label=f"{n} células, P = {r['P']:.3f} W")
    e[1, 2].set_xlim(18.5, 21.5)
    e[1, 2].set_xlabel("t no décimo período (µs)")
    e[1, 2].set_ylabel("I (mA)")
    e[1, 2].set_title("convergência em malha, pico de corrente")
    for x in e.flat:
        x.grid(alpha=0.3)
        x.legend(fontsize=8)
    fig.suptitle("Stankov 2020, 100 mbar e 1,8 kV: varreduras de γ e de ε_r", fontsize=11)
    fig.tight_layout()
    fig.savefig(saida, dpi=100)


if __name__ == "__main__":
    main()
