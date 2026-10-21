# SPDX-License-Identifier: GPL-3.0-or-later
"""V5 of validacao/LEIAME.md: energy coupled in the fifth pulse (restarts of
remoto/validacao_takashima_pulso.sh or scripts/retoma_fina.py) versus pressure,
compared with Takashima et al. 2012 (fig. 8, CPT generator, 20 and 40 kHz).
E = integral of V dQ over the pulse, with Q = eps0 eps_d (V - phi_left) A / Delta
and A = 9.1 cm2 (electrodes of 14 mm x 65 mm).

Usage: python3 scripts/compara_takashima.py DIR_OF_THE_RESTARTS [OUTPUT.png]
"""
import json
import pathlib
import sys

import numpy as np

RAIZ = pathlib.Path(__file__).resolve().parent.parent
DADOS = RAIZ / "docs" / "literatura" / "dados" / "takashima2012"
EPS0 = 8.8541878128e-12
AREA = 9.1e-4
TORR = 133.322


def energia(d):
    a = json.loads((d / "parametros.json").read_text())["argumentos"]
    s = np.loadtxt(d / "superficie.dat", skiprows=1, ndmin=2)
    q = EPS0 * a["eps_dieletrico"] * (s[:, 1] - s[:, 2]) / a["espessura"] * AREA
    return a["pressao"] / TORR, float(np.trapezoid(s[:, 1], q)), s


def main():
    base = pathlib.Path(sys.argv[1])
    res = sorted(energia(d) for d in base.glob("T*_p5") if d.is_dir())
    m40 = np.loadtxt(DADOS / "fig8_energia_CPT_40kHz.csv", delimiter=",")
    m20 = np.loadtxt(DADOS / "fig8_energia_CPT_20kHz.csv", delimiter=",")
    print(f"{'p (Torr)':>8s} {'E dbd1d (mJ)':>12s} {'E meas. 40 kHz (mJ)':>20s}")
    for p, e, _ in res:
        perto = np.abs(m40[:, 0] - p) < 0.1 * p + 3
        med = f"{m40[perto, 1].mean():.2f}" if perto.any() else "-"
        print(f"{p:8.0f} {e * 1e3:12.2f} {med:>20s}")
    if len(sys.argv) < 3:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4.8))
    a.plot(m40[:, 0], m40[:, 1], "o", color="tab:red", ms=4, label="measured, 40 kHz (fig. 8)")
    a.plot(m20[:, 0], m20[:, 1], "s", color="tab:blue", ms=4, label="measured, 20 kHz (fig. 8)")
    a.plot([r[0] for r in res], [r[1] * 1e3 for r in res], "k^-", label="dbd1d, fifth pulse at 40 kHz")
    a.set_xlabel("pressure (Torr)")
    a.set_ylabel("coupled energy per pulse (mJ)")
    a.set_title("dry air, CPT generator (Takashima 2012)")
    for p, e, s in res:
        b.plot(s[:, 0] * 1e9 - s[0, 0] * 1e9, s[:, 1] / 1e3, lw=0.8, label=f"V, {p:.0f} Torr" if p == res[0][0] else None)
        break
    for p, e, s in res:
        a_ = json.loads((base / f"T{p:.0f}_p5" / "parametros.json").read_text())["argumentos"]
        q = EPS0 * a_["eps_dieletrico"] * (s[:, 1] - s[:, 2]) / a_["espessura"] * AREA
        b.plot(s[:, 0] * 1e9 - s[0, 0] * 1e9, np.concatenate([[0], np.cumsum(0.5 * (s[1:, 1] + s[:-1, 1]) * np.diff(q))]) * 1e3,
               label=f"cumulative energy, {p:.0f} Torr")
    b.set_xlabel("t in the pulse (ns)")
    b.set_ylabel("V (kV) and energy (mJ)")
    b.set_title("fifth pulse in dbd1d")
    for x in (a, b):
        x.grid(alpha=0.3)
        x.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(sys.argv[2], dpi=110)


if __name__ == "__main__":
    main()
