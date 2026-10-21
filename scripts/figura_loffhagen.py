# SPDX-License-Identifier: GPL-3.0-or-later
"""Figure of V2 (ignition of Loffhagen et al. 2020): ratio of the current peak to
the peak of the capacitive current in the tenth period versus the amplitude, with
the measured ignition and that of their model, and the current at some amplitudes.

Usage: python3 scripts/figura_loffhagen.py RUNS_DIR OUTPUT.png
"""
import json
import pathlib
import sys

import numpy as np

EPS0 = 8.8541878128e-12
AREA = 8e-4


def main():
    base, saida = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4.8))
    us, rz = [], []
    for d in sorted(x for x in base.glob("L[0-9]*") if x.is_dir()):
        p = json.loads((d / "parametros.json").read_text())["argumentos"]
        s = np.loadtxt(d / "superficie.dat", skiprows=1)
        t, v, phi = s[:, 0], s[:, 1], s[:, 2]
        i = np.gradient(EPS0 * p["eps_dieletrico"] * (v - phi) / p["espessura"] * AREA, t)
        per = p["periodo"]
        m = t >= 9 * per - 1e-12
        c = EPS0 * AREA / (p["gap"] + 2 * p["espessura"] / p["eps_dieletrico"])
        us.append(p["v_pico"] / 1e3)
        rz.append(np.abs(i[m]).max() / np.abs(np.gradient(c * v, t))[m].max())
        if p["v_pico"] in (2800, 2850, 3000, 4000):
            b.plot((t[m] - 9 * per) * 1e6, i[m] * 1e3, label=f"U0 = {p['v_pico'] / 1e3:.2f} kV")
    o = np.argsort(us)
    a.plot(np.array(us)[o], np.array(rz)[o], "ko-", label="dbd1d (tenth period)")
    a.axvspan(2.865, 3.174, color="tab:orange", alpha=0.25, label="measured ignition (3.02 kV, bar)")
    a.axvline(3.649, color="tab:blue", ls="--", label="model of Loffhagen 2020 (3.65 kV)")
    a.axhline(2, color="gray", ls=":", lw=1)
    a.set_xlabel("amplitude U0 (kV)")
    a.set_ylabel("peak of I / peak of the capacitive current")
    a.set_title("ignition in argon at 1 atm (1 mm gap, 86.2 kHz)")
    b.set_xlabel("t in the tenth period (us)")
    b.set_ylabel("I (mA), area of 8 cm2")
    b.set_title("dbd1d current")
    for x in (a, b):
        x.grid(alpha=0.3)
        x.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(saida, dpi=110)


if __name__ == "__main__":
    main()
