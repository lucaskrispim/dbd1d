# SPDX-License-Identifier: GPL-3.0-or-later
"""Figura do V2 (ignição de Loffhagen et al. 2020): razão entre o pico de
corrente e o pico da corrente capacitiva no décimo período contra a amplitude,
com a ignição medida e a do modelo deles, e a corrente em algumas amplitudes.

Uso: python3 scripts/figura_loffhagen.py DIR_DAS_EXECUCOES SAIDA.png
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
            b.plot((t[m] - 9 * per) * 1e6, i[m] * 1e3, label=f"U₀ = {p['v_pico'] / 1e3:.2f} kV")
    o = np.argsort(us)
    a.plot(np.array(us)[o], np.array(rz)[o], "ko-", label="1D (décimo período)")
    a.axvspan(2.865, 3.174, color="tab:orange", alpha=0.25, label="ignição medida (3,02 kV, barra)")
    a.axvline(3.649, color="tab:blue", ls="--", label="modelo de Loffhagen 2020 (3,65 kV)")
    a.axhline(2, color="gray", ls=":", lw=1)
    a.set_xlabel("amplitude U₀ (kV)")
    a.set_ylabel("pico de I / pico da corrente capacitiva")
    a.set_title("Ignição em argônio a 1 atm (gap 1 mm, 86,2 kHz)")
    b.set_xlabel("t no décimo período (µs)")
    b.set_ylabel("I (mA), área de 8 cm²")
    b.set_title("Corrente no 1D")
    for x in (a, b):
        x.grid(alpha=0.3)
        x.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(saida, dpi=110)


if __name__ == "__main__":
    main()
