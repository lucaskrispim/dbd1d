# SPDX-License-Identifier: GPL-3.0-or-later
"""V2 of validacao/LEIAME.md: amplitude scan of remoto/validacao_loffhagen.sh
against the ignition voltage of Loffhagen et al. 2020 (3.02 kV measured in pure
Ar, 2.87 to 3.17 kV; 3.65 kV in their model).

In each run, over the last complete period: the ratio of the current peak (I =
dQ/dt, Q = eps0 eps_d (V - phi_left) A / Delta) to the peak of the capacitive
current without plasma (C dV/dt, with C = eps0 A / (d + 2 Delta/eps_d)), the
charge transferred through the gas in a half period and the maximum n_e. Ignited:
ratio above 2. The ignition voltage is the lowest ignited amplitude; the
estimate between the last non-ignited and the first ignited one is given as a range.

Usage: python3 scripts/compara_loffhagen.py DIR [DIR ...]
"""
import json
import pathlib
import sys

import numpy as np

EPS0 = 8.8541878128e-12
AREA = 8e-4  # m2
LIMIAR = 2.0


def analisa(d):
    a = json.loads((d / "parametros.json").read_text())["argumentos"]
    per = a["periodo"]
    sup = np.loadtxt(d / "superficie.dat", skiprows=1, ndmin=2)
    t, v, phi_e, sig_e = sup[:, 0], sup[:, 1], sup[:, 2], sup[:, 4]
    q = EPS0 * a["eps_dieletrico"] * (v - phi_e) / a["espessura"] * AREA
    i = np.gradient(q, t)
    c = EPS0 * AREA / (a["gap"] + 2.0 * a["espessura"] / a["eps_dieletrico"])
    k = int(np.floor(t[-1] / per + 1e-9))
    m = (t >= (k - 1) * per - 1e-12) & (t <= k * per + 1e-12)
    i_cap = np.abs(np.gradient(c * v, t))[m].max()
    razao = np.abs(i[m]).max() / i_cap
    # charge transferred through the gas: excursion of the surface charge in the period
    dq = (sig_e[m].max() - sig_e[m].min()) * AREA
    return a["v_pico"], k, razao, dq


def main():
    linhas = sorted(analisa(pathlib.Path(d)) for d in sys.argv[1:])
    print(f"{'U0 (kV)':>8s} {'period':>7s} {'I/I_cap':>8s} {'dQ_gas (nC)':>12s}  state")
    for u0, k, r, dq in linhas:
        print(f"{u0 / 1e3:8.2f} {k:7d} {r:8.2f} {dq * 1e9:12.3f}  {'ignited' if r > LIMIAR else 'off'}")
    acesas = [u for u, _, r, _ in linhas if r > LIMIAR]
    apagadas = [u for u, _, r, _ in linhas if r <= LIMIAR and (not acesas or u < min(acesas))]
    if acesas:
        baixo = max(apagadas) / 1e3 if apagadas else float("nan")
        print(f"dbd1d ignites between {baixo:.2f} and {min(acesas) / 1e3:.2f} kV; measured 3.02 kV (2.87 to 3.17); "
              "model of Loffhagen 3.65 kV")
    else:
        print("no amplitude ignited; measured 3.02 kV")


if __name__ == "__main__":
    main()
