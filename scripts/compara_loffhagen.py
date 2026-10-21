# SPDX-License-Identifier: GPL-3.0-or-later
"""V2 de validacao/LEIAME.md: varredura de amplitude de remoto/validacao_loffhagen.sh
contra a tensão de ignição de Loffhagen et al. 2020 (3,02 kV medidos no Ar
puro, de 2,87 a 3,17 kV; 3,65 kV no modelo deles).

Em cada execução, no último período inteiro: a razão entre o pico da corrente
(I = dQ/dt, Q = ε₀ ε_d (V − φ_esq) A / Δ) e o pico da corrente capacitiva sem
plasma (C dV/dt, com C = ε₀ A / (d + 2Δ/ε_d)), a carga transferida pelo gás
num meio período e o n_e máximo. Acesa: razão acima de 2. A tensão de ignição
é a menor amplitude acesa; a estimativa entre a última apagada e a primeira
acesa vai como faixa.

Uso: python3 scripts/compara_loffhagen.py DIR [DIR ...]
"""
import json
import pathlib
import sys

import numpy as np

EPS0 = 8.8541878128e-12
AREA = 8e-4  # m²
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
    # carga transferida pelo gás: excursão da carga superficial no período
    dq = (sig_e[m].max() - sig_e[m].min()) * AREA
    return a["v_pico"], k, razao, dq


def main():
    linhas = sorted(analisa(pathlib.Path(d)) for d in sys.argv[1:])
    print(f"{'U0 (kV)':>8s} {'período':>7s} {'I/I_cap':>8s} {'ΔQ_gás (nC)':>12s}  estado")
    for u0, k, r, dq in linhas:
        print(f"{u0 / 1e3:8.2f} {k:7d} {r:8.2f} {dq * 1e9:12.3f}  {'acesa' if r > LIMIAR else 'apagada'}")
    acesas = [u for u, _, r, _ in linhas if r > LIMIAR]
    apagadas = [u for u, _, r, _ in linhas if r <= LIMIAR and (not acesas or u < min(acesas))]
    if acesas:
        baixo = max(apagadas) / 1e3 if apagadas else float("nan")
        print(f"ignição no 1D entre {baixo:.2f} e {min(acesas) / 1e3:.2f} kV; medida 3,02 kV (2,87 a 3,17); "
              "modelo de Loffhagen 3,65 kV")
    else:
        print("nenhuma amplitude acesa; medida 3,02 kV")


if __name__ == "__main__":
    main()
