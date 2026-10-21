# SPDX-License-Identifier: GPL-3.0-or-later
"""V3 of validacao/LEIAME.md: runs of remoto/validacao_yang.sh (and the restarts of
scripts/retoma_fina.py) against Yang et al. 2022 (docs/literatura/dados/yang2022/).

From the main run (records every 1 us): N, H and N2(A3) averaged over the gap and
at the centre, and the average T_g. N and H of pulse n are those just before pulse
n + 1 (the article measures 100 us after a burst of n pulses; the decay over that
interval is a few percent, figs. 10 and 15). N2(A3): maximum and minimum of each
period, against the measured envelope of v = 0 and v = 1 (the mechanism does not
resolve the vibration of N2(A)). From the restarts: energy of the pulse E =
integral of V dQ over 240 ns, with Q = eps0 eps_d (V - phi_left) A / Delta and
A = 7.2 cm2.

Usage: python3 scripts/compara_yang.py MAIN [RESTARTS ...]
"""
import argparse
import json
import pathlib
import re
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from resumo import blocos  # noqa: E402

RAIZ = pathlib.Path(__file__).resolve().parent.parent
DADOS = RAIZ / "docs" / "literatura" / "dados" / "yang2022"
EPS0 = 8.8541878128e-12
AREA = 7.2e-4
PERIODO = 1e-5


def le(f):
    return np.loadtxt(DADOS / f, delimiter=",", ndmin=2)


def series(d):
    """Times and, per species, (gap average, centre); average T_g."""
    t, med, cen, tg = [], {}, {}, []
    for tb, nomes, linhas in blocos(d / "perfil.dat"):
        a = np.array(linhas[1:-1])
        faces = [0.0]
        for c in a[:, 0]:
            faces.append(2.0 * c - faces[-1])
        h = np.diff(faces)
        k = int(np.argmin(np.abs(a[:, 0] - 0.5 * faces[-1])))
        t.append(tb)
        for e in ("N", "H", "N2(A3)"):
            if e in nomes:
                med.setdefault(e, []).append(float((a[:, nomes.index(e)] * h).sum() / h.sum()))
                cen.setdefault(e, []).append(a[k, nomes.index(e)])
        col = next((n for n in ("Temperature", "Tgas") if n in nomes), None)
        if col:
            tg.append(float((a[:, nomes.index(col)] * h).sum() / h.sum()))
    return (np.array(t), {e: np.array(v) for e, v in med.items()}, {e: np.array(v) for e, v in cen.items()},
            np.array(tg))


def simulados(t_us, t):
    """Measured times (us) that the run has already reached (t in s)."""
    return [x for x in t_us if x * 1e-6 <= t[-1] + 1e-9]


def mistura(d):
    """N2 or N2+1%H2, from the case of the run."""
    caso = json.loads((d / "parametros.json").read_text())["argumentos"]["caso"]
    return "N2+1%H2" if "1pcH2" in pathlib.Path(caso).name else "N2"


def energia(d):
    a = json.loads((d / "parametros.json").read_text())["argumentos"]
    sup = np.loadtxt(d / "superficie.dat", skiprows=1, ndmin=2)
    q = EPS0 * a["eps_dieletrico"] * (sup[:, 1] - sup[:, 2]) / a["espessura"] * AREA
    return float(np.trapezoid(sup[:, 1], q))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("principal", type=pathlib.Path)
    p.add_argument("retomadas", nargs="*", type=pathlib.Path)
    p.add_argument("--figure", "--figura", dest="figura", type=pathlib.Path)
    a = p.parse_args()
    t, med, cen, tg = series(a.principal)
    n_pulsos = int(np.floor(t[-1] / PERIODO + 1e-9))
    pulsos = np.arange(1, n_pulsos + 1)
    antes = np.array([np.argmin(np.abs(t - n * PERIODO + 1e-6)) for n in pulsos])  # 1 us before pulse n + 1
    gas = mistura(a.principal)
    print(f"{a.principal.name}: {n_pulsos} pulses ({gas})")
    medida_n = le("fig9_N_acumulado_N2.csv") if gas == "N2" else le("fig14_N_acumulado_1pcH2.csv")
    print(f"{'pulse':>5s} {'N avg':>9s} {'N centre':>9s} {'N meas.':>9s} {'H avg':>9s} {'T_g (K)':>8s}")
    for n, nm in zip(medida_n[:, 0].astype(int), medida_n[:, 1]):
        if n <= n_pulsos:
            j = antes[n - 1]
            h = med.get("H", np.full(len(t), np.nan))[j]
            print(f"{n:5d} {med['N'][j]:9.2e} {cen['N'][j]:9.2e} {nm:9.2e} {h:9.2e} "
                  f"{tg[j] if len(tg) else float('nan'):8.1f}")
    if gas != "N2":
        hm = le("fig14_H_acumulado_1pcH2.csv")
        print("H measured (fig. 14(b)) against dbd1d, gap average:")
        for tt, hv in hm[:, :2]:
            if not simulados([tt], t):
                continue
            j = int(np.argmin(np.abs(t - tt * 1e-6)))
            print(f"  {tt:7.0f} us: measured {hv:9.2e}, dbd1d {med['H'][j]:9.2e}")
    if "N2(A3)" in med:
        mx = [med["N2(A3)"][(t > (n - 1) * PERIODO) & (t <= n * PERIODO)].max() for n in pulsos]
        print(f"N2(A3) average: maximum in pulse 5 {mx[min(4, n_pulsos - 1)]:.2e}, in the last {mx[-1]:.2e} cm-3 "
              f"(measured v=0 + v=1: peak of ~1.2e14 near 50 us in N2)")
    for r in a.retomadas:
        m = re.search(r"_p(\d+)$", r.name)
        print(f"energy of pulse {m.group(1) if m else '?'}: {energia(r) * 1e3:.2f} mJ")
    print("measured in N2: 3.48 mJ in pulse 2 to 3.08 in pulse 150 (with reflections); 2.2 to 2.0 mJ for the "
          "incident pulse alone")


if __name__ == "__main__":
    main()
