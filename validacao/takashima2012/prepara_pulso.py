# SPDX-License-Identifier: GPL-3.0-or-later
"""Input voltage of the case of Takashima et al. 2012 (CPT generator) from the
applied field measured at 60 Torr (docs/literatura/dados/takashima2012/
fig7_campo_aplicado_CPT.csv), for `simula.py --waveform table`.

The applied field of their model is the voltage divided by d + 2 l/eps (10 mm
gap, 1.75 mm quartz walls, eps = 3.8 here): V = E x 1.092 cm. The curve starts
at 67 ns (zero before); it ends at 180 ns with +3.4 kV/cm and returns to zero in
a linear ramp of 30 ns. Resampling at 0.5 ns, t shifted to start at zero. The
same shape serves for every pressure (the article only shows that of 60 Torr).

Usage: python3 validacao/takashima2012/prepara_pulso.py [EPS_R]
  (writes pulso_cpt.txt with eps = 3.8, or pulso_cpt_epsEPS_R.txt)
"""
import sys
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parents[1]

def main():
    eps = float(sys.argv[1]) if len(sys.argv) > 1 else 3.8
    DISTANCIA_CM = 1.0 + 2 * 0.175 / eps
    nome = "pulso_cpt.txt" if eps == 3.8 else f"pulso_cpt_eps{sys.argv[1]}.txt"
    d = np.loadtxt(RAIZ / "docs" / "literatura" / "dados" / "takashima2012" / "fig7_campo_aplicado_CPT.csv",
                   delimiter=",")
    t, e = d[:, 0], d[:, 1]
    t0 = t[0] - 1.0
    t, e = np.concatenate([[t0], t, [t[-1] + 30.0]]), np.concatenate([[0.0], e, [0.0]])
    grade = np.arange(0.0, t[-1] - t0 + 0.25, 0.5)
    v = np.interp(grade + t0, t, e) * DISTANCIA_CM
    v[0] = v[-1] = 0.0
    np.savetxt(AQUI / nome, np.column_stack([grade * 1e-9, v]), fmt="%.6e",
               header=f"Takashima 2012, CPT generator, from the applied field at 60 Torr (fig. 7), eps = {eps}; "
                      "t (s), V (V)")
    print(f"{nome}: {len(grade)} points up to {grade[-1]:.1f} ns, from {v.min() / 1e3:.1f} to {v.max() / 1e3:.1f} kV")


if __name__ == "__main__":
    main()
