# SPDX-License-Identifier: GPL-3.0-or-later
"""Input voltage of the case of Yang et al. 2022 from the measured pulse 50
(docs/literatura/dados/yang2022/fig2_tensao_pulso50.csv), for
`simula.py --waveform table --voltage-table pulso50.txt --period 1e-5`.

Treatment, all recorded here:
- before -1 ns the voltage is zero (the plateau of 0.2 to 1.1 kV in the
  digitization before the rise is the zero line crossing the curve, not signal);
- the curve goes up to 198.4 ns, where the figure ends at -5.8 kV; from there the
  voltage returns to zero in a linear ramp of 30 ns (what comes after was not
  measured);
- uniform resampling at 0.5 ns (the rise takes ~3 ns), with t shifted to start
  at zero;
- the median per column cuts the narrow peak: 25.2 kV here against 26.2 kV from
  the edge of the line (4%).

Usage: python3 validacao/yang2022/prepara_pulso.py  (writes pulso50.txt next to it)
"""
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parents[1]


def main():
    d = np.loadtxt(RAIZ / "docs" / "literatura" / "dados" / "yang2022" / "fig2_tensao_pulso50.csv", delimiter=",")
    t_ns, v_kv = d[:, 0], d[:, 1]
    ok = t_ns >= -1.0
    t_ns, v_kv = np.concatenate([[-1.0], t_ns[ok]]), np.concatenate([[0.0], v_kv[ok]])
    fim = t_ns[-1]
    t_ns, v_kv = np.concatenate([t_ns, [fim + 30.0]]), np.concatenate([v_kv, [0.0]])
    grade = np.arange(0.0, fim + 1.0 + 30.0 + 0.25, 0.5)
    v = np.interp(grade - 1.0, t_ns, v_kv, left=0.0, right=0.0) * 1e3
    v[0] = v[-1] = 0.0
    np.savetxt(AQUI / "pulso50.txt", np.column_stack([grade * 1e-9, v]), fmt="%.6e",
               header="Yang 2022, pulse 50 in N2 at 150 Torr (fig. 2(a)); t (s), V (V); see prepara_pulso.py")
    print(f"pulso50.txt: {len(grade)} points, from 0 to {grade[-1]:.1f} ns, peak {v.max() / 1e3:.1f} kV, "
          f"minimum {v.min() / 1e3:.1f} kV")


if __name__ == "__main__":
    main()
