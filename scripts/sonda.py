# SPDX-License-Identifier: GPL-3.0-or-later
"""Time series of a column of perfil.dat in the cell closest to a position.

Usage: python3 scripts/sonda.py DIR/perfil.dat COLUMN X [COLUMN ...]
       (X in m; writes t and the values of the columns)
"""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from resumo import blocos  # noqa: E402


def serie(arquivo, coluna, x):
    """Times (s) and values of a column in the cell whose centre is closest to x."""
    t, v = [], []
    for tb, nomes, linhas in blocos(pathlib.Path(arquivo)):
        a = np.array(linhas[1:-1])
        k = np.argmin(np.abs(a[:, 0] - x))
        t.append(tb)
        v.append(a[k, nomes.index(coluna)])
    return np.array(t), np.array(v)


def main():
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    arquivo, x, colunas = sys.argv[1], float(sys.argv[3]), [sys.argv[2]] + sys.argv[4:]
    series = [serie(arquivo, c, x) for c in colunas]
    print("".join(f"{n:>24s}" for n in ["Time"] + colunas))
    for i, t in enumerate(series[0][0]):
        print("".join(f" {v:24.16E}" for v in [t] + [s[1][i] for s in series]))


if __name__ == "__main__":
    main()
