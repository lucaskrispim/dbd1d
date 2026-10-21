# SPDX-License-Identifier: GPL-3.0-or-later
"""Série temporal de uma coluna do perfil.dat na célula mais próxima de uma
posição, como borda0.dat, centro.dat e borda1.dat do código Fortran.

Uso: python3 scripts/sonda.py DIR/perfil.dat COLUNA X [COLUNA ...]
     (X em m; escreve t e os valores das colunas)
"""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from resumo import blocos  # noqa: E402


def serie(arquivo, coluna, x):
    """Instantes (s) e valores de coluna na célula cujo centro está mais perto de x."""
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
