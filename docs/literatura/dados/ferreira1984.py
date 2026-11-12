# SPDX-License-Identifier: GPL-3.0-or-later
"""Vibrational populations of N2(B, v = 0 to 12) in the positive column of a pure N2
glow discharge: Ferreira, Touzeau, Hochard and Cernogora 1984, J. Phys. B 17, 4439,
figs. 1 (0.7 Torr) and 2 (2 Torr) at 5, 20 and 50 mA (tube of 2 cm inner diameter,
stagnant gas), and fig. 3, the total N2(B) against the current.

The figures are scanned (raster, black and white, with five marker shapes for the
five sequences Delta v = 0 to 4), so the series cannot be separated by colour as in
ferramentas/digitaliza.py. The points were read on page 5 of the PDF rendered at
400 dpi, against the decades 1e8 to 1e11 cm-3 of the y axis; where a level has points
from two or three sequences, the value is their geometric mean. The sum over v agrees
with the total of fig. 3 (5.2, 2.2 and 0.5e11 cm-3 at 2 Torr; 4.3, 1.75 and 0.45e11 at
0.7 Torr; read on the points where the figure has them, as 2.2e11 at 2 Torr and 20 mA) within 14%,
which checks the reading.

Used for the fraction f(v = 2) of N2(B) in v = 2: Levaton et al. (2020, fig. 10; 2021,
figs. 2 and 3) measured N2(B, v = 2) only, against the total N2(C) (docs/auditoria/
2026-11-10/ERROS.md V3-1).

Usage: python3 docs/literatura/dados/ferreira1984.py  (writes to ferreira1984/)
"""
from pathlib import Path

# [N2(B, v)] in cm-3, v = 0 to 12, as read (see above)
POPULACOES = {
    ("2", 50): [1.5e11, 1.1e11, 8.2e10, 3.2e10, 3.1e10, 2.8e10, 1.7e10, 1.0e10, 9.2e9, 8.0e9, 5.8e9, 4.8e9, 2.9e9],
    ("2", 20): [8.6e10, 5.2e10, 3.8e10, 1.2e10, 1.05e10, 7.8e9, 4.8e9, 3.0e9, 1.8e9, 1.3e9, 1.3e9, 9.5e8, 5.7e8],
    ("2", 5): [2.5e10, 1.2e10, 7.5e9, 2.0e9, 1.8e9, 9.5e8, 7.0e8, 4.2e8, 2.4e8, 1.75e8, 1.8e8, 1.4e8, 1.1e8],
    ("0.7", 50): [1.5e11, 1.1e11, 8.5e10, 3.2e10, 2.6e10, 2.3e10, 1.9e10, 1.2e10, 8.7e9, 7.0e9, 5.7e9, 4.9e9, 3.3e9],
    ("0.7", 20): [6.4e10, 4.2e10, 3.1e10, 1.0e10, 8.8e9, 7.7e9, 5.3e9, 2.6e9, 1.55e9, 1.3e9, 1.15e9, 8.7e8, 6.0e8],
    ("0.7", 5): [2.0e10, 1.2e10, 8.7e9, 2.3e9, 2.1e9, 1.8e9, 1.1e9, 5.5e8, 3.8e8, 2.8e8, 2.5e8, 1.85e8, 2.3e8],
}
# total N2(B) of fig. 3 (cm-3), read on the measured points or, without one, on the fitted line
TOTAL_FIG3 = {("2", 50): 5.2e11, ("2", 20): 2.2e11, ("2", 5): 5.0e10,
              ("0.7", 50): 4.3e11, ("0.7", 20): 1.75e11, ("0.7", 5): 4.5e10}


def fracao_v2():
    """f(v = 2) = [N2(B, v = 2)] / sum over v = 0 to 12, per (pressure in Torr, current in mA)."""
    return {k: n[2] / sum(n) for k, n in POPULACOES.items()}


def main():
    pasta = Path(__file__).resolve().parent / "ferreira1984"
    pasta.mkdir(exist_ok=True)
    with open(pasta / "populacoes_b.csv", "w") as f:
        f.write("# Ferreira et al. 1984, J. Phys. B 17, 4439, figs. 1 and 2: [N2(B, v)] (cm-3), read at 400 dpi\n")
        f.write("pressao_Torr,corrente_mA," + ",".join(f"v{v}" for v in range(13)) + ",soma,total_fig3,f_v2\n")
        for (p, i), n in POPULACOES.items():
            f.write(f"{p},{i}," + ",".join(f"{x:.3g}" for x in n)
                    + f",{sum(n):.4g},{TOTAL_FIG3[(p, i)]:.3g},{n[2] / sum(n):.4f}\n")
    print(pasta / "populacoes_b.csv")


if __name__ == "__main__":
    main()
