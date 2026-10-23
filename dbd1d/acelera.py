# SPDX-License-Identifier: GPL-3.0-or-later
"""Cycle acceleration for slow chemistry in a periodic discharge (projective
integration over cycles). The discharge settles in a few cycles; the neutral
species and the gas temperature drift over hundreds or thousands (a residence
time of 0.05 s is 1000 cycles at 20 kHz). After a few direct cycles, each slow
variable is projected many cycles ahead from its changes between the ends of the
last three cycles, d0 = y1 - y0 and d1 = y2 - y1 (states at the same phase).

With r = d1/d0 in (0, 1) the variable relaxes and the projection over M cycles is
the geometric sum d1 r (1 - r^M) / (1 - r), exact for an exponential relaxation
and bounded for a fast one; with r >= 1 it is linear, M d1; with r <= 0 (the
change alternates in sign: a fast mode or noise) the variable is not projected.
Each change is limited to the fraction limite of the value (half by default):
the densities stay positive, and a species that starts from zero, which no
relaxation estimate covers yet, still grows by half at each jump. Electrons,
ions, the energy and the surface charges are not projected: they adjust to the
new neutrals within the next direct cycle."""
import numpy as np

LIMITE = 0.5


def indices_lentos(md):
    """Indices in the state vector of the neutral species (charge 0) in all cells
    and, with --heating, of the gas temperature."""
    linhas = [k for k in range(md.n_esp) if md.carga[k] == 0]
    if md.aquecimento:
        linhas.append(md.i_t)
    return np.concatenate([k * md.m + np.arange(md.m) for k in linhas]).astype(int)


def projeta(y0, y1, y2, lentos, ciclos, limite=LIMITE):
    """State y2 with the slow variables projected ciclos cycles ahead, and the
    number of them whose change was limited."""
    if ciclos < 1 or not 0.0 < limite < 1.0:
        raise ValueError("the projection needs at least one cycle and a limit in (0, 1)")
    d0, d1 = y1[lentos] - y0[lentos], y2[lentos] - y1[lentos]
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(d0 != 0.0, d1 / d0, 0.0)
        soma = np.where(np.abs(1.0 - r) > 1e-12, r * (1.0 - r ** ciclos) / (1.0 - r), ciclos)
    passo = np.where(r >= 1.0, ciclos * d1, np.where(r > 0.0, d1 * soma, 0.0))
    teto = limite * np.abs(y2[lentos])
    limitados = int(np.count_nonzero(np.abs(passo) > teto))
    novo = y2.copy()
    novo[lentos] = y2[lentos] + np.clip(passo, -teto, teto)
    return novo, limitados
