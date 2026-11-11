# SPDX-License-Identifier: GPL-3.0-or-later
"""Repetitively pulsed discharge in 0D: a pulse at imposed E/N and electron density,
then an afterglow with the electrons free, repeated until the state at the end of a
period equals the state at its start (the periodic regime). Written for the pulsed
N2 jet of Kuhfeld et al. 2021 (J. Phys. D 54, 305204 and 305205; docs/jato_n2), on
top of Coluna0D (coluna.py), of which it uses integra, carga, i_e, especies and
n_gas.

The breakdown is not modelled. At the start of the pulse the ion N2^+ receives what
the ions lack to carry the imposed electron density; at its end the electrons are
set to the charge of the ions, so the afterglow starts neutral and the electrons
recombine with the ions that the pulse actually has. With ions_no_fim, the most
abundant ion also receives, at the end of the pulse, what the ions lack to carry the
imposed density (the imposed n_e consumes ions that the chemistry at the bulk field
need not replace).

With perfil_ne, the pulse imposes n_e(t) interpolated in that profile (t from the
start of the pulse, s; cm-3) instead of the constant ne, which then only sets the ion
seed; the integrator step is bounded by MAX_PASSO_PERFIL so that the rise of the
profile is not stepped over. passo_saida adds outputs at that step inside the pulse.
"""
from dataclasses import dataclass

import numpy as np

N_PULSO = 20          # outputs inside the pulse, log spaced
LIMIAR = 1e-8         # species below this fraction of the gas do not enter the convergence test
MAX_PASSO_PERFIL = 2e-9   # largest integrator step (s) in a pulse with a profile of n_e


@dataclass(frozen=True)
class Pulso:
    e_n: float        # E/N during the pulse (Td)
    ne: float         # electron density during the pulse (cm-3)
    duracao: float    # pulse length (s)
    periodo: float    # repetition period (s)
    e_n_pos: float    # E/N in the afterglow (Td)
    ions_no_fim: bool = False   # also complete the ions at the end of the pulse (sensitivity)
    perfil_ne: tuple = None     # ((t, ...), (n_e, ...)): n_e(t) imposed in the pulse
    passo_saida: float = None   # extra outputs at this step (s) inside the pulse

    def __post_init__(self):
        if not 0.0 < self.duracao < self.periodo:
            raise ValueError(f"duracao {self.duracao:g} s must lie in (0, periodo = {self.periodo:g} s)")
        if self.perfil_ne is not None:
            t = np.asarray(self.perfil_ne[0], dtype=float)
            if t[0] > 0.0 or t[-1] < self.duracao or np.any(np.diff(t) <= 0.0):
                raise ValueError("perfil_ne must have increasing times covering the pulse, 0 to duracao")

    def ne_no_pulso(self):
        """What Coluna0D.integra takes as ne in the pulse: the constant or n_e(t)."""
        if self.perfil_ne is None:
            return self.ne
        t, n = (np.asarray(a, dtype=float) for a in self.perfil_ne)
        return lambda x: np.interp(x, t, n)

    def saidas_no_pulso(self):
        """Outputs inside the pulse: N_PULSO log spaced, plus one every passo_saida."""
        t = np.geomspace(self.duracao * 1e-4, self.duracao, N_PULSO)
        if self.passo_saida is not None:
            t = np.union1d(t, np.arange(self.passo_saida, self.duracao, self.passo_saida))
        return t


def carga_dos_ions(modelo, y):
    """Positive charge density of the ions of the state (cm-3)."""
    ions = modelo.carga > 0
    return float(np.sum(modelo.carga[ions] * y[ions]))


def periodo(modelo, pulso, y0, saidas_pos, rtol=1e-6):
    """One period from the state y0: (t, y[instant, species]), with t = 0 (after the
    ion seed), N_PULSO instants up to the end of the pulse and the afterglow
    instants saidas_pos (absolute, in (duracao, periodo], the last one the period)."""
    saidas_pos = np.asarray(saidas_pos, dtype=float)
    if (saidas_pos[0] <= pulso.duracao or saidas_pos[-1] > pulso.periodo
            or not np.isclose(saidas_pos[-1], pulso.periodo, rtol=1e-12)):
        raise ValueError(f"saidas_pos must lie in ({pulso.duracao:g}, {pulso.periodo:g}] s and end at the period")
    y = np.array(y0, dtype=float)
    falta = pulso.ne - carga_dos_ions(modelo, y)
    if falta > 0.0:
        y[[s.upper() for s in modelo.especies].index("N2^+")] += falta
    y[modelo.i_e] = pulso.ne
    t_p = pulso.saidas_no_pulso()
    passo = {} if pulso.perfil_ne is None else dict(max_step=MAX_PASSO_PERFIL)
    _, y_p = modelo.integra(pulso.e_n, pulso.ne_no_pulso(), pulso.duracao, y0=y, saidas=t_p, rtol=rtol, **passo)
    fim = y_p[-1].copy()
    if pulso.ions_no_fim:
        ions = np.flatnonzero(modelo.carga > 0)
        dominante = ions[np.argmax(fim[ions])]
        fim[dominante] += max(pulso.ne - carga_dos_ions(modelo, fim), 0.0) / modelo.carga[dominante]
    fim[modelo.i_e] = carga_dos_ions(modelo, fim)
    _, y_a = modelo.integra(pulso.e_n_pos, None, pulso.periodo - pulso.duracao, y0=fim,
                            saidas=saidas_pos - pulso.duracao, rtol=rtol)
    return np.concatenate([[0.0], t_p, saidas_pos]), np.vstack([y, y_p, y_a])


def mudanca(modelo, inicio, fim):
    """Largest relative change of a species between the start and the end of a period."""
    vale = fim > LIMIAR * modelo.n_gas
    if not np.any(vale):
        raise RuntimeError(f"no species above the threshold {LIMIAR:g} of the gas density at the end of the period")
    return float(np.max(np.abs(fim[vale] - inicio[vale]) / fim[vale]))


def mudanca_com_piso(modelo, inicio, fim):
    """Largest change of a species between the start and the end of a period,
    relative to its density or to LIMIAR * n_gas, whichever is larger: every species
    counts, and one that climbs through the threshold does not enter with a jump."""
    return float(np.max(np.abs(fim - inicio) / np.maximum(np.abs(fim), LIMIAR * modelo.n_gas)))


def convergiu(historico, tol, criterio):
    """antigo: the last two changes below tol. novo: also the geometric rest of the
    sequence, h r/(1 - r) with r the ratio of the last two, below tol (a slow
    contraction leaves the state farther from the fixed point than the last change)."""
    if len(historico) < 2 or max(historico[-2:]) >= tol:
        return False
    if criterio == "antigo":
        return True
    r = historico[-1] / historico[-2] if historico[-2] > 0.0 else 0.0
    return r < 1.0 and historico[-1] * r / (1.0 - r) < tol


def regime_periodico(modelo, pulso, y0, saidas_pos, tol=1e-3, max_periodos=80, rtol=1e-6, criterio="antigo"):
    """Periods from y0 until the change between the start and the end of a period
    converges (convergiu): (t, y) of the last period and the history of the change.
    criterio antigo is that of the first runs of the jet (mudanca, species above
    LIMIAR); novo uses mudanca_com_piso and the geometric rest."""
    if criterio not in ("antigo", "novo"):
        raise ValueError(f"criterio {criterio!r}: antigo or novo")
    medida = mudanca if criterio == "antigo" else mudanca_com_piso
    y = np.array(y0, dtype=float)
    historico = []
    for _ in range(max_periodos):
        t, ys = periodo(modelo, pulso, y, saidas_pos, rtol=rtol)
        historico.append(medida(modelo, y, ys[-1]))
        if convergiu(historico, tol, criterio):
            return t, ys, historico
        y = ys[-1]
    raise RuntimeError(f"no periodic regime after {max_periodos} periods (last change {historico[-1]:.3g})")
