# SPDX-License-Identifier: GPL-3.0-or-later
"""Positive column of a DC discharge in 0D: the chemistry of a case (quimica.json
and tabela_energia_tg.dat) integrated along the residence time of the gas, at
imposed E/N and T_g, with the local field approximation for the electrons and the
radial losses of a tube of radius R (cm). This is the model of Levaton et al.
2020 (Phys. Plasmas 27, 063503) with our chemistry, for the comparison in
docs/coluna_positiva_n2.

Radial losses, with Lambda = R/2.405 (the first zero of J0):
  ions: ambipolar diffusion, nu = D_a/Lambda^2 with D_a = mu_i kTe/e;
  neutrals other than the background: diffusion in series with the reaction at
  the wall (Chantry), 1/nu = Lambda^2/D + R(2 - gamma)/(gamma vbar), with D and
  gamma of transporte.json or gamma given per species;
  electrons: the sum of the ion losses (quasi-neutrality), unless n_e is imposed.
What leaves the volume returns as background gas. With a background of one
species, the atoms are conserved: the background is the gas minus what the other
species hold (an N atom is half an N2, an N4+ two); with a mixed background, it is
held fixed.
"""
import json
import pathlib
import re

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from modelo import N_LOSCHMIDT, mobilidade_de
from quimica_zdp import Quimica
from tabela import TabelaTg

KB = 1.380649e-23
QE = 1.602176634e-19
UMA = 1.66053906660e-27
LAMBDA_J0 = 2.404825557695773


def atomos(especie):
    """Atoms of each element in a species name: N2(A3) -> {N: 2}, N4^+ -> {N: 4}."""
    nome = especie.split("^")[0].split("(")[0]
    if nome.upper() == "E":
        return {}
    conta = {}
    for el, n in re.findall(r"([A-Z][a-z]?)(\d*)", nome):
        conta[el] = conta.get(el, 0) + (int(n) if n else 1)
    return conta


def perda_ambipolar(mu, te_ev, raio):
    """Loss frequency (1/s) of an ion by ambipolar diffusion: mu in cm2/(V s),
    te_ev in eV, raio in cm."""
    return mu * te_ev / (raio / LAMBDA_J0) ** 2


def perda_neutro(d, gama, raio, vbar):
    """Loss frequency (1/s) of a neutral at the wall: diffusion (d in cm2/s) in
    series with the reaction of probability gama (at most 1); vbar in cm/s."""
    if gama > 1.0:
        raise ValueError(f"wall loss probability {gama:g} above 1")
    if gama <= 0.0:
        return 0.0
    return 1.0 / ((raio / LAMBDA_J0) ** 2 / d + raio * (2.0 - gama) / (gama * vbar))


class Coluna0D:
    def __init__(self, caso, pressao_pa, t_gas, raio_cm, gamas=None, fatores=None, estrito=True):
        """gamas: wall loss probability per neutral species, replacing the one of
        transporte.json; a name that is not a species of the case is an error, unless
        estrito=False (one set of probabilities for several chemistries)."""
        caso = pathlib.Path(caso)
        self.q = Quimica.de_json(caso / "quimica.json")
        self.tab = TabelaTg(caso / "tabela_energia_tg.dat")
        gas = json.loads((caso / "gas.json").read_text())
        transporte = json.loads((caso / "transporte.json").read_text())
        self.transporte = {k.upper(): v for k, v in transporte.items() if not k.startswith("_")}
        self.pressao_pa, self.t_gas, self.raio = pressao_pa, t_gas, raio_cm
        self.n_gas = pressao_pa / (KB * t_gas) * 1e-6
        nomes = [s.upper() for s in self.q.especies]
        self.fundo = {nomes.index(s.upper()): f for s, f in gas["fundo"].items()}
        self.indices = [i for i in range(len(nomes)) if i not in self.fundo]
        self.especies = [self.q.especies[i] for i in self.indices]
        self.i_e = [s.upper() for s in self.especies].index("E")
        self.carga = self.q.carga[self.indices]
        self.gamas = {k.upper(): v for k, v in (gamas or {}).items()}
        do_estado = {s.upper(): k for k, s in enumerate(self.especies)}
        ions = sorted(g for g in gamas or {} if g.upper() in do_estado and self.carga[do_estado[g.upper()]] != 0)
        if ions:
            raise ValueError(f"wall probability given for ions ({', '.join(ions)}): they are lost by ambipolar diffusion")
        ausentes = sorted(g for g in gamas or {} if g.upper() not in do_estado)
        if ausentes and estrito:
            raise ValueError(f"wall probability for species not in the case: {', '.join(ausentes)}")
        # multiplier of the coefficient of chosen reactions (index: factor), for sensitivity
        self.fatores = dict(fatores or {})
        # background molecules held by each species of the state (single background only)
        self.no_fundo = np.zeros(len(self.especies))
        if len(self.fundo) == 1:
            el, n_fundo = next(iter(atomos(self.q.especies[next(iter(self.fundo))]).items()))
            self.no_fundo = np.array([atomos(s).get(el, 0) / n_fundo for s in self.especies])

    def _dados(self, k):
        return self.transporte[self.especies[k].upper()]

    def densidades(self, y):
        """All species of the chemistry (cm-3): the state and the background."""
        d = np.zeros(len(self.q.especies))
        d[self.indices] = y
        for i, f in self.fundo.items():
            d[i] = f * self.n_gas - self.no_fundo @ y
        return d

    def coeficientes(self, e_n):
        """Rate coefficients (reactions,) and Te (K) at E/N (Td), local field."""
        v = self.tab(np.array([e_n]), np.array([self.t_gas]))
        te = v[:, 0]
        neutras = self.n_gas if self.q.usa_any_neutral else None
        k = self.q.coeficientes(v[:, 5:], np.array([self.t_gas]), te, np.array([e_n]),
                                None if neutras is None else np.array([neutras]))
        k = k[:, 0]
        for r, f in self.fatores.items():
            k[r] *= f
        return k, float(te[0])

    def mobilidade(self, k, e_n):
        """Mobility of the ion k of the state (cm2/(V s))."""
        return float(mobilidade_de(self._dados(k)["mobilidade"], e_n, N_LOSCHMIDT / self.n_gas))

    def difusao(self, k):
        """Diffusion coefficient of the neutral k of the state (cm2/s)."""
        d = self._dados(k)["difusao"]
        p_atm = self.pressao_pa / 101325.0
        return d["D0"] / p_atm * (self.t_gas / d["T0"]) ** d["expoente"] * 1e4

    def frequencias_de_perda(self, e_n, te):
        """Radial loss frequency (1/s) of each species of the state; zero for the
        electrons, whose loss follows the ions (rhs)."""
        te_ev = te * KB / QE
        nu = np.zeros(len(self.especies))
        for k, s in enumerate(self.especies):
            if k == self.i_e:
                continue
            if self.carga[k] != 0:
                nu[k] = perda_ambipolar(self.mobilidade(k, e_n), te_ev, self.raio)
            else:
                dados = self._dados(k)
                gama = self.gamas.get(s.upper(), dados["perda_na_parede"])
                vbar = np.sqrt(8.0 * KB * self.t_gas / (np.pi * dados["massa_u"] * UMA)) * 100.0
                nu[k] = perda_neutro(self.difusao(k), gama, self.raio, vbar)
        return nu

    def rhs(self, t, y, e_n, ne=None):
        """dy/dt: chemistry minus radial losses; with ne, the electron density is
        held at that value."""
        if ne is not None:
            y = y.copy()
            y[self.i_e] = ne
        k, te = self.coeficientes(e_n)
        fontes = self.q.fontes(self.densidades(y)[:, None], k[:, None])[:, 0][self.indices]
        nu = self.frequencias_de_perda(e_n, te)
        dy = fontes - nu * y
        if ne is not None:
            dy[self.i_e] = 0.0
        else:
            ions = self.carga > 0
            dy[self.i_e] -= np.sum(nu[ions] * y[ions] * self.carga[ions])
        return dy

    def ne_da_corrente(self, corrente_a, e_n):
        """Electron density (cm-3) that carries the current I (A) in the tube at
        E/N: I = e n_e mu_e E pi R^2, with mu_e of the table (BOLSIG+)."""
        mu = self.tab(np.array([e_n]), np.array([self.t_gas]))[0, 1] / self.n_gas
        campo = e_n * 1e-17 * self.n_gas
        return corrente_a / (QE * mu * campo * np.pi * self.raio ** 2)

    def balanco_eletrons(self, e_n, corrente_a, t_res):
        """Electron balance at the end of the residence time, with n_e from the
        current: (net production by the chemistry, radial loss), cm-3/s. The loss is
        n_e times the ambipolar frequency of the ions weighted by their densities
        (quasi-neutrality: the imposed n_e is not tied to the ions of the state)."""
        ne = self.ne_da_corrente(corrente_a, e_n)
        _, y = self.integra(e_n, ne, t_res, n_saidas=2)
        k, te = self.coeficientes(e_n)
        quimica = self.q.fontes(self.densidades(y[-1])[:, None], k[:, None])[:, 0][self.indices][self.i_e]
        ions = self.carga > 0
        nu = self.frequencias_de_perda(e_n, te)
        media = np.sum(nu[ions] * y[-1][ions]) / np.sum(y[-1][ions])
        return quimica, ne * media

    def en_sustentado(self, corrente_a, t_res, faixa=(20.0, 300.0)):
        """E/N (Td) that sustains the column: the electron balance at the end of the
        residence time is zero (ionization equals the radial loss)."""
        def liquido(en):
            q, perda = self.balanco_eletrons(en, corrente_a, t_res)
            return (q - perda) / perda
        try:
            return brentq(liquido, *faixa, rtol=1e-6)
        except ValueError as erro:
            if "different signs" not in str(erro):
                raise
            raise ValueError(f"no sustaining E/N between {faixa[0]:g} and {faixa[1]:g} Td: the electron "
                             "balance has the same sign at both ends") from None

    def estado_inicial(self, ne):
        """Gas entering the column: electrons at ne with N2+ to match, the rest zero."""
        y = np.zeros(len(self.especies))
        y[self.i_e] = ne
        ion = [s.upper() for s in self.especies].index("N2^+")
        y[ion] = ne
        return y

    def integra(self, e_n, ne, t_fim, n_saidas=50, y0=None, rtol=1e-6, atol=1.0):
        """Densities at n_saidas instants from t_fim*1e-6 to t_fim (log spaced);
        returns (t, y[instant, species])."""
        y0 = self.estado_inicial(ne) if y0 is None else y0
        saidas = np.geomspace(t_fim * 1e-6, t_fim, n_saidas)
        sol = solve_ivp(self.rhs, (0.0, t_fim), y0, method="BDF", t_eval=saidas, args=(e_n, ne),
                        rtol=rtol, atol=atol)
        if not sol.success:
            raise RuntimeError(sol.message)
        y = sol.y.T.copy()
        if ne is not None:
            y[:, self.i_e] = ne
        return sol.t, y
