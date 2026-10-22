# SPDX-License-Identifier: GPL-3.0-or-later
"""Coupled 1D fluid model: transport of every species, Poisson equation,
chemistry and surface charges of the DBD in a single system dy/dt = f(t, y), for
an implicit integrator with adaptive step (scipy's BDF).

State y: densities (cm-3) of the transported species, cell by cell, and the left
and right surface charges (C/m2). The background gas (one species or a mixture,
gas.json) is fixed, with constant density, at the given T_g.

Discretization (the geometry and wall conditions of the original Fortran code):
- mesh clustered towards the walls by a hyperbolic tangent (agrupamento = 0: uniform);
- Poisson equation in the cells and at the wall nodes, with the dielectric
  barrier and the surface charge in Gauss's law at each interface;
- interior Scharfetter-Gummel fluxes with the mobility and diffusion of the
  cells and their harmonic mean at the faces;
- absorbing walls: Gamma = n*(f*v_th/2 + max(w_n, 0)), f the Boltzmann factor of
  half a cell when the drift points away from the wall; secondary emission
  gamma*Gamma_ions;
- chemistry: reactions of the case with the electron impact rate coefficients
  and Te of the table in E/N, at the field of the cell centre.

With energia=True, the electron energy density n_eps (eV cm-3) is in the state vector
(local mean energy approximation, Hagelaar and Pitchford 2005): the electron
coefficients (mobility, diffusion, Te, rates) come from the mean energy
n_eps/n_e and T_g (tabela_energia_tg.dat through TabelaEnergiaTg, or
tabela_energia.dat at 300 K), and
  d(n_eps)/dt + d(Gamma_eps)/dx = -Gamma_e*E - n_e*N*[(P_el + P_inel)/N](eps),
with Gamma_eps of Scharfetter-Gummel (mu_eps, D_eps, charge -1), the field work
computed with the same electron fluxes as the transport and the loss from the
tabulated elastic and inelastic powers. The total power P/N of BOLSIG+ cannot be
used as the loss: in the temporal growth mode of BOLSIG+ it includes k_ion*eps,
the energy of the created electrons, which in the fluid already appears as a
dilution of n_eps by the growth of n_e (counting it again lowered the mean
energy and the ionization). At the wall, each electron leaving by thermal motion
takes (5/2)kTe, as in Hagelaar et al. 2000, and the drift takes mu_eps*E*n_eps;
secondary electrons enter with eps_sec.

With aquecimento=True, the gas temperature T (K) is in the state vector, at constant
volume, as in the Fortran code and in ZDPlasKin with GAS_HEATING:
  c_v N k_B dT/dt = d/dx(kappa(T) dT/dx) + Q_el + Q_J,
with N the heavy particles of the cell, c_v and kappa(T) from gas.json (in
argon, c_v = 3/2), harmonic mean at the faces, the wall at t_parede half a cell
away, Q_el the elastic electron loss (P_el/N of the table) and Q_J the Joule
heating of the ions, |z| n_i mu_i E^2. Local T and N enter the rates, the ion
diffusion (Einstein), the neutral diffusion, the ion mobilities and the thermal
wall fluxes. With the energy table in T_g (TabelaEnergiaTg), the local T_g also
enters the electron coefficients, the elastic loss and the floor of the mean
energy; with the table at 300 K, they stay those of 300 K.

With gas_solto=True, the background gas is in the state vector: it diffuses with its
coefficient from transporte.json, takes part in the chemistry and receives back
from the wall, as neutral atoms, the particles the wall absorbs, each with its
atomos_do_fundo (Ar2+ counts two). The N of the electrons (E/N, coefficients of
the table, losses) becomes that of the neutrals of the cell, as the all_neutral
of ZDPlasKin.
"""
import copy
import json
import pathlib

import numpy as np
from scipy.linalg import solve_banded

from quimica_zdp import ErroQuimica
from tabela import E_DEN, E_DN, E_K0, E_MUEN, E_MUN, E_NUN, E_PEL, E_PINEL, E_POT, NUN, PEL

Q = 1.602176634e-19
EPS0 = 8.8541878128e-12
KB = 1.380649e-23
ME = 9.1093837015e-31
U = 1.66053906660e-27
N_LOSCHMIDT = 2.6867811e19     # cm-3


def bernoulli(x):
    """B(x) = x/(e^x - 1), with B(0) = 1."""
    x = np.asarray(x, dtype=float)
    pequeno = np.abs(x) < 1e-8
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        b = np.where(pequeno, 1.0 - 0.5 * x, x / np.expm1(np.where(pequeno, 1.0, x)))
    return np.where(np.isnan(b), 0.0, b)


def faces_tanh(m, comprimento, agrupamento):
    """Faces of m cells in [0, comprimento], clustered towards both walls by a
    hyperbolic tangent of strength agrupamento (0: uniform mesh)."""
    j = np.arange(m + 1)
    if agrupamento <= 0.0:
        return comprimento * j / m
    f = 0.5 * comprimento * (1.0 + np.tanh(agrupamento * (2.0 * j / m - 1.0)) / np.tanh(agrupamento))
    f[0], f[-1] = 0.0, comprimento
    return f


def condutividade_termica(t):
    """Thermal conductivity of argon (W/(m K)), as in the Fortran code: the
    reference that the gas.json of argon reproduces (test_gas, test_verificacao)."""
    k = (0.0307313 * t + 10.006) * 1e-3
    return np.where(t > 3000.0, k * np.sqrt(np.maximum(t, 3000.0) / 3000.0), k)


def media_harmonica(a, b):
    """2ab/(a + b), zero where a + b vanishes (coefficient at the faces)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        h = 2.0 * a * b / (a + b)
    return np.where((a + b) > 0.0, h, 0.0)


RAIZ = pathlib.Path(__file__).resolve().parent.parent


def _le_json(valor, nome):
    """Copy of the given dictionary, of the given file or, without a value, of
    the file nome at the root of the repository."""
    if valor is None:
        valor = RAIZ / nome
    if not isinstance(valor, dict):
        valor = json.loads(pathlib.Path(valor).read_text())
    return copy.deepcopy(valor)


class ErroGas(ValueError):
    """Missing or invalid properties of the background gas (gas.json)."""


class ErroTransporte(ValueError):
    """Missing or invalid transport data (transporte.json)."""


class Modelo:
    def __init__(self, quimica, tabela, n_celulas=100, gap=8.0e-4, agrupamento=0.0,
                 v_pico=1.0e3, periodo=1.0e-7, espessura=0.0, eps_dieletrico=4.8,
                 gama_sec=0.05, polarizabilidade=1.6411, sigma_cx=1.0e-18,
                 fundo=None, t_gas=300.0, n_gas=None,
                 tabela_energia=None, eps_sec=5.0, faces=None, freio=True,
                 aquecimento=False, t_parede=300.0, gas_solto=False, bolsigdb=None,
                 k0_ar=None, en_aquecimento_ar=None, k0_ar2=None, ganho_tres_corpos=0.0, eps_gas=1.0,
                 transporte=None, gas=None, tabela_calor=None,
                 forma="seno", subida=0.0, largura=0.0, descida=0.0,
                 espessura_dir=None, eps_dieletrico_dir=None, tabela_tensao=None, reflexao_eletrons=0.0):
        self.q, self.tab = quimica, tabela
        if quimica.bolsig_tres_corpos:
            raise ErroQuimica("BOLSIG+ reactions with two reacting electrons: "
                              + ", ".join(quimica.bolsig_tres_corpos) + ". The BOLSIG+ rate is a two-body one "
                              "(cm3/s) and cannot be a three-body coefficient; correct kinet.inp")
        # properties of the background gas (gas.json at the root, or the given one)
        self.gas = _le_json(gas, "gas.json")
        # folder of the files of the case (simula.py changes it with --case)
        self.caso = RAIZ
        fundos = self._le_gas(quimica, fundo)
        self.tab_e = tabela_energia
        self.energia = tabela_energia is not None
        self.eps_sec = eps_sec
        # Boltzmann factor of half a cell in the thermal wall flux; a subgrid model
        # that tends to 1 as the mesh is refined
        self.freio = freio
        self.m = n_celulas
        self.L = gap
        self.base = faces_tanh(n_celulas, gap, agrupamento)             # reference mesh
        self.troca_malha(self.base if faces is None else np.asarray(faces, dtype=float))
        self._define_tensao(v_pico, periodo, forma, subida, largura, descida, tabela_tensao)
        self._define_barreiras(espessura, eps_dieletrico, espessura_dir, eps_dieletrico_dir)
        # relative permittivity of the gas in the Poisson equation (the Fortran code
        # used 1.000516, a value at 0 C; at 300 K it would be ~1.00047; the default
        # 1 is vacuum)
        self.eps_gas = eps_gas
        self.gama = gama_sec
        # electron reflection at the walls (Hagelaar et al. 2000): the absorbed flux,
        # thermal and drift, and the energy it carries are multiplied by (1 - r)/(1 + r)
        if not 0.0 <= reflexao_eletrons < 1.0:
            raise ValueError(f"electron reflection {reflexao_eletrons}: must be in [0, 1)")
        self.reflexao_eletrons = reflexao_eletrons
        self.absorcao_eletrons = (1.0 - reflexao_eletrons) / (1.0 + reflexao_eletrons)
        self.alfa, self.sigma_cx = polarizabilidade, sigma_cx
        self.t_gas = t_gas
        # gas density at 1 atm and the given temperature, cm-3
        self.n_gas = 101325.0 / (KB * t_gas) * 1e-6 if n_gas is None else n_gas
        self._define_especies(quimica, fundos, gas_solto)
        # transport data per species (transporte.json at the root, or the given one)
        self.transporte = _le_json(transporte, "transporte.json")
        self._aplica_opcoes_do_argonio(k0_ar, en_aquecimento_ar, k0_ar2)
        self._le_massas(fundos)
        self._le_mobilidades()
        self._define_grupos(gas_solto)
        self.n_esp = len(self.especies)
        self.n_var = self.n_esp + (1 if self.energia else 0)
        if self.energia:
            self._prepara_energia(quimica, fundos, bolsigdb, ganho_tres_corpos)
        self.calor_reacao = self._calor_das_reacoes(quimica, ganho_tres_corpos)
        # fast heating by electrons (BOLSIG+ processes listed in gas.json, such as
        # the rotational excitation of N2): tabela_calor_tg.dat of the case
        self.tab_calor = None
        if self.gas.get("calor", {}).get("processos_rapidos"):
            if tabela_calor is None:
                raise ErroGas("gas.json lists processos_rapidos: tabela_calor_tg.dat is missing (ferramentas/caso.py "
                              "heat and tables) or the parameter tabela_calor")
            self.tab_calor = tabela_calor
        self.aquecimento = aquecimento
        self.t_parede = t_parede
        if aquecimento:
            self.i_t = self.n_var
            self.n_var += 1
        self._le_difusao_e_perda(fundos[0][0], gas_solto)

    # ------------------------------------------------- steps of the constructor
    def _define_tensao(self, v_pico, periodo, forma, subida, largura, descida, tabela=None):
        """Waveform of the voltage: seno, a sine (default); degrau, a linear rise in
        subida s and constant afterwards (DC voltage); pulso, a trapezoid of rise,
        plateau largura and fall, repeated every periodo; tabela, pairs (t, V)
        within one period (a measured pulse), interpolated and repeated every
        periodo, with V zero outside the table."""
        if forma not in ("seno", "degrau", "pulso", "tabela"):
            raise ValueError(f"unknown voltage waveform: {forma}")
        if forma == "pulso" and subida + largura + descida >= periodo:
            raise ValueError("pulse: rise + width + fall must be shorter than the period")
        if forma == "tabela":
            t_tab, v_tab = (np.asarray(x, dtype=float) for x in tabela)
            if len(t_tab) < 2 or np.any(np.diff(t_tab) <= 0.0) or t_tab[0] < 0.0:
                raise ValueError("voltage table: times must increase, from t >= 0")
            if t_tab[-1] >= periodo:
                raise ValueError("voltage table: the last time must be shorter than the period")
            # V zero at both ends: outside the table the voltage is zero, and a jump
            # would give an infinite displacement current
            if v_tab[0] != 0.0 or v_tab[-1] != 0.0:
                raise ValueError("voltage table: must start and end at zero")
            self.t_tab, self.v_tab = t_tab, v_tab
        self.v_pico, self.omega = v_pico, 2.0 * np.pi / periodo
        self.forma, self.periodo = forma, periodo
        self.subida, self.largura, self.descida = subida, largura, descida

    def _define_barreiras(self, espessura, eps_dieletrico, espessura_dir, eps_dieletrico_dir):
        """Dielectric barrier of each electrode; the right one, without a value,
        equal to the left one. Zero thickness is a bare metal electrode, and only
        on both sides."""
        self.espessura, self.eps_d = espessura, eps_dieletrico
        self.espessura_dir = espessura if espessura_dir is None else espessura_dir
        self.eps_d_dir = eps_dieletrico if eps_dieletrico_dir is None else eps_dieletrico_dir
        if (self.espessura > 0.0) != (self.espessura_dir > 0.0):
            raise ValueError("thickness (espessura): a barrier on one side only is not supported (both or none)")

    def _define_especies(self, quimica, fundos, gas_solto):
        """Transported species: all but those of the background gas, which enters
        only when free; in a mixture, each background species with its fraction of N."""
        self.fundos = [(quimica.indice[nome], fracao) for nome, fracao in fundos]
        self.gas_solto = gas_solto
        indices_fundo = {i for i, _ in self.fundos}
        self.especies = [k for k in range(len(quimica.especies)) if gas_solto or k not in indices_fundo]
        self.nomes = [quimica.especies[k].upper() for k in self.especies]
        self.carga = quimica.carga[self.especies]
        # the mobility of transporte.json is that of the ion itself (v = K E): the
        # drift uses only the sign of the charge, which enters whole in the Poisson equation
        self.sinal = np.sign(self.carga)
        self.i_e = self.nomes.index("E")

    def _aplica_opcoes_do_argonio(self, k0_ar, en_aquecimento_ar, k0_ar2):
        """Command line options for argon, which override the mobilities of Ar+
        and Ar2+ in transporte.json."""
        opcoes = (("AR^+", "K0", k0_ar), ("AR^+", "EN_aquecimento", en_aquecimento_ar), ("AR2^+", "K0", k0_ar2))
        for nome, chave, valor in opcoes:
            if valor is None:
                continue
            if nome not in self.nomes:
                raise ErroTransporte(f"transport: argon option for {nome}, which is not in the chemistry")
            mob = self._objeto(nome, "mobilidade")
            if chave == "EN_aquecimento" and mob.get("tipo") != "frost":
                raise ErroTransporte(f"transport: heating field given for {nome}, whose mobility "
                                     f"is not frost ({mob.get('tipo')})")
            mob[chave] = valor

    def _le_massas(self, fundos):
        """Mass of each transported species and of the background gas, for the
        elastic exchange of the electrons with it; in a mixture, 1/M = sum of
        x_i/M_i (an approximation: weighted by the fraction, not by the collision
        frequency)."""
        self.massa = np.array([ME if s == "E" else self._numero(s, "massa_u") * U for s in self.nomes])
        if len(fundos) == 1:
            self.m_fundo = self._numero(fundos[0][0], "massa_u") * U
        else:
            self.m_fundo = 1.0 / sum(fracao / (self._numero(nome, "massa_u") * U) for nome, fracao in fundos)

    def _le_mobilidades(self):
        """Mobility of each ion: constant, Frost or a table in E/N."""
        self._mob = {}
        for k, s in enumerate(self.nomes):
            if s == "E" or self.carga[k] == 0:
                continue
            mob = self._objeto(s, "mobilidade")
            if mob.get("tipo") == "tabela":
                self._confere_tabela_de_mobilidade(s, mob)
            else:
                faltam = {"constante": ["K0"], "frost": ["K0", "EN_aquecimento"]}.get(mob.get("tipo"))
                if faltam is None:
                    raise ErroTransporte(f"transport: unknown mobility type {mob.get('tipo')} ({s})")
                for chave in faltam:
                    self._numero(s, "mobilidade", chave)
            self._mob[k] = mob

    @staticmethod
    def _confere_tabela_de_mobilidade(s, mob):
        e_n, k0 = mob.get("E_N"), mob.get("K0")
        listas = isinstance(e_n, list) and isinstance(k0, list) and len(e_n) == len(k0) and len(e_n) >= 2
        if not listas or np.any(np.diff(e_n) <= 0) or np.any(np.asarray(k0, dtype=float) <= 0):
            raise ErroTransporte(f"transport: {s} mobility as a table needs increasing E_N and positive "
                                 "K0, of the same length")
        if mob.get("acima", "preso") not in ("preso", "raiz"):
            raise ErroTransporte(f"transport: {s} mobility as a table: acima = {mob['acima']!r}, "
                                 "must be preso or raiz")

    def _define_grupos(self, gas_solto):
        """Positive ions, negative ions, neutrals, the initial ion and, with the free
        gas, the atoms the wall returns to the gas per absorbed particle. Negative
        ions: transport and Joule heating as the positive ones, without secondary
        emission; the field of the walls confines them to the volume (fluxos)."""
        nomes, carga = self.nomes, self.carga
        self.ions = [k for k in range(len(nomes)) if carga[k] > 0]
        self.ions_negativos = [k for k, s in enumerate(nomes) if carga[k] < 0 and s != "E"]
        ion = str(self.gas["ion_inicial"]).upper()
        if ion not in nomes or carga[nomes.index(ion)] <= 0:
            raise ErroGas(f"gas.json: ion_inicial {self.gas['ion_inicial']} is not a positive ion of the chemistry")
        self.ion_inicial = nomes.index(ion)
        self.neutros = [k for k, s in enumerate(nomes) if carga[k] == 0 and s != "E"]
        if gas_solto:
            # each background species in the state vector; atomos[j, k]: particles of the
            # background j that the wall returns per absorbed particle k (in a
            # mixture, atomos_do_fundo is an object per background species, such as
            # {"N2": 0.5, "H2": 1.5} for NH3)
            self.i_fundos = [nomes.index(self.q.especies[i].upper()) for i, _ in self.fundos]
            nomes_fundo = [nomes[i] for i in self.i_fundos]
            self.atomos = np.zeros((len(self.i_fundos), len(nomes)))
            for k, s in enumerate(nomes):
                if s == "E" or k in self.i_fundos:
                    continue
                if len(nomes_fundo) == 1:
                    self.atomos[0, k] = self._numero(s, "atomos_do_fundo", faixa="non-negative")
                    continue
                self._objeto(s, "atomos_do_fundo")
                for j, f in enumerate(nomes_fundo):
                    self.atomos[j, k] = self._numero(s, "atomos_do_fundo", f, faixa="non-negative")

    def _prepara_energia(self, quimica, fundos, bolsigdb, ganho_tres_corpos):
        """Energy exchanged by the electrons in each reaction outside the table:
        collisions with species that are not the background gas (in argon, those
        of Ar*), absent from the power tabulated for the background, and the gain
        of the electron left in three-body recombination (two electrons react, one
        remains). Assumption for the gain: from 0 (default) to the ionization
        energy of the gas (15.76 eV in argon). With the heat block of gas.json,
        what the electron does not take becomes gas heat (an upper bound:
        radiation is not subtracted). The BOLSIG+ reactions are left out: their
        energy comes from bolsigdb.dat."""
        # chemistry from quimica.json: the thresholds come from it, never from the
        # bolsigdb.dat at the repository root (argon); read from zdplaskin_m.F90 without a given
        # bolsigdb, the one at the repository root
        if bolsigdb is None and not getattr(quimica, "do_json", False):
            bolsigdb = RAIZ / "bolsigdb.dat"
        energia = quimica.energia_eletronica(bolsigdb, [nome for nome, _ in fundos])
        ie = quimica.indice["E"]
        tres = ((quimica.ordem[ie] >= 2) & (quimica.estequiometria[ie] == -1)
                & np.array([c is None for c in quimica.coluna_bolsig]))
        self.energia_reacao = energia + np.where(tres, ganho_tres_corpos, 0.0)

    def _le_difusao_e_perda(self, fundo, gas_solto):
        """Diffusion of the neutrals, m2/s, at the T and N of the gas, and the wall
        loss factor: Chantry, s/(2 - s) of the thermal flux n v/2 for a sticking
        coefficient s (1 for ions, electrons and the background gas, which the
        wall returns)."""
        self._dif = {}
        self.fator_parede = np.ones(len(self.nomes))
        fundo_solto = set(self.i_fundos) if gas_solto else set()
        for k, s in enumerate(self.nomes):
            if s == "E" or self.carga[k] != 0:
                continue
            dif = self._objeto(s, "difusao")
            self._numero(s, "difusao", "D0")
            self._numero(s, "difusao", "T0")
            self._numero(s, "difusao", "expoente", faixa="finite")
            self._dif[k] = dif
            if k not in fundo_solto:
                gama = self._numero(s, "perda_na_parede", faixa="between 0 and 1")
                self.fator_parede[k] = gama / (2.0 - gama)
        p_atm = (self.n_gas * 1e6) * (1.38e-23 * self.t_gas) / 101325.0
        self.difusao = np.zeros(len(self.nomes))
        for k, dif in self._dif.items():
            self.difusao[k] = dif["D0"] / p_atm * (self.t_gas / dif["T0"]) ** dif["expoente"]

    def _le_gas(self, quimica, fundo):
        """Validates gas.json and returns the list (NAME, fraction) of the background gas."""
        g = self.gas
        for chave in ("fundo", "cv_por_kb", "condutividade", "ion_inicial"):
            if chave not in g:
                raise ErroGas(f"gas.json: {chave} is missing")
        if not isinstance(g["fundo"], dict) or not g["fundo"]:
            raise ErroGas(f"gas.json: fundo {g['fundo']!r}: must be a dictionary of species and fraction")
        fundos = []
        for nome, fracao in g["fundo"].items():
            if nome.upper() not in quimica.indice:
                raise ErroGas(f"gas.json: the background gas {nome} is not in the chemistry")
            if not isinstance(fracao, (int, float)) or isinstance(fracao, bool) or not fracao > 0.0:
                raise ErroGas(f"gas.json: fraction of the background gas {nome} = {fracao!r}: must be positive")
            if nome.upper() in [n for n, _ in fundos]:
                raise ErroGas(f"gas.json: the background gas {nome.upper()} appears twice")
            fundos.append((nome.upper(), float(fracao)))
        if abs(sum(f for _, f in fundos) - 1.0) > 1e-9:
            raise ErroGas(f"gas.json: the fractions of the background gas add up to {sum(f for _, f in fundos):g}, not 1")
        if fundo is not None and [fundo.upper()] != [n for n, _ in fundos]:
            raise ErroGas(f"background gas {fundo} differs from gas.json ({', '.join(n for n, _ in fundos)})")
        cv = g["cv_por_kb"]
        if not isinstance(cv, (int, float)) or isinstance(cv, bool) or not np.isfinite(cv) or cv <= 0.0:
            raise ErroGas(f"gas.json: cv_por_kb = {cv!r}: must be a positive number")
        self.cv_por_kb = float(cv)
        c = g["condutividade"]
        if not isinstance(c, dict) or c.get("tipo") not in ("linear_com_raiz", "tabela"):
            raise ErroGas(f"gas.json: condutividade {c!r}: tipo must be linear_com_raiz or tabela")
        faltam = [k for k in {"linear_com_raiz": ("a", "b", "escala", "t_raiz"), "tabela": ("T", "kappa")}[c["tipo"]]
                  if k not in c]
        if faltam:
            raise ErroGas(f"gas.json: condutividade without {', '.join(faltam)}")
        if c["tipo"] == "tabela" and (len(c["T"]) != len(c["kappa"]) or len(c["T"]) < 2
                                      or np.any(np.diff(c["T"]) <= 0) or np.any(np.asarray(c["kappa"]) <= 0)):
            raise ErroGas("gas.json: condutividade as a table needs increasing T and positive kappa, of the same length")
        return fundos

    def _calor_das_reacoes(self, quimica, ganho_tres_corpos):
        """Heat (eV per event) of each reaction, -sum(nu h), with the enthalpies of
        the heat block (calor) of gas.json; None without the block. The BOLSIG+
        reactions (the energy comes from the electrons, in P_inel) and those with a
        single reactant (radiative: the photon leaves) are left out. In three-body
        recombination, the gain of the remaining electron is subtracted."""
        calor = self.gas.get("calor")
        if calor is None:
            return None
        ent = {k.upper(): v for k, v in calor.get("entalpias", {}).items()}
        ent["E"] = 0.0
        ie = quimica.indice["E"]
        nu, ordem = quimica.estequiometria, quimica.ordem
        q = np.zeros(len(quimica.reacoes))
        faltam = set()
        for r in range(len(quimica.reacoes)):
            if quimica.coluna_bolsig[r] is not None or ordem[:, r].sum() <= 1:
                continue
            for s in np.nonzero(nu[:, r])[0]:
                nome = quimica.especies[s].upper()
                if nome not in ent:
                    faltam.add(nome)
                else:
                    q[r] -= nu[s, r] * ent[nome]
            if ordem[ie, r] >= 2 and nu[ie, r] == -1:
                q[r] -= ganho_tres_corpos
        if faltam:
            raise ErroGas(f"gas.json: the enthalpy of {', '.join(sorted(faltam))} is missing in the calor block")
        self._eletrons_perdidos = np.where(np.array([c is None for c in quimica.coluna_bolsig]),
                                           np.minimum(nu[ie], 0.0), 0.0)
        return q

    def condutividade(self, t):
        """Thermal conductivity of the gas (W/(m K)) at the temperature t (K)."""
        c = self.gas["condutividade"]
        t = np.asarray(t, dtype=float)
        if c["tipo"] == "tabela":
            return np.interp(t, c["T"], c["kappa"])
        k = (c["a"] * t + c["b"]) * c["escala"]
        return np.where(t > c["t_raiz"], k * np.sqrt(np.maximum(t, c["t_raiz"]) / c["t_raiz"]), k)

    def _dado(self, especie, campo):
        """Field of the transport data of a species; an explicit error if missing."""
        if especie not in self.transporte:
            raise ErroTransporte(f"transport: no data for the species {especie}")
        if campo not in self.transporte[especie]:
            raise ErroTransporte(f"transport: the species {especie} has no {campo}")
        return self.transporte[especie][campo]

    def _objeto(self, especie, campo):
        valor = self._dado(especie, campo)
        if not isinstance(valor, dict):
            raise ErroTransporte(f"transport: {especie} {campo} = {valor!r}: must be an object")
        return valor

    FAIXAS = {"positive": lambda v: v > 0.0, "non-negative": lambda v: v >= 0.0,
              "between 0 and 1": lambda v: 0.0 <= v <= 1.0, "finite": lambda v: True}

    def _numero(self, especie, campo, chave=None, faixa="positive"):
        """Finite number in the field (or field.key) of a species, in the given range."""
        if chave:
            alvo = self._objeto(especie, campo)
            if chave not in alvo:
                raise ErroTransporte(f"transport: the species {especie} has no {campo}.{chave}")
            valor = alvo[chave]
        else:
            valor = self._dado(especie, campo)
        numero = isinstance(valor, (int, float)) and not isinstance(valor, bool) and np.isfinite(valor)
        if not (numero and self.FAIXAS[faixa](valor)):
            nome = f"{campo}.{chave}" if chave else campo
            raise ErroTransporte(f"transport: {especie} {nome} = {valor!r}: must be a {faixa} number")
        return float(valor)

    def troca_malha(self, faces):
        """Uses the mesh of the given faces (same number of cells, ends 0 and L)."""
        if len(faces) != self.m + 1 or faces[0] != 0.0 or abs(faces[-1] - self.L) > 1e-15 * self.L:
            raise ValueError("mesh incompatible with the model")
        if np.any(np.diff(faces) <= 0.0):
            raise ValueError("faces of the mesh out of order")
        self.faces = np.array(faces, dtype=float)
        self.faces[-1] = self.L
        self.h = np.diff(self.faces)                                     # widths
        self.xc = 0.5 * (self.faces[1:] + self.faces[:-1])               # centres
        nos = np.concatenate([[0.0], self.xc, [self.L]])
        self.d = np.diff(nos)                                            # m+1 distances between nodes

    # ------------------------------------------------------------------ field
    def tensao(self, t):
        """Voltage applied to the left electrode (V) at time t (s)."""
        if self.forma == "seno":
            return self.v_pico * np.sin(self.omega * t)
        if self.forma == "degrau":
            return self.v_pico * min(t / self.subida, 1.0)
        s = t % self.periodo
        if self.forma == "tabela":
            return float(np.interp(s, self.t_tab, self.v_tab, left=0.0, right=0.0))
        if s < self.subida:
            return self.v_pico * s / self.subida
        if s < self.subida + self.largura:
            return self.v_pico
        if s < self.subida + self.largura + self.descida:
            return self.v_pico * (self.subida + self.largura + self.descida - s) / self.descida
        return 0.0

    def poisson(self, rho, sigma_esq, sigma_dir, v):
        """phi at the nodes (walls and centres) and the field at the m+1 faces. rho in C/m3."""
        m, d = self.m, self.d
        n = m + 2
        ab = np.zeros((3, n))       # banded form of solve_banded: upper, diagonal, lower
        rhs = np.zeros(n)
        # cells: (phi_{i+1}-phi_i)/d_i - (phi_i-phi_{i-1})/d_{i-1} = -rho h/eps0
        for_cel = np.arange(1, m + 1)
        g_w = self.eps_gas / d[for_cel - 1]
        g_e = self.eps_gas / d[for_cel]
        ab[1, for_cel] = -(g_w + g_e)
        ab[0, for_cel + 1] = g_e
        ab[2, for_cel - 1] = g_w
        rhs[for_cel] = -rho * self.h / EPS0
        if self.espessura > 0.0:
            c = self.eps_d / self.espessura
            c_dir = self.eps_d_dir / self.espessura_dir
            # c*(V - phi_0) + (phi_1 - phi_0)/d_0 + sigma/eps0 = 0
            ab[1, 0] = -(c + self.eps_gas / d[0])
            ab[0, 1] = self.eps_gas / d[0]
            rhs[0] = -c * v - sigma_esq / EPS0
            # -c_dir*phi_n + (phi_{n-1} - phi_n)/d_m + sigma/eps0 = 0
            ab[1, n - 1] = -(c_dir + self.eps_gas / d[-1])
            ab[2, n - 2] = self.eps_gas / d[-1]
            rhs[n - 1] = -sigma_dir / EPS0
        else:
            ab[1, 0], rhs[0] = 1.0, v
            ab[1, n - 1], rhs[n - 1] = 1.0, 0.0
        phi = solve_banded((1, 1), ab, rhs)
        campo = -np.diff(phi) / d
        return phi, campo

    def campo_nas_celulas(self, campo):
        """Field at the cell centres, interpolated from the field at the faces."""
        nos = np.concatenate([[0.0], self.xc, [self.L]])
        x_f = 0.5 * (nos[1:] + nos[:-1])                 # midpoint of each pair of nodes
        w = (self.xc - x_f[:-1]) / (x_f[1:] - x_f[:-1])
        return (1.0 - w) * campo[:-1] + w * campo[1:]

    # ------------------------------------------------------------ coefficients
    # measured reduced mobilities (cm2/(V s), at the Loschmidt density), constant or
    # from the formula of Frost, K = K0*(1 + (E/N)/(E/N)_h)**(-1/2); values and
    # sources in transporte.json

    def mobilidade_ion(self, k, campo, n_pesadas=None):
        """Mobility (m2/(V s)) of the ion k in the cells, from the measurements,
        with mu = K*N0/N; N is that of the gas or, if given, that of the heavy particles."""
        mob = self._mob[k]
        escala = N_LOSCHMIDT / (self.n_gas if n_pesadas is None else n_pesadas) * 1e-4
        if mob["tipo"] == "constante":
            return np.full_like(np.asarray(campo, dtype=float), mob["K0"] * escala)
        e_n = np.abs(campo) / (self.n_gas if n_pesadas is None else n_pesadas) * 1e15
        if mob["tipo"] == "tabela":
            # K0 measured versus E/N, linear; below, held; above, held or with the
            # high-field tail K ~ (E/N)^(-1/2) from the last measured point
            k0 = np.interp(e_n, mob["E_N"], mob["K0"])
            if mob.get("acima", "preso") == "raiz":
                ultimo = mob["E_N"][-1]
                k0 = np.where(e_n > ultimo, mob["K0"][-1] * np.sqrt(ultimo / np.maximum(e_n, ultimo)), k0)
            return k0 * escala
        return mob["K0"] * escala / np.sqrt(1.0 + e_n / mob["EN_aquecimento"])

    def n_pesadas(self, n):
        """Heavy particles per cell (cm-3): the background gas and the species that
        are not electrons."""
        pesadas = [k for k in range(self.n_esp) if k != self.i_e]
        soma = np.maximum(n[pesadas], 0.0).sum(axis=0)
        return soma if self.gas_solto else self.n_gas + soma

    def n_neutro(self, n):
        """N of the electrons (cm-3): that of the gas with the fixed gas; with the
        free gas, the neutrals of the cell (background gas and transported
        neutrals), as the all_neutral of ZDPlasKin."""
        if not self.gas_solto:
            return self.n_gas
        return np.maximum(n[self.neutros], 0.0).sum(axis=0)

    def coeficientes(self, campo_cel, n=None, n_eps=None, t_gas=None):
        """Mobility (m2/(V s)), diffusion (m2/s) and temperature (K) of each species
        in the cells, electron impact rate coefficients (cells, columns), E/N (Td)
        and the extra coefficients: P_el/N (eV cm3/s) and, with the energy
        equation, those of the electron energy. With t_gas (K, per cell), the local
        temperature and heavy particles enter the coefficients of ions and neutrals."""
        n_e_neutro = self.n_neutro(n)
        e_n = np.abs(campo_cel) / n_e_neutro * 1e15
        mob = np.zeros((self.n_esp, self.m))
        dif = np.zeros((self.n_esp, self.m))
        local = t_gas is not None or self.gas_solto
        tg = np.full(self.m, self.t_gas) if t_gas is None else np.asarray(t_gas, dtype=float)
        # the ion mobility always uses the N of the heavy particles of the cell; the
        # diffusion of the neutrals, only with heating or the free gas
        n_pes = self.n_pesadas(n) if n is not None else None
        temp = np.tile(tg, (self.n_esp, 1))
        extra = {}
        if self.energia:
            eps, eps_bruta = self.energia_media(n, n_eps, e_n, tg, com_bruta=True)
            v = self.tab_e(eps, tg)
            # electrons below the floor of the table (colder than the gas): the table
            # does not have them (BOLSIG+ gives the regime with a field), and the
            # value at the floor would keep heating the gas; the elastic loss gains
            # the relaxation (2m/M)(nu/N)(eps - floor) and becomes negative, taking
            # them to the gas
            pel = v[:, E_PEL]
            abaixo = eps_bruta < eps
            if np.any(abaixo):
                pel = pel + np.where(abaixo, 2.0 * ME / self.m_fundo * v[:, E_NUN] * (eps_bruta - eps), 0.0)
            mob[self.i_e] = v[:, E_MUN] / n_e_neutro * 1e-4
            dif[self.i_e] = v[:, E_DN] / n_e_neutro * 1e-4
            temp[self.i_e] = (2.0 / 3.0) * eps * Q / KB
            k_bolsig = v[:, E_K0:]
            extra = {"eps": eps, "k_bolsig": k_bolsig, "mu_eps": v[:, E_MUEN] / n_e_neutro * 1e-4,
                     "d_eps": v[:, E_DEN] / n_e_neutro * 1e-4, "pot": v[:, E_POT],
                     "pel": pel, "perda": pel + v[:, E_PINEL],
                     "nu": v[:, E_NUN]}
        else:
            v = self.tab(e_n, tg)
            mob[self.i_e] = v[:, 1] / n_e_neutro * 1e-4
            dif[self.i_e] = v[:, 2] / n_e_neutro * 1e-4
            temp[self.i_e] = v[:, 0]
            k_bolsig = v[:, 5:]
            extra = {"pel": v[:, PEL], "nu": v[:, NUN]}
        extra["n"] = n_e_neutro
        extra["tg"] = tg
        extra["e_n"] = e_n                      # local E/N (Td), for the rates that use EN
        for k in range(self.n_esp):
            if k == self.i_e:
                continue
            if self.carga[k] != 0:
                mob[k] = self.mobilidade_ion(k, campo_cel, n_pes)
                dif[k] = mob[k] * KB * tg / (abs(self.carga[k]) * Q)       # Einstein, charge z
            elif local:
                p_atm = (n_pes * 1e6) * (1.38e-23 * tg) / 101325.0
                d = self._dif[k]
                dif[k] = d["D0"] / p_atm * (tg / d["T0"]) ** d["expoente"]
            else:
                dif[k] = self.difusao[k]
        return mob, dif, temp, k_bolsig, e_n, extra

    # electron density below which the mean energy tends to the local field value
    N_MIN_ENERGIA = 1.0e-2        # cm-3

    def energia_media(self, n, n_eps, e_n, t_gas=None, com_bruta=False):
        """Mean energy (eV) per cell, held within the range of the table at the T_g
        of the cell (with the table in T_g, the floor is ~(3/2) k T_g):
        (n_eps + n_min*eps_LFA)/(n_e + n_min). With n_e >> n_min it is n_eps/n_e;
        where there are no electrons (n_e zero or slightly negative from the
        integrator) the ratio has no meaning and the local field energy applies.
        n_min is well above the noise of the densities in the integrator
        (absolute tolerance)."""
        n_min = self.N_MIN_ENERGIA
        tg = np.full(self.m, self.t_gas) if t_gas is None else np.asarray(t_gas, dtype=float)
        lfa = self.tab_e.energia_de_campo_local(e_n, tg)
        piso, teto = self.tab_e.limites(tg)
        ne = np.maximum(n[self.i_e], 0.0)
        # energy outside the range of the table, counted before holding, where there
        # are enough electrons for the ratio to have a meaning
        com_eletrons = ne > 1.0e3 * n_min
        if np.any(com_eletrons):
            bruta = n_eps[com_eletrons] / ne[com_eletrons]
            self.tab_e.fora += int(np.count_nonzero(bruta > teto[com_eletrons] * (1.0 + 1e-12)))
            self.tab_e.fora_abaixo += int(np.count_nonzero(bruta < piso[com_eletrons] * (1.0 - 1e-12)))
        # n_eps does not exceed eps_max*n_e: without electrons there is no electron energy
        n_eps = np.clip(n_eps, 0.0, teto * ne)
        eps = (n_eps + n_min * lfa) / (ne + n_min)
        # com_bruta: also the energy before holding it at the floor, for the
        # relaxation of electrons colder than the gas
        return (np.clip(eps, piso, teto), eps) if com_bruta else np.clip(eps, piso, teto)

    # ------------------------------------------------------------------ fluxes
    def sg(self, n, carga, mu_c, d_c, campo):
        """Scharfetter-Gummel fluxes at the interior faces for densities n (rows,
        cells) of the given charge, with the mobility and diffusion of the cells."""
        d_int = self.d[1:-1]
        mu_f = media_harmonica(mu_c[:, :-1], mu_c[:, 1:])
        d_f = media_harmonica(d_c[:, :-1], d_c[:, 1:])
        w = carga[:, None] * mu_f * campo[None, 1:-1]
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            pe = np.where(d_f > 0.0, w * d_int / d_f, np.sign(w) * 1e300)
            gsg = (d_f / d_int) * (bernoulli(-pe) * n[:, :-1] - bernoulli(pe) * n[:, 1:])
        gup = np.maximum(w, 0.0) * n[:, :-1] + np.minimum(w, 0.0) * n[:, 1:]
        return np.where(d_f <= 0.0, gup, gsg)

    def fluxos(self, n, campo, mob, dif, temp):
        """Fluxes at the m+1 faces (cm-3 m/s, positive towards +x) and the fluxes
        reaching each wall, per species."""
        G = np.zeros((self.n_esp, self.m + 1))
        G[:, 1:-1] = self.sg(n, self.sinal, mob, dif, campo)
        # walls
        v_term = np.sqrt(8.0 * KB * temp / (np.pi * self.massa[:, None]))
        out = np.zeros((self.n_esp, 2))
        for lado, (cel, face, sinal) in enumerate(((0, 0, -1.0), (self.m - 1, self.m, 1.0))):
            ce = self.carga * campo[face]                  # force per e: the Boltzmann factor uses |z| and E
            w_n = sinal * mob[:, cel] * self.sinal * campo[face]      # drift towards the wall
            meia = self.d[face]
            with np.errstate(over="ignore"):
                f = np.where((w_n < 0.0) & (temp[:, cel] > 0.0) & self.freio,
                             np.exp(-Q * np.abs(ce) * meia / (KB * temp[:, cel])), 1.0)
            out[:, lado] = n[:, cel] * (self.fator_parede * f * 0.5 * v_term[:, cel] + np.maximum(w_n, 0.0))
        out[self.i_e] *= self.absorcao_eletrons
        gama_ions = self.gama * out[self.ions].sum(axis=0)
        G[:, 0] = -out[:, 0]
        G[:, -1] = out[:, 1]
        if self.gas_solto:
            # the background gas is not absorbed: each of its species receives the
            # atoms the wall absorbed (the background has zero weight in atomos)
            retorno = self.atomos @ out
            for j, i in enumerate(self.i_fundos):
                G[i, 0] = retorno[j, 0]
                G[i, -1] = -retorno[j, 1]
        G[self.i_e, 0] += gama_ions[0]
        G[self.i_e, -1] -= gama_ions[1]
        return G, out, gama_ions

    def fluxos_energia(self, n, n_eps, campo, mob, temp, extra, gama_ions):
        """Electron energy flux at the m+1 faces (eV cm-3 m/s, positive towards +x):
        Scharfetter-Gummel in the interior; at the walls, 5/2 kTe per electron of
        the thermal flux reduced by the Boltzmann factor (energy flux (5/6) v_th n_eps of Hagelaar et al.,
        Phys. Rev. E 62, 1452, 2000), mu_eps*E*n_eps in the drift towards the wall
        and eps_sec per incoming secondary electron."""
        mu_e = extra["mu_eps"][None, :]
        d_e = extra["d_eps"][None, :]
        Ge = np.zeros(self.m + 1)
        Ge[1:-1] = self.sg(n_eps[None, :], np.array([-1.0]), mu_e, d_e, campo)[0]
        ie = self.i_e
        te = temp[ie]
        v_term = np.sqrt(8.0 * KB * te / (np.pi * ME))
        saida = np.zeros(2)
        for lado, (cel, face, sinal) in enumerate(((0, 0, -1.0), (self.m - 1, self.m, 1.0))):
            w_n = -sinal * mob[ie, cel] * campo[face]            # drift of the electrons towards the wall
            w_eps = -sinal * extra["mu_eps"][cel] * campo[face]
            with np.errstate(over="ignore"):
                f = 1.0
                if w_n < 0.0 and self.freio:
                    f = np.exp(-Q * abs(campo[face]) * self.d[face] / (KB * te[cel]))
            saida[lado] = self.absorcao_eletrons * (n[ie, cel] * f * 0.5 * v_term[cel] * 2.5 * KB * te[cel] / Q
                                                    + max(w_eps, 0.0) * n_eps[cel])
        Ge[0] = -saida[0] + self.eps_sec * gama_ions[0]
        Ge[-1] = saida[1] - self.eps_sec * gama_ions[1]
        return Ge

    # ------------------------------------------------------------------- RHS
    def separa(self, y):
        """Densities of the species (without the energy), surface charges."""
        z = y[:-2].reshape(self.n_var, self.m)
        return z[:self.n_esp], y[-2], y[-1]

    def energia_do_estado(self, y):
        return y[:-2].reshape(self.n_var, self.m)[self.n_esp]

    def temperatura_do_estado(self, y):
        return y[:-2].reshape(self.n_var, self.m)[self.i_t]

    def densidade_neutra(self, n):
        """Total density of neutrals per cell (cm-3): the ALL_NEUTRAL of ZDPlasKin,
        third body of the reactions with ANY_NEUTRAL."""
        todas = self.densidades_completas(n)
        neutras = [s for s, nome in enumerate(self.q.especies) if self.q.carga[s] == 0 and nome.upper() != "E"]
        return todas[neutras].sum(axis=0)

    def densidades_completas(self, n):
        """Densities of every species of the chemistry (cm-3), with the fixed
        background gas at its fraction of N; negative values from the integrator
        become zero."""
        todas = np.empty((len(self.q.especies), self.m))
        todas[self.especies] = np.maximum(n, 0.0)
        if not self.gas_solto:
            for i, fracao in self.fundos:
                todas[i] = fracao * self.n_gas
        return todas

    def campo(self, t, n, sigma_esq, sigma_dir):
        """Potential at the nodes and field at the faces for the densities n."""
        rho = Q * 1e6 * (self.carga @ n)
        return self.poisson(rho, sigma_esq, sigma_dir, self.tensao(t))

    def rhs(self, t, y):
        """dy/dt of the full state, with the field of the Poisson equation at time t."""
        n, s_esq, s_dir = self.separa(y)
        _, campo = self.campo(t, n, s_esq, s_dir)
        return self.rhs_campo(t, y, campo)

    def rhs_campo(self, t, y, campo):
        """Right-hand side with the given field at the faces (rhs solves it first;
        the structured Jacobian perturbs it)."""
        n, s_esq, s_dir = self.separa(y)
        n_eps = self.energia_do_estado(y) if self.energia else None
        t_gas = self.temperatura_do_estado(y) if self.aquecimento else None
        campo_cel = self.campo_nas_celulas(campo)
        mob, dif, temp, k_bolsig, e_n, extra = self.coeficientes(campo_cel, n, n_eps, t_gas)
        G, out, gama_ions = self.fluxos(n, campo, mob, dif, temp)
        dn = -np.diff(G, axis=1) / self.h
        tg = t_gas if self.aquecimento else np.full(self.m, self.t_gas)
        k = self.q.coeficientes(k_bolsig, tg, temp[self.i_e], e_n,
                                self.densidade_neutra(n) if self.q.usa_any_neutral else None)
        if self.calor_reacao is not None:
            taxas = self.q.taxas(self.densidades_completas(n), k)
            extra["taxas"] = taxas
            fontes = self.q.estequiometria @ taxas
        else:
            fontes = self.q.fontes(self.densidades_completas(n), k)
        dn += fontes[self.especies]
        partes = [dn.ravel()]
        if self.energia:
            partes.append(self.d_energia(n, n_eps, campo, G, mob, temp, extra, gama_ions))
        if self.aquecimento:
            partes.append(self.d_temperatura(n, t_gas, campo_cel, mob, extra))
        # surface charges: the arriving particles and the emitted electrons
        d_sigma = Q * 1e6 * (self.carga @ out + gama_ions)
        return np.concatenate(partes + [d_sigma])

    def d_temperatura(self, n, t_gas, campo_cel, mob, extra):
        """dT/dt (K/s): conduction with kappa(T) (harmonic mean at the faces, wall at
        t_parede half a cell away), elastic electron loss and Joule heating of the
        ions, divided by c_v N k_B with N the heavy particles."""
        kap = self.condutividade(t_gas)
        k_par = self.condutividade(np.array(self.t_parede))
        F = np.empty(self.m + 1)                         # W/m2, positive towards +x
        F[1:-1] = -media_harmonica(kap[:-1], kap[1:]) * np.diff(t_gas) / self.d[1:-1]
        F[0] = -media_harmonica(k_par, kap[0]) * (t_gas[0] - self.t_parede) / self.d[0]
        F[-1] = -media_harmonica(kap[-1], k_par) * (self.t_parede - t_gas[-1]) / self.d[-1]
        ne = np.maximum(n[self.i_e], 0.0)
        q_el = Q * extra["pel"] * extra["n"] * ne * 1e6                 # W/m3
        q_j = self.joule_ions(n, mob, campo_cel)
        fonte = -np.diff(F) / self.h + q_el + q_j
        if self.calor_reacao is not None:
            # heat of the reactions (enthalpies of gas.json) and, with the energy
            # equation, the mean energy carried by the consumed electrons (d_energia removes it)
            q_reac = self.calor_reacao @ extra["taxas"]
            if self.energia:
                q_reac = q_reac - extra["eps"] * (self._eletrons_perdidos @ extra["taxas"])
            fonte = fonte + Q * 1e6 * q_reac
        if self.tab_calor is not None:
            # P/N of the fast processes at the mean energy (with --energy) or at the
            # local field energy of the table itself, at the E/N and T_g of the cell
            tg = extra["tg"]
            eps = extra["eps"] if self.energia else self.tab_calor.energia_de_campo_local(extra["e_n"], tg)
            p_rap = self.tab_calor(eps, tg)[:, 1]
            fonte = fonte + Q * 1e6 * p_rap * extra["n"] * ne
        return fonte / (self.cv_por_kb * KB * self.n_pesadas(n) * 1e6)

    def joule_ions(self, n, mob, campo_cel):
        """Joule heating of the ions (W/m3), |z| e n mu E^2, positive and negative."""
        ions = self.ions + self.ions_negativos
        return sum(abs(self.carga[k]) * Q * np.maximum(n[k], 0.0) * 1e6 * mob[k] * campo_cel ** 2 for k in ions)

    def ganho_de_energia(self, G, campo):
        """Work of the field on the electrons per cell (eV cm-3/s), -Gamma_e*E
        averaged over the two faces of the cell. In the wall cells only that of the
        inner face counts: the absorption flux at the wall face is dominated by the
        thermal motion (v_th/2 ~ 1e6 m/s), which the field does not do, and it
        heated the cell several times more than the volume."""
        trabalho = -G[self.i_e] * campo
        ganho = 0.5 * (trabalho[:-1] + trabalho[1:])
        ganho[0] = trabalho[1]
        ganho[-1] = trabalho[-2]
        return ganho

    def d_energia(self, n, n_eps, campo, G, mob, temp, extra, gama_ions):
        """d(n_eps)/dt (eV cm-3/s): transport, field work with the electron fluxes of
        the two faces of the cell, tabulated elastic and inelastic loss (P_el +
        P_inel), energy exchanged in collisions with species outside the
        background (Ar* in argon), mean energy carried away by each lost electron
        and the optional gain of the third electron in three-body recombination."""
        Ge = self.fluxos_energia(n, n_eps, campo, mob, temp, extra, gama_ions)
        ganho = self.ganho_de_energia(G, campo)
        perda = np.maximum(n[self.i_e], 0.0) * extra["n"] * extra["perda"]
        # reactions: superelastic collisions and stepwise ionization with Ar* give or
        # take the threshold energy; each electron lost in recombination takes the
        # mean energy (new electrons enter without energy: ionization already pays
        # the threshold in P_inel and the rest stays with the electrons)
        k = self.q.coeficientes(extra["k_bolsig"], extra["tg"], temp[self.i_e], extra["e_n"],
                                self.densidade_neutra(n) if self.q.usa_any_neutral else None)
        taxas = self.q.taxas(self.densidades_completas(n), k)
        perdidos = np.minimum(self.q.estequiometria[self.q.indice["E"]], 0.0)
        outras = self.energia_reacao @ taxas + extra["eps"] * (perdidos @ taxas)
        return -np.diff(Ge) / self.h + ganho - perda + outras

    def fora_da_tabela(self, y):
        """Cells of a state with mean energy above and below the range of the
        table, where there are enough electrons (n_e > 10 cm-3), for the log."""
        n = self.separa(y)[0]
        ne = n[self.i_e]
        com = ne > 1.0e3 * self.N_MIN_ENERGIA
        bruta = self.energia_do_estado(y)[com] / ne[com]
        tg = self.temperatura_do_estado(y) if self.aquecimento else np.full(self.m, self.t_gas)
        piso, teto = self.tab_e.limites(tg)
        return (int(np.count_nonzero(bruta > teto[com] * (1.0 + 1e-12))),
                int(np.count_nonzero(bruta < piso[com] * (1.0 - 1e-12))))

    def frequencia_de_colisao(self, n, extra):
        """Collision frequency of the electrons (1/s) per cell, as in the Fortran
        code: (nu/N) of the table times the sum of all densities of the cell."""
        return extra["nu"] * (self.n_pesadas(n) + np.maximum(n[self.i_e], 0.0))

    def celula_muito_ionizada(self, y):
        """First cell with n_e/N above 1e-3 with the fixed gas, where the assumption
        of a fixed background stops holding; None if there is none."""
        if self.gas_solto:
            return None
        acima = np.nonzero(self.separa(y)[0][self.i_e] > 1e-3 * self.n_gas)[0]
        return int(acima[0]) if len(acima) else None

    T_ALTA = 1000.0     # K: above it the extrapolated kappa stops holding

    def celula_quente(self, y):
        """First cell with the gas above T_ALTA (with heating); None if there is none."""
        if not self.aquecimento:
            return None
        acima = np.nonzero(self.temperatura_do_estado(y) > self.T_ALTA)[0]
        return int(acima[0]) if len(acima) else None

    def estado_inicial(self, n_e0=1.0, neutro=False):
        """As in the Fortran code: only electrons at n_e0 cm-3, uncharged surfaces;
        with neutro, the initial ion of gas.json also at n_e0 (a quasi-neutral
        initial plasma); with the energy equation, the mean energy of zero field."""
        n = np.zeros((self.n_var, self.m))
        n[self.i_e] = n_e0
        if neutro:
            n[self.ion_inicial] = n_e0
        if self.energia:
            n[self.n_esp] = n_e0 * self.tab_e.limites(np.array([self.t_gas]))[0][0]
        if self.aquecimento:
            n[self.i_t] = self.t_gas
        if self.gas_solto:
            for (_, fracao), i in zip(self.fundos, self.i_fundos):
                n[i] = fracao * self.n_gas
        return np.concatenate([n.ravel(), [0.0, 0.0]])
