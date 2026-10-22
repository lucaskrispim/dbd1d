# SPDX-License-Identifier: GPL-3.0-or-later
"""Walls, Poisson with charge, field in the cells and constants, with answers
derived by hand and CODATA constants written here, without importing them from the
model (gaps VT1 to VT4 of docs/lacunas.md; they correspond to teste_parede.f90,
teste_dbd.f90, teste_poisson.f90, teste_malha.f90 and teste_quimica.f90)."""
import numpy as np
import pytest

import modelo
from conftest import RAIZ, quimica_do
from modelo import Modelo
from quimica_zdp import Quimica
from tabela import Tabela

KB = 1.380649e-23
QE = 1.602176634e-19
EPS0 = 8.8541878128e-12
UMA = 1.66053906660e-27
M_AR = 39.948 * UMA
N0 = 2.6867811e19                      # cm-3, Loschmidt
N_GAS = 101325.0 / (KB * 300.0) * 1e-6  # cm-3


@pytest.fixture(scope="module")
def pecas():
    return quimica_do(RAIZ), Tabela(RAIZ / "tabela_bolsig.dat")


def v_term(t, m):
    return np.sqrt(8.0 * KB * t / (np.pi * m))


def test_constantes_codata():
    assert modelo.Q == 1.602176634e-19
    assert modelo.KB == 1.380649e-23
    assert modelo.EPS0 == 8.8541878128e-12
    assert modelo.U == UMA                  # the mass of each species comes from transporte.json
    assert modelo.ME == 9.1093837015e-31
    assert modelo.N_LOSCHMIDT == 2.6867811e19


# ------------------------------------------------------------------ paredes
def estado_paredes(md):
    n = np.zeros((md.n_esp, md.m))
    for nome, valor in (("E", 1e9), ("AR^+", 2e9), ("AR2^+", 3e9), ("AR*", 4e9)):
        n[md.nomes.index(nome)] = valor
    return n


def test_fluxo_termico_de_cada_especie_sem_campo(pecas):
    # without field, each species carries n v_th/2 to each wall: Ar+ and Ar* with the mass
    # of argon, Ar2+ with twice it (v_th/sqrt 2), at 300 K
    md = Modelo(*pecas, n_celulas=10)
    n = estado_paredes(md)
    mob, dif, temp, *_ = md.coeficientes(np.zeros(md.m), n)
    _, out, _ = md.fluxos(n, np.zeros(md.m + 1), mob, dif, temp)
    for nome, dens, massa in (("AR^+", 2e9, M_AR), ("AR2^+", 3e9, 2 * M_AR), ("AR*", 4e9, M_AR)):
        esperado = dens * v_term(300.0, massa) / 2.0
        assert out[md.nomes.index(nome)] == pytest.approx([esperado, esperado], rel=1e-12, abs=0), nome


def test_deriva_dos_ions_nas_duas_paredes(pecas):
    # uniform field +E: the ions drift to the right. At the right wall
    # Gamma = n (v_th/2 + mu E); on the left the drift moves away and the thermal
    # flux is reduced by exp(-e E (h/2)/(k T)); mu of Ar+ by Frost with
    # K0 = 1.61 cm2/(V s) and 210 Td, of Ar2+ constant with K0 = 1.95. With 500 V/m
    # and a half cell of 40 um the Boltzmann factor is ~0.46, so the half cell counts
    md = Modelo(*pecas, n_celulas=10)
    n = estado_paredes(md)
    campo_v = 500.0
    campo = np.full(md.m + 1, campo_v)
    mob, dif, temp, *_ = md.coeficientes(np.full(md.m, campo_v), n)
    _, out, _ = md.fluxos(n, campo, mob, dif, temp)
    n_pes = N_GAS + 9e9                     # gas and the heavy species (Ar+, Ar2+, Ar*)
    e_n = campo_v / n_pes * 1e15
    mu = {"AR^+": 1.61 * N0 / n_pes * 1e-4 / np.sqrt(1.0 + e_n / 210.0),
          "AR2^+": 1.95 * N0 / n_pes * 1e-4}
    meia = md.L / md.m / 2.0
    freio = np.exp(-QE * campo_v * meia / (KB * 300.0))
    assert 0.1 < freio < 0.9
    for nome, dens, massa in (("AR^+", 2e9, M_AR), ("AR2^+", 3e9, 2 * M_AR)):
        k = md.nomes.index(nome)
        vt = v_term(300.0, massa)
        assert out[k, 1] == pytest.approx(dens * (vt / 2.0 + mu[nome] * campo_v), rel=1e-10, abs=0), nome
        assert out[k, 0] == pytest.approx(dens * freio * vt / 2.0, rel=1e-10, abs=0), nome


def test_emissao_secundaria_em_cada_parede(pecas):
    # each wall emits gamma times the ion flux that reaches it
    md = Modelo(*pecas, n_celulas=10, gama_sec=0.07)
    n = estado_paredes(md)
    campo = np.full(md.m + 1, 2.0e5)
    mob, dif, temp, *_ = md.coeficientes(np.full(md.m, 2.0e5), n)
    G, out, gama = md.fluxos(n, campo, mob, dif, temp)
    ions = [md.nomes.index("AR^+"), md.nomes.index("AR2^+")]
    esperado = 0.07 * out[ions].sum(axis=0)
    assert gama == pytest.approx(esperado, rel=1e-14, abs=0)
    ie = md.i_e
    assert G[ie, 0] == pytest.approx(-out[ie, 0] + esperado[0], rel=1e-14, abs=0)
    assert G[ie, -1] == pytest.approx(out[ie, 1] - esperado[1], rel=1e-14, abs=0)


# ------------------------------------------------------------------ Poisson
def test_campo_da_carga_superficial(pecas):
    # V = 0, +sigma at the left interface and -sigma at the right, without charge in the gas:
    # E_gas = sigma / (eps0 (1 + eps_d L/(2 d))), in the +x direction
    d, eps_d = 1.0e-3, 4.8
    md = Modelo(*pecas, n_celulas=20, agrupamento=1.5, espessura=d, eps_dieletrico=eps_d)
    sigma = 3.0e-6
    _, campo = md.poisson(np.zeros(md.m), sigma, -sigma, 0.0)
    esperado = sigma / (EPS0 * (1.0 + eps_d * md.L / (2.0 * d)))
    np.testing.assert_allclose(campo, esperado, rtol=1e-10)


def test_potencial_na_superficie_do_dieletrico(pecas):
    # capacitive divider: E_gas = V/(L + 2d/eps_d); the left surface is
    # at V - (E_gas/eps_d) d
    d, eps_d, v = 1.75e-3, 4.8, 1.0e3
    md = Modelo(*pecas, n_celulas=20, agrupamento=1.5, espessura=d, eps_dieletrico=eps_d)
    phi, campo = md.poisson(np.zeros(md.m), 0.0, 0.0, v)
    e_gas = v / (md.L + 2.0 * d / eps_d)
    np.testing.assert_allclose(campo, e_gas, rtol=1e-10)
    assert phi[0] == pytest.approx(v - e_gas / eps_d * d, rel=1e-10, abs=0)
    assert phi[-1] == pytest.approx(e_gas / eps_d * d, rel=1e-10, abs=0)


def test_poisson_com_carga_uniforme_de_segunda_ordem_na_malha_agrupada(pecas):
    # grounded electrodes and uniform rho: phi = rho x (L - x)/(2 eps0)
    rho = 1.0e-2                                   # C/m3
    erros = []
    for m in (20, 40, 80):
        md = Modelo(*pecas, n_celulas=m, agrupamento=2.0)
        phi, _ = md.poisson(np.full(m, rho), 0.0, 0.0, 0.0)
        exato = rho * md.xc * (md.L - md.xc) / (2.0 * EPS0)
        erros.append(np.max(np.abs(phi[1:-1] - exato)) / np.max(exato))
    ordens = np.log2(np.array(erros[:-1]) / np.array(erros[1:]))
    assert np.all(np.abs(ordens - 2.0) < 0.05), ordens


def test_campo_linear_exato_nas_celulas_da_malha_agrupada(pecas):
    # phi = x^2 at the nodes gives the exact field -2x at the midpoints between nodes; the
    # interpolation to the cell centres must give -2 x_c, including
    # in the wall cells, where the nodes lie half a cell away (N17)
    md = Modelo(*pecas, n_celulas=15, agrupamento=2.0)
    nos = np.concatenate([[0.0], md.xc, [md.L]])
    campo = -np.diff(nos ** 2) / np.diff(nos)
    np.testing.assert_allclose(md.campo_nas_celulas(campo), -2.0 * md.xc, rtol=1e-12)
    # linear field that changes sign: it remains exact and bounded by the extremes
    x_med = 0.5 * (nos[1:] + nos[:-1])
    campo = 3.0e5 * (2.0 * x_med / md.L - 1.0)
    cel = md.campo_nas_celulas(campo)
    np.testing.assert_allclose(cel, 3.0e5 * (2.0 * md.xc / md.L - 1.0), rtol=0, atol=1e-9 * 3.0e5)
    assert np.all(np.abs(cel) <= np.abs(campo).max())


def test_faces_da_malha_agrupada(pecas):
    # faces x_j = (L/2)(1 + tanh(beta(2j/m - 1))/tanh(beta)), as malha.f90
    md = Modelo(*pecas, n_celulas=12, agrupamento=2.0)
    j = np.arange(13)
    esperado = 0.5 * 8.0e-4 * (1.0 + np.tanh(2.0 * (2.0 * j / 12 - 1.0)) / np.tanh(2.0))
    np.testing.assert_allclose(md.faces, esperado, rtol=1e-12, atol=1e-18)


# ------------------------------------------------------------- coeficientes
def test_mobilidade_dos_eletrons_pela_tabela(pecas):
    # mu_e = (mu N)/N with mu N in 1/(cm V s), N in cm-3 and 1e-4 for m2/(V s) (F2)
    md = Modelo(*pecas, n_celulas=4)
    e_n = 100.0
    campo = e_n * 1e-21 * N_GAS * 1e6
    mob, dif, *_ = md.coeficientes(np.full(md.m, campo), np.zeros((md.n_esp, md.m)))
    mun, dn = md.tab(e_n)[0, 1], md.tab(e_n)[0, 2]
    assert mob[md.i_e] == pytest.approx(mun / N_GAS * 1e-4, rel=1e-12, abs=0)
    assert dif[md.i_e] == pytest.approx(dn / N_GAS * 1e-4, rel=1e-12, abs=0)


def test_gauss_e_conservacao_durante_a_blindagem(pecas):
    # quasi-neutral plasma of 1e11 cm-3 under constant voltage (the peak of a sine
    # of long period) with dielectric on both walls, integrated for 100 ns. There is no
    # simple final state: the electrons escape by thermal motion to both
    # walls and the plasma empties. Exact identities are checked at
    # any instant, which catch sign and factor errors of the surface
    # charge: (1) charge conservation, sigma_left + sigma_right + sum of
    # rho h = 0; (2) Gauss at the left electrode for each face k,
    # eps0 E_k = eps0 eps_d (V - phi_0)/d + sigma_left + sum up to k of rho h.
    # And the shielding: the field at the centre falls below 1% of the initial one.
    from scipy.integrate import solve_ivp
    from jacobiano import ModeloRapido
    v, d, eps_d = 50.0, 1.0e-3, 4.8
    md = ModeloRapido(*pecas, n_celulas=24, agrupamento=1.5, v_pico=v, periodo=1.0,
                      espessura=d, eps_dieletrico=eps_d)
    y0 = md.estado_inicial(1.0e11, neutro=True)
    t0 = 0.25                                       # voltage peak
    atol = np.concatenate([np.full(md.n_var * md.m, 1e-4), [1e-12, 1e-12]])
    sol = solve_ivp(md.rhs, (t0, t0 + 1e-7), y0, method="BDF", rtol=1e-6, atol=atol,
                    jac=lambda t, y: md.jacobiano(t, y, 1e-4))
    assert sol.success
    n, s_esq, s_dir = md.separa(sol.y[:, -1])
    phi, campo = md.campo(sol.t[-1], n, s_esq, s_dir)
    _, campo0 = md.campo(t0, *md.separa(y0))
    rho_h = QE * 1e6 * (md.carga @ n) * md.h                     # C/m2 per cell
    capacitor = EPS0 * eps_d * v / d
    assert abs(s_esq + s_dir + rho_h.sum()) < 1e-12 * capacitor
    v_t = md.tensao(sol.t[-1])
    gauss = EPS0 * eps_d * (v_t - phi[0]) / d + s_esq + np.concatenate([[0.0], np.cumsum(rho_h)])
    np.testing.assert_allclose(EPS0 * campo, gauss, rtol=0, atol=1e-10 * capacitor)
    assert abs(campo[md.m // 2]) < 1e-2 * np.abs(campo0).max()


@pytest.mark.parametrize("eps_gas", [1.0, 1.000516])
def test_permissividade_do_gas_no_poisson(pecas, eps_gas):
    # T4: with the relative permittivity of the gas, the capacitive divider gives
    # E_gas = V/(L + 2 d eps_gas/eps_d), and the surface charge gives
    # E = sigma/(eps0 (eps_gas + eps_d L/(2 d))); the same in the DAE matrix
    from jacobiano import ModeloRapido
    d, eps_d, v, sigma = 1.75e-3, 4.8, 1.0e3, 3.0e-6
    md = ModeloRapido(*pecas, n_celulas=20, agrupamento=1.5, espessura=d, eps_dieletrico=eps_d, eps_gas=eps_gas)
    _, campo = md.poisson(np.zeros(md.m), 0.0, 0.0, v)
    np.testing.assert_allclose(campo, v / (md.L + 2.0 * d * eps_gas / eps_d), rtol=1e-10)
    _, campo = md.poisson(np.zeros(md.m), sigma, -sigma, 0.0)
    np.testing.assert_allclose(campo, sigma / (EPS0 * (eps_gas + eps_d * md.L / (2.0 * d))), rtol=1e-10)
    # the DAE matrix and the response G use the same eps_gas
    from scipy.linalg import solve_banded
    phi = solve_banded((1, 1), md._matriz_poisson(), md._rhs_poisson(np.zeros(md.m), sigma, -sigma, 0.0))
    np.testing.assert_allclose(-np.diff(phi) / md.d, campo, rtol=1e-10)
    np.testing.assert_allclose(md.G[:, md.m] * sigma - md.G[:, md.m + 1] * sigma, campo, rtol=1e-10)
