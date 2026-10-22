# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests of the coupled model (operators and system), with answers derived by hand."""
import numpy as np
import pytest
from scipy.integrate import solve_ivp

from conftest import RAIZ, quimica_do
from modelo import EPS0, Q, Modelo, bernoulli
from quimica_zdp import Quimica
from tabela import Tabela


@pytest.fixture(scope="module")
def pecas():
    return quimica_do(RAIZ), Tabela(RAIZ / "tabela_bolsig.dat")


def test_bernoulli():
    x = np.array([-50.0, -1.0, -1e-10, 0.0, 1e-10, 1.0, 50.0, 800.0])
    b = bernoulli(x)
    assert b[3] == 1.0
    np.testing.assert_allclose(b - bernoulli(-x), -x, rtol=1e-12, atol=1e-12)


def test_scharfetter_gummel_exato_para_perfil_de_equilibrio(pecas):
    # with uniform mobility, diffusion and field, n = exp(w x/D) has zero
    # flux at every inner face: the SG scheme is exact for it, on a clustered mesh
    md = Modelo(*pecas, n_celulas=40, agrupamento=2.0)
    mu, dd, e = 0.03, 0.2, 2.0e4
    w = -mu * e                      # electrons
    n = np.zeros((md.n_esp, md.m))
    n[md.i_e] = np.exp(w * md.xc / dd)
    campo = np.full(md.m + 1, e)
    mob = np.full((md.n_esp, md.m), mu)
    dif = np.full((md.n_esp, md.m), dd)
    temp = np.full((md.n_esp, md.m), 300.0)
    G, _, _ = md.fluxos(n, campo, mob, dif, temp)
    assert np.max(np.abs(G[md.i_e, 1:-1])) < 1e-12 * dd * np.max(n[md.i_e]) / np.min(md.d)


def test_fluxo_na_face_com_salto_de_difusao_e_o_de_resistencias_em_serie(pecas):
    # without field and with D piecewise constant (D1 in one cell, D2 in the next, uniform
    # mesh), the steady flux between the two centres is that of two half cells in series,
    # (n1 - n2) / (h/(2 D1) + h/(2 D2)): the harmonic mean of D at the face over h
    md = Modelo(*pecas, n_celulas=4, agrupamento=0.0)
    h = md.d[1]
    n = np.array([[3.0e9, 1.0e9, 1.0e9, 1.0e9]])
    d_c = np.array([[1.0e-3, 9.0e-3, 9.0e-3, 9.0e-3]])
    mu_c = np.array([[0.1, 0.9, 0.9, 0.9]])
    g = md.sg(n, np.array([1.0]), mu_c, d_c, np.zeros(md.m + 1))
    esperado = (3.0e9 - 1.0e9) / (h / (2 * 1.0e-3) + h / (2 * 9.0e-3))
    assert g[0, 0] == pytest.approx(esperado, rel=1e-12)
    # with field and without diffusion, the upwind drift uses the same mean of the mobility
    e = np.full(md.m + 1, 1.0e4)
    g = md.sg(n, np.array([1.0]), mu_c, np.zeros_like(d_c), e)
    assert g[0, 0] == pytest.approx(2 * 0.1 * 0.9 / (0.1 + 0.9) * 1.0e4 * 3.0e9, rel=1e-12)


def test_fluxo_que_sai_pela_parede_esquerda_aponta_para_menos_x(pecas):
    # fluxes are positive towards +x: what leaves through the left wall is -out and
    # what leaves through the right wall is +out, out >= 0 being the absorbed flux;
    # for the electrons the secondary emission enters back (+ at the left)
    md = Modelo(*pecas, n_celulas=8, agrupamento=0.0, gama_sec=0.05)
    r = np.random.default_rng(1)
    n = 10.0 ** r.uniform(8, 11, (md.n_esp, md.m))
    mob = r.uniform(0.01, 0.1, (md.n_esp, md.m))
    dif = r.uniform(1e-4, 1e-3, (md.n_esp, md.m))
    temp = np.full((md.n_esp, md.m), 300.0)
    temp[md.i_e] = 2e4
    campo = np.linspace(-3e4, 5e4, md.m + 1)
    G, out, gama = md.fluxos(n, campo, mob, dif, temp)
    assert (out > 0.0).all()
    outros = [k for k in range(md.n_esp) if k != md.i_e]
    np.testing.assert_allclose(G[outros, 0], -out[outros, 0], rtol=1e-15)
    np.testing.assert_allclose(G[outros, -1], out[outros, 1], rtol=1e-15)
    assert G[md.i_e, 0] == pytest.approx(-out[md.i_e, 0] + gama[0], rel=1e-14)
    assert G[md.i_e, -1] == pytest.approx(out[md.i_e, 1] - gama[1], rel=1e-14)


def test_poisson_linear_na_malha_agrupada(pecas):
    md = Modelo(*pecas, n_celulas=40, agrupamento=2.0)
    phi, campo = md.poisson(np.zeros(md.m), 0.0, 0.0, 1.0e3)
    x = np.concatenate([[0.0], md.xc, [md.L]])
    np.testing.assert_allclose(phi, 1.0e3 * (1.0 - x / md.L), atol=1e-9)
    np.testing.assert_allclose(campo, 1.0e3 / md.L, rtol=1e-10)


def test_poisson_com_dieletrico_e_capacitores_em_serie(pecas):
    # without charge: continuous D, E_gas = V/(L + 2*thickness/eps_d)
    md = Modelo(*pecas, n_celulas=40, agrupamento=2.0, espessura=1.75e-3)
    _, campo = md.poisson(np.zeros(md.m), 0.0, 0.0, 1.0e3)
    np.testing.assert_allclose(campo, 1.0e3 / (md.L + 2.0 * 1.75e-3 / 4.8), rtol=1e-10)


def test_carga_total_conservada(pecas):
    # transport, walls, secondary emission and chemistry: the rate of change of the
    # charge of the gas plus that of the surfaces is zero
    md = Modelo(*pecas, n_celulas=30, agrupamento=1.5, espessura=1.75e-3, v_pico=5.0e3)
    gerador = np.random.default_rng(3)
    n = 10.0 ** gerador.uniform(8, 13, size=(md.n_esp, md.m))
    y = np.concatenate([n.ravel(), [1e-6, -2e-6]])
    dy = md.rhs(2.5e-8, y)
    dn, ds_esq, ds_dir = md.separa(dy)
    d_gas = Q * 1e6 * np.sum((md.carga @ dn) * md.h)
    escala = Q * 1e6 * np.sum(np.abs(md.carga[:, None] * dn) * md.h[None, :])
    assert abs(d_gas + ds_esq + ds_dir) < 1e-12 * escala


def test_relaxacao_de_maxwell(pecas):
    # uniform quasi-neutral plasma with a charge perturbation sin(k x), k = pi/L,
    # grounded electrodes. Linearizing (zero-field mobilities and diffusions),
    # the charge relaxes at the rate nu (1 + eps), nu = e(mu_e n_e + mu_i n_i)/eps0 and
    # eps = (D_e + D_i) k^2/nu ~ (lambda_D k)^2, down to the Boltzmann plateau
    # eps/(1 + eps): after t = 2/nu, R = [eps + exp(-2(1 + eps))]/(1 + eps).
    # The perturbation of 1e-6 gives a field below the first node of the table, where the
    # coefficients are constant (with 1e-4 the field reaches 2e-3 Td, where mu_e
    # varies by 27%, Ramsauer region, and the linearization fails)
    md = Modelo(*pecas, n_celulas=60, v_pico=0.0)
    n0 = 1.0e11
    n = np.zeros((md.n_esp, md.m))
    n[md.i_e] = n0
    i_ar = md.nomes.index("AR^+")
    n[i_ar] = n0 * (1.0 + 1e-6 * np.sin(np.pi * md.xc / md.L))
    y0 = np.concatenate([n.ravel(), [0.0, 0.0]])
    _, campo = md.campo(0.0, *md.separa(y0))
    assert np.abs(campo).max() / md.n_gas * 1e15 < md.tab.e_n[0]
    mob, dif, _, _, _, _ = md.coeficientes(np.zeros(md.m))
    nu = Q * 1e6 * n0 * (mob[md.i_e, 0] + mob[i_ar, 0]) / EPS0
    eps = (dif[md.i_e, 0] + dif[i_ar, 0]) * (np.pi / md.L) ** 2 / nu
    centro = slice(md.m // 4, 3 * md.m // 4)

    def carga(y):
        nn, _, _ = md.separa(y)
        return np.sum((md.carga @ nn)[centro])

    sol = solve_ivp(md.rhs, (0.0, 2.0 / nu), y0, method="BDF", rtol=1e-8, atol=1e-6)
    assert sol.success
    esperado = (eps + np.exp(-2.0 * (1.0 + eps))) / (1.0 + eps)
    np.testing.assert_allclose(carga(sol.y[:, -1]) / carga(y0), esperado, rtol=2e-4)


def test_fluxo_de_parede_freado(pecas):
    # uniform electrons with Te = 1e5 K and field E = +1e6 V/m (drift towards -x):
    # the left wall receives n*(v_term/2 + mu*E); the right one, from which the drift
    # moves away, only n*f*v_term/2 with f = exp(-e*E*(h/2)/(k*Te))
    md = Modelo(*pecas, n_celulas=40, agrupamento=1.5)
    n0, mu, te, e = 1.0e10, 0.03, 1.0e5, 1.0e6
    n = np.zeros((md.n_esp, md.m))
    n[md.i_e] = n0
    campo = np.full(md.m + 1, e)
    mob = np.zeros((md.n_esp, md.m))
    mob[md.i_e] = mu
    dif = np.zeros((md.n_esp, md.m))
    temp = np.full((md.n_esp, md.m), 300.0)
    temp[md.i_e] = te
    _, out, _ = md.fluxos(n, campo, mob, dif, temp)
    vt = np.sqrt(8.0 * 1.380649e-23 * te / (np.pi * 9.1093837015e-31))
    f = np.exp(-Q * e * 0.5 * md.h[-1] / (1.380649e-23 * te))
    np.testing.assert_allclose(out[md.i_e, 0], n0 * (0.5 * vt + mu * e), rtol=1e-12)
    np.testing.assert_allclose(out[md.i_e, 1], n0 * f * 0.5 * vt, rtol=1e-12)


def test_mobilidades_ionicas_medidas(pecas):
    # Ar+ in Ar: Frost formula with K0 = 1.61 cm2/(V s) and heating field of
    # 210 Td (Golyatina and Maiorov, Plasma Phys. Rep. 43, 75,
    # 2017; Ellis et al. 1976 measure 1.53); Ar2+: K0 = 1.95 cm2/(V s) (Phys. Rev.
    # 160, 130, 1967), constante. Mobilidade = K0*N0/N.
    md = Modelo(*pecas, n_celulas=4)
    n0 = 2.6867811e19
    e_n = np.array([0.0, 210.0, 2100.0, 50.0])
    campo = e_n * 1e-21 * md.n_gas * 1e6
    i_ar, i_ar2 = md.nomes.index("AR^+"), md.nomes.index("AR2^+")
    mu_ar = md.mobilidade_ion(i_ar, campo)
    esperado = 1.61e-4 * n0 / md.n_gas / np.sqrt(1.0 + e_n / 210.0)
    np.testing.assert_allclose(mu_ar, esperado, rtol=1e-12)
    np.testing.assert_allclose(md.mobilidade_ion(i_ar2, campo), 1.95e-4 * n0 / md.n_gas, rtol=1e-12)


def test_ordem_do_scharfetter_gummel_com_coeficientes_variaveis(pecas):
    # manufactured solution: smooth n(x), D(x) and mu(x), uniform field, clustered
    # mesh; the divergence of the discrete flux in the inner cells (away from the
    # walls, whose flux is the boundary condition) converges to the exact one with
    # order near 2 when the cells are doubled
    # field of 1e6 V/m: the drift (~200 m/s) dominates the diffusion (~60 m/s), so that
    # a first order error in the face mobility shows up; the mesh Peclet
    # number stays below 0.1
    L = 8.0e-4
    k = 2 * np.pi / L
    e = 1.0e6

    def exato(x):
        n = 1e12 * (1.2 + np.sin(k * x))
        dn = 1e12 * k * np.cos(k * x)
        d2n = -1e12 * k ** 2 * np.sin(k * x)
        D = 0.05 * (1.0 + 0.5 * np.cos(k * x))
        dD = -0.05 * 0.5 * k * np.sin(k * x)
        mu = 2.0e-4 * (1.0 + 0.3 * np.sin(k * x))
        dmu = 2.0e-4 * 0.3 * k * np.cos(k * x)
        # ion (charge +1): Gamma = -D n' + mu E n
        dG = -(dD * dn + D * d2n) + e * (dmu * n + mu * dn)
        return n, D, mu, dG

    erros = []
    for m in (40, 80, 160):
        md = Modelo(*pecas, n_celulas=m, agrupamento=2.0)
        i_ar = md.nomes.index("AR^+")
        n_c, D_c, mu_c, dG = exato(md.xc)
        n = np.zeros((md.n_esp, md.m))
        n[i_ar] = n_c
        mob = np.zeros((md.n_esp, md.m))
        dif = np.zeros((md.n_esp, md.m))
        mob[i_ar], dif[i_ar] = mu_c, D_c
        temp = np.full((md.n_esp, md.m), 300.0)
        G, _, _ = md.fluxos(n, np.full(md.m + 1, e), mob, dif, temp)
        div = np.diff(G[i_ar]) / md.h
        interno = slice(m // 8, m - m // 8)
        erros.append(np.max(np.abs(div[interno] - dG[interno])) / np.max(np.abs(dG)))
    ordens = np.log2(np.array(erros[:-1]) / np.array(erros[1:]))
    assert np.all(np.abs(ordens - 2.0) < 0.1), (erros, ordens)


def test_sem_freio_a_parede_recebe_o_fluxo_termico_inteiro(pecas):
    md = Modelo(*pecas, n_celulas=40, agrupamento=1.5, freio=False)
    n0, mu, te, e = 1.0e10, 0.03, 1.0e5, 1.0e6
    n = np.zeros((md.n_esp, md.m))
    n[md.i_e] = n0
    mob = np.zeros((md.n_esp, md.m))
    mob[md.i_e] = mu
    temp = np.full((md.n_esp, md.m), 300.0)
    temp[md.i_e] = te
    _, out, _ = md.fluxos(n, np.full(md.m + 1, e), mob, np.zeros((md.n_esp, md.m)), temp)
    vt = np.sqrt(8.0 * 1.380649e-23 * te / (np.pi * 9.1093837015e-31))
    np.testing.assert_allclose(out[md.i_e, 1], n0 * 0.5 * vt, rtol=1e-12)


def test_aviso_de_grau_de_ionizacao(pecas):
    # as in the Fortran code: with fixed gas, it warns when n_e/N exceeds 1e-3
    md = Modelo(*pecas, n_celulas=10)
    y = md.estado_inicial(1.0)
    assert md.celula_muito_ionizada(y) is None
    n = y[:-2].reshape(md.n_var, md.m)
    n[md.i_e, 4] = 2e-3 * md.n_gas
    assert md.celula_muito_ionizada(y) == 4


def test_frost_sempre_com_as_particulas_pesadas_da_celula(pecas):
    # as ions.f90:59-60, even with fixed gas and without heating:
    # mu = K0 (N0/N_heavy)/sqrt(1 + (E/N_heavy)/210 Td)
    md = Modelo(*pecas, n_celulas=4)
    n = np.zeros((md.n_esp, md.m))
    n[md.nomes.index("AR*")] = 1.0e17
    campo = 3.0e6
    mob, *_ = md.coeficientes(np.full(md.m, campo), n)
    n_pes = md.n_gas + 1.0e17
    e_n = campo / n_pes * 1e15
    esperado = 1.61 * 2.6867811e19 / n_pes * 1e-4 / np.sqrt(1.0 + e_n / 210.0)
    assert mob[md.nomes.index("AR^+"), 0] == pytest.approx(esperado, rel=1e-12)


def test_densidade_do_gas_coerente_com_a_temperatura(pecas):
    # at 1 atm: N = p/(k_B T); giving the temperature changes the default density
    md = Modelo(*pecas, n_celulas=4, t_gas=600.0)
    assert md.n_gas == pytest.approx(101325.0 / (1.380649e-23 * 600.0) * 1e-6, rel=1e-15)
    # and the diffusion of Ar* (DK of the Fortran code, with the k_B of 1.38e-23 of termico.f90)
    # sai a 1 atm: (17e-6/p)(T/273)^1,9
    p = md.n_gas * 1e6 * 1.38e-23 * 600.0 / 101325.0
    assert md.difusao[md.nomes.index("AR*")] == pytest.approx(17e-6 / p * (600.0 / 273.0) ** 1.9, rel=1e-14)


def test_frequencia_de_colisao_com_o_n_total(pecas):
    # como quimica.f90:97: (nu/N) da tabela vezes a soma de todas as densidades
    # of the cell (gas, Ar*, ions and electrons)
    md = Modelo(*pecas, n_celulas=4)
    n = np.zeros((md.n_esp, md.m))
    n[md.nomes.index("AR*")] = 0.1 * md.n_gas
    n[md.i_e] = 1e12
    e_n = 50.0
    campo = e_n * 1e-21 * md.n_gas * 1e6
    *_, extra = md.coeficientes(np.full(md.m, campo), n)
    nu_n = md.tab(e_n)[0, 3]
    np.testing.assert_allclose(md.frequencia_de_colisao(n, extra), nu_n * (1.1 * md.n_gas + 1e12), rtol=1e-12)


def test_mobilidades_ionicas_pelo_construtor(pecas):
    md = Modelo(*pecas, n_celulas=4, k0_ar=1.5, en_aquecimento_ar=200.0, k0_ar2=1.9)
    campo = np.full(md.m, 3.0e6)
    e_n = 3.0e6 / md.n_gas * 1e15
    escala = 2.6867811e19 / md.n_gas * 1e-4
    np.testing.assert_allclose(md.mobilidade_ion(md.nomes.index("AR^+"), campo),
                               1.5 * escala / np.sqrt(1.0 + e_n / 200.0), rtol=1e-12)
    np.testing.assert_allclose(md.mobilidade_ion(md.nomes.index("AR2^+"), campo), 1.9 * escala, rtol=1e-12)


def test_aviso_de_temperatura_do_gas(pecas):
    # with heating: first cell above 1000 K (where the 300 K table and the
    # extrapolated kappa stop being valid); None if there is none
    md = Modelo(*pecas, n_celulas=5, aquecimento=True)
    y = md.estado_inicial(1.0)
    assert md.celula_quente(y) is None
    y[md.i_t * md.m + 3] = 1500.0
    assert md.celula_quente(y) == 3
