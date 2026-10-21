# SPDX-License-Identifier: GPL-3.0-or-later
"""Testes do modelo acoplado (operadores e sistema), com respostas deduzidas à mão."""
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
    # com mobilidade, difusão e campo uniformes, n = exp(w x/D) tem fluxo
    # nulo em toda face interna: o esquema SG é exato para ele, em malha agrupada
    md = Modelo(*pecas, n_celulas=40, agrupamento=2.0)
    mu, dd, e = 0.03, 0.2, 2.0e4
    w = -mu * e                      # elétrons
    n = np.zeros((md.n_esp, md.m))
    n[md.i_e] = np.exp(w * md.xc / dd)
    campo = np.full(md.m + 1, e)
    mob = np.full((md.n_esp, md.m), mu)
    dif = np.full((md.n_esp, md.m), dd)
    temp = np.full((md.n_esp, md.m), 300.0)
    G, _, _ = md.fluxos(n, campo, mob, dif, temp)
    assert np.max(np.abs(G[md.i_e, 1:-1])) < 1e-12 * dd * np.max(n[md.i_e]) / np.min(md.d)


def test_poisson_linear_na_malha_agrupada(pecas):
    md = Modelo(*pecas, n_celulas=40, agrupamento=2.0)
    phi, campo = md.poisson(np.zeros(md.m), 0.0, 0.0, 1.0e3)
    x = np.concatenate([[0.0], md.xc, [md.L]])
    np.testing.assert_allclose(phi, 1.0e3 * (1.0 - x / md.L), atol=1e-9)
    np.testing.assert_allclose(campo, 1.0e3 / md.L, rtol=1e-10)


def test_poisson_com_dieletrico_e_capacitores_em_serie(pecas):
    # sem carga: D contínuo, E_gas = V/(L + 2*espessura/eps_d)
    md = Modelo(*pecas, n_celulas=40, agrupamento=2.0, espessura=1.75e-3)
    _, campo = md.poisson(np.zeros(md.m), 0.0, 0.0, 1.0e3)
    np.testing.assert_allclose(campo, 1.0e3 / (md.L + 2.0 * 1.75e-3 / 4.8), rtol=1e-10)


def test_carga_total_conservada(pecas):
    # transporte, paredes, emissão secundária e química: a taxa de variação da
    # carga do gás mais a das superfícies é zero
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
    # plasma uniforme e quase neutro com perturbação de carga sin(k x), k = pi/L,
    # eletrodos aterrados. Linearizando (mobilidades e difusões de campo nulo),
    # a carga relaxa com taxa nu (1 + eps), nu = e(mu_e n_e + mu_i n_i)/eps0 e
    # eps = (D_e + D_i) k^2/nu ~ (lambda_D k)^2, até o patamar de Boltzmann
    # eps/(1 + eps): depois de t = 2/nu, R = [eps + exp(-2(1 + eps))]/(1 + eps).
    # A perturbação de 1e-6 dá campo abaixo do primeiro nó da tabela, onde os
    # coeficientes são constantes (com 1e-4 o campo chega a 2e-3 Td, onde mu_e
    # varia 27%, região de Ramsauer, e a linearização falha)
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
    # elétrons uniformes com Te = 1e5 K e campo E = +1e6 V/m (deriva para -x):
    # a parede esquerda recebe n*(v_term/2 + mu*E); a direita, de onde a deriva
    # se afasta, só n*f*v_term/2 com f = exp(-e*E*(h/2)/(k*Te))
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
    # Ar+ em Ar: fórmula de Frost com K0 = 1,61 cm2/(V s) e campo de
    # aquecimento de 210 Td (Golyatina e Maiorov, Plasma Phys. Rep. 43, 75,
    # 2017; Ellis et al. 1976 medem 1,53); Ar2+: K0 = 1,95 cm2/(V s) (Phys. Rev.
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
    # solução manufaturada: n(x), D(x) e mu(x) suaves, campo uniforme, malha
    # agrupada; a divergência do fluxo discreto nas células internas (longe das
    # paredes, cujo fluxo é a condição de contorno) converge para a exata com
    # ordem perto de 2 ao dobrar as células
    # campo de 1e6 V/m: a deriva (~200 m/s) domina a difusão (~60 m/s), para que
    # um erro de primeira ordem na mobilidade das faces apareça; o número de
    # Péclet de malha fica abaixo de 0,1
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
        # íon (carga +1): Gamma = -D n' + mu E n
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
    # como no Fortran: com o gás preso, avisa quando n_e/N passa de 1e-3
    md = Modelo(*pecas, n_celulas=10)
    y = md.estado_inicial(1.0)
    assert md.celula_muito_ionizada(y) is None
    n = y[:-2].reshape(md.n_var, md.m)
    n[md.i_e, 4] = 2e-3 * md.n_gas
    assert md.celula_muito_ionizada(y) == 4


def test_frost_sempre_com_as_particulas_pesadas_da_celula(pecas):
    # como ions.f90:59-60, mesmo com o gás preso e sem aquecimento:
    # mu = K0 (N0/N_pes)/sqrt(1 + (E/N_pes)/210 Td)
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
    # a 1 atm: N = p/(k_B T); dar a temperatura muda a densidade padrão
    md = Modelo(*pecas, n_celulas=4, t_gas=600.0)
    assert md.n_gas == pytest.approx(101325.0 / (1.380649e-23 * 600.0) * 1e-6, rel=1e-15)
    # e a difusão do Ar* (DK do Fortran, com o k_B de 1,38e-23 de termico.f90)
    # sai a 1 atm: (17e-6/p)(T/273)^1,9
    p = md.n_gas * 1e6 * 1.38e-23 * 600.0 / 101325.0
    assert md.difusao[md.nomes.index("AR*")] == pytest.approx(17e-6 / p * (600.0 / 273.0) ** 1.9, rel=1e-14)


def test_frequencia_de_colisao_com_o_n_total(pecas):
    # como quimica.f90:97: (nu/N) da tabela vezes a soma de todas as densidades
    # da célula (gás, Ar*, íons e elétrons)
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
    # com aquecimento: primeira célula acima de 1000 K (onde a tabela a 300 K e o
    # kappa extrapolado deixam de valer); None se não há
    md = Modelo(*pecas, n_celulas=5, aquecimento=True)
    y = md.estado_inicial(1.0)
    assert md.celula_quente(y) is None
    y[md.i_t * md.m + 3] = 1500.0
    assert md.celula_quente(y) == 3
