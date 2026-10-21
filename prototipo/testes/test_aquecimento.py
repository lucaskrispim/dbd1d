# SPDX-License-Identifier: GPL-3.0-or-later
"""Temperatura do gás a volume constante, como no código Fortran (termico.f90,
cinetica.f90 e ZDPlasKin com GAS_HEATING):
    (3/2) N k_B dT/dt = d/dx(kappa(T) dT/dx) + Q_el + Q_J,
com N as partículas pesadas, kappa(T) = (0,0307313 T + 10,006)e-3 W/(m K),
parede a 300 K a meia célula, Q_el a perda elástica dos elétrons (tabela) e
Q_J o Joule dos íons, e n_i mu_i E^2."""
import numpy as np
import pytest

from conftest import RAIZ, quimica_do
from jacobiano import ModeloRapido
from modelo import KB, Q
from quimica_zdp import Quimica
from tabela import PEL, Tabela
from test_jacobiano import jacobiano_por_diferencas_centrais


@pytest.fixture(scope="module")
def pecas():
    return quimica_do(RAIZ), Tabela(RAIZ / "tabela_bolsig.dat")


def modelo(pecas, **k):
    return ModeloRapido(*pecas, aquecimento=True, **k)


def estado(md, n, t_gas):
    """y a partir das densidades (espécies, células) e da temperatura (células)."""
    return np.concatenate([n.ravel(), t_gas, [0.0, 0.0]])


def kappa(t):
    return (0.0307313 * t + 10.006) * 1e-3


def test_conducao_de_calor_de_segunda_ordem(pecas):
    # sem partículas carregadas não há fonte; T = 300 + A sen(pi x/L) vale 300
    # nas paredes, e d/dx(kappa T') = a T'^2 + kappa T'' com kappa = a T + b
    erros = []
    for m in (40, 80, 160):
        md = modelo(pecas, n_celulas=m, v_pico=0.0)
        a_, L = 50.0, md.L
        t_gas = 300.0 + a_ * np.sin(np.pi * md.xc / L)
        y = estado(md, np.zeros((md.n_esp, m)), t_gas)
        dtdt = md.rhs(0.0, y)[md.i_t * m:(md.i_t + 1) * m]
        d1 = a_ * np.pi / L * np.cos(np.pi * md.xc / L)
        d2 = -a_ * (np.pi / L) ** 2 * np.sin(np.pi * md.xc / L)
        exato = (0.0307313e-3 * d1 ** 2 + kappa(t_gas) * d2) / (1.5 * KB * md.n_gas * 1e6)
        # células internas: na da parede o contorno fica a meia célula e o erro de
        # truncamento é de ordem 1 (a solução continua de 2ª ordem); ela tem
        # teste próprio abaixo
        erros.append(np.max(np.abs(dtdt - exato)[1:-1]) / np.max(np.abs(exato)))
    ordens = np.log2(np.array(erros[:-1]) / np.array(erros[1:]))
    assert erros[-1] < 1e-3, erros
    assert np.all(np.abs(ordens - 2.0) < 0.1), ordens


def test_parede_a_300_K_a_meia_celula(pecas):
    # T uniforme a 500 K: só as células das paredes perdem calor, pelo fluxo
    # kappa_f (T - 300)/(h/2), kappa_f a média harmônica de kappa(300) e kappa(500)
    m = 20
    md = modelo(pecas, n_celulas=m, v_pico=0.0)
    y = estado(md, np.zeros((md.n_esp, m)), np.full(m, 500.0))
    dtdt = md.rhs(0.0, y)[md.i_t * m:(md.i_t + 1) * m]
    k_f = 2.0 * kappa(300.0) * kappa(500.0) / (kappa(300.0) + kappa(500.0))
    h = md.L / m
    esperado = -k_f * 200.0 / (h / 2) / h / (1.5 * KB * md.n_gas * 1e6)
    assert dtdt[0] == pytest.approx(esperado, rel=1e-12)
    assert dtdt[-1] == pytest.approx(esperado, rel=1e-12)
    assert np.all(dtdt[1:-1] == 0.0)


def test_face_interna_com_media_harmonica(pecas):
    # uma célula a 400 K no meio de células a 300 K (paredes também a 300 K):
    # a vizinha recebe kappa_f*100/h/h, kappa_f a média harmônica, como kk no Fortran
    m = 20
    md = modelo(pecas, n_celulas=m, v_pico=0.0)
    t_gas = np.full(m, 300.0)
    t_gas[10] = 400.0
    dtdt = md.rhs(0.0, estado(md, np.zeros((md.n_esp, m)), t_gas))[md.i_t * m:(md.i_t + 1) * m]
    k_f = 2.0 * kappa(300.0) * kappa(400.0) / (kappa(300.0) + kappa(400.0))
    h = md.L / m
    assert dtdt[9] == pytest.approx(k_f * 100.0 / h / h / (1.5 * KB * md.n_gas * 1e6), rel=1e-12)


def test_condutividade_termica_nos_dois_ramos():
    from modelo import condutividade_termica
    assert condutividade_termica(np.array(1000.0)) == pytest.approx(kappa(1000.0), rel=1e-15)
    assert condutividade_termica(np.array(4000.0)) == pytest.approx(
        kappa(4000.0) * np.sqrt(4000.0 / 3000.0), rel=1e-15)


def test_fonte_de_calor_eletrons_e_ions(pecas):
    # plasma neutro e uniforme entre eletrodos metálicos: campo uniforme V/L,
    # T uniforme a 300 K (sem condução); no centro, dT/dt = (Q_el + Q_J)/(3/2 N k_B)
    m = 20
    md = modelo(pecas, n_celulas=m, v_pico=1.0e3, periodo=1e-7)
    t = 2.5e-8                                   # pico da tensão
    n = np.zeros((md.n_esp, m))
    ne = 1.0e10
    n[md.i_e] = ne
    n[md.nomes.index("AR^+")] = ne
    y = estado(md, n, np.full(m, 300.0))
    dtdt = md.rhs(t, y)[md.i_t * m:(md.i_t + 1) * m]
    campo = 1.0e3 / md.L
    e_n = campo / md.n_gas * 1e15
    q_el = Q * md.tab(e_n)[0, PEL] * md.n_gas * ne * 1e6        # W/m3
    k_ar = md.nomes.index("AR^+")
    q_j = Q * ne * 1e6 * md.mobilidade_ion(k_ar, np.array([campo]))[0] * campo ** 2
    esperado = (q_el + q_j) / (1.5 * KB * (md.n_gas + ne) * 1e6)
    assert dtdt[m // 2] == pytest.approx(esperado, rel=1e-6)


def test_taxa_com_a_temperatura_local(pecas):
    # só Ar+, uniforme, sem campo: o Ar2+ nasce por Ar+ + Ar + Ar -> Ar2+ + Ar,
    # com k10 = 2,25e-31 (T/300)^-0,4; a 600 K a taxa cai pelo fator 2^-0,4
    m = 10
    md = modelo(pecas, n_celulas=m, v_pico=0.0)
    n = np.zeros((md.n_esp, m))
    n[md.nomes.index("AR^+")] = 1.0e10
    i2 = md.nomes.index("AR2^+")
    taxa = {}
    for t_gas in (300.0, 600.0):
        dy = md.rhs(0.0, estado(md, n, np.full(m, t_gas)))
        taxa[t_gas] = dy[i2 * m + m // 2]
    assert taxa[300.0] == pytest.approx(2.25e-31 * md.n_gas ** 2 * 1.0e10, rel=1e-10)
    assert taxa[600.0] / taxa[300.0] == pytest.approx(2.0 ** -0.4, rel=1e-12)


def test_difusao_dos_ions_e_do_ar_metaestavel_com_a_temperatura_local(pecas):
    m = 10
    md = modelo(pecas, n_celulas=m, v_pico=0.0)
    n = np.zeros((md.n_esp, m))
    t_gas = np.full(m, 600.0)
    mob, dif, temp, *_ = md.coeficientes(np.zeros(m), n, None, t_gas)
    k_ar = md.nomes.index("AR^+")
    assert np.allclose(dif[k_ar], mob[k_ar] * KB * 600.0 / Q, rtol=1e-12)
    assert np.allclose(temp[k_ar], 600.0)
    # Ar*: (17e-6/p)(T/273)^1,9 com p = N k_B T (atm), N fixo
    k_m = md.nomes.index("AR*")
    p = md.n_gas * 1e6 * 1.38e-23 * 600.0 / 101325.0
    assert np.allclose(dif[k_m], 17e-6 / p * (600.0 / 273.0) ** 1.9, rtol=1e-12)


def test_sem_aquecimento_nada_muda(pecas):
    md0 = ModeloRapido(*pecas, n_celulas=12, v_pico=1e3, espessura=1e-3)
    md1 = ModeloRapido(*pecas, n_celulas=12, v_pico=1e3, espessura=1e-3, aquecimento=True)
    g = np.random.default_rng(3)
    n = 10.0 ** g.uniform(8, 11, size=(md0.n_esp, 12))
    y0 = np.concatenate([n.ravel(), [1e-7, -2e-7]])
    y1 = estado(md1, n, np.full(12, 300.0))
    y1[-2:] = [1e-7, -2e-7]
    f0, f1 = md0.rhs(1e-8, y0), md1.rhs(1e-8, y1)
    m = 12
    # densidades e cargas iguais; com aquecimento a mobilidade dos íons e a
    # difusão do Ar* usam N das partículas pesadas, que difere do N do gás em ~1e-8
    np.testing.assert_allclose(f1[:md0.n_esp * m], f0[:-2], rtol=1e-6, atol=1e-6 * np.abs(f0).max())
    np.testing.assert_allclose(f1[-2:], f0[-2:], rtol=1e-6)


def test_jacobiano_estruturado_com_aquecimento(pecas):
    q, tab = pecas
    md = ModeloRapido(q, tab, n_celulas=16, agrupamento=1.5, v_pico=5e3, espessura=1.75e-3,
                      aquecimento=True)
    g = np.random.default_rng(5)
    n = 10.0 ** g.uniform(9, 12, size=(md.n_esp, md.m))
    y = estado(md, n, 300.0 + 200.0 * g.uniform(0, 1, md.m))
    y[-2:] = [2e-7, -3e-7]
    t = 2.2e-8
    J_ref = jacobiano_por_diferencas_centrais(md, t, y)
    J = md.jacobiano(t, y)
    peso = np.abs(y)
    escala = np.abs(J_ref * peso).max(axis=1, keepdims=True) + 1e-300
    assert np.max(np.abs(J - J_ref) * peso / escala) < 1e-5
