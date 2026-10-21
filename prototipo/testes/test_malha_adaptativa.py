# SPDX-License-Identifier: GPL-3.0-or-later
"""Testes da malha adaptativa: transferência conservativa entre malhas e geração
da malha por equidistribuição, e os operadores do modelo numa malha qualquer."""
import numpy as np
import pytest

from conftest import RAIZ, quimica_do
from malha_adaptativa import faces_por_equidistribuicao, monitor, transfere, transfere_energia
from modelo import Q, Modelo, faces_tanh
from quimica_zdp import Quimica
from tabela import Tabela, TabelaEnergia

L = 8.0e-4


def malha_aleatoria(m, semente):
    g = np.random.default_rng(semente)
    h = g.uniform(0.3, 1.7, m)
    return np.concatenate([[0.0], np.cumsum(h) / h.sum() * L])


# ---------------------------------------------------------------- transferência
def test_transferencia_conserva_o_total():
    velhas, novas = malha_aleatoria(80, 1), malha_aleatoria(80, 2)
    u = 10.0 ** np.random.default_rng(3).uniform(5, 13, (4, 80))
    v = transfere(u, velhas, novas)
    np.testing.assert_allclose((v * np.diff(novas)).sum(axis=1), (u * np.diff(velhas)).sum(axis=1), rtol=1e-13)


def test_transferencia_preserva_positividade():
    velhas, novas = malha_aleatoria(60, 4), malha_aleatoria(60, 5)
    u = np.zeros((1, 60))
    u[0, 20:25] = 1e14                    # degrau: a reconstrução não pode criar negativos
    v = transfere(u, velhas, novas)
    assert np.all(v >= 0.0)
    assert v.max() <= u.max() * (1 + 1e-12)


def test_transferencia_exata_para_perfil_linear():
    velhas, novas = malha_aleatoria(50, 6), malha_aleatoria(50, 7)
    xc = 0.5 * (velhas[1:] + velhas[:-1])
    u = (2.0e12 + 3.0e15 * xc)[None, :]
    v = transfere(u, velhas, novas)[0]
    exato = 2.0e12 + 3.0e15 * 0.5 * (novas[1:] + novas[:-1])
    # células cujo intervalo novo cai dentro do domínio das reconstruções
    # inteiras; nas duas de borda a inclinação é a de um lado só
    np.testing.assert_allclose(v[1:-1], exato[1:-1], rtol=1e-12)


def test_transferencia_identidade_na_mesma_malha():
    f = malha_aleatoria(40, 8)
    u = np.random.default_rng(9).uniform(1, 2, (3, 40))
    np.testing.assert_allclose(transfere(u, f, f), u, rtol=1e-14)


# --------------------------------------------------------------- equidistribuição
def test_monitor_constante_reproduz_a_malha_base():
    base = faces_tanh(100, L, 2.0)
    novas = faces_por_equidistribuicao(base, np.zeros(100), base)
    np.testing.assert_allclose(novas, base, atol=1e-12 * L)


def test_celulas_se_concentram_na_frente():
    base = faces_tanh(150, L, 2.0)
    xc = 0.5 * (base[1:] + base[:-1])
    # frente: n_e cai seis ordens em ~5 um em torno de x = 400 um
    ne = 1e13 * 0.5 * (1.0 - np.tanh((xc - 4.0e-4) / 2.5e-6)) + 1e7
    mon = monitor(base, [np.log(ne)])
    novas = faces_por_equidistribuicao(base, mon, base)
    h_novo = np.diff(novas)
    xn = 0.5 * (novas[1:] + novas[:-1])
    na_frente = np.argmin(np.abs(xn - 4.0e-4))
    h_base_frente = np.diff(base)[np.argmin(np.abs(xc - 4.0e-4))]
    assert h_novo[na_frente] < 0.3 * h_base_frente
    # malha válida: faces crescentes, extremos fixos, vizinhas com razão limitada
    assert novas[0] == 0.0 and novas[-1] == L
    assert np.all(h_novo > 0.0)
    razao = h_novo[1:] / h_novo[:-1]
    assert np.max(np.maximum(razao, 1.0 / razao)) < 1.5


# ----------------------------------------------- operadores numa malha qualquer
@pytest.fixture(scope="module")
def pecas():
    return (quimica_do(RAIZ), Tabela(RAIZ / "tabela_bolsig.dat"),
            TabelaEnergia(RAIZ / "tabela_energia.dat"))


def test_operadores_numa_malha_qualquer(pecas):
    q, t, te = pecas
    md = Modelo(q, t, n_celulas=60, tabela_energia=te, espessura=1.75e-3, v_pico=5e3)
    md.troca_malha(malha_aleatoria(60, 11))
    # Poisson: com dielétrico e sem carga, campo uniforme de capacitores em série
    _, campo = md.poisson(np.zeros(md.m), 0.0, 0.0, 1.0e3)
    np.testing.assert_allclose(campo, 1.0e3 / (L + 2.0 * 1.75e-3 / 4.8), rtol=1e-10)
    # carga do gás mais superfícies conservada pelo lado direito
    g = np.random.default_rng(12)
    n = 10.0 ** g.uniform(8, 13, size=(md.n_esp, md.m))
    n_eps = n[md.i_e] * g.uniform(1, 8, md.m)
    y = np.concatenate([n.ravel(), n_eps, [1e-6, -2e-6]])
    dy = md.rhs(2.5e-8, y)
    dn, ds_esq, ds_dir = md.separa(dy)
    d_gas = Q * 1e6 * np.sum((md.carga @ dn) * md.h)
    escala = Q * 1e6 * np.sum(np.abs(md.carga[:, None] * dn) * md.h[None, :])
    assert abs(d_gas + ds_esq + ds_dir) < 1e-12 * escala


def test_tamanho_minimo_de_celula():
    # readaptações sucessivas sobre uma frente estreitíssima (como na simulação,
    # cada uma sobre a malha da anterior) esmagariam as células; nenhuma fica
    # abaixo de h_min (sem o limite, a menor chega a 0,84 um)
    base = faces_tanh(150, L, 2.0)
    faces = base
    for _ in range(15):
        xc = 0.5 * (faces[1:] + faces[:-1])
        ne = 1e13 * 0.5 * (1.0 - np.tanh((xc - 4.0e-4) / 1.0e-9)) + 1e-3
        faces = faces_por_equidistribuicao(faces, monitor(faces, [np.log(ne)]), base, h_min=2.0e-6)
    assert np.min(np.diff(faces)) >= 2.0e-6 * (1 - 1e-6)


def test_energia_media_transferida_sem_picos():
    # frente íngreme de n_e com energia média variando: na malha nova a energia
    # média fica entre o menor e o maior valor da velha, e a energia total se
    # conserva até a diferença entre as duas transferências de n_e
    velhas, novas = malha_aleatoria(80, 13), malha_aleatoria(80, 14)
    xc = 0.5 * (velhas[1:] + velhas[:-1])
    ne = 1e13 * 0.5 * (1.0 - np.tanh((xc - 4.0e-4) / 5.0e-6)) + 1e3
    eps = 2.0 + 8.0 * (xc > 4.0e-4)
    ne_novo = transfere(ne[None, :], velhas, novas)[0]
    n_eps_novo = transfere_energia(ne, ne * eps, velhas, novas, ne_novo)
    eps_novo = n_eps_novo / ne_novo
    assert eps_novo.min() >= 2.0 - 1e-12 and eps_novo.max() <= 10.0 + 1e-12
    np.testing.assert_allclose(np.sum(n_eps_novo * np.diff(novas)), np.sum(ne * eps * np.diff(velhas)), rtol=2e-2)
