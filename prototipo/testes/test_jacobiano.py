# SPDX-License-Identifier: GPL-3.0-or-later
"""Testes do jacobiano estruturado e da formulação DAE (potencial no estado).

O jacobiano do lado direito tem a forma J = J_local + S*G: J_local é a derivada
a campo fixo (bloco-tridiagonal), S a derivada em relação ao campo nas faces e
G a resposta linear do Poisson à carga das células e das superfícies. Na forma
DAE o potencial entra no estado com 0 = A*phi - b(n, sigma, t), e eliminar phi
(complemento de Schur) tem de devolver o mesmo J.
"""
import numpy as np
import pytest

from conftest import RAIZ, quimica_do
from jacobiano import ModeloRapido, SistemaDAE
from quimica_zdp import Quimica
from tabela import Tabela, TabelaEnergia


@pytest.fixture(scope="module")
def caso():
    md = ModeloRapido(quimica_do(RAIZ), Tabela(RAIZ / "tabela_bolsig.dat"),
                      n_celulas=20, agrupamento=1.5, v_pico=5e3, espessura=1.75e-3,
                      tabela_energia=TabelaEnergia(RAIZ / "tabela_energia.dat"))
    g = np.random.default_rng(21)
    n = 10.0 ** g.uniform(9, 12, size=(md.n_esp, md.m))
    n_eps = n[md.i_e] * g.uniform(2.0, 8.0, md.m)
    y = np.concatenate([n.ravel(), n_eps, [2e-7, -3e-7]])
    return md, 2.2e-8, y


def jacobiano_por_diferencas_centrais(md, t, y):
    f0 = md.rhs(t, y)
    J = np.zeros((len(f0), len(y)))
    for j in range(len(y)):
        h = 1e-6 * max(abs(y[j]), 1e-9 if j >= len(y) - 2 else 1.0)
        yp, ym = y.copy(), y.copy()
        yp[j] += h
        ym[j] -= h
        J[:, j] = (md.rhs(t, yp) - md.rhs(t, ym)) / (2.0 * h)
    return J


def test_jacobiano_estruturado_igual_a_diferencas_centrais(caso):
    md, t, y = caso
    J_ref = jacobiano_por_diferencas_centrais(md, t, y)
    J = md.jacobiano(t, y)
    erro = np.linalg.norm(J - J_ref) / np.linalg.norm(J_ref)
    assert erro < 1e-4, erro
    # cada linha também, com as colunas pesadas pelo valor da variável (ver o
    # teste do Schur): sem o peso, a carga de superfície esconde as densidades
    peso = np.abs(y)
    escala = np.abs(J_ref * peso).max(axis=1, keepdims=True) + 1e-300
    assert np.max(np.abs(J - J_ref) * peso / escala) < 1e-5


def test_dae_consistente_com_o_modelo(caso):
    # no estado consistente (phi do Poisson), a parte diferencial do lado direito
    # é a do modelo e o resíduo algébrico é nulo
    md, t, y = caso
    dae = SistemaDAE(md)
    z = dae.estado(t, y)
    g = dae.rhs(t, z)
    f = md.rhs(t, y)
    np.testing.assert_allclose(g[:dae.N], f, rtol=1e-12, atol=1e-12 * np.abs(f).max())
    b = np.abs(md._rhs_poisson(1.0, 1e-7, 1e-7, md.tensao(t))).max()
    assert np.abs(g[dae.N:]).max() < 1e-9 * b


def test_schur_da_dae_devolve_o_jacobiano(caso):
    md, t, y = caso
    dae = SistemaDAE(md)
    Jz = dae.jacobiano(t, dae.estado(t, y)).toarray()
    N = dae.N
    Jyy, Jyp, Jpy, Jpp = Jz[:N, :N], Jz[:N, N:], Jz[N:, :N], Jz[N:, N:]
    schur = Jyy - Jyp @ np.linalg.solve(Jpp, Jpy)
    J = md.jacobiano(t, y)
    # cada coluna pesada pelo valor da variável (|J_ij y_j|): por unidade, a
    # derivada do campo em relação à densidade (cm-3) é uns 18 decimais menor que
    # em relação à carga de superfície (C/m2) e sumiria na escala da linha
    peso = np.abs(y)
    Jp, schur_p = J * peso, schur * peso
    escala = np.abs(Jp).max(axis=1, keepdims=True) + 1e-300
    assert np.max(np.abs(schur_p - Jp) / escala) < 1e-8
    # e o termo do campo, sozinho, também confere
    campo_p = Jp - Jyy * peso
    escala_campo = np.abs(campo_p).max(axis=1, keepdims=True) + 1e-300
    assert np.max(np.abs((schur_p - Jyy * peso) - campo_p) / escala_campo) < 1e-8
