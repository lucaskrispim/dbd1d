# SPDX-License-Identifier: GPL-3.0-or-later
"""Íons negativos (H- na amônia de Sun et al. 2022, O- e O2- no ar): mobilidade
e difusão como os positivos, deriva no sentido oposto, presos no volume pelo
campo das paredes e com aquecimento Joule. Química de teste: o argônio com a
carga do Ar2+ trocada para -1 (a mecânica do modelo não depende da reação)."""
import copy

import numpy as np
import pytest

from conftest import RAIZ, quimica_do
from modelo import KB, Q, Modelo
from quimica_zdp import Quimica
from tabela import TabelaTg


@pytest.fixture(scope="module")
def pecas():
    q = quimica_do(RAIZ)
    neg = copy.deepcopy(q)
    neg.carga = neg.carga.copy()
    neg.carga[neg.indice["AR2^+"]] = -1.0
    return q, neg, TabelaTg(RAIZ / "tabela_energia_tg.dat")


def _modelos(pecas, **kw):
    q, neg, tab = pecas
    return (Modelo(q, tab, n_celulas=10, **kw), Modelo(neg, tab, n_celulas=10, **kw))


def test_ion_negativo_e_aceito_com_a_mobilidade_do_arquivo(pecas):
    pos, neg = _modelos(pecas)
    k = neg.nomes.index("AR2^+")
    campo = np.full(neg.m, 3e6)
    assert neg.carga[k] == -1.0
    np.testing.assert_array_equal(neg.mobilidade_ion(k, campo), pos.mobilidade_ion(k, campo))


def test_deriva_no_sentido_oposto(pecas):
    # densidade uniforme e campo uniforme: fluxo interno -mu E n para o negativo
    pos, neg = _modelos(pecas)
    k = neg.nomes.index("AR2^+")
    n = np.full((neg.n_esp, neg.m), 1e10)
    campo = np.full(neg.m + 1, 2e5)
    mob = np.full((neg.n_esp, neg.m), 1e-4)
    dif = np.full((neg.n_esp, neg.m), 1e-6)
    temp = np.full((neg.n_esp, neg.m), 300.0)
    g_pos = pos.fluxos(n, campo, mob, dif, temp)[0][k, 1:-1]
    g_neg = neg.fluxos(n, campo, mob, dif, temp)[0][k, 1:-1]
    np.testing.assert_allclose(g_pos, 1e-4 * 2e5 * 1e10, rtol=1e-12)
    np.testing.assert_allclose(g_neg, -1e-4 * 2e5 * 1e10, rtol=1e-12)


def test_preso_no_volume_pelo_campo_e_sem_emissao_secundaria(pecas):
    # campo +x: o negativo vai para a parede esquerda (deriva entra inteira) e
    # na direita só o térmico freado por exp(-e E d/(k T)); não arranca elétrons
    _, neg = _modelos(pecas)
    k = neg.nomes.index("AR2^+")
    n = np.zeros((neg.n_esp, neg.m))
    n[k] = 1e10
    campo = np.full(neg.m + 1, 2e5)
    mob = np.full((neg.n_esp, neg.m), 1e-4)
    dif = np.full((neg.n_esp, neg.m), 1e-6)
    temp = np.full((neg.n_esp, neg.m), 300.0)
    _, out, gama = neg.fluxos(n, campo, mob, dif, temp)
    v_term = np.sqrt(8 * KB * 300.0 / (np.pi * neg.massa[k]))
    assert out[k, 0] == pytest.approx(1e10 * (0.5 * v_term + 1e-4 * 2e5), rel=1e-12)
    freio = np.exp(-Q * 2e5 * neg.d[-1] / (KB * 300.0))
    assert out[k, 1] == pytest.approx(1e10 * 0.5 * v_term * freio, rel=1e-12)
    assert np.all(gama == 0.0)


def test_joule_dos_ions_negativos(pecas):
    # o aquecimento Joule soma todos os íons, positivos e negativos
    _, neg = _modelos(pecas, aquecimento=True, v_pico=3e3, espessura=1.75e-3)
    y = neg.estado_inicial(n_e0=0.0)
    z = y[:-2].reshape(neg.n_var, neg.m)
    k = neg.nomes.index("AR2^+")
    z[k] = 1e12
    z[neg.nomes.index("AR^+")] = 1e12           # neutro: o campo é o aplicado
    n = neg.separa(y)[0]
    campo = neg.campo_nas_celulas(neg.campo(1e-8, n, 0.0, 0.0)[1])
    sem = copy.deepcopy(neg)
    sem.ions_negativos = []
    d_com = neg.rhs(1e-8, y)[:-2].reshape(neg.n_var, neg.m)[neg.i_t]
    d_sem = sem.rhs(1e-8, y)[:-2].reshape(neg.n_var, neg.m)[neg.i_t]
    q_neg = Q * 1e12 * 1e6 * neg.mobilidade_ion(k, campo, neg.n_pesadas(n)) * campo ** 2
    rho_cv = neg.cv_por_kb * KB * neg.n_pesadas(n) * 1e6
    np.testing.assert_allclose(d_com - d_sem, q_neg / rho_cv, rtol=1e-9)
