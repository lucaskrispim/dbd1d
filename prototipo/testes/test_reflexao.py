# SPDX-License-Identifier: GPL-3.0-or-later
"""Reflexão de elétrons na parede (Hagelaar et al., Phys. Rev. E 62, 1452,
2000): com coeficiente r, o fluxo de elétrons absorvido, térmico e deriva, e a
energia que ele leva ficam multiplicados por (1 - r)/(1 + r); íons, neutros e
a emissão secundária não mudam."""
import numpy as np
import pytest

from conftest import RAIZ, quimica_do
from modelo import Modelo
from tabela import Tabela, TabelaEnergia


@pytest.fixture(scope="module")
def pecas():
    return quimica_do(RAIZ), Tabela(RAIZ / "tabela_bolsig.dat")


def _estado(md):
    n = np.zeros((md.n_esp, md.m))
    for nome, valor in (("E", 1e9), ("AR^+", 2e9), ("AR2^+", 3e9), ("AR*", 4e9)):
        n[md.nomes.index(nome)] = valor
    return n


def test_fluxo_de_particulas_com_reflexao(pecas):
    # campo que leva os elétrons à parede esquerda: lá entram térmico e deriva
    campo_v = 2.0e4
    sem = Modelo(*pecas, n_celulas=10, gama_sec=0.07)
    com = Modelo(*pecas, n_celulas=10, gama_sec=0.07, reflexao_eletrons=0.7)
    n = _estado(sem)
    campo = np.full(sem.m + 1, campo_v)
    saidas = []
    for md in (sem, com):
        mob, dif, temp, *_ = md.coeficientes(np.full(md.m, campo_v), n)
        saidas.append(md.fluxos(n, campo, mob, dif, temp))
    (g0, out0, gi0), (g1, out1, gi1) = saidas
    ie = sem.i_e
    np.testing.assert_allclose(out1[ie], out0[ie] * 0.3 / 1.7, rtol=1e-13)
    outras = [k for k in range(sem.n_esp) if k != ie]
    np.testing.assert_array_equal(out1[outras], out0[outras])
    np.testing.assert_array_equal(gi1, gi0)
    assert g1[ie, 0] == pytest.approx(-out1[ie, 0] + gi1[0], rel=1e-13)


def test_fluxo_de_energia_com_reflexao():
    # mesmo estado, sem emissão secundária: a energia que sai fica multiplicada por (1 - r)/(1 + r)
    tabela = Tabela(RAIZ / "tabela_bolsig.dat")
    q = quimica_do(RAIZ)
    saidas = []
    for r in (0.0, 0.5):
        md = Modelo(q, tabela, n_celulas=6, tabela_energia=TabelaEnergia(RAIZ / "tabela_energia.dat"),
                    reflexao_eletrons=r)
        n = _estado(md)
        n_eps = n[md.i_e] * 4.0
        campo = np.full(md.m + 1, 3.0e4)
        mob, dif, temp, _, _, extra = md.coeficientes(md.campo_nas_celulas(campo), n, n_eps)
        saidas.append(md.fluxos_energia(n, n_eps, campo, mob, temp, extra, np.zeros(2)))
    np.testing.assert_allclose(saidas[1][[0, -1]], saidas[0][[0, -1]] / 3.0, rtol=1e-13)
    np.testing.assert_array_equal(saidas[1][1:-1], saidas[0][1:-1])


def test_reflexao_fora_da_faixa(pecas):
    for r in (-0.1, 1.0):
        with pytest.raises(ValueError):
            Modelo(*pecas, n_celulas=10, reflexao_eletrons=r)
