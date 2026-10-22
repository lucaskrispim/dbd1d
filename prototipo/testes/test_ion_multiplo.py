# SPDX-License-Identifier: GPL-3.0-or-later
"""Ions of charge greater than 1 (doubly ionized Ar2+, N^2+...): the mobility of
transporte.json is the measured one of the ion itself (v = K E), so the drift uses
only the sign of the charge; the Einstein diffusion is mu k T/(|z| e), the Joule
heating |z| e n mu E^2, and the charge z enters Poisson, the surface charge and
the Boltzmann factor at the wall. Test chemistry: argon with the charge of Ar2+ changed to +2."""
import copy

import numpy as np
import pytest

from conftest import RAIZ, quimica_do
from modelo import KB, Q, Modelo
from quimica_zdp import Quimica
from tabela import TabelaTg


@pytest.fixture(scope="module")
def modelos():
    q = quimica_do(RAIZ)
    dupla = copy.deepcopy(q)
    dupla.carga = dupla.carga.copy()
    dupla.carga[dupla.indice["AR2^+"]] = 2.0
    tab = TabelaTg(RAIZ / "tabela_energia_tg.dat")
    return Modelo(q, tab, n_celulas=10), Modelo(dupla, tab, n_celulas=10)


def _estado(md):
    n = np.full((md.n_esp, md.m), 1e10)
    campo = np.full(md.m + 1, 2e5)
    mob = np.full((md.n_esp, md.m), 1e-4)
    dif = np.full((md.n_esp, md.m), 1e-6)
    temp = np.full((md.n_esp, md.m), 300.0)
    return n, campo, mob, dif, temp


def test_deriva_com_a_mobilidade_do_proprio_ion(modelos):
    um, dois = modelos
    k = dois.nomes.index("AR2^+")
    g1 = um.fluxos(*_estado(um))[0][k, 1:-1]
    g2 = dois.fluxos(*_estado(dois))[0][k, 1:-1]
    np.testing.assert_allclose(g2, 1e-4 * 2e5 * 1e10, rtol=1e-12)
    np.testing.assert_allclose(g2, g1, rtol=1e-12)


def test_deriva_para_a_parede(modelos):
    # field +x: the drift takes the ion to the right wall with mu E, not 2 mu E
    _, dois = modelos
    k = dois.nomes.index("AR2^+")
    n, campo, mob, dif, temp = _estado(dois)
    out = dois.fluxos(n, campo, mob, dif, temp)[1]
    termico = 0.5 * np.sqrt(8.0 * KB * 300.0 / (np.pi * dois.massa[k]))
    assert out[k, 1] == pytest.approx(1e10 * (termico + 1e-4 * 2e5), rel=1e-12)


def test_difusao_de_einstein_com_a_carga(modelos):
    um, dois = modelos
    k = dois.nomes.index("AR2^+")
    n = dois.separa(dois.estado_inicial(n_e0=1e9, neutro=True))[0]
    campo = np.full(dois.m, 1e5)
    mob1, dif1, *_ = um.coeficientes(campo, n)
    mob2, dif2, *_ = dois.coeficientes(campo, n)
    np.testing.assert_array_equal(mob2[k], mob1[k])
    np.testing.assert_allclose(dif2[k], mob2[k] * KB * um.t_gas / (2.0 * Q), rtol=1e-14)


def test_joule_com_a_carga(modelos):
    _, dois = modelos
    k = dois.nomes.index("AR2^+")
    n = np.zeros((dois.n_esp, dois.m))
    n[k] = 1e10
    mob = np.full((dois.n_esp, dois.m), 1e-4)
    campo = np.full(dois.m, 1e5)
    np.testing.assert_allclose(dois.joule_ions(n, mob, campo), 2.0 * Q * 1e10 * 1e6 * 1e-4 * 1e10, rtol=1e-14)


def test_freio_da_parede_com_a_carga(modelos):
    # field +x: the positive ion moves away from the left wall, and the thermal flux reaching
    # it is reduced by exp(-|z| e E d/(k T)); on the right the drift does not depend on z
    um, dois = modelos
    k = dois.nomes.index("AR2^+")
    out1 = um.fluxos(*_estado(um))[1]
    out2 = dois.fluxos(*_estado(dois))[1]
    a = Q * 2e5 * dois.d[0] / (KB * 300.0)
    assert out2[k, 0] == pytest.approx(out1[k, 0] * np.exp(-a), rel=1e-10)
    assert out2[k, 1] == pytest.approx(out1[k, 1], rel=1e-12)
