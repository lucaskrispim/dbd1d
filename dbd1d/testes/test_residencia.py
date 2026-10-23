# SPDX-License-Identifier: GPL-3.0-or-later
"""Gas flow through the reactor (residence time tau, --residence-time): everything carried
by the gas leaves at the rate n/tau and the gas that enters is clean background at the
inlet temperature, so the right-hand side gains -n/tau for every species in the state,
-n_eps/tau for the electron energy and (T_in - T)/tau for the gas temperature; the surface
charges, bound to the barriers, do not change."""
import numpy as np
import pytest

from conftest import RAIZ, quimica_do
from modelo import Modelo
from tabela import Tabela, TabelaEnergia


@pytest.fixture(scope="module")
def pecas():
    return (quimica_do(RAIZ), Tabela(RAIZ / "tabela_bolsig.dat"), TabelaEnergia(RAIZ / "tabela_energia.dat"))


def _estado(md, r):
    y = md.estado_inicial(n_e0=1e10, neutro=True)
    z = y[:-2].reshape(md.n_var, md.m)
    z[: md.n_esp] *= r.uniform(0.5, 2.0, (md.n_esp, md.m))
    if md.energia:
        z[md.n_esp] = z[md.i_e] * r.uniform(2.0, 6.0, md.m)
    if md.aquecimento:
        z[md.i_t] = r.uniform(310.0, 420.0, md.m)
    y[-2:] = [1e-6, -2e-6]
    return y


@pytest.mark.parametrize("energia, aquecimento", [(False, False), (True, False), (True, True)])
def test_residencia_retira_o_que_o_gas_leva(pecas, energia, aquecimento):
    q, t, te = pecas
    tau, t_in = 0.05, 300.0
    kw = dict(n_celulas=8, agrupamento=1.5, espessura=1e-3, t_gas=t_in, t_parede=t_in, aquecimento=aquecimento,
              tabela_energia=te if energia else None)
    sem = Modelo(q, t, **kw)
    com = Modelo(q, t, residencia=tau, **kw)
    y = _estado(sem, np.random.default_rng(4))
    base = sem.rhs(1e-9, y)
    d = com.rhs(1e-9, y) - base
    # the difference of two right-hand sides keeps their round-off: tolerance on the scale of each row
    escala = np.abs(base[:-2].reshape(sem.n_var, sem.m)).max(axis=1, keepdims=True) * 1e-12
    z = y[:-2].reshape(sem.n_var, sem.m)
    dz = d[:-2].reshape(sem.n_var, sem.m)
    assert np.all(np.abs(dz[: sem.n_esp] + z[: sem.n_esp] / tau) <= escala[: sem.n_esp] + 1e-9 * np.abs(z[: sem.n_esp] / tau))
    if energia:
        np.testing.assert_allclose(dz[sem.n_esp], -z[sem.n_esp] / tau, rtol=1e-9, atol=escala[sem.n_esp, 0])
    if aquecimento:
        np.testing.assert_allclose(dz[sem.i_t], (t_in - z[sem.i_t]) / tau, rtol=1e-9, atol=escala[sem.i_t, 0])
    assert d[-2] == 0.0 and d[-1] == 0.0


def test_sem_residencia_nada_muda(pecas):
    q, t, _ = pecas
    md = Modelo(q, t, n_celulas=8, espessura=1e-3)
    y = _estado(md, np.random.default_rng(5))
    np.testing.assert_array_equal(Modelo(q, t, n_celulas=8, espessura=1e-3, residencia=None).rhs(1e-9, y),
                                  md.rhs(1e-9, y))
