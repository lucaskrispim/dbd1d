# SPDX-License-Identifier: GPL-3.0-or-later
"""Chemistry read from zdplaskin_m.F90 and table in Python against the Fortran code
(modules tabela and cinetica, the latter checked against ZDPlasKin in teste_cinetica)."""
import numpy as np
import pytest

from conftest import RAIZ, quimica_do
from quimica_zdp import Quimica
from tabela import Tabela, K0, TE

N_AR = 2.4474637681159418e19


@pytest.fixture(scope="module")
def quimica():
    return quimica_do(RAIZ)


@pytest.fixture(scope="module")
def tabela():
    return Tabela(RAIZ / "tabela_bolsig.dat")


def referencia(tipo=None):
    tab, fon, reac, aquec = {}, {}, {}, {}
    for l in (RAIZ / "prototipo/testes/referencia_cinetica.txt").read_text().splitlines():
        c = l.split()
        if c[0] == "tabela":
            tab[float(c[1])] = np.array([float(x) for x in c[2:]])
        elif c[0] == "fontes":
            fon[(float(c[1]), int(c[2]))] = np.array([float(x) for x in c[3:]])
        elif c[0] == "reacoes":
            reac[(float(c[1]), int(c[2]), float(c[3]))] = np.array([float(x) for x in c[4:]])
        elif c[0] == "aquecimento":
            aquec[(float(c[1]), int(c[2]), float(c[3]))] = float(c[4])
    if tipo == "reacoes":
        return reac
    if tipo == "aquecimento":
        return aquec
    return tab, fon


COMPOSICOES = {1: [1.0e10, N_AR, 1.0e8, 6.0e9, 4.0e9],
               2: [1.0e14, N_AR, 6.0e14, 3.0e13, 7.0e13],
               3: [3.0e11, N_AR, 2.0e9, 2.9e11, 1.0e10]}


def test_especies_e_cargas(quimica):
    assert quimica.especies == ["E", "AR", "AR*", "AR^+", "AR2^+"]
    assert list(quimica.carga) == [-1, 0, 0, 1, 1]


def test_conservacao_por_reacao(quimica):
    nucleos = np.array([0, 1, 1, 1, 2])
    assert np.all(quimica.carga @ quimica.estequiometria == 0)
    assert np.all(nucleos @ quimica.estequiometria == 0)


def test_tabela_igual_ao_fortran(tabela):
    tab, _ = referencia()
    for e_n, ref in tab.items():
        np.testing.assert_allclose(tabela(e_n)[0], ref, rtol=1e-12, atol=0)


def test_fontes_iguais_ao_fortran(quimica, tabela):
    _, fon = referencia()
    for (e_n, c), ref in fon.items():
        dens = np.array(COMPOSICOES[c], dtype=float)[:, None]
        v = tabela(e_n)
        k = quimica.coeficientes(v[:, K0:], np.array([300.0]), v[:, TE])
        s = quimica.fontes(dens, k)[:, 0]
        assert np.max(np.abs(s - ref)) <= 1e-10 * np.max(np.abs(ref)), (e_n, c, s, ref)


def test_tabela_em_e_n_ate_campo_quase_nulo():
    # Q3: the table goes down to 1e-4 Td, as the energy one; without field Te tends to the
    # gas temperature, as in ZDPlasKin directly (352 K at 1e-3 Td; before, the table started at
    # 0.01 Td and held Te at 2358 K, changing k8 by up to 1e4)
    from tabela import TE, Tabela
    tab = Tabela(RAIZ / "tabela_bolsig.dat")
    assert tab.e_n[0] <= 1.0e-4 * (1 + 1e-12)
    te = tab([1.0e-4, 1.0e-3])[:, TE]
    assert 300.0 <= te[0] < 310.0
    assert 330.0 < te[1] < 380.0


def test_cada_reacao_igual_ao_fortran_a_300_e_320_K(quimica, tabela):
    # VT6: reaction by reaction, with tolerance relative to the rate itself (the sum of the
    # sources, scaled by the largest term, hid reaction 6, up to 8e-19 of it)
    reac = referencia("reacoes")
    assert len(reac) == 30
    for (e_n, c, t_gas), ref in reac.items():
        dens = np.array(COMPOSICOES[c], dtype=float)[:, None]
        v = tabela(e_n)
        k = quimica.coeficientes(v[:, K0:], np.array([t_gas]), v[:, TE])
        r = quimica.taxas(dens, k)[:, 0]
        np.testing.assert_allclose(r, ref, rtol=1e-11, atol=0, err_msg=f"{e_n} Td, composicao {c}, {t_gas} K")


def test_aquecimento_pela_perda_elastica_igual_ao_fortran():
    # same composition (free gas) and same E/N, uniform temperature equal to that of the
    # wall (no conduction) and without ion Joule heating. Two model differences,
    # undone before comparing at 1e-12: the Fortran code uses gamma - 1 = 0.667 and the
    # prototype 2/3 (G1), and the elastic loss of the prototype is per N of neutrals
    # divided by the heat capacity with N of the heavy species, while the Fortran one lacks
    # the ratio N_neutrals/N_heavy (up to 4e-6 in composition 2)
    from modelo import Modelo
    aquec = referencia("aquecimento")
    q, tab = quimica_do(RAIZ), Tabela(RAIZ / "tabela_bolsig.dat")
    for (e_n, c, t_gas), ref in aquec.items():
        md = Modelo(q, tab, n_celulas=4, aquecimento=True, gas_solto=True, t_parede=t_gas)
        dens = np.array(COMPOSICOES[c], dtype=float)
        n = np.repeat(dens[:, None], md.m, axis=1)[[q.indice[nome] for nome in md.nomes]]
        n_neutro = dens[q.indice["AR"]] + dens[q.indice["AR*"]]
        campo = np.full(md.m, e_n * 1e-21 * n_neutro * 1e6)
        mob, _, _, _, _, extra = md.coeficientes(campo, n, None, np.full(md.m, t_gas))
        mob[md.ions] = 0.0                                   # without ion Joule heating
        dtdt = md.d_temperatura(n, np.full(md.m, t_gas), campo, mob, extra)
        como_fortran = dtdt * (0.667 * 1.5) * md.n_pesadas(n) / extra["n"]
        np.testing.assert_allclose(como_fortran, ref, rtol=1e-12, err_msg=f"{e_n} Td, composicao {c}, {t_gas} K")
