# SPDX-License-Identifier: GPL-3.0-or-later
"""Energy table with the gas temperature (gap G4): the BOLSIG+ electron
coefficients per mean energy and per T_g, from tabela_energia_tg.dat. The Fortran
code passes the local T_g to BOLSIG+; the prototype used the table at 300 K, and
with hot gas the elastic loss (source of the heating) and the floor of the
electron mean energy ((3/2) k T_g) came out wrong at weak fields."""
import numpy as np
import pytest

from conftest import RAIZ, quimica_do
from tabela import E_PEL, TabelaEnergia, TabelaEnergiaTg

KB_EV = 1.380649e-23 / 1.602176634e-19


@pytest.fixture(scope="module")
def tab():
    return TabelaEnergiaTg(RAIZ / "tabela_energia_tg.dat")


@pytest.fixture(scope="module")
def tab300():
    return TabelaEnergia(RAIZ / "tabela_energia.dat")


def test_nos_de_temperatura(tab):
    assert tab.t_gas[0] == 300.0 and tab.t_gas[-1] >= 1500.0
    assert np.all(np.diff(tab.t_gas) > 0.0)


def test_bloco_de_300_k_e_a_tabela_atual(tab, tab300):
    # same BOLSIG+ scan: at 300 K the new table reproduces the current one bit for bit
    eps = np.geomspace(0.03, 60.0, 400)
    assert np.array_equal(tab(eps, np.full(eps.size, 300.0)), tab300(eps))
    assert np.array_equal(tab.limites(np.array([300.0]))[0], [tab300.eps[0]])


@pytest.mark.parametrize("t_gas", [600.0, 1000.0, 1500.0])
def test_piso_da_energia_media_e_a_do_gas(tab, t_gas):
    # with a field of 1e-4 Td the electrons stay at the gas temperature
    eps_min = tab.limites(np.array([t_gas]))[0][0]
    assert eps_min == pytest.approx(1.5 * KB_EV * t_gas, rel=1e-2)


def test_perda_elastica_depende_da_temperatura_do_gas(tab):
    # at the same mean energy, hotter gas gives energy back to the electrons:
    # the elastic loss falls, strongly when eps approaches (3/2) k T_g
    eps = np.array([0.1])
    p300 = tab(eps, np.array([300.0]))[0, E_PEL]
    p600 = tab(eps, np.array([600.0]))[0, E_PEL]
    assert 0.0 < p600 < 0.5 * p300


@pytest.mark.parametrize("t_gas", [300.0, 425.0, 650.0, 1125.0, 1375.0])
def test_interpolacao_contra_o_bolsig_fora_dos_nos(tab, t_gas):
    # check blocks of the generator off the nodes (at 300 K, with E/N half a
    # step from the nodes: only the interpolation in mean energy, gap V61), over the whole
    # range above the floor, including near it, where the elastic loss goes to
    # zero: mobility and collision frequency to 1%, elastic loss to 2%, k0 and
    # k1 to 1.5%, k2 to 1%; k3 (Ar* -> Ar+) to 1% outside 0.45 to 0.7 Td (1.7 to
    # 2.0 eV), where the BOLSIG+ mean energy grows irregularly between
    # nodes, and to 10% there
    from tabela import E_K0, E_MUN, E_NUN
    conf = TabelaEnergiaTg(RAIZ / "tabela_energia_tg_conferencia.dat")
    bloco = conf.blocos[list(conf.t_gas).index(t_gas)]
    sel = bloco.eps < 50.0
    eps, ref, e_n = bloco.eps[sel], bloco.valores[sel], bloco.e_n[sel]
    val = tab(eps, np.full(eps.size, t_gas))
    for col, tol in ((E_MUN, 1e-2), (E_NUN, 1e-2), (E_PEL, 2e-2)):
        np.testing.assert_allclose(val[:, col], ref[:, col], rtol=tol, err_msg=str(col))
    for j, tol in zip(range(E_K0, E_K0 + 3), (1.5e-2, 1.5e-2, 1e-2)):
        g = ref[:, j] > 1e-6 * ref[:, j].max()
        np.testing.assert_allclose(val[g, j], ref[g, j], rtol=tol, err_msg=str(j))
    j = E_K0 + 3
    g = ref[:, j] > 1e-6 * ref[:, j].max()
    irregular = (e_n > 0.45) & (e_n < 0.7)
    np.testing.assert_allclose(val[g & ~irregular, j], ref[g & ~irregular, j], rtol=1e-2)
    np.testing.assert_allclose(val[g & irregular, j], ref[g & irregular, j], rtol=0.1)


@pytest.fixture(scope="module")
def pecas():
    from quimica_zdp import Quimica
    from tabela import Tabela
    return quimica_do(RAIZ), Tabela(RAIZ / "tabela_bolsig.dat")


def _estado(md, eps, t_gas):
    """State with electrons and Ar+ at 1e10 cm-3, mean energy eps and gas at t_gas."""
    y = md.estado_inicial(n_e0=1e10, neutro=True)
    n = y[:-2].reshape(md.n_var, md.m)
    n[md.n_esp] = 1e10 * eps
    n[md.i_t] = t_gas
    return y


@pytest.mark.parametrize("t_gas", [300.0, 320.0])
def test_modelo_igual_ao_de_300_k_com_o_gas_a_300_k(pecas, tab, tab300, t_gas):
    # with the gas at 300 K the new table gives the same right-hand side as the 300 K one,
    # bit for bit; at 320 K, a different one
    from modelo import Modelo
    md_novo = Modelo(*pecas, n_celulas=6, v_pico=3e3, espessura=1.75e-3, aquecimento=True, tabela_energia=tab)
    md_velho = Modelo(*pecas, n_celulas=6, v_pico=3e3, espessura=1.75e-3, aquecimento=True, tabela_energia=tab300)
    y = _estado(md_novo, 0.06, t_gas)
    iguais = np.array_equal(md_novo.rhs(2.5e-8, y), md_velho.rhs(2.5e-8, y))
    assert iguais == (t_gas == 300.0)


def test_modelo_usa_a_temperatura_local_do_gas(pecas, tab):
    # the elastic loss and the floor of the mean energy follow the T_g of each cell
    from modelo import Modelo
    md = Modelo(*pecas, n_celulas=4, aquecimento=True, tabela_energia=tab)
    y = _estado(md, 0.3, 300.0)
    n = md.separa(y)[0]
    tg = np.array([300.0, 600.0, 1000.0, 1500.0])
    _, _, _, _, _, extra = md.coeficientes(np.zeros(md.m), n, md.energia_do_estado(y), t_gas=tg)
    np.testing.assert_allclose(extra["pel"], tab(np.full(4, 0.3), tg)[:, E_PEL], rtol=1e-9)
    # electrons below the 600 K floor rise to the local floor, ~(3/2) k T_g
    eps = md.energia_media(n, n[md.i_e] * 0.02, np.zeros(md.m), tg)
    np.testing.assert_allclose(eps, tab.limites(tg)[0], rtol=1e-12)
    assert eps[1] == pytest.approx(1.5 * KB_EV * 600.0, rel=1e-2)


def test_coeficientes_prendem_a_energia_no_piso_local(pecas, tab):
    # through the right-hand side: cold electrons rise to the floor of the local T_g
    from modelo import Modelo
    md = Modelo(*pecas, n_celulas=4, aquecimento=True, tabela_energia=tab)
    y = _estado(md, 0.3, 300.0)
    n = md.separa(y)[0]
    tg = np.array([300.0, 600.0, 1000.0, 1500.0])
    _, _, _, _, _, extra = md.coeficientes(np.zeros(md.m), n, n[md.i_e] * 0.02, t_gas=tg)
    np.testing.assert_allclose(extra["eps"], tab.limites(tg)[0], rtol=1e-12)


def test_abaixo_do_piso_os_eletrons_ganham_energia_do_gas(pecas, tab):
    # electrons colder than the gas (gas heating fast, or electrons that
    # reach a hotter cell): the elastic loss becomes negative, with the
    # relaxation (2m/M)(nu/N)(eps - floor) beyond the value at the floor
    from modelo import ME, Modelo
    from tabela import E_NUN, E_PINEL
    md = Modelo(*pecas, n_celulas=4, aquecimento=True, tabela_energia=tab)
    y = _estado(md, 0.3, 300.0)
    n = md.separa(y)[0]
    tg = np.full(md.m, 900.0)
    piso = tab.limites(tg)[0]
    eps_bruta = np.array([0.5, 0.8, 1.0, 2.0]) * piso
    _, _, _, _, _, extra = md.coeficientes(np.zeros(md.m), n, n[md.i_e] * eps_bruta, t_gas=tg)
    v = tab(piso, tg)
    m_ar = md.transporte["AR"]["massa_u"] * 1.66053906660e-27
    esperado = v[:, E_PEL] + 2.0 * ME / m_ar * v[:, E_NUN] * (eps_bruta - piso)
    np.testing.assert_allclose(extra["pel"][:2], esperado[:2], rtol=1e-6)
    assert np.all(extra["pel"][:2] < 0.0)
    np.testing.assert_allclose(extra["perda"][:2], extra["pel"][:2] + v[:2, E_PINEL], rtol=1e-9)
    assert extra["pel"][2] == pytest.approx(v[2, E_PEL], rel=1e-6)     # at the floor, the value of the table
    assert extra["pel"][3] > 0.0


def test_jacobiano_estruturado_com_o_gas_quente(tab):
    # the structured Jacobian captures the dependence of the coefficients on T_g
    from jacobiano import ModeloRapido
    from quimica_zdp import Quimica
    from tabela import Tabela
    from test_jacobiano import jacobiano_por_diferencas_centrais
    md = ModeloRapido(quimica_do(RAIZ), Tabela(RAIZ / "tabela_bolsig.dat"), n_celulas=12,
                      agrupamento=1.5, v_pico=5e3, espessura=1.75e-3, aquecimento=True, tabela_energia=tab)
    g = np.random.default_rng(5)
    n = 10.0 ** g.uniform(9, 12, size=(md.n_esp, md.m))
    n_eps = n[md.i_e] * g.uniform(0.1, 3.0, md.m)
    temp = g.uniform(400.0, 1200.0, md.m)
    y = np.concatenate([n.ravel(), n_eps, temp, [2e-7, -3e-7]])
    J_ref = jacobiano_por_diferencas_centrais(md, 2.2e-8, y)
    J = md.jacobiano(2.2e-8, y)
    peso = np.abs(y)
    escala = np.abs(J_ref * peso).max(axis=1, keepdims=True) + 1e-300
    assert np.max(np.abs(J - J_ref) * peso / escala) < 1e-4


# ------------------------------------------- local field with T_g (G4 in local field)
@pytest.fixture(scope="module")
def tab_lfa():
    from tabela import TabelaTg
    return TabelaTg(RAIZ / "tabela_energia_tg.dat")


def test_campo_local_a_300_k_e_a_tabela_atual(tab_lfa):
    # at 300 K, the columns of tabela_bolsig.dat (Te back from the mean energy: 2e-16)
    from tabela import TE, Tabela
    velha = Tabela(RAIZ / "tabela_bolsig.dat")
    e_n = np.geomspace(1e-3, 3000.0, 500)
    v, ref = tab_lfa(e_n, np.full(e_n.size, 300.0)), velha(e_n)
    np.testing.assert_allclose(v[:, TE], ref[:, TE], rtol=1e-14)
    assert np.array_equal(v[:, TE + 1:], ref[:, TE + 1:])


@pytest.mark.parametrize("t_gas", [300.0, 425.0, 650.0, 1125.0, 1375.0])
def test_campo_local_contra_o_bolsig_fora_dos_nos(tab_lfa, t_gas):
    # same check blocks, per E/N: mobility and collision frequency
    # to 1%; diffusion and elastic loss to 2% (up to 1.5% and 0.8% measured, below
    # 0.1 Td, with thermal electrons, in the range from 1000 to 1250 K); rates to
    # 1.5%; k3 between 0.4 and 0.7 Td to 5% (up to 3.1% measured at 1125 and 1375 K and
    # 0.13% at 300 K: the BOLSIG+ irregularity in that range moves
    # between T_g nodes, and the linear interpolation in T_g does not follow it)
    from tabela import DN, K0, MUN, NUN, PEL
    conf = TabelaEnergiaTg(RAIZ / "tabela_energia_tg_conferencia.dat")
    b = conf.blocos[list(conf.t_gas).index(t_gas)]
    sel = (b.e_n > 1e-3) & (b.e_n < 3000.0)
    e_n, ref = b.e_n[sel], b.valores[sel]
    v = tab_lfa(e_n, np.full(e_n.size, t_gas))
    from tabela import E_DN, E_K0, E_MUN, E_NUN
    for col, ecol, tol in ((MUN, E_MUN, 1e-2), (DN, E_DN, 2e-2), (NUN, E_NUN, 1e-2), (PEL, E_PEL, 2e-2)):
        np.testing.assert_allclose(v[:, col], ref[:, ecol], rtol=tol, err_msg=str(col))
    for j in range(4):
        r = ref[:, E_K0 + j]
        g = r > 1e-6 * r.max()
        if j == 3:
            janela = (e_n > 0.4) & (e_n < 0.7)
            np.testing.assert_allclose(v[g & janela, K0 + j], r[g & janela], rtol=5e-2)
            g &= ~janela
        np.testing.assert_allclose(v[g, K0 + j], r[g], rtol=1.5e-2, err_msg=str(j))


def test_modelo_de_campo_local_usa_a_temperatura_do_gas(pecas, tab_lfa):
    from modelo import Modelo
    from tabela import PEL
    md = Modelo(pecas[0], tab_lfa, n_celulas=4, aquecimento=True)
    n = md.separa(md.estado_inicial(n_e0=1e10, neutro=True))[0]
    tg = np.array([300.0, 600.0, 1000.0, 1500.0])
    campo = np.full(md.m, 2.5e6)
    _, _, _, _, e_n, extra = md.coeficientes(campo, n, None, t_gas=tg)
    np.testing.assert_allclose(extra["pel"], tab_lfa(e_n, tg)[:, PEL], rtol=1e-14)
    assert extra["pel"][1] != extra["pel"][0]


def test_eedf_de_campo_local_usa_a_temperatura_do_gas(pecas, tab_lfa, tab):
    # the EEDF recorded in local field comes from the local field energy at the T_g of the cell
    from modelo import Modelo
    from simula import energia_da_eedf
    md = Modelo(pecas[0], tab_lfa, n_celulas=4, aquecimento=True)
    tg = np.array([300.0, 600.0, 1125.0, 1500.0])
    e_n = np.array([0.01, 0.1, 1.0, 10.0])
    np.testing.assert_allclose(energia_da_eedf(md, e_n, {"tg": tg}), tab.energia_de_campo_local(e_n, tg), rtol=1e-14)


@pytest.mark.parametrize("t_gas", [425.0, 650.0, 1125.0, 1375.0])
def test_energia_de_campo_local_com_a_temperatura(tab, t_gas):
    # steady mean energy per E/N and T_g against the check blocks
    conf = TabelaEnergiaTg(RAIZ / "tabela_energia_tg_conferencia.dat")
    b = conf.blocos[list(conf.t_gas).index(t_gas)]
    sel = (b.e_n > 1e-3) & (b.e_n < 3000.0)
    np.testing.assert_allclose(tab.energia_de_campo_local(b.e_n[sel], np.full(sel.sum(), t_gas)), b.eps[sel],
                               rtol=5e-3)


def test_jacobiano_de_campo_local_com_o_gas_quente(tab_lfa):
    from jacobiano import ModeloRapido
    from quimica_zdp import Quimica
    from test_jacobiano import jacobiano_por_diferencas_centrais
    md = ModeloRapido(quimica_do(RAIZ), tab_lfa, n_celulas=12, agrupamento=1.5,
                      v_pico=5e3, espessura=1.75e-3, aquecimento=True)
    g = np.random.default_rng(7)
    n = 10.0 ** g.uniform(9, 12, size=(md.n_esp, md.m))
    temp = g.uniform(400.0, 1200.0, md.m)
    y = np.concatenate([n.ravel(), temp, [2e-7, -3e-7]])
    J_ref = jacobiano_por_diferencas_centrais(md, 2.2e-8, y)
    J = md.jacobiano(2.2e-8, y)
    peso = np.abs(y)
    escala = np.abs(J_ref * peso).max(axis=1, keepdims=True) + 1e-300
    assert np.max(np.abs(J - J_ref) * peso / escala) < 1e-4
