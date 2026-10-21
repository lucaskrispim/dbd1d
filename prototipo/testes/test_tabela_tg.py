# SPDX-License-Identifier: GPL-3.0-or-later
"""Tabela de energia com a temperatura do gás (lacuna G4): os coeficientes dos
elétrons do BOLSIG+ por energia média e por T_g, de tabela_energia_tg.dat. O
Fortran passa a T_g local ao BOLSIG+; o protótipo usava a tabela a 300 K, e
com o gás quente a perda elástica (fonte do aquecimento) e o piso da energia
média dos elétrons ((3/2) k T_g) saíam errados nos campos fracos."""
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
    # mesma varredura do BOLSIG+: a 300 K a tabela nova reproduz a atual bit a bit
    eps = np.geomspace(0.03, 60.0, 400)
    assert np.array_equal(tab(eps, np.full(eps.size, 300.0)), tab300(eps))
    assert np.array_equal(tab.limites(np.array([300.0]))[0], [tab300.eps[0]])


@pytest.mark.parametrize("t_gas", [600.0, 1000.0, 1500.0])
def test_piso_da_energia_media_e_a_do_gas(tab, t_gas):
    # com campo de 1e-4 Td os elétrons ficam na temperatura do gás
    eps_min = tab.limites(np.array([t_gas]))[0][0]
    assert eps_min == pytest.approx(1.5 * KB_EV * t_gas, rel=1e-2)


def test_perda_elastica_depende_da_temperatura_do_gas(tab):
    # na mesma energia média, o gás mais quente devolve energia aos elétrons:
    # a perda elástica cai, e muito quando eps se aproxima de (3/2) k T_g
    eps = np.array([0.1])
    p300 = tab(eps, np.array([300.0]))[0, E_PEL]
    p600 = tab(eps, np.array([600.0]))[0, E_PEL]
    assert 0.0 < p600 < 0.5 * p300


@pytest.mark.parametrize("t_gas", [300.0, 425.0, 650.0, 1125.0, 1375.0])
def test_interpolacao_contra_o_bolsig_fora_dos_nos(tab, t_gas):
    # blocos de conferência do gerador fora dos nós (a 300 K, com E/N a meio
    # passo dos nós: só a interpolação em energia média, lacuna V61), em toda a
    # faixa acima do piso, inclusive perto dele, onde a perda elástica vai a
    # zero: mobilidade e frequência de colisão a 1%, perda elástica a 2%, k0 e
    # k1 a 1,5%, k2 a 1%; o k3 (Ar* -> Ar+) a 1% fora de 0,45 a 0,7 Td (1,7 a
    # 2,0 eV), onde a energia média do BOLSIG+ cresce de forma irregular entre
    # nós, e a 10% ali
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
    """Estado com elétrons e Ar+ a 1e10 cm-3, energia média eps e gás a t_gas."""
    y = md.estado_inicial(n_e0=1e10, neutro=True)
    n = y[:-2].reshape(md.n_var, md.m)
    n[md.n_esp] = 1e10 * eps
    n[md.i_t] = t_gas
    return y


@pytest.mark.parametrize("t_gas", [300.0, 320.0])
def test_modelo_igual_ao_de_300_k_com_o_gas_a_300_k(pecas, tab, tab300, t_gas):
    # com o gás a 300 K a tabela nova dá o mesmo lado direito que a de 300 K,
    # bit a bit; a 320 K, diferente
    from modelo import Modelo
    md_novo = Modelo(*pecas, n_celulas=6, v_pico=3e3, espessura=1.75e-3, aquecimento=True, tabela_energia=tab)
    md_velho = Modelo(*pecas, n_celulas=6, v_pico=3e3, espessura=1.75e-3, aquecimento=True, tabela_energia=tab300)
    y = _estado(md_novo, 0.06, t_gas)
    iguais = np.array_equal(md_novo.rhs(2.5e-8, y), md_velho.rhs(2.5e-8, y))
    assert iguais == (t_gas == 300.0)


def test_modelo_usa_a_temperatura_local_do_gas(pecas, tab):
    # a perda elástica e o piso da energia média seguem a T_g de cada célula
    from modelo import Modelo
    md = Modelo(*pecas, n_celulas=4, aquecimento=True, tabela_energia=tab)
    y = _estado(md, 0.3, 300.0)
    n = md.separa(y)[0]
    tg = np.array([300.0, 600.0, 1000.0, 1500.0])
    _, _, _, _, _, extra = md.coeficientes(np.zeros(md.m), n, md.energia_do_estado(y), t_gas=tg)
    np.testing.assert_allclose(extra["pel"], tab(np.full(4, 0.3), tg)[:, E_PEL], rtol=1e-9)
    # elétrons abaixo do piso de 600 K sobem ao piso local, ~(3/2) k T_g
    eps = md.energia_media(n, n[md.i_e] * 0.02, np.zeros(md.m), tg)
    np.testing.assert_allclose(eps, tab.limites(tg)[0], rtol=1e-12)
    assert eps[1] == pytest.approx(1.5 * KB_EV * 600.0, rel=1e-2)


def test_coeficientes_prendem_a_energia_no_piso_local(pecas, tab):
    # pelo caminho do lado direito: elétrons frios sobem ao piso da T_g local
    from modelo import Modelo
    md = Modelo(*pecas, n_celulas=4, aquecimento=True, tabela_energia=tab)
    y = _estado(md, 0.3, 300.0)
    n = md.separa(y)[0]
    tg = np.array([300.0, 600.0, 1000.0, 1500.0])
    _, _, _, _, _, extra = md.coeficientes(np.zeros(md.m), n, n[md.i_e] * 0.02, t_gas=tg)
    np.testing.assert_allclose(extra["eps"], tab.limites(tg)[0], rtol=1e-12)


def test_abaixo_do_piso_os_eletrons_ganham_energia_do_gas(pecas, tab):
    # elétrons mais frios que o gás (gás aquecendo depressa, ou elétrons que
    # chegam a uma célula mais quente): a perda elástica fica negativa, com a
    # relaxação (2m/M)(nu/N)(eps - piso) além do valor no piso
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
    assert extra["pel"][2] == pytest.approx(v[2, E_PEL], rel=1e-6)     # no piso, o valor da tabela
    assert extra["pel"][3] > 0.0


def test_jacobiano_estruturado_com_o_gas_quente(tab):
    # o jacobiano estruturado pega a dependência dos coeficientes em T_g
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


# ------------------------------------------- campo local com T_g (G4 em campo local)
@pytest.fixture(scope="module")
def tab_lfa():
    from tabela import TabelaTg
    return TabelaTg(RAIZ / "tabela_energia_tg.dat")


def test_campo_local_a_300_k_e_a_tabela_atual(tab_lfa):
    # a 300 K, as colunas da tabela_bolsig.dat (Te volta da energia média: 2e-16)
    from tabela import TE, Tabela
    velha = Tabela(RAIZ / "tabela_bolsig.dat")
    e_n = np.geomspace(1e-3, 3000.0, 500)
    v, ref = tab_lfa(e_n, np.full(e_n.size, 300.0)), velha(e_n)
    np.testing.assert_allclose(v[:, TE], ref[:, TE], rtol=1e-14)
    assert np.array_equal(v[:, TE + 1:], ref[:, TE + 1:])


@pytest.mark.parametrize("t_gas", [300.0, 425.0, 650.0, 1125.0, 1375.0])
def test_campo_local_contra_o_bolsig_fora_dos_nos(tab_lfa, t_gas):
    # mesmos blocos de conferência, por E/N: mobilidade e frequência de colisão
    # a 1%; difusão e perda elástica a 2% (até 1,5% e 0,8% medidos, abaixo de
    # 0,1 Td, com elétrons térmicos, no intervalo de 1000 a 1250 K); taxas a
    # 1,5%; o k3 entre 0,4 e 0,7 Td a 5% (até 3,1% medido a 1125 e 1375 K e
    # 0,13% a 300 K: a irregularidade do BOLSIG+ nessa faixa muda de posição
    # entre nós de T_g, e a interpolação linear em T_g não a acompanha)
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
    # a EEDF gravada em campo local sai da energia de campo local na T_g da célula
    from modelo import Modelo
    from simula import energia_da_eedf
    md = Modelo(pecas[0], tab_lfa, n_celulas=4, aquecimento=True)
    tg = np.array([300.0, 600.0, 1125.0, 1500.0])
    e_n = np.array([0.01, 0.1, 1.0, 10.0])
    np.testing.assert_allclose(energia_da_eedf(md, e_n, {"tg": tg}), tab.energia_de_campo_local(e_n, tg), rtol=1e-14)


@pytest.mark.parametrize("t_gas", [425.0, 650.0, 1125.0, 1375.0])
def test_energia_de_campo_local_com_a_temperatura(tab, t_gas):
    # energia média de regime por E/N e T_g contra os blocos de conferência
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
