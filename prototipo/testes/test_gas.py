# SPDX-License-Identifier: GPL-3.0-or-later
"""Properties of the background gas per case (gas.json): background gas, c_v,
thermal conductivity and ion of the initial plasma. The default (gas.json at the
root) is argon, with the values that were in the code."""
import copy
import json

import numpy as np
import pytest

from conftest import RAIZ, quimica_do, precisa_arquivo
from modelo import ErroGas, Modelo, condutividade_termica
from quimica_zdp import Quimica
from tabela import TabelaTg


@pytest.fixture(scope="module")
def pecas():
    return quimica_do(RAIZ), TabelaTg(RAIZ / "tabela_energia_tg.dat")


@pytest.fixture
def gas():
    return json.loads((RAIZ / "gas.json").read_text())


def test_padrao_e_o_argonio(pecas):
    md = Modelo(*pecas, n_celulas=4, aquecimento=True)
    t = np.array([300.0, 1000.0, 2999.0, 3500.0, 8000.0])
    assert np.array_equal(md.condutividade(t), condutividade_termica(t))       # bit a bit
    assert md.cv_por_kb == 1.5 and md.nomes[md.ion_inicial] == "AR^+"


def test_cv_do_gas_entra_no_aquecimento(pecas, gas):
    # dT/dt is the power divided by cv N k_B: with cv = 2.5 (molecular), 1.5/2.5
    # of that of argon, with the same sources
    md_ar = Modelo(*pecas, n_celulas=6, aquecimento=True, v_pico=3e3, espessura=1.75e-3)
    gas["cv_por_kb"] = 2.5
    md_mol = Modelo(*pecas, n_celulas=6, aquecimento=True, v_pico=3e3, espessura=1.75e-3, gas=gas)
    y = md_ar.estado_inicial(n_e0=1e12, neutro=True)
    y[:-2].reshape(md_ar.n_var, md_ar.m)[md_ar.i_t] = np.linspace(300.0, 340.0, md_ar.m)
    d_ar = md_ar.rhs(2.5e-8, y)[:-2].reshape(md_ar.n_var, md_ar.m)[md_ar.i_t]
    d_mol = md_mol.rhs(2.5e-8, y)[:-2].reshape(md_mol.n_var, md_mol.m)[md_mol.i_t]
    np.testing.assert_allclose(d_mol, d_ar * 1.5 / 2.5, rtol=1e-12)


def test_condutividade_por_tabela(pecas, gas):
    gas["condutividade"] = {"tipo": "tabela", "T": [300.0, 600.0, 1200.0], "kappa": [0.026, 0.044, 0.073]}
    md = Modelo(*pecas, n_celulas=4, gas=gas)
    np.testing.assert_allclose(md.condutividade(np.array([300.0, 450.0, 900.0, 1200.0])),
                               [0.026, 0.035, 0.0585, 0.073], rtol=1e-14)


def test_ion_inicial_do_gas(pecas, gas):
    gas["ion_inicial"] = "AR2^+"
    md = Modelo(*pecas, n_celulas=4, gas=gas)
    n = md.separa(md.estado_inicial(n_e0=5.0, neutro=True))[0]
    assert np.all(n[md.nomes.index("AR2^+")] == 5.0) and np.all(n[md.nomes.index("AR^+")] == 0.0)


@pytest.mark.parametrize("mudanca, mensagem", [
    (lambda g: g.pop("cv_por_kb"), "cv_por_kb"),
    (lambda g: g.update(cv_por_kb=-1.0), "cv_por_kb"),
    (lambda g: g.update(fundo={"AR": 0.8, "AR*": 0.3}), "add up"),
    (lambda g: g.update(fundo={"AR": 1.2, "AR*": -0.2}), "positive"),
    (lambda g: g.update(fundo={"XE": 1.0}), "XE"),
    (lambda g: g.update(ion_inicial="AR*"), "ion_inicial"),
    (lambda g: g.update(condutividade={"tipo": "outra"}), "condutividade")])
def test_gas_invalido_e_erro(pecas, gas, mudanca, mensagem):
    g = copy.deepcopy(gas)
    mudanca(g)
    with pytest.raises(ErroGas, match=mensagem):
        Modelo(*pecas, n_celulas=4, gas=g)


# -------------------------------------------------- mixed background gas
def test_mistura_no_fundo(pecas, gas):
    # background Ar 0.8 + Ar* 0.2: both leave the transport and enter the chemistry with
    # the fraction times N
    gas["fundo"] = {"AR": 0.8, "AR*": 0.2}
    md = Modelo(*pecas, n_celulas=4, gas=gas)
    assert "AR" not in md.nomes and "AR*" not in md.nomes
    n = md.separa(md.estado_inicial(n_e0=1e9, neutro=True))[0]
    todas = md.densidades_completas(n)
    i = md.q.indice
    np.testing.assert_allclose(todas[i["AR"]], 0.8 * md.n_gas, rtol=1e-15)
    np.testing.assert_allclose(todas[i["AR*"]], 0.2 * md.n_gas, rtol=1e-15)


def test_mistura_tira_do_fundo_a_energia_das_colisoes(pecas, gas):
    # with Ar* in the background, the collisions with Ar* are already in the tabulated power of the
    # mixture: their energy per reaction is zero
    from tabela import TabelaEnergiaTg
    te = TabelaEnergiaTg(RAIZ / "tabela_energia_tg.dat")
    so_ar = Modelo(*pecas, n_celulas=4, tabela_energia=te)
    gas["fundo"] = {"AR": 0.8, "AR*": 0.2}
    mist = Modelo(*pecas, n_celulas=4, tabela_energia=te, gas=gas)
    assert np.any(so_ar.energia_reacao != 0.0)
    assert np.all(mist.energia_reacao == 0.0)


@precisa_arquivo("casos/n2/quimica.json")
def test_massa_da_troca_elastica_na_mistura(pecas_n2):
    # 1/M = sum of x_i/M_i (approximation: weighted by the fraction, not by the collision
    # frequency of each gas)
    gas = json.loads((N2 / "gas.json").read_text())
    gas.pop("calor", None)
    gas["fundo"] = {"N2": 0.75, "N": 0.25}
    md = Modelo(*pecas_n2, n_celulas=4, gas=gas, transporte=N2 / "transporte.json")
    u = 1.66053906660e-27
    t = json.loads((N2 / "transporte.json").read_text())
    esperado = 1.0 / (0.75 / (t["N2"]["massa_u"] * u) + 0.25 / (t["N"]["massa_u"] * u))
    assert md.m_fundo == pytest.approx(esperado, rel=1e-12)


# -------------------------------------------------- heat of the reactions (gas.json)
N2 = RAIZ / "casos" / "n2"
ENTALPIAS_N2 = {"N2": 0.0, "N": 4.88, "N2(A)": 6.17, "N2(B)": 7.35, "N2(A')": 8.40, "N2(C)": 11.03,
                "N^+": 19.41, "N2^+": 15.58, "N3^+": 18.0, "N4^+": 14.5}


@pytest.fixture(scope="module")
def pecas_n2():
    return quimica_do(N2), TabelaTg(N2 / "tabela_energia_tg.dat")


def _modelo_n2(pecas_n2, calor, **kw):
    gas = json.loads((N2 / "gas.json").read_text())
    gas.pop("calor", None)
    if calor is not None:
        gas["calor"] = calor
    return Modelo(*pecas_n2, n_celulas=6, aquecimento=True, v_pico=3e3, espessura=1.75e-3, gas=gas,
                  transporte=N2 / "transporte.json", **kw)


@precisa_arquivo("casos/n2/quimica.json")
def test_calor_de_cada_reacao_pelas_entalpias(pecas_n2):
    md = _modelo_n2(pecas_n2, {"entalpias": ENTALPIAS_N2})
    q = md.q
    def calor(reacao):
        return md.calor_reacao[q.reacoes.index(reacao)]
    assert calor("N2(A)+N2=>N2+N2") == pytest.approx(6.17)             # quenching
    assert calor("N2^++E=>N+N") == pytest.approx(15.58 - 2 * 4.88)     # dissociative recombination
    assert calor("N2^++N2+N2=>N4^++N2") == pytest.approx(15.58 - 14.5)
    assert calor("N2(A)=>N2") == 0.0                                    # radiative: the photon leaves
    assert calor("bolsig:N2->N2(B3)") == 0.0                            # electron energy, already in P_inel


@precisa_arquivo("casos/n2/quimica.json")
def test_calor_das_reacoes_entra_no_aquecimento(pecas_n2):
    # dT/dt with heat = without heat + sum of the rates times the heat of each reaction,
    # divided by cv N k_B (in local field; with --energy the mean energy of the
    # consumed electrons also enters)
    sem = _modelo_n2(pecas_n2, None)
    com = _modelo_n2(pecas_n2, {"entalpias": ENTALPIAS_N2})
    y = sem.estado_inicial(n_e0=1e12, neutro=True)
    z = y[:-2].reshape(sem.n_var, sem.m)
    for nome, d in (("N2(A)", 1e15), ("N4^+", 5e11), ("N", 1e14)):
        z[sem.nomes.index(nome)] = d
    d_sem = sem.rhs(2.5e-8, y)[:-2].reshape(sem.n_var, sem.m)[sem.i_t]
    d_com = com.rhs(2.5e-8, y)[:-2].reshape(com.n_var, com.m)[com.i_t]
    n = com.separa(y)[0]
    campo = com.campo(2.5e-8, n, y[-2], y[-1])[1]
    _, _, temp, k_bolsig, e_n, extra = com.coeficientes(com.campo_nas_celulas(campo), n, com.energia_do_estado(y)
                                                        if com.energia else None, com.temperatura_do_estado(y))
    taxas = com.q.taxas(com.densidades_completas(n), com.q.coeficientes(k_bolsig, extra["tg"], temp[com.i_e], e_n))
    q_reac = com.calor_reacao @ taxas * 1e6 * 1.602176634e-19
    esperado = d_sem + q_reac / (com.cv_por_kb * 1.380649e-23 * com.n_pesadas(n) * 1e6)
    np.testing.assert_allclose(d_com, esperado, rtol=1e-9)
    assert np.all(d_com > d_sem)


@precisa_arquivo("casos/n2/quimica.json")
def test_entalpia_faltando_e_erro(pecas_n2):
    ent = dict(ENTALPIAS_N2)
    del ent["N4^+"]
    with pytest.raises(ErroGas, match=r"enthalpy.*N4\^\+"):
        _modelo_n2(pecas_n2, {"entalpias": ent})


@precisa_arquivo("casos/n2/quimica.json")
def test_com_energia_os_eletrons_consumidos_levam_a_energia_media_ao_gas(pecas_n2):
    # with --energy, the mean energy of the electrons lost in common reactions
    # (which d_energia removes) also becomes gas heat
    from tabela import TabelaEnergiaTg
    te = TabelaEnergiaTg(N2 / "tabela_energia_tg.dat")
    sem = _modelo_n2(pecas_n2, None, tabela_energia=te)
    com = _modelo_n2(pecas_n2, {"entalpias": ENTALPIAS_N2}, tabela_energia=te)
    y = sem.estado_inicial(n_e0=1e12, neutro=True)
    z = y[:-2].reshape(sem.n_var, sem.m)
    z[sem.n_esp] = 1e12 * 1.5                                     # n_eps: mean energy of 1.5 eV
    d_sem = sem.rhs(2.5e-8, y)[:-2].reshape(sem.n_var, sem.m)[sem.i_t]
    d_com = com.rhs(2.5e-8, y)[:-2].reshape(com.n_var, com.m)[com.i_t]
    n = com.separa(y)[0]
    campo = com.campo(2.5e-8, n, y[-2], y[-1])[1]
    _, _, temp, k_bolsig, e_n, extra = com.coeficientes(com.campo_nas_celulas(campo), n, com.energia_do_estado(y),
                                                        com.temperatura_do_estado(y))
    taxas = com.q.taxas(com.densidades_completas(n), com.q.coeficientes(k_bolsig, extra["tg"], temp[com.i_e], e_n))
    ie = com.q.indice["E"]
    perdidos = np.where([c is None for c in com.q.coluna_bolsig], np.minimum(com.q.estequiometria[ie], 0.0), 0.0)
    q = (com.calor_reacao @ taxas - extra["eps"] * (perdidos @ taxas)) * 1e6 * 1.602176634e-19
    np.testing.assert_allclose(d_com, d_sem + q / (com.cv_por_kb * 1.380649e-23 * com.n_pesadas(n) * 1e6), rtol=1e-9)


@precisa_arquivo("casos/n2/quimica.json")
def test_calor_rapido_dos_eletrons_entra_no_aquecimento(pecas_n2, tmp_path):
    # fast heat table (constant P/N, eV cm3/s): the gas receives P N n_e
    from tabela import TabelaEnergiaTg
    linhas = ["# T_g, E/N, energia media, P_rapido/N"]
    for tg in (300.0, 1500.0):
        for en, eps in ((1e-4, 0.039), (1e3, 20.0)):
            linhas.append(f"{tg} {en} {eps} 3e-10")
        linhas.append("")
    (tmp_path / "c.dat").write_text("\n".join(linhas))
    sem = _modelo_n2(pecas_n2, {"entalpias": ENTALPIAS_N2})
    com = _modelo_n2(pecas_n2, {"entalpias": ENTALPIAS_N2, "processos_rapidos": ["N2->N2(ROT)"]},
                     tabela_calor=TabelaEnergiaTg(tmp_path / "c.dat"))
    y = sem.estado_inicial(n_e0=1e12, neutro=True)
    d_sem = sem.rhs(2.5e-8, y)[:-2].reshape(sem.n_var, sem.m)[sem.i_t]
    d_com = com.rhs(2.5e-8, y)[:-2].reshape(com.n_var, com.m)[com.i_t]
    n = com.separa(y)[0]
    q = 3e-10 * com.n_neutro(n) * n[com.i_e] * 1e6 * 1.602176634e-19
    np.testing.assert_allclose(d_com, d_sem + q / (com.cv_por_kb * 1.380649e-23 * com.n_pesadas(n) * 1e6), rtol=1e-9)


@precisa_arquivo("casos/n2/quimica.json")
def test_processos_rapidos_sem_tabela_e_erro(pecas_n2):
    with pytest.raises(ErroGas, match="tabela_calor"):
        _modelo_n2(pecas_n2, {"entalpias": ENTALPIAS_N2, "processos_rapidos": ["N2->N2(ROT)"]})


@precisa_arquivo("casos/n2/quimica.json")
def test_caso_n2_monta_com_o_proprio_calor(pecas_n2):
    # the gas.json of the case has the enthalpy of every species and the generated fast
    # heat table; the fast heat of the table is positive in a discharge field
    from tabela import TabelaEnergiaTg
    gas = json.loads((N2 / "gas.json").read_text())
    tab = TabelaEnergiaTg(N2 / "tabela_calor_tg.dat")
    md = Modelo(*pecas_n2, n_celulas=6, aquecimento=True, gas=gas, transporte=N2 / "transporte.json",
                tabela_calor=tab)
    assert md.calor_reacao[md.q.reacoes.index("N2^++E=>N+N")] == pytest.approx(15.581 - 2 * 4.899)
    assert tab.energia_de_campo_local(np.array([100.0]), np.array([300.0]))[0] > 1.0
    assert tab(np.array([2.0]), np.array([300.0]))[0, 1] > 0.0


def test_mesmo_gas_duas_vezes_no_fundo_e_erro(pecas, gas):
    gas["fundo"] = {"AR": 0.5, "ar": 0.5}
    with pytest.raises(ErroGas, match="twice"):
        Modelo(*pecas, n_celulas=4, gas=gas)
