# SPDX-License-Identifier: GPL-3.0-or-later
"""Transport data per species (transporte.json): the default file reproduces the
argon values that were written in the code, each species uses its own data and a
missing value is an explicit error, instead of the Ar+ mobility or the Ar*
diffusion applied silently."""
import copy
import json

import numpy as np
import pytest

from conftest import RAIZ, quimica_do, precisa_arquivo
from modelo import KB, ME, ErroTransporte, Modelo
from quimica_zdp import Quimica
from tabela import Tabela

U = 1.66053906660e-27
N_LOSCHMIDT = 2.6867811e19


@pytest.fixture(scope="module")
def pecas():
    return quimica_do(RAIZ), Tabela(RAIZ / "tabela_bolsig.dat")


@pytest.fixture
def dados():
    return json.loads((RAIZ / "transporte.json").read_text())


def test_padrao_reproduz_os_valores_do_argonio(pecas):
    md = Modelo(*pecas, n_celulas=4, gas_solto=True)
    m_ar = 39.948 * U
    esperado = {"E": ME, "AR": m_ar, "AR*": m_ar, "AR^+": m_ar, "AR2^+": 2.0 * m_ar}
    for k, nome in enumerate(md.nomes):
        assert md.massa[k] == esperado[nome], nome                   # bit a bit
    atomos = {"E": 0.0, "AR": 0.0, "AR*": 1.0, "AR^+": 1.0, "AR2^+": 2.0}
    assert list(md.atomos[0]) == [atomos[n] for n in md.nomes]
    p_atm = (md.n_gas * 1e6) * (1.38e-23 * 300.0) / 101325.0
    k = md.nomes.index("AR*")
    assert md.difusao[k] == 17.0e-6 / p_atm * (300.0 / 273.0) ** 1.9
    campo = np.array([0.0, 1e6, 1e7])
    escala = N_LOSCHMIDT / md.n_gas * 1e-4
    e_n = campo / md.n_gas * 1e15
    assert np.array_equal(md.mobilidade_ion(md.nomes.index("AR^+"), campo),
                          1.61 * escala / np.sqrt(1.0 + e_n / 210.0))
    assert np.array_equal(md.mobilidade_ion(md.nomes.index("AR2^+"), campo), np.full(3, 1.95 * escala))


def test_cada_especie_usa_os_seus_dados(pecas, dados):
    # Ar2+ with its own Frost mobility: the model uses it, not that of Ar+
    dados["AR2^+"]["mobilidade"] = {"tipo": "frost", "K0": 2.0, "EN_aquecimento": 100.0}
    dados["AR*"]["difusao"]["D0"] = 2.0e-5
    dados["AR2^+"]["massa_u"] = 80.0
    md = Modelo(*pecas, n_celulas=4, transporte=dados)
    campo = np.array([1e6])
    escala = N_LOSCHMIDT / md.n_gas * 1e-4
    mu = md.mobilidade_ion(md.nomes.index("AR2^+"), campo)
    assert mu[0] == pytest.approx(2.0 * escala / np.sqrt(1.0 + campo[0] / md.n_gas * 1e15 / 100.0), rel=1e-14)
    p_atm = (md.n_gas * 1e6) * (1.38e-23 * 300.0) / 101325.0
    assert md.difusao[md.nomes.index("AR*")] == pytest.approx(2.0e-5 / p_atm * (300.0 / 273.0) ** 1.9, rel=1e-14)
    assert md.massa[md.nomes.index("AR2^+")] == pytest.approx(80.0 * U, rel=1e-15)


def test_difusao_local_usa_os_dados_da_especie(pecas, dados):
    # with heating the diffusion of the neutrals follows the local T, with the same data
    dados["AR*"]["difusao"] = {"D0": 3.0e-5, "T0": 300.0, "expoente": 1.5}
    md = Modelo(*pecas, n_celulas=4, aquecimento=True, transporte=dados)
    n = md.separa(md.estado_inicial())[0]
    tg = np.full(md.m, 600.0)
    _, dif, *_ = md.coeficientes(np.zeros(md.m), n, t_gas=tg)
    p_atm = (md.n_pesadas(n) * 1e6) * (1.38e-23 * 600.0) / 101325.0
    np.testing.assert_allclose(dif[md.nomes.index("AR*")], 3.0e-5 / p_atm * (600.0 / 300.0) ** 1.5, rtol=1e-14)


@pytest.mark.parametrize("especie, campo, mensagem", [
    ("AR2^+", "mobilidade", "AR2^+"), ("AR*", "difusao", "AR*"), ("AR^+", "massa_u", "AR^+")])
def test_dado_ausente_e_erro(pecas, dados, especie, campo, mensagem):
    del dados[especie][campo]
    with pytest.raises(ValueError, match=r"transport.*" + mensagem.replace("^", r"\^").replace("*", r"\*")
                       + r".*" + campo):
        Modelo(*pecas, n_celulas=4, transporte=dados)


def test_especie_ausente_e_erro(pecas, dados):
    del dados["AR2^+"]
    with pytest.raises(ValueError, match=r"AR2\^\+"):
        Modelo(*pecas, n_celulas=4, transporte=dados)


def test_atomos_so_sao_exigidos_com_gas_solto(pecas, dados):
    del dados["AR*"]["atomos_do_fundo"]
    Modelo(*pecas, n_celulas=4, transporte=dados)
    with pytest.raises(ValueError, match=r"AR\*.*atomos_do_fundo"):
        Modelo(*pecas, n_celulas=4, gas_solto=True, transporte=dados)


def test_tipo_de_mobilidade_desconhecido_e_erro(pecas, dados):
    dados["AR^+"]["mobilidade"]["tipo"] = "langevin"
    with pytest.raises(ValueError, match="langevin"):
        Modelo(*pecas, n_celulas=4, transporte=dados)


def test_ion_negativo_sem_mobilidade_e_erro(pecas):
    # a negative ion without mobility in the file cannot silently become a neutral (without
    # drift)
    q = copy.copy(pecas[0])
    q.carga = q.carga.copy()
    q.carga[q.indice["AR*"]] = -1
    with pytest.raises(ValueError, match="AR\\*.*mobilidade|mobilidade.*AR\\*"):
        Modelo(q, pecas[1], n_celulas=4)


def test_opcoes_do_argonio_sobrepoem_o_arquivo(pecas):
    md = Modelo(*pecas, n_celulas=4, k0_ar=1.5, en_aquecimento_ar=200.0, k0_ar2=1.9)
    assert md.transporte["AR^+"]["mobilidade"] == {"tipo": "frost", "K0": 1.5, "EN_aquecimento": 200.0}
    assert md.transporte["AR2^+"]["mobilidade"]["K0"] == 1.9
    # the file on disk does not change
    assert json.loads((RAIZ / "transporte.json").read_text())["AR^+"]["mobilidade"]["K0"] == 1.61



@pytest.mark.parametrize("especie, caminho, valor", [
    ("AR^+", ("mobilidade", "K0"), -1.61), ("AR^+", ("mobilidade", "EN_aquecimento"), 0.0),
    ("AR*", ("difusao", "D0"), 0.0), ("AR*", ("difusao", "T0"), -273.0),
    ("AR*", ("difusao", "expoente"), float("nan")), ("AR^+", ("massa_u",), "39.948"),
    ("AR2^+", ("mobilidade",), 1.95), ("AR*", ("difusao",), 17e-6),
    ("AR*", ("perda_na_parede",), 1.5), ("AR*", ("perda_na_parede",), -0.1)])
def test_valor_invalido_e_erro(pecas, dados, especie, caminho, valor):
    # the presence of the keys alone is not enough: a negative K0 would silently invert the
    # drift, and a wrong type would end in a traceback
    alvo = dados[especie]
    for chave in caminho[:-1]:
        alvo = alvo[chave]
    alvo[caminho[-1]] = valor
    with pytest.raises(ErroTransporte, match=especie.replace("^", r"\^").replace("*", r"\*") + ".*" + caminho[-1]):
        Modelo(*pecas, n_celulas=4, transporte=dados)


def test_atomos_negativos_e_erro(pecas, dados):
    dados["AR2^+"]["atomos_do_fundo"] = -2
    with pytest.raises(ErroTransporte, match=r"AR2\^\+.*atomos_do_fundo"):
        Modelo(*pecas, n_celulas=4, gas_solto=True, transporte=dados)


def test_erros_de_transporte_sao_valueerror(pecas, dados):
    del dados["AR2^+"]
    with pytest.raises(ValueError):
        Modelo(*pecas, n_celulas=4, transporte=dados)
    assert issubclass(ErroTransporte, ValueError)


def test_opcao_do_argonio_incompativel_e_erro(pecas, dados):
    # a heating field for an Ar+ of constant mobility would have no effect
    dados["AR^+"]["mobilidade"] = {"tipo": "constante", "K0": 1.61}
    with pytest.raises(ErroTransporte, match="frost"):
        Modelo(*pecas, n_celulas=4, en_aquecimento_ar=200.0, transporte=dados)


@pytest.mark.parametrize("gama", [1.0, 0.5, 0.0])
def test_perda_na_parede_dos_neutros(pecas, dados, gama):
    # Chantry flux at the wall: gamma/(2 - gamma) of the thermal flux n v/2
    dados["AR*"]["perda_na_parede"] = gama
    md = Modelo(*pecas, n_celulas=4, transporte=dados)
    k = md.nomes.index("AR*")
    n = np.zeros((md.n_esp, md.m))
    n[k] = 1e10
    temp = np.full((md.n_esp, md.m), 300.0)
    _, out, _ = md.fluxos(n, np.zeros(md.m + 1), np.zeros((md.n_esp, md.m)), np.zeros((md.n_esp, md.m)), temp)
    v_term = np.sqrt(8.0 * KB * 300.0 / (np.pi * md.massa[k]))
    np.testing.assert_allclose(out[k], gama / (2.0 - gama) * 1e10 * 0.5 * v_term, rtol=1e-14, atol=0)


def test_perda_na_parede_e_exigida_dos_neutros(pecas, dados):
    del dados["AR*"]["perda_na_parede"]
    with pytest.raises(ErroTransporte, match=r"AR\*.*perda_na_parede"):
        Modelo(*pecas, n_celulas=4, transporte=dados)


def test_estado_inicial_neutro_semeia_o_primeiro_ion(pecas):
    md = Modelo(*pecas, n_celulas=4)
    n = md.separa(md.estado_inicial(n_e0=7.0, neutro=True))[0]
    assert md.nomes[md.ions[0]] == "AR^+" and np.all(n[md.ions[0]] == 7.0)
    assert np.all(n[[k for k in md.ions[1:]]] == 0.0)


@precisa_arquivo("src/parametros.f90")
def test_mobilidades_iguais_as_do_fortran(dados):
    # fortran_contra_prototipo depends on the defaults of the two codes being
    # equal; src/parametros.f90 has those of the Fortran code
    import re
    texto = (RAIZ / "src" / "parametros.f90").read_text()
    valor = {m.group(1): float(m.group(2).replace("d", "e"))
             for m in re.finditer(r"(k0_ar_ion|aquecimento_ar_ion|k0_ar2_ion)\s*=\s*([0-9.]+d[+-]?[0-9]+)", texto)}
    assert dados["AR^+"]["mobilidade"]["K0"] == valor["k0_ar_ion"]
    assert dados["AR^+"]["mobilidade"]["EN_aquecimento"] == valor["aquecimento_ar_ion"]
    assert dados["AR2^+"]["mobilidade"]["K0"] == valor["k0_ar2_ion"]


def test_mobilidade_em_tabela_por_e_n(pecas, dados):
    # K0 measured as a function of E/N (as in the tables of Ellis et al. 1976),
    # interpolated linearly and clamped at the ends
    dados["AR2^+"]["mobilidade"] = {"tipo": "tabela", "E_N": [10.0, 100.0, 300.0], "K0": [2.0, 1.8, 1.2]}
    md = Modelo(*pecas, n_celulas=4, transporte=dados)
    escala = N_LOSCHMIDT / md.n_gas * 1e-4
    e_n = np.array([1.0, 10.0, 55.0, 200.0, 1000.0])
    campo = e_n * md.n_gas * 1e-15
    np.testing.assert_allclose(md.mobilidade_ion(md.nomes.index("AR2^+"), campo),
                               np.array([2.0, 2.0, 1.9, 1.5, 1.2]) * escala, rtol=1e-12)


@pytest.mark.parametrize("mob", [{"tipo": "tabela", "E_N": [10.0, 5.0], "K0": [2.0, 1.8]},
                                 {"tipo": "tabela", "E_N": [10.0, 50.0], "K0": [2.0]},
                                 {"tipo": "tabela", "E_N": [10.0, 50.0], "K0": [2.0, -1.0]}])
def test_mobilidade_em_tabela_invalida(pecas, dados, mob):
    dados["AR2^+"]["mobilidade"] = mob
    with pytest.raises(ErroTransporte, match=r"AR2\^\+.*table"):
        Modelo(*pecas, n_celulas=4, transporte=dados)


def test_mobilidade_em_tabela_com_cauda_de_campo_alto(pecas, dados):
    # above the last measured E/N, K = K_last (E/N_last / (E/N))^(1/2)
    dados["AR2^+"]["mobilidade"] = {"tipo": "tabela", "E_N": [10.0, 40.0], "K0": [2.3, 2.4], "acima": "raiz"}
    md = Modelo(*pecas, n_celulas=4, transporte=dados)
    escala = N_LOSCHMIDT / md.n_gas * 1e-4
    e_n = np.array([20.0, 40.0, 160.0, 640.0])
    mu = md.mobilidade_ion(md.nomes.index("AR2^+"), e_n * md.n_gas * 1e-15)
    np.testing.assert_allclose(mu, np.array([2.3 + 0.1 / 3.0, 2.4, 1.2, 0.6]) * escala, rtol=1e-12)


def test_cauda_desconhecida_e_erro(pecas, dados):
    dados["AR2^+"]["mobilidade"] = {"tipo": "tabela", "E_N": [10.0, 40.0], "K0": [2.3, 2.4], "acima": "outra"}
    with pytest.raises(ErroTransporte, match="acima"):
        Modelo(*pecas, n_celulas=4, transporte=dados)
