# SPDX-License-Identifier: GPL-3.0-or-later
"""Gás de fundo solto, como gas_fixo = .false. no código Fortran: o Ar neutro
entra no estado, difunde com o coeficiente do Ar* (DK), participa da química e
recebe de volta pela parede, como átomos neutros, as partículas de argônio que
ela absorve (achado N15). O N dos elétrons (E/N e coeficientes da tabela) é o
dos neutros da célula, Ar + Ar*."""
import numpy as np
import pytest

from conftest import RAIZ, quimica_do
from jacobiano import ModeloRapido
from quimica_zdp import Quimica
from tabela import Tabela
from test_jacobiano import jacobiano_por_diferencas_centrais


@pytest.fixture(scope="module")
def pecas():
    return quimica_do(RAIZ), Tabela(RAIZ / "tabela_bolsig.dat")


def aleatorio(md, semente):
    g = np.random.default_rng(semente)
    n = 10.0 ** g.uniform(9, 12, size=(md.n_esp, md.m))
    n[md.nomes.index("AR")] = md.n_gas * g.uniform(0.9, 1.1, md.m)
    return n


def test_atomos_de_argonio_conservados(pecas):
    # transporte, paredes e química: a soma de Ar, Ar*, Ar+ e 2 Ar2+ no gap não
    # muda; os elétrons entram e saem, os átomos não
    md = ModeloRapido(*pecas, n_celulas=16, agrupamento=1.5, v_pico=5e3, espessura=1.75e-3,
                      gas_solto=True)
    n = aleatorio(md, 7)
    y = np.concatenate([n.ravel(), [2e-7, -3e-7]])
    dn = md.rhs(2.2e-8, y)[:-2].reshape(md.n_esp, md.m)
    atomos = np.array([0.0 if s == "E" else (2.0 if s.startswith("AR2") else 1.0) for s in md.nomes])
    variacao = atomos @ (dn * md.h).sum(axis=1)
    escala = np.abs(dn[md.nomes.index("AR")] * md.h).sum()
    assert abs(variacao) < 1e-12 * escala


def test_parede_devolve_os_atomos_como_ar_neutro(pecas):
    # Ar uniforme e 1e6 cm-3 de Ar* só na célula da parede, sem campo: a química
    # que pesa (Ar* + Ar -> 2 Ar e Ar* + 2 Ar -> 3 Ar) troca Ar* por Ar um a um, e
    # a parede devolve como Ar o Ar* que absorve; a soma dn(Ar) + dn(Ar*) nessa
    # célula fica só com a difusão do Ar* para a vizinha
    md = ModeloRapido(*pecas, n_celulas=10, v_pico=0.0, gas_solto=True)
    n = np.zeros((md.n_esp, md.m))
    i_ar, i_m = md.nomes.index("AR"), md.nomes.index("AR*")
    n[i_ar] = md.n_gas
    n[i_m, 0] = 1.0e6
    dn = md.rhs(0.0, np.concatenate([n.ravel(), [0.0, 0.0]]))[:-2].reshape(md.n_esp, md.m)
    difusao = md.difusao[md.nomes.index("AR")] * 1.0e6 / md.d[1] / md.h[0]
    assert dn[i_ar, 0] + dn[i_m, 0] == pytest.approx(-difusao, rel=1e-6)


def test_campo_reduzido_com_os_neutros_da_celula(pecas):
    md = ModeloRapido(*pecas, n_celulas=10, v_pico=0.0, gas_solto=True)
    n = np.zeros((md.n_esp, md.m))
    n[md.nomes.index("AR")] = md.n_gas
    n[md.nomes.index("AR")][3] = 0.5 * md.n_gas
    *_, e_n, _ = md.coeficientes(np.full(md.m, 1.0e6), n)
    assert e_n[3] == pytest.approx(2.0 * e_n[2], rel=1e-12)


def test_gas_preso_nada_muda(pecas):
    md0 = ModeloRapido(*pecas, n_celulas=12, v_pico=1e3, espessura=1e-3)
    md1 = ModeloRapido(*pecas, n_celulas=12, v_pico=1e3, espessura=1e-3, gas_solto=True)
    g = np.random.default_rng(3)
    n0 = 10.0 ** g.uniform(8, 11, size=(md0.n_esp, 12))
    n1 = np.zeros((md1.n_esp, 12))
    for k, s in enumerate(md0.nomes):
        n1[md1.nomes.index(s)] = n0[k]
    n1[md1.nomes.index("AR")] = md1.n_gas
    f0 = md0.rhs(1e-8, np.concatenate([n0.ravel(), [1e-7, -2e-7]]))
    f1 = md1.rhs(1e-8, np.concatenate([n1.ravel(), [1e-7, -2e-7]]))
    d0 = f0[:-2].reshape(md0.n_esp, 12)
    d1 = f1[:-2].reshape(md1.n_esp, 12)
    # as espécies comuns mudam só pelo N dos neutros, que passa a incluir o Ar*
    # (~1e-8 do N do gás)
    for k, s in enumerate(md0.nomes):
        np.testing.assert_allclose(d1[md1.nomes.index(s)], d0[k], rtol=1e-6, atol=1e-6 * np.abs(d0).max())
    np.testing.assert_allclose(f1[-2:], f0[-2:], rtol=1e-6)


def test_jacobiano_estruturado_com_gas_solto(pecas):
    md = ModeloRapido(*pecas, n_celulas=16, agrupamento=1.5, v_pico=5e3, espessura=1.75e-3,
                      gas_solto=True, aquecimento=True)
    n = aleatorio(md, 11)
    t_gas = 300.0 + 200.0 * np.random.default_rng(12).uniform(0, 1, md.m)
    y = np.concatenate([n.ravel(), t_gas, [2e-7, -3e-7]])
    t = 2.2e-8
    J_ref = jacobiano_por_diferencas_centrais(md, t, y)
    J = md.jacobiano(t, y)
    peso = np.abs(y)
    escala = np.abs(J_ref * peso).max(axis=1, keepdims=True) + 1e-300
    assert np.max(np.abs(J - J_ref) * peso / escala) < 1e-5


def test_frost_com_o_n_local(pecas):
    # como ions.f90: K = K0 (N0/N) / sqrt(1 + (E/N)/210 Td), com N as partículas
    # pesadas da célula também dentro do E/N
    md = ModeloRapido(*pecas, n_celulas=10, v_pico=0.0, gas_solto=True)
    n = np.zeros((md.n_esp, md.m))
    n[md.nomes.index("AR")] = md.n_gas
    n[md.nomes.index("AR")][3] = 0.5 * md.n_gas
    campo = np.full(md.m, 3.0e6)
    mob, *_ = md.coeficientes(campo, n)
    n_loc = 0.5 * md.n_gas
    e_n = 3.0e6 / n_loc * 1e15
    esperado = 1.61 * 2.6867811e19 / n_loc * 1e-4 / np.sqrt(1.0 + e_n / 210.0)
    assert mob[md.nomes.index("AR^+"), 3] == pytest.approx(esperado, rel=1e-12)


# ------------------------------------------- gás solto com o fundo em mistura
AMONIA = RAIZ / "casos" / "amonia_sun2022"


def _atomos(nome):
    """Átomos de N e de H na espécie, pela fórmula do nome."""
    import re
    s = nome.upper().replace("^+", "").replace("^-", "").split("(")[0]
    c = {"N": 0, "H": 0}
    for el, num in re.findall(r"([NH])(\d*)", s):
        c[el] += int(num or 1)
    return c


@pytest.fixture(scope="module")
def amonia():
    import json
    from tabela import TabelaTg
    q = quimica_do(AMONIA)
    transporte = json.loads((AMONIA / "transporte.json").read_text())
    # a parede devolve cada partícula como N2 e H2 com os seus átomos
    for nome, dados in transporte.items():
        if not nome.startswith("_"):
            a = _atomos(nome)
            dados["atomos_do_fundo"] = {"N2": a["N"] / 2.0, "H2": a["H"] / 2.0}
    for fundo in ("N2", "H2"):
        transporte[fundo]["difusao"] = dict(transporte[fundo + "(V1)"]["difusao"])
    return q, TabelaTg(AMONIA / "tabela_energia_tg.dat"), transporte


def _modelo_amonia(amonia, **kw):
    q, tab, transporte = amonia
    return ModeloRapido(q, tab, n_celulas=12, agrupamento=1.5, gap=5e-3, espessura=1e-3, espessura_dir=2e-3,
                        eps_dieletrico=4.3, v_pico=7e3, t_gas=573.0, n_gas=2e4 / (1.380649e-23 * 573.0) * 1e-6,
                        gas=AMONIA / "gas.json", transporte=transporte, gas_solto=True, **kw)


def test_mistura_solta_comeca_nas_fracoes(amonia):
    md = _modelo_amonia(amonia)
    n = md.separa(md.estado_inicial(n_e0=1e9, neutro=True))[0]
    for nome in ("N2", "H2"):
        np.testing.assert_allclose(n[md.nomes.index(nome)], 0.5 * md.n_gas, rtol=1e-15)


def test_mistura_solta_conserva_os_atomos_de_cada_elemento(amonia):
    # transporte, paredes (devolução como N2 e H2) e química: os átomos de N e
    # os de H no gap não mudam, cada um por si
    md = _modelo_amonia(amonia)
    g = np.random.default_rng(3)
    n = 10.0 ** g.uniform(9, 12, size=(md.n_esp, md.m))
    for nome in ("N2", "H2"):
        n[md.nomes.index(nome)] = 0.5 * md.n_gas * g.uniform(0.9, 1.1, md.m)
    y = np.concatenate([n.ravel(), [2e-7, -3e-7]])
    dn = md.rhs(1e-9, y)[:-2].reshape(md.n_esp, md.m)
    total = (dn * md.h).sum(axis=1)
    for el in ("N", "H"):
        a = np.array([0.0 if s == "E" else _atomos(s)[el] for s in md.nomes])
        escala = np.abs(a[:, None] * dn * md.h).sum()
        assert abs(a @ total) < 1e-12 * escala, el


def test_mistura_solta_pede_os_atomos_por_fundo(amonia):
    from modelo import ErroTransporte
    q, tab, transporte = amonia
    import copy
    t = copy.deepcopy(transporte)
    t["NH3"]["atomos_do_fundo"] = 2.0          # numa mistura, um número só não diz de qual fundo
    with pytest.raises(ErroTransporte, match=r"NH3.*atomos_do_fundo"):
        _modelo_amonia((q, tab, t))
