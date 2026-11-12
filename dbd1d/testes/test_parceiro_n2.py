# SPDX-License-Identifier: GPL-3.0-or-later
"""N2 as an unchanged collision partner (V-T, quenching, three-body) is N2(X) in all its
vibrational levels, the M = N2 of Gordiets et al. 1995 (sec. II-B), not the species N2,
which in a chemistry with N2(vk) is v = 0 only (docs/auditoria/2026-11-10/ERROS.md E14).
The rule lives in Quimica.taxas, so the column, the pulsed jet, the 1D model and every
script that computes rates from the chemistry use the same partner."""
import numpy as np
import pytest

from conftest import RAIZ, precisa_arquivo, quimica_do
from modelo import Modelo
from quimica_zdp import Quimica
from tabela import Tabela

CASO = RAIZ / "casos" / "n2_amonia" / "quimica.json"


def _estado(q, semente=4):
    r = np.random.default_rng(semente)
    dens = 10.0 ** r.uniform(8, 16, (len(q.especies), 3))
    k = 10.0 ** r.uniform(-16, -9, (len(q.reacoes), 3))
    return dens, k


@precisa_arquivo("casos/n2_amonia/quimica.json")
def test_taxas_com_parceiro_em_todos_os_niveis():
    antigo, q = Quimica.de_json(CASO), Quimica.de_json(CASO)
    q.usa_parceiro_total()
    dens, k = _estado(q)
    base, nova = antigo.taxas(dens, k), q.taxas(dens, k)
    n2 = q.indice["N2"]
    total = dens[n2] + sum(dens[i] for s, i in q.indice.items() if s.startswith("N2(V"))
    for reacao, fator in (("N2(A3)+N2=>N2+N2", total / dens[n2]), ("N2(V1)+N2=>N2+N2", total / dens[n2]),
                          ("N2(B3)+N2=>N2(A3)+N2(V6)", 1.0), ("bolsig:N2->N2(V1)", 1.0)):
        r = q.reacoes.index(reacao)
        np.testing.assert_allclose(nova[r], base[r] * fator, rtol=1e-12)


def test_quimica_sem_n2_nao_muda():
    q = quimica_do(RAIZ)
    dens, k = _estado(q)
    base = q.taxas(dens, k)
    q.usa_parceiro_total()
    np.testing.assert_array_equal(q.taxas(dens, k), base)


def test_modelo_usa_o_parceiro_total_so_com_gas_solto():
    # with the background held fixed (the default), the species N2 is the whole N2 of the gas
    # (densidades_completas: fraction times n_gas), so adding the N2(vk) would count them twice
    # (ERROS.md C1); with --free-gas, N2 evolves and is v = 0
    tab = Tabela(RAIZ / "tabela_bolsig.dat")
    assert not Modelo(quimica_do(RAIZ), tab, n_celulas=4).q.parceiro_total
    assert Modelo(quimica_do(RAIZ), tab, n_celulas=4, gas_solto=True).q.parceiro_total
    assert not Modelo(quimica_do(RAIZ), tab, n_celulas=4, gas_solto=True, parceiro_n2="v0").q.parceiro_total
    with pytest.raises(ValueError, match="parceiro_n2"):
        Modelo(quimica_do(RAIZ), tab, n_celulas=4, parceiro_n2="outro")


def test_v0_desliga_o_parceiro_de_uma_quimica_ja_usada():
    # a chemistry object reused after a model with the total partner (ERROS.md C3)
    q = quimica_do(RAIZ)
    q.usa_parceiro_total()
    assert not Modelo(q, Tabela(RAIZ / "tabela_bolsig.dat"), n_celulas=4, gas_solto=True, parceiro_n2="v0").q.parceiro_total
    q.usa_parceiro_total(False)
    assert not q.parceiro_total
