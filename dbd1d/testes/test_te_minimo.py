# SPDX-License-Identifier: GPL-3.0-or-later
"""Electron temperature in the rate coefficients held at or above T_g (--te-floor-tg).

The tables in T_g of BOLSIG+ have no superelastic collisions, so at the floor of the
table (E/N of 1e-4 Td) the electrons are colder than the gas: 306 K at T_g = 375 K and
341 K at 800 K in casos/amonia_sun2022. In the afterglow the mean energy sits at that
floor and the recombinations in (300/Te)^a run too fast, more at high T_g
(docs/campanha_amonia/onda3/PLANO.md, D5). The option keeps the old behaviour by default,
so the frozen runs of article 1 are reproduced."""
import json

import numpy as np
import pytest

from conftest import RAIZ, precisa_arquivo, quimica_do
from modelo import Modelo
from tabela import Tabela

SUN = RAIZ / "casos" / "amonia_sun2022"


def test_te_das_taxas_so_muda_com_a_opcao():
    tab = Tabela(RAIZ / "tabela_bolsig.dat")
    te, tg = np.array([306.0, 900.0, 773.0]), np.array([375.0, 773.0, 773.0])
    np.testing.assert_array_equal(Modelo(quimica_do(RAIZ), tab, n_celulas=4).te_das_taxas(te, tg), te)
    np.testing.assert_array_equal(
        Modelo(quimica_do(RAIZ), tab, n_celulas=4, te_minimo_tg=True).te_das_taxas(te, tg), [375.0, 900.0, 773.0])


def _te_no_rhs(opcoes):
    import simula
    a = simula.cria_parser().parse_args([str(RAIZ / "x"), "--caso", str(SUN), "--energy", "--cells", "8",
                                         "--t-gas", "773", "--t-wall", "773", "--pressure", "2e4",
                                         "--neutral", "--ne0", "1e9", *opcoes])
    md = simula.monta_modelo(a, SUN, None)
    vistos = []
    original = md.q.coeficientes

    def espia(k_bolsig, tgas, te, *resto, **kw):
        vistos.append((np.array(te, dtype=float), np.array(tgas, dtype=float)))
        return original(k_bolsig, tgas, te, *resto, **kw)

    md.q.coeficientes = espia
    md.rhs(0.0, md.estado_inicial(a.ne0, a.neutro))
    return md, vistos


@precisa_arquivo("casos/amonia_sun2022/tabela_energia_tg.dat")
def test_rhs_usa_te_ao_menos_tg_com_a_opcao():
    # without field the mean energy is at the floor of the table, colder than the gas at 773 K
    _, sem = _te_no_rhs([])
    assert sem and all((te < tg).any() for te, tg in sem)
    md, com = _te_no_rhs(["--te-floor-tg"])
    assert md.te_minimo_tg
    assert com and all((te >= tg).all() for te, tg in com)


def test_simula_grava_e_retomada_exige_a_mesma_opcao(tmp_path):
    import simula
    a = simula.cria_parser().parse_args([str(tmp_path / "a"), "--te-floor-tg"])
    assert a.te_minimo_tg is True
    assert simula.cria_parser().parse_args([str(tmp_path / "a")]).te_minimo_tg is False
    origem = tmp_path / "origem"
    origem.mkdir()
    (origem / "parametros.json").write_text(json.dumps({"argumentos": {"te_minimo_tg": False}}))
    with pytest.raises(SystemExit):
        p = simula.cria_parser()
        simula.estado_da_retomada(p, p.parse_args([str(tmp_path / "b"), "--te-floor-tg", "--restart", str(origem), "0"]),
                                  None)


@precisa_arquivo("casos/amonia_sun2022/tabela_energia_tg.dat")
def test_fluxos_reconstroi_as_taxas_com_o_mesmo_piso():
    # scripts/fluxos.py recomputes the rates of a run; it has to give them the T_e of the run
    import sys
    sys.path.insert(0, str(RAIZ / "scripts"))
    import fluxos
    md, vistos = _te_no_rhs(["--te-floor-tg"])
    vistos.clear()
    fluxos.taxas_no_gap(md, 0.0, md.estado_inicial(1e9, True))
    assert vistos and all((te >= tg).all() for te, tg in vistos)


def test_retomada_de_origem_sem_a_chave_e_de_rodada_sem_o_piso(tmp_path, capsys):
    # parametros.json written before the option has no te_minimo_tg: the origin ran without
    # the floor, so a restart with --te-floor-tg is refused and one without it is not
    import simula
    origem = tmp_path / "origem"
    origem.mkdir()
    (origem / "parametros.json").write_text(json.dumps({"argumentos": {}}))

    def retoma(*opcoes):
        p = simula.cria_parser()
        a = p.parse_args([str(tmp_path / "b"), *opcoes, "--restart", str(origem), "0"])
        try:
            simula.estado_da_retomada(p, a, None)
        except (SystemExit, Exception):
            pass
        return capsys.readouterr().err

    assert "--te-floor-tg: the origin used False" in retoma("--te-floor-tg")
    assert "--te-floor-tg" not in retoma()
