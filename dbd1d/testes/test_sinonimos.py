# SPDX-License-Identifier: GPL-3.0-or-later
"""Every option of simula.py: each of its names (English, the Portuguese alias of earlier
versions and the earlier English ones) gives the same destination and the same value."""
import pathlib
import sys

import pytest

from conftest import RAIZ

sys.path.insert(0, str(RAIZ / "dbd1d"))
from simula import cria_parser  # noqa: E402

# a value of each kind of option, written as on the command line
VALORES = {"forma": ["pulse"], "jac": ["fd"], "parceiro_n2": ["v0"], "acelera": ["4", "50"], "retoma": ["/m/origem", "1e-6"]}


def _valor(acao):
    if acao.dest in VALORES:
        return VALORES[acao.dest]
    if acao.nargs == 0:
        return []
    if acao.type is int:
        return ["7"]
    if acao.type is float:
        return ["0.25"]
    if acao.type is pathlib.Path:
        return ["/m/arquivo"]
    raise AssertionError(f"option without a test value: {acao.dest}")


ACOES = [a for a in cria_parser()._actions if a.option_strings and a.dest != "help"]


@pytest.mark.parametrize("acao", ACOES, ids=[a.dest for a in ACOES])
def test_todos_os_nomes_dao_o_mesmo_valor(acao):
    padrao = vars(cria_parser().parse_args(["/m/x"]))[acao.dest]
    lidos = [vars(cria_parser().parse_args(["/m/x", nome, *_valor(acao)]))[acao.dest]
             for nome in acao.option_strings]
    assert all(v == lidos[0] for v in lidos), acao.option_strings
    assert lidos[0] != padrao, acao.dest


def test_o_destino_e_o_nome_portugues():
    # the destination (key of parametros.json) is the Portuguese name when there is one
    for a in ACOES:
        nomes = [n[2:].replace("-", "_") for n in a.option_strings]
        assert a.dest in nomes, a.option_strings


def test_valores_internos_e_ingleses_das_escolhas():
    p = cria_parser()
    for ingles, interno in (("sine", "seno"), ("step", "degrau"), ("pulse", "pulso"), ("table", "tabela")):
        assert p.parse_args(["/m/x", "--waveform", ingles]).forma == interno
        assert p.parse_args(["/m/x", "--forma", interno]).forma == interno
    for ingles, interno in (("dae", "dae"), ("structured", "estruturado"), ("fd", "fd")):
        assert p.parse_args(["/m/x", "--jac", ingles]).jac == interno


def test_cada_nome_e_cada_destino_numa_opcao_so():
    # an alias added as a separate option, with its own destination, would be read into another
    # key of parametros.json: each name and each destination belong to one option only
    nomes = [n for a in ACOES for n in a.option_strings]
    destinos = [a.dest for a in ACOES]
    assert len(nomes) == len(set(nomes))
    assert len(destinos) == len(set(destinos))
