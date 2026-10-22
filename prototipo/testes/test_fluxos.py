# SPDX-License-Identifier: GPL-3.0-or-later
"""Flux analysis per reaction (scripts/fluxos.py): rates of each reaction rebuilt
from the recorded states, integrated over the gap and in time, and the fraction of
the production and consumption of each species per reaction."""
import subprocess
import sys

import numpy as np
import pytest

from conftest import RAIZ

sys.path.insert(0, str(RAIZ / "scripts"))
import fluxos  # noqa: E402


@pytest.fixture(scope="module")
def execucao(tmp_path_factory):
    d = tmp_path_factory.mktemp("fluxo") / "ar"
    subprocess.run([sys.executable, str(RAIZ / "prototipo" / "simula.py"), str(d), "--n-celulas", "12", "--agrupamento", "2.0",
                    "--v-pico", "5e3", "--espessura", "1.75e-3", "--energia", "--neutro", "--ne0", "1e9",
                    "--t-end", "2e-8", "--dt-saida", "1e-9"], check=True, capture_output=True,
                   env={"OPENBLAS_NUM_THREADS": "1", "PATH": ""})
    return d


def test_integral_das_taxas_bate_com_a_variacao_quimica(execucao):
    # the sum of the reactions that produce and consume Ar* is its chemical source;
    # with Ar* hardly leaving through the wall in 20 ns, it stays close to the
    # change of the Ar* integrated over the gap
    md, t, integrais = fluxos.taxas_integradas([execucao])
    q = md.q
    s = q.indice["AR*"]
    fonte = q.estequiometria[s] @ integrais
    estados = fluxos.estados([execucao], md)
    n0, n1 = estados[0][1], estados[-1][1]
    k = md.nomes.index("AR*")
    variacao = ((n1[k] - n0[k]) * md.h).sum()
    assert fonte == pytest.approx(variacao, rel=0.05)


def test_fracoes_somam_um_e_a_ionizacao_produz_o_ar_mais(execucao):
    md, _, integrais = fluxos.taxas_integradas([execucao])
    prod, cons = fluxos.contribuicoes(md.q, integrais, "AR^+")
    assert sum(f for _, f in prod) == pytest.approx(1.0, abs=1e-12)
    assert sum(f for _, f in cons) == pytest.approx(1.0, abs=1e-12)
    assert "AR->AR^+" in prod[0][0].upper().replace(" ", "")


def test_execucao_antiga_sem_opcoes_novas_e_lida(execucao, tmp_path):
    # parametros.json of a run made before an option existed (reflexao_eletrons came in 1.1):
    # the missing options take the default of simula.py
    import json
    import shutil
    velha = tmp_path / "velha"
    shutil.copytree(execucao, velha)
    p = json.loads((velha / "parametros.json").read_text())
    del p["argumentos"]["reflexao_eletrons"]
    (velha / "parametros.json").write_text(json.dumps(p))
    md = fluxos.modelo_da_execucao(velha)
    assert md.absorcao_eletrons == 1.0
