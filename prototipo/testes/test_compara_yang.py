# SPDX-License-Identifier: GPL-3.0-or-later
"""Comparação com Yang et al. 2022 (scripts/compara_yang.py): a mistura sai do
caso gravado no parametros.json, não do nome da execução."""
import json
import sys

from conftest import RAIZ

sys.path.insert(0, str(RAIZ / "scripts"))
import compara_yang  # noqa: E402


def _execucao(pasta, caso):
    pasta.mkdir()
    (pasta / "parametros.json").write_text(json.dumps({"argumentos": {"caso": caso}}))
    return pasta


def test_mistura_pelo_caso(tmp_path):
    # nomes que não dizem a mistura: a de H2 se chama YH2_n50 na mercury
    assert compara_yang.mistura(_execucao(tmp_path / "YH2_n50", "/m/casos/yang2022_1pcH2")) == "N2+1%H2"
    assert compara_yang.mistura(_execucao(tmp_path / "Y1pc_x", "/m/casos/yang2022_N2")) == "N2"


def test_so_os_instantes_ja_simulados():
    # execução parcial até 200 µs: as medidas de 300 µs em diante ficam de fora,
    # em vez de receber o último valor calculado
    import numpy as np
    t = np.arange(0, 201) * 1e-6
    assert compara_yang.simulados([104, 200, 301, 1500], t) == [104, 200]
