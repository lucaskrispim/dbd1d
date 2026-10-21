# SPDX-License-Identifier: GPL-3.0-or-later
"""Comparison with Yang et al. 2022 (scripts/compara_yang.py): the mixture comes
from the case recorded in parametros.json, not from the name of the run."""
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
    # names that do not tell the mixture: the H2 one is called YH2_n50 on mercury
    assert compara_yang.mistura(_execucao(tmp_path / "YH2_n50", "/m/casos/yang2022_1pcH2")) == "N2+1%H2"
    assert compara_yang.mistura(_execucao(tmp_path / "Y1pc_x", "/m/casos/yang2022_N2")) == "N2"


def test_so_os_instantes_ja_simulados():
    # partial run up to 200 µs: the measurements from 300 µs on are left out,
    # instead of receiving the last computed value
    import numpy as np
    t = np.arange(0, 201) * 1e-6
    assert compara_yang.simulados([104, 200, 301, 1500], t) == [104, 200]
