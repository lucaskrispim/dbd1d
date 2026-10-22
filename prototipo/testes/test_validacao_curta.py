# SPDX-License-Identifier: GPL-3.0-or-later
"""Short validation (validacao/curta.py): the reduced comparisons run and pass; the
complete set, a few minutes long, is marked lento."""
import json
import subprocess
import sys

import pytest

from conftest import RAIZ

CURTA = str(RAIZ / "validacao" / "curta.py")
AMBIENTE = {"OPENBLAS_NUM_THREADS": "1", "PATH": ""}


def _roda(tmp_path, *itens):
    return subprocess.run([sys.executable, CURTA, *itens, "--output", str(tmp_path)], capture_output=True, text=True,
                          env=AMBIENTE)


def test_itens_rapidos_passam(tmp_path):
    r = _roda(tmp_path, "townsend", "enxame")
    assert r.returncode == 0, r.stdout + r.stderr
    linhas = json.loads((tmp_path / "curta.json").read_text())
    assert [(l["item"], l["ok"]) for l in linhas] == [("townsend", True), ("townsend", True), ("enxame", True)]


def test_item_desconhecido_e_recusado(tmp_path):
    r = _roda(tmp_path, "xyz")
    assert r.returncode != 0 and "unknown items" in r.stderr


@pytest.mark.lento
def test_validacao_curta_completa(tmp_path):
    r = _roda(tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    linhas = json.loads((tmp_path / "curta.json").read_text())
    assert {l["item"] for l in linhas} == {"stankov", "loffhagen", "townsend", "enxame"}
