# SPDX-License-Identifier: GPL-3.0-or-later
"""Charge and atoms conserved by every reaction of every exported chemistry present
(quimica.json of the root and of each case folder): the stoichiometry of each reaction
moves no charge and no atom of N, H, O or Ar."""
import json
import re

import numpy as np
import pytest

from conftest import RAIZ

QUIMICAS = sorted([RAIZ / "quimica.json", *RAIZ.glob("casos/*/quimica.json")])
ELEMENTOS = re.compile(r"(AR|N|H|O)(\d*)")


def atomos(especie):
    """Atoms of each element in a species name, case-insensitive (AR2^+ -> {AR: 2},
    N2(A3) -> {N: 2}, NH3 -> {N: 1, H: 3}, the cluster O2^+N2 -> {O: 2, N: 2}); the
    electron has none."""
    nome = re.sub(r"\^[+-]+", "", re.sub(r"\([^)]*\)", "", especie.upper()))
    nome = re.sub(r"[^A-Z0-9]", "", nome)
    if nome == "E":
        return {}
    conta, resto = {}, nome
    for el, n in ELEMENTOS.findall(nome):
        conta[el] = conta.get(el, 0) + (int(n) if n else 1)
        resto = resto.replace(el + n, "", 1)
    assert resto == "", f"species {especie} with an unknown element"
    return conta


def test_atomos():
    assert atomos("AR2^+") == {"AR": 2} and atomos("AR*") == {"AR": 1}
    assert atomos("N2(A3)") == {"N": 2} and atomos("N4^+") == {"N": 4}
    assert atomos("NH3") == {"N": 1, "H": 3} and atomos("N2O5") == {"N": 2, "O": 5}
    assert atomos("O2^+N2") == {"O": 2, "N": 2} and atomos("N(2D)") == {"N": 1}
    assert atomos("E") == {}


@pytest.mark.parametrize("arquivo", QUIMICAS, ids=[str(q.relative_to(RAIZ)) for q in QUIMICAS])
def test_carga_e_atomos_conservados_por_reacao(arquivo):
    q = json.loads(arquivo.read_text())
    est = np.asarray(q["estequiometria"], dtype=float)          # (species, reactions)
    especies = q["especies"]
    assert est.shape == (len(especies), len(q["reacoes"]))
    assert np.all(np.asarray(q["carga"], dtype=float) @ est == 0), "charge"
    elementos = sorted({el for e in especies for el in atomos(e)})
    for el in elementos:
        conta = np.array([atomos(e).get(el, 0) for e in especies], dtype=float)
        falha = np.nonzero(conta @ est)[0]
        assert falha.size == 0, f"{el} not conserved in " + "; ".join(q["reacoes"][i] for i in falha[:5])
