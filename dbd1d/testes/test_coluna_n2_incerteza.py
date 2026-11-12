# SPDX-License-Identifier: GPL-3.0-or-later
"""Uncertainty band of the model in scripts/coluna_n2.py: one variable at a time (E/N, n_e,
T_g by their measured bars) gives a band narrower than the worst joint case; the eight corners
are computed too (ERROS.md A3)."""
import itertools
import sys

from conftest import RAIZ

sys.path.insert(0, str(RAIZ / "scripts"))
import coluna_n2  # noqa: E402

COND = dict(en=100.0, en_err=10.0, ne=1e10, ne_err=2e9, tg=400.0, tg_err=20.0)


def test_perturbacoes_um_a_um_e_cantos():
    p = coluna_n2.perturbacoes(COND)
    assert len(p["eixos"]) == 6 and len(p["cantos"]) == 8
    cantos = {(en, ne, tg) for en, ne, tg in itertools.product((90.0, 110.0), (8e9, 1.2e10), (380.0, 420.0))}
    assert set(p["cantos"]) == cantos
    # each axis point moves one variable only
    for en, ne, tg in p["eixos"]:
        assert sum((en != 100.0, ne != 1e10, tg != 400.0)) == 1
