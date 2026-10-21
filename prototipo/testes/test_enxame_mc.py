# SPDX-License-Identifier: GPL-3.0-or-later
"""Reading of the LXCat format by the swarm Monte Carlo (ferramentas/enxame_mc.py):
threshold glued to the slash ("2.9e-1/threshold energy") and ATTACHMENT without a
parameter line, as in the bolsigdb.dat of the ammonia case of Sun et al. 2022."""
import sys

import pytest

from conftest import RAIZ

sys.path.insert(0, str(RAIZ / "ferramentas"))
import enxame_mc  # noqa: E402


def test_le_o_bolsigdb_da_amonia():
    s = enxame_mc.le_secoes(RAIZ / "casos" / "amonia_sun2022" / "bolsigdb.dat")
    assert s[("EXCITATION", "N2 -> N2(v1)")][0] == pytest.approx(0.29)
    assert s[("EXCITATION", "H2 -> H2(v1)")][0] == pytest.approx(0.516)
    assert s[("ATTACHMENT", "H2 -> H^-")][0] == 0.0
    assert len(s[("ATTACHMENT", "H2 -> H^-")][1]) > 2


def test_le_o_bolsigdb_do_argonio_como_antes():
    s = enxame_mc.le_secoes(RAIZ / "bolsigdb.dat")
    assert s[("IONIZATION", "Ar -> Ar^+")][0] == 15.8
