# SPDX-License-Identifier: GPL-3.0-or-later
"""Variants of casos/amonia_sun2022 for the sensitivity runs of article 2
(ferramentas/casos_sun.py): the N2-N2 V-V block corrected (A5, A7) and NH + H2 + M frozen at
373 K. The model reads zdplaskin_m.F90 and quimica.json, not kinet.inp, so the tests look at
the rates of the chemistry the model uses."""
import math
import sys

import numpy as np
import pytest

from conftest import RAIZ

# the mechanism of the Sun case is of other authors and is not distributed (casos/README.md);
# ferramentas/casos_sun.py is, and builds the variants from the user's copy
if not (RAIZ / "ferramentas/casos_sun.py").exists() or not (RAIZ / "casos/amonia_sun2022/kinet.inp").exists():
    pytest.skip("needs ferramentas/casos_sun.py and casos/amonia_sun2022, which are not distributed",
                allow_module_level=True)

sys.path.insert(0, str(RAIZ / "ferramentas"))
import casos_sun  # noqa: E402
from escada_vibracional import energia  # noqa: E402
from quimica_zdp import Quimica  # noqa: E402
from vv_sun import corrige as corrige_vv  # noqa: E402

BASE = RAIZ / "casos" / "amonia_sun2022"
IGUAIS = ("bolsigdb.dat", "bolsigdb.log", "transporte.json", "gas.json", "composicao.txt",
          "tabela_energia_tg.dat", "tabela_energia.dat", "tabela_eedf.dat")


def _k(caso, reacao, t):
    q = Quimica.de_json(RAIZ / "casos" / caso / "quimica.json")
    kb = np.zeros((1, 1000))
    k = q.coeficientes(kb, np.array([t]), np.array([t]), np.array([1.0]), np.array([4e18]))
    return float(k[q.reacoes.index(reacao), 0])


def test_receitas():
    texto = (BASE / "kinet.inp").read_text()
    assert casos_sun.monta("amonia_sun2022_vv") == corrige_vv(texto)
    congelado = casos_sun.monta("amonia_sun2022_nh_congelado")
    assert "(373.0d0/300.0d0)*exp(1700.0d0/373.0d0)" in congelado
    assert congelado.count("\n") == texto.count("\n")
    with pytest.raises(ValueError):
        casos_sun.monta("outro")


@pytest.mark.parametrize("caso", ["amonia_sun2022_vv", "amonia_sun2022_nh_congelado"])
def test_caso_no_disco_em_dia(caso):
    pasta = RAIZ / "casos" / caso
    assert (pasta / "kinet.inp").read_text() == casos_sun.monta(caso)
    for nome in IGUAIS:
        assert (pasta / nome).read_bytes() == (BASE / nome).read_bytes(), nome
    # the chemistry the model reads was regenerated from the new kinet.inp
    assert (pasta / "zdplaskin_m.F90").read_text() != (BASE / "zdplaskin_m.F90").read_text()


@pytest.mark.parametrize("t", [373.0, 773.0])
def test_vv_do_caso_tem_balanco_detalhado(t):
    direta, inversa = "N2(V1)+N2(V7)=>N2+N2(V8)", "N2(V8)+N2=>N2(V7)+N2(V1)"
    de = energia(1) + energia(7) - energia(0) - energia(8)
    kf, kr = _k("amonia_sun2022_vv", direta, t), _k("amonia_sun2022_vv", inversa, t)
    if de > 0:
        assert kr / kf == pytest.approx(math.exp(-de / t), rel=1e-4)
    else:
        assert kf / kr == pytest.approx(math.exp(de / t), rel=1e-4)
    # the base keeps the published block, the same coefficient both ways
    assert _k("amonia_sun2022", inversa, t) == pytest.approx(_k("amonia_sun2022", direta, t), rel=1e-12)


def test_nh_congelado_usa_o_coeficiente_de_373_k():
    r = "NH+H2+N2=>NH3+N2"
    assert _k("amonia_sun2022_nh_congelado", r, 773.0) == pytest.approx(_k("amonia_sun2022", r, 373.0), rel=1e-12)
