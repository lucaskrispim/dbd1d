# SPDX-License-Identifier: GPL-3.0-or-later
"""Continuous reference of the Townsend criterion (validacao/townsend.py), with
synthetic coefficients: without diffusion and without thermal flux it reduces to
the classical criterion gamma (exp(alpha d) - 1) = 1; the thermal reabsorption at
the cathode requires more multiplication (larger gap) to break down."""
import sys

import numpy as np
import pytest
from scipy.optimize import brentq

from conftest import RAIZ

sys.path.insert(0, str(RAIZ / "validacao"))
import townsend  # noqa: E402

W, NU, GAMA = 1.0e5, 2.0e9, 0.05          # m/s, 1/s; alfa = 2e4 1/m
ALFA = NU / W
GAP_CLASSICO = np.log1p(1.0 / GAMA) / ALFA


def gap_de_ruptura(d_e, v_th, a_e=1.0):
    f = lambda g: townsend.determinante(W, d_e, NU, v_th, GAMA, g, a_e)  # noqa: E731
    return brentq(f, 0.3 * GAP_CLASSICO, 5.0 * GAP_CLASSICO, xtol=1e-12)


def test_recai_no_criterio_classico_sem_difusao_nem_fluxo_termico():
    assert gap_de_ruptura(d_e=1e-9, v_th=0.0) == pytest.approx(GAP_CLASSICO, rel=1e-4)


def test_difusao_pequena_muda_pouco_e_fluxo_termico_pede_mais_multiplicacao():
    g_dif = gap_de_ruptura(d_e=0.3, v_th=0.0)
    assert abs(g_dif / GAP_CLASSICO - 1.0) < 0.1
    g_ter = gap_de_ruptura(d_e=0.3, v_th=1.0e6)
    assert g_ter > 1.05 * g_dif
    # with reflection, fewer electrons return to the cathode and breakdown is easier
    assert gap_de_ruptura(d_e=0.3, v_th=1.0e6, a_e=0.5) < g_ter


def test_raizes_satisfazem_a_equacao_caracteristica():
    for lam in townsend.raizes(W, 0.3, NU):
        assert 0.3 * lam * lam - W * lam + NU == pytest.approx(0.0, abs=1e-6 * NU)


@pytest.mark.parametrize("gama, reflexao, sinal", [(0.05, 0.0, 1.0), (0.05, 0.5, 1.0), (0.05, 0.0, -1.0)],
                         ids=["catodo_direito", "catodo_direito_reflexao", "catodo_esquerdo"])
def test_ruptura_do_simulador_bate_com_a_referencia(tmp_path, gama, reflexao, sinal):
    # end to end: argon at 100 Torr, gap of 1.5 mm (p d = 15 Torr cm),
    # metal electrodes, local field. With 200 cells the discrete breakdown lies
    # 0.2 to 0.3% below the continuous reference (mesh study from 100 to 800
    # cells in validacao/townsend.py); the tolerance is 0.5%
    gap, pressao = 1.5e-3, 100.0 * townsend.TORR
    v_ref, _ = townsend.ruptura_referencia(gap, pressao, gama, r=reflexao)
    vs = v_ref * np.array([0.99, 1.01])
    taxas = []
    for v in vs:
        # negative voltage on the left electrode: the cathode is the left wall
        d = tmp_path / f"v{v:.1f}"
        townsend.roda(d, gap, pressao, gama, sinal * v, 200, sys.executable, 3e-5, reflexao)
        taxas.append(townsend.taxa_de_crescimento(d))
    assert taxas[0] < 0.0 < taxas[1]          # below and above breakdown: the growth rate changes sign
    v_b = vs[0] - taxas[0] * (vs[1] - vs[0]) / (taxas[1] - taxas[0])
    assert v_b == pytest.approx(v_ref, rel=5e-3)
