# SPDX-License-Identifier: GPL-3.0-or-later
"""Reactor geometry: dielectric barriers with different thickness and permittivity
on each electrode (Sun et al. 2022: 1 and 2 mm of quartz) and the gas pressure as
a parameter (10 to 50 kPa)."""
import json
import subprocess
import sys

import numpy as np
import pytest
from scipy.linalg import solve_banded

from conftest import PROTOTIPO, RAIZ, quimica_do
from jacobiano import ModeloRapido
from modelo import Modelo
from quimica_zdp import Quimica
from tabela import TabelaTg


@pytest.fixture(scope="module")
def pecas():
    return quimica_do(RAIZ), TabelaTg(RAIZ / "tabela_energia_tg.dat")


GEOM = dict(n_celulas=40, gap=5e-3, espessura=1e-3, eps_dieletrico=4.3, espessura_dir=2e-3,
            eps_dieletrico_dir=3.8, agrupamento=1.5)


def campo_vazio(v):
    # capacitor in series: the three layers have the same D = eps0 eps E
    return v / (5e-3 + 1e-3 / 4.3 + 2e-3 / 3.8)


@pytest.mark.parametrize("classe", [Modelo, ModeloRapido])
def test_barreiras_diferentes_no_poisson_sem_carga(pecas, classe):
    md = classe(*pecas, **GEOM)
    phi, campo = md.poisson(np.zeros(md.m), 0.0, 0.0, 3e3)
    np.testing.assert_allclose(campo, campo_vazio(3e3), rtol=1e-12)
    # drop across the barriers: V - phi_0 = E l1 eps_gas/eps1 and phi_n = E l2 eps_gas/eps2
    assert 3e3 - phi[0] == pytest.approx(campo_vazio(3e3) * 1e-3 / 4.3, rel=1e-12)
    assert phi[-1] == pytest.approx(campo_vazio(3e3) * 2e-3 / 3.8, rel=1e-12)


def test_formulacao_dae_com_barreiras_diferentes(pecas):
    # the matrix and the right-hand side used by the DAE give the same potential
    md = ModeloRapido(*pecas, **GEOM)
    rho = np.linspace(-1.0, 2.0, md.m) * 1e-3
    phi, _ = md.poisson(rho, 2e-6, -1e-6, 3e3)
    b = md._rhs_poisson(rho, 2e-6, -1e-6, 3e3)
    np.testing.assert_allclose(solve_banded((1, 1), md.ab, b), phi, rtol=1e-12)


def test_padrao_e_simetrico_bit_a_bit(pecas):
    sim = Modelo(*pecas, n_celulas=20, espessura=1.75e-3, eps_dieletrico=4.8)
    exp = Modelo(*pecas, n_celulas=20, espessura=1.75e-3, eps_dieletrico=4.8, espessura_dir=1.75e-3,
                 eps_dieletrico_dir=4.8)
    rho = np.linspace(-1.0, 2.0, 20) * 1e-3
    assert np.array_equal(sim.poisson(rho, 1e-6, 2e-6, 5e3)[0], exp.poisson(rho, 1e-6, 2e-6, 5e3)[0])


def test_barreira_so_de_um_lado_e_recusada(pecas):
    with pytest.raises(ValueError, match="espessura"):
        Modelo(*pecas, n_celulas=10, espessura=1e-3, espessura_dir=0.0)


def _roda(destino, *extra):
    return subprocess.run([sys.executable, str(PROTOTIPO / "simula.py"), str(destino), "--n-celulas", "10",
                           "--t-end", "1e-9", "--dt-saida", "1e-9", *extra],
                          capture_output=True, text=True, env={"OPENBLAS_NUM_THREADS": "1", "PATH": ""})


def test_pressao_e_barreiras_pela_linha_de_comando(tmp_path):
    r = _roda(tmp_path / "a", "--pressao", "2e4", "--t-gas", "573", "--gap", "5e-3", "--espessura", "1e-3",
              "--espessura-dir", "2e-3", "--eps-dieletrico", "4.3", "--eps-dieletrico-dir", "4.3")
    assert r.returncode == 0, r.stderr
    par = json.loads((tmp_path / "a" / "parametros.json").read_text())
    assert par["modelo"]["n_gas"] == pytest.approx(2e4 / (1.380649e-23 * 573.0) * 1e-6)
    assert par["argumentos"]["espessura_dir"] == 2e-3


@pytest.mark.parametrize("extra, mensagem", [
    (["--pressure", "0"], "--pressure"),
    (["--thickness", "1e-3", "--thickness-right", "-1"], "--thickness-right"),
    (["--thickness-right", "1e-3"], "--thickness-right"),
    (["--thickness", "1e-3", "--eps-dielectric-right", "0.5"], "--eps-dielectric-right"),
])
def test_geometria_invalida_e_recusada(tmp_path, extra, mensagem):
    r = _roda(tmp_path / "x", *extra)
    erro = [l for l in (r.stderr + r.stdout).splitlines() if "error:" in l]
    assert r.returncode != 0 and erro and mensagem in erro[-1], r.stderr
