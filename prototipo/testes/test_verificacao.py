# SPDX-License-Identifier: GPL-3.0-or-later
"""Verification tests that were missing to reach the level of the Fortran code (VT7 to
VT12 of docs/lacunas.md): global energy balance of the gas, conduction and energy
table against independent answers, the DAE integrator against an exact solution,
robustness at zero field with a dense plasma (N21), free gas and the modes of
simula.py that had no test."""
import subprocess
import sys

import numpy as np
import pytest

import bdf_dae
from bdf_dae import BDFMassa
from conftest import PROTOTIPO, RAIZ, quimica_do
from modelo import Modelo, condutividade_termica
from quimica_zdp import Quimica
from tabela import PEL, Tabela, TabelaEnergia

KB = 1.380649e-23
QE = 1.602176634e-19


@pytest.fixture(scope="module")
def pecas():
    return quimica_do(RAIZ), Tabela(RAIZ / "tabela_bolsig.dat")


# ------------------------------------------------------------- gas (VT7, VT8)
def test_balanco_global_de_energia_do_gas(pecas):
    # sum of (3/2) N k_B dT/dt h over the cells = sum of (Q_el + Q_J) h minus the
    # heat leaving through the two walls, k_f (T - T_wall)/(h/2); to 1e-12
    md = Modelo(*pecas, n_celulas=30, agrupamento=1.5, espessura=1e-3, v_pico=3e3, aquecimento=True)
    g = np.random.default_rng(8)
    n = 10.0 ** g.uniform(9, 12, size=(md.n_esp, md.m))
    t_gas = 300.0 + 100.0 * g.uniform(size=md.m)
    y = np.concatenate([n.ravel(), t_gas, [1e-7, -2e-7]])
    t = 2e-8
    dtdt = md.rhs(t, y)[md.i_t * md.m:(md.i_t + 1) * md.m]
    _, campo = md.campo(t, n, 1e-7, -2e-7)
    cel = md.campo_nas_celulas(campo)
    mob, _, _, _, e_n, _ = md.coeficientes(cel, n, None, t_gas)
    q_el = QE * md.tab(e_n)[:, PEL] * md.n_gas * n[md.i_e] * 1e6
    q_j = sum(QE * n[k] * 1e6 * mob[k] * cel ** 2 for k in md.ions)
    k_par = condutividade_termica(np.array(300.0))
    sai = 0.0
    for c, meia in ((0, md.d[0]), (md.m - 1, md.d[-1])):
        kc = condutividade_termica(t_gas[c])
        sai += 2 * k_par * kc / (k_par + kc) * (t_gas[c] - 300.0) / meia
    n_pes = md.n_gas + n[[k for k in range(md.n_esp) if k != md.i_e]].sum(axis=0)
    armazenado = np.sum(1.5 * KB * n_pes * 1e6 * dtdt * md.h)
    fontes = np.sum((q_el + q_j) * md.h)
    assert armazenado == pytest.approx(fontes - sai, rel=1e-12, abs=0)


def test_joule_dos_ions_independe_do_sentido_do_campo(pecas):
    # Q_J = e n_i mu_i E^2: fields +E and -E heat equally
    md = Modelo(*pecas, n_celulas=10, aquecimento=True)
    n = np.zeros((md.n_esp, md.m))
    n[md.nomes.index("AR^+")] = 1e12
    t_gas = np.full(md.m, 300.0)
    d = []
    for sinal in (1.0, -1.0):
        cel = np.full(md.m, sinal * 2e5)
        mob, _, _, _, _, extra = md.coeficientes(cel, n, None, t_gas)
        d.append(md.d_temperatura(n, t_gas, cel, mob, extra))
    assert np.all(d[0] > 0.0)
    np.testing.assert_allclose(d[1], d[0], rtol=1e-14)


def test_conducao_de_segunda_ordem_na_malha_agrupada(pecas):
    # the same manufactured solution of the uniform mesh (test_aquecimento), now
    # with clustering 2: the distance between centres differs from the cell width
    erros = []
    for m in (40, 80, 160):
        md = Modelo(*pecas, n_celulas=m, v_pico=0.0, agrupamento=2.0, aquecimento=True)
        a_, L = 50.0, md.L
        t_gas = 300.0 + a_ * np.sin(np.pi * md.xc / L)
        y = np.concatenate([np.zeros(md.n_esp * m), t_gas, [0.0, 0.0]])
        dtdt = md.rhs(0.0, y)[md.i_t * m:(md.i_t + 1) * m]
        d1 = a_ * np.pi / L * np.cos(np.pi * md.xc / L)
        d2 = -a_ * (np.pi / L) ** 2 * np.sin(np.pi * md.xc / L)
        kap = (0.0307313 * t_gas + 10.006) * 1e-3
        exato = (0.0307313e-3 * d1 ** 2 + kap * d2) / (1.5 * KB * md.n_gas * 1e6)
        erros.append(np.max(np.abs(dtdt - exato)[1:-1]) / np.max(np.abs(exato)))
    ordens = np.log2(np.array(erros[:-1]) / np.array(erros[1:]))
    # inner cells: in the wall cell the local truncation error of the half cell is
    # first order (the solution remains second order); the wall closure has its
    # own test in test_aquecimento.py
    assert np.all(np.abs(ordens - 2.0) < 0.1), ordens


# ----------------------------------------------------- energy table (VT9)
def tabela_sintetica(caminho, funcao):
    """tabela_energia.dat with eps from 0.04 to 40 eV and columns given by funcao(eps)."""
    eps = 0.04 * 10.0 ** (np.arange(31) / 10.0)
    with open(caminho, "w") as f:
        f.write("# sintetica\n# colunas\n")
        for e in eps:
            f.write("".join(f" {v:24.16E}" for v in [e * 10.0, e, *funcao(e)]) + "\n")


def test_tabela_de_energia_exata_para_leis_de_potencia(tmp_path):
    # the interpolation is linear in (ln eps, ln value): power laws come out exact
    # outside the nodes
    potencias = np.arange(1, 13) * 0.25
    tabela_sintetica(tmp_path / "t.dat", lambda e: e ** potencias)
    tab = TabelaEnergia(tmp_path / "t.dat")
    eps = np.array([0.0517, 0.331, 2.77, 19.3])
    np.testing.assert_allclose(tab(eps)[:, 1:], eps[:, None] ** potencias, rtol=1e-12)


def test_tabela_de_energia_de_segunda_ordem(tmp_path):
    # smooth function as that of teste_tabela.f90: error between nodes of order 2 in ln eps
    def g(e):
        return e ** 1.5 * np.exp(-3.0 / e)

    erros = []
    for por_decada in (20, 40, 80, 160):
        eps = 0.04 * 10.0 ** (np.arange(3 * por_decada + 1) / por_decada)
        with open(tmp_path / "t.dat", "w") as arq:
            arq.write("#\n#\n")
            for e in eps:
                arq.write("".join(f" {v:24.16E}" for v in [e * 10.0, e, *np.full(12, g(e))]) + "\n")
        tab = TabelaEnergia(tmp_path / "t.dat")
        meios = np.sqrt(eps[1:] * eps[:-1])
        erros.append(np.max(np.abs(tab(meios)[:, 1] / g(meios) - 1.0)))
    ordens = np.log2(np.array(erros[:-1]) / np.array(erros[1:]))
    assert np.all(np.diff(ordens) > 0.0) and abs(ordens[-1] - 2.0) < 0.05, ordens


# ------------------------------------------------------------ integrador (VT10)
def test_bdf_com_massa_numa_dae_linear_exata():
    # y' = -y + z, 0 = z - y/2: y = exp(-t/2), z = y/2
    def f(t, w):
        return np.array([-w[0] + w[1], w[1] - 0.5 * w[0]])

    def jac(t, w):
        return np.array([[-1.0, 1.0], [-0.5, 1.0]])

    s = BDFMassa(f, 0.0, np.array([1.0, 0.5]), 4.0, np.array([1.0, 0.0]), rtol=1e-9,
                 atol=np.array([1e-12, 1e30]), jac=jac)
    while s.status == "running":
        s.step()
    assert s.status == "finished"
    assert s.y[0] == pytest.approx(np.exp(-2.0), rel=1e-7, abs=0)
    assert s.y[1] == pytest.approx(0.5 * s.y[0], rel=1e-12)


def test_bdf_com_massa_corta_o_passo_quando_o_newton_falha():
    # the exact linear DAE with f undefined (NaN) more than 0.05 beyond the last accepted
    # step: Newton fails even with the current Jacobian, the step is halved
    # until it fits, and the solution remains exact
    estado = {"t": 0.0}

    def f(t, w):
        if t - estado["t"] > 0.05:
            return np.full(2, np.nan)
        return np.array([-w[0] + w[1], w[1] - 0.5 * w[0]])

    def jac(t, w):
        return np.array([[-1.0, 1.0], [-0.5, 1.0]])

    s = BDFMassa(f, 0.0, np.array([1.0, 0.5]), 2.0, np.array([1.0, 0.0]), rtol=1e-9,
                 atol=np.array([1e-12, 1e30]), jac=jac)
    while s.status == "running":
        s.step()
        estado["t"] = s.t
    assert s.status == "finished" and s.contador["cortes"] > 0
    assert s.y[0] == pytest.approx(np.exp(-1.0), rel=1e-7, abs=0)


# ---------------------------------------------------------- zero field (VT11)
@pytest.mark.parametrize("energia", [False, True])
def test_celula_do_n21_integrada_sem_densidade_negativa(pecas, energia):
    # zero field in a dense plasma (the cell that brought BOLSIG+ down in the Fortran code,
    # N21: n_e = 2.9e14, Ar* = 6.6e14): integrated for 1 ns, as V55 and V57 of the
    # Fortran code, it stays finite and without negative density beyond the tolerance
    from scipy.integrate import solve_ivp
    from jacobiano import ModeloRapido
    te = TabelaEnergia(RAIZ / "tabela_energia.dat") if energia else None
    md = ModeloRapido(*pecas, n_celulas=6, v_pico=0.0, tabela_energia=te)
    y = md.estado_inicial(2.9e14, neutro=True)
    z = y[:-2].reshape(md.n_var, md.m)
    z[md.nomes.index("AR*")] = 6.6e14
    atol = np.concatenate([np.full(md.n_var * md.m, 1e-4), [1e-12, 1e-12]])
    sol = solve_ivp(md.rhs, (0.0, 1e-9), y, method="BDF", rtol=1e-6, atol=atol,
                    jac=lambda t, w: md.jacobiano(t, w, 1e-4))
    assert sol.success and np.all(np.isfinite(sol.y))
    assert sol.y[:-2].min() > -1e-2


# ------------------------------------------------------------ free gas (VT12)
def test_eletrons_com_o_n_local_no_gas_solto(pecas):
    # mu_e = (mu N)/N and D_e = (D N)/N with the N of the cell
    md = Modelo(*pecas, n_celulas=4, gas_solto=True)
    n = np.zeros((md.n_esp, md.m))
    n[md.nomes.index("AR")] = md.n_gas
    n[md.nomes.index("AR")][2] = 0.5 * md.n_gas
    n[md.nomes.index("AR^+")] = 1e17          # ions do not enter the N of the electrons
    e_n = 50.0
    n_local = n[md.nomes.index("AR")]
    campo = e_n * 1e-21 * n_local * 1e6                    # same E/N in every cell
    mob, dif, *_ = md.coeficientes(campo, n)
    assert mob[md.i_e, 2] == pytest.approx(2.0 * mob[md.i_e, 0], rel=1e-12)
    assert dif[md.i_e, 2] == pytest.approx(2.0 * dif[md.i_e, 0], rel=1e-12)


def roda_simula(destino, *extra):
    r = subprocess.run([sys.executable, str(PROTOTIPO / "simula.py"), str(destino), "--v-pico", "5e3",
                        "--n-celulas", "12", "--espessura", "1.75e-3", "--ne0", "1e9", "--neutro",
                        "--t-end", "2e-9", "--dt-saida", "1e-9", "--rtol", "1e-6", *extra],
                       capture_output=True, text=True, env={"OPENBLAS_NUM_THREADS": "1", "PATH": ""})
    assert r.returncode == 0, r.stdout + r.stderr
    return np.loadtxt(destino / "superficie.dat", skiprows=1)


@pytest.mark.parametrize("extra, rtol", [(["--jac", "estruturado"], 1e-4), (["--jac", "fd", "--adaptativa"], 5e-2),
                                         (["--gas-solto"], 1e-4)])
def test_modos_do_simula_concordam_com_o_padrao(tmp_path, extra, rtol):
    # same case in the default mode (DAE, fixed gas): structured Jacobian
    # equal; free gas equal with low ionization; adaptive mesh close
    base = roda_simula(tmp_path / "base")
    outro = roda_simula(tmp_path / "outro", *extra)
    np.testing.assert_allclose(outro[:, 2:6], base[:, 2:6], rtol=rtol, atol=rtol * np.abs(base[:, 2:6]).max())


def test_gas_solto_com_aquecimento_e_energia_roda(tmp_path):
    roda_simula(tmp_path / "x", "--gas-solto", "--aquecimento", "--energia")
