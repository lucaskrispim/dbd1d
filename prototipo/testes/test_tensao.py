# SPDX-License-Identifier: GPL-3.0-or-later
"""Shape of the applied voltage: sine (default), step with a linear rise (DC
voltage) and repeated trapezoidal pulse, for short convergence studies."""
import subprocess
import sys

import numpy as np
import pytest

from conftest import PROTOTIPO, RAIZ, quimica_do
from modelo import Modelo
from quimica_zdp import Quimica
from tabela import TabelaTg


@pytest.fixture(scope="module")
def pecas():
    return quimica_do(RAIZ), TabelaTg(RAIZ / "tabela_energia_tg.dat")


def test_seno_e_o_padrao(pecas):
    md = Modelo(*pecas, n_celulas=4, v_pico=2e3, periodo=1e-7)
    t = np.array([0.0, 1e-8, 2.5e-8, 7.3e-8])
    assert np.array_equal([md.tensao(x) for x in t], 2e3 * np.sin(2 * np.pi / 1e-7 * t))


def test_degrau_sobe_linear_e_fica(pecas):
    md = Modelo(*pecas, n_celulas=4, v_pico=3e3, forma="degrau", subida=2e-9)
    assert md.tensao(0.0) == 0.0
    assert md.tensao(0.5e-9) == pytest.approx(750.0)
    assert md.tensao(2e-9) == pytest.approx(3e3)
    assert md.tensao(1e-6) == 3e3


def test_pulso_trapezoidal_repetido(pecas):
    # rise 2 ns, plateau 10 ns, fall 4 ns, repetition every 100 ns
    md = Modelo(*pecas, n_celulas=4, v_pico=3e3, forma="pulso", subida=2e-9, largura=10e-9,
                descida=4e-9, periodo=1e-7)
    for t0 in (0.0, 1e-7, 3e-7):
        assert md.tensao(t0 + 1e-9) == pytest.approx(1.5e3)
        assert md.tensao(t0 + 5e-9) == pytest.approx(3e3)
        assert md.tensao(t0 + 14e-9) == pytest.approx(1.5e3)
        assert md.tensao(t0 + 16e-9) == pytest.approx(0.0, abs=1e-9)
        assert md.tensao(t0 + 50e-9) == 0.0


def test_pulso_mais_longo_que_o_periodo_e_erro(pecas):
    with pytest.raises(ValueError, match="period"):
        Modelo(*pecas, n_celulas=4, forma="pulso", subida=2e-9, largura=1e-7, descida=2e-9, periodo=1e-7)


def test_passo_maximo_nao_pula_o_pulso():
    from simula import passo_maximo
    from argparse import Namespace
    assert passo_maximo(Namespace(forma="seno")) == np.inf
    assert passo_maximo(Namespace(forma="degrau", subida=2e-9)) == np.inf
    assert passo_maximo(Namespace(forma="pulso", subida=2e-9, largura=10e-9, descida=4e-9)) == 2e-9


@pytest.mark.parametrize("extra, mensagem", [
    (["--waveform", "pulse", "--rise", "1e-9"], "--width and --fall must"),
    (["--waveform", "pulse"], "--rise must"),
    (["--waveform", "step", "--rise", "0"], "--rise must"),
    (["--forma", "pulso", "--subida", "1e-9", "--largura", "1e-8", "--descida", "1e-9", "--periodo", "5e-9"],
     "shorter than --period"),
])
def test_forma_invalida_e_recusada(tmp_path, extra, mensagem):
    r = subprocess.run([sys.executable, str(PROTOTIPO / "simula.py"), str(tmp_path / "x"), "--t-end", "2e-9", *extra],
                       capture_output=True, text=True, env={"OPENBLAS_NUM_THREADS": "1", "PATH": ""})
    erro = [l for l in (r.stderr + r.stdout).splitlines() if "error:" in l]
    assert r.returncode != 0 and erro and mensagem in erro[-1]
    assert "Traceback" not in r.stderr


def test_pulso_roda_e_grava_a_tensao(tmp_path):
    r = subprocess.run([sys.executable, str(PROTOTIPO / "simula.py"), str(tmp_path / "p"), "--n-celulas", "10",
                        "--forma", "pulso", "--v-pico", "2e3", "--subida", "1e-9", "--largura", "3e-9",
                        "--descida", "1e-9", "--periodo", "1e-8", "--t-end", "1.2e-8", "--dt-saida", "1e-9",
                        "--espessura", "1e-3"],
                       capture_output=True, text=True, env={"OPENBLAS_NUM_THREADS": "1", "PATH": ""})
    assert r.returncode == 0, r.stderr
    v = np.loadtxt(tmp_path / "p" / "superficie.dat", skiprows=1)[:, 1]
    np.testing.assert_allclose(v, [0, 2e3, 2e3, 2e3, 2e3, 0, 0, 0, 0, 0, 0, 2e3, 2e3], atol=1e-6)


def test_gas_pre_aquecido(tmp_path):
    # preheated mixture (850 K, as in the ammonia case): N from the pressure of 1
    # atm at 850 K and the initial temperature of the profile at 850 K
    import json
    r = subprocess.run([sys.executable, str(PROTOTIPO / "simula.py"), str(tmp_path / "q"), "--n-celulas", "10",
                        "--t-gas", "850", "--t-parede", "850", "--aquecimento", "--t-end", "1e-9",
                        "--dt-saida", "1e-9", "--espessura", "1e-3"],
                       capture_output=True, text=True, env={"OPENBLAS_NUM_THREADS": "1", "PATH": ""})
    assert r.returncode == 0, r.stderr
    par = json.loads((tmp_path / "q" / "parametros.json").read_text())
    assert par["modelo"]["n_gas"] == pytest.approx(101325.0 / (1.380649e-23 * 850.0) * 1e-6)
    linhas = (tmp_path / "q" / "perfil.dat").read_text().splitlines()
    nomes = [linhas[1][i:i + 24].strip() for i in range(0, len(linhas[1]), 24)]
    assert float(linhas[3].split()[nomes.index("Temperature")]) == pytest.approx(850.0)


# tabulated voltage (measured pulse, as that of Yang 2022): pairs (t, V) within
# one period, repeated every --period, zero outside the table
T_TAB = np.array([0.0, 2e-9, 5e-9, 9e-9, 20e-9])
V_TAB = np.array([0.0, 20e3, -4e3, 6e3, 0.0])


def test_tabela_interpola_e_repete(pecas):
    md = Modelo(*pecas, n_celulas=4, forma="tabela", tabela_tensao=(T_TAB, V_TAB), periodo=1e-6)
    for t0 in (0.0, 1e-6, 7e-6):
        assert md.tensao(t0 + 1e-9) == pytest.approx(10e3)
        assert md.tensao(t0 + 3.5e-9) == pytest.approx(8e3)
        assert md.tensao(t0 + 2e-9) == pytest.approx(20e3)
        assert md.tensao(t0 + 14.5e-9) == pytest.approx(3e3)
        assert md.tensao(t0 + 0.5e-6) == 0.0


@pytest.mark.parametrize("t, v, mensagem", [
    (T_TAB, np.array([1e3, 20e3, -4e3, 6e3, 0.0]), "start and end at zero"),
    (T_TAB, np.array([0.0, 20e3, -4e3, 6e3, 5.0]), "start and end at zero"),
    (np.array([0.0, 3e-9, 2e-9, 9e-9, 20e-9]), V_TAB, "increase"),
    (T_TAB * 1e2, V_TAB, "shorter than the period"),
])
def test_tabela_invalida_e_recusada(pecas, t, v, mensagem):
    with pytest.raises(ValueError, match=mensagem):
        Modelo(*pecas, n_celulas=4, forma="tabela", tabela_tensao=(t, v), periodo=1e-6)


def test_passo_maximo_da_tabela_nao_pula_o_pulso():
    # inside the pulse window, the step of the table (the smallest separation between
    # points); outside, never beyond the start of the next pulse
    from simula import passo_maximo_em
    from argparse import Namespace
    a = Namespace(forma="tabela", periodo=1e-6, tabela=(T_TAB, V_TAB))
    assert passo_maximo_em(a, 0.0) == pytest.approx(2e-9)
    assert passo_maximo_em(a, 3e-6 + 10e-9) == pytest.approx(2e-9)
    for t in (25e-9, 0.3e-6, 0.999e-6, 4.5e-6):
        inicio_seguinte = (np.floor(t / 1e-6) + 1) * 1e-6
        assert 0.0 < passo_maximo_em(a, t) <= inicio_seguinte - t + 1e-18
    # far from the pulse the step may grow
    assert passo_maximo_em(a, 0.3e-6) > 0.5e-6
    assert passo_maximo_em(Namespace(forma="seno"), 1.0) == np.inf


def test_tabela_roda_e_grava_a_tensao(tmp_path):
    arq = tmp_path / "pulso.txt"
    np.savetxt(arq, np.column_stack([T_TAB, V_TAB / 10]), header="t_s V_V")
    # with 10 cells and local field the BDF with mass collapses the step at the field
    # reversal at t = 20 ns (mesh that does not resolve the sheath); with 40, it does not
    r = subprocess.run([sys.executable, str(PROTOTIPO / "simula.py"), str(tmp_path / "p"), "--n-celulas", "40",
                        "--forma", "tabela", "--tensao-tabela", str(arq), "--periodo", "1e-7",
                        "--t-end", "1.2e-7", "--dt-saida", "1e-9", "--espessura", "1e-3"],
                       capture_output=True, text=True, env={"OPENBLAS_NUM_THREADS": "1", "PATH": ""})
    assert r.returncode == 0, r.stderr
    sup = np.loadtxt(tmp_path / "p" / "superficie.dat", skiprows=1)
    esperado = np.interp(np.mod(sup[:, 0], 1e-7), T_TAB, V_TAB / 10, right=0.0)
    np.testing.assert_allclose(sup[:, 1], esperado, atol=1e-6)
    assert np.abs(sup[:, 1]).max() > 1.9e3


def test_tabela_sem_arquivo_e_recusada(tmp_path):
    r = subprocess.run([sys.executable, str(PROTOTIPO / "simula.py"), str(tmp_path / "x"), "--t-end", "2e-9",
                        "--forma", "tabela"],
                       capture_output=True, text=True, env={"OPENBLAS_NUM_THREADS": "1", "PATH": ""})
    erro = [l for l in (r.stderr + r.stdout).splitlines() if "error:" in l]
    assert r.returncode != 0 and erro and "--voltage-table" in erro[-1]


def test_tabela_no_meio_do_periodo_nao_e_pulada(tmp_path):
    # pulse of 20 kV at 0.5 µs in a period of 1 µs, with outputs only at 0.27 and
    # 0.54 µs: without the dynamic maximum step the integrator goes from 0.27 to 0.54 µs
    # in a few steps, skips the pulse and the barrier stays uncharged (~5e-11 C/m2);
    # with it, the discharge leaves ~1e-5 C/m2
    arq = tmp_path / "pulso.txt"
    np.savetxt(arq, np.column_stack([T_TAB + 0.5e-6, V_TAB]))
    r = subprocess.run([sys.executable, str(PROTOTIPO / "simula.py"), str(tmp_path / "p"), "--n-celulas", "15",
                        "--forma", "tabela", "--tensao-tabela", str(arq), "--periodo", "1e-6",
                        "--t-end", "5.4e-7", "--dt-saida", "2.7e-7", "--espessura", "1e-3", "--ne0", "1e6"],
                       capture_output=True, text=True, env={"OPENBLAS_NUM_THREADS": "1", "PATH": ""})
    assert r.returncode == 0, r.stderr
    sup = np.loadtxt(tmp_path / "p" / "superficie.dat", skiprows=1)
    assert abs(sup[-1, 4]) > 1e-6


def test_tabela_retoma_em_outra_pasta(tmp_path):
    # the origin records the absolute path of the table; the restart with the same
    # file by a relative path is the same table
    np.savetxt(tmp_path / "pulso.txt", np.column_stack([T_TAB, V_TAB / 10]))
    comum = ["--n-celulas", "40", "--forma", "tabela", "--tensao-tabela", "pulso.txt", "--periodo", "1e-7",
             "--espessura", "1e-3"]
    env = {"OPENBLAS_NUM_THREADS": "1", "PATH": ""}
    r = subprocess.run([sys.executable, str(PROTOTIPO / "simula.py"), "r0", *comum, "--t-end", "1e-8",
                        "--dt-saida", "1e-9"], capture_output=True, text=True, env=env, cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    r = subprocess.run([sys.executable, str(PROTOTIPO / "simula.py"), "r1", "--retoma", "r0", "5e-9", *comum,
                        "--t-end", "1e-8", "--dt-saida", "5e-10"], capture_output=True, text=True, env=env, cwd=tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    a = np.loadtxt(tmp_path / "r0" / "superficie.dat", skiprows=1)
    b = np.loadtxt(tmp_path / "r1" / "superficie.dat", skiprows=1)
    np.testing.assert_allclose(b[-1, 1], a[-1, 1])
