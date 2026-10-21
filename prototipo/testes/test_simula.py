# SPDX-License-Identifier: GPL-3.0-or-later
"""End-to-end test of simula.py: a small case runs, writes the outputs in the format
of the Fortran code and the restart from a recorded instant reproduces the
continuous run within the integrator tolerance (with the DAE, the continuous run
does not restart the integrator at the outputs and the restart does)."""
import json
import subprocess
import sys

import numpy as np
import pytest

from conftest import PROTOTIPO, RAIZ, quimica_do, precisa_caso

sys.path.insert(0, str(RAIZ / "scripts"))
from resumo import blocos  # noqa: E402

BASE = ["--v-peak", "5e3", "--cells", "20", "--clustering", "1.5", "--thickness", "1.75e-3",
        "--energy", "--dt-output", "2e-9", "--ne0", "1e9", "--neutral"]


def roda(destino, *extra):
    subprocess.run([sys.executable, str(PROTOTIPO / "simula.py"), str(destino), *BASE, *extra],
                   check=True, capture_output=True, env={"OPENBLAS_NUM_THREADS": "1", "PATH": ""})


def superficie(d):
    return np.loadtxt(d / "superficie.dat", skiprows=1)


DENSIDADES = ["E", "AR*", "AR^+", "AR2^+", "E/N"]


def iguais(a, b, rtol, fator_atol):
    """Compares column by column, with an absolute tolerance relative to the maximum of the column."""
    for j in range(a.shape[1]):
        np.testing.assert_allclose(a[:, j], b[:, j], rtol=rtol, atol=fator_atol * np.abs(b[:, j]).max())


def colunas(d, nomes_col, k=-1):
    """Columns by name in block k of the profile, cells only."""
    tb, nomes, linhas = list(blocos(d / "perfil.dat"))[k]
    a = np.array(linhas[1:-1])
    return np.column_stack([a[:, nomes.index(n)] for n in nomes_col])


def test_roda_grava_e_retoma(tmp_path):
    continua, parte, resto = tmp_path / "continua", tmp_path / "parte", tmp_path / "resto"
    roda(continua, "--t-end", "8e-9", "--rtol", "1e-7")
    roda(parte, "--t-end", "4e-9", "--rtol", "1e-7")
    roda(resto, "--t-end", "8e-9", "--retoma", str(parte), "4e-9", "--rtol", "1e-7")
    s = superficie(continua)
    assert s.shape == (5, 6)                               # t = 0, 2, 4, 6, 8 ns
    perfis = list(blocos(continua / "perfil.dat"))
    assert len(perfis) == 5 and len(perfis[-1][2]) == 22   # 20 cells and the two wall nodes
    np.testing.assert_allclose(superficie(resto)[-1], s[-1], rtol=1e-5, atol=1e-20)
    p_resto, p_cont = colunas(resto, DENSIDADES), colunas(continua, DENSIDADES)
    iguais(p_resto, p_cont, 1e-4, 1e-4)


def test_dae_reproduz_o_caminho_original(tmp_path):
    # same physics by the two paths: finite difference Jacobian of scipy
    # with Poisson inside the right-hand side, and DAE with the potential in the state and
    # structured Jacobian; with a tight tolerance, the outputs coincide
    fd, dae = tmp_path / "fd", tmp_path / "dae"
    roda(fd, "--t-end", "8e-9", "--jac", "fd", "--rtol", "1e-7")
    roda(dae, "--t-end", "8e-9", "--jac", "dae", "--rtol", "1e-7")
    s_fd, s_dae = superficie(fd), superficie(dae)
    np.testing.assert_allclose(s_dae[:, 2:4], s_fd[:, 2:4], rtol=1e-5)
    p_fd, p_dae = colunas(fd, DENSIDADES), colunas(dae, DENSIDADES)
    iguais(p_dae, p_fd, 1e-4, 1e-3)


def test_aquecimento_grava_e_retoma_a_temperatura(tmp_path):
    # with --heating the Temperature column starts at 300 K, only
    # grows with the discharge and is read back on restart
    continua, parte, resto = tmp_path / "continua", tmp_path / "parte", tmp_path / "resto"
    roda(continua, "--t-end", "8e-9", "--rtol", "1e-7", "--aquecimento")
    roda(parte, "--t-end", "4e-9", "--rtol", "1e-7", "--aquecimento")
    roda(resto, "--t-end", "8e-9", "--retoma", str(parte), "4e-9", "--rtol", "1e-7", "--aquecimento")
    perfis = list(blocos(continua / "perfil.dat"))
    nomes = perfis[0][1]
    j = nomes.index("Temperature")
    t0 = np.array(perfis[0][2][1:-1])[:, j]
    t_fim = np.array(perfis[-1][2][1:-1])[:, j]
    assert np.all(t0 == 300.0)
    assert np.all(t_fim >= 300.0 - 1e-6) and t_fim.max() > 300.0
    p_resto = np.array(list(blocos(resto / "perfil.dat"))[-1][2][1:-1])
    np.testing.assert_allclose(p_resto[:, j], t_fim, rtol=1e-7)


def test_perfil_com_coeficientes_dos_eletrons_e_do_gas(tmp_path):
    # as the profile of the Fortran code: De, Me and ve = Me*E in the cells and, with
    # heating, kappa and rho*cv
    d = tmp_path / "c"
    roda(d, "--t-end", "2e-9", "--aquecimento")
    tb, nomes, linhas = list(blocos(d / "perfil.dat"))[-1]
    a = np.array(linhas[1:-1])
    for nome in ("De", "Me", "veloc", "diff T", "rho cv"):
        assert nome in nomes
    me, ve, e_n = a[:, nomes.index("Me")], a[:, nomes.index("veloc")], a[:, nomes.index("E/N")]
    n_gas = 101325.0 / (1.380649e-23 * 300.0)
    assert np.all(me > 0.0) and np.all(a[:, nomes.index("De")] > 0.0)
    np.testing.assert_allclose(ve, me * e_n * 1e-21 * n_gas, rtol=1e-10)
    t = a[:, nomes.index("Temperature")]
    np.testing.assert_allclose(a[:, nomes.index("diff T")], (0.0307313 * t + 10.006) * 1e-3, rtol=1e-12)
    # collision frequency as the Fortran code: (nu/N) of the table times the total N of the
    # cell (gas and every species); here from the mean energy of Te
    from tabela import E_NUN, TabelaEnergia
    tab = TabelaEnergia(RAIZ / "tabela_energia.dat")
    eps = 1.5 * 1.380649e-23 * a[:, nomes.index("Te")] / 1.602176634e-19
    especies = nomes[nomes.index("Te") + 1:nomes.index("E/N")]           # all, with the gas (AR)
    assert "AR" in especies
    n_total = sum(a[:, nomes.index(c)] for c in especies)                  # cm-3
    np.testing.assert_allclose(a[:, nomes.index("freq")], tab(eps)[:, E_NUN] * n_total, rtol=1e-6)


def test_serie_temporal_numa_posicao(tmp_path):
    # scripts/sonda.py: the series of a column at the cell closest to x, as
    # borda0.dat, centro.dat and borda1.dat of the Fortran code
    from sonda import serie
    d = tmp_path / "c"
    roda(d, "--t-end", "4e-9")
    t, v = serie(d / "perfil.dat", "E", 4.0e-4)
    perfis = list(blocos(d / "perfil.dat"))
    assert len(t) == len(perfis) == 3
    for (tb, nomes, linhas), tv, vv in zip(perfis, t, v):
        a = np.array(linhas[1:-1])
        k = np.argmin(np.abs(a[:, 0] - 4.0e-4))
        assert tv == tb and vv == a[k, nomes.index("E")]


@pytest.mark.parametrize("extra, mensagem", [
    (["--period", "0"], "period"),
    (["--dt-output", "0"], "dt-output"),
    (["--thickness", "-1e-3"], "thickness"),
    (["--eps-dielectric", "0.5"], "eps-dielectric"),
    (["--cells", "2"], "cells"),
    (["--t-end", "5e-9", "--dt-output", "2e-9"], "multiple"),
    (["--gap", "0"], "gap"),
    (["--k0-ar-ion", "0"], "k0-ar-ion"),
    (["--clustering", "-1"], "clustering"),
    (["--eps-sec", "-1"], "eps-sec"),
    (["--adaptive", "--jac", "fd", "--dt-mesh", "0"], "dt-mesh"),
    (["--restart", "x", "abc"], "restart"),
    (["--restart", "/nao/existe", "1e-9"], "restart"),
    (["--ne0", "-1"], "ne0"),
    (["--transport", "/nao/existe.json"], "transport"),
    (["--eps-gas", "0.5"], "eps-gas"),
    # initial or wall T_g outside the nodes of the table (from 300 to 1500 K in
    # argon): the table would clamp at the end node without warning
    (["--t-gas", "250"], "t_nos.txt"),
    (["--aquecimento", "--t-parede", "2000"], "t_nos.txt"),
])
def test_entrada_invalida_e_recusada(tmp_path, extra, mensagem):
    # as entrada.f90: meaningless parameters stop the run with a message
    r = subprocess.run([sys.executable, str(PROTOTIPO / "simula.py"), str(tmp_path / "x"), "--t-end", "2e-9", *extra],
                       capture_output=True, text=True, env={"OPENBLAS_NUM_THREADS": "1", "PATH": ""})
    assert r.returncode != 0
    assert mensagem in r.stderr + r.stdout
    assert "Traceback" not in r.stderr               # refused with a message, not with a crash
    assert not (tmp_path / "x" / "perfil.dat").exists()


def test_parametros_gravados_e_mobilidades_configuraveis(tmp_path):
    # the run records the parameters (as the entrada.nml in the directory of the
    # Fortran code); the ion mobilities of entrada.nml are options
    d = tmp_path / "c"
    roda(d, "--t-end", "2e-9", "--k0-ar-ion", "1.5", "--en-aquecimento-ar-ion", "200",
         "--k0-ar2-ion", "1.9", "--t-parede", "310", "--aquecimento")
    p = json.loads((d / "parametros.json").read_text())
    assert p["argumentos"]["k0_ar_ion"] == 1.5 and p["argumentos"]["t_parede"] == 310.0
    assert p["argumentos"]["v_pico"] == 5e3
    # default tolerances: rtol 1e-4 and atol 1e-10 (see the test of depleted electrons)
    assert (p["argumentos"]["rtol"], p["argumentos"]["atol"]) == (1e-4, 1e-10)
    assert p["modelo"]["eps_gas"] == 1.0
    mob = p["modelo"]["transporte"]
    assert mob["AR^+"]["mobilidade"] == {"tipo": "frost", "K0": 1.5, "EN_aquecimento": 200.0}
    assert mob["AR2^+"]["mobilidade"]["K0"] == 1.9
    for chave in ("commit", "numpy", "scipy", "python", "OPENBLAS_NUM_THREADS"):
        assert chave in p
    assert p["OPENBLAS_NUM_THREADS"] == "1"
    # the wall temperature reaches the model (wall nodes in the profile)
    tb, nomes, linhas = list(blocos(d / "perfil.dat"))[-1]
    assert linhas[0][nomes.index("Temperature")] == 310.0 and linhas[-1][nomes.index("Temperature")] == 310.0
    # and the mobilities reach the model: Ar+ changes with respect to the default
    padrao = tmp_path / "padrao"
    roda(padrao, "--t-end", "2e-9", "--aquecimento")
    ar = [np.array(list(blocos(x / "perfil.dat"))[-1][2][1:-1]) for x in (d, padrao)]
    j = list(blocos(d / "perfil.dat"))[-1][1].index("AR^+")
    assert np.max(np.abs(ar[0][:, j] / ar[1][:, j] - 1.0)) > 1e-4


def test_eletrodo_metalico_e_aceito(tmp_path):
    roda(tmp_path / "m", "--t-end", "2e-9", "--espessura", "0")
    assert (tmp_path / "m" / "perfil.dat").exists()


def test_parametros_com_malha_por_arquivo_e_identificacao_do_codigo(tmp_path):
    # with --faces the number of cells is that of the file; the path is recorded
    # absolute, with the sha256 of the content; the data and the code are identified
    # by sha256; the effective values of the model also go to the record
    import hashlib
    faces = np.linspace(0.0, 8e-4, 13)
    arq = tmp_path / "faces.txt"
    np.savetxt(arq, faces)
    d = tmp_path / "f"
    roda(d, "--t-end", "2e-9", "--faces", str(arq))
    p = json.loads((d / "parametros.json").read_text())
    assert p["argumentos"]["n_celulas"] == 12
    assert p["argumentos"]["faces"] == str(arq.resolve())
    assert p["sha256"]["faces"] == hashlib.sha256(arq.read_bytes()).hexdigest()
    for nome in ("tabela_energia.dat", "tabela_energia_tg.dat", "tabela_bolsig.dat", "zdplaskin_m.F90", "quimica.json",
                 "bolsigdb.dat", "transporte.json", "prototipo/modelo.py"):
        # the distributed package has neither zdplaskin_m.F90 nor bolsigdb.dat: it records what exists
        if (RAIZ / nome).exists():
            assert p["sha256"][nome] == hashlib.sha256((RAIZ / nome).read_bytes()).hexdigest()
    assert p["modelo"]["transporte"]["AR^+"]["mobilidade"]["K0"] == 1.61 and p["modelo"]["n_gas"] > 2.4e19
    assert p["argumentos"]["k0_ar_ion"] is None              # without the option, transporte.json applies
    assert "argv" in p and "cwd" in p


@pytest.mark.parametrize("faces, mensagem", [([0.0, 4e-4, 8e-4], "faces"),
                                             ([0.0, 5e-4, 3e-4, 8e-4], "faces")])
def test_malha_por_arquivo_invalida(tmp_path, faces, mensagem):
    # fewer than 3 cells or faces out of order are refused
    arq = tmp_path / "faces.txt"
    np.savetxt(arq, faces)
    r = subprocess.run([sys.executable, str(PROTOTIPO / "simula.py"), str(tmp_path / "x"), "--t-end", "2e-9",
                        "--faces", str(arq)], capture_output=True, text=True,
                       env={"OPENBLAS_NUM_THREADS": "1", "PATH": ""})
    assert r.returncode != 0 and mensagem in r.stderr + r.stdout


FORTRAN = ["Pos", "Temperature", "volt", "De", "Me", "veloc", "diff T", "rho cv", "alfa", "freq", "Te",
           "E", "AR", "AR*", "AR^+", "AR2^+", "E/N", "campo"]


def test_perfil_com_as_colunas_do_fortran(tmp_path):
    # same names as saida.f90, always present (without the CFL, which does not apply
    # to the implicit integrator); without heating the temperature is that of the gas; the background
    # gas appears even when fixed; in the wall lines, as the Fortran code, the
    # temperature and kappa of the wall and rho cv of the neighbouring cell
    # (--t-wall without --heating has no effect: the walls stay at the
    # gas temperature)
    d = tmp_path / "c"
    roda(d, "--t-end", "2e-9", "--t-parede", "350")
    tb, nomes, linhas = list(blocos(d / "perfil.dat"))[-1]
    for nome in FORTRAN:
        assert nome in nomes, nome
    a = np.array(linhas)
    col = {nome: a[:, nomes.index(nome)] for nome in FORTRAN}
    assert np.all(col["Temperature"] == 300.0)
    n_gas = 101325.0 / (1.380649e-23 * 300.0) * 1e-6
    assert np.all(col["AR"][1:-1] == n_gas)
    kappa300 = (0.0307313 * 300.0 + 10.006) * 1e-3
    np.testing.assert_allclose(col["diff T"], kappa300, rtol=1e-12)
    np.testing.assert_allclose(col["alfa"][1:-1], col["diff T"][1:-1] / col["rho cv"][1:-1], rtol=1e-12)
    assert col["rho cv"][0] == col["rho cv"][1] and col["rho cv"][-1] == col["rho cv"][-2]


def test_estado_gravado_e_lido_sem_perda(tmp_path):
    # the profile keeps n_eps with the energy equation: writing and reading returns the
    # exact state (before, the energy was rebuilt from the recorded Te, with loss)
    sys.path.insert(0, str(PROTOTIPO))
    from simula import estado_gravado, grava
    from jacobiano import ModeloRapido
    from quimica_zdp import Quimica
    from tabela import Tabela, TabelaEnergia
    md = ModeloRapido(quimica_do(RAIZ), Tabela(RAIZ / "tabela_bolsig.dat"), n_celulas=8,
                      espessura=1e-3, tabela_energia=TabelaEnergia(RAIZ / "tabela_energia.dat"),
                      aquecimento=True)
    g = np.random.default_rng(4)
    n = 10.0 ** g.uniform(6, 11, size=(md.n_esp, md.m))
    y = np.concatenate([n.ravel(), n[md.i_e] * g.uniform(0.5, 8, md.m), 300 + 50 * g.uniform(size=md.m),
                        [1e-7, -2e-7]])
    grava(tmp_path, md, [3e-9], y[:, None], True)
    np.testing.assert_allclose(estado_gravado(md, tmp_path, 3e-9), y, rtol=1e-15, atol=0)


def test_retomada_recusa_malha_diferente(tmp_path):
    roda(tmp_path / "a", "--t-end", "2e-9")
    r = subprocess.run([sys.executable, str(PROTOTIPO / "simula.py"), str(tmp_path / "b"), *BASE, "--t-end", "4e-9",
                        "--retoma", str(tmp_path / "a"), "2e-9", "--agrupamento", "0"],
                       capture_output=True, text=True, env={"OPENBLAS_NUM_THREADS": "1", "PATH": ""})
    assert r.returncode != 0 and "malha" in r.stderr + r.stdout


def test_retomada_no_mesmo_diretorio_preserva_o_historico(tmp_path):
    # restart in the middle: the records after 4 ns are redone, not duplicated
    # restart in the middle: the records up to 4 ns stay intact byte by byte, the
    # later ones are redone without duplication; the previous files and parameters
    # are kept until the new run finishes
    d = tmp_path / "a"
    roda(d, "--t-end", "8e-9")
    antes = (d / "perfil.dat").read_text().split("\n\n")
    params_antes = (d / "parametros.json").read_text()
    roda(d, "--t-end", "8e-9", "--retoma", str(d), "4e-9")
    assert [round(tb * 1e9) for tb, _, _ in blocos(d / "perfil.dat")] == [0, 2, 4, 6, 8]
    np.testing.assert_allclose(superficie(d)[:, 0], [0, 2e-9, 4e-9, 6e-9, 8e-9], atol=1e-20)
    assert (d / "perfil.dat").read_text().split("\n\n")[:3] == antes[:3]
    assert (d / "parametros_ate_4e-09.json").read_text() == params_antes
    p = json.loads((d / "parametros.json").read_text())
    assert p["argumentos"]["retoma"][1] == "4e-9"


def test_retomada_no_mesmo_diretorio_que_falha_nao_perde_dados(tmp_path):
    # restart from an instant that does not exist: nothing changes in the directory
    d = tmp_path / "a"
    roda(d, "--t-end", "4e-9")
    antes = {f: (d / f).read_text() for f in ("perfil.dat", "superficie.dat", "parametros.json")}
    r = subprocess.run([sys.executable, str(PROTOTIPO / "simula.py"), str(d), *BASE, "--t-end", "8e-9",
                        "--retoma", str(d), "3e-9"], capture_output=True, text=True,
                       env={"OPENBLAS_NUM_THREADS": "1", "PATH": ""})
    assert r.returncode != 0
    assert {f: (d / f).read_text() for f in antes} == antes


@pytest.mark.parametrize("origem, retomada, mensagem", [(["--heating"], [], "heating"),
                                                         (["--free-gas"], [], "free-gas"),
                                                         (["--waveform", "step", "--rise", "1e-9"], [], "--waveform"),
                                                         (["--t-gas", "400"], [], "--t-gas")])
def test_retomada_recusa_descartar_variaveis_da_origem(tmp_path, origem, retomada, mensagem):
    # restarting without --heating (or --free-gas) from a run that had it
    # would discard the temperature (or the Ar) without warning
    roda(tmp_path / "a", "--t-end", "2e-9", *origem)
    r = subprocess.run([sys.executable, str(PROTOTIPO / "simula.py"), str(tmp_path / "b"), *BASE, "--t-end", "4e-9",
                        "--retoma", str(tmp_path / "a"), "2e-9", *retomada], capture_output=True, text=True,
                       env={"OPENBLAS_NUM_THREADS": "1", "PATH": ""})
    assert r.returncode != 0 and mensagem in r.stderr + r.stdout


def test_retomada_com_gas_solto(tmp_path):
    continua, parte, resto = tmp_path / "c", tmp_path / "p", tmp_path / "r"
    roda(continua, "--t-end", "8e-9", "--rtol", "1e-7", "--gas-solto")
    roda(parte, "--t-end", "4e-9", "--rtol", "1e-7", "--gas-solto")
    roda(resto, "--t-end", "8e-9", "--retoma", str(parte), "4e-9", "--rtol", "1e-7", "--gas-solto")
    nomes = DENSIDADES + ["AR"]
    iguais(colunas(resto, nomes), colunas(continua, nomes), 1e-4, 1e-4)


def test_estado_lido_do_perfil_do_fortran(tmp_path):
    # the profile of the Fortran code (regression reference, 100 uniform cells, local
    # field) is read by name: the densities of the cells come back equal
    sys.path.insert(0, str(PROTOTIPO))
    from simula import estado_gravado
    from modelo import Modelo
    from quimica_zdp import Quimica
    from tabela import Tabela
    import shutil
    ref = tmp_path / "ref"
    ref.mkdir()
    shutil.copy(RAIZ / "referencias" / "release_np4_20ns" / "perfil.dat", ref)
    (ref / "superficie.dat").write_text("cabecalho\n" + " 1.0E-08 0 0 0 0 0\n")
    md = Modelo(quimica_do(RAIZ), Tabela(RAIZ / "tabela_bolsig.dat"), n_celulas=100)
    y = estado_gravado(md, ref, 1e-8)
    tb, nomes, linhas = [b for b in blocos(ref / "perfil.dat") if abs(b[0] - 1e-8) < 1e-15][0]
    a = np.array(linhas[1:-1])
    np.testing.assert_array_equal(md.separa(y)[0][md.i_e], a[:, nomes.index("E")])


def test_aviso_de_ionizacao_com_a_posicao(tmp_path):
    # n_e/N above 1e-3 already at the start: the warning appears once, with the position
    r = subprocess.run([sys.executable, str(PROTOTIPO / "simula.py"), str(tmp_path / "x"), "--t-end", "1e-12",
                        "--dt-saida", "1e-12", "--ne0", "3e16", "--neutro", "--n-celulas", "6"],
                       capture_output=True, text=True, env={"OPENBLAS_NUM_THREADS": "1", "PATH": ""})
    assert r.stdout.count("WARNING: ionization degree") == 1
    # 6 uniform cells in 0.8 mm: the first (node 2 of the Fortran code) at x = 66.67 um
    assert "node 2 (x = 66.67 um)" in r.stdout


def test_avisos_verificados_a_cada_passo(tmp_path, monkeypatch):
    # the warnings do not wait for the next output: they are checked at every
    # accepted step of the integrator, including between outputs
    sys.path.insert(0, str(PROTOTIPO))
    import simula
    instantes = []
    original = simula.avisa
    monkeypatch.setattr(simula, "avisa", lambda md, t, y: (instantes.append(t), original(md, t, y)))
    monkeypatch.setattr(sys, "argv", ["simula.py", str(tmp_path / "x"), *BASE, "--t-end", "2e-9"])
    simula.main()
    entre = [t for t in instantes if 1e-15 < t < 2e-9 * (1 - 1e-9)]
    assert len(entre) > 3


def test_tolerancia_padrao_controla_os_eletrons_esgotados(tmp_path):
    # local field, 5 kV, 1600 cells, up to 25 ns: between 20 and 25 ns ~700 cells
    # have n_e between 1e-8 and 1e-4 cm-3, seed of the next avalanche. With atol of
    # 1e-4 they had no error control and n_e came out up to 5 times wrong (40 to
    # 85 times with 3200 cells); with the default (rtol 1e-4, atol 1e-10) it stays within
    # 1% of the tight integration (rtol 1e-8, atol 1e-14)
    caso = ["--v-pico", "5e3", "--espessura", "1.75e-3", "--n-celulas", "1600", "--t-end", "2.5e-8",
            "--dt-saida", "1e-9", "--aquecimento"]
    for nome, extra in (("padrao", []), ("apertado", ["--rtol", "1e-8", "--atol", "1e-14"])):
        subprocess.run([sys.executable, str(PROTOTIPO / "simula.py"), str(tmp_path / nome), *caso, *extra],
                       check=True, capture_output=True, env={"OPENBLAS_NUM_THREADS": "1", "PATH": ""})
    for t in (22, 24, 25):
        a = colunas(tmp_path / "padrao", ["E"], k=t).ravel()
        b = colunas(tmp_path / "apertado", ["E"], k=t).ravel()
        h = np.diff(np.linspace(0.0, 8e-4, 1601))
        assert np.sum(a * h) == pytest.approx(np.sum(b * h), rel=1e-2), t


def test_arquivo_de_transporte_por_opcao(tmp_path):
    # --transport replaces the transport data: the value reaches the model, the
    # path and the sha256 of the file go to the record; a species without data is
    # refused with the message of the model
    import hashlib
    dados = json.loads((RAIZ / "transporte.json").read_text())
    dados["AR2^+"]["mobilidade"]["K0"] = 2.5
    arq = tmp_path / "transporte.json"
    arq.write_text(json.dumps(dados))
    d = tmp_path / "t"
    roda(d, "--t-end", "2e-9", "--transporte", str(arq))
    p = json.loads((d / "parametros.json").read_text())
    assert p["modelo"]["transporte"]["AR2^+"]["mobilidade"]["K0"] == 2.5
    assert p["argumentos"]["transporte"] == str(arq.resolve())
    assert p["sha256"]["transporte"] == hashlib.sha256(arq.read_bytes()).hexdigest()
    del dados["AR*"]
    arq.write_text(json.dumps(dados))
    r = subprocess.run([sys.executable, str(PROTOTIPO / "simula.py"), str(tmp_path / "u"), *BASE, "--t-end", "2e-9",
                        "--transporte", str(arq)], capture_output=True, text=True,
                       env={"OPENBLAS_NUM_THREADS": "1", "PATH": ""})
    assert r.returncode != 0 and "error: transport: no data for the species AR*" in r.stderr
    assert "Traceback" not in r.stderr


def test_log_conta_temperatura_fora_da_tabela(tmp_path):
    # T_g above 1500 K is clamped at the last node of the T_g table: the log counts it
    r = subprocess.run([sys.executable, str(PROTOTIPO / "simula.py"), str(tmp_path / "t"), *BASE, "--t-end", "2e-9"],
                       capture_output=True, text=True, env={"OPENBLAS_NUM_THREADS": "1", "PATH": ""})
    assert r.returncode == 0, r.stderr
    import re
    assert re.search(r"out of table \(above/below/T_g\) \d+/\d+/0 ", r.stdout), r.stdout[-500:]


def test_log_de_campo_local_conta_temperatura_fora_da_tabela(tmp_path):
    import re
    base = [a for a in BASE if a != "--energy"]
    r = subprocess.run([sys.executable, str(PROTOTIPO / "simula.py"), str(tmp_path / "t"), *base, "--t-end", "2e-9"],
                       capture_output=True, text=True, env={"OPENBLAS_NUM_THREADS": "1", "PATH": ""})
    assert r.returncode == 0, r.stderr
    assert re.search(r"out of table \(above/below/T_g\) \d+/-/0 ", r.stdout), r.stdout[-500:]


def test_caso_na_raiz_e_o_padrao(tmp_path):
    # --case with the root of the repository is the same as without the option
    roda(tmp_path / "a", "--t-end", "2e-9")
    roda(tmp_path / "b", "--t-end", "2e-9", "--caso", str(RAIZ))
    for nome in ("perfil.dat", "superficie.dat"):
        assert (tmp_path / "a" / nome).read_bytes() == (tmp_path / "b" / nome).read_bytes()
    p = json.loads((tmp_path / "b" / "parametros.json").read_text())
    assert p["argumentos"]["caso"] == str(RAIZ)
    assert "gas.json" in p["sha256"]


def test_caso_incompleto_lista_o_que_falta(tmp_path):
    caso = tmp_path / "caso"
    caso.mkdir()
    for nome in ("zdplaskin_m.F90", "bolsigdb.dat", "tabela_energia_tg.dat"):
        (caso / nome).write_text("x")
    r = subprocess.run([sys.executable, str(PROTOTIPO / "simula.py"), str(tmp_path / "x"), *BASE, "--t-end", "2e-9",
                        "--caso", str(caso)], capture_output=True, text=True,
                       env={"OPENBLAS_NUM_THREADS": "1", "PATH": ""})
    assert r.returncode != 0 and "Traceback" not in r.stderr
    for nome in ("tabela_eedf.dat", "transporte.json", "gas.json"):
        assert nome in r.stderr


def _copia_caso_n2(destino):
    import shutil
    shutil.copytree(RAIZ / "casos" / "n2", destino)
    return destino


@precisa_caso("n2")
def test_composicao_do_bolsig_diferente_do_fundo_e_erro(tmp_path):
    # the table has only the losses of the species with a positive fraction in
    # composicao.txt; the model excludes the reactions of the background of gas.json: the two
    # must coincide, otherwise losses are counted twice or not at all
    caso = _copia_caso_n2(tmp_path / "caso")
    (caso / "composicao.txt").write_text("N2 0.8\nO2 0.2\nN2^+ traco\n")
    r = subprocess.run([sys.executable, str(PROTOTIPO / "simula.py"), str(tmp_path / "x"), *BASE, "--t-end", "2e-9",
                        "--caso", str(caso)], capture_output=True, text=True,
                       env={"OPENBLAS_NUM_THREADS": "1", "PATH": ""})
    assert r.returncode != 0 and "Traceback" not in r.stderr
    assert "composicao.txt" in r.stderr and "O2" in r.stderr


@precisa_caso("n2")
def test_caso_n2_roda_pelo_simula(tmp_path):
    # N2 without code written for it: 2 ns, with the profile recorded with the
    # species of the case and the initial N2+ converted into N4+
    r = subprocess.run([sys.executable, str(PROTOTIPO / "simula.py"), str(tmp_path / "x"), *BASE, "--t-end", "2e-9",
                        "--caso", str(RAIZ / "casos" / "n2")], capture_output=True, text=True,
                       env={"OPENBLAS_NUM_THREADS": "1", "PATH": ""})
    assert r.returncode == 0, r.stderr[-2000:]
    tb, nomes, linhas = list(blocos(tmp_path / "x" / "perfil.dat"))[-1]
    for esp in ("N2^+", "N4^+", "N2(A)", "N"):
        assert esp in nomes
    a = np.array(linhas[1:-1])
    assert a[:, nomes.index("N4^+")].max() > a[:, nomes.index("N2^+")].max()


@precisa_caso("n2")
def test_composicao_com_frações_diferentes_do_fundo_e_erro(tmp_path):
    # in a mixture, the BOLSIG+ table holds for the fractions of composicao.txt;
    # gas.json needs the same ones
    from simula import confere_composicao
    caso = _copia_caso_n2(tmp_path / "caso")
    (caso / "composicao.txt").write_text("N2 0.5\nN 0.5\nN2^+ traco\n")
    g = json.loads((caso / "gas.json").read_text())
    g["fundo"] = {"N2": 0.75, "N": 0.25}
    (caso / "gas.json").write_text(json.dumps(g))
    msg = confere_composicao(caso)
    assert msg is not None and "fraction" in msg and "N2" in msg
    g["fundo"] = {"N2": 0.5, "N": 0.5}
    (caso / "gas.json").write_text(json.dumps(g))
    assert confere_composicao(caso) is None


def test_reflexao_de_eletrons_chega_ao_modelo(tmp_path):
    # --electron-reflection stays in the arguments and in the recorded model, and changes the electrons
    d, padrao = tmp_path / "r", tmp_path / "p"
    roda(d, "--t-end", "2e-9", "--reflexao-eletrons", "0.7")
    roda(padrao, "--t-end", "2e-9")
    p = json.loads((d / "parametros.json").read_text())
    assert p["argumentos"]["reflexao_eletrons"] == 0.7
    assert p["modelo"]["reflexao_eletrons"] == 0.7
    ne = [colunas(x, ["E"])[:, 0] for x in (d, padrao)]
    assert np.max(np.abs(ne[0] / ne[1] - 1.0)) > 1e-6


def test_reflexao_de_eletrons_fora_da_faixa_e_recusada(tmp_path):
    r = subprocess.run([sys.executable, str(PROTOTIPO / "simula.py"), str(tmp_path / "x"), "--t-end", "2e-9",
                        "--reflexao-eletrons", "1.0"], capture_output=True, text=True,
                       env={"OPENBLAS_NUM_THREADS": "1", "PATH": ""})
    assert r.returncode != 0 and "--electron-reflection" in r.stderr
    assert "Traceback" not in r.stderr


def test_nomes_antigos_das_opcoes_continuam_aceitos(tmp_path):
    # the options have English names; the Portuguese ones of earlier versions
    # still work and give the same result
    novo, antigo = tmp_path / "novo", tmp_path / "antigo"
    roda(novo, "--t-end", "2e-9", "--waveform", "sine", "--pressure", "9e4", "--gamma", "0.07")
    roda(antigo, "--t-end", "2e-9", "--forma", "seno", "--pressao", "9e4", "--gama", "0.07")
    for nome in ("superficie.dat", "perfil.dat"):
        assert (novo / nome).read_text() == (antigo / nome).read_text()
    a = json.loads((antigo / "parametros.json").read_text())["argumentos"]
    assert (a["pressao"], a["gama"], a["forma"]) == (9e4, 0.07, "seno")
