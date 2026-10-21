# SPDX-License-Identifier: GPL-3.0-or-later
"""Observables of the time series (scripts/serie.py): energy per period from the
Lissajous curve, maximum difference along time against a reference and first
validity warning in the log."""
import json
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "scripts"))
from serie import carga_por_semiperiodo, diferenca_maxima, energia_por_periodo, primeiro_aviso  # noqa: E402

EPS0 = 8.8541878128e-12


def test_energia_por_periodo_de_lissajous():
    # Q = A sin(wt - delta) per area at the electrode: ∮ V dQ = pi V0 A sin(delta)
    periodo, v0, a, delta = 1e-7, 5e3, 2e-4, 0.3
    t = np.linspace(0.0, 2 * periodo, 4001)
    v = v0 * np.sin(2 * np.pi * t / periodo)
    q = a * np.sin(2 * np.pi * t / periodo - delta)
    esp, eps_d = 1.75e-3, 4.8
    phi_esq = v - q * esp / (EPS0 * eps_d)
    e = energia_por_periodo(t, v, phi_esq, esp, eps_d, periodo)
    assert len(e) == 2
    np.testing.assert_allclose(e, np.pi * v0 * a * np.sin(delta), rtol=1e-5)


def test_energia_sem_dissipacao_e_nula():
    periodo = 1e-7
    t = np.linspace(0.0, periodo, 2001)
    v = 3e3 * np.sin(2 * np.pi * t / periodo)
    phi_esq = 0.4 * v                                # Q in phase with V: pure capacitor
    assert abs(energia_por_periodo(t, v, phi_esq, 1e-3, 10.0, periodo)[0]) < 1e-12


def test_diferenca_maxima_normalizada_pelo_maximo_da_referencia():
    # the difference of 0.5 at the instant the reference crosses zero does not blow up:
    # it is divided by the maximum of the series (10)
    t = np.arange(5.0)
    ref = {"x": np.array([0.0, 10.0, 0.0, -10.0, 0.0])}
    outra = {"x": np.array([0.0, 10.0, 0.5, -10.0, 0.0])}
    d = diferenca_maxima(t, outra, t, ref)
    assert d["x"] == pytest.approx((0.05, 2.0))


def test_diferenca_maxima_so_nos_instantes_comuns():
    d = diferenca_maxima(np.array([0.0, 1.0, 2.0]), {"x": np.array([1.0, 2.0, 9.0])},
                         np.array([0.0, 1.0]), {"x": np.array([1.0, 2.5])})
    assert d["x"] == pytest.approx((0.2, 1.0))


def test_primeiro_aviso(tmp_path):
    log = tmp_path / "caso.log"
    log.write_text("t = 1.0e-09 s ...\nAVISO: gas acima de 1000 K em t = 8.3e-08 s, no 2 (x = 1.00 um)\n"
                   "AVISO: grau de ionizacao acima de 1e-3 em t = 8.1e-08 s, no 2 (x = 1.00 um)\n")
    assert primeiro_aviso(log) == pytest.approx(8.1e-8)
    log.write_text("t = 1.0e-09 s ...\ntempo total: 3 s\n")
    assert primeiro_aviso(log) is None
    # energy outside the table is a diagnostic, not a validity warning
    log.write_text("AVISO: energia media fora da tabela em t = 5.0000e-09 s: 0 celulas acima, 41 abaixo\n")
    assert primeiro_aviso(log) is None
    # the warning of the Fortran code also counts
    log.write_text("WARNING: ionization degree above 1e-3 at t = 7.5e-08 s, node 2 (x = 1.00 um)\n"
                   "WARNING: gas above 1000 K at t = 9.0e-08 s, node 2 (x = 1.00 um)\n")
    assert primeiro_aviso(log) == pytest.approx(7.5e-8)     # logs of the English version
    log.write_text(" AVISO: grau de ionizacao acima de 1e-3 em t =  8.316E-08 s, celula 2\n")
    assert primeiro_aviso(log) == pytest.approx(8.316e-8)


def test_script_le_a_execucao(tmp_path):
    # synthetic run with two instants; the script prints the energy per
    # period and the maximum difference against the reference
    import subprocess
    def execucao(d, fator):
        d.mkdir()
        (d / "parametros.json").write_text(json.dumps({"argumentos": {
            "espessura": 1.75e-3, "eps_dieletrico": 4.8, "periodo": 1e-7, "t_parede": 300.0}}))
        t = np.linspace(0.0, 1e-7, 101)
        v = 5e3 * np.sin(2 * np.pi * t / 1e-7)
        linhas = "".join(f"{a:.16e} {b:.16e} {0.9 * b * fator:.16e} 0.0 0.0 0.0\n" for a, b in zip(t, v))
        (d / "superficie.dat").write_text("t V phi_esq phi_dir sigma_esq sigma_dir\n" + linhas)
        texto = ""
        for tb in (0.0, 1e-7):
            texto += f"{tb:24.16E}\n" + "".join(f"{n:>24s}" for n in ("Pos", "E", "AR^+", "Temperature")) + "\n"
            for x in (0.0, 1e-4, 3e-4, 4e-4):
                texto += f" {x:24.16E} {1e9 * fator:24.16E} {2e9:24.16E} {301.0:24.16E}\n"
            texto += "\n"
        (d / "perfil.dat").write_text(texto)
    execucao(tmp_path / "a", 1.0)
    execucao(tmp_path / "ref", 1.1)
    r = subprocess.run([sys.executable, str(pathlib.Path(__file__).resolve().parents[2] / "scripts" / "serie.py"),
                        str(tmp_path / "a"), "--referencia", str(tmp_path / "ref")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "energy per period" in r.stdout and "int ne" in r.stdout and "no warning" in r.stdout


def test_carga_por_semiperiodo():
    # sigma = t/T sampled every T/8: the change is 1/2 per half period; the
    # last quarter period does not close a half period and is left out
    periodo = 2e-5
    t = np.linspace(0.0, 1.25 * periodo, 11)
    np.testing.assert_allclose(carga_por_semiperiodo(t, t / periodo, periodo), [0.5, 0.5], rtol=1e-12)


def test_script_convergencia_por_periodo(tmp_path):
    # three synthetic meshes with energy and charge per half period first
    # order in the mesh: the script classifies as monotonic with order 1
    import subprocess
    dirs = []
    for k in range(3):
        d = tmp_path / f"n{k}"
        d.mkdir()
        (d / "parametros.json").write_text(json.dumps({"argumentos": {
            "espessura": 1.75e-3, "eps_dieletrico": 4.8, "periodo": 1e-7}}))
        t = np.linspace(0.0, 1e-7, 201)
        v = 5e3 * np.sin(2 * np.pi * t / 1e-7)
        fase = 0.3 * (1 + 0.1 / 2 ** k)
        q = 2e-4 * np.sin(2 * np.pi * t / 1e-7 - fase)
        phi = v - q * 1.75e-3 / (EPS0 * 4.8)
        sigma = 1e-5 * (1 + 0.1 / 2 ** k) * t / 1e-7
        (d / "superficie.dat").write_text("t V phi_esq phi_dir sigma_esq sigma_dir\n" + "".join(
            f"{a:.16e} {b:.16e} {c:.16e} 0.0 {e:.16e} 0.0\n" for a, b, c, e in zip(t, v, phi, sigma)))
        texto = f"{0.0:24.16E}\n" + "".join(f"{n:>24s}" for n in ("Pos", "E", "AR^+")) + "\n"
        texto += "".join(f" {x:24.16E} {1.0:24.16E} {1.0:24.16E}\n" for x in (0.0, 1e-4, 2e-4)) + "\n"
        (d / "perfil.dat").write_text(texto)
        dirs.append(d)
    r = subprocess.run([sys.executable, str(pathlib.Path(__file__).resolve().parents[2] / "scripts" / "serie.py"),
                        *map(str, dirs), "--malhas"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    linhas = [l for l in r.stdout.splitlines() if "monotonic" in l]
    assert any("energy period 1" in l for l in linhas) and any("charge half period 1" in l for l in linhas)
    assert all(" 1.00 " in l for l in linhas if "carga" in l)
    assert all("U (E-H)" in l for l in r.stdout.splitlines() if "semiperíodo" in l and "carga" in l and "(C/m2)" not in l)
    # with a tolerance floor above the fine difference (2.5% of 1.1e-5), the
    # charge falls below the floor
    r = subprocess.run([sys.executable, str(pathlib.Path(__file__).resolve().parents[2] / "scripts" / "serie.py"),
                        *map(str, dirs), "--malhas", "--piso", "0.05"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert all("below the floor" in l for l in r.stdout.splitlines() if "charge half period" in l)


@pytest.mark.parametrize("forma, tem, nao_tem", [
    ("degrau", ["no quantities per period (step waveform)"], ["energy per period", "charge per half period"]),
    ("pulso", ["energy per period"], ["charge per half period"]),
])
def test_metricas_por_periodo_conforme_a_forma(tmp_path, forma, tem, nao_tem):
    # a step has no period; a pulse has the energy per repetition period, but
    # no polarity half periods
    import subprocess
    d = tmp_path / "a"
    d.mkdir()
    (d / "parametros.json").write_text(json.dumps({"argumentos": {
        "espessura": 1e-3, "eps_dieletrico": 4.8, "periodo": 1e-7, "t_parede": 300.0, "forma": forma}}))
    t = np.linspace(0.0, 1e-7, 101)
    v = np.where(t > 1e-9, 5e3, 5e3 * t / 1e-9)
    (d / "superficie.dat").write_text("t V phi_esq phi_dir sigma_esq sigma_dir\n" + "".join(
        f"{a:.16e} {b:.16e} {0.9 * b:.16e} 0.0 0.0 0.0\n" for a, b in zip(t, v)))
    texto = ""
    for tb in (0.0, 1e-7):
        texto += f"{tb:24.16E}\n" + "".join(f"{n:>24s}" for n in ("Pos", "E", "AR^+", "Temperature")) + "\n"
        for x in (0.0, 1e-4, 3e-4, 4e-4):
            texto += f" {x:24.16E} {1e9:24.16E} {2e9:24.16E} {301.0:24.16E}\n"
        texto += "\n"
    (d / "perfil.dat").write_text(texto)
    r = subprocess.run([sys.executable, str(pathlib.Path(__file__).resolve().parents[2] / "scripts" / "serie.py"),
                        str(d)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    for s in tem:
        assert s in r.stdout, s
    for s in nao_tem:
        assert s not in r.stdout, s
