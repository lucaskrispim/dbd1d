# SPDX-License-Identifier: GPL-3.0-or-later
"""Observables of the ionization front used in the convergence study
(scripts/convergencia.py), on a synthetic profile with a known answer."""
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "scripts"))
from convergencia import observaveis_frente, ordem_e_gci  # noqa: E402


def perfil_sintetico(cátodo_a_direita):
    # gap of 800 um with cells of 1 um; Ar+ with a plateau of 2e12 behind the front and
    # a peak of 9e12 in one cell next to it; electrons above 1e12 up to the front
    faces = np.linspace(0.0, 800e-6, 801)
    x = 0.5 * (faces[1:] + faces[:-1])
    h = np.diff(faces)
    um = x * 1e6
    ai = np.where(um < 600.0, 2e12, 1e10)
    ai[599] = 9e12
    ne = np.where(um < 605.0, 5e12, 1e9)
    if not cátodo_a_direita:
        x, h, ai, ne = 800e-6 - x[::-1], h[::-1], ai[::-1], ne[::-1]
    return x, h, ne, ai


@pytest.mark.parametrize("direita", [True, False])
def test_platô_janela_e_frente(direita):
    x, h, ne, ai = perfil_sintetico(direita)
    # reference 0.25 um beside the centre of the peak, so that no cell lies
    # exactly on the edge of the 60 um window: 29 plateau cells, the peak
    # and 30 cells outside the front enter, where Ar+ is 1e10
    x_ref = 599.75e-6 if direita else 800e-6 - 599.75e-6
    o = observaveis_frente(x, h, ne, ai, x_ref=x_ref)
    assert o["plateau Ar+ (cm-3)"] == pytest.approx(2e12)
    esperado = (29 * 2e12 + 9e12 + 30 * 1e10) * 1e-6
    assert o["Ar+ in 60 um (cm-3 m)"] == pytest.approx(esperado, rel=1e-12)
    # the front is the edge of the region with n_e > 1e12 on the cathode side
    frente = 605e-6 if direita else 800e-6 - 605e-6
    assert o["front (m)"] == pytest.approx(frente, abs=0.5e-6)


def test_scripts_leem_os_parametros_da_execucao(tmp_path):
    # G23: the analysis scripts take the geometry from the parametros.json written
    # by simula.py; without it, the defaults or the given values stay
    import json
    from resumo import parametros_da_execucao
    assert parametros_da_execucao(tmp_path) == {}
    (tmp_path / "parametros.json").write_text(json.dumps({"argumentos": {"espessura": 1.5e-3, "eps_dieletrico": 7.0}}))
    p = parametros_da_execucao(tmp_path)
    assert p["espessura"] == 1.5e-3 and p["eps_dieletrico"] == 7.0


def _perfil(d, nomes, linhas):
    d.mkdir()
    texto = f"{1e-9:24.16E}\n" + "".join(f"{n:>24s}" for n in nomes) + "\n"
    texto += "".join("".join(f" {v:24.16E}" for v in l) + "\n" for l in linhas) + "\n"
    (d / "perfil.dat").write_text(texto)


def test_compara_por_nome_de_coluna(tmp_path):
    # G24: the same profile with the columns in another order is equal; a different
    # column is pointed out by name; without the Fortran series, only the profile
    import subprocess
    script = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "compara.py"
    _perfil(tmp_path / "a", ["Pos", "E", "AR^+"], [[0.0, 1.0, 2.0], [1e-4, 3.0, 4.0]])
    _perfil(tmp_path / "b", ["Pos", "AR^+", "E"], [[0.0, 2.0, 1.0], [1e-4, 4.0, 3.0]])
    _perfil(tmp_path / "c", ["Pos", "AR^+", "E", "extra"], [[0.0, 2.0, 1.5, 9.0], [1e-4, 4.0, 3.0, 9.0]])
    r = subprocess.run([sys.executable, str(script), str(tmp_path / "a"), str(tmp_path / "b")],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout
    r = subprocess.run([sys.executable, str(script), str(tmp_path / "a"), str(tmp_path / "c")],
                       capture_output=True, text=True)
    assert r.returncode == 1 and "E " in r.stdout and "AR^+" not in r.stdout.split("DIFFERS")[1]
    assert "extra" in r.stdout                      # a column on one side only is reported


def test_compara_estrito(tmp_path):
    # strict mode (make regressao): fails with a series missing on one side, a column
    # on one side only (except CFL and n_eps, specific to each code) and an instant on
    # one side only; the shorter last line of the Fortran code (without field) is not an error
    import subprocess
    script = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "compara.py"
    def roda(a, b):
        return subprocess.run([sys.executable, str(script), str(a), str(b), "--estrito"],
                              capture_output=True, text=True)
    _perfil(tmp_path / "a", ["Pos", "E", "campo"], [[0.0, 1.0, 5.0], [1e-4, 3.0]])
    _perfil(tmp_path / "b", ["Pos", "E", "campo", "n_eps"], [[0.0, 1.0, 5.0, 7.0], [1e-4, 3.0, 6.0, 8.0]])
    assert roda(tmp_path / "a", tmp_path / "b").returncode == 0
    _perfil(tmp_path / "c", ["Pos", "E", "campo", "outra"], [[0.0, 1.0, 5.0, 7.0], [1e-4, 3.0, 6.0, 8.0]])
    assert roda(tmp_path / "a", tmp_path / "c").returncode == 1
    (tmp_path / "a" / "borda0.dat").write_text("Time\n0.0\n")
    assert roda(tmp_path / "a", tmp_path / "b").returncode == 1
    (tmp_path / "a" / "borda0.dat").unlink()
    _perfil(tmp_path / "d", ["Pos", "E", "campo"], [[0.0, 1.0, 5.0], [1e-4, 3.0]])
    with open(tmp_path / "d" / "perfil.dat", "a") as f:
        f.write(f"{2e-9:24.16E}\n" + "".join(f"{n:>24s}" for n in ["Pos", "E", "campo"]) + "\n"
                + f" {0.0:24.16E} {1.0:24.16E} {5.0:24.16E}\n\n")
    assert roda(tmp_path / "a", tmp_path / "d").returncode == 1


def _execucao(d, t, x, colunas, sigma=(1e-6, -1e-6), phi=(10.0, -10.0)):
    """Synthetic run: perfil.dat with the wall nodes and superficie.dat."""
    d.mkdir()
    nomes = ["Pos", *colunas]
    nos = np.concatenate([[0.0], x, [2.0 * x[-1] - x[-2] + 0.0 if len(x) > 1 else 1.0]])
    linhas = [[nos[i]] + [colunas[c][min(max(i - 1, 0), len(x) - 1)] for c in colunas] for i in range(len(nos))]
    texto = f"{t:24.16E}\n" + "".join(f"{n:>24s}" for n in nomes) + "\n"
    texto += "".join("".join(f" {v:24.16E}" for v in l) + "\n" for l in linhas) + "\n"
    (d / "perfil.dat").write_text(texto)
    (d / "superficie.dat").write_text("t V phi_esq phi_dir sigma_esq sigma_dir\n"
                                      + f"{t:.16e} 0.0 {phi[0]:.16e} {phi[1]:.16e} {sigma[0]:.16e} {sigma[1]:.16e}\n")


def _colunas(m, ne=5e12, temperatura=310.0, fator=1.0):
    return {"E": np.full(m, ne * fator), "AR^+": np.linspace(1e12, 3e12, m) * fator,
            "E/N": np.full(m, 50.0), "Temperature": np.full(m, temperatura),
            "AR*": np.full(m, 1e13 * fator), "AR2^+": np.full(m, 2e11 * fator)}


def _roda(script, *args):
    import subprocess
    caminho = pathlib.Path(__file__).resolve().parents[2] / "scripts" / script
    return subprocess.run([sys.executable, str(caminho), *map(str, args)], capture_output=True, text=True)


def test_metricas_roda_e_nao_quebra_sem_frente(tmp_path):
    # review of the campaign design: v.update(perfil(...)) broke because
    # perfil returns (cells, dictionary); without n_e > 1e12 the front is nan
    x = np.linspace(5e-6, 795e-6, 80)
    _execucao(tmp_path / "a", 1e-8, x, _colunas(80, fator=1.01))
    _execucao(tmp_path / "b", 1e-8, x, _colunas(80, ne=1e9))
    r = _roda("metricas.py", "--t", "1e-8", tmp_path / "a", tmp_path / "b")
    assert r.returncode == 0, r.stderr
    assert "int ne" in r.stdout and "front" in r.stdout


@pytest.mark.parametrize("razao", [2.0, 1.5])
def test_ordem_e_gci_monotona(razao):
    # f = 1 + h, exact first order: p = 1, Roache GCI with Fs = 1.25
    h = np.array([1.0, 1.0 / razao, 1.0 / razao ** 2])
    f = 1.0 + 0.01 * h
    r = ordem_e_gci(f, razao, ordem_formal=1.0)
    assert r["tipo"] == "monotonic"
    assert r["ordens"][-1] == pytest.approx(1.0, rel=1e-9)
    assert r["gci"] == pytest.approx(1.25 * abs(f[1] - f[2]) / f[2] / (razao - 1.0), rel=1e-9)


def test_ordem_longe_da_formal_usa_fs_3_e_limita_p():
    # observed p 3 against formal 1: Fs = 3 and p bounded by the formal order
    f = 1.0 + 0.01 * np.array([1.0, 1.0 / 8, 1.0 / 64])
    r = ordem_e_gci(f, 2.0, ordem_formal=1.0)
    assert r["ordens"][-1] == pytest.approx(3.0, rel=1e-9)
    assert r["gci"] == pytest.approx(3.0 * abs(f[1] - f[2]) / f[2] / (2.0 - 1.0), rel=1e-9)


@pytest.mark.parametrize("f, tipo", [([1.0, 1.1, 1.05], "oscillatory"), ([1.0, 1.01, 1.03], "divergent")])
def test_ordem_classifica_e_nao_da_gci_sem_convergencia(f, tipo):
    r = ordem_e_gci(np.array(f), 2.0, ordem_formal=1.0)
    assert r["tipo"] == tipo and np.isnan(r["gci"])


def test_ordem_abaixo_do_piso_de_ruido():
    # fine difference of 1e-6 relative, below a tolerance floor of 1e-4
    f = np.array([1.0, 1.0 + 1e-3, 1.0 + 1e-3 + 1e-6])
    r = ordem_e_gci(f, 2.0, ordem_formal=1.0, piso=1e-4)
    assert r["tipo"] == "below the floor" and np.isnan(r["gci"])


def test_convergencia_observaveis_novos_e_frente_sem_eletrons(tmp_path):
    # maximum and integrated ΔT, integrated Ar* and Ar2+; --front without n_e > 1e12 gives
    # nan instead of breaking (case D)
    dirs = []
    for k, m in enumerate((40, 80, 160)):
        faces = np.linspace(0.0, 800e-6, m + 1)
        x = 0.5 * (faces[1:] + faces[:-1])
        dirs.append(tmp_path / f"n{m}")
        _execucao(dirs[-1], 1e-8, x, _colunas(m, ne=1e9, temperatura=300.0 + 3.0 * (1 + 0.1 / 2 ** k)))
    r = _roda("convergencia.py", "--t", "1e-8", "--frente", *dirs)
    assert r.returncode == 0, r.stderr
    for nome in ("max dT (K)", "int dT (K m)", "int Ar* (cm-3 m)", "int Ar2+ (cm-3 m)", "front (m)"):
        assert nome in r.stdout, nome
    linha = next(l for l in r.stdout.splitlines() if l.startswith("max dT (K)"))
    assert float(linha.split()[-1]) == pytest.approx(3.0 * (1 + 0.1 / 4))


def test_ordem_sem_valor_e_indefinida():
    # front nan on every mesh (case D) cannot come out as monotonic
    r = ordem_e_gci(np.array([np.nan, np.nan, np.nan]), 2.0)
    assert r["tipo"] == "undefined" and np.isnan(r["gci"])


def test_ordem_perto_do_piso_nao_e_aceita():
    # the fine difference carries the noise of two runs: between 1 and 3 times the
    # floor, the order is not reliable (review of the production campaign)
    f = np.array([1.0, 1.0 + 4e-3, 1.0 + 4e-3 + 2e-4])
    r = ordem_e_gci(f, 2.0, ordem_formal=1.0, piso=1e-4)
    assert r["tipo"] == "near the floor" and np.isnan(r["gci"])
    f = np.array([1.0, 1.0 + 4e-3, 1.0 + 4e-3 + 4e-4])
    assert ordem_e_gci(f, 2.0, ordem_formal=1.0, piso=1e-4)["tipo"] == "monotonic"


def test_metricas_mostra_todas_as_grandezas(tmp_path):
    # sigma_right and E/N at the centre were missing from the report; their tolerance also
    # needs to be measured
    x = np.linspace(5e-6, 795e-6, 80)
    _execucao(tmp_path / "a", 1e-8, x, _colunas(80, fator=1.01))
    _execucao(tmp_path / "b", 1e-8, x, _colunas(80))
    r = _roda("metricas.py", "--t", "1e-8", tmp_path / "a", tmp_path / "b")
    assert r.returncode == 0, r.stderr
    for nome in ("sigma_dir", "E/N at centre", "int Ar*", "int Ar2+", "max dT", "int dT"):
        assert nome in r.stdout, nome


# ---------------------------------------------------------- Eça and Hoekstra (K2)
from convergencia import incerteza_eca_hoekstra  # noqa: E402

from conftest import precisa_arquivo


def _h(n=5):
    return 0.5 ** np.arange(n)          # from the coarsest to the finest


def test_eh_dados_exatos_de_primeira_ordem():
    # phi = 1 + 0.1 h: exact free fit, p = 1, Fs = 1.25 and U = 1.25 eps with
    # eps = |phi_fit(h_fine) - phi0| = 0.1 h_fine
    h = _h()
    r = incerteza_eca_hoekstra(1.0 + 0.1 * h, h, ordem_formal=1.0)
    assert r["ajuste"] == "free" and r["fs"] == 1.25
    assert r["p"] == pytest.approx(1.0, abs=1e-3)
    assert r["phi0"] == pytest.approx(1.0, abs=1e-6)
    assert r["U"] == pytest.approx(1.25 * 0.1 * h[-1], rel=1e-3)


def test_eh_segunda_ordem_usa_o_ajuste_de_menor_desvio():
    # phi = 1 + 0.1 h^2 with formal order 1: p = 2 out of range, Fs = 3 and the
    # fixed-exponent fit of smallest deviation (h^2, exact): U = 3 * 0.1 h_fine^2
    h = _h()
    r = incerteza_eca_hoekstra(1.0 + 0.1 * h ** 2, h, ordem_formal=1.0)
    assert r["fs"] == 3.0 and r["ajuste"] == "h2"
    assert r["U"] == pytest.approx(3.0 * 0.1 * h[-1] ** 2, rel=1e-6)


def test_eh_mistura_de_primeira_e_segunda_ordem():
    # phi = 1 + 0.02 h + 0.2 h^2: the fit h + h^2 is exact and is the chosen one
    h = _h()
    f = 1.0 + 0.02 * h + 0.2 * h ** 2
    r = incerteza_eca_hoekstra(f, h, ordem_formal=1.0)
    assert r["ajuste"] == "h12" and r["fs"] == 3.0
    assert r["U"] == pytest.approx(3.0 * abs(f[-1] - 1.0), rel=1e-6)


def test_eh_ordem_baixa_usa_a_amplitude_dos_dados():
    # p = 0.3 (the regime of case B): U = 3 Delta_M, the amplitude of the data, which
    # covers the true error (the fit with fixed p did not)
    h = _h()
    f = 1.0 + 0.1 * h ** 0.3
    r = incerteza_eca_hoekstra(f, h, ordem_formal=1.0)
    assert r["ajuste"] == "amplitude" and r["U"] == pytest.approx(3.0 * (f.max() - f.min()), rel=1e-12)
    assert r["U"] >= abs(f[-1] - 1.0)


def test_eh_desvio_com_os_graus_de_liberdade_do_ajuste():
    # sigma of the fit with fixed exponent (2 parameters) uses n - 2
    h = _h(6)
    f = 1.0 + 0.1 * h ** 2 + np.array([1, -1, 1, -1, 1, -1]) * 1e-4
    r = incerteza_eca_hoekstra(f, h, ordem_formal=1.0)
    npar = {"h1": 2, "h2": 2, "h12": 3}[r["ajuste"]]
    cols = {"h1": [h], "h2": [h ** 2], "h12": [h, h ** 2]}[r["ajuste"]]
    A = np.column_stack([np.ones(6), *cols])
    c, *_ = np.linalg.lstsq(A, f, rcond=None)
    res = f - A @ c
    assert r["sigma"] == pytest.approx(np.sqrt(np.sum(res ** 2) / (6 - npar)), rel=1e-9)


def test_eh_dados_ruidosos_ampliam_a_incerteza():
    # noise larger than the spread: U = 3 sigma/dphi (eps + sigma + |res|)
    h = _h(6)
    ruido = np.array([1, -1, 1, -1, 1, -1]) * 4e-3
    f = 1.0 + 0.01 * h + ruido
    r = incerteza_eca_hoekstra(f, h, ordem_formal=1.0)
    dphi = (f.max() - f.min()) / (len(f) - 1)
    assert r["sigma"] > dphi
    esperado = 3.0 * r["sigma"] / dphi * (r["eps"] + r["sigma"] + abs(r["residuo"]))
    assert r["U"] == pytest.approx(esperado, rel=1e-12) and r["U"] > abs(f[-1] - 1.0)


def test_eh_dados_constantes():
    r = incerteza_eca_hoekstra(np.full(5, 2.0), _h(), ordem_formal=1.0)
    assert r["U"] == 0.0


def test_eh_com_tres_malhas_nao_ajusta():
    r = incerteza_eca_hoekstra(np.array([1.2, 1.1, 1.05]), _h(3), ordem_formal=1.0)
    assert np.isnan(r["U"])


def test_convergencia_imprime_a_incerteza_eh(tmp_path):
    dirs = []
    for k, m in enumerate((40, 80, 160, 320)):
        faces = np.linspace(0.0, 800e-6, m + 1)
        x = 0.5 * (faces[1:] + faces[:-1])
        dirs.append(tmp_path / f"n{m}")
        _execucao(dirs[-1], 1e-8, x, _colunas(m, temperatura=300.0 + 3.0 * (1 + 0.1 / 2 ** k)))
    r = _roda("convergencia.py", "--t", "1e-8", *dirs)
    assert r.returncode == 0, r.stderr
    linha = next(l for l in r.stdout.splitlines() if "max dT" in l)
    i = r.stdout.splitlines().index(linha)
    assert "U (E-H)" in r.stdout.splitlines()[i + 1]


def test_eh_ajuste_livre_com_ruido_pequeno():
    # clean branch with residual: U = 1.25 eps + sigma + |residual|, and the sigma of the
    # free fit has 3 parameters (phi0, alpha and p): n - 3
    h = _h(6)
    f = 1.0 + 0.1 * h + np.array([1, -1, 1, -1, 1, -1]) * 2e-5
    r = incerteza_eca_hoekstra(f, h, ordem_formal=1.0)
    assert r["ajuste"] == "free" and abs(r["residuo"]) > 0.0
    A = np.column_stack([np.ones(6), h ** r["p"]])
    c, *_ = np.linalg.lstsq(A, f, rcond=None)
    assert r["sigma"] == pytest.approx(np.sqrt(np.sum((f - A @ c) ** 2) / 3), rel=1e-6)
    assert r["U"] == pytest.approx(1.25 * r["eps"] + r["sigma"] + abs(r["residuo"]), rel=1e-12)


def test_eh_desvio_do_ajuste_de_dois_parametros():
    # first order data with formal order 0.5: p = 1 out of range and the
    # fit h (2 parameters) is the one of smallest deviation: sigma with n - 2
    h = _h(6)
    f = 1.0 + 0.1 * h + np.array([1, -1, 1, -1, 1, -1]) * 2e-5
    r = incerteza_eca_hoekstra(f, h, ordem_formal=0.5)
    assert r["ajuste"] == "h1"
    A = np.column_stack([np.ones(6), h])
    c, *_ = np.linalg.lstsq(A, f, rcond=None)
    assert r["sigma"] == pytest.approx(np.sqrt(np.sum((f - A @ c) ** 2) / 4), rel=1e-9)


# ------------------------------------------- metrics of any case (N2)
def _execucao_n2(d, m, fator=1.0):
    import json
    faces = np.linspace(0.0, 800e-6, m + 1)
    x = 0.5 * (faces[1:] + faces[:-1])
    colunas = {"Temperature": np.full(m, 310.0), "E": np.full(m, 5e12 * fator),
               "N": np.full(m, 1e13 * fator), "N2": np.full(m, 2.4e19),
               "N2(A)": np.full(m, 3e13 * fator), "N2^+": np.linspace(1e12, 3e12, m) * fator,
               "N4^+": np.full(m, 4e11 * fator), "E/N": np.full(m, 50.0)}
    _execucao(d, 1e-8, x, colunas)
    caso = pathlib.Path(__file__).resolve().parents[2] / "casos" / "n2"
    (d / "parametros.json").write_text(json.dumps({"argumentos": {"caso": str(caso), "t_parede": 300.0}}))


@precisa_arquivo("casos/n2/kinet.inp")
def test_convergencia_usa_as_especies_do_caso(tmp_path):
    # N2: the main ion is the ion_inicial of gas.json (N2^+); each species of the profile
    # enters integrated, except the background gas
    dirs = []
    for k, m in enumerate((40, 80, 160)):
        dirs.append(tmp_path / f"n{m}")
        _execucao_n2(dirs[-1], m, fator=1 + 0.1 / 2 ** k)
    r = _roda("convergencia.py", "--t", "1e-8", "--frente", *dirs)
    assert r.returncode == 0, r.stderr
    for nome in ("max N2^+ (cm-3)", "int N2^+ (cm-3 m)", "int N4^+ (cm-3 m)", "int N2(A) (cm-3 m)",
                 "int N (cm-3 m)", "plateau N2^+ (cm-3)", "N2^+ in 60 um (cm-3 m)"):
        assert nome in r.stdout, nome
    assert "int N2 (cm-3 m)" not in r.stdout and "Ar" not in r.stdout
    linha = next(l for l in r.stdout.splitlines() if l.startswith("int N4^+ (cm-3 m)"))
    assert float(linha.split()[-1]) == pytest.approx(4e11 * (1 + 0.1 / 4) * 800e-6, rel=1e-9)


@precisa_arquivo("casos/n2/kinet.inp")
def test_serie_usa_as_especies_do_caso(tmp_path):
    _execucao_n2(tmp_path / "a", 80)
    _execucao_n2(tmp_path / "b", 80, fator=1.01)
    r = _roda("serie.py", "--referencia", tmp_path / "b", tmp_path / "a")
    assert r.returncode == 0, r.stderr
    assert "int N2^+ (cm-3 m)" in r.stdout and "int N2(A) (cm-3 m)" in r.stdout
    assert "int N2 (cm-3 m)" not in r.stdout


@precisa_arquivo("casos/n2/kinet.inp")
def test_metricas_usa_o_ion_do_caso(tmp_path):
    _execucao_n2(tmp_path / "a", 80)
    _execucao_n2(tmp_path / "b", 80, fator=1.01)
    r = _roda("metricas.py", "--t", "1e-8", tmp_path / "a", tmp_path / "b")
    assert r.returncode == 0, r.stderr
    assert "peak N2^+ at (um)" in r.stdout and "int N2(A) (cm-3 m)" in r.stdout


@precisa_arquivo("casos/n2/kinet.inp")
def test_caso_gravado_em_outra_maquina(tmp_path):
    # parametros.json from mercury: the absolute path does not exist here; a case of the
    # casos/ folder becomes the local casos/NAME, the root becomes the local root
    import json
    from resumo import ErroCaso, especies_do_caso
    for caso, ion in (("/maquina/remota/argonio/casos/n2", "N2^+"), ("/maquina/remota/argonio", "AR^+")):
        d = tmp_path / ion.replace("^", "")
        d.mkdir()
        (d / "parametros.json").write_text(json.dumps({"argumentos": {"caso": caso}}))
        assert especies_do_caso(d)[1] == ion
    (d / "parametros.json").write_text(json.dumps({"argumentos": {"caso": "/outra/casos/nao_existe"}}))
    with pytest.raises(ErroCaso, match="nao_existe"):
        especies_do_caso(d)


def test_kinet_sem_species_da_erro_claro(tmp_path):
    import json
    from resumo import ErroCaso, especies_do_caso
    (tmp_path / "kinet.inp").write_text("ELEMENTS\ne Ar\nEND\n")
    (tmp_path / "parametros.json").write_text(json.dumps({"argumentos": {"caso": str(tmp_path)}}))
    with pytest.raises(ErroCaso, match="SPECIES"):
        especies_do_caso(tmp_path)


@precisa_arquivo("casos/n2/kinet.inp")
def test_resumo_usa_as_especies_do_caso(tmp_path):
    _execucao_n2(tmp_path / "a", 40)
    r = _roda("resumo.py", tmp_path / "a")
    assert r.returncode == 0, r.stderr
    assert "N2^+" in r.stdout and "N4^+" in r.stdout and "NaN: 0" in r.stdout and "negative density: 0" in r.stdout


@precisa_arquivo("casos/n2/kinet.inp")
def test_resumo_mostra_a_densidade_mais_negativa(tmp_path):
    # the count alone does not say whether the negative value is integrator noise (below
    # atol) or an error: the summary shows the most negative value and the atol of the run
    import json
    _execucao_n2(tmp_path / "a", 40)
    texto = (tmp_path / "a" / "perfil.dat").read_text().splitlines()
    nomes = [texto[1][i:i + 24].strip() for i in range(0, len(texto[1]), 24)]
    campos = texto[5].split()
    campos[nomes.index("N4^+")] = "-2.5000000000000000E-12"
    texto[5] = "".join(f" {c:>24s}" for c in campos)
    (tmp_path / "a" / "perfil.dat").write_text("\n".join(texto) + "\n")
    par = json.loads((tmp_path / "a" / "parametros.json").read_text())
    par["argumentos"]["atol"] = 1e-10
    (tmp_path / "a" / "parametros.json").write_text(json.dumps(par))
    r = _roda("resumo.py", tmp_path / "a")
    assert r.returncode == 0, r.stderr
    assert "negative density: 1, the most negative -2.5e-12 cm-3 (N4^+); atol 1e-10" in r.stdout
