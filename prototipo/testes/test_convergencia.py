# SPDX-License-Identifier: GPL-3.0-or-later
"""Observáveis da frente de ionização usados no estudo de convergência
(scripts/convergencia.py), num perfil sintético com resposta conhecida."""
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "scripts"))
from convergencia import observaveis_frente, ordem_e_gci  # noqa: E402


def perfil_sintetico(cátodo_a_direita):
    # gap de 800 um com células de 1 um; Ar+ com platô de 2e12 atrás da frente e
    # um pico de 9e12 numa célula junto dela; elétrons acima de 1e12 até a frente
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
    # referência 0,25 um ao lado do centro do pico, para nenhuma célula ficar
    # exatamente na borda da janela de 60 um: entram 29 células de platô, o pico
    # e 30 células do lado de fora da frente, onde Ar+ vale 1e10
    x_ref = 599.75e-6 if direita else 800e-6 - 599.75e-6
    o = observaveis_frente(x, h, ne, ai, x_ref=x_ref)
    assert o["platô Ar+ (cm-3)"] == pytest.approx(2e12)
    esperado = (29 * 2e12 + 9e12 + 30 * 1e10) * 1e-6
    assert o["Ar+ em 60 um (cm-3 m)"] == pytest.approx(esperado, rel=1e-12)
    # a frente é a borda da região com n_e > 1e12 do lado do cátodo
    frente = 605e-6 if direita else 800e-6 - 605e-6
    assert o["frente (m)"] == pytest.approx(frente, abs=0.5e-6)


def test_scripts_leem_os_parametros_da_execucao(tmp_path):
    # G23: os scripts de análise tiram a geometria do parametros.json gravado
    # pelo simula.py; sem ele, ficam os padrões ou os valores dados
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
    # G24: o mesmo perfil com as colunas em outra ordem é igual; uma coluna
    # diferente é apontada pelo nome; sem as séries do Fortran, só o perfil
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
    assert r.returncode == 1 and "E " in r.stdout and "AR^+" not in r.stdout.split("DIFERE")[1]
    assert "extra" in r.stdout                      # coluna só de um lado é avisada


def test_compara_estrito(tmp_path):
    # modo estrito (make regressao): falha com série ausente de um lado, coluna
    # só de um lado (exceto CFL e n_eps, próprias de cada código) e instante só
    # de um lado; a última linha mais curta do Fortran (sem campo) não é erro
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
    """Execução sintética: perfil.dat com os nós das paredes e superficie.dat."""
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
    # revisão do desenho da campanha: v.update(perfil(...)) quebrava porque
    # perfil devolve (células, dicionário); sem n_e > 1e12 a frente é nan
    x = np.linspace(5e-6, 795e-6, 80)
    _execucao(tmp_path / "a", 1e-8, x, _colunas(80, fator=1.01))
    _execucao(tmp_path / "b", 1e-8, x, _colunas(80, ne=1e9))
    r = _roda("metricas.py", "--t", "1e-8", tmp_path / "a", tmp_path / "b")
    assert r.returncode == 0, r.stderr
    assert "int ne" in r.stdout and "frente" in r.stdout


@pytest.mark.parametrize("razao", [2.0, 1.5])
def test_ordem_e_gci_monotona(razao):
    # f = 1 + h, primeira ordem exata: p = 1, GCI de Roache com Fs = 1,25
    h = np.array([1.0, 1.0 / razao, 1.0 / razao ** 2])
    f = 1.0 + 0.01 * h
    r = ordem_e_gci(f, razao, ordem_formal=1.0)
    assert r["tipo"] == "monótona"
    assert r["ordens"][-1] == pytest.approx(1.0, rel=1e-9)
    assert r["gci"] == pytest.approx(1.25 * abs(f[1] - f[2]) / f[2] / (razao - 1.0), rel=1e-9)


def test_ordem_longe_da_formal_usa_fs_3_e_limita_p():
    # p observado 3 contra formal 1: Fs = 3 e p limitado à ordem formal
    f = 1.0 + 0.01 * np.array([1.0, 1.0 / 8, 1.0 / 64])
    r = ordem_e_gci(f, 2.0, ordem_formal=1.0)
    assert r["ordens"][-1] == pytest.approx(3.0, rel=1e-9)
    assert r["gci"] == pytest.approx(3.0 * abs(f[1] - f[2]) / f[2] / (2.0 - 1.0), rel=1e-9)


@pytest.mark.parametrize("f, tipo", [([1.0, 1.1, 1.05], "oscilatória"), ([1.0, 1.01, 1.03], "divergente")])
def test_ordem_classifica_e_nao_da_gci_sem_convergencia(f, tipo):
    r = ordem_e_gci(np.array(f), 2.0, ordem_formal=1.0)
    assert r["tipo"] == tipo and np.isnan(r["gci"])


def test_ordem_abaixo_do_piso_de_ruido():
    # diferença fina de 1e-6 relativa, abaixo de um piso de tolerância de 1e-4
    f = np.array([1.0, 1.0 + 1e-3, 1.0 + 1e-3 + 1e-6])
    r = ordem_e_gci(f, 2.0, ordem_formal=1.0, piso=1e-4)
    assert r["tipo"] == "abaixo do piso" and np.isnan(r["gci"])


def test_convergencia_observaveis_novos_e_frente_sem_eletrons(tmp_path):
    # ΔT máximo e integrado, Ar* e Ar2+ integrados; --frente sem n_e > 1e12 dá
    # nan em vez de quebrar (caso D)
    dirs = []
    for k, m in enumerate((40, 80, 160)):
        faces = np.linspace(0.0, 800e-6, m + 1)
        x = 0.5 * (faces[1:] + faces[:-1])
        dirs.append(tmp_path / f"n{m}")
        _execucao(dirs[-1], 1e-8, x, _colunas(m, ne=1e9, temperatura=300.0 + 3.0 * (1 + 0.1 / 2 ** k)))
    r = _roda("convergencia.py", "--t", "1e-8", "--frente", *dirs)
    assert r.returncode == 0, r.stderr
    for nome in ("max dT (K)", "int dT (K m)", "int Ar* (cm-3 m)", "int Ar2+ (cm-3 m)", "frente (m)"):
        assert nome in r.stdout, nome
    linha = next(l for l in r.stdout.splitlines() if l.startswith("max dT (K)"))
    assert float(linha.split()[-1]) == pytest.approx(3.0 * (1 + 0.1 / 4))


def test_ordem_sem_valor_e_indefinida():
    # frente nan em todas as malhas (caso D) não pode sair como monótona
    r = ordem_e_gci(np.array([np.nan, np.nan, np.nan]), 2.0)
    assert r["tipo"] == "indefinida" and np.isnan(r["gci"])


def test_ordem_perto_do_piso_nao_e_aceita():
    # a diferença fina carrega o ruído de duas execuções: entre 1 e 3 vezes o
    # piso, a ordem não é confiável (revisão da campanha de produção)
    f = np.array([1.0, 1.0 + 4e-3, 1.0 + 4e-3 + 2e-4])
    r = ordem_e_gci(f, 2.0, ordem_formal=1.0, piso=1e-4)
    assert r["tipo"] == "perto do piso" and np.isnan(r["gci"])
    f = np.array([1.0, 1.0 + 4e-3, 1.0 + 4e-3 + 4e-4])
    assert ordem_e_gci(f, 2.0, ordem_formal=1.0, piso=1e-4)["tipo"] == "monótona"


def test_metricas_mostra_todas_as_grandezas(tmp_path):
    # sigma_dir e E/N no centro saíam do relatório; a tolerância deles também
    # precisa ser medida
    x = np.linspace(5e-6, 795e-6, 80)
    _execucao(tmp_path / "a", 1e-8, x, _colunas(80, fator=1.01))
    _execucao(tmp_path / "b", 1e-8, x, _colunas(80))
    r = _roda("metricas.py", "--t", "1e-8", tmp_path / "a", tmp_path / "b")
    assert r.returncode == 0, r.stderr
    for nome in ("sigma_dir", "E/N no centro", "int Ar*", "int Ar2+", "max dT", "int dT"):
        assert nome in r.stdout, nome


# ---------------------------------------------------------- Eça e Hoekstra (K2)
from convergencia import incerteza_eca_hoekstra  # noqa: E402


def _h(n=5):
    return 0.5 ** np.arange(n)          # do mais grosso ao mais fino


def test_eh_dados_exatos_de_primeira_ordem():
    # phi = 1 + 0,1 h: ajuste livre exato, p = 1, Fs = 1,25 e U = 1,25 eps com
    # eps = |phi_ajuste(h_fina) - phi0| = 0,1 h_fina
    h = _h()
    r = incerteza_eca_hoekstra(1.0 + 0.1 * h, h, ordem_formal=1.0)
    assert r["ajuste"] == "livre" and r["fs"] == 1.25
    assert r["p"] == pytest.approx(1.0, abs=1e-3)
    assert r["phi0"] == pytest.approx(1.0, abs=1e-6)
    assert r["U"] == pytest.approx(1.25 * 0.1 * h[-1], rel=1e-3)


def test_eh_segunda_ordem_usa_o_ajuste_de_menor_desvio():
    # phi = 1 + 0,1 h^2 com ordem formal 1: p = 2 fora da faixa, Fs = 3 e o
    # ajuste de expoente fixo de menor desvio (h^2, exato): U = 3 * 0,1 h_fina^2
    h = _h()
    r = incerteza_eca_hoekstra(1.0 + 0.1 * h ** 2, h, ordem_formal=1.0)
    assert r["fs"] == 3.0 and r["ajuste"] == "h2"
    assert r["U"] == pytest.approx(3.0 * 0.1 * h[-1] ** 2, rel=1e-6)


def test_eh_mistura_de_primeira_e_segunda_ordem():
    # phi = 1 + 0,02 h + 0,2 h^2: o ajuste h + h^2 é exato e é o escolhido
    h = _h()
    f = 1.0 + 0.02 * h + 0.2 * h ** 2
    r = incerteza_eca_hoekstra(f, h, ordem_formal=1.0)
    assert r["ajuste"] == "h12" and r["fs"] == 3.0
    assert r["U"] == pytest.approx(3.0 * abs(f[-1] - 1.0), rel=1e-6)


def test_eh_ordem_baixa_usa_a_amplitude_dos_dados():
    # p = 0,3 (o regime do caso B): U = 3 Delta_M, a amplitude dos dados, que
    # cobre o erro verdadeiro (o ajuste com p fixo não cobria)
    h = _h()
    f = 1.0 + 0.1 * h ** 0.3
    r = incerteza_eca_hoekstra(f, h, ordem_formal=1.0)
    assert r["ajuste"] == "amplitude" and r["U"] == pytest.approx(3.0 * (f.max() - f.min()), rel=1e-12)
    assert r["U"] >= abs(f[-1] - 1.0)


def test_eh_desvio_com_os_graus_de_liberdade_do_ajuste():
    # sigma do ajuste com expoente fixo (2 parâmetros) usa n - 2
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
    # ruído maior que o espalhamento: U = 3 sigma/dphi (eps + sigma + |res|)
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
    # ramo limpo com resíduo: U = 1,25 eps + sigma + |residuo|, e o sigma do
    # ajuste livre tem 3 parâmetros (phi0, alfa e p): n - 3
    h = _h(6)
    f = 1.0 + 0.1 * h + np.array([1, -1, 1, -1, 1, -1]) * 2e-5
    r = incerteza_eca_hoekstra(f, h, ordem_formal=1.0)
    assert r["ajuste"] == "livre" and abs(r["residuo"]) > 0.0
    A = np.column_stack([np.ones(6), h ** r["p"]])
    c, *_ = np.linalg.lstsq(A, f, rcond=None)
    assert r["sigma"] == pytest.approx(np.sqrt(np.sum((f - A @ c) ** 2) / 3), rel=1e-6)
    assert r["U"] == pytest.approx(1.25 * r["eps"] + r["sigma"] + abs(r["residuo"]), rel=1e-12)


def test_eh_desvio_do_ajuste_de_dois_parametros():
    # dados de primeira ordem com ordem formal 0,5: p = 1 fora da faixa e o
    # ajuste h (2 parâmetros) é o de menor desvio: sigma com n - 2
    h = _h(6)
    f = 1.0 + 0.1 * h + np.array([1, -1, 1, -1, 1, -1]) * 2e-5
    r = incerteza_eca_hoekstra(f, h, ordem_formal=0.5)
    assert r["ajuste"] == "h1"
    A = np.column_stack([np.ones(6), h])
    c, *_ = np.linalg.lstsq(A, f, rcond=None)
    assert r["sigma"] == pytest.approx(np.sqrt(np.sum((f - A @ c) ** 2) / 4), rel=1e-9)


# ------------------------------------------- métricas de qualquer caso (N2)
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


def test_convergencia_usa_as_especies_do_caso(tmp_path):
    # N2: íon principal é o ion_inicial do gas.json (N2^+); cada espécie do perfil
    # entra integrada, menos o gás de fundo
    dirs = []
    for k, m in enumerate((40, 80, 160)):
        dirs.append(tmp_path / f"n{m}")
        _execucao_n2(dirs[-1], m, fator=1 + 0.1 / 2 ** k)
    r = _roda("convergencia.py", "--t", "1e-8", "--frente", *dirs)
    assert r.returncode == 0, r.stderr
    for nome in ("max N2^+ (cm-3)", "int N2^+ (cm-3 m)", "int N4^+ (cm-3 m)", "int N2(A) (cm-3 m)",
                 "int N (cm-3 m)", "platô N2^+ (cm-3)", "N2^+ em 60 um (cm-3 m)"):
        assert nome in r.stdout, nome
    assert "int N2 (cm-3 m)" not in r.stdout and "Ar" not in r.stdout
    linha = next(l for l in r.stdout.splitlines() if l.startswith("int N4^+ (cm-3 m)"))
    assert float(linha.split()[-1]) == pytest.approx(4e11 * (1 + 0.1 / 4) * 800e-6, rel=1e-9)


def test_serie_usa_as_especies_do_caso(tmp_path):
    _execucao_n2(tmp_path / "a", 80)
    _execucao_n2(tmp_path / "b", 80, fator=1.01)
    r = _roda("serie.py", "--referencia", tmp_path / "b", tmp_path / "a")
    assert r.returncode == 0, r.stderr
    assert "int N2^+ (cm-3 m)" in r.stdout and "int N2(A) (cm-3 m)" in r.stdout
    assert "int N2 (cm-3 m)" not in r.stdout


def test_metricas_usa_o_ion_do_caso(tmp_path):
    _execucao_n2(tmp_path / "a", 80)
    _execucao_n2(tmp_path / "b", 80, fator=1.01)
    r = _roda("metricas.py", "--t", "1e-8", tmp_path / "a", tmp_path / "b")
    assert r.returncode == 0, r.stderr
    assert "pico N2^+ em (um)" in r.stdout and "int N2(A) (cm-3 m)" in r.stdout


def test_caso_gravado_em_outra_maquina(tmp_path):
    # parametros.json do mercury: o caminho absoluto não existe aqui; caso da
    # pasta casos/ vira o casos/NOME local, a raiz vira a raiz local
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


def test_resumo_usa_as_especies_do_caso(tmp_path):
    _execucao_n2(tmp_path / "a", 40)
    r = _roda("resumo.py", tmp_path / "a")
    assert r.returncode == 0, r.stderr
    assert "N2^+" in r.stdout and "N4^+" in r.stdout and "NaN: 0" in r.stdout and "densidade negativa: 0" in r.stdout


def test_resumo_mostra_a_densidade_mais_negativa(tmp_path):
    # a contagem sozinha não diz se o negativo é ruído do integrador (abaixo do
    # atol) ou erro: o resumo mostra o valor mais negativo e o atol da execução
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
    assert "densidade negativa: 1, a mais negativa -2.5e-12 cm-3 (N4^+); atol 1e-10" in r.stdout
