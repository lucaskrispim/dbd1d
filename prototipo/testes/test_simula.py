# SPDX-License-Identifier: GPL-3.0-or-later
"""Teste de ponta a ponta do simula.py: um caso pequeno roda, grava as saídas no
formato do Fortran e a retomada a partir de um instante gravado reproduz a
execução contínua até a tolerância do integrador (com a DAE, a execução
contínua não reinicia o integrador nas saídas e a retomada reinicia)."""
import json
import subprocess
import sys

import numpy as np
import pytest

from conftest import PROTOTIPO, RAIZ, quimica_do, precisa_caso

sys.path.insert(0, str(RAIZ / "scripts"))
from resumo import blocos  # noqa: E402

BASE = ["--v-pico", "5e3", "--n-celulas", "20", "--agrupamento", "1.5", "--espessura", "1.75e-3",
        "--energia", "--dt-saida", "2e-9", "--ne0", "1e9", "--neutro"]


def roda(destino, *extra):
    subprocess.run([sys.executable, str(PROTOTIPO / "simula.py"), str(destino), *BASE, *extra],
                   check=True, capture_output=True, env={"OPENBLAS_NUM_THREADS": "1", "PATH": ""})


def superficie(d):
    return np.loadtxt(d / "superficie.dat", skiprows=1)


DENSIDADES = ["E", "AR*", "AR^+", "AR2^+", "E/N"]


def iguais(a, b, rtol, fator_atol):
    """Compara coluna a coluna, com tolerância absoluta relativa ao máximo da coluna."""
    for j in range(a.shape[1]):
        np.testing.assert_allclose(a[:, j], b[:, j], rtol=rtol, atol=fator_atol * np.abs(b[:, j]).max())


def colunas(d, nomes_col, k=-1):
    """Colunas pelo nome no bloco k do perfil, só as células."""
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
    assert len(perfis) == 5 and len(perfis[-1][2]) == 22   # 20 células e os dois nós das paredes
    np.testing.assert_allclose(superficie(resto)[-1], s[-1], rtol=1e-5, atol=1e-20)
    p_resto, p_cont = colunas(resto, DENSIDADES), colunas(continua, DENSIDADES)
    iguais(p_resto, p_cont, 1e-4, 1e-4)


def test_dae_reproduz_o_caminho_original(tmp_path):
    # mesma física pelos dois caminhos: jacobiano por diferenças finitas do scipy
    # com o Poisson dentro do lado direito, e DAE com o potencial no estado e
    # jacobiano estruturado; com tolerância apertada, as saídas coincidem
    fd, dae = tmp_path / "fd", tmp_path / "dae"
    roda(fd, "--t-end", "8e-9", "--jac", "fd", "--rtol", "1e-7")
    roda(dae, "--t-end", "8e-9", "--jac", "dae", "--rtol", "1e-7")
    s_fd, s_dae = superficie(fd), superficie(dae)
    np.testing.assert_allclose(s_dae[:, 2:4], s_fd[:, 2:4], rtol=1e-5)
    p_fd, p_dae = colunas(fd, DENSIDADES), colunas(dae, DENSIDADES)
    iguais(p_dae, p_fd, 1e-4, 1e-3)


def test_aquecimento_grava_e_retoma_a_temperatura(tmp_path):
    # com --aquecimento a coluna Temperature começa em 300 K, só
    # cresce com a descarga e é lida de volta na retomada
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
    # como o perfil do Fortran: De, Me e ve = Me*E nas células e, com
    # aquecimento, kappa e rho*cv
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
    # frequência de colisão como o Fortran: (nu/N) da tabela vezes N total da
    # célula (gás e todas as espécies); aqui pela energia média de Te
    from tabela import E_NUN, TabelaEnergia
    tab = TabelaEnergia(RAIZ / "tabela_energia.dat")
    eps = 1.5 * 1.380649e-23 * a[:, nomes.index("Te")] / 1.602176634e-19
    especies = nomes[nomes.index("Te") + 1:nomes.index("E/N")]           # todas, com o gás (AR)
    assert "AR" in especies
    n_total = sum(a[:, nomes.index(c)] for c in especies)                  # cm-3
    np.testing.assert_allclose(a[:, nomes.index("freq")], tab(eps)[:, E_NUN] * n_total, rtol=1e-6)


def test_serie_temporal_numa_posicao(tmp_path):
    # scripts/sonda.py: a série de uma coluna na célula mais próxima de x, como
    # borda0.dat, centro.dat e borda1.dat do Fortran
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
    (["--periodo", "0"], "periodo"),
    (["--dt-saida", "0"], "dt-saida"),
    (["--espessura", "-1e-3"], "espessura"),
    (["--eps-dieletrico", "0.5"], "eps-dieletrico"),
    (["--n-celulas", "2"], "n-celulas"),
    (["--t-end", "5e-9", "--dt-saida", "2e-9"], "multiplo"),
    (["--gap", "0"], "gap"),
    (["--k0-ar-ion", "0"], "k0-ar-ion"),
    (["--agrupamento", "-1"], "agrupamento"),
    (["--eps-sec", "-1"], "eps-sec"),
    (["--adaptativa", "--jac", "fd", "--dt-malha", "0"], "dt-malha"),
    (["--retoma", "x", "abc"], "retoma"),
    (["--retoma", "/nao/existe", "1e-9"], "retoma"),
    (["--ne0", "-1"], "ne0"),
    (["--transporte", "/nao/existe.json"], "transporte"),
    (["--eps-gas", "0.5"], "eps-gas"),
    # T_g inicial ou das paredes fora dos nós da tabela (de 300 a 1500 K no
    # argônio): a tabela prenderia o nó extremo sem aviso
    (["--t-gas", "250"], "t_nos.txt"),
    (["--aquecimento", "--t-parede", "2000"], "t_nos.txt"),
])
def test_entrada_invalida_e_recusada(tmp_path, extra, mensagem):
    # como entrada.f90: parâmetros sem sentido param a execução com mensagem
    r = subprocess.run([sys.executable, str(PROTOTIPO / "simula.py"), str(tmp_path / "x"), "--t-end", "2e-9", *extra],
                       capture_output=True, text=True, env={"OPENBLAS_NUM_THREADS": "1", "PATH": ""})
    assert r.returncode != 0
    assert mensagem in r.stderr + r.stdout
    assert "Traceback" not in r.stderr               # recusa com mensagem, não com falha
    assert not (tmp_path / "x" / "perfil.dat").exists()


def test_parametros_gravados_e_mobilidades_configuraveis(tmp_path):
    # a execução grava os parâmetros (como o entrada.nml no diretório do
    # Fortran); as mobilidades iônicas do entrada.nml são opções
    d = tmp_path / "c"
    roda(d, "--t-end", "2e-9", "--k0-ar-ion", "1.5", "--en-aquecimento-ar-ion", "200",
         "--k0-ar2-ion", "1.9", "--t-parede", "310", "--aquecimento")
    p = json.loads((d / "parametros.json").read_text())
    assert p["argumentos"]["k0_ar_ion"] == 1.5 and p["argumentos"]["t_parede"] == 310.0
    assert p["argumentos"]["v_pico"] == 5e3
    # tolerâncias padrão: rtol 1e-4 e atol 1e-10 (ver o teste de elétrons esgotados)
    assert (p["argumentos"]["rtol"], p["argumentos"]["atol"]) == (1e-4, 1e-10)
    assert p["modelo"]["eps_gas"] == 1.0
    mob = p["modelo"]["transporte"]
    assert mob["AR^+"]["mobilidade"] == {"tipo": "frost", "K0": 1.5, "EN_aquecimento": 200.0}
    assert mob["AR2^+"]["mobilidade"]["K0"] == 1.9
    for chave in ("commit", "numpy", "scipy", "python", "OPENBLAS_NUM_THREADS"):
        assert chave in p
    assert p["OPENBLAS_NUM_THREADS"] == "1"
    # a temperatura da parede chega ao modelo (nós das paredes no perfil)
    tb, nomes, linhas = list(blocos(d / "perfil.dat"))[-1]
    assert linhas[0][nomes.index("Temperature")] == 310.0 and linhas[-1][nomes.index("Temperature")] == 310.0
    # e as mobilidades chegam ao modelo: o Ar+ muda em relação ao padrão
    padrao = tmp_path / "padrao"
    roda(padrao, "--t-end", "2e-9", "--aquecimento")
    ar = [np.array(list(blocos(x / "perfil.dat"))[-1][2][1:-1]) for x in (d, padrao)]
    j = list(blocos(d / "perfil.dat"))[-1][1].index("AR^+")
    assert np.max(np.abs(ar[0][:, j] / ar[1][:, j] - 1.0)) > 1e-4


def test_eletrodo_metalico_e_aceito(tmp_path):
    roda(tmp_path / "m", "--t-end", "2e-9", "--espessura", "0")
    assert (tmp_path / "m" / "perfil.dat").exists()


def test_parametros_com_malha_por_arquivo_e_identificacao_do_codigo(tmp_path):
    # com --faces o número de células é o do arquivo; o caminho é gravado
    # absoluto, com o sha256 do conteúdo; os dados e o código são identificados
    # por sha256; os valores efetivos do modelo também vão para o registro
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
        # no pacote distribuído não há zdplaskin_m.F90 nem bolsigdb.dat: registra o que existe
        if (RAIZ / nome).exists():
            assert p["sha256"][nome] == hashlib.sha256((RAIZ / nome).read_bytes()).hexdigest()
    assert p["modelo"]["transporte"]["AR^+"]["mobilidade"]["K0"] == 1.61 and p["modelo"]["n_gas"] > 2.4e19
    assert p["argumentos"]["k0_ar_ion"] is None              # sem a opção, vale o transporte.json
    assert "argv" in p and "cwd" in p


@pytest.mark.parametrize("faces, mensagem", [([0.0, 4e-4, 8e-4], "faces"),
                                             ([0.0, 5e-4, 3e-4, 8e-4], "faces")])
def test_malha_por_arquivo_invalida(tmp_path, faces, mensagem):
    # menos de 3 células ou faces fora de ordem são recusadas
    arq = tmp_path / "faces.txt"
    np.savetxt(arq, faces)
    r = subprocess.run([sys.executable, str(PROTOTIPO / "simula.py"), str(tmp_path / "x"), "--t-end", "2e-9",
                        "--faces", str(arq)], capture_output=True, text=True,
                       env={"OPENBLAS_NUM_THREADS": "1", "PATH": ""})
    assert r.returncode != 0 and mensagem in r.stderr + r.stdout


FORTRAN = ["Pos", "Temperature", "volt", "De", "Me", "veloc", "diff T", "rho cv", "alfa", "freq", "Te",
           "E", "AR", "AR*", "AR^+", "AR2^+", "E/N", "campo"]


def test_perfil_com_as_colunas_do_fortran(tmp_path):
    # mesmos nomes do saida.f90, sempre presentes (sem a CFL, que não se aplica
    # ao integrador implícito); sem aquecimento a temperatura é a do gás; o gás
    # de fundo aparece mesmo preso; nas linhas das paredes, como o Fortran, a
    # temperatura e kappa da parede e rho cv da célula vizinha
    # (--t-parede sem --aquecimento não tem efeito: as paredes ficam na
    # temperatura do gás)
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
    # o perfil guarda n_eps com a equação de energia: gravar e ler devolve o
    # estado exato (antes a energia era refeita pela Te gravada, com perda)
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
    # retomada no meio: os registros posteriores a 4 ns são refeitos, não duplicados
    # retomada no meio: os registros até 4 ns ficam intactos byte a byte, os
    # posteriores são refeitos sem duplicar; os arquivos e os parâmetros
    # anteriores ficam guardados até a nova execução terminar
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
    # retomada de um instante que não existe: nada muda no diretório
    d = tmp_path / "a"
    roda(d, "--t-end", "4e-9")
    antes = {f: (d / f).read_text() for f in ("perfil.dat", "superficie.dat", "parametros.json")}
    r = subprocess.run([sys.executable, str(PROTOTIPO / "simula.py"), str(d), *BASE, "--t-end", "8e-9",
                        "--retoma", str(d), "3e-9"], capture_output=True, text=True,
                       env={"OPENBLAS_NUM_THREADS": "1", "PATH": ""})
    assert r.returncode != 0
    assert {f: (d / f).read_text() for f in antes} == antes


@pytest.mark.parametrize("origem, retomada, mensagem", [(["--aquecimento"], [], "aquecimento"),
                                                         (["--gas-solto"], [], "gas-solto"),
                                                         (["--forma", "degrau", "--subida", "1e-9"], [], "--forma"),
                                                         (["--t-gas", "400"], [], "--t-gas")])
def test_retomada_recusa_descartar_variaveis_da_origem(tmp_path, origem, retomada, mensagem):
    # retomar sem --aquecimento (ou --gas-solto) de uma execução que o tinha
    # descartaria a temperatura (ou o Ar) sem aviso
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
    # o perfil do Fortran (referência de regressão, 100 células uniformes, campo
    # local) é lido pelos nomes: as densidades das células voltam iguais
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
    # n_e/N acima de 1e-3 já no início: o aviso sai uma vez, com a posição
    r = subprocess.run([sys.executable, str(PROTOTIPO / "simula.py"), str(tmp_path / "x"), "--t-end", "1e-12",
                        "--dt-saida", "1e-12", "--ne0", "3e16", "--neutro", "--n-celulas", "6"],
                       capture_output=True, text=True, env={"OPENBLAS_NUM_THREADS": "1", "PATH": ""})
    assert r.stdout.count("AVISO: grau de ionizacao") == 1
    # 6 células uniformes em 0,8 mm: a primeira (nó 2 do Fortran) em x = 66,67 um
    assert "no 2 (x = 66.67 um)" in r.stdout


def test_avisos_verificados_a_cada_passo(tmp_path, monkeypatch):
    # os avisos não esperam a próxima gravação: são verificados a cada passo
    # aceito do integrador, inclusive entre as saídas
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
    # campo local, 5 kV, 1600 células, até 25 ns: entre 20 e 25 ns ~700 células
    # têm n_e entre 1e-8 e 1e-4 cm-3, semente da avalanche seguinte. Com atol de
    # 1e-4 eles ficavam sem controle de erro e n_e saía até 5 vezes errado (40 a
    # 85 vezes com 3200 células); com o padrão (rtol 1e-4, atol 1e-10) fica a
    # 1% da integração apertada (rtol 1e-8, atol 1e-14)
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
    # --transporte troca os dados de transporte: o valor chega ao modelo, o
    # caminho e o sha256 do arquivo vão para o registro; espécie sem dados é
    # recusada com a mensagem do modelo
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
    assert r.returncode != 0 and "erro: transporte: sem dados da espécie AR*" in r.stderr
    assert "Traceback" not in r.stderr


def test_log_conta_temperatura_fora_da_tabela(tmp_path):
    # T_g acima de 1500 K fica presa no último nó da tabela em T_g: o log conta
    r = subprocess.run([sys.executable, str(PROTOTIPO / "simula.py"), str(tmp_path / "t"), *BASE, "--t-end", "2e-9"],
                       capture_output=True, text=True, env={"OPENBLAS_NUM_THREADS": "1", "PATH": ""})
    assert r.returncode == 0, r.stderr
    import re
    assert re.search(r"avaliacoes fora da tabela \(acima/abaixo/T_g\) \d+/\d+/0 ", r.stdout), r.stdout[-500:]


def test_log_de_campo_local_conta_temperatura_fora_da_tabela(tmp_path):
    import re
    base = [a for a in BASE if a != "--energia"]
    r = subprocess.run([sys.executable, str(PROTOTIPO / "simula.py"), str(tmp_path / "t"), *base, "--t-end", "2e-9"],
                       capture_output=True, text=True, env={"OPENBLAS_NUM_THREADS": "1", "PATH": ""})
    assert r.returncode == 0, r.stderr
    assert re.search(r"avaliacoes fora da tabela \(acima/abaixo/T_g\) \d+/-/0 ", r.stdout), r.stdout[-500:]


def test_caso_na_raiz_e_o_padrao(tmp_path):
    # --caso com a raiz do repositório é o mesmo que sem a opção
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
    # a tabela só tem as perdas das espécies com fração positiva na
    # composicao.txt; o modelo exclui as reações do fundo do gas.json: os dois
    # precisam coincidir, senão perdas contadas duas vezes ou nenhuma
    caso = _copia_caso_n2(tmp_path / "caso")
    (caso / "composicao.txt").write_text("N2 0.8\nO2 0.2\nN2^+ traco\n")
    r = subprocess.run([sys.executable, str(PROTOTIPO / "simula.py"), str(tmp_path / "x"), *BASE, "--t-end", "2e-9",
                        "--caso", str(caso)], capture_output=True, text=True,
                       env={"OPENBLAS_NUM_THREADS": "1", "PATH": ""})
    assert r.returncode != 0 and "Traceback" not in r.stderr
    assert "composicao.txt" in r.stderr and "O2" in r.stderr


@precisa_caso("n2")
def test_caso_n2_roda_pelo_simula(tmp_path):
    # o N2 sem código escrito para ele: 2 ns, com o perfil gravado com as
    # espécies do caso e o N2+ inicial convertido em N4+
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
    # numa mistura, a tabela do BOLSIG+ vale para as frações da composicao.txt;
    # o gas.json precisa das mesmas
    from simula import confere_composicao
    caso = _copia_caso_n2(tmp_path / "caso")
    (caso / "composicao.txt").write_text("N2 0.5\nN 0.5\nN2^+ traco\n")
    g = json.loads((caso / "gas.json").read_text())
    g["fundo"] = {"N2": 0.75, "N": 0.25}
    (caso / "gas.json").write_text(json.dumps(g))
    msg = confere_composicao(caso)
    assert msg is not None and "fração" in msg and "N2" in msg
    g["fundo"] = {"N2": 0.5, "N": 0.5}
    (caso / "gas.json").write_text(json.dumps(g))
    assert confere_composicao(caso) is None


def test_reflexao_de_eletrons_chega_ao_modelo(tmp_path):
    # --reflexao-eletrons fica nos argumentos e no modelo gravado, e muda os elétrons
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
    assert r.returncode != 0 and "--reflexao-eletrons" in r.stderr
    assert "Traceback" not in r.stderr
