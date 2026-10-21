# SPDX-License-Identifier: GPL-3.0-or-later
"""Ferramenta de caso (ferramentas/caso.py): a partir de uma pasta com o
kinet.inp e o bolsigdb.dat de uma mistura, gera tudo o que o protótipo usa,
sem editar código para cada gás."""
import pathlib
import shutil
import subprocess
import sys

import pytest

from conftest import RAIZ, precisa_terceiros

sys.path.insert(0, str(RAIZ / "ferramentas"))
import caso  # noqa: E402


@precisa_terceiros
def test_zdplaskin_do_argonio_e_o_do_repositorio(tmp_path):
    # o pré-processador mais as três edições registradas reproduzem o
    # zdplaskin_m.F90 do repositório, menos a linha da data de geração
    shutil.copy(RAIZ / "kinet.inp", tmp_path / "kinet.inp")
    caso.gera_zdplaskin(tmp_path)
    novo = (tmp_path / "zdplaskin_m.F90").read_text().splitlines()
    velho = (RAIZ / "zdplaskin_m.F90").read_text().splitlines()
    assert len(novo) == len(velho)
    diferentes = [i for i, (a, b) in enumerate(zip(novo, velho)) if a != b]
    assert diferentes == [14]                     # a data, na linha 15


@precisa_terceiros
def test_edicao_que_nao_casa_e_erro(tmp_path, monkeypatch):
    # se o pré-processador mudar e uma edição não casar exatamente uma vez,
    # a ferramenta para em vez de gerar um arquivo diferente em silêncio
    shutil.copy(RAIZ / "kinet.inp", tmp_path / "kinet.inp")
    monkeypatch.setattr(caso, "EDICOES_ZDPLASKIN", caso.EDICOES_ZDPLASKIN + [("linha que não existe", "x")])
    with pytest.raises(caso.ErroCaso, match="linha que não existe"):
        caso.gera_zdplaskin(tmp_path)


@precisa_terceiros
def test_kinet_com_erro_e_erro(tmp_path):
    (tmp_path / "kinet.inp").write_text("ELEMENTS\ne Ar\nEND\nSPECIES\ne Ar Xx\nEND\n")
    with pytest.raises(caso.ErroCaso, match="pré-processador"):
        caso.gera_zdplaskin(tmp_path)


# ------------------------------------------------- leitor de química genérico
import numpy as np  # noqa: E402

from quimica_zdp import ErroQuimica, Quimica  # noqa: E402


@pytest.fixture(scope="module")
def n2():
    return Quimica(RAIZ / "casos" / "n2" / "zdplaskin_m.F90")


def test_le_especies_cargas_e_reacoes_do_n2(n2):
    assert n2.especies == ["E", "N", "N2", "N2(A)", "N2(B)", "N2(A')", "N2(C)", "N^+", "N2^+", "N3^+", "N4^+"]
    assert n2.reacoes[0] == "bolsig:N2->N2(A3,V0-4)"
    assert list(n2.carga) == [-1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
    assert len(n2.reacoes) == 40
    assert sum(c is not None for c in n2.coluna_bolsig) == 7


def test_variavel_local_en_e_max(n2):
    # TEFF = TGAS + 0,12 EN**2 antes das taxas; EN é o E/N local (Td)
    tg, te, en = np.array([300.0, 600.0]), np.array([1e4, 2e4]), np.array([10.0, 100.0])
    k = n2.coeficientes(np.zeros((2, 7)), tg, te, en)
    np.testing.assert_allclose(k[7], 2.08e-19 * (300.0 / te) ** 4.5, rtol=1e-14)
    teff = tg + 0.12 * en ** 2
    # 2.1 sem D é real simples no Fortran: o expoente é 2,0999999046...
    np.testing.assert_allclose(k[15], 1.7e-29 * (300.0 / teff) ** float(np.float32(2.1)), rtol=1e-14)
    np.testing.assert_allclose(k[18], 9.0e-30 * np.exp(400.0 / teff), rtol=1e-14)
    np.testing.assert_allclose(k[38], np.maximum(8.3e-34 * np.exp(500.0 / tg), 1.91e-33), rtol=1e-14)
    np.testing.assert_allclose(k[36], 5.4e-8 * (1 - np.exp(-3354.0 / tg)) * np.exp(-113200.0 / tg), rtol=1e-14)


def test_en_e_exigido_quando_a_quimica_usa(n2):
    with pytest.raises(ErroQuimica, match="EN"):
        n2.coeficientes(np.zeros((1, 7)), np.array([300.0]), np.array([1e4]))


def test_nome_desconhecido_e_erro_ao_carregar(tmp_path):
    texto = (RAIZ / "casos" / "n2" / "zdplaskin_m.F90").read_text()
    texto = texto.replace("rrt(15) = 1.0D-29", "rrt(15) = 1.0D-29*XYZ")
    (tmp_path / "z.F90").write_text(texto)
    with pytest.raises(ErroQuimica, match=r"reação 15.*XYZ"):
        Quimica(tmp_path / "z.F90")


def test_argonio_continua_igual():
    # o argônio não usa EN: as taxas são as mesmas com ou sem ele
    q = Quimica(RAIZ / "zdplaskin_m.F90")
    tg, te = np.array([300.0, 450.0]), np.array([5e3, 3e4])
    kb = np.random.default_rng(1).uniform(1e-12, 1e-9, size=(2, 4))
    np.testing.assert_array_equal(q.coeficientes(kb, tg, te), q.coeficientes(kb, tg, te, np.array([1.0, 2.0])))


# ----------------------------------------------------- tabelas pelo BOLSIG+
TABELAS = ("tabela_energia.dat", "tabela_eedf.dat", "tabela_energia_tg.dat", "tabela_energia_tg_conferencia.dat")


def _limpa_compilacao(pasta):
    # a compilação do caso fica em build/<perfil>/caso/<pasta>: os testes não a deixam para trás
    for d in (RAIZ / "build").glob("*/caso/" + str(pathlib.Path(pasta).resolve()).replace("/", "_")):
        shutil.rmtree(d)


@pytest.mark.lento
@precisa_terceiros
def test_tabelas_do_argonio_pela_ferramenta(tmp_path):
    # uma pasta com o kinet.inp e o bolsigdb.dat do argônio, sem composicao.txt
    # (padrão: argônio puro), gera as tabelas idênticas às do repositório
    for nome in ("kinet.inp", "bolsigdb.dat"):
        shutil.copy(RAIZ / nome, tmp_path / nome)
    caso.gera_zdplaskin(tmp_path)
    caso.gera_tabelas(tmp_path)
    _limpa_compilacao(tmp_path)
    for nome in TABELAS:
        # dados bit a bit; o cabeçalho (#) traz os nomes dos processos do caso
        dados = [[l for l in (d / nome).read_text().splitlines() if not l.startswith("#")] for d in (tmp_path, RAIZ)]
        assert dados[0] == dados[1], nome


@pytest.mark.lento
@precisa_terceiros
def test_tabelas_do_n2_pela_ferramenta(tmp_path):
    # o N2 com composicao.txt (N2 puro, N2+ como traço): 7 taxas do BOLSIG+
    from tabela import E_K0, TabelaEnergiaTg
    for nome in ("kinet.inp", "bolsigdb.dat", "composicao.txt"):
        shutil.copy(RAIZ / "casos" / "n2" / nome, tmp_path / nome)
    caso.gera_zdplaskin(tmp_path)
    caso.gera_tabelas(tmp_path)
    _limpa_compilacao(tmp_path)
    t = TabelaEnergiaTg(tmp_path / "tabela_energia_tg.dat")
    assert t.blocos[0].valores.shape[1] == E_K0 + 7
    cabecalho = (tmp_path / "tabela_energia.dat").read_text().splitlines()[1]
    assert "N2->N2(A3,V0-4)" in cabecalho and "N2->N2^+" in cabecalho and "N2^+->N2" not in cabecalho


# ------------------------------------------------- transporte por biblioteca
import json  # noqa: E402


@pytest.mark.parametrize("especie, massa", [("AR", 39.948), ("AR*", 39.948), ("AR^+", 39.948), ("AR2^+", 79.896),
                                            ("N2(A)", 28.014), ("N4^+", 56.028), ("N", 14.007), ("N2(A')", 28.014),
                                            ("H2O", 18.015), ("OH^-", 17.007), ("O^+^+", 15.999)])
def test_massa_pela_formula(especie, massa):
    assert caso.massa_u(especie) == pytest.approx(massa, abs=2e-3)


def test_transporte_do_argonio_pela_biblioteca(tmp_path):
    # química e gas.json do argônio: a biblioteca mais a massa calculada
    # reproduzem o transporte.json da raiz
    for nome in ("zdplaskin_m.F90", "gas.json"):
        shutil.copy(RAIZ / nome, tmp_path / nome)
    faltam = caso.gera_transporte(tmp_path)
    assert faltam == {}
    novo = json.loads((tmp_path / "transporte.json").read_text())
    velho = json.loads((RAIZ / "transporte.json").read_text())
    for esp in ("AR", "AR*", "AR^+", "AR2^+"):
        for campo in ("massa_u", "mobilidade", "difusao", "perda_na_parede", "atomos_do_fundo"):
            if campo in velho[esp]:
                assert novo[esp][campo] == pytest.approx(velho[esp][campo], rel=1e-4) if campo == "massa_u" \
                    else novo[esp][campo] == velho[esp][campo], (esp, campo)


def test_transporte_lista_o_que_falta_e_preserva_o_que_existe(tmp_path):
    # biblioteca vazia: só a massa sai preenchida; o que o usuário já tinha
    # escrito no transporte.json do caso não é sobrescrito
    for nome in ("zdplaskin_m.F90", "gas.json"):
        shutil.copy(RAIZ / nome, tmp_path / nome)
    (tmp_path / "transporte.json").write_text(json.dumps({"AR*": {"perda_na_parede": 0.5}}))
    faltam = caso.gera_transporte(tmp_path, biblioteca={})
    assert set(faltam["AR^+"]) == {"mobilidade"} and set(faltam["AR*"]) == {"difusao"}
    t = json.loads((tmp_path / "transporte.json").read_text())
    assert t["AR*"]["perda_na_parede"] == 0.5 and t["AR^+"]["massa_u"] == pytest.approx(39.948, abs=2e-3)


@pytest.fixture(scope="module")
def pecas_argonio():
    from tabela import TabelaTg
    return Quimica(RAIZ / "zdplaskin_m.F90"), TabelaTg(RAIZ / "tabela_energia_tg.dat")


# ------------------------------- construções do Fortran no ZDPlasKin_reac_rates
def _com_taxas(tmp_path, *linhas, depois="rrt(15) = 1.0D-29"):
    """zdplaskin_m.F90 do N2 com linhas novas antes das taxas e a taxa 15 trocada."""
    texto = (RAIZ / "casos" / "n2" / "zdplaskin_m.F90").read_text()
    texto = texto.replace("  TEFF = TGAS + 0.12D0 * EN**2", "  TEFF = TGAS + 0.12D0 * EN**2\n" + "\n".join(linhas))
    texto = texto.replace("  rrt(15) = 1.0D-29", "  " + depois)
    (tmp_path / "z.F90").write_text(texto)
    return Quimica(tmp_path / "z.F90")


def _k15(q):
    return q.coeficientes(np.zeros((1, 7)), np.array([300.0]), np.array([1e4]), np.array([10.0]))[14, 0]


def test_literal_com_ponto_antes_do_d(tmp_path):
    assert _k15(_com_taxas(tmp_path, depois="rrt(15) = 1.D-7*2.d0")) == pytest.approx(2e-7, rel=1e-14, abs=0)


def test_linha_de_continuacao(tmp_path):
    q = _com_taxas(tmp_path, depois="rrt(15) = &\n    1.0D-7 * &\n    2.0D0")
    assert _k15(q) == pytest.approx(2e-7, rel=1e-14, abs=0)


def _k15_celulas(q, tgas, te=1e4):
    tgas = np.asarray(tgas, dtype=float)
    return q.coeficientes(np.zeros((len(tgas), 7)), tgas, np.full(len(tgas), te), np.full(len(tgas), 10.0))[14]


@pytest.mark.parametrize("expr, k", [("2.0D-7*(1/2) + 1.0D-7*(3/2)", 1e-7),      # 1/2 = 0 e 3/2 = 1
                                     ("1.0D-7*(-3/2)", -1e-7),                    # -(3/2) = -1, truncado para zero
                                     ("1.0D-7*(TGAS/150.)**(3/2)", 2e-7),         # expoente 1
                                     # da esquerda para a direita: TGAS*3 é real antes de /2
                                     ("1.0D-9*TGAS*3/2", 450e-9),
                                     ("1.0D-7*TGAS/300/2", 0.5e-7),
                                     ("1.0D-9*(TGAS/300.)**3/2", 0.5e-9),
                                     ("1.0D-7*(3/2**2)", 0.0),                    # 2**2 antes: 3/4 = 0
                                     ("1.0D-7*(3/(-2))", -1e-7)])
def test_divisao_inteira_segue_o_fortran(tmp_path, expr, k):
    # o ZDPlasKin compilado faz a divisão inteira; o leitor faz o mesmo e avisa
    q = _com_taxas(tmp_path, depois=f"rrt(15) = {expr}")
    assert _k15(q) == pytest.approx(k, rel=1e-14, abs=1e-30)
    # avisa só quando há divisão entre inteiros de fato
    inteira = "TGAS" not in expr or "**(3/2)" in expr
    assert any("inteir" in a and "reação 15" in a for a in q.avisos) == inteira


@pytest.mark.parametrize("expr, valor", [
    # 1.0/3.0 é conta em precisão simples: 0,33333334 (o mecanismo de ar do
    # ZDPlasKin usa TGAS**(1.0/3.0) nas taxas V-T)
    ("1.0D-12*TGAS**(1.0/3.0)", 1.0e-12 * 300.0 ** float(np.float32(1.0) / np.float32(3.0))),
    ("1.0D-12*TGAS**(1.0D0/3.0)", 1.0e-12 * 300.0 ** (1.0 / 3.0)),            # um duplo: conta dupla
    ("1.0D-12*(0.5*2.1 + TGAS)", 1.0e-12 * (float(np.float32(np.float32(0.5) * np.float32(2.1))) + 300.0)),
    ("1.0D-12*(0.1*3.0)", 1.0e-12 * float(np.float32(0.1) * np.float32(3.0))),    # arredonda em float32
])
def test_conta_entre_reais_simples_fica_simples(tmp_path, expr, valor):
    assert _k15(_com_taxas(tmp_path, depois=f"rrt(15) = {expr}")) == pytest.approx(valor, rel=1e-15, abs=0)


def test_if_de_uma_linha(tmp_path):
    q = _com_taxas(tmp_path, "  IF(TGAS > 350.) TEFF = 1.0D0", depois="rrt(15) = TEFF*1.0D-7")
    np.testing.assert_allclose(_k15_celulas(q, [300.0, 400.0]), [(300.0 + 12.0) * 1e-7, 1e-7], rtol=1e-14)


def test_bloco_if_com_operadores_do_fortran(tmp_path):
    q = _com_taxas(tmp_path,
                   "  if (TGAS .gt. 350.0D0 .and. TE >= 1.0D4) then",
                   "    A1 = 2.0D0",
                   "  else if (TGAS /= 300.0D0 .and. .not. (TGAS .eq. 330.0D0)) then",
                   "    A1 = 3.0D0",
                   "  else",
                   "    A1 = 5.0D0",
                   "  end if",
                   depois="rrt(15) = A1*1.0D-7")
    np.testing.assert_allclose(_k15_celulas(q, [300.0, 320.0, 330.0, 400.0]), [5e-7, 3e-7, 5e-7, 2e-7], rtol=1e-14)
    # TE abaixo de 1e4 em toda parte: o primeiro ramo não vale
    np.testing.assert_allclose(_k15_celulas(q, [400.0], te=5e3), [3e-7], rtol=1e-14)


def test_condicao_que_comeca_com_not(tmp_path):
    q = _com_taxas(tmp_path, "  A1 = 1.0D0", "  if (.not. TGAS .gt. 350.0D0) A1 = 2.0D0", depois="rrt(15) = A1*1.0D-7")
    np.testing.assert_allclose(_k15_celulas(q, [300.0, 400.0]), [2e-7, 1e-7], rtol=1e-14)


def test_variavel_sem_valor_fora_do_ramo_e_erro_na_avaliacao(tmp_path):
    # definida só dentro de um IF sem ELSE: no Fortran o valor fica indefinido
    q = _com_taxas(tmp_path, "  if (TGAS > 350.) then", "    A1 = 2.0D0", "  end if", depois="rrt(15) = A1*1.0D-7")
    with pytest.raises(ErroQuimica, match="reação 15"):
        _k15_celulas(q, [300.0, 400.0])


def test_taxa_do_bolsig_misturada_com_expressao_e_recusada(tmp_path):
    with pytest.raises(ErroQuimica, match="BOLSIG"):
        _com_taxas(tmp_path, depois="rrt(15) = 1.0D-29\n  if (TGAS > 350.) rrt(1) = 0.0D0")


@pytest.mark.parametrize("linha", ["  if (TGAS > 350.) return", "  else if (TGAS > 1."])
def test_controle_que_o_leitor_nao_segue_e_recusado(tmp_path, linha):
    with pytest.raises(ErroQuimica):
        _com_taxas(tmp_path, "  if (TGAS > 300.) then", linha, "  end if")


def test_configuracao_sem_uso_e_ignorada(tmp_path):
    # o pré-processador pode emitir De = ZDPlasKin_cfg(6) (difusão dos
    # elétrons) sem uso: só uma taxa que a use é recusada
    q = _com_taxas(tmp_path, "  De  = ZDPlasKin_cfg(6)")
    assert _k15(q) == pytest.approx(1.0e-29, rel=1e-14, abs=0)
    with pytest.raises(ErroQuimica, match=r"ZDPLASKIN_CFG\(6\)"):
        _com_taxas(tmp_path, "  De  = ZDPlasKin_cfg(6)", depois="rrt(15) = 1.0D-29*De")


def test_taxa_redefinida_dentro_de_if(tmp_path):
    q = _com_taxas(tmp_path, depois="rrt(15) = 1.0D-7\n  if(TGAS > 350.) rrt(15) = 2.0D-7")
    np.testing.assert_allclose(_k15_celulas(q, [300.0, 400.0]), [1e-7, 2e-7], rtol=1e-14)


def test_configuracao_do_zdplaskin_na_taxa(tmp_path):
    # ZDPlasKin_cfg(1), (3) e (4): T_g, E/N e T_e, as entradas da rotina de taxas
    q = _com_taxas(tmp_path, "  A2 = ZDPlasKin_cfg(1)*ZDPlasKin_cfg(4) + ZDPlasKin_cfg(3)",
                   depois="rrt(15) = A2*1.0D-12")
    assert _k15(q) == pytest.approx((300.0 * 1e4 + 10.0) * 1e-12, rel=1e-14, abs=0)
    with pytest.raises(ErroQuimica, match=r"ZDPLASKIN_CFG\(7\)"):
        _com_taxas(tmp_path, "  A2 = ZDPlasKin_cfg(7)", depois="rrt(15) = A2")


def test_constante_declarada_com_parameter(tmp_path):
    q = _com_taxas(tmp_path, "  DOUBLE PRECISION, PARAMETER :: A0 = 2.0D0", depois="rrt(15) = A0*1.0D-7")
    assert _k15(q) == pytest.approx(2e-7, rel=1e-14, abs=0)


def test_varias_constantes_na_mesma_declaracao(tmp_path):
    # kinet.inp de Sun et al. 2022: parameter :: a=..., b=...; a vírgula dentro
    # de parênteses não separa declarações
    q = _com_taxas(tmp_path, "  DOUBLE PRECISION, PARAMETER :: A0 = 2.0D0, B0 = MAX(3.0D0, 1.0D0)*A0, C0=1.0D-7",
                   depois="rrt(15) = B0*C0")
    assert _k15(q) == pytest.approx(6e-7, rel=1e-14, abs=0)


def test_lista_em_varios_blocos_data():
    # com muitas reações o pré-processador divide a lista em data x(1:54),
    # data x(55:108)...: o leitor junta os blocos na ordem dos índices
    from quimica_zdp import _lista_data
    texto = """  data reaction_sign(3:4) &
  /"C=>D","N2(A'1)=>N2"/
  data reaction_sign(1:2) &
  /"A=>B", &
   "B=>C"/
"""
    assert _lista_data(texto, "reaction_sign") == ["A=>B", "B=>C", "C=>D", "N2(A'1)=>N2"]


def test_taxa_igual_a_de_outra_reacao(tmp_path):
    # rrt(46) = rrt(45) no kinet.inp de Sun et al. 2022: o coeficiente de uma
    # reação anterior (antes de multiplicar pelas densidades)
    q = _com_taxas(tmp_path, depois="rrt(15) = 2.0D0*rrt(14)")
    k = q.coeficientes(np.zeros((1, 7)), np.array([300.0]), np.array([1e4]), np.array([10.0]))
    assert k[14, 0] == pytest.approx(2.0 * k[13, 0], rel=1e-14, abs=0) and k[13, 0] > 0.0


def test_taxa_de_reacao_posterior_e_recusada(tmp_path):
    with pytest.raises(ErroQuimica, match="posterior"):
        _com_taxas(tmp_path, depois="rrt(15) = rrt(16)")


def test_densidade_total_de_neutros(tmp_path):
    # ANY_NEUTRAL do ZDPlasKin (cm-3), terceiro corpo genérico no kinet.inp de
    # Sun et al. 2022: entra como Tgas, Te e EN
    q = _com_taxas(tmp_path, "  call ZDPlasKin_get_density_total(ALL_NEUTRAL=ANY_NEUTRAL)",
                   depois="rrt(15) = 1.0D-29*ANY_NEUTRAL")
    k = q.coeficientes(np.zeros((1, 7)), np.array([300.0]), np.array([1e4]), np.array([10.0]),
                       any_neutral=np.array([2e19]))
    assert k[14, 0] == pytest.approx(2e-10, rel=1e-14, abs=0)
    with pytest.raises(ErroQuimica, match="ANY_NEUTRAL"):
        _k15(q)


def test_modelo_passa_a_densidade_de_neutros(pecas_argonio):
    # gás preso: o fundo e os neutros transportados (Ar*), sem os íons
    from modelo import Modelo
    md = Modelo(*pecas_argonio, n_celulas=4)
    n = md.separa(md.estado_inicial(n_e0=1e9, neutro=True))[0]
    n[md.nomes.index("AR*")] = 3e12
    np.testing.assert_allclose(md.densidade_neutra(n), md.n_gas + 3e12, rtol=1e-15)


# call executa código que o leitor não tem; density() a rotina das taxas do
# ZDPlasKin nem enxerga (o use do módulo não a importa)
@pytest.mark.parametrize("linha, mensagem", [("  call minha_rotina(TGAS)", "call"),
                                             ("  TEFF = density(1)", "DENSITY"),
                                             ("  if (TGAS > 300.) then", "IF")])
def test_construcao_nao_suportada_e_erro_claro(tmp_path, linha, mensagem):
    with pytest.raises(ErroQuimica, match=mensagem):
        _com_taxas(tmp_path, linha)


def test_sintaxe_invalida_vira_erro_claro(tmp_path):
    with pytest.raises(ErroQuimica, match="reação 15"):
        _com_taxas(tmp_path, depois="rrt(15) = 1.0D-29 * * 2")


@precisa_terceiros
def test_reacao_do_bolsig_com_dois_eletrons_e_recusada(tmp_path_factory):
    # e + N2^+ + e => e + N2 com a taxa do BOLSIG+ (cm3/s, dois corpos) usada
    # como de três corpos: erro de ordens de grandeza (revisão da ferramenta)
    from modelo import Modelo
    from tabela import TabelaTg
    # o kinet.inp original do exemplo do ZDPlasKin, antes da troca no caso
    pasta = tmp_path_factory.mktemp("n2_original")
    kinet = (RAIZ / "casos" / "n2" / "kinet.inp").read_text()
    kinet = kinet.replace("e + N2^+ + e => e + N2             !   2.08d-19*(300.0d0/Te)**4.50",
                          "e + N2^+ + e => e + N2             !   BOLSIG N2^+ -> N2")
    (pasta / "kinet.inp").write_text(kinet)
    caso.gera_zdplaskin(pasta)
    q = Quimica(pasta / "zdplaskin_m.F90")
    assert q.bolsig_tres_corpos == ["bolsig:N2^+->N2"]
    assert Quimica(RAIZ / "casos" / "n2" / "zdplaskin_m.F90").bolsig_tres_corpos == []
    with pytest.raises(ErroQuimica, match=r"N2\^\+->N2.*dois corpos"):
        Modelo(q, TabelaTg(RAIZ / "casos" / "n2" / "tabela_energia_tg.dat"), n_celulas=4,
               gas=RAIZ / "casos" / "n2" / "gas.json", transporte=RAIZ / "casos" / "n2" / "transporte.json")


def test_attachment_sem_limiar(tmp_path):
    # no formato do LXCat o ATTACHMENT não tem linha de limiar
    banco = (RAIZ / "bolsigdb.dat").read_text() + ("\nATTACHMENT\nAR -> AR^-\nCOMMENT: sintético\n"
                                                    "-----------------------------\n 1.0 1e-22\n 2.0 1e-22\n"
                                                    "-----------------------------\n")
    (tmp_path / "b.dat").write_text(banco)
    q = Quimica(RAIZ / "zdplaskin_m.F90")
    np.testing.assert_array_equal(q.energia_eletronica(tmp_path / "b.dat", "AR"), q.energia_eletronica(RAIZ / "bolsigdb.dat", "AR"))


# ---------------------------------------------- calor rápido dos elétrons
def _pasta_n2(destino, processos):
    for nome in ("kinet.inp", "bolsigdb.dat", "composicao.txt", "zdplaskin_m.F90", "transporte.json"):
        shutil.copy(RAIZ / "casos" / "n2" / nome, destino / nome)
    gas = json.loads((RAIZ / "casos" / "n2" / "gas.json").read_text())
    gas["calor"] = {"entalpias": {}, "processos_rapidos": processos}
    (destino / "gas.json").write_text(json.dumps(gas))
    return destino


def test_lista_de_calor_rapido_com_os_limiares(tmp_path):
    pasta = _pasta_n2(tmp_path, ["N2 -> N2(rot)"])
    caso.gera_calor_rapido(pasta)
    assert (pasta / "calor_rapido.txt").read_text().split() == ["N2->N2(ROT)", "0.02"]


def test_processo_rapido_desconhecido_e_erro(tmp_path):
    pasta = _pasta_n2(tmp_path, ["N2 -> N2(xyz)"])
    with pytest.raises(caso.ErroCaso, match=r"N2->N2\(XYZ\)"):
        caso.gera_calor_rapido(pasta)


@pytest.mark.lento
@precisa_terceiros
def test_tabela_de_calor_rapido_do_n2(tmp_path):
    from tabela import E_PINEL, TabelaEnergiaTg
    pasta = _pasta_n2(tmp_path, ["N2 -> N2(rot)"])
    caso.gera_calor_rapido(pasta)
    caso.gera_tabelas(pasta)
    _limpa_compilacao(pasta)
    calor = TabelaEnergiaTg(pasta / "tabela_calor_tg.dat")
    energia = TabelaEnergiaTg(pasta / "tabela_energia_tg.dat")
    assert np.array_equal(calor.t_gas, energia.t_gas)
    b0, e0 = calor.blocos[0], energia.blocos[0]
    assert np.array_equal(b0.e_n, e0.e_n) and np.array_equal(b0.eps, e0.eps)
    p_rot, p_inel = b0.valores[:, 1], e0.valores[:, E_PINEL]
    assert np.all(p_rot >= 0.0) and np.all(p_rot <= p_inel * (1 + 1e-12))
    assert p_rot[np.argmin(abs(b0.e_n - 10.0))] > 0.0


def _caso_argonio(destino, fundo):
    for nome in ("zdplaskin_m.F90", "gas.json"):
        shutil.copy(RAIZ / nome, destino / nome)
    g = json.loads((destino / "gas.json").read_text())
    g["fundo"] = fundo
    (destino / "gas.json").write_text(json.dumps(g))


def test_transporte_com_fundo_em_mistura(tmp_path):
    # as espécies do fundo não são transportadas: nenhuma exige difusão; a
    # mobilidade de um íon vem da biblioteca só com o dado na mistura (chave
    # "ESPECIE em A+B", na ordem do gas.json), e não do dado num gás só
    _caso_argonio(tmp_path, {"AR": 0.8, "AR*": 0.2})
    bib = {"AR^+ em AR": {"mobilidade": {"tipo": "constante", "K0": 1.5}},
           "AR2^+ em AR+AR*": {"mobilidade": {"tipo": "constante", "K0": 1.9}}}
    faltam = caso.gera_transporte(tmp_path, biblioteca=bib)
    assert faltam == {"AR^+": ["mobilidade"]}
    t = json.loads((tmp_path / "transporte.json").read_text())
    assert t["AR2^+"]["mobilidade"]["K0"] == 1.9


def test_transporte_exige_mobilidade_do_ion_negativo(tmp_path):
    from quimica_zdp import Quimica
    _caso_argonio(tmp_path, {"AR": 1.0})
    q = Quimica(tmp_path / "zdplaskin_m.F90")
    q.carga = q.carga.copy()
    q.carga[q.indice["AR2^+"]] = -1.0
    faltam = caso.gera_transporte(tmp_path, biblioteca={}, quimica=q)
    assert faltam["AR2^+"] == ["mobilidade"]


# ------------------------------- seções de choque com mais de 1000 pontos
def _bolsigdb(pontos):
    tabela = "\n".join(f" {e:.6e}\t{1e-20 * (1 + e):.6e}" for e in np.linspace(2.0, 100.0, pontos))
    return ("EXCITATION\nN -> N(2D)\n 2.0e+0 / threshold\nPROCESS: E + N -> E + N(2D)\n"
            "-----------------------------\n" + tabela + "\n-----------------------------\n")


@precisa_terceiros
def test_secao_com_mais_de_900_pontos_e_recusada(tmp_path):
    # o bolsig_x86_64.so cai com tabelas grandes (caso da amônia, N -> N(2D)
    # com 1291 pontos; 999 cai, 900 passa)
    (tmp_path / "bolsigdb.dat").write_text(_bolsigdb(1291))
    (tmp_path / "zdplaskin_m.F90").write_text("")
    with pytest.raises(caso.ErroCaso, match=r"N->N\(2D\).*1291"):
        caso.gera_tabelas(tmp_path)


def test_reducao_das_secoes_grandes(tmp_path):
    # tira, um a um, o ponto cuja falta menos muda a curva, até sobrarem 900;
    # mantém o primeiro e o último; um pico estreito (ressonância) fica
    e = np.linspace(2.0, 100.0, 1291)
    s = 1e-20 * (1 + e) + 5e-19 * np.exp(-((e - 40.0) / 0.2) ** 2)
    tabela = "\n".join(f" {a:.6e}\t{b:.6e}" for a, b in zip(e, s))
    bloco = ("EXCITATION\nN -> N(2D)\n 2.0e+0 / threshold\nPROCESS: E + N -> E + N(2D)\n"
             "-----------------------------\n" + tabela + "\n-----------------------------\n")
    (tmp_path / "bolsigdb.dat").write_text(bloco + _bolsigdb(50).replace("N(2D)", "N(2P)"))
    relatorio = caso.reduz_secoes(tmp_path / "bolsigdb.dat", tmp_path / "reduzido.dat")
    texto = (tmp_path / "reduzido.dat").read_text()
    assert set(relatorio) == {"N->N(2D)"}
    assert relatorio["N->N(2D)"]["pontos"] == (1291, 900)
    assert relatorio["N->N(2D)"]["erro_max"] < 1e-3
    blocos = texto.split("-----------------------------")
    assert len(blocos[1].split("\n")) - 2 == 900 and len(blocos[3].split("\n")) - 2 == 50
    assert " 1.000000e+02" in blocos[1] and " 2.000000e+00" in blocos[1]


@precisa_terceiros
def test_falha_do_gerador_apaga_as_tabelas_parciais(tmp_path, monkeypatch):
    import subprocess as sp
    (tmp_path / "bolsigdb.dat").write_text(_bolsigdb(10))
    (tmp_path / "zdplaskin_m.F90").write_text("")
    def falha(*a, **k):
        (tmp_path / "tabela_energia.dat").write_text("parcial")
        return sp.CompletedProcess(a, 2, "", "SIGSEGV")
    monkeypatch.setattr(caso.subprocess, "run", falha)
    with pytest.raises(caso.ErroCaso):
        caso.gera_tabelas(tmp_path)
    assert not (tmp_path / "tabela_energia.dat").exists()


def test_tira_as_linhas_que_quebram_a_monotonia(tmp_path):
    # o BOLSIG+ refaz a grade entre E/N vizinhos e a energia média pode cair
    # 0,1% (amônia, perto de 350 Td): a linha sai, em cada bloco de T_g, e fica
    # registrada; o protótipo inverte a tabela pela energia e exige monotonia
    arq = tmp_path / "tabela_energia_tg.dat"
    arq.write_text("# cabeçalho\n"
                   "300 1.0 0.50 7\n300 2.0 0.60 7\n300 3.0 0.59 7\n300 4.0 0.70 7\n\n"
                   "600 1.0 0.55 7\n600 2.0 0.65 7\n600 3.0 0.66 7\n")
    retiradas = caso.tira_nao_monotonas(arq, coluna_t=0, coluna_eps=2)
    assert retiradas == ["300 3.0 0.59 7"]
    linhas = [l for l in arq.read_text().splitlines() if l and not l.startswith("#")]
    assert linhas == ["300 1.0 0.50 7", "300 2.0 0.60 7", "300 4.0 0.70 7",
                      "600 1.0 0.55 7", "600 2.0 0.65 7", "600 3.0 0.66 7"]
    assert arq.read_text().startswith("# cabeçalho\n")
    assert "\n\n600" in arq.read_text()            # a linha em branco entre blocos fica


def test_limiar_colado_a_barra(tmp_path):
    # banco de Sun et al. 2022: "2.000000e-2/threshold energy", sem espaço
    (tmp_path / "b.dat").write_text("EXCITATION\nN2 -> N2(rot)\n2.000000e-2/threshold energy\n-----\n 1 1\n-----\n"
                                    "EXCITATION\nN2 -> N2(v1)\n 2.900e-1 / threshold\n-----\n 1 1\n-----\n")
    assert caso.processos_do_bolsigdb(tmp_path / "b.dat") == {"N2->N2(ROT)": 0.02, "N2->N2(V1)": 0.29}


def test_energia_eletronica_com_limiar_colado_a_barra(tmp_path, pecas_argonio):
    q = pecas_argonio[0]
    texto = (RAIZ / "bolsigdb.dat").read_text()
    colado = texto.replace("  1.150e+1  1.000e+0 / threshold energy", "1.150e+1/threshold energy", 1)
    assert colado != texto
    (tmp_path / "b.dat").write_text(colado)
    np.testing.assert_array_equal(q.energia_eletronica(tmp_path / "b.dat", "AR"),
                                  q.energia_eletronica(RAIZ / "bolsigdb.dat", "AR"))


# ------------------------------------------- limites do BOLSIG+ sem passo manual
def _bolsigdb_varios(pastas_pontos):
    """bolsigdb.dat sintético: um processo EXCITATION por item, com n pontos."""
    linhas = []
    for nome, n in pastas_pontos:
        linhas += ["EXCITATION", nome, " 1.0", "COMMENT: sintetico", "-----------------------------"]
        linhas += [f"{1.0 + i:.6e} {1e-20 * (1 + (i % 7)):.6e}" for i in range(n)]
        linhas += ["-----------------------------", ""]
    return "\n".join(linhas) + "\n"


def test_tudo_reduz_sozinho_as_secoes_grandes(tmp_path):
    (tmp_path / "bolsigdb.dat").write_text(_bolsigdb_varios([("N2 -> N2*", 1200), ("N2 -> N2**", 50)]))
    relatorio = caso.reduz_se_preciso(tmp_path)
    assert set(relatorio) == {"N2->N2*"}
    assert caso.secoes_grandes(tmp_path / "bolsigdb.dat") == {}
    assert caso.secoes_grandes(tmp_path / "bolsigdb_original.dat") == {"N2->N2*": 1200}
    assert caso.reduz_se_preciso(tmp_path) == {}            # já reduzido: nada a fazer
    assert caso.reduz_se_preciso in caso.ACOES["tudo"]


def test_reduz_nao_sobrescreve_bolsigdb_trocado(tmp_path):
    # com o bolsigdb_original.dat de uma redução anterior, um bolsigdb.dat novo
    # (trocado pelo usuário) não pode ser substituído pela redução do antigo
    (tmp_path / "bolsigdb.dat").write_text(_bolsigdb_varios([("N2 -> N2*", 1200)]))
    caso.reduz_bolsigdb(tmp_path)
    novo = _bolsigdb_varios([("H2 -> H2*", 1300)])
    (tmp_path / "bolsigdb.dat").write_text(novo)
    with pytest.raises(caso.ErroCaso, match="bolsigdb_original.dat"):
        caso.reduz_bolsigdb(tmp_path)
    assert (tmp_path / "bolsigdb.dat").read_text() == novo


def _gerador_que_cai_acima_de(limite_td):
    """Gerador falso: grava o progresso como o gera_tabela_energia.f90 e cai ao
    passar de limite_td, a menos que o en_max.txt fique abaixo."""
    chamadas = []

    def roda(pasta):
        en_max = float((pasta / "en_max.txt").read_text()) if (pasta / "en_max.txt").exists() else 5000.0
        chamadas.append(en_max)
        for e_n in (1.0, 100.0, 1000.0, limite_td * 1.01, en_max):
            if e_n > en_max:
                break
            (pasta / "progresso_tabelas.txt").write_text(f"300.0 {e_n}\n")
            if e_n > limite_td:
                return 1, "WARNING: BOLSIG+ convergence failed"
        for nome in caso.TABELAS:
            (pasta / nome).write_text("# falsa\n")
        return 0, ""
    return roda, chamadas


def _pasta_minima(pasta):
    for nome in ("zdplaskin_m.F90", "bolsigdb.dat"):
        (pasta / nome).write_text("")


def test_topo_de_en_automatico_quando_o_bolsig_cai(tmp_path):
    _pasta_minima(tmp_path)
    roda, chamadas = _gerador_que_cai_acima_de(4300.0)
    caso.gera_tabelas(tmp_path, roda=roda)
    topo = float((tmp_path / "en_max.txt").read_text())
    assert 3000.0 < topo < 4300.0
    assert len(chamadas) == 2 and chamadas[0] == 5000.0
    assert "4343" in (tmp_path / "tabelas_topo.txt").read_text()     # onde caiu, registrado
    assert not (tmp_path / "progresso_tabelas.txt").exists()


def test_progresso_velho_nao_vira_queda_e_en_max_volta(tmp_path):
    # um progresso_tabelas.txt de uma execução interrompida e uma falha de
    # compilação: nada de topo novo, e o en_max.txt do usuário fica como era
    _pasta_minima(tmp_path)
    (tmp_path / "progresso_tabelas.txt").write_text("300.0 4500.0\n")
    (tmp_path / "en_max.txt").write_text("4800\n")
    chamadas = []

    def falha_de_compilacao(pasta):
        chamadas.append(1)
        return 2, "Error: compilation failed"
    with pytest.raises(caso.ErroCaso, match="compilation"):
        caso.gera_tabelas(tmp_path, roda=falha_de_compilacao)
    assert len(chamadas) == 1 and (tmp_path / "en_max.txt").read_text() == "4800\n"


def test_topo_que_nao_resolve_devolve_o_en_max_original(tmp_path):
    # o BOLSIG+ cai sempre no último ponto: o topo desce até as tentativas
    # acabarem, e então o en_max.txt original volta
    _pasta_minima(tmp_path)
    (tmp_path / "en_max.txt").write_text("4800\n")

    def cai_sempre(pasta):
        topo = float((pasta / "en_max.txt").read_text())
        (pasta / "progresso_tabelas.txt").write_text(f"300.0 {topo}\n")
        return 1, "WARNING: BOLSIG+ convergence failed"
    with pytest.raises(caso.ErroCaso):
        caso.gera_tabelas(tmp_path, roda=cai_sempre)
    assert (tmp_path / "en_max.txt").read_text() == "4800\n"


def test_queda_em_en_baixo_nao_vira_topo(tmp_path):
    # cair em E/N baixo não é limite de campo alto: erro com a saída, sem en_max.txt
    _pasta_minima(tmp_path)
    roda, chamadas = _gerador_que_cai_acima_de(50.0)
    with pytest.raises(caso.ErroCaso, match="convergence failed"):
        caso.gera_tabelas(tmp_path, roda=roda)
    assert not (tmp_path / "en_max.txt").exists() and len(chamadas) == 1


def test_reduz_aceita_o_reduzido_sem_o_fim_de_linha(tmp_path):
    (tmp_path / "bolsigdb.dat").write_text(_bolsigdb_varios([("N2 -> N2*", 1200)]))
    caso.reduz_bolsigdb(tmp_path)
    (tmp_path / "bolsigdb.dat").write_text((tmp_path / "bolsigdb.dat").read_text().rstrip("\n"))
    caso.reduz_bolsigdb(tmp_path)
    assert not (tmp_path / "bolsigdb_reduzido.tmp").exists()


def test_simula_mostra_os_avisos_da_quimica(tmp_path, capsys):
    sys.path.insert(0, str(RAIZ / "prototipo"))
    import simula
    q = _com_taxas(tmp_path, depois="rrt(15) = 1.0D-7*(3/2)")

    class M:
        pass
    md = M()
    md.q = q
    simula.mostra_avisos_da_quimica(md)
    assert "AVISO: reação 15: divisão entre inteiros" in capsys.readouterr().out


# --------------------------------------- estimativa dos dados de transporte
def test_mobilidade_de_langevin_e_lei_de_blanc():
    # K0 = 13,853/sqrt(alfa mu) (cm2/(V s), alfa em Å3, mu em u) em cada gás;
    # na mistura, 1/K = soma de x_i/K_i
    k_n2 = 13.853 / np.sqrt(1.7403 * caso.massa_u("N2^+") * caso.massa_u("N2") / (caso.massa_u("N2^+") + caso.massa_u("N2")))
    assert caso.mobilidade_estimada("N2^+", {"N2": 1.0}) == pytest.approx(k_n2, rel=1e-12)
    k_h2 = 13.853 / np.sqrt(0.8045 * caso.massa_u("N2^+") * caso.massa_u("H2") / (caso.massa_u("N2^+") + caso.massa_u("H2")))
    assert caso.mobilidade_estimada("N2^+", {"N2": 0.25, "H2": 0.75}) == pytest.approx(
        1.0 / (0.25 / k_n2 + 0.75 / k_h2), rel=1e-12)
    # polarizabilidade dada no gas.json vale sobre a da tabela
    assert caso.mobilidade_estimada("N2^+", {"N2": 1.0}, {"N2": 4 * 1.7403}) == pytest.approx(k_n2 / 2, rel=1e-12)
    with pytest.raises(caso.ErroCaso, match=r"SF6.*polarizabilidades"):
        caso.mobilidade_estimada("SF6^-", {"SF6": 1.0})


def _pasta_n2_sem_transporte(destino):
    for nome in ("zdplaskin_m.F90", "gas.json"):
        shutil.copy(RAIZ / "casos" / "n2" / nome, destino / nome)
    return destino


def test_estima_preenche_so_o_que_falta_e_marca_a_fonte(tmp_path):
    pasta = _pasta_n2_sem_transporte(tmp_path)
    faltam = caso.gera_transporte(pasta, biblioteca={}, estimar=True)
    t = json.loads((pasta / "transporte.json").read_text())
    for ion in ("N2^+", "N4^+"):
        assert t[ion]["mobilidade"]["tipo"] == "constante"
        assert t[ion]["mobilidade"]["K0"] == pytest.approx(caso.mobilidade_estimada(ion, {"N2": 1.0}), rel=1e-12)
        assert "estimativa" in t[ion]["fonte"]
    # a perda na parede não se estima: continua listada
    assert all("perda_na_parede" in c for c in faltam.values())
    assert not any("mobilidade" in c for c in faltam.values())


def test_estima_a_difusao_pela_teoria_cinetica(tmp_path):
    pytest.importorskip("cantera")
    pasta = _pasta_n2_sem_transporte(tmp_path)
    caso.gera_transporte(pasta, biblioteca={}, estimar=True)
    t = json.loads((pasta / "transporte.json").read_text())
    # estados excitados com a difusão do estado fundamental (N2(A) como N2)
    assert t["N2(A)"]["difusao"] == t["N2(B)"]["difusao"]
    assert 1e-5 < t["N"]["difusao"]["D0"] < 1e-4 and 1.5 < t["N"]["difusao"]["expoente"] < 2.0
    assert "estimativa" in t["N"]["fonte"]
    assert "difusao" in t["N2"]                 # o fundo também, para --gas-solto


# ------------------------------------------------ entalpias para o bloco calor
def test_entalpias_pelos_limiares_do_bolsig():
    # do N2 (0 eV) pelos processos do BOLSIG+ do caso: excitados e N2+ com o
    # menor limiar de cada produto (v = 0), iguais aos curados no gas.json do N2
    q = Quimica(RAIZ / "casos" / "n2" / "zdplaskin_m.F90")
    curadas = json.loads((RAIZ / "casos" / "n2" / "gas.json").read_text())["calor"]["entalpias"]
    h, origem = caso.propaga_entalpias(q, caso.limiares_do_bolsigdb(RAIZ / "casos" / "n2" / "bolsigdb.dat"),
                                       {"N2": 0.0})
    for esp in ("N2(A)", "N2(B)", "N2(C)"):
        assert h[esp] == pytest.approx(curadas[esp], abs=1e-3), esp
        assert "limiar" in origem[esp]
    # o bolsigdb.dat dá a ionização arredondada (15,6 eV); a curada é a do NIST
    assert h["N2^+"] == 15.6 and h["N2^+"] == pytest.approx(curadas["N2^+"], abs=0.02)
    assert "N4^+" not in h and "N" not in h          # sem processo que chegue a eles


def test_entalpias_incompletas_nao_ligam_o_calor(tmp_path):
    # sem bloco calor, gravar um incompleto faria o modelo recusar o caso: fica
    # num arquivo à parte, e o gas.json não muda
    for nome in ("zdplaskin_m.F90", "bolsigdb.dat"):
        shutil.copy(RAIZ / "casos" / "n2" / nome, tmp_path / nome)
    gas = json.loads((RAIZ / "casos" / "n2" / "gas.json").read_text())
    gas.pop("calor")
    (tmp_path / "gas.json").write_text(json.dumps(gas))
    antes = (tmp_path / "gas.json").read_text()
    faltam = caso.estima_entalpias(tmp_path)
    assert faltam and (tmp_path / "gas.json").read_text() == antes
    proposta = json.loads((tmp_path / "entalpias_propostas.json").read_text())
    assert proposta["faltam"] == faltam
    if caso.tem_cantera():
        assert proposta["entalpias"]["N2(A)"] == pytest.approx(6.17, abs=1e-3)


def test_entalpias_completam_o_gas_json(tmp_path):
    pytest.importorskip("cantera")
    for nome in ("zdplaskin_m.F90", "bolsigdb.dat"):
        shutil.copy(RAIZ / "casos" / "n2" / nome, tmp_path / nome)
    gas = json.loads((RAIZ / "casos" / "n2" / "gas.json").read_text())
    gas["calor"] = {"entalpias": {"N4^+": 14.481}}
    (tmp_path / "gas.json").write_text(json.dumps(gas))
    faltam = caso.estima_entalpias(tmp_path)
    h = json.loads((tmp_path / "gas.json").read_text())["calor"]["entalpias"]
    assert h["N4^+"] == 14.481                                        # o que existia fica
    assert h["N"] == pytest.approx(4.899, abs=0.01) and abs(h["N2"]) < 0.01
    assert "N3^+" in faltam and "N4^+" not in faltam


def test_terceiros_por_variavel_de_ambiente(tmp_path, monkeypatch):
    # ZDPlasKin e BOLSIG+ não vão no pacote (a licença não permite
    # redistribuir): a ferramenta os acha por ZDPLASKIN_DIR e BOLSIG_SO
    zdp = tmp_path / "zdp"
    zdp.mkdir()
    for nome in ("preprocessor", "dvode_f90_m.F90", "bolsig_x86_64.so"):
        (zdp / nome).write_text("x")
    monkeypatch.setenv("ZDPLASKIN_DIR", str(zdp))
    monkeypatch.delenv("BOLSIG_SO", raising=False)
    t = caso.terceiros()
    assert t["preprocessor"] == zdp / "preprocessor" and t["dvode"] == zdp / "dvode_f90_m.F90"
    assert t["bolsig"] == zdp / "bolsig_x86_64.so"
    outro = tmp_path / "libbolsig.so"
    outro.write_text("x")
    monkeypatch.setenv("BOLSIG_SO", str(outro))
    assert caso.terceiros()["bolsig"] == outro


def test_terceiros_ausentes_dao_erro_claro(tmp_path, monkeypatch):
    monkeypatch.setenv("ZDPLASKIN_DIR", str(tmp_path / "nao_existe"))
    monkeypatch.setenv("BOLSIG_SO", str(tmp_path / "nao_existe.so"))
    monkeypatch.setattr(caso, "RAIZ", tmp_path)
    with pytest.raises(caso.ErroCaso, match="ZDPLASKIN_DIR"):
        caso.terceiros()


def test_exporta_grava_quimica_json(tmp_path):
    for nome in ("zdplaskin_m.F90", "bolsigdb.dat"):
        shutil.copy(RAIZ / "casos" / "n2" / nome, tmp_path / nome)
    caso.exporta(tmp_path)
    sys.path.insert(0, str(RAIZ / "prototipo"))
    from quimica_zdp import Quimica
    j = Quimica.de_json(tmp_path / "quimica.json")
    assert j.reacoes == Quimica(tmp_path / "zdplaskin_m.F90").reacoes
    assert caso.exporta in caso.ACOES["tudo"] and caso.ACOES["tudo"][-1] is caso.exporta
