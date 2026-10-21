# SPDX-License-Identifier: GPL-3.0-or-later
"""Case tool (ferramentas/caso.py): from a folder with the kinet.inp and the
bolsigdb.dat of a mixture, it generates everything the prototype uses, without
editing code for each gas."""
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
    # the preprocessor plus the three recorded edits reproduce the
    # zdplaskin_m.F90 of the repository, except the line with the generation date
    shutil.copy(RAIZ / "kinet.inp", tmp_path / "kinet.inp")
    caso.gera_zdplaskin(tmp_path)
    novo = (tmp_path / "zdplaskin_m.F90").read_text().splitlines()
    velho = (RAIZ / "zdplaskin_m.F90").read_text().splitlines()
    assert len(novo) == len(velho)
    diferentes = [i for i, (a, b) in enumerate(zip(novo, velho)) if a != b]
    assert diferentes == [14]                     # the date, on line 15


@precisa_terceiros
def test_edicao_que_nao_casa_e_erro(tmp_path, monkeypatch):
    # if the preprocessor changes and an edit does not match exactly once,
    # the tool stops instead of silently generating a different file
    shutil.copy(RAIZ / "kinet.inp", tmp_path / "kinet.inp")
    monkeypatch.setattr(caso, "EDICOES_ZDPLASKIN", caso.EDICOES_ZDPLASKIN + [("linha que não existe", "x")])
    with pytest.raises(caso.ErroCaso, match="linha que não existe"):
        caso.gera_zdplaskin(tmp_path)


@precisa_terceiros
def test_kinet_com_erro_e_erro(tmp_path):
    (tmp_path / "kinet.inp").write_text("ELEMENTS\ne Ar\nEND\nSPECIES\ne Ar Xx\nEND\n")
    with pytest.raises(caso.ErroCaso, match="preprocessor"):
        caso.gera_zdplaskin(tmp_path)


# ------------------------------------------------- generic chemistry reader
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
    # TEFF = TGAS + 0.12 EN**2 before the rates; EN is the local E/N (Td)
    tg, te, en = np.array([300.0, 600.0]), np.array([1e4, 2e4]), np.array([10.0, 100.0])
    k = n2.coeficientes(np.zeros((2, 7)), tg, te, en)
    np.testing.assert_allclose(k[7], 2.08e-19 * (300.0 / te) ** 4.5, rtol=1e-14)
    teff = tg + 0.12 * en ** 2
    # 2.1 without D is single precision real in Fortran: the exponent is 2.0999999046...
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
    with pytest.raises(ErroQuimica, match=r"reaction 15.*XYZ"):
        Quimica(tmp_path / "z.F90")


def test_argonio_continua_igual():
    # argon does not use EN: the rates are the same with or without it
    q = Quimica(RAIZ / "zdplaskin_m.F90")
    tg, te = np.array([300.0, 450.0]), np.array([5e3, 3e4])
    kb = np.random.default_rng(1).uniform(1e-12, 1e-9, size=(2, 4))
    np.testing.assert_array_equal(q.coeficientes(kb, tg, te), q.coeficientes(kb, tg, te, np.array([1.0, 2.0])))


# ----------------------------------------------------- tables from BOLSIG+
TABELAS = ("tabela_energia.dat", "tabela_eedf.dat", "tabela_energia_tg.dat", "tabela_energia_tg_conferencia.dat")


def _limpa_compilacao(pasta):
    # the case build lives in build/<profile>/caso/<folder>: the tests do not leave it behind
    for d in (RAIZ / "build").glob("*/caso/" + str(pathlib.Path(pasta).resolve()).replace("/", "_")):
        shutil.rmtree(d)


@pytest.mark.lento
@precisa_terceiros
def test_tabelas_do_argonio_pela_ferramenta(tmp_path):
    # a folder with the kinet.inp and bolsigdb.dat of argon, without composicao.txt
    # (default: pure argon), generates tables identical to those of the repository
    for nome in ("kinet.inp", "bolsigdb.dat"):
        shutil.copy(RAIZ / nome, tmp_path / nome)
    caso.gera_zdplaskin(tmp_path)
    caso.gera_tabelas(tmp_path)
    _limpa_compilacao(tmp_path)
    for nome in TABELAS:
        # data bit for bit; the header (#) carries the names of the processes of the case
        dados = [[l for l in (d / nome).read_text().splitlines() if not l.startswith("#")] for d in (tmp_path, RAIZ)]
        assert dados[0] == dados[1], nome


@pytest.mark.lento
@precisa_terceiros
def test_tabelas_do_n2_pela_ferramenta(tmp_path):
    # N2 with composicao.txt (pure N2, N2+ as a trace): 7 BOLSIG+ rates
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


# ------------------------------------------------- transport from the library
import json  # noqa: E402


@pytest.mark.parametrize("especie, massa", [("AR", 39.948), ("AR*", 39.948), ("AR^+", 39.948), ("AR2^+", 79.896),
                                            ("N2(A)", 28.014), ("N4^+", 56.028), ("N", 14.007), ("N2(A')", 28.014),
                                            ("H2O", 18.015), ("OH^-", 17.007), ("O^+^+", 15.999)])
def test_massa_pela_formula(especie, massa):
    assert caso.massa_u(especie) == pytest.approx(massa, abs=2e-3)


def test_transporte_do_argonio_pela_biblioteca(tmp_path):
    # chemistry and gas.json of argon: the library plus the computed mass
    # reproduce the transporte.json of the root
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
    # empty library: only the mass comes out filled; what the user had already
    # written in the transporte.json of the case is not overwritten
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


# ------------------------------- Fortran constructs in ZDPlasKin_reac_rates
def _com_taxas(tmp_path, *linhas, depois="rrt(15) = 1.0D-29"):
    """zdplaskin_m.F90 of N2 with new lines before the rates and rate 15 replaced."""
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
                                     ("1.0D-7*(-3/2)", -1e-7),                    # -(3/2) = -1, truncated towards zero
                                     ("1.0D-7*(TGAS/150.)**(3/2)", 2e-7),         # expoente 1
                                     # from left to right: TGAS*3 is real before /2
                                     ("1.0D-9*TGAS*3/2", 450e-9),
                                     ("1.0D-7*TGAS/300/2", 0.5e-7),
                                     ("1.0D-9*(TGAS/300.)**3/2", 0.5e-9),
                                     ("1.0D-7*(3/2**2)", 0.0),                    # 2**2 first: 3/4 = 0
                                     ("1.0D-7*(3/(-2))", -1e-7)])
def test_divisao_inteira_segue_o_fortran(tmp_path, expr, k):
    # the compiled ZDPlasKin does integer division; the reader does the same and warns
    q = _com_taxas(tmp_path, depois=f"rrt(15) = {expr}")
    assert _k15(q) == pytest.approx(k, rel=1e-14, abs=1e-30)
    # it warns only when there is an actual division between integers
    inteira = "TGAS" not in expr or "**(3/2)" in expr
    assert any("integer" in a and "reaction 15" in a for a in q.avisos) == inteira


@pytest.mark.parametrize("expr, valor", [
    # 1.0/3.0 is a single precision operation: 0.33333334 (the air mechanism of
    # ZDPlasKin uses TGAS**(1.0/3.0) in the V-T rates)
    ("1.0D-12*TGAS**(1.0/3.0)", 1.0e-12 * 300.0 ** float(np.float32(1.0) / np.float32(3.0))),
    ("1.0D-12*TGAS**(1.0D0/3.0)", 1.0e-12 * 300.0 ** (1.0 / 3.0)),            # a double: double precision operation
    ("1.0D-12*(0.5*2.1 + TGAS)", 1.0e-12 * (float(np.float32(np.float32(0.5) * np.float32(2.1))) + 300.0)),
    ("1.0D-12*(0.1*3.0)", 1.0e-12 * float(np.float32(0.1) * np.float32(3.0))),    # rounds in float32
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
    # TE below 1e4 everywhere: the first branch does not apply
    np.testing.assert_allclose(_k15_celulas(q, [400.0], te=5e3), [3e-7], rtol=1e-14)


def test_condicao_que_comeca_com_not(tmp_path):
    q = _com_taxas(tmp_path, "  A1 = 1.0D0", "  if (.not. TGAS .gt. 350.0D0) A1 = 2.0D0", depois="rrt(15) = A1*1.0D-7")
    np.testing.assert_allclose(_k15_celulas(q, [300.0, 400.0]), [2e-7, 1e-7], rtol=1e-14)


def test_variavel_sem_valor_fora_do_ramo_e_erro_na_avaliacao(tmp_path):
    # defined only inside an IF without ELSE: in Fortran the value is undefined
    q = _com_taxas(tmp_path, "  if (TGAS > 350.) then", "    A1 = 2.0D0", "  end if", depois="rrt(15) = A1*1.0D-7")
    with pytest.raises(ErroQuimica, match="reaction 15"):
        _k15_celulas(q, [300.0, 400.0])


def test_taxa_do_bolsig_misturada_com_expressao_e_recusada(tmp_path):
    with pytest.raises(ErroQuimica, match="BOLSIG"):
        _com_taxas(tmp_path, depois="rrt(15) = 1.0D-29\n  if (TGAS > 350.) rrt(1) = 0.0D0")


@pytest.mark.parametrize("linha", ["  if (TGAS > 350.) return", "  else if (TGAS > 1."])
def test_controle_que_o_leitor_nao_segue_e_recusado(tmp_path, linha):
    with pytest.raises(ErroQuimica):
        _com_taxas(tmp_path, "  if (TGAS > 300.) then", linha, "  end if")


def test_configuracao_sem_uso_e_ignorada(tmp_path):
    # the preprocessor may emit De = ZDPlasKin_cfg(6) (electron
    # diffusion) unused: only a rate that uses it is refused
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
    # kinet.inp of Sun et al. 2022: parameter :: a=..., b=...; the comma inside
    # parentheses does not separate declarations
    q = _com_taxas(tmp_path, "  DOUBLE PRECISION, PARAMETER :: A0 = 2.0D0, B0 = MAX(3.0D0, 1.0D0)*A0, C0=1.0D-7",
                   depois="rrt(15) = B0*C0")
    assert _k15(q) == pytest.approx(6e-7, rel=1e-14, abs=0)


def test_lista_em_varios_blocos_data():
    # with many reactions the preprocessor splits the list into data x(1:54),
    # data x(55:108)...: the reader joins the blocks in the order of the indices
    from quimica_zdp import _lista_data
    texto = """  data reaction_sign(3:4) &
  /"C=>D","N2(A'1)=>N2"/
  data reaction_sign(1:2) &
  /"A=>B", &
   "B=>C"/
"""
    assert _lista_data(texto, "reaction_sign") == ["A=>B", "B=>C", "C=>D", "N2(A'1)=>N2"]


def test_taxa_igual_a_de_outra_reacao(tmp_path):
    # rrt(46) = rrt(45) in the kinet.inp of Sun et al. 2022: the coefficient of an
    # earlier reaction (before multiplying by the densities)
    q = _com_taxas(tmp_path, depois="rrt(15) = 2.0D0*rrt(14)")
    k = q.coeficientes(np.zeros((1, 7)), np.array([300.0]), np.array([1e4]), np.array([10.0]))
    assert k[14, 0] == pytest.approx(2.0 * k[13, 0], rel=1e-14, abs=0) and k[13, 0] > 0.0


def test_taxa_de_reacao_posterior_e_recusada(tmp_path):
    with pytest.raises(ErroQuimica, match="later"):
        _com_taxas(tmp_path, depois="rrt(15) = rrt(16)")


def test_densidade_total_de_neutros(tmp_path):
    # ANY_NEUTRAL of ZDPlasKin (cm-3), generic third body in the kinet.inp of
    # Sun et al. 2022: it enters as Tgas, Te and EN
    q = _com_taxas(tmp_path, "  call ZDPlasKin_get_density_total(ALL_NEUTRAL=ANY_NEUTRAL)",
                   depois="rrt(15) = 1.0D-29*ANY_NEUTRAL")
    k = q.coeficientes(np.zeros((1, 7)), np.array([300.0]), np.array([1e4]), np.array([10.0]),
                       any_neutral=np.array([2e19]))
    assert k[14, 0] == pytest.approx(2e-10, rel=1e-14, abs=0)
    with pytest.raises(ErroQuimica, match="ANY_NEUTRAL"):
        _k15(q)


def test_modelo_passa_a_densidade_de_neutros(pecas_argonio):
    # fixed gas: the background and the transported neutrals (Ar*), without the ions
    from modelo import Modelo
    md = Modelo(*pecas_argonio, n_celulas=4)
    n = md.separa(md.estado_inicial(n_e0=1e9, neutro=True))[0]
    n[md.nomes.index("AR*")] = 3e12
    np.testing.assert_allclose(md.densidade_neutra(n), md.n_gas + 3e12, rtol=1e-15)


# call runs code the reader does not have; density() the rate routine of
# ZDPlasKin does not even see (the use of the module does not import it)
@pytest.mark.parametrize("linha, mensagem", [("  call minha_rotina(TGAS)", "call"),
                                             ("  TEFF = density(1)", "DENSITY"),
                                             ("  if (TGAS > 300.) then", "IF")])
def test_construcao_nao_suportada_e_erro_claro(tmp_path, linha, mensagem):
    with pytest.raises(ErroQuimica, match=mensagem):
        _com_taxas(tmp_path, linha)


def test_sintaxe_invalida_vira_erro_claro(tmp_path):
    with pytest.raises(ErroQuimica, match="reaction 15"):
        _com_taxas(tmp_path, depois="rrt(15) = 1.0D-29 * * 2")


@precisa_terceiros
def test_reacao_do_bolsig_com_dois_eletrons_e_recusada(tmp_path_factory):
    # e + N2^+ + e => e + N2 with the BOLSIG+ rate (cm3/s, two-body) used
    # as three-body: an error of orders of magnitude (review of the tool)
    from modelo import Modelo
    from tabela import TabelaTg
    # the original kinet.inp of the ZDPlasKin example, before the replacement in the case
    pasta = tmp_path_factory.mktemp("n2_original")
    kinet = (RAIZ / "casos" / "n2" / "kinet.inp").read_text()
    kinet = kinet.replace("e + N2^+ + e => e + N2             !   2.08d-19*(300.0d0/Te)**4.50",
                          "e + N2^+ + e => e + N2             !   BOLSIG N2^+ -> N2")
    (pasta / "kinet.inp").write_text(kinet)
    caso.gera_zdplaskin(pasta)
    q = Quimica(pasta / "zdplaskin_m.F90")
    assert q.bolsig_tres_corpos == ["bolsig:N2^+->N2"]
    assert Quimica(RAIZ / "casos" / "n2" / "zdplaskin_m.F90").bolsig_tres_corpos == []
    with pytest.raises(ErroQuimica, match=r"N2\^\+->N2.*two-body"):
        Modelo(q, TabelaTg(RAIZ / "casos" / "n2" / "tabela_energia_tg.dat"), n_celulas=4,
               gas=RAIZ / "casos" / "n2" / "gas.json", transporte=RAIZ / "casos" / "n2" / "transporte.json")


def test_attachment_sem_limiar(tmp_path):
    # in the LXCat format ATTACHMENT has no threshold line
    banco = (RAIZ / "bolsigdb.dat").read_text() + ("\nATTACHMENT\nAR -> AR^-\nCOMMENT: sintético\n"
                                                    "-----------------------------\n 1.0 1e-22\n 2.0 1e-22\n"
                                                    "-----------------------------\n")
    (tmp_path / "b.dat").write_text(banco)
    q = Quimica(RAIZ / "zdplaskin_m.F90")
    np.testing.assert_array_equal(q.energia_eletronica(tmp_path / "b.dat", "AR"), q.energia_eletronica(RAIZ / "bolsigdb.dat", "AR"))


# ---------------------------------------------- fast heat from the electrons
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
    # the background species are not transported: none requires diffusion; the
    # mobility of an ion comes from the library only with the value in the mixture (key
    # "ESPECIE em A+B", in the order of gas.json), and not from the value in a single gas
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


# ------------------------------- cross sections with more than 1000 points
def _bolsigdb(pontos):
    tabela = "\n".join(f" {e:.6e}\t{1e-20 * (1 + e):.6e}" for e in np.linspace(2.0, 100.0, pontos))
    return ("EXCITATION\nN -> N(2D)\n 2.0e+0 / threshold\nPROCESS: E + N -> E + N(2D)\n"
            "-----------------------------\n" + tabela + "\n-----------------------------\n")


@precisa_terceiros
def test_secao_com_mais_de_900_pontos_e_recusada(tmp_path):
    # bolsig_x86_64.so crashes with large tables (ammonia case, N -> N(2D)
    # with 1291 points; 999 crashes, 900 passes)
    (tmp_path / "bolsigdb.dat").write_text(_bolsigdb(1291))
    (tmp_path / "zdplaskin_m.F90").write_text("")
    with pytest.raises(caso.ErroCaso, match=r"N->N\(2D\).*1291"):
        caso.gera_tabelas(tmp_path)


def test_reducao_das_secoes_grandes(tmp_path):
    # removes, one by one, the point whose absence changes the curve least, until 900 remain;
    # keeps the first and the last; a narrow peak (resonance) stays
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
    # BOLSIG+ rebuilds the grid between neighbouring E/N and the mean energy may fall
    # 0.1% (ammonia, near 350 Td): the line is removed, in each T_g block, and
    # recorded; the prototype inverts the table by energy and requires monotonicity
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
    assert "\n\n600" in arq.read_text()            # the blank line between blocks stays


def test_limiar_colado_a_barra(tmp_path):
    # database of Sun et al. 2022: "2.000000e-2/threshold energy", without a space
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


# ------------------------------------------- BOLSIG+ limits without a manual step
def _bolsigdb_varios(pastas_pontos):
    """Synthetic bolsigdb.dat: one EXCITATION process per item, with n points."""
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
    assert caso.reduz_se_preciso(tmp_path) == {}            # already reduced: nothing to do
    assert caso.reduz_se_preciso in caso.ACOES["tudo"]


def test_reduz_nao_sobrescreve_bolsigdb_trocado(tmp_path):
    # with the bolsigdb_original.dat of an earlier reduction, a new bolsigdb.dat
    # (replaced by the user) cannot be overwritten by the reduction of the old one
    (tmp_path / "bolsigdb.dat").write_text(_bolsigdb_varios([("N2 -> N2*", 1200)]))
    caso.reduz_bolsigdb(tmp_path)
    novo = _bolsigdb_varios([("H2 -> H2*", 1300)])
    (tmp_path / "bolsigdb.dat").write_text(novo)
    with pytest.raises(caso.ErroCaso, match="bolsigdb_original.dat"):
        caso.reduz_bolsigdb(tmp_path)
    assert (tmp_path / "bolsigdb.dat").read_text() == novo


def _gerador_que_cai_acima_de(limite_td):
    """Fake generator: writes the progress as gera_tabela_energia.f90 and crashes when
    passing limite_td, unless en_max.txt lies below."""
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
    assert "4343" in (tmp_path / "tabelas_topo.txt").read_text()     # where it crashed, recorded
    assert not (tmp_path / "progresso_tabelas.txt").exists()


def test_progresso_velho_nao_vira_queda_e_en_max_volta(tmp_path):
    # a progresso_tabelas.txt of an interrupted run and a compilation
    # failure: no new top, and the en_max.txt of the user stays as it was
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
    # BOLSIG+ always crashes at the last point: the top goes down until the attempts
    # run out, and then the original en_max.txt comes back
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
    # crashing at low E/N is not a high-field limit: an error with the output, without en_max.txt
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
    assert "WARNING: reaction 15: integer division" in capsys.readouterr().out


# --------------------------------------- estimate of the transport data
def test_mobilidade_de_langevin_e_lei_de_blanc():
    # K0 = 13.853/sqrt(alpha mu) (cm2/(V s), alpha in Å3, mu in u) in each gas;
    # in the mixture, 1/K = sum of x_i/K_i
    k_n2 = 13.853 / np.sqrt(1.7403 * caso.massa_u("N2^+") * caso.massa_u("N2") / (caso.massa_u("N2^+") + caso.massa_u("N2")))
    assert caso.mobilidade_estimada("N2^+", {"N2": 1.0}) == pytest.approx(k_n2, rel=1e-12)
    k_h2 = 13.853 / np.sqrt(0.8045 * caso.massa_u("N2^+") * caso.massa_u("H2") / (caso.massa_u("N2^+") + caso.massa_u("H2")))
    assert caso.mobilidade_estimada("N2^+", {"N2": 0.25, "H2": 0.75}) == pytest.approx(
        1.0 / (0.25 / k_n2 + 0.75 / k_h2), rel=1e-12)
    # a polarisability given in gas.json overrides that of the table
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
        assert "estimate" in t[ion]["fonte"]
    # the wall loss is not estimated: it stays listed
    assert all("perda_na_parede" in c for c in faltam.values())
    assert not any("mobilidade" in c for c in faltam.values())


def test_estima_a_difusao_pela_teoria_cinetica(tmp_path):
    pytest.importorskip("cantera")
    pasta = _pasta_n2_sem_transporte(tmp_path)
    caso.gera_transporte(pasta, biblioteca={}, estimar=True)
    t = json.loads((pasta / "transporte.json").read_text())
    # excited states with the diffusion of the ground state (N2(A) as N2)
    assert t["N2(A)"]["difusao"] == t["N2(B)"]["difusao"]
    assert 1e-5 < t["N"]["difusao"]["D0"] < 1e-4 and 1.5 < t["N"]["difusao"]["expoente"] < 2.0
    assert "estimate" in t["N"]["fonte"]
    assert "difusao" in t["N2"]                 # the background too, for --free-gas


# ------------------------------------------------ enthalpies for the heat block
def test_entalpias_pelos_limiares_do_bolsig():
    # of N2 (0 eV) from the BOLSIG+ processes of the case: excited states and N2+ with the
    # lowest threshold of each product (v = 0), equal to the curated ones in the gas.json of N2
    q = Quimica(RAIZ / "casos" / "n2" / "zdplaskin_m.F90")
    curadas = json.loads((RAIZ / "casos" / "n2" / "gas.json").read_text())["calor"]["entalpias"]
    h, origem = caso.propaga_entalpias(q, caso.limiares_do_bolsigdb(RAIZ / "casos" / "n2" / "bolsigdb.dat"),
                                       {"N2": 0.0})
    for esp in ("N2(A)", "N2(B)", "N2(C)"):
        assert h[esp] == pytest.approx(curadas[esp], abs=1e-3), esp
        assert "threshold" in origem[esp]
    # bolsigdb.dat gives the rounded ionisation (15.6 eV); the curated one is that of NIST
    assert h["N2^+"] == 15.6 and h["N2^+"] == pytest.approx(curadas["N2^+"], abs=0.02)
    assert "N4^+" not in h and "N" not in h          # no process reaches them


def test_entalpias_incompletas_nao_ligam_o_calor(tmp_path):
    # without a heat block, writing an incomplete one would make the model refuse the case: it goes
    # to a separate file, and gas.json does not change
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
    assert h["N4^+"] == 14.481                                        # what existed stays
    assert h["N"] == pytest.approx(4.899, abs=0.01) and abs(h["N2"]) < 0.01
    assert "N3^+" in faltam and "N4^+" not in faltam


def test_terceiros_por_variavel_de_ambiente(tmp_path, monkeypatch):
    # ZDPlasKin and BOLSIG+ are not in the package (the licence does not allow
    # redistribution): the tool finds them through ZDPLASKIN_DIR and BOLSIG_SO
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
