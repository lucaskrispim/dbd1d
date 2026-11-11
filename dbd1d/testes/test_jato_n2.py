# SPDX-License-Identifier: GPL-3.0-or-later
"""Pulsed N2 jet of Kuhfeld et al. 2021 (docs/jato_n2): conditions, run and
comparison scripts (scripts/jato_n2*.py)."""
import json
import sys

import numpy as np
import pytest

from conftest import RAIZ, precisa_arquivo

# the jet scripts are not distributed in the package (see INCLUI in empacotamento/monta_pacote.py)
_SCRIPTS_JATO = ("scripts/jato_n2.py", "scripts/jato_n2_ttdf.py", "scripts/jato_n2_compara.py",
                 "scripts/jato_n2_acoplado.py")
if not all((RAIZ / c).exists() for c in _SCRIPTS_JATO):
    pytest.skip("needs the scripts/jato_n2*.py scripts, which are not distributed", allow_module_level=True)

sys.path.insert(0, str(RAIZ / "scripts"))
import jato_n2  # noqa: E402

precisa_dados = precisa_arquivo("docs/literatura/dados/kuhfeld2021/II_fig6_3kV_200ns.csv")


def test_troca_fluxo_vale_83():
    # Kuhfeld et al. 2021 II, eq. 17: C Q p_atm / (V p) with half of the 20 sccm in each half
    assert jato_n2.troca_fluxo() == pytest.approx(83.33, rel=1e-3)


@precisa_dados
def test_plato_3kv_200ns():
    # mean of figure 6 from its first point (31 ns, after the breakdown) to the nominal end of the pulse
    t_ini, ne = jato_n2.plato("3kV_200ns", 200e-9)
    assert t_ini == pytest.approx(31e-9, abs=1e-9)
    assert ne == pytest.approx(2.07e13, rel=0.01)
    assert jato_n2.plato("4kV_200ns", 200e-9)[1] == pytest.approx(2.92e13, rel=0.01)


@precisa_dados
def test_condicoes_tem_tres():
    conds = jato_n2.condicoes()
    assert [c["id"] for c in conds] == ["3kV_200ns", "3kV_250ns", "4kV_200ns"]
    for c in conds:
        assert c["ne"] == pytest.approx(c["ne_csv"], rel=0.01)
        assert c["periodo"] == 1e-3
    assert [c["duracao"] for c in conds] == pytest.approx([200e-9, 250e-9, 200e-9], rel=1e-12)


@precisa_arquivo("casos/n2_amonia_v45/quimica.json")
def test_gamas_jato_cobre_a_escada():
    from coluna import Coluna0D
    c = Coluna0D(RAIZ / "casos" / "n2_amonia_v45", 2e4, 330.0, 0.0541)
    g = jato_n2.gamas_jato(c, 1e-3)
    escada = [s for s in c.especies if s.upper().startswith("N2(V")]
    assert len(escada) == 45 and all(g[s] == 1e-3 for s in escada)
    assert g["N"] == 3.2e-4 and g["N2(A3)"] == 1.0


def test_variacoes_conhecidas():
    primeiras = {"base", "parede_he24", "fluxo_tk", "ne_menos", "ne_mais", "gama_1e-4", "gama_1e-2", "ions_no_fim"}
    assert primeiras <= set(jato_n2.VARIACOES)
    assert all(n.startswith("corrigido") for n in set(jato_n2.VARIACOES) - primeiras)
    with pytest.raises(KeyError):
        jato_n2.opcoes("nenhuma")


def test_opcoes_da_base_e_das_variacoes():
    base = jato_n2.opcoes("base")
    assert base == dict(gama_v=1e-3, quantum_na_parede=False, fator_troca=1.0, fator_ne=1.0, ions_no_fim=False,
                        pulso="retangulo", parede="tubo", en="fixo", criterio="antigo")
    assert jato_n2.opcoes("parede_he24")["gama_v"] == 3e-3 and jato_n2.opcoes("parede_he24")["quantum_na_parede"]
    assert jato_n2.opcoes("fluxo_tk")["fator_troca"] == pytest.approx(330.0 / 273.15)
    assert jato_n2.opcoes("ne_mais")["fator_ne"] == 1.2 and jato_n2.opcoes("ne_menos")["fator_ne"] == 0.8


def test_saidas_pos_depois_do_pulso_mais_longo():
    assert jato_n2.SAIDAS_POS[0] > 2.75e-7 and jato_n2.SAIDAS_POS[-1] == 1e-3 and len(jato_n2.SAIDAS_POS) == 60


import jato_n2_ttdf  # noqa: E402


def test_ttdf_soma_um():
    assert jato_n2_ttdf.ttdf(np.arange(46), 0.05, 1300.0, 5000.0).sum() == pytest.approx(1.0, abs=1e-12)


def test_ttdf_parte_quente_so_acima_de_v0():
    # Kuhfeld et al. 2021 I, eq. 25: the hot molecules have v >= 1, so with R_h the ground state loses
    sem = jato_n2_ttdf.ttdf(np.arange(46), 0.0, 1300.0, 5000.0)
    com = jato_n2_ttdf.ttdf(np.arange(46), 0.1, 1300.0, 5000.0)
    assert com[0] == pytest.approx(0.9 * sem[0], rel=1e-12)


def test_ajusta_ttdf_recupera_parametros():
    pop = jato_n2_ttdf.ttdf(np.arange(9), 0.05, 1300.0, 5000.0)
    r_h, t_h = jato_n2_ttdf.ajusta_ttdf(pop, 1300.0)
    assert r_h == pytest.approx(0.05, rel=1e-3) and t_h == pytest.approx(5000.0, rel=1e-3)


def test_tvib_01_de_boltzmann():
    pop = jato_n2_ttdf.ttdf(np.arange(9), 0.0, 1300.0, 5000.0)
    assert jato_n2_ttdf.tvib_01(pop) == pytest.approx(1300.0, rel=1e-10)


import jato_n2_compara  # noqa: E402


def test_metrica_identica_da_zero():
    t = np.geomspace(1e-6, 1e-3, 30)
    y = 1.0 + np.log10(t / 1e-6)
    m = jato_n2_compara.metrica(t, y, t[::3], y[::3], 0.9 * y[::3], 1.1 * y[::3])
    assert m["rms_log10"] == pytest.approx(0.0, abs=1e-12) and m["frac_na_barra"] == 1.0 and m["n"] == 10


def test_metrica_fator_dois():
    t = np.geomspace(1e-6, 1e-3, 30)
    y = 1.0 + np.log10(t / 1e-6)
    m = jato_n2_compara.metrica(t, 2 * y, t, y, 0.9 * y, 1.1 * y)
    assert m["rms_log10"] == pytest.approx(np.log10(2.0), abs=1e-12) and m["frac_na_barra"] == 0.0


def test_metrica_ignora_pontos_nao_positivos_e_fora_da_janela():
    t = np.geomspace(1e-7, 1e-3, 30)
    y = np.ones(30)
    t_exp = np.array([5e-7, 2e-6, 3e-6, 4e-6])
    y_exp = np.array([1.0, 1.0, 0.0, -1.0])
    assert jato_n2_compara.metrica(t, y, t_exp, y_exp, y_exp, y_exp, t_min=1e-6)["n"] == 1
    assert jato_n2_compara.metrica(t, y, t_exp, y_exp, y_exp, y_exp)["n"] == 2


def test_tempo_sim_eixo_log():
    # log time axes of papers I and II: the CARS delays 20, 40, 60 ns are -40, -20, 0 ns of the linear
    # axes (KI: the first three points are before the ignition), so the pulse starts at 60 ns
    t = jato_n2_compara.tempo_sim("fig8_v1_medidos", np.array([60e-9, 1e-3]))
    np.testing.assert_allclose(t, [0.0, 1e-3 - 60e-9])


def test_tempo_sim_eixo_linear():
    t = jato_n2_compara.tempo_sim("I_fig15_3kV_200ns", np.array([0.0, 200e-9]))
    np.testing.assert_allclose(t, [0.0, 200e-9])
    np.testing.assert_allclose(jato_n2_compara.tempo_sim("I_fig15_3kV_200ns", np.array([0.0]), 50e-9), [50e-9])


def test_figuras_das_populacoes():
    assert jato_n2_compara.FIGURA_POPULACOES == {"3kV_200ns": "fig8", "3kV_250ns": "figA1", "4kV_200ns": "figA2"}


def test_metrica_ignora_simulacao_nao_finita():
    t = np.geomspace(1e-7, 1e-3, 30)
    y = np.ones(30)
    y[:10] = np.nan
    m = jato_n2_compara.metrica(t, y, t, np.ones(30), np.ones(30), np.ones(30))
    assert m["n"] == 20 and m["rms_log10"] == pytest.approx(0.0, abs=1e-12)


def test_ttdf_simulada_descarta_temperatura_sem_parte_quente():
    # with R_h below RH_MIN the hot temperature is not identifiable: NaN
    t = np.array([0.0, 1e-8, 1e-7])
    frac = np.array([jato_n2_ttdf.ttdf(np.arange(9), r, 1300.0, 5000.0) for r in (0.0, 1e-3, 0.03)])
    sim = {"t_s": t, **{f"frac_v{v}": frac[:, v] for v in range(9)}}
    tt, r_h, t_h = jato_n2_compara.ttdf_simulada(sim)
    assert np.isnan(t_h[0]) and t_h[1] == pytest.approx(5000.0, rel=1e-3)



import jato_n2_acoplado  # noqa: E402


def test_media_no_pulso_trapezio():
    t = np.array([0.0, 1e-7, 2e-7, 5e-7])
    vdfs = np.array([[1.0, 0.0], [0.8, 0.2], [0.6, 0.4], [0.0, 1.0]])
    np.testing.assert_allclose(jato_n2_acoplado.media_no_pulso(t, vdfs, 2e-7), [0.8, 0.2], rtol=1e-12)


@precisa_arquivo("casos/n2_amonia/quimica.json")
def test_vdf_soma_um_e_inclui_v0():
    from coluna import Coluna0D
    c = Coluna0D(RAIZ / "casos" / "n2_amonia", 2e4, 330.0, 0.0541)
    y = c.estado_inicial(0.0)
    y[c.especies.index("N2(V1)")] = 0.1 * c.n_gas
    f = jato_n2_acoplado.vdf(c, y)
    assert len(f) == 9 and f.sum() == pytest.approx(1.0) and f[1] == pytest.approx(0.1, rel=1e-3)


@precisa_dados
@precisa_arquivo("casos/n2_amonia/quimica.json")
def test_monta_aceita_caminho_absoluto_do_caso():
    # the coupled cases live outside casos/; jato_n2.monta joins an absolute path as it is
    c, _ = jato_n2.monta(str(RAIZ / "casos" / "n2_amonia"), jato_n2.condicoes()[0])
    assert len(jato_n2.niveis(c)) == 8


def test_mudanca_ln_ignora_populacoes_pequenas():
    a = np.array([0.9, 0.1, 1e-12])
    b = np.array([0.9, 0.11, 5e-12])
    assert jato_n2_acoplado.mudanca_ln(a, b) == pytest.approx(np.log(1.1), rel=1e-12)


def test_curvas_das_variacoes_em_ordem_fixa_e_so_as_que_existem():
    rodadas = {"acoplado": "c", "base": "a", "gama_1e-4": "x"}
    curvas = jato_n2_compara.curvas_variacoes(rodadas)
    assert list(curvas) == ["A45 base", "A45 acoplado"]
    assert curvas["A45 acoplado"] == ("c", jato_n2_compara.VARIACOES_FIGURA["acoplado"][1])


def test_metrica_deixa_fora_pontos_no_piso_da_digitalizacao():
    # v = 4 of figures 8, A1 and A2 has points drawn on the lower edge of the axis (about 1e-4 %)
    t = np.geomspace(1e-6, 1e-3, 30)
    y = np.full(30, 0.05)
    t_exp = np.array([2e-6, 3e-6, 4e-6])
    y_exp = np.array([0.05, 9.3e-5, 0.05])
    m = jato_n2_compara.metrica(t, y, t_exp, y_exp, 0.9 * y_exp, 1.1 * y_exp, piso=jato_n2_compara.PISO_PCT)
    assert m["n"] == 2 and m["rms_log10"] == pytest.approx(0.0, abs=1e-12)


def test_frac_na_barra_so_conta_pontos_com_barra():
    t = np.geomspace(1e-6, 1e-3, 30)
    y = np.ones(30)
    t_exp = np.array([2e-6, 3e-6])
    y_exp = np.array([1.0, 1.1])
    m = jato_n2_compara.metrica(t, y, t_exp, y_exp, np.array([0.9, 1.1]), np.array([1.1, 1.1]))
    assert m["frac_na_barra"] == 1.0 and m["n"] == 2


def test_t_vib_h_so_onde_ki_diz_que_e_confiavel():
    # KI, fig. 16: not reliable before 100 ns (3 kV) and 50 ns (4 kV)
    assert jato_n2_compara.T_H_CONFIAVEL == {"3kV_200ns": 100e-9, "3kV_250ns": 100e-9, "4kV_200ns": 50e-9}


def test_linha_de_base_e_a_media_antes_do_pulso():
    t = np.array([-40e-9, -20e-9, 0.0, 20e-9])
    np.testing.assert_allclose(jato_n2_compara.linha_de_base(t, np.array([0.01, 0.02, 0.03, 0.5])), 0.02)


def test_ajuste_ttdf_so_nos_niveis_detectaveis():
    # KI: the measured spectra show v up to 3 early in the pulse and the afterglow populations go to v = 4
    np.testing.assert_array_equal(jato_n2_ttdf.V_AJUSTE, np.arange(5))


@precisa_arquivo("docs/literatura/dados/kuhfeld2021/fig8_v4_medidos.csv")
def test_modelo_de_kuhfeld_ii_tambem_sem_os_pontos_no_piso():
    med = jato_n2_compara.le("fig8_v4_medidos")
    calc = jato_n2_compara.le("fig8_v4_calculado")
    na_janela = (med[:, 0] >= max(jato_n2_compara.T_AFTERGLOW + jato_n2_compara.INICIO_LOG, calc[:, 0].min())) & \
        (med[:, 0] <= calc[:, 0].max())
    linha = [x for x in jato_n2_compara.linhas_kuhfeld("3kV_200ns") if x["v"] == 4][0]
    assert linha["n"] == int(np.sum(na_janela & (med[:, 1] > jato_n2_compara.PISO_PCT)))


def test_piso_vira_limite_superior_na_barra_do_menor_valor_medido():
    t = np.geomspace(1e-6, 1e-3, 30)
    y = np.full(30, 0.05)
    t_exp = np.array([2e-6, 3e-6, 4e-6, 5e-6])
    y_exp = np.array([0.05, 9.3e-5, 9.3e-5, 0.03])
    m = jato_n2_compara.metrica(t, y, t_exp, y_exp, 0.9 * y_exp, 1.1 * y_exp, piso=jato_n2_compara.PISO_PCT)
    # detection limit = smallest measured value above the floor (0.03); the simulation (0.05) is above it
    assert m["n_limite"] == 2 and m["frac_limite"] == 0.0
    m = jato_n2_compara.metrica(t, 0.01 * y, t_exp, y_exp, 0.9 * y_exp, 1.1 * y_exp, piso=jato_n2_compara.PISO_PCT)
    assert m["frac_limite"] == 1.0
    # the limit is the top of the bar of that smallest point (0.033), not the point itself
    m = jato_n2_compara.metrica(t, 0.032 / 0.05 * y, t_exp, y_exp, 0.9 * y_exp, 1.1 * y_exp,
                                piso=jato_n2_compara.PISO_PCT)
    assert m["frac_limite"] == 1.0


@precisa_arquivo("docs/jato_n2/condicoes.csv")
def test_t_vib_c_de_ki_igual_ao_csv():
    import csv
    with open(RAIZ / "docs" / "jato_n2" / "condicoes.csv") as f:
        linhas = list(csv.DictReader(l for l in f if not l.startswith("#")))
    assert jato_n2_compara.TVIB_C_KI == {r["id"]: float(r["Tvib_c_K"]) for r in linhas}


def test_ttdf_simulada_aceita_t_vib_c_fixo():
    pop = jato_n2_ttdf.ttdf(np.arange(9), 0.04, 1200.0, 5000.0)
    sim = {"t_s": np.array([0.0, 1e-7])}
    for v in range(9):
        sim[f"frac_v{v}"] = np.array([pop[v], pop[v]])
    t, r_h, _ = jato_n2_compara.ttdf_simulada(sim, tvib_c=1200.0)
    assert r_h[0] == pytest.approx(0.04, rel=1e-3)


def test_quanta_comparado_e_o_de_ki():
    # KI, eq. 31 and 32: quanta from the populations up to the highest detected level (v = 4)
    assert jato_n2_compara.QUANTA_KI == "quanta_v4"


def test_metrica_antes_do_pulso_compara_o_fim_do_periodo_com_os_pontos_antes_da_ignicao():
    t_exp = np.array([-40e-9, -20e-9, 0.0, 20e-9])
    y_exp = np.array([1.0, 1.0, 1.0, 5.0])
    m = jato_n2_compara.metrica_pre(2.0, t_exp, y_exp, 0.9 * y_exp, 1.1 * y_exp)
    assert m["n"] == 3 and m["rms_log10"] == pytest.approx(np.log10(2.0)) and m["frac_na_barra"] == 0.0


def test_janela_tolera_o_arredondamento_da_digitalizacao():
    # I_fig15_3kV_200ns has a point at 99.97 ns that belongs to the window starting at 100 ns
    t = np.geomspace(1e-9, 1e-6, 30)
    y = np.ones(30)
    m = jato_n2_compara.metrica(t, y, np.array([99.97e-9, 150e-9]), np.ones(2), 0.9 * np.ones(2), 1.1 * np.ones(2),
                                t_min=100e-9)
    assert m["n"] == 2


@precisa_arquivo("docs/jato_n2/resultados/n2_amonia_v45_3kV_200ns.csv")
def test_pontos_antes_da_ignicao_nao_entram_tambem_na_metrica_do_periodo():
    sim = jato_n2_compara.le_simulacao(RAIZ / "docs" / "jato_n2" / "resultados" / "n2_amonia_v45_3kV_200ns.csv")
    linhas = {x["grandeza"]: x for x in jato_n2_compara.linhas_de("A45", "3kV_200ns", "base", sim)}
    d = jato_n2_compara.le("I_fig20_3kV_200ns")
    t = jato_n2_compara.tempo_sim("I_fig20_3kV_200ns", d[:, 0])
    assert linhas["quanta_v4"]["n"] + linhas["quanta_v4_pre"]["n"] == int(np.sum(t <= sim["t_s"].max()))


def test_barra_que_nao_contem_o_ponto_nao_entra_na_fracao():
    # fig8_v4_medidos at 45 us: value 0.0177 with the bar from 1e-4 to 0.0138 (bar paired with another point)
    t = np.geomspace(1e-6, 1e-3, 30)
    y = np.ones(30)
    t_exp = np.array([2e-6, 3e-6])
    m = jato_n2_compara.metrica(t, y, t_exp, np.array([1.0, 1.5]), np.array([0.9, 1.2]), np.array([1.1, 1.4]))
    assert m["frac_na_barra"] == 1.0 and m["n"] == 2


def test_legenda_da_quimica_b_diz_que_ela_nao_tem_v_v():
    # casos/n2_tese has no N2(v) + N2(w) reaction; the v45 ladders have 1981
    assert "sem V-V" in jato_n2_compara.LEGENDA["B"]


# corrected runs (docs/superpowers/plans/2026-11-10-jato-n2-correcao.md)

def _cond(cid):
    return next(c for c in jato_n2.condicoes() if c["id"] == cid)


# dose from 0 to the last point above 1e12; the rest of the tail (0.4 to 0.6 %) goes to the afterglow
@pytest.mark.parametrize("cid, dose, fim", [("3kV_200ns", 3.921e6, 252e-9), ("3kV_250ns", 4.843e6, 302e-9),
                                            ("4kV_200ns", 5.441e6, 258e-9)])
def test_perfil_de_ne_da_fig6(cid, dose, fim):
    t, n = jato_n2.perfil_ne(cid)
    assert t[0] == 0.0 and t[-1] == pytest.approx(fim, abs=1e-10)
    assert np.trapezoid(n, t) == pytest.approx(dose, rel=2e-3)


@pytest.mark.parametrize("cid, en", [("3kV_200ns", 86.8), ("3kV_250ns", 78.7), ("4kV_200ns", 77.0)])
def test_en_medido_da_fig13(cid, en):
    assert jato_n2.en_medido(cid) == pytest.approx(en, abs=0.1)


def _nu_n2v(c):
    k, te = c.coeficientes(1e-4)
    nu = c.frequencias_de_perda(1e-4, te)
    return np.array([nu[i] for i, s in enumerate(c.especies) if s.upper().startswith("N2(V")])


@precisa_arquivo("casos/n2_amonia/quimica.json")
def test_parede_do_canal():
    c, _ = jato_n2.monta("n2_amonia", _cond("3kV_200ns"), "corrigido")
    i = next(i for i, s in enumerate(c.especies) if s.upper() == "N2(V1)")
    vbar = np.sqrt(8 * 1.380649e-23 * 330.0 / (np.pi * 28.0134 * 1.66053906660e-27)) * 100.0
    nu_s = vbar * 1e-3 * 40.0 / (2.0 * (2.0 - 1e-3))
    lam = 0.1 / (np.pi * np.sqrt(2.0))
    alvo = 1.0 / (1.0 / nu_s + lam ** 2 / c.difusao(i))
    np.testing.assert_allclose(_nu_n2v(c), alvo, rtol=1e-4)


@precisa_arquivo("casos/n2_amonia/quimica.json")
def test_parede_kii_eq19():
    c, _ = jato_n2.monta("n2_amonia", _cond("3kV_200ns"), "corrigido_parede_kii")
    np.testing.assert_allclose(_nu_n2v(c), 499.7, rtol=2e-4)


@precisa_arquivo("casos/n2_amonia/quimica.json")
def test_corrigido_impoe_o_perfil_e_a_grade():
    _, p = jato_n2.monta("n2_amonia", _cond("3kV_200ns"), "corrigido")
    assert p.duracao == pytest.approx(252e-9, abs=1e-10) and p.perfil_ne is not None and p.passo_saida == 5e-9
    assert p.ne == pytest.approx(_cond("3kV_200ns")["ne"])          # the ion seed stays the plateau mean
    _, q = jato_n2.monta("n2_amonia", _cond("3kV_200ns"), "corrigido_so_parede")
    assert q.perfil_ne is None and q.duracao == pytest.approx(200e-9)


def test_dose_de_ne():
    from pulsado import Pulso
    assert jato_n2.dose_ne(Pulso(81.0, 2e13, 2e-7, 1e-3, 1e-4)) == pytest.approx(4e6)
    perfil = ((0.0, 1e-7, 2e-7), (0.0, 2e13, 2e13))
    assert jato_n2.dose_ne(Pulso(81.0, 2e13, 2e-7, 1e-3, 1e-4, perfil_ne=perfil)) == pytest.approx(3e6)


def test_saidas_da_pos_descarga_corrigida():
    s = jato_n2.saidas_pos(252e-9, corrigido=True)
    assert s[0] == pytest.approx(262e-9) and s[-1] == pytest.approx(1e-3) and np.all(np.diff(s) > 0)
    assert np.sum(s <= 400e-9) >= 14
    np.testing.assert_array_equal(jato_n2.saidas_pos(200e-9, corrigido=False), jato_n2.SAIDAS_POS)


def test_variacoes_corrigidas():
    nomes = {"corrigido", "corrigido_so_pulso", "corrigido_so_parede", "corrigido_parede_kii", "corrigido_parede_he24",
             "corrigido_nenhum",
             "corrigido_en_medido", "corrigido_ne_menos", "corrigido_ne_mais"}
    assert nomes <= set(jato_n2.VARIACOES)
    for n in nomes:
        assert jato_n2.opcoes(n)["criterio"] == "novo"
    assert jato_n2.opcoes("base")["criterio"] == "antigo" and jato_n2.opcoes("base")["pulso"] == "retangulo"


def test_acoplado_corrigido_tem_nome_e_pasta_proprios():
    assert jato_n2_acoplado.nomes("3kV_200ns", "base") == ("n2_amonia_v45_3kV_200ns_acoplado", "jato_3kV_200ns")
    assert jato_n2_acoplado.nomes("3kV_200ns", "corrigido") == ("n2_amonia_v45_3kV_200ns_corrigido_acoplado",
                                                                 "jato_3kV_200ns_corrigido")


def test_figura_dois_por_dois_separa_pulso_e_parede():
    # the cell with neither correction is corrigido_nenhum (same criterion and grid as the others), not base
    rodadas = {"corrigido_nenhum": "a", "corrigido_so_pulso": "b", "corrigido_so_parede": "c", "corrigido": "d",
               "base": "x"}
    curvas = jato_n2_compara.curvas_variacoes(rodadas, jato_n2_compara.FIGURA_DOIS_POR_DOIS)
    assert [v[0] for v in curvas.values()] == ["a", "b", "c", "d"]


def test_figura_das_variacoes_corrigidas():
    rodadas = {"corrigido": "a", "corrigido_ne_menos": "b", "corrigido_acoplado": "c", "base": "x"}
    curvas = jato_n2_compara.curvas_variacoes(rodadas, jato_n2_compara.VARIACOES_FIGURA_CORRIGIDO)
    assert [v[0] for v in curvas.values()] == ["a", "b", "c"]


@precisa_arquivo("casos/n2_amonia/quimica.json")
def test_corrigido_nenhum_so_muda_criterio_e_grade():
    # the cell of the 2 x 2 with neither correction, run like the corrected cells
    op = jato_n2.opcoes("corrigido_nenhum")
    assert (op["pulso"], op["parede"], op["criterio"]) == ("retangulo", "tubo", "novo")
    assert {k: v for k, v in op.items() if k != "criterio"} == {k: v for k, v in jato_n2.opcoes("base").items()
                                                                  if k != "criterio"}
    c, p = jato_n2.monta("n2_amonia", _cond("3kV_200ns"), "corrigido_nenhum")
    cb, pb = jato_n2.monta("n2_amonia", _cond("3kV_200ns"), "base")
    assert p.perfil_ne is None and p.duracao == pytest.approx(200e-9) and p.passo_saida == 5e-9
    assert pb.passo_saida is None and p.ne == pb.ne


def test_versao_registra_entradas_e_mantem_chaves_antigas():
    v = jato_n2.versao()
    assert {"commit", "sha256", "sha256_entradas", "sujo"} <= set(v)
    assert set(v["sha256"]) == set(jato_n2.CODIGO)
    assert "kinet.inp" in v["sha256_entradas"]


@precisa_dados
def test_versao_do_jato_cobre_os_dados_que_le():
    v = jato_n2.versao()
    lidos = ["docs/jato_n2/condicoes.csv"] + [f"docs/literatura/dados/kuhfeld2021/{p}_{c}.csv"
                                              for p in ("II_fig6", "I_fig13") for c in ("3kV_200ns", "3kV_250ns", "4kV_200ns")]
    for a in lidos:
        assert v["sha256_entradas"].get(a) is not None, a


def test_versao_do_jato_acoplado_cobre_o_banco_de_base():
    acoplado = pytest.importorskip("jato_n2_acoplado")
    v = acoplado.versao_do_acoplado()
    assert v["sha256_entradas"]["casos/n2_amonia/bolsigdb.dat"] is not None
    assert "docs/jato_n2/condicoes.csv" in v["sha256_entradas"]


def test_roda_recusa_arvore_suja_antes_de_integrar(monkeypatch):
    monkeypatch.delenv("DBD1D_PERMITE_SUJO", raising=False)
    monkeypatch.setattr(jato_n2, "versao", lambda *a: dict(commit="x", sujo=["dbd1d/coluna.py"]))
    monkeypatch.setattr(jato_n2, "monta", lambda *a, **k: pytest.fail("integration started"))
    with pytest.raises(RuntimeError, match="coluna.py"):
        jato_n2.roda("n2_amonia", dict(id="x"))


def test_raiz_da_coluna_registra_versao():
    raiz = pytest.importorskip("coluna_n2_raiz")
    for a in raiz.CODIGO:
        assert (RAIZ / a).is_file(), a
    v = raiz.versao()
    assert {"commit", "sha256", "sha256_entradas", "sujo"} <= set(v)
    assert v["sha256_entradas"]["tabela_energia_tg.dat"] is not None
    # the coupled bank is built from the base case and the conditions come from the CSV files
    for a in ("casos/n2_amonia/bolsigdb.dat", "docs/coluna_positiva_n2/condicoes.csv",
              "docs/literatura/dados/gordiets1995/n2_puro.csv"):
        assert v["sha256_entradas"].get(a) is not None, a
