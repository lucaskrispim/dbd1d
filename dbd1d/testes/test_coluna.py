# SPDX-License-Identifier: GPL-3.0-or-later
"""Positive column in 0D (coluna.py): chemistry of a case at imposed E/N and T_g,
along the residence time, with the radial losses of a tube of radius R."""
import numpy as np
import pytest

from conftest import RAIZ, precisa_arquivo
from coluna import Coluna0D, atomos, perda_ambipolar, perda_neutro

J0 = 2.404825557695773   # first zero of the Bessel function J0
KB = 1.380649e-23
QE = 1.602176634e-19
CASO = RAIZ / "casos" / "n2_amonia"
precisa_caso = precisa_arquivo("casos/n2_amonia/quimica.json")
# wall probabilities of the article (scripts/coluna_n2.py, GAMAS)
GAMAS = {"N": 3.2e-4, "N2(A3)": 1.0, "N2(a'1)": 1.0, "N(2D)": 1.0, "N(2P)": 1.0}


def test_perda_ambipolar():
    # D_a = mu_i kTe/e, nu = D_a / Lambda^2 with Lambda = R / J0
    mu, te_ev, r = 2000.0, 1.5, 0.3
    assert perda_ambipolar(mu, te_ev, r) == pytest.approx(mu * te_ev / (r / J0) ** 2, rel=1e-12)


def test_perda_neutro_em_serie_difusão_e_parede():
    # Capitelli et al. 2000, eq. (11.7): -D dn/dr = (2 gamma/(2 - gamma)) n vbar/4 at r = R; with the
    # wall area over the volume 2/R of a cylinder the surface term is 2 R (2 - gamma) / (4 gamma vbar),
    # the 2R/(gamma vbar) of Gordiets et al. 1995, eq. (23), for gamma << 1:
    # 1/nu = Lambda^2 / D + R (2 - gamma) / (gamma vbar)
    d, gama, r, vbar = 150.0, 1e-3, 0.3, 4.7e4
    esperado = 1.0 / ((r / J0) ** 2 / d + r * (2 - gama) / (gama * vbar))
    assert perda_neutro(d, gama, r, vbar) == pytest.approx(esperado, rel=1e-12)
    # gamma -> 1: the diffusion limit dominates when the wall term is small
    assert perda_neutro(d, 1.0, r, 1e12) == pytest.approx(d / (r / J0) ** 2, rel=1e-6)
    assert perda_neutro(d, 0.0, r, vbar) == 0.0


@precisa_caso
def test_estado_e_fundo():
    c = Coluna0D(CASO, pressao_pa=230.0, t_gas=290.0, raio_cm=0.3)
    assert "N2" not in c.especies and "E" in c.especies and "N2(A3)" in c.especies
    assert c.n_gas == pytest.approx(230.0 / (KB * 290.0) * 1e-6, rel=1e-12)


@precisa_caso
def test_rhs_é_fonte_química_menos_perda_radial():
    c = Coluna0D(CASO, pressao_pa=230.0, t_gas=290.0, raio_cm=0.3)
    rng = np.random.default_rng(1)
    y = 10.0 ** rng.uniform(6, 12, len(c.especies))
    e_n = 110.0
    k, te = c.coeficientes(e_n)
    fontes = c.q.fontes(c.densidades(y)[:, None], k[:, None])[:, 0][c.indices]
    nu = c.frequencias_de_perda(e_n, te)
    dy = c.rhs(0.0, y, e_n, ne=None)
    outras = np.arange(len(y)) != c.i_e          # the electrons: test_eletrons_perdidos_junto_com_os_íons
    np.testing.assert_allclose(dy[outras], (fontes - nu * y)[outras], rtol=1e-12, atol=1e-30)
    # with the electron density imposed, it does not change
    assert c.rhs(0.0, y, e_n, ne=y[c.i_e])[c.i_e] == 0.0


@precisa_caso
def test_perdas_por_classe_de_espécie():
    c = Coluna0D(CASO, pressao_pa=230.0, t_gas=290.0, raio_cm=0.3, gamas={"N": 3.2e-4})
    e_n = 110.0
    _, te = c.coeficientes(e_n)
    nu = c.frequencias_de_perda(e_n, te)
    i = c.especies.index
    te_ev = te * KB / QE
    mu = c.mobilidade(c.especies.index("N2^+"), e_n)
    assert nu[i("N2^+")] == pytest.approx(perda_ambipolar(mu, te_ev, 0.3), rel=1e-12)
    vbar = np.sqrt(8 * KB * 290.0 / (np.pi * 14.007 * 1.66053906660e-27)) * 100
    assert nu[i("N")] == pytest.approx(perda_neutro(c.difusao(i("N")), 3.2e-4, 0.3, vbar), rel=1e-12)
    # the electrons go with the ions (quasi-neutrality): their loss is set in rhs, not here
    assert nu[i("E")] == 0.0


@precisa_caso
def test_eletrons_perdidos_junto_com_os_íons():
    c = Coluna0D(CASO, pressao_pa=230.0, t_gas=290.0, raio_cm=0.3)
    y = np.zeros(len(c.especies))
    y[c.i_e] = 1e10
    y[c.especies.index("N2^+")] = 6e9
    y[c.especies.index("N4^+")] = 4e9
    e_n = 110.0
    k, te = c.coeficientes(e_n)
    nu = c.frequencias_de_perda(e_n, te)
    perda_ions = sum(nu[c.especies.index(s)] * y[c.especies.index(s)] for s in ("N2^+", "N4^+"))
    fonte_e = c.q.fontes(c.densidades(y)[:, None], k[:, None])[:, 0][c.indices][c.i_e]
    assert c.rhs(0.0, y, e_n, ne=None)[c.i_e] == pytest.approx(fonte_e - perda_ions, rel=1e-12)


@precisa_caso
def test_integração_curta_dá_densidades_positivas_e_n2a_crescente():
    c = Coluna0D(CASO, pressao_pa=230.0, t_gas=290.0, raio_cm=0.3, gamas={"N": 3.2e-4})
    t, y = c.integra(e_n=118.0, ne=1.31e10, t_fim=1e-5, n_saidas=5)
    assert np.all(y > -1.0)
    a = y[:, c.especies.index("N2(A3)")]
    assert np.all(np.diff(a) > 0) and a[-1] > 1e9
    assert np.all(y[:, c.i_e] == 1.31e10)


@precisa_caso
def test_difusão_do_n_em_cm2_por_s():
    # transporte.json of the case: D0 = 3.3e-5 m2/s at 1 atm and 300 K, exponent 1.75
    c = Coluna0D(CASO, pressao_pa=230.0, t_gas=290.0, raio_cm=0.3)
    esperado = 3.3e-5 * 1e4 / (230.0 / 101325.0) * (290.0 / 300.0) ** 1.75
    assert c.difusao(c.especies.index("N")) == pytest.approx(esperado, rel=1e-12)


@precisa_arquivo("casos/xu2025/quimica.json")
def test_fundo_misto_entra_com_as_frações():
    c = Coluna0D(RAIZ / "casos" / "xu2025", pressao_pa=1333.2, t_gas=295.0, raio_cm=0.3)
    d = c.densidades(np.zeros(len(c.especies)))
    nomes = [s.upper() for s in c.q.especies]
    assert d[nomes.index("N2")] == pytest.approx(0.25 * c.n_gas, rel=1e-12)
    assert d[nomes.index("H2")] == pytest.approx(0.75 * c.n_gas, rel=1e-12)


@precisa_caso
def test_fundo_perde_o_que_está_nas_outras_espécies():
    # conservation of N atoms: the background N2 is the gas minus the N2 held in
    # the other species (N2(v), excited states, ions; an N atom is half an N2)
    c = Coluna0D(CASO, pressao_pa=230.0, t_gas=290.0, raio_cm=0.3)
    y = np.zeros(len(c.especies))
    i = c.especies.index
    y[i("N2(V1)")] = 1e15
    y[i("N")] = 2e14
    y[i("N4^+")] = 1e10
    y[c.i_e] = 1e10
    n2 = c.densidades(y)[[s.upper() for s in c.q.especies].index("N2")]
    assert n2 == pytest.approx(c.n_gas - 1e15 - 0.5 * 2e14 - 2 * 1e10, rel=1e-12)


@precisa_caso
def test_ne_da_corrente():
    # j = e ne mu_e E over the section pi R^2
    c = Coluna0D(CASO, pressao_pa=230.0, t_gas=290.0, raio_cm=0.3)
    e_n, i_a = 110.0, 0.01
    mu = c.tab(np.array([e_n]), np.array([290.0]))[0, 1] / c.n_gas
    campo = e_n * 1e-17 * c.n_gas
    esperado = i_a / (QE * mu * campo * np.pi * 0.3 ** 2)
    assert c.ne_da_corrente(i_a, e_n) == pytest.approx(esperado, rel=1e-12)


@precisa_caso
def test_fator_multiplica_só_a_reação_escolhida():
    base = Coluna0D(CASO, pressao_pa=230.0, t_gas=290.0, raio_cm=0.3)
    r = [i for i, txt in enumerate(base.q.reacoes) if "=>" in txt and "N4^+" in txt.split("=>")[1]
         and "N2(A'1)" in txt.upper()]
    assert r
    c = Coluna0D(CASO, pressao_pa=230.0, t_gas=290.0, raio_cm=0.3, fatores={r[0]: 3.0})
    k0, _ = base.coeficientes(110.0)
    k1, _ = c.coeficientes(110.0)
    assert k1[r[0]] == pytest.approx(3.0 * k0[r[0]], rel=1e-14)
    outras = np.arange(len(k0)) != r[0]
    np.testing.assert_array_equal(k1[outras], k0[outras])


@precisa_caso
def test_en_sustentado_zera_o_balanço_dos_elétrons():
    c = Coluna0D(CASO, pressao_pa=230.0, t_gas=290.0, raio_cm=0.3, gamas={"N": 3.2e-4})
    e_n = c.en_sustentado(corrente_a=0.01, t_res=4.3e-4)
    producao, perda = c.balanco_eletrons(e_n, 0.01, 4.3e-4)
    assert 25.0 < e_n < 250.0 and producao > 0
    assert abs(producao - perda) < 1e-4 * perda
    # the loss is n_e times the ambipolar frequency of the ions weighted by their densities
    ne = c.ne_da_corrente(0.01, e_n)
    _, y = c.integra(e_n, ne, 4.3e-4, n_saidas=2)
    _, te = c.coeficientes(e_n)
    nu = c.frequencias_de_perda(e_n, te)
    ions = c.carga > 0
    assert perda == pytest.approx(ne * np.sum(nu[ions] * y[-1][ions]) / np.sum(y[-1][ions]), rel=1e-10)


# ---------------------------------------------------------------- limits of the wall loss

def test_perda_neutro_limite_de_superfície_para_gama_pequeno():
    # gamma << 1: the wall reaction is the bottleneck, nu -> gamma vbar / (R (2 - gamma))
    d, r, vbar = 150.0, 0.3, 4.7e4
    for gama in (1e-8, 1e-7):
        assert perda_neutro(d, gama, r, vbar) == pytest.approx(gama * vbar / (r * (2 - gama)), rel=1e-6)


def test_perda_neutro_cresce_com_gama_e_satura_na_difusão():
    d, r, vbar = 150.0, 0.3, 4.7e4
    gamas = np.geomspace(1e-7, 1.0, 40)
    nu = np.array([perda_neutro(d, g, r, vbar) for g in gamas])
    assert np.all(np.diff(nu) > 0)
    assert np.all(nu < d / (r / J0) ** 2)                  # never faster than diffusion alone
    # gamma = 1 exactly: the surface term is R / vbar
    assert nu[-1] == pytest.approx(1.0 / ((r / J0) ** 2 / d + r / vbar), rel=1e-12)


@pytest.mark.parametrize("gama", [1.0000001, 1.5, 2.0, 3.0])
def test_perda_neutro_recusa_probabilidade_acima_de_um(gama):
    with pytest.raises(ValueError, match=f"^wall loss probability {gama:g} above 1$"):
        perda_neutro(150.0, gama, 0.3, 4.7e4)


def test_perda_neutro_negativa_ou_nula_não_perde():
    assert perda_neutro(150.0, -1e-3, 0.3, 4.7e4) == 0.0


# ---------------------------------------------------------------- errors with a clear message

@precisa_caso
def test_gama_de_espécie_que_não_existe_é_recusado():
    with pytest.raises(ValueError, match="^wall probability for species not in the case: N2\\(XYZ\\), N9$"):
        Coluna0D(CASO, pressao_pa=230.0, t_gas=290.0, raio_cm=0.3, gamas={"N2(XYZ)": 1e-3, "N9": 1e-3, "N": 1e-3})
    # with estrito=False the names that are not in the case are ignored
    c = Coluna0D(CASO, pressao_pa=230.0, t_gas=290.0, raio_cm=0.3, gamas={"N2(XYZ)": 1e-3, "N": 2e-3}, estrito=False)
    assert c.gamas["N"] == 2e-3


@precisa_caso
def test_gama_de_íon_é_recusado():
    # the ions are lost by ambipolar diffusion; a wall probability for them would be ignored
    with pytest.raises(ValueError, match="^wall probability given for ions \\(N2\\^\\+, N4\\^\\+\\): they are lost by "
                                         "ambipolar diffusion$"):
        Coluna0D(CASO, pressao_pa=230.0, t_gas=290.0, raio_cm=0.3, gamas={"N4^+": 1e-3, "N2^+": 1e-3}, estrito=False)


@precisa_caso
def test_faixa_de_en_sem_raiz_dá_mensagem_clara():
    c = Coluna0D(CASO, pressao_pa=230.0, t_gas=290.0, raio_cm=0.3)
    with pytest.raises(ValueError, match="^no sustaining E/N between 20 and 30 Td: the electron balance has the same "
                                         "sign at both ends$"):
        c.en_sustentado(corrente_a=0.01, t_res=4.3e-4, faixa=(20.0, 30.0))


# ---------------------------------------------------------------- the two modes of the electrons

@precisa_caso
def test_elétrons_livres_evoluem_e_seguem_os_íons():
    # ne=None: the electrons are a variable of the state, lost with the ions
    c = Coluna0D(CASO, pressao_pa=230.0, t_gas=290.0, raio_cm=0.3)
    y0 = c.estado_inicial(1e10)
    t, y = c.integra(e_n=118.0, ne=None, t_fim=1e-6, n_saidas=3, y0=y0)
    assert np.all(np.isfinite(y)) and np.all(y > -1.0)
    assert y[-1, c.i_e] != 1e10
    ions = c.carga > 0
    # quasi-neutrality is not imposed but kept by the shared loss: within 1% at the end
    assert y[-1, c.i_e] == pytest.approx(np.sum(y[-1][ions] * c.carga[ions]), rel=1e-2)


@precisa_caso
def test_elétrons_impostos_não_mudam():
    c = Coluna0D(CASO, pressao_pa=230.0, t_gas=290.0, raio_cm=0.3)
    _, y = c.integra(e_n=118.0, ne=2e10, t_fim=1e-6, n_saidas=3, y0=c.estado_inicial(5e9))
    np.testing.assert_array_equal(y[:, c.i_e], 2e10)


# ---------------------------------------------------------------- physical properties

@precisa_caso
def test_química_conserva_átomos_de_n():
    # the source of N atoms summed over all the species (the background included) is zero
    from coluna import atomos
    c = Coluna0D(CASO, pressao_pa=230.0, t_gas=290.0, raio_cm=0.3)
    rng = np.random.default_rng(7)
    n_atomos = np.array([atomos(s).get("N", 0) for s in c.q.especies])
    for e_n in (60.0, 110.0, 200.0):
        y = 10.0 ** rng.uniform(6, 13, len(c.especies))
        k, _ = c.coeficientes(e_n)
        f = c.q.fontes(c.densidades(y)[:, None], k[:, None])[:, 0]
        escala = np.sum(np.abs(f) * n_atomos)
        assert abs(np.sum(f * n_atomos)) <= 1e-12 * escala


@precisa_caso
def test_en_sustentado_cai_com_a_corrente():
    # more current, more electrons and more stepwise and associative ionization: a lower field
    c = Coluna0D(CASO, pressao_pa=230.0, t_gas=290.0, raio_cm=0.3, gamas=GAMAS)
    assert c.en_sustentado(0.02, 4.3e-4) < c.en_sustentado(0.01, 4.3e-4)


# ---------------------------------------------------------------- regression

@precisa_caso
def test_regressão_coluna_de_levaton_10_ma():
    # condition L20_10mA, chemistry A: docs/coluna_positiva_n2/etapa5/en.csv (109.775 Td) and the
    # vibrational population and dissociation at the measured E/N (118 Td) with n_e = 1.31e10 cm-3
    c = Coluna0D(CASO, pressao_pa=230.0, t_gas=290.0, raio_cm=0.3, gamas=GAMAS)
    assert c.en_sustentado(0.01, 4.3e-4) == pytest.approx(109.77532698340232, rel=1e-6)
    _, y = c.integra(118.0, 1.31e10, 4.3e-4, n_saidas=3)
    nomes = [s.upper() for s in c.especies]
    n0 = c.densidades(y[-1])[[s.upper() for s in c.q.especies].index("N2")]
    assert y[-1, nomes.index("N2(V1)")] / n0 == pytest.approx(0.09148908352544494, rel=1e-5)
    assert y[-1, nomes.index("N")] / n0 == pytest.approx(0.0002810043611996352, rel=1e-5)
    assert y[-1, nomes.index("N2(A3)")] == pytest.approx(20542153965444.375, rel=1e-5)


@precisa_arquivo("scripts/coluna_n2_lote.py")
@pytest.mark.parametrize("variante", ["N_x5", "vib_y", "vib", "parede", "meta"])
def test_variante_de_parede_inexistente_é_recusada(variante):
    import sys
    sys.path.insert(0, str(RAIZ / "scripts"))
    from coluna_n2_lote import gamas
    with pytest.raises(ValueError, match="unknown wall variant"):
        gamas(variante)


@precisa_arquivo("scripts/coluna_n2_lote.py")
@pytest.mark.parametrize("variante", ["meta_0", "meta_1.5", "meta_x"])
def test_variante_de_metaestáveis_fora_de_0_a_1_é_recusada(variante):
    import sys
    sys.path.insert(0, str(RAIZ / "scripts"))
    from coluna_n2_lote import gamas
    with pytest.raises(ValueError, match="unknown wall variant"):
        gamas(variante)
    assert gamas("meta_0.01")["N2(A3)"] == 0.01 and gamas("meta_1")["N(2D)"] == 1.0


@precisa_caso
def test_falha_do_integrador_é_relatada(monkeypatch):
    import coluna
    c = Coluna0D(CASO, pressao_pa=230.0, t_gas=290.0, raio_cm=0.3)

    class Falha:
        success, message = False, "step size too small"
    monkeypatch.setattr(coluna, "solve_ivp", lambda *a, **k: Falha())
    with pytest.raises(RuntimeError, match="step size too small"):
        c.integra(118.0, 1e10, 1e-6)


@precisa_caso
def test_erro_que_não_é_de_sinal_passa_intacto(monkeypatch):
    c = Coluna0D(CASO, pressao_pa=230.0, t_gas=290.0, raio_cm=0.3)

    def quebra(*a):
        raise ValueError("E/N outside the table")
    monkeypatch.setattr(c, "balanco_eletrons", quebra)
    with pytest.raises(ValueError, match="E/N outside the table"):
        c.en_sustentado(0.01, 4.3e-4)



def test_átomos_de_um_nome():
    # the state in parentheses and the charge are not part of the formula
    assert atomos("N2(A3)") == {"N": 2} and atomos("N2(a'1)") == {"N": 2}
    assert atomos("N4^+") == {"N": 4} and atomos("N^+") == {"N": 1}
    assert atomos("NH3") == {"N": 1, "H": 3} and atomos("N2H^+") == {"N": 2, "H": 1}
    assert atomos("E") == {} and atomos("e") == {}



@precisa_caso
def test_instantes_de_saída_da_integração():
    # n_saidas instants, log spaced from t_fim * 1e-6 to t_fim; 50 by default
    c = Coluna0D(CASO, pressao_pa=230.0, t_gas=290.0, raio_cm=0.3)
    t, y = c.integra(118.0, 1e10, 1e-6)
    assert y.shape == (50, len(c.especies))
    np.testing.assert_allclose(t, np.geomspace(1e-12, 1e-6, 50), rtol=1e-12)
    t, y = c.integra(118.0, 1e10, 1e-6, n_saidas=4)
    np.testing.assert_allclose(t, [1e-12, 1e-10, 1e-8, 1e-6], rtol=1e-12)
