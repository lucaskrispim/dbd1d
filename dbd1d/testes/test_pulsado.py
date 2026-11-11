# SPDX-License-Identifier: GPL-3.0-or-later
"""Pulsed cycle (pulsado.py): pulse at imposed E/N and n_e, afterglow with n_e
free, repeated up to the periodic regime. A linear model with an analytic
solution stands in for Coluna0D, with the same interface."""
import numpy as np
import pytest

from conftest import RAIZ, precisa_arquivo
from pulsado import Pulso, periodo, regime_periodico


class Linear:
    """dX/dt = s - k X while n_e is imposed and -k X after; the single ion N2^+
    decays at r_pulso in the pulse, and with the electrons at r in the afterglow."""
    especies = ["X", "E", "N2^+"]
    carga = np.array([0, -1, 1])
    i_e = 1
    n_gas = 1e12

    def __init__(self, s=1e15, k=1e3, r=1e6, r_pulso=0.0):
        self.s, self.k, self.r, self.r_pulso = s, k, r, r_pulso

    def integra(self, e_n, ne, t_fim, n_saidas=50, y0=None, rtol=1e-6, atol=1.0, saidas=None, max_step=np.inf):
        t = np.asarray(saidas, dtype=float)
        y = np.zeros((len(t), 3))
        if ne is not None:
            alvo = self.s / self.k
            y[:, 0] = alvo + (y0[0] - alvo) * np.exp(-self.k * t)
            y[:, 1] = ne(t) if callable(ne) else ne
            y[:, 2] = y0[2] * np.exp(-self.r_pulso * t)
        else:
            y[:, 0] = y0[0] * np.exp(-self.k * t)
            y[:, 1] = y0[1] * np.exp(-self.r * t)
            y[:, 2] = y0[2] * np.exp(-self.r * t)
        return t, y


PULSO = Pulso(e_n=81.0, ne=2e13, duracao=2.2e-7, periodo=1e-3, e_n_pos=1e-4)
SAIDAS = np.geomspace(3e-7, 1e-3, 40)


def test_regime_periodico_linear_bate_com_analitico():
    m = Linear()
    t, y, _ = regime_periodico(m, PULSO, np.zeros(3), SAIDAS, tol=1e-6)
    tau, per, k, s = PULSO.duracao, PULSO.periodo, m.k, m.s
    analitico = s / k * (1 - np.exp(-k * tau)) * np.exp(-k * (per - tau)) / (1 - np.exp(-k * per))
    assert t[0] == 0.0 and t[-1] == pytest.approx(per)
    assert y[0, 0] == pytest.approx(analitico, rel=2e-3)
    assert y[-1, 0] == pytest.approx(analitico, rel=2e-3)


def test_semente_de_ions_no_inicio_do_pulso():
    m = Linear()
    y0 = np.array([0.0, 0.0, 5e12])
    t, y = periodo(m, PULSO, y0, SAIDAS)
    assert y[0, 2] == pytest.approx(PULSO.ne)        # the ion completes the imposed charge
    assert y[0, 1] == pytest.approx(PULSO.ne)


def test_semente_nao_retira_ions_em_excesso():
    m = Linear()
    y0 = np.array([0.0, 0.0, 5e13])
    _, y = periodo(m, PULSO, y0, SAIDAS)
    assert y[0, 2] == pytest.approx(5e13)


def test_neutralidade_no_inicio_da_pos_descarga():
    m = Linear()
    t, y = periodo(m, PULSO, np.array([0.0, 0.0, 3e13]), SAIDAS)
    fim = np.searchsorted(t, PULSO.duracao, side="right") - 1
    assert t[fim] == pytest.approx(PULSO.duracao)
    # first afterglow output: electrons and ions decayed from the same value (the ion charge)
    k = fim + 1
    assert y[k, 1] == pytest.approx(y[k, 2], rel=1e-12)
    assert y[k, 2] == pytest.approx(3e13 * np.exp(-m.r * (t[k] - PULSO.duracao)), rel=1e-12)


def test_historico_decresce_e_para_abaixo_de_tol():
    m = Linear(k=1e3)
    _, _, hist = regime_periodico(m, PULSO, np.zeros(3), SAIDAS, tol=1e-3)
    assert hist[-1] < 1e-3 and hist[-2] < 1e-3
    assert all(b <= a for a, b in zip(hist[2:], hist[3:]))


def test_sem_convergencia_e_erro():
    with pytest.raises(RuntimeError, match="periodic"):
        regime_periodico(Linear(k=10.0), PULSO, np.zeros(3), SAIDAS, tol=1e-3, max_periodos=2)


def test_saidas_pos_fora_do_intervalo_e_erro():
    with pytest.raises(ValueError, match="saidas_pos"):
        periodo(Linear(), PULSO, np.zeros(3), np.array([1e-7, 1e-3]))
    with pytest.raises(ValueError, match="saidas_pos"):
        periodo(Linear(), PULSO, np.zeros(3), np.array([1e-6, 5e-4]))


def test_pulso_mais_longo_que_o_periodo_e_erro():
    with pytest.raises(ValueError, match="duracao"):
        Pulso(e_n=81.0, ne=2e13, duracao=2e-3, periodo=1e-3, e_n_pos=1e-4)


def test_nenhuma_especie_acima_do_limiar_e_erro():
    m = Linear()
    m.n_gas = 1e30
    with pytest.raises(RuntimeError, match="threshold"):
        regime_periodico(m, PULSO, np.zeros(3), SAIDAS)


@precisa_arquivo("casos/n2_amonia/quimica.json")
def test_um_periodo_do_jato_com_coluna0d():
    from coluna import Coluna0D
    c = Coluna0D(RAIZ / "casos" / "n2_amonia", 2e4, 330.0, 0.0541, troca_fluxo=83.3, quantum_na_parede=False)
    pulso = Pulso(e_n=81.0, ne=2e13, duracao=2.2e-7, periodo=1e-3, e_n_pos=float(c.tab.e_n[0]))
    t, y = periodo(c, pulso, c.estado_inicial(0.0), SAIDAS)
    v1 = c.especies.index("N2(V1)")
    fim = np.searchsorted(t, pulso.duracao, side="right") - 1
    assert np.all(np.isfinite(y)) and np.all(y > -1e-3 * c.n_gas * 1e-12)
    assert y[fim, v1] > 1e-4 * c.n_gas                      # the pulse excites vibration
    assert y[-1, c.i_e] < 1e-2 * pulso.ne                    # the electrons recombine in the afterglow


def test_ions_no_fim_completa_a_carga_no_ion_dominante():
    # sensitivity: the afterglow starts with n_e equal to the imposed value, the missing
    # charge given to the most abundant ion
    m = Linear(r_pulso=1e7)       # the ions fall to 2e13 e^-2.2 during the pulse
    p = Pulso(e_n=81.0, ne=2e13, duracao=2.2e-7, periodo=1e-3, e_n_pos=1e-4, ions_no_fim=True)
    t, y = periodo(m, p, np.array([0.0, 0.0, 5e12]), SAIDAS)
    k = np.searchsorted(t, p.duracao, side="right")
    assert y[k, 1] == pytest.approx(2e13 * np.exp(-m.r * (t[k] - p.duracao)), rel=1e-12)
    assert y[k, 2] == pytest.approx(y[k, 1], rel=1e-12)


def test_sem_ions_no_fim_a_pos_descarga_comeca_com_os_ions_do_pulso():
    m = Linear(r_pulso=1e7)
    t, y = periodo(m, PULSO, np.zeros(3), SAIDAS)
    k = np.searchsorted(t, PULSO.duracao, side="right")
    assert y[k, 1] == pytest.approx(2e13 * np.exp(-1e7 * PULSO.duracao - m.r * (t[k] - PULSO.duracao)), rel=1e-12)


PERFIL = ((0.0, 3.1e-8, 3.2e-8, 2.0e-7, 2.5e-7), (1e9, 1e9, 2.4e13, 1.9e13, 1e12))
PULSO_PERFIL = Pulso(e_n=81.0, ne=2.08e13, duracao=2.5e-7, periodo=1e-3, e_n_pos=1e-4, perfil_ne=PERFIL,
                     passo_saida=5e-9)


def test_pulso_com_perfil_impoe_o_ne_interpolado():
    m = Linear()
    t, y = periodo(m, PULSO_PERFIL, np.zeros(3), SAIDAS)
    no_pulso = (t > 0) & (t <= PULSO_PERFIL.duracao)
    np.testing.assert_allclose(y[no_pulso, 1], np.interp(t[no_pulso], *PERFIL))


def test_pulso_com_perfil_limita_o_passo_do_integrador():
    registro = []

    class Registra(Linear):
        def integra(self, *a, **k):
            registro.append(k.get("max_step", np.inf))
            return super().integra(*a, **k)

    periodo(Registra(), PULSO_PERFIL, np.zeros(3), SAIDAS)
    assert registro[0] <= 2e-9 and registro[1] == np.inf


def test_passo_de_saida_no_pulso():
    t, _ = periodo(Linear(), PULSO_PERFIL, np.zeros(3), SAIDAS)
    no_pulso = t[(t > 0) & (t <= PULSO_PERFIL.duracao)]
    assert np.all(np.isin(np.round(np.arange(5e-9, 2.5e-7, 5e-9) * 1e12), np.round(no_pulso * 1e12)))
    assert no_pulso[-1] == PULSO_PERFIL.duracao and np.all(np.diff(no_pulso) > 0)


def test_perfil_que_nao_cobre_o_pulso_e_erro():
    with pytest.raises(ValueError):
        Pulso(e_n=81.0, ne=2e13, duracao=3e-7, periodo=1e-3, e_n_pos=1e-4, perfil_ne=PERFIL)


def test_pulso_sem_perfil_nao_muda_as_saidas():
    t, _ = periodo(Linear(), PULSO, np.zeros(3), SAIDAS)
    assert np.sum((t > 0) & (t <= PULSO.duracao)) == 20


def _cauda_que_cruza_o_limiar():
    """Start and end of six periods of a state with a fast species and a slow one that
    climbs through LIMIAR * n_gas (the N2(v = 14) of the two G2 failures)."""
    m = Linear()
    piso = 1e-8 * m.n_gas
    rapida = 0.5 * m.n_gas * (1 - 0.1 ** np.arange(7))
    lenta = piso * 2.0 * (1 - 0.8 ** np.arange(1, 8))
    estados = [np.array([rapida[k], 0.0, lenta[k]]) for k in range(7)]
    return m, estados


def test_mudanca_antiga_salta_quando_uma_especie_cruza_o_limiar():
    from pulsado import mudanca
    m, e = _cauda_que_cruza_o_limiar()
    h = [mudanca(m, a, b) for a, b in zip(e[:-1], e[1:])]
    assert any(b > a for a, b in zip(h[1:], h[2:]))


def test_mudanca_com_piso_nao_salta():
    from pulsado import mudanca_com_piso
    m, e = _cauda_que_cruza_o_limiar()
    h = [mudanca_com_piso(m, a, b) for a, b in zip(e[:-1], e[1:])]
    assert all(b <= a for a, b in zip(h, h[1:]))


def test_criterio_novo_limita_a_distancia_ao_ponto_fixo():
    # contraction exp(-k T) = 0.86 per period, as the gamma = 1e-4 runs
    m = Linear(k=-np.log(0.86) / PULSO.periodo)
    tau, per, k, s = PULSO.duracao, PULSO.periodo, m.k, m.s
    fixo = s / k * (1 - np.exp(-k * tau)) * np.exp(-k * (per - tau)) / (1 - np.exp(-k * per))
    _, y_ant, _ = regime_periodico(m, PULSO, np.zeros(3), SAIDAS, tol=1e-3)
    _, y_nov, _ = regime_periodico(m, PULSO, np.zeros(3), SAIDAS, tol=1e-3, criterio="novo")
    assert abs(y_ant[-1, 0] / fixo - 1) > 1e-3
    assert abs(y_nov[-1, 0] / fixo - 1) < 1e-3


def test_criterio_novo_no_linear_analitico():
    m = Linear()
    t, y, _ = regime_periodico(m, PULSO, np.zeros(3), SAIDAS, tol=1e-6, criterio="novo")
    tau, per, k, s = PULSO.duracao, PULSO.periodo, m.k, m.s
    assert y[-1, 0] == pytest.approx(s / k * (1 - np.exp(-k * tau)) * np.exp(-k * (per - tau)) / (1 - np.exp(-k * per)),
                                     rel=2e-6)


def test_criterio_desconhecido_e_erro():
    with pytest.raises(ValueError):
        regime_periodico(Linear(), PULSO, np.zeros(3), SAIDAS, criterio="outro")
