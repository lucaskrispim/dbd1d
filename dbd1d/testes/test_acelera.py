# SPDX-License-Identifier: GPL-3.0-or-later
"""Cycle acceleration (acelera.py): projection of the slow variables over many
cycles from the changes between the ends of the last direct cycles."""
from types import SimpleNamespace

import numpy as np
import pytest

from acelera import indices_lentos, projeta

LENTOS = np.arange(6)


def serie(y_eq, amp, r, k):
    """End of cycle k of a relaxation y_eq + amp r^k, element by element."""
    return y_eq + amp * r ** k


def test_relaxacao_exponencial_e_projetada_exatamente():
    y_eq = np.full(6, 100.0)
    amp = np.array([5.0, -5.0, 3.0, -1.0, 2.0, 4.0])
    r = np.array([0.999, 0.99, 0.9, 0.5, 0.1, 0.95])
    y0, y1, y2 = (serie(y_eq, amp, r, k) for k in (0, 1, 2))
    novo, limitados = projeta(y0, y1, y2, LENTOS, 40, limite=0.5)
    np.testing.assert_allclose(novo, serie(y_eq, amp, r, 42), rtol=1e-12)
    assert limitados == 0


def test_deriva_constante_e_projetada_linearmente():
    y0 = np.full(6, 100.0)
    y1, y2 = y0 + 0.1, y0 + 0.2
    novo, _ = projeta(y0, y1, y2, LENTOS, 30, limite=0.5)
    np.testing.assert_allclose(novo, y2 + 30 * 0.1, rtol=1e-12)


def test_variaveis_rapidas_ficam_como_estao():
    y0 = np.arange(10.0) + 50.0
    y1, y2 = y0 + 1.0, y0 + 1.5
    novo, _ = projeta(y0, y1, y2, np.array([1, 4]), 10, limite=0.5)
    rapidas = np.setdiff1d(np.arange(10), [1, 4])
    np.testing.assert_array_equal(novo[rapidas], y2[rapidas])
    assert np.all(novo[[1, 4]] != y2[[1, 4]])


def test_oscilacao_nao_e_projetada():
    # alternating sign between cycles: a fast mode or noise, not a drift
    y0 = np.full(6, 100.0)
    y1, y2 = y0 + 1.0, y0 + 0.5
    novo, _ = projeta(y0, y1, y2, LENTOS, 50, limite=0.5)
    np.testing.assert_array_equal(novo, y2)


def test_mudanca_limitada_a_uma_fracao_do_valor():
    y0 = np.array([100.0, 100.0, 1.0, 1.0, 100.0, 100.0])
    y1 = y0 - np.array([1.0, 0.0, 0.1, 0.0, 0.0, 0.0])
    y2 = y1 - np.array([1.0, 0.0, 0.1, 0.0, 0.0, 0.0])
    novo, limitados = projeta(y0, y1, y2, LENTOS, 100, limite=0.2)
    # 100 cycles of -0.1 would take the third to -9: it stops at 20% below
    np.testing.assert_allclose(novo[2], 0.8 * y2[2])
    np.testing.assert_allclose(novo[0], 0.8 * y2[0])
    assert limitados == 2 and np.all(novo > 0)


def test_variavel_sem_mudanca_fica_igual():
    y = np.full(6, 7.0)
    novo, limitados = projeta(y, y, y, LENTOS, 100, limite=0.2)
    np.testing.assert_array_equal(novo, y)
    assert limitados == 0


def _modelo(aquecimento):
    # species E, N2+, N, NH3, H-, energy row, temperature row; 3 cells
    md = SimpleNamespace(carga=np.array([-1, 1, 0, 0, -1]), i_e=0, n_esp=5, m=3, energia=True)
    md.n_var = 6 + (1 if aquecimento else 0)
    md.aquecimento = aquecimento
    if aquecimento:
        md.i_t = 6
    return md


def test_indices_lentos_sao_os_neutros_e_a_temperatura():
    np.testing.assert_array_equal(indices_lentos(_modelo(False)), [6, 7, 8, 9, 10, 11])
    np.testing.assert_array_equal(indices_lentos(_modelo(True)), [6, 7, 8, 9, 10, 11, 18, 19, 20])


@pytest.mark.parametrize("ciclos, limite", [(0, 0.2), (10, 0.0), (10, 1.0)])
def test_parametros_invalidos_sao_recusados(ciclos, limite):
    y = np.ones(6)
    with pytest.raises(ValueError, match="^the projection needs at least one cycle and a limit in \\(0, 1\\)$"):
        projeta(y, y, y, LENTOS, ciclos, limite=limite)


# ---------------------------------------------------------------- edges and limits

def test_um_ciclo_projetado_é_o_próximo_ciclo_exato():
    y_eq, amp, r = np.full(6, 50.0), np.full(6, 4.0), np.array([0.2, 0.4, 0.6, 0.8, 0.9, 0.99])
    y0, y1, y2 = (serie(y_eq, amp, r, k) for k in (0, 1, 2))
    novo, _ = projeta(y0, y1, y2, LENTOS, 1, limite=0.5)
    np.testing.assert_allclose(novo, serie(y_eq, amp, r, 3), rtol=1e-13)


def test_razão_quase_um_é_contínua_com_a_deriva_linear():
    # |1 - r| below 1e-12 takes the linear sum; just above it the geometric one: both give M d1
    y0 = np.full(6, 100.0)
    d = 1e-3
    for eps in (1e-13, 1e-11):
        y1, y2 = y0 + d, y0 + d + d * (1.0 - eps)
        novo, _ = projeta(y0, y1, y2, LENTOS, 20, limite=0.5)
        np.testing.assert_allclose(novo, y2 + 20 * d, rtol=1e-9)


def test_variável_parada_que_começa_a_mudar_não_é_projetada():
    # d0 = 0 and d1 != 0: no estimate of the relaxation yet, the variable waits for the next jump
    y0 = np.full(6, 10.0)
    y1, y2 = y0.copy(), y0 + 1.0
    novo, _ = projeta(y0, y1, y2, LENTOS, 30, limite=0.5)
    np.testing.assert_array_equal(novo, y2)


def test_mudança_exatamente_no_teto_não_conta_como_limitada():
    y0 = np.full(6, 100.0)
    y1, y2 = y0 + 1.0, y0 + 2.0
    # 10 cycles of +1 from 102 is +10, the ceiling at limite = 10/102
    novo, limitados = projeta(y0, y1, y2, LENTOS, 10, limite=10.0 / 102.0)
    np.testing.assert_allclose(novo, 112.0, rtol=1e-14)
    assert limitados == 0


def test_valor_negativo_usa_o_módulo_no_teto():
    y0 = np.full(6, -100.0)
    y1, y2 = y0 - 1.0, y0 - 2.0
    novo, limitados = projeta(y0, y1, y2, LENTOS, 1000, limite=0.1)
    np.testing.assert_allclose(novo, 1.1 * y2, rtol=1e-14)
    assert limitados == 6


def test_aceleração_desligada_por_padrão_e_marcada_experimental():
    from simula import cria_parser
    p = cria_parser()
    a = p.parse_args(["destino"])
    assert a.acelera is None
    ajuda = next(x.help for x in p._actions if "--accelerate" in x.option_strings)
    assert "experimental" in ajuda.lower() and "off by default" in ajuda.lower()


def test_limite_padrão_é_metade_do_valor():
    import acelera
    assert acelera.LIMITE == 0.5
    y0 = np.full(6, 100.0)
    y1, y2 = y0 + 10.0, y0 + 20.0
    novo, limitados = projeta(y0, y1, y2, LENTOS, 1000)
    np.testing.assert_allclose(novo, 1.5 * y2, rtol=1e-14)
    assert limitados == 6


def test_deriva_que_acelera_é_projetada_linearmente():
    # r >= 1 (the change grows from cycle to cycle): no geometric sum, M d1
    y0 = np.full(6, 1000.0)
    y1, y2 = y0 + 1.0, y0 + 1.0 + 1.5
    novo, _ = projeta(y0, y1, y2, LENTOS, 10, limite=0.5)
    np.testing.assert_allclose(novo, y2 + 10 * 1.5, rtol=1e-14)
