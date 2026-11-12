# SPDX-License-Identifier: GPL-3.0-or-later
"""NH3 at the reactor outlet in the conditions of Sun et al. 2022 (scripts/residencia_sun.py)."""
import numpy as np
import pytest

from conftest import RAIZ, precisa_arquivo

precisa_script = precisa_arquivo("scripts/residencia_sun.py")


def modulo():
    import sys
    sys.path.insert(0, str(RAIZ / "scripts"))
    import residencia_sun
    return residencia_sun


@precisa_script
def test_pulsos_na_residência_do_caso_base():
    # 200 sccm at 20 kPa and 573 K: 35.4 cm3/s through 3.14 cm3, 0.0887 s, 2661 pulses at 30 kHz
    m = modulo()
    assert m.pulsos_na_residencia(573.0, 2e4, 3e4) == pytest.approx(2661, rel=2e-3)


@precisa_script
def test_residência_cai_com_a_temperatura_e_sobe_com_a_pressão():
    m = modulo()
    assert m.pulsos_na_residencia(773.0, 2e4, 3e4) == pytest.approx(m.pulsos_na_residencia(373.0, 2e4, 3e4) * 373 / 773)
    assert m.pulsos_na_residencia(573.0, 4e4, 3e4) == pytest.approx(2 * m.pulsos_na_residencia(573.0, 2e4, 3e4))


@precisa_script
def test_taxa_na_saída_igual_à_por_pulso_quando_o_acúmulo_é_linear():
    m = modulo()
    # NH3 growing by 1e11 cm-2 per pulse: the outlet over N pulses gives the rate of one pulse
    assert m.taxa_saida(1e11 * 2661, 2661, 3e4) == pytest.approx(m.taxa(1e11, 3e4))


@precisa_script
def test_extrapolação_por_potência_reproduz_a_lei():
    m = modulo()
    n = np.arange(1, 301)
    p = 1e11 * n ** -0.6
    a = np.cumsum(p)
    k = np.arange(1, 2001)
    assert m.extrapola(n, a, 2000, "potencia") == pytest.approx(np.sum(1e11 * k ** -0.6), rel=1e-6)


@precisa_script
def test_extrapolações_em_ordem():
    m = modulo()
    n = np.arange(1, 301)
    a = np.cumsum(1e11 * np.exp(-n / 200.0))
    parado, pot, lin = (m.extrapola(n, a, 2000, x) for x in ("parado", "potencia", "linear"))
    assert parado == a[-1]
    assert parado < pot < lin
    assert lin == pytest.approx(a[-1] + (a[-1] - a[-2]) * 1700)


@precisa_script
def test_extrapolação_não_encurta():
    m = modulo()
    n = np.arange(1, 301)
    with pytest.raises(ValueError):
        m.extrapola(n, np.cumsum(np.ones(300)), 200, "linear")
