# SPDX-License-Identifier: GPL-3.0-or-later
"""Positive column (coluna.py) on a synthetic N2 case that the package distributes
(testes/dados/coluna_sintetica_zdp.F90 and a table written here), with closed-form
answers: the rate coefficients are power laws of E/N, which the log-log interpolation
of the table reproduces exactly, the mean energy and the ion mobility are constant, and
the only reactions are electron impact on the background. So at fixed E/N and n_e
- N2^+ and N relax to their sources over their loss frequencies, n(t) = (S/nu)(1 - exp(-nu t));
- N2(v2) does the same, and the quantum it loses at the wall feeds N2(v1);
- the sustaining field is the root of k_i(E/N) N = nu_a, with nu_a = mu_i kTe/(e Lambda^2)."""
import json
import warnings

import numpy as np
import pytest

from conftest import PROTOTIPO
from coluna import Coluna0D
from quimica_zdp import Quimica

KB, QE, UMA, N_LOSCHMIDT = 1.380649e-23, 1.602176634e-19, 1.66053906660e-27, 2.6867811e19
J0 = 2.404825557695773                            # first zero of J0: Lambda = R / J0
P, TG, R = 230.0, 300.0, 0.3                      # Pa, K, cm
EPS = 4.0                                         # mean energy, eV: Te = (2/3) eps
K0_ION = 1.9                                      # reduced mobility of N2+, cm2/(V s)
MU_N = 1.0e22                                     # mu_e N, 1/(cm V s)
# k = A (E/N / 100 Td)^p, cm3/s: ionization, dissociation, N2(v1), N2(v2)
LEIS = [(2.0e-12, 6.0), (5.0e-13, 4.0), (3.0e-12, -1.0), (1.0e-12, -0.5)]
GAMAS = {"N": 1e-3, "N2(V1)": 4.5e-4, "N2(V2)": 7e-3}


def coef(e_n):
    return np.array([a * (np.asarray(e_n) / 100.0) ** p for a, p in LEIS])


@pytest.fixture(scope="module")
def caso(tmp_path_factory):
    d = tmp_path_factory.mktemp("coluna_sintetica")
    Quimica(PROTOTIPO / "testes" / "dados" / "coluna_sintetica_zdp.F90").salva(d / "quimica.json")
    e_n = np.geomspace(1.0, 2000.0, 60)
    linhas = []
    for t in (280.0, 900.0):
        for x, k in zip(e_n, coef(e_n).T):
            # T_g, E/N, eps, mu N, D N, mu_eps N, D_eps N, P/N, Pel/N, Pinel/N, nu/N, k
            linhas.append([t, x, EPS, MU_N, MU_N * 2 * EPS / 3, MU_N, MU_N, 1e-18, 1e-20, 1e-18, 1e-8, *k])
    np.savetxt(d / "tabela_energia_tg.dat", linhas, header="synthetic table of test_coluna_sintetica.py")
    (d / "gas.json").write_text(json.dumps({"fundo": {"N2": 1.0}}))
    neutro = {"massa_u": 28.0, "difusao": {"D0": 2.0e-5, "T0": 300.0, "expoente": 1.75}, "perda_na_parede": 0.0}
    (d / "transporte.json").write_text(json.dumps({
        "N2(v1)": neutro, "N2(v2)": neutro, "N2": neutro,
        "N": {"massa_u": 14.0, "difusao": {"D0": 4.0e-5, "T0": 300.0, "expoente": 1.75}, "perda_na_parede": 0.0},
        "N2^+": {"massa_u": 28.0, "mobilidade": {"tipo": "constante", "K0": K0_ION}}}))
    return d


def nu_ambipolar(c):
    mu = K0_ION * N_LOSCHMIDT / c.n_gas                          # cm2/(V s)
    # Capitelli et al. 2000, eq. (6.10): D_a = mu_i (D_e/mu_e + k T_g/e); the table above has D_e/mu_e = (2/3) eps
    return mu * ((2.0 / 3.0) * EPS + KB * TG / QE) / (R / J0) ** 2


def nu_parede(c, especie):
    dados = c.transporte[especie]
    vbar = np.sqrt(8.0 * KB * TG / (np.pi * dados["massa_u"] * UMA)) * 100.0
    d = dados["difusao"]["D0"] / (P / 101325.0) * (TG / 300.0) ** 1.75 * 1e4
    # diffusion in series with the wall reaction (Chantry 1987), written here, not taken from coluna.py
    g = GAMAS[especie]
    return 1.0 / ((R / J0) ** 2 / d + R * (2.0 - g) / (g * vbar))


def test_densidades_relaxam_as_suas_fontes(caso):
    c = Coluna0D(caso, P, TG, R, gamas=GAMAS)
    e_n, ne, t = 120.0, 1.0e6, 2.0e-3
    _, y = c.integra(e_n, ne, t, n_saidas=2, rtol=1e-10, atol=1e-12)
    nomes = [s.upper() for s in c.especies]
    k = coef(e_n)
    n = c.n_gas
    nu_i, nu_n, nu_1, nu_2 = nu_ambipolar(c), nu_parede(c, "N"), nu_parede(c, "N2(V1)"), nu_parede(c, "N2(V2)")
    relaxa = lambda s, nu: s / nu * (1.0 - np.exp(-nu * t))   # noqa: E731
    assert y[-1, nomes.index("N2^+")] == pytest.approx(relaxa(k[0] * ne * n, nu_i), rel=1e-6)
    assert y[-1, nomes.index("N")] == pytest.approx(relaxa(2 * k[1] * ne * n, nu_n), rel=1e-6)
    v2 = y[-1, nomes.index("N2(V2)")]
    assert v2 == pytest.approx(relaxa(k[3] * ne * n, nu_2), rel=1e-6)
    # N2(v1): its own source plus the quantum lost by N2(v2) at the wall
    s1, s2 = k[2] * ne * n, k[3] * ne * n
    v1 = (relaxa(s1, nu_1) + nu_2 * s2 / nu_2 * ((1 - np.exp(-nu_1 * t)) / nu_1
                                                - (np.exp(-nu_2 * t) - np.exp(-nu_1 * t)) / (nu_1 - nu_2)))
    assert y[-1, nomes.index("N2(V1)")] == pytest.approx(v1, rel=1e-6)
    # the background holds what the state takes from it (an N atom is half a molecule)
    fundo = c.densidades(y[-1])[[s.upper() for s in c.q.especies].index("N2")]
    tomado = y[-1, nomes.index("N2^+")] + v1 + v2 + 0.5 * y[-1, nomes.index("N")]
    assert fundo == pytest.approx(n - tomado, rel=1e-12)


def test_campo_sustentado_e_a_raiz_analitica(caso):
    c = Coluna0D(caso, P, TG, R, gamas=GAMAS)
    a, p = LEIS[0]
    esperado = 100.0 * (nu_ambipolar(c) / (a * c.n_gas)) ** (1.0 / p)
    with warnings.catch_warnings():
        warnings.simplefilter("error")                      # inside the table: no warning
        en = c.en_sustentado(1e-4, 4e-4)
    # the background is reduced by what the column took from it, below 1e-7 here
    assert en == pytest.approx(esperado, rel=1e-6)
    # and the current only sets n_e, which this linear chemistry does not see
    assert c.en_sustentado(1e-5, 4e-4) == pytest.approx(en, rel=1e-6)


def test_quantum_na_parede_desligado_leva_tudo_ao_fundo(caso):
    c = Coluna0D(caso, P, TG, R, gamas=GAMAS, quantum_na_parede=False)
    e_n, ne, t = 120.0, 1.0e6, 2.0e-3
    _, y = c.integra(e_n, ne, t, n_saidas=2, rtol=1e-10, atol=1e-12)
    nomes = [s.upper() for s in c.especies]
    nu_1 = nu_parede(c, "N2(V1)")
    s1 = coef(e_n)[2] * ne * c.n_gas
    assert y[-1, nomes.index("N2(V1)")] == pytest.approx(s1 / nu_1 * (1 - np.exp(-nu_1 * t)), rel=1e-6)


def test_avisa_fora_da_tabela(caso):
    # T_g and E/N outside the table are taken at its edge node: the column says so
    with pytest.warns(RuntimeWarning, match="T_g = 1000 K outside the table"):
        Coluna0D(caso, P, 1000.0, R, gamas=GAMAS)
    c = Coluna0D(caso, P, TG, R, gamas=GAMAS)
    with pytest.warns(RuntimeWarning, match="above the table"):
        c.integra(1.1 * c.tab.e_n[-1], 1e6, 1e-9, n_saidas=2)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        c.integra(100.0, 1e6, 1e-9, n_saidas=2)
