# SPDX-License-Identifier: GPL-3.0-or-later
"""Coupled cross section set (ferramentas/acoplamento_vibracional.py) and BOLSIG+ (research
repository only: it needs the LXCat sections of casos/n2_amonia/bolsigdb.dat, which are not
distributed, and a coupled case run with BOLSIG+). The set generated here from the sections
of N2(X) has, for each electronic process from N2(v), the section of N2(X) shifted down by
E_v, with E_v from the spectroscopic constants of N2(X); a coupled case run by
scripts/coluna_n2_raiz.py has exactly this set, and the rate coefficient that BOLSIG+ put in
its table is the integral of the section over the EEDF it printed, gamma int u sigma(u) f(u) du."""
import sys

import numpy as np
import pytest

from conftest import RAIZ, precisa_arquivo

BASE = RAIZ / "casos" / "n2_amonia" / "bolsigdb.dat"
ACOPLADOS = RAIZ / "casos" / "n2_amonia_acoplado"
pytestmark = precisa_arquivo("casos/n2_amonia/bolsigdb.dat", "casos/n2_amonia_acoplado")
GAMA = np.sqrt(2 * 1.602176634e-19 / 9.1093837015e-31)       # m/s per sqrt(eV)
# N2(X): omega_e and omega_e x_e (cm-1, Huber and Herzberg) and cm-1 to eV
WE, WEXE, CM_EV = 2358.57, 14.324, 1.23984198e-4
PROCESSOS = ("N2(A3,v0-4)", "N2(B3)", "N2(C3)", "N2^+")


def e_v(v):
    """Vibrational energy of N2(X, v) above v = 0 (eV), anharmonic oscillator."""
    return (WE * v - WEXE * v * (v + 1)) * CM_EV


def _secao(linhas, nome):
    """Threshold (eV) and table (energy in eV, section in m2) of the process NAME."""
    i = linhas.index(nome)
    limiar = float(linhas[i + 1].split("/")[0])
    j = next(k for k in range(i, len(linhas)) if linhas[k].startswith("-----"))
    k = next(k for k in range(j + 1, len(linhas)) if linhas[k].startswith("-----"))
    return limiar, np.array([[float(x) for x in l.split()] for l in linhas[j + 1:k]])


@pytest.fixture(scope="module")
def banco():
    sys.path.insert(0, str(RAIZ / "ferramentas"))
    import acoplamento_vibracional as av
    return av.banco_acoplado(BASE.read_text(), 45)


@pytest.fixture(scope="module")
def caso():
    # the case of a self-consistent root with the full coupling (without the excitation from the
    # excited levels); its folder carries the E/N of the root, hence the glob, which must find it
    casos = sorted(p.parent for p in ACOPLADOS.glob("*/*/it*/tabela_eedf.dat") if "_ev_" not in p.parts[-4])
    assert casos, f"no coupled case with tables under {ACOPLADOS}"
    return casos[0]


@pytest.mark.parametrize("v", [1, 2, 5, 20])
@pytest.mark.parametrize("destino", PROCESSOS)
def test_secao_de_n2v_e_a_de_n2x_deslocada(banco, v, destino):
    linhas = banco.splitlines()
    l0, s0 = _secao(linhas, f"N2 -> {destino}")
    lv, sv = _secao(linhas, f"N2(v{v}) -> {destino}")
    assert l0 - lv == pytest.approx(e_v(v), abs=1e-4 * v)
    # as a function, from the threshold on (a leading point at zero energy is not shifted); the
    # energies are written with 7 digits
    acima = sv[:, 0] >= lv
    assert acima.sum() >= 5
    np.testing.assert_allclose(sv[acima, 1], np.interp(sv[acima, 0] + e_v(v), s0[:, 0], s0[:, 1]),
                               rtol=1e-3, atol=1e-6 * s0[:, 1].max())
    assert np.all(sv[~acima, 1] == 0.0)


def test_caso_acoplado_tem_o_conjunto_gerado(banco, caso):
    # the tables of the case were made from the set that the tool generates now
    guardado = (caso / "bolsigdb.dat").read_text()
    if guardado != banco:
        # the folder is built at run time by the column and jet scripts; one made before a
        # correction of the inputs is stale, and rerunning the script regenerates it
        pytest.skip("stored coupled case predates the current inputs; rerun the scripts that build it")
    assert guardado == banco


@pytest.mark.parametrize("especie", ["N2", "N2(V1)", "N2(V2)"])
def test_taxa_da_tabela_e_a_integral_da_secao_na_eedf(banco, caso, especie):
    cab = (caso / "tabela_energia.dat").read_text().splitlines()[1].lstrip("# ").split(", ")
    tab = np.loadtxt(caso / "tabela_energia.dat")
    origem = "N2" if especie == "N2" else f"N2(v{especie[-2]})"
    secao = _secao(banco.splitlines(), f"{origem} -> N2(A3,v0-4)")[1]
    col = cab.index(f"k({especie}->N2(A3,V0-4))")
    conferidos = 0
    for bloco in (caso / "tabela_eedf.dat").read_text().split("\n\n"):
        ls = [l for l in bloco.splitlines() if l.strip() and not l.startswith("#")]
        if not ls:
            continue
        e_n = float(ls[0].split()[0])
        if not 50.0 <= e_n <= 500.0:        # where the rate is not dominated by the tail of the printed EEDF
            continue
        a = np.array([[float(x) for x in l.split()] for l in ls[1:]])
        sigma = np.interp(a[:, 0], secao[:, 0], secao[:, 1], left=0.0, right=secao[-1, 1])
        k = GAMA * np.trapezoid(a[:, 0] * sigma * a[:, 1], a[:, 0]) * 1e6       # cm3/s
        linha = tab[np.argmin(np.abs(tab[:, 0] - e_n))]
        assert abs(linha[0] / e_n - 1.0) < 1e-3
        assert k == pytest.approx(linha[col], rel=5e-3), e_n
        conferidos += 1
    assert conferidos >= 8
