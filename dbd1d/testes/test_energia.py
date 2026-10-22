# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests of the electron energy equation (local mean energy approximation)."""
import numpy as np
import pytest

from conftest import RAIZ, quimica_do, precisa_arquivo
from modelo import KB, ME, Q, Modelo
from quimica_zdp import Quimica
from tabela import E_K0, E_MUN, E_PEL, E_PINEL, E_POT, Tabela, TabelaEnergia


@pytest.fixture(scope="module")
def pecas():
    return (quimica_do(RAIZ), Tabela(RAIZ / "tabela_bolsig.dat"),
            TabelaEnergia(RAIZ / "tabela_energia.dat"))


def modelo(pecas, **kw):
    q, t, te = pecas
    return Modelo(q, t, tabela_energia=te, **kw)


def test_tabela_por_energia_reproduz_os_nos(pecas):
    te = pecas[2]
    np.testing.assert_allclose(te(te.eps[::37]), te.valores[::37], rtol=1e-13)
    np.testing.assert_allclose(te.energia_de_campo_local(te.e_n[::37]), te.eps[::37], rtol=1e-12)


def test_potencia_tabelada_igual_ao_ganho_de_campo_local(pecas):
    # in steady state, the power absorbed per N is mu*N*(E/N)**2 in each row
    te = pecas[2]
    ganho = te.valores[:, E_MUN] * (te.e_n * 1e-17) ** 2
    np.testing.assert_allclose(te.valores[:, E_POT], ganho, rtol=1e-6)


@pytest.mark.parametrize("e_n", [30.0, 100.0, 300.0, 1000.0, 3000.0])
def test_avalanche_uniforme_mantem_a_energia_de_campo_local(pecas, e_n):
    # uniform plasma and field with eps = eps(E/N): the mean energy does not change,
    # counting the dilution by the electrons that ionization creates,
    #   d(eps)/dt = (d(n_eps)/dt - eps d(n_e)/dt) / n_e = 0.
    # The BOLSIG+ power P/N is the one the field delivers and includes the growth
    # k_ion*eps; only P_el + P_inel enter as loss. With eps 10% above the
    # energy falls, 10% below it rises.
    md = modelo(pecas, n_celulas=40)
    campo = np.full(md.m + 1, e_n * 1e-21 * md.n_gas * 1e6)
    n = np.zeros((md.n_esp, md.m))
    n[md.i_e] = 1.0e10
    eps0 = md.tab_e.energia_de_campo_local(e_n)[0]
    c = md.m // 2
    taxas = []
    for fator in (1.0, 1.1, 0.9):
        n_eps = n[md.i_e] * eps0 * fator
        mob, dif, temp, k_b, _, extra = md.coeficientes(md.campo_nas_celulas(campo), n, n_eps)
        G, _, gi = md.fluxos(n, campo, mob, dif, temp)
        d_eps = md.d_energia(n, n_eps, campo, G, mob, temp, extra, gi)
        k = md.q.coeficientes(k_b, np.full(md.m, 300.0), temp[md.i_e])
        d_ne = md.q.fontes(md.densidades_completas(n), k)[md.q.indice["E"]]
        eps = eps0 * fator
        escala = n[md.i_e, c] * md.n_gas * extra["pot"][c]
        taxas.append((d_eps[c] - eps * d_ne[c]) / escala)
    # residual measured below 4e-5 of P above 10 Td (interpolation and noise of
    # BOLSIG+); the corrected defect (Q1) gave 0.37% at 30 Td and 7% at 100 Td
    assert abs(taxas[0]) < 2e-4
    assert taxas[1] < 0.0 < taxas[2]


@precisa_arquivo("bolsigdb.dat")
def test_potencias_da_tabela_de_energia(pecas):
    # P_inel is the sum threshold*k of the processes of the background gas in bolsigdb.dat (without
    # superelastic nor Ar*), and P - P_el - P_inel = k_ion*eps: the table comes from the
    # temporal growth model of BOLSIG+, which the loss of the fluid assumes
    linhas = (RAIZ / "bolsigdb.dat").read_text().splitlines()
    limiar = {}
    for i, l in enumerate(linhas):
        if l.strip() in ("EXCITATION", "IONIZATION"):
            limiar[linhas[i + 1].replace(" ", "")] = float(linhas[i + 2].split()[0])
    te = pecas[2]
    v = te.valores
    k1, k2 = v[:, E_K0], v[:, E_K0 + 1]
    p_inel = limiar["Ar->Ar^+"] * k1 + limiar["Ar<->Ar*"] * k2
    np.testing.assert_allclose(v[:, E_PINEL], p_inel, rtol=1e-12, atol=1e-300)
    acima = te.e_n >= 10.0
    resto = v[acima, E_POT] - v[acima, E_PEL] - v[acima, E_PINEL]
    np.testing.assert_allclose(resto, k1[acima] * te.eps[acima], rtol=0, atol=1e-4 * v[acima, E_POT].max())
    assert np.max(np.abs(resto - k1[acima] * te.eps[acima]) / v[acima, E_POT]) < 1e-4


def test_fluxo_de_energia_na_parede(pecas):
    # electrons with eps = 15 eV (Te = 10 eV) and field +1e6 V/m: at the left
    # wall (drift towards it) 5/2 kTe per electron of the thermal flux leave (energy
    # flux (5/6) v_term n_eps, Hagelaar et al. 2000) plus
    # mu_eps*E*n_eps; on the right only the thermal flux reduced by the Boltzmann factor. No ions, no secondaries.
    md = modelo(pecas, n_celulas=40, agrupamento=1.5)
    n0, eps, e = 1.0e10, 15.0, 1.0e6
    n = np.zeros((md.n_esp, md.m))
    n[md.i_e] = n0
    n_eps = n[md.i_e] * eps
    campo = np.full(md.m + 1, e)
    mob, dif, temp, _, _, extra = md.coeficientes(md.campo_nas_celulas(campo), n, n_eps)
    ge = md.fluxos_energia(n, n_eps, campo, mob, temp, extra, np.zeros(2))
    te = (2.0 / 3.0) * eps * Q / KB
    vt = np.sqrt(8.0 * KB * te / (np.pi * ME))
    termico = n0 * 0.5 * vt * 2.5 * KB * te / Q
    f = np.exp(-Q * e * md.d[-1] / (KB * te))
    np.testing.assert_allclose(-ge[0], termico + extra["mu_eps"][0] * e * n0 * eps, rtol=1e-9)
    np.testing.assert_allclose(ge[-1], f * termico, rtol=1e-9)


def test_energia_no_lado_direito(pecas):
    # the energy row of the right-hand side is -div(Gamma_eps) + gain - loss +
    # collisions with Ar* - energy of the recombined electrons, with the gain computed from the electron fluxes at the faces
    # and the loss from the elastic and inelastic powers of the table
    md = modelo(pecas, n_celulas=30, agrupamento=1.5, espessura=1.75e-3, v_pico=5.0e3)
    gerador = np.random.default_rng(5)
    n = 10.0 ** gerador.uniform(9, 12, size=(md.n_esp, md.m))
    n_eps = n[md.i_e] * gerador.uniform(1.0, 8.0, md.m)
    y = np.concatenate([n.ravel(), n_eps, [1e-6, -2e-6]])
    t = 2.5e-8
    d_eps = md.energia_do_estado(md.rhs(t, y))
    _, campo = md.campo(t, n, 1e-6, -2e-6)
    mob, dif, temp, k_b, _, extra = md.coeficientes(md.campo_nas_celulas(campo), n, n_eps)
    G, _, gi = md.fluxos(n, campo, mob, dif, temp)
    ge = md.fluxos_energia(n, n_eps, campo, mob, temp, extra, gi)
    k = md.q.coeficientes(k_b, np.full(md.m, 300.0), temp[md.i_e])
    i_m = md.nomes.index("AR*")
    ar_m = n[md.i_e] * n[i_m] * (11.5 * k[md.q.reacoes.index("bolsig:AR*->AR")]
                                 - 4.3 * k[md.q.reacoes.index("bolsig:AR*->AR^+")])
    v = md.tab_e(extra["eps"])
    perda = v[:, E_PEL] + v[:, E_PINEL]
    # recombination: each lost electron carries the mean energy
    eps = extra["eps"]
    te = (2.0 / 3.0) * eps * 1.602176634e-19 / 1.380649e-23
    ne, ar1, ar2 = n[md.i_e], n[md.nomes.index("AR^+")], n[md.nomes.index("AR2^+")]
    perdidos = (8.5e-7 * (te / 300.0) ** -0.67 * ne * ar2 + 8.75e-27 * (te / 11600.0) ** -4.5 * ar1 * ne ** 2
                + 3.0e-14 * ne * ar2)
    esperado = (-np.diff(ge) / md.h + md.ganho_de_energia(G, campo)
                - n[md.i_e] * md.n_gas * perda + ar_m - eps * perdidos)
    np.testing.assert_allclose(d_eps, esperado, rtol=1e-12)


def test_fluxo_de_energia_interno_exato_no_equilibrio(pecas):
    # with uniform mu_eps and D_eps and uniform field, n_eps = exp(-mu_eps*E*x/D_eps)
    # (electron drift towards -x balanced by diffusion) has zero energy flux
    # at every inner face, also on the clustered mesh
    md = modelo(pecas, n_celulas=40, agrupamento=2.0)
    mu, dd, e = 0.04, 0.3, 2.0e4
    n_eps = np.exp(-mu * e * md.xc / dd)
    n = np.zeros((md.n_esp, md.m))
    n[md.i_e] = 1.0
    campo = np.full(md.m + 1, e)
    extra = {"mu_eps": np.full(md.m, mu), "d_eps": np.full(md.m, dd)}
    mob = np.zeros((md.n_esp, md.m))
    temp = np.full((md.n_esp, md.m), 300.0)
    ge = md.fluxos_energia(n, n_eps, campo, mob, temp, extra, np.zeros(2))
    assert np.max(np.abs(ge[1:-1])) < 1e-12 * dd * np.max(n_eps) / np.min(md.d)


def test_energia_media_sem_eletrons_e_a_de_campo_local(pecas):
    # where there are no electrons (n_e zero or slightly negative by rounding of the
    # integrator) the ratio n_eps/n_e makes no sense: the mean energy is that of the
    # local field; with abundant electrons, it is n_eps/n_e
    md = modelo(pecas, n_celulas=4)
    e_n = np.array([30.0, 30.0, 30.0, 30.0])
    n = np.zeros((md.n_esp, 4))
    n[md.i_e] = [-1.0e-6, 0.0, 1.0e6, 1.0e12]
    n_eps = np.array([3.0e-3, 0.0, 7.0e6, 7.0e12])
    eps = md.energia_media(n, n_eps, e_n)
    lfa = md.tab_e.energia_de_campo_local(30.0)[0]
    np.testing.assert_allclose(eps[:2], lfa, rtol=1e-3)
    np.testing.assert_allclose(eps[2:], 7.0, rtol=1e-6)


def test_ganho_de_energia_na_celula_da_parede(pecas):
    # uniform plasma and field: the work of the field per electron is the same in every
    # cell, including the wall cells. The absorption flux at the wall face is
    # dominated by thermal motion, which the field does not perform, and cannot enter
    # no ganho
    md = modelo(pecas, n_celulas=40)
    e_n = 50.0
    campo = np.full(md.m + 1, e_n * 1e-21 * md.n_gas * 1e6)
    n = np.zeros((md.n_esp, md.m))
    n[md.i_e] = 1.0e10
    eps0 = md.tab_e.energia_de_campo_local(e_n)[0]
    n_eps = n[md.i_e] * eps0
    mob, dif, temp, _, _, extra = md.coeficientes(md.campo_nas_celulas(campo), n, n_eps)
    G, _, _ = md.fluxos(n, campo, mob, dif, temp)
    ganho = md.ganho_de_energia(G, campo)
    np.testing.assert_allclose(ganho[[0, -1]], ganho[md.m // 2], rtol=1e-12)


def test_fora_da_tabela_e_contado(pecas):
    # E/N or mean energy outside the range of the table use the end value and are counted
    tab, tab_e = pecas[1], pecas[2]
    tab.fora = tab_e.fora = 0
    tab(np.array([1.0, 1.0e4, 2.0e4]))
    tab_e(np.array([tab_e.eps[-1] * 2.0, 3.0]))
    assert tab.fora == 2 and tab_e.fora == 1


@precisa_arquivo("bolsigdb.dat")
def test_energia_das_colisoes_com_o_ar_metaestavel(pecas):
    # the BOLSIG+ of the Fortran code receives the mixture Ar and Ar* (BOLSIG block of kinet.inp);
    # the table is for pure Ar, so the tabulated power lacks the collisions with
    # Ar*: the superelastic Ar* + e -> Ar + e returns 11.5 eV to the electron and the
    # stepwise ionization e + Ar* -> Ar+ costs 4.3 eV (thresholds of bolsigdb.dat)
    q = pecas[0]
    dE = q.energia_eletronica(RAIZ / "bolsigdb.dat", "AR")
    esperado = {"bolsig:AR*->AR": 11.5, "bolsig:AR*->AR^+": -4.3}
    for i, r in enumerate(q.reacoes):
        assert dE[i] == pytest.approx(esperado.get(r, 0.0)), r


def test_ganho_superelastico_na_equacao_de_energia(pecas):
    md = modelo(pecas, n_celulas=20)
    campo = np.full(md.m + 1, 1.0e5)
    n = np.zeros((md.n_esp, md.m))
    n[md.i_e] = 1.0e10
    n_eps = n[md.i_e] * 2.0
    i_m = md.nomes.index("AR*")
    d = {}
    for ar_m in (0.0, 1.0e13):
        n[i_m] = ar_m
        mob, dif, temp, k_b, _, extra = md.coeficientes(md.campo_nas_celulas(campo), n, n_eps)
        G, _, gi = md.fluxos(n, campo, mob, dif, temp)
        d[ar_m] = md.d_energia(n, n_eps, campo, G, mob, temp, extra, gi)
    k = md.q.coeficientes(k_b, np.full(md.m, 300.0), temp[md.i_e])
    j3 = md.q.reacoes.index("bolsig:AR*->AR")
    j4 = md.q.reacoes.index("bolsig:AR*->AR^+")
    esperado = 1.0e10 * 1.0e13 * (11.5 * k[j3] - 4.3 * k[j4])
    np.testing.assert_allclose(d[1.0e13] - d[0.0], esperado, rtol=1e-10)


def test_recombinacao_tira_a_energia_dos_eletrons(pecas):
    # each electron that vanishes in recombination (Ar2+ + e -> Ar* + Ar, Ar+ + 2e ->
    # Ar + e, e + Ar2+ -> 2Ar) carries the mean energy: d(n_eps) gains -eps*rate.
    # Rates from the expressions of kinet.inp, with Te = (2/3) eps
    md = modelo(pecas, n_celulas=10)
    campo = np.zeros(md.m + 1)
    eps = 2.0
    n = np.zeros((md.n_esp, md.m))
    n[md.i_e] = 1.0e12
    n_eps = n[md.i_e] * eps
    d = {}
    for com_ions in (False, True):
        n[md.nomes.index("AR^+")] = 3.0e12 if com_ions else 0.0
        n[md.nomes.index("AR2^+")] = 2.0e12 if com_ions else 0.0
        mob, dif, temp, _, _, extra = md.coeficientes(md.campo_nas_celulas(campo), n, n_eps)
        G, _, gi = md.fluxos(n, campo, mob, dif, temp)
        d[com_ions] = md.d_energia(n, n_eps, campo, G, mob, temp, extra, gi)
    te = (2.0 / 3.0) * eps * 1.602176634e-19 / 1.380649e-23
    k5 = 8.5e-7 * (te / 300.0) ** -0.67
    k8 = 8.75e-27 * (te / 11600.0) ** -4.5
    k12 = 3.0e-14
    perdidos = k5 * 1e12 * 2e12 + k8 * 3e12 * 1e12 ** 2 + k12 * 1e12 * 2e12
    c = md.m // 2
    assert d[True][c] - d[False][c] == pytest.approx(-eps * perdidos, rel=1e-10)


def test_fora_da_tabela_contado_antes_de_prender(pecas):
    # the mean energy is clamped to the range of the table, but the occurrence must be
    # counted before, above and below (I3; the Fortran code counts both sides)
    md = modelo(pecas, n_celulas=4)
    n = np.zeros((md.n_esp, md.m))
    n[md.i_e] = 1.0e10
    tab = md.tab_e
    tab.fora = tab.fora_abaixo = 0
    md.energia_media(n, n[md.i_e] * 5.0 * tab.eps[-1], np.zeros(md.m))
    assert tab.fora == md.m
    md.energia_media(n, n[md.i_e] * 0.5 * tab.eps[0], np.zeros(md.m))
    assert tab.fora_abaixo == md.m
    # without electrons, or with few (1 cm-3, integrator noise), the mean
    # energy is not defined and nothing is counted
    tab.fora = tab.fora_abaixo = 0
    md.energia_media(np.zeros((md.n_esp, md.m)), np.zeros(md.m), np.zeros(md.m))
    poucos = np.zeros((md.n_esp, md.m))
    poucos[md.i_e] = 1.0
    md.energia_media(poucos, 5.0 * tab.eps[-1] * poucos[md.i_e], np.zeros(md.m))
    assert tab.fora == 0 and tab.fora_abaixo == 0


def test_energia_dos_eletrons_secundarios_na_parede(pecas):
    # without electrons and without field, the only energy crossing the wall face is that
    # of the secondaries: eps_sec per electron, gamma per arriving ion, and the ions
    # arrive with n v_th/2 (Ar2+ with twice the mass); it enters the gas (sign +x
    # on the left, -x on the right)
    md = modelo(pecas, n_celulas=10, gama_sec=0.1, eps_sec=4.0)
    n = np.zeros((md.n_esp, md.m))
    n[md.nomes.index("AR^+")] = 2e9
    n[md.nomes.index("AR2^+")] = 3e9
    campo = np.zeros(md.m + 1)
    n_eps = np.zeros(md.m)
    mob, dif, temp, _, _, extra = md.coeficientes(md.campo_nas_celulas(campo), n, n_eps)
    _, _, gi = md.fluxos(n, campo, mob, dif, temp)
    ge = md.fluxos_energia(n, n_eps, campo, mob, temp, extra, gi)
    uma = 1.66053906660e-27
    vt = lambda m: np.sqrt(8.0 * 1.380649e-23 * 300.0 / (np.pi * m))   # noqa: E731
    chegam = 2e9 * vt(39.948 * uma) / 2.0 + 3e9 * vt(2 * 39.948 * uma) / 2.0
    assert ge[0] == pytest.approx(4.0 * 0.1 * chegam, rel=1e-12, abs=0)
    assert ge[-1] == pytest.approx(-4.0 * 0.1 * chegam, rel=1e-12, abs=0)


def test_ganho_do_terceiro_eletron_na_recombinacao_a_tres_corpos(pecas):
    # Ar+ + e + e -> Ar + e: with ganho_tres_corpos = G, the remaining electron
    # receives G eV per event (limiting hypothesis: 15.76 eV, capture into the ground state;
    # default 0). The difference in the right-hand side is G k8 ne^2 n_Ar+
    eps = 0.2
    d = {}
    for g in (0.0, 15.76):
        md = modelo(pecas, n_celulas=6, ganho_tres_corpos=g)
        n = np.zeros((md.n_esp, md.m))
        n[md.i_e] = 1e12
        n[md.nomes.index("AR^+")] = 1e12
        campo = np.zeros(md.m + 1)
        n_eps = n[md.i_e] * eps
        mob, dif, temp, _, _, extra = md.coeficientes(md.campo_nas_celulas(campo), n, n_eps)
        G, _, gi = md.fluxos(n, campo, mob, dif, temp)
        d[g] = md.d_energia(n, n_eps, campo, G, mob, temp, extra, gi)
    te = (2.0 / 3.0) * eps * 1.602176634e-19 / 1.380649e-23
    k8 = 8.75e-27 * (te / 11600.0) ** -4.5
    assert d[15.76][3] - d[0.0][3] == pytest.approx(15.76 * k8 * 1e36, rel=1e-10, abs=0)


def test_fora_da_tabela_nos_estados_gravados(pecas):
    # count per state (for the output log): cells above and below the
    # range of the table, only where there are enough electrons (n_e > 10 cm-3)
    md = modelo(pecas, n_celulas=5)
    n = np.zeros((md.n_var, md.m))
    n[md.i_e] = [1e10, 1e10, 1e10, 1.0, 0.0]
    eps_max, eps_min = md.tab_e.eps[-1], md.tab_e.eps[0]
    n[md.n_esp] = n[md.i_e] * np.array([5 * eps_max, 0.5 * eps_min, 2.0, 5 * eps_max, 0.0])
    y = np.concatenate([n.ravel(), [0.0, 0.0]])
    assert md.fora_da_tabela(y) == (1, 1)
