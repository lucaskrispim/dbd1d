# SPDX-License-Identifier: GPL-3.0-or-later
"""Vibrational ladder of N2(X) up to v = 45 (ferramentas/escada_vibracional.py),
with the V-T and V-V rate coefficients of Capitelli, Ferreira, Gordiets and
Osipov 2000 (chapter 7)."""
import math
import sys

import numpy as np
import pytest

from conftest import RAIZ, precisa_arquivo, precisa_terceiros

sys.path.insert(0, str(RAIZ / "ferramentas"))
import escada_vibracional as ev  # noqa: E402

HC_K = 1.438776877


def test_energias_de_morse():
    assert ev.energia(0) == 0.0
    assert ev.energia(1) == pytest.approx((2358.57 - 2 * 14.324) * HC_K, rel=1e-12, abs=0)
    quanta = np.diff([ev.energia(v) for v in range(47)])
    assert np.all(quanta > 0) and np.all(np.diff(quanta) < 0)     # anharmonic: the quantum shrinks


def test_vt_do_nivel_1_é_a_tabela_7_1():
    for t in (300.0, 600.0):
        e10 = ev.energia(1)
        k10 = 7.8e-12 * t * math.exp(-218.0 / t ** (1 / 3) + 690.0 / t) / (1.0 - math.exp(-e10 / t))
        assert ev.k_vt(1, t) == pytest.approx(k10, rel=1e-12, abs=0)


def test_vt_escalonado_por_ssh():
    # k_{v,v-1} = k10 Z_v exp((v - 1) delta_VT), eqs. (7.5) to (7.8), gamma_0 >= 20
    t, v = 400.0, 20
    xe = 14.324 / 2358.57
    z = v * (1 - xe) / (1 - v * xe)
    de = 14.324 * HC_K                      # Delta E = omega_e x_e, eq. (3.67), p. 36
    delta = 1.87 * de * 0.2 ** (2 / 3) * 14.0 ** (1 / 3) / (ev.energia(1) ** (1 / 3) * t ** (1 / 3))
    assert ev.k_vt(v, t) == pytest.approx(ev.k_vt(1, t) * z * math.exp((v - 1) * delta), rel=1e-12, abs=0)


def test_vv_exotérmico_cai_com_o_desajuste():
    # A(v) -> A(v-1) with B(w-1) -> B(w), v < w: exothermic, smaller for a larger |v - w|
    t = 400.0
    assert ev.k_vv(5, 6, t) > ev.k_vv(5, 15, t) > ev.k_vv(5, 30, t)
    with pytest.raises(ValueError, match="^k_vv gives the exothermic direction, v < w$"):
        ev.k_vv(6, 5, t)                     # the endothermic direction comes from detailed balance
    with pytest.raises(ValueError, match="exothermic"):
        ev.k_vv(5, 5, t)                     # no exchange between equal levels


def _f_sobre_f0(y):
    """F(y)/F(y_10^01) of eqs. (7.34) and (7.36), p. 114; y_10^01 = 0 and F(0) = 1."""
    if y <= 20:
        return 0.5 * (3 - math.exp(-2 * y / 3)) * math.exp(-2 * y / 3)
    return 8 * math.sqrt(math.pi / 3) * y ** (7 / 3) * math.exp(-3 * y ** (2 / 3))


def test_vv_de_capitelli():
    # eq. (7.40): k01 G_S(v, w) g(v, w), with (7.26a), (7.36), (7.39) and (7.41); Delta E = omega_e x_e
    # and 2y/3 = delta_VV |v - w| (eq. 7.37 with the 0.427 of eq. 7.39)
    t, v, w = 500.0, 3, 12
    xe = 14.324 / 2358.57
    z = lambda n: n * (1 - xe) / (1 - n * xe)  # noqa: E731
    delta = 0.427 * 14.324 * HC_K * 0.2 * math.sqrt(14.0 / t)
    g = 0.645 + 0.617 * ((w - 10) / 30) ** 3
    esperado = 0.9e-14 * (t / 300) ** 1.5 * z(v) * z(w) * _f_sobre_f0(1.5 * delta * abs(v - w)) * g
    assert ev.k_vv(v, w, t) == pytest.approx(esperado, rel=1e-12, abs=0)


# ---------------------------------------------------------------- against the book and independent sources

def _coeficiente(nome):
    """Coefficient c of the $ line escv_NAME = c/Tgas**(...) or c/sqrt(Tgas) that the tool writes."""
    import re
    linha = next(l for l in ev.variaveis() if l.startswith(f"$ escv_{nome} ="))
    return float(re.match(r"\$ escv_\w+ = ([0-9.]+)d([+-]\d+)/", linha).expand(r"\1e\2"))


# constants of N2(X) and of the SSH model as printed in Capitelli et al. 2000, typed here
# so that the test does not reuse the constants of the tool
WE_LIVRO, WEXE_LIVRO = 2358.57, 14.324     # cm-1
HCK_LIVRO = 1.438776877                    # cm K
L_LIVRO, MU_LIVRO = 0.2, 14.0              # A (eq. 7.40), reduced mass of N2-N2 (u)


def test_delta_vt_igual_ao_de_capitelli():
    # eq. (7.8), p. 107, gamma_0 >= 20: delta_VT = 1.87 Delta E L^(2/3) mu^(1/3) / (E10^(1/3) T^(1/3)),
    # with Delta E = omega_e x_e and E10 = omega_e - 2 omega_e x_e (eq. 3.67, p. 36)
    de = WEXE_LIVRO * HCK_LIVRO
    e10 = (WE_LIVRO - 2 * WEXE_LIVRO) * HCK_LIVRO
    livro = 1.87 * de * L_LIVRO ** (2 / 3) * MU_LIVRO ** (1 / 3) / e10 ** (1 / 3)
    assert _coeficiente("dvt") == pytest.approx(livro, rel=5e-3)


def test_delta_vt_perto_de_sun():
    # Sun et al. 2022 (casos/n2_amonia/kinet.inp:150): 0.86 (14/T)^(1/3) per quantum; Sun's 0.86 follows
    # from eq. (7.8) with omega_e x_e rounded to 14.0 cm-1, hence 3% and not 0.5%
    assert _coeficiente("dvt") == pytest.approx(0.86 * 14.0 ** (1 / 3), rel=0.03)


def test_delta_vv_igual_ao_de_capitelli():
    # eq. (7.39), p. 114: delta_VV = 0.427 Delta E L sqrt(mu / T), Delta E = omega_e x_e, L = 0.2 A (eq. 7.40)
    livro = 0.427 * WEXE_LIVRO * HCK_LIVRO * L_LIVRO * math.sqrt(MU_LIVRO)
    assert _coeficiente("dvv") == pytest.approx(livro, rel=5e-3)


def test_delta_vv_perto_de_gordiets():
    # Gordiets et al. 1995: delta_VV = 6.8 / sqrt(T)
    assert _coeficiente("dvv") == pytest.approx(6.8, rel=0.05)


def test_vv_tem_o_fator_de_capitelli():
    # N2(10) + N2(0) -> N2(9) + N2(1) is k_{v,v-1}^{w-1,w} of eq. (7.40), p. 115, with v = 10 and w = 1:
    # the index of G_S(v, w) is Delta = |v - w| = 9, since E_{v,v-1} - E_{w,w-1} = 2 omega_e x_e (w - v)
    # in eq. (7.37) and 2y/3 = delta_VV Delta. Against the simplified form k01 Z_v Z_w g exp(-delta_VV Delta)
    # the rate has the factor F(y)/F(0) exp(+delta_VV Delta) = 1.5 - 0.5 exp(-delta_VV Delta), both ways
    t, delta_n = 500.0, 9
    k = {(tuple(r.reagentes), tuple(r.produtos)): kk for r, kk in _avalia(t)}
    xe = 14.324 / 2358.57
    z = lambda n: n * (1 - xe) / (1 - n * xe)  # noqa: E731
    dvv = _coeficiente("dvv") / math.sqrt(t)
    simples = 0.9e-14 * (t / 300) ** 1.5 * z(10) * z(1) * 0.645 * math.exp(-dvv * delta_n)
    q = (ev.energia(10) - ev.energia(9)) - (ev.energia(1) - ev.energia(0))
    fator = 1.5 - 0.5 * math.exp(-dvv * delta_n)
    assert k[(("N2(v1)", "N2(v9)"), ("N2", "N2(v10)"))] / simples == pytest.approx(fator, rel=1e-6)
    assert k[(("N2", "N2(v10)"), ("N2(v1)", "N2(v9)"))] / (simples * math.exp(q / t)) == pytest.approx(fator, rel=1e-6)


def test_vv_no_ramo_de_y_acima_de_20():
    # |v - w| = 45 at 300 K (N2(1) + N2(45) -> N2(0) + N, N from the top, v = 1, w = 46): y > 20 and
    # F(y) = 8 (pi/3)^(1/2) y^(7/3) exp(-3 y^(2/3)), eq. (7.34), p. 114, with y of eq. (7.37) from the
    # Morse levels of eq. (3.67). The tool takes 2y/3 = delta_VV |v - w| with the 0.427 of eq. (7.39),
    # which is 0.32 x 2 x 2/3 = 0.4267 rounded: 0.08% in y, about 1.5% through exp(-3 y^(2/3)) at y = 26
    t, v, w = 300.0, 1, 46
    e = lambda n: (WE_LIVRO * n - WEXE_LIVRO * n * (n + 1)) * HCK_LIVRO  # noqa: E731
    y = 0.32 * ((e(v) - e(v - 1)) - (e(w) - e(w - 1))) * L_LIVRO * math.sqrt(MU_LIVRO / t)
    assert y > 20
    xe = WEXE_LIVRO / WE_LIVRO
    z = lambda n: n * (1 - xe) / (1 - n * xe)  # noqa: E731
    g = 0.645 + 0.617 * ((w - 10) / 30) ** 3
    livro = 0.9e-14 * (t / 300) ** 1.5 * z(v) * z(w) * 8 * math.sqrt(math.pi / 3) * y ** (7 / 3) \
        * math.exp(-3 * y ** (2 / 3)) * g
    assert ev.k_vv(v, w, t) == pytest.approx(livro, rel=0.02, abs=0)
    k = {(tuple(r.reagentes), tuple(r.produtos)): kk for r, kk in _avalia(t)}
    assert k[(("N2(v1)", "N2(v45)"), ("N2", "N", "N"))] == pytest.approx(ev.k_vv(v, w, t), rel=1e-11, abs=0)


@precisa_arquivo("casos/n2_amonia_v45/kinet.inp")
def test_kinet_de_n2_amonia_v45_tem_os_deltas_corretos():
    # the case as committed carries the expressions the tool writes now
    texto = (RAIZ / "casos" / "n2_amonia_v45" / "kinet.inp").read_text().splitlines()
    for linha in ev.variaveis():
        assert linha in texto, linha


def test_vt_com_n_multiquântico():
    # eqs. (7.13) and (7.14): k_v split equally among Delta v = 1..min(v, 5)
    t, v = 500.0, 12
    kv = 4e-10 * (t / 300) ** 0.5 * math.exp(-7280.0 / t + 0.065 * ev.energia(v) / t)
    assert ev.k_vt_n(v, 3, t) == pytest.approx(kv / 5, rel=1e-12, abs=0)
    assert ev.k_vt_n(3, 3, t) == pytest.approx(4e-10 * (t / 300) ** 0.5
                                               * math.exp(-7280.0 / t + 0.065 * ev.energia(3) / t) / 3, rel=1e-12, abs=0)
    assert ev.k_vt_n(v, 6, t) == 0.0


def test_reações_conservam_átomos_e_contam_certo():
    reacoes = ev.reacoes(45)
    n = lambda s: 0 if s == "N2" and False else (1 if s == "N" else 2)  # noqa: E731
    for r in reacoes:
        assert sum(n(s) for s in r.reagentes) == sum(n(s) for s in r.produtos), r
    vt = [r for r in reacoes if r.tipo == "VT"]
    vv = [r for r in reacoes if r.tipo == "VV"]
    assert len(vt) == 2 * 45 and len(vv) == 2 * (45 * 44 // 2)
    assert any(r.produtos.count("N") == 2 for r in reacoes)            # dissociation from the top of the ladder


@precisa_terceiros
def test_coeficientes_do_kinet_gerado_batem_com_as_fórmulas(tmp_path):
    import caso
    sys.path.insert(0, str(RAIZ / "dbd1d"))
    from quimica_zdp import Quimica
    texto = ev.substitui_vibracao((RAIZ / "casos" / "n2_amonia" / "kinet.inp").read_text(), 45)
    (tmp_path / "kinet.inp").write_text(texto)
    caso.gera_zdplaskin(tmp_path)
    q = Quimica(tmp_path / "zdplaskin_m.F90")
    t = 450.0
    k = q.coeficientes(np.zeros((1, 64)), np.array([t]), np.array([20000.0]), np.array([100.0]),
                       np.array([3e16]))[:, 0]
    nomes = [r.upper() for r in q.reacoes]

    def coef(texto_reacao):
        return k[nomes.index(texto_reacao.upper())]
    assert coef("N2(V20)+N2=>N2(V19)+N2") == pytest.approx(ev.k_vt(20, t), rel=1e-10, abs=0)
    assert coef("N2(V19)+N2=>N2(V20)+N2") == pytest.approx(
        ev.k_vt(20, t) * math.exp(-(ev.energia(20) - ev.energia(19)) / t), rel=1e-10, abs=0)
    assert coef("N2(V3)+N2(V11)=>N2(V2)+N2(V12)") == pytest.approx(ev.k_vv(3, 12, t), rel=1e-10, abs=0)
    q_ex = (ev.energia(3) - ev.energia(2)) - (ev.energia(12) - ev.energia(11))
    assert coef("N2(V2)+N2(V12)=>N2(V3)+N2(V11)") == pytest.approx(ev.k_vv(3, 12, t) * math.exp(-q_ex / t), rel=1e-10, abs=0)
    assert coef("N2(V12)+N=>N2(V9)+N") == pytest.approx(ev.k_vt_n(12, 3, t), rel=1e-10, abs=0)
    # |v - w| = 45 at 450 K is in the branch y > 20 of F(y), eq. (7.34)
    assert 1.5 * ev._delta_vv(t) * 45 > 20
    assert coef("N2(V1)+N2(V45)=>N2+N+N") == pytest.approx(ev.k_vv(1, 46, t), rel=1e-10, abs=0)
    # the vibrational kinetics of the base chemistry (v <= 8) is replaced, the rest stays
    assert "N2(V9)" in " ".join(q.especies).upper() and "N2(V45)" in " ".join(q.especies).upper()
    assert not any("KVV10_N2N2" in r.upper() for r in texto.splitlines() if "=>" in r)
    assert "N+N+N2=>N2(A3)+N2" in nomes and any(n.startswith("BOLSIG:N2->N2(V1)") for n in nomes)


def test_reações_vibracionais_em_blocos_arroba_também_saem():
    # the N2-O2 chemistry of the thesis writes the V-T with @ lists
    texto = """SPECIES
N2  N2(v1)  N2(v2)  N  N2^+  e
END
REACTIONS
@A + N2 => @B + N2                                 !   kVT10_N2N2 * @R
    @A = N2(v1)  N2(v2)
    @B = N2  N2(v1)
    @R = 1.0d0  2.0d0
e + N2^+ => N  + @B                                !   1.8d-7 * @R
    @B = N  N
    @R = 0.5d0  0.5d0
N2(v1) + N2^+ => N2 + N2^+                         !   1.0d-10
END
"""
    novo = ev.substitui_vibracao(texto, 3)
    assert "kVT10_N2N2" not in novo                     # vibrational only, after expanding the @ lists
    assert "e + N2^+ => N  + @B" in novo                 # has an electron and an ion: kept
    assert "N2(v1) + N2^+ => N2 + N2^+" in novo          # has an ion: kept


# ---------------------------------------------------------------- the generated text, evaluated

def _avalia(t, vmax=45):
    """Rate coefficient (cm3/s) of each reaction of the ladder at T, from the text that
    goes to the kinet.inp: the $ variables and the expressions, in Python."""
    import re
    ambiente = {"exp": math.exp, "sqrt": math.sqrt, "Tgas": t}
    py = lambda s: re.sub(r"(\d)d([+-]?\d)", r"\1e\2", s)  # noqa: E731
    for linha in ev.variaveis(vmax):
        if linha.startswith("$ double precision"):
            continue
        m = re.match(r"\$ if \((.*)\) (\w+) = (.*)$", linha)       # one-line IF of the F(y) branch y > 20
        if m:
            if eval(py(m.group(1)), ambiente):
                ambiente[m.group(2)] = eval(py(m.group(3)), ambiente)
            continue
        nome, expr = linha[1:].split("=", 1)
        ambiente[nome.strip()] = eval(py(expr), ambiente)
    return [(r, eval(py(r.taxa), ambiente)) for r in ev.reacoes(vmax)]


@pytest.mark.parametrize("t", [300.0, 797.0])
def test_texto_gerado_dá_as_fórmulas(t):
    for r, k in _avalia(t):
        v = lambda s: 0 if s == "N2" else int(s[4:-1])  # noqa: E731
        if r.tipo == "VT" and v(r.reagentes[0]) > v(r.produtos[0]):
            assert k == pytest.approx(ev.k_vt(v(r.reagentes[0]), t), rel=1e-11, abs=0), r
        elif r.tipo == "VV" and v(r.reagentes[0]) > v(r.produtos[0]):
            assert k == pytest.approx(ev.k_vv(v(r.reagentes[0]), v(r.produtos[1]), t), rel=1e-11, abs=0), r
        elif r.tipo == "VTN":
            vi, vf = v(r.reagentes[0]), v(r.produtos[0])
            assert k == pytest.approx(ev.k_vt_n(vi, vi - vf, t), rel=1e-11, abs=0), r


@pytest.mark.parametrize("t", [300.0, 797.0])
def test_balanço_detalhado_de_todos_os_pares(t):
    # forward over backward = exp(Delta E / kT): the products of the inverse are the reactants of the direct
    k = {(tuple(r.reagentes), tuple(r.produtos)): kk for r, kk in _avalia(t) if r.tipo in ("VT", "VV")}
    e = lambda s: 0.0 if s == "N2" else ev.energia(int(s[4:-1]))  # noqa: E731
    pares = 0
    for (reag, prod), kf in k.items():
        if (prod, reag) in k:
            de = sum(e(s) for s in reag) - sum(e(s) for s in prod)
            assert kf / k[(prod, reag)] == pytest.approx(math.exp(de / t), rel=1e-10), (reag, prod)
            pares += 1
    assert pares == 2 * (45 + 45 * 44 // 2)


# ---------------------------------------------------------------- edges and limits

def test_nível_zero_não_relaxa():
    assert ev.k_vt(0, 400.0) == 0.0


def test_vt_com_n_no_salto_máximo_e_acima():
    t = 500.0
    assert ev.k_vt_n(4, 4, t) > 0.0 and ev.k_vt_n(4, 5, t) == 0.0     # Delta v up to v
    assert ev.k_vt_n(30, 5, t) > 0.0 and ev.k_vt_n(30, 6, t) == 0.0   # and up to 5


@pytest.mark.parametrize("dv", [0, -1])
def test_vt_com_n_recusa_salto_não_positivo(dv):
    with pytest.raises(ValueError, match=f"^Delta v must be at least 1, not {dv}$"):
        ev.k_vt_n(10, dv, 500.0)


def test_vt_com_n_sem_ativação_no_alto_da_escada():
    # 0.065 E_v above 7280 K (v >= 47): the activation term goes, k = k0 / 5
    t, v = 500.0, 47
    assert 0.065 * ev.energia(v) > 7280.0 >= 0.065 * ev.energia(45)
    assert ev.k_vt_n(v, 2, t) == pytest.approx(4e-10 * (t / 300) ** 0.5 / 5, rel=1e-12, abs=0)


def test_escada_mínima_de_um_nível():
    reacoes = ev.reacoes(1)
    tipos = [r.tipo for r in reacoes]
    assert tipos.count("VT") == 2 and tipos.count("VV") == 0 and tipos.count("VTN") == 1
    assert tipos.count("DIS") == 2


@pytest.mark.parametrize("vmax", [0, -3])
def test_escada_vazia_é_recusada(vmax):
    with pytest.raises(ValueError, match=f"^vmax must be at least 1, not {vmax}$"):
        ev.reacoes(vmax)
    with pytest.raises(ValueError, match=f"^vmax must be at least 1, not {vmax}$"):
        ev.substitui_vibracao(BASE, vmax)


def test_kinet_sem_seções_é_recusado():
    with pytest.raises(ValueError, match="^kinet.inp without SPECIES or REACTIONS$"):
        ev.substitui_vibracao("SPECIES\nN2 E\nEND\n", 3)


# ---------------------------------------------------------------- regression of the text

BASE = """ELEMENTS
N  E
END
SPECIES
N2  N2(v1)  N2(v2)  N  N2^+  E
END
BOLSIG
N2
END
REACTIONS
E + N2 => E + N2(v1)                       ! BOLSIG N2 -> N2(v1)
N2(v1) + N2 => N2 + N2                     ! 1.0d-15
N2(v2) + N2(v1) => N2(v1) + N2(v2)         ! 1.0d-14
N + N + N2 => N2 + N2                      ! 8.3d-34
E + N2^+ => N + N                          ! 1.8d-7
END
"""


def test_retrato_do_kinet_com_escada_até_v_3():
    referencia = RAIZ / "dbd1d" / "testes" / "dados" / "escada_v3.inp"
    assert ev.substitui_vibracao(BASE, 3) == referencia.read_text()


def test_espécies_novas_em_linhas_de_oito_sem_repetir():
    texto = ev.substitui_vibracao(BASE, 20)
    especies = texto.split("SPECIES\n")[1].split("\nEND")[0].splitlines()
    nomes = " ".join(especies).split()
    assert len(nomes) == len(set(nomes)) and "N2(v20)" in nomes and "N2(v21)" not in nomes
    assert max(len(l.split()) for l in especies[1:]) <= 8


def test_linha_de_comando(tmp_path):
    (tmp_path / "base.inp").write_text(BASE)
    import subprocess
    r = subprocess.run([sys.executable, str(RAIZ / "ferramentas" / "escada_vibracional.py"),
                        str(tmp_path / "base.inp"), str(tmp_path / "novo.inp"), "3"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "novo.inp").read_text() == ev.substitui_vibracao(BASE, 3)


def test_linha_de_comando_sem_argumentos_explica_o_uso():
    import subprocess
    r = subprocess.run([sys.executable, str(RAIZ / "ferramentas" / "escada_vibracional.py")], capture_output=True,
                       text=True)
    assert r.returncode != 0
    assert r.stderr.strip() == ("Usage: python3 ferramentas/escada_vibracional.py ORIGIN/kinet.inp "
                                "DESTINATION/kinet.inp [VMAX]")



# ---------------------------------------------------------------- found by mutation testing

def test_fator_g_muda_de_ramo_em_v_10():
    # eq. (7.41): 1 - 0.04 max(v, w) below 10, 0.645 + 0.617 ((m - 10)/30)^3 from 10 on
    t = 500.0
    xe = 14.324 / 2358.57
    z = lambda n: n * (1 - xe) / (1 - n * xe)  # noqa: E731
    delta = 0.427 * 14.324 * HC_K * 0.2 * math.sqrt(14.0 / t)
    sem_g = lambda v, w: 0.9e-14 * (t / 300) ** 1.5 * z(v) * z(w) * _f_sobre_f0(1.5 * delta * abs(v - w))  # noqa: E731
    assert ev.k_vv(3, 9, t) == pytest.approx(sem_g(3, 9) * (1 - 0.04 * 9), rel=1e-12, abs=0)
    assert ev.k_vv(3, 10, t) == pytest.approx(sem_g(3, 10) * 0.645, rel=1e-12, abs=0)
    assert ev.k_vv(3, 11, t) == pytest.approx(sem_g(3, 11) * (0.645 + 0.617 / 30 ** 3), rel=1e-12, abs=0)


def test_contagem_de_cada_tipo_de_reação():
    for vmax in (1, 4, 45):
        tipos = [r.tipo for r in ev.reacoes(vmax)]
        assert tipos.count("VT") == 2 * vmax and tipos.count("VV") == vmax * (vmax - 1)
        assert tipos.count("VTN") == sum(min(v, 5) for v in range(1, vmax + 1))
        assert tipos.count("DIS") == 1 + vmax
        nomes = {s for r in ev.reacoes(vmax) for s in r.reagentes + r.produtos}
        assert f"N2(v{vmax})" in nomes and f"N2(v{vmax + 1})" not in nomes


def test_texto_da_vt_com_n_sem_ativação_no_alto():
    # vmax = 48: from v = 47 on the generated text has no activation term, as k_vt_n
    t = 600.0
    for r, k in _avalia(t, vmax=48):
        if r.tipo == "VTN":
            vi, vf = int(r.reagentes[0][4:-1]), (0 if r.produtos[0] == "N2" else int(r.produtos[0][4:-1]))
            assert k == pytest.approx(ev.k_vt_n(vi, vi - vf, t), rel=1e-11, abs=0), r


def test_vmax_padrão_é_45_e_um_nível_é_aceito():
    texto = ev.substitui_vibracao(BASE)
    assert "N2(v45)" in texto and "N2(v46)" not in texto
    um = ev.substitui_vibracao(BASE, 1)
    assert "N2(v1) + N2 => N2 + N2" in um and "N2(v2) + N2 =>" not in um


def test_vt_da_base_com_n_também_sai():
    base = BASE.replace("N + N + N2 => N2 + N2", "N2(v2) + N => N2(v1) + N                  ! 1.0d-13\n"
                                                    "N + N + N2 => N2 + N2")
    novo = ev.substitui_vibracao(base, 3)
    assert "1.0d-13" not in novo and "8.3d-34" in novo


def test_comentário_em_reactions_fica():
    base = BASE.replace("REACTIONS\n", "REACTIONS\n# N2(v1) + N2 => N2 + N2 is replaced\n")
    assert "# N2(v1) + N2 => N2 + N2 is replaced" in ev.substitui_vibracao(base, 3).splitlines()


def test_espécie_vibracional_só_no_primeiro_bloco_arroba():
    texto = """SPECIES
N2  N2(v1)  N  N2^+  e
END
REACTIONS
@A + N2 => N2 + @B                                 !   1.0d-16 * @R
    @A = N2(v1)  N2(v1)
    @B = N2  N2
    @R = 1.0d0  2.0d0
END
"""
    assert "1.0d-16" not in ev.substitui_vibracao(texto, 2)


def test_espécie_explícita_conta_junto_com_as_listas():
    # an ion among the explicit species keeps the reaction even if its @ lists are vibrational
    texto = """SPECIES
N2  N2(v1)  N  N2^+  e
END
REACTIONS
@A + N2^+ => N2 + N2^+                             !   2.0d-16 * @R
    @A = N2(v1)
    @R = 1.0d0
END
"""
    assert "2.0d-16" in ev.substitui_vibracao(texto, 2)


def test_base_sem_níveis_vibracionais_ganha_todos():
    base = "ELEMENTS\nN  E\nEND\nSPECIES\nN2  N  E\nEND\nREACTIONS\nN + N + N2 => N2 + N2  ! 8.3d-34\nEND\n"
    especies = ev.substitui_vibracao(base, 3).split("SPECIES\n")[1].split("\nEND")[0].split()
    assert especies == ["N2", "N", "E", "N2(v1)", "N2(v2)", "N2(v3)"]


def test_linha_de_comando_sem_vmax_usa_45(tmp_path):
    import subprocess
    (tmp_path / "base.inp").write_text(BASE)
    r = subprocess.run([sys.executable, str(RAIZ / "ferramentas" / "escada_vibracional.py"),
                        str(tmp_path / "base.inp"), str(tmp_path / "novo.inp")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "novo.inp").read_text() == ev.substitui_vibracao(BASE, 45)
