# SPDX-License-Identifier: GPL-3.0-or-later
"""EEDF gravada (Q15): o gerador grava a EEDF do BOLSIG+ em nós da tabela de
energia (tabela_eedf.dat), e o simula.py grava, como o Fortran
(eedf_borda0.dat, eedf_centro.dat, eedf_borda1.dat), a EEDF interpolada pela
energia média da célula nos nós 2, n/2 e n-1. Normalização do BOLSIG+:
integral de sqrt(u) f du = 1, e a energia média é integral de u^(3/2) f du."""
import subprocess
import sys

import numpy as np
import pytest

from conftest import PROTOTIPO, RAIZ

sys.path.insert(0, str(RAIZ / "scripts"))


def momentos(u, f):
    # ln f linear entre os pontos e exponencial de 0 ao primeiro, integrado em
    # grade fina (independente de momentos_da_eedf: 200 subdivisões)
    ln_f = np.log(f)
    b = (ln_f[1] - ln_f[0]) / (u[1] - u[0])
    uf = np.concatenate([np.linspace(0.0, u[0], 400)[:-1],
                         np.concatenate([np.linspace(u[i], u[i + 1], 201)[:-1] for i in range(len(u) - 1)]),
                         u[-1:]])
    ff = np.exp(np.where(uf < u[0], ln_f[0] + b * (uf - u[0]), np.interp(uf, u, ln_f)))
    trap = getattr(np, "trapezoid", None) or np.trapz
    return trap(np.sqrt(uf) * ff, uf), trap(uf ** 1.5 * ff, uf)


def test_tabela_da_eedf_normalizada_e_coerente_com_a_energia_media():
    from tabela import TabelaEEDF, TabelaEnergia
    te = TabelaEnergia(RAIZ / "tabela_energia.dat")
    tab = TabelaEEDF(RAIZ / "tabela_eedf.dat")
    assert len(tab.eps) >= 300      # todos abaixo de 0,1 Td, 10 por década acima
    for k in range(len(tab.eps)):
        u, f = tab.u[k], tab.f[k]
        norma, media = momentos(u, f)
        # com ln f linear e o trecho inicial exponencial: medido até 0,3% na
        # norma e 0,36% na energia média (5000 Td, grade grossa)
        assert norma == pytest.approx(1.0, rel=5e-3)
        # a energia média do nó é a da tabela de energia no mesmo E/N
        assert tab.eps[k] == pytest.approx(np.interp(np.log(tab.e_n[k]), np.log(te.e_n), te.eps), rel=1e-6)
        assert media / norma == pytest.approx(tab.eps[k], rel=5e-3)


def test_eedf_interpolada_tem_a_energia_media_pedida():
    from tabela import TabelaEEDF
    tab = TabelaEEDF(RAIZ / "tabela_eedf.dat")
    # medido: até 0,36% (perto de 2 eV)
    for eps in np.geomspace(0.04, 30.0, 400):
        u, f = tab(eps)
        norma, media = momentos(u, f)
        assert media / norma == pytest.approx(eps, rel=5e-3)


def test_interpolacao_reproduz_um_no_retirado():
    # tirando o nó k, a EEDF interpolada entre os vizinhos na energia média dele
    # reproduz a energia média e a forma no miolo (f acima de 1e-4 do máximo);
    # fora dos nós de 2 a 5 Td (início das inelásticas, a cauda acima de 11,55 eV
    # muda de forma)
    from tabela import TabelaEEDF
    tab = TabelaEEDF(RAIZ / "tabela_eedf.dat")
    for k in range(5, len(tab.eps) - 5, 3):
        if 1.5 < tab.e_n[k] < 6.0:
            continue
        sem = TabelaEEDF.__new__(TabelaEEDF)
        manter = [j for j in range(len(tab.eps)) if j != k]
        sem.e_n, sem.eps = tab.e_n[manter], tab.eps[manter]
        sem.u, sem.f = [tab.u[j] for j in manter], [tab.f[j] for j in manter]
        sem.ln_eps = np.log(sem.eps)
        u, f = sem(tab.eps[k])
        norma, media = momentos(u, f)
        assert media / norma == pytest.approx(tab.eps[k], rel=2e-2), tab.e_n[k]
        miolo = tab.f[k] > 1e-4 * tab.f[k].max()
        f_no = np.exp(np.interp(tab.u[k][miolo], u, np.log(f)))
        assert np.max(np.abs(np.log(f_no / tab.f[k][miolo]))) < 0.3, tab.e_n[k]


def test_simula_grava_a_eedf_nos_tres_nos(tmp_path):
    from resumo import blocos
    d = tmp_path / "e"
    subprocess.run([sys.executable, str(PROTOTIPO / "simula.py"), str(d), "--v-pico", "5e3", "--n-celulas", "20",
                    "--espessura", "1.75e-3", "--energia", "--ne0", "1e9", "--neutro", "--t-end", "4e-9",
                    "--dt-saida", "2e-9"], check=True, capture_output=True,
                   env={"OPENBLAS_NUM_THREADS": "1", "PATH": ""})
    perfis = list(blocos(d / "perfil.dat"))
    for arquivo, no in (("eedf_borda0.dat", 2), ("eedf_centro.dat", 22 // 2), ("eedf_borda1.dat", 21)):
        registros = list(blocos(d / arquivo))
        assert len(registros) == 3
        for (t, nomes, linhas), (tp, nomes_p, linhas_p) in zip(registros, perfis):
            assert nomes == ["Energy", "EEDF"] and t == tp
            u, f = np.array(linhas).T
            # nó do Fortran (1 = parede): a linha no-1 do perfil
            eps = linhas_p[no - 1][nomes_p.index("n_eps")] / linhas_p[no - 1][nomes_p.index("E")]
            norma, media = momentos(u, f)
            assert media / norma == pytest.approx(min(max(eps, 0.0388), 1e3), rel=1.5e-2)


@pytest.mark.parametrize("m, esperado", [(20, (0, 9, 19)), (21, (0, 9, 20)), (6, (0, 2, 5))])
def test_celulas_da_eedf_sao_os_nos_do_fortran(m, esperado):
    # saida.f90 grava a EEDF nos nós 2, n/2 (divisão inteira) e n-1, com n = m + 2
    # nós contando as paredes (nó 1); a célula do nó k é a de índice k - 2
    sys.path.insert(0, str(PROTOTIPO))
    from simula import celulas_da_eedf

    class Malha:
        pass
    md = Malha()
    md.m = m
    assert celulas_da_eedf(md) == esperado
