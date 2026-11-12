# SPDX-License-Identifier: GPL-3.0-or-later
"""Measured conditions of the N2 column (scripts/coluna_n2.py) against their sources."""
import sys

import pytest

from conftest import RAIZ, precisa_arquivo

pytestmark = precisa_arquivo("scripts/coluna_n2.py", "docs/literatura/dados/gordiets1995/n2_puro.csv")


def _condicoes():
    sys.path[:0] = [str(RAIZ / "scripts")]
    import coluna_n2
    return {c["id"]: c for c in coluna_n2.condicoes()}


@pytest.mark.parametrize("cid, eixo, parede", [("G95_15mA", 504.0, 316.0), ("G95_30mA", 590.0, 322.0),
                                               ("G95_80mA", 797.0, 355.0)])
def test_tg_de_gordiets_é_a_média_radial(cid, eixo, parede):
    # Gordiets et al. 1995, fig. 4: measured T_g "in the central region of the tube"; their
    # parabolic profile has axis 2 T_g - T_w, so the radial mean is (axis + T_w)/2, with T_w at
    # 0% O2 from Gordiets et al. 1996, sec. 3.4 (lower end of the ranges 316-326, 322-344 and
    # 355-392 K; the upper end moves the mean by 5 to 19 K, inside the 10% bar)
    c = _condicoes()[cid]
    assert c["tg"] == pytest.approx((eixo + parede) / 2.0)
    assert c["tg_eixo"] == pytest.approx(eixo)


def test_tg_de_levaton_fica_como_medido():
    # Levaton et al. use their measured T_g in their own 0D model (2020, table VIII)
    assert _condicoes()["L20_10mA"]["tg"] == pytest.approx(290.0)


def _ferreira():
    import importlib.util
    spec = importlib.util.spec_from_file_location("ferreira1984", RAIZ / "docs" / "literatura" / "dados" / "ferreira1984.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@precisa_arquivo("docs/literatura/dados/ferreira1984.py")
def test_leitura_de_ferreira_bate_com_o_total_da_fig3():
    # the sum of the levels read in figs. 1 and 2 against the total of fig. 3 of the same article
    f = _ferreira()
    for k, n in f.POPULACOES.items():
        assert sum(n) == pytest.approx(f.TOTAL_FIG3[k], rel=0.15), k
    assert all(0.14 < x < 0.18 for x in f.fracao_v2().values())


@precisa_arquivo("scripts/coluna_n2_v3.py", "docs/literatura/dados/levaton2021/fig2_medidos.csv",
                 "docs/literatura/dados/levaton2021/fig3_medidos.csv", "docs/literatura/dados/levaton2020/fig10_medidos.csv")
@pytest.mark.parametrize("cid, arquivo", [("L20_10mA", "levaton2020/fig10_medidos.csv"),
                                          ("B21_230Pa", "levaton2021/fig2_medidos.csv"),
                                          ("B21_530Pa", "levaton2021/fig3_medidos.csv")])
def test_razao_medida_usa_o_n2b_total(cid, arquivo):
    # Levaton et al. measured N2(B, v = 2) only (2020, fig. 10; 2021, p. 2 and figs. 2 and 3); the
    # measured ratio compared with the total N2(B)/N2(C) of the models is
    # [N2(B, v = 2)] / f(v = 2) / [N2(C)], with f the median and range of Ferreira et al. 1984
    sys.path[:0] = [str(RAIZ / "scripts")]
    import coluna_n2_v3 as v3
    f = sorted(_ferreira().fracao_v2().values())
    assert v3.F_V2 == round(f[len(f) // 2], 3)
    assert v3.F_V2_FAIXA == (round(f[0], 3), round(f[-1], 3))
    linhas = {l.split(",")[0]: [float(x) for x in l.split(",")[1:]]
              for l in open(RAIZ / "docs/literatura/dados" / arquivo) if not l.startswith("#")}
    (_, b, b_inf, b_sup), (_, c, c_inf, c_sup) = linhas["N2_B"], linhas["N2_C"]
    razao, (inf, sup), _ = v3.medido(cid)
    assert razao == pytest.approx(b / c / v3.F_V2, rel=1e-12)
    assert inf == pytest.approx(b_inf / c_sup / v3.F_V2_FAIXA[1], rel=1e-12)
    assert sup == pytest.approx(b_sup / c_inf / v3.F_V2_FAIXA[0], rel=1e-12)
