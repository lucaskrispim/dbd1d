# SPDX-License-Identifier: GPL-3.0-or-later
"""Summary of the self-consistent coupled solutions (scripts/coluna_n2_raiz_resumo.py)."""
import json

import pytest

from conftest import RAIZ, precisa_arquivo

precisa_script = precisa_arquivo("scripts/coluna_n2_raiz_resumo.py")


def modulo():
    import sys
    sys.path.insert(0, str(RAIZ / "scripts"))
    import coluna_n2_raiz_resumo
    return coluna_n2_raiz_resumo


@precisa_script
def test_frações_das_etapas_somam_um():
    # ladder 110, targets 100, shifted 98, full 95: fall 15 = 10 + 2 + 3
    f = modulo().fracoes_etapas(110.0, 100.0, 98.0, 95.0)
    assert f == pytest.approx((10 / 15, 2 / 15, 3 / 15))


@precisa_script
def test_frações_sem_etapa_intermediária_são_nulas():
    assert modulo().fracoes_etapas(110.0, 100.0, None, 95.0) is None


def escreve(pasta, nome, convergiu, en, frac=0.3):
    d = {"convergiu": convergiu, "en": en, "tv": 3000.0, "dis": 1e-3,
         "pontos": [{"en": en + 1, "fracao_v_maior_0": 0.0},
                    {"en": en, "fracao_v_maior_0": frac, "fracao_v_maior_10": 1e-6},
                    {"en": en + 0.01, "fracao_v_maior_0": 0.99}]}
    (pasta / f"{nome}_raiz.json").write_text(json.dumps(d))


@precisa_script
def test_linha_lê_a_solução_e_ignora_cadeia_sem_convergência(tmp_path):
    m = modulo()
    escreve(tmp_path, "L20_50mA", True, 92.0, frac=0.31)
    escreve(tmp_path, "L20_50mA_alvo", True, 96.0)
    escreve(tmp_path, "L20_50mA_deslocado", None, 94.0)
    linha = m.linha(tmp_path, "L20_50mA", "base", 113.0)
    assert linha["en_completo"] == 92.0 and linha["en_alvo"] == 96.0
    assert linha["en_deslocado"] is None and linha["fracao_alvo"] is None
    # the populations of the point at the root, not of the first or the last one evaluated
    assert linha["fracao_v_maior_0"] == 0.31
    assert linha["desvio"] == pytest.approx(92.0 / 90.0 - 1)
    assert m.linha(tmp_path, "G95_15mA", "base", 108.5)["en_completo"] is None


@precisa_script
def test_nome_do_arquivo_por_variante_e_modo():
    m = modulo()
    assert m.nome("B21_530Pa", "base", "completo") == "B21_530Pa"
    assert m.nome("B21_530Pa", "vib_gordiets", "alvo") == "B21_530Pa_vib_gordiets_alvo"


@precisa_script
def test_nome_com_ev():
    m = modulo()
    assert m.nome("B21_530Pa", "base", "completo", ev=True) == "B21_530Pa_completo_ev"
    assert m.nome("G95_80mA", "vib_gordiets", "alvo", ev=True) == "G95_80mA_vib_gordiets_alvo_ev"


@precisa_script
def test_linha_com_ev_lê_as_cadeias_ev(tmp_path):
    m = modulo()
    escreve(tmp_path, "L20_50mA", True, 92.0)
    escreve(tmp_path, "L20_50mA_completo_ev", True, 99.0)
    assert m.linha(tmp_path, "L20_50mA", "base", 113.0, ev=True)["en_completo"] == 99.0
