# SPDX-License-Identifier: GPL-3.0-or-later
"""Regression of the results published with the article on the N2 positive column
(research repository only: the chemistries are of other authors): rows of
docs/coluna_positiva_n2 recomputed with the scripts that wrote them, so that a change of
coluna.py, of the chemistries or of the scripts that moves a published number fails here.
The coupled roots (pendencias/raiz) take hours and are not recomputed; their summary is
checked against the root files in test_raiz_resumo.py. The published rows are those of the reruns
of wave 3 of the chain audit (docs/coluna_positiva_n2/onda3, inputs corrected on 11/11/2026); the
rows before the corrections stay in docs/coluna_positiva_n2 for the comparison."""
import csv
import sys

import numpy as np
import pytest

from conftest import RAIZ, precisa_arquivo

D = RAIZ / "docs" / "coluna_positiva_n2" / "onda3"
pytestmark = [precisa_arquivo("scripts/coluna_n2.py", "scripts/coluna_n2_en.py", "scripts/coluna_n2_lote.py",
                              "casos/n2_amonia/quimica.json", "casos/n2_tese/quimica.json",
                              "casos/n2_exemplo/quimica.json", "casos/n2_amonia_v45/quimica.json"),
              pytest.mark.lento]


def _scripts():
    sys.path[:0] = [str(RAIZ / "scripts")]
    import coluna_n2
    import coluna_n2_en
    import coluna_n2_lote
    return coluna_n2, coluna_n2_en, coluna_n2_lote


def _linha(arquivo, **chaves):
    linhas = [l for l in csv.DictReader(open(arquivo)) if all(l[k] == v for k, v in chaves.items())]
    assert len(linhas) == 1, (arquivo, chaves)
    return linhas[0]


def _condicao(cid):
    coluna_n2, _, _ = _scripts()
    return next(c for c in coluna_n2.condicoes() if c["id"] == cid)


@pytest.mark.parametrize("cid, quimica", [
    ("L20_50mA", "n2_amonia"),
    ("G95_80mA", "n2_tese"),
    ("B21_530Pa", "n2_exemplo")])
def test_campo_sustentado_da_etapa5(cid, quimica):
    _, coluna_n2_en, _ = _scripts()
    publicado = float(_linha(D / "etapa5" / "en.csv", id=cid, quimica=quimica)["en"])
    assert coluna_n2_en.en(quimica, _condicao(cid)) == pytest.approx(publicado, rel=1e-6)


@pytest.mark.parametrize("cid, quimica", [
    ("B21_530Pa", "n2_amonia"),
    ("L20_20mA", "n2_tese")])
def test_tv_e_dissociacao_da_etapa4(cid, quimica):
    coluna_n2, _, _ = _scripts()
    cond = _condicao(cid)
    r = coluna_n2.roda(quimica, cond, cond["en"], cond["ne"], cond["tg"])
    publicado = _linha(D / "etapa4" / "resultados.csv", id=cid, quimica=quimica)
    assert r["tv"] == pytest.approx(float(publicado["tv"]), rel=1e-5)
    assert r["dis"] == pytest.approx(float(publicado["dis"]), rel=1e-5)


def test_escada_ate_v45(tmp_path, monkeypatch):
    # falta/escada_v45.csv, chemistry A with the ladder, narrow tube at 230 Pa and 60 mA
    _, _, coluna_n2_lote = _scripts()
    arquivo = tmp_path / "linha.csv"
    monkeypatch.setattr(sys, "argv", ["coluna_n2_lote.py", "n2_amonia_v45", "B21_230Pa", "base", str(arquivo)])
    coluna_n2_lote.main()
    novo = next(csv.DictReader(open(arquivo)))
    publicado = _linha(D / "falta" / "escada_v45.csv", caso="n2_amonia_v45", id="B21_230Pa", variante="base")
    for k in ("tv", "dis", "en"):
        # the published file has 6 significant digits
        assert float(novo[k]) == pytest.approx(float(publicado[k]), rel=1e-5), k
    assert np.isfinite(float(novo["tv"]))
