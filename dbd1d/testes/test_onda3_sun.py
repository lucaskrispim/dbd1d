# SPDX-License-Identifier: GPL-3.0-or-later
"""Runs of Sun et al. 2022 read by the post-processing of article 2 come from the wave 3 folder
(scripts/onda3_sun.py), never from the old runs that still sit in execucoes/ under the same
names (docs/campanha_amonia/onda3/PLANO.md, section 9, item 4)."""
import json
import re
import sys

import pytest

from conftest import RAIZ

# the scripts of article 2 are not distributed (see INCLUI in empacotamento/monta_pacote.py)
if not (RAIZ / "scripts/onda3_sun.py").exists():
    pytest.skip("needs scripts/onda3_sun.py, which is not distributed", allow_module_level=True)

sys.path.insert(0, str(RAIZ / "scripts"))
import onda3_sun  # noqa: E402


def _rodada(pasta, versao):
    pasta.mkdir(parents=True)
    reg = {"argumentos": {}}
    if versao:
        reg["versao"] = versao
    (pasta / "parametros.json").write_text(json.dumps(reg))
    return pasta


def test_le_a_pasta_da_variavel_e_recusa_rodada_antiga(tmp_path, monkeypatch):
    monkeypatch.setenv("SUN_ONDA3", str(tmp_path))
    _rodada(tmp_path / "periodo_sun" / "T573_p13_pulso", "1.6.0")
    assert onda3_sun.rodada("periodo_sun", "T573_p13_pulso") == tmp_path / "periodo_sun" / "T573_p13_pulso"
    _rodada(tmp_path / "periodo_sun" / "antiga", "1.5.0")
    _rodada(tmp_path / "periodo_sun" / "sem_versao", None)
    for nome in ("antiga", "sem_versao", "falta"):
        with pytest.raises(onda3_sun.RodadaAntiga):
            onda3_sun.rodada("periodo_sun", nome)


@pytest.mark.parametrize("script", ["figuras_amonia.py", "linha_base_0d_sun.py", "posdescarga_sun.py"])
def test_scripts_do_artigo_2_nao_leem_execucoes_direto(script):
    codigo = (RAIZ / "scripts" / script).read_text()
    assert not re.search(r'RAIZ\s*/\s*"execucoes"', codigo), script
