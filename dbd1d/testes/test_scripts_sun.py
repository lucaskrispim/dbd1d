# SPDX-License-Identifier: GPL-3.0-or-later
"""Launch scripts of the Sun et al. 2022 runs (remoto/*_sun.sh): each group writes to its
own folder, with its own case and extra options, so that a variant never writes over the
runs of another group (docs/campanha_amonia/onda3/PLANO.md, section 9, item 2). The python
is replaced by a stub that only writes its arguments to the log."""
import os
import shutil
import stat
import subprocess

import pytest

from conftest import RAIZ

# the launch scripts of article 2 are not distributed (see INCLUI in empacotamento/monta_pacote.py)
if not (RAIZ / "remoto/pulsos_sun.sh").exists():
    pytest.skip("needs remoto/pulsos_sun.sh, which is not distributed", allow_module_level=True)

pytestmark = pytest.mark.skipif(shutil.which("bash") is None or not os.path.exists("/usr/bin/time"),
                                reason="needs bash and /usr/bin/time")


def _stub(tmp_path):
    py = tmp_path / "py"
    py.write_text("#!/bin/bash\necho \"$@\"\n")
    py.chmod(py.stat().st_mode | stat.S_IEXEC)
    return py


def _roda(script, args, env):
    return subprocess.run(["bash", str(RAIZ / "remoto" / script), *args], env={**os.environ, **env},
                          capture_output=True, text=True, timeout=120, check=True)


def test_pulsos_sun_grava_na_pasta_e_com_o_caso_pedidos(tmp_path):
    py = _stub(tmp_path)
    saida = tmp_path / "vv"
    _roda("pulsos_sun.sh", ["2", str(py), "300", "573"],
          {"DIR": str(saida), "CASO": "casos/amonia_sun2022_vv", "SUFIXO": "_vv", "EXTRA": "--te-floor-tg"})
    log = (saida / "T573_vv.log").read_text()
    assert f"{saida}/T573_vv " in log
    assert "--caso casos/amonia_sun2022_vv " in log
    assert "--te-floor-tg" in log
    assert not (RAIZ / "execucoes" / "pulsos_sun" / "T573_vv").exists()


def test_pulsos_sun_sem_variaveis_mantem_o_antigo(tmp_path):
    py = _stub(tmp_path)
    saida = tmp_path / "base"
    _roda("pulsos_sun.sh", ["2", str(py), "300", "773"], {"DIR": str(saida)})
    log = (saida / "T773.log").read_text()
    assert "--caso casos/amonia_sun2022 " in log and "--te-floor-tg" not in log


@pytest.mark.parametrize("script,args,arquivo", [
    ("sensibilidade_sun.sh", ["1"], "gH1e-2.log"),
    ("sensibilidade_sun.sh", ["1"], "hongW98.log"),
    ("sensibilidade_sun.sh", ["1"], "base.log"),
    ("pacote_revisao_sun.sh", ["1"], "T773_s10.log"),
    ("campanha_amonia.sh", ["20"], "S_n150.log"),
    ("convergencia_pulso_sun.sh", ["1"], "P_n150.log"),
])
def test_scripts_aceitam_pasta_e_opcoes_extras(tmp_path, script, args, arquivo):
    py = _stub(tmp_path)
    saida = tmp_path / "g"
    extra = ["P_n150:150"] if script.startswith("convergencia") else []
    _roda(script, [*args, str(py), *extra], {"DIR": str(saida), "EXTRA": "--te-floor-tg"})
    log = (saida / arquivo).read_text()
    assert "--te-floor-tg" in log and str(saida) in log
    if arquivo == "hongW98.log":
        assert "--caso casos/amonia_sun2022_hongW98 " in log


@pytest.mark.parametrize("script", ["periodo_sun.sh", "pulso_fino_sun.sh"])
def test_retomadas_aceitam_pasta_e_opcoes_extras(tmp_path, script):
    py = _stub(tmp_path)
    origem = tmp_path / "origem"
    origem.mkdir()
    (origem / "parametros.json").write_text('{"argumentos": {"t_gas": 573.0}}')
    saida = tmp_path / "g"
    _roda(script, [str(py), f"T573_p13:{origem}:13"], {"DIR": str(saida), "EXTRA": "--te-floor-tg"})
    log = (saida / "T573_p13_pulso.log").read_text()
    assert "--te-floor-tg" in log and f"--retoma {origem} " in log
