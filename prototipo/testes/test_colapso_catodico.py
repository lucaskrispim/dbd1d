# SPDX-License-Identifier: GPL-3.0-or-later
"""C1: colapso junto ao cátodo em campo local (ACHADOS, "Colapso junto ao
cátodo em campo local"). No caso de 5 kV com quartzo e aquecimento, com 100
células, a coluna de plasma em campo local avança até o cátodo e a célula da
parede dispara (n_e/N acima de 1). Retomando o mesmo estado aos 65 ns, com a
coluna ainda a 170 µm da parede, o colapso acontece também sem emissão
secundária, e com a equação de energia não acontece: a coluna para a mais de
20 µm do cátodo. Se o campo local deixar de colapsar, o registro precisa ser
revisto; se a energia colapsar, o modelo de produção perdeu a validade nesse
caso."""
import subprocess
import sys

import numpy as np
import pytest

from conftest import PROTOTIPO, RAIZ

sys.path.insert(0, str(RAIZ / "scripts"))
from resumo import blocos  # noqa: E402

CASO = ["--v-pico", "5e3", "--espessura", "1.75e-3", "--n-celulas", "100", "--dt-saida", "1e-9",
        "--aquecimento", "--eps-gas", "1.000516"]
N_GAS = 101325.0 / (1.380649e-23 * 300.0) * 1e-6     # cm-3, gás preso a 300 K
AVISO = "grau de ionizacao acima de 1e-3"


def roda(destino, *extra):
    return subprocess.run([sys.executable, str(PROTOTIPO / "simula.py"), str(destino), *CASO, *extra],
                          capture_output=True, text=True, env={"OPENBLAS_NUM_THREADS": "1", "PATH": ""})


@pytest.fixture(scope="module")
def origem(tmp_path_factory):
    d = tmp_path_factory.mktemp("campo_local")
    r = roda(d, "--t-end", "6.5e-8")
    assert r.returncode == 0 and AVISO not in r.stdout, r.stdout[-2000:] + r.stderr[-2000:]
    return d


def ionizacao_maxima(d):
    """Maior n_e/N em todas as células e instantes gravados e a posição (m) dele."""
    melhor = (0.0, 0.0)
    for _, nomes, linhas in blocos(d / "perfil.dat"):
        a = np.array(linhas[1:-1])
        i = int(np.argmax(a[:, nomes.index("E")]))
        melhor = max(melhor, (a[i, nomes.index("E")] / N_GAS, a[i, 0]))
    return melhor


@pytest.mark.parametrize("extra, colapsa", [([], True), (["--gama", "0"], True), (["--energia"], False)],
                         ids=["campo_local", "sem_emissao_secundaria", "energia"])
def test_colapso_catodico(tmp_path, origem, extra, colapsa):
    r = roda(tmp_path, "--t-end", "7.5e-8", "--retoma", str(origem), "6.5e-8", *extra)
    assert (AVISO in r.stdout) == colapsa, r.stdout[-2000:]
    x, pos = ionizacao_maxima(tmp_path)
    if colapsa:
        assert f"{AVISO} em t" in r.stdout and " no 2 (" in r.stdout     # na célula da parede
        assert x > 0.1
    else:
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
        assert x < 1e-5 and pos > 20e-6
