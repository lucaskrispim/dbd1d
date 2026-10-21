# SPDX-License-Identifier: GPL-3.0-or-later
"""C1: collapse next to the cathode in local field (ACHADOS, "Colapso junto ao
cátodo em campo local"). In the 5 kV case with quartz and heating, with 100
cells, the plasma column in local field advances to the cathode and the wall cell
blows up (n_e/N above 1). Restarting the same state at 65 ns, with the column
still 170 µm from the wall, the collapse also happens without secondary emission,
and with the energy equation it does not happen: the column stops more than
20 µm from the cathode. If the local field stops collapsing, the record must be
revised; if the energy model collapses, the production model lost its validity
in this case."""
import subprocess
import sys

import numpy as np
import pytest

from conftest import PROTOTIPO, RAIZ

sys.path.insert(0, str(RAIZ / "scripts"))
from resumo import blocos  # noqa: E402

CASO = ["--v-pico", "5e3", "--espessura", "1.75e-3", "--n-celulas", "100", "--dt-saida", "1e-9",
        "--aquecimento", "--eps-gas", "1.000516"]
N_GAS = 101325.0 / (1.380649e-23 * 300.0) * 1e-6     # cm-3, gas held at 300 K
AVISO = "ionization degree above 1e-3"


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
    """Largest n_e/N over all cells and recorded instants and its position (m)."""
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
        assert f"{AVISO} at t" in r.stdout and " node 2 (" in r.stdout     # in the wall cell
        assert x > 0.1
    else:
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
        assert x < 1e-5 and pos > 20e-6
