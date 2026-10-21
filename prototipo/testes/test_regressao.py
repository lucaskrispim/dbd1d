# SPDX-License-Identifier: GPL-3.0-or-later
"""Regression of the prototype against recorded references (G34) and determinism,
as make regressao and N5 in the Fortran code. Two short cases (Makefile targets):
with the energy equation and in local field (which uses the table in E/N). The
references are in referencias/prototipo_curto and prototipo_curto_lfa and are
rebuilt with make referencia-regressao-prototipo, which writes the difference
measured against the previous one in mudanca.txt (own commit).

Tolerance: with the same versions of Python, numpy and scipy as the reference,
1e-9; with others, 2e-3. With the old tolerances of the integrator (rtol and atol
1e-4) the last bit reached about 4e-4 in the densities (measured with n_e0
perturbed by 1e-15); with the new atol (1e-10) it is 9e-5. A gamma 1% different
changes 1e-2: 2e-3 separates the two."""
import json
import platform
import subprocess
import sys

import numpy as np
import pytest
import scipy

from conftest import PROTOTIPO, RAIZ

AMBIENTE = {"OPENBLAS_NUM_THREADS": "1", "PATH": ""}


def caso(nome):
    """Arguments of the case from the Makefile variable."""
    texto = (RAIZ / "Makefile").read_text().replace("\\\n", " ")
    linha = next(l for l in texto.splitlines() if l.startswith(nome + " ") or l.startswith(nome + "="))
    return linha.split("=", 1)[1].split()


CASOS = [("CURTO_PROTOTIPO", "prototipo_curto"), ("CURTO_PROTOTIPO_LFA", "prototipo_curto_lfa")]


def roda(destino, args):
    subprocess.run([sys.executable, str(PROTOTIPO / "simula.py"), str(destino), *args],
                   check=True, capture_output=True, env=AMBIENTE)


def tolerancia(referencia):
    p = json.loads((referencia / "parametros.json").read_text())
    agora = {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__}
    diferentes = {k: (p[k], v) for k, v in agora.items() if p[k] != v}
    return (1e-9 if not diferentes else 2e-3), diferentes


@pytest.mark.parametrize("variavel, referencia", CASOS)
def test_caso_curto_igual_a_referencia(tmp_path, variavel, referencia):
    ref = RAIZ / "referencias" / referencia
    roda(tmp_path / "c", caso(variavel))
    tol, diferentes = tolerancia(ref)
    r = subprocess.run([sys.executable, str(RAIZ / "scripts" / "compara.py"), str(ref), str(tmp_path / "c"),
                        "--tol", str(tol), "--estrito"], capture_output=True, text=True, env=AMBIENTE)
    assert r.returncode == 0, f"{r.stdout}\nversões diferentes da referência: {diferentes}"
    # the local field case covers the table in E/N (Q3)
    assert ("--energia" in caso(variavel)) == (variavel == "CURTO_PROTOTIPO")


def test_execucao_deterministica(tmp_path):
    args = caso("CURTO_PROTOTIPO")
    roda(tmp_path / "a", args)
    roda(tmp_path / "b", args)
    for nome in ("perfil.dat", "superficie.dat"):
        assert (tmp_path / "a" / nome).read_bytes() == (tmp_path / "b" / nome).read_bytes(), nome
