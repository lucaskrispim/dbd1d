# SPDX-License-Identifier: GPL-3.0-or-later
"""Regression of the prototype against recorded references (G34) and determinism,
as make regressao and N5 in the Fortran code. Three short cases (Makefile targets):
with the energy equation and in local field (which uses the table in E/N), both before
the breakdown, and with the breakdown and gas heating up to half a period. The
references are in referencias/prototipo_curto, prototipo_curto_lfa and
prototipo_curto_ruptura and are rebuilt with make referencia-regressao-prototipo,
which writes the difference measured against the previous one in mudanca.txt (own
commit).

Tolerance: with the same versions of Python, numpy and scipy as the reference,
1e-9; with others, 2e-3. With the old tolerances of the integrator (rtol and atol
1e-4) the last bit reached about 4e-4 in the densities (measured with n_e0
perturbed by 1e-15); with the new atol (1e-10) it is 9e-5. A gamma 1% different
changes 1e-2: 2e-3 separates the two. Through the breakdown the last bit grows, and
pointwise most where a quantity crosses zero: n_e0 perturbed by 1e-15 to 1e-13 changes the
field of the case with breakdown by 5e-2 to 1.1e-1 at some point (4/11/2026). With other
versions that case is compared relative to the largest value of each column
(compara.py --relative-to-max), where the same perturbations give 1.9e-3 to 4.0e-3 and a
gamma 1% different 8.9e-3, at 2e-2: it fails on a change that moves the result by
about 2%."""
import json
import platform
import subprocess
import sys

import numpy as np
import pytest
import scipy

from conftest import PROTOTIPO, RAIZ

AMBIENTE = {"OPENBLAS_NUM_THREADS": "1", "PATH": ""}


# the two short cases, as the variables of the same name in the Makefile of the
# research repository (test_argumentos_iguais_aos_do_makefile keeps them equal)
ARGUMENTOS = {
    "CURTO_PROTOTIPO": ["--v-peak", "5e3", "--cells", "20", "--clustering", "1.5", "--thickness", "1.75e-3",
                        "--energy", "--dt-output", "2e-9", "--ne0", "1e9", "--neutral", "--t-end", "8e-9"],
    "CURTO_PROTOTIPO_LFA": ["--v-peak", "5e3", "--cells", "20", "--clustering", "1.5", "--thickness", "1.75e-3",
                            "--dt-output", "2e-9", "--ne0", "1e9", "--neutral", "--t-end", "8e-9"],
    # breakdown (n_e from 1e9 to 1.2e13 cm-3 between 8 and 28 ns, E/N up to 235 Td), secondary
    # emission and gas heating, up to half a period
    "CURTO_PROTOTIPO_RUPTURA": ["--v-peak", "5e3", "--cells", "20", "--clustering", "1.5", "--thickness",
                                "1.75e-3", "--energy", "--heating", "--dt-output", "2e-9", "--ne0", "1e9",
                                "--neutral", "--t-end", "5e-8"],
}


def caso(nome):
    return ARGUMENTOS[nome]


@pytest.mark.skipif(not (RAIZ / "Makefile").exists(), reason="Makefile of the research repository, not distributed")
def test_argumentos_iguais_aos_do_makefile():
    # the Makefile writes the options with their Portuguese names (internal name with - for _)
    sys.path.insert(0, str(PROTOTIPO))
    from simula import NOMES
    interno = {v: k for k, v in NOMES.items()}
    texto = (RAIZ / "Makefile").read_text().replace("\\\n", " ")
    for nome, args in ARGUMENTOS.items():
        linha = next(l for l in texto.splitlines() if l.startswith(nome + " ") or l.startswith(nome + "="))
        portugues = ["--" + interno.get(x[2:], x[2:]).replace("_", "-") if x.startswith("--") else x for x in args]
        assert linha.split("=", 1)[1].split() == portugues, nome


CASOS = [("CURTO_PROTOTIPO", "prototipo_curto"), ("CURTO_PROTOTIPO_LFA", "prototipo_curto_lfa"),
         ("CURTO_PROTOTIPO_RUPTURA", "prototipo_curto_ruptura")]


def roda(destino, args):
    subprocess.run([sys.executable, str(PROTOTIPO / "simula.py"), str(destino), *args],
                   check=True, capture_output=True, env=AMBIENTE)


# comparison with versions of Python, numpy or scipy other than those of the reference:
# tolerance and options of compara.py
OUTRAS_VERSOES = {"prototipo_curto_ruptura": (2e-2, ["--relative-to-max"])}


def tolerancia(referencia):
    """(tolerance, extra options of compara.py, versions that differ from the reference)."""
    p = json.loads((referencia / "parametros.json").read_text())
    agora = {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__}
    diferentes = {k: (p[k], v) for k, v in agora.items() if p[k] != v}
    if not diferentes:
        return 1e-9, [], diferentes
    return (*OUTRAS_VERSOES.get(referencia.name, (2e-3, [])), diferentes)


@pytest.mark.parametrize("variavel, referencia", CASOS)
def test_caso_curto_igual_a_referencia(tmp_path, variavel, referencia):
    ref = RAIZ / "referencias" / referencia
    roda(tmp_path / "c", caso(variavel))
    tol, opcoes, diferentes = tolerancia(ref)
    r = subprocess.run([sys.executable, str(RAIZ / "scripts" / "compara.py"), str(ref), str(tmp_path / "c"),
                        "--tol", str(tol), "--estrito", *opcoes], capture_output=True, text=True, env=AMBIENTE)
    assert r.returncode == 0, f"{r.stdout}\nversions different from the reference: {diferentes}"
    # the local field case covers the table in E/N (Q3)
    assert ("--energy" in caso(variavel)) == (variavel != "CURTO_PROTOTIPO_LFA")


def test_execucao_deterministica(tmp_path):
    args = caso("CURTO_PROTOTIPO")
    roda(tmp_path / "a", args)
    roda(tmp_path / "b", args)
    for nome in ("perfil.dat", "superficie.dat"):
        assert (tmp_path / "a" / nome).read_bytes() == (tmp_path / "b" / nome).read_bytes(), nome


def test_caso_com_ruptura_com_outras_versoes(tmp_path, monkeypatch):
    # the comparison used with other library versions (relative to the column maximum) sees a
    # secondary emission 1% higher: 8.9e-3 measured, above a fifth of its tolerance
    ref = RAIZ / "referencias" / "prototipo_curto_ruptura"
    tol, opcoes = OUTRAS_VERSOES["prototipo_curto_ruptura"]
    args = caso("CURTO_PROTOTIPO_RUPTURA")
    roda(tmp_path / "c", args + ["--gamma", "0.0505"])
    r = subprocess.run([sys.executable, str(RAIZ / "scripts" / "compara.py"), str(ref), str(tmp_path / "c"),
                        "--tol", str(tol / 5), "--estrito", *opcoes], capture_output=True, text=True, env=AMBIENTE)
    assert r.returncode == 1, r.stdout
