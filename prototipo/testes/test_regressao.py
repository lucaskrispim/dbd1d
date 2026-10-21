# SPDX-License-Identifier: GPL-3.0-or-later
"""Regressão do protótipo contra referências gravadas (G34) e determinismo,
como make regressao e N5 no Fortran. Dois casos curtos (alvos do Makefile):
com a equação de energia e em campo local (que usa a tabela em E/N). As
referências ficam em referencias/prototipo_curto e prototipo_curto_lfa e são
refeitas com make referencia-regressao-prototipo, que grava a diferença medida
contra a anterior em mudanca.txt (commit próprio).

Tolerância: com as mesmas versões de Python, numpy e scipy da referência,
1e-9; com outras, 2e-3. Com as tolerâncias antigas do integrador (rtol e atol
1e-4) o último bit chegava a uns 4e-4 nas densidades (medido com n_e0
perturbado em 1e-15); com o atol novo (1e-10) é de 9e-5. Um gama 1% diferente
muda 1e-2: 2e-3 separa os dois."""
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
    """Argumentos do caso pela variável do Makefile."""
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
    # o caso de campo local cobre a tabela em E/N (Q3)
    assert ("--energia" in caso(variavel)) == (variavel == "CURTO_PROTOTIPO")


def test_execucao_deterministica(tmp_path):
    args = caso("CURTO_PROTOTIPO")
    roda(tmp_path / "a", args)
    roda(tmp_path / "b", args)
    for nome in ("perfil.dat", "superficie.dat"):
        assert (tmp_path / "a" / nome).read_bytes() == (tmp_path / "b" / nome).read_bytes(), nome
