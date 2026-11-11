# SPDX-License-Identifier: GPL-3.0-or-later
"""ferramentas/corrige_secoes.py applied to the cross-section files as they were before the
input corrections (commit d3d9b24 of the research repository) gives, byte for byte, the files
corrected by hand and checked by test_entradas.py."""
import subprocess
import sys

import pytest

from conftest import RAIZ, precisa_arquivo

sys.path.insert(0, str(RAIZ / "ferramentas"))
import corrige_secoes as cb  # noqa: E402

ANTES = "d3d9b24"
CASOS = ("xu2025", "amonia_sun2022", "yang2022_N2", "ar_n2_o2", "n2_tese", "n2")


def _antigo(caso):
    r = subprocess.run(["git", "show", f"{ANTES}:casos/{caso}/bolsigdb.dat"], cwd=RAIZ,
                       capture_output=True, text=True)
    if r.returncode:
        pytest.skip(f"needs the git history of the research repository ({ANTES})")
    return r.stdout


@pytest.mark.parametrize("caso", CASOS)
def test_corrige_reproduz_o_arquivo_corrigido(caso):
    if not (RAIZ / "casos" / caso / "bolsigdb.dat").exists():
        pytest.skip("cross sections are not distributed")
    assert cb.corrige(_antigo(caso)) == (RAIZ / "casos" / caso / "bolsigdb.dat").read_text()


@precisa_arquivo("casos/ar_n2_o2/bolsigdb.dat")
def test_corrige_e_idempotente():
    texto = (RAIZ / "casos" / "ar_n2_o2" / "bolsigdb.dat").read_text()
    assert cb.corrige(texto) == texto


def test_corrige_bloco_sintetico():
    texto = ("EXCITATION\nN2 -> N + N(2D)\n1.2e+1/threshold energy\n-----\n0 0\n-----\n\n"
             "EXCITATION\nN2 <-> N2(A3)\n  6.170e+0 1.0d45 / threshold energy\n-----\n0 0\n-----\n")
    assert cb.corrige(texto) == ("EXCITATION\nN2 <-> N2(A3)\n  6.170e+0 3.0d0 / threshold energy\n"
                                 "-----\n0 0\n-----\n")


@precisa_arquivo("casos/ar_n2_o2/original/bolsigdb_ar_air2D.dat")
def test_monta_de_ar_n2_o2_escreve_o_banco_corrigido(tmp_path):
    sys.path.insert(0, str(RAIZ / "casos" / "ar_n2_o2"))
    texto = cb.corrige((RAIZ / "casos" / "ar_n2_o2" / "original" / "bolsigdb_ar_air2D.dat").read_text().rstrip()
                       + "\n\n" + (RAIZ / "bolsigdb.dat").read_text())
    assert texto == (RAIZ / "casos" / "ar_n2_o2" / "bolsigdb.dat").read_text()
    assert 'corrige' in (RAIZ / "casos" / "ar_n2_o2" / "monta.py").read_text()
