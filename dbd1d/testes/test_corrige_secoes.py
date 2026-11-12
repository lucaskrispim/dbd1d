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


@precisa_arquivo("casos/ar_n2_o2/original/bolsigdb_ar_air2D.dat", "casos/ar_n2_o2/original/kinet_ar_air2D.inp",
                 "bolsigdb.dat")
def test_monta_de_ar_n2_o2_escreve_o_banco_corrigido(tmp_path):
    # ERROS.md T8: monta.py run on a copy of the tree writes the corrected bank of the case
    import shutil
    (tmp_path / "casos").mkdir()
    shutil.copytree(RAIZ / "casos" / "ar_n2_o2" / "original", tmp_path / "casos" / "ar_n2_o2" / "original")
    shutil.copy(RAIZ / "casos" / "ar_n2_o2" / "monta.py", tmp_path / "casos" / "ar_n2_o2")
    shutil.copytree(RAIZ / "ferramentas", tmp_path / "ferramentas", ignore=shutil.ignore_patterns("__pycache__"))
    for f in ("kinet.inp", "bolsigdb.dat"):
        shutil.copy(RAIZ / f, tmp_path / f)
    subprocess.run([sys.executable, str(tmp_path / "casos" / "ar_n2_o2" / "monta.py")], check=True,
                   capture_output=True)
    assert ((tmp_path / "casos" / "ar_n2_o2" / "bolsigdb.dat").read_text()
            == (RAIZ / "casos" / "ar_n2_o2" / "bolsigdb.dat").read_text())


def test_razao_zero_fica_zero():
    # an exact zero switches the superelastic inverse off on purpose (casos ar_n2, n2): kept
    texto = "EXCITATION\nN2 <-> N2(V1)\n  2.910e-1 0.0d0 / threshold energy\n-----\n0 0\n-----\n"
    assert cb.corrige(texto) == texto


def test_corrige_tira_a_captura_dissociativa_repetida():
    # the ZDPlasKin N2-O2 file has two O2 -> O2^- blocks; the "Two body attachment" one carries
    # the dissociative attachment cross section of O2 -> O^-+O (onset 4.4 eV), loaded twice by
    # BOLSIG+ (ERROS.md T3-1); the three-body block stays
    bloco = "ATTACHMENT\n{p}\nCOMMENT: {c}\n-----\n{x}\n-----\n\n"
    texto = (bloco.format(p="O2 -> O^-+O", c="2-body attachment", x="4.4 0\n5 1e-22")
             + bloco.format(p="O2 -> O2^-", c="Three body attachment (energy loss taken to be kT @ 300K)", x="0 1e-21")
             + bloco.format(p="O2 -> O2^-", c="Two body attachment (energy loss taken to be kT @ 300K)", x="4.4 0\n5 1e-22"))
    saida = cb.corrige(texto)
    assert "Two body attachment" not in saida
    assert "Three body attachment" in saida and "O2 -> O^-+O" in saida
    assert cb.corrige(saida) == saida
