# SPDX-License-Identifier: GPL-3.0-or-later
"""Subset of a chemistry without an element (ferramentas/subconjunto.py): the N2
part of the N2-H2 mechanism of the ammonia, the N2 part of the N2-O2 mechanism."""
import sys

import pytest

from conftest import RAIZ, precisa_terceiros

sys.path.insert(0, str(RAIZ / "ferramentas"))
from subconjunto import elementos, subconjunto  # noqa: E402

KINET = """ELEMENTS
N  H  E
END
SPECIES
N2  H2  N2(A3)  N2(v1)  H2(v1)
N  H  NH
E  N2^+  H^-  N2H^+  NH^+
END
BOLSIG
N2   H2   N    H
DENSITY N2 += N2(v1)
DENSITY H2 += H2(v1)
END
REACTIONS
$ double precision :: kx
$ kx = 1.0d-10
E + N2 => E + N2(A3)                       ! BOLSIG N2 -> @R
    @R = N2(A3,v0-4)  N2(A3,v5-9)
E + H2 => E + H + H                        ! BOLSIG H2 -> @R
    @R = H2(B3SIG)  H2(C3PI)
N + N + @M => N2(A3) + @M                  ! 1.70d-33*@R
     @M = N2     H2
     @R = 1.0d0  2.0d0
N + H + @M => NH + @M                      ! 1.00d-33*@R
     @M = N2     H2
     @R = 1.0d0  1.0d0
N2(A3) + N2 => N2 + N2                     ! kx
N2^+ + H^- => N2 + H                       ! 2.0d-7
N2H^+ + E => N2 + H                        ! 1.0d-7
NH^+ + N2 => N2H^+ + N                     ! 6.5d-10
END
"""


def blocos(texto):
    return [l.strip() for l in texto.splitlines() if l.strip()]


def test_espécies_e_elementos_sem_h():
    novo = blocos(subconjunto(KINET, "H"))
    assert novo[:3] == ["ELEMENTS", "N  E", "END"]
    i = novo.index("SPECIES")
    especies = " ".join(novo[i + 1:novo.index("END", i)]).split()
    assert especies == ["N2", "N2(A3)", "N2(v1)", "N", "E", "N2^+"]
    i = novo.index("BOLSIG")
    assert [l.split() for l in novo[i + 1:novo.index("END", i)]] == [["N2", "N"], ["DENSITY", "N2", "+=", "N2(v1)"]]


def test_elemento_sem_espécie_atômica_sai_de_elements():
    texto = "ELEMENTS\nN  O  E\nEND\nSPECIES\nN2  O2  E\nEND\nREACTIONS\nE + O2 => E + O2  ! 1.0d-9\nEND\n"
    assert blocos(subconjunto(texto, "O"))[:3] == ["ELEMENTS", "N  E", "END"]


def test_reações_com_h_saem_e_as_outras_ficam_iguais():
    novo = subconjunto(KINET, "H")
    assert "E + H2 => E + H + H" not in novo and "H2(B3SIG)" not in novo
    assert "N + H + @M => NH + @M" not in novo
    assert "N2^+ + H^- => N2 + H" not in novo and "N2H^+ + E" not in novo
    assert "NH^+ + N2" not in novo                          # an ion: the + of the charge is not a separator
    for linha in ("E + N2 => E + N2(A3)                       ! BOLSIG N2 -> @R",
                  "    @R = N2(A3,v0-4)  N2(A3,v5-9)",
                  "N2(A3) + N2 => N2 + N2                     ! kx",
                  "$ kx = 1.0d-10"):
        assert linha in novo.splitlines()


def test_bloco_de_terceiro_corpo_perde_só_a_posição_com_h():
    linhas = subconjunto(KINET, "H").splitlines()
    i = linhas.index("N + N + @M => N2(A3) + @M                  ! 1.70d-33*@R")
    assert linhas[i + 1].split() == ["@M", "=", "N2"]
    assert linhas[i + 2].split() == ["@R", "=", "1.0d0"]   # the value paired with N2, not with H2


def test_elemento_que_não_existe_é_recusado():
    with pytest.raises(ValueError, match="^no species of O in the chemistry$"):
        subconjunto(KINET, "O")


@precisa_terceiros
def test_subconjunto_de_n2_da_amônia_passa_no_pré_processador(tmp_path):
    import caso
    (tmp_path / "kinet.inp").write_text(subconjunto((RAIZ / "casos" / "xu2025" / "kinet.inp").read_text(), "H"))
    caso.gera_zdplaskin(tmp_path)
    texto = (tmp_path / "zdplaskin_m.F90").read_text()
    assert "NH3" not in texto and "N2(A3)" in texto


# ---------------------------------------------------------------- edges

def test_terceiro_corpo_só_de_h_tira_a_reação():
    texto = KINET.replace("N + N + @M => N2(A3) + @M                  ! 1.70d-33*@R\n     @M = N2     H2\n"
                          "     @R = 1.0d0  2.0d0", "N + N + @M => N2(A3) + @M                  ! 1.70d-33*@R\n"
                          "     @M = H2\n     @R = 2.0d0")
    assert "N + N + @M" in texto and "N + N + @M" not in subconjunto(texto, "H")


def test_lista_de_outro_tamanho_fica_inteira():
    # a list whose length differs from the longest one (a single factor for all) is not cut
    texto = KINET.replace("     @R = 1.0d0  2.0d0", "     @R = 1.0d0")
    linhas = subconjunto(texto, "H").splitlines()
    i = linhas.index("N + N + @M => N2(A3) + @M                  ! 1.70d-33*@R")
    assert linhas[i + 1].split() == ["@M", "=", "N2"] and linhas[i + 2].split() == ["@R", "=", "1.0d0"]


def test_comentários_e_linhas_em_branco_ficam():
    texto = KINET.replace("SPECIES\n", "SPECIES\n# the species\n\n")
    novo = subconjunto(texto, "H")
    assert "# the species" in novo.splitlines()


def test_espécie_carregada_com_o_elemento_sai():
    novo = subconjunto(KINET, "H")
    especies = novo.split("SPECIES\n")[1].split("\nEND")[0].split()
    assert "H^-" not in especies and "N2H^+" not in especies and "NH^+" not in especies


def test_resultado_é_idempotente():
    uma = subconjunto(KINET, "H")
    with pytest.raises(ValueError, match="no species of H"):
        subconjunto(uma, "H")


def test_retrato_do_subconjunto_sem_h():
    assert subconjunto(KINET, "H") == (RAIZ / "dbd1d" / "testes" / "dados" / "subconjunto_sem_h.inp").read_text()


def test_linha_de_comando(tmp_path):
    import subprocess
    (tmp_path / "base.inp").write_text(KINET)
    r = subprocess.run([sys.executable, str(RAIZ / "ferramentas" / "subconjunto.py"), str(tmp_path / "base.inp"),
                        str(tmp_path / "novo.inp"), "H"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "novo.inp").read_text() == subconjunto(KINET, "H")


def test_linha_de_comando_sem_argumentos_explica_o_uso():
    import subprocess
    r = subprocess.run([sys.executable, str(RAIZ / "ferramentas" / "subconjunto.py")], capture_output=True, text=True)
    assert r.returncode != 0
    assert r.stderr.strip() == "Usage: python3 ferramentas/subconjunto.py ORIGIN/kinet.inp DESTINATION/kinet.inp ELEMENT"



# ---------------------------------------------------------------- found by mutation testing

def test_elementos_de_um_nome():
    assert elementos("N2(A3)") == {"N"}                    # the state in parentheses is not an element
    assert elementos("N2(a'1)") == {"N"} and elementos("H2(B3SIG)") == {"H"}
    assert elementos("NH^+") == {"N", "H"} and elementos("N2H^+") == {"N", "H"}
    assert elementos("E") == set() and elementos("e") == set()


def test_comentário_no_fim_da_linha_de_espécies():
    texto = KINET.replace("N2  H2  N2(A3)  N2(v1)  H2(v1)", "N2  H2  N2(A3)  N2(v1)  H2(v1)  # H2 and its levels")
    especies = subconjunto(texto, "H").split("SPECIES\n")[1].split("\nEND")[0].split()
    assert especies == ["N2", "N2(A3)", "N2(v1)", "N", "E", "N2^+"]


def test_espécie_citada_no_comentário_da_reação_não_conta():
    texto = KINET.replace("N2(A3) + N2 => N2 + N2                     ! kx",
                          "N2(A3) + N2 => N2 + N2                     ! kx + H2")
    assert "N2(A3) + N2 => N2 + N2                     ! kx + H2" in subconjunto(texto, "H").splitlines()


def test_linha_de_comentário_com_seta_fica():
    texto = KINET.replace("REACTIONS\n", "REACTIONS\n# N + H => NH, see below\n")
    texto = "# kinetics of N + H => NH and N2\n" + texto
    novo = subconjunto(texto, "H").splitlines()
    assert "# N + H => NH, see below" in novo and "# kinetics of N + H => NH and N2" in novo


def test_reações_depois_de_uma_que_sai_inteira_continuam():
    texto = KINET.replace("     @M = N2     H2\n     @R = 1.0d0  2.0d0", "     @M = H2\n     @R = 2.0d0")
    novo = subconjunto(texto, "H")
    assert "N2(A3) + N2 => N2 + N2" in novo and novo.rstrip().endswith("END")
