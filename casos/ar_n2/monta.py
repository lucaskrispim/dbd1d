# SPDX-License-Identifier: GPL-3.0-or-later
"""Case Ar:N2 90:10 assembled from the two mechanisms of the repository, to test
the tool on a new mixture: kinet.inp joins the species, the BOLSIG+ processes and
the reactions of argon (root) and N2 (casos/n2), plus three cross reactions;
bolsigdb.dat joins the two cross-section databases.

Cross reactions, with typical coefficients from the literature (here they serve
the test of the tool, not a study of Ar:N2):
  Ar^+ + N2 => N2^+ + Ar        1.0e-11 cm3/s (charge exchange)
  Ar* + N2 => Ar + N2(C)        3.0e-11 cm3/s (energy transfer)
  Ar2^+ + N2 => N2^+ + Ar + Ar  1.0e-11 cm3/s

Usage: python3 casos/ar_n2/monta.py   (writes kinet.inp, bolsigdb.dat and composicao.txt here)
"""
import pathlib
import re

AQUI = pathlib.Path(__file__).resolve().parent
RAIZ = AQUI.parent.parent


def bloco(texto, nome):
    """Lines between NAME and the next END."""
    m = re.search(rf"^{nome}\s*\n(.*?)^END", texto, re.S | re.M)
    return [l for l in m.group(1).splitlines() if l.strip() and not l.strip().startswith("#")]


def reacoes(texto):
    m = re.search(r"^REACTIONS\s*\n(.*)^END", texto, re.S | re.M)
    return m.group(1).rstrip()


ar = (RAIZ / "kinet.inp").read_text()
n2 = (RAIZ / "casos" / "n2" / "kinet.inp").read_text()
especies = " ".join(bloco(ar, "SPECIES") + bloco(n2, "SPECIES")).split()
especies = list(dict.fromkeys(especies))
bolsig = list(dict.fromkeys(" ".join(bloco(ar, "BOLSIG") + bloco(n2, "BOLSIG")).split()))
kinet = f"""#
# Ar:N2 montado de kinet.inp (argônio) e casos/n2/kinet.inp por casos/ar_n2/monta.py
#

ELEMENTS
e Ar N
END

BOLSIG
{" ".join(bolsig)}
END

SPECIES
{" ".join(especies)}
END

REACTIONS
#--------------------------------------------------------------------------- argônio
{reacoes(ar)}

#--------------------------------------------------------------------------- N2
{reacoes(n2)}

#--------------------------------------------------------------------------- cruzadas
Ar^+ + N2 => N2^+ + Ar             !   1.0d-11
Ar* + N2 => Ar + N2(C)             !   3.0d-11
Ar2^+ + N2 => N2^+ + Ar + Ar       !   1.0d-11

END
"""
(AQUI / "kinet.inp").write_text(kinet)
(AQUI / "bolsigdb.dat").write_text((RAIZ / "bolsigdb.dat").read_text().rstrip() + "\n\n"
                                   + (RAIZ / "casos" / "n2" / "bolsigdb.dat").read_text())
(AQUI / "composicao.txt").write_text("Ar 0.9\nN2 0.1\nAr* traco\nN2^+ traco\n")
print(f"{len(especies)} species, BOLSIG: {' '.join(bolsig)}")
