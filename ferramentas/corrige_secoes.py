# SPDX-License-Identifier: GPL-3.0-or-later
"""Corrections of the published cross-section files used by the cases (docs/auditoria/2026-11-10/ERROS.md).

The files come from other authors and are not distributed, so the corrections are applied
to the user's copy:

- the three blocks of the Biagi database of the supplementary file of Sun et al. 2022
  labelled as dissociation (N2 -> N + N(2D), N + N(2P), N(2D) + N(2D)) are, by their PROCESS
  lines, excitations of a''1Sigma, b'1Sigma and the sum of the singlets already in the Phelps set: removed (E4, V2-1);
- the four N2 blocks of the N2-O2 file of the ZDPlasKin website that repeat the E3, a''
  and singlet cross sections of the Morgan set (N2*(11.88eV), N2*(12.25eV), N2(a"1),
  N+N(13eV)) are removed (E6, V1-2); the copies kept are those of the file (N2(E3), N2(a''1),
  N2(SUM)), which the chemistry names; N2(E3) has, besides the Morgan points, the resonance
  peak at 11.92 eV (4.96e-22 m2), so chemistry B is Morgan except for that point (decision of the
  authors, 11/11/2026; ERROS.md N3);
- the second O2 -> O2^- block of that file ("Two body attachment"), whose points are those of
  the dissociative attachment O2 -> O^-+O, is removed (T3-1);
- the statistical weight ratios of the <-> processes of that file, 1e4 to 1e45, are set to the
  physical values: 1 for N2(v) and O2(v), 3 for N2(A3), 2/3 for O2(a1) (V1-3).

Usage: python3 ferramentas/corrige_secoes.py FILE [FILE ...]   (rewrites each file in place;
a file already corrected is left unchanged)
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from acoplamento_vibracional import blocos  # noqa: E402

BIAGI = {("EXCITATION", "N2 -> N + N(2D)"), ("EXCITATION", "N2 -> N + N(2P)"),
         ("EXCITATION", "N2 -> N(2D) + N(2D)")}
# removed only when the four are in the file, as in the N2-O2 file: the N2 file of the
# ZDPlasKin example has N2(a"1) alone, as its only a'' cross section, and keeps it
MORGAN_REPETIDOS = {("EXCITATION", "N2 -> N2*(11.88eV)"), ("EXCITATION", "N2 -> N2*(12.25eV)"),
                    ("EXCITATION", 'N2 -> N2(a"1)'), ("EXCITATION", "N2 -> N+N(13eV)")}

# the second O2 -> O2^- block of the N2-O2 file ("Two body attachment") carries the
# dissociative attachment cross section of O2 -> O^-+O (onset 4.4 eV, same points up to
# 11.35 eV), which the file already has: removed (T3-1); told apart from the three-body block,
# which has the same PROCESS, by the start of its COMMENT line
REPETIDOS_POR_COMENTARIO = {("ATTACHMENT", "O2 -> O2^-", "COMMENT: Two body attachment")}

# ratio g(final)/g(initial) of the superelastic inverse, as written in the threshold line
RAZOES = {
    **{f"N2 <-> N2(V{v})": "1.0d0" for v in range(1, 9)},
    "N2 <-> N2(A3)": "3.0d0",
    **{f"O2 <-> O2(V{v})": "1.0d0" for v in range(1, 5)},
    "O2 <-> O2(A1)": "6.667d-1",
}


def _razao(limiar, nova):
    """Threshold line with its second field (the weight ratio) replaced, spacing kept."""
    campos = limiar.split()
    inicio = limiar.index(campos[1], limiar.index(campos[0]) + len(campos[0]))
    return limiar[:inicio] + nova + limiar[inicio + len(campos[1]):]


def corrige(texto):
    """Text of a bolsigdb.dat with the blocks of BIAGI, of REPETIDOS_POR_COMENTARIO, and of
    MORGAN_REPETIDOS when all four are present, removed (each with the blank lines that follow it), and the ratios of RAZOES
    set, except an exact zero, which switches the inverse off on purpose. Idempotent."""
    partes = blocos(texto)
    chaves = {(l[0].strip(), l[1].strip()) for t, l in partes if t == "bloco"}
    removidos = BIAGI | (MORGAN_REPETIDOS if MORGAN_REPETIDOS <= chaves else set())
    saida, pular_brancos = [], False
    for tipo, linhas in partes:
        if tipo == "bloco":
            chave = (linhas[0].strip(), linhas[1].strip())
            if chave in removidos or any(chave == r[:2] and len(linhas) > 2 and linhas[2].strip().startswith(r[2])
                                         for r in REPETIDOS_POR_COMENTARIO):
                pular_brancos = True
                continue
            if chave[1] in RAZOES and float(linhas[2].split()[1].replace("d", "e")) != 0.0:
                linhas = [*linhas[:2], _razao(linhas[2], RAZOES[chave[1]]), *linhas[3:]]
        elif pular_brancos and not linhas[0].strip():
            continue
        pular_brancos = False
        saida.extend(linhas)
    return "\n".join(saida) + ("\n" if texto.endswith("\n") else "")


def main(arquivos):
    for nome in arquivos:
        p = pathlib.Path(nome)
        antes = p.read_text()
        depois = corrige(antes)
        if depois != antes:
            p.write_text(depois)
        print(f"{p}: {'corrected' if depois != antes else 'already corrected'}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
