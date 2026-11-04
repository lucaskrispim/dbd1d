# SPDX-License-Identifier: GPL-3.0-or-later
"""The N2 chemistries of the positive column (docs/coluna_positiva_n2), rebuilt from
the cases they come from, whose mechanisms belong to other authors and are not
distributed (casos/README.md):

- n2_exemplo: the N2 example of ZDPlasKin, the kinet.inp of casos/n2;
- n2_amonia: the N2 part of the N2-H2 mechanism of Sun et al. 2022, casos/xu2025
  without H (ferramentas/subconjunto.py);
- n2_tese: the N2 part of the N2-O2 mechanism, casos/ar_n2_o2 without O and Ar;
- n2_amonia_v45, n2_tese_v45: the two above with the vibrational ladder of N2(X)
  up to v = 45 (ferramentas/escada_vibracional.py);
- n2_amonia_a2, n2_tese_a2: n2_amonia and n2_tese with N2(a''1) as a species of its
  own instead of lumped into N2(C3) (separa_a2). In the N2 example of ZDPlasKin
  the excitation to N2(a''1) produces no species and there is nothing to separate.

Each case gets the kinet.inp and the bolsigdb.dat of the case it comes from; the
other files of the case (gas.json, transporte.json, composicao.txt, t_nos.txt) are
distributed. Then `python3 ferramentas/caso.py all casos/NAME` builds the chemistry
and the tables (needs ZDPlasKin and BOLSIG+).

Usage: python3 ferramentas/casos_coluna.py [--casos CASES_DIR] [NAME ...]
"""
import argparse
import pathlib
import re
import sys

from escada_vibracional import substitui_vibracao
from subconjunto import subconjunto

RAIZ = pathlib.Path(__file__).resolve().parent.parent
SUBCONJUNTO = "# Subset of casos/{origem}/kinet.inp without {sem}, made by ferramentas/subconjunto.py\n" \
              "# for the N2 positive column (docs/coluna_positiva_n2).\n"
ESCADA = "# {origem} with the vibrational ladder of N2(X) up to v = 45 (ferramentas/escada_vibracional.py)\n"


def _sem(*elementos):
    def receita(texto, origem):
        for el in elementos:
            texto = subconjunto(texto, el)
        return SUBCONJUNTO.format(origem=origem, sem=" and ".join(elementos)) + texto
    return receita


def _escada(texto, origem):
    return ESCADA.format(origem=origem) + substitui_vibracao(texto, 45)


# the preprocessor of ZDPlasKin refuses '' in a species name (as chemistry B writes N2(a`1) for a')
A2_ESPECIE = "N2(a``1)"
A2 = "# {origem} with N2(a''1) separated from N2(C3) (ferramentas/casos_coluna.py, separa_a2)\n"


def separa_a2(texto):
    """kinet.inp with N2(a''1) taken out of the list of processes lumped into
    N2(C3) and made a species (named A2_ESPECIE): its own electron excitation (the same BOLSIG+
    process), loss by quenching with N2 to the ground state, 1e-14 cm3/s (Gordiets
    et al. 1995, table II, R29), and at the wall (transporte.json of the case)."""
    linhas = texto.splitlines()
    saida, achou = [], False
    for i, l in enumerate(linhas):
        anterior = linhas[i - 1] if i else ""
        excita_c = re.match(r"\s*([eE])\s*\+\s*N2\s*=>\s*[eE]\s*\+\s*N2\(C3\)\s*!", anterior)
        if excita_c and l.lstrip().startswith("@R") and "N2(a''1)" in l.split():
            nomes = [n for n in l.split("=", 1)[1].split() if n != "N2(a''1)"]
            e = excita_c.group(1)
            saida += ["    @R = " + "  ".join(nomes), f"{e} + N2 => {e} + {A2_ESPECIE}".ljust(43) + "! BOLSIG N2 -> N2(a''1)"]
            achou = True
            continue
        saida.append(l)
    if not achou:
        raise ValueError("no @R list with N2(a''1) after an excitation to N2(C3)")
    secao = None
    for i, l in enumerate(saida):
        cabeca = l.strip().upper()
        if cabeca in ("ELEMENTS", "SPECIES", "BOLSIG", "REACTIONS"):
            secao = cabeca
        elif cabeca == "END" and secao == "SPECIES":
            saida.insert(i, A2_ESPECIE)
            break
    fim = max(i for i, l in enumerate(saida) if l.strip().upper() == "END")
    saida[fim:fim] = ["# N2(a'') separated from N2(C): quenching by N2, Gordiets et al. 1995, table II, R29",
                      f"{A2_ESPECIE} + N2 => N2 + N2 ! 1.0d-14"]
    return "\n".join(saida) + "\n"


def _a2(texto, origem):
    return A2.format(origem=origem) + separa_a2(texto)


# name: (case it comes from, kinet.inp of the origin -> kinet.inp), in order of dependence
RECEITAS = {
    "n2_exemplo": ("n2", lambda texto, origem: texto),
    "n2_amonia": ("xu2025", _sem("H")),
    "n2_tese": ("ar_n2_o2", _sem("O", "Ar")),
    "n2_amonia_v45": ("n2_amonia", _escada),
    "n2_tese_v45": ("n2_tese", _escada),
    "n2_amonia_a2": ("n2_amonia", _a2),
    "n2_tese_a2": ("n2_tese", _a2),
}


def _le(casos, origem, arquivo):
    caminho = casos / origem / arquivo
    if not caminho.exists():
        raise FileNotFoundError(f"{caminho} is missing: rebuild the case {origem} first (casos/README.md)")
    return caminho.read_text()


def monta(nome, casos):
    """(kinet.inp, bolsigdb.dat) of the case nome, from the cases in the folder casos."""
    if nome not in RECEITAS:
        raise ValueError(f"unknown case {nome}: {', '.join(RECEITAS)}")
    origem, receita = RECEITAS[nome]
    casos = pathlib.Path(casos)
    if origem in RECEITAS:
        texto, banco = monta(origem, casos)
    else:
        texto, banco = _le(casos, origem, "kinet.inp"), _le(casos, origem, "bolsigdb.dat")
    return receita(texto, origem), banco


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("nomes", nargs="*", default=list(RECEITAS), metavar="NAME", help="cases (default: all)")
    p.add_argument("--casos", type=pathlib.Path, default=RAIZ / "casos", help="folder of the cases")
    a = p.parse_args()
    for nome in a.nomes:
        try:
            kinet, banco = monta(nome, a.casos)
        except (ValueError, FileNotFoundError) as erro:
            sys.exit(str(erro))
        destino = a.casos / nome
        destino.mkdir(parents=True, exist_ok=True)
        (destino / "kinet.inp").write_text(kinet)
        (destino / "bolsigdb.dat").write_text(banco)
        print(f"{destino}: kinet.inp and bolsigdb.dat written; next: python3 ferramentas/caso.py all {destino}")


if __name__ == "__main__":
    main()
