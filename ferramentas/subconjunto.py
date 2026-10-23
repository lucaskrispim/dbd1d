# SPDX-License-Identifier: GPL-3.0-or-later
"""Subset of a kinet.inp (ZDPlasKin format) without one element: the species that
contain it leave ELEMENTS, SPECIES and BOLSIG (and the DENSITY lines that use
them), the reactions that use them are dropped, and in a reaction with @ blocks
(third bodies, @M = N2 H2 with @R = 1.0d0 1.0d0) only the positions that bring the
element are dropped, in all the lists of the reaction (the @R lists of the
BOLSIG+ reactions name processes of the cross sections, not species, and stay
whole). The $ lines are kept (unused variables are
harmless). The rest of the text is copied unchanged.

Usage: python3 ferramentas/subconjunto.py ORIGIN/kinet.inp DESTINATION/kinet.inp ELEMENT
"""
import re
import sys

SECOES = ("ELEMENTS", "SPECIES", "BOLSIG", "REACTIONS")


def elementos(especie):
    """Elements of a species name: N2(A3) -> {N}, NH2^+ -> {N, H}; the electron, none."""
    nome = especie.split("^")[0].split("(")[0]
    if nome.upper() == "E":
        return set()
    return set(re.findall(r"[A-Z][a-z]?", nome))


def _tokens(linha):
    return linha.split("#")[0].split()


def _especies_da_reacao(linha):
    # the + that separates species has spaces around it; the one of a charge (N2^+) does not
    equacao = linha.split("!")[0]
    return [t.strip() for t in re.split(r"\s\+\s|=>", equacao) if t.strip()]


def subconjunto(texto, elemento):
    linhas = texto.splitlines()
    especies = set()
    secao = None
    for linha in linhas:
        cabeca = linha.strip().upper()
        if cabeca in SECOES:
            secao = cabeca
        elif cabeca == "END":
            secao = None
        elif secao == "SPECIES":
            especies.update(_tokens(linha))
    fora = {e for e in especies if elemento in elementos(e)}
    if not fora:
        raise ValueError(f"no species of {elemento} in the chemistry")

    saida, secao, i = [], None, 0
    while i < len(linhas):
        linha = linhas[i]
        cabeca = linha.strip().upper()
        if cabeca in SECOES:
            secao = cabeca
            saida.append(linha)
            i += 1
            continue
        if cabeca == "END":
            secao = None
            saida.append(linha)
            i += 1
            continue
        if secao in ("ELEMENTS", "SPECIES", "BOLSIG") and linha.strip() and not linha.strip().startswith("#"):
            toks = _tokens(linha)
            if toks and toks[0].upper() == "DENSITY":
                if not any(t in fora for t in toks):
                    saida.append(linha)
            else:
                manter = [t for t in toks if (t != elemento if secao == "ELEMENTS" else t not in fora)]
                if manter:
                    saida.append("  ".join(manter))
            i += 1
            continue
        if secao == "REACTIONS" and "=>" in linha and not linha.lstrip().startswith(("#", "$")):
            blocos = []
            j = i + 1
            while j < len(linhas) and linhas[j].lstrip().startswith("@"):
                blocos.append(linhas[j])
                j += 1
            explicitas = [t for t in _especies_da_reacao(linha) if not t.startswith("@")]
            if any(t in fora for t in explicitas):
                i = j
                continue
            if blocos:
                listas = []
                for b in blocos:
                    nome, valores = b.split("=", 1)
                    listas.append((b[:len(b) - len(b.lstrip())], nome.strip(), valores.split()))
                n = max(len(v) for _, _, v in listas)
                ficam = [k for k in range(n) if not any(len(v) == n and v[k] in fora for _, _, v in listas)]
                if not ficam:
                    i = j
                    continue
                saida.append(linha)
                for recuo, nome, valores in listas:
                    novos = [valores[k] for k in ficam] if len(valores) == n else valores
                    saida.append(f"{recuo}{nome} = " + "  ".join(novos))
            else:
                saida.append(linha)
            i = j
            continue
        saida.append(linha)
        i += 1
    return "\n".join(saida) + "\n"


def main():
    if len(sys.argv) != 4:
        sys.exit("Usage: python3 ferramentas/subconjunto.py ORIGIN/kinet.inp DESTINATION/kinet.inp ELEMENT")
    origem, destino, elemento = sys.argv[1:4]
    texto = subconjunto(open(origem).read(), elemento)
    with open(destino, "w") as f:
        f.write(texto)


if __name__ == "__main__":
    main()
