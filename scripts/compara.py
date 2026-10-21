# SPDX-License-Identifier: GPL-3.0-or-later
"""Compares the outputs of two runs of the simulator, column by column.

Usage: python3 scripts/compara.py DIR_A DIR_B [--tol 1e-12] [--max-time 2e-8]

Reads perfil.dat and superficie.dat (and borda0.dat, centro.dat and borda1.dat
of the original Fortran code, if present in both directories), aligns the
records by time and the columns by name, and reports the largest relative
difference of each common column; columns present on one side only are listed.
With --strict (the regression test uses it), it also fails on a series missing
on one side, a column on one side only (except CFL and n_eps, specific to each
code) and a time on one side only. The relative difference is |a-b| /
max(|a|, |b|, floor), with the floor 1e-30 times the largest absolute value of
the column, so as not to amplify noise around zero. Exits with code 1 if any
column exceeds --tol.
"""
import argparse
import math
import pathlib
import sys

SERIES = ["superficie.dat", "borda0.dat", "centro.dat", "borda1.dat"]


def numero(texto):
    try:
        return float(texto)
    except ValueError:
        # the format 1pe24.16 loses the E when the exponent has three digits
        if "-" in texto[1:] or "+" in texto[1:]:
            i = max(texto.rfind("-"), texto.rfind("+"))
            return float(texto[:i] + "E" + texto[i:])
        raise


def cabecalho(linha):
    """The writer puts the names in fields of 24 characters; some contain spaces."""
    return [linha[i:i + 24].strip() for i in range(0, len(linha.rstrip()), 24)]


def le_serie(caminho, tempo_max):
    """File with a header and one line per time; the first column is the time."""
    linhas = caminho.read_text().splitlines()
    nomes = confere_nomes(cabecalho(linhas[0]), caminho)
    dados = {}
    for linha in linhas[1:]:
        campos = linha.split()
        if not campos:
            continue
        valores = [numero(c) for c in campos]
        if valores[0] <= tempo_max * (1 + 1e-9):
            dados[round(valores[0] / 1e-12)] = dict(zip(nomes, valores))
    return nomes, dados


PROPRIAS = {"CFL", "n_eps"}


def confere_nomes(nomes, caminho):
    repetidos = sorted({n for n in nomes if nomes.count(n) > 1})
    if repetidos:
        sys.exit(f"{caminho}: repeated columns: {', '.join(repetidos)}")
    return nomes


def le_perfil(caminho, tempo_max):
    """Blocks separated by an empty line: time, header, one line per node. Each
    block is read with its own header and the values are kept by name."""
    dados = {}
    nomes = None
    bloco = []
    for linha in caminho.read_text().splitlines() + [""]:
        if linha.strip():
            bloco.append(linha)
            continue
        if len(bloco) > 2:
            t = numero(bloco[0].split()[0])
            proprios = confere_nomes(cabecalho(bloco[1]), caminho)
            nomes = nomes or proprios
            if t <= tempo_max * (1 + 1e-9):
                for no, l in enumerate(bloco[2:]):
                    valores = [numero(c) for c in l.split()]
                    dados[(round(t / 1e-12), no)] = dict(zip(proprios, valores))
        bloco = []
    return nomes, dados


def compara(nomes_a, nomes_b, a, b, rotulo, tol, estrito=False):
    comuns = sorted(set(a) & set(b))
    if not comuns:
        print(f"{rotulo}: no record in common")
        return False
    nomes = [n for n in nomes_a if n in nomes_b]
    so_um = [n for n in nomes_a if n not in nomes_b] + [n for n in nomes_b if n not in nomes_a]
    so_registro = len(set(a) ^ set(b))
    maior_abs = dict.fromkeys(nomes, 0.0)
    pior = dict.fromkeys(nomes, 0.0)
    for k in comuns:
        for n in nomes:
            if n in a[k] and n in b[k]:
                maior_abs[n] = max(maior_abs[n], abs(a[k][n]), abs(b[k][n]))
    for k in comuns:
        for n in nomes:
            if n not in a[k] or n not in b[k]:
                continue                       # the last line of the Fortran output lacks the field
            x, y = a[k][n], b[k][n]
            if math.isnan(x) or math.isnan(y):
                d = 0.0 if (math.isnan(x) and math.isnan(y)) else math.inf
            else:
                piso = max(1e-30 * maior_abs[n], 1e-300)
                d = abs(x - y) / max(abs(x), abs(y), piso)
            pior[n] = max(pior[n], d)
    ok = all(d <= tol for d in pior.values())
    if estrito:
        ok = ok and so_registro == 0 and all(n in PROPRIAS for n in so_um)
    print(f"{rotulo}: {len(comuns)} records, {len(nomes)} common columns, {'OK' if ok else 'DIFFERS'}")
    for n, d in pior.items():
        if d > tol:
            print(f"    {n:>14s}  max. relative diff. {d:.3e}")
    if so_um:
        print(f"    columns on one side only: {', '.join(so_um)}")
    if so_registro:
        print(f"    records on one side only: {so_registro}")
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("a", type=pathlib.Path)
    p.add_argument("b", type=pathlib.Path)
    p.add_argument("--tol", type=float, default=1e-12)
    p.add_argument("--max-time", "--tempo-max", dest="tempo_max", type=float, default=math.inf)
    p.add_argument("--strict", "--estrito", dest="estrito", action="store_true",
                   help="fail on a missing series, or a column or time on one side only")
    args = p.parse_args()

    ok = True
    for nome in SERIES:
        existe = ((args.a / nome).exists(), (args.b / nome).exists())
        if not all(existe):
            if any(existe):
                print(f"{nome}: only in {'A' if existe[0] else 'B'}")
                ok &= not args.estrito
            continue
        na, da = le_serie(args.a / nome, args.tempo_max)
        nb, db = le_serie(args.b / nome, args.tempo_max)
        ok &= compara(na, nb, da, db, nome, args.tol, args.estrito)
    na, da = le_perfil(args.a / "perfil.dat", args.tempo_max)
    nb, db = le_perfil(args.b / "perfil.dat", args.tempo_max)
    ok &= compara(na, nb, da, db, "perfil.dat", args.tol, args.estrito)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
