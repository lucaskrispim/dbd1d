# SPDX-License-Identifier: GPL-3.0-or-later
"""Compara as saídas de duas execuções do simulador, coluna a coluna.

Uso: python3 scripts/compara.py DIR_A DIR_B [--tol 1e-12] [--tempo-max 2e-8]

Lê perfil.dat, superficie.dat, borda0.dat, centro.dat e borda1.dat dos dois
diretórios (as séries só se existirem nos dois, porque o protótipo não grava as
três últimas), alinha os registros
pelo instante e as colunas pelo nome, e reporta a maior diferença relativa de
cada coluna comum; colunas que só um lado tem são listadas. Serve para Fortran
contra Fortran, protótipo contra protótipo e um contra o outro. Com --estrito (o
make regressao usa), também reprova série ausente de um lado, coluna só de um
lado (exceto CFL e n_eps, próprias de cada código) e instante só de um lado.
A diferença relativa usa |a-b| / max(|a|, |b|, piso), em que o piso é 1e-30
vezes o maior valor absoluto da coluna, para não amplificar ruído em torno de
zero. Sai com código 1 se alguma coluna passar de --tol.
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
        # o formato 1pe24.16 perde o E quando o expoente tem três dígitos
        if "-" in texto[1:] or "+" in texto[1:]:
            i = max(texto.rfind("-"), texto.rfind("+"))
            return float(texto[:i] + "E" + texto[i:])
        raise


def cabecalho(linha):
    """O writer grava os nomes em campos de 24 caracteres; alguns têm espaço."""
    return [linha[i:i + 24].strip() for i in range(0, len(linha.rstrip()), 24)]


def le_serie(caminho, tempo_max):
    """Arquivo com cabeçalho e uma linha por instante; a primeira coluna é o tempo."""
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
        sys.exit(f"{caminho}: colunas repetidas: {', '.join(repetidos)}")
    return nomes


def le_perfil(caminho, tempo_max):
    """Blocos separados por linha vazia: instante, cabeçalho, uma linha por nó.
    Cada bloco é lido com o seu cabeçalho e os valores ficam por nome."""
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
        print(f"{rotulo}: nenhum registro em comum")
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
                continue                       # a última linha do Fortran não tem campo
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
    print(f"{rotulo}: {len(comuns)} registros, {len(nomes)} colunas comuns, {'OK' if ok else 'DIFERE'}")
    for n, d in pior.items():
        if d > tol:
            print(f"    {n:>14s}  dif. relativa máx. {d:.3e}")
    if so_um:
        print(f"    colunas só de um lado: {', '.join(so_um)}")
    if so_registro:
        print(f"    registros só de um lado: {so_registro}")
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("a", type=pathlib.Path)
    p.add_argument("b", type=pathlib.Path)
    p.add_argument("--tol", type=float, default=1e-12)
    p.add_argument("--tempo-max", type=float, default=math.inf)
    p.add_argument("--estrito", action="store_true",
                   help="reprova série ausente, coluna ou instante só de um lado")
    args = p.parse_args()

    ok = True
    for nome in SERIES:
        existe = ((args.a / nome).exists(), (args.b / nome).exists())
        if not all(existe):
            if any(existe):
                print(f"{nome}: só em {'A' if existe[0] else 'B'}")
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
