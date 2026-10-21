# SPDX-License-Identifier: GPL-3.0-or-later
"""Summary of a run: evolution of the maxima in time.

Usage: python3 -I scripts/resumo.py DIRECTORY [--every N]

Reads perfil.dat and shows, every N records, the largest value in the gap of the
electron density and of each species of the case except the background gas
(cm-3; in argon, Ar+, Ar2+ and Ar*), E/N (Td), Te (K) and the gas temperature
(K). Counts NaN and negative densities and shows the most negative one with the
atol of the run: a negative value below atol is noise of the integrator.
Also provides blocos, the reader of the block files (perfil.dat, EEDF).
"""
import argparse
import math
import pathlib


def parametros_da_execucao(diretorio):
    """Arguments written by simula.py in parametros.json ({} if there is none)."""
    import json
    arquivo = pathlib.Path(diretorio) / "parametros.json"
    return json.loads(arquivo.read_text())["argumentos"] if arquivo.exists() else {}


RAIZ = pathlib.Path(__file__).resolve().parent.parent


class ErroCaso(Exception):
    """Case of the run not found, or kinet.inp without SPECIES."""
# names of the argon quantities, as in the earlier records
ROTULOS = {"AR^+": "Ar+", "AR2^+": "Ar2+", "AR*": "Ar*"}


def rotulo(especie):
    return ROTULOS.get(especie, especie)


def especies_do_caso(diretorio):
    """Species of the case of the run, in upper case as in perfil.dat, without the
    electron and without the background gas (SPECIES block of kinet.inp), and the
    main ion (ion_inicial of gas.json). Without a recorded case, the root: argon."""
    import json
    caso = pathlib.Path(parametros_da_execucao(diretorio).get("caso") or RAIZ)
    if not caso.exists():
        # run from another machine: the same case in the local root
        if caso.parent.name == "casos":
            if not (RAIZ / "casos" / caso.name).exists():
                raise ErroCaso(f"{diretorio}: case {caso} does not exist, not even as casos/{caso.name} here")
            caso = RAIZ / "casos" / caso.name
        else:
            caso = RAIZ
    linhas = (caso / "kinet.inp").read_text().splitlines()
    i = next((k for k, l in enumerate(linhas) if l.strip().upper() == "SPECIES"), None)
    if i is None:
        raise ErroCaso(f"{caso / 'kinet.inp'}: no SPECIES line")
    especies = []
    for linha in linhas[i + 1:]:
        linha = linha.split("#")[0].split("!")[0].strip().upper()
        if linha == "END":
            break
        especies += linha.split()
    gas = json.loads((caso / "gas.json").read_text())
    fundo = {k.upper() for k in gas["fundo"]}
    return [e for e in especies if e != "E" and e not in fundo], gas["ion_inicial"].upper()


def numero(texto):
    try:
        return float(texto)
    except ValueError:
        i = max(texto.rfind("-"), texto.rfind("+"))
        return float(texto[:i] + "E" + texto[i:])


def cabecalho(linha):
    return [linha[i:i + 24].strip() for i in range(0, len(linha.rstrip()), 24)]


def blocos(caminho):
    bloco = []
    for linha in caminho.read_text().splitlines() + [""]:
        if linha.strip():
            bloco.append(linha)
            continue
        if len(bloco) > 2:
            yield numero(bloco[0].split()[0]), cabecalho(bloco[1]), \
                [[numero(c) for c in l.split()] for l in bloco[2:]]
        bloco = []


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dir", type=pathlib.Path)
    p.add_argument("--every", "--cada", dest="cada", type=int, default=1, help="show every N records")
    a = p.parse_args()
    especies, ion = especies_do_caso(a.dir)
    densidades = ["E", ion] + [e for e in especies if e != ion]
    colunas = densidades + ["E/N", "Te", "Temperature"]
    print(f"{'t (ns)':>8s}" + "".join(f"{c:>12s}" for c in colunas))
    nans, negativos, pior = 0, 0, (0.0, "")
    for k, (t, nomes, linhas) in enumerate(blocos(a.dir / "perfil.dat")):
        colunas_aqui = [c for c in colunas if c in nomes]
        idx = [nomes.index(c) for c in colunas_aqui]
        interior = linhas[1:-1]
        for l in interior:
            for j in idx:
                if j < len(l) and math.isnan(l[j]):
                    nans += 1
            for c in densidades:
                if c not in nomes:
                    continue
                j = nomes.index(c)
                if l[j] < 0:
                    negativos += 1
                    pior = min(pior, (l[j], c))
        if k % a.cada == 0:
            maximos = [max(l[j] for l in interior if j < len(l)) for j in idx]
            print(f"{t*1e9:8.1f}" + "".join(f"{m:12.3e}" for m in maximos))
    print("NaN:", nans)
    texto = f"negative density: {negativos}"
    if negativos:
        atol = parametros_da_execucao(a.dir).get("atol")
        texto += f", the most negative {pior[0]:.1e} cm-3 ({pior[1]})" + (f"; atol {atol:g}" if atol else "")
    print(texto)


if __name__ == "__main__":
    main()
