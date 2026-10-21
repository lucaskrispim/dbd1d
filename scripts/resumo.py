# SPDX-License-Identifier: GPL-3.0-or-later
"""Resumo de uma execução: evolução dos máximos ao longo do tempo.

Uso: python3 -I scripts/resumo.py DIRETORIO [--cada N]

Lê perfil.dat e mostra, a cada N registros, o maior valor no gap de densidade
de elétrons e de cada espécie do caso menos o gás de fundo (cm-3; no argônio,
Ar+, Ar2+ e Ar*), E/N (Td), Te (K) e temperatura do gás (K).
Conta NaN e densidades negativas e mostra a mais negativa com o atol da
execução: negativo abaixo do atol é ruído do integrador.
"""
import argparse
import math
import pathlib


def parametros_da_execucao(diretorio):
    """Argumentos gravados pelo simula.py em parametros.json ({} se não houver)."""
    import json
    arquivo = pathlib.Path(diretorio) / "parametros.json"
    return json.loads(arquivo.read_text())["argumentos"] if arquivo.exists() else {}


RAIZ = pathlib.Path(__file__).resolve().parent.parent


class ErroCaso(Exception):
    """Caso da execução que não se acha ou kinet.inp sem SPECIES."""
# nomes das métricas do argônio, os mesmos dos registros anteriores
ROTULOS = {"AR^+": "Ar+", "AR2^+": "Ar2+", "AR*": "Ar*"}


def rotulo(especie):
    return ROTULOS.get(especie, especie)


def especies_do_caso(diretorio):
    """Espécies do caso da execução, em maiúsculas como no perfil.dat, sem o
    elétron e sem o gás de fundo (bloco SPECIES do kinet.inp), e o íon principal
    (ion_inicial do gas.json). Sem caso registrado, a raiz: o argônio."""
    import json
    caso = pathlib.Path(parametros_da_execucao(diretorio).get("caso") or RAIZ)
    if not caso.exists():
        # execução de outra máquina: o mesmo caso na raiz local
        if caso.parent.name == "casos":
            if not (RAIZ / "casos" / caso.name).exists():
                raise ErroCaso(f"{diretorio}: caso {caso} não existe nem como casos/{caso.name} aqui")
            caso = RAIZ / "casos" / caso.name
        else:
            caso = RAIZ
    linhas = (caso / "kinet.inp").read_text().splitlines()
    i = next((k for k, l in enumerate(linhas) if l.strip().upper() == "SPECIES"), None)
    if i is None:
        raise ErroCaso(f"{caso / 'kinet.inp'}: sem a linha SPECIES")
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
    p.add_argument("--cada", type=int, default=1)
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
    texto = f"densidade negativa: {negativos}"
    if negativos:
        atol = parametros_da_execucao(a.dir).get("atol")
        texto += f", a mais negativa {pior[0]:.1e} cm-3 ({pior[1]})" + (f"; atol {atol:g}" if atol else "")
    print(texto)


if __name__ == "__main__":
    main()
