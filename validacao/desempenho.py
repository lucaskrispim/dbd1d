# SPDX-License-Identifier: GPL-3.0-or-later
"""Tabela de desempenho das execuções de validação: para cada execução, a
mistura (espécies e reações), as células, o tempo simulado, o tempo de parede
e as contagens do integrador (avaliações do lado direito, jacobianos e
fatorações), com o custo por período ou por pulso. Lê o .log e o
parametros.json de cada execução.

Uso: python3 validacao/desempenho.py DIR_COM_AS_PASTAS_validacao_* [--markdown]
"""
import json
import pathlib
import re
import sys

RAIZ = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "prototipo"))


def mistura(caso):
    """Espécies e reações da química do caso (zdplaskin_m.F90 ou quimica.json)."""
    from quimica_zdp import Quimica
    # o caminho gravado é o da máquina que rodou: casos/NOME vira o caso local de
    # mesmo nome, e a raiz de lá (o argônio) vira a raiz daqui
    caso = pathlib.Path(caso)
    caso = RAIZ / "casos" / caso.name if caso.parent.name == "casos" else RAIZ
    q = Quimica(caso / "zdplaskin_m.F90") if (caso / "zdplaskin_m.F90").exists() \
        else Quimica.de_json(caso / "quimica.json")
    return len(q.especies), len(q.reacoes)


def linha(log):
    texto = log.read_text(errors="replace")
    ult = [l for l in texto.splitlines() if l.startswith("t = ")]
    parede = re.search(r"tempo de parede: ([\d.]+) s", texto)
    if not ult or not parede:
        return None
    m = re.search(r"t = (\S+) s\s+avaliacoes (\d+)\s+jacobianos (\d+)\s+fatoracoes (\d+)", ult[-1])
    return float(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)), float(parede.group(1))


def main():
    base = pathlib.Path(sys.argv[1])
    md = "--markdown" in sys.argv
    cab = ["execução", "caso", "espécies", "reações", "células", "períodos", "t simulado (s)", "parede (s)",
           "parede por período (s)", "avaliações", "jacobianos", "fatorações"]
    linhas = []
    for log in sorted(base.glob("validacao_*/*.log")):
        d = log.with_suffix("")
        if not (d / "parametros.json").exists():
            continue
        a = json.loads((d / "parametros.json").read_text())["argumentos"]
        if a.get("retoma"):
            continue
        r = linha(log)
        if r is None:
            continue
        t, aval, jac, fat, parede = r
        ne, nr = mistura(a.get("caso", RAIZ))
        periodos = t / a["periodo"]
        nome_caso = pathlib.Path(a.get("caso", "")).name if pathlib.Path(a.get("caso", "")).parent.name == "casos" \
            else "argônio"
        linhas.append([d.name, nome_caso, ne, nr, a["n_celulas"],
                       f"{periodos:.0f}", f"{t:.3g}", f"{parede:.0f}", f"{parede / periodos:.0f}", aval, jac, fat])
    if md:
        print("| " + " | ".join(cab) + " |")
        print("|" + "---|" * len(cab))
        for l in linhas:
            print("| " + " | ".join(str(x) for x in l) + " |")
    else:
        print(" ".join(f"{c:>14s}" for c in cab))
        for l in linhas:
            print(" ".join(f"{str(x):>14s}" for x in l))


if __name__ == "__main__":
    main()
