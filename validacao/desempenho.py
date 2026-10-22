# SPDX-License-Identifier: GPL-3.0-or-later
"""Performance table of the validation runs: for each run, the mixture (species
and reactions), the cells, the simulated time, the wall time and the counts of the
integrator (right-hand side evaluations, Jacobians and factorizations), with the
cost per period or per pulse. Reads the .log and the parametros.json of each run
(logs of the English and of the earlier Portuguese versions).

Usage: python3 validacao/desempenho.py DIR_WITH_THE_validacao_*_FOLDERS [--markdown]
"""
import json
import pathlib
import re
import sys

RAIZ = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "dbd1d"))


def mistura(caso):
    """Species and reactions of the chemistry of the case (zdplaskin_m.F90 or quimica.json)."""
    from quimica_zdp import Quimica
    # the recorded path is that of the machine that ran: casos/NAME becomes the
    # local case of the same name, and the root there (argon) the local root
    caso = pathlib.Path(caso)
    caso = RAIZ / "casos" / caso.name if caso.parent.name == "casos" else RAIZ
    q = Quimica(caso / "zdplaskin_m.F90") if (caso / "zdplaskin_m.F90").exists() \
        else Quimica.de_json(caso / "quimica.json")
    return len(q.especies), len(q.reacoes)


def linha(log):
    texto = log.read_text(errors="replace")
    ult = [l for l in texto.splitlines() if l.startswith("t = ")]
    parede = re.search(r"(?:tempo de parede|wall time): ([\d.]+) s", texto)
    if not ult or not parede:
        return None
    m = re.search(r"t = (\S+) s\s+(?:avaliacoes|evaluations) (\d+)\s+(?:jacobianos|jacobians) (\d+)\s+"
                  r"(?:fatoracoes|factorizations) (\d+)", ult[-1])
    return float(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)), float(parede.group(1))


def main():
    base = pathlib.Path(sys.argv[1])
    md = "--markdown" in sys.argv
    cab = ["run", "case", "species", "reactions", "cells", "periods", "t simulated (s)", "wall (s)",
           "wall per period (s)", "evaluations", "jacobians", "factorizations"]
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
            else "argon"
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
