# SPDX-License-Identifier: GPL-3.0-or-later
"""Version of the runs (phase 5 of docs/auditoria/2026-11-11_cadeia/PLANO.md): for each
parametros.json under the given folders (and each <csv>.origem.json, see confere_origem), compares the sha256 recorded for the inputs with the
current files of the case, and says which inputs changed since the run. A run whose inputs
changed is to be redone before its numbers go to an article.

The case is argumentos.caso (the repository root when absent); recorded paths are relative to
the case, except those with a folder (code, relative to the repository root). A path recorded on
another machine (/home/lucas/...) is mapped to this repository by its part after the repository
name, when that exists here.

Usage: python3 ferramentas/versoes.py [FOLDER ...] > versoes.csv   (default: execucoes resultados docs)
"""
import csv
import hashlib
import json
import pathlib
import sys

RAIZ = pathlib.Path(__file__).resolve().parent.parent


def _sha(caminho):
    return hashlib.sha256(caminho.read_bytes()).hexdigest()


def _local(caminho, raiz=RAIZ):
    """Path of this repository for a path recorded here or on another machine."""
    p = pathlib.Path(caminho)
    if p.exists():
        return p
    partes = p.parts
    for i, parte in enumerate(partes):
        if parte.startswith("Argonio") or parte.startswith("argonio"):
            candidato = raiz.joinpath(*partes[i + 1:])
            if candidato.exists():
                return candidato
    return None


def confere(arquivo, raiz=RAIZ):
    """(case, changed inputs, missing inputs) of one parametros.json."""
    d = json.loads(pathlib.Path(arquivo).read_text())
    caso_gravado = (d.get("argumentos") or {}).get("caso") or str(raiz)
    caso = _local(caso_gravado, raiz)
    mudados, ausentes = [], []
    for nome, sha in (d.get("sha256") or {}).items():
        base = raiz if "/" in nome else caso
        atual = base / nome if base is not None else None
        if atual is None or not atual.is_file():
            ausentes.append(nome)
        elif _sha(atual) != sha:
            mudados.append(nome)
    return caso_gravado, mudados, ausentes


def confere_origem(arquivo, raiz=RAIZ):
    """Files that changed since a CSV was written, from its <csv>.origem.json
    (proveniencia.grava_origem): the CSV itself, the code (paths relative to the repository) and
    the inputs of each case (as CASE/name). Missing files count as changed."""
    arquivo = pathlib.Path(arquivo)
    d = json.loads(arquivo.read_text())
    mudados = []
    csv_ = arquivo.with_name(d["csv"])
    if not csv_.is_file() or _sha(csv_) != d["sha256_csv"]:
        mudados.append(d["csv"])
    if "codigo" in d:   # record without a case: code and files read, relative to the repository
        for nome, sha in {**d["codigo"].get("sha256", {}), **d["codigo"].get("sha256_entradas", {})}.items():
            atual = raiz / nome
            if not atual.is_file() or _sha(atual) != sha:
                mudados.append(nome)
    for caso, reg in d.get("casos", {}).items():
        for nome, sha in (reg.get("sha256") or {}).items():
            atual = raiz / nome
            if (not atual.is_file() or _sha(atual) != sha) and nome not in mudados:
                mudados.append(nome)
        for nome, sha in (reg.get("sha256_entradas") or {}).items():
            atual = raiz / "casos" / caso / nome
            if not atual.is_file() or _sha(atual) != sha:
                mudados.append(f"{caso}/{nome}")
    return mudados


def csv_sem_registro(pasta, raiz=RAIZ):
    """CSVs under `pasta` without <csv>.origem.json beside them (paths relative to `raiz`), except
    the literature data (docs/literatura), which are transcriptions and not results."""
    saida = []
    for csv_ in sorted(pathlib.Path(pasta).glob("**/*.csv")):
        rel = csv_.relative_to(raiz)
        if rel.parts[:2] == ("docs", "literatura"):
            continue
        if not csv_.with_name(csv_.name + ".origem.json").exists():
            saida.append(str(rel))
    return saida


def main(pastas):
    w = csv.writer(sys.stdout, delimiter=";")
    w.writerow(["rodada", "commit", "caso", "situacao", "entradas_mudadas", "entradas_ausentes"])
    for pasta in pastas:
        for arquivo in sorted((RAIZ / pasta).glob("**/parametros.json")):
            d = json.loads(arquivo.read_text())
            caso, mudados, ausentes = confere(arquivo)
            situacao = "refazer" if mudados else ("sem_entrada_local" if ausentes else "atual")
            w.writerow([arquivo.parent.relative_to(RAIZ), (d.get("commit") or "")[:7], caso, situacao,
                        " ".join(mudados), " ".join(ausentes)])
        for arquivo in sorted((RAIZ / pasta).glob("**/*.origem.json")):
            d = json.loads(arquivo.read_text())
            commit = next((r.get("commit") or "" for r in d.get("casos", {}).values()), "")[:7]
            mudados = confere_origem(arquivo)
            w.writerow([arquivo.relative_to(RAIZ).with_suffix("").with_suffix(""), commit,
                        " ".join(d.get("casos", {})), "refazer" if mudados else "atual", " ".join(mudados), ""])
        for csv_ in csv_sem_registro(RAIZ / pasta):
            w.writerow([csv_, "", "", "sem_registro", "", ""])


if __name__ == "__main__":
    main(sys.argv[1:] or ["execucoes", "resultados", "docs"])
