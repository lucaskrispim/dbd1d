# SPDX-License-Identifier: GPL-3.0-or-later
"""Variants of casos/amonia_sun2022 (Sun et al. 2022) for the sensitivity runs of the
ammonia article (docs/campanha_amonia/onda3/PLANO.md):

- amonia_sun2022_vv: the N2-N2 V-V block corrected by ferramentas/vv_sun.py (falling form of
  Capitelli et al. 2000, eq. 3.69, and detailed balance; docs/auditoria/2026-11-10/ERROS.md
  A5, A7); the base case keeps the published block;
- amonia_sun2022_nh_congelado: NH + H2 + M -> NH3 + M with its coefficient fixed at the value
  of 373 K, (373/300) exp(1700/373), at every T_g.

Only kinet.inp changes, in heavy particle reactions: the cross sections, the tables and the
transport are those of the base case, copied. Then the chemistry the model reads
(zdplaskin_m.F90 and quimica.json) is regenerated with ferramentas/caso.py (needs ZDPlasKin).

Usage: python3 ferramentas/casos_sun.py [--casos CASES_DIR] [NAME ...]
"""
import argparse
import pathlib
import re
import shutil
import subprocess
import sys

from vv_sun import corrige as corrige_vv

RAIZ = pathlib.Path(__file__).resolve().parent.parent
BASE = "amonia_sun2022"
# files of the base that the variants share: everything except the chemistry
COPIADOS = ("bolsigdb.dat", "bolsigdb.log", "bolsigdb_original.dat", "composicao.txt", "dados_fortran.txt",
            "gas.json", "reducao_secoes.txt", "tabela_eedf.dat", "tabela_energia.dat", "tabela_energia_tg.dat",
            "tabela_energia_tg_conferencia.dat", "tabelas_monotonia.txt", "transporte.json")
_NH = re.compile(r"^(NH \+ H2 \+ @M => NH3 \+ @M +! 2\.50d-35\*)\(Tgas/300\.0d0\)\*exp\(1700\.0d0/Tgas\)", re.M)


def _congela_nh(texto):
    novo, n = _NH.subn(r"\1(373.0d0/300.0d0)*exp(1700.0d0/373.0d0)", texto)
    if n != 1:
        raise ValueError(f"NH + H2 + M of Sun found {n} times in kinet.inp, expected 1")
    return novo


RECEITAS = {"amonia_sun2022_vv": corrige_vv, "amonia_sun2022_nh_congelado": _congela_nh}


def monta(nome, casos=RAIZ / "casos"):
    """kinet.inp of the variant nome, from the kinet.inp of the base case."""
    if nome not in RECEITAS:
        raise ValueError(f"unknown case {nome}: {', '.join(RECEITAS)}")
    return RECEITAS[nome]((pathlib.Path(casos) / BASE / "kinet.inp").read_text())


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("nomes", nargs="*", default=list(RECEITAS), metavar="NAME", help="cases (default: all)")
    p.add_argument("--casos", type=pathlib.Path, default=RAIZ / "casos", help="folder of the cases")
    a = p.parse_args()
    for nome in a.nomes:
        try:
            kinet = monta(nome, a.casos)
        except ValueError as erro:
            sys.exit(str(erro))
        destino = a.casos / nome
        destino.mkdir(parents=True, exist_ok=True)
        for arquivo in COPIADOS:
            shutil.copy2(a.casos / BASE / arquivo, destino / arquivo)
        (destino / "kinet.inp").write_text(kinet)
        for acao in ("zdplaskin", "export"):
            subprocess.run([sys.executable, str(RAIZ / "ferramentas" / "caso.py"), acao, str(destino)], check=True)
        print(f"{destino}: done")


if __name__ == "__main__":
    main()
