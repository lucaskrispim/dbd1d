# SPDX-License-Identifier: GPL-3.0-or-later
"""Vibrational distribution of N2(X) in the positive column against the state-to-state
models of Levaton et al. 2020 (fig. 11) and Gordiets et al. 1995 (fig. 13), digitized
in docs/literatura/dados (levaton2020.py, gordiets1995.py). A comparison between
models: neither VDF is measured.

- Levaton, 10 and 50 mA: their measured E/N and n_e (the n_e of their model, from
  their mobility) and T_g, the initial distribution a Boltzmann one at T_g (their
  t = 0), at t = 1e-6 s, 1e-5 s and the end of the column; chemistry A truncated at
  v = 8 and with the ladder up to v = 45, with their vibrational wall probability
  (4.5e-4).
- Gordiets, 30 and 80 mA: the steady state (0.1 s) at the measured E/N and n_e, the
  ladder up to v = 45 with the two vibrational wall probabilities, as fractions
  N2(v)/N2; their curves are for 0.1% O2, which changes only the levels above ~12.
  With the self-consistent coupled solutions (scripts/coluna_n2_raiz.py), their
  distributions too, if the JSON are given.

Usage: python3 scripts/coluna_n2_vdf.py SAIDA_DIR [RAIZ_DIR]
Writes vdf.csv.
"""
import csv
import json
import pathlib
import sys

import numpy as np

RAIZ = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "dbd1d"))
sys.path.insert(0, str(RAIZ / "scripts"))
from coluna import Coluna0D  # noqa: E402
from coluna_n2 import condicoes, energia_vib  # noqa: E402
from coluna_n2_lote import gamas  # noqa: E402

TEMPOS_L20 = {"L20_10mA": (1e-6, 1e-5, 4.3e-4), "L20_50mA": (1e-6, 1e-5, 3.5e-4)}


def niveis(c):
    nomes = [s.upper() for s in c.especies]
    return [(int(s[4:-1]), nomes.index(s)) for s in nomes if s.startswith("N2(V") and s[4:-1].isdigit()]


def boltzmann(c, ne):
    """Initial state: electrons and N2+ at ne, N2(v) in Boltzmann at T_g."""
    y = c.estado_inicial(ne)
    ks = niveis(c)
    pesos = np.exp(-energia_vib(np.array([0] + [v for v, _ in ks])) / c.t_gas)
    for (v, i), w in zip(ks, pesos[1:]):
        y[i] = c.n_gas * w / pesos.sum()
    return y


def vdf(c, y):
    """{v: density} with v = 0 the background; fractions are of the sum over v."""
    n0 = c.densidades(y)[[s.upper() for s in c.q.especies].index("N2")]
    return {0: n0, **{v: y[i] for v, i in niveis(c)}}


def main():
    saida = pathlib.Path(sys.argv[1])
    saida.mkdir(parents=True, exist_ok=True)
    linhas = []
    conds = {c["id"]: c for c in condicoes()}
    for cid, tempos in TEMPOS_L20.items():
        cond = conds[cid]
        for caso in ("n2_amonia", "n2_amonia_v45"):
            c = Coluna0D(RAIZ / "casos" / caso, cond["p"], cond["tg"], cond["r"], gamas=gamas("base"), estrito=False)
            t, y = c.integra(cond["en"], cond["ne"], max(tempos), n_saidas=400, y0=boltzmann(c, cond["ne"]))
            for alvo in tempos:
                k = int(np.argmin(np.abs(np.log(t / alvo))))
                d = vdf(c, y[k])
                for v, n in d.items():
                    linhas.append(dict(id=cid, caso=caso, parede="levaton", t=alvo, t_calculado=t[k], v=v, n=n,
                                       fracao=n / sum(d.values())))
        print(cid, "done", flush=True)
    for cid in ("G95_30mA", "G95_80mA"):
        cond = conds[cid]
        for variante, parede in (("base", "levaton"), ("vib_gordiets", "gordiets")):
            c = Coluna0D(RAIZ / "casos" / "n2_amonia_v45", cond["p"], cond["tg"], cond["r"], gamas=gamas(variante),
                         estrito=False)
            _, y = c.integra(cond["en"], cond["ne"], cond["t"], n_saidas=3)
            d = vdf(c, y[-1])
            for v, n in d.items():
                linhas.append(dict(id=cid, caso="n2_amonia_v45", parede=parede, t=cond["t"], t_calculado=cond["t"],
                                   v=v, n=n, fracao=n / sum(d.values())))
        print(cid, "done", flush=True)
    if len(sys.argv) > 2:
        for arq in sorted(pathlib.Path(sys.argv[2]).glob("G95_*_raiz.json")):
            d = json.loads(arq.read_text())
            if not d.get("convergiu") or d["modo"] != "completo":
                continue
            final = next(p for p in d["pontos"] if p["en"] == d["en"])
            parede = "gordiets" if d["variante"] == "vib_gordiets" else "levaton"
            for v, f in enumerate(final["populacoes"]):
                linhas.append(dict(id=d["id"], caso="acoplado", parede=parede, t=0.1, t_calculado=0.1, v=v, n=np.nan,
                                   fracao=f))
    with open(saida / "vdf.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(linhas[0]))
        w.writeheader()
        w.writerows(linhas)


if __name__ == "__main__":
    main()
