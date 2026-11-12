# SPDX-License-Identifier: GPL-3.0-or-later
"""Dissociation of N2 in the DC glow of Kumar and Ghosh 1991 (Chem. Phys. Lett. 179,
463; table 1 in docs/literatura/dados/kumar1991/tabela1.csv), with the 0D column
(dbd1d/coluna.py): Pyrex tube of radius 0.6 cm, 10 cm column, linear flow 2.9 m/s
(residence time 34.5 ms), 1 to 3 Torr, 30 to 80 mA. T_g is not measured; 400 K as
the authors, and 350 and 450 K for the range. For each condition and chemistry: the
sustaining E/N at the end of the residence time with n_e from the current, the mean
n_e against the measured axial one times <J0>/J0(0) = 2 J1(2.405)/2.405 = 0.432 (the
bell-shaped profile they report), and [N]/[N2] at half and at the end of the
residence time, against the measured value 20 cm downstream and 1.2 times it (their
measurement midway between the electrodes). Wall probabilities as scripts/coluna_n2.py;
the ladder up to v = 45 with the two vibrational walls.

Usage: python3 scripts/coluna_n2_kumar.py SAIDA_DIR
Writes kumar.csv.
"""
import csv
import pathlib
import sys

import numpy as np

RAIZ = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "dbd1d"))
sys.path.insert(0, str(RAIZ / "scripts"))
sys.path.insert(0, str(RAIZ / "ferramentas"))
import proveniencia  # noqa: E402
from coluna import Coluna0D  # noqa: E402
from coluna_n2 import QUIMICAS, n2_total  # noqa: E402
from coluna_n2_lote import gamas  # noqa: E402

TORR = 133.322
RAIO = 0.6
T_RES = 0.10 / 2.9
MEDIA_J0 = 0.43190      # 2 J1(2.405)/2.405
CASOS = [(q, "base") for q in QUIMICAS] + [("n2_amonia_v45", "base"), ("n2_amonia_v45", "vib_gordiets")]


def condicoes():
    with open(RAIZ / "docs" / "literatura" / "dados" / "kumar1991" / "tabela1.csv") as f:
        linhas = [l.split(",") for l in f if not l.startswith("#")]
    return [dict(p=float(a) * TORR, torr=float(a), i=float(b) * 1e-3, ne_eixo=float(c) * 1e10, medido=float(h) / 100.0)
            for a, b, c, d, e, f_, g, h in linhas]


def roda(caso, variante, cond, tg):
    c = Coluna0D(RAIZ / "casos" / caso, cond["p"], tg, RAIO, gamas=gamas(variante), estrito=False)
    en = c.en_sustentado(cond["i"], T_RES)
    ne = c.ne_da_corrente(cond["i"], en)
    t, y = c.integra(en, ne, T_RES, n_saidas=60)
    nomes = [s.upper() for s in c.especies]
    dis = [y[k, nomes.index("N")] / n2_total(c, y[k]) for k in range(len(t))]
    meio = float(np.interp(np.log(T_RES / 2), np.log(t), dis))
    return dict(en=en, ne=ne, dis_meio=meio, dis_fim=dis[-1])


def main():
    saida = pathlib.Path(sys.argv[1])
    saida.mkdir(parents=True, exist_ok=True)
    linhas = []
    for cond in condicoes():
        for caso, variante in CASOS:
            r = roda(caso, variante, cond, 400.0)
            faixa = [roda(caso, variante, cond, tg)["dis_fim"] for tg in (350.0, 450.0)]
            linha = dict(torr=cond["torr"], corrente_mA=cond["i"] * 1e3, quimica=caso, variante=variante, **r,
                         dis_fim_min=min(faixa + [r["dis_fim"]]), dis_fim_max=max(faixa + [r["dis_fim"]]),
                         ne_medio_medido=cond["ne_eixo"] * MEDIA_J0, medido=cond["medido"])
            linhas.append(linha)
            print(f"{cond['torr']:.0f} Torr {cond['i'] * 1e3:3.0f} mA {caso:14s} {variante:12s} E/N {r['en']:6.1f}  "
                  f"ne {r['ne']:.2e} (measured mean {linha['ne_medio_medido']:.2e})  [N]/[N2] {100 * r['dis_meio']:.3f}% "
                  f"(t/2) {100 * r['dis_fim']:.3f}% (end; Tg 350-450 K: {100 * linha['dis_fim_min']:.3f} to "
                  f"{100 * linha['dis_fim_max']:.3f})  measured {100 * cond['medido']:.2f}% "
                  f"(midway {120 * cond['medido']:.2f}%)", flush=True)
    with open(saida / "kumar.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(linhas[0]))
        w.writeheader()
        w.writerows(linhas)
    proveniencia.grava_origem(saida / "kumar.csv", __file__, ['n2_amonia', 'n2_tese', 'n2_exemplo', 'n2_amonia_v45'])  # code and inputs beside the CSV


if __name__ == "__main__":
    main()
