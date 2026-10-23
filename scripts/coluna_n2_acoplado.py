# SPDX-License-Identifier: GPL-3.0-or-later
"""Test of a mechanism for the high-excitation disagreement
(docs/coluna_positiva_n2/README.md), fourth test: the
electron kinetics coupled to the vibrational distribution of N2(X) up to v = 45
(ferramentas/acoplamento_vibracional.py). For one condition, iteration k:
  1. the vibrational distribution at the end of the residence time, at the
     sustaining E/N of iteration k - 1 (k = 0: the ladder without coupling,
     casos/n2_amonia_v45) and n_e from the current;
  2. the coupled case casos/n2_amonia_acoplado/ID_itk: bolsigdb with the copies
     for N2(v1..v45) and the superelastic inverse, kinet.inp with the reactions
     from each level, composicao.txt with the populations of step 1, T_g nodes at
     T_g -+ 10 K;
  3. the tables of BOLSIG+ and quimica.json (ferramentas/caso.py; ZDPLASKIN_DIR
     where the third-party programs are);
  4. the sustaining E/N, Tv and [N]/[N2] with the coupled case.

From k = 2 on the E/N that builds the distribution is relaxed, the mean of the
input and of the result of iteration k - 1 (the plain iteration oscillates at 530
Pa). VARIANTE sets the wall probabilities as in scripts/coluna_n2_lote.py. MODO
is a step of the decomposition of the coupling: alvo (N2(v) as electron targets
with the cross sections of N2 as they are, no superelastic), deslocado (the
shifted cross sections, no superelastic) or completo (the default, both); meio
(half the shift) and limiar01 (minimum shifted threshold 0.1 eV instead of 0.3)
are tests of sensitivity of completo.

Usage: python3 scripts/coluna_n2_acoplado.py ID K SAIDA_DIR [VARIANTE [MODO]]
Writes SAIDA_DIR/ID[_VARIANTE][_MODO]_itK.json.
"""
import json
import pathlib
import shutil
import subprocess
import sys

import numpy as np

RAIZ = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "dbd1d"))
sys.path.insert(0, str(RAIZ / "scripts"))
sys.path.insert(0, str(RAIZ / "ferramentas"))
import acoplamento_vibracional as av  # noqa: E402
from coluna import Coluna0D  # noqa: E402
from coluna_n2 import GAMAS, condicoes, treanor  # noqa: E402
from coluna_n2_en import CORRENTE_MA  # noqa: E402
from coluna_n2_lote import gamas as gamas_da  # noqa: E402

V45 = RAIZ / "casos" / "n2_amonia_v45"
BASE = RAIZ / "casos" / "n2_amonia"
DESTINO = RAIZ / "casos" / "n2_amonia_acoplado"
VMAX = 45


def resultado(caso, cond, en, gamas=GAMAS):
    """Tv, [N]/[N2], populations {v: fraction} at E/N with n_e from the current."""
    c = Coluna0D(caso, cond["p"], cond["tg"], cond["r"], gamas=gamas, estrito=False)
    i_a = CORRENTE_MA[cond["id"]] * 1e-3
    _, y = c.integra(en, c.ne_da_corrente(i_a, en), cond["t"], n_saidas=3)
    nomes = [s.upper() for s in c.especies]
    n0 = c.densidades(y[-1])[[s.upper() for s in c.q.especies].index("N2")]
    vib = np.r_[n0, [y[-1, nomes.index(f"N2(V{k})")] for k in range(1, VMAX + 1)]]
    tv = treanor(vib[:5], cond["tg"])
    return tv, y[-1, nomes.index("N")] / n0, {v: float(x / vib.sum()) for v, x in enumerate(vib)}


# (fraction of E_v in the shift, superelastic, minimum shifted threshold in eV)
MODOS = {"completo": (1.0, True, av.LIMIAR_MINIMO), "deslocado": (1.0, False, av.LIMIAR_MINIMO),
         "alvo": (0.0, False, av.LIMIAR_MINIMO), "meio": (0.5, True, av.LIMIAR_MINIMO),
         "limiar01": (1.0, True, 0.1)}


def monta(pasta, cond, populacoes, modo="completo"):
    deslocar, superelastica, minimo = MODOS[modo]
    pasta.mkdir(parents=True, exist_ok=True)
    banco_base = (BASE / "bolsigdb.dat").read_text()
    (pasta / "bolsigdb.dat").write_text(av.banco_acoplado(banco_base, VMAX, deslocar, superelastica, minimo))
    (pasta / "kinet.inp").write_text(av.kinet_acoplado((V45 / "kinet.inp").read_text(), VMAX, banco_base, deslocar, minimo))
    (pasta / "composicao.txt").write_text(av.composicao({v: f for v, f in populacoes.items() if f > 1e-14}))
    (pasta / "t_nos.txt").write_text(f"{cond['tg'] - 10:g}\n{cond['tg'] + 10:g}\n")
    (pasta / "en_max.txt").write_text("1000\n")
    for nome in ("gas.json", "transporte.json"):
        shutil.copy(V45 / nome, pasta / nome)
    for acao in ("zdplaskin", "tabelas", "exporta"):
        r = subprocess.run([sys.executable, str(RAIZ / "ferramentas" / "caso.py"), acao, str(pasta)],
                           capture_output=True, text=True)
        if r.returncode:
            sys.exit(f"{pasta}: {acao} failed:\n{r.stdout[-3000:]}{r.stderr[-3000:]}")


def main():
    cid, k, saida = sys.argv[1], int(sys.argv[2]), pathlib.Path(sys.argv[3])
    variante = sys.argv[4] if len(sys.argv) > 4 else "base"
    modo = sys.argv[5] if len(sys.argv) > 5 else "completo"
    if modo not in MODOS:
        sys.exit(f"MODO must be one of {', '.join(MODOS)}, not {modo}")
    g = gamas_da(variante)
    nome = cid if variante == "base" else f"{cid}_{variante}"
    nome += "" if modo == "completo" else f"_{modo}"
    saida.mkdir(parents=True, exist_ok=True)
    cond = next(c for c in condicoes() if c["id"] == cid)
    i_a = CORRENTE_MA[cid] * 1e-3
    if k == 0:
        anterior = V45
        en_anterior = Coluna0D(V45, cond["p"], cond["tg"], cond["r"], gamas=g, estrito=False).en_sustentado(i_a, cond["t"])
    else:
        anterior = DESTINO / f"{nome}_it{k - 1}"
        previo = json.loads((saida / f"{nome}_it{k - 1}.json").read_text())
        en_anterior = previo["en"] if k == 1 else 0.5 * (previo["en"] + previo["en_entrada"])
    _, _, populacoes = resultado(anterior, cond, en_anterior, g)
    pasta = DESTINO / f"{nome}_it{k}"
    monta(pasta, cond, populacoes, modo)
    c = Coluna0D(pasta, cond["p"], cond["tg"], cond["r"], gamas=g, estrito=False)
    en = c.en_sustentado(i_a, cond["t"])
    tv, dis, pops = resultado(pasta, cond, en, g)
    dados = dict(id=cid, variante=variante, modo=modo, iteracao=k, en_entrada=en_anterior, en=en, tv=tv, dis=dis,
                 en_medido=cond["en"], fracao_v_maior_10=sum(f for v, f in pops.items() if v > 10))
    (saida / f"{nome}_it{k}.json").write_text(json.dumps(dados, indent=1))
    print(json.dumps(dados), flush=True)


if __name__ == "__main__":
    main()
