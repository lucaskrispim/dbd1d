# SPDX-License-Identifier: GPL-3.0-or-later
"""Summary of the self-consistent coupled solutions of the N2 positive column
(scripts/coluna_n2_raiz.py, remoto/raiz_coluna.sh): for each condition and vibrational
wall probability, the sustaining E/N of the ladder up to v = 45 without coupling
(falta/escada_v45.csv, chemistry A) and of the three coupling steps, the share of
the fall of E/N due to each step, and T_v, dissociation and fractions in v >= 1 and
v > 10 at the root of the full coupling. Chains that did not converge are left
empty.

Usage: python3 scripts/coluna_n2_raiz_resumo.py RAIZ_DIR ESCADA_CSV [ev]
Writes RAIZ_DIR/resumo.csv, or RAIZ_DIR/resumo_ev.csv with the chains of the modes
with e-V from the excited levels (MODO_ev in scripts/coluna_n2_acoplado.py).
"""
import csv
import json
import pathlib
import sys

EN_MEDIDO = {"L20_10mA": 118.0, "L20_20mA": 108.0, "L20_50mA": 90.0, "B21_230Pa": 94.0,
             "B21_530Pa": 48.0, "G95_15mA": 66.7, "G95_30mA": 63.0, "G95_80mA": 69.9}
VARIANTES = ("base", "vib_gordiets")
MODOS = ("alvo", "deslocado", "completo")


def nome(cid, variante, modo, ev=False):
    """Name of the chain as in scripts/coluna_n2_raiz.py (base and completo are omitted)."""
    modo = f"{modo}_ev" if ev else modo
    partes = [cid] + ([variante] if variante != "base" else []) + ([modo] if modo != "completo" else [])
    return "_".join(partes)


def solucao(pasta, n):
    arq = pathlib.Path(pasta) / f"{n}_raiz.json"
    if not arq.exists():
        return None
    d = json.loads(arq.read_text())
    return d if d.get("convergiu") else None


def fracoes_etapas(escada, alvo, deslocado, completo):
    """Share of the fall of E/N from the ladder to the full coupling due to the
    targets, the shifted thresholds and the superelastic collisions."""
    if None in (escada, alvo, deslocado, completo):
        return None
    queda = escada - completo
    return (escada - alvo) / queda, (alvo - deslocado) / queda, (deslocado - completo) / queda


def linha(pasta, cid, variante, en_escada, ev=False):
    en = {}
    for modo in MODOS:
        d = solucao(pasta, nome(cid, variante, modo, ev))
        en[modo] = d["en"] if d else None
    comp = solucao(pasta, nome(cid, variante, "completo", ev))
    raiz = next(p for p in comp["pontos"] if p["en"] == comp["en"]) if comp else {}
    f = fracoes_etapas(en_escada, en["alvo"], en["deslocado"], en["completo"])
    return {"id": cid, "variante": variante, "en_medido": EN_MEDIDO.get(cid), "en_escada": en_escada,
            "en_alvo": en["alvo"], "en_deslocado": en["deslocado"], "en_completo": en["completo"],
            "desvio": en["completo"] / EN_MEDIDO[cid] - 1 if comp and cid in EN_MEDIDO else None,
            "fracao_alvo": f[0] if f else None, "fracao_deslocado": f[1] if f else None,
            "fracao_superelastica": f[2] if f else None,
            "tv": comp["tv"] if comp else None, "dis": comp["dis"] if comp else None,
            "fracao_v_maior_0": raiz.get("fracao_v_maior_0"), "fracao_v_maior_10": raiz.get("fracao_v_maior_10")}


def escada(arq):
    with open(arq) as f:
        return {(r["id"], r["variante"]): float(r["en"]) for r in csv.DictReader(f)
                if r["caso"] == "n2_amonia_v45"}


def main():
    pasta, esc = pathlib.Path(sys.argv[1]), escada(sys.argv[2])
    ev = sys.argv[3:] == ["ev"]
    linhas = [linha(pasta, cid, v, esc[(cid, v)], ev) for v in VARIANTES for cid in EN_MEDIDO]
    with open(pasta / ("resumo_ev.csv" if ev else "resumo.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(linhas[0]))
        w.writeheader()
        for r in linhas:
            w.writerow({k: ("" if x is None else f"{x:.6g}" if isinstance(x, float) else x) for k, x in r.items()})


if __name__ == "__main__":
    main()
