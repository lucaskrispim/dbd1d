# SPDX-License-Identifier: GPL-3.0-or-later
"""Self-consistent coupled solution of one condition of the N2 positive column
(docs/coluna_positiva_n2), by a root method instead of the fixed-point iteration
of scripts/coluna_n2_acoplado.py, which does not converge at 530 Pa.

The unknown is the sustaining E/N, x. For each x (inner loop,
acoplamento_vibracional.ponto_fixo): the vibrational distribution at the end of the
residence time, at x and n_e from the current, with the coupled case built from
the previous distribution, until the distribution stops changing (largest change of
ln of the populations above 1e-14 below TOL). The first x starts from the ladder
without coupling (casos/n2_amonia_v45), the next ones from the distribution of the
previous x. Then the electron balance r(x) = (production - loss)/loss with the last
case, and phi(x), the E/N that this case sustains. Outer loop
(acoplamento_vibracional.raiz_autoconsistente): from x0, the sustaining E/N of the
ladder, steps along phi and then the secant until r changes sign, then Brent.

VARIANTE and MODO as in scripts/coluna_n2_acoplado.py. ANTERIOR: a *_raiz.json of an
earlier solution of the same condition; the search starts from its root (x0) and
from the distribution of that point, instead of from the ladder (to redo the roots
after a change of the coupling). Only the last iteration of each x is kept on disk.
Usage: python3 scripts/coluna_n2_raiz.py ID SAIDA_DIR [VARIANTE [MODO [ANTERIOR]]]
Writes SAIDA_DIR/NAME_raiz.json (the solution and every x evaluated) and the
cases in casos/n2_amonia_acoplado/NAME_raiz/.
"""
import json
import pathlib
import shutil
import sys

import numpy as np

RAIZ = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "scripts"))
import coluna_n2_acoplado as ca  # noqa: E402
from coluna_n2_acoplado import av, Coluna0D, condicoes, CORRENTE_MA, gamas_da  # noqa: E402
import proveniencia  # noqa: E402

CODIGO = proveniencia.codigo_do_script(RAIZ, pathlib.Path(__file__))


EXTRAS = ("casos/n2_amonia/bolsigdb.dat", "docs/coluna_positiva_n2/condicoes.csv",
          "docs/literatura/dados/gordiets1995/n2_puro.csv")


def versao():
    """Provenance record of the root search: commit, hashes of the code and of the inputs of
    casos/n2_amonia_v45 and of the other files the search reads (the base bank of the coupled
    cases, the measured conditions), files changed outside a commit."""
    return proveniencia.registro(RAIZ, CODIGO, RAIZ / "casos" / "n2_amonia_v45", extras=EXTRAS)

TOL = 1e-3          # inner loop: largest change of ln(population)
MAX_IT = 30         # inner loop: at 530 Pa the change falls by about 0.58 per iteration
XTOL = 0.05         # outer loop: Td
X_MIN = 5.0         # Td


def main():
    registro = proveniencia.exige_arvore_limpa(versao())   # fail before the long search
    cid, saida = sys.argv[1], pathlib.Path(sys.argv[2])
    variante = sys.argv[3] if len(sys.argv) > 3 else "base"
    modo = sys.argv[4] if len(sys.argv) > 4 else "completo"
    if modo not in ca.MODOS:
        sys.exit(f"MODO must be one of {', '.join(ca.MODOS)}, not {modo}")
    g = gamas_da(variante)
    nome = cid + ("" if variante == "base" else f"_{variante}") + ("" if modo == "completo" else f"_{modo}")
    cond = next(c for c in condicoes() if c["id"] == cid)
    i_a = CORRENTE_MA[cid] * 1e-3
    base = ca.DESTINO / f"{nome}_raiz"
    saida.mkdir(parents=True, exist_ok=True)
    pontos = []
    estado = {"pops": None}
    anterior = json.loads(pathlib.Path(sys.argv[5]).read_text()) if len(sys.argv) > 5 else None
    if anterior is not None:
        if not anterior.get("convergiu"):
            sys.exit(f"{sys.argv[5]}: the earlier solution did not converge")
        ponto = next(pt for pt in anterior["pontos"] if pt["en"] == anterior["en"])
        estado["pops"] = np.array(ponto["populacoes"])

    def avalia(x):
        pasta_x = base / f"en{x:.4f}"
        if estado["pops"] is None:
            _, _, pops = ca.resultado(ca.V45, cond, x, g)
            estado["pops"] = np.array([pops[v] for v in range(ca.VMAX + 1)])
        feitos = []

        def mapa(p):
            pasta = pasta_x / f"it{len(feitos)}"
            ca.monta(pasta, cond, {v: float(f) for v, f in enumerate(p)}, modo)
            feitos.append(pasta)
            _, _, pops = ca.resultado(pasta, cond, x, g)
            if len(feitos) > 1:
                shutil.rmtree(feitos[-2])
            return np.array([pops[v] for v in range(ca.VMAX + 1)])

        p, n, mudancas = av.ponto_fixo(mapa, estado["pops"], tol=TOL, max_it=MAX_IT)
        estado["pops"] = p
        c = Coluna0D(feitos[-1], cond["p"], cond["tg"], cond["r"], gamas=g, estrito=False)
        q, perda = c.balanco_eletrons(x, i_a, cond["t"])
        try:
            phi = c.en_sustentado(i_a, cond["t"], faixa=(2.0, 300.0))
        except ValueError:
            phi = None
        tv, dis, pops = ca.resultado(feitos[-1], cond, x, g)
        pontos.append(dict(en=x, r=(q - perda) / perda, phi=phi, iteracoes=n, mudancas=mudancas, tv=tv, dis=dis,
                           fracao_v_maior_0=1.0 - pops[0], fracao_v_maior_10=sum(f for v, f in pops.items() if v > 10),
                           populacoes=[pops[v] for v in range(ca.VMAX + 1)], caso=str(feitos[-1])))
        print(json.dumps({k: v for k, v in pontos[-1].items() if k != "populacoes"}), flush=True)
        (saida / f"{nome}_raiz.json").write_text(json.dumps(dict(id=cid, variante=variante, modo=modo,
                                                                   versao=registro, pontos=pontos), indent=1))
        return pontos[-1]["r"], phi

    if anterior is not None:
        x0 = anterior["en"]
    else:
        x0 = Coluna0D(ca.V45, cond["p"], cond["tg"], cond["r"], gamas=g, estrito=False).en_sustentado(i_a, cond["t"])
    try:
        x, _ = av.raiz_autoconsistente(avalia, x0, X_MIN, xtol=XTOL)
        final = next(pt for pt in pontos if pt["en"] == x) if any(pt["en"] == x for pt in pontos) else None
        if final is None:
            avalia(x)
            final = pontos[-1]
        resultado = dict(convergiu=True, en=x, tv=final["tv"], dis=final["dis"], r=final["r"])
    except (ValueError, RuntimeError) as erro:
        resultado = dict(convergiu=False, erro=str(erro))
    (saida / f"{nome}_raiz.json").write_text(json.dumps(dict(id=cid, variante=variante, modo=modo, en_medido=cond["en"],
                                                               x0=x0, **resultado, versao=registro, pontos=pontos), indent=1))
    print(json.dumps(dict(id=cid, variante=variante, modo=modo, **resultado)), flush=True)


if __name__ == "__main__":
    main()
