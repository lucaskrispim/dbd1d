# SPDX-License-Identifier: GPL-3.0-or-later
"""Quantities of several runs side by side, at one time, with the relative
difference of each from the last one (the reference).

Usage: python3 -I scripts/metricas.py --t T DIR [DIR ...] REFERENCE

Quantities: charge and potential at the surfaces, n_e and the main ion of the
case (Ar+ in argon), maximum and integrated over the gap, and the other species
integrated (convergencia.py), position of the front (cathode side edge with
n_e > 1e12 cm-3), position of the peak of the main ion and the smallest cell.
"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from convergencia import observaveis_frente, perfil, superficie  # noqa: E402
from resumo import blocos, especies_do_caso, rotulo  # noqa: E402


def extras(d, t):
    for tb, nomes, linhas in blocos(d / "perfil.dat"):
        if abs(tb - t) <= 1e-6 * t:
            interior = linhas[1:-1]
            x = [l[0] for l in interior]
            ne = [l[nomes.index("E")] for l in interior]
            ion = especies_do_caso(d)[1]
            ar = [l[nomes.index(ion)] for l in interior]
            faces = [0.0]
            for c in x:
                faces.append(2.0 * c - faces[-1])
            k = max(range(len(ar)), key=lambda i: ar[i])
            # front on the cathode side, as in convergencia.py; nan without n_e > 1e12
            frente = observaveis_frente(x, [b - a for a, b in zip(faces[:-1], faces[1:])], ne, ar,
                                        x_ref=x[k])["front (m)"]
            return {"front (um)": frente * 1e6, f"peak {rotulo(ion)} at (um)": x[k] * 1e6,
                    "smallest cell (um)": min(b - a for a, b in zip(faces[:-1], faces[1:])) * 1e6}
    raise SystemExit(f"{d}: time {t} not found")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--t", type=float, required=True)
    p.add_argument("dirs", nargs="+", type=pathlib.Path)
    a = p.parse_args()
    tabela = []
    for d in a.dirs:
        v = superficie(d, a.t)
        v.update(perfil(d, a.t)[1])
        v.update(extras(d, a.t))
        tabela.append(v)
    ref = tabela[-1]
    print(f"t = {a.t:.3e} s; relative difference from {a.dirs[-1].name}")
    print(f"{'':22s}" + "".join(f"{d.name:>24s}" for d in a.dirs))
    for nome in ref:
        linha = f"{nome:22s}"
        for v in tabela:
            dif = (v[nome] - ref[nome]) / abs(ref[nome]) if ref[nome] != 0 else float("nan")
            linha += f"{v[nome]:13.4e} ({dif:+7.2%})"
        print(linha)


if __name__ == "__main__":
    main()
