# SPDX-License-Identifier: GPL-3.0-or-later
"""Métricas de várias execuções lado a lado, num instante, com a diferença
relativa de cada uma para a última (a referência).

Uso: python3 -I scripts/metricas.py --t T DIR [DIR ...] REFERENCIA

Métricas: carga e potencial nas superfícies, n_e e o íon principal do caso (Ar+
no argônio) máximos e integrados no gap e as outras espécies integradas
(convergencia.py), posição da frente (maior x com n_e > 1e12 cm-3), posição do
pico do íon principal e a menor célula da malha.
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
            # frente do lado do cátodo, como em convergencia.py; nan sem n_e > 1e12
            frente = observaveis_frente(x, [b - a for a, b in zip(faces[:-1], faces[1:])], ne, ar,
                                        x_ref=x[k])["frente (m)"]
            return {"frente (um)": frente * 1e6, f"pico {rotulo(ion)} em (um)": x[k] * 1e6,
                    "menor celula (um)": min(b - a for a, b in zip(faces[:-1], faces[1:])) * 1e6}
    raise SystemExit(f"{d}: instante {t} não encontrado")


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
    print(f"t = {a.t:.3e} s; diferença relativa para {a.dirs[-1].name}")
    print(f"{'':22s}" + "".join(f"{d.name:>24s}" for d in a.dirs))
    for nome in ref:
        linha = f"{nome:22s}"
        for v in tabela:
            dif = (v[nome] - ref[nome]) / abs(ref[nome]) if ref[nome] != 0 else float("nan")
            linha += f"{v[nome]:13.4e} ({dif:+7.2%})"
        print(linha)


if __name__ == "__main__":
    main()
