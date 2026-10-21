# SPDX-License-Identifier: GPL-3.0-or-later
"""Fluxos por reação de uma execução, como a análise de caminhos de Sun et al.
2022: a taxa de cada reação reconstruída em cada estado gravado (os mesmos
coeficientes do modelo), integrada no gap e no tempo (trapézio), e a fração da
produção e do consumo de cada espécie que cabe a cada reação.

Uso: python3 scripts/fluxos.py DIR [DIR ...] [--especies NH,NH3,...] [--n 6]
Vários DIR são trechos seguidos da mesma execução (retomadas), em ordem; o
instante repetido na emenda conta uma vez. A resolução no tempo é a da
gravação (--dt-saida): ela precisa acompanhar o pulso.
"""
import argparse
import json
import pathlib
import sys

import numpy as np

AQUI = pathlib.Path(__file__).resolve().parent
RAIZ = AQUI.parent
sys.path.insert(0, str(AQUI))
sys.path.insert(0, str(RAIZ / "prototipo"))
from resumo import blocos  # noqa: E402

PADRAO = ("N(2D)", "NH", "NH2", "NH3", "NH4^+", "H")


def modelo_da_execucao(d):
    """Modelo montado com os argumentos gravados no parametros.json; o caso de
    outra máquina vira o casos/NOME daqui."""
    import simula
    a = argparse.Namespace(**json.loads((d / "parametros.json").read_text())["argumentos"])
    caso = pathlib.Path(a.caso)
    if not caso.exists():
        caso = RAIZ / "casos" / caso.name if caso.parent.name == "casos" else RAIZ
    a.caso = caso
    if a.transporte and not pathlib.Path(a.transporte).exists():
        a.transporte = d / pathlib.Path(a.transporte).name if (d / pathlib.Path(a.transporte).name).exists() else None
    a.transporte = pathlib.Path(a.transporte) if a.transporte else None
    faces = np.loadtxt(a.faces) if a.faces else None
    return simula.monta_modelo(a, caso, faces)


def estados(dirs, md):
    """[(t, n, y)] de todos os registros dos trechos, em ordem, sem repetir a emenda."""
    saida, vistos = [], set()
    for d in dirs:
        sup = {}
        for linha in (d / "superficie.dat").read_text().splitlines()[1:]:
            v = [float(x) for x in linha.split()]
            sup[round(v[0], 15)] = (v[4], v[5])
        for t, nomes, linhas in blocos(d / "perfil.dat"):
            chave = round(t, 15)
            if chave in vistos:
                continue
            vistos.add(chave)
            interior = np.array(linhas[1:-1])
            n = np.array([interior[:, nomes.index(nome)] for nome in md.nomes])
            partes = [n]
            if md.energia:
                partes.append(interior[:, nomes.index("n_eps")][None, :])
            if md.aquecimento:
                partes.append(interior[:, nomes.index("Temperature")][None, :])
            y = np.concatenate([np.vstack(partes).ravel(), list(sup[chave])])
            saida.append((t, n, y))
    return sorted(saida, key=lambda e: e[0])


def taxas_no_gap(md, t, y):
    """Taxa de cada reação integrada no gap (cm-3 m/s) no estado y."""
    n, s_esq, s_dir = md.separa(y)
    _, campo = md.campo(t, n, s_esq, s_dir)
    campo_cel = md.campo_nas_celulas(campo)
    n_eps = md.energia_do_estado(y) if md.energia else None
    t_gas = md.temperatura_do_estado(y) if md.aquecimento else None
    _, _, temp, k_bolsig, e_n, _ = md.coeficientes(campo_cel, n, n_eps, t_gas)
    tg = t_gas if md.aquecimento else np.full(md.m, md.t_gas)
    k = md.q.coeficientes(k_bolsig, tg, temp[md.i_e], e_n,
                          md.densidade_neutra(n) if md.q.usa_any_neutral else None)
    return md.q.taxas(md.densidades_completas(n), k) @ md.h


def taxas_integradas(dirs):
    """(modelo, instantes, integral no tempo da taxa no gap de cada reação, cm-3 m)."""
    dirs = [pathlib.Path(d) for d in dirs]
    md = modelo_da_execucao(dirs[0])
    lista = estados(dirs, md)
    t = np.array([e[0] for e in lista])
    r = np.array([taxas_no_gap(md, ti, y) for ti, _, y in lista])
    return md, t, np.trapezoid(r, t, axis=0)


def contribuicoes(q, integrais, especie):
    """([(reação, fração da produção)], [(reação, fração do consumo)]) da
    espécie, das maiores para as menores."""
    s = q.indice[especie.upper()]
    nu = q.estequiometria[s]
    lados = []
    for sinal in (1.0, -1.0):
        parte = np.where(sinal * nu > 0, sinal * nu * integrais, 0.0)
        total = parte.sum()
        ordem = np.argsort(parte)[::-1]
        lados.append([(q.reacoes[r], parte[r] / total) for r in ordem if parte[r] > 0] if total > 0 else [])
    return lados[0], lados[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dirs", nargs="+", type=pathlib.Path)
    p.add_argument("--especies", default=",".join(PADRAO))
    p.add_argument("--n", type=int, default=6)
    a = p.parse_args()
    md, t, integrais = taxas_integradas(a.dirs)
    print(f"{len(t)} estados de {t[0]:.6e} a {t[-1]:.6e} s")
    for especie in a.especies.split(","):
        prod, cons = contribuicoes(md.q, integrais, especie)
        s = md.q.indice[especie.upper()]
        liquido = md.q.estequiometria[s] @ integrais
        print(f"\n== {especie}  (produção líquida no período: {liquido:.4e} cm-3 m)")
        for titulo, lado in (("produção", prod), ("consumo", cons)):
            print(f"  {titulo}:")
            for nome, f in lado[:a.n]:
                print(f"    {100 * f:5.1f}%  {nome}")


if __name__ == "__main__":
    main()
