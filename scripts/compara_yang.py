# SPDX-License-Identifier: GPL-3.0-or-later
"""V3 de validacao/LEIAME.md: execuções de remoto/validacao_yang.sh (e as retomadas
de remoto/validacao_yang_pulso.sh) contra Yang et al. 2022
(docs/literatura/dados/yang2022/).

Da execução principal (registros a cada 1 µs): N, H e N2(A3) médios no gap e
no centro, e T_g médio. N e H do pulso n são os do instante logo antes do
pulso n + 1 (o artigo mede 100 µs depois de uma rajada de n pulsos; o
decaimento nesse intervalo é de poucos por cento, figs. 10 e 15). N2(A3):
máximo e mínimo de cada período, contra a envoltória medida de v = 0 e v = 1
(o mecanismo não resolve a vibração do N2(A)). Das retomadas: energia do pulso
E = ∫ V dQ em 240 ns, com Q = ε₀ ε_d (V − φ_esq) A / Δ e A = 7,2 cm².

Uso: python3 scripts/compara_yang.py PRINCIPAL [RETOMADAS ...] [--figura SAIDA.png]
"""
import argparse
import json
import pathlib
import re
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from resumo import blocos  # noqa: E402

RAIZ = pathlib.Path(__file__).resolve().parent.parent
DADOS = RAIZ / "docs" / "literatura" / "dados" / "yang2022"
EPS0 = 8.8541878128e-12
AREA = 7.2e-4
PERIODO = 1e-5


def le(f):
    return np.loadtxt(DADOS / f, delimiter=",", ndmin=2)


def series(d):
    """Instantes e, por espécie, (média no gap, centro); T_g médio."""
    t, med, cen, tg = [], {}, {}, []
    for tb, nomes, linhas in blocos(d / "perfil.dat"):
        a = np.array(linhas[1:-1])
        faces = [0.0]
        for c in a[:, 0]:
            faces.append(2.0 * c - faces[-1])
        h = np.diff(faces)
        k = int(np.argmin(np.abs(a[:, 0] - 0.5 * faces[-1])))
        t.append(tb)
        for e in ("N", "H", "N2(A3)"):
            if e in nomes:
                med.setdefault(e, []).append(float((a[:, nomes.index(e)] * h).sum() / h.sum()))
                cen.setdefault(e, []).append(a[k, nomes.index(e)])
        col = next((n for n in ("Temperature", "Tgas") if n in nomes), None)
        if col:
            tg.append(float((a[:, nomes.index(col)] * h).sum() / h.sum()))
    return (np.array(t), {e: np.array(v) for e, v in med.items()}, {e: np.array(v) for e, v in cen.items()},
            np.array(tg))


def energia(d):
    a = json.loads((d / "parametros.json").read_text())["argumentos"]
    sup = np.loadtxt(d / "superficie.dat", skiprows=1, ndmin=2)
    q = EPS0 * a["eps_dieletrico"] * (sup[:, 1] - sup[:, 2]) / a["espessura"] * AREA
    return float(np.trapezoid(sup[:, 1], q))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("principal", type=pathlib.Path)
    p.add_argument("retomadas", nargs="*", type=pathlib.Path)
    p.add_argument("--figura", type=pathlib.Path)
    a = p.parse_args()
    t, med, cen, tg = series(a.principal)
    n_pulsos = int(np.floor(t[-1] / PERIODO + 1e-9))
    pulsos = np.arange(1, n_pulsos + 1)
    antes = np.array([np.argmin(np.abs(t - n * PERIODO + 1e-6)) for n in pulsos])  # 1 µs antes do pulso n + 1
    gas = "N2+1%H2" if "1pc" in str(a.principal) else "N2"
    print(f"{a.principal.name}: {n_pulsos} pulsos ({gas})")
    medida_n = le("fig9_N_acumulado_N2.csv") if gas == "N2" else le("fig14_N_acumulado_1pcH2.csv")
    print(f"{'pulso':>5s} {'N méd':>9s} {'N centro':>9s} {'N medido':>9s} {'H méd':>9s} {'T_g (K)':>8s}")
    for n, nm in zip(medida_n[:, 0].astype(int), medida_n[:, 1]):
        if n <= n_pulsos:
            j = antes[n - 1]
            h = med.get("H", np.full(len(t), np.nan))[j]
            print(f"{n:5d} {med['N'][j]:9.2e} {cen['N'][j]:9.2e} {nm:9.2e} {h:9.2e} "
                  f"{tg[j] if len(tg) else float('nan'):8.1f}")
    if gas != "N2":
        hm = le("fig14_H_acumulado_1pcH2.csv")
        print("H medido (fig. 14(b)) contra o 1D, média no gap:")
        for tt, hv in hm[:, :2]:
            j = int(np.argmin(np.abs(t - tt * 1e-6)))
            print(f"  {tt:7.0f} µs: medido {hv:9.2e}, 1D {med['H'][j]:9.2e}")
    if "N2(A3)" in med:
        mx = [med["N2(A3)"][(t > (n - 1) * PERIODO) & (t <= n * PERIODO)].max() for n in pulsos]
        print(f"N2(A3) médio: máximo no pulso 5 {mx[min(4, n_pulsos - 1)]:.2e}, no último {mx[-1]:.2e} cm-3 "
              f"(medido v=0 + v=1: pico de ~1,2e14 perto de 50 µs em N2)")
    for r in a.retomadas:
        m = re.search(r"_p(\d+)$", r.name)
        print(f"energia do pulso {m.group(1) if m else '?'}: {energia(r) * 1e3:.2f} mJ")
    print("medido em N2: 3,48 mJ no pulso 2 a 3,08 no 150 (com refletidos); 2,2 a 2,0 mJ só o incidente")


if __name__ == "__main__":
    main()
