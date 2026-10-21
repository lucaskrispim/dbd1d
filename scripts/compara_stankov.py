# SPDX-License-Identifier: GPL-3.0-or-later
"""V1 de validacao/LEIAME.md: execuções de remoto/validacao_stankov.sh contra
Stankov et al. 2020 (docs/literatura/dados/stankov2020/).

Do superficie.dat: carga no eletrodo energizado Q = ε₀ ε_d (V − φ_esq) A / Δ,
corrente I = dQ/dt (diferença centrada), potência média P = (1/T) ∫ V I dt no
último período inteiro, tensão no gap U_gap = φ_esq − φ_dir e a tensão de
ruptura (|U_gap| no pico principal de corrente de cada meio período). Do
perfil.dat: n_e médio no gap e o seu máximo no período. Tudo no último
período, com t medido do início dele (o mesmo zero das figuras do artigo).

Uso: python3 scripts/compara_stankov.py DIR [DIR ...] [--figura SAIDA.png]
"""
import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from resumo import blocos  # noqa: E402

RAIZ = pathlib.Path(__file__).resolve().parent.parent
DADOS = RAIZ / "docs" / "literatura" / "dados" / "stankov2020"
EPS0 = 8.8541878128e-12
AREA = 7.26e-4  # m²
MEDIDA = {100: 0.559, 300: 0.935, 500: 1.535, 650: 2.474}  # W, fig. 7


def ultimo_periodo(d):
    """Grandezas elétricas no último período inteiro gravado."""
    a = json.loads((d / "parametros.json").read_text())["argumentos"]
    per = a["periodo"]
    sup = np.loadtxt(d / "superficie.dat", skiprows=1, ndmin=2)
    t, v, phi_e, phi_d = sup[:, 0], sup[:, 1], sup[:, 2], sup[:, 3]
    q = EPS0 * a["eps_dieletrico"] * (v - phi_e) / a["espessura"] * AREA
    i = np.gradient(q, t)
    k = int(np.floor(t[-1] / per + 1e-9))
    m = (t >= (k - 1) * per - 1e-12) & (t <= k * per + 1e-12)
    tt = t[m] - (k - 1) * per
    p = np.trapezoid(v[m] * i[m], t[m]) / per
    ugap = phi_e[m] - phi_d[m]
    rupturas = []
    for meio in (tt < 0.5 * per, tt >= 0.5 * per):
        j = np.argmax(np.abs(i[m][meio]))
        rupturas.append((tt[meio][j], abs(ugap[meio][j]), abs(i[m][meio][j])))
    ne = []
    for tb, nomes, linhas in blocos(d / "perfil.dat"):
        if (k - 1) * per - 1e-12 <= tb <= k * per + 1e-12:
            arr = np.array(linhas[1:-1])
            faces = [0.0]
            for c in arr[:, 0]:
                faces.append(2.0 * c - faces[-1])
            h = np.diff(faces)
            ne.append(float((arr[:, nomes.index("E")] * h).sum() / h.sum()) * 1e6)
    return {"periodo": k, "a": a, "t": tt, "I": i[m], "V": v[m], "Ugap": ugap, "P": p, "rupturas": rupturas,
            "ne_max": max(ne) if ne else np.nan}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dirs", nargs="+", type=pathlib.Path)
    p.add_argument("--figura", type=pathlib.Path)
    args = p.parse_args()
    print(f"{'execução':<18s} {'p (mbar)':>8s} {'U0 (kV)':>7s} {'per.':>4s} {'P (W)':>7s} {'medida':>7s} "
          f"{'desvio':>7s} {'t pico (µs)':>11s} {'I pico (mA)':>11s} {'U_rupt (V)':>10s} {'n_e máx (m-3)':>13s}")
    res = []
    for d in args.dirs:
        r = ultimo_periodo(d)
        pres = round(r["a"]["pressao"] / 100.0)
        med = MEDIDA.get(pres, np.nan)
        tp, up, ip = max(r["rupturas"], key=lambda x: x[2])
        print(f"{d.name:<18s} {pres:8d} {abs(r['a']['v_pico']) / 1e3:7.2f} {r['periodo']:4d} {r['P']:7.3f} {med:7.3f} "
              f"{(r['P'] / med - 1) * 100:+6.1f}% {tp * 1e6:11.2f} {ip * 1e3:11.1f} {up:10.1f} {r['ne_max']:13.3e}")
        res.append((d, r))
    print("Stankov (γ = 0,02, ε_r = 4,2, 100 mbar): pico medido 11,8 mA em 18,53 µs; U_rupt 608,6 V e n_e,max "
          "3,38e16 m-3 calculados por eles (tabela 2; o I_max da tabela é dez vezes o da fig. 3, não usar)")
    if not args.figura:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, eixos = plt.subplots(2, 1, figsize=(10, 8))
    med = np.loadtxt(DADOS / "fig3_corrente_medida_100mbar.csv", delimiter=",")
    eixos[0].plot(med[:, 0], med[:, 1] * 1e3, "k", lw=1, label="medida (Stankov 2020, fig. 3)")
    for d, r in res:
        if round(r["a"]["pressao"] / 100.0) == 100:
            eixos[0].plot(r["t"] * 1e6, r["I"] * 1e3, lw=1, label=f"1D: {d.name} (período {r['periodo']})")
    eixos[0].set_xlabel("t (µs)")
    eixos[0].set_ylabel("I (mA)")
    eixos[0].set_title("100 mbar, 1,8 kV")
    pr = sorted(MEDIDA)
    eixos[1].plot(pr, [MEDIDA[x] for x in pr], "ks-", label="medida (fig. 7)")
    calc = {}
    for d, r in res:
        calc.setdefault(d.name.split("_")[0], []).append((round(r["a"]["pressao"] / 100.0), r["P"]))
    for nome, pts in calc.items():
        pts.sort()
        eixos[1].plot([x for x, _ in pts], [y for _, y in pts], "o-", label=f"1D: {nome}")
    eixos[1].set_xlabel("p (mbar)")
    eixos[1].set_ylabel("P (W)")
    for e in eixos:
        e.grid(alpha=0.3)
        e.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.figura, dpi=110)


if __name__ == "__main__":
    main()
