# SPDX-License-Identifier: GPL-3.0-or-later
"""Comparações de detalhe do V1 (Stankov et al. 2020), além da potência e do pico
de compara_stankov.py:

- U_gap(t) e n̄_e(t) do último período contra os do modelo deles (fig. 4, γ = 0,02);
- cauda de corrente depois do pico principal (média de 21 a 28 µs) contra o
  detalhe medido da fig. 3;
- figura de Lissajous Q × V e carga que passa por meio período (excursão de Q),
  a medida com Q = ∫ I dt da corrente da fig. 3;
- repetição período a período: instante e altura do pico positivo de cada
  período, contra os da corrente medida em dez períodos a 650 mbar (fig. 6).

Uso: python3 scripts/detalha_stankov.py DIR_100MBAR [DIR_650MBAR] [--figura SAIDA.png]
"""
import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from resumo import blocos  # noqa: E402
from compara_stankov import AREA, DADOS, EPS0  # noqa: E402


def le(nome):
    return np.loadtxt(DADOS / nome, delimiter=",", ndmin=2)


def eletrica(d):
    a = json.loads((d / "parametros.json").read_text())["argumentos"]
    sup = np.loadtxt(d / "superficie.dat", skiprows=1, ndmin=2)
    t, v, pe, pd = sup[:, 0], sup[:, 1], sup[:, 2], sup[:, 3]
    q = EPS0 * a["eps_dieletrico"] * (v - pe) / a["espessura"] * AREA
    return a, t, v, q, np.gradient(q, t), pe - pd


def ne_medio(d, t0, t1):
    tt, ne = [], []
    for tb, nomes, linhas in blocos(d / "perfil.dat"):
        if t0 - 1e-12 <= tb <= t1 + 1e-12:
            arr = np.array(linhas[1:-1])
            faces = [0.0]
            for c in arr[:, 0]:
                faces.append(2.0 * c - faces[-1])
            h = np.diff(faces)
            tt.append(tb - t0)
            ne.append(float((arr[:, nomes.index("E")] * h).sum() / h.sum()) * 1e6)
    return np.array(tt), np.array(ne)


def picos_por_periodo(t, i, per, n):
    """Instante (no período) e altura do pico positivo de cada período."""
    out = []
    for k in range(n):
        m = (t >= k * per) & (t < (k + 1) * per)
        if m.sum() < 3:
            continue
        j = np.argmax(i[m])
        out.append((k + 1, t[m][j] - k * per, i[m][j]))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("d100", type=pathlib.Path)
    p.add_argument("d650", type=pathlib.Path, nargs="?")
    p.add_argument("--figura", type=pathlib.Path)
    args = p.parse_args()
    a, t, v, q, i, ug = eletrica(args.d100)
    per = a["periodo"]
    k = int(np.floor(t[-1] / per + 1e-9))
    m = (t >= (k - 1) * per - 1e-12) & (t <= k * per + 1e-12)
    tt = t[m] - (k - 1) * per
    # tensão no gap e n_e contra o modelo deles
    ug_ref = le("fig4_ugap_modelo_gama0.02.csv")
    ne_ref = le("fig4_ne_medio_modelo_gama0.02.csv")
    tn, ne = ne_medio(args.d100, (k - 1) * per, k * per)
    print(f"{args.d100.name}, período {k}")
    print(f"  U_gap máx/mín: 1D {ug[m].max():.0f}/{ug[m].min():.0f} V; Stankov (modelo) {ug_ref[:, 1].max():.0f}/"
          f"{ug_ref[:, 1].min():.0f} V")
    meio = tn < 0.5 * per
    print(f"  n_e médio: máx 1D {ne.max():.2e}, Stankov {ne_ref[:, 1].max():.2e} m-3; mínimo antes da ruptura "
          f"1D {ne[meio].min():.2e}, Stankov {ne_ref[ne_ref[:, 0] < 20.8, 1].min():.2e} m-3")
    # cauda depois do pico: a medida não tem corcova nítida (degraus pequenos sobre a
    # queda); compara a corrente média de 21 a 28 µs
    med = le("fig3_corrente_medida_100mbar.csv")
    cor = le("fig3_corcova_medida_100mbar.csv")
    cauda = (tt > 21e-6) & (tt < 28e-6)
    mc = (cor[:, 0] > 21.0) & (cor[:, 0] < 28.0)
    print(f"  corrente média de 21 a 28 µs: 1D {i[m][cauda].mean() * 1e3:.2f} mA; medida {cor[mc, 1].mean() * 1e3:.2f} mA")
    # Lissajous e carga por meio período
    tm = med[:, 0] * 1e-6
    qm = np.concatenate([[0.0], np.cumsum(0.5 * (med[1:, 1] + med[:-1, 1]) * np.diff(tm))])
    va = -1800.0 * np.sin(2 * np.pi * tm / per)
    print(f"  carga que passa no período (excursão de Q): 1D {np.ptp(q[m]) * 1e9:.1f} nC; medida "
          f"{np.ptp(qm) * 1e9:.1f} nC (Q = ∫ I dt da fig. 3)")
    rep = picos_por_periodo(t, i, per, k)
    print("  pico positivo por período (1D): " + "; ".join(f"{n}: {tp * 1e6:.2f} µs {ip * 1e3:.1f} mA" for n, tp, ip in rep))
    if args.d650:
        a6, t6, v6, q6, i6, _ = eletrica(args.d650)
        r6 = picos_por_periodo(t6, i6, per, int(np.floor(t6[-1] / per + 1e-9)))
        m6 = le("fig6_corrente_medida_650mbar.csv")
        rm = picos_por_periodo(m6[:, 0] * 1e-6, m6[:, 1], per, 10)
        print(f"{args.d650.name}: pico positivo por período")
        print("  1D:     " + "; ".join(f"{n}: {tp * 1e6:.1f} µs {ip * 1e3:.0f} mA" for n, tp, ip in r6))
        print("  medido: " + "; ".join(f"{n}: {tp * 1e6:.1f} µs {ip * 1e3:.0f} mA" for n, tp, ip in rm)
              + "  (curva do PDF com passo de 0,5 µs: instantes valem, alturas não)")
    if not args.figura:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, e = plt.subplots(2, 3, figsize=(17, 9))
    e[0, 0].plot(tt * 1e6, ug[m], "C0", label="1D")
    e[0, 0].plot(ug_ref[:, 0], ug_ref[:, 1], "k--", label="modelo de Stankov (fig. 4a, γ = 0,02)")
    e[0, 0].plot(tt * 1e6, v[m] / 5, "C7", lw=0.8, label="U_a / 5")
    e[0, 0].set_ylabel("U_gap (V)")
    e[0, 0].set_title("tensão no gap, 100 mbar")
    e[0, 1].semilogy(tn * 1e6, ne, "C0", label="1D")
    e[0, 1].semilogy(ne_ref[:, 0], ne_ref[:, 1], "k--", label="modelo de Stankov (fig. 4b)")
    e[0, 1].set_ylabel("n̄_e (m⁻³)")
    e[0, 1].set_title("n_e médio no gap, 100 mbar")
    e[0, 2].plot(tt * 1e6, i[m] * 1e3, "C0", label="1D")
    e[0, 2].plot(cor[:, 0], cor[:, 1] * 1e3, "k", lw=0.8, label="medida (detalhe da fig. 3)")
    e[0, 2].set_xlim(18, 30)
    e[0, 2].set_ylim(0, 8)
    e[0, 2].set_ylabel("I (mA)")
    e[0, 2].set_title("cauda depois do pico")
    e[1, 0].plot(v[m] / 1e3, (q[m] - q[m].mean()) * 1e9, "C0", label="1D")
    e[1, 0].plot(va / 1e3, (qm - qm.mean()) * 1e9, "k", lw=0.8, label="medida (Q = ∫ I dt)")
    e[1, 0].set_xlabel("U_a (kV)")
    e[1, 0].set_ylabel("Q (nC)")
    e[1, 0].set_title("Lissajous Q × V, 100 mbar")
    nrep = [x[0] for x in rep]
    e[1, 1].plot(nrep, [x[1] * 1e6 for x in rep], "C0o-", label="1D, 100 mbar")
    e[1, 1].axhline(18.53, color="k", ls="--", label="medida 100 mbar (periódica)")
    if args.d650:
        e[1, 1].plot([x[0] for x in r6], [x[1] * 1e6 for x in r6], "C3s-", label="1D, 650 mbar")
        e[1, 1].plot([x[0] for x in rm], [x[1] * 1e6 for x in rm], "k^:", label="medida 650 mbar (fig. 6)")
        e[1, 2].plot((t6 - (np.floor(t6[-1] / per) - 3) * per) * 1e6, i6 * 1e3, "C3", lw=0.8, label="1D")
        e[1, 2].plot(m6[:, 0] - 7 * per * 1e6, m6[:, 1] * 1e3, "k", lw=0.8,
                     label="medida (fig. 6), períodos 8 a 10")
        e[1, 2].set_xlim(0, 3 * per * 1e6)
        e[1, 2].set_title("corrente nos três últimos períodos, 650 mbar")
        e[1, 2].set_xlabel("t (µs)")
        e[1, 2].set_ylabel("I (mA)")
    e[1, 1].set_xlabel("período")
    e[1, 1].set_ylabel("instante do pico positivo no período (µs)")
    e[1, 1].set_title("repetição período a período")
    for x in (e[0, 0], e[0, 1], e[0, 2]):
        x.set_xlabel("t no último período (µs)")
    for x in e.flat:
        x.grid(alpha=0.3)
        if x.get_legend_handles_labels()[0]:
            x.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.figura, dpi=100)


if __name__ == "__main__":
    main()
