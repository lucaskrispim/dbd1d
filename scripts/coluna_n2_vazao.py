# SPDX-License-Identifier: GPL-3.0-or-later
"""V4 of docs/coluna_positiva_n2: electron density and E/N against the gas flow
rate at 60 mA and 230 Pa (Levaton et al. 2020, figs. 3 and 4), from the
self-consistent 0D column (dbd1d/coluna.py, en_sustentado), with the chemistries
of the ammonia and of the thesis.

Assumptions, stated in the results:
- T_g = 390 K, the value measured at 60 mA and 230 Pa (Levaton et al. 2021, table 1);
- residence time at the probes: Levaton gives 9.6e-5 to 1.8e-3 s between the
  probes over the flow rates of 1 to 0.1 Slm; we take the geometric mean of the
  probe interval at 1 Slm, sqrt(9.6e-5 * 1.8e-4) = 1.3e-4 s, scaled by 1/Q;
- the measured n_e is recalculated with our electron mobility at the measured
  E/N (scripts/coluna_n2.py, ne_nosso).
The 1070 Pa series is left out: the article gives no T_g for it.

Usage: python3 scripts/coluna_n2_vazao.py SAIDA_DIR
Writes vazao.csv and ne_vazao.png.
"""
import csv
import pathlib
import sys

import numpy as np

RAIZ = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "dbd1d"))
sys.path.insert(0, str(RAIZ / "scripts"))
from coluna import Coluna0D  # noqa: E402
from coluna_n2 import GAMAS, ne_nosso  # noqa: E402

P, TG, R, I_A = 230.0, 390.0, 0.3, 0.060
T_1SLM = np.sqrt(9.6e-5 * 1.8e-4)
DADOS = RAIZ / "docs" / "literatura" / "dados" / "levaton2020"


def main():
    saida = pathlib.Path(sys.argv[1])
    saida.mkdir(parents=True, exist_ok=True)
    en_med = np.loadtxt(DADOS / "fig3_EN_230Pa.csv", delimiter=",")
    ne_med = np.loadtxt(DADOS / "fig4_ne_230Pa.csv", delimiter=",")
    linhas = []
    for caso in ("n2_amonia", "n2_tese"):
        c = Coluna0D(RAIZ / "casos" / caso, P, TG, R, gamas=GAMAS, estrito=False)
        for (q, en_m, *_), (_, ne_m, ne_inf, ne_sup) in zip(en_med, ne_med):
            t = T_1SLM / q
            en = c.en_sustentado(I_A, t)
            ne = c.ne_da_corrente(I_A, en)
            ne_m_nosso = ne_nosso(c, ne_m, en_m)
            linhas.append(dict(quimica=caso, Q_slm=q, t_s=t, EN_medido=en_m, EN_previsto=en, ne_medido=ne_m,
                               ne_medido_nossa_mobilidade=ne_m_nosso, ne_previsto=ne))
            print(f"{caso:10s} Q {q:4.2f}  t {t:.2e}  E/N {en_m:5.1f} -> {en:5.1f}  "
                  f"ne {ne_m_nosso:.3g} -> {ne:.3g}", flush=True)
    with open(saida / "vazao.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(linhas[0]))
        w.writeheader()
        w.writerows(linhas)
    figura(linhas, saida / "ne_vazao.png")


def figura(linhas, arquivo):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    base = [l for l in linhas if l["quimica"] == "n2_amonia"]
    q = [l["Q_slm"] for l in base]
    ax[0].plot(q, [l["ne_medido_nossa_mobilidade"] for l in base], "ks", label="measured (our mobility)")
    ax[1].plot(q, [l["EN_medido"] for l in base], "ks", label="measured")
    for caso, cor in (("n2_amonia", "tab:red"), ("n2_tese", "tab:blue")):
        sel = [l for l in linhas if l["quimica"] == caso]
        ax[0].plot([l["Q_slm"] for l in sel], [l["ne_previsto"] for l in sel], "o-", color=cor, label=caso)
        ax[1].plot([l["Q_slm"] for l in sel], [l["EN_previsto"] for l in sel], "o-", color=cor, label=caso)
    ax[0].set(xlabel="Q (Slm)", ylabel="n$_e$ (cm$^{-3}$)", title="n$_e$ against the flow rate, 60 mA, 230 Pa (V4)")
    ax[1].set(xlabel="Q (Slm)", ylabel="E/N (Td)", title="E/N against the flow rate, 60 mA, 230 Pa")
    for a in ax:
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(arquivo, dpi=110)


if __name__ == "__main__":
    main()
