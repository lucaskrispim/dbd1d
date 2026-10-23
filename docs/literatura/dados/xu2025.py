# SPDX-License-Identifier: GPL-3.0-or-later
"""Extraction of figures 1(d-f) and 3 of Xu et al. 2025, Nat. Commun., doi
10.1038/s41467-025-66403-6: plate-to-plate DBD with an Al2O3 barrier (5 mm gap,
10 Torr, 20 kHz, 1.3 kV amplitude) and the ferroelectric (PZT) barrier, called
FBD, in N2, H2 and 25% N2 / 75% H2. Vector figures, pages 26 and 28 of the PDF
(version "article in press").

Figure 1(d-f): charge Q (uC/cm2) versus the cell voltage, measured with a
reference capacitor in series, in pure N2, the mixture and pure H2.
Figure 3(a,b): N and H densities (cm^-3) by TALIF along the central line over one
cycle (0 to 50 us), with the cell voltage in the upper panels. The time averages
of figure 3(c,d) are printed on the bars and are recorded here as text; the
integral of each extracted curve over 50 us reproduces its bar within 3%, which
checks the calibration.

Calibration (pt, y downwards) from the major ticks of each panel, found by
tracos_de_escala and checked against the labels:
  fig. 1(d), (e), (f): x ticks at -0.5, 0, 0.5 kV; y ticks at 0.2 ... -0.2 uC/cm2;
  fig. 3: x ticks at 0, 10, ..., 50 us; voltage panels at 1, 0, -1 kV; DBD panel
  of N at 1.2e16, 8e15, 4e15, 0; DBD panel of H at 2.5e17, 2e17, ..., 0.

Colours: DBD in N2 (0, 0.28, 0.45) in fig. 3(a), DBD in H2 (0, 0.42, 0.67) in
fig. 3(b) (the voltage of both pure gases is drawn in the first), DBD in the mixture (0.49, 0.68, 0.88); FBD in the pure gas
(0.53, 0.01, 0.01) and in the mixture (0.91, 0.61, 0.58). In fig. 1 the DBD is
(0.49, 0.68, 0.88) and the FBD (0.91, 0.61, 0.58); the shaded min-max bands are
fills and are not extracted.

Usage: python3 docs/literatura/dados/xu2025.py  (writes to xu2025/)
"""
import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(RAIZ / "ferramentas"))
import digitaliza as dg  # noqa: E402

PDF = RAIZ / "docs" / "artigos" / "s41467-025-66403-6_reference.pdf"
SAIDA = Path(__file__).with_name("xu2025")

DBD_PURO_N2, DBD_PURO_H2, DBD_MISTURA = (0.0, 0.28, 0.45), (0.0, 0.42, 0.67), (0.49, 0.68, 0.88)
FBD_PURO, FBD_MISTURA = (0.53, 0.01, 0.01), (0.91, 0.61, 0.58)

# time averages printed on the bars of fig. 3(c,d), cm^-3
MEDIAS = [
    ("N", "DBD", "mixture", 7.1e13), ("N", "DBD", "N2", 2.3e15),
    ("N", "FBD", "mixture", 2.8e15), ("N", "FBD", "N2", 3.0e15),
    ("H", "DBD", "mixture", 3.8e16), ("H", "DBD", "H2", 2.2e15),
    ("H", "FBD", "mixture", 7.4e16), ("H", "FBD", "H2", 3.5e16),
]


def grava(nome, cabecalho, colunas, nota):
    SAIDA.mkdir(exist_ok=True)
    np.savetxt(SAIDA / nome, np.column_stack(colunas), fmt="%.6g", delimiter=",",
               header=f"{nota}\n{cabecalho}", comments="# ")
    print(f"{nome}: {len(colunas[0])} points")


def escala(tr, moldura, eixo, n):
    t = dg.tracos_de_escala(tr, moldura, eixo, min_comp=1.5)
    assert len(t) == n, (moldura, eixo, t)
    return t


def curva(tr, cor, janela, ex, ey):
    c = dg.curva_vetorial(tr, cor, janela, min_pontos=10)
    assert len(c) > 50, (cor, janela, len(c))
    return ex.valor(c[:, 0]), ey.valor(c[:, 1])


def ciclo(tr, cor, janela, ex, ey):
    """Closed Q-U loop: the vertices in the order of the path, not sorted in x."""
    partes = [t["p"] for t in tr if dg._mesma_cor(t["traco"], cor) and len(t["p"]) >= 10
              and dg._dentro(t["p"].mean(0)[None, :], janela)[0]]
    assert partes, (cor, janela)
    p = np.concatenate(partes)
    return ex.valor(p[:, 0]), ey.valor(p[:, 1])


def figura1(tr):
    paineis = {"d_N2": (34.3, 352.9, 163.0, 456.1), "e_mistura": (202.4, 352.9, 330.2, 456.1),
               "f_H2": (369.4, 352.9, 497.4, 456.1)}
    for nome, m in paineis.items():
        tx, ty = escala(tr, m, "x", 3), escala(tr, m, "y", 5)
        ex, ey = dg.Eixo(tx[0], -0.5, tx[-1], 0.5), dg.Eixo(ty[0], 0.2, ty[-1], -0.2)
        for barreira, cor in (("dbd", DBD_MISTURA), ("fbd", FBD_MISTURA)):
            u, q = ciclo(tr, cor, m, ex, ey)
            grava(f"fig1{nome}_{barreira}.csv", "U_cell_kV,Q_uC_cm2", (u, q),
                  f"Xu 2025 fig. 1({nome[0]}), {barreira.upper()}, mean over 10 cycles, path order")


def confere_media(t, n, esp, barreira, gas):
    """The integral over the cycle divided by 50 us must give the bar of fig. 3(c,d)
    (the curves stop a little before 50 us; the missing part is near zero)."""
    ref = next(v for e, b, g, v in MEDIAS if (e, b, g) == (esp, barreira, gas))
    media = np.trapezoid(n, t) / 50.0
    assert abs(media / ref - 1.0) < 0.05, (esp, barreira, gas, media, ref)


def figura3(tr):
    m_un, m_dn = (65.2, 18.6, 235.3, 77.9), (65.2, 186.9, 235.3, 295.7)
    m_uh, m_dh = (302.4, 18.6, 472.5, 77.9), (302.4, 186.9, 472.5, 295.7)
    m_fn, m_fh = (65.2, 77.9, 235.3, 186.9), (302.4, 77.9, 472.5, 186.9)
    for m_u, m_d, m_f, esp, puro_dbd, gas, topo, nty in (
            (m_un, m_dn, m_fn, "N", DBD_PURO_N2, "N2", 1.2e16, 4),
            (m_uh, m_dh, m_fh, "H", DBD_PURO_H2, "H2", 2.5e17, 6)):
        tx = escala(tr, m_d, "x", 6)
        ex = dg.Eixo(tx[0], 0.0, tx[-1], 50.0)
        ty = escala(tr, m_d, "y", nty)
        ey = dg.Eixo(ty[0], topo, ty[-1], 0.0)
        for rotulo, cor in (("dbd_" + gas, puro_dbd), ("dbd_mistura", DBD_MISTURA)):
            t, n = curva(tr, cor, m_d, ex, ey)
            confere_media(t, n, esp, "DBD", gas if rotulo.endswith(gas) else "mixture")
            grava(f"fig3_n{esp}_{rotulo}.csv", f"t_us,n{esp}_cm3", (t, n),
                  f"Xu 2025 fig. 3, {esp} density by TALIF, Al2O3 DBD, {rotulo.split('_')[1]}")
        tu = escala(tr, m_u, "y", 3)
        eu = dg.Eixo(tu[0], 1.0, tu[-1], -1.0)
        txu = escala(tr, m_u, "x", 6)
        exu = dg.Eixo(txu[0], 0.0, txu[-1], 50.0)
        # the voltage of the pure gas has the colour of N2 in both panels
        for rotulo, cor in (("dbd_" + gas, DBD_PURO_N2), ("dbd_mistura", DBD_MISTURA)):
            t, u = curva(tr, cor, m_u, exu, eu)
            grava(f"fig3_U_{esp}_{rotulo}.csv", "t_us,U_cell_kV", (t, u),
                  f"Xu 2025 fig. 3, cell voltage of the TALIF run of {esp}, Al2O3 DBD, {rotulo.split('_')[1]}")
        # FBD, for reference: its panel has the same scale as the DBD panel
        tf = escala(tr, m_f, "y", nty)
        ef = dg.Eixo(tf[0], topo, tf[-1], 0.0)
        for rotulo, cor in (("fbd_" + gas, FBD_PURO), ("fbd_mistura", FBD_MISTURA)):
            t, n = curva(tr, cor, m_f, ex, ef)
            confere_media(t, n, esp, "FBD", gas if rotulo.endswith(gas) else "mixture")
            grava(f"fig3_n{esp}_{rotulo}.csv", f"t_us,n{esp}_cm3", (t, n),
                  f"Xu 2025 fig. 3, {esp} density by TALIF, PZT FBD, {rotulo.split('_')[1]}")
    with open(SAIDA / "fig3_medias.csv", "w") as f:
        f.write("# Xu 2025 fig. 3(c,d), time averages over one cycle printed on the bars\n")
        f.write("# species,barrier,gas,mean_cm3\n")
        for linha in MEDIAS:
            f.write(",".join(str(c) for c in linha) + "\n")
    print("fig3_medias.csv:", len(MEDIAS), "values")


def main():
    figura1(dg.tracos_pdf(PDF, 26))
    figura3(dg.tracos_pdf(PDF, 28))


if __name__ == "__main__":
    main()
