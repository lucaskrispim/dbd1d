# SPDX-License-Identifier: GPL-3.0-or-later
"""Short validation: reduced versions of the comparisons of the article that run in
a few minutes on a laptop, to check an installation or a code change without the
full runs (hours to days) listed in artigo/README.md.

Each item runs short simulations (in parallel) and checks a few quantities: against
the value that the same short run gave for version 1.2.2 (REFERENCIA, with a tolerance
that covers other versions of numpy and scipy) and, where it makes sense, against
the measurement or the published model. The full runs of the article give the
numbers of the text; these short runs only show that the program still gives them.

  stankov    argon DBD of Stankov et al. 2020 at 100 mbar, 3 periods with 50 cells
             (exemplos/quickstart.sh): mean power and current peak of the third period
  loffhagen  argon at 1 atm, Loffhagen et al. 2020: 2.6 kV (below the measured ignition
             band, 2.87 to 3.17 kV) must stay off and 3.2 kV (above it) must ignite,
             5 periods with 50 cells
  townsend   Townsend breakdown in argon at 100 Torr (validacao/townsend.py): the discrete
             breakdown with 200 cells within 0.5% of the continuous reference
  enxame     electron swarm parameters of the argon tables against the measurements
             compiled by Crispim et al. 2024 (no simulation)

Usage: python3 validacao/curta.py [ITEM ...] [--output DIR] [--python PY]
Exit code 0 when every check passes.
"""
import argparse
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import time

import numpy as np

RAIZ = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "scripts"))
sys.path.insert(0, str(RAIZ / "validacao"))
sys.path.insert(0, str(RAIZ / "dbd1d"))

T_STANKOV = 4.1666666666666665e-05
T_LOFFHAGEN = 1.1600928074245939e-05

# values of the short runs with version 1.2.2 (python 3.10, numpy 2.2, scipy 1.15)
REFERENCIA = {
    "stankov": {"P (W)": 0.6467, "t peak (us)": 19.58, "I peak (mA)": 24.4},
}


def stankov_args(destino):
    return [str(destino), "--cells", "50", "--clustering", "3.0", "--energy", "--gap", "3e-3", "--thickness",
            "1e-3", "--eps-dielectric", "4.2", "--gamma", "0.02", "--eps-sec", "2", "--pressure", "1e4",
            "--t-gas", "300", "--waveform", "sine", "--period", repr(T_STANKOV), "--v-peak", "-1800",
            "--neutral", "--ne0", "1e6", "--t-end", repr(3 * T_STANKOV), "--dt-output", repr(T_STANKOV / 400)]


def loffhagen_args(destino, u0):
    return [str(destino), "--cells", "50", "--clustering", "3.0", "--energy", "--gap", "1e-3", "--thickness",
            "2e-3", "--eps-dielectric", "4.6", "--gamma", "0.02", "--eps-sec", "2", "--pressure", "101325",
            "--t-gas", "300", "--waveform", "sine", "--period", repr(T_LOFFHAGEN), "--v-peak", str(u0),
            "--neutral", "--ne0", "1e6", "--t-end", repr(5 * T_LOFFHAGEN), "--dt-output", repr(T_LOFFHAGEN / 400)]


def simulacoes(itens, saida):
    """Runs of the selected items: name -> list of simula.py arguments."""
    runs = {}
    if "stankov" in itens:
        runs["stankov"] = stankov_args(saida / "stankov")
    if "loffhagen" in itens:
        for u0 in (2600, 3200):
            runs[f"loffhagen_{u0}"] = loffhagen_args(saida / f"loffhagen_{u0}", u0)
    return runs


def roda(runs, py):
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    procs = {nome: subprocess.Popen([py, str(RAIZ / "dbd1d" / "simula.py"), *args], stdout=subprocess.DEVNULL,
                                    stderr=subprocess.PIPE, text=True, env=env)
             for nome, args in runs.items()}
    falhas = {}
    for nome, pr in procs.items():
        _, erro = pr.communicate()
        if pr.returncode != 0:
            falhas[nome] = erro.strip().splitlines()[-1] if erro.strip() else f"exit code {pr.returncode}"
    return falhas


def confere(nome, valor, ok, esperado):
    return {"item": nome[0], "quantity": nome[1], "value": valor, "expected": esperado, "ok": bool(ok)}


def perto(valor, ref, tol):
    return abs(valor / ref - 1.0) <= tol


def verifica_stankov(saida):
    from compara_stankov import ultimo_periodo
    r = ultimo_periodo(saida / "stankov")
    tp, _, ip = max(r["rupturas"], key=lambda x: x[2])
    ref = REFERENCIA["stankov"]
    return [
        confere(("stankov", "period"), r["periodo"], r["periodo"] == 3, "3"),
        confere(("stankov", "P (W)"), round(r["P"], 4), perto(r["P"], ref["P (W)"], 0.02),
                f"{ref['P (W)']} +- 2% (measured 0.559 W, 10th period of the full run 0.642 W)"),
        confere(("stankov", "t peak (us)"), round(tp * 1e6, 2), perto(tp * 1e6, ref["t peak (us)"], 0.01),
                f"{ref['t peak (us)']} +- 1% (measured 18.53 us)"),
        confere(("stankov", "I peak (mA)"), round(ip * 1e3, 1), perto(ip * 1e3, ref["I peak (mA)"], 0.05),
                f"{ref['I peak (mA)']} +- 5% (measured 11.8 mA)"),
    ]


def verifica_loffhagen(saida):
    from compara_loffhagen import analisa, LIMIAR
    linhas = []
    for u0, acesa in ((2600, False), (3200, True)):
        _, k, razao, _ = analisa(saida / f"loffhagen_{u0}")
        linhas.append(confere(("loffhagen", f"I/I_cap at {u0 / 1e3:.1f} kV"), round(razao, 2),
                              (razao > LIMIAR) == acesa and k == 5,
                              f"{'above' if acesa else 'at most'} {LIMIAR:g} ({'ignited' if acesa else 'off'}; "
                              "measured ignition 2.87 to 3.17 kV)"))
    return linhas


def verifica_townsend(saida, py):
    import townsend
    gap, pressao, gama = 1.5e-3, 100.0 * townsend.TORR, 0.05
    v_ref = townsend.ruptura_referencia(gap, pressao, gama)[0]
    vs = v_ref * np.array([0.99, 1.01])
    taxas = []
    for v in vs:
        d = saida / f"townsend_{v:.1f}"
        townsend.roda(d, gap, pressao, gama, v, 200, py, 3e-5)
        taxas.append(townsend.taxa_de_crescimento(d))
    v_b = vs[0] - taxas[0] * (vs[1] - vs[0]) / (taxas[1] - taxas[0])
    return [confere(("townsend", "growth rate changes sign"), f"{taxas[0]:.3g} / {taxas[1]:.3g} 1/s",
                    taxas[0] < 0.0 < taxas[1], "negative below, positive above"),
            confere(("townsend", "breakdown (V)"), round(v_b, 1), abs(v_b / v_ref - 1.0) <= 5e-3,
                    f"{v_ref:.1f} +- 0.5% (continuous reference)")]


def verifica_enxame():
    from compara_enxame import compara
    _, _, linhas = compara()
    pior = 0.0
    for nome, fig, x, dev in linhas:
        m = (x >= 1.0) & (x < 100.0)
        if m.any() and "medida" in nome:
            pior = max(pior, abs(np.median(dev[m])))
    return [confere(("enxame", "largest median deviation, 1 to 100 Td (%)"), round(100 * pior, 1), pior <= 0.16,
                    "at most 16 (Section of the swarm data)")]


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("itens", nargs="*", default=["stankov", "loffhagen", "townsend", "enxame"],
                   help="items to run (default: all)")
    p.add_argument("--output", type=pathlib.Path, help="folder for the runs (default: a temporary folder)")
    p.add_argument("--python", default=sys.executable, help="Python interpreter for the runs (default: the current one)")
    a = p.parse_args()
    desconhecidos = set(a.itens) - {"stankov", "loffhagen", "townsend", "enxame"}
    if desconhecidos:
        p.error(f"unknown items: {', '.join(sorted(desconhecidos))}")
    saida = a.output or pathlib.Path(tempfile.mkdtemp(prefix="dbd1d_curta_"))
    saida.mkdir(parents=True, exist_ok=True)
    inicio = time.time()
    falhas = roda(simulacoes(a.itens, saida), a.python)
    linhas = []
    for nome, erro in falhas.items():
        linhas.append(confere((nome, "run"), "failed", False, erro))
    if "stankov" in a.itens and "stankov" not in falhas:
        linhas += verifica_stankov(saida)
    if "loffhagen" in a.itens and not any(n.startswith("loffhagen") for n in falhas):
        linhas += verifica_loffhagen(saida)
    if "townsend" in a.itens:
        linhas += verifica_townsend(saida, a.python)
    if "enxame" in a.itens:
        linhas += verifica_enxame()
    for l in linhas:
        print(f"{'ok  ' if l['ok'] else 'FAIL'} {l['item']:<10s} {l['quantity']:<42s} {str(l['value']):>14s}   "
              f"expected {l['expected']}")
    (saida / "curta.json").write_text(json.dumps(linhas, indent=1) + "\n")
    print(f"{sum(l['ok'] for l in linhas)} of {len(linhas)} checks passed in {time.time() - inicio:.0f} s; "
          f"runs in {saida}")
    sys.exit(0 if all(l["ok"] for l in linhas) else 1)


if __name__ == "__main__":
    main()
