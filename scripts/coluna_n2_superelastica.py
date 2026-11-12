# SPDX-License-Identifier: GPL-3.0-or-later
"""Test of a mechanism for the high-excitation disagreement (docs/coluna_positiva_n2/README.md): is the
overestimated E/N of the N2 positive column caused by the missing superelastic
collisions of the electrons with N2(v)? The tables of BOLSIG+ of the cases are
built with N2 in the ground state only. Here the tables of n2_amonia are rebuilt
with a vibrationally excited N2 at several Tv:
- each N2(vk), k = 1 to 8, gets a copy of all the cross sections of N2 except the
  vibrational excitations themselves (same thresholds: an approximation that
  slightly underestimates the effect, since from v > 0 the thresholds are lower);
- the excitations N2 -> N2(vk) get the double arrow, which turns on the
  superelastic inverse process in BOLSIG+ (statistical weight ratio 1); the resonant
  v = 1 excitation (N2 -> N2(v1res)) is summed into N2 <-> N2(v1), so that it has an
  inverse too, and its own block is kept with zero cross section (ERROS.md E11);
- composicao.txt: N2(vk) fractions of a harmonic Boltzmann distribution at Tv
  (vibrational quantum 0.29 eV), truncated at v = 8 and normalized.
The chemistry (quimica.json) is that of n2_amonia: the BOLSIG+ columns that the
reactions use keep their order. Then, for each condition, the sustaining E/N is
found with the table whose Tv is the one the model predicts there (fixed point
over the Tv nodes, interpolating E/N in Tv).

The step en refuses to run when the transporte.json or bolsigdb.dat of a tv* folder differs
from what the step casos would write from the current n2_amonia case.

Usage: python3 scripts/coluna_n2_superelastica.py casos   (builds the tables, minutes per Tv)
       python3 scripts/coluna_n2_superelastica.py en SAIDA_DIR
"""
import csv
import math
import pathlib
import shutil
import subprocess
import sys

import numpy as np

RAIZ = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "dbd1d"))
sys.path.insert(0, str(RAIZ / "scripts"))
sys.path.insert(0, str(RAIZ / "ferramentas"))
import proveniencia  # noqa: E402
BASE = RAIZ / "casos" / "n2_amonia"
DESTINO = RAIZ / "casos" / "n2_amonia_superelastica"
TVS = (1000, 2000, 3000, 4500, 6000)
VIB = [f"N2(v{k})" for k in range(1, 9)]
TIPOS = ("ELASTIC", "EFFECTIVE", "EXCITATION", "IONIZATION", "ATTACHMENT", "ROTATION")


def blocos(texto):
    """Blocks of a bolsigdb.dat: (type, lines of the block) and the text between them."""
    linhas = texto.splitlines()
    saida, i = [], 0
    while i < len(linhas):
        if linhas[i].strip() in TIPOS:
            j, tracos = i + 1, 0
            while tracos < 2:
                if linhas[j].startswith("-----"):
                    tracos += 1
                j += 1
            saida.append(("bloco", linhas[i:j]))
            i = j
        else:
            saida.append(("texto", [linhas[i]]))
            i += 1
    return saida


def _pontos(linhas):
    """Indices of the two dashed lines and the points (energy, cross section) between them."""
    i, j = [k for k, l in enumerate(linhas) if l.startswith("-----")][:2]
    return i, j, np.array([[float(x) for x in l.split()] for l in linhas[i + 1:j]])


def _com_pontos(linhas, pontos):
    i, j, _ = _pontos(linhas)
    return linhas[:i + 1] + [f" {e:.6e}\t{s:.6e}" for e, s in pontos] + linhas[j:]


def _junta_v1res(partes):
    """The resonant part of the v = 1 excitation (N2 -> N2(v1res)) goes into the N2 -> N2(v1)
    block, summed on the union of the energy grids (linear interpolation, zero outside each
    grid), so that its superelastic inverse uses the density of N2(v1); the v1res block keeps
    its energies with zero cross section, so that the @R sum of the chemistry and the column
    order stay as in the base case (ERROS.md E11: before, v = 1 entered with about 1/19)."""
    nomes = [l[1].strip() if t == "bloco" else None for t, l in partes]
    if "N2 -> N2(v1)" not in nomes or "N2 -> N2(v1res)" not in nomes:
        return partes
    i1, ir = nomes.index("N2 -> N2(v1)"), nomes.index("N2 -> N2(v1res)")
    a, r = _pontos(partes[i1][1])[2], _pontos(partes[ir][1])[2]
    e = np.union1d(a[:, 0], r[:, 0])
    soma = (np.interp(e, a[:, 0], a[:, 1], left=0.0, right=0.0)
            + np.interp(e, r[:, 0], r[:, 1], left=0.0, right=0.0))
    partes = list(partes)
    partes[i1] = ("bloco", _com_pontos(partes[i1][1], np.c_[e, soma]))
    partes[ir] = ("bloco", _com_pontos(partes[ir][1], np.c_[r[:, 0], np.zeros(len(r))]))
    return partes


def bolsigdb_superelastico(texto):
    """Copies of the N2 cross sections for each N2(vk) and the double arrow on N2 -> N2(vk),
    with the resonant v = 1 excitation inside N2 <-> N2(v1) (_junta_v1res)."""
    saida, copias = [], []
    for tipo, linhas in _junta_v1res(blocos(texto)):
        if tipo == "texto":
            saida.extend(linhas)
            continue
        alvo = linhas[1].split("->")[0].split("<->")[0].strip()
        destino = linhas[1].split("->")[-1].strip()
        vib = alvo == "N2" and destino.startswith("N2(v")
        if vib and destino in VIB:
            linhas = [linhas[0], f"N2 <-> {destino}"] + linhas[2:]
        saida.extend(linhas)
        if alvo == "N2" and not vib:
            for v in VIB:
                novo = linhas[1].replace("N2", v, 1)
                copias.extend([""] + [linhas[0], novo] + [l.replace("SPECIES: e / N2", f"SPECIES: e / {v}")
                                                          for l in linhas[2:]])
    return "\n".join(saida + copias) + "\n"


def composicao(tv):
    w = [math.exp(-0.29 * v * 11604.5 / tv) for v in range(9)]
    s = sum(w)
    return f"N2 {w[0] / s:.6g}\n" + "".join(f"N2(v{v}) {w[v] / s:.6g}\n" for v in range(1, 9)) + "N2^+ traco\n"


def conteudo_esperado(base=BASE):
    """Text that monta writes to each tv* folder for the files taken from the base case."""
    return {"bolsigdb.dat": bolsigdb_superelastico((base / "bolsigdb.dat").read_text()),
            "transporte.json": (base / "transporte.json").read_text()}


def defasados(base=BASE, destino=DESTINO):
    """Files (tv*/name) of the derived cases whose content, or existence, differs from what monta
    would write from the base case."""
    esperado = conteudo_esperado(base)
    saida = []
    for pasta in sorted(destino.glob("tv*")):
        for nome, texto in esperado.items():
            arquivo = pasta / nome
            if not arquivo.is_file() or arquivo.read_text() != texto:
                saida.append(f"{pasta.name}/{nome}")
    return saida


def monta():
    texto_kinet = (BASE / "kinet.inp").read_text()
    antigo = "BOLSIG\nN2  N\nEND"
    assert antigo in texto_kinet
    kinet = texto_kinet.replace(antigo, "BOLSIG\nN2  N  " + "  ".join(VIB) + "\nEND")
    banco = conteudo_esperado()["bolsigdb.dat"]
    for tv in TVS:
        d = DESTINO / f"tv{tv}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "kinet.inp").write_text(kinet)
        (d / "bolsigdb.dat").write_text(banco)
        (d / "composicao.txt").write_text(composicao(tv))
        (d / "t_nos.txt").write_text("280\n400\n600\n800\n")
        for nome in ("gas.json", "transporte.json", "quimica.json"):
            shutil.copy(BASE / nome, d / nome)
        for acao in ("zdplaskin", "tabelas"):
            r = subprocess.run([sys.executable, str(RAIZ / "ferramentas" / "caso.py"), acao, str(d)],
                               capture_output=True, text=True)
            if r.returncode:
                sys.exit(f"{d}: {acao} failed:\n{r.stdout[-2000:]}{r.stderr[-2000:]}")
        print(f"tv{tv}: tables ready", flush=True)


def en_com_tv(cond, tv, corrente):
    from coluna import Coluna0D
    from coluna_n2 import GAMAS, treanor
    c = Coluna0D(DESTINO / f"tv{tv}", cond["p"], cond["tg"], cond["r"], gamas=GAMAS, estrito=False)
    en = c.en_sustentado(corrente, cond["t"])
    _, y = c.integra(en, c.ne_da_corrente(corrente, en), cond["t"], n_saidas=3)
    nomes = [s.upper() for s in c.especies]
    n0 = c.densidades(y[-1])[[s.upper() for s in c.q.especies].index("N2")]
    tv_modelo = treanor(np.r_[n0, [y[-1, nomes.index(f"N2(V{k})")] for k in range(1, 5)]], cond["tg"])
    return en, tv_modelo


def en(saida):
    from coluna_n2 import condicoes
    from coluna_n2_en import CORRENTE_MA
    velhos = defasados()
    if velhos:
        sys.exit("derived cases out of date with casos/n2_amonia: " + ", ".join(velhos)
                 + ". Run the step casos first.")
    saida = pathlib.Path(saida)
    saida.mkdir(parents=True, exist_ok=True)
    linhas = []
    for cond in condicoes():
        corrente = CORRENTE_MA[cond["id"]] * 1e-3
        resultados = {tv: en_com_tv(cond, tv, corrente) for tv in TVS}
        # fixed point: the table Tv equal to the Tv of the model (interpolated between nodes)
        dif = np.array([resultados[tv][1] - tv for tv in TVS])
        en_tv = np.array([resultados[tv][0] for tv in TVS])
        troca = np.nonzero(np.diff(np.sign(dif)))[0]
        if len(troca):
            k = troca[0]
            f = dif[k] / (dif[k] - dif[k + 1])
            tv_fp = TVS[k] + f * (TVS[k + 1] - TVS[k])
            en_fp = en_tv[k] + f * (en_tv[k + 1] - en_tv[k])
        else:
            tv_fp, en_fp = float("nan"), float("nan")
        linhas.append(dict(id=cond["id"], en_med=cond["en"], en_err=cond["en_err"], tv_med=cond["tv"],
                           **{f"en_tv{tv}": resultados[tv][0] for tv in TVS},
                           **{f"tvmodelo_tv{tv}": resultados[tv][1] for tv in TVS}, tv_ponto_fixo=tv_fp,
                           en_ponto_fixo=en_fp))
        print(f"{cond['id']:10s} measured {cond['en']:6.1f}  " + "  ".join(
            f"Tv{tv}: {resultados[tv][0]:5.1f} (model {resultados[tv][1]:.0f})" for tv in TVS)
            + f"  ponto fixo: Tv {tv_fp:.0f}, E/N {en_fp:.1f}", flush=True)
    with open(saida / "superelastica.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(linhas[0]))
        w.writeheader()
        w.writerows(linhas)
    proveniencia.grava_origem(saida / "superelastica.csv", __file__, ['n2_amonia_superelastica/tv1000', 'n2_amonia_superelastica/tv2000', 'n2_amonia_superelastica/tv3000', 'n2_amonia_superelastica/tv4500', 'n2_amonia_superelastica/tv6000'])  # code and inputs beside the CSV


if __name__ == "__main__":
    if sys.argv[1] == "casos":
        monta()
    else:
        en(sys.argv[2])
