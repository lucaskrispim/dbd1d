# SPDX-License-Identifier: GPL-3.0-or-later
"""Test of a mechanism for the high-excitation disagreement
(docs/coluna_positiva_n2/README.md), second test:
electron impact ionization from the excited states N2(A), N2(B) and N2(a')
(Capitelli et al. 2000, processes 12.3 to 12.5), which the three chemistries lack.
casos/n2_amonia_etapas is n2_amonia with three reactions more,
  E + N2(A3)  => E + E + N2^+   (threshold 15.58 - 6.17 = 9.41 eV)
  E + N2(B3)  => E + E + N2^+   (15.58 - 7.35 = 8.23 eV)
  E + N2(a'1) => E + E + N2^+   (15.58 - 8.40 = 7.18 eV)
from BOLSIG+, with the cross section of the ionization of N2(X) shifted down by
the excitation energy, sigma(eps) = sigma_X(eps + E_exc): a usual approximation
(the excitation energies are those of casos/n2/gas.json). The excited states
enter the BOLSIG+ mixture as traces, so the EEDF is that of n2_amonia. Then the
sustaining E/N of every condition, as in scripts/coluna_n2_en.py.

Usage: python3 scripts/coluna_n2_etapas.py caso
       python3 scripts/coluna_n2_etapas.py en SAIDA_DIR
"""
import csv
import pathlib
import shutil
import subprocess
import sys

RAIZ = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "dbd1d"))
sys.path.insert(0, str(RAIZ / "scripts"))
BASE = RAIZ / "casos" / "n2_amonia"
DESTINO = RAIZ / "casos" / "n2_amonia_etapas"
EXCITADOS = {"N2(A3)": 6.17, "N2(B3)": 7.35, "N2(a'1)": 8.40}
LIMIAR_X = 15.58


def secao_ionizacao_x(texto):
    """(energy, cross section) of the N2 -> N2^+ block of a bolsigdb.dat."""
    linhas = texto.splitlines()
    i = linhas.index("N2 -> N2^+")
    j = next(k for k in range(i, len(linhas)) if linhas[k].startswith("-----"))
    pontos = []
    for linha in linhas[j + 1:]:
        if linha.startswith("-----"):
            break
        e, s = linha.split()
        pontos.append((float(e), float(s)))
    return pontos


def bloco(especie, e_exc, pontos):
    limiar = LIMIAR_X - e_exc
    dados = [(0.0, 0.0)] + [(e - e_exc, s) for e, s in pontos if e >= LIMIAR_X]
    return "\n".join(["", "IONIZATION", f"{especie} -> N2^+", f"{limiar:.6e}/threshold energy",
                      f"SPECIES: e / {especie}",
                      f"PROCESS: E + {especie} -> E + E + N2+, Ionization (N2(X) shifted by {e_exc} eV)",
                      "COLUMNS: Energy (eV) | Cross section (m2)", "-----------------------------"]
                     + [f" {e:.6e}\t{s:.6e}" for e, s in dados] + ["-----------------------------", ""])


def monta():
    DESTINO.mkdir(parents=True, exist_ok=True)
    banco = (BASE / "bolsigdb.dat").read_text()
    pontos = secao_ionizacao_x(banco)
    (DESTINO / "bolsigdb.dat").write_text(banco + "".join(bloco(s, e, pontos) for s, e in EXCITADOS.items()))
    kinet = (BASE / "kinet.inp").read_text()
    antigo = "BOLSIG\nN2  N\nEND"
    assert antigo in kinet
    kinet = kinet.replace(antigo, "BOLSIG\nN2  N  " + "  ".join(EXCITADOS) + "\nEND")
    ionizacao = "E + N2 => E + E + N2^+                     ! BOLSIG N2 -> N2^+"
    assert ionizacao in kinet
    novas = "".join(f"\nE + {s} => E + E + N2^+ ! BOLSIG {s} -> N2^+" for s in EXCITADOS)
    kinet = kinet.replace(ionizacao, ionizacao + novas)
    (DESTINO / "kinet.inp").write_text(kinet)
    (DESTINO / "composicao.txt").write_text("N2 1.0\nN2^+ traco\n" + "".join(f"{s} traco\n" for s in EXCITADOS))
    for nome in ("gas.json", "transporte.json", "t_nos.txt"):
        shutil.copy(BASE / nome, DESTINO / nome)
    for acao in ("zdplaskin", "tabelas", "exporta"):
        r = subprocess.run([sys.executable, str(RAIZ / "ferramentas" / "caso.py"), acao, str(DESTINO)],
                           capture_output=True, text=True)
        if r.returncode:
            sys.exit(f"{acao} failed:\n{r.stdout[-3000:]}{r.stderr[-3000:]}")
        print(acao, "ok", flush=True)


def en(saida):
    from coluna import Coluna0D
    from coluna_n2 import GAMAS, condicoes
    from coluna_n2_en import CORRENTE_MA
    saida = pathlib.Path(saida)
    saida.mkdir(parents=True, exist_ok=True)
    linhas = []
    for cond in condicoes():
        i_a = CORRENTE_MA[cond["id"]] * 1e-3
        resultado = {}
        for nome, gamas in (("base", GAMAS), ("meta_0.01", dict(GAMAS, **{s: 0.01 for s in
                                                                            ("N2(A3)", "N2(a'1)", "N(2D)", "N(2P)")}))):
            c = Coluna0D(DESTINO, cond["p"], cond["tg"], cond["r"], gamas=gamas, estrito=False)
            e_n = c.en_sustentado(i_a, cond["t"], faixa=(10.0, 300.0))
            # share of the ionization from the excited states at that E/N
            ne = c.ne_da_corrente(i_a, e_n)
            _, y = c.integra(e_n, ne, cond["t"], n_saidas=2)
            k, _ = c.coeficientes(e_n)
            taxas = c.q.taxas(c.densidades(y[-1])[:, None], k[:, None])[:, 0]
            ion = [r for r, txt in enumerate(c.q.reacoes) if txt.upper().startswith("BOLSIG:") and "->N2^+" in txt.upper()]
            excit = [r for r in ion if not txt_x(c.q.reacoes[r])]
            fracao = taxas[excit].sum() / taxas[ion].sum()
            resultado[nome] = (e_n, fracao)
        linhas.append(dict(id=cond["id"], en_med=cond["en"], en_base=resultado["base"][0],
                           fracao_etapas_base=resultado["base"][1], en_meta001=resultado["meta_0.01"][0],
                           fracao_etapas_meta001=resultado["meta_0.01"][1]))
        print(f"{cond['id']:10s} measured {cond['en']:6.1f}  with stepwise {resultado['base'][0]:6.1f} "
              f"(stepwise fraction {resultado['base'][1]:.2f})  with stepwise and gamma_meta 0.01 "
              f"{resultado['meta_0.01'][0]:6.1f} ({resultado['meta_0.01'][1]:.2f})", flush=True)
    with open(saida / "etapas.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(linhas[0]))
        w.writeheader()
        w.writerows(linhas)


def txt_x(reacao):
    return reacao.upper().startswith("BOLSIG:N2->")


if __name__ == "__main__":
    if sys.argv[1] == "caso":
        monta()
    else:
        en(sys.argv[2])
