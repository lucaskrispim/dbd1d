# SPDX-License-Identifier: GPL-3.0-or-later
"""Gas and transport data of the case Ar:N2 90:10: conductivity of the mixture by
kinetic theory (Cantera and GRI-Mech 3.0), cv of 0.9 x 1.5 + 0.1 x 2.5 k_B,
mobilities and diffusions estimated by ferramentas/caso.py estimate (Langevin with
Blanc's law and kinetic theory) and the wall loss of the neutrals from the argon
and N2 cases (it is a property of the surface, not of the gas).
Usage (with Cantera): python3 casos/ar_n2/gera_dados.py
The "fonte" strings are data written to the JSON files and stay as they are there.
"""
import json
import pathlib
import sys

import cantera as ct

AQUI = pathlib.Path(__file__).resolve().parent
RAIZ = AQUI.parent.parent
sys.path.insert(0, str(RAIZ / "ferramentas"))
import caso  # noqa: E402

X = {"AR": 0.9, "N2": 0.1}
g = ct.Solution("gri30.yaml")
g.transport_model = "mixture-averaged"
T = [260, 300, 400, 500, 600, 800, 1000, 1500]
kappa = []
for t in T:
    g.TPX = t, ct.one_atm, X
    kappa.append(round(float(g.thermal_conductivity), 5))
gas = {
    "_formato": json.loads((RAIZ / "casos" / "n2" / "gas.json").read_text())["_formato"],
    "fundo": {"Ar": 0.9, "N2": 0.1},
    "cv_por_kb": 0.9 * 1.5 + 0.1 * 2.5,
    "condutividade": {"tipo": "tabela", "T": T, "kappa": kappa},
    "ion_inicial": "Ar^+",
    "fonte": ["mistura de teste da ferramenta (casos/ar_n2/monta.py)",
              "condutividade pela teoria cinética (Cantera, GRI-Mech 3.0), a 1 atm",
              "cv: média pela fração de 1,5 k_B (Ar) e 2,5 k_B (N2 sem vibração)"],
}
(AQUI / "gas.json").write_text(json.dumps(gas, ensure_ascii=False, indent=2) + "\n")
faltam = caso.gera_transporte(AQUI, biblioteca={}, estimar=True)
t = json.loads((AQUI / "transporte.json").read_text())
perdas = {**{k: v for k, v in json.loads((RAIZ / "transporte.json").read_text()).items() if isinstance(v, dict)},
          **{k: v for k, v in json.loads((RAIZ / "casos" / "n2" / "transporte.json").read_text()).items()
             if isinstance(v, dict)}}
for nome, faltando in faltam.items():
    if faltando == ["perda_na_parede"]:
        origem = next(v for k, v in perdas.items() if k.upper() == nome.upper())
        t[nome]["perda_na_parede"] = origem["perda_na_parede"]
        t[nome]["fonte"] = t[nome].get("fonte", "") + "; perda na parede do caso de um gás só (propriedade da superfície)"
(AQUI / "transporte.json").write_text(json.dumps(t, ensure_ascii=False, indent=2) + "\n")
print("gas.json and transporte.json written; missing:", faltam)
