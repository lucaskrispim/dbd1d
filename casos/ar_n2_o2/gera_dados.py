# SPDX-License-Identifier: GPL-3.0-or-later
"""Gas and transport data of the case of dry air with argon (N2 78%, O2 21%,
Ar 1%), to test the tool on a new mixture; not meant for a study of air.
Conductivity of the mixture by kinetic theory (Cantera, GRI-Mech 3.0); cv by
fraction (2.5 k_B for N2 and O2, 1.5 for Ar); mobilities and diffusions by
ferramentas/caso.py estimate. What the estimate does not cover:
- diffusion of O3, NO3 and N2O5 (outside GRI-Mech): that of O2 for O3 and that
  of NO2 for the other two;
- wall loss (not estimated): from the existing cases (amonia_sun2022 for N2(v),
  excited states of N2, N, N(2D), N(2P); root for Ar*) and, for oxygen and the
  NOx, test values: short-lived excited states 1, O2(a1) 1e-4, O2(b1) 1e-2,
  O2(v) as N2(v), O 1e-3, stable molecules 0.
Usage (with Cantera): python3 casos/ar_n2_o2/gera_dados.py
"""
import json
import pathlib
import sys

import cantera as ct

AQUI = pathlib.Path(__file__).resolve().parent
RAIZ = AQUI.parent.parent
sys.path.insert(0, str(RAIZ / "ferramentas"))
import caso  # noqa: E402

X = {"N2": 0.78, "O2": 0.21, "AR": 0.01}
g = ct.Solution("gri30.yaml")
g.transport_model = "mixture-averaged"
T = [260, 300, 400, 500, 600, 800, 1000, 1500]
kappa = []
for t in T:
    g.TPX = t, ct.one_atm, X
    kappa.append(round(float(g.thermal_conductivity), 5))
gas = {
    "_formato": json.loads((RAIZ / "casos" / "n2" / "gas.json").read_text())["_formato"],
    "fundo": {"N2": 0.78, "O2": 0.21, "Ar": 0.01},
    "cv_por_kb": 0.78 * 2.5 + 0.21 * 2.5 + 0.01 * 1.5,
    "condutividade": {"tipo": "tabela", "T": T, "kappa": kappa},
    "ion_inicial": "N2^+",
    "fonte": ["test mixture of the tool (casos/ar_n2_o2/monta.py)",
              "conductivity by kinetic theory (Cantera, GRI-Mech 3.0), at 1 atm",
              "cv: average by fraction of 2.5 k_B (N2, O2 without vibration) and 1.5 k_B (Ar)"],
}
(AQUI / "gas.json").write_text(json.dumps(gas, ensure_ascii=False, indent=2) + "\n")
caso.gera_transporte(AQUI, biblioteca={}, estimar=True)
t = json.loads((AQUI / "transporte.json").read_text())
SEMELHANTE = {"O3": "O2", "NO3": "NO2", "N2O5": "NO2"}
for nome, base in SEMELHANTE.items():
    t[nome]["difusao"] = dict(t[base]["difusao"])
    t[nome]["fonte"] = f"diffusion as that of {base} (outside GRI-Mech; test value)"
existentes = {}
for arq in (RAIZ / "transporte.json", RAIZ / "casos" / "amonia_sun2022" / "transporte.json"):
    for k, v in json.loads(arq.read_text()).items():
        if isinstance(v, dict) and "perda_na_parede" in v:
            existentes[k.upper().replace("'", "`").replace("A`1", "A`1")] = v["perda_na_parede"]
existentes["N2(A`1)"] = existentes.get("N2(A'1)".upper(), existentes.get("N2(A`1)", 1.0))
TESTE = {"O2(V1)": "N2(V1)", "O2(V2)": "N2(V2)", "O2(V3)": "N2(V3)", "O2(V4)": "N2(V4)"}
VALORES = {"O2(A1)": 1e-4, "O2(B1)": 1e-2, "O2(4.5EV)": 1.0, "O(1D)": 1.0, "O(1S)": 1.0, "O": 1e-3,
           "O3": 0.0, "NO": 0.0, "N2O": 0.0, "NO2": 0.0, "NO3": 0.0, "N2O5": 0.0}
for nome, e in t.items():
    if not isinstance(e, dict) or "difusao" not in e or "perda_na_parede" in e or nome in ("N2", "O2", "AR"):
        continue
    if nome.upper() in existentes:
        e["perda_na_parede"] = existentes[nome.upper()]
        nota = "wall loss of the existing case (property of the surface)"
    elif nome.upper() in TESTE:
        e["perda_na_parede"] = existentes[TESTE[nome.upper()]]
        nota = f"wall loss as that of {TESTE[nome.upper()]} (test value)"
    else:
        e["perda_na_parede"] = VALORES[nome.upper()]
        nota = "wall loss: test value"
    e["fonte"] = (e.get("fonte", "") + "; " + nota).lstrip("; ")
(AQUI / "transporte.json").write_text(json.dumps(t, ensure_ascii=False, indent=2) + "\n")
print("gas.json and transporte.json written")
