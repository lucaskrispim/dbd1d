# SPDX-License-Identifier: GPL-3.0-or-later
"""Dados do gás e do transporte do caso de ar seco com argônio (N2 78%, O2 21%,
Ar 1%), para testar a ferramenta numa mistura nova; não servem a um estudo do
ar. Condutividade da mistura pela teoria cinética (Cantera, GRI-Mech 3.0); cv
pela fração (2,5 k_B para N2 e O2, 1,5 para Ar); mobilidades e difusões por
ferramentas/caso.py estima. O que a estimativa não cobre:
- difusão de O3, NO3 e N2O5 (fora do GRI-Mech): a do O2 para o O3 e a do NO2
  para os outros dois;
- perda na parede (não se estima): dos casos existentes (amonia_sun2022 para
  N2(v), excitados do N2, N, N(2D), N(2P); raiz para Ar*) e, para o oxigênio e os
  NOx, valores de teste: excitados de vida curta 1, O2(a1) 1e-4, O2(b1) 1e-2,
  O2(v) como N2(v), O 1e-3, moléculas estáveis 0.
Uso (com o Cantera): python3 casos/ar_n2_o2/gera_dados.py
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
    "fonte": ["mistura de teste da ferramenta (casos/ar_n2_o2/monta.py)",
              "condutividade pela teoria cinética (Cantera, GRI-Mech 3.0), a 1 atm",
              "cv: média pela fração de 2,5 k_B (N2, O2 sem vibração) e 1,5 k_B (Ar)"],
}
(AQUI / "gas.json").write_text(json.dumps(gas, ensure_ascii=False, indent=2) + "\n")
caso.gera_transporte(AQUI, biblioteca={}, estimar=True)
t = json.loads((AQUI / "transporte.json").read_text())
SEMELHANTE = {"O3": "O2", "NO3": "NO2", "N2O5": "NO2"}
for nome, base in SEMELHANTE.items():
    t[nome]["difusao"] = dict(t[base]["difusao"])
    t[nome]["fonte"] = f"difusão como a do {base} (fora do GRI-Mech; valor de teste)"
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
        nota = "perda na parede do caso existente (propriedade da superfície)"
    elif nome.upper() in TESTE:
        e["perda_na_parede"] = existentes[TESTE[nome.upper()]]
        nota = f"perda na parede como a do {TESTE[nome.upper()]} (valor de teste)"
    else:
        e["perda_na_parede"] = VALORES[nome.upper()]
        nota = "perda na parede: valor de teste"
    e["fonte"] = (e.get("fonte", "") + "; " + nota).lstrip("; ")
(AQUI / "transporte.json").write_text(json.dumps(t, ensure_ascii=False, indent=2) + "\n")
print("gas.json e transporte.json gravados")
