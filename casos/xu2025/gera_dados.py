# SPDX-License-Identifier: GPL-3.0-or-later
"""Gas and transport data of the case xu2025 (N2:H2 25:75), by the same method as
the case amonia_sun2022: conductivity and diffusion by kinetic theory (Cantera and
GRI-Mech 3.0), wall losses as in biblioteca/transporte.json (entries
"em N2+H2") and mobilities by Blanc's law with the fractions 0.25 and 0.75.
Usage (in a Python with Cantera): python3 gera_dados.py
Writes gas.json and transporte.json in this folder; the mass comes afterwards from
ferramentas/caso.py transport. The "fonte" strings are data written to the JSON
files and stay as they are there.
"""
import json
import pathlib
import sys

import cantera as ct
import numpy as np

AQUI = pathlib.Path(__file__).resolve().parent
RAIZ = AQUI.parent.parent
sys.path.insert(0, str(RAIZ / "ferramentas"))
import caso  # noqa: E402

X = {"N2": 0.25, "H2": 0.75}
g = ct.Solution("gri30.yaml")
g.transport_model = "mixture-averaged"

T = [200, 250, 295, 350, 400, 500, 600, 800, 1000]
kappa = []
for t in T:
    g.TPX = t, 1333.2, X
    kappa.append(round(float(g.thermal_conductivity), 5))
gas = {
    "_formato": json.loads((RAIZ / "casos" / "n2" / "gas.json").read_text())["_formato"],
    "fundo": dict(X),
    "cv_por_kb": 2.5,
    "condutividade": {"tipo": "tabela", "T": T, "kappa": kappa},
    "ion_inicial": "N2^+",
    "fonte": ["mistura N2:H2 25:75 de Xu et al. 2025 (Nat. Commun., doi 10.1038/s41467-025-66403-6)",
              "condutividade da mistura pela teoria cinética (Cantera e GRI-Mech 3.0), como no caso amonia_sun2022",
              "cv de gás diatômico sem vibração, 2,5 k_B"],
}
(AQUI / "gas.json").write_text(json.dumps(gas, ensure_ascii=False, indent=2) + "\n")

# trace diffusion in the mixture: 1/D = sum x_j/D_kj, at 1 atm; exponent from 250 to 1000 K
iN2, iH2 = g.species_index("N2"), g.species_index("H2")
def difusao(esp):
    k = g.species_index(esp)
    ts = np.array([250.0, 295.0, 400.0, 600.0, 1000.0])
    d = []
    for t in ts:
        g.TPX = t, ct.one_atm, X
        b = g.binary_diff_coeffs
        d.append(1.0 / (X["N2"] / b[k, iN2] + X["H2"] / b[k, iH2]))
    n = np.polyfit(np.log(ts / 300.0), np.log(np.array(d) / np.interp(300.0, ts, d)), 1)[0]
    return {"D0": float(np.interp(300.0, ts, d)), "T0": 300.0, "expoente": round(float(n), 3)}

bib = json.loads((RAIZ / "biblioteca" / "transporte.json").read_text())
transporte = {"_formato": json.loads((RAIZ / "transporte.json").read_text())["_formato"]}
base = {"N2": "N2", "H2": "H2"}
for chave, ent in bib.items():
    if not chave.endswith(" em N2+H2"):
        continue
    esp = chave.split(" em ")[0]
    novo = {}
    if "difusao" in ent:
        ref = ent["fonte"].split("(como ")[1].split(")")[0] if "(como " in ent["fonte"] else esp
        novo["difusao"] = difusao(ref)
        novo["perda_na_parede"] = ent["perda_na_parede"]
        novo["fonte"] = f"difusão de traço em N2:H2 25:75 pela teoria cinética (como {ref}); perda na parede " \
                        f"como na entrada {chave} da biblioteca"
    if "mobilidade" in ent:
        alfa = {"N2": 1.740, "H2": 0.8045}
        def lang(gas):
            m1, m2 = caso.massa_u(esp), caso.massa_u(gas)
            return 13.853 / (alfa[gas] * m1 * m2 / (m1 + m2)) ** 0.5
        medida = bib.get(f"{esp} em N2", {}).get("mobilidade")
        if medida and medida.get("tipo") == "tabela":
            k0 = [round(1.0 / (X["N2"] / k + X["H2"] / lang("H2")), 4) for k in medida["K0"]]
            novo["mobilidade"] = {"tipo": "tabela", "E_N": medida["E_N"], "K0": k0, "acima": medida.get("acima", "preso")}
            novo["fonte"] = "lei de Blanc em N2:H2 25:75 com a medida em N2 (Ellis et al. 1976) e Langevin em H2"
        else:
            novo["mobilidade"] = {"tipo": "constante", "K0": round(1.0 / (X["N2"] / lang("N2") + X["H2"] / lang("H2")), 4)}
            novo["fonte"] = "lei de Blanc em N2:H2 25:75 com o limite de Langevin nos dois gases (estimativa)"
    transporte[esp] = novo
(AQUI / "transporte.json").write_text(json.dumps(transporte, ensure_ascii=False, indent=2) + "\n")
print("gas.json and transporte.json written;", len(transporte) - 1, "species")
