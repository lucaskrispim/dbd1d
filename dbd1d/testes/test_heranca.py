# SPDX-License-Identifier: GPL-3.0-or-later
"""Inheritance between cases (phase 3 of docs/auditoria/2026-11-11_cadeia/PLANO.md): a transport
value (diffusion or mobility) that appears, identical, in cases with different backgrounds was
copied from one gas to the other, which is the class of ERROS.md E1, E2, E3 and E6. Every such
coincidence fails unless it is listed in EXPLICADAS with the reason."""
import collections
import json

from conftest import RAIZ

# (field, value as sorted JSON): reason
EXPLICADAS = {
    ("difusao", json.dumps({"D0": 1.7e-05, "T0": 273.0, "expoente": 1.9}, sort_keys=True)):
        "self-diffusion of the background Ar (DK of the Fortran code with fundo, used only with "
        "--free-gas, no primary source) equals N2-N2 of Capitelli et al. 2000, table 6.2; Ar* took "
        "the same value until 11/11/2026 (ERROS.md A8); Ar self-diffusion near 0.16 cm2/s at 273 K "
        "makes the value plausible for Ar, to be sourced",
}


def _fundo(gas):
    return json.dumps({k.upper(): float(v) for k, v in gas["fundo"].items()}, sort_keys=True)


def coincidencias(casos):
    """{(field, value): {background: [case:species]}} for values found in more than one background."""
    grupos = collections.defaultdict(lambda: collections.defaultdict(list))
    for pasta in sorted(p for p in casos.iterdir() if p.is_dir()):
        g, t = pasta / "gas.json", pasta / "transporte.json"
        if not (g.exists() and t.exists()):
            continue
        fundo = _fundo(json.loads(g.read_text()))
        for esp, e in json.loads(t.read_text()).items():
            if esp.startswith("_") or not isinstance(e, dict):
                continue
            for campo in ("difusao", "mobilidade"):
                if campo in e:
                    grupos[(campo, json.dumps(e[campo], sort_keys=True))][fundo].append(f"{pasta.name}:{esp}")
    return {k: dict(v) for k, v in grupos.items() if len(v) > 1}


def test_valor_de_transporte_nao_passa_de_um_fundo_para_outro():
    novas = {k: v for k, v in coincidencias(RAIZ / "casos").items() if k not in EXPLICADAS}
    assert not novas, novas


def test_coincidencia_explicada_ainda_existe():
    # an explanation without the coincidence is stale and must go
    assert set(EXPLICADAS) <= set(coincidencias(RAIZ / "casos"))


def test_pega_difusao_copiada_de_outro_gas(tmp_path):
    for nome, fundo, d0 in (("a", {"N2": 1}, 4.64e-5), ("b", {"N2": 0.9, "H2": 0.1}, 4.64e-5), ("c", {"N2": 1.0}, 2e-5)):
        (tmp_path / nome).mkdir()
        (tmp_path / nome / "gas.json").write_text(json.dumps({"fundo": fundo}))
        (tmp_path / nome / "transporte.json").write_text(
            json.dumps({"N2(V1)": {"difusao": {"D0": d0, "T0": 300.0, "expoente": 1.7}}}))
    achadas = coincidencias(tmp_path)
    assert len(achadas) == 1
    (fundos,) = achadas.values()
    assert sorted(sum(fundos.values(), [])) == ["a:N2(V1)", "b:N2(V1)"]
