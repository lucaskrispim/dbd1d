# SPDX-License-Identifier: GPL-3.0-or-later
"""Fine restart (scripts/retoma_fina.py): the command line rebuilt from the
parametros.json of a run repeats all its options and changes only the destination,
the restart, the end and the output interval."""
import json
import pathlib
import sys

from conftest import RAIZ

sys.path.insert(0, str(RAIZ / "scripts"))
sys.path.insert(0, str(RAIZ / "dbd1d"))
import retoma_fina  # noqa: E402
from simula import cria_parser  # noqa: E402

# arguments recorded by a sensitivity run of Yang on mercury
ORIGINAL = {
    "dir": "/m/execucoes/validacao_yang/YS_e1.5", "n_celulas": 50, "agrupamento": 3.0, "v_pico": 1000.0,
    "periodo": 1e-05, "forma": "tabela", "tensao_tabela": "/m/validacao/yang2022/pulso50.txt", "t_gas": 300.0,
    "subida": 0.0, "largura": 0.0, "descida": 0.0, "espessura": 0.0015, "espessura_dir": None,
    "eps_dieletrico_dir": None, "pressao": 19998.0, "gap": 0.008, "eps_dieletrico": 3.8, "eps_gas": 1.0,
    "gama": 0.05, "ne0": 1000000.0, "neutro": True, "sem_freio": False, "t_end": 5e-05, "dt_saida": 1e-06,
    "rtol": 0.0001, "atol": 1e-10, "retoma": None, "energia": True, "eps_sec": 5.0, "gas_solto": False,
    "caso": "/m/casos/yang2022_N2", "transporte": None, "k0_ar_ion": None, "en_aquecimento_ar_ion": None,
    "k0_ar2_ion": None, "t_parede": 300.0, "ganho_tres_corpos": 0.0, "aquecimento": True, "adaptativa": False,
    "dt_malha": 2e-11, "peso_adaptativo": 1.0, "jac": "dae", "faces": None}


def test_repete_as_opcoes_e_troca_so_a_janela():
    linha = retoma_fina.linha_de_comando(ORIGINAL, "/m/destino", 4e-5, 4.024e-5, 5e-10)
    a = vars(cria_parser().parse_args(linha))
    trocadas = {"dir", "retoma", "t_end", "dt_saida"}
    for k, v in ORIGINAL.items():
        if k not in trocadas:
            assert (str(a[k]) if v is not None and k in ("tensao_tabela", "caso") else a[k]) == v, k
    assert str(a["dir"]) == "/m/destino"
    assert a["retoma"] == ["/m/execucoes/validacao_yang/YS_e1.5", repr(4e-5)]
    assert a["t_end"] == 4.024e-5 and a["dt_saida"] == 5e-10


def test_inicio_do_pulso():
    # pulse n starts at (n - 1) periods
    assert retoma_fina.inicio_do_pulso(5, 1e-5) == 4e-5
    assert retoma_fina.inicio_do_pulso(1, 2.5e-5) == 0.0


def _gravados(linha):
    """Arguments as simula.py records them in parametros.json (paths as text, through JSON)."""
    a = vars(cria_parser().parse_args(linha))
    return json.loads(json.dumps({k: str(v) if isinstance(v, pathlib.Path) else v for k, v in a.items()}))


def test_ida_e_volta_com_todas_as_opcoes():
    # every option of simula.py set away from its default, including the two-value ones
    linha = ["/m/origem", "--cells", "40", "--clustering", "2.5", "--v-peak", "7e3", "--period", "3.3e-5",
             "--waveform", "pulse", "--t-gas", "373", "--rise", "2e-8", "--width", "3.6e-7", "--fall", "2e-8",
             "--thickness", "1e-3", "--thickness-right", "2e-3", "--eps-dielectric-right", "4.3",
             "--pressure", "2e4", "--gap", "5e-3", "--eps-dielectric", "4.3", "--eps-gas", "1.1",
             "--gamma", "0.02", "--electron-reflection", "0.1", "--residence-time", "0.07",
             "--accelerate", "4", "50", "--ne0", "1e9", "--neutral", "--no-wall-boltzmann", "--t-end", "1e-3",
             "--dt-output", "3.3e-6", "--rtol", "1e-5", "--atol", "1e-9", "--energy", "--eps-sec", "4.0",
             "--free-gas", "--case", "/m/casos/sun", "--transport", "/m/t.json", "--k0-ar-ion", "1.5",
             "--en-heating-ar-ion", "50", "--k0-ar2-ion", "1.8", "--t-wall", "373", "--three-body-gain", "0.5",
             "--heating", "--adaptive", "--dt-mesh", "1e-11", "--adaptive-weight", "2.0", "--jac", "fd",
             "--faces", "/m/faces.txt"]
    original = _gravados(linha)
    padrao = vars(cria_parser().parse_args(["/m/x"]))
    trocadas = {"dir", "retoma", "t_end", "dt_saida", "acelera"}
    # the line sets every option, so that an option added later without a test fails here
    assert {k for k in padrao if k not in trocadas | {"tensao_tabela"} and original[k] == padrao[k]} == set()
    a = _gravados(retoma_fina.linha_de_comando(original, "/m/destino", 4e-5, 4.024e-5, 5e-10))
    for k, v in original.items():
        if k not in trocadas:
            assert a[k] == v, k
    assert a["acelera"] is None                                   # the fine window does not jump
    assert a["retoma"] == ["/m/origem", repr(4e-5)] and a["dir"] == "/m/destino"
