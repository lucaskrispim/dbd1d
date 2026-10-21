# SPDX-License-Identifier: GPL-3.0-or-later
"""Retomada fina (scripts/retoma_fina.py): a linha de comando remontada do
parametros.json de uma execução repete todas as opções dela e troca só o
destino, a retomada, o fim e o intervalo de gravação."""
import sys

from conftest import RAIZ

sys.path.insert(0, str(RAIZ / "scripts"))
sys.path.insert(0, str(RAIZ / "prototipo"))
import retoma_fina  # noqa: E402
from simula import cria_parser  # noqa: E402

# argumentos gravados por uma execução de sensibilidade do Yang na mercury
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
    # pulso n começa em (n - 1) períodos
    assert retoma_fina.inicio_do_pulso(5, 1e-5) == 4e-5
    assert retoma_fina.inicio_do_pulso(1, 2.5e-5) == 0.0
