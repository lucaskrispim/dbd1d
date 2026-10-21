# SPDX-License-Identifier: GPL-3.0-or-later
"""Química exportada (quimica.json): o modelo roda sem o zdplaskin_m.F90, que é
código do ZDPlasKin e não pode ser redistribuído. A química carregada do JSON
tem de dar os mesmos coeficientes, taxas, fontes e energia por reação que a
lida do zdplaskin_m.F90, bit a bit, em estados sorteados."""
import json

import numpy as np
import pytest

from conftest import RAIZ
from quimica_zdp import Quimica

CASOS = [RAIZ, RAIZ / "casos" / "n2", RAIZ / "casos" / "amonia_sun2022", RAIZ / "casos" / "ar_n2_o2"]


def _fundos(caso):
    return list(json.loads((caso / "gas.json").read_text())["fundo"])


def _estados(q, n=7, semente=3):
    r = np.random.default_rng(semente)
    ncol = max([c for c in q.coluna_bolsig if c is not None], default=-1) + 1
    kb = 10.0 ** r.uniform(-14, -8, (n, ncol))
    tg = r.uniform(300.0, 1200.0, n)
    te = r.uniform(300.0, 8e4, n)
    en = r.uniform(0.1, 900.0, n)
    neutros = r.uniform(1e17, 3e19, n)
    dens = 10.0 ** r.uniform(6, 19, (len(q.especies), n))
    return kb, tg, te, en, neutros, dens


def _avalia(q, estado):
    kb, tg, te, en, neutros, dens = estado
    with np.errstate(all="ignore"):
        k = q.coeficientes(kb, tg, te, en if q.usa_en else None, neutros if q.usa_any_neutral else None)
        return k, q.taxas(dens, k), q.fontes(dens, k)


@pytest.mark.parametrize("caso", CASOS, ids=lambda c: c.name)
def test_json_igual_ao_zdplaskin_m(tmp_path, caso):
    q = Quimica(caso / "zdplaskin_m.F90")
    arq = tmp_path / "quimica.json"
    q.salva(arq, bolsigdb=caso / "bolsigdb.dat")
    j = Quimica.de_json(arq)
    assert j.especies == q.especies and j.reacoes == q.reacoes and j.coluna_bolsig == q.coluna_bolsig
    assert np.array_equal(j.carga, q.carga) and np.array_equal(j.ordem, q.ordem)
    assert np.array_equal(j.estequiometria, q.estequiometria)
    assert (j.usa_en, j.usa_any_neutral, j.bolsig_tres_corpos) == (q.usa_en, q.usa_any_neutral, q.bolsig_tres_corpos)
    est = _estados(q)
    for a, b in zip(_avalia(q, est), _avalia(j, est)):
        assert np.array_equal(a, b, equal_nan=True)
    fundos = _fundos(caso)
    assert np.array_equal(j.energia_eletronica(None, fundos), q.energia_eletronica(caso / "bolsigdb.dat", fundos))


def test_json_com_literal_alterado_e_detectado(tmp_path):
    # um número do programa alterado em 1 % tem de mudar os coeficientes: a
    # comparação bit a bit do teste acima não deixa passar texto mal copiado
    import re
    caso = RAIZ / "casos" / "ar_n2_o2"
    q = Quimica(caso / "zdplaskin_m.F90")
    arq = tmp_path / "quimica.json"
    q.salva(arq, bolsigdb=caso / "bolsigdb.dat")
    d = json.loads(arq.read_text())
    numero = re.compile(r"\d+\.\d*e[+-]?\d+")
    p = next(p for p in d["programa"] if "texto" in p and isinstance(p["alvo"], int) and numero.search(p["texto"]))
    p["texto"] = numero.sub(lambda m: repr(float(m.group(0)) * 1.01), p["texto"], count=1)
    arq.write_text(json.dumps(d))
    est = _estados(q)
    assert not np.array_equal(_avalia(q, est)[0], _avalia(Quimica.de_json(arq), est)[0], equal_nan=True)


def test_json_sem_reacao_e_recusado(tmp_path):
    caso = RAIZ / "casos" / "n2"
    q = Quimica(caso / "zdplaskin_m.F90")
    arq = tmp_path / "quimica.json"
    q.salva(arq, bolsigdb=caso / "bolsigdb.dat")
    d = json.loads(arq.read_text())
    d["programa"] = d["programa"][:-1]
    arq.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="reação"):
        _avalia(Quimica.de_json(arq), _estados(q))


def test_simulacao_so_com_quimica_json(tmp_path):
    # caso sem zdplaskin_m.F90 e sem bolsigdb.dat, só com o quimica.json: a
    # execução é idêntica à do caso original
    import shutil
    import subprocess
    import sys
    from conftest import PROTOTIPO
    original = RAIZ / "casos" / "n2"
    copia = tmp_path / "n2"
    shutil.copytree(original, copia, ignore=shutil.ignore_patterns("zdplaskin_m.F90", "bolsigdb.dat", "*.log"))
    Quimica(original / "zdplaskin_m.F90").salva(copia / "quimica.json", bolsigdb=original / "bolsigdb.dat")
    comum = ["--n-celulas", "20", "--energia", "--t-end", "4e-9", "--dt-saida", "2e-9", "--espessura", "1e-3",
             "--v-pico", "5e3", "--ne0", "1e6", "--neutro"]
    env = {"OPENBLAS_NUM_THREADS": "1", "PATH": ""}
    for nome, caso in (("a", original), ("b", copia)):
        r = subprocess.run([sys.executable, str(PROTOTIPO / "simula.py"), str(tmp_path / nome), "--caso", str(caso),
                            *comum], capture_output=True, text=True, env=env)
        assert r.returncode == 0, r.stdout + r.stderr
    for arq in ("superficie.dat", "perfil.dat"):
        assert (tmp_path / "a" / arq).read_text() == (tmp_path / "b" / arq).read_text()
    registro = json.loads((tmp_path / "b" / "parametros.json").read_text())
    assert "quimica.json" in registro["sha256"] and "zdplaskin_m.F90" not in registro["sha256"]


def test_json_sem_limiar_necessario_e_erro_e_nao_usa_o_bolsigdb_da_raiz(tmp_path):
    # um caso exportado sem os limiares de que precisa não pode rodar com o
    # bolsigdb.dat do argônio da raiz, que é de outro gás
    import shutil
    import subprocess
    import sys
    from conftest import PROTOTIPO
    # no argônio as reações do Ar* com elétrons precisam dos limiares
    copia = tmp_path / "ar"
    copia.mkdir()
    for nome in ("tabela_energia_tg.dat", "tabela_eedf.dat", "gas.json", "transporte.json", "quimica.json"):
        shutil.copy(RAIZ / nome, copia / nome)
    d = json.loads((copia / "quimica.json").read_text())
    assert d["limiares_bolsig"]
    d["limiares_bolsig"] = []
    (copia / "quimica.json").write_text(json.dumps(d))
    r = subprocess.run([sys.executable, str(PROTOTIPO / "simula.py"), str(tmp_path / "x"), "--caso", str(copia),
                        "--n-celulas", "10", "--energia", "--t-end", "2e-9", "--dt-saida", "1e-9",
                        "--espessura", "1e-3"], capture_output=True, text=True,
                       env={"OPENBLAS_NUM_THREADS": "1", "PATH": ""})
    assert r.returncode != 0 and "limiar" in r.stdout + r.stderr and "Traceback" not in r.stderr


def test_json_de_outro_formato_e_recusado(tmp_path):
    arq = tmp_path / "quimica.json"
    arq.write_text(json.dumps({"formato": "outra coisa"}))
    from quimica_zdp import ErroQuimica
    with pytest.raises(ErroQuimica, match="formato"):
        Quimica.de_json(arq)


def test_json_cobre_if_divisao_inteira_e_any_neutral(tmp_path):
    # módulo sintético do grupo (testes/dados), que roda também no pacote:
    # IF / ELSE IF / ELSE, IF de uma linha, divisão inteira, literal simples,
    # PARAMETER e ANY_NEUTRAL passam pelo JSON sem mudar um bit
    from conftest import PROTOTIPO
    q = Quimica(PROTOTIPO / "testes" / "dados" / "sintetico_zdp.F90")
    arq = tmp_path / "quimica.json"
    q.salva(arq)
    j = Quimica.de_json(arq)
    assert q.usa_en and q.usa_any_neutral and len(q.avisos) == 2 and j.avisos == q.avisos
    est = _estados(q, n=40)
    a, b = _avalia(q, est), _avalia(j, est)
    for x, y in zip(a, b):
        assert np.array_equal(x, y, equal_nan=True)
    # os ramos do IF aparecem nas células sorteadas: o fator da reação 3 muda
    tg, en = est[1], est[3]
    ramo = np.where((tg > 600) & (en < 100), 2.0, np.where(tg > 450, 3.0, 1.0))
    assert len(set(ramo)) == 3
