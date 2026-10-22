# SPDX-License-Identifier: GPL-3.0-or-later
"""Exported chemistry (quimica.json): the model runs without zdplaskin_m.F90, which
is ZDPlasKin code and cannot be redistributed. The chemistry loaded from the JSON
must give the same coefficients, rates, sources and energy per reaction as the one
read from zdplaskin_m.F90, bit for bit, in random states."""
import json

import numpy as np
import pytest

from conftest import RAIZ, precisa_arquivo
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


@pytest.mark.parametrize("caso", [pytest.param(c, marks=precisa_arquivo(str((c / "zdplaskin_m.F90").relative_to(RAIZ))))
                                  for c in CASOS], ids=lambda c: c.name or "argonio")
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


@precisa_arquivo("casos/ar_n2_o2/zdplaskin_m.F90")
def test_json_com_literal_alterado_e_detectado(tmp_path):
    # a number of the program changed by 1% must change the coefficients: the
    # bit for bit comparison of the test above does not let badly copied text through
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


@precisa_arquivo("casos/n2/zdplaskin_m.F90")
def test_json_sem_reacao_e_recusado(tmp_path):
    caso = RAIZ / "casos" / "n2"
    q = Quimica(caso / "zdplaskin_m.F90")
    arq = tmp_path / "quimica.json"
    q.salva(arq, bolsigdb=caso / "bolsigdb.dat")
    d = json.loads(arq.read_text())
    d["programa"] = d["programa"][:-1]
    arq.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="reaction"):
        _avalia(Quimica.de_json(arq), _estados(q))


@precisa_arquivo("casos/n2/zdplaskin_m.F90")
def test_simulacao_so_com_quimica_json(tmp_path):
    # case without zdplaskin_m.F90 and without bolsigdb.dat, only with quimica.json: the
    # run is identical to that of the original case
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
    # a case exported without the thresholds it needs cannot run with the
    # bolsigdb.dat of the argon case at the repository root, which is of another gas
    import shutil
    import subprocess
    import sys
    from conftest import PROTOTIPO
    # in argon the reactions of Ar* with electrons need the thresholds
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
    assert r.returncode != 0 and "threshold" in r.stdout + r.stderr and "Traceback" not in r.stderr


def test_json_de_outro_formato_e_recusado(tmp_path):
    arq = tmp_path / "quimica.json"
    arq.write_text(json.dumps({"formato": "outra coisa"}))
    from quimica_zdp import ErroQuimica
    with pytest.raises(ErroQuimica, match="formato"):
        Quimica.de_json(arq)


def test_json_cobre_if_divisao_inteira_e_any_neutral(tmp_path):
    # synthetic module of the group (testes/dados), which also runs in the package:
    # IF / ELSE IF / ELSE, one-line IF, integer division, single precision literal,
    # PARAMETER and ANY_NEUTRAL go through the JSON without changing a bit
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
    # the IF branches appear in the random cells: the factor of reaction 3 changes
    tg, en = est[1], est[3]
    ramo = np.where((tg > 600) & (en < 100), 2.0, np.where(tg > 450, 3.0, 1.0))
    assert len(set(ramo)) == 3


def _com_texto(tmp_path, texto):
    """quimica.json of the argon root with the first expression replaced."""
    d = json.loads((RAIZ / "quimica.json").read_text())
    p = next(p for p in d["programa"] if "texto" in p)
    p["texto"] = texto
    arq = tmp_path / "quimica.json"
    arq.write_text(json.dumps(d))
    return arq


@pytest.mark.parametrize("texto", [
    "().__class__.__mro__[1].__subclasses__().__len__()",
    "__import__('os').getcwd()",
    "exp.__globals__",
    "(lambda: 1)()",
    "[x for x in (1, 2)][0]",
    "'texto'",
    "exp(tgas, out=None)",
    "open",
])
def test_json_com_codigo_fora_da_aritmetica_e_recusado(tmp_path, texto):
    # quimica.json is the format for sharing cases: its expressions are evaluated,
    # so anything beyond arithmetic on the known names is refused when loading
    from quimica_zdp import ErroQuimica
    with pytest.raises(ErroQuimica, match="not allowed"):
        Quimica.de_json(_com_texto(tmp_path, texto))


def test_json_com_aritmetica_valida_carrega(tmp_path):
    q = Quimica.de_json(_com_texto(tmp_path, "_s(1.0e-10) * exp(-_s(100.0) / tgas) * (te / 300.0) ** 0.5"))
    k = q.coeficientes(np.full((2, 4), 1e-9), np.array([300.0, 600.0]), np.array([1e4, 2e4]))
    assert np.isfinite(k).all()


def _com_entrada(tmp_path, muda):
    """quimica.json of the argon root with the first BOLSIG+ entry of the program changed."""
    d = json.loads((RAIZ / "quimica.json").read_text())
    p = next(p for p in d["programa"] if "texto" not in p)
    muda(p)
    arq = tmp_path / "quimica.json"
    arq.write_text(json.dumps(d))
    return arq


@pytest.mark.parametrize("muda", [
    lambda p: p.update(coluna_bolsig="().__class__.__base__.__subclasses__() or 0*tgas"),
    lambda p: p.update(coluna_bolsig=-1),
    lambda p: p.update(coluna_bolsig=10**6),
    lambda p: p.update(coluna_bolsig=1.5),
    lambda p: p.update(alvo="__import__"),
    lambda p: p.update(alvo=10**6),
], ids=["texto", "negativa", "grande", "real", "alvo_dunder", "alvo_fora"])
def test_entrada_do_bolsig_invalida_e_recusada(tmp_path, muda):
    # the entries without expression are a column of the BOLSIG+ table: an integer in
    # range, for a reaction in range; anything else would reach eval or index outside
    from quimica_zdp import ErroQuimica
    with pytest.raises(ErroQuimica, match="not allowed"):
        Quimica.de_json(_com_entrada(tmp_path, muda))


@pytest.mark.parametrize("texto", ["9**9**9**9", "tgas ** 1000", "2 ** -100"])
def test_potencia_com_expoente_enorme_e_recusada(tmp_path, texto):
    # an integer power tower would block the process before any evaluation
    from quimica_zdp import ErroQuimica
    with pytest.raises(ErroQuimica, match="not allowed"):
        Quimica.de_json(_com_texto(tmp_path, texto))
