# SPDX-License-Identifier: GPL-3.0-or-later
"""ferramentas/versoes.py: a run whose inputs changed after it is marked to be redone (phase 5 of
docs/auditoria/2026-11-11_cadeia/PLANO.md)."""
import hashlib
import json
import sys

from conftest import RAIZ

sys.path.insert(0, str(RAIZ / "ferramentas"))
import versoes  # noqa: E402


def _rodada(tmp_path, caso_gravado):
    caso = tmp_path / "Argonio_X" / "casos" / "c"
    caso.mkdir(parents=True)
    (caso / "kinet.inp").write_text("a")
    rodada = tmp_path / "rodada"
    rodada.mkdir()
    sha = hashlib.sha256(b"a").hexdigest()
    (rodada / "parametros.json").write_text(json.dumps(
        {"argumentos": {"caso": caso_gravado or str(caso)}, "sha256": {"kinet.inp": sha, "falta.dat": sha}}))
    return caso, rodada / "parametros.json"


def test_entrada_igual_e_atual_e_mudada_e_acusada(tmp_path):
    caso, arquivo = _rodada(tmp_path, None)
    assert versoes.confere(arquivo, tmp_path)[1:] == ([], ["falta.dat"])
    (caso / "kinet.inp").write_text("b")
    assert versoes.confere(arquivo, tmp_path)[1] == ["kinet.inp"]


def test_caminho_de_outra_maquina_e_mapeado(tmp_path):
    caso, arquivo = _rodada(tmp_path, "/home/lucas/argonio_val/casos/c")
    assert versoes._local("/home/lucas/argonio_val/casos/c", tmp_path / "Argonio_X") == caso
    assert versoes.confere(arquivo, tmp_path / "Argonio_X")[1] == []


def _origem(tmp_path):
    caso = tmp_path / "casos" / "c"
    caso.mkdir(parents=True)
    (caso / "kinet.inp").write_text("a")
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "s.py").write_text("x")
    csv_ = tmp_path / "r.csv"
    csv_.write_text("id\n1\n")
    sha = lambda b: hashlib.sha256(b).hexdigest()
    reg = {"csv": "r.csv", "sha256_csv": sha(b"id\n1\n"),
           "casos": {"c": {"sha256": {"scripts/s.py": sha(b"x")}, "sha256_entradas": {"kinet.inp": sha(b"a")}}}}
    origem = tmp_path / "r.csv.origem.json"
    origem.write_text(json.dumps(reg))
    return caso, csv_, origem


def test_origem_de_csv_confere_csv_codigo_e_entradas(tmp_path):
    # the CSVs of the reruns carry <csv>.origem.json (proveniencia.grava_origem); versoes.py
    # checked only parametros.json of the runs (wave 3 of the chain audit)
    caso, csv_, origem = _origem(tmp_path)
    assert versoes.confere_origem(origem, tmp_path) == []
    (caso / "kinet.inp").write_text("b")
    (tmp_path / "scripts" / "s.py").write_text("y")
    csv_.write_text("id\n2\n")
    assert sorted(versoes.confere_origem(origem, tmp_path)) == ["c/kinet.inp", "r.csv", "scripts/s.py"]


def test_csv_sem_registro_e_apontado(tmp_path):
    # a CSV without <csv>.origem.json passed in silence (the glm reviewer of wave 3: three pieces of
    # the v45 ladder); the literature data are transcriptions, not results, and stay out
    _, csv_, origem = _origem(tmp_path)
    (tmp_path / "solto.csv").write_text("id\n1\n")
    lit = tmp_path / "docs" / "literatura"
    lit.mkdir(parents=True)
    (lit / "medida.csv").write_text("x\n1\n")
    assert versoes.csv_sem_registro(tmp_path, tmp_path) == ["solto.csv"]


def test_origem_sem_caso_confere_o_codigo(tmp_path):
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "s.py").write_text("x")
    saida = tmp_path / "r.json"
    saida.write_text("{}")
    sha = lambda b: hashlib.sha256(b).hexdigest()
    origem = tmp_path / "r.json.origem.json"
    origem.write_text(json.dumps({"csv": "r.json", "sha256_csv": sha(b"{}"), "casos": {},
                                  "codigo": {"sha256": {"scripts/s.py": sha(b"x")}, "sha256_entradas": {}}}))
    assert versoes.confere_origem(origem, tmp_path) == []
    (tmp_path / "scripts" / "s.py").write_text("y")
    assert versoes.confere_origem(origem, tmp_path) == ["scripts/s.py"]


def test_arvore_de_nome_qualquer_na_maquina_remota_e_mapeada(tmp_path):
    # the runs of the ammonia wave 3 live in ~/amonia_onda3_<date> or ~/onda3_20261112, whose
    # names do not start with Argonio (docs/campanha_amonia/onda3/PLANO.md, section 9)
    raiz = tmp_path / "Argonio_X"
    caso, arquivo = _rodada(tmp_path, "/home/lucas/amonia_onda3_20261121/casos/c")
    assert versoes._local("/home/lucas/amonia_onda3_20261121/casos/c", raiz) == caso
    assert versoes.confere(arquivo, raiz)[1:] == ([], ["falta.dat"])


def test_transporte_e_origem_da_retomada_sao_conferidos(tmp_path):
    # --transport and --restart record sha256 of files outside the case (keys transporte and
    # parametros_da_origem); without them, every such run was sem_entrada_local
    raiz = tmp_path / "Argonio_X"
    caso, _ = _rodada(tmp_path, None)
    sens = raiz / "execucoes" / "sens"
    sens.mkdir(parents=True)
    (sens / "gH.json").write_text("t")
    origem = raiz / "execucoes" / "pulsos" / "T573"
    origem.mkdir(parents=True)
    (origem / "parametros.json").write_text("p")
    sha = lambda b: hashlib.sha256(b).hexdigest()
    arquivo = tmp_path / "r.json"
    arquivo.write_text(json.dumps({
        "argumentos": {"caso": "/home/lucas/amonia_onda3/casos/c",
                       "transporte": "/home/lucas/amonia_onda3/execucoes/sens/gH.json",
                       "retoma": ["/home/lucas/amonia_onda3/execucoes/pulsos/T573", "0.0004"]},
        "sha256": {"kinet.inp": sha(b"a"), "transporte": sha(b"t"), "parametros_da_origem": sha(b"p")}}))
    assert versoes.confere(arquivo, raiz)[1:] == ([], [])
    (sens / "gH.json").write_text("u")
    (origem / "parametros.json").write_text("q")
    assert versoes.confere(arquivo, raiz)[1] == ["transporte", "parametros_da_origem"]


def test_caminho_relativo_e_lido_a_partir_do_cwd_da_rodada(tmp_path, monkeypatch):
    # the launch scripts pass --restart execucoes/... relative to the folder of the run, which
    # simula.py records as cwd; the check must not depend on the folder it is run from
    raiz = tmp_path / "Argonio_X"
    _rodada(tmp_path, None)
    origem = raiz / "execucoes" / "pulsos" / "T573"
    origem.mkdir(parents=True)
    (origem / "parametros.json").write_text("p")
    arquivo = tmp_path / "r.json"
    arquivo.write_text(json.dumps({
        "cwd": "/home/lucas/amonia_onda3_20261121",
        "argumentos": {"caso": "/home/lucas/amonia_onda3_20261121/casos/c",
                       "retoma": ["execucoes/pulsos/T573", "0.0004"]},
        "sha256": {"parametros_da_origem": hashlib.sha256(b"p").hexdigest()}}))
    # from another folder that holds a different file under the same relative path
    outra = tmp_path / "outra"
    (outra / "execucoes" / "pulsos" / "T573").mkdir(parents=True)
    (outra / "execucoes" / "pulsos" / "T573" / "parametros.json").write_text("outro")
    for pasta in (outra, raiz, tmp_path):
        monkeypatch.chdir(pasta)
        assert versoes.confere(arquivo, raiz)[1:] == ([], []), pasta
