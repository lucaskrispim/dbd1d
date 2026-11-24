# SPDX-License-Identifier: GPL-3.0-or-later
"""Assembly of the public package (empacotamento/monta_pacote.py, research repository
only): a listed file that is not in git is reported, not dropped without notice."""
import sys

import pytest

from conftest import RAIZ, precisa_arquivo

pytestmark = precisa_arquivo("empacotamento/monta_pacote.py")


def _montador():
    sys.path.insert(0, str(RAIZ / "empacotamento"))
    import monta_pacote
    return monta_pacote


def test_padrao_sem_arquivo_rastreado_e_apontado():
    m = _montador()
    no_git = m.rastreados()
    assert RAIZ / "dbd1d" / "modelo.py" in no_git
    # without the tracked file, its pattern is reported
    sem = no_git - {RAIZ / "dbd1d" / "versao.py"}
    antes = m.INCLUI
    try:
        m.INCLUI = ["dbd1d/versao.py", "dbd1d/modelo.py"]
        assert m.padroes_vazios(sem) == ["dbd1d/versao.py"]
        assert m.fora_do_git(sem) == [RAIZ / "dbd1d" / "versao.py"]
        assert m.padroes_vazios(no_git) == [] and m.fora_do_git(no_git) == []
    finally:
        m.INCLUI = antes


def test_nenhum_padrao_sem_arquivo_no_disco():
    # every pattern of the list matches some file on disk (a typo in the list fails here;
    # whether the files are tracked is checked when the package is built)
    m = _montador()
    assert [p for p in m.INCLUI if not any(f.is_file() for f in RAIZ.glob(p))] == []


def test_montagem_para_com_padrao_vazio_ou_arquivo_fora_do_git(tmp_path, monkeypatch):
    m = _montador()
    sem = m.rastreados() - {RAIZ / "dbd1d" / "versao.py"}
    monkeypatch.setattr(m, "rastreados", lambda: sem)
    monkeypatch.setattr(m, "INCLUI", ["dbd1d/versao.py"])
    monkeypatch.setattr(sys, "argv", ["monta_pacote.py", str(tmp_path / "p")])
    with pytest.raises(SystemExit, match="padrões sem nenhum arquivo rastreado"):
        m.main()
    # a pattern with tracked files and one untracked: stops, unless accepted
    monkeypatch.setattr(m, "INCLUI", ["dbd1d/versao.py", "dbd1d/modelo.py"])
    monkeypatch.setattr(m, "padroes_vazios", lambda no_git=None: [])
    with pytest.raises(SystemExit, match="fora do git"):
        m.main()
    assert not (tmp_path / "p").exists()


def test_arquivo_modificado_vai_como_no_commit_salvo_se_aceito(tmp_path, monkeypatch):
    # a tracked file changed on disk and not committed goes into the package as it is in
    # HEAD; only the files named with --aceita-modificado (the version bump of the release)
    # go as they are on disk
    m = _montador()
    alvo, outro = RAIZ / "dbd1d" / "versao.py", RAIZ / "dbd1d" / "tabela.py"
    commit = {alvo: b"VERSAO = 'do commit'\n", outro: b"# tabela do commit\n"}
    monkeypatch.setattr(m, "modificados", lambda: {alvo, outro})
    monkeypatch.setattr(m, "do_commit", lambda f: commit[f])
    monkeypatch.setattr(m, "INCLUI", ["dbd1d/versao.py", "dbd1d/tabela.py"])
    monkeypatch.setattr(sys, "argv", ["monta_pacote.py", str(tmp_path / "p"), "--aceita-modificado", "dbd1d/versao.py"])
    m.main()
    assert (tmp_path / "p" / "dbd1d" / "versao.py").read_text().endswith(alvo.read_text())
    assert (tmp_path / "p" / "dbd1d" / "tabela.py").read_text() == \
        "# SPDX-License-Identifier: GPL-3.0-or-later\n# tabela do commit\n"
    # a name given to --aceita-modificado that is not a modified file of the package stops
    monkeypatch.setattr(sys, "argv", ["monta_pacote.py", str(tmp_path / "q"), "--aceita-modificado", "dbd1d/simula.py"])
    with pytest.raises(SystemExit, match="não é arquivo modificado do pacote"):
        m.main()
    # the forms that would be ignored without notice stop, and so does a file missing from HEAD,
    # before anything is written
    for argv in (["--aceita-modificado=dbd1d/versao.py"], ["--aceita-modificado"]):
        monkeypatch.setattr(sys, "argv", ["monta_pacote.py", str(tmp_path / "r"), *argv])
        with pytest.raises(SystemExit, match="um arquivo por vez"):
            m.main()

    def fora_do_head(f):
        raise SystemExit(f"{f} não está no HEAD")
    monkeypatch.setattr(m, "do_commit", fora_do_head)
    monkeypatch.setattr(sys, "argv", ["monta_pacote.py", str(tmp_path / "s")])
    with pytest.raises(SystemExit, match="não está no HEAD"):
        m.main()
    assert not (tmp_path / "q").exists() and not (tmp_path / "r").exists() and not (tmp_path / "s").exists()


def test_modificados_e_do_commit_leem_o_git():
    m = _montador()
    assert all(f.is_relative_to(RAIZ) for f in m.modificados())
    assert b"class Modelo" in m.do_commit(RAIZ / "dbd1d" / "modelo.py")


def test_testes_que_rodam_lancadores_nao_distribuidos_pulam_no_pacote():
    # a test that runs shell scripts of remoto/ which the package does not ship has to skip
    # there (bash returns 127 otherwise): it names one of those scripts in a skip guard
    import re
    m = _montador()
    entregues = {f for p in m.INCLUI for f in RAIZ.glob(p) if f.is_file()}
    sem_guarda = []
    for teste in sorted((RAIZ / "dbd1d" / "testes").glob("*.py")):
        fonte = teste.read_text()
        if not re.search(r'RAIZ\s*/\s*"remoto"', fonte):
            continue
        nomes = set(re.findall(r'"([\w.]+\.sh)"', fonte))
        faltam = sorted(n for n in nomes if RAIZ / "remoto" / n not in entregues)
        guardas = re.findall(r'(?:skip|exists)\([^\n]*remoto/([\w.]+\.sh)', fonte)
        if faltam and not any(g in faltam for g in guardas):
            sem_guarda.append(f"{teste.name}: {', '.join(faltam)}")
    assert sem_guarda == []


def test_cabecalho_spdx_proprio_nao_e_duplicado(tmp_path):
    # bdf_dae.py carries code of scipy and declares GPL-3.0-or-later AND BSD-3-Clause itself
    m = _montador()
    m.copia(RAIZ / "dbd1d" / "bdf_dae.py", tmp_path / "bdf_dae.py")
    texto = (tmp_path / "bdf_dae.py").read_text()
    assert texto.count("SPDX-License-Identifier") == 1
    assert texto.startswith("# SPDX-License-Identifier: GPL-3.0-or-later AND BSD-3-Clause")
    assert (RAIZ / "empacotamento" / "raiz" / "LICENSE-scipy.txt").is_file()
    m.copia(RAIZ / "dbd1d" / "modelo.py", tmp_path / "modelo.py")
    assert (tmp_path / "modelo.py").read_text().startswith("# SPDX-License-Identifier: GPL-3.0-or-later\n")


def test_modulos_importados_pelos_testes_sao_distribuidos():
    # the package ships dbd1d/testes/*.py: every repository module that those tests import
    # (through the sys.path insertions of ferramentas, scripts, ...) has to be in INCLUI,
    # unless the importing test names the file in a skip guard (precisa_arquivo or a module
    # level skip), which is how the package leaves out what it does not distribute
    import ast
    m = _montador()
    entregues = {f for p in m.INCLUI for f in RAIZ.glob(p) if f.is_file()}
    pastas = ["ferramentas", "scripts", "artigo", "validacao", "empacotamento", "docs/literatura/dados", "dbd1d"]
    faltam = []
    for teste in sorted((RAIZ / "dbd1d" / "testes").glob("*.py")):
        fonte = teste.read_text()
        for no in ast.walk(ast.parse(fonte)):
            if isinstance(no, ast.Import):
                nomes = [a.name for a in no.names]
            elif isinstance(no, ast.ImportFrom) and no.module and no.level == 0:
                nomes = [no.module]
            else:
                continue
            for nome in nomes:
                for pasta in pastas:
                    f = RAIZ / pasta / (nome.split(".")[0] + ".py")
                    rel = f.relative_to(RAIZ).as_posix()
                    if f.is_file() and f not in entregues and f'"{rel}"' not in fonte:
                        faltam.append(f"{teste.name} imports {rel}")
    assert sorted(set(faltam)) == []
