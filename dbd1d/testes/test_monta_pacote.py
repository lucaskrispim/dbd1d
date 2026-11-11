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
